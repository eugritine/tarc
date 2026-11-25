"""Load a final checkpoint and predict from the audited processed NPZ interface.

No text, audio or visual backbone is silently substituted. This interface needs
the recorded word-aligned token arrays and supplied audio/visual feature arrays;
it is not a raw-video/ASR feature extraction pipeline. Historical ``.pt`` files
contain Python/NumPy metadata and must be obtained from a trusted source.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIMS = {
    "mosi": {"text": 1024, "audio": 5, "vision": 20},
    "mosei": {"text": 1024, "audio": 74, "vision": 35},
    "iemocap": {"text": 1024, "audio": 100, "vision": 512},
}
CLASS_NAMES = ["happy_or_excited", "sad", "neutral", "angry"]


def read_config(dataset: str, config: str | Path | dict | None = None) -> dict:
    """Read the shipped frozen configuration, or an explicitly supplied copy."""
    dataset = dataset.lower()
    if dataset not in DATASET_DIMS:
        raise ValueError(f"Unknown final dataset: {dataset}")
    if isinstance(config, dict):
        value = copy.deepcopy(config)
    else:
        path = Path(config) if config else RELEASE_ROOT / "configs" / f"{dataset}.json"
        if config is None and not path.is_file():
            path = Path(__file__).resolve().parent / "resources" / f"{dataset}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
    value["model"].update(
        task="classification" if dataset == "iemocap" else "sentiment",
        input_dims=DATASET_DIMS[dataset].copy(),
        num_classes=4,
    )
    value["dataset"] = dataset
    return value


def encoder_configuration(path: str | Path | dict | None = None):
    from transformers import AutoConfig

    if isinstance(path, dict):
        fields = copy.deepcopy(path)
    else:
        target = (
            Path(path) if path else RELEASE_ROOT / "configs" / "encoder_config.json"
        )
        if path is None and not target.is_file():
            target = (
                Path(__file__).resolve().parent / "resources" / "encoder_config.json"
            )
        if target.is_dir():
            target = target / "config.json"
        fields = json.loads(target.read_text(encoding="utf-8"))
    model_type = fields.pop("model_type")
    return AutoConfig.for_model(model_type, **fields)


def build_model(config: dict, encoder, *, train_last_n: int = 24):
    """Instantiate the evidence core and task-specific readout."""
    from tarc.config import ModelConfig
    from tarc.bert_model import TARCTextEncoder
    from tarc.readouts.unified import UnifiedReadoutTARC

    cfg = ModelConfig(**config["model"]).validate()
    core = TARCTextEncoder(cfg, bert_model=encoder, train_last_n=train_last_n)
    # The classification branch inherits SelectiveTARC execution unchanged.
    return UnifiedReadoutTARC(core)


@dataclass
class LoadedTARC:
    model: Any
    normalizer: dict
    config: dict
    dataset: str
    metadata: dict
    device: Any

    def predict_batch(
        self, batch: dict, available_modalities: Iterable[str] | None = None
    ) -> dict:
        import torch
        from tarc.data import to_device

        batch = to_device(batch, self.device)
        available = None
        if available_modalities is not None:
            chosen = set(available_modalities)
            if not chosen or not chosen <= set(self.model.cfg.modalities):
                raise ValueError("Choose a nonempty subset of text, audio and vision")
            keep = torch.tensor(
                [m in chosen for m in self.model.cfg.modalities],
                dtype=torch.bool,
                device=self.device,
            )
            available = batch["available"] & keep[None, :]
            if not available.any(-1).all():
                raise ValueError("A selected sample would have no observed input")
        self.model.eval()
        with torch.inference_mode():
            out = dict(self.model(batch, available=available))
            if self.model.sentiment:
                # The pure evidence endpoint is tanh(f)*a_F. It is not q_s=q_a=1.
                f = self.model.core.head(out["z"])
                out["evidence_prediction"] = torch.tanh(f) * out["fused_magnitude"]
        return out


def load_model(
    checkpoint: str | Path,
    dataset: str | None = None,
    config: str | Path | dict | None = None,
    encoder_config: str | Path | dict | None = None,
    device: str = "cpu",
) -> LoadedTARC:
    """Strictly load an original final checkpoint without base-model weights.

    The checkpoint already contains the entire fine-tuned language encoder.
    Construction on the meta device avoids an unnecessary second 1.74 GB copy.
    The sole nonpersistent DeBERTa position-ID buffer is restored explicitly.
    """
    import numpy as np
    import torch
    from transformers import AutoModel

    path = Path(checkpoint)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if (
        not isinstance(state, dict)
        or not {"state_dict", "normalizer", "registration"} <= state.keys()
    ):
        raise ValueError(
            "Expected a complete final TARC checkpoint, not encoder initialization"
        )
    registration = state["registration"]
    inferred = registration.get("dataset")
    dataset = (dataset or inferred or "").lower()
    if inferred and dataset != inferred.lower():
        raise ValueError("Requested dataset differs from the checkpoint registration")
    cfg = read_config(dataset, config)
    # Some historical IEM registrations carry pre-adapter sentiment widths;
    # the shipped config and actual checkpoint tensors resolve that record.
    for modality, expected in cfg["model"]["input_dims"].items():
        key = f"core.encoders.{modality}.projection.weight"
        if (
            key not in state["state_dict"]
            or state["state_dict"][key].shape[1] != expected
        ):
            raise ValueError(
                f"Checkpoint projection does not match {dataset}/{modality}"
            )
    embedded_encoder = registration.get("encoder_config")
    enc_cfg = encoder_configuration(encoder_config or embedded_encoder)
    with torch.device("meta"):
        encoder = AutoModel.from_config(enc_cfg)
        model = build_model(cfg, encoder)
    # Strict keys and shapes prevent a wrong readout or an initialization-only
    # checkpoint from being accepted as the final model.
    model.load_state_dict(state["state_dict"], strict=True, assign=True)
    for module_name, module in model.named_modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            if name == "position_ids" and tuple(buffer.shape) == (
                1,
                enc_cfg.max_position_embeddings,
            ):
                module._buffers[name] = torch.arange(
                    enc_cfg.max_position_embeddings
                ).expand(1, -1)
            else:
                raise ValueError(
                    f"Unmaterialized checkpoint buffer: {module_name}.{name}"
                )
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    model.to(target).eval()
    normalizer = {
        m: {
            k: (
                v.detach().cpu().numpy()
                if isinstance(v, torch.Tensor)
                else np.asarray(v)
            )
            for k, v in values.items()
        }
        for m, values in state["normalizer"].items()
    }
    metadata = {
        "checkpoint": path.name,
        "epoch": state.get("epoch"),
        "updates": state.get("updates"),
        "seed": registration.get("seed"),
        "protocol_sha256": registration.get("protocol_sha256"),
        "strict_state_load": True,
    }
    return LoadedTARC(model, normalizer, cfg, dataset, metadata, target)


def predict_npz(
    loaded: LoadedTARC,
    npz_path: str | Path,
    indices: Iterable[int] | None = None,
    batch_size: int = 4,
    available_modalities: Iterable[str] | None = None,
) -> list[dict]:
    """Return real predictions from processed features, with optional fixed masks."""
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from tarc.data import FeatureDataset, collate

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if available_modalities is not None:
        available_modalities = tuple(available_modalities)
    task = "classification" if loaded.dataset == "iemocap" else "sentiment"
    data = FeatureDataset(
        npz_path,
        ["text", "audio", "vision"],
        task,
        normalizer=loaded.normalizer,
        num_classes=4,
    )
    if not data.has_bert:
        raise ValueError("Processed NPZ must include the recorded text token arrays")
    for m in ["audio", "vision"]:
        if data.dims[m] != loaded.config["model"]["input_dims"][m]:
            raise ValueError(f"Processed {m} width differs from the checkpoint")
    selected = list(range(len(data))) if indices is None else list(indices)
    if any(i < 0 or i >= len(data) for i in selected):
        raise IndexError("Sample index is outside the processed split")
    results = []
    for batch in DataLoader(
        Subset(data, selected), batch_size=batch_size, shuffle=False, collate_fn=collate
    ):
        out = loaded.predict_batch(batch, available_modalities)
        values = {
            k: v.detach().cpu().numpy() for k, v in out.items() if hasattr(v, "detach")
        }
        for i, sample_id in enumerate(batch["ids"]):
            row = {
                "id": sample_id,
                "dataset": loaded.dataset,
                "mode": "live_model_inference",
                "y": float(batch["y"][i]),
                "checkpoint": loaded.metadata.copy(),
            }
            if task == "classification":
                p = values["probabilities"][i]
                pt = out["text_logits"][i].softmax(-1).detach().cpu().tolist()
                pf = out["fused_logits"][i].softmax(-1).detach().cpu().tolist()
                prediction = int(p.argmax())
                row.update(
                    prediction=prediction,
                    label=CLASS_NAMES[prediction],
                    probabilities=p.tolist(),
                    text_probabilities=pt,
                    evidence_probabilities=pf,
                    text_prediction=int(np.argmax(pt)),
                    evidence_prediction=int(np.argmax(pf)),
                    q=float(values["q"][i, 0]),
                )
            else:
                prediction = float(values["logits"][i].reshape(-1)[0])
                row.update(
                    prediction=prediction,
                    label="nonnegative" if prediction >= 0 else "negative",
                    text_prediction=float(values["text_prediction"][i].reshape(-1)[0]),
                    evidence_prediction=float(
                        values["evidence_prediction"][i].reshape(-1)[0]
                    ),
                    q_sign=float(values["q_sign"][i, 0]),
                    q_magnitude=float(values["q_magnitude"][i, 0]),
                )
            for k in [
                "rho",
                "u",
                "available",
                "expert_prob",
                "expert_gate",
                "expert_selected",
            ]:
                if k in values:
                    row[k] = values[k][i].tolist()
            results.append(row)
    # Reject nonfinite values before producing JSON or demonstration claims.
    json.dumps(results, allow_nan=False)
    return results
