"""Train TARC with validation-based checkpoint selection.

The default protocol uses 20 epochs and batch size 16. Test evaluation is separate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mosi", "mosei", "iemocap"], required=True)
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing train.npz and val.npz",
    )
    p.add_argument(
        "--encoder",
        type=Path,
        required=True,
        help="Local DeBERTa-v3-large encoder directory",
    )
    p.add_argument(
        "--initialization",
        type=Path,
        required=True,
        help="Shared SST-2 encoder-only safetensors with bert.* keys",
    )
    p.add_argument(
        "--output", type=Path, required=True, help="New/empty output directory"
    )
    p.add_argument("--config", type=Path, help="Override bundled dataset JSON")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--epochs", type=int, help="Override configured epoch count (default: 20)"
    )
    p.add_argument("--device", default="cpu")
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(metrics, dataset):
    if dataset == "iemocap":
        return metrics["weighted_f1"], metrics["accuracy"]
    return int(metrics["mae"] <= 0.82), metrics["acc2_including_zero"], -metrics["mae"]


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Use a new output directory; existing runs are never overwritten"
        )

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoModel, get_linear_schedule_with_warmup
    from safetensors.torch import load_file
    from tarc.data import FeatureDataset, collate, fit_normalizer, to_device
    from tarc.ops import nonempty_dropout
    from tarc.runtime import seed_everything, source_snapshot
    from tarc.metrics import compute_metrics
    from tarc.readouts.selective import selective_loss
    from .api import read_config, build_model

    cfg = read_config(args.dataset, args.config)
    args.epochs = cfg["train"]["epochs"] if args.epochs is None else args.epochs
    cfg["train"]["epochs"] = args.epochs
    settings = cfg["train"]
    batch_size = settings["micro_batch_size"]
    if args.epochs < 1 or batch_size < 1 or settings["eval_batch_size"] < 1:
        raise ValueError("Epoch count and batch sizes must be positive")
    if settings["gradient_accumulation_steps"] != 1:
        raise ValueError(
            "This runner supports the recorded accumulation setting of one"
        )
    if settings["effective_batch_size"] != batch_size:
        raise ValueError("Effective batch size must match the micro batch size")
    if (
        not settings["missing_consistency"]
        or settings["encoder_gradient_policy"] != "none"
    ):
        raise ValueError("Unsupported consistency or encoder gradient policy")
    if settings["selection_metric"] != (
        "weighted_f1_then_accuracy"
        if args.dataset == "iemocap"
        else "acc2_including_zero_subject_to_mae_0p82"
    ):
        raise ValueError("Use the recorded validation-selection convention")
    if (
        not 0 <= settings["missing_probability"] < 1
        or not 0 <= settings["warmup_fraction"] <= 1
    ):
        raise ValueError("Dropout must be in [0,1); warmup must be in [0,1]")
    if (
        min(settings["bert_lr"], settings["tarc_lr"], settings["clip_grad"]) <= 0
        or settings["weight_decay"] < 0
    ):
        raise ValueError(
            "Learning rates/gradient clipping must be positive and weight decay nonnegative"
        )
    if (
        cfg["selective"]["initial_gate"] != 0.1
        or cfg["selective"]["beta"] != 0.5
        or cfg["selective"]["max_magnitude"] != 3.0
    ):
        raise ValueError(
            "The released readout fixes initial_gate=.1, beta=.5, max_magnitude=3"
        )
    if cfg["selective"]["label_smoothing"] != 0:
        raise ValueError("The final loss uses no label smoothing")
    cfg["bert"]["local_path"] = None  # Do not export a private machine path.
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    seed_everything(args.seed, settings["threads"])
    task = "classification" if args.dataset == "iemocap" else "sentiment"
    data = {
        name: FeatureDataset(
            args.data / f"{name}.npz", ["text", "audio", "vision"], task, num_classes=4
        )
        for name in ["train", "val"]
    }
    if set(data["train"].ids) & set(data["val"].ids):
        raise ValueError("Train/validation sample ID overlap")
    for split in data.values():
        if not split.has_bert:
            raise ValueError("The final model requires online DeBERTa token inputs")
        if any(
            split.dims[m] != cfg["model"]["input_dims"][m] for m in ["audio", "vision"]
        ):
            raise ValueError(
                "Processed AV widths do not match the final dataset interface"
            )
    normalizer = fit_normalizer(data["train"])
    for split in data.values():
        split.normalizer = normalizer

    encoder = AutoModel.from_pretrained(
        args.encoder, local_files_only=True, use_safetensors=True
    )
    initialization = load_file(str(args.initialization), device="cpu")
    initialization = {
        k[len("bert.") :]: value
        for k, value in initialization.items()
        if k.startswith("bert.")
    }
    if not initialization:
        raise ValueError(
            "Initialization must contain the recorded encoder-only bert.* state"
        )
    encoder.load_state_dict(initialization, strict=True)
    del initialization
    model = build_model(cfg, encoder, train_last_n=cfg["bert"]["train_last_n"]).to(
        device
    )
    text_parameters = list(model.bert.parameters())
    text_ids = {id(p) for p in text_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in text_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": text_parameters, "lr": settings["bert_lr"]},
            {"params": other_parameters, "lr": settings["tarc_lr"]},
        ],
        weight_decay=settings["weight_decay"],
    )

    class Indexed(FeatureDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["index"] = index
            return item

    # Reuse the validated arrays/normalizer without reading another split.
    indexed = Indexed.__new__(Indexed)
    indexed.__dict__.update(data["train"].__dict__)

    def indexed_collate(samples):
        batch = collate(samples)
        batch["index"] = torch.tensor([s["index"] for s in samples], dtype=torch.long)
        return batch

    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if settings["video_balanced_sampling"]:
        groups = [str(value).split("$_$")[0] for value in data["train"].ids]
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        sampler = WeightedRandomSampler(
            torch.tensor(1 / counts[inverse], dtype=torch.double),
            len(groups),
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        indexed,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        collate_fn=indexed_collate,
    )
    validation = DataLoader(
        data["val"],
        batch_size=settings["eval_batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    total_updates = args.epochs * math.ceil(len(indexed) / batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(settings["warmup_fraction"] * total_updates), total_updates
    )
    args.output.mkdir(parents=True, exist_ok=True)
    import platform
    from importlib.metadata import version

    environment = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: version(name)
            for name in [
                "torch",
                "numpy",
                "transformers",
                "safetensors",
                "scipy",
                "scikit-learn",
                "PyYAML",
            ]
        },
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "threads": torch.get_num_threads(),
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    registration = {
        "dataset": args.dataset,
        "seed": args.seed,
        "config": copy.deepcopy(cfg),
        "source_sha256": source_snapshot(args.output),
        "environment": environment,
        "encoder_config": encoder.config.to_dict(),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "precision": "FP32",
        "test_loaded": False,
        "data_sha256": {
            split: digest(args.data / f"{split}.npz") for split in ["train", "val"]
        },
        "initialization_sha256": digest(args.initialization),
        "entry": "tarc.train",
        "mode": "new_portable_training_run",
        "published_budget_preserved": args.epochs == 20 and batch_size == 16,
    }
    (args.output / "registration.json").write_text(
        json.dumps(registration, indent=2) + "\n"
    )
    (args.output / "encoder_config.json").write_text(
        json.dumps(encoder.config.to_dict(), indent=2) + "\n"
    )
    np.savez_compressed(
        args.output / "train_only_normalizer.npz",
        **{
            f"{m}__{key}": value
            for m, stats in normalizer.items()
            for key, value in stats.items()
        },
    )

    best = None
    updates = 0
    history, streams = [], {}
    for epoch in range(1, args.epochs + 1):
        losses, order = [], []
        for batch in loader:
            order.extend(batch["index"].tolist())
            batch = to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(batch)
            full = model.decode(encoded)
            dropped = model.decode(
                encoded,
                nonempty_dropout(batch["available"], settings["missing_probability"]),
            )
            loss, _ = selective_loss(
                model,
                full,
                dropped,
                batch["y"],
                cfg["loss"],
                utility_weight=cfg["selective"]["utility_weight"],
                endpoint_weight=cfg["selective"]["endpoint_weight"],
                temperature=cfg["selective"]["temperature"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["clip_grad"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            updates += 1
            losses.append(float(loss.detach()))
        model.eval()
        ys, predictions, ids = [], [], []
        with torch.inference_mode():
            for batch in validation:
                result = model(to_device(batch, device))["logits"]
                if not torch.isfinite(result).all():
                    raise FloatingPointError("Nonfinite validation prediction")
                ys.append(batch["y"].numpy())
                predictions.append(result.detach().cpu().numpy())
                ids.extend(batch["ids"])
        arrays = {
            "y": np.concatenate(ys),
            "logits": np.concatenate(predictions),
            "ids": np.asarray(ids),
        }
        metrics = compute_metrics(arrays["y"], arrays["logits"], task)
        selected = best is None or rank(metrics, args.dataset) > rank(
            best["metrics"], args.dataset
        )
        if selected:
            temporary = args.output / "best.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalizer": normalizer,
                    "registration": registration,
                    "epoch": epoch,
                    "updates": updates,
                },
                temporary,
            )
            temporary.replace(args.output / "best.pt")
            np.savez_compressed(
                args.output / "best_validation_predictions.npz", **arrays
            )
            best = {"epoch": epoch, "metrics": metrics}
        streams[f"epoch_{epoch}"] = np.asarray(order, dtype=np.int64)
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "train_loss": float(np.mean(losses)),
                "validation": metrics,
                "selected": selected,
            }
        )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(args.output / "training_index_streams.npz", **streams)
        print(json.dumps(history[-1], allow_nan=False), flush=True)
    if updates != total_updates:
        raise RuntimeError("Unexpected optimizer update count")
    summary = {
        "status": "completed_train_validation_only",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "updates": updates,
        "best": best,
        "test_loaded": False,
        "checkpoint_sha256": digest(args.output / "best.pt"),
    }
    (args.output / "completed_training.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
