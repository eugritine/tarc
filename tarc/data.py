"""Safe numeric-feature NPZ interface. Raw text/images are not feature tensors."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, path, modalities, task, normalizer=None, num_classes=None):
        self.path = Path(path)
        self.modalities = list(modalities)
        self.task = task
        self.normalizer = normalizer
        with np.load(self.path, allow_pickle=False) as f:
            self.arrays = {k: f[k] for k in f.files}
        a = self.arrays
        for k in ["y", "ids"] + self.modalities:
            if k not in a:
                raise ValueError(f"{self.path}: missing NPZ key {k}")
        self.n = len(a["y"])
        if self.n == 0 or a["y"].shape != (self.n,):
            raise ValueError("y must be a nonempty one-dimensional array")
        if not np.isfinite(a["y"]).all():
            raise ValueError("Labels contain NaN/Inf")
        if a["ids"].shape != (self.n,) or a["ids"].dtype.kind not in {"U", "S"}:
            raise ValueError(
                "ids must be a one-dimensional Unicode/bytes array; no object arrays"
            )
        self.ids = a["ids"].astype(str)
        if len(set(self.ids)) != self.n:
            raise ValueError("Duplicate sample IDs within split")
        if "groups" in a and (
            a["groups"].shape != (self.n,) or a["groups"].dtype.kind not in {"U", "S"}
        ):
            raise ValueError("groups must be Unicode/bytes IDs")
        if task == "classification":
            if not np.equal(a["y"], np.floor(a["y"])).all() or np.min(a["y"]) < 0:
                raise ValueError(
                    "Classification y must contain nonnegative integer class IDs"
                )
            if num_classes is not None and np.max(a["y"]) >= num_classes:
                raise ValueError("Class ID exceeds configured num_classes")
        avail = a.get("available", np.ones((self.n, len(modalities)), bool))
        if avail.shape != (self.n, len(modalities)) or not np.isin(avail, [0, 1]).all():
            raise ValueError("available must be a binary [N,M] array")
        self.available = avail.astype(bool)
        bert_keys = ("text_input_ids", "text_attention_mask", "text_token_type_ids")
        self.has_bert = any(k in a for k in bert_keys)
        if self.has_bert:
            if not all(k in a for k in bert_keys):
                raise ValueError(
                    "BERT inputs require IDs, attention mask, and token type IDs"
                )
            if "text" not in self.modalities:
                raise ValueError("BERT inputs require a text modality")
            text_shape = a["text"].shape[:2]
            # A verified word map can join an encoder's subword grid [N,S]
            # to the audio/vision-aligned word grid [N,T], with S != T.
            token_shape = (
                a["text_input_ids"].shape if "text_word_index" in a else text_shape
            )
            if len(token_shape) != 2 or token_shape[0] != self.n or token_shape[1] < 1:
                raise ValueError("Encoder inputs must be a nonempty [N,S] grid")
            for k in bert_keys:
                if a[k].shape != token_shape or a[k].dtype.kind not in {"i", "u", "b"}:
                    raise ValueError(
                        f"{k} must be integer on the common encoder token grid"
                    )
            if (
                np.any(a["text_input_ids"] < 0)
                or not np.isin(a["text_attention_mask"], [0, 1]).all()
            ):
                raise ValueError("Invalid BERT IDs or attention mask")
            if "text_word_index" in a:
                mapping = a["text_word_index"]
                if mapping.shape != token_shape or mapping.dtype.kind not in {"i", "u"}:
                    raise ValueError(
                        "text_word_index must be integer on the encoder token grid"
                    )
                if "bert_lengths" not in a or a["bert_lengths"].shape != (self.n,):
                    raise ValueError("word-aligned BERT inputs require bert_lengths")
                if np.any(a["bert_lengths"] < 1) or np.any(
                    a["bert_lengths"] > token_shape[1]
                ):
                    raise ValueError("Invalid bert_lengths")
                # Stored unsigned maps cannot represent the required -1 sentinel.
                if mapping.dtype.kind == "u" or np.any(mapping < -1):
                    raise ValueError("Invalid text_word_index sentinel")
                if np.any((mapping >= 0) & ~a["text_attention_mask"].astype(bool)):
                    raise ValueError("Mapped wordpiece lies outside BERT attention")
        self.dims = {}
        for mi, m in enumerate(self.modalities):
            x = a[m]
            if (
                x.ndim != 3
                or x.shape[0] != self.n
                or x.shape[1] < 1
                or x.shape[2] < 1
                or x.dtype.kind != "f"
            ):
                raise ValueError(f"{m} must be float [N,T,D] numeric features")
            self.dims[m] = x.shape[-1]
            length = a.get(m + "_lengths")
            if length is None:
                raise ValueError(
                    f"{m}_lengths is required; padded zeros cannot safely define lengths"
                )
            if (
                length.shape != (self.n,)
                or not np.equal(length, np.floor(length)).all()
                or np.any(length < 0)
                or np.any(length > x.shape[1])
            ):
                raise ValueError(f"Invalid {m}_lengths")
            prefix = np.arange(x.shape[1])[None, :] < length[:, None]
            supplied = a.get(m + "_mask", prefix)
            if supplied.shape != prefix.shape or not np.isin(supplied, [0, 1]).all():
                raise ValueError(f"Invalid {m}_mask")
            valid = supplied.astype(bool) & prefix & self.available[:, mi, None]
            self.available[:, mi] &= valid.any(1)
            a[m + "_valid"] = valid
            if not np.isfinite(x[valid]).all():
                raise ValueError(
                    f"{m}: NaN/Inf at observed tokens; correct preprocessing explicitly"
                )
            ts = a.get(m + "_time")
            if ts is None:
                ts = np.zeros(prefix.shape, np.float32)
                for b, n in enumerate(length.astype(int)):
                    if n:
                        ts[b, :n] = np.linspace(0, 1, n, dtype=np.float32)
            if ts.shape != prefix.shape or not np.isfinite(ts[valid]).all():
                raise ValueError(f"Invalid {m}_time")
            for b in range(self.n):
                times = ts[b, valid[b]]
                if len(times) and (
                    np.any(np.diff(times) < 0) or times.min() < 0 or times.max() > 1
                ):
                    raise ValueError(
                        "Timestamps must be sorted and normalized to a shared [0,1] timeline"
                    )
            a[m + "_time"] = ts.astype(np.float32)
        if self.has_bert:
            content = a["text_valid"]
            attention = a["text_attention_mask"].astype(bool)
            if "text_word_index" not in a and np.any(content & ~attention):
                raise ValueError(
                    "Downstream text mask must be a subset of BERT attention mask"
                )
            if "text_word_index" in a:
                mapping = a["text_word_index"]
                for b in range(self.n):
                    observed = np.unique(mapping[b, mapping[b] >= 0])
                    expected = np.arange(int(content[b].sum()))
                    if not np.array_equal(observed, expected):
                        raise ValueError(
                            "wordpiece map does not cover the downstream word grid"
                        )
        if not self.available.any(1).all():
            raise ValueError("Dataset contains samples without any observed modality")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        a = self.arrays
        x, mask, times = {}, {}, {}
        for m in self.modalities:
            n = max(1, int(a[m + "_lengths"][index]))
            valid = a[m + "_valid"][index, :n]
            values = np.where(valid[:, None], a[m][index, :n], 0).astype(np.float32)
            if self.normalizer is not None:
                values = (values - self.normalizer[m]["mean"]) / self.normalizer[m][
                    "std"
                ]
                values[~valid] = 0
            x[m] = torch.from_numpy(values.copy())
            mask[m] = torch.from_numpy(valid.copy())
            times[m] = torch.from_numpy(a[m + "_time"][index, :n].copy())
        dtype = torch.long if self.task == "classification" else torch.float32
        item = {
            "x": x,
            "mask": mask,
            "time": times,
            "y": torch.tensor(a["y"][index], dtype=dtype),
            "available": torch.from_numpy(self.available[index].copy()),
            "id": str(self.ids[index]),
        }
        if self.has_bert:
            n = max(1, int(a.get("bert_lengths", a["text_lengths"])[index]))
            item["bert"] = {
                "input_ids": torch.from_numpy(
                    a["text_input_ids"][index, :n].astype(np.int64, copy=True)
                ),
                "attention_mask": torch.from_numpy(
                    a["text_attention_mask"][index, :n].astype(np.int64, copy=True)
                ),
                "token_type_ids": torch.from_numpy(
                    a["text_token_type_ids"][index, :n].astype(np.int64, copy=True)
                ),
            }
            if "text_word_index" in a:
                item["bert"]["word_index"] = torch.from_numpy(
                    a["text_word_index"][index, :n].astype(np.int64, copy=True)
                )
        return item


def collate(samples):
    mods = list(samples[0]["x"])
    x, mask, time = {}, {}, {}
    for m in mods:
        x[m] = torch.nn.utils.rnn.pad_sequence(
            [s["x"][m] for s in samples], batch_first=True
        )
        mask[m] = torch.nn.utils.rnn.pad_sequence(
            [s["mask"][m] for s in samples], batch_first=True
        )
        time[m] = torch.nn.utils.rnn.pad_sequence(
            [s["time"][m] for s in samples], batch_first=True
        )
    result = {
        "x": x,
        "mask": mask,
        "time": time,
        "y": torch.stack([s["y"] for s in samples]),
        "available": torch.stack([s["available"] for s in samples]),
        "ids": [s["id"] for s in samples],
    }
    if "bert" in samples[0]:
        if not all("bert" in s for s in samples):
            raise ValueError("Mixed BERT and non-BERT samples in one batch")
        result["bert"] = {
            k: torch.nn.utils.rnn.pad_sequence(
                [s["bert"][k] for s in samples],
                batch_first=True,
                padding_value=-1 if k == "word_index" else 0,
            )
            for k in samples[0]["bert"]
        }
    return result


def to_device(batch, device):
    return {
        k: (
            {m: t.to(device) for m, t in v.items()}
            if isinstance(v, dict)
            else v.to(device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def fit_normalizer(dataset):
    result = {}
    for m in dataset.modalities:
        # Fit only on the training split and observed tokens. Double-precision accumulators.
        sx = np.zeros(dataset.dims[m], np.float64)
        ss = np.zeros_like(sx)
        count = 0
        for b in range(dataset.n):
            x = dataset.arrays[m][b, dataset.arrays[m + "_valid"][b]].astype(np.float64)
            sx += x.sum(0)
            ss += (x * x).sum(0)
            count += len(x)
        if not count:
            raise ValueError(f"No training observations for modality {m}")
        mean = sx / count
        std = np.sqrt(np.maximum(ss / count - mean * mean, 0)).clip(1e-6)
        result[m] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return result


def normalizer_to_state(norm):
    return {
        m: {k: torch.as_tensor(v) for k, v in d.items()}
        for m, d in (norm or {}).items()
    }


def normalizer_from_state(state):
    return {
        m: {k: v.cpu().numpy() for k, v in d.items()} for m, d in state.items()
    } or None


def audit_splits(data_dir, modalities, task, num_classes=None, group_disjoint=False):
    root = Path(data_dir)
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise ValueError(
            "metadata.json is required; record dataset, feature provenance and split origin"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ["dataset", "feature_provenance", "split_origin", "synthetic"]:
        if key not in meta:
            raise ValueError(f"metadata.json missing {key}")
    splits = {
        s: FeatureDataset(root / f"{s}.npz", modalities, task, num_classes=num_classes)
        for s in ("train", "val", "test")
    }
    for s, ds in splits.items():
        if ds.dims != splits["train"].dims:
            raise ValueError("Feature dimensions differ between splits")
    for s, t in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(splits[s].ids) & set(splits[t].ids):
            raise ValueError(f"Sample ID leakage between {s} and {t}")
        if group_disjoint:
            if "groups" not in splits[s].arrays or "groups" not in splits[t].arrays:
                raise ValueError(
                    "Group-disjoint validation requested but group IDs are missing"
                )
            if set(splits[s].arrays["groups"]) & set(splits[t].arrays["groups"]):
                raise ValueError(f"Group leakage between {s} and {t}")
    return splits, {
        "metadata": meta,
        "dimensions": splits["train"].dims,
        "counts": {s: len(d) for s, d in splits.items()},
        "sha256": {s: sha256(root / f"{s}.npz") for s in splits},
        "group_disjoint_checked": group_disjoint,
    }
