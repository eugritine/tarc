"""Evaluate a fixed prediction file without training or selecting a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def evaluate_file(path, dataset):
    import numpy as np
    from .metrics import compute_metrics

    path = Path(path)
    if dataset not in {"mosi", "mosei", "iemocap"}:
        raise ValueError("Unknown dataset")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            y = np.asarray(data["y"]).reshape(-1)
            logits = np.asarray(data["logits"])
            ids = np.asarray(data["ids"]).reshape(-1).tolist()
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") != "live_model_inference" or data.get("dataset") != dataset:
            raise ValueError(
                "Expected matching tarc.predict JSON or an NPZ with y/logits/ids"
            )
        rows = data["samples"]
        y = np.asarray([row["y"] for row in rows]).reshape(-1)
        ids = [row["id"] for row in rows]
        if dataset == "iemocap":
            probabilities = np.asarray(
                [row["probabilities"] for row in rows], dtype=float
            )
            if (
                probabilities.shape != (len(y), 4)
                or not np.isfinite(probabilities).all()
            ):
                raise ValueError("Expected four finite class probabilities per sample")
            if np.any(probabilities < 0) or not np.allclose(
                probabilities.sum(1), 1, atol=1e-6
            ):
                raise ValueError("Invalid class probabilities")
            logits = np.log(np.maximum(probabilities, 1e-12))
        else:
            logits = np.asarray([row["prediction"] for row in rows])
    if not len(y) or len(ids) != len(y) or len(set(map(str, ids))) != len(y):
        raise ValueError("Sample IDs must be unique and aligned with nonempty targets")
    if not np.isfinite(y).all() or not np.isfinite(logits).all():
        raise ValueError("Targets and predictions must be finite")
    if dataset == "iemocap":
        if (
            logits.shape != (len(y), 4)
            or np.any(y != y.astype(int))
            or np.any((y < 0) | (y >= 4))
        ):
            raise ValueError(
                "IEMOCAP requires [N,4] logits and integer labels in [0,3]"
            )
        y = y.astype(int)
        task = "classification"
    else:
        if logits.shape not in {(len(y),), (len(y), 1)}:
            raise ValueError("Sentiment predictions must have shape [N] or [N,1]")
        logits = logits.reshape(-1)
        task = "sentiment"
    return {
        "dataset": dataset,
        "n": len(y),
        "metrics": compute_metrics(y, logits, task),
        "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metric_convention": "Percent accuracy/F1; sentiment y=0 is nonnegative; IEMOCAP support-weighted F1",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=["mosi", "mosei", "iemocap"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_file(args.predictions, args.dataset)
    result.update(seed=args.seed, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
