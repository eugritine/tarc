"""Check the original fifteen test caches and recompute the reported main table.

Requires locally held original prediction caches, not included in this repository.
File SHA-256 is checked before reading labels or calculating metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def recompute(sentiment_dir, emotion_dir, manifest_path=None, results_path=None):
    from tarc.evaluate import evaluate_file
    from tarc.aggregate import aggregate

    manifest = json.loads(
        Path(manifest_path or ROOT / "assets/experiment_manifest.json").read_text()
    )
    expected = json.loads(
        Path(results_path or ROOT / "results/final_results.json").read_text()
    )
    report = {
        "mode": "original_prediction_cache_recomputation",
        "datasets": {},
        "new_training_runs": 0,
    }
    for dataset in ("mosi", "mosei", "iemocap"):
        runs = []
        for source in manifest["datasets"][dataset]["runs"]:
            seed = source["seed"]
            path = (
                Path(emotion_dir) / f"seed_{seed}.npz"
                if dataset == "iemocap"
                else Path(sentiment_dir) / f"{dataset}_seed_{seed}.npz"
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source["test_prediction_sha256"]:
                raise ValueError(
                    f"Original prediction identity mismatch: {dataset}/seed_{seed}"
                )
            result = evaluate_file(path, dataset)
            if result["n"] != manifest["datasets"][dataset]["counts"]["test"]:
                raise ValueError("Test cohort size mismatch")
            archived = next(
                r for r in expected["datasets"][dataset]["runs"] if r["seed"] == seed
            )
            for metric, value in archived["metrics_recomputed"].items():
                if abs(result["metrics"][metric] - value) > 1e-8:
                    raise ValueError(f"Metric mismatch: {dataset}/seed_{seed}/{metric}")
            result.update(
                seed=seed, split="test", checkpoint_sha256=source["checkpoint_sha256"]
            )
            runs.append(result)
        report["datasets"][dataset] = {"runs": runs, "aggregate": aggregate(runs)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentiment-dir",
        required=True,
        type=Path,
        help="Folder with mosi_seed_1.npz ... mosei_seed_5.npz",
    )
    parser.add_argument(
        "--emotion-dir",
        required=True,
        type=Path,
        help="Folder with IEMOCAP seed_1.npz ... seed_5.npz",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = recompute(args.sentiment_dir, args.emotion_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Verified and recomputed all 15 original test prediction caches.")


if __name__ == "__main__":
    main()
