"""Aggregate matched dataset/split runs using sample standard deviation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def aggregate(records, expected_seeds=(1, 2, 3, 4, 5)):
    if len(records) < 2:
        raise ValueError("At least two runs are required for sample standard deviation")
    expected_seeds = tuple(expected_seeds)
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Expected seeds must be unique")
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != len(seeds) or set(seeds) != set(expected_seeds):
        raise ValueError("Missing, extra or repeated seeds")
    datasets = {r["dataset"] for r in records}
    splits = {r["split"] for r in records}
    sizes = {r["n"] for r in records}
    if len(datasets) != 1 or len(splits) != 1 or len(sizes) != 1:
        raise ValueError("Runs must have the same dataset, split and cohort size")
    dataset = next(iter(datasets))
    if dataset not in {"mosi", "mosei", "iemocap"}:
        raise ValueError("Unknown dataset")
    keys = (
        ["accuracy", "weighted_f1"]
        if dataset == "iemocap"
        else ["acc2_including_zero", "f1_including_zero", "mae"]
    )
    summary = {}
    for key in keys:
        values = [r["metrics"][key] for r in records]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Invalid metric: {key}")
        summary[key] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
    return {
        "dataset": dataset,
        "split": next(iter(splits)),
        "n": next(iter(sizes)),
        "seeds": sorted(seeds),
        "sample_sd_ddof": 1,
        "summary": summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="1,2,3,4,5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(
        [json.loads(p.read_text()) for p in args.inputs],
        tuple(int(v) for v in args.expected_seeds.split(",")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
