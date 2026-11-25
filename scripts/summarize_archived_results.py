"""Recompute the archived five-seed summaries using only the standard library."""

import json
from pathlib import Path
import statistics


def main():
    results = json.loads(
        (Path(__file__).resolve().parents[1] / "results/final_results.json").read_text()
    )
    for dataset, record in results["datasets"].items():
        summary = {}
        for metric, expected in record["summary_recomputed"].items():
            values = [run["metrics_recomputed"][metric] for run in record["runs"]]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            if (
                abs(mean - expected["mean"]) > 1e-10
                or abs(sd - expected["sample_sd"]) > 1e-10
            ):
                raise ValueError(f"Archived summary mismatch: {dataset}/{metric}")
            summary[metric] = {"mean": mean, "sample_sd": sd}
        print(
            json.dumps(
                {"dataset": dataset, "seeds": results["seeds"], "summary": summary}
            )
        )


if __name__ == "__main__":
    main()
