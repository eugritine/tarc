"""Release integrity and archived statistics, with no model dependencies."""

import json
from pathlib import Path
import statistics
import unittest

from tarc.aggregate import aggregate

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_archived_summary_matches_each_seed(self):
        value = json.loads((ROOT / "results/final_results.json").read_text())
        for dataset, record in value["datasets"].items():
            self.assertEqual(sorted(r["seed"] for r in record["runs"]), [1, 2, 3, 4, 5])
            for key, expected in record["summary_recomputed"].items():
                values = [r["metrics_recomputed"][key] for r in record["runs"]]
                self.assertAlmostEqual(
                    statistics.mean(values), expected["mean"], places=10
                )
                self.assertAlmostEqual(
                    statistics.stdev(values), expected["sample_sd"], places=10
                )

    def test_checkout_and_packaged_configs_are_identical(self):
        for path in (ROOT / "configs").glob("*.json"):
            self.assertEqual(
                json.loads(path.read_text()),
                json.loads((ROOT / "tarc/resources" / path.name).read_text()),
            )

    def test_aggregate_sample_sd_and_reject_incomparable_runs(self):
        runs = [
            {
                "dataset": "iemocap",
                "split": "test",
                "n": 10,
                "seed": s,
                "metrics": {"accuracy": s, "weighted_f1": s},
            }
            for s in [1, 2, 3, 4, 5]
        ]
        value = aggregate(runs)
        self.assertEqual(value["summary"]["accuracy"]["mean"], 3)
        self.assertAlmostEqual(value["summary"]["accuracy"]["sample_sd"], 2.5**0.5)
        with self.assertRaises(ValueError):
            aggregate(runs[:4])
        with self.assertRaises(ValueError):
            aggregate(runs[:4] + [runs[0]])
        with self.assertRaises(ValueError):
            aggregate(runs[:4] + [{**runs[4], "split": "validation"}])
        with self.assertRaises(ValueError):
            aggregate(runs[:4] + [{**runs[4], "n": 9}])

    def test_static_demo_has_empty_private_overlay(self):
        self.assertEqual(
            json.loads((ROOT / "demo/local-texts.json").read_text()), {"datasets": {}}
        )
        self.assertIn("fetch('local-texts.json')", (ROOT / "demo/app.js").read_text())


if __name__ == "__main__":
    unittest.main()
