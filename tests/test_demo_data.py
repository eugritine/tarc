"""Independent numerical acceptance checks; runs without model dependencies."""

import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "demo/data.json").read_text(encoding="utf-8"))

    def test_complete_cohorts_and_labels(self):
        self.assertEqual(self.data["mode"], "archived_model_outputs")
        self.assertEqual(self.data["total_records"], 3131)
        for key, count, epoch in [
            ("mosi", 229, 7),
            ("mosei", 1871, 2),
            ("iemocap", 1031, 20),
        ]:
            d = self.data["datasets"][key]
            self.assertEqual(
                (len(d["samples"]), d["n"], d["seed"], d["epoch"], d["split"]),
                (count, count, 3, epoch, "validation"),
            )
            self.assertEqual(len({s["id"] for s in d["samples"]}), count)
        self.assertEqual(
            self.data["datasets"]["iemocap"]["class_names"],
            ["happy_or_excited", "sad", "neutral", "angry"],
        )

    def test_categorical_probability_identity_and_outcome_labels(self):
        counts = dict.fromkeys(["helpful", "kept", "harmful", "other"], 0)
        for s in self.data["datasets"]["iemocap"]["samples"]:
            for k in ["text_prob", "evidence_prob", "corrected_prob"]:
                self.assertEqual(len(s[k]), 4)
                self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in s[k]))
                self.assertAlmostEqual(sum(s[k]), 1, places=6)
            self.assertTrue(0 <= s["q"] <= 1)
            mixture = [
                (1 - s["q"]) * t + s["q"] * f
                for t, f in zip(s["text_prob"], s["evidence_prob"])
            ]
            self.assertLess(
                max(abs(a - b) for a, b in zip(mixture, s["corrected_prob"])), 1e-6
            )
            old = max(range(4), key=s["text_prob"].__getitem__) == s["y"]
            new = max(range(4), key=s["corrected_prob"].__getitem__) == s["y"]
            outcome = (
                "kept"
                if old and new
                else "harmful"
                if old
                else "helpful"
                if new
                else "other"
            )
            self.assertEqual(s["scenario"], outcome)
            counts[outcome] += 1
        self.assertEqual(
            counts, {"helpful": 83, "kept": 708, "harmful": 7, "other": 233}
        )

    def test_scalar_readout_and_error_direction(self):
        for dataset in ["mosi", "mosei"]:
            for s in self.data["datasets"][dataset]["samples"]:
                r = s["readout_state"]
                residual = (abs(r["text_sign_logit"]) + r["beta"]) * math.tanh(
                    r["fusion_sign_logit"]
                )
                polarity = r["text_sign_logit"] + s["q_sign"] * residual
                magnitude = (1 - s["q_magnitude"]) * r["text_magnitude"] + s[
                    "q_magnitude"
                ] * r["fused_magnitude"]
                self.assertLess(
                    abs(math.tanh(polarity) * magnitude - s["corrected"]), 1e-6
                )
                self.assertLess(
                    abs(
                        math.tanh(r["text_sign_logit"]) * r["text_magnitude"]
                        - s["text"]
                    ),
                    1e-6,
                )
                gain = abs(s["text"] - s["y"]) - abs(s["corrected"] - s["y"])
                outcome = (
                    "helpful" if gain > 1e-6 else "harmful" if gain < -1e-6 else "kept"
                )
                self.assertEqual(s["scenario"], outcome)
        self.assertTrue(
            all(s["rho"] is None for s in self.data["datasets"]["mosi"]["samples"])
        )


if __name__ == "__main__":
    unittest.main()
