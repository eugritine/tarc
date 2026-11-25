"""Metric conventions and malformed-prediction rejection."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_METRICS = all(
    importlib.util.find_spec(n) is not None for n in ("numpy", "scipy", "sklearn")
)
if HAS_METRICS:
    import numpy as np
    from tarc.evaluate import evaluate_file


@unittest.skipUnless(HAS_METRICS, "Install model extras to test prediction metrics")
class EvaluationTests(unittest.TestCase):
    def test_zero_is_nonnegative_and_targets_are_flattened(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.npz"
            np.savez(
                path,
                ids=np.array(["a", "b", "c"]),
                y=np.array([[-1.0], [0.0], [1.0]]),
                logits=np.array([[-2.0], [0.0], [-0.1]]),
            )
            out = evaluate_file(path, "mosi")
            self.assertEqual(out["n"], 3)
            self.assertAlmostEqual(out["metrics"]["acc2_including_zero"], 200 / 3)
            self.assertAlmostEqual(out["metrics"]["mae"], 0.7)

    def test_duplicate_ids_and_bad_class_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.npz"
            np.savez(
                path,
                ids=np.array(["a", "a"]),
                y=np.array([0, 1]),
                logits=np.ones((2, 4)),
            )
            with self.assertRaises(ValueError):
                evaluate_file(path, "iemocap")
            np.savez(
                path,
                ids=np.array(["a", "b"]),
                y=np.array([0, 1]),
                logits=np.ones((2, 3)),
            )
            with self.assertRaises(ValueError):
                evaluate_file(path, "iemocap")


if __name__ == "__main__":
    unittest.main()
