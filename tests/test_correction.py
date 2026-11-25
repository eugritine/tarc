"""Readout semantics, label isolation and missing-evidence behavior."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tarc.readouts.correction import (
        SelectiveCategoricalCorrection,
        SelectiveSentimentCorrection,
    )


@unittest.skipUnless(HAS_TORCH, "Install model extras to test the neural readouts")
class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_categorical_mixture_and_missing_inputs(self):
        model = SelectiveCategoricalCorrection(4, 4)
        text, evidence, logits = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
        visible = torch.tensor([[True], [True], [False]])
        extra = torch.tensor([[True], [False], [True]])
        out = model(text, evidence, logits, visible, extra)
        anchor = out["text_logits"].softmax(-1)
        expected = (1 - out["q"]) * anchor + out["q"] * logits.softmax(-1)
        torch.testing.assert_close(out["probabilities"][0], expected[0])
        torch.testing.assert_close(out["probabilities"][1], anchor[1], rtol=0, atol=0)
        torch.testing.assert_close(
            out["probabilities"][2], logits.softmax(-1)[2], rtol=0, atol=0
        )
        self.assertEqual(out["q"][1].item(), 0)
        self.assertEqual(out["q"][2].item(), 0)

    def test_utility_updates_gate_without_moving_targets_or_features(self):
        model = SelectiveCategoricalCorrection(4, 4)
        text = torch.randn(3, 4, requires_grad=True)
        evidence = torch.randn(3, 4, requires_grad=True)
        logits = torch.randn(3, 4, requires_grad=True)
        visible = torch.ones(3, 1, dtype=torch.bool)
        out = model(text, evidence, logits, visible, visible)
        loss = model.utility_loss(out, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertIsNone(text.grad)
        self.assertIsNone(evidence.grad)
        self.assertIsNone(logits.grad)
        self.assertIsNone(model.text_head.weight.grad)
        self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)

    def test_scalar_equation_and_missing_inputs(self):
        model = SelectiveSentimentCorrection(4, 4)
        text, evidence = torch.randn(3, 4), torch.randn(3, 4)
        t, f, a = (
            torch.tensor([[1.0], [-2.0], [0.5]]),
            torch.tensor([[-1.0], [2.0], [-0.5]]),
            torch.randn(3, 1),
        )
        visible = torch.tensor([[True], [True], [False]])
        extra = torch.tensor([[True], [False], [True]])
        out = model(text, evidence, t, f, a, visible, extra)
        residual = (t.abs() + 0.5) * f.tanh()
        magnitude = (1 - out["q_magnitude"]) * out["text_magnitude"] + out[
            "q_magnitude"
        ] * out["fused_magnitude"]
        expected = (t + out["q_sign"] * residual).tanh() * magnitude
        torch.testing.assert_close(out["logits"][:2], expected[:2])
        torch.testing.assert_close(
            out["logits"][1], out["text_prediction"][1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            out["logits"][2], (f.tanh() * out["fused_magnitude"])[2], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
