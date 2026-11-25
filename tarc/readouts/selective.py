"""Text-anchored prediction with endpoint and utility supervision.

Gate inputs are detached; endpoint and utility losses are combined with
the common evidence-core training objective."""

import torch
from torch import nn
import torch.nn.functional as F

from tarc.ops import masked_mean
from tarc.losses import objective
from tarc.readouts.correction import (
    SelectiveSentimentCorrection,
    SelectiveCategoricalCorrection,
)


class SelectiveTARC(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.cfg = core.cfg
        self.text_index = self.cfg.modalities.index("text")
        self.sentiment = self.cfg.task == "sentiment"
        if self.sentiment:
            dim = core.bert.config.hidden_size * (
                4 if core.text_pooling == "multipool" else 1
            )
            self.correction = SelectiveSentimentCorrection(dim, self.cfg.hidden_dim)
        elif self.cfg.task == "classification":
            self.correction = SelectiveCategoricalCorrection(
                self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.num_classes
            )
        else:
            raise ValueError("Unsupported task")

    @property
    def bert(self):
        return self.core.bert

    def encode(self, batch):
        return self.core.encode(batch)

    def decode(self, encoded, available=None, intervention="R_full"):
        base = self.core.decode(encoded, available, intervention)
        visible = base["available"][:, self.text_index : self.text_index + 1]
        other = [i for i in range(len(self.cfg.modalities)) if i != self.text_index]
        evidence = base["available"][:, other].any(-1, keepdim=True)
        if self.sentiment:
            out = self.correction(
                encoded["text_sign_features"],
                base["z"],
                base["text_sign_logit"],
                base["fusion_sign_logit"],
                self.core.magnitude_head(base["z"]),
                visible,
                evidence,
            )
        else:
            text = masked_mean(
                encoded["primary"]["z"]["text"], encoded["masks"]["text"]
            )
            out = self.correction(text, base["z"], base["logits"], visible, evidence)
        return {**base, **out}

    def forward(self, batch, available=None, intervention="R_full"):
        return self.decode(self.encode(batch), available, intervention)


def selective_loss(
    model,
    full,
    dropped,
    y,
    weights,
    *,
    utility_weight=0.1,
    endpoint_weight=0.25,
    temperature=0.25,
):
    """Only training labels supervise detached utility targets and endpoints.

    Explicit endpoint losses keep both anchor and evidence branches useful even
    when a gate is small. The original consistency/KL/routing terms are retained.
    """
    if utility_weight < 0 or endpoint_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    base, parts = objective(full, dropped, y, model.cfg, weights)
    if model.sentiment:
        target = (y.reshape(-1, 1) >= 0).to(full["logits"].dtype)
        text = full["text_prediction"].reshape(-1)
        polarity = F.binary_cross_entropy_with_logits(full["text_sign_logit"], target)
        polarity = polarity + F.binary_cross_entropy_with_logits(
            full["polarity_evidence_logit"], target
        )
        intensity = F.l1_loss(full["text_magnitude"].reshape(-1), y.abs().reshape(-1))
        intensity = intensity + F.l1_loss(
            full["fused_magnitude"].reshape(-1), y.abs().reshape(-1)
        )
        endpoints = F.l1_loss(text, y.reshape(-1)) + 0.5 * polarity + 0.5 * intensity
        primary = parts["task"] + endpoint_weight * endpoints
    else:
        endpoints = 0.5 * (
            F.cross_entropy(full["text_logits"], y.long())
            + F.cross_entropy(full["fused_logits"], y.long())
        )
        primary = parts["task"] + endpoint_weight * endpoints
    utility = model.correction.utility_loss(full, y, temperature)
    total = base + endpoint_weight * endpoints + utility_weight * utility
    return total, {
        **parts,
        "endpoint": endpoints,
        "utility": utility,
        "encoder_primary_loss": primary,
        "total": total,
    }
