"""Utility-gated categorical and polarity/magnitude correction heads.

Labels are used only in the training utility loss, never in inference."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class SelectiveSentimentCorrection(nn.Module):
    def __init__(self, text_dim, fused_dim, max_magnitude=3.0, beta=0.5):
        super().__init__()
        self.max_magnitude = float(max_magnitude)
        self.beta = float(beta)
        self.text_magnitude = nn.Linear(text_dim, 1)
        self.gates = nn.Sequential(
            nn.LayerNorm(text_dim + fused_dim + 1),
            nn.Linear(text_dim + fused_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.constant_(self.gates[-1].bias, math.log(0.1 / 0.9))

    def forward(
        self,
        text_features,
        fused_features,
        text_sign,
        fusion_sign,
        fused_magnitude_logit,
        text_visible,
        evidence_visible,
    ):
        # Gate supervision cannot move encoder features to manufacture its own targets.
        inputs = torch.cat(
            [text_features, fused_features, text_sign.abs()], -1
        ).detach()
        q = torch.sigmoid(self.gates(inputs))
        eligible = (text_visible & evidence_visible).to(q.dtype)
        q_sign, q_magnitude = q[:, :1] * eligible, q[:, 1:] * eligible
        text_magnitude = self.max_magnitude * torch.sigmoid(
            self.text_magnitude(text_features)
        )
        fused_magnitude = self.max_magnitude * torch.sigmoid(fused_magnitude_logit)
        residual = (text_sign.abs() + self.beta) * torch.tanh(fusion_sign)
        corrected_sign = text_sign + residual
        sign = torch.where(text_visible, text_sign + q_sign * residual, fusion_sign)
        magnitude = torch.where(
            text_visible,
            text_magnitude + q_magnitude * (fused_magnitude - text_magnitude),
            fused_magnitude,
        )
        anchor = torch.tanh(text_sign) * text_magnitude
        return dict(
            logits=torch.tanh(sign) * magnitude,
            sign_logit=sign,
            text_prediction=anchor,
            polarity_evidence_prediction=torch.tanh(corrected_sign) * text_magnitude,
            magnitude_evidence_prediction=torch.tanh(text_sign) * fused_magnitude,
            text_sign_logit=text_sign,
            polarity_evidence_logit=corrected_sign,
            q_sign=q_sign,
            q_magnitude=q_magnitude,
            eligible=eligible,
            magnitude=magnitude,
            text_magnitude=text_magnitude,
            fused_magnitude=fused_magnitude,
        )

    @staticmethod
    def utility_loss(out, y, temperature=0.25):
        if temperature <= 0:
            raise ValueError("Positive temperature required")
        y = y.reshape(-1, 1)
        target = (y >= 0).to(y.dtype)
        with torch.no_grad():
            old_sign = F.binary_cross_entropy_with_logits(
                out["text_sign_logit"], target, reduction="none"
            )
            new_sign = F.binary_cross_entropy_with_logits(
                out["polarity_evidence_logit"], target, reduction="none"
            )
            old_mag = (out["text_prediction"] - y).abs()
            new_mag = (out["magnitude_evidence_prediction"] - y).abs()
            target_sign = torch.sigmoid((old_sign - new_sign) / temperature)
            target_magnitude = torch.sigmoid((old_mag - new_mag) / temperature)
        mask = out["eligible"]
        denom = mask.sum().clamp_min(1.0)
        qs = out["q_sign"].clamp(1e-6, 1 - 1e-6)
        qm = out["q_magnitude"].clamp(1e-6, 1 - 1e-6)
        return (
            (
                F.binary_cross_entropy(qs, target_sign, reduction="none")
                + F.binary_cross_entropy(qm, target_magnitude, reduction="none")
            )
            * mask
        ).sum() / denom


class SelectiveCategoricalCorrection(nn.Module):
    def __init__(self, text_dim, fused_dim, classes=4):
        super().__init__()
        self.text_head = nn.Linear(text_dim, classes)
        self.gate = nn.Sequential(
            nn.LayerNorm(text_dim + fused_dim),
            nn.Linear(text_dim + fused_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(0.1 / 0.9))

    def forward(
        self,
        text_features,
        fused_features,
        fused_logits,
        text_visible,
        evidence_visible,
    ):
        text_logits = self.text_head(text_features)
        anchor = text_logits.softmax(-1)
        evidence = fused_logits.softmax(-1)
        q = torch.sigmoid(
            self.gate(torch.cat([text_features, fused_features], -1).detach())
        )
        eligible = (text_visible & evidence_visible).to(q.dtype)
        q = q * eligible
        mixture = anchor + q * (evidence - anchor)
        mixture = torch.where(text_visible, mixture, evidence)
        return dict(
            logits=mixture.clamp_min(1e-8).log(),
            text_logits=text_logits,
            fused_logits=fused_logits,
            q=q,
            eligible=eligible,
            probabilities=mixture,
        )

    @staticmethod
    def utility_loss(out, y, temperature=0.25):
        if temperature <= 0:
            raise ValueError("Positive temperature required")
        with torch.no_grad():
            old = F.cross_entropy(out["text_logits"], y, reduction="none")
            new = F.cross_entropy(out["fused_logits"], y, reduction="none")
            target = torch.sigmoid((old - new) / temperature).unsqueeze(-1)
        mask = out["eligible"]
        loss = F.binary_cross_entropy(
            out["q"].clamp(1e-6, 1 - 1e-6), target, reduction="none"
        )
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
