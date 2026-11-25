import torch
from torch import nn
import torch.nn.functional as F
from .ops import masked_mean, masked_softmax


class ReliabilityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.means = nn.ModuleDict(
            {m: nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for m in cfg.modalities}
        )
        self.logvars = nn.ModuleDict(
            {m: nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for m in cfg.modalities}
        )
        # A residual correction conditioned on the current visible set. Zero
        # initialization reproduces the original reliability rule initially.
        self.context = None
        if cfg.contextual_reliability:
            width = len(cfg.modalities) * (cfg.hidden_dim + 3)
            # Do not shift initialization of the unchanged coupling/experts/head
            # when comparing a zero-residual candidate with the same base seed.
            with torch.random.fork_rng(devices=[]):
                self.context = nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, cfg.reliability_context_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.reliability_context_hidden, len(cfg.modalities)),
                )
                nn.init.zeros_(self.context[-1].weight)
                nn.init.zeros_(self.context[-1].bias)

    def encode(self, features, masks):
        samples, uncertainties, kl = {}, [], []
        for m in self.cfg.modalities:
            mask = masks[m]
            mu = self.means[m](features[m])
            lv = self.logvars[m](features[m]).clamp(
                self.cfg.logvar_min, self.cfg.logvar_max
            )
            if self.cfg.use_uncertainty:
                z = (
                    mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
                    if self.training
                    else mu
                )
                u = F.softplus(lv).mean(-1)
                k = 0.5 * (mu.square() + lv.exp() - 1 - lv).mean(-1)
                denom = mask.sum(1).clamp_min(1)
                uncertainties.append((u * mask).sum(1) / denom)
                kl.append((k * mask).sum(1) / denom)
            else:
                z = mu
                uncertainties.append(mu.new_zeros(len(mu)))
                kl.append(mu.new_zeros(len(mu)))
            samples[m] = z.masked_fill(~mask[:, :, None], 0)
        descriptors = torch.stack(
            [masked_mean(samples[m], masks[m]) for m in self.cfg.modalities], dim=1
        )
        normalized = F.normalize(descriptors, dim=-1, eps=self.cfg.epsilon)
        cos = normalized @ normalized.transpose(1, 2)
        return {
            "z": samples,
            "u": torch.stack(uncertainties, dim=1),
            "kl": torch.stack(kl, dim=1),
            "cosine": cos,
            "descriptors": descriptors,
        }

    def state(self, encoded, available, mode="full"):
        if not available.any(-1).all():
            raise ValueError("At least one modality must be observed")
        if mode not in {"full", "without_u", "without_agreement", "uniform"}:
            raise ValueError(f"Unknown reliability intervention: {mode}")
        m = len(self.cfg.modalities)
        peers = (
            available[:, None, :]
            & ~torch.eye(m, dtype=torch.bool, device=available.device)[None]
        )
        kappa = (encoded["cosine"] * peers).sum(-1) / peers.sum(-1).clamp_min(1)
        if mode == "full":
            logits = -encoded["u"] / self.cfg.tau_u + self.cfg.lambda_kappa * kappa
        elif mode == "without_u":
            logits = self.cfg.lambda_kappa * kappa
        elif mode == "without_agreement":
            logits = -encoded["u"] / self.cfg.tau_u
        else:
            logits = torch.zeros_like(encoded["u"])
        if self.context is not None and mode != "uniform":
            # Mask before the learned mapping: a missing stream's content,
            # uncertainty and agreement must never influence visible scores.
            visible = available.to(logits.dtype)
            descriptors = encoded["descriptors"] * visible[:, :, None]
            unc = (
                encoded["u"] * visible
                if mode != "without_u"
                else torch.zeros_like(logits)
            )
            agreement = (
                kappa * visible
                if mode != "without_agreement"
                else torch.zeros_like(logits)
            )
            if not self.cfg.reliability_context_content:
                # Control: learn only subset-dependent prior offsets. Keep the
                # original analytic uncertainty/agreement terms above intact.
                descriptors = torch.zeros_like(descriptors)
                unc = torch.zeros_like(unc)
                agreement = torch.zeros_like(agreement)
            features = torch.cat(
                [descriptors.flatten(1), unc, agreement, visible], dim=-1
            )
            correction = self.cfg.reliability_context_bound * torch.tanh(
                self.context(features)
            )
            logits = logits + correction
        return masked_softmax(logits, available), kappa
