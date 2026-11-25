import torch
from torch import nn
from .config import ModelConfig
from .temporal import MultiScaleTemporalEncoder
from .reliability import ReliabilityHead
from .coupling import IntermodalCoupling
from .experts import SparseExperts, GatedExpert
from .ops import masked_mean
from .text_message_pooling import text_message_pool
from .sign_readout import sign_readout


class TARC(nn.Module):
    """TARC multiscale latent encoding, source assessment and sparse evidence core."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoders = nn.ModuleDict(
            {
                m: MultiScaleTemporalEncoder(cfg.input_dims[m], cfg)
                for m in cfg.modalities
            }
        )
        self.reliability = ReliabilityHead(cfg)
        self.second_reliability = ReliabilityHead(cfg) if cfg.separate_rho else None
        self.missing_prototypes = nn.Parameter(
            torch.empty(len(cfg.modalities), cfg.hidden_dim)
        )
        nn.init.normal_(self.missing_prototypes, std=0.02)
        self.coupling = IntermodalCoupling(cfg)
        self.experts = (
            SparseExperts(cfg)
            if cfg.use_moe
            else GatedExpert(cfg.hidden_dim, cfg.expert_dim)
        )
        self.head = nn.Linear(
            cfg.hidden_dim, cfg.num_classes if cfg.task == "classification" else 1
        )

    def encode(self, batch):
        features, alphas = {}, {}
        pair_count = 0
        masks = {
            m: batch["mask"][m] & batch["available"][:, i, None]
            for i, m in enumerate(self.cfg.modalities)
        }
        for m in self.cfg.modalities:
            features[m], alphas[m], pairs = self.encoders[m](batch["x"][m], masks[m])
            pair_count += pairs
        main = self.reliability.encode(features, masks)
        second = (
            self.second_reliability.encode(features, masks)
            if self.second_reliability is not None
            else None
        )
        return {
            "primary": main,
            "second": second,
            "masks": masks,
            "times": batch["time"],
            "available": batch["available"],
            "scale_weights": alphas,
            "temporal_score_pairs": pair_count,
        }

    def decode(self, encoded, available=None, intervention="R_full"):
        cfg = self.cfg
        intervention_modes = {
            "R_full": ("full", False),
            "R_without_u_in_rho": ("without_u", False),
            "R_without_agreement_in_rho": ("without_agreement", False),
            "R_uniform": ("uniform", False),
            "R_without_router_u": ("full", True),
        }
        if intervention not in intervention_modes:
            raise ValueError(f"Unknown inference intervention: {intervention}")
        rho_mode, zero_router_u = intervention_modes[intervention]
        avail = (
            encoded["available"]
            if available is None
            else available & encoded["available"]
        )
        if not avail.any(-1).all():
            raise ValueError(
                "No-observation samples are unsupported; at least one modality is required"
            )
        primary = encoded["primary"]
        rho, kappa = self.reliability.state(primary, avail, rho_mode)
        if self.second_reliability is None:
            erho, eu = rho, primary["u"]
        else:
            erho, _ = self.second_reliability.state(encoded["second"], avail, rho_mode)
            eu = encoded["second"]["u"]
        router_u = torch.zeros_like(eu) if zero_router_u else eu
        zhat, masks, times = {}, {}, {}
        for i, m in enumerate(cfg.modalities):
            z = primary["z"][m]
            prototype = self.missing_prototypes[i].reshape(1, 1, -1)
            zhat[m] = torch.where(avail[:, i, None, None], z, prototype)
            masks[m] = encoded["masks"][m].clone()
            times[m] = encoded["times"][m].clone()
            # A dropped stream must not reveal its original length/timestamps.
            # Missing modalities are represented by exactly one prototype token.
            missing = ~avail[:, i]
            masks[m][missing] = False
            times[m][missing] = 0
            empty = ~masks[m].any(-1)
            masks[m][empty, 0] = True
            times[m][empty, 0] = 0.5
        coupled, mass, cross_pairs = self.coupling(zhat, masks, times, rho, avail)
        pooled = torch.stack(
            [masked_mean(coupled[m], masks[m]) for m in cfg.modalities], dim=1
        )
        h = (pooled * rho[:, :, None]).sum(1)
        if cfg.text_message_pooling:
            h = text_message_pool(
                coupled["text"],
                zhat["text"],
                masks["text"],
                avail[:, cfg.modalities.index("text")],
                h,
            )
        if cfg.use_moe:
            result = self.experts(h, erho, router_u * avail)
        else:
            zz = self.experts(h)
            ones = zz.new_ones(len(zz), 1)
            zero = zz.sum() * 0
            result = {
                "z": zz,
                "expert_prob": ones,
                "expert_gate": ones,
                "expert_selected": ones,
                "expert_load_raw": ones.mean(0),
                "expert_load_normalized": ones.mean(0),
                "balance": zero,
                "balance_raw": zero,
                "balance_normalized": zero,
                "expert_evaluations": len(zz),
            }
        if cfg.expert_residual:
            # Preserve the reliability-weighted coupled representation alongside
            # the expert transformation. No new parameters or routing decisions.
            result = {**result, "z": h + result["z"]}
        if cfg.sign_readout_pooling != "legacy":
            result = {
                **result,
                **sign_readout(
                    cfg,
                    self.experts,
                    h,
                    result["z"],
                    coupled,
                    zhat,
                    masks,
                    avail,
                    erho,
                    router_u,
                    result["expert_evaluations"],
                ),
            }
        denom = encoded["available"].sum().clamp_min(1)
        kl = (primary["kl"] * encoded["available"]).sum() / denom
        if encoded["second"] is not None:
            kl = 0.5 * (
                kl + (encoded["second"]["kl"] * encoded["available"]).sum() / denom
            )
        readout = (
            result["z"] if cfg.sign_readout_pooling == "legacy" else result["sign_z"]
        )
        return {
            **result,
            "logits": self.head(readout),
            "rho": rho,
            "kappa": kappa,
            "u": primary["u"] * avail,
            "expert_rho": erho,
            "router_u": router_u * avail,
            "available": avail,
            "attention_mass": mass,
            "kl": kl,
            "h": h,
            "scale_weights": encoded["scale_weights"],
            "temporal_score_pairs": encoded["temporal_score_pairs"],
            "cross_score_pairs": cross_pairs,
        }

    def forward(self, batch, available=None, intervention="R_full"):
        return self.decode(self.encode(batch), available, intervention)
