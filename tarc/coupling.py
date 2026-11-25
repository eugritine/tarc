import torch
from torch import nn
from .ops import masked_softmax
from .rho_credit import add_matched_rho_bridge


class IntermodalCoupling(nn.Module):
    """Joint cross-source TopK, Eq. (26)-(33), with explicit candidate generation."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rho_credit_strength = 0.0
        self.rho_credit_coefficient = 1.0
        d = cfg.hidden_dim
        self.q = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in cfg.modalities})
        self.k = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in cfg.modalities})
        self.v = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in cfg.modalities})

    def _candidates(self, tq, ts):
        n = len(ts)
        cap = min(n, self.cfg.candidate_per_source)
        if self.cfg.cross_mode == "dense" or cap == n:
            ids = torch.arange(n, device=ts.device)[None, :].expand(len(tq), -1)
            return ids, torch.ones_like(ids, dtype=torch.bool)
        # Approximation choice: half timestamp-local, half stratified global keys.
        local = max(1, cap // 2)
        global_n = cap - local
        center = torch.searchsorted(ts.contiguous(), tq.contiguous())
        offsets = torch.arange(local, device=ts.device) - local // 2
        ids = (center[:, None] + offsets[None, :]).clamp(0, n - 1)
        if global_n:
            uniform = torch.div(
                torch.arange(global_n, device=ts.device) * n,
                global_n,
                rounding_mode="floor",
            )
            ids = torch.cat([ids, uniform[None, :].expand(len(tq), -1)], dim=-1)
        ids = ids.sort(-1).values
        unique = torch.ones_like(ids, dtype=torch.bool)
        unique[:, 1:] = ids[:, 1:] != ids[:, :-1]
        return ids, unique

    def forward(self, z, masks, times, rho, available):
        cfg = self.cfg
        mods = cfg.modalities
        q = {m: self.q[m](z[m]) for m in mods}
        k = {m: self.k[m](z[m]) for m in mods}
        v = {m: self.v[m](z[m]) for m in mods}
        outputs = {m: [] for m in mods}
        all_mass = []
        score_pairs = 0
        for b in range(len(rho)):
            target_masses = []
            for mi, m in enumerate(mods):
                if not bool(available[b, mi]):
                    outputs[m].append(z[m][b] * 0)
                    target_masses.append(rho.new_zeros(len(mods)))
                    continue
                target_ids = masks[m][b].nonzero(as_tuple=True)[0]
                qb = q[m][b, target_ids]
                tq = times[m][b, target_ids]
                logits, values, valids, source_ids = [], [], [], []
                for ni, n in enumerate(mods):
                    if n == m or (
                        cfg.missing_policy == "exclude" and not bool(available[b, ni])
                    ):
                        continue
                    source_valid = masks[n][b].nonzero(as_tuple=True)[0]
                    if not len(source_valid):
                        continue
                    ts = times[n][b, source_valid]
                    ids, unique = self._candidates(tq, ts)
                    idx = source_valid[ids]
                    score = (qb[:, None, :] * k[n][b, idx]).sum(
                        -1
                    ) / cfg.hidden_dim**0.5
                    score = score + cfg.lambda_r * torch.log(rho[b, ni] + cfg.epsilon)
                    score = (
                        score
                        - cfg.lambda_time
                        * (tq[:, None] - times[n][b, idx]).abs()
                        / cfg.tau_t
                    )
                    logits.append(score)
                    values.append(v[n][b, idx])
                    valids.append(unique)
                    source_ids.append(torch.full_like(ids, ni))
                    score_pairs += score.numel()
                if not logits:
                    outputs[m].append(z[m][b])
                    target_masses.append(rho.new_zeros(len(mods)))
                    continue
                scores = torch.cat(logits, dim=-1)
                vals = torch.cat(values, dim=1)
                valid = torch.cat(valids, dim=-1)
                source = torch.cat(source_ids, dim=-1)
                # Discrete edge selection is detached; retained score gradients are exact.
                credit_inputs = None
                if (
                    self.training
                    and torch.is_grad_enabled()
                    and self.rho_credit_strength > 0
                ):
                    credit_inputs = (scores, vals, valid, source)
                chosen = (
                    scores.detach()
                    .masked_fill(~valid, -torch.inf)
                    .topk(min(cfg.kc, scores.shape[-1]), dim=-1)
                    .indices
                )
                scores = scores.gather(1, chosen)
                valid = valid.gather(1, chosen)
                source = source.gather(1, chosen)
                vals = vals.gather(1, chosen[:, :, None].expand(-1, -1, cfg.hidden_dim))
                weights = masked_softmax(scores / cfg.tau_c, valid)
                message = (weights[:, :, None] * vals).sum(1)
                if credit_inputs is not None:
                    message = add_matched_rho_bridge(
                        message,
                        *credit_inputs,
                        rho[b],
                        strength=self.rho_credit_strength,
                        credit_coefficient=self.rho_credit_coefficient,
                        reliability_weight=cfg.lambda_r,
                        temperature=cfg.tau_c,
                        epsilon=cfg.epsilon,
                    )
                coupled = z[m][b].index_add(0, target_ids, message)
                outputs[m].append(coupled)
                mass = weights.new_zeros(len(qb), len(mods)).scatter_add(
                    1, source, weights
                )
                target_masses.append(mass.mean(0))
            all_mass.append(torch.stack(target_masses))
        return (
            {m: torch.stack(outputs[m]) for m in mods},
            torch.stack(all_mass),
            score_pairs,
        )
