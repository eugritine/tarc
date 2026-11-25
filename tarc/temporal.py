import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .ops import log_budget


class SparseTemporalScale(nn.Module):
    """Equations (6)-(13), single attention head as written in the manuscript."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.local = nn.Conv1d(
            d, d, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=d
        )
        self.cfg = cfg

    def forward(self, h: Tensor, valid: Tensor):
        # One sequence at a time: query budgets never depend on another sample's padding.
        h = h.masked_fill(~valid[:, None], 0)
        ids = valid.nonzero(as_tuple=True)[0]
        n = len(ids)
        if not n:
            return h, 0
        q, k, v = self.q(h[ids]), self.k(h[ids]), self.v(h[ids])
        if self.cfg.sparse_temporal:
            ns = log_budget(n, self.cfg.sample_factor)
            if self.training:
                sample = torch.randint(n, (n, ns), device=h.device)
            else:
                # Deterministic stratified keys, shifted per query; no full score matrix.
                base = torch.div(
                    torch.arange(ns, device=h.device) * n, ns, rounding_mode="floor"
                )
                sample = (base[None, :] + torch.arange(n, device=h.device)[:, None]) % n
            with torch.no_grad():
                estimates = (q.detach()[:, None, :] * k.detach()[sample]).sum(
                    -1
                ) / h.shape[-1] ** 0.5
                score = estimates.max(-1).values - estimates.mean(-1)
                chosen = score.topk(log_budget(n, self.cfg.ki_factor)).indices
            pairs = n * ns + len(chosen) * n
        else:
            chosen = torch.arange(n, device=h.device)
            pairs = n * n
        attention = torch.softmax(q[chosen] @ k.T / h.shape[-1] ** 0.5, dim=-1)
        selected = attention @ v
        local = h + self.local(h.T[None]).squeeze(0).T
        result = local.index_copy(0, ids[chosen], selected)
        return result.masked_fill(~valid[:, None], 0), pairs


class MultiScaleTemporalEncoder(nn.Module):
    """Modality-specific encoding with fixed integer resampling scales."""

    def __init__(self, input_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Linear(input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(SparseTemporalScale(cfg) for _ in cfg.scales)
        self.scale_score = nn.Linear(cfg.hidden_dim, 1, bias=False)

    def forward(self, x: Tensor, mask: Tensor):
        h = self.projection(x.masked_fill(~mask[:, :, None], 0))
        h = h.masked_fill(~mask[:, :, None], 0)
        outputs, scale_weights = [], []
        pair_count = 0
        full_len = x.shape[1]
        for b in range(len(x)):
            valid_ids = mask[b].nonzero(as_tuple=True)[0]
            if not len(valid_ids):
                outputs.append(h[b] * 0)
                scale_weights.append(h.new_zeros(len(self.cfg.scales)))
                continue
            end = int(valid_ids[-1]) + 1
            hb, mb = h[b, :end], mask[b, :end]
            representations, descriptors = [], []
            for stride, block in zip(self.cfg.scales, self.blocks):
                pad = (-end) % stride
                hp = F.pad(hb, (0, 0, 0, pad)).reshape(-1, stride, hb.shape[-1])
                mp = F.pad(mb, (0, pad)).reshape(-1, stride)
                down = (hp * mp[:, :, None]).sum(1) / mp.sum(1)[:, None].clamp_min(1)
                down_mask = mp.any(1)
                y, pairs = block(down, down_mask)
                pair_count += pairs
                descriptors.append(y.sum(0) / down_mask.sum().clamp_min(1))
                # Interpolate the scale output back to this modality's temporal grid.
                up = F.interpolate(
                    y.T[None], size=end, mode="linear", align_corners=False
                )[0].T
                representations.append(up.masked_fill(~mb[:, None], 0))
            alpha = torch.softmax(
                self.scale_score(torch.stack(descriptors)).squeeze(-1), dim=0
            )
            z = (torch.stack(representations) * alpha[:, None, None]).sum(0)
            outputs.append(F.pad(z, (0, 0, 0, full_len - end)))
            scale_weights.append(alpha)
        return torch.stack(outputs), torch.stack(scale_weights), pair_count
