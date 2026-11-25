import math
import torch
from torch import Tensor


def log_budget(n: int, factor: float) -> int:
    """Natural log; rounded upward, clamped to the number of valid tokens."""
    return min(n, max(1, math.ceil(factor * math.log(max(n, 2))))) if n else 0


def masked_softmax(logits: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """All-masked rows return zeros, never NaNs."""
    valid = valid.expand_as(logits)
    z = logits.masked_fill(~valid, -torch.inf)
    peak = z.amax(dim=dim, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(z - peak).masked_fill(~valid, 0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1)


def nonempty_dropout(available: Tensor, probability: float, generator=None) -> Tensor:
    """Independent Bernoulli dropout conditioned on at least one visible modality."""
    if not 0 <= probability < 1:
        raise ValueError(
            "Dropout probability must lie in [0,1); use named subsets for one-modality evaluation"
        )
    if not available.any(dim=-1).all():
        raise ValueError("Cannot drop modalities from an already empty sample")
    if probability == 0:
        return available.clone()
    keep = (
        torch.rand(available.shape, device=available.device, generator=generator)
        >= probability
    ) & available
    bad = ~keep.any(dim=-1)
    while bad.any():
        count = int(bad.sum())
        draws = torch.rand(
            (count, available.shape[1]), device=available.device, generator=generator
        )
        keep[bad] = (draws >= probability) & available[bad]
        bad = ~keep.any(dim=-1)
    return keep
