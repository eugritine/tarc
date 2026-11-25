"""Matched extra expert readout, leaving the original shared h/z in place.

The sign and base paths share parameters. This is a readout allocation, not
independent optimization or an AV-only/causal representation. The original
balance regularizer remains on the base call; no new auxiliary loss is added.
Both enabled roles always pay the extra expert call, including fallback rows.
"""

import torch
from .text_message_pooling import text_message_pool


def sign_readout(
    cfg,
    experts,
    h,
    base_z,
    coupled,
    original,
    masks,
    available,
    expert_rho,
    router_u,
    base_evaluations,
):
    text_index = cfg.modalities.index("text")
    text_visible = available[:, text_index]
    active = text_visible & (available.sum(-1) > 1)
    if cfg.sign_readout_pooling == "text_message":
        branch_h = text_message_pool(
            coupled["text"], original["text"], masks["text"], text_visible, h
        )
    elif cfg.sign_readout_pooling == "full_pool":
        branch_h = h
    else:
        raise ValueError("sign_readout only accepts an enabled pooling role")
    if cfg.use_moe:
        branch = experts(branch_h, expert_rho, router_u * available)
        branch_z = branch["z"]
        evaluated = branch["expert_evaluations"]
        load = branch["expert_load_raw"]
    else:
        branch_z = experts(branch_h)
        evaluated = len(branch_h)
        load = branch_z.new_ones(1)
    if cfg.expert_residual:
        branch_z = branch_h + branch_z
    return {
        "sign_z": torch.where(active[:, None], branch_z, base_z),
        "sign_h": branch_h,
        "sign_branch_active": active,
        "sign_expert_load_raw": load,
        "base_expert_evaluations": base_evaluations,
        "sign_expert_evaluations": evaluated,
        "expert_evaluations": base_evaluations + evaluated,
    }
