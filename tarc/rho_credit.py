"""Training-only rho gradient bridge using the known straight-through principle.

Forward values and hard selected-score gradients are unchanged. Only rho
receives an additional dense-candidate surrogate derivative. This estimator
is biased and does not differentiate the discrete TopK objective exactly.
Disabled by default; enabled only for training with gradients.
"""

import math
import torch


def add_rho_credit_bridge(
    message,
    scores,
    values,
    valid,
    sources,
    rho,
    *,
    strength=1.0,
    reliability_weight=1.0,
    temperature=1.0,
    epsilon=1e-8,
):
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("Bridge strength must be finite nonnegative")
    if strength == 0:
        return message
    if (
        scores.ndim != 2
        or values.ndim != 3
        or values.shape[:2] != scores.shape
        or message.shape != (len(scores), values.shape[-1])
        or valid.shape != scores.shape
        or valid.dtype != torch.bool
        or sources.shape != scores.shape
        or sources.dtype != torch.long
        or rho.ndim != 1
        or rho.numel() < 1
        or len(scores) == 0
    ):
        raise ValueError("Incompatible candidate/message/rho shapes")
    if not bool(valid.any(-1).all()):
        raise ValueError("Need at least one valid candidate per query")
    for name, x in [("temperature", temperature), ("epsilon", epsilon)]:
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError(name + " must be finite positive")
    if (
        type(reliability_weight) not in (int, float)
        or not math.isfinite(reliability_weight)
        or reliability_weight < 0
    ):
        raise ValueError("Reliability weight must be finite nonnegative")
    if not bool(torch.isfinite(rho.detach()).all()) or bool((rho.detach() < 0).any()):
        raise ValueError("Invalidrho")
    if bool((sources < 0).any()) or bool((sources >= len(rho)).any()):
        raise ValueError("Candidate source outside rho")
    # Stop content/time scores and all value/encoder gradients in this branch.
    # Reintroduce only the reliability log-prior derivative. At current rho,
    # scores equal the ordinary scores up to a zero-valued algebraic term.
    prior = reliability_weight * torch.log(rho[sources] + epsilon)
    surrogate_scores = scores.detach() + (prior - prior.detach())
    weights = torch.softmax(
        (surrogate_scores / temperature).masked_fill(~valid, -torch.inf), dim=-1
    )
    dense_message = (weights[:, :, None] * values.detach()).sum(1)
    # Subtract before adding to preserve exact forward values, including scale.
    return message + strength * (dense_message - dense_message.detach())


def add_matched_rho_bridge(
    message,
    scores,
    values,
    valid,
    sources,
    rho,
    *,
    credit_coefficient=1.0,
    strength=1.0,
    **kwargs,
):
    if type(credit_coefficient) not in (int, float) or credit_coefficient not in (
        0.0,
        1.0,
    ):
        raise ValueError("Matched credit coefficient must be0 or1")
    # Preserve the old lazy disabled path separately from the compute control.
    if strength == 0:
        return add_rho_credit_bridge(
            message, scores, values, valid, sources, rho, strength=strength, **kwargs
        )
    # Both choices keep the same eager autograd graph and exact rho values.
    # Zero sends a zero derivative through rho, rather than skipping the work.
    proxy = rho.detach() + credit_coefficient * (rho - rho.detach())
    return add_rho_credit_bridge(
        message, scores, values, valid, sources, proxy, strength=strength, **kwargs
    )


def rho_credit_settings(settings):
    recipe = settings.get("rho_credit")
    if recipe is None:
        return None
    if not isinstance(recipe, dict) or set(recipe) - {"strength", "credit_coefficient"}:
        raise ValueError("rho_credit must contain only strength and credit_coefficient")
    strength = recipe.get("strength", 0.0)
    coefficient = recipe.get("credit_coefficient", 1.0)
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("Rho credit strength must be finite nonnegative")
    if type(coefficient) not in (int, float) or coefficient not in (0.0, 1.0):
        raise ValueError("Rho credit coefficient must be0 or1")
    if strength == 0:
        return None
    if set(recipe) != {"strength", "credit_coefficient"}:
        raise ValueError(
            "Positive rho credit requires explicit matched control coefficient"
        )
    return recipe


def configure_rho_credit(model, settings):
    recipe = rho_credit_settings(settings)
    if recipe is None:
        return model
    cfg = model.cfg
    if (
        cfg.task != "sentiment"
        or not cfg.contextual_reliability
        or cfg.separate_rho
        or not cfg.use_moe
        or not cfg.expert_reliability
        or cfg.lambda_r <= 0
    ):
        raise ValueError(
            "Rho credit requires sentiment shared contextual token/expert reliability"
        )
    model.coupling.rho_credit_strength = float(recipe["strength"])
    model.coupling.rho_credit_coefficient = float(recipe["credit_coefficient"])
    return model
