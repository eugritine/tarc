import torch
import torch.nn.functional as F


def task_loss(logits, y, task, regression_loss="mae"):
    if task == "classification":
        return F.cross_entropy(logits, y.long())
    if regression_loss == "mae":
        return F.l1_loss(logits.squeeze(-1), y.float())
    if regression_loss == "mse":
        return F.mse_loss(logits.squeeze(-1), y.float())
    raise ValueError("regression_loss must be mae or mse")


def sentiment_sign_loss(
    logits, y, zero_policy="including", boundary_scale=0.0, boundary_floor=0.25
):
    """Binary sign loss on the existing sentiment regression output.

    No extra classification head is introduced.  Optional weights reduce the
    influence of labels close to zero without changing dataset membership.
    """
    if zero_policy not in {"including", "excluding"}:
        raise ValueError("sign_zero_policy must be including or excluding")
    if boundary_scale < 0 or not 0 <= boundary_floor <= 1:
        raise ValueError("Invalid sign boundary weighting")
    score = logits.squeeze(-1)
    target = (y >= 0).to(score.dtype)
    keep = (
        torch.ones_like(y, dtype=torch.bool) if zero_policy == "including" else y != 0
    )
    if not keep.any():
        return score.sum() * 0
    raw = F.binary_cross_entropy_with_logits(
        score[keep], target[keep], reduction="none"
    )
    if boundary_scale > 0:
        strength = (y[keep].abs() / boundary_scale).clamp(max=1).to(raw.dtype)
        weight = boundary_floor + (1 - boundary_floor) * strength
        return (raw * weight).sum() / weight.sum().clamp_min(torch.finfo(raw.dtype).eps)
    return raw.mean()


def consistency_losses(full, dropped, tau_d, task, epsilon=1e-8, sample_weight=None):
    teacher_z = full["z"].detach()
    con_per_sample = 1 - F.cosine_similarity(
        teacher_z, dropped["z"], dim=-1, eps=epsilon
    )
    if task == "classification":
        teacher = F.softmax(full["logits"].detach() / tau_d, dim=-1)
        kd_per_sample = tau_d**2 * F.kl_div(
            F.log_softmax(dropped["logits"] / tau_d, dim=-1), teacher, reduction="none"
        ).sum(-1)
    else:
        kd_per_sample = (
            (dropped["logits"] - full["logits"].detach()).square().flatten(1).mean(-1)
        )
    if sample_weight is None:
        sample_weight = torch.ones_like(con_per_sample)
    con = (con_per_sample * sample_weight).mean()
    kd = (kd_per_sample * sample_weight).mean()
    reference = full["rho"].detach() * dropped["available"]
    reference = reference / reference.sum(-1, keepdim=True).clamp_min(epsilon)
    rho = ((dropped["rho"] - reference).abs() * dropped["available"]).sum(-1).mean()
    return con, kd, rho


def distillation_weights(full, dropped, y, task, mode="none", temperature=0.25):
    """Training-only teacher advantage; labels never enter model inference."""
    if mode not in {"none", "advantage"}:
        raise ValueError("teacher_filter must be none or advantage")
    if temperature <= 0:
        raise ValueError("teacher_filter_temperature must be positive")
    with torch.no_grad():
        if mode == "none":
            return torch.ones_like(y, dtype=full["logits"].dtype)
        if task == "classification":
            teacher_error = F.cross_entropy(full["logits"], y.long(), reduction="none")
            student_error = F.cross_entropy(
                dropped["logits"], y.long(), reduction="none"
            )
        else:
            teacher_error = (full["logits"].squeeze(-1) - y).abs()
            student_error = (dropped["logits"].squeeze(-1) - y).abs()
        return torch.sigmoid((student_error - teacher_error) / temperature)


def objective(full, dropped, y, cfg, weights, full_task_logits=None):
    zero = full["logits"].sum() * 0
    task = task_loss(
        full["logits"] if full_task_logits is None else full_task_logits,
        y,
        cfg.task,
        weights.get("regression", "mae"),
    )
    sign = (
        sentiment_sign_loss(
            full["logits"],
            y,
            weights.get("sign_zero_policy", "including"),
            weights.get("sign_boundary_scale", 0.0),
            weights.get("sign_boundary_floor", 0.25),
        )
        if cfg.task == "sentiment"
        else zero
    )
    if dropped is None:
        con, kd, rho, student = zero, zero, zero, zero
        teacher_weight = zero + 1
        balance = full["balance"]
    else:
        filter_mode = weights.get("teacher_filter", "none")
        if filter_mode != "none" and weights.get("student_task", 0.0) <= 0:
            raise ValueError(
                "Filtered distillation requires positive student_task supervision"
            )
        sample_weight = distillation_weights(
            full,
            dropped,
            y,
            cfg.task,
            filter_mode,
            weights.get("teacher_filter_temperature", 0.25),
        )
        teacher_weight = sample_weight.mean()
        con, kd, rho = consistency_losses(
            full, dropped, cfg.tau_d, cfg.task, cfg.epsilon, sample_weight
        )
        student = task_loss(
            dropped["logits"], y, cfg.task, weights.get("regression", "mae")
        )
        balance = (full["balance"] + dropped["balance"]) / 2
    total = (
        task
        + weights.get("sign", 0.0) * sign
        + weights.get("student_task", 0.0) * student
        + weights.get("unc", 1e-4) * full["kl"]
        + weights.get("bal", 0.01) * balance
        + weights.get("con", 0.1) * con
        + weights.get("kd", 0.1) * kd
        + weights.get("rho", 0.1) * rho
    )
    return total, {
        "total": total,
        "task": task,
        "sign": sign,
        "student_task": student,
        "unc": full["kl"],
        "bal": balance,
        "con": con,
        "kd": kd,
        "rho": rho,
        "teacher_weight": teacher_weight,
    }
