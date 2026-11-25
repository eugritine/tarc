import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def ece(prob, y, bins=15):
    conf = prob.max(1)
    correct = prob.argmax(1) == y
    result = 0.0
    for lo, hi in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]
    ):
        chosen = (conf > lo) & (conf <= hi)
        if chosen.any():
            result += chosen.mean() * abs(correct[chosen].mean() - conf[chosen].mean())
    return float(result)


def compute_metrics(y, logits, task):
    y = np.asarray(y)
    logits = np.asarray(logits)
    if not len(y) or not np.isfinite(logits).all():
        raise ValueError("Invalid predictions")
    if task == "classification":
        p = softmax(logits, axis=-1)
        pred = p.argmax(1)
        out = {
            "accuracy": 100 * accuracy_score(y, pred),
            "weighted_f1": 100 * f1_score(y, pred, average="weighted", zero_division=0),
            "macro_f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
            "ece": ece(p, y),
            "nll": float(
                -np.log(np.maximum(p[np.arange(len(y)), y.astype(int)], 1e-12)).mean()
            ),
        }
        try:
            out["auroc"] = float(
                roc_auc_score(y, p[:, 1])
                if p.shape[1] == 2
                else roc_auc_score(
                    y,
                    p,
                    average="weighted",
                    multi_class="ovr",
                    labels=np.arange(p.shape[1]),
                )
            )
        except ValueError:
            out["auroc"] = None
        return out
    pred = logits.reshape(-1)
    out = {
        "mae": float(np.abs(pred - y).mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pearson": float(np.corrcoef(y, pred)[0, 1])
        if y.std() > 0 and pred.std() > 0
        else None,
    }
    if task == "sentiment":
        # Report both conventions; don't quietly choose the higher score.
        out["acc2_including_zero"] = 100 * accuracy_score(y >= 0, pred >= 0)
        out["f1_including_zero"] = 100 * f1_score(
            y >= 0, pred >= 0, average="weighted", zero_division=0
        )
        nz = y != 0
        out["acc2_excluding_zero"] = (
            100 * accuracy_score(y[nz] > 0, pred[nz] > 0) if nz.any() else None
        )
        out["f1_excluding_zero"] = (
            100 * f1_score(y[nz] > 0, pred[nz] > 0, average="weighted", zero_division=0)
            if nz.any()
            else None
        )
    return out


def normalized_aurc(rates, scores):
    r, p = np.asarray(rates, float), np.asarray(scores, float)
    if (
        r.ndim != 1
        or r.shape != p.shape
        or len(r) < 2
        or r[0] != 0
        or np.any(np.diff(r) <= 0)
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "AURC requires finite scores, strictly increasing rates, and a rate-zero endpoint"
        )
    return float(np.sum((p[:-1] + p[1:]) * (r[1:] - r[:-1]) / 2) / r[-1])


def paired_statistics(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or len(a) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "Paired arrays must have the same shape with at least two finite observations"
        )
    delta = a - b
    sd = delta.std(ddof=1)
    n = len(delta)
    mean = delta.mean()
    if sd > 0:
        t = mean / (sd / np.sqrt(n))
        p = 2 * stats.t.sf(abs(t), n - 1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        dz = mean / sd
        ci = [float(mean - half), float(mean + half)]
    else:
        t = p = dz = None
        ci = [float(mean), float(mean)]
    d = np.round(delta, 12)
    if np.all(d == 0):
        wp = 1.0
        wplus = wminus = 0.0
        rbc = 0.0
    else:
        nz = d[d != 0]
        ranks = stats.rankdata(np.abs(nz))
        wplus = float(ranks[nz > 0].sum())
        wminus = float(ranks[nz < 0].sum())
        rbc = (wplus - wminus) / (wplus + wminus)
        # Exact sign enumeration handles absolute-value ties for small paired samples.
        if len(nz) <= 20:
            sums = np.array([0.0])
            for rank in ranks:
                sums = np.r_[sums, sums + rank]
            center = ranks.sum() / 2
            wp = float(np.mean(np.abs(sums - center) >= abs(wplus - center) - 1e-10))
        else:
            wp = float(stats.wilcoxon(d, method="approx").pvalue)
    return {
        "n": n,
        "mean_difference": float(mean),
        "difference_sample_std": float(sd),
        "t": None if t is None else float(t),
        "df": n - 1,
        "p_two_sided": None if p is None else float(p),
        "cohen_dz": None if dz is None else float(dz),
        "ci95_mean_difference": ci,
        "wilcoxon_w_plus": wplus,
        "wilcoxon_w_minus": wminus,
        "wilcoxon_w_min": min(wplus, wminus),
        "wilcoxon_p_two_sided": wp,
        "rank_biserial": float(rbc),
    }
