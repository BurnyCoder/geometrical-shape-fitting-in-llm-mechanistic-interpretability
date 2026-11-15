"""Statistics: bootstrap CIs, paired comparisons, multiple-testing correction, calibration.

Global context: implements the preregistered analysis plan (reports/preregistration.md, Sec. 5):
percentile bootstrap CIs (Efron & Tibshirani 1993), optionally over clusters of dependent test clouds
(cluster bootstrap, added after the 2026-11-14 audit), hierarchical bootstrap over seeds then samples
(Saravanan et al. 2020, "Application of the hierarchical bootstrap to multi-level data in
neuroscience", arXiv:2007.07797), paired bootstrap + Wilcoxon signed-rank tests with Holm's
step-down correction (Holm 1979), expected calibration error and temperature scaling
(Guo et al. 2017, arXiv:1706.04599).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


def _sampler(n: int, groups, rng: np.random.Generator):
    """Callable that returns one bootstrap index array over n samples.

    groups=None: i.i.d. resampling of samples. Otherwise a cluster bootstrap (Davison & Hinkley 1997, Sec. 3.8; Field &
    Welsh 2007, doi:10.1111/j.1467-9868.2007.00593.x): whole groups are drawn with replacement and keep all their
    members. Needed because test clouds are generated in chunks that share N and the R^D-embedding draw
    (data/testsets.py), so clouds of one chunk are not independent (audit 2026-11-14, reports/deviations.md).
    """
    if groups is None:
        return lambda: rng.integers(0, n, n)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")                                    # member indices grouped by group id
    members = np.split(order, np.cumsum(np.bincount(inv))[:-1])               # one index array per group
    return lambda: np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def _pct(boots, alpha: float) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_ci(values, stat=np.mean, reps: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None) -> tuple[float, float, float]:
    """Percentile bootstrap: (estimate, lower, upper); groups -> cluster bootstrap."""
    v = np.asarray(values)
    draw = _sampler(len(v), groups, np.random.default_rng(seed))
    boots = np.array([stat(v[draw()]) for _ in range(reps)])
    return (float(stat(v)), *_pct(boots, alpha))


def bootstrap_ci_pairs(y_true, y_pred, stat, reps: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None) -> tuple[float, float, float]:
    """Bootstrap for statistics of (label, prediction) pairs, e.g. macro-F1; groups -> cluster bootstrap."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    draw = _sampler(len(yt), groups, np.random.default_rng(seed))
    boots = [stat(yt[i], yp[i]) for i in (draw() for _ in range(reps))]
    return (float(stat(yt, yp)), *_pct(boots, alpha))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))


def paired_bootstrap(a, b, stat=np.mean, reps: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None) -> tuple[float, float, float]:
    """CI of stat(a) - stat(b) with the same resampled indices for both (paired design); groups -> cluster bootstrap."""
    a, b = np.asarray(a), np.asarray(b)
    draw = _sampler(len(a), groups, np.random.default_rng(seed))
    diffs = [stat(a[i]) - stat(b[i]) for i in (draw() for _ in range(reps))]
    return (float(stat(a) - stat(b)), *_pct(diffs, alpha))


def paired_bootstrap_f1(y, pa, pb, reps: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None) -> tuple[float, float, float]:
    """CI of macro-F1(pa) - macro-F1(pb) on the same resampled clouds; groups -> cluster bootstrap."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    draw = _sampler(len(y), groups, np.random.default_rng(seed))
    diffs = [macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i]) for i in (draw() for _ in range(reps))]
    return (macro_f1(y, pa) - macro_f1(y, pb), *_pct(diffs, alpha))


def hierarchical_bootstrap(groups: list, stat=np.mean, reps: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Resample groups (seeds) with replacement, then samples within each chosen group."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g) for g in groups]
    boots = []
    for _ in range(reps):
        gs = [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        boots.append(stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in gs])))
    return float(stat(np.concatenate(groups))), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilcoxon_p(a, b, groups=None) -> float:
    """Wilcoxon signed-rank p-value of the paired differences a - b; with groups, of the per-group mean differences,
    so that the test's independence assumption holds for clustered samples."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if groups is not None:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        d = np.bincount(inv, weights=d) / np.bincount(inv)                   # one mean difference per group
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction (Guo et al. 2017, Eq. 3)."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(e)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature T minimising validation NLL (Guo et al. 2017, Sec. 4.2), via LBFGS on log T."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    lg, lb = logits.detach().float().cpu(), labels.cpu()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def binom_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial CI."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)
