"""Model selection among candidate shape families: calibrated classifier, geometric AIC, or hybrid.

Global context: after refinement, several families may fit a cloud (nested families: a full circle is
also an ellipse, a helix with tiny pitch is almost a circle). Selection rules (ablation A5):
- classifier: argmax of temperature-scaled softmax probabilities (Guo et al. 2017, "On Calibration of
  Modern Neural Networks", arXiv:1706.04599).
- gaic: Kanatani's geometric AIC, AIC_G = J + 2 (d N + p) eps^2, with J the sum of squared point-to-shape
  distances, d the manifold dimension, p the number of shape parameters and eps^2 the noise variance,
  estimated from the most general candidate as eps^2 = J_gen / ((3 - d_gen) N - p_gen)
  (Kanatani 2004, "Uncertainty Modeling and Model Selection for Geometric Inference", IEEE TPAMI 26(10);
  doi:10.1109/TPAMI.2004.93; author copy https://iim.cs.tut.ac.jp/member/kanatani/papers/pamodel2.pdf). The volumetric blob has d = 3; its J is computed
  like every other family's (refine.py). Deviation from Kanatani: J is the distance to the nearest of a
  finite set of shape samples (512 per curve, a 40 x 40 grid per surface), evaluated at the two-sided
  refinement optimum rather than at the minimiser of J, so for surfaces it includes discretisation error
  comparable to the lowest noise levels (reports/deviations.md).
- hybrid: among candidates whose calibrated probability is >= 0.1 x the top probability, the lowest AIC_G.
"""

from __future__ import annotations

import torch

from neurogeom.shapes.registry import FAMILIES


def n_free(f, p: dict) -> torch.Tensor:
    if f.name == "clusters":
        return (3 * p["k"] + 1).to(torch.float32)
    return torch.full_like(p["center"][:, 0] if "center" in p else p["spread"], float(f.n_free))


def _stack(cands: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(J, d, p) of every candidate: (B, n_cands), (n_cands,), (B, n_cands)."""
    js = torch.stack([c["J"] for c in cands], -1)
    ds = torch.tensor([FAMILIES[c["fi"]].manifold_dim for c in cands], dtype=torch.float32, device=js.device)
    return js, ds, torch.stack([c["p"] for c in cands], -1)


def _general(cands: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """J (B,), p (B,) and d of the most general non-volumetric candidate (highest d < 3; ties: smallest J)."""
    js, ds, ps = _stack(cands)
    gen_mask = ds < 3
    dmax = ds[gen_mask].max() if bool(gen_mask.any()) else torch.tensor(2.0)
    jg = js.masked_fill(~(gen_mask & (ds == dmax)), float("inf")).min(-1)
    return jg.values, torch.gather(ps, -1, jg.indices.unsqueeze(-1)).squeeze(-1), dmax


def noise_eps2(cands: list[dict], n_points: torch.Tensor) -> torch.Tensor:
    """(B,) noise variance estimated from the most general non-volumetric candidate g, eps^2 = J_g / ((3 - d_g) N - p_g).
    Because the noise is estimated from g itself, AIC_G(blob) - AIC_G(g) = J_blob + J_g + 2 p_blob eps^2 > 0, i.e. the
    d = 3 blob can never be selected (audit 2026-11-14; B1-fixed decides the blob with general_rms, baselines/classical.py)."""
    jg, p_g, dmax = _general(cands)
    dof = ((3 - dmax) * n_points.to(torch.float32) - p_g).clamp_min(1.0)
    return (jg / dof).clamp_min(1e-8)


def general_rms(cands: list[dict], n_points: torch.Tensor) -> torch.Tensor:
    """(B,) RMS point-to-shape distance of that most general candidate, sqrt(J_g / N): the noise level the cloud would have
    if it were that surface. B1-fixed's blob statistic; unlike sqrt(eps^2) it stays bounded when N <= p_g, where the
    degrees of freedom (3 - d) N - p of the G-AIC noise estimate are clamped to 1 (observed in the smoke run: 5-point
    lines got eps_hat ~ 0.8)."""
    return (_general(cands)[0] / n_points.to(torch.float32).clamp_min(1.0)).sqrt()


def gaic_scores(cands: list[dict], n_points: torch.Tensor) -> torch.Tensor:
    """cands: list over families of {"fi", "J" (B,), "p" (B,)}; returns AIC_G (B, n_cands)."""
    n = n_points.to(torch.float32)
    js, ds, ps = _stack(cands)
    return js + 2 * (ds.unsqueeze(0) * n.unsqueeze(-1) + ps) * noise_eps2(cands, n_points).unsqueeze(-1)


def select(probs: torch.Tensor, cand_fi: list[int], aic: torch.Tensor | None, rule: str) -> torch.Tensor:
    """Index into cand_fi chosen per sample. probs: (B, 13) calibrated; aic: (B, n_cands)."""
    pc = probs[:, cand_fi]
    if rule == "classifier" or aic is None:
        return pc.argmax(-1)
    if rule == "gaic":
        return aic.argmin(-1)
    if rule == "hybrid":
        ok = pc >= 0.1 * pc.max(-1, keepdim=True).values
        return aic.masked_fill(~ok, float("inf")).argmin(-1)
    raise ValueError(rule)
