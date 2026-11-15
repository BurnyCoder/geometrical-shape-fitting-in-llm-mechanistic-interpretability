"""Test-time refinement: gradient-based fitting of one family's parameters to a point cloud.

Global context: the network gives an amortised first guess (as in SPIN, Kolotouros et al. 2019,
arXiv:1909.12828, where a regressor initialises an optimisation-based fit); refinement then makes the
shape pass as close to the input points as possible. The objective is two-sided:
    J(theta) = mean_i d(x_i, S_theta) + lambda * sum_j w_j d(s_j, X)
The first term alone (points -> shape) is degenerate: a sphere's radius or an arc's extent can grow
to cover every point, and cluster centroids can multiply; the second term (shape -> points, weighted
by length/area) penalises shape regions with no supporting points. The same routine, started from
heuristic initialisations instead of the network, is the classical baseline B1.
"""

from __future__ import annotations

import torch

from neurogeom.geometry.chamfer import nearest_dist, nearest_index, one_sided, residual


def objective(f, raw: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select, dense: dict, back_w: float) -> torch.Tensor:
    p = f.decode(raw, select)
    pts, w = f.dense(p, dense)
    fwd = residual(x, mask, pts, w)                                   # input points -> shape
    back = one_sided(pts, w, x, mask.to(x.dtype))                     # shape -> input points
    return fwd + back_w * back


def refine(f, raw0: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select=None, steps: int = 300,
           lr: float = 1e-2, dense: dict | None = None, back_w: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam on the raw (unconstrained) parameters, independently per sample. Returns (raw, objective)."""
    dense = dense or {"curve": 512, "surface_side": 40, "blob": 256}
    raw = raw0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    best_raw, best_obj = raw.detach().clone(), torch.full((raw.shape[0],), float("inf"), device=raw.device)
    for _ in range(steps):
        obj = objective(f, raw, x, mask, select, dense, back_w)
        with torch.no_grad():                                       # keep the best iterate per sample
            better = obj < best_obj
            best_obj = torch.where(better, obj.detach(), best_obj)
            best_raw[better] = raw.detach()[better]
        opt.zero_grad(set_to_none=True)
        obj.sum().backward()                                        # samples are independent: sum = per-sample steps
        opt.step()
    with torch.no_grad():
        obj = objective(f, raw, x, mask, select, dense, back_w)
        better = obj < best_obj
        best_obj = torch.where(better, obj, best_obj)
        best_raw[better] = raw.detach()[better]
    return best_raw, best_obj


@torch.no_grad()
def grid_dist(f, p: dict, x: torch.Tensor, dense: dict) -> torch.Tensor:
    """(B, N) distance from each point to the nearest of the shape's dense samples."""
    pts, w = f.dense(p, dense)
    return nearest_dist(x, pts, w > 0)


@torch.no_grad()
def projected_dist(f, p: dict, x: torch.Tensor, dense: dict, steps: int = 10, h: float = 1e-4) -> torch.Tensor:
    """(B, N) orthogonal point-to-shape distance: start at the nearest dense sample and take damped Gauss-Newton steps
    on the shape's sampling parameter u in [0, 1]^d (point inversion on a parametric curve or surface: Ma & Hewitt
    2003, CAGD 20(2), doi:10.1016/S0167-8396(03)00021-9; Hu & Wallner 2005, CAGD 22(3), doi:10.1016/j.cagd.2004.11.003).
    Each step is backtracked (scales 1, 1/2, 1/4, 1/8, the best one per point; a full step overshoots near a sphere's
    pole or for points far outside a strongly curved curve) and kept only where it shortens the distance, so the
    result is never larger than the grid distance.
    Point sets (centroid/vertex distances are already exact) and the blob (no parametric surface) keep the grid value.
    Added by the 2026-11-14 audit: the grid distance alone inflates geometric-AIC residuals at low noise."""
    pts, w = f.dense(p, dense)
    idx = nearest_index(x, pts, w > 0)                                          # (B, N) nearest dense sample
    best = torch.sqrt(((x - torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))) ** 2).sum(-1))
    if not f.posed or f.name == "blob":
        return best
    u = f.grid(dense["curve"], dense["surface_side"], x.device)[idx]           # (B, N, d) parameter of that sample
    eye = torch.eye(u.shape[-1], device=x.device)
    for _ in range(steps):
        cols = []
        for k in range(u.shape[-1]):                                            # finite-difference Jacobian dS/du_k
            up, um = (u + h * eye[k]).clamp(0, 1), (u - h * eye[k]).clamp(0, 1)   # stay inside the domain (sqrt at u = 0)
            cols.append((f.points(p, up)[0] - f.points(p, um)[0]) / (up - um)[..., k:k + 1])
        jac = torch.stack(cols, -1)                                             # (B, N, 3, d)
        r = (x - f.points(p, u)[0]).unsqueeze(-1)                               # residual vector (B, N, 3, 1)
        jtj = jac.transpose(-1, -2) @ jac
        lam = 1e-3 * jtj.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1) + 1e-12   # Levenberg damping
        step = torch.nan_to_num(torch.linalg.solve(jtj + lam * eye, jac.transpose(-1, -2) @ r).squeeze(-1))
        u0 = u                                                                  # every trial scale starts from the same u
        for scale in (1.0, 0.5, 0.25, 0.125):                                  # backtracking line search
            cand = (u0 + scale * step).clamp(0, 1)
            dc = torch.sqrt(((x - f.points(p, cand)[0]) ** 2).sum(-1))
            better = dc < best
            u, best = torch.where(better.unsqueeze(-1), cand, u), torch.where(better, dc, best)
    return best


@torch.no_grad()
def sq_residual_sum(f, raw: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select, dense: dict, exact: bool = False) -> torch.Tensor:
    """J = sum_i d(x_i, S)^2 over valid points (used by geometric AIC); exact=True projects onto the shape (B1-fixed)."""
    p = f.decode(raw, select)
    d = projected_dist(f, p, x, dense) if exact else grid_dist(f, p, x, dense)
    return (d ** 2 * mask.to(d.dtype)).sum(-1)
