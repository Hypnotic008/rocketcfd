"""Average-based adaptive refinement for LES (Mozaffari, Guilmineau, Visonneau & Wackers,
Comput. Fluids 232, 2022), with a cell budget as in NASA LAVA's Cartesian AMR (Kiris et al.,
Aerosp. Sci. Technol. 55, 2016).

An LES grid that follows every eddy changes under the turbulence it is meant to resolve. The
paper's answer: refine on the TIME AVERAGE of the instantaneous refinement criterion, which
settles to a fixed region (the shock layers, the wake, the shear layers off the chines and fins)
instead of chasing the instantaneous vortices. Their criterion (Eq. 4, "FCH") is the
second-derivative (Hessian) error indicator of a second-order scheme,

    C = max(|H(p)|, rho U |H(u)|, rho U |H(v)|, rho U |H(w)|) ** 1/2 ,

and a cell needs refining while C h > Tr (Eq. 1). LAVA instead sets the threshold from a cell
budget; here both: the threshold Tr is the one that refines the requested number of new cells.

  * CriterionAverager: hook for a run, accumulates the average of C on the base grid (GPU) and
    writes adapt_criterion.npz into the results folder.
  * adapted_patches: patches of cell size h_new over the cells with the largest C h, within a
    cell budget, outside the regions an existing patch already refines at least as finely.
"""
import os
import numpy as np
import torch

from .solver import NG


def _hess_norm(f, ax):
    """Frobenius norm of the Hessian of f (3-D tensor) on the rectilinear axes ax."""
    g = torch.gradient(f, spacing=ax)
    s = torch.zeros_like(f)
    for i in range(3):
        gi = torch.gradient(g[i], spacing=ax)
        for j in range(3):
            s += gi[j] * gi[j]
    return torch.sqrt(s)


class CriterionAverager:
    """Time average of the FCH criterion on a solver's interior cells (call record(step, t))."""

    def __init__(self, solver):
        self.s = solver
        dev = solver.U.device
        self.ax = tuple(torch.as_tensor(np.asarray(a[NG:-NG], np.float64), dtype=solver.dtype, device=dev)
                        for a in solver.axes)
        self.fluid = torch.as_tensor(np.asarray(solver.fluid)[NG:-NG, NG:-NG, NG:-NG], device=dev)
        self.sum = torch.zeros(self.fluid.shape, dtype=torch.float64, device=dev)
        self.n = 0

    @torch.no_grad()
    def record(self, step=None, t=None):
        s = self.s
        if s.backend == "warp":
            s._wprim("U")
            W = s.W_buf
        else:
            W = s._prim(s.U)
        W = W[:, NG:-NG, NG:-NG, NG:-NG]
        rho, u, v, w, p = (W[c] for c in range(5))
        U = torch.sqrt(u * u + v * v + w * w)
        c = _hess_norm(p, self.ax)
        rU = rho * U
        for comp in (u, v, w):
            c = torch.maximum(c, rU * _hess_norm(comp, self.ax))
        c = torch.sqrt(c)
        c = torch.where(self.fluid & torch.isfinite(c), c, torch.zeros_like(c))
        self.sum += c.double()
        self.n += 1

    def save(self, path):
        s = self.s
        avg = (self.sum / max(self.n, 1)).cpu().numpy().astype(np.float32)
        np.savez_compressed(path, C=avg, samples=self.n,
                            fx=np.asarray(s.faces[0]), fy=np.asarray(s.faces[1]), fz=np.asarray(s.faces[2]),
                            fluid=np.asarray(s.fluid)[NG:-NG, NG:-NG, NG:-NG])
        return path


def load_criterion(path):
    if os.path.isdir(path):
        path = os.path.join(path, "adapt_criterion.npz")
    d = np.load(path)
    return dict(C=d["C"].astype(np.float64), faces=[d["fx"], d["fy"], d["fz"]], fluid=d["fluid"],
                samples=int(d["samples"]))


def goal_weight(crit, sdf, length=6.0):
    """Hybrid feature / goal weight (Vivarelli, Qin & Shahpar, Fluids 10, 2025, Sec. 5): the
    error in the surface loads comes from the flow near the body, so the averaged criterion is
    weighted by exp(-d / length) with d the distance to the body. A cheap stand-in for an adjoint
    weighting (Kast & Fidkowski 2013): the shock layer and the flow over the chines and fins are
    favoured over the far wake."""
    cf = crit["faces"]
    X = np.meshgrid(*[0.5 * (f[1:] + f[:-1]) for f in cf], indexing="ij")
    d = np.maximum(sdf(*X), 0.0)
    return np.exp(-d / length)


def adapted_patches(crit, base_faces, h_new, budget, existing=(), region=None, box=(64, 64, 64),
                    verbose=True, weight=None):
    """Patches of cell size h_new where the averaged criterion times the (criterion grid's) cell
    size is largest, sized to about `budget` new cells.

    crit: load_criterion(...). existing: patch face lists already in the run; cells inside one
    at least as fine as h_new are skipped. region: ((x0, y0, z0), (x1, y1, z1)) bounding box
    for the refinement (LAVA-style refinement box). weight: optional per-cell factor on C h
    (same shape as the criterion), e.g. goal_weight() for refinement aimed at the surface loads."""
    from .patches import cell_patches
    C, cf = crit["C"], crit["faces"]
    cen = np.meshgrid(*[0.5 * (f[1:] + f[:-1]) for f in cf], indexing="ij")
    wid = np.meshgrid(*[np.diff(f) for f in cf], indexing="ij")
    hc = np.maximum.reduce(wid)
    err = C * hc
    if weight is not None:
        err = err * weight
    ok = crit["fluid"].astype(bool) & (err > 0) & (hc > 1.2 * h_new)
    if region is not None:
        lo, hi = np.asarray(region[0], float), np.asarray(region[1], float)
        for d in range(3):
            ok &= (cen[d] >= lo[d]) & (cen[d] <= hi[d])
    for pf in existing:
        hp = max(float(np.min(np.diff(f))) for f in pf)
        if hp > h_new * 1.01:
            continue
        inside = np.ones_like(ok)
        for d in range(3):
            inside &= (cen[d] >= pf[d][0]) & (cen[d] <= pf[d][-1])
        ok &= ~inside
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return [], 0.0
    order = idx[np.argsort(-err.ravel()[idx])]
    vol = np.prod([w.ravel()[order] for w in wid], axis=0)
    fine_cells = np.cumsum(vol / h_new ** 3)

    def build(k):
        sel = order[:k]
        c3 = np.stack([c.ravel()[sel] for c in cen])
        h3 = np.stack([0.5 * w.ravel()[sel] for w in wid])
        pf = cell_patches(base_faces, (h_new,) * 3, c3, h3, box=box, margin=2, verbose=False)
        return pf, sum(int(np.prod([len(f) - 1 for f in p])) for p in pf)

    # the patches (tiled, with margins) hold more cells than the flagged ones: bisect on the
    # number of flagged cells until the patch total is within the budget
    lo_k, hi_k = 1, int(np.searchsorted(fine_cells, budget)) + 1
    hi_k = min(max(hi_k, 1), len(order))
    best = build(lo_k)
    for _ in range(12):
        if hi_k - lo_k <= max(1, len(order) // 2000):
            break
        mid = (lo_k + hi_k) // 2
        pf, n = build(mid)
        if n <= budget:
            lo_k, best = mid, (pf, n)
        else:
            hi_k = mid
    pf, n = best
    thr = float(err.ravel()[order[lo_k - 1]])
    if verbose:
        print(f"adapt: {len(pf)} patches, {n / 1e6:.2f} M cells of {h_new} m over the {lo_k} base cells with "
              f"the largest averaged criterion (C h > {thr:.3g}; {crit['samples']} samples)")
    return pf, thr
