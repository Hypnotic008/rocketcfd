"""Restart files: save every grid of a run (base + patches, with k / omega), and start a new run
from one -- on the same grids (exact copy) or on different ones (trilinear interpolation of the
primitive variables from the finest old grid at each point). So a run can be continued, refined,
or switched to other physics (e.g. an LES or RANS run started from a converged inviscid one).

  python cases/super_heavy_descent.py ... --restart results/glide_patch      (folder or .npz)
"""
import numpy as np
import torch

from .gas import cons_to_prim, prim_to_cons
from .solver import NG


def _grids(runner):
    return runner.grids if hasattr(runner, "grids") else [runner]


def save(runner, path):
    out = {}
    gs = _grids(runner)
    out["n_grids"] = len(gs)
    for i, g in enumerate(gs):
        out[f"U{i}"] = g.U.detach().cpu().numpy().astype(np.float32)
        for d in range(3):
            out[f"f{i}_{d}"] = np.asarray(g.faces[d], float)
        if g.turbulence:
            out[f"K{i}"] = g.K_buf.detach().cpu().numpy().astype(np.float32)
            out[f"O{i}"] = g.O_buf.detach().cpu().numpy().astype(np.float32)
    base = gs[0]
    out.update(t=float(base.t), step=int(base.step_count), gamma=float(base.gamma), version=2)
    np.savez_compressed(path, **out)
    return path


def _read(path):
    import os
    if os.path.isdir(path):
        path = os.path.join(path, "restart.npz")
    d = np.load(path)
    if "n_grids" not in d:                         # old single-grid restart (Solver.save)
        return [dict(U=d["U"], faces=[d["fx"], d["fy"], d["fz"]])], float(d["t"]), int(d["step"])
    grids = []
    for i in range(int(d["n_grids"])):
        g = dict(U=d[f"U{i}"], faces=[d[f"f{i}_{k}"] for k in range(3)])
        if f"K{i}" in d:
            g.update(K=d[f"K{i}"], O=d[f"O{i}"])
        grids.append(g)
    return grids, float(d["t"]), int(d["step"])


def _centres(faces):
    """Padded cell centres (NG ghost layers mirrored at the ends), as the solver lays them out."""
    out = []
    for f in faces:
        f = np.asarray(f, float)
        c = 0.5 * (f[1:] + f[:-1])
        w0, w1 = f[1] - f[0], f[-1] - f[-2]
        lo = c[0] - w0 * np.arange(NG, 0, -1)
        hi = c[-1] + w1 * np.arange(1, NG + 1)
        out.append(np.concatenate([lo, c, hi]))
    return out


def _reseed_near_walls(n, vals, h_src, verbose=True, band=1.5):
    """New fluid cells within `band` old cells of a wall (body or thin fin plate) take the state of
    the nearest new fluid cell clear of the walls.

    Interpolating from the old grid there mixes in its solid and immersed-boundary ghost cells,
    which hold mirrored (wall-reflected) states; on a finer new grid around thin grid-fin plates
    those land in the free flow and an LES restarted that way blew up within 50 steps at the
    windward fin. The solver rebuilds the near-wall flow from the re-seeded state in a few steps."""
    shape = h_src.shape
    if not hasattr(n, "phi") or n.phi.shape != shape:
        return
    dist = np.asarray(n.phi, float).copy()
    td = getattr(n, "_thin_dist", None)
    if td is not None:
        sub, d_thin = td
        dist[sub] = np.minimum(dist[sub], d_thin)
    fluid = np.asarray(n.fluid, bool)
    near = fluid & (dist < band * h_src)
    far = fluid & ~near
    if not near.any() or not far.any():
        return
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~far, return_distances=False, return_indices=True)
    flat = np.ravel_multi_index(tuple(idx), shape)[near]
    V = vals.reshape(vals.shape[0], -1)
    nz = np.flatnonzero(near.ravel())
    V[:, nz] = V[:, flat]
    if verbose:
        print(f"restart: {len(nz)} near-wall cells re-seeded from the flow clear of the walls")


def load(runner, path, gamma=None, verbose=True, rotate_x=0.0):
    """Fill `runner` (Solver or PatchGrid) from a restart file. Returns 'exact' or 'interpolated'.

    rotate_x [rad]: turn the old velocity vectors about the x axis (a restart at a different
    windward side). Any rotation, or a different grid, goes through the interpolation path."""
    from scipy.interpolate import RegularGridInterpolator
    old, t, step = _read(path)
    new = _grids(runner)
    g0 = new[0]
    gam = gamma or g0.gamma
    same = len(old) == len(new) and all(
        all(len(o["faces"][d]) == len(n.faces[d]) and np.allclose(o["faces"][d], n.faces[d]) for d in range(3))
        for o, n in zip(old, new))
    with torch.no_grad():
        if same and not rotate_x:
            for o, n in zip(old, new):
                n.U.copy_(torch.as_tensor(o["U"], dtype=n.dtype, device=n.device))
                if n.turbulence and "K" in o:
                    n.K_buf.copy_(torch.as_tensor(o["K"], dtype=n.dtype, device=n.device))
                    n.O_buf.copy_(torch.as_tensor(o["O"], dtype=n.dtype, device=n.device))
                if n.backend == "warp":
                    n.U1.copy_(n.U)
            for n in new:
                n.t, n.step_count = t, step
            if verbose:
                print(f"restart: exact copy of {len(old)} grid(s) at step {step}, t = {t:.4g} s")
            return "exact"
        # interpolation: primitive variables from the finest old grid containing each point
        vol = [float(np.prod([np.min(np.diff(o["faces"][d])) for d in range(3)])) for o in old]
        order = np.argsort(vol)                                    # finest first
        interps, h_old = [], []
        for i in order:
            h_old.append(float(max(np.max(np.diff(o["faces"][d])) for d in range(3))))
            o = old[i]
            ax = _centres(o["faces"])
            W = cons_to_prim(torch.as_tensor(o["U"], dtype=torch.float64), gam, 1e-9, 1e-6).numpy()
            if rotate_x:
                cr, sr = np.cos(rotate_x), np.sin(rotate_x)
                W[2], W[3] = cr * W[2] - sr * W[3], sr * W[2] + cr * W[3]
            fields = [W[c] for c in range(5)]
            if "K" in o:
                fields += [o["K"] / np.maximum(W[0], 1e-12), o["O"] / np.maximum(W[0], 1e-12)]
            lo = [o["faces"][d][0] for d in range(3)]
            hi = [o["faces"][d][-1] for d in range(3)]
            interps.append((lo, hi, [RegularGridInterpolator(ax, f, bounds_error=False, fill_value=None)
                                     for f in fields]))
        for n in new:
            X, Y, Z = np.meshgrid(*n.axes, indexing="ij")
            P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
            done = np.zeros(len(P), bool)
            vals = np.zeros((7, len(P)))
            h_src = np.zeros(len(P))                              # cell size of the old grid used
            have_turb = False
            for (lo, hi, its), hs in zip(interps, h_old):
                ins = ~done & np.all((P >= lo) & (P <= hi), axis=1)
                if not ins.any():
                    continue
                for c, it in enumerate(its):
                    vals[c, ins] = it(P[ins])
                h_src[ins] = hs
                have_turb |= len(its) == 7
                done |= ins
            if (~done).any():                                     # outside every old grid: nearest
                lo, hi, its = interps[-1]
                for c, it in enumerate(its):
                    vals[c, ~done] = it(np.clip(P[~done], lo, hi))
                h_src[~done] = h_old[-1]
            shape = X.shape
            _reseed_near_walls(n, vals, h_src.reshape(shape), verbose)
            W = vals[:5].reshape((5,) + shape)
            W[0] = np.maximum(W[0], 1e-6)
            W[4] = np.maximum(W[4], 1e-3)
            n.U.copy_(prim_to_cons(torch.as_tensor(W, dtype=n.dtype, device=n.device), n.gamma))
            if n.turbulence and have_turb:
                rho = torch.as_tensor(W[0], dtype=n.dtype, device=n.device)
                n.K_buf.copy_(rho * torch.as_tensor(np.maximum(vals[5].reshape(shape), 0.0), dtype=n.dtype, device=n.device))
                n.O_buf.copy_(rho * torch.as_tensor(np.maximum(vals[6].reshape(shape), 1e-6), dtype=n.dtype, device=n.device))
            if n.backend == "warp":
                n.U1.copy_(n.U)
                if n.turbulence:
                    n.K1_buf.copy_(n.K_buf)
                    n.O1_buf.copy_(n.O_buf)
    if verbose:
        rot = f", crossflow turned {np.degrees(rotate_x):+.0f} deg" if rotate_x else ""
        print(f"restart: interpolated from {len(old)} old grid(s) onto {len(new)} new grid(s){rot}")
    return "interpolated"
