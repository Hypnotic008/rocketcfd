"""Probe points: time histories of rho, u, v, w, p at fixed points during a run (unsteady /
LES runs: buffeting, wake and base-pressure fluctuations, shedding frequencies). Values are
interpolated trilinearly from the finest grid containing each point, gathered on the device
every `every` steps and written to <folder>/probes.npz (t, step, points, names, data[t, point, var]).

  python cases/super_heavy_descent.py ... --les --probes            (a default set around the booster)
  python cases/super_heavy_descent.py ... --probe-points "75,0,0; 80,0,3"
The app plots them (Plots -> Probes: history and spectrum).
"""
import os

import numpy as np
import torch

from .solver import NG


class ProbeRecorder:
    NAMES = ("rho", "u", "v", "w", "p")

    def __init__(self, runner, points, folder, names=None, every=1, flush_every=2000):
        self.grids = runner.grids if hasattr(runner, "grids") else [runner]
        self.points = np.asarray(points, float).reshape(-1, 3)
        self.names = list(names) if names is not None else [f"P{i}" for i in range(len(self.points))]
        self.folder = folder
        self.every = max(1, int(every))
        self.flush_every = flush_every
        self.buf, self.t, self.step = [], [], []
        # for each point: the finest grid whose interior contains it, and its trilinear stencil
        vol = [float(np.prod(g.h)) for g in self.grids]
        self.plan = []                         # per grid: (point ids, flat corner idx (P, 8), weights (P, 8))
        owner = np.full(len(self.points), -1)
        for gi in np.argsort(vol):
            g = self.grids[gi]
            ax = g.axes
            ins = np.ones(len(self.points), bool)
            for d in range(3):
                ins &= (self.points[:, d] >= ax[d][NG]) & (self.points[:, d] <= ax[d][-NG - 1])
            ins &= owner < 0
            owner[ins] = gi
        for gi in np.unique(owner[owner >= 0]):
            g = self.grids[gi]
            ids = np.flatnonzero(owner == gi)
            shape = tuple(int(v) for v in g.U.shape[1:])
            corners, weights = [], []
            i0s, frs = [], []
            for d in range(3):
                a = g.axes[d]
                i0 = np.clip(np.searchsorted(a, self.points[ids, d]) - 1, 0, len(a) - 2)
                fr = (self.points[ids, d] - a[i0]) / (a[i0 + 1] - a[i0])
                i0s.append(i0)
                frs.append(np.clip(fr, 0.0, 1.0))
            for a_ in (0, 1):
                for b_ in (0, 1):
                    for c_ in (0, 1):
                        corners.append(np.ravel_multi_index((i0s[0] + a_, i0s[1] + b_, i0s[2] + c_), shape))
                        weights.append((frs[0] if a_ else 1 - frs[0]) * (frs[1] if b_ else 1 - frs[1])
                                       * (frs[2] if c_ else 1 - frs[2]))
            dev = g.device
            self.plan.append((ids, torch.as_tensor(np.stack(corners, 1), device=dev),
                              torch.as_tensor(np.stack(weights, 1), dtype=g.dtype, device=dev), g))
        self.missing = np.flatnonzero(owner < 0)
        os.makedirs(folder, exist_ok=True)

    @torch.no_grad()
    def record(self, step, t):
        if step % self.every:
            return
        out = torch.full((len(self.points), 5), float("nan"), dtype=torch.float32, device=self.grids[0].device)
        for ids, cidx, w, g in self.plan:
            if g.backend == "warp":
                g._wprim("U")
                W = g.W_buf
            else:
                W = g._prim(g.U)
            Wf = W.reshape(5, -1)
            vals = (Wf[:, cidx] * w[None]).sum(-1)          # (5, P)
            out[torch.as_tensor(ids, device=out.device)] = vals.T.float()
        self.buf.append(out)
        self.t.append(float(t))
        self.step.append(int(step))
        if len(self.buf) >= self.flush_every:
            self.save()

    def save(self):
        if not self.t:
            return
        data = torch.stack(self.buf).cpu().numpy() if self.buf else np.zeros((0, len(self.points), 5))
        path = os.path.join(self.folder, "probes.npz")
        if os.path.exists(path) and getattr(self, "_saved", 0):
            old = np.load(path)
            data = np.concatenate([old["data"], data])
            t = np.concatenate([old["t"], self.t])
            st = np.concatenate([old["step"], self.step])
        else:
            t, st = np.array(self.t), np.array(self.step)
        np.savez_compressed(path, t=t, step=st, data=data, points=self.points, names=np.array(self.names),
                            vars=np.array(self.NAMES))
        self._saved = 1
        self.buf, self.t, self.step = [], [], []
