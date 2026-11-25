"""Snapshots during a run, for animations of the flow developing.

Each snapshot is small: surface Cp at the body-surface vertices (gathered on the GPU),
Mach on the two planes through the booster axis (x-z: rudder fin / tower side; x-y: catch
fins), and optionally a coarse 3-D Mach field. Saved as float16 .npz files in
<run folder>/snapshots/, played back by `python -m rocketcfd.viewer <folder> --movie evolve`.

For a physically time-accurate movie run with --time-accurate (global time step, every
cell every step); in steady mode the frames show the path to convergence instead.
"""
import os
import numpy as np
import torch

NG = 2


class SnapshotRecorder:
    def __init__(self, solver, folder, q_inf, p_inf, region, half=15.0, volume_stride=0):
        from skimage.measure import marching_cubes
        self.s = solver
        self.dir = os.path.join(folder, "snapshots")
        os.makedirs(self.dir, exist_ok=True)
        for f in os.listdir(self.dir):
            if f.startswith("snap_"):
                os.remove(os.path.join(self.dir, f))
        self.q, self.p_inf = q_inf, p_inf
        ax = solver.axes
        shape = solver.phi.shape
        # body surface (fixed geometry) and its trilinear gather stencil
        rng = [np.flatnonzero((ax[d] >= region[0][d]) & (ax[d] <= region[1][d])) for d in range(3)]
        sl = tuple(slice(r[0], r[-1] + 1) for r in rng)
        v, faces, _, _ = marching_cubes(solver.phi[sl], 0.0)
        idx = v + np.array([r[0] for r in rng])
        xyz = np.stack([np.interp(idx[:, d], np.arange(len(ax[d])), ax[d]) for d in range(3)], 1)
        i0 = np.floor(idx).astype(np.int64)
        fr = idx - i0
        corners, weights = [], []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    ii = np.minimum(i0[:, 0] + a, shape[0] - 1)
                    jj = np.minimum(i0[:, 1] + b, shape[1] - 1)
                    kk = np.minimum(i0[:, 2] + c, shape[2] - 1)
                    corners.append(np.ravel_multi_index((ii, jj, kk), shape))
                    weights.append((fr[:, 0] if a else 1 - fr[:, 0]) * (fr[:, 1] if b else 1 - fr[:, 1])
                                   * (fr[:, 2] if c else 1 - fr[:, 2]))
        dev = solver.device
        self.c_t = torch.as_tensor(np.stack(corners, 1), device=dev)
        self.w_t = torch.as_tensor(np.stack(weights, 1), dtype=solver.dtype, device=dev)
        # the two planes through the axis, cropped to +/- half around the vehicle
        x, y, z = (a_[NG:-NG] for a_ in ax)
        self.jy0 = int(np.argmin(np.abs(y))) + NG
        self.kz0 = int(np.argmin(np.abs(z))) + NG
        self.jr = np.flatnonzero(np.abs(y) <= half) + NG
        self.kr = np.flatnonzero(np.abs(z) <= half) + NG
        self.ir = np.arange(NG, NG + len(x))
        solid = ~solver.fluid
        mesh = dict(xyz=xyz.astype(np.float32), faces=faces.astype(np.int32),
                    x=x.astype(np.float32), y=ax[1][self.jr].astype(np.float32), z=ax[2][self.kr].astype(np.float32),
                    solid_xz=solid[self.ir][:, self.jy0][:, self.kr], solid_xy=solid[self.ir][:, self.jr][:, :, self.kz0],
                    q_inf=q_inf, p_inf=p_inf)
        self.vs = volume_stride
        if volume_stride:
            self.vi = self.ir[::volume_stride]
            self.vj = self.jr[::volume_stride]
            self.vk = self.kr[::volume_stride]
            mesh.update(vx=ax[0][self.vi].astype(np.float32), vy=ax[1][self.vj].astype(np.float32),
                        vz=ax[2][self.vk].astype(np.float32),
                        vol_solid=solid[self.vi][:, self.vj][:, :, self.vk], rho_inf=float(solver.W_inf[0]),
                        gamma=float(solver.gamma))
        np.savez_compressed(os.path.join(self.dir, "mesh.npz"), **mesh)
        self.k = 0

    def _device_primitive(self):
        s = self.s
        if s.backend == "warp":
            s._wprim("U")
            return s.W_buf
        return s._prim(s.U)

    @torch.no_grad()
    def record(self, step, t):
        W = self._device_primitive()
        g = self.s.gamma
        p = W[4]
        mach = torch.sqrt(W[1] ** 2 + W[2] ** 2 + W[3] ** 2) / torch.sqrt(g * p / W[0])
        pv = (p.reshape(-1)[self.c_t] * self.w_t).sum(1)
        cp = ((pv - self.p_inf) / self.q).half().cpu().numpy()
        ir = torch.as_tensor(self.ir, device=W.device)
        jr = torch.as_tensor(self.jr, device=W.device)
        kr = torch.as_tensor(self.kr, device=W.device)
        m_xz = mach[ir][:, self.jy0][:, kr].half().cpu().numpy()
        m_xy = mach[ir][:, jr][:, :, self.kz0].half().cpu().numpy()
        out = dict(cp=cp, mach_xz=m_xz, mach_xy=m_xy, step=step, t=t)
        if self.vs:
            vi, vj, vk = (torch.as_tensor(a_, device=W.device) for a_ in (self.vi, self.vj, self.vk))
            sub = lambda a_: a_[vi][:, vj][:, :, vk]
            out["mach_vol"] = sub(mach).half().cpu().numpy()
            # full state for 3-D movies (rocketcfd.cinematic: Q-criterion, schlieren); float16 is
            # plenty for pictures. Pressure is stored relative to the freestream to keep precision.
            out["vol_rho"] = sub(W[0]).half().cpu().numpy()
            out["vol_u"] = sub(W[1]).half().cpu().numpy()
            out["vol_v"] = sub(W[2]).half().cpu().numpy()
            out["vol_w"] = sub(W[3]).half().cpu().numpy()
            out["vol_dp"] = (sub(W[4]) - self.p_inf).half().cpu().numpy()
        np.savez_compressed(os.path.join(self.dir, f"snap_{self.k:04d}.npz"), **out)
        self.k += 1
