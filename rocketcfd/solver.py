"""3-D compressible Euler solver on a uniform Cartesian grid with a ghost-cell
immersed boundary (inviscid slip wall) defined by a signed distance function.

Numerics
  * finite volume, MUSCL reconstruction of primitive variables (van Leer / minmod)
  * HLLC Riemann solver with HLLE fallback at strong shocks (carbuncle cure)
  * SSP-RK2 time integration, global CFL time step
  * ghost-cell immersed boundary: each solid cell next to the fluid gets the
    mirror-image state, interpolated from fluid cells, with the normal velocity reflected
Runs with PyTorch on CPU or an NVIDIA GPU (Windows, Linux) with the same code.
"""
import os
import time
import numpy as np
import torch
from scipy import ndimage

from .gas import prim_to_cons, cons_to_prim, hllc_flux

NG = 2  # ghost layers on the domain boundary


def default_device():
    if os.environ.get("ROCKETCFD_DEVICE"):
        return torch.device(os.environ["ROCKETCFD_DEVICE"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sl(a, ax, start, stop):
    return a.narrow(ax, start, stop - start)


def _limit(a, b, kind):
    if kind == "minmod":
        return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))
    ab = a * b
    return torch.where(ab > 0, 2.0 * ab / (a + b + (ab <= 0).to(a.dtype)), torch.zeros_like(a))  # van Leer


class Solver:
    def __init__(self, lo, hi, h, freestream, sdf=None, gamma=1.4, bcs="auto", cfl=0.7,
                 limiter="vanleer", shock_threshold=0.5, dtype=torch.float32, device=None,
                 verbose=True, faces=None, backend="auto", thin_sdf=None, thin_box=None,
                 viscous=None, wall=None, turbulence=None):
        """
        lo, hi     : domain corners (m)
        h          : cell size, scalar or (hx, hy, hz)
        faces      : optional [x_faces, y_faces, z_faces] for a stretched grid
                     (see rocketcfd.grid); overrides lo, hi and h
        freestream : dict(rho, u, v, w, p)  (SI)
        sdf        : callable(X, Y, Z) -> phi  (phi < 0 inside solid), or None
        bcs        : 'auto' or {(axis, side): 'freestream'|'extrapolate'|'periodic'}
        device     : torch device; default CUDA if available, else CPU
        backend    : 'warp' (fused kernels, float32), 'torch' (reference), or 'auto'
        thin_sdf   : callable(X, Y, Z) -> phi for thin-walled parts (grid fins). These are
                     represented by blocked cell faces instead of ghost cells, so plates
                     thinner than a cell stay solid. Needs the warp backend.
        thin_box   : ((x0, y0, z0), (x1, y1, z1)) bounding box of the thin parts (speeds up setup)
        viscous    : None (Euler) or dict(mu=..., model='constant'|'sutherland', T0=273.15, S=110.4,
                     R=287.05, Pr=0.72). Sutherland defaults are for air (mu=1.716e-5 at 273.15 K).
        wall       : 'slip' (default for Euler) or 'noslip' (default when viscous) for immersed walls
        turbulence : None or dict(model='sst', k_inf=..., omega_inf=...): k-omega SST with a
                     Spalding wall function on immersed walls and thin walls (needs viscous).
                     Defaults: k_inf = 9e-9 a^2, omega_inf = 1e-6 rho a^2 / mu (NASA TMR).
        """
        self.gamma, self.cfl, self.limiter = gamma, cfl, limiter
        self.viscous = None
        if viscous is not None:
            v = dict(model="sutherland", mu=1.716e-5, T0=273.15, S=110.4, R=287.05, Pr=0.72)
            v.update(viscous)
            self.viscous = v
        self.turbulence = None
        if turbulence is not None:
            if self.viscous is None:
                raise ValueError("turbulence needs viscous=...")
            self.turbulence = dict(model="sst")
            self.turbulence.update(turbulence)
        # With the wall function the wall shear comes from the law of the wall, so immersed-wall
        # ghost cells mirror the velocity as a slip wall (a no-slip mirror would put a spurious
        # steep gradient into the first cell's strain rate and turbulence production).
        self.wall = wall or ("slip" if self.turbulence else "noslip" if self.viscous else "slip")
        self.shock_threshold = shock_threshold
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else default_device()
        self.verbose = verbose
        if faces is None:
            lo_ = np.asarray(lo, float)
            h_ = np.broadcast_to(np.asarray(h, float), (3,))
            n_ = [int(round(v)) for v in (np.asarray(hi, float) - lo_) / h_]
            faces = [lo_[d] + h_[d] * np.arange(n_[d] + 1) for d in range(3)]
        self.faces = [np.asarray(f, float) for f in faces]
        self.n = tuple(len(f) - 1 for f in self.faces)
        self.lo = np.array([f[0] for f in self.faces])
        self.hi = np.array([f[-1] for f in self.faces])
        w = [np.diff(f) for f in self.faces]
        self.h = np.array([wd.min() for wd in w])          # smallest cell size per axis
        self.h_max = np.array([wd.max() for wd in w])
        # padded cell widths and centres (ghost layers copy the edge cell width)
        self.widths = [np.concatenate([np.full(NG, wd[0]), wd, np.full(NG, wd[-1])]) for wd in w]
        self.axes = []
        for d in range(3):
            c = 0.5 * (self.faces[d][:-1] + self.faces[d][1:])
            left = c[0] - w[d][0] * np.arange(NG, 0, -1)
            right = c[-1] + w[d][-1] * np.arange(1, NG + 1)
            self.axes.append(np.concatenate([left, c, right]))
        self.fs = freestream
        self.W_inf = np.array([freestream[k] for k in ("rho", "u", "v", "w", "p")], float)
        self.rho_min = 1e-4 * freestream["rho"]
        self.p_min = 1e-4 * freestream["p"]

        t0 = time.time()
        shape = tuple(m + 2 * NG for m in self.n)
        if sdf is None:
            self.phi = np.full(shape, 1e9)
        else:
            X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
            self.phi = sdf(X, Y, Z).astype(np.float64)
            del X, Y, Z
        self._setup_bcs(bcs)
        self._setup_thin(thin_sdf, thin_box)
        self._setup_ib()
        dev = self.device
        if verbose:
            name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
            hs = ", ".join(f"{a:.3g}" if abs(a - b) < 1e-9 else f"{a:.3g}..{b:.3g}"
                           for a, b in zip(self.h, self.h_max))
            print(f"grid {self.n} = {np.prod(self.n)/1e6:.2f} M cells, cell size [{hs}] m, "
                  f"{self.n_ghost} IB ghost cells, setup {time.time()-t0:.1f} s, device {name}")

        interior = np.zeros(shape, bool)
        interior[NG:-NG, NG:-NG, NG:-NG] = True
        self.update_mask = torch.as_tensor((self.fluid & interior)[NG:-NG, NG:-NG, NG:-NG], device=dev)
        self.n_fluid = int(self.update_mask.sum())
        self.W_inf_t = torch.as_tensor(self.W_inf, dtype=dtype, device=dev)
        self.invw = []
        for d in range(3):
            shp = [1, 1, 1]
            shp[d] = self.n[d]
            self.invw.append(torch.as_tensor(1.0 / w[d], dtype=dtype, device=dev).reshape(shp))
        W0 = self.W_inf_t.reshape(5, 1, 1, 1).expand((5,) + shape).contiguous()
        self.U = prim_to_cons(W0, gamma)
        self.t = 0.0
        self.step_count = 0
        self.history = []
        self.backend = self._pick_backend(backend)
        if self.backend == "warp":
            self._setup_warp()
        if verbose:
            print(f"backend: {self.backend}")

    def _pick_backend(self, backend):
        if backend == "torch":
            if self.has_thin or self.viscous or self.wall != "slip":
                raise RuntimeError("thin walls, viscous terms and no-slip walls need the warp backend")
            return "torch"
        ok = self.dtype == torch.float32
        try:
            import warp  # noqa: F401
        except ImportError:
            ok = False
        if (backend == "warp" or self.has_thin or self.viscous or self.wall != "slip" or self.turbulence) and not ok:
            raise RuntimeError("warp backend (needed for thin walls) needs `pip install warp-lang` and float32")
        return "warp" if ok else "torch"

    def _setup_warp(self):
        import warp as wp
        from . import kernels as K
        self._wp, self._K = wp, K
        wp.init()
        dev = self.device
        self._wdev = f"cuda:{dev.index or 0}" if dev.type == "cuda" else "cpu"
        shape = self.U.shape[1:]
        self.W_buf = torch.empty_like(self.U)
        self.U1 = self.U.clone()
        self.R_buf = torch.zeros((5,) + self.n, dtype=self.dtype, device=dev)
        self.S_buf = torch.zeros(shape, dtype=torch.int32, device=dev)
        self.rate_buf = torch.zeros(self.n, dtype=self.dtype, device=dev)
        self.dt_buf = torch.zeros(1, dtype=self.dtype, device=dev)
        self.dtl_buf = torch.zeros(self.n, dtype=self.dtype, device=dev)
        nt = tuple((m + 7) // 8 for m in self.n)
        self.act_buf = torch.zeros(nt, dtype=self.dtype, device=dev)
        self.blk_buf = torch.as_tensor(self.blk, device=dev)
        self.mask_f = self.update_mask.to(self.dtype)
        self.invw1 = [self.invw[d].reshape(-1).contiguous() for d in range(3)]
        f = wp.from_torch
        self._w = dict(U=f(self.U), U1=f(self.U1), W=f(self.W_buf), R=f(self.R_buf), S=f(self.S_buf),
                       rate=f(self.rate_buf), dt=f(self.dt_buf), mask=f(self.mask_f), blk=f(self.blk_buf),
                       dtl=f(self.dtl_buf), act=f(self.act_buf),
                       ix=f(self.invw1[0]), iy=f(self.invw1[1]), iz=f(self.invw1[2]))
        self._active_flux = wp.vec3i(*[0 if (self.n[d] == 1 and self.bcs[(d, 0)] == "periodic") else 1
                                       for d in range(3)])
        self._active_sensor = wp.vec3i(*[1 if self.n[d] > 1 else 0 for d in range(3)])
        self._wp_ib = None
        if self.ib is not None:
            shape_np = tuple(int(v) for v in shape)
            gi = self.ib["g"].cpu().numpy()
            ci = self.ib["c"].cpu().numpy()
            g3 = np.stack(np.unravel_index(gi, shape_np), -1).astype(np.int32)
            c3 = np.stack(np.unravel_index(ci, shape_np), -1).astype(np.int32)
            self._wp_ib = dict(
                n=len(gi),
                ghost=wp.array(g3, dtype=wp.vec3i, device=self._wdev),
                corner=wp.array(c3, dtype=wp.vec3i, device=self._wdev),
                weight=wp.array(self.ib["w"].cpu().numpy().astype(np.float32), dtype=wp.float32,
                                device=self._wdev),
                normal=wp.array(self.ib["n"].cpu().numpy().T.astype(np.float32), dtype=wp.vec3,
                                device=self._wdev))
        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(dev)) if dev.type == "cuda" else None
        vp = K.Visc()
        v = self.viscous or {}
        vp.mode = 0 if not v else (1 if v["model"] == "constant" else 2)
        vp.mu0 = float(v.get("mu", 0.0))
        vp.T0 = float(v.get("T0", 273.15))
        vp.S = float(v.get("S", 110.4))
        vp.R = float(v.get("R", 287.05))
        vp.Pr = float(v.get("Pr", 0.72))
        self._centres = [torch.as_tensor(self.axes[d].astype(np.float32), device=dev) for d in range(3)]
        vp.cx, vp.cy, vp.cz = (wp.from_torch(c_) for c_ in self._centres)
        vp.turb = 1 if self.turbulence else 0
        shape3 = tuple(self.U.shape[1:])
        if self.turbulence:
            dist = np.where(self.phi > 0, self.phi, 0.0).astype(np.float32)
            dist[self.phi >= 1e8] = 1e3
            if getattr(self, "_thin_dist", None) is not None:
                sub, td = self._thin_dist
                dist[sub] = np.minimum(dist[sub], td)
            self.dist_buf = torch.as_tensor(np.maximum(dist, 1e-6), device=dev)
            self.fluid_buf = torch.as_tensor(self.fluid.astype(np.float32), device=dev)
            self.mut_buf = torch.zeros(shape3, dtype=self.dtype, device=dev)
            self._init_turbulence()
            for name in ("K1", "O1"):
                setattr(self, name + "_buf", torch.empty(shape3, dtype=self.dtype, device=dev))
            self.K1_buf.copy_(self.K_buf)
            self.O1_buf.copy_(self.O_buf)
            for name in ("Rk", "Rw", "Dk", "Dw", "utau"):
                setattr(self, name + "_buf", torch.zeros(self.n, dtype=self.dtype, device=dev))
            f = wp.from_torch
            self._w.update(K=f(self.K_buf), O=f(self.O_buf), K1=f(self.K1_buf), O1=f(self.O1_buf),
                           Rk=f(self.Rk_buf), Rw=f(self.Rw_buf), Dk=f(self.Dk_buf), Dw=f(self.Dw_buf),
                           utau=f(self.utau_buf))
        else:
            self.dist_buf = torch.ones((1, 1, 1), dtype=self.dtype, device=dev)
            self.fluid_buf = torch.ones((1, 1, 1), dtype=self.dtype, device=dev)
            self.mut_buf = torch.zeros((1, 1, 1), dtype=self.dtype, device=dev)
        vp.mut = wp.from_torch(self.mut_buf)
        vp.dist = wp.from_torch(self.dist_buf)
        vp.fluid = wp.from_torch(self.fluid_buf)
        self._vp = vp

    def _init_turbulence(self):
        """Freestream k and omega (NASA Turbulence Modeling Resource SST defaults) everywhere."""
        rho, p = self.W_inf[0], self.W_inf[4]
        a2 = self.gamma * p / rho
        v = self.viscous
        T = p / (rho * v["R"])
        mu = v["mu"] if v["model"] == "constant" else v["mu"] * (T / v["T0"]) ** 1.5 * (v["T0"] + v["S"]) / (T + v["S"])
        tb = self.turbulence
        self.k_inf = float(tb.get("k_inf", 9e-9 * a2))
        self.omega_inf = float(tb.get("omega_inf", 1e-6 * rho * a2 / mu))
        shape3 = tuple(self.U.shape[1:])
        if getattr(self, "K_buf", None) is None:
            self.K_buf = torch.empty(shape3, dtype=self.dtype, device=self.device)
            self.O_buf = torch.empty(shape3, dtype=self.dtype, device=self.device)
        self.K_buf.fill_(rho * self.k_inf)
        self.O_buf.fill_(rho * self.omega_inf)

    def _fill_bc_turb(self, Kt, Om):
        """k / omega ghost layers: freestream on inflow faces, zero gradient elsewhere."""
        n = self.n
        vel = self.W_inf[1:4]
        rho = self.W_inf[0]
        for d in range(3):
            N = n[d]
            for side in (0, 1):
                kind = self.bcs[(d, side)]
                for T, val in ((Kt, rho * self.k_inf), (Om, rho * self.omega_inf)):
                    dst = _sl(T, d, 0, NG) if side == 0 else _sl(T, d, N + NG, N + 2 * NG)
                    if kind == "periodic":
                        dst.copy_(_sl(T, d, N, N + NG) if side == 0 else _sl(T, d, NG, 2 * NG))
                        continue
                    into = (vel[d] > 1e-9) if side == 0 else (vel[d] < -1e-9)
                    if kind in ("freestream", "farfield") and into:
                        dst.fill_(val)
                    else:
                        edge = _sl(T, d, NG, NG + 1) if side == 0 else _sl(T, d, N + NG - 1, N + NG)
                        dst.copy_(edge.expand_as(dst))

    # ------------------------------------------------------------------ setup
    def _setup_bcs(self, bcs):
        self._bcs_arg = bcs
        vel = self.W_inf[1:4]
        self.bcs = {}
        for d in range(3):
            for side in (0, 1):
                kind = "auto" if bcs == "auto" else bcs.get((d, side), "auto")
                if kind == "auto":
                    c_inf = np.sqrt(self.gamma * self.W_inf[4] / self.W_inf[0])
                    if np.linalg.norm(vel) < c_inf:
                        # subsonic: waves run both ways, so fixing everything at the inflow and
                        # copying everything at the outflow lets the pressure level drift and slows
                        # the whole domain down; use characteristic (Riemann-invariant) farfield
                        kind = "farfield"
                    else:
                        into = vel[d] > 1e-9 if side == 0 else vel[d] < -1e-9
                        kind = "freestream" if into else "extrapolate"
                self.bcs[(d, side)] = kind

    def _setup_thin(self, thin_sdf, thin_box):
        """Blocked faces for thin parts: face (i -> i+1 along d) is a wall if either cell centre
        is inside the part, or the segment between the centres passes through it. The segment
        test samples phi at quarter points; with a distance function, phi(a) + phi(b) < |a - b|
        between consecutive samples means the part may lie between them (conservative)."""
        shape = self.phi.shape
        self.thin_solid = np.zeros(shape, bool)
        self.blk = np.zeros(shape, np.int32)
        self.has_thin = thin_sdf is not None
        if not self.has_thin:
            return
        r0, r1 = [], []
        for d in range(3):
            c = self.axes[d]
            if thin_box is None:
                a, b = 0, len(c) - 1
            else:
                a = max(int(np.searchsorted(c, thin_box[0][d])) - 3, 0)
                b = min(int(np.searchsorted(c, thin_box[1][d])) + 3, len(c) - 1)
            r0.append(a)
            r1.append(b)
        ax = [self.axes[d][r0[d]:r1[d] + 1] for d in range(3)]
        sub = tuple(slice(r0[d], r1[d] + 1) for d in range(3))
        G = np.meshgrid(*ax, indexing="ij")
        pc = thin_sdf(*G)
        ts = pc < 0
        self.thin_solid[sub] = ts
        self._thin_dist = (sub, np.abs(pc).astype(np.float32))
        for d in range(3):
            if len(ax[d]) < 2 or (self.n[d] == 1 and self.bcs[(d, 0)] == "periodic"):
                continue
            lo_sl = [slice(None)] * 3
            hi_sl = [slice(None)] * 3
            lo_sl[d] = slice(0, -1)
            hi_sl[d] = slice(1, None)
            lo_sl, hi_sl = tuple(lo_sl), tuple(hi_sl)
            Gs = [g_[lo_sl] for g_ in G]
            step = (G[d][hi_sl] - G[d][lo_sl])
            blocked = ts[lo_sl] | ts[hi_sl]
            # quarter-point samples along each face segment
            tq = (0.0, 0.25, 0.5, 0.75, 1.0)
            samples = [pc[lo_sl]]
            for t in tq[1:-1]:
                P = list(Gs)
                P[d] = Gs[d] + t * step
                samples.append(thin_sdf(*P))
            samples.append(pc[hi_sl])
            for sm in samples:
                blocked |= sm < 0
            # Sub-segments that the distance function cannot rule out (phi_a + phi_b < length)
            # are bisected until the part is found (phi < 0 somewhere) or the piece is short
            # enough to exclude it. The bare test over-blocks faces that run parallel to a plate
            # close by, e.g. on stretched cells next to a wall.
            flat_b = blocked.ravel()
            Gf = [g_.ravel() for g_ in Gs]
            stepf = step.ravel()
            fid, ta, tb, pa, pb = [], [], [], [], []
            for k in range(4):
                sa, sb = samples[k].ravel(), samples[k + 1].ravel()
                cand = (~flat_b) & (sa + sb < (tq[k + 1] - tq[k]) * stepf)
                idx_ = np.flatnonzero(cand)
                fid.append(idx_)
                ta.append(np.full(idx_.size, tq[k]))
                tb.append(np.full(idx_.size, tq[k + 1]))
                pa.append(sa[idx_])
                pb.append(sb[idx_])
            fid, ta, tb, pa, pb = map(np.concatenate, (fid, ta, tb, pa, pb))
            for _level in range(18):
                keep = ~flat_b[fid]
                fid, ta, tb, pa, pb = fid[keep], ta[keep], tb[keep], pa[keep], pb[keep]
                if fid.size == 0:
                    break
                tm = 0.5 * (ta + tb)
                P = [g_[fid] for g_ in Gf]
                P[d] = P[d] + tm * stepf[fid]
                pm = thin_sdf(*P)
                hit = pm < 0
                flat_b[fid[hit]] = True
                half = 0.5 * (tb - ta) * stepf[fid]
                c1 = (~hit) & (pa + pm < half)
                c2 = (~hit) & (pm + pb < half)
                fid = np.concatenate([fid[c1], fid[c2]])
                ta, tb = np.concatenate([ta[c1], tm[c2]]), np.concatenate([tm[c1], tb[c2]])
                pa, pb = np.concatenate([pa[c1], pm[c2]]), np.concatenate([pm[c1], pb[c2]])
            if fid.size:                       # still ambiguous after 18 halvings: block (conservative)
                flat_b[fid] = True
            blocked = flat_b.reshape(blocked.shape)
            tgt = tuple(slice(r0[e], r1[e] + 1) if e != d else slice(r0[e], r1[e]) for e in range(3))
            self.blk[tgt] |= (blocked.astype(np.int32) << d)
        if self.verbose:
            nb = [int(((self.blk >> d) & 1).sum()) for d in range(3)]
            print(f"thin walls: {int(ts.sum())} solid cells, blocked faces x/y/z = {nb}")

    def _setup_ib(self):
        phi = self.phi
        self.fluid = (phi > 0) & ~self.thin_solid
        solid = phi <= 0                    # thick (ghost-cell) solids only; thin parts use blocked faces
        cross = ndimage.generate_binary_structure(3, 1)
        near = ndimage.binary_dilation(self.fluid, structure=cross, iterations=NG)
        ghost = solid & near
        gi = np.flatnonzero(ghost)
        self.n_ghost = gi.size
        if gi.size == 0:
            self.ib = None
            return
        grads = np.gradient(phi, *self.axes)
        nrm = np.stack([gd.ravel()[gi] for gd in grads])
        nrm /= np.linalg.norm(nrm, axis=0) + 1e-30
        shape = phi.shape
        ijk = np.stack(np.unravel_index(gi, shape)).astype(float)
        xg = np.stack([self.axes[d][ijk[d].astype(int)] for d in range(3)])
        hloc = np.minimum.reduce([self.widths[d][ijk[d].astype(int)] for d in range(3)])
        dist0 = np.maximum(-2.0 * phi.ravel()[gi], 0.5 * hloc)
        fluid_f = self.fluid.ravel()

        cidx = np.zeros((gi.size, 8), np.int64)
        cw = np.zeros((gi.size, 8))
        todo = np.ones(gi.size, bool)
        for extra in (0.0, 1.0, 2.0, 3.0):
            if not todo.any():
                break
            sel = np.flatnonzero(todo)
            xi = xg[:, sel] + (dist0[sel] + extra * hloc[sel]) * nrm[:, sel]
            i0 = np.empty((3, sel.size), np.int64)
            fr = np.empty((3, sel.size))
            for d in range(3):
                c = self.axes[d]
                k = np.clip(np.searchsorted(c, xi[d]) - 1, 0, len(c) - 2)
                i0[d] = k
                fr[d] = np.clip((xi[d] - c[k]) / (c[k + 1] - c[k]), 0.0, 1.0)
            k = 0
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        ii = np.clip(i0[0] + a, 0, shape[0] - 1)
                        jj = np.clip(i0[1] + b, 0, shape[1] - 1)
                        kk = np.clip(i0[2] + c, 0, shape[2] - 1)
                        flat = np.ravel_multi_index((ii, jj, kk), shape)
                        w = ((fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1])
                             * (fr[2] if c else 1 - fr[2])) * fluid_f[flat]
                        cidx[sel, k], cw[sel, k] = flat, w
                        k += 1
            s = cw[sel].sum(1)
            ok = s > 1e-3
            cw[sel[ok]] /= s[ok, None]
            todo[sel[ok]] = False
        if todo.any():  # fall back to nearest fluid cell
            _, inds = ndimage.distance_transform_edt(~self.fluid, return_indices=True)
            sel = np.flatnonzero(todo)
            near_idx = np.ravel_multi_index(tuple(ind.ravel()[gi[sel]] for ind in inds), shape)
            cidx[sel] = near_idx[:, None]
            cw[sel] = 0.0
            cw[sel, 0] = 1.0
        self.n_ib_fallback = int(todo.sum())
        dev = self.device
        self.ib = dict(g=torch.as_tensor(gi, device=dev), c=torch.as_tensor(cidx, device=dev),
                       w=torch.as_tensor(cw, dtype=self.dtype, device=dev),
                       n=torch.as_tensor(nrm, dtype=self.dtype, device=dev))

    # ---------------------------------------------------------------- kernels
    def _fill_bc(self, W):
        """Fill the domain-boundary ghost layers in place."""
        n = self.n
        for d in range(3):
            ax = d + 1
            N = n[d]
            for side in (0, 1):
                kind = self.bcs[(d, side)]
                dst = _sl(W, ax, 0, NG) if side == 0 else _sl(W, ax, N + NG, N + 2 * NG)
                if kind == "periodic":
                    dst.copy_(_sl(W, ax, N, N + NG) if side == 0 else _sl(W, ax, NG, 2 * NG))
                elif kind == "farfield":
                    edge = _sl(W, ax, NG, NG + 1) if side == 0 else _sl(W, ax, N + NG - 1, N + NG)
                    dst.copy_(self._farfield(edge, d, -1.0 if side == 0 else 1.0).expand_as(dst))
                elif kind in ("wall", "slipwall"):
                    # mirror the first NG interior layers; no-slip negates all velocity
                    # components (zero velocity on the boundary face), slip only the normal one
                    src = _sl(W, ax, NG, 2 * NG) if side == 0 else _sl(W, ax, N, N + NG)
                    src = torch.flip(src, dims=(ax,)).clone()
                    if kind == "wall":
                        src[1:4] = -src[1:4]
                    else:
                        src[1 + d] = -src[1 + d]
                    dst.copy_(src)
                elif kind == "extrapolate":
                    edge = _sl(W, ax, NG, NG + 1) if side == 0 else _sl(W, ax, N + NG - 1, N + NG)
                    dst.copy_(edge.expand_as(dst))
                else:  # freestream
                    dst.copy_(self.W_inf_t.reshape(5, 1, 1, 1).expand_as(dst))
        return W

    def _farfield(self, Wi, d, sgn):
        """Characteristic farfield state from the edge cells Wi (5, ...) and the freestream,
        for a boundary with outward normal sgn * e_d (Riemann invariants of the normal flow)."""
        g = self.gamma
        Wf = self.W_inf_t.reshape(5, *([1] * (Wi.dim() - 1)))
        rho_i, p_i = Wi[0], Wi[4]
        c_i = torch.sqrt(g * p_i / rho_i)
        c_f = torch.sqrt(g * Wf[4] / Wf[0])
        un_i = sgn * Wi[1 + d]
        un_f = sgn * Wf[1 + d]
        Rp = un_i + 2.0 * c_i / (g - 1.0)              # leaves the domain: from the interior
        Rm = un_f - 2.0 * c_f / (g - 1.0)              # enters the domain: from the freestream
        un_b = 0.5 * (Rp + Rm)
        c_b = 0.25 * (g - 1.0) * (Rp - Rm)
        inflow = un_b < 0
        ent = torch.where(inflow, Wf[4] / Wf[0] ** g, p_i / rho_i ** g)
        rho_b = (c_b * c_b / (g * ent)) ** (1.0 / (g - 1.0))
        p_b = rho_b * c_b * c_b / g
        out = torch.empty_like(Wi)
        out[0], out[4] = rho_b, p_b
        for k in range(3):
            if k == d:
                out[1 + k] = sgn * un_b
            else:
                out[1 + k] = torch.where(inflow, Wf[1 + k].expand_as(Wi[1 + k]), Wi[1 + k])
        # supersonic normal flow: everything from upstream
        sup_in = un_f < -c_f
        sup_out = un_i > c_i
        out = torch.where(sup_out, Wi, out)
        out = torch.where(sup_in, Wf.expand_as(Wi), out)
        return out

    def _fill_ib(self, W):
        """Set immersed-boundary ghost cells in place (mirror state, normal velocity reflected)."""
        if self.ib is None:
            return W
        ib = self.ib
        Wf = W.view(5, -1)
        qi = (Wf[:, ib["c"]] * ib["w"]).sum(-1)            # (5, Ng)
        vel = qi[1:4]
        vn = (vel * ib["n"]).sum(0)
        vel = vel - 2.0 * vn * ib["n"]
        Wf[:, ib["g"]] = torch.cat([qi[:1], vel, qi[4:]], 0)
        return W

    def _residual(self, W, order):
        n, g = self.n, self.gamma
        p = W[4]
        s = torch.zeros_like(p)
        for d in range(3):
            if n[d] > 1:
                pp, pm = torch.roll(p, -1, d), torch.roll(p, 1, d)
                s = torch.maximum(s, (pp - pm).abs() / torch.minimum(pp, pm))
        shock = s > self.shock_threshold
        R = None
        for d in range(3):
            if n[d] == 1 and self.bcs[(d, 0)] == "periodic":
                continue
            ax = d + 1
            Wc, Sc = W, shock
            for o in range(3):
                if o != d:
                    Wc = _sl(Wc, o + 1, NG, NG + n[o])
                    Sc = _sl(Sc, o, NG, NG + n[o])
            N = n[d]
            qm1, q0, qp1, qp2 = (_sl(Wc, ax, k, N + 1 + k) for k in range(4))
            dC = qp1 - q0
            WL = q0 + (0.5 * order) * _limit(q0 - qm1, dC, self.limiter)
            WR = qp1 - (0.5 * order) * _limit(dC, qp2 - qp1, self.limiter)
            bad = (WL[0] <= 0) | (WL[4] <= 0) | (WR[0] <= 0) | (WR[4] <= 0)
            WL = torch.where(bad, q0, WL)
            WR = torch.where(bad, qp1, WR)
            hl = _sl(Sc, d, 1, N + 2) | _sl(Sc, d, 2, N + 3)
            F = hllc_flux(WL, WR, d, g, hl)
            div = (_sl(F, ax, 1, N + 1) - _sl(F, ax, 0, N)) * self.invw[d]
            R = -div if R is None else R - div
        return R

    def _prim(self, U):
        W = cons_to_prim(U, self.gamma, self.rho_min, self.p_min)
        return self._fill_ib(self._fill_bc(W))

    def _interior(self, A):
        return A[:, NG:-NG, NG:-NG, NG:-NG]

    # ------------------------------------------------------------ warp path
    def _wprim(self, Ukey, tkey=None):
        wp, K, w = self._wp, self._K, self._w
        shape = tuple(self.U.shape[1:])
        wp.launch(K.prim_kernel, dim=shape, inputs=[w[Ukey], self.gamma, self.rho_min, self.p_min, w["W"]],
                  device=self._wdev, stream=self._wp_stream)
        self._fill_bc(self.W_buf)
        ib = self._wp_ib
        if ib is not None:
            wp.launch(K.ib_kernel, dim=ib["n"], inputs=[w["W"], ib["ghost"], ib["corner"], ib["weight"],
                                                       ib["normal"], int(self.wall == "noslip")],
                      device=self._wdev, stream=self._wp_stream)
        wp.launch(K.shock_kernel, dim=shape, inputs=[w["W"], self._active_sensor, self.shock_threshold, w["S"]],
                  device=self._wdev, stream=self._wp_stream)
        if self.turbulence:
            kk, oo = (("K", "O") if tkey in (None, "") else (tkey, "O1"))
            self._fill_bc_turb(getattr(self, kk + "_buf"), getattr(self, oo + "_buf"))
            wp.launch(K.mut_kernel, dim=shape, inputs=[w["W"], w[kk], w[oo], self._vp, self._vp.mut],
                      device=self._wdev, stream=self._wp_stream)

    def _wturb(self, kk, oo, outk, outo, a0, a1, local):
        """SST residual on stage state (kk, oo) and point-implicit update into (outk, outo)."""
        wp, K, w = self._wp, self._K, self._w
        wp.launch(K.turb_residual_kernel, dim=self.n,
                  inputs=[w["W"], w[kk], w[oo], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"], self._vp,
                          w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["utau"]], device=self._wdev, stream=self._wp_stream)
        wp.launch(K.turb_update_kernel, dim=self.n,
                  inputs=[w["K"], w["O"], w[kk], w[oo], w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["dt"], w["dtl"],
                          int(local), self._vp, w["W"], w["utau"], a0, a1, 1e-6 * self.k_inf, 1e-3 * self.omega_inf,
                          w[outk], w[outo]], device=self._wdev, stream=self._wp_stream)

    def _wresidual(self, order, tiles=None):
        wp, K, w = self._wp, self._K, self._w
        kind = 1 if self.limiter == "minmod" else 0
        args = [w["W"], w["S"], w["ix"], w["iy"], w["iz"], self._active_flux, self.gamma, float(order), kind, w["blk"],
                self._vp]
        if tiles is None:
            wp.launch(K.residual_kernel, dim=self.n, inputs=args + [w["R"]],
                      device=self._wdev, stream=self._wp_stream)
        else:
            wp.launch(K.residual_tiles_kernel, dim=tiles.shape[0] * 512, inputs=args + [tiles, w["R"]],
                      device=self._wdev, stream=self._wp_stream)

    def _wrk(self, Ua, a0, a1, out, local, tiles=None):
        wp, K, w = self._wp, self._K, self._w
        args = [w["U"], w[Ua], w["R"], w["dt"], w["dtl"], int(local), w["mask"], a0, a1]
        if tiles is None:
            wp.launch(K.rk_kernel, dim=self.n, inputs=args + [w[out]], device=self._wdev, stream=self._wp_stream)
        else:
            wp.launch(K.rk_tiles_kernel, dim=tiles.shape[0] * 512, inputs=args + [tiles, w[out]],
                      device=self._wdev, stream=self._wp_stream)

    def _step_warp(self, order, want_res=False, local=False, tiles=None):
        """SSP-RK2 step. local: per-cell time steps (steady runs only). tiles: update only
        these 8x8x8 tiles (active-region updating, steady runs only)."""
        wp, K, w = self._wp, self._K, self._w
        self._wprim("U")
        wp.launch(K.rate_kernel, dim=self.n, inputs=[w["W"], w["ix"], w["iy"], w["iz"], self._active_flux,
                                                     w["mask"], self.gamma, self._vp, w["rate"]],
                  device=self._wdev, stream=self._wp_stream)
        torch.div(self.cfl, self.rate_buf.max(), out=self.dt_buf[0])
        if local:
            torch.div(self.cfl, torch.clamp(self.rate_buf, min=1e-30), out=self.dtl_buf)
        elif tiles is None and getattr(self, "_tiles_on", False):
            self.dtl_buf.copy_(self.dt_buf[0].expand_as(self.dtl_buf))   # for the activity measure
        self._wresidual(order, tiles)
        res = torch.sqrt(((self.R_buf[0] * self.mask_f) ** 2).sum() / self.n_fluid) if want_res else None
        self._wrk("U", 0.0, 1.0, "U1", local, tiles)
        if self.turbulence:
            self._wturb("K", "O", "K1", "O1", 0.0, 1.0, local)
        self._wprim("U1", "K1" if self.turbulence else None)
        self._wresidual(order, tiles)
        self._wrk("U1", 0.5, 0.5, "U", local, tiles)
        if self.turbulence:
            self._wturb("K1", "O1", "K", "O", 0.5, 0.5, local)
        return self.dt_buf[0].clone(), res

    def _refresh_tiles(self, frac):
        """Active tiles: those whose largest relative density change per step exceeds `frac`
        times the largest anywhere, plus one tile around them (so changes can propagate)."""
        wp, K, w = self._wp, self._K, self._w
        self.act_buf.zero_()
        wp.launch(K.activity_kernel, dim=self.n, inputs=[w["R"], w["W"], w["dtl"], w["mask"], w["act"]],
                  device=self._wdev, stream=self._wp_stream)
        a = self.act_buf
        on = (a > frac * a.max()).to(self.dtype)[None, None]
        on = torch.nn.functional.max_pool3d(on, 3, stride=1, padding=1)[0, 0] > 0
        idx = torch.nonzero(on).to(torch.int32).contiguous()
        self._tiles_t = idx                      # keep the tensor alive for the warp view
        return wp.from_torch(idx, dtype=wp.vec3i) if idx.shape[0] else None, idx.shape[0] / on.numel()

    # ----------------------------------------------------------- torch path
    def _step(self, order, want_res=True):
        """One SSP-RK2 step. Returns (dt, density residual) as device tensors (no host sync)."""
        if self.backend == "warp":
            return self._step_warp(order, want_res)
        g, m = self.gamma, self.update_mask
        U = self.U
        W = self._prim(U)
        Wi = self._interior(W)
        c = torch.sqrt(g * Wi[4] / Wi[0])
        rate = sum((Wi[1 + d].abs() + c) * self.invw[d] for d in range(3) if self.n[d] > 1)
        dt = self.cfl / torch.where(m, rate, torch.zeros_like(rate)).max()
        R0 = self._residual(W, order)
        Ui = self._interior(U)
        U1 = U.clone()
        self._interior(U1).copy_(torch.where(m, Ui + dt * R0, Ui))
        R1 = self._residual(self._prim(U1), order)
        U1i = self._interior(U1)
        U2i = torch.where(m, 0.5 * Ui + 0.5 * (U1i + dt * R1), Ui)
        Ui.copy_(U2i)
        res = torch.sqrt(((R0[0] * m) ** 2).sum() / self.n_fluid)
        return dt, res

    # -------------------------------------------------------------- driver
    def run(self, nsteps, chunk=100, first_order_steps=200, report=None, steady=False,
            active_tiles=None, tile_refresh=50, tile_frac=1e-3, local_dt=False, snapshot=None):
        """Advance nsteps time steps. `report(solver)` is called after every chunk; it can
        return True to stop.

        steady=True (warp backend) turns on the steady-state accelerations, which change the
        path to the solution but not the converged answer:
          * active tiles (default on when steady): every `tile_refresh` steps a full step
            measures how fast each 8x8x8 tile is changing; until the next refresh only tiles
            changing faster than `tile_frac` x the fastest (plus a one-tile margin) are updated;
          * local_dt=True: every cell advances at its own stable step (off by default: on the
            stretched grids tested it slowed convergence; worth trying with strong stretching).
        Time-accurate runs (steady=False) always update every cell with one global step.
        snapshot = (every_steps, fn): fn(step, t) is called every `every_steps` steps.
        """
        steady = steady and self.backend == "warp"
        if active_tiles is None:
            active_tiles = steady
        active_tiles = active_tiles and steady
        if self.turbulence and active_tiles:
            active_tiles = False                # the turbulence kernels always cover the whole grid
        self._tiles_on = active_tiles
        local = steady and local_dt
        done = 0
        t0 = time.time()
        work = 0.0                       # cell updates actually done, in units of full steps
        tiles = None
        self.active_fraction = 1.0
        with torch.no_grad():
            while done < nsteps:
                k = min(chunk, nsteps - done)
                order = 0.0 if self.step_count < first_order_steps else 1.0
                dt_sum = torch.zeros((), dtype=self.dtype, device=self.device)
                for kk in range(k):
                    if self.backend == "warp":
                        full = (not active_tiles) or ((self.step_count + kk) % tile_refresh == 0) \
                            or self.step_count + kk < first_order_steps
                        dt, r = self._step_warp(order, want_res=(kk == k - 1), local=local,
                                                tiles=None if full else tiles)
                        if active_tiles and full and self.step_count + kk >= first_order_steps:
                            tiles, self.active_fraction = self._refresh_tiles(tile_frac)
                            if tiles is None:
                                tiles, self.active_fraction = None, 1.0
                        work += 1.0 if (full or tiles is None) else self.active_fraction
                    else:
                        dt, r = self._step(order, want_res=(kk == k - 1))
                        work += 1.0
                    if r is not None:
                        res = r
                    dt_sum += dt
                    if snapshot is not None and (self.step_count + kk + 1) % snapshot[0] == 0:
                        snapshot[1](self.step_count + kk + 1, self.t + float(dt_sum))
                res = float(res)              # one host sync per chunk
                self.t += float(dt_sum)
                self.step_count += k
                done += k
                self.history.append((self.step_count, self.t, res))
                if not np.isfinite(res):
                    raise FloatingPointError(f"solution diverged at step {self.step_count}")
                if self.verbose:
                    el = time.time() - t0
                    cps = np.prod(self.n) * done / el / 1e6
                    act = f"  active {100*self.active_fraction:.0f}%" if active_tiles else ""
                    mem = (f"  GPU mem {torch.cuda.max_memory_allocated(self.device)/2**30:.2f} GB"
                           if self.device.type == "cuda" else "")
                    print(f"step {self.step_count:6d}  t = {self.t:.4e} s  "
                          f"res(rho) = {res/self.fs['rho']:.3e} 1/s  [{cps:.1f} Mcell-steps/s]{mem}{act}",
                          flush=True)
                if report and report(self):
                    break                      # the report callback asked to stop (converged)
        self.work_full_steps = work
        return self

    def set_freestream(self, freestream, keep_state=False):
        """Change the flight condition without rebuilding the grid, geometry or wall data
        (all of which are independent of the freestream). keep_state=True warm-starts from
        the current solution instead of a uniform field."""
        self.fs = freestream
        self.W_inf = np.array([freestream[k] for k in ("rho", "u", "v", "w", "p")], float)
        self.rho_min = 1e-4 * freestream["rho"]
        self.p_min = 1e-4 * freestream["p"]
        self._setup_bcs(self._bcs_arg)
        self.W_inf_t.copy_(torch.as_tensor(self.W_inf, dtype=self.dtype, device=self.device))
        if not keep_state:
            shape = self.U.shape[1:]
            self.U.copy_(prim_to_cons(self.W_inf_t.reshape(5, 1, 1, 1).expand((5,) + tuple(shape)), self.gamma))
            if self.turbulence:
                self._init_turbulence()
        if self.backend == "warp":
            self.U1.copy_(self.U)
        self.t = 0.0
        self.step_count = 0
        self.history = []

    def set_primitive(self, W):
        """Set the state from a primitive-variable array (5, nx+4, ny+4, nz+4) (numpy or tensor)."""
        W = torch.as_tensor(np.asarray(W), dtype=self.dtype, device=self.device)
        self.U.copy_(prim_to_cons(W, self.gamma))       # in place: warp views stay valid
        if self.backend == "warp":
            self.U1.copy_(self.U)

    # ------------------------------------------------------------- outputs
    def primitive(self):
        """Primitive variables on the padded grid (numpy), ghosts filled."""
        with torch.no_grad():
            return self._prim(self.U).cpu().numpy().astype(np.float64)

    def fields(self):
        out = self._fields()
        if self.turbulence:
            with torch.no_grad():
                if self.backend == "warp":
                    self._wprim("U")
                sl = (slice(NG, -NG),) * 3
                rho = out["rho"]
                out["k"] = (self.K_buf[sl].double().cpu().numpy()) / rho
                out["omega"] = (self.O_buf[sl].double().cpu().numpy()) / rho
                out["mut"] = self.mut_buf[sl].double().cpu().numpy()
        return out

    def _fields(self):
        W = self.primitive()[:, NG:-NG, NG:-NG, NG:-NG]
        rho, u, v, w, p = W
        c = np.sqrt(self.gamma * p / rho)
        mach = np.sqrt(u * u + v * v + w * w) / c
        solid = ~self.fluid[NG:-NG, NG:-NG, NG:-NG]
        return dict(rho=rho, u=u, v=v, w=w, p=p, mach=mach, solid=solid,
                    phi=self.phi[NG:-NG, NG:-NG, NG:-NG],
                    x=self.axes[0][NG:-NG], y=self.axes[1][NG:-NG], z=self.axes[2][NG:-NG])

    def loads(self, S_ref, L_ref, x_ref=(0.0, 0.0, 0.0), band=1.5, parts=None):
        """Integrate surface pressure: -∮ (p - p_inf) n dA via a smoothed delta function.
        Returns body-axis force/moment coefficients and wind-axis drag/lift.
        The surface band and its weights are computed once; each call only gathers the
        pressure in the band (a few thousand values) from the device."""
        sf = self._surface(band)
        tf = sf.get("thin")
        with torch.no_grad():
            if self.backend == "warp":
                self._wprim("U")
                Wsrc = self.W_buf
            else:
                Wsrc = self._prim(self.U)
            pflat = Wsrc[4].reshape(-1)
            pb = pflat[sf["flat_t"]]
            if tf is not None:
                pa = pflat[tf["a_t"]].double().cpu().numpy()
                pbb = pflat[tf["b_t"]].double().cpu().numpy()
        dp = pb.double().cpu().numpy() - self.W_inf[4]
        dF = sf["wF"] * dp                                  # (3, Nb)
        F_ib = dF.sum(1)
        r = sf["pos"] - np.asarray(x_ref, float)[:, None]
        M_ib = np.cross(r.T, dF.T).sum(0)
        F_thin = np.zeros(3)
        M_thin = np.zeros(3)
        if tf is not None:
            pinf = self.W_inf[4]
            mag = ((pa - pinf) * tf["fa"] - (pbb - pinf) * tf["fb"]) * tf["area"]
            dFt = np.zeros((3, mag.size))
            dFt[tf["dir"], np.arange(mag.size)] = mag      # the a side pushes the wall toward +d
            F_thin = dFt.sum(1)
            rt = tf["pos"] - np.asarray(x_ref, float)[:, None]
            M_thin = np.cross(rt.T, dFt.T).sum(0)
        F = F_ib + F_thin
        M = M_ib + M_thin
        part_out = {}
        for name, spec in (parts or {}).items():
            if callable(spec):                     # spec(pos) -> mask, pos is (3, N)
                sel_of = spec
            else:
                axis, lo_, hi_ = spec
                sel_of = lambda pos, axis=axis, lo_=lo_, hi_=hi_: (pos[axis] >= lo_) & (pos[axis] < hi_)
            sel = sel_of(sf["pos"])
            Fp = dF[:, sel].sum(1)
            Mp = np.cross(r[:, sel].T, dF[:, sel].T).sum(0)
            if tf is not None:
                selt = sel_of(tf["pos"])
                Fp = Fp + dFt[:, selt].sum(1)
                Mp = Mp + np.cross(rt[:, selt].T, dFt[:, selt].T).sum(0)
            part_out[name] = (Fp, Mp)
        V = self.W_inf[1:4]
        Vm = np.linalg.norm(V)
        q = 0.5 * self.W_inf[0] * Vm ** 2
        Cf = F / (q * S_ref)
        Cm = M / (q * S_ref * L_ref)
        e_d = V / Vm
        ex = np.array([1.0, 0.0, 0.0])
        # crossflow direction: the part of the freestream normal to the body axis
        cross = e_d - (e_d @ ex) * ex
        if np.linalg.norm(cross) < 1e-9:
            cross = np.array([0.0, 0.0, 1.0])
        cross /= np.linalg.norm(cross)
        side = np.cross(ex, cross)
        lift_dir = cross - (cross @ e_d) * e_d          # perpendicular to V, in the crossflow plane
        lift_dir /= np.linalg.norm(lift_dir)
        # axis about which a positive moment turns the body axis toward the freestream
        restore_axis = np.cross(ex, e_d)
        restore_axis = restore_axis / np.linalg.norm(restore_axis) if np.linalg.norm(restore_axis) > 1e-9 \
            else -side
        qs = q * S_ref
        parts_c = {k: dict(CD=float(v[0] @ e_d / qs), CN=float(v[0] @ cross / qs),
                           C_restore=float(v[1] @ restore_axis / (qs * L_ref)),
                           CX=float(v[0][0] / qs), CY=float(v[0][1] / qs), CZ=float(v[0][2] / qs))
                   for k, v in part_out.items()}
        return dict(F=F, M=M, F_thin=F_thin, M_thin=M_thin, q_inf=q, parts=parts_c,
                    C_thin=F_thin / (q * S_ref), Cm_thin=M_thin / (q * S_ref * L_ref),
                    CX=Cf[0], CY=Cf[1], CZ=Cf[2],
                    Cl=Cm[0], Cm=Cm[1], Cn=Cm[2], CD=Cf @ e_d, CL=Cf @ lift_dir,
                    CN=Cf @ cross, CS=Cf @ side, C_roll=Cm[0], C_restore=Cm @ restore_axis)

    def _surface(self, band):
        key = ("surface", band)
        if getattr(self, "_surf_key", None) != key:
            phi = self.phi
            wx, wy, wz = self.widths
            hloc = np.minimum(np.minimum(wx[:, None, None], wy[None, :, None]), wz[None, None, :])
            eps = band * hloc
            idx = np.nonzero(np.abs(phi) < eps)
            e = eps[idx]
            delta = (1 + np.cos(np.pi * phi[idx] / e)) / (2 * e)
            # gradient of phi only at band cells (central differences on the padded grid)
            g = []
            for d in range(3):
                c = self.axes[d]
                ip = [i.copy() for i in idx]
                im = [i.copy() for i in idx]
                ip[d] = np.minimum(idx[d] + 1, phi.shape[d] - 1)
                im[d] = np.maximum(idx[d] - 1, 0)
                g.append((phi[tuple(ip)] - phi[tuple(im)]) / (c[ip[d]] - c[im[d]]))
            g = np.stack(g)
            g /= np.linalg.norm(g, axis=0) + 1e-30
            dV = wx[idx[0]] * wy[idx[1]] * wz[idx[2]]
            flat = np.ravel_multi_index(idx, phi.shape)
            self._surf = dict(wF=-(delta * dV) * g,
                              pos=np.stack([self.axes[d][idx[d]] for d in range(3)]),
                              flat_t=torch.as_tensor(flat, device=self.device))
            if self.has_thin:
                interior = np.zeros(phi.shape, bool)
                interior[NG:-NG, NG:-NG, NG:-NG] = True
                A_, B_, D_, AR, PS, FA, FB = [], [], [], [], [], [], []
                for d in range(3):
                    ia = np.nonzero(((self.blk >> d) & 1).astype(bool))
                    ib_ = list(ia)
                    ib_[d] = ia[d] + 1
                    ib_ = tuple(ib_)
                    fa = self.fluid[ia] & interior[ia]
                    fb = self.fluid[ib_] & interior[ib_]
                    keep = fa | fb
                    ia = tuple(v[keep] for v in ia)
                    ib_ = tuple(v[keep] for v in ib_)
                    o = [e for e in range(3) if e != d]
                    area = self.widths[o[0]][ia[o[0]]] * self.widths[o[1]][ia[o[1]]]
                    pos = np.stack([self.axes[e][ia[e]] for e in range(3)]).astype(float)
                    pos[d] += 0.5 * self.widths[d][ia[d]]
                    A_.append(np.ravel_multi_index(ia, phi.shape))
                    B_.append(np.ravel_multi_index(ib_, phi.shape))
                    D_.append(np.full(area.size, d))
                    AR.append(area)
                    PS.append(pos)
                    FA.append(fa[keep].astype(float))
                    FB.append(fb[keep].astype(float))
                self._surf["thin"] = dict(a_t=torch.as_tensor(np.concatenate(A_), device=self.device),
                                          b_t=torch.as_tensor(np.concatenate(B_), device=self.device),
                                          dir=np.concatenate(D_), area=np.concatenate(AR),
                                          pos=np.concatenate(PS, axis=1), fa=np.concatenate(FA),
                                          fb=np.concatenate(FB))
            self._surf_key = key
        return self._surf

    def save(self, path):
        np.savez_compressed(path, U=self.U.cpu().numpy(), t=self.t, step=self.step_count,
                            fx=self.faces[0], fy=self.faces[1], fz=self.faces[2],
                            history=np.array(self.history))

    def load(self, path):
        d = np.load(path)
        self.U.copy_(torch.as_tensor(d["U"], dtype=self.dtype, device=self.device))
        if self.backend == "warp":
            self.U1.copy_(self.U)
        self.t, self.step_count = float(d["t"]), int(d["step"])
        self.history = [tuple(r) for r in d["history"]]
