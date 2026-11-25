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


_CPU_WARNED = False


def default_device():
    global _CPU_WARNED
    if os.environ.get("ROCKETCFD_DEVICE"):
        return torch.device(os.environ["ROCKETCFD_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not _CPU_WARNED:
        _CPU_WARNED = True
        gpu = False
        try:
            import shutil
            import subprocess
            if shutil.which("nvidia-smi"):
                gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).returncode == 0
        except Exception:                                          # noqa: BLE001
            pass
        if gpu:
            print(f"WARNING: an NVIDIA GPU is present but this PyTorch ({torch.__version__}) cannot use CUDA, so "
                  "rocketcfd runs on the CPU. Reinstall the CUDA build:\n"
                  "  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126",
                  flush=True)
    return torch.device("cpu")


def _sl(a, ax, start, stop):
    return a.narrow(ax, start, stop - start)


def _limit(a, b, kind):
    """Limited slope; a = difference away from the face, b = across it (matters for Koren)."""
    if kind == "minmod":
        return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))
    if kind == "koren":
        aa, bb = a.abs(), b.abs()
        v = torch.sign(a) * torch.minimum(torch.minimum(2.0 * bb, (aa + 2.0 * bb) / 3.0), 2.0 * aa)
        return torch.where(a * b > 0, v, torch.zeros_like(a))
    ab = a * b
    return torch.where(ab > 0, 2.0 * ab / (a + b + (ab <= 0).to(a.dtype)), torch.zeros_like(a))  # van Leer


class Solver:
    def __init__(self, lo, hi, h, freestream, sdf=None, gamma=1.4, bcs="auto", cfl=0.7,
                 limiter="koren", shock_threshold=0.5, dtype=torch.float32, device=None,
                 verbose=True, faces=None, backend="auto", thin_sdf=None, thin_box=None,
                 viscous=None, wall=None, turbulence=None, alloc=None, thin_normals=True, low_mach="auto",
                 jets=None):
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
                     Or dict(model='les', sgs='none'|'vreman', cv=0.07, wall_model=True): large-eddy
                     simulation for time-accurate runs. sgs='none' (default): implicit LES, the
                     upwind scheme's own dissipation acts as the subgrid model; 'vreman': Vreman (2004)
                     eddy viscosity on top (tests/taylor_green.py, 32^3: dissipation peak at t 7.0
                     without, 5.5 with it, DNS 9.0: the model over-dissipates on coarse grids with
                     this scheme). With wall_model the
                     equilibrium wall model of wall-modelled LES (Bose & Park 2018): the wall shear
                     from Spalding's law with the instantaneous velocity of the first cell, as the
                     RANS wall function. wall_model=False: wall-resolved LES (no-slip walls).
        limiter    : 'koren' (default: kappa = 1/3 MUSCL, third-order upwind-biased where smooth;
                     tests/vortex_order.py), 'vanleer' (the earlier default) or 'minmod'
        low_mach   : Thornber et al. (2008) low-Mach correction of the reconstructed velocities (less
                     dissipation in slow flow; tests/low_mach_cylinder.py: Mach 0.1 cylinder drag +0.010
                     vs -0.037 without, rear stagnation Cp 1.07 vs 0.69). 'auto' (default) = on for a
                     subsonic freestream only: behind strong shocks it costs ~2 % stagnation pressure
        jets       : list of rocket-engine exits, each dict(center=(x, y, z), direction=(dx, dy, dz) (the
                     way the exhaust leaves), radius, mach, p [Pa], T [K]): the solid cells within
                     ~2.5 cells behind each exit disk hold the exhaust state, so the plume is part of the
                     flow (landing burns, hot staging, retropropulsion). One perfect gas (the freestream
                     gamma and R): choose T so the exit velocity mach * sqrt(gamma R T) is right.
                     self.jet_throttle (0..1, or a list with one value per jet) blends from ambient
                     (ignition transients). Warp backend.
        thin_normals : thin-wall faces use the real plate normal for the wall pressure (default);
                     False = axis-aligned mirror (staircase), the earlier behaviour
        alloc      : optional callable(name, shape) -> tensor for the state arrays (U, W, K, O);
                     used by rocketcfd.patches to place several grids in one buffer
        """
        self._alloc = alloc or (lambda name, shape: torch.empty(shape, dtype=dtype, device=self.device))
        self.thin_normals = thin_normals
        self.jets = list(jets) if jets else []
        self.jet_throttle = 1.0
        self.low_mach = low_mach
        # implicit relaxation of the mean flow: 'rb' red-black Gauss-Seidel (default: same
        # convergence as 8 Jacobi sweeps at about half the cost, tests/steady_accel.py) or 'jacobi'
        self.relax = "rb"
        # residual as face-flux kernels + divergence (each face once, far fewer registers on
        # the GPU) instead of the fused one-thread-per-cell kernel
        self.split_residual = True
        self.face_scratch = None    # optional callable(shape) -> shared scratch tensor (patches)
        self.cache_lam = True       # implicit sweeps read cached spectral radii (lam_kernel)
        # implicit turbulent runs: pseudo-time CFL cap in wall-function cells (None = off)
        self.cfl_wall = None
        self._iface = None          # interface ghost-cell fill (rocketcfd.patches)
        self.owned = None           # cells whose surface loads this grid reports (rocketcfd.patches)
        self.gamma, self.cfl, self.limiter = gamma, cfl, limiter
        self.viscous = None
        if viscous is not None:
            v = dict(model="sutherland", mu=1.716e-5, T0=273.15, S=110.4, R=287.05, Pr=0.72)
            v.update(viscous)
            self.viscous = v
        self.turbulence = None          # RANS (k-omega SST) settings
        self.les = None                 # LES settings
        if turbulence is not None:
            if self.viscous is None:
                raise ValueError("turbulence needs viscous=...")
            if turbulence.get("model", "sst") == "les":
                self.les = dict(model="les", sgs="none", cv=0.07, wall_model=True)
                self.les.update(turbulence)
            else:
                self.turbulence = dict(model="sst")
                self.turbulence.update(turbulence)
        # wall shear from the law of the wall (RANS wall function / LES equilibrium wall model)
        self.wallmodel = bool(self.turbulence) or bool(self.les and self.les["wall_model"])
        # With the wall function the wall shear comes from the law of the wall, so immersed-wall
        # ghost cells mirror the velocity as a slip wall (a no-slip mirror would put a spurious
        # steep gradient into the first cell's strain rate and turbulence production).
        self.wall = wall or ("slip" if self.wallmodel else "noslip" if self.viscous else "slip")
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
        self.U = self._alloc("U", (5,) + shape)
        self.U.copy_(prim_to_cons(W0, gamma))
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
        if (backend == "warp" or self.has_thin or self.viscous or self.wall != "slip" or self.turbulence
                or self.les) and not ok:
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
        self.W_buf = self._alloc("W", tuple(self.U.shape))
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
                                device=self._wdev),
                ratio=wp.array(self.ib["r"].cpu().numpy().astype(np.float32), dtype=wp.float32,
                               device=self._wdev))
        self._wp_jets = self._setup_jets(wp, K) if self.jets else None
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
        # turb: 0 laminar, 1 eddy viscosity + law-of-the-wall shear (SST, wall-modelled LES),
        # 2 eddy viscosity with no-slip walls (wall-resolved LES)
        vp.turb = 1 if self.wallmodel else (2 if self.les else 0)
        shape3 = tuple(self.U.shape[1:])
        if self.turbulence or self.les:
            dist = np.where(self.phi > 0, self.phi, 0.0).astype(np.float32)
            dist[self.phi >= 1e8] = 1e3
            if getattr(self, "_thin_dist", None) is not None:
                sub, td = self._thin_dist
                dist[sub] = np.minimum(dist[sub], td)
            self.dist_buf = torch.as_tensor(np.maximum(dist, 1e-6), device=dev)
            self.fluid_buf = torch.as_tensor(self.fluid.astype(np.float32), device=dev)
            self.mut_buf = torch.zeros(shape3, dtype=self.dtype, device=dev)
        if self.turbulence:
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
        if not (self.turbulence or self.les):
            self.dist_buf = torch.ones((1, 1, 1), dtype=self.dtype, device=dev)
            self.fluid_buf = torch.ones((1, 1, 1), dtype=self.dtype, device=dev)
            self.mut_buf = torch.zeros((1, 1, 1), dtype=self.dtype, device=dev)
        vp.mut = wp.from_torch(self.mut_buf)
        vp.dist = wp.from_torch(self.dist_buf)
        vp.fluid = wp.from_torch(self.fluid_buf)
        self._vp = vp

    def _setup_jets(self, wp, K):
        """Cells that hold each engine's exhaust state (see the `jets` argument)."""
        cells, states, groups = [], [], []
        g = self.gamma
        self.jet_cells = []
        for ji, jt in enumerate(self.jets):
            c = np.asarray(jt["center"], float)
            d = np.asarray(jt["direction"], float)
            d = d / np.linalg.norm(d)
            r = float(jt["radius"])
            Rg = float(jt.get("R", (self.viscous or {}).get("R", 287.05)))
            # cell size seen along the jet axis at the exit, and a box around the exit disk
            ic = [int(np.clip(np.searchsorted(self.axes[k], c[k]), 0, len(self.axes[k]) - 1)) for k in range(3)]
            hd = float(np.sqrt(sum((d[k] * self.widths[k][ic[k]]) ** 2 for k in range(3))))
            depth = float(jt.get("depth", 2.5 * hd))
            ext = r + depth + 2 * max(self.widths[k][ic[k]] for k in range(3))
            rng = [np.flatnonzero(np.abs(self.axes[k] - c[k]) <= ext) for k in range(3)]
            if any(len(q) == 0 for q in rng):
                self.jet_cells.append(0)
                continue
            X, Y, Z = np.meshgrid(*[self.axes[k][rng[k]] for k in range(3)], indexing="ij")
            P = np.stack([X - c[0], Y - c[1], Z - c[2]], -1)
            sa = P @ d
            rad = np.linalg.norm(P - sa[..., None] * d, axis=-1)
            ph = self.phi[np.ix_(*rng)]
            sel = (sa < 0.0) & (sa > -depth) & (rad <= r) & (ph <= 0)
            ii = np.argwhere(sel)
            idx = np.stack([rng[k][ii[:, k]] for k in range(3)], -1)
            T, pe, M = float(jt["T"]), float(jt["p"]), float(jt["mach"])
            ce = np.sqrt(g * Rg * T)
            u = M * ce * d
            q = np.array([pe / (Rg * T), u[0], u[1], u[2], pe], np.float32)
            cells.append(idx)
            states.append(np.repeat(q[None], len(idx), 0))
            groups.append(np.full(len(idx), ji, np.int32))
            self.jet_cells.append(len(idx))
        if not cells or sum(len(cc) for cc in cells) == 0:
            return None
        cells = np.concatenate(cells).astype(np.int32)
        states = np.concatenate(states).astype(np.float32)
        self._jet_thr = torch.ones(len(self.jets), dtype=torch.float32, device=self.device)
        self._jet_thr_np = np.ones(len(self.jets), np.float32)
        return dict(n=len(cells), cells=wp.array(cells, dtype=wp.vec3i, device=self._wdev),
                    state=wp.array(states, dtype=K.vec5, device=self._wdev),
                    group=wp.array(np.concatenate(groups), dtype=wp.int32, device=self._wdev),
                    ambient=K.vec5(*[float(v) for v in self.W_inf]), throttle=wp.from_torch(self._jet_thr))

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
            self.K_buf = self._alloc("K", shape3)
            self.O_buf = self._alloc("O", shape3)
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
                    if kind == "interface":
                        continue
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
                # a segment hugging a plate surface stays ambiguous at every level and the
                # candidates double each time (2^18 per face: tens of GB); below 1 % of the
                # cell spacing it passes within that distance of the plate: block it
                tiny = half < 0.01 * np.abs(stepf[fid])
                flat_b[fid[tiny & (c1 | c2)]] = True
                c1 &= ~tiny
                c2 &= ~tiny
                fid = np.concatenate([fid[c1], fid[c2]])
                ta, tb = np.concatenate([ta[c1], tm[c2]]), np.concatenate([tm[c1], tb[c2]])
                pa, pb = np.concatenate([pa[c1], pm[c2]]), np.concatenate([pm[c1], pb[c2]])
            if fid.size:                       # still ambiguous after 18 halvings: block (conservative)
                flat_b[fid] = True
            blocked = flat_b.reshape(blocked.shape)
            tgt = tuple(slice(r0[e], r1[e] + 1) if e != d else slice(r0[e], r1[e]) for e in range(3))
            self.blk[tgt] |= (blocked.astype(np.int32) << d)
        if self.thin_normals:
            self._pack_thin_normals(thin_sdf)
        if self.verbose:
            nb = [int(((self.blk >> d) & 1).sum()) for d in range(3)]
            print(f"thin walls: {int(ts.sum())} solid cells, blocked faces x/y/z = {nb}")

    def _pack_thin_normals(self, thin_sdf):
        """Store the plate normal of every cell that owns a blocked face in bits 8-31 of its
        blocked-face word (3 x 8-bit components, ~0.5 deg), and in bit 3 + d whether the face
        toward +d may use it: the wall flux then takes the velocity normal to the real plate
        (kernels.wall_flux_n). Normal = gradient of thin_sdf at the cell centre. Only faces whose
        normal is within ~70 deg of the face axis (|n_d| >= 0.3) use it -- the diagonal grid-fin
        lattice (|n_d| = 0.71) does. Next to a plate edge or tooth tip the gradient points toward
        the edge, along the flow, and the wall would read the oncoming flow as an impact (the
        isolated fin diverged); shallow staircase steps keep the axis mirror as well."""
        cells = np.nonzero(self.blk & 7)
        if cells[0].size == 0:
            return

        def grad_at(idx):
            X = [self.axes[d][idx[d]] for d in range(3)]
            hl = np.minimum.reduce([self.widths[d][idx[d]] for d in range(3)])
            eps = 0.05 * hl
            gr = []
            for d in range(3):
                P = list(X)
                P[d] = X[d] + eps
                fp = np.asarray(thin_sdf(*P), float)
                P[d] = X[d] - eps
                fm = np.asarray(thin_sdf(*P), float)
                gr.append((fp - fm) / (2 * eps))
            gv = np.stack(gr)
            mag = np.linalg.norm(gv, axis=0)
            return gv / np.maximum(mag, 1e-30), (mag > 0.5) & (mag < 2.0)

        gA, okA = grad_at(cells)
        qv = np.clip(np.round(gA * 127.0).astype(np.int64) + 128, 1, 255)
        code = (qv[0] | (qv[1] << 8) | (qv[2] << 16)) * okA
        packed = ((code << 8) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
        self.blk[cells] |= packed
        n_valid = 0
        for d in range(3):
            fa = np.nonzero((self.blk >> d) & 1)
            if fa[0].size == 0:
                continue
            ga, oka = grad_at(fa)
            valid = oka & (np.abs(ga[d]) >= 0.3)
            self.blk[tuple(v[valid] for v in fa)] |= np.int32(1 << (3 + d))
            n_valid += int(valid.sum())
        self.n_thin_normals = n_valid

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
        d_img = dist0.copy()                    # ghost-to-image distance actually used
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
            d_img[sel[ok]] = dist0[sel[ok]] + extra * hloc[sel[ok]]
            todo[sel[ok]] = False
        if todo.any():  # fall back to nearest fluid cell
            _, inds = ndimage.distance_transform_edt(~self.fluid, return_indices=True)
            sel = np.flatnonzero(todo)
            near_idx = np.ravel_multi_index(tuple(ind.ravel()[gi[sel]] for ind in inds), shape)
            cidx[sel] = near_idx[:, None]
            cw[sel] = 0.0
            cw[sel, 0] = 1.0
        self.n_ib_fallback = int(todo.sum())
        # no-slip walls: the tangential velocity goes linearly through zero at the wall, ghost value =
        # -(d_ghost / d_image) x image value (distances from the wall). The image is the ghost's
        # mirror point (ratio 1) unless it had to move further out (ghost almost on the wall, or the
        # mirror stencil not in the fluid); plain reflection then overstates the shear. The normal
        # component is always mirrored (kernels.ib_kernel), which keeps the wall watertight.
        dg = np.maximum(-phi.ravel()[gi], 0.0)
        ratio = np.clip(dg / np.maximum(d_img - dg, 1e-12), 0.0, 1.0)
        ratio[todo] = 1.0
        dev = self.device
        self.ib = dict(g=torch.as_tensor(gi, device=dev), c=torch.as_tensor(cidx, device=dev),
                       w=torch.as_tensor(cw, dtype=self.dtype, device=dev),
                       n=torch.as_tensor(nrm, dtype=self.dtype, device=dev),
                       r=torch.as_tensor(ratio, dtype=self.dtype, device=dev))

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
                if kind == "interface":
                    continue                     # filled from the neighbouring grids (patches)
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
        vel = vel - 2.0 * vn * ib["n"]                     # slip wall: mirror the normal component
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
            WR = qp1 - (0.5 * order) * _limit(qp2 - qp1, dC, self.limiter)
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
    def _L(self, kernel, dim, inputs):
        """Warp launch through a cache of recorded launches: the argument packing (tens of us
        per launch in Python, which dominates many small patches) is done once per distinct set
        of arrays; later calls only refresh the scalar arguments and replay."""
        scalar = (int, float, bool, np.number)
        inputs = [x.item() if isinstance(x, np.number) else x for x in inputs]
        key = (id(kernel), tuple(int(v) for v in dim) if isinstance(dim, (tuple, list)) else int(dim),
               tuple(id(x) for x in inputs if not isinstance(x, scalar)))
        cache = self.__dict__.setdefault("_lcache", {})
        c = cache.get(key)
        if c is None:
            c = self._wp.launch(kernel, dim=dim, inputs=inputs, device=self._wdev, stream=self._wp_stream,
                                record_cmd=True)
            cache[key] = c
        else:
            for i, x in enumerate(inputs):
                if isinstance(x, scalar):
                    c.set_param_at_index(i, x)
        c.launch(stream=self._wp_stream)

    def _wprim(self, Ukey, tkey=None):
        wp, K, w = self._wp, self._K, self._w
        shape = tuple(self.U.shape[1:])
        self._L(K.prim_kernel, shape, [w[Ukey], self.gamma, self.rho_min, self.p_min, w["W"]])
        if self._iface is not None:
            self._iface(self)
        self._fill_bc(self.W_buf)
        ib = self._wp_ib
        if ib is not None:
            self._L(K.ib_kernel, ib["n"], [w["W"], ib["ghost"], ib["corner"], ib["weight"],
                                                       ib["normal"], ib["ratio"], int(self.wall == "noslip")])
        jt = getattr(self, "_wp_jets", None)
        if jt is not None:
            thr = np.broadcast_to(np.asarray(self.jet_throttle, np.float32), self._jet_thr_np.shape)
            if np.any(thr != self._jet_thr_np):                 # upload only when it changes
                self._jet_thr_np = thr.copy()
                self._jet_thr.copy_(torch.as_tensor(self._jet_thr_np))
            self._L(K.jet_kernel, jt["n"], [w["W"], jt["cells"], jt["state"], jt["group"], jt["ambient"], jt["throttle"]])
        self._L(K.shock_kernel, shape, [w["W"], self._active_sensor, self.shock_threshold, w["S"]])
        if self.les and self.les["sgs"] == "vreman":
            self._L(K.vreman_kernel, shape, [w["W"], self._vp, float(self.les["cv"]), self._vp.mut])
        if self.turbulence:
            kk, oo = (("K", "O") if tkey in (None, "") else (tkey, "O1"))
            self._fill_bc_turb(getattr(self, kk + "_buf"), getattr(self, oo + "_buf"))
            self._L(K.mut_kernel, shape, [w["W"], w[kk], w[oo], self._vp, self._vp.mut])

    def _wturb(self, kk, oo, outk, outo, a0, a1, local, dual=None):
        """SST residual on stage state (kk, oo) and point-implicit update into (outk, outo).
        dual = (b0, b1, b2, 1/dt): add the physical-time terms of dual time stepping."""
        wp, K, w = self._wp, self._K, self._w
        wp.launch(K.turb_residual_kernel, dim=self.n,
                  inputs=[w["W"], w[kk], w[oo], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"], self._vp,
                          w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["utau"]], device=self._wdev, stream=self._wp_stream)
        if dual is not None:
            wp.launch(K.dual_time_turb_kernel, dim=self.n,
                      inputs=[w["Kn"], w["On"], w["Km"], w["Om"], w["Rk"], w["Rw"], w["Dk"], w["Dw"], *dual],
                      device=self._wdev, stream=self._wp_stream)
        wp.launch(K.turb_update_kernel, dim=self.n,
                  inputs=[w["K"], w["O"], w[kk], w[oo], w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["dt"], w["dtl"],
                          int(local), self._vp, w["W"], w["utau"], a0, a1, 1e-6 * self.k_inf, 1e-3 * self.omega_inf,
                          w[outk], w[outo]], device=self._wdev, stream=self._wp_stream)

    def _wresidual(self, order, tiles=None):
        wp, K, w = self._wp, self._K, self._w
        lm = self.low_mach
        if lm == "auto":
            lm = np.linalg.norm(self.W_inf[1:4]) < np.sqrt(self.gamma * self.W_inf[4] / self.W_inf[0])
        kind = {"vanleer": 0, "minmod": 1, "koren": 2}[self.limiter] + (4 if lm else 0)
        args = [w["W"], w["S"], w["ix"], w["iy"], w["iz"], self._active_flux, self.gamma, float(order), kind, w["blk"],
                self._vp]
        if tiles is None and self.split_residual:
            # one direction at a time: face fluxes into a scratch array, then R (+)= their
            # difference (kernels.FACE_KERNELS / DIV_KERNELS). The scratch can be shared by
            # several grids (rocketcfd.patches) since grids are updated one after another.
            if "F" not in w:
                F = self.face_scratch(tuple(self.W_buf.shape)) if self.face_scratch else torch.empty_like(self.W_buf)
                self._F_buf = F
                w["F"] = wp.from_torch(F)
            visc = 1 if self.viscous else 0
            first = 1
            for d in range(3):
                if self._active_flux[d] != 0:
                    dims = list(self.n)
                    dims[d] += 1
                    self._L(K.FACE_KERNELS[visc][d], tuple(dims), [w["W"], w["S"], w["blk"], self.gamma, float(order),
                                                                  kind, self._vp, w["F"]])
                    self._L(K.DIV_KERNELS[visc][d], self.n, [w["W"], w[("ix", "iy", "iz")[d]], self.gamma, w["blk"],
                                                             self._vp, w["F"], first, w["R"]])
                    first = 0
        elif tiles is None:
            self._L(K.residual_kernel, self.n, args + [w["R"]])
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

    def _stable_dt(self):
        """Largest stable global time step (tensor) at the current state."""
        K, w = self._K, self._w
        self._wprim("U")
        self._L(K.rate_kernel, self.n, [w["W"], w["ix"], w["iy"], w["iz"], self._active_flux,
                                        w["mask"], self.gamma, self._vp, w["rate"]])
        return self.cfl / self.rate_buf.max()

    def _step_warp(self, order, want_res=False, local=False, tiles=None, dt_fixed=None):
        """SSP-RK2 step. local: per-cell time steps (steady runs only). tiles: update only
        these 8x8x8 tiles (active-region updating, steady runs only). dt_fixed: use this global
        time step (tensor) instead of this grid's own (time-accurate runs on composite grids)."""
        wp, K, w = self._wp, self._K, self._w
        self._wprim("U")
        if dt_fixed is not None:
            self.dt_buf[0].copy_(dt_fixed)
        else:
            self._L(K.rate_kernel, self.n, [w["W"], w["ix"], w["iy"], w["iz"], self._active_flux,
                                            w["mask"], self.gamma, self._vp, w["rate"]])
            torch.div(self.cfl, self.rate_buf.max(), out=self.dt_buf[0])
        if local and dt_fixed is None:
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

    def _step_implicit(self, order, cfl, sweeps=8, want_res=False, maxrel=0.2, dual=None):
        """One backward-Euler pseudo-time step (steady runs only): residual at the current state,
        `sweeps` Jacobi sweeps of the linearised system with local pseudo time steps at CFL `cfl`
        (DP-LUR), then U += dU limited to `maxrel` relative change in density / pressure.
        k and omega advance by a point-implicit step at their own local stable step.
        dual = (b0, b1, b2, 1/dt): inner iteration of dual time stepping (see run_unsteady)."""
        wp, K, w = self._wp, self._K, self._w
        if "dU2" not in w:
            self.dU2_buf = torch.zeros_like(self.U)
            w["dU2"] = wp.from_torch(self.dU2_buf)
        self._wprim("U")
        self._wresidual(order)
        if dual is not None:
            self._L(K.dual_time_kernel, self.n, [w["U"], w["Un"], w["Um"], w["R"], w["mask"], *dual])
        res = torch.sqrt(((self.R_buf[0] * self.mask_f) ** 2).sum() / self.n_fluid) if want_res else None
        if self.turbulence:
            # implicit k-omega: residual, then Jacobi sweeps at the turbulence's own local
            # pseudo time step (convective |u| + diffusion) at the same CFL number
            if "dK2" not in w:
                self.dK2_buf = torch.zeros_like(self.K_buf)
                self.dO2_buf = torch.zeros_like(self.O_buf)
                w.update(dK2=wp.from_torch(self.dK2_buf), dO2=wp.from_torch(self.dO2_buf))
            self._L(K.turb_rate_kernel, self.n, [w["W"], w["ix"], w["iy"], w["iz"], self._active_flux, self._vp, float(cfl), w["dtl"]])
            self._L(K.turb_residual_kernel, self.n, [w["W"], w["K"], w["O"], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"],
                              self._vp, w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["utau"]])
            if dual is not None:
                self._L(K.dual_time_turb_kernel, self.n, [w["Kn"], w["On"], w["Km"], w["Om"], w["Rk"], w["Rw"], w["Dk"], w["Dw"], *dual])
            tb = (("K1", "O1"), ("dK2", "dO2"))
            for it in range(sweeps):
                (ko, oo), (kn, on) = tb[(it + 1) % 2], tb[it % 2]
                self._L(K.turb_sweep_kernel, self.n, [w["W"], w["K"], w["O"], w["Rk"], w["Rw"], w["Dk"], w["Dw"], w["dtl"], self._vp,
                                  w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"], int(it == 0),
                                  w[ko], w[oo], w[kn], w[on]])
            kl, ol = tb[(sweeps - 1) % 2]
            self._L(K.turb_apply_kernel, self.n, [w["K"], w["O"], w[kl], w[ol], w["W"], self._vp, w["utau"],
                              1e-6 * self.k_inf, 1e-3 * self.omega_inf])
        bufs = ("U1", "dU2")
        dd = 0.0 if dual is None else dual[0] * dual[3]
        if "wst" not in w:
            self.wst_buf = torch.zeros(self.n, dtype=self.dtype, device=self.device)
            w["wst"] = wp.from_torch(self.wst_buf)
        if self.turbulence and getattr(self, "wall_stiffness", False):
            # implicit diagonal of the wall-function shear (optional: tested on the flat plate, it
            # did not change where the implicit k-omega path settles)
            self._L(K.wall_stiff_kernel, self.n, [w["W"], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"],
                                                  self._vp, w["wst"]])
        if self.cache_lam:
            # spectral radii once per iteration (they do not change during the sweeps)
            if "lam" not in w:
                self.lam_buf = torch.empty((3,) + tuple(self.n), dtype=self.dtype, device=self.device)
                w["lam"] = wp.from_torch(self.lam_buf)
            self._L(K.lam_kernel, self.n, [w["W"], w["ix"], w["iy"], w["iz"], self._active_flux, self._vp,
                                           self.gamma, w["lam"]])
            if self.turbulence and self.cfl_wall and not getattr(self, "wall_stiffness", False):
                self._L(K.wall_cfl_kernel, self.n, [w["lam"], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"],
                                                    self._vp, float(cfl), float(self.cfl_wall), w["wst"]])
            sweep = K.implicit_sweep_lam_kernel
            common = [w["U"], w["lam"], w["R"], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"], w["mask"],
                      w["wst"], self.gamma, float(cfl), dd]
        else:
            sweep = K.implicit_sweep_kernel
            common = [w["U"], w["W"], w["R"], w["ix"], w["iy"], w["iz"], self._active_flux, w["blk"], w["mask"],
                      w["wst"], self._vp, self.gamma, float(cfl), dd]
        if self.relax == "rb":
            # red-black Gauss-Seidel: a Jacobi start, then in-place half sweeps by colour; converges
            # per sweep about twice as fast as Jacobi, so `sweeps` Jacobi-equivalents -> sweeps / 2
            self._L(sweep, self.n, common + [1, -1, w["U1"], w["U1"]])
            for it in range(max(1, (sweeps + 1) // 2) - 1):
                for col in (0, 1):
                    self._L(sweep, self.n, common + [0, col, w["U1"], w["U1"]])
            last = "U1"
        else:
            for it in range(sweeps):
                src, dst = w[bufs[(it + 1) % 2]], w[bufs[it % 2]]
                self._L(sweep, self.n, common + [int(it == 0), -1, src, dst])
            last = bufs[(sweeps - 1) % 2]
        self._L(K.implicit_apply_kernel, self.n, [w["U"], w[last], w["mask"], self.gamma, maxrel])
        return res

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
            active_tiles=None, tile_refresh=50, tile_frac=1e-3, local_dt=False, snapshot=None,
            implicit=False, cfl_max=20.0, cfl_start=1.0, cfl_ramp=300, sweeps=8, explicit_finish=None):
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

        implicit=True (warp backend, steady runs): backward-Euler pseudo-time stepping with
        DP-LUR Jacobi sweeps and local time steps; the CFL number grows geometrically from
        `cfl_start` to `cfl_max` over `cfl_ramp` steps. Each step costs about 1.5-2 explicit
        RK2 steps but converges in far fewer steps, especially on stretched or very fine
        near-wall grids. `t` does not advance (pseudo time has no physical meaning).
        """
        if implicit:
            if self.backend != "warp":
                raise RuntimeError("the implicit solver needs the warp backend")
            self._run_implicit(nsteps, chunk, first_order_steps, report, snapshot,
                               cfl_max, cfl_start, cfl_ramp, sweeps)
            # Turbulent runs: the implicit k-omega path settles a few per cent away from the
            # steady state the explicit scheme reaches from the same start (flat plate: Cf -1 %
            # vs -6 % from White). Finish with explicit local-time-step steps, which then move
            # it there in a fraction of the explicit-only cost. Default: on for turbulent runs.
            nfin = explicit_finish if explicit_finish is not None else (nsteps * 3 if self.turbulence else 0)
            if nfin:
                if self.verbose:
                    print(f"explicit finish: up to {nfin} local-time-step steps", flush=True)
                start = self.step_count
                # the report's convergence test is already met at the switch: do at least a third
                fin_report = None if report is None else \
                    (lambda sv: (sv.step_count - start >= nfin // 3) and report(sv))
                return self.run(nfin, chunk=max(chunk, 500), first_order_steps=0, report=fin_report,
                                steady=True, active_tiles=False, local_dt=True, snapshot=snapshot)
            return self
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

    def _run_implicit(self, nsteps, chunk, first_order_steps, report, snapshot, cfl_max, cfl_start,
                      cfl_ramp, sweeps):
        done = 0
        t0 = time.time()
        self._tiles_on = False
        self.active_fraction = 1.0
        n0 = self.step_count
        with torch.no_grad():
            while done < nsteps:
                k = min(chunk, nsteps - done)
                for kk in range(k):
                    it = self.step_count + kk
                    order = 0.0 if it < first_order_steps else 1.0
                    ramp = min(1.0, (it - n0) / max(cfl_ramp, 1))
                    cfl = cfl_start * (cfl_max / cfl_start) ** ramp
                    r = self._step_implicit(order, cfl, sweeps, want_res=(kk == k - 1))
                    if r is not None:
                        res = r
                    if snapshot is not None and (it + 1) % snapshot[0] == 0:
                        snapshot[1](it + 1, self.t)
                res = float(res)
                self.step_count += k
                done += k
                self.history.append((self.step_count, self.t, res))
                if not np.isfinite(res):
                    raise FloatingPointError(f"solution diverged at step {self.step_count}")
                if self.verbose:
                    el = time.time() - t0
                    cps = np.prod(self.n) * done / el / 1e6
                    mem = (f"  GPU mem {torch.cuda.max_memory_allocated(self.device)/2**30:.2f} GB"
                           if self.device.type == "cuda" else "")
                    print(f"iter {self.step_count:6d}  CFL {cfl:6.1f}  res(rho) = {res/self.fs['rho']:.3e} 1/s  "
                          f"[{cps:.1f} Mcell-iters/s]{mem}", flush=True)
                if report and report(self):
                    break
        self.work_full_steps = float(done)
        return self

    def _dual_shift(self, bdf, dt=None):
        """Start a physical time step of dual time stepping: U^(n-1) <- U^n <- U (and k, omega).
        Returns the BDF coefficients and 1/dt for _step_implicit(dual=...)."""
        wp, w = self._wp, self._w
        if "Un" not in w:
            self.Un_buf = self.U.clone()
            self.Um_buf = self.U.clone()
            w.update(Un=wp.from_torch(self.Un_buf), Um=wp.from_torch(self.Um_buf))
            if self.turbulence:
                for nm in ("Kn", "On", "Km", "Om"):
                    setattr(self, nm + "_buf", (self.K_buf if nm[0] == "K" else self.O_buf).clone())
                    w[nm] = wp.from_torch(getattr(self, nm + "_buf"))
            self._dual_steps = 0
        self.Um_buf.copy_(self.Un_buf)
        self.Un_buf.copy_(self.U)
        if self.turbulence:
            self.Km_buf.copy_(self.Kn_buf)
            self.Om_buf.copy_(self.On_buf)
            self.Kn_buf.copy_(self.K_buf)
            self.On_buf.copy_(self.O_buf)
        first = self._dual_steps == 0 or bdf == 1
        b = (1.0, -1.0, 0.0) if first else (1.5, -2.0, 0.5)
        self._dual_steps += 1
        return (b[0], b[1], b[2], 1.0 / (dt if dt is not None else self._dual_dt))

    def run_unsteady(self, nsteps, dt, inner=30, inner_tol=1e-2, cfl=20.0, sweeps=8, order=1.0,
                     report=None, every=10, snapshot=None, bdf=2):
        """Time-accurate run with dual time stepping (warp backend): each physical step of
        size `dt` solves the BDF2 (bdf=1: backward Euler) equations by implicit pseudo-time
        iterations until the density residual falls by `inner_tol` or `inner` iterations.
        dt is set by the physics (e.g. ~1/100 of a shedding period), not by the cell size, so
        fine near-wall cells no longer force tiny steps. report(solver) is called every
        `every` physical steps and may return True to stop; snapshot = (every_steps, fn)."""
        if self.backend != "warp":
            raise RuntimeError("dual time stepping needs the warp backend")
        self._dual_dt = dt
        t0 = time.time()
        self._tiles_on = False
        iters = 0
        with torch.no_grad():
            for n in range(nsteps):
                dual = self._dual_shift(bdf)
                r0 = None
                for m in range(inner):
                    r = float(self._step_implicit(order, cfl, sweeps, want_res=True, dual=dual))
                    iters += 1
                    if not np.isfinite(r):
                        raise FloatingPointError(f"diverged at physical step {self.step_count}, inner {m}")
                    r0 = r if r0 is None else r0
                    if m > 0 and r < inner_tol * r0:
                        break
                self.step_count += 1
                self.t += dt
                self.last_inner = (m + 1, r / max(r0, 1e-30))
                if snapshot is not None and self.step_count % snapshot[0] == 0:
                    snapshot[1](self.step_count, self.t)
                if (n + 1) % every == 0 or n == nsteps - 1:
                    self.history.append((self.step_count, self.t, r0))
                    if self.verbose:
                        el = time.time() - t0
                        print(f"step {self.step_count:6d}  t = {self.t:.4e} s  inner {m+1:3d} "
                              f"(res drop {r/max(r0,1e-30):.1e})  {iters/(n+1):.1f} inner/step  "
                              f"[{np.prod(self.n)*iters/el/1e6:.1f} Mcell-iters/s]", flush=True)
                    if report and report(self):
                        break
        self.work_full_steps = float(iters)
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

    def state(self):
        """Copy of the solution, for set_state (warm starts in parameter sweeps)."""
        st = dict(U=self.U.clone())
        if self.turbulence:
            st.update(K=self.K_buf.clone(), O=self.O_buf.clone())
        return st

    def set_state(self, st):
        self.U.copy_(st["U"])
        if self.turbulence and "K" in st:
            self.K_buf.copy_(st["K"])
            self.O_buf.copy_(st["O"])
        if self.backend == "warp":
            self.U1.copy_(self.U)

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
        if self.les and self.backend == "warp":
            with torch.no_grad():
                self._wprim("U")
                out["mut"] = self.mut_buf[(slice(NG, -NG),) * 3].double().cpu().numpy()
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

    def loads(self, S_ref, L_ref, x_ref=(0.0, 0.0, 0.0), band=1.5, parts=None, friction=True, method="tri"):
        """Integrate surface pressure: -∮ (p - p_inf) n dA via a smoothed delta function, plus
        (viscous runs, friction=True) the wall shear -- from the velocity interpolated 1.5 cells
        out along the wall normal, through the same law of the wall as the solver's wall function
        when turbulent, mu u_t / y when laminar (thin walls: the adjacent cells). All coefficients,
        parts and moments include friction; CD_friction / C_friction give its share.
        Returns body-axis force/moment coefficients and wind-axis drag/lift.

        method="tri" (default): immersed walls are triangulated (marching cubes of phi) and the
        pressure is interpolated at each triangle's centroid; exact for any cell shape. "band": the
        earlier smoothed-delta band, which mis-integrates flat walls facing the long side of
        stretched cells (-90 % .. +150 % on a flat end face with 4:1 cells). Thin walls use the
        blocked faces either way. The surface data are built once; each call only gathers values."""
        sf = self._surface_tri() if method == "tri" else self._surface(band)
        tf = self._surface(band).get("thin") if method == "tri" else sf.get("thin")
        with torch.no_grad():
            if self.backend == "warp":
                self._wprim("U")
                Wsrc = self.W_buf
            else:
                Wsrc = self._prim(self.U)
            pflat = Wsrc[4].reshape(-1)
            if "p_idx" in sf:
                pb = (pflat[sf["p_idx"]] * sf["p_w"]).sum(-1)
            else:
                pb = pflat[sf["flat_t"]]
            if tf is not None:
                pa = pflat[tf["a_t"]].double().cpu().numpy()
                pbb = pflat[tf["b_t"]].double().cpu().numpy()
            if self.viscous and friction:
                Wf = Wsrc.reshape(5, -1)
                nimg = sf["y_img"].shape[0]
                Wb = [(Wf[:, sf["img_idx"][:, 8 * m:8 * m + 8]] * sf["img_w"][:, 8 * m:8 * m + 8]).sum(-1)
                      .double().cpu().numpy() for m in range(nimg)]
                if tf is not None:
                    Wa_t = Wf[:, tf["a_t"]].double().cpu().numpy()
                    Wb_t = Wf[:, tf["b_t"]].double().cpu().numpy()
        dp = pb.double().cpu().numpy() - self.W_inf[4]
        dF = sf["wF"] * dp                                  # (3, Nb)
        F_fric = np.zeros(3)
        if self.viscous and friction:
            # wall shear on immersed walls: law of the wall (turbulent) or mu u_t / y (laminar)
            # at the fluid-side band cells; the smoothed delta puts half its weight there
            fr = self._shear(Wb, sf["nrm"], sf["y_img"]) * sf["wfr"]
            dF = dF + fr
            F_fric = F_fric + fr.sum(1)
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
            if self.viscous and friction:
                # shear on both wetted sides of each blocked face (tangential = the other two axes)
                for Wt, flag, ykey in ((Wa_t, tf["fa"], "ya"), (Wb_t, tf["fb"], "yb")):
                    nrm = np.zeros((3, mag.size))
                    nrm[tf["dir"], np.arange(mag.size)] = 1.0
                    ft = self._shear(Wt, nrm, tf[ykey]) * (tf["area"] * flag)
                    dFt = dFt + ft
                    F_fric = F_fric + ft.sum(1)
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
                           CX=float(v[0][0] / qs), CY=float(v[0][1] / qs), CZ=float(v[0][2] / qs),
                           F=np.asarray(v[0], float), M=np.asarray(v[1], float))   # N, N m about x_ref
                   for k, v in part_out.items()}
        return dict(F=F, M=M, F_thin=F_thin, M_thin=M_thin, q_inf=q, parts=parts_c,
                    F_friction=F_fric, CD_friction=float(F_fric @ e_d / qs),
                    C_friction=F_fric / qs,
                    C_thin=F_thin / (q * S_ref), Cm_thin=M_thin / (q * S_ref * L_ref),
                    CX=Cf[0], CY=Cf[1], CZ=Cf[2],
                    Cl=Cm[0], Cm=Cm[1], Cn=Cm[2], CD=Cf @ e_d, CL=Cf @ lift_dir,
                    CN=Cf @ cross, CS=Cf @ side, C_roll=Cm[0], C_restore=Cm @ restore_axis)

    def _shear(self, Wc, nrm, y):
        """Wall shear force per unit area on the wall [Pa] (3, N) from cell states Wc (5, N)
        at distance y along the wall normal nrm (3, N): along the tangential velocity.
        Wc, y may be lists of two samples (laminar): u = a y + b y^2 through the no-slip wall."""
        from .wallfunction import spalding_utau
        v = self.viscous
        if isinstance(Wc, (list, tuple)) and len(Wc) == 2 and not self.wallmodel:
            (W1, W2), (y1, y2) = Wc, y
            T = W1[4] / (W1[0] * v["R"])
            mu = v["mu"] if v["model"] == "constant" else \
                v["mu"] * (T / v["T0"]) ** 1.5 * (v["T0"] + v["S"]) / (T + v["S"])
            u1 = W1[1:4] - (W1[1:4] * nrm).sum(0) * nrm
            u2 = W2[1:4] - (W2[1:4] * nrm).sum(0) * nrm
            return mu * (u1 * y2 ** 2 - u2 * y1 ** 2) / (y1 * y2 * (y2 - y1))
        if isinstance(Wc, (list, tuple)):
            Wc, y = Wc[0], y[0]
        rho, vel, p = Wc[0], Wc[1:4], Wc[4]
        T = p / (rho * v["R"])
        mu = v["mu"] if v["model"] == "constant" else \
            v["mu"] * (T / v["T0"]) ** 1.5 * (v["T0"] + v["S"]) / (T + v["S"])
        ut = vel - (vel * nrm).sum(0) * nrm
        um = np.maximum(np.linalg.norm(ut, axis=0), 1e-12)
        if self.wallmodel:
            utau = spalding_utau(um, y, mu / rho)
            tau = rho * utau ** 2
        else:
            tau = mu * um / y
        return tau * ut / um

    def _surface_tri(self):
        """Triangulated immersed walls for the loads: marching cubes of phi in index space (in
        slabs, so memory stays bounded), oriented toward the fluid, mapped to physical space.
        Per triangle: area vector, centroid, the trilinear stencil for the pressure at the centroid
        (ghost cells hold the wall state, so no masking), and wall-shear sample stencils."""
        if getattr(self, "_tri", None) is not None:
            return self._tri
        from skimage.measure import marching_cubes
        from scipy.ndimage import map_coordinates
        phi = self.phi
        shape = phi.shape
        wx, wy, wz = self.widths
        near = np.abs(phi) < 3.0 * np.maximum(np.maximum(wx[:, None, None], wy[None, :, None]), wz[None, None, :])
        cen_l, area_l = [], []
        if near.any() and phi.min() < 0 < phi.max():
            lo_i = [int(np.flatnonzero(near.any(axis=tuple(e for e in range(3) if e != d)))[0]) for d in range(3)]
            hi_i = [int(np.flatnonzero(near.any(axis=tuple(e for e in range(3) if e != d)))[-1]) for d in range(3)]
            lo_i = [max(v - 1, 0) for v in lo_i]
            hi_i = [min(v + 2, shape[d]) for d, v in enumerate(hi_i)]
            sl12 = (slice(lo_i[1], hi_i[1]), slice(lo_i[2], hi_i[2]))
            step = 48
            for k0 in range(lo_i[0], hi_i[0] - 1, step):
                k1 = min(k0 + step + 1, hi_i[0])
                blk = phi[k0:k1][:, sl12[0], sl12[1]]
                if blk.shape[0] < 2 or blk.min() >= 0 or blk.max() <= 0:
                    continue
                if min(blk.shape) < 2:
                    continue
                v, f, _, _ = marching_cubes(blk, 0.0)
                v = v + np.array([k0, lo_i[1], lo_i[2]], float)        # padded index space
                t = v[f]                                              # (T, 3, 3)
                c = t.mean(1)
                n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                nn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
                pp = map_coordinates(phi, (c + 0.25 * nn).T, order=1, mode="nearest")
                pm = map_coordinates(phi, (c - 0.25 * nn).T, order=1, mode="nearest")
                flip = pp < pm                                        # normal must point to the fluid
                tp = np.stack([np.stack([np.interp(t[:, j, d], np.arange(shape[d]), self.axes[d])
                                         for d in range(3)], 1) for j in range(3)], 1)
                a = 0.5 * np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
                a[flip] *= -1.0
                cen_l.append(c)
                area_l.append(a)
        if cen_l:
            c_idx = np.concatenate(cen_l)
            avec = np.concatenate(area_l)
        else:
            c_idx = np.zeros((0, 3))
            avec = np.zeros((0, 3))
        # keep triangles whose centroid lies in an interior cell this grid owns
        cell = np.floor(c_idx + 0.5).astype(np.int64)
        for d in range(3):
            cell[:, d] = np.clip(cell[:, d], 0, shape[d] - 1)
        cflat = np.ravel_multi_index(cell.T, shape) if len(cell) else np.zeros(0, np.int64)
        inner = np.zeros(shape, bool)
        inner[NG:-NG, NG:-NG, NG:-NG] = True
        keep = inner.ravel()[cflat] if len(cflat) else np.zeros(0, bool)
        if self.owned is not None and len(cflat):
            keep &= self.owned.ravel()[cflat] > 0.5
        c_idx, avec, cell = c_idx[keep], avec[keep], cell[keep]
        pos = np.stack([np.interp(c_idx[:, d], np.arange(shape[d]), self.axes[d]) for d in range(3)])
        area = np.linalg.norm(avec, axis=1)
        nrm = (avec / np.maximum(area, 1e-30)[:, None]).T

        def stencil(pts, fluid_only):
            i0, fr = [], []
            for d in range(3):
                cax = self.axes[d]
                k = np.clip(np.searchsorted(cax, pts[d]) - 1, 0, len(cax) - 2)
                i0.append(k)
                fr.append(np.clip((pts[d] - cax[k]) / (cax[k + 1] - cax[k]), 0.0, 1.0))
            ii = np.zeros((pts.shape[1], 8), np.int64)
            ww = np.zeros((pts.shape[1], 8))
            fluid_f = self.fluid.ravel()
            m_ = 0
            for a_ in (0, 1):
                for b_ in (0, 1):
                    for c_ in (0, 1):
                        f_ = np.ravel_multi_index((i0[0] + a_, i0[1] + b_, i0[2] + c_), shape)
                        ii[:, m_] = f_
                        w_ = ((fr[0] if a_ else 1 - fr[0]) * (fr[1] if b_ else 1 - fr[1])
                              * (fr[2] if c_ else 1 - fr[2]))
                        ww[:, m_] = w_ * fluid_f[f_] if fluid_only else w_
                        m_ += 1
            tw = ww.sum(1)
            ok = tw > 1e-6
            ww[ok] /= tw[ok, None]
            return ii, ww, ok

        p_idx, p_w, _ = stencil(pos, False)
        hloc = np.minimum.reduce([self.widths[d][cell[:, d]] for d in range(3)]) if len(cell) else np.zeros(0)
        y_img = (np.stack([1.0 * hloc, 2.0 * hloc]) if not self.wallmodel else 1.5 * hloc[None])
        st_ = [stencil(pos + y_ * nrm, True) for y_ in y_img]
        dev = self.device
        self._tri = dict(
            wF=-avec.T, pos=pos, nrm=nrm, y_img=y_img, area=area,
            wfr=area * np.logical_and.reduce([t_[2] for t_ in st_]) if st_ else area,
            p_idx=torch.as_tensor(p_idx, device=dev), p_w=torch.as_tensor(p_w, dtype=self.dtype, device=dev),
            img_idx=torch.as_tensor(np.concatenate([t_[0] for t_ in st_], 1), device=dev),
            img_w=torch.as_tensor(np.concatenate([t_[1] for t_ in st_], 1), dtype=self.dtype, device=dev))
        return self._tri

    def _surface(self, band):
        key = ("surface", band)
        if getattr(self, "_surf_key", None) != key:
            phi = self.phi
            wx, wy, wz = self.widths
            hloc = np.minimum(np.minimum(wx[:, None, None], wy[None, :, None]), wz[None, None, :])
            eps = band * hloc
            inner_ = np.zeros(phi.shape, bool)            # interior cells only: the padding layers
            inner_[NG:-NG, NG:-NG, NG:-NG] = True         # repeat the edge (5x on one-cell-thick 2-D grids)
            idx = np.nonzero((np.abs(phi) < eps) & inner_)
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
            ph = phi[idx]
            if self.owned is not None:
                dV = dV * self.owned[idx]
            # wall-shear sample points along the wall normal: 1 and 2 local cells out (laminar:
            # quadratic fit through the no-slip wall), 1.5 cells out (turbulent: law of the wall)
            pos = np.stack([self.axes[d][idx[d]] for d in range(3)])
            fluid_f = self.fluid.ravel()

            def stencil(y_s):
                xi = pos + (y_s - ph) * g
                i0, fr = [], []
                for d in range(3):
                    c = self.axes[d]
                    k = np.clip(np.searchsorted(c, xi[d]) - 1, 0, len(c) - 2)
                    i0.append(k)
                    fr.append(np.clip((xi[d] - c[k]) / (c[k + 1] - c[k]), 0.0, 1.0))
                ii = np.zeros((ph.size, 8), np.int64)
                ww = np.zeros((ph.size, 8))
                m_ = 0
                for a_ in (0, 1):
                    for b_ in (0, 1):
                        for c_ in (0, 1):
                            f_ = np.ravel_multi_index((i0[0] + a_, i0[1] + b_, i0[2] + c_), phi.shape)
                            ii[:, m_] = f_
                            ww[:, m_] = ((fr[0] if a_ else 1 - fr[0]) * (fr[1] if b_ else 1 - fr[1])
                                         * (fr[2] if c_ else 1 - fr[2])) * fluid_f[f_]
                            m_ += 1
                tw = ww.sum(1)
                ok = tw > 1e-6
                ww[ok] /= tw[ok, None]
                return ii, ww, ok

            y_img = (np.stack([1.0 * hloc[idx], 2.0 * hloc[idx]]) if not self.wallmodel
                     else 1.5 * hloc[idx][None])
            st_ = [stencil(y_) for y_ in y_img]
            img_idx = np.concatenate([t_[0] for t_ in st_], 1)
            img_w = np.concatenate([t_[1] for t_ in st_], 1)
            ok_img = np.logical_and.reduce([t_[2] for t_ in st_])
            self._surf = dict(wF=-(delta * dV) * g,
                              pos=np.stack([self.axes[d][idx[d]] for d in range(3)]),
                              flat_t=torch.as_tensor(flat, device=self.device),
                              nrm=g, phi=ph, wfr=2.0 * delta * dV * (ph > 0) * self.fluid[idx] * ok_img,
                              y_img=y_img,
                              img_idx=torch.as_tensor(img_idx, device=self.device),
                              img_w=torch.as_tensor(img_w, dtype=self.dtype, device=self.device))
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
                    if self.owned is not None:
                        area = area * self.owned[ia]
                    pos = np.stack([self.axes[e][ia[e]] for e in range(3)]).astype(float)
                    pos[d] += 0.5 * self.widths[d][ia[d]]
                    A_.append(np.ravel_multi_index(ia, phi.shape))
                    B_.append(np.ravel_multi_index(ib_, phi.shape))
                    D_.append(np.full(area.size, d))
                    AR.append(area)
                    PS.append(pos)
                    FA.append(fa[keep].astype(float))
                    FB.append(fb[keep].astype(float))
                Aall, Ball, Dall = np.concatenate(A_), np.concatenate(B_), np.concatenate(D_)
                # wall distance of the cells on each side (wall-function distance when turbulent)
                if getattr(self, "dist_buf", None) is not None and self.wallmodel:
                    dist = self.dist_buf.reshape(-1).cpu().numpy()
                    ya, yb = dist[Aall], dist[Ball]
                else:
                    wa = np.stack([self.widths[d_][np.unravel_index(Aall, phi.shape)[d_]] for d_ in range(3)])
                    ya = yb = 0.5 * wa[Dall, np.arange(Dall.size)]
                self._surf["thin"] = dict(ya=np.maximum(ya, 1e-9), yb=np.maximum(yb, 1e-9),
                                          a_t=torch.as_tensor(np.concatenate(A_), device=self.device),
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
