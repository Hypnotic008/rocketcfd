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
                 verbose=True, faces=None):
        """
        lo, hi     : domain corners (m)
        h          : cell size, scalar or (hx, hy, hz)
        faces      : optional [x_faces, y_faces, z_faces] for a stretched grid
                     (see rocketcfd.grid); overrides lo, hi and h
        freestream : dict(rho, u, v, w, p)  (SI)
        sdf        : callable(X, Y, Z) -> phi  (phi < 0 inside solid), or None
        bcs        : 'auto' or {(axis, side): 'freestream'|'extrapolate'|'periodic'}
        device     : torch device; default CUDA if available, else CPU
        """
        self.gamma, self.cfl, self.limiter = gamma, cfl, limiter
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

    # ------------------------------------------------------------------ setup
    def _setup_bcs(self, bcs):
        vel = self.W_inf[1:4]
        self.bcs = {}
        for d in range(3):
            for side in (0, 1):
                kind = "auto" if bcs == "auto" else bcs.get((d, side), "auto")
                if kind == "auto":
                    into = vel[d] > 1e-9 if side == 0 else vel[d] < -1e-9
                    kind = "freestream" if into else "extrapolate"
                self.bcs[(d, side)] = kind

    def _setup_ib(self):
        phi = self.phi
        self.fluid = phi > 0
        solid = ~self.fluid
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
            _, inds = ndimage.distance_transform_edt(solid, return_indices=True)
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
                elif kind == "extrapolate":
                    edge = _sl(W, ax, NG, NG + 1) if side == 0 else _sl(W, ax, N + NG - 1, N + NG)
                    dst.copy_(edge.expand_as(dst))
                else:  # freestream
                    dst.copy_(self.W_inf_t.reshape(5, 1, 1, 1).expand_as(dst))
        return W

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

    def _step(self, order):
        """One SSP-RK2 step. Returns (dt, density residual) as device tensors (no host sync)."""
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
    def run(self, nsteps, chunk=100, first_order_steps=200, report=None):
        """Advance nsteps time steps. `report(solver)` is called after every chunk."""
        done = 0
        t0 = time.time()
        with torch.no_grad():
            while done < nsteps:
                k = min(chunk, nsteps - done)
                order = 0.0 if self.step_count < first_order_steps else 1.0
                dt_sum = torch.zeros((), dtype=self.dtype, device=self.device)
                for _ in range(k):
                    dt, res = self._step(order)
                    dt_sum += dt
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
                    mem = (f"  GPU mem {torch.cuda.max_memory_allocated(self.device)/2**30:.2f} GB"
                           if self.device.type == "cuda" else "")
                    print(f"step {self.step_count:6d}  t = {self.t:.4e} s  "
                          f"res(rho) = {res/self.fs['rho']:.3e} 1/s  [{cps:.1f} Mcell-steps/s]{mem}",
                          flush=True)
                if report:
                    report(self)
        return self

    def set_primitive(self, W):
        """Set the state from a primitive-variable array (5, nx+4, ny+4, nz+4) (numpy or tensor)."""
        W = torch.as_tensor(np.asarray(W), dtype=self.dtype, device=self.device)
        self.U = prim_to_cons(W, self.gamma)

    # ------------------------------------------------------------- outputs
    def primitive(self):
        """Primitive variables on the padded grid (numpy), ghosts filled."""
        with torch.no_grad():
            return self._prim(self.U).cpu().numpy().astype(np.float64)

    def fields(self):
        W = self.primitive()[:, NG:-NG, NG:-NG, NG:-NG]
        rho, u, v, w, p = W
        c = np.sqrt(self.gamma * p / rho)
        mach = np.sqrt(u * u + v * v + w * w) / c
        solid = ~self.fluid[NG:-NG, NG:-NG, NG:-NG]
        return dict(rho=rho, u=u, v=v, w=w, p=p, mach=mach, solid=solid,
                    phi=self.phi[NG:-NG, NG:-NG, NG:-NG],
                    x=self.axes[0][NG:-NG], y=self.axes[1][NG:-NG], z=self.axes[2][NG:-NG])

    def loads(self, S_ref, L_ref, x_ref=(0.0, 0.0, 0.0), band=1.5):
        """Integrate surface pressure: -∮ (p - p_inf) n dA via a smoothed delta function.
        Returns body-axis force/moment coefficients and wind-axis drag/lift."""
        W = self.primitive()
        phi = self.phi
        wx, wy, wz = self.widths
        hloc = np.minimum(np.minimum(wx[:, None, None], wy[None, :, None]), wz[None, None, :])
        eps = band * hloc
        bandm = np.abs(phi) < eps
        idx = np.nonzero(bandm)
        e = eps[idx]
        delta = (1 + np.cos(np.pi * phi[idx] / e)) / (2 * e)
        grads = np.gradient(phi, *self.axes)
        gn = np.sqrt(sum(gd[idx] ** 2 for gd in grads)) + 1e-30
        dV = wx[idx[0]] * wy[idx[1]] * wz[idx[2]]
        dp = (W[4][idx] - self.W_inf[4]) * delta * dV
        dF = np.stack([-(dp * grads[d][idx] / gn) for d in range(3)])
        F = dF.sum(1)
        r = np.stack([self.axes[d][idx[d]] - x_ref[d] for d in range(3)])
        M = np.cross(r.T, dF.T).sum(0)
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
        return dict(F=F, M=M, q_inf=q, CX=Cf[0], CY=Cf[1], CZ=Cf[2],
                    Cl=Cm[0], Cm=Cm[1], Cn=Cm[2], CD=Cf @ e_d, CL=Cf @ lift_dir,
                    CN=Cf @ cross, CS=Cf @ side, C_roll=Cm[0], C_restore=Cm @ restore_axis)

    def save(self, path):
        np.savez_compressed(path, U=self.U.cpu().numpy(), t=self.t, step=self.step_count,
                            fx=self.faces[0], fy=self.faces[1], fz=self.faces[2],
                            history=np.array(self.history))

    def load(self, path):
        d = np.load(path)
        self.U = torch.as_tensor(d["U"], dtype=self.dtype, device=self.device)
        self.t, self.step_count = float(d["t"]), int(d["step"])
        self.history = [tuple(r) for r in d["history"]]
