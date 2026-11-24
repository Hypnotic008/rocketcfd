"""3-D compressible Euler solver on a uniform Cartesian grid with a ghost-cell
immersed boundary (inviscid slip wall) defined by a signed distance function.

Numerics
  * finite volume, MUSCL reconstruction of primitive variables (van Leer / minmod)
  * HLLC Riemann solver with HLLE fallback at strong shocks (carbuncle cure)
  * SSP-RK2 time integration, global CFL time step
  * ghost-cell immersed boundary: each solid cell next to the fluid gets the
    mirror-image state, interpolated from fluid cells, with the normal velocity reflected
Runs on CPU or GPU through JAX (same code).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy import ndimage

from .gas import prim_to_cons, cons_to_prim, hllc_flux

NG = 2  # ghost layers on the domain boundary


def _sl(a, ax, start, stop):
    return lax.slice_in_dim(a, start, stop, axis=ax)


def _limit(a, b, kind):
    if kind == "minmod":
        return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
    ab = a * b
    return jnp.where(ab > 0, 2.0 * ab / (a + b + (ab <= 0)), 0.0)  # van Leer


class Solver:
    def __init__(self, lo, hi, h, freestream, sdf=None, gamma=1.4, bcs="auto", cfl=0.7,
                 limiter="vanleer", shock_threshold=0.5, dtype=jnp.float32, verbose=True):
        """
        lo, hi     : domain corners (m)
        h          : cell size, scalar or (hx, hy, hz)
        freestream : dict(rho, u, v, w, p)  (SI)
        sdf        : callable(X, Y, Z) -> phi  (phi < 0 inside solid), or None
        bcs        : 'auto' or {(axis, side): 'freestream'|'extrapolate'|'periodic'}
        """
        self.gamma, self.cfl, self.limiter = gamma, cfl, limiter
        self.shock_threshold = shock_threshold
        self.dtype = dtype
        self.verbose = verbose
        self.lo = np.asarray(lo, float)
        self.h = np.broadcast_to(np.asarray(h, float), (3,)).copy()
        self.n = tuple(int(round(v)) for v in (np.asarray(hi, float) - self.lo) / self.h)
        self.hi = self.lo + self.h * np.array(self.n)
        self.fs = freestream
        self.W_inf = np.array([freestream[k] for k in ("rho", "u", "v", "w", "p")], float)
        self.rho_min = 1e-4 * freestream["rho"]
        self.p_min = 1e-4 * freestream["p"]

        t0 = time.time()
        self.axes = [self.lo[d] + (np.arange(-NG, self.n[d] + NG) + 0.5) * self.h[d] for d in range(3)]
        shape = tuple(m + 2 * NG for m in self.n)
        if sdf is None:
            self.phi = np.full(shape, 1e9)
        else:
            X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
            self.phi = sdf(X, Y, Z).astype(np.float64)
            del X, Y, Z
        self._setup_bcs(bcs)
        self._setup_ib()
        if verbose:
            print(f"grid {self.n} = {np.prod(self.n)/1e6:.2f} M cells, h = {self.h}, "
                  f"{self.n_ghost} IB ghost cells, setup {time.time()-t0:.1f} s, device {jax.devices()[0]}")

        interior = np.zeros(shape, bool)
        interior[NG:-NG, NG:-NG, NG:-NG] = True
        self.update_mask = jnp.asarray((self.fluid & interior)[NG:-NG, NG:-NG, NG:-NG], dtype)
        self.W_inf_j = jnp.asarray(self.W_inf, dtype)
        W0 = np.broadcast_to(self.W_inf[:, None, None, None], (5,) + shape)
        self.U = prim_to_cons(jnp.asarray(W0, dtype), gamma)
        self.t = 0.0
        self.step_count = 0
        self.history = []
        self._advance = jax.jit(self._advance_impl, static_argnums=(1,))

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
        phi, h = self.phi, self.h
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
        grads = np.gradient(phi, *h)
        nrm = np.stack([gd.ravel()[gi] for gd in grads])
        nrm /= np.linalg.norm(nrm, axis=0) + 1e-30
        shape = phi.shape
        ijk = np.stack(np.unravel_index(gi, shape)).astype(float)
        xg = np.stack([self.axes[d][ijk[d].astype(int)] for d in range(3)])
        dist0 = np.maximum(-2.0 * phi.ravel()[gi], 0.5 * h.min())
        fluid_f = self.fluid.ravel()

        cidx = np.zeros((gi.size, 8), np.int64)
        cw = np.zeros((gi.size, 8))
        todo = np.ones(gi.size, bool)
        for extra in (0.0, 1.0, 2.0, 3.0):
            if not todo.any():
                break
            sel = np.flatnonzero(todo)
            xi = xg[:, sel] + (dist0[sel] + extra * h.min()) * nrm[:, sel]
            f = np.stack([(xi[d] - self.lo[d]) / h[d] - 0.5 + NG for d in range(3)])
            i0 = np.floor(f).astype(np.int64)
            fr = f - i0
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
        self.ib = dict(g=jnp.asarray(gi), c=jnp.asarray(cidx), w=jnp.asarray(cw, self.dtype),
                       n=jnp.asarray(nrm, self.dtype))

    # ---------------------------------------------------------------- kernels
    def _fill_bc(self, W):
        n = self.n
        for d in range(3):
            ax = d + 1
            N = n[d]
            for side in (0, 1):
                kind = self.bcs[(d, side)]
                lo_idx = (0, NG) if side == 0 else (N + NG, N + 2 * NG)
                if kind == "periodic":
                    src = _sl(W, ax, N, N + NG) if side == 0 else _sl(W, ax, NG, 2 * NG)
                elif kind == "extrapolate":
                    edge = _sl(W, ax, NG, NG + 1) if side == 0 else _sl(W, ax, N + NG - 1, N + NG)
                    src = jnp.repeat(edge, NG, axis=ax)
                else:  # freestream
                    shp = list(W.shape)
                    shp[ax] = NG
                    src = jnp.broadcast_to(self.W_inf_j.reshape(5, 1, 1, 1), shp)
                W = lax.dynamic_update_slice_in_dim(W, src.astype(W.dtype), lo_idx[0], axis=ax)
        return W

    def _fill_ib(self, W):
        if self.ib is None:
            return W
        ib = self.ib
        Wf = W.reshape(5, -1)
        qi = jnp.sum(Wf[:, ib["c"]] * ib["w"][None], axis=-1)  # (5, Ng)
        vel = qi[1:4]
        vn = jnp.sum(vel * ib["n"], axis=0)
        vel = vel - 2.0 * vn[None] * ib["n"]
        qg = jnp.concatenate([qi[:1], vel, qi[4:]], axis=0)
        return Wf.at[:, ib["g"]].set(qg).reshape(W.shape)

    def _residual(self, W, order):
        n, g = self.n, self.gamma
        p = W[4]
        s = jnp.zeros_like(p)
        for d in range(3):
            if n[d] > 1:
                pp, pm = jnp.roll(p, -1, d), jnp.roll(p, 1, d)
                s = jnp.maximum(s, jnp.abs(pp - pm) / jnp.minimum(pp, pm))
        shock = s > self.shock_threshold
        R = 0.0
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
            dL, dC, dR = q0 - qm1, qp1 - q0, qp2 - qp1
            WL = q0 + 0.5 * order * _limit(dL, dC, self.limiter)
            WR = qp1 - 0.5 * order * _limit(dC, dR, self.limiter)
            bad = (WL[0] <= 0) | (WL[4] <= 0) | (WR[0] <= 0) | (WR[4] <= 0)
            WL = jnp.where(bad[None], q0, WL)
            WR = jnp.where(bad[None], qp1, WR)
            hl = _sl(Sc, d, 1, N + 2) | _sl(Sc, d, 2, N + 3)
            F = hllc_flux(WL, WR, d, g, hl)
            R = R - (_sl(F, ax, 1, N + 1) - _sl(F, ax, 0, N)) / self.h[d]
        return R

    def _prim(self, U):
        W = cons_to_prim(U, self.gamma, self.rho_min, self.p_min)
        return self._fill_ib(self._fill_bc(W))

    def _interior(self, A):
        return A[:, NG:-NG, NG:-NG, NG:-NG]

    def _advance_impl(self, U, nsteps, order):
        g, m = self.gamma, self.update_mask

        def update(U, Ui_new):
            Ui = self._interior(U)
            Ui_new = jnp.where(m[None] > 0, Ui_new, Ui)
            return lax.dynamic_update_slice(U, Ui_new, (0, NG, NG, NG))

        def body(_, carry):
            U, t, _r = carry
            W = self._prim(U)
            Wi = self._interior(W)
            c = jnp.sqrt(g * Wi[4] / Wi[0])
            rate = sum((jnp.abs(Wi[1 + d]) + c) / self.h[d] for d in range(3) if self.n[d] > 1)
            dt = self.cfl / jnp.max(rate * m)
            R0 = self._residual(W, order)
            U1 = update(U, self._interior(U) + dt * R0)
            R1 = self._residual(self._prim(U1), order)
            U2 = update(U, 0.5 * self._interior(U) + 0.5 * (self._interior(U1) + dt * R1))
            res = jnp.sqrt(jnp.sum((R0[0] * m) ** 2) / jnp.sum(m))
            return U2, t + dt, res

        return lax.fori_loop(0, nsteps, body, (U, jnp.asarray(0.0, U.dtype), jnp.asarray(0.0, U.dtype)))

    # -------------------------------------------------------------- driver
    def run(self, nsteps, chunk=100, first_order_steps=200, report=None):
        """Advance nsteps time steps. `report(solver)` is called after every chunk."""
        done = 0
        t0 = time.time()
        while done < nsteps:
            k = min(chunk, nsteps - done)
            order = 0.0 if self.step_count < first_order_steps else 1.0
            self.U, dt_sum, res = self._advance(self.U, k, order)
            res = float(res)
            self.t += float(dt_sum)
            self.step_count += k
            done += k
            self.history.append((self.step_count, self.t, res))
            if not np.isfinite(res):
                raise FloatingPointError(f"solution diverged at step {self.step_count}")
            if self.verbose:
                el = time.time() - t0
                cps = np.prod(self.n) * done / el / 1e6
                print(f"step {self.step_count:6d}  t = {self.t:.4e} s  "
                      f"res(rho) = {res/self.fs['rho']:.3e} 1/s  [{cps:.1f} Mcell-steps/s]", flush=True)
            if report:
                report(self)
        return self

    # ------------------------------------------------------------- outputs
    def primitive(self):
        """Primitive variables on the padded grid (numpy), ghosts filled."""
        return np.asarray(jax.jit(self._prim)(self.U))

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
        eps = band * self.h.min()
        bandm = np.abs(phi) < eps
        delta = np.where(bandm, (1 + np.cos(np.pi * phi / eps)) / (2 * eps), 0.0)
        grads = np.gradient(phi, *self.h)
        gn = np.sqrt(sum(gd * gd for gd in grads)) + 1e-30
        dV = np.prod(self.h)
        dp = (W[4] - self.W_inf[4]) * delta * dV
        idx = np.nonzero(bandm)
        F = np.array([-(dp[idx] * grads[d][idx] / gn[idx]).sum() for d in range(3)])
        r = np.stack([self.axes[d][idx[d]] - x_ref[d] for d in range(3)])
        dF = np.stack([-(dp[idx] * grads[d][idx] / gn[idx]) for d in range(3)])
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
        np.savez_compressed(path, U=np.asarray(self.U), t=self.t, step=self.step_count,
                            lo=self.lo, h=self.h, n=self.n, history=np.array(self.history))

    def load(self, path):
        d = np.load(path)
        self.U = jnp.asarray(d["U"], self.dtype)
        self.t, self.step_count = float(d["t"]), int(d["step"])
        self.history = [tuple(r) for r in d["history"]]
