"""Calorically perfect gas: variable conversions and approximate Riemann solvers.

Primitive W = [rho, u, v, w, p];  conservative U = [rho, rho u, rho v, rho w, E].
Arrays have the variable index first: shape (5, ...).
"""
import jax.numpy as jnp


def prim_to_cons(W, gamma):
    rho, u, v, w, p = W
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
    return jnp.stack([rho, rho * u, rho * v, rho * w, E])


def cons_to_prim(U, gamma, rho_min, p_min):
    rho = jnp.maximum(U[0], rho_min)
    u, v, w = U[1] / rho, U[2] / rho, U[3] / rho
    p = (gamma - 1.0) * (U[4] - 0.5 * rho * (u * u + v * v + w * w))
    return jnp.stack([rho, u, v, w, jnp.maximum(p, p_min)])


def _phys_flux(W, U, d):
    """Physical flux in direction d (0, 1, 2)."""
    un = W[1 + d]
    p = W[4]
    F = U * un
    F = F.at[1 + d].add(p)
    F = F.at[4].add(p * un)
    return F


def _wave_speeds(WL, WR, d, gamma):
    """Einfeldt (Roe-average based) signal speeds."""
    rL, rR = WL[0], WR[0]
    unL, unR = WL[1 + d], WR[1 + d]
    cL = jnp.sqrt(gamma * WL[4] / rL)
    cR = jnp.sqrt(gamma * WR[4] / rR)
    sL, sR = jnp.sqrt(rL), jnp.sqrt(rR)
    wt = 1.0 / (sL + sR)
    HL = (gamma / (gamma - 1.0)) * WL[4] / rL + 0.5 * (WL[1] ** 2 + WL[2] ** 2 + WL[3] ** 2)
    HR = (gamma / (gamma - 1.0)) * WR[4] / rR + 0.5 * (WR[1] ** 2 + WR[2] ** 2 + WR[3] ** 2)
    ut = [(sL * WL[1 + k] + sR * WR[1 + k]) * wt for k in range(3)]
    Ht = (sL * HL + sR * HR) * wt
    ct = jnp.sqrt(jnp.maximum((gamma - 1.0) * (Ht - 0.5 * (ut[0] ** 2 + ut[1] ** 2 + ut[2] ** 2)), 1e-12))
    SL = jnp.minimum(unL - cL, ut[d] - ct)
    SR = jnp.maximum(unR + cR, ut[d] + ct)
    return SL, SR


def hlle_flux(WL, WR, d, gamma):
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    FL, FR = _phys_flux(WL, UL, d), _phys_flux(WR, UR, d)
    SL, SR = _wave_speeds(WL, WR, d, gamma)
    SL = jnp.minimum(SL, 0.0)
    SR = jnp.maximum(SR, 0.0)
    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-30)


def hllc_flux(WL, WR, d, gamma, use_hlle=None):
    """HLLC flux (Toro). If use_hlle (bool array) is given, fall back to HLLE there
    (cures the carbuncle / odd-even decoupling at strong grid-aligned shocks)."""
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    FL, FR = _phys_flux(WL, UL, d), _phys_flux(WR, UR, d)
    SL, SR = _wave_speeds(WL, WR, d, gamma)
    rL, rR = WL[0], WR[0]
    unL, unR = WL[1 + d], WR[1 + d]
    pL, pR = WL[4], WR[4]
    mL = rL * (SL - unL)
    mR = rR * (SR - unR)
    Ss = (pR - pL + unL * mL - unR * mR) / (mL - mR + 1e-30)

    def ustar(W, U, S, m):
        r = m / (S - Ss + 1e-30)
        comps = [r]
        for k in range(3):
            comps.append(r * (Ss if k == d else W[1 + k]))
        un = W[1 + d]
        comps.append(r * (U[4] / W[0] + (Ss - un) * (Ss + W[4] / (W[0] * (S - un) + 1e-30))))
        return jnp.stack(comps)

    FsL = FL + SL * (ustar(WL, UL, SL, mL) - UL)
    FsR = FR + SR * (ustar(WR, UR, SR, mR) - UR)
    F = jnp.where(SL >= 0.0, FL, jnp.where(Ss >= 0.0, FsL, jnp.where(SR > 0.0, FsR, FR)))
    if use_hlle is not None:
        SLe, SRe = jnp.minimum(SL, 0.0), jnp.maximum(SR, 0.0)
        Fe = (SRe * FL - SLe * FR + SLe * SRe * (UR - UL)) / (SRe - SLe + 1e-30)
        F = jnp.where(use_hlle, Fe, F)
    return F
