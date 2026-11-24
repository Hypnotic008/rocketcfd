"""Validation 1: Sod shock tube vs the exact Riemann solution (Toro, ch. 4)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rocketcfd import Solver

G = 1.4


def exact(x, t, L=(1.0, 0.0, 1.0), R=(0.125, 0.0, 0.1), x0=0.5):
    rL, uL, pL = L
    rR, uR, pR = R
    cL, cR = np.sqrt(G * pL / rL), np.sqrt(G * pR / rR)

    def f(p, r, pk, c):
        if p > pk:
            A, B = 2 / ((G + 1) * r), (G - 1) / (G + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * c / (G - 1) * ((p / pk) ** ((G - 1) / (2 * G)) - 1)

    ps = brentq(lambda p: f(p, rL, pL, cL) + f(p, rR, pR, cR) + uR - uL, 1e-8, 10)
    us = 0.5 * (uL + uR) + 0.5 * (f(ps, rR, pR, cR) - f(ps, rL, pL, cL))
    rsL = rL * (ps / pL) ** (1 / G)
    csL = cL * (ps / pL) ** ((G - 1) / (2 * G))
    rsR = rR * ((ps / pR + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * ps / pR + 1))
    SR = uR + cR * np.sqrt((G + 1) / (2 * G) * ps / pR + (G - 1) / (2 * G))
    s = (x - x0) / t
    rho, u, p = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    for i, si in enumerate(s):
        if si < uL - cL:
            rho[i], u[i], p[i] = rL, uL, pL
        elif si < us - csL:
            u[i] = 2 / (G + 1) * (cL + (G - 1) / 2 * uL + si)
            c = 2 / (G + 1) * (cL + (G - 1) / 2 * (uL - si))
            rho[i] = rL * (c / cL) ** (2 / (G - 1))
            p[i] = pL * (c / cL) ** (2 * G / (G - 1))
        elif si < us:
            rho[i], u[i], p[i] = rsL, us, ps
        elif si < SR:
            rho[i], u[i], p[i] = rsR, us, ps
        else:
            rho[i], u[i], p[i] = rR, uR, pR
    return rho, u, p


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    N = 400
    s = Solver((0, 0, 0), (1, 1.0 / N, 1.0 / N), 1.0 / N, dict(rho=1, u=0, v=0, w=0, p=1),
               bcs={(0, 0): "extrapolate", (0, 1): "extrapolate", (1, 0): "periodic",
                    (1, 1): "periodic", (2, 0): "periodic", (2, 1): "periodic"},
               cfl=0.5, verbose=False)
    x = s.axes[0]
    W = np.zeros((5,) + s.phi.shape)
    left = (x < 0.5)[:, None, None]
    W[0] = np.where(left, 1.0, 0.125)
    W[4] = np.where(left, 1.0, 0.1)
    s.set_primitive(W)
    while s.t < 0.2:  # run to t = 0.2 (last chunk may slightly overshoot; use exact t)
        s.run(5, chunk=5, first_order_steps=0)
    f = s.fields()
    xc = f["x"]
    re, ue, pe = exact(xc, s.t)
    err = np.mean(np.abs(f["rho"][:, 0, 0] - re))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for a, num, ex, lab in zip(axs, (f["rho"], f["u"], f["p"]), (re, ue, pe), ("density", "velocity", "pressure")):
        a.plot(xc, ex, "k-", lw=1.2, label="exact")
        a.plot(xc, num[:, 0, 0], "o", ms=2.5, color="C3", label=f"rocketcfd (N={N})")
        a.set_title(lab)
        a.set_xlabel("x")
    axs[0].legend()
    fig.suptitle(f"Sod shock tube, t = {s.t:.4f}: mean |density error| = {err:.4f}")
    fig.savefig(f"{out}/validate_sod.png", dpi=130)
    print(f"Sod: t = {s.t:.4f}, steps {s.step_count}, L1(rho) error = {err:.5f}")
    return err


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
