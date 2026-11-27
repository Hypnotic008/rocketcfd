"""Order of accuracy on smooth flow: isentropic vortex advected one period through a
periodic box (Yee et al. 1999); the exact solution after one period is the initial field.

  python tests/vortex_order.py            # van Leer vs Koren (kappa = 1/3), N = 32, 64, 128
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver

G = 1.4
BETA = 5.0


def vortex(X, Y):
    r2 = X ** 2 + Y ** 2
    f = np.exp(0.5 * (1.0 - r2))
    T = 1.0 - (G - 1.0) * BETA ** 2 / (8.0 * G * np.pi ** 2) * f ** 2
    rho = T ** (1.0 / (G - 1.0))
    return np.stack([rho, 1.0 - BETA / (2 * np.pi) * Y * f, BETA / (2 * np.pi) * X * f, 0.0 * X, rho * T])


def run(n, limiter, backend="warp"):
    h = 10.0 / n
    s = Solver((-5.0, -5.0, 0.0), (5.0, 5.0, h), h, dict(rho=1.0, u=1.0, v=0.0, w=0.0, p=1.0),
               bcs={(d, k): "periodic" for d in range(3) for k in (0, 1)}, limiter=limiter, cfl=0.4,
               verbose=False, backend=backend)
    X, Y = np.meshgrid(s.axes[0], s.axes[1], indexing="ij")
    W0 = vortex(X, Y)[:, :, :, None] * np.ones((1, 1, 1, s.U.shape[3]))
    s.set_primitive(W0)
    t_end = 10.0
    while s.t < t_end - 1e-9:
        s.run(1, chunk=1, first_order_steps=0)
        if s.t > t_end:
            break
    f = s.fields()
    Xi, Yi = np.meshgrid(f["x"], f["y"], indexing="ij")
    # compare with the exact field shifted by the overshoot past t_end
    ex = vortex(Xi - (s.t - t_end), Yi)[0]
    return np.mean(np.abs(f["rho"][:, :, 0] - ex))


if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1:] if v.isdigit()] or [32, 64, 128]
    for lim in ("vanleer", "koren", "weno5z"):
        errs = [run(n, lim) for n in ns]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        print(f"{lim:8s} L1(rho) " + "  ".join(f"N={n}: {e:.2e}" for n, e in zip(ns, errs))
              + "   order " + " ".join(f"{o:.2f}" for o in orders), flush=True)
