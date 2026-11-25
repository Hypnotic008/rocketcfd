"""Robustness of the reconstruction choices with strong shocks: Sod shock tube (exact
solution) and a Mach 4 cylinder (bow-shock standoff, stagnation pressure).

  python tests/limiter_shocks.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cases"))


def sod(limiter, n=400, low_mach=False):
    from validate_sod import exact
    h = 1.0 / n
    s = Solver((0.0, 0.0, 0.0), (1.0, h, h), h, dict(rho=1.0, u=0.0, v=0.0, w=0.0, p=1.0),
               bcs={(0, 0): "extrapolate", (0, 1): "extrapolate", (1, 0): "periodic", (1, 1): "periodic",
                    (2, 0): "periodic", (2, 1): "periodic"}, limiter=limiter, cfl=0.5, verbose=False,
               low_mach=low_mach,
               backend="warp")
    W = s.primitive()
    left = s.axes[0] < 0.5
    W[0][left], W[4][left] = 1.0, 1.0
    W[0][~left], W[4][~left] = 0.125, 0.1
    W[1:4] = 0.0
    s.set_primitive(W)
    while s.t < 0.2:
        s.run(1, chunk=1, first_order_steps=0)
    f = s.fields()
    ex = exact(f["x"], s.t)
    return np.mean(np.abs(f["rho"][:, 0, 0] - ex[0]))


def cylinder(limiter, h=0.02, low_mach=False):
    G = 1.4
    M = 4.0
    faces = [axis_faces(-2.0, 1.5, 4 * h, [(-1.2, 0.6, h)], 1.08),
             axis_faces(-2.5, 2.5, 4 * h, [(-1.4, 1.4, h)], 1.08), np.array([0.0, h])]
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5
    s = Solver(None, None, None, dict(rho=G, u=M, v=0.0, w=0.0, p=1.0), sdf=sdf, faces=faces,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, limiter=limiter, verbose=False, low_mach=low_mach)
    s.run(1200, chunk=400, first_order_steps=100, steady=True, implicit=True, cfl_ramp=100)
    f = s.fields()
    j = np.argmin(np.abs(f["y"]))
    x = f["x"]
    p = f["p"][:, j, 0]
    up = x < -0.5
    i_sh = np.flatnonzero(up & (p > 1.5 * p[0]))[0]
    standoff = -0.5 - x[i_sh]
    i_st = np.flatnonzero(up & ~f["solid"][:, j, 0])[-1]
    # Rayleigh pitot
    p02 = ((G + 1) * M ** 2 / 2) ** (G / (G - 1)) * ((G + 1) / (2 * G * M ** 2 - (G - 1))) ** (1 / (G - 1))
    billig = 0.5 * 0.386 * np.exp(4.67 / M ** 2)
    return standoff / billig - 1, p[i_st] / p02 - 1, s.history[-1][2]


if __name__ == "__main__":
    modes = sys.argv[1:] or ["vanleer", "koren", "koren+lowmach"]
    for mode in modes:
        lim, lm = mode.split("+")[0], "lowmach" in mode
        e = sod(lim, low_mach=lm)
        so, ps, res = cylinder(lim, low_mach=lm)
        print(f"{mode:14s} Sod L1(rho) {e:.5f}   Mach 4 cylinder: standoff {100*so:+.1f} % (Billig), "
              f"p_stag {100*ps:+.1f} % (Rayleigh), final res {res:.2e}", flush=True)
