"""WENO5-Z against an immersed wall with a strong shock, time-accurate (the LES setting): the
Mach 4 cylinder of tests/limiter_shocks.py, explicit global time steps from a freestream start.
Checks the run stays finite and the bow-shock standoff and stagnation pressure match Billig and
Rayleigh as well as with Koren MUSCL.

  python tests/weno_ib_shock.py [steps]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def cylinder_unsteady(limiter, steps=3000, h=0.02):
    G, M = 1.4, 4.0
    faces = [axis_faces(-2.0, 1.5, 4 * h, [(-1.2, 0.6, h)], 1.08),
             axis_faces(-2.5, 2.5, 4 * h, [(-1.4, 1.4, h)], 1.08), np.array([0.0, h])]
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5
    s = Solver(None, None, None, dict(rho=G, u=M, v=0.0, w=0.0, p=1.0), sdf=sdf, faces=faces,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, limiter=limiter, verbose=False, cfl=0.5)
    s.run(steps, chunk=500, first_order_steps=200)
    f = s.fields()
    if not np.isfinite(f["rho"]).all():
        return np.nan, np.nan
    j = np.argmin(np.abs(f["y"]))
    x, p = f["x"], f["p"][:, j, 0]
    up = x < -0.5
    i_sh = np.flatnonzero(up & (p > 1.5 * p[0]))[0]
    standoff = -0.5 - x[i_sh]
    i_st = np.flatnonzero(up & ~f["solid"][:, j, 0])[-1]
    p02 = ((G + 1) * M ** 2 / 2) ** (G / (G - 1)) * ((G + 1) / (2 * G * M ** 2 - (G - 1))) ** (1 / (G - 1))
    billig = 0.5 * 0.386 * np.exp(4.67 / M ** 2)
    return standoff / billig - 1, p[i_st] / p02 - 1


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    for lim in ("koren", "weno5z"):
        so, ps = cylinder_unsteady(lim, steps)
        print(f"{lim:7s} Mach 4 cylinder, time-accurate {steps} steps: standoff {100*so:+.1f} % (Billig), "
              f"p_stag {100*ps:+.1f} % (Rayleigh)", flush=True)
