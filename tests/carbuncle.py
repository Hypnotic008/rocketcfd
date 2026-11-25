"""High-Mach robustness: Mach 8 and 12 cylinders. A carbuncle (the odd-even decoupling of
HLLC-type fluxes on grid-aligned strong shocks) shows as a bow shock pushed far forward on
the stagnation line; compare the standoff with Billig's correlation and the stagnation
pressure with the Rayleigh pitot formula.

  python tests/carbuncle.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def run(M, h=0.02):
    G = 1.4
    faces = [axis_faces(-2.0, 1.5, 4 * h, [(-1.0, 0.6, h)], 1.08),
             axis_faces(-3.0, 3.0, 4 * h, [(-1.6, 1.6, h)], 1.08), np.array([0.0, h])]
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5
    s = Solver(None, None, None, dict(rho=G, u=M, v=0.0, w=0.0, p=1.0), sdf=sdf, faces=faces,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, verbose=False)
    s.run(1500, chunk=500, first_order_steps=150, steady=True, implicit=True, cfl_ramp=150)
    f = s.fields()
    j = np.argmin(np.abs(f["y"]))
    x, p = f["x"], f["p"][:, j, 0]
    up = x < -0.5
    i_sh = np.flatnonzero(up & (p > 1.5))[0]
    standoff = -0.5 - x[i_sh]
    i_st = np.flatnonzero(up & ~f["solid"][:, j, 0])[-1]
    p02 = ((G + 1) * M ** 2 / 2) ** (G / (G - 1)) * ((G + 1) / (2 * G * M ** 2 - (G - 1))) ** (1 / (G - 1))
    billig = 0.5 * 0.386 * np.exp(4.67 / M ** 2)
    # shock shape symmetry: shock x on rows y = +-0.4
    def shock_x(yv):
        jj = np.argmin(np.abs(f["y"] - yv))
        pp = f["p"][:, jj, 0]
        return x[np.flatnonzero((x < 0.0) & (pp > 1.5))[0]]
    asym = abs(shock_x(0.4) - shock_x(-0.4))
    return standoff / billig - 1, p[i_st] / p02 - 1, asym, s.history[-1][2]


if __name__ == "__main__":
    for M in [float(v) for v in sys.argv[1:]] or [8.0, 12.0]:
        so, ps, asym, res = run(M)
        print(f"Mach {M}: standoff {100*so:+.1f} % (Billig), stagnation pressure {100*ps:+.1f} % (Rayleigh), "
              f"shock asymmetry {asym:.3f} m, final residual {res:.2e}", flush=True)
