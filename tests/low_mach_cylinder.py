"""Low-Mach accuracy: inviscid flow round a cylinder at Mach 0.1 (Guillard & Viozat test).

The exact (potential-flow) answer has zero drag and Cp = 1 at both stagnation points.
Upwind schemes add dissipation that does not fall with Mach number, which shows as a
spurious wake, drag and a low rear-stagnation pressure. Compares the HLLC baseline with
the Thornber et al. (2008) low-Mach correction (Solver(low_mach=True)).

  python tests/low_mach_cylinder.py [mach]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def run(mach, low_mach, h=0.04, iters=1500):
    G = 1.4
    fs = dict(rho=G, u=mach, v=0.0, w=0.0, p=1.0)
    faces = [axis_faces(-12.0, 12.0, 30 * h, [(-0.7, 0.7, h)], 1.08),
             axis_faces(-12.0, 12.0, 30 * h, [(-0.7, 0.7, h)], 1.08), np.array([0.0, h])]
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5
    s = Solver(None, None, None, fs, sdf=sdf, faces=faces, bcs={(2, 0): "periodic", (2, 1): "periodic"},
               low_mach=low_mach, verbose=False)
    s.run(iters, chunk=500, first_order_steps=100, steady=True, implicit=True, cfl_ramp=100)
    ld = s.loads(S_ref=1.0 * h, L_ref=1.0)
    f = s.fields()
    q = 0.5 * G * mach ** 2
    j = np.argmin(np.abs(f["y"]))
    xs = f["x"]
    i_front = np.flatnonzero((xs < -0.5) & ~f["solid"][:, j, 0])[-1]
    i_rear = np.flatnonzero((xs > 0.5) & ~f["solid"][:, j, 0])[0]
    cp = lambda i: (f["p"][i, j, 0] - 1.0) / q
    # whole surface: Cp 1.5 cells off the wall vs potential flow, Cp = 1 - 4 sin^2(theta) at the wall
    from scipy.ndimage import map_coordinates
    W = s.primitive()
    th = np.radians(np.arange(0.0, 360.0, 5.0))
    r = 0.5 + 1.5 * h
    pts = np.stack([-r * np.cos(th), r * np.sin(th), np.full(th.size, 0.5 * h)])
    idx = [np.interp(pts[d], s.axes[d], np.arange(len(s.axes[d]))) for d in range(3)]
    cps = (map_coordinates(W[4], idx, order=1) - 1.0) / q
    # potential flow at radius r: Cp = 2 (R/r)^2 cos(2 theta) - (R/r)^4
    a = (0.5 / r) ** 2
    exact = 2 * a * np.cos(2 * th) - a * a
    rms = float(np.sqrt(np.mean((cps - exact) ** 2)))
    return ld["CD"], cp(i_front), cp(i_rear), s.history[-1][2], rms


if __name__ == "__main__":
    M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    for lm in (False, True):
        cd, cpf, cpr, res, rms = run(M, lm)
        print(f"Mach {M} {'low-Mach fix' if lm else 'baseline   '}: CD {cd:+.4f} (exact 0)  Cp front {cpf:.3f}  "
              f"Cp rear {cpr:.3f} (exact ~1)  rms Cp error round the body {rms:.3f}", flush=True)
