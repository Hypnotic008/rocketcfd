"""Validation 3: thin flat plate at incidence in supersonic flow (tests thin walls).

A 1 mm plate is far thinner than a cell, so it exists only as blocked faces. Exact
inviscid reference: shock-expansion theory (oblique shock below, Prandtl-Meyer
expansion above), which gives the pressure on each face and the normal force.

  python cases/validate_flat_plate.py               # plate along the grid, flow tilted by alpha
  python cases/validate_flat_plate.py --inclined    # plate tilted across the grid (staircase walls)
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import brentq
from rocketcfd import Solver
from rocketcfd.grid import Zone, stretched_faces

G = 1.4


def oblique_p(M, theta):
    """p2/p1 behind the weak oblique shock that turns the flow by theta."""
    f = lambda b: np.tan(theta) - 2 / np.tan(b) * (M**2 * np.sin(b)**2 - 1) / (M**2 * (G + np.cos(2 * b)) + 2)
    mu = np.arcsin(1 / M)
    beta = brentq(f, mu + 1e-9, np.radians(64.0))
    return 1 + 2 * G / (G + 1) * (M**2 * np.sin(beta)**2 - 1)


def pm_p(M, theta):
    """p3/p1 after a Prandtl-Meyer expansion turning the flow by theta."""
    nu = lambda m: (np.sqrt((G + 1) / (G - 1)) * np.arctan(np.sqrt((G - 1) / (G + 1) * (m**2 - 1)))
                    - np.arctan(np.sqrt(m**2 - 1)))
    M3 = brentq(lambda m: nu(m) - nu(M) - theta, M, 50)
    return ((1 + 0.5 * (G - 1) * M**2) / (1 + 0.5 * (G - 1) * M3**2)) ** (G / (G - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--inclined", action="store_true")
    a = ap.parse_args()
    M, al, h = a.mach, np.radians(a.alpha), a.h
    t = 0.001
    if a.inclined:
        # plate rotated by -alpha (leading edge up), flow along +x
        fs = dict(rho=G, u=M, v=0.0, w=0.0, p=1.0)
        ca, sa = np.cos(-al), np.sin(-al)

        def sdf(X, Y, Z):
            xs = X * ca + Y * sa
            ys = -X * sa + Y * ca
            return np.sqrt(np.maximum(np.abs(xs) - 0.5, 0) ** 2 + np.maximum(np.abs(ys) - 0.5 * t, 0) ** 2) \
                + np.minimum(np.maximum(np.abs(xs) - 0.5, np.abs(ys) - 0.5 * t), 0)
        normal = np.array([np.sin(-al) * -1, np.cos(-al), 0.0])     # plate normal (upper side)
        normal = np.array([-sa, ca, 0.0])
    else:
        fs = dict(rho=G, u=M * np.cos(al), v=M * np.sin(al), w=0.0, p=1.0)

        def sdf(X, Y, Z):
            return np.sqrt(np.maximum(np.abs(X) - 0.5, 0) ** 2 + np.maximum(np.abs(Y) - 0.5 * t, 0) ** 2) \
                + np.minimum(np.maximum(np.abs(X) - 0.5, np.abs(Y) - 0.5 * t), 0)
        normal = np.array([0.0, 1.0, 0.0])

    faces = stretched_faces((-1.5, -1.5, 0.0), (2.5, 1.5, h), 4 * h,
                            [Zone((-0.6, -0.15, 0.0), (0.6, 0.15, h), h)], growth=1.1)
    faces[2] = np.array([0.0, h])
    s = Solver(None, None, None, fs, faces=faces, thin_sdf=sdf,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, cfl=0.7)
    s.run(3000, chunk=500, first_order_steps=300)
    ld = s.loads(S_ref=1.0 * h, L_ref=1.0, x_ref=(0.0, 0.0, 0.0))
    q = 0.5 * G * M**2
    # force on the plate is pushed from the high-pressure (lower) side: CN along +normal
    CN = ld["F_thin"] @ normal / (q * h)
    p_low, p_up = oblique_p(M, al), pm_p(M, al)
    CN_exact = (p_low - p_up) / q
    print(f"flat plate, M {M}, alpha {a.alpha} deg, h {h}, {'inclined (staircase)' if a.inclined else 'grid-aligned'}")
    print(f"  CN  CFD {CN:.4f}   shock-expansion {CN_exact:.4f}   ({100*(CN/CN_exact-1):+.1f} %)")
    # face pressures at mid-chord
    f = s.fields()
    if not a.inclined:
        i = np.argmin(np.abs(f["x"] - 0.0))
        j_up = np.flatnonzero(f["y"] > 0)[0]
        j_lo = np.flatnonzero(f["y"] < 0)[-1]
        print(f"  p_lower/p_inf  CFD {f['p'][i, j_lo, 0]:.4f}   exact {p_low:.4f}")
        print(f"  p_upper/p_inf  CFD {f['p'][i, j_up, 0]:.4f}   exact {p_up:.4f}")
    return CN, CN_exact


if __name__ == "__main__":
    main()
