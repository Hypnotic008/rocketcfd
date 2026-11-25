"""Transonic validation: NACA 0012, Mach 0.8, 1.25 deg, inviscid (AGARD AR-211 test case 1).

Euler reference solutions give CL ~ 0.35-0.36, CD ~ 0.022-0.023 (wave drag), with a strong
shock on the upper surface near 62 % chord and a weak one below near 35 %. Tests shocks in a
subsonic freestream (where the low-Mach correction is on), the implicit solver in transonic
flow, the characteristic farfield and the immersed boundary on a thin body with a sharp edge.

  python tests/naca0012.py [h_near]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def naca_polygon(n=400):
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    t = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    t[-1] = 0.0                                     # sharp trailing edge
    return np.concatenate([np.stack([x, t], 1), np.stack([x[::-1], -t[::-1]], 1)[1:]])


def polygon_sdf(poly):
    A, B = poly[:-1], poly[1:]

    def sdf(X, Y, Z):
        P = np.stack([X.ravel(), Y.ravel()], 1)
        out = np.empty(P.shape[0])
        for i0 in range(0, P.shape[0], 20000):
            p = P[i0:i0 + 20000, None, :]
            ab = B - A
            t = np.clip(((p - A) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
            d = np.sqrt((((A + t[..., None] * ab) - p) ** 2).sum(-1)).min(1)
            # inside test: |y| below the thickness at that x
            x, y = p[:, 0, 0], p[:, 0, 1]
            xx = np.clip(x, 0, 1)
            th = 0.6 * (0.2969 * np.sqrt(xx) - 0.1260 * xx - 0.3516 * xx ** 2 + 0.2843 * xx ** 3 - 0.1036 * xx ** 4)
            inside = (x > 0) & (x < 1) & (np.abs(y) < th)
            out[i0:i0 + 20000] = np.where(inside, -d, d)
        return out.reshape(X.shape)
    return sdf


def run(h=0.004, M=0.8, alpha=1.25, iters=3000):
    G = 1.4
    a = np.radians(alpha)
    fs = dict(rho=G, u=M * np.cos(a), v=M * np.sin(a), w=0.0, p=1.0)
    faces = [axis_faces(-20.0, 21.0, 60 * h, [(-0.08, 1.08, h)], 1.08),
             axis_faces(-20.0, 20.0, 60 * h, [(-0.12, 0.12, h)], 1.08), np.array([0.0, h])]
    s = Solver(None, None, None, fs, sdf=polygon_sdf(naca_polygon()), faces=faces,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, verbose=False)
    print(f"{int(np.prod(s.n))} cells, h {h}", flush=True)
    s.run(iters, chunk=500, first_order_steps=200, steady=True, implicit=True, cfl_ramp=200)
    ld = s.loads(S_ref=1.0 * h, L_ref=1.0, x_ref=(0.25, 0.0, 0.0))
    # lift / drag in wind axes (loads' CL uses the body x axis as reference; recompute)
    F = ld["F"] / (0.5 * G * M ** 2 * h)
    CL = -F[0] * np.sin(a) + F[1] * np.cos(a)
    CD = F[0] * np.cos(a) + F[1] * np.sin(a)
    f = s.fields()
    # shock position on the upper surface: largest pressure rise along the first row above the wall
    x = f["x"]
    sel = (x > 0.2) & (x < 0.95)
    xs = x[sel]
    yu = 0.6 * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs ** 2 + 0.2843 * xs ** 3 - 0.1036 * xs ** 4) + 1.5 * h
    pu = np.array([np.interp(yy, f["y"], f["p"][i, :, 0]) for i, yy in zip(np.flatnonzero(sel), yu)])
    shock = xs[np.argmax(np.gradient(pu, xs))]
    return CL, CD, shock, s.history[-1][2]


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.004
    CL, CD, xs, res = run(h)
    print(f"NACA 0012 M 0.8 a 1.25 (h {h}): CL {CL:.4f} (ref 0.35-0.36)  CD {CD:.4f} (ref 0.022-0.023)  "
          f"upper shock x/c {xs:.2f} (ref ~0.62)  final res {res:.2e}", flush=True)
