"""Validation 2: supersonic flow over a circular cylinder (tests the immersed boundary).

Checks against three independent references:
  * bow shock standoff distance      Billig (1967): delta/R = 0.386 exp(4.67 / M^2)
  * stagnation-point pressure        Rayleigh pitot formula (exact for inviscid flow)
  * windward surface pressure        modified Newtonian: Cp = Cp_max cos^2(theta)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rocketcfd import Solver
from rocketcfd.output import plot_slices

G = 1.4


def pitot_ratio(M):
    """p0' / p_inf behind a normal shock (Rayleigh)."""
    return ((G + 1) ** 2 * M ** 2 / (4 * G * M ** 2 - 2 * (G - 1))) ** (G / (G - 1)) \
        * (1 - G + 2 * G * M ** 2) / (G + 1)


def main(M=4.0, cells_per_R=40, out="results", stretched=False):
    os.makedirs(out, exist_ok=True)
    R = 1.0
    h = R / cells_per_R
    fs = dict(rho=G, u=M, v=0.0, w=0.0, p=1.0)  # speed of sound = 1
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - R
    faces = None
    if stretched:
        # fine only in a zone around the body and shock layer; 4x coarser in the farfield
        from rocketcfd.grid import Zone, stretched_faces, describe
        zone = Zone((-1.8, -1.6, 0.0), (1.2, 1.6, h), h, "body")
        faces = stretched_faces((-3.0, -4.5, 0.0), (3.5, 4.5, h), 4 * h, [zone], growth=1.1)
        faces[2] = np.array([0.0, h])
        print(describe(faces))
    s = Solver((-3.0, -4.5, 0.0), (3.5, 4.5, h), (h, h, h), fs, sdf=sdf, faces=faces,
               bcs={(2, 0): "periodic", (2, 1): "periodic"}, cfl=0.7)
    s.run(4000, chunk=500, first_order_steps=500)
    f = s.fields()
    x, y = f["x"], f["y"]
    j0 = np.argmin(np.abs(y))
    rho_line = f["rho"][:, j0, 0]
    p_line = f["p"][:, j0, 0]
    # shock = first cell (from upstream) where density exceeds the mean of pre/post-shock values
    rho_post = G * (G + 1) * M ** 2 / ((G - 1) * M ** 2 + 2)
    i_sh = np.argmax(rho_line > 0.5 * (G + rho_post))
    xs = np.interp(0.5 * (G + rho_post), rho_line[i_sh - 1:i_sh + 1], x[i_sh - 1:i_sh + 1])
    standoff = -R - xs
    billig = 0.386 * np.exp(4.67 / M ** 2) * R
    # stagnation pressure: last fluid cell on the stagnation line
    i_w = np.flatnonzero(~f["solid"][:, j0, 0] & (x < 0))[-1]
    p_stag = p_line[i_w]
    p0 = pitot_ratio(M)
    # surface pressure distribution vs modified Newtonian
    q = 0.5 * G * M ** 2
    Cpmax = (p0 - 1.0) / q
    th = np.linspace(0, np.radians(80), 41)
    xi, yi = -(R + 0.6 * h) * np.cos(th), (R + 0.6 * h) * np.sin(th)
    from scipy.interpolate import RegularGridInterpolator
    P = np.ma.filled(np.ma.masked_where(f["solid"][:, :, 0], f["p"][:, :, 0]), np.nan)
    # nearest-fluid sampling just off the wall
    pi = RegularGridInterpolator((x, y), P, method="nearest", bounds_error=False)(np.c_[xi, yi])
    cp_num = (pi - 1.0) / q
    cp_newt = Cpmax * np.cos(th) ** 2

    print(f"M = {M}, {cells_per_R} cells/R")
    print(f"  standoff   CFD {standoff:.4f} R   Billig {billig:.4f} R   ({100*(standoff/billig-1):+.1f} %)")
    print(f"  p_stag     CFD {p_stag:.3f}       Rayleigh {p0:.3f}     ({100*(p_stag/p0-1):+.1f} %)")
    tag = "_stretched" if stretched else ""
    plot_slices(f, f"{out}/validate_cylinder_M{M:g}{tag}.png",
                title=f"Cylinder, M = {M}: standoff {standoff:.3f}R (Billig {billig:.3f}R), "
                      f"p_stag {p_stag:.2f} (Rayleigh {p0:.2f})", p_inf=1.0)
    fig, a = plt.subplots(figsize=(6, 4), constrained_layout=True)
    a.plot(np.degrees(th), cp_newt, "k-", label="modified Newtonian")
    a.plot(np.degrees(th), cp_num, "o", color="C3", ms=4, label="rocketcfd")
    a.set_xlabel("angle from stagnation point [deg]")
    a.set_ylabel("Cp")
    a.set_title(f"Cylinder surface pressure, M = {M}")
    a.legend()
    fig.savefig(f"{out}/validate_cylinder_cp_M{M:g}{tag}.png", dpi=130)
    return dict(standoff=standoff, billig=billig, p_stag=p_stag, p0=p0)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(float(args[0]) if args else 4.0, stretched="--stretched" in sys.argv)
