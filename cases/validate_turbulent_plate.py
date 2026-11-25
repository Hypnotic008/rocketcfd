"""Validation 5: turbulent flat plate -- k-omega SST with the Spalding wall function.

Thin plate (both sides wetted), Mach 0.4, Re_L = 1e7 (default) or 1e9 (--re 1e9, Super Heavy
scale). The first cell centre sits in the log layer (y+ ~ 150 at Re 1e7), so the wall shear
comes from the law of the wall. References:
  * skin friction: White, Cf = 0.455 / ln^2(0.06 Re_x); Schlichting, Cf = 0.0592 Re_x^(-1/5)
  * velocity profile: log law u+ = ln(y+) / 0.41 + 5.0

  py cases\\validate_turbulent_plate.py               (Re 1e7, a few minutes on the GPU)
  py cases\\validate_turbulent_plate.py --re 1e9
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rocketcfd import Solver
from rocketcfd.grid import axis_faces
from rocketcfd.convergence import settled

KAPPA, B = 0.41, 5.0


def spalding_utau(u, y, nu):
    """Vectorised friction velocity from Spalding's law (same as the solver's wall function)."""
    ut = np.sqrt(nu * u / y)
    lg = y * ut / nu > 11
    for _ in range(8):
        ut = np.where(lg, KAPPA * u / np.log(np.maximum(9.0 * y * ut / nu, 1.0001)), ut)
    for _ in range(12):
        up = u / ut
        kp = KAPPA * up
        e = np.exp(-KAPPA * B)
        f = up + e * (np.exp(kp) - 1 - kp - 0.5 * kp ** 2 - kp ** 3 / 6) - y * ut / nu
        dfdup = 1 + e * KAPPA * (np.exp(kp) - 1 - kp - 0.5 * kp ** 2)
        df = dfdup * (-up / ut) - y / nu
        ut = np.maximum(ut - f / df, 0.1 * ut)
    return ut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=1e7)
    ap.add_argument("--mach", type=float, default=0.4)
    ap.add_argument("--hy", type=float, default=0.0008, help="cell height next to the plate [m]")
    ap.add_argument("--hx", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T, p, R, g = 288.15, 101325.0, 287.05, 1.4
    rho = p / (R * T)
    U = a.mach * np.sqrt(g * R * T)
    mu = rho * U * 1.0 / a.re
    nu = mu / rho
    fs = dict(rho=rho, u=U, v=0.0, w=0.0, p=p)
    faces = [axis_faces(-0.3, 4.0, 6 * a.hx, [(-0.05, 1.05, a.hx)], 1.08),
             axis_faces(-0.6, 0.6, 60 * a.hy, [(-0.03, 0.03, a.hy)], 1.08),
             np.array([0.0, a.hx])]
    t = 1e-4
    plate = lambda X, Y, Z: np.sqrt(np.maximum(np.abs(X - 0.5) - 0.5, 0) ** 2 + np.maximum(np.abs(Y) - t / 2, 0) ** 2) \
        + np.minimum(np.maximum(np.abs(X - 0.5) - 0.5, np.abs(Y) - t / 2), 0)
    s = Solver(None, None, None, fs, faces=faces, thin_sdf=plate, cfl=0.7,
               bcs={(2, 0): "periodic", (2, 1): "periodic"},
               viscous=dict(model="constant", mu=mu, R=R, Pr=0.72), turbulence=dict(model="sst"))
    hist = []
    st = (0.3, 0.6, 0.9)

    def report(sv):
        f = sv.fields()
        j = np.flatnonzero(f["y"] > 0)[0]
        vals = tuple(f["u"][np.argmin(np.abs(f["x"] - xs)), j, 0] / U for xs in st)
        hist.append(vals)
        print(f"      near-wall u/U at x = 0.3 / 0.6 / 0.9: {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f}   "
              f"max mu_t/mu {np.nanmax(f['mut'])/mu:.1f}", flush=True)
        return settled(hist, 1e-3 * np.maximum(np.abs(vals), 1e-3), window=4, trend_window=8, horizon=10)

    t0 = time.time()
    s.run(a.max_steps, chunk=2000, first_order_steps=500, report=report, steady=True, active_tiles=False)
    wall = time.time() - t0
    f = s.fields()
    x, y = f["x"], f["y"]
    ju = np.flatnonzero(y > 0)[0]
    jl = np.flatnonzero(y < 0)[-1]
    sel = (x > 0.05) & (x < 0.95)
    ut_u = spalding_utau(np.abs(f["u"][sel, ju, 0]), y[ju], nu)
    ut_l = spalding_utau(np.abs(f["u"][sel, jl, 0]), -y[jl], nu)
    Cf = 0.5 * (ut_u ** 2 + ut_l ** 2) / (0.5 * U ** 2)
    Rex = U * x[sel] / nu
    cf_white = 0.455 / np.log(0.06 * Rex) ** 2
    cf_schl = 0.0592 * Rex ** -0.2
    mid = (x[sel] > 0.3) & (x[sel] < 0.9)
    err = np.mean(Cf[mid] / cf_white[mid] - 1)
    # profile at x = 0.7 in wall units
    i = np.argmin(np.abs(x - 0.7))
    yy = y[y > 0]
    uu = f["u"][i, y > 0, 0]
    utau = spalding_utau(abs(uu[0]), yy[0], nu)
    yp, upl = yy * utau / nu, uu / utau
    yplus1 = yp[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax[0].plot(x[sel], cf_white, "k-", label="White 0.455 / ln^2(0.06 Re_x)")
    ax[0].plot(x[sel], cf_schl, "k--", lw=0.8, label="Schlichting 0.0592 Re_x^-0.2")
    ax[0].plot(x[sel], Cf, "o", ms=3, color="C3", label="rocketcfd SST + wall function")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("Cf")
    ax[0].set_ylim(0, 2.0 * cf_white.max())
    ax[0].legend()
    ax[0].set_title(f"Cf: mean deviation from White {100*err:+.1f} % on 0.3 < x < 0.9")
    yl = np.logspace(0, np.log10(max(yp.max(), 20)), 100)
    ax[1].semilogx(yl, np.log(yl) / KAPPA + B, "k-", label="log law")
    ax[1].semilogx(yl[yl < 12], yl[yl < 12], "k:", label="u+ = y+")
    keep = yp < 0.4 * yp.max()
    ax[1].semilogx(yp[keep], upl[keep], "o", ms=4, color="C3", label="rocketcfd, x = 0.7 m")
    ax[1].set_xlabel("y+")
    ax[1].set_ylabel("u+")
    ax[1].legend()
    ax[1].set_title(f"profile in wall units (first cell y+ = {yplus1:.0f})")
    fig.suptitle(f"Turbulent flat plate, k-omega SST + wall function, Mach {a.mach}, Re_L = {a.re:.0e}")
    tag = f"_re{a.re:.0e}".replace("+", "")
    fig.savefig(f"{a.out}/validate_turbulent_plate{tag}.png", dpi=120)
    print(f"Turbulent plate Re {a.re:.0e}: Cf mean deviation from White {100*err:+.2f} % (0.3 < x < 0.9); "
          f"first-cell y+ {yplus1:.0f}; {s.step_count} steps, {wall/60:.1f} min")


if __name__ == "__main__":
    main()
