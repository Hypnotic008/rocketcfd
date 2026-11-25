"""Validation 4: laminar flat plate (Blasius) -- tests the viscous terms and no-slip walls.

A thin plate (both sides wetted) in Mach 0.4 air, Re_L = 1e5 with constant viscosity.
References: Blasius skin friction Cf = 0.664 / sqrt(Re_x) and the Blasius velocity profile
u/U = f'(eta), eta = y sqrt(U / (nu x)). At Mach 0.4 compressibility lowers Cf by ~1 %.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def blasius_profile(eta_max=8.0):
    f = lambda e, y: [y[1], y[2], -0.5 * y[0] * y[2]]
    sol = solve_ivp(f, [0, eta_max], [0.0, 0.0, 0.332057], dense_output=True, rtol=1e-9, atol=1e-12)
    return sol


def main(out="results", Re=1e5, M=0.4, hy=0.0008, hx=0.01):
    os.makedirs(out, exist_ok=True)
    T, p = 288.15, 101325.0
    R, g = 287.05, 1.4
    rho = p / (R * T)
    U = M * np.sqrt(g * R * T)
    L = 1.0
    mu = rho * U * L / Re
    fs = dict(rho=rho, u=U, v=0.0, w=0.0, p=p)
    faces = [axis_faces(-0.3, 1.2, hx, []),
             axis_faces(-0.12, 0.12, 12 * hy, [(-0.02, 0.02, hy)], 1.08),
             np.array([0.0, hx])]
    t = 1e-4
    plate = lambda X, Y, Z: np.sqrt(np.maximum(np.abs(X - 0.5) - 0.5, 0) ** 2 + np.maximum(np.abs(Y) - t / 2, 0) ** 2) \
        + np.minimum(np.maximum(np.abs(X - 0.5) - 0.5, np.abs(Y) - t / 2), 0)
    s = Solver(None, None, None, fs, faces=faces, thin_sdf=plate, cfl=0.7,
               bcs={(2, 0): "periodic", (2, 1): "periodic"},
               viscous=dict(model="constant", mu=mu, R=R, Pr=0.72))
    hist = []

    def report(sv):
        f = sv.fields()
        j = np.flatnonzero(f["y"] > 0)[0]
        i = np.argmin(np.abs(f["x"] - 0.5))
        hist.append(f["u"][i, j, 0])
        return len(hist) > 6 and abs(hist[-1] - hist[-4]) < 1e-4 * U

    s.run(60000, chunk=2000, first_order_steps=500, report=report, steady=True)
    f = s.fields()
    x, y = f["x"], f["y"]
    ju = np.flatnonzero(y > 0)[0]
    jl = np.flatnonzero(y < 0)[-1]
    sel = (x > 0.05) & (x < 0.95)
    tau = mu * 0.5 * (f["u"][sel, ju, 0] / y[ju] + f["u"][sel, jl, 0] / (-y[jl]))
    Cf = tau / (0.5 * rho * U ** 2)
    Rex = rho * U * x[sel] / mu
    Cf_b = 0.664 / np.sqrt(Rex)
    mid = (x[sel] > 0.2) & (x[sel] < 0.8)
    err = np.mean(Cf[mid] / Cf_b[mid] - 1)
    # velocity profile at x = 0.5
    i = np.argmin(np.abs(x - 0.5))
    yy = y[y > 0]
    uu = f["u"][i, y > 0, 0] / U
    eta = yy * np.sqrt(rho * U / (mu * x[i]))
    bl = blasius_profile()
    fp = bl.sol(np.clip(eta, 0, 8.0))[1]
    keep = eta < 7
    prof_err = np.max(np.abs(uu[keep] - fp[keep]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ax[0].plot(x[sel], Cf_b, "k-", label="Blasius 0.664/sqrt(Re_x)")
    ax[0].plot(x[sel], Cf, "o", ms=3, color="C3", label="rocketcfd")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("skin friction Cf")
    ax[0].set_ylim(0, 0.01)
    ax[0].legend()
    ax[0].set_title(f"Cf: mean deviation {100*err:+.1f} % on 0.2 < x < 0.8")
    e = np.linspace(0, 8, 200)
    ax[1].plot(bl.sol(e)[1], e, "k-", label="Blasius f'(eta)")
    ax[1].plot(uu[keep], eta[keep], "o", ms=4, color="C3", label="rocketcfd, x = 0.5 m")
    ax[1].set_xlabel("u / U")
    ax[1].set_ylabel("eta = y sqrt(U / nu x)")
    ax[1].legend()
    ax[1].set_title(f"profile: max |error| {prof_err:.3f}")
    fig.suptitle(f"Laminar flat plate, Mach {M}, Re_L = {Re:.0e}, {int(0.04/hy)} cells across +/-2 cm")
    fig.savefig(f"{out}/validate_blasius.png", dpi=120)
    print(f"Blasius: Cf mean deviation {100*err:+.2f} % (0.2 < x < 0.8); profile max |u/U error| {prof_err:.3f}; "
          f"{s.step_count} steps")
    return err, prof_err


if __name__ == "__main__":
    main()
