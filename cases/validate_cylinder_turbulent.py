"""Validation 6: circular cylinder in crossflow at high Reynolds number (turbulent separation).

Time-accurate k-omega SST with the wall function on an immersed-boundary cylinder, D = 1 m,
Mach 0.3, Re = 1e7 (transcritical). The drag depends on where the boundary layer separates,
the physics behind the sheltered side of the booster and the chines. Experiments at
Re ~ 4e6 - 1e7 (Roshko 1961; Achenbach 1968; Jones et al. 1969): mean drag coefficient about
0.5 - 0.75, Strouhal number about 0.25 - 0.30. RANS typically lands near the low end of Cd.

  py cases\\validate_cylinder_turbulent.py

Options for resolving the boundary layer affordably:
  --patch-h H   refine a shell of --shell metres around the cylinder to cells of size H
                (local refinement patches; the base grid stays at --h)
  --dual        dual time stepping (implicit inner iterations, --spp physical steps per
                shedding period) instead of explicit time steps limited by the smallest cell
  --laminar     no turbulence model (with --re 100: laminar shedding, St 0.164, Cd 1.33)

  py cases\\validate_cylinder_turbulent.py --dual --patch-h 0.001 --shell 0.06
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rocketcfd import Solver
from rocketcfd.grid import axis_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=1e7)
    ap.add_argument("--mach", type=float, default=0.3)
    ap.add_argument("--h", type=float, default=0.008, help="cell size near the cylinder [m] (D = 1 m)")
    ap.add_argument("--periods", type=float, default=25.0, help="run length in shedding periods (St ~ 0.27)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--laminar", action="store_true", help="no turbulence model (e.g. --re 100)")
    ap.add_argument("--dual", action="store_true", help="dual time stepping (implicit inner iterations)")
    ap.add_argument("--spp", type=int, default=80, help="dual: physical steps per shedding period")
    ap.add_argument("--inner", type=int, default=20, help="dual: max inner iterations per step")
    ap.add_argument("--cfl", type=float, default=20.0, help="dual: pseudo-time CFL")
    ap.add_argument("--patch-h", type=float, default=0.0, help="refine near the wall to this cell size [m]")
    ap.add_argument("--shell", type=float, default=0.06, help="thickness of the refined shell [m]")
    a = ap.parse_args()
    St_ref, cd_band = (0.165, (1.30, 1.36)) if a.laminar else (0.27, (0.5, 0.75))
    os.makedirs(a.out, exist_ok=True)
    T, p, R, g = 288.15, 101325.0, 287.05, 1.4
    rho = p / (R * T)
    U = a.mach * np.sqrt(g * R * T)
    D = 1.0
    mu = rho * U * D / a.re
    fs = dict(rho=rho, u=U, v=0.0, w=0.0, p=p)
    h = a.h
    faces = [axis_faces(-15.0, 30.0, 40 * h, [(-0.7, 2.5, h)], 1.06),
             axis_faces(-15.0, 15.0, 40 * h, [(-0.8, 0.8, h)], 1.06),
             np.array([0.0, h])]
    sdf = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5 * D
    kw = dict(sdf=sdf, cfl=0.7, bcs={(2, 0): "periodic", (2, 1): "periodic"},
              viscous=dict(model="constant", mu=mu, R=R, Pr=0.72),
              turbulence=None if a.laminar else dict(model="sst"))
    if a.patch_h > 0:
        from rocketcfd.patches import PatchGrid, wall_patches
        pf = wall_patches(faces, (a.patch_h, a.patch_h, h), a.shell, sdf=sdf, box=(96, 96, 1))
        s = PatchGrid(faces, pf, fs, **kw)
    else:
        s = Solver(None, None, None, fs, faces=faces, **kw)
    # a small transverse kick in the near wake breaks the top/bottom symmetry, so shedding
    # starts within a few periods instead of waiting for round-off to grow
    for g in (s.grids if a.patch_h > 0 else [s]):
        W = g.primitive()
        X, Y = np.meshgrid(g.axes[0], g.axes[1], indexing="ij")
        W[2] += (0.1 * U * np.exp(-((X - 1.0) ** 2 + Y ** 2) / 0.2))[:, :, None]
        g.set_primitive(W)
    q = 0.5 * rho * U ** 2
    span = h
    t_end = a.periods * D / (St_ref * U)
    hist = []

    def report(sv):
        ld = sv.loads(S_ref=D * span, L_ref=D, x_ref=(0.0, 0.0, 0.0))
        hist.append((sv.t, ld["F"][0] / (q * D * span), ld["F"][1] / (q * D * span)))
        return sv.t >= t_end

    t0 = time.time()
    if a.dual or a.patch_h > 0:
        dt = D / (St_ref * U) / a.spp
        nsteps = int(np.ceil(t_end / dt))
        print(f"dual time stepping: dt = {dt:.3e} s ({a.spp} steps per period), {nsteps} steps")
        s.run_unsteady(nsteps, dt, inner=a.inner, cfl=a.cfl, report=report, every=1)
    else:
        s.run(10 ** 7, chunk=200, first_order_steps=500, report=report, steady=False)
    wall = time.time() - t0
    hs = np.array(hist)
    tt, cd, cl = hs[:, 0], hs[:, 1], hs[:, 2]
    keep = tt > 0.4 * tt[-1]                     # discard the start-up transient
    cd_mean = cd[keep].mean()
    # Strouhal number from the lift oscillation
    tk, clk = tt[keep], cl[keep] - cl[keep].mean()
    ti = np.linspace(tk[0], tk[-1], 4096)
    ci = np.interp(ti, tk, clk)
    spec = np.abs(np.fft.rfft(ci * np.hanning(len(ci))))
    freq = np.fft.rfftfreq(len(ci), ti[1] - ti[0])
    fpk = freq[1:][np.argmax(spec[1:])]
    St = fpk * D / U
    f = s.fields() if a.patch_h <= 0 else s.base.fields()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    ax[0].plot(tt * U / D, cd, label=f"Cd (mean {cd_mean:.3f})")
    ax[0].plot(tt * U / D, cl, label=f"Cl (St = {St:.3f})")
    ax[0].axhspan(*cd_band, color="k", alpha=0.08, label=f"reference Cd {cd_band[0]} - {cd_band[1]}")
    ax[0].set_xlabel("convective time t U / D")
    ax[0].legend()
    x, y = f["x"], f["y"]
    w = (x > -1.5) & (x < 6)
    v = (y > -2.5) & (y < 2.5)
    vort = np.gradient(f["v"][:, :, 0], x, axis=0) - np.gradient(f["u"][:, :, 0], y, axis=1)
    vm = np.ma.masked_where(f["solid"][:, :, 0], vort)[np.ix_(w, v)]
    lim = np.percentile(np.abs(vm.compressed()), 98)
    im = ax[1].pcolormesh(x[w], y[v], vm.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax[1].set_aspect("equal")
    ax[1].set_title("vorticity at the end of the run (shedding wake)")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    model = "laminar" if a.laminar else "k-omega SST + wall function"
    near = f"h {h} m" + (f", patches {a.patch_h} m within {a.shell} m" if a.patch_h > 0 else "")
    fig.suptitle(f"Cylinder in crossflow, {model}, Re = {a.re:.0e}, Mach {a.mach}, {near}")
    tag = ("_laminar" if a.laminar else "") + ("_patch" if a.patch_h > 0 else "") + ("_dual" if a.dual else "")
    fig.savefig(f"{a.out}/validate_cylinder_turbulent{tag}.png", dpi=120)
    cells = sum(int(np.prod(g.n)) for g in s.grids) if a.patch_h > 0 else int(np.prod(s.n))
    ref = "reference St 0.164, Cd 1.33" if a.laminar else "experiments Cd ~0.5-0.75, St ~0.25-0.30"
    print(f"Cylinder Re {a.re:.0e} ({model}): mean Cd {cd_mean:.3f}, St {St:.3f} ({ref}); "
          f"{s.step_count} steps, {wall/60:.1f} min, {cells} cells")


if __name__ == "__main__":
    main()
