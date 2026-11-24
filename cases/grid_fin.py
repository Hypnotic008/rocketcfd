"""Isolated V3 grid fin (CatchSim CAD, scaled to 3.5 x 4.2 x 0.5 m) over a Mach x incidence sweep.

The lattice plates (~4.5 cm) are represented as thin walls (blocked faces), so they stay
solid at cell sizes of 2-3 cm. The grid and wall data are built once and reused for
every flight condition.

Fin frame (the CAD's own axes): x along the booster axis (flow passes through the
lattice along +x), y around the hull, z radial (root at z = 0, tip at 4.2 m).
Incidence alpha tilts the flow in the x-y plane, which for an isolated fin is the same
as a hinge deflection. The hinge axis is the z axis through the root centre.

Outputs per case (fin reference area S = outline planform area, c = 3.5 m width, b = 4.2 m span):
  CN = F_y / (q S)       normal force (what steers the booster)
  CA = F_x / (q S)       axial force (fin drag)
  Ch = M_z / (q S c)     hinge moment (sizes the actuator)
  Cb = M_x / (q S b)     root bending moment

  py cases\\grid_fin.py --h 0.025                       (full sweep on the GPU)
  py cases\\grid_fin.py --h 0.025 --mach 2.82 --alpha 5  (one case)
"""
import sys, os, json, argparse, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mach", type=float, nargs="+", default=[0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.8, 4.0])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--alt", type=float, default=10745.0)
    ap.add_argument("--h", type=float, default=0.025, help="cell size around the fin [m]")
    ap.add_argument("--h-far", type=float, default=0.0, help="farfield cell size (default 24 x h)")
    ap.add_argument("--fin", default="parametric", choices=["parametric", "cad"],
                    help="parametric: V3 model with teeth (vehicles/grid_fin.py); cad: CatchSim's CAD scaled")
    ap.add_argument("--plate", type=float, default=0.0, help="parametric fin: lattice plate thickness [m]")
    ap.add_argument("--no-teeth", action="store_true", help="parametric fin without the toothed face")
    ap.add_argument("--plate-offset", type=float, default=0.0,
                    help="thin every plate by this much on each side [m] (e.g. 0.01 turns 4.5 cm into 2.5 cm)")
    ap.add_argument("--local-dt", action="store_true",
                    help="per-cell time steps (steady): speeds up the subsonic cases, whose large farfield "
                         "cells otherwise wait on the fin's small ones")
    ap.add_argument("--ascending", action="store_true",
                    help="sweep Mach upward, each Mach started from rest (default: downward with warm starts, "
                         "like the real descent, which keeps a started lattice started)")
    ap.add_argument("--growth", type=float, default=1.12)
    ap.add_argument("--flowthroughs", type=float, default=6.0, help="maximum per case")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="results/grid_fin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--time-accurate", action="store_true",
                    help="global time step, all cells every step (off: local time steps + active tiles)")
    a = ap.parse_args()

    from rocketcfd import Solver, atmosphere
    from rocketcfd.grid import Zone, stretched_faces, describe
    from rocketcfd.vehicles.super_heavy import load_fin_cad, SuperHeavyParams
    from rocketcfd.output import plot_slices

    off = a.plate_offset
    if a.fin == "parametric":
        from rocketcfd.vehicles.grid_fin import GridFinParams, grid_fin_sdf, grid_fin_box, planform_area
        GP = GridFinParams(teeth=not a.no_teeth, **({"plate": a.plate} if a.plate else {}))
        fin_sdf = lambda X, Y, Z: grid_fin_sdf(X, Y, Z, GP) + off
        S_ref, c_ref, b_ref = planform_area(GP), GP.width, GP.z_tip
        box = grid_fin_box(GP, pad=0.05)
        print(GP.summary(), f"| outline area S = {S_ref:.2f} m^2")
    else:
        P = SuperHeavyParams()
        sx, sy, sz = P.fin_scale
        cad = load_fin_cad(P.fin_cad_path, P.fin_cad_res, False)
        smin = min(sx, sy, sz)
        fin_sdf = lambda X, Y, Z: smin * cad(X / sx, Y / sy, Z / sz) + off
        outline = load_fin_cad(P.fin_cad_path, P.fin_cad_res, True)
        S_ref = outline.solid.any(axis=0).sum() * outline.res ** 2 * sy * sz
        c_ref, b_ref = 6.0 * sy, 7.0 * sz
        box = ((-0.75 * sx / 2 - 0.05, -3.0 * sy - 0.05, -0.05), (0.75 * sx / 2 + 0.05, 3.0 * sy + 0.05, 7.0 * sz + 0.05))
        print(f"fin (CatchSim CAD): {0.75*sx:.2f} m deep x {6*sy:.2f} m wide x {7*sz:.2f} m span, "
              f"outline area S = {S_ref:.2f} m^2")

    # generous farfield: choked / subsonic cases spill flow around the fin and need room
    lo, hi = (-9.0, -9.0, -6.0), (12.0, 9.0, 11.0)
    m = 0.15
    zone = Zone((box[0][0] - m, box[0][1] - m, box[0][2] - m), (box[1][0] + m, box[1][1] + m, box[1][2] + m), a.h, "fin")
    faces = stretched_faces(lo, hi, a.h_far or 24 * a.h, [zone], growth=a.growth)
    print(describe(faces))
    if a.dry_run:
        n = np.prod([len(f) - 1 for f in faces])
        print(f"estimated GPU memory ~{n * 120 / 2**30:.1f} GB")
        return

    os.makedirs(a.out, exist_ok=True)
    T, p, rho, c = atmosphere(a.alt)

    def freestream(M, al_deg):
        V = M * c
        al = np.radians(al_deg)
        return dict(rho=rho, u=V * np.cos(al), v=V * np.sin(al), w=0.0, p=p), 0.5 * rho * V * V, V

    machs = sorted(a.mach) if a.ascending else sorted(a.mach, reverse=True)
    alphas = sorted(a.alpha)
    fs0, _, _ = freestream(machs[0], alphas[0])
    t0 = time.time()
    s = Solver(None, None, None, fs0, faces=faces, thin_sdf=fin_sdf, thin_box=box, cfl=0.7)
    print(f"setup {time.time()-t0:.1f} s")
    rows = []
    zmid = 0.5 * (box[0][2] + box[1][2]) + 0.11      # a mid-span plane, off the lattice nodes
    # Flight-like path: Mach descends along the alpha = 0 (symmetric) states; each incidence is
    # an excursion from that Mach's alpha-0 state, swept upward. (Sweeping incidence back down
    # first left asymmetric, partly choked lattices behind: real hysteresis, but not this path.)
    first = True
    base = None                                   # alpha-0 state of the previous Mach
    for mi, M in enumerate(machs):
        for ai, al in enumerate(alphas):
            fs, q, V = freestream(M, al)
            if ai == 0 and base is not None and not a.ascending:
                s.U.copy_(base)
            warm = (not first) and (ai > 0 or not a.ascending)
            s.set_freestream(fs, keep_state=warm)
            first = False
            hmin = s.h.min()
            dt_est = s.cfl / ((V + c) / hmin + 2 * (c + V * np.sin(np.radians(al))) / hmin)
            nmax = a.max_steps or int(a.flowthroughs * (hi[0] - lo[0]) / V / dt_est)
            chunk = max(100, int(nmax / 40 / 50) * 50)
            hist = []

            def report(sv):
                ld = sv.loads(S_ref=S_ref, L_ref=c_ref, x_ref=(0.0, 0.0, 0.0))
                Ft, Mt = ld["F_thin"], ld["M_thin"]
                CN, CA, Ch, Cb = Ft[1] / (q * S_ref), Ft[0] / (q * S_ref), Mt[2] / (q * S_ref * c_ref), Mt[0] / (q * S_ref * b_ref)
                hist.append((CN, CA, Ch))
                print(f"   step {sv.step_count:6d}  CN {CN:+.4f}  CA {CA:+.4f}  Ch {Ch:+.5f}  Cb {Cb:+.5f}", flush=True)
                if len(hist) >= 4 and sv.step_count > nmax / a.flowthroughs:
                    r = np.array(hist[-4:])
                    if np.all(np.abs(r - r[-1]).max(0) < a.tol):
                        return True
                return False

            print(f"=== M {M}  alpha {al} deg  (up to {nmax} steps)", flush=True)
            t1 = time.time()
            s.run(nmax, chunk=chunk, first_order_steps=(0 if warm else 300), report=report,
                  steady=not a.time_accurate, local_dt=a.local_dt)
            if ai == 0:
                base = s.U.clone()
            ld = s.loads(S_ref=S_ref, L_ref=c_ref, x_ref=(0.0, 0.0, 0.0))
            Ft, Mt = ld["F_thin"], ld["M_thin"]
            row = dict(mach=M, alpha=al, CN=Ft[1] / (q * S_ref), CA=Ft[0] / (q * S_ref),
                       Ch=Mt[2] / (q * S_ref * c_ref), Cb=Mt[0] / (q * S_ref * b_ref),
                       steps=s.step_count, wall_s=time.time() - t1)
            rows.append(row)
            print(f"    -> CN {row['CN']:+.4f}  CA {row['CA']:+.4f}  Ch {row['Ch']:+.5f}  ({row['steps']} steps, "
                  f"{row['wall_s']:.0f} s)", flush=True)
            if al == max(alphas) or (al == 0 and M in (machs[0], machs[-1])):
                f = s.fields()
                plot_slices(f, f"{a.out}/fin_M{M:g}_a{al:g}_midspan.png",
                            title=f"V3 grid fin ({a.fin}), M {M}, alpha {al} deg: mid-span plane (z = {zmid:.1f} m)"
                                  f"{', plates thinned by %g m/side' % off if off else ''}",
                            p_inf=p, normal_axis=2, at=zmid, window=((-1.5, 2.5), (-2.6, 2.6)))
            with open(f"{a.out}/grid_fin_sweep.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    rows.sort(key=lambda r: (r["mach"], r["alpha"]))
    json.dump(dict(S_ref=S_ref, c_ref=c_ref, b_ref=b_ref, h=a.h, plate_offset=off,
                   order="ascending" if a.ascending else "descending", cells=int(np.prod(s.n)), rows=rows),
              open(f"{a.out}/grid_fin_sweep.json", "w"), indent=2, default=float)
    # slopes and plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    machs = sorted(set(r["mach"] for r in rows))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for al in sorted(set(r["alpha"] for r in rows)):
        sel = sorted([r for r in rows if r["alpha"] == al], key=lambda r: r["mach"])
        for ax, key in zip(axs, ("CN", "CA", "Ch")):
            ax.plot([r["mach"] for r in sel], [r[key] for r in sel], "o-", label=f"alpha {al:g} deg")
    slope = []
    for M in machs:
        pts = sorted([(r["alpha"], r["CN"]) for r in rows if r["mach"] == M])
        if len(pts) >= 2:
            x_, y_ = np.radians([p_[0] for p_ in pts]), [p_[1] for p_ in pts]
            slope.append((M, np.polyfit(x_, y_, 1)[0]))
    for ax, t_ in zip(axs, ("normal force CN", "axial force CA", "hinge moment Ch")):
        ax.set_xlabel("Mach")
        ax.set_title(t_)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"V3 grid fin ({'parametric' + (', no teeth' if a.no_teeth else ', teeth') if a.fin == 'parametric' else 'CatchSim CAD'}"
                 f", isolated, inviscid), h = {a.h} m, S = {S_ref:.2f} m^2, "
                 f"Mach {'ascending from rest' if a.ascending else 'descending, warm-started'}"
                 f"{', plates thinned %g m/side' % off if off else ''}")
    fig.savefig(f"{a.out}/grid_fin_sweep.png", dpi=130)
    if slope:
        print("\nnormal-force slope dCN/dalpha per rad (fin area):")
        for M, sl in slope:
            print(f"  M {M:4.2f}: {sl:6.3f}")
        json.dump(dict(slope=slope), open(f"{a.out}/grid_fin_slope.json", "w"), indent=2)


if __name__ == "__main__":
    main()
