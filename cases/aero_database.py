"""Super Heavy V3 aero database for CatchSim: Mach x angle of attack x roll, in one process.

The grid and geometry are built once (roll is set by turning the crossflow, not the
vehicle), and every case warm-starts from a converged neighbour: Mach runs from high to
low, angle of attack from 0 upward, and each roll angle restarts from that Mach's zero-
incidence solution. The implicit solver is used throughout; grid fins are resolved by
local refinement patches (--fin-h) instead of whole slabs of fine cells.

Per case it records the six body-axis force and moment coefficients, the crossflow-frame
coefficients CatchSim uses (CD, CN, CS, C_restore, C_roll), the breakdown by region, and
each fin's forces and hinge moment. Results are appended to <out>/cases.csv as they come
(the sweep resumes where it stopped) and collected into <out>/aero_db.json, the table
that integrations/catchsim/aero_cfd.py interpolates.

Inviscid coefficients depend on Mach number only, so one altitude serves the whole sweep.
The V3 layout is mirror-symmetric about the rudder-fin / raceway plane, so roll angles
0-180 deg cover the full circle (the export mirrors them).

  python cases/aero_database.py --dry-run                          # grid, patches, memory, case list
  python cases/aero_database.py                                    # default sweep (GPU, hours)
  python cases/aero_database.py --mach 2.8 2.0 --aoa 0 8 16 --roll 0 180 --out results/db_quick
"""
import sys, os, json, csv, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

MACH = [4.0, 3.2, 2.6, 2.0, 1.6, 1.3, 1.1, 0.95, 0.8, 0.6]
AOA = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
ROLL = [0.0, 90.0, 180.0]           # windward side, dimensions.md azimuth: 0 rudder fin, 90 catch fin, 180 raceway
KEYS = ("CD", "CL", "CN", "CS", "C_roll", "C_restore", "CX", "CY", "CZ", "Cl", "Cm", "Cn", "CD_friction")


def build(a):
    import torch
    from rocketcfd import Solver, atmosphere
    from rocketcfd.vehicles.super_heavy import v3_params, super_heavy_sdf, super_heavy_fins_sdf, \
        super_heavy_fins_box
    from rocketcfd.grid import Zone, stretched_faces, describe
    from rocketcfd.patches import PatchGrid, wall_patches
    P = v3_params(fin_model=a.fins, fin_deflection_deg=tuple(a.fin_deflection))
    sdf = lambda X, Y, Z: super_heavy_sdf(X, Y, Z, P)
    thin_sdf = thin_box = None
    if a.fins == "parametric":
        thin_sdf = lambda X, Y, Z: super_heavy_fins_sdf(X, Y, Z, P)
        thin_box = super_heavy_fins_box(P)
    L = P.length
    lo, hi = (-15.5, -22.0, -22.0), (L + 18.0, 22.0, 22.0)
    r_out = P.radius + 1.35 + 0.5
    zones = [Zone((-1.5, -r_out, -r_out), (L + 1.0, r_out, r_out), a.h, "vehicle")]
    faces = stretched_faces(lo, hi, a.h_far or 4 * a.h, zones, growth=a.growth)
    print(describe(faces))
    patches = []
    if a.fin_h and thin_sdf is not None:
        patches += wall_patches(faces, a.fin_h, a.fin_shell, thin_sdf=thin_sdf, thin_box=thin_box, box=(64, 64, 64))
    if a.engine_h:
        R_eng = P.radius + 1.0
        reg = ((-2.0, -R_eng, -R_eng), (P.skirt_bottom_x + 1.0, R_eng, R_eng))
        patches += wall_patches(faces, a.engine_h, a.engine_shell, sdf=sdf, region=reg, box=(64, 64, 64))
    if a.wall_h:
        hw = tuple(float(v) for v in a.wall_h.split(","))
        patches += wall_patches(faces, hw * 3 if len(hw) == 1 else hw, a.wall_shell, sdf=sdf,
                                box=tuple(int(v) for v in a.patch_box.split(",")))
    n = int(np.prod([len(f) - 1 for f in faces])) + sum(int(np.prod([len(f) - 1 for f in pf])) for pf in patches)
    per = (200 if a.turbulent else 140)
    print(f"{len(patches)} patches, {n/1e6:.2f} M cells in all, ~{n*per/2**30:.1f} GB GPU memory")
    if a.dry_run:
        return None, P, None
    T, p, rho, c = atmosphere(a.alt)
    fs0 = make_fs(a, a.mach[0], 0.0, 0.0)
    visc = dict(model="sutherland") if a.turbulent else None
    turb = dict(model="sst") if a.turbulent else None
    kw = dict(sdf=sdf, cfl=0.7, thin_sdf=thin_sdf, thin_box=thin_box, viscous=visc, turbulence=turb)
    runner = PatchGrid(faces, patches, fs0, **kw) if patches else Solver(None, None, None, fs0, faces=faces, **kw)
    return runner, P, (T, p, rho, c)


def make_fs(a, mach, aoa, roll):
    from rocketcfd import atmosphere
    from rocketcfd.vehicles.super_heavy import _az
    T, p, rho, c = atmosphere(a.alt)
    V = mach * c
    al, tw = np.radians(aoa), _az(roll)
    return dict(rho=rho, u=V * np.cos(al), v=-V * np.sin(al) * np.cos(tw), w=-V * np.sin(al) * np.sin(tw), p=p)


def solve_case(runner, a, parts, cold):
    """Implicit iterations until CD, CN, C_restore settle (or max iterations); returns the
    averaged loads over the last checks and convergence info."""
    from rocketcfd.convergence import settled
    from rocketcfd.vehicles.super_heavy import REFERENCE
    hist, full = [], []
    min_it = a.min_iters_cold if cold else a.min_iters

    def report(sv):
        ld = sv.loads(**REFERENCE, parts=parts)
        hist.append((ld["CD"], ld["CN"], ld["C_restore"]))
        full.append(ld)
        if sv.step_count >= min_it:
            tolv = (a.tol * max(abs(hist[-1][0]), 1e-6), a.tol, a.tol)
            return settled(hist, tolv)
        return False

    kw = dict(chunk=a.check, report=report, cfl_max=a.cfl_max)
    if cold:
        kw.update(first_order_steps=150, cfl_start=1.0, cfl_ramp=200)
    else:
        kw.update(first_order_steps=0, cfl_start=min(5.0, a.cfl_max), cfl_ramp=60)
    t0 = time.time()
    if hasattr(runner, "grids"):
        runner.run(a.max_iters, **kw)
    else:
        runner.run(a.max_iters, steady=True, implicit=True, **kw)
    tail = full[-max(3, len(full) // 5):]
    conv = len(hist) >= 4 and settled(hist, (a.tol * max(abs(hist[-1][0]), 1e-6), a.tol, a.tol))
    return tail, dict(iters=runner.step_count, wall_s=time.time() - t0, converged=bool(conv),
                      res=float(runner.history[-1][-1]) if runner.history else float("nan"))


def averaged(tail, P):
    from rocketcfd.vehicles.super_heavy import fin_hinge_moments
    row = {}
    for k in KEYS:
        v = [float(t[k]) for t in tail if k in t]
        if v:
            row[k] = float(np.mean(v))
            row[k + "_std"] = float(np.std(v))
    for name in tail[-1]["parts"]:
        for k in ("CD", "CN", "C_restore", "CX", "CY", "CZ"):
            row[f"{k}[{name}]"] = float(np.mean([t["parts"][name][k] for t in tail]))
    hm = [fin_hinge_moments(t, P) for t in tail]
    for name in hm[-1]:
        row[f"C_hinge[{name}]"] = float(np.mean([h[name]["C_h"] for h in hm]))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mach", type=float, nargs="+", default=MACH)
    ap.add_argument("--aoa", type=float, nargs="+", default=AOA, help="deg away from engines-first")
    ap.add_argument("--roll", type=float, nargs="+", default=ROLL,
                    help="windward side (dimensions.md azimuth), 0-180: the layout is mirror-symmetric")
    ap.add_argument("--alt", type=float, default=10745.0, help="altitude [m]; inviscid results depend on Mach only")
    ap.add_argument("--h", type=float, default=0.3, help="cell size near the vehicle [m]")
    ap.add_argument("--h-far", type=float, default=0.0)
    ap.add_argument("--growth", type=float, default=1.12)
    ap.add_argument("--fins", default="parametric", choices=["none", "parametric"])
    ap.add_argument("--fin-h", type=float, default=0.06, help="fin patch cell size [m] (0 = no patches)")
    ap.add_argument("--fin-shell", type=float, default=0.3)
    ap.add_argument("--engine-h", type=float, default=0.06,
                    help="local refinement of the engine end (bells, thrust plate, skirt) to this cell size [m], "
                         "e.g. 0.06: resolves the ~4 cm gaps between the outer nozzles that make drag grid-sensitive")
    ap.add_argument("--engine-shell", type=float, default=0.5, help="refined distance around the engine end [m]")
    ap.add_argument("--fin-deflection", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--wall-h", default="", help="hull shell patches hx,hy,hz (viscous runs)")
    ap.add_argument("--wall-shell", type=float, default=0.4)
    ap.add_argument("--patch-box", default="32,96,96")
    ap.add_argument("--turbulent", action="store_true")
    ap.add_argument("--cfl-max", type=float, default=20.0)
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--min-iters", type=int, default=200, help="warm-started cases")
    ap.add_argument("--min-iters-cold", type=int, default=500)
    ap.add_argument("--check", type=int, default=50, help="iterations between load checks")
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--out", default="results/aero_db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.mach = sorted(a.mach, reverse=True)
    a.aoa = sorted(a.aoa)
    os.makedirs(a.out, exist_ok=True)

    cases = []
    for m in a.mach:
        for r in a.roll:
            for al in a.aoa:
                if al == 0.0 and r != a.roll[0]:
                    continue                          # zero incidence: roll does not matter
                cases.append((m, al, r))
    print(f"{len(cases)} cases: Mach {a.mach}, aoa {a.aoa}, roll {a.roll}")
    runner, P, atm = build(a)
    if runner is None:
        return
    from rocketcfd.vehicles.super_heavy import fin_selectors, REFERENCE
    xf0 = P.fin_x_center - 0.8
    parts = {"aft end": (0, -5.0, P.skirt_bottom_x + 0.6), "hull": (0, P.skirt_bottom_x + 0.6, xf0),
             "fin band": (0, xf0, P.tank_top_x - 0.05), "top": (0, P.tank_top_x - 0.05, 1e9)}
    if a.fins != "none":
        parts.update(fin_selectors(P))

    csv_path = f"{a.out}/cases.csv"
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                done[(float(row["mach"]), float(row["aoa"]), float(row["roll"]))] = row
    rows = list(done.values())
    zero_state = {}            # Mach -> converged zero-incidence state
    zero_latest = None
    last = None                # case solved just before in this process
    t_all = time.time()
    for (m, al, r) in cases:
        if (m, al, r) in done:
            print(f"M {m} aoa {al} roll {r}: done earlier, skipping")
            last = None
            continue
        fs = make_fs(a, m, al, r)
        if al == 0.0:
            src = zero_latest                              # previous Mach's zero-incidence solution
        elif last is not None and last[0] == m and last[2] == r and last[1] < al:
            src = "keep"                                   # next incidence at the same roll
        else:
            src = zero_state.get(m)                        # new roll: from this Mach's alpha = 0
        cold = src is None
        if cold:
            runner.set_freestream(fs, keep_state=False)
        else:
            if src != "keep":
                runner.set_state(src)
            runner.set_freestream(fs, keep_state=True)
        print(f"\n=== M {m}  aoa {al}  roll {r}  ({'cold start' if cold else 'warm start'}) ===", flush=True)
        try:
            tail, info = solve_case(runner, a, parts, cold)
        except FloatingPointError as e:
            # retry from a uniform start at a lower CFL before giving up on this case
            print(f"    diverged ({e}); retrying cold at CFL {a.cfl_max / 4:g}", flush=True)
            runner.set_freestream(fs, keep_state=False)
            cfl0, a.cfl_max = a.cfl_max, a.cfl_max / 4
            try:
                tail, info = solve_case(runner, a, parts, True)
            except FloatingPointError as e2:
                print(f"    diverged again ({e2}); case skipped", flush=True)
                runner.set_freestream(fs, keep_state=False)
                last = None
                a.cfl_max = cfl0
                continue
            a.cfl_max = cfl0
        row = dict(mach=m, aoa=al, roll=r, alt=a.alt, **info, **averaged(tail, P))
        rows.append(row)
        if al == 0.0:
            zero_state[m] = zero_latest = runner.state()
        last = (m, al, r)
        new = not os.path.exists(csv_path)
        fields = list(row)
        if not new:
            with open(csv_path) as fh:
                fields = next(csv.reader(fh))
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"    CD {row['CD']:.4f}  CN {row['CN']:+.4f}  C_restore {row['C_restore']:+.5f}  "
              f"C_roll {row['C_roll']:+.5f}  {info['iters']} iters, {info['wall_s']/60:.1f} min, "
              f"{'converged' if info['converged'] else 'NOT settled (averaged)'}", flush=True)
    print(f"\nsweep finished in {(time.time()-t_all)/3600:.2f} h")
    from rocketcfd.catchsim_export import build_database
    db = build_database(csv_path, reference=REFERENCE)
    with open(f"{a.out}/aero_db.json", "w") as fh:
        json.dump(db, fh, indent=1)
    print(f"wrote {a.out}/aero_db.json")


if __name__ == "__main__":
    main()
