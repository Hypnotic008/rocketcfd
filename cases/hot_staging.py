"""Hot staging: Starship's engines light while it still sits on the booster's vented interstage;
the plumes hit the booster's forward dome and blow out through the ring's vents. Time-accurate
(explicit, one global time step), with an ignition ramp, snapshots for movies and probes.

Geometry: rocketcfd.vehicles.super_heavy (booster) + starship (ship, ring), approximate public
dimensions. Ascent: the flow comes from the ship's nose (-x). Inviscid by default.

  python cases/hot_staging.py --dry-run                                     grid / memory
  python cases/hot_staging.py --h 0.4 --ring-h 0.1 --time 0.25 --snapshots 150 --snap-volume 1 --out results/hot_staging
  python -m rocketcfd.cinematic results/hot_staging --mode evolve --style plume

Caution: the plume / ambient pressure ratio at staging altitude is ~10^4; the solver's density
and pressure floors keep it running, and the outer plume (near vacuum) is only qualitative.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mach", type=float, default=4.5)
    ap.add_argument("--alt", type=float, default=65000.0, help="altitude [m]")
    ap.add_argument("--aoa", type=float, default=0.0, help="angle of attack [deg]")
    ap.add_argument("--h", type=float, default=0.5, help="cell size near the stack [m]")
    ap.add_argument("--ring-h", type=float, default=0.0, help="refine the interstage region to this cell size [m]")
    ap.add_argument("--growth", type=float, default=1.12)
    ap.add_argument("--booster-engines", default="centre", choices=["none", "centre", "inner"],
                    help="booster engines still firing during staging")
    ap.add_argument("--ship-engines", default="all", choices=["none", "sea-level", "all"])
    ap.add_argument("--ignition", type=float, default=0.02, help="ship engine ramp-up time [s]")
    ap.add_argument("--time", type=float, default=0.15, help="physical time to simulate [s]")
    ap.add_argument("--cfl", type=float, default=0.6)
    ap.add_argument("--snapshots", type=int, default=0)
    ap.add_argument("--snap-volume", type=int, default=0)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/hot_staging")
    a = ap.parse_args(argv)

    import torch
    from rocketcfd import Solver, atmosphere
    from rocketcfd.grid import Zone, stretched_faces, describe
    from rocketcfd.vehicles.super_heavy import super_heavy_sdf, v3_params, engine_positions
    from rocketcfd.vehicles.starship import StarshipParams, starship_sdf, hot_stage_ring_sdf, engines as ship_engines
    from rocketcfd.output import write_vtk

    os.makedirs(a.out, exist_ok=True)
    T, p, rho, c = atmosphere(a.alt)
    V = a.mach * c
    al = np.radians(a.aoa)
    fs = dict(rho=rho, u=-V * np.cos(al), v=0.0, w=V * np.sin(al), p=p)       # ascent: flow toward -x
    q = 0.5 * rho * V * V
    print(f"alt {a.alt/1e3:.1f} km  p {p:.2f} Pa  rho {rho:.2e}  V {V:.0f} m/s  M {a.mach}  q {q/1e3:.2f} kPa")
    PB = v3_params(fin_model="none")
    S = StarshipParams()
    sdf = lambda X, Y, Z: np.minimum(np.minimum(super_heavy_sdf(X, Y, Z, PB), hot_stage_ring_sdf(X, Y, Z, S)),
                                     starship_sdf(X, Y, Z, S))
    # engines: (center, direction, radius, mach, p, T)
    jets = []
    if a.booster_engines != "none":
        n = {"centre": 3, "inner": 13}[a.booster_engines]
        for y, z, xe in engine_positions(PB)[:n]:
            jets.append(dict(center=(float(xe), float(y), float(z)), direction=(-1.0, 0.0, 0.0),
                             radius=0.95 * PB.bell_exit_radius, mach=3.5, p=80e3, T=2200.0))
    n_ship = 0
    if a.ship_engines != "none":
        for kind, y, z, xe, re_, bl in ship_engines(S):
            if a.ship_engines == "sea-level" and kind != "sea level":
                continue
            # sea-level Raptor: Mach 3.5, 80 kPa; vacuum Raptor: Mach 5, 10 kPa (both ~3300 m/s as "air")
            st = dict(mach=3.5, p=80e3, T=2200.0) if kind == "sea level" else dict(mach=5.0, p=10e3, T=1100.0)
            jets.append(dict(center=(float(xe), float(y), float(z)), direction=(-1.0, 0.0, 0.0), radius=0.95 * re_, **st))
            n_ship += 1
    print(f"{len(jets)} jets ({n_ship} ship engines, ramp {a.ignition} s)")

    lo = (-12.0, -26.0, -26.0)
    hi = (S.x_nose + 25.0, 26.0, 26.0)
    zones = [Zone((-1.5, -6.5, -6.5), (S.x_nose + 1.0, 6.5, 6.5), a.h, "stack")]
    faces = stretched_faces(lo, hi, 4 * a.h, zones, growth=a.growth)
    print(describe(faces))
    patches = []
    if a.ring_h:
        from rocketcfd.patches import wall_patches
        reg = ((S.ring_x0 - 3.0, -8.0, -8.0), (S.x_aft + S.skirt_height + 1.0, 8.0, 8.0))
        patches = wall_patches(faces, a.ring_h, 0.0, sdf=lambda X, Y, Z: np.zeros_like(X) - 1e-9, region=reg,
                               box=(64, 64, 64), inner=1e9)
    if a.dry_run:
        n = np.prod([len(f) - 1 for f in faces]) + sum(int(np.prod([len(f) - 1 for f in pf])) for pf in patches)
        print(f"{n/1e6:.2f} M cells ({len(patches)} patches), ~{n * 115 / 2**30:.1f} GB GPU memory")
        return None
    if patches:
        from rocketcfd.patches import PatchGrid
        runner = PatchGrid(faces, patches, fs, sdf=sdf, cfl=a.cfl, jets=jets)
        s = runner.base
    else:
        s = runner = Solver(None, None, None, fs, faces=faces, sdf=sdf, cfl=a.cfl, jets=jets)
    grids = runner.grids if patches else [runner]
    ship_jet = np.zeros(len(jets), bool)
    ship_jet[len(jets) - n_ship:] = True

    # time step and run length
    dt_est = min(float(g_.cfl / (((V + c) + 2 * 3300.0) / g_.h.min())) for g_ in grids)
    nsteps = int(a.time / dt_est) + 1
    chunk = max(10, nsteps // 200)
    print(f"~{nsteps} steps of ~{dt_est:.2e} s")

    hooks = []
    if a.snapshots:
        from rocketcfd.snapshots import SnapshotRecorder
        rec = SnapshotRecorder(s, a.out, q, p, ((-1.5, -8.0, -8.0), (S.x_nose + 1.0, 8.0, 8.0)),
                               half=18.0, volume_stride=a.snap_volume)
        rec.record(0, 0.0)
        hooks.append((max(1, nsteps // a.snapshots), rec.record))
    prb = None
    if a.probes:
        from rocketcfd.probes import ProbeRecorder
        pts = {"ring vent, outside": (70.3, S.radius + 1.0, 0.0), "between the stages, axis": (71.8, 0.0, 0.0),
               "ship aft, outside": (S.x_aft + 4.0, S.radius + 2.0, 0.0), "booster dome": (71.4, 2.0, 0.0),
               "behind the ship's flap": (S.x_aft + 13.0, S.radius + 2.0, 0.0)}
        prb = ProbeRecorder(runner, list(pts.values()), a.out, names=list(pts))
        hooks.append((1, prb.record))
    snap = None
    if hooks:
        import math
        per = 0
        for e_, _ in hooks:
            per = math.gcd(per, int(e_))

        def _hook(step, t_):
            for every, fn in hooks:
                if step % every == 0:
                    fn(step, t_)
        snap = (max(per, 1), _hook)

    # ship engines ramp up with time (set per chunk); booster engines run throughout
    def set_throttle(t_):
        thr = min(1.0, t_ / max(a.ignition, 1e-9))
        vals = np.where(ship_jet, thr, 1.0).astype(np.float32)
        for g_ in grids:
            g_.jet_throttle = vals
        return thr

    hist = []
    t0 = time.time()
    parts = {"ship": (0, S.x_aft - 0.05, 1e9), "booster": (0, -10.0, S.x_aft - 0.05)}
    REF = dict(S_ref=63.6, L_ref=72.3, x_ref=(0.4 * 72.3, 0.0, 0.0))
    done = 0
    while done < nsteps:
        k = min(chunk, nsteps - done)
        thr = set_throttle(s.t)
        if patches:
            runner.run_explicit(k, chunk=k, first_order_steps=50, time_accurate=True, snapshot=snap)
        else:
            s.run(k, chunk=k, first_order_steps=50, steady=False, snapshot=snap)
        done += k
        ld = runner.loads(**REF, parts=parts)
        Fs, Fb = ld["parts"]["ship"]["F"], ld["parts"]["booster"]["F"]
        hist.append((s.step_count, s.t, thr, float(Fs[0]), float(Fb[0])))
        print(f"step {s.step_count:6d}  t = {s.t*1e3:7.2f} ms  ignition {thr:4.2f}  axial force: ship {Fs[0]/1e3:+9.1f} kN, "
              f"booster {Fb[0]/1e3:+9.1f} kN   ({time.time()-t0:.0f} s)", flush=True)
    if prb is not None:
        prb.save()
    f = runner.fields(0) if patches else s.fields()
    write_vtk(f, f"{a.out}/flow.vtk")
    if patches:
        runner.write_patches_vtm(f"{a.out}/flow_patches.vtm")
    from rocketcfd import restart
    restart.save(runner, f"{a.out}/restart.npz")
    from rocketcfd.visual3d import surface_pressure, write_vtk_surface
    xyz, tris, cp_s = surface_pressure(s, q, ((-1.5, -9.0, -9.0), (S.x_nose + 0.5, 9.0, 9.0)),
                                       grids=runner.grids if patches else None)
    write_vtk_surface(f"{a.out}/surface_cp.vtk", xyz, tris, cp_s)
    res = dict(condition=dict(alt=a.alt, mach=a.mach, aoa=a.aoa, V=V, q=q, p=p, rho=rho, T=T),
               grid=dict(h=a.h, ring_h=a.ring_h, cells=int(sum(np.prod(g_.n) for g_ in grids)), steps=s.step_count,
                         wall_s=time.time() - t0, solver="explicit time-accurate"),
               jets=dict(n=len(jets), ship=n_ship, ignition=a.ignition),
               history=hist)
    with open(f"{a.out}/loads.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {a.out}: flow.vtk, surface_cp.vtk, loads.json" + (", snapshots/" if a.snapshots else ""))
    return res


if __name__ == "__main__":
    main()
