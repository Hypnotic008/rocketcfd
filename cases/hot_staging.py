"""Hot staging: Starship's engines light while it still sits on the booster's hot stage; the
plumes hit the booster's forward dome and its protection plates and blow out between the truss
tubes. Time-accurate (explicit, one global time step), with an ignition ramp, snapshots for
movies and probes.

Geometry: rocketcfd.vehicles.super_heavy (booster) + starship (ship, hot stage), approximate
public dimensions. --ring v3 (default): the V3 hot stage of the photos, a zig-zag truss of round
tubes between two rings, and the "flower" of protection plates on the forward dome (a centre
plate and a ring of seven). --ring vented: the earlier ring with straight struts. Ascent: the flow
comes from the ship's nose (-x). Inviscid.

Data: gas pressure and temperature next to the dome, each plate, the truss and the ship's aft
end (rocketcfd.hotstage: time histories, peaks, per-vertex values in every snapshot).

  python cases/hot_staging.py --dry-run --ring-h 0.07 --wall-h 0.035        grid / memory
  python cases/hot_staging.py --h 0.4 --ring-h 0.07 --wall-h 0.035 --time 0.08 --snapshots 160 --snap-volume 2 --out results/hot_staging
  python -m rocketcfd.hotstage report results/hot_staging
  python -m rocketcfd.cinematic results/hot_staging --mode hotstage

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
    ap.add_argument("--ring", default="v3", choices=["v3", "vented"], help="hot-stage geometry")
    ap.add_argument("--sl-cant", type=float, default=0.0,
                    help="ship sea-level engines canted outward by this angle [deg] (bell and plume, about the gimbal point)")
    ap.add_argument("--vac-cant", type=float, default=0.0, help="the same for the vacuum engines [deg]")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="separation: the ship this far above the hot stage [m] (a frozen moment of the separation)")
    ap.add_argument("--ring-h", type=float, default=0.0,
                    help="refine a cylinder round the hot stage (r < --ring-r, the ship's engines to below the dome) [m]")
    ap.add_argument("--ring-r", type=float, default=6.0)
    ap.add_argument("--wall-h", type=float, default=0.0,
                    help="and a finer shell on the dome, plates, truss and ship's engines / aft end [m]")
    ap.add_argument("--wall-shell", type=float, default=0.25, help="thickness of that shell [m]")
    ap.add_argument("--snap-patch-cells", type=float, default=6.0, help="cap on the 3-D snapshot patch cells [M]")
    ap.add_argument("--growth", type=float, default=1.12)
    ap.add_argument("--booster-engines", default="centre", choices=["none", "centre", "inner"],
                    help="booster engines still firing during staging")
    ap.add_argument("--booster-throttle", type=float, default=1.0, help="their chamber pressure / rated")
    ap.add_argument("--rho-min", type=float, default=5e-3,
                    help="gas temperature counts only above this density [kg/m^3] (thin gas: no heat)")
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
    from rocketcfd.vehicles.starship import StarshipParams, starship_sdf, interstage_sdf, engine_frames
    from rocketcfd.output import write_vtk

    os.makedirs(a.out, exist_ok=True)
    T, p, rho, c = atmosphere(a.alt)
    V = a.mach * c
    al = np.radians(a.aoa)
    fs = dict(rho=rho, u=-V * np.cos(al), v=0.0, w=V * np.sin(al), p=p)       # ascent: flow toward -x
    q = 0.5 * rho * V * V
    print(f"alt {a.alt/1e3:.1f} km  p {p:.2f} Pa  rho {rho:.2e}  V {V:.0f} m/s  M {a.mach}  q {q/1e3:.2f} kPa")
    PB = v3_params(fin_model="none")
    S = StarshipParams(ring_style="v3_truss" if a.ring == "v3" else "vented", sl_cant=a.sl_cant, vac_cant=a.vac_cant,
                       sep_gap=a.gap)
    ring_sdf = lambda X, Y, Z: interstage_sdf(X, Y, Z, S, PB.tank_top_x, PB.dome_rise)
    sdf = lambda X, Y, Z: np.minimum(np.minimum(super_heavy_sdf(X, Y, Z, PB), ring_sdf(X, Y, Z)),
                                     starship_sdf(X, Y, Z, S))
    # engines: Raptor 3 nozzle-exit states (rocketcfd.engines: CH4/LOX, 350 bar, 1-D nozzle sized
    # to the rated thrust), imposed with the exhaust's own density, velocity and pressure
    import dataclasses
    from rocketcfd import engines as EN
    jets = []
    if a.booster_engines != "none":
        n = {"centre": 3, "inner": 13}[a.booster_engines]
        bspec = dataclasses.replace(EN.RAPTOR3_SL, exit_diameter=2 * PB.bell_exit_radius)
        for y, z, xe in engine_positions(PB)[:n]:
            j = EN.jet(bspec, (xe, y, z), (-1.0, 0.0, 0.0), p_amb=p)
            if a.booster_throttle != 1.0:
                e = EN.exit_state(bspec, p, throttle=a.booster_throttle)
                j.update(p=float(e["p_e"]), engine=e)
            jets.append(j)
    n_ship = 0
    if a.ship_engines != "none":
        for fr in engine_frames(S):
            kind = fr["kind"]
            if a.ship_engines == "sea-level" and kind != "sea level":
                continue
            spec = EN.RAPTOR3_SL if kind == "sea level" else EN.RAPTOR3_VAC
            spec = dataclasses.replace(spec, exit_diameter=2 * fr["exit_r"])
            jets.append(EN.jet(spec, fr["exit"], fr["dir"], p_amb=p))
            n_ship += 1
    print(f"{len(jets)} jets ({n_ship} ship engines, ramp {a.ignition} s):")
    seen = set()
    for j in jets:
        e = j["engine"]
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        print(f"  {e['name']}: {e['propellant']}, p_c {e['p_c']/1e5:.0f} bar, T_c {e['T_c']:.0f} K, "
              f"exit {e['exit_diameter']:.2f} m (area ratio {e['area_ratio']:.0f}), M_e {e['M_e']:.2f}, "
              f"V_e {e['V_e']:.0f} m/s, T_e {e['T_e']:.0f} K, p_e {e['p_e']/1e3:.0f} kPa, {e['mdot']:.0f} kg/s, "
              f"thrust here {e['thrust']/1e3:.0f} kN, exhaust {e['molar_mass']:.1f} g/mol")
    engine_info = [{k: (v if isinstance(v, str) else float(v)) for k, v in j["engine"].items()} for j in jets]
    for j in jets:
        j.pop("engine")

    lo = (-12.0, -26.0, -26.0)
    hi = (S.x_nose + 25.0, 26.0, 26.0)
    zones = [Zone((-1.5, -6.5, -6.5), (S.x_nose + 1.0, 6.5, 6.5), a.h, "stack")]
    faces = stretched_faces(lo, hi, 4 * a.h, zones, growth=a.growth)
    print(describe(faces))
    patches, ring_patches = [], []
    x_lo, x_hi = PB.tank_top_x - 2.5, S.x_ship_aft + S.skirt_height + 1.5      # below the dome .. above the bells
    if a.ring_h:
        from rocketcfd.patches import wall_patches
        Rr = a.ring_r
        reg = ((x_lo, -Rr - 1, -Rr - 1), (x_hi, Rr + 1, Rr + 1))
        # a cylinder: the gap between the stages and the plume leaving between the truss tubes
        cyl = lambda X, Y, Z: np.maximum(np.sqrt(Y * Y + Z * Z) - Rr, np.maximum(x_lo - X, X - x_hi))
        ring_patches = wall_patches(faces, a.ring_h, 0.0, sdf=cyl, region=reg, box=(64, 64, 64), inner=1e9)
        patches += ring_patches
    if a.wall_h:
        from rocketcfd.patches import wall_patches
        ship_aft = lambda X, Y, Z: np.maximum(starship_sdf(X, Y, Z, S), X - (S.x_ship_aft + S.skirt_height + 0.3))
        dome = lambda X, Y, Z: np.maximum(super_heavy_sdf(X, Y, Z, PB), PB.tank_top_x - 0.3 - X)
        walls = lambda X, Y, Z: np.minimum(np.minimum(ring_sdf(X, Y, Z), ship_aft(X, Y, Z)), dome(X, Y, Z))
        reg = ((PB.tank_top_x - 0.3, -S.radius - 0.6, -S.radius - 0.6),
               (S.x_ship_aft + S.skirt_height + 0.3, S.radius + 0.6, S.radius + 0.6))
        patches += wall_patches(faces, a.wall_h, a.wall_shell, sdf=walls, region=reg, box=(48, 48, 48))
    if a.dry_run:
        n = np.prod([len(f) - 1 for f in faces]) + sum(int(np.prod([len(f) - 1 for f in pf])) for pf in patches)
        npt = lambda L: sum(int(np.prod([len(f) - 1 for f in pf])) for pf in L)
        print(f"{n/1e6:.2f} M cells ({len(patches)} patches: hot-stage cylinder {npt(ring_patches)/1e6:.2f} M, "
              f"wall shell {npt(patches[len(ring_patches):])/1e6:.2f} M), ~{n * 115 / 2**30:.1f} GB GPU memory")
        return None
    if patches:
        from rocketcfd.patches import PatchGrid
        runner = PatchGrid(faces, patches, fs, sdf=sdf, cfl=a.cfl, jets=jets)
        s = runner.base
    else:
        s = runner = Solver(None, None, None, fs, faces=faces, sdf=sdf, cfl=a.cfl, jets=jets)
    grids = runner.grids if patches else [runner]
    # each engine on the finest grid holding its exit: cells and the density factor that keeps
    # its mass flow and thrust on the staircased exit disk
    for ji, j in enumerate(jets):
        best = None
        for g_ in grids:
            nc = g_.jet_cells[ji] if ji < len(getattr(g_, "jet_cells", [])) else 0
            if nc:
                h_ = float(min(np.min(np.diff(f_)) for f_ in g_.faces))
                if best is None or h_ < best[0]:
                    best = (h_, nc, g_.jet_density_scale[ji])
        if best:
            print(f"  jet {ji}: {best[1]} exhaust cells of {best[0]:.3f} m, density x{best[2]:.3f} for the rated mass flow")
    ship_jet = np.zeros(len(jets), bool)
    ship_jet[len(jets) - n_ship:] = True

    # time step and run length
    # (the summed wave speeds in the hot plume, ~13 km/s with the Raptor 3 exit states, as seen in runs)
    dt_est = min(float(g_.cfl / (13000.0 / g_.h.min())) for g_ in grids)
    nsteps = int(a.time / dt_est) + 1
    chunk = max(10, nsteps // 200)
    print(f"~{nsteps} steps of ~{dt_est:.2e} s")

    # gas pressure and temperature next to the dome, plates, truss and ship aft end
    from rocketcfd import hotstage as HS
    thermo = HS.thermo_constants(T, V, EN.exit_state(EN.RAPTOR3_SL, p), gamma_solver=float(s.gamma), rho_min=a.rho_min)
    print(f"exhaust temperature model: cp_exh {thermo['cp_exh']:.0f} J/kg/K, gas temperature where rho > {a.rho_min:g} kg/m^3")
    wall = HS.WallSampler(grids, HS.part_meshes(S, PB, h=0.06), thermo, S,
                          folder=os.path.join(a.out, "snapshots") if a.snapshots else a.out)
    hooks = []
    if a.snapshots:
        from rocketcfd.snapshots import SnapshotRecorder
        rec = SnapshotRecorder(s, a.out, q, p, ((-1.5, -8.0, -8.0), (S.x_nose + 1.0, 8.0, 8.0)),
                               half=18.0, volume_stride=a.snap_volume, patches=grids[1:],
                               volume_patches=[g_ for g_ in grids[1:1 + len(ring_patches)]],
                               max_patch_cells=a.snap_patch_cells * 1e6, planes_full=True)
        rec.extras.append(wall.record)
        rec.record(0, 0.0)
        hooks.append((max(1, nsteps // a.snapshots), rec.record))
    else:
        wall.record(0, 0.0)
        hooks.append((max(1, nsteps // 300), wall.record))
    prb = None
    if a.probes:
        from rocketcfd.probes import ProbeRecorder
        from rocketcfd.vehicles.starship import dome_plate_centroids
        cy, cz = dome_plate_centroids(S)["outer plate 1"] if S.ring_style != "vented" else (2.5, 0.0)
        apex = PB.tank_top_x + PB.dome_rise
        pts = {"between the truss tubes, outside": (70.3, S.radius + 1.0, 0.0),
               "between the stages, axis": (0.5 * (apex + S.x_ship_aft + S.sea_level[4]), 0.0, 0.0),
               "above the centre plate": (apex + 0.25, 0.0, 0.0),
               "above an outer plate": (float(PB.tank_top_x + PB.dome_rise * np.sqrt(max(0.0, 1 - (np.hypot(cy, cz) / S.radius) ** 2))) + 0.25, cy, cz),
               "ship aft, outside": (S.x_ship_aft + 4.0, S.radius + 2.0, 0.0),
               "behind the ship's flap": (S.x_ship_aft + 13.0, S.radius + 2.0, 0.0)}
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
    x_split = S.x_aft - 0.05 if a.gap < 0.4 else S.x_aft + 0.02         # (the RVacs hang 0.3 m below the ship)
    parts = {"ship": (0, x_split, 1e9), "booster": (0, -10.0, x_split)}
    REF = dict(S_ref=63.6, L_ref=72.3, x_ref=(0.4 * 72.3, 0.0, 0.0))
    done = 0
    while s.t < a.time - 1e-12:
        # the step count is an estimate (the plume's speed of sound sets dt): run to the time
        k = chunk
        if done:
            k = max(1, min(chunk, int(np.ceil((a.time - s.t) / (s.t / done)))))
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
    if a.snapshots:
        rec.close()
    wall.save(a.out, meta=dict(alt=a.alt, mach=a.mach, ring=a.ring, ring_h=a.ring_h, wall_h=a.wall_h, ignition=a.ignition,
                               sl_cant=a.sl_cant, vac_cant=a.vac_cant, gap=a.gap,
                               booster_engines=a.booster_engines, ship_engines=a.ship_engines, **thermo))
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
               geometry=dict(ring=a.ring, sl_cant=a.sl_cant, vac_cant=a.vac_cant, gap=a.gap),
               grid=dict(h=a.h, ring=a.ring, ring_h=a.ring_h, wall_h=a.wall_h, cells=int(sum(np.prod(g_.n) for g_ in grids)), steps=s.step_count,
                         wall_s=time.time() - t0, solver="explicit time-accurate"),
               jets=dict(n=len(jets), ship=n_ship, ignition=a.ignition, engines=engine_info),
               history=hist)
    with open(f"{a.out}/loads.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {a.out}: flow.vtk, surface_cp.vtk, loads.json, hotstage.json, hotstage_peaks.npz, hotstage_*.vtk"
          + (", snapshots/" if a.snapshots else ""))
    return res


if __name__ == "__main__":
    main()
