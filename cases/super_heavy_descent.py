"""Super Heavy V3 unpowered descent (engines first) at one flight condition.

Body frame: +x from the engines toward the top. Falling engines-first means the air
arrives at the aft end, so the freestream velocity is +x, tilted by `aoa` (degrees
away from exactly engines-first) toward +z. CatchSim calls this a total angle of
attack of 180 - aoa.

Coefficients use CatchSim's references (S = 63.6 m^2, L = 72.3 m, moments about the
centre of mass at 0.40 L from the aft end), so they compare directly with
quatsim/aero.py:  drag Cd * fin_drag_factor, cn_alpha (per sin aoa), cm_alpha (per sin aoa).

Examples
  python cases/super_heavy_descent.py --h 0.5 --fins none           # CPU, ~1 h
  python cases/super_heavy_descent.py --h 0.1 --fins cad_filled     # one GPU
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def catchsim_reference(mach, aoa_deg):
    """The current placeholder values in CatchSim's quatsim/aero.py, for comparison."""
    m = abs(mach)
    if m < 0.8:
        cd = 0.85
    elif m < 1.2:
        cd = 0.85 + 0.75 * (m - 0.8) / 0.4
    elif m < 3.0:
        cd = 1.60 - 0.35 * (m - 1.2) / 1.8
    else:
        cd = 1.25
    return dict(cd_body=cd, cd_with_fins=2.0 * cd, cn_alpha=2.0, cm_alpha=0.35)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=10745.0, help="altitude [m] (CatchSim peak-q: 10.7 km)")
    ap.add_argument("--mach", type=float, default=2.82)
    ap.add_argument("--aoa", type=float, default=5.0, help="deg away from engines-first")
    ap.add_argument("--windward", type=float, default=180.0,
                    help="side facing the crossflow, azimuth in dimensions.md convention: "
                         "0 = rudder fin side, 90/270 = catch fins, 180 = raceway / taller chines")
    ap.add_argument("--h", type=float, default=0.5, help="cell size near the vehicle [m]")
    ap.add_argument("--h-far", type=float, default=0.0, help="farfield cell size [m] (default 4 x --h)")
    ap.add_argument("--h-fins", type=float, default=0.0,
                    help="cell size around the grid fins [m] (0 = same as --h). Refines whole slabs: costly")
    ap.add_argument("--growth", type=float, default=1.12, help="max cell-size ratio between neighbours")
    ap.add_argument("--uniform", action="store_true", help="uniform grid at --h everywhere (old behaviour)")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="build the case, time N explicit steps, N implicit iterations and the load "
                         "integration (per-kernel breakdown), then exit")
    ap.add_argument("--snapshots", type=int, default=0,
                    help="save about this many frames over the run for `viewer --movie evolve` "
                         "(use with --time-accurate for a physically time-accurate movie)")
    ap.add_argument("--snap-volume", type=int, default=0, help="also save a coarse 3-D Mach field (stride)")
    ap.add_argument("--probes", action="store_true",
                    help="record rho, u, v, w, p every step at a default set of points (wake, leeward hull, "
                         "fins, engine end) into probes.npz: time histories / spectra of unsteady runs")
    ap.add_argument("--probe-points", default="", help='extra probe points "x,y,z; x,y,z" [m] (implies --probes)')
    ap.add_argument("--probe-every", type=int, default=1, help="record the probes every n steps")
    ap.add_argument("--time-accurate", action="store_true",
                    help="global time step, all cells every step (off: local time steps + active tiles)")
    ap.add_argument("--fins", default="none", choices=["none", "parametric", "cad", "cad_filled"],
                    help="parametric = V3 fin model as thin walls (needs cells well below the 0.31 m "
                         "lattice spacing near the fins, e.g. --h-fins 0.03)")
    ap.add_argument("--fin-deflection", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--no-chines", action="store_true", help="remove all four chines (to isolate their effect)")
    ap.add_argument("--turbulent", action="store_true",
                    help="viscous air (Sutherland) with k-omega SST and the wall function (default: inviscid)")
    ap.add_argument("--les", action="store_true",
                    help="unsteady wall-modelled LES (Vreman subgrid model + equilibrium wall model), "
                         "time-accurate explicit steps: unsteady wakes, movies (with --snapshots/--snap-volume)")
    ap.add_argument("--engines", default="none", choices=["none", "centre", "inner", "all"],
                    help="engines firing (landing / boostback burns, retropropulsion, plume visuals): "
                         "centre = 3, inner = 13 (centre + inner ring), all = 33. One perfect gas: the exhaust "
                         "is air at --exit-T chosen to give the right exit velocity")
    ap.add_argument("--exit-mach", type=float, default=3.5, help="nozzle exit Mach number")
    ap.add_argument("--exit-p", type=float, default=80e3, help="nozzle exit pressure [Pa]")
    ap.add_argument("--exit-T", type=float, default=2200.0,
                    help="exit temperature [K] for air (2200 K with Mach 3.5 gives ~3300 m/s, Raptor-like)")
    ap.add_argument("--throttle", type=float, default=1.0, help="0..1, blends the exit state from ambient")
    ap.add_argument("--plume-h", type=float, default=0.0,
                    help="refine a block in front of the engines (the plume) to this cell size [m] (0 = off)")
    ap.add_argument("--plume-length", type=float, default=30.0, help="length of the refined plume block [m]")
    ap.add_argument("--upstream", type=float, default=15.5,
                    help="domain length ahead of the engines [m] (more for long plumes, e.g. 45 with --engines)")
    ap.add_argument("--chines", choices=["all", "tall", "short", "none"], default="all",
                    help="which chines to fit: tall = the raceway-side pair, short = the rudder-side pair")
    ap.add_argument("--implicit", action="store_true",
                    help="implicit steady solver (DP-LUR pseudo-time stepping): far fewer iterations "
                         "than the explicit solver, most useful with fine near-wall cells")
    ap.add_argument("--cfl-max", type=float, default=0.0, help="implicit solver CFL number (default 20; 5 with engines firing)")
    ap.add_argument("--explicit", action="store_true",
                    help="with patches: explicit local-time-step steps instead of the implicit solver "
                         "(cheaper per step; often faster for inviscid runs without very thin cells)")
    ap.add_argument("--wall-h", default="",
                    help="local refinement: a shell of patches around the hull with cells hx,hy,hz [m], "
                         "e.g. 0.25,0.03,0.03 (fine across the wall, long along the booster). Implies --implicit")
    ap.add_argument("--wall-shell", type=float, default=0.45, help="thickness of the refined hull shell [m]")
    ap.add_argument("--fin-h", type=float, default=0.0,
                    help="local refinement: patches of this cell size [m] around the grid fins only, "
                         "e.g. 0.03 (replaces --h-fins slab refinement). Implies --implicit")
    ap.add_argument("--fin-shell", type=float, default=0.3, help="refined distance around the fins [m]")
    ap.add_argument("--engine-h", type=float, default=0.0,
                    help="local refinement of the engine end (bells, thrust plate, skirt) to this cell size [m], "
                         "e.g. 0.06: resolves the ~4 cm gaps between the outer nozzles that make drag grid-sensitive")
    ap.add_argument("--engine-shell", type=float, default=0.5, help="refined distance around the engine end [m]")
    ap.add_argument("--patch-box", default="32,96,96", help="max patch size in cells (x,y,z)")
    ap.add_argument("--iters", type=int, default=4000, help="implicit solver: maximum iterations")
    ap.add_argument("--flowthroughs", type=float, default=4.0, help="maximum run length")
    ap.add_argument("--min-flowthroughs", type=float, default=1.0, help="never stop before this")
    ap.add_argument("--tol", type=float, default=5e-4,
                    help="stop when CD, CN, C_restore change less than this over the last 3 checks (0 = off)")
    ap.add_argument("--steps", type=int, default=0, help="override step count")
    ap.add_argument("--out", default="results/sh_descent")
    ap.add_argument("--x64", action="store_true", help="double precision")
    ap.add_argument("--restart", default="",
                    help="start from a previous run's restart.npz (or its folder): same grid = continue exactly; "
                         "other grid / physics = interpolated start (e.g. LES or RANS from a converged inviscid run)")
    a = ap.parse_args(argv)
    if not a.cfl_max:
        # plumes against the oncoming flow are strongly unsteady: smaller pseudo time steps
        a.cfl_max = 5.0 if a.engines != "none" else 20.0
    if a.les:
        # LES is unsteady: physical time steps, explicit, and the run never stops early (the
        # reported coefficients are averages over the last quarter of the run)
        a.time_accurate, a.explicit, a.implicit, a.tol = True, True, False, 0.0

    import torch
    from rocketcfd import Solver, atmosphere
    from rocketcfd.convergence import settled
    from rocketcfd.vehicles.super_heavy import super_heavy_sdf, v3_params, REFERENCE
    from rocketcfd.output import plot_slices, write_vtk

    os.makedirs(a.out, exist_ok=True)
    T, p, rho, c = atmosphere(a.alt)
    V = a.mach * c
    from rocketcfd.vehicles.super_heavy import _az
    al = np.radians(a.aoa)
    tw = _az(a.windward)
    # air arrives on the windward side and moves across toward the leeward side
    fs = dict(rho=rho, u=V * np.cos(al), v=-V * np.sin(al) * np.cos(tw), w=-V * np.sin(al) * np.sin(tw), p=p)
    q = 0.5 * rho * V * V
    print(f"alt {a.alt/1e3:.2f} km  T {T:.1f} K  p {p:.0f} Pa  rho {rho:.4f}  V {V:.1f} m/s  "
          f"M {a.mach}  q {q/1e3:.1f} kPa  aoa {a.aoa} deg from engines-first, windward side {a.windward} deg")

    P = v3_params(fin_model=a.fins, fin_deflection_deg=tuple(a.fin_deflection))
    which = "none" if a.no_chines else a.chines
    if which != "all":
        tall = tuple(c for c in P.chines if c[2] > 32.0)
        P.chines = {"none": (), "tall": tall, "short": tuple(c for c in P.chines if c not in tall)}[which]
        print(f"chines fitted: {which} ({len(P.chines)})")
    sdf = lambda X, Y, Z: super_heavy_sdf(X, Y, Z, P)
    thin_sdf = thin_box = None
    if a.fins == "parametric":
        from rocketcfd.vehicles.super_heavy import super_heavy_fins_sdf, super_heavy_fins_box
        thin_sdf = lambda X, Y, Z: super_heavy_fins_sdf(X, Y, Z, P)
        thin_box = super_heavy_fins_box(P)

    L = P.length
    lo = (-a.upstream, -22.0, -22.0)
    hi = (L + 18.0, 22.0, 22.0)
    dtype = torch.float64 if a.x64 else torch.float32
    faces = None
    if not a.uniform:
        from rocketcfd.grid import Zone, stretched_faces, describe
        r_out = P.radius + 1.35 + 0.5                     # chines reach 5.85 m; outer bells 5.05 m
        zones = [Zone((-1.5, -r_out, -r_out), (L + 1.0, r_out, r_out), a.h, "vehicle")]
        if a.fins != "none" and a.h_fins and a.h_fins < a.h:
            from rocketcfd.vehicles.super_heavy import super_heavy_fins_box
            fb = super_heavy_fins_box(P, margin=0.3)      # only where fins are (none on the tower side)
            zones.append(Zone(fb[0], fb[1], a.h_fins, "grid fins"))
        faces = stretched_faces(lo, hi, a.h_far or 4 * a.h, zones, growth=a.growth)
        print(describe(faces))
    else:
        h3 = np.broadcast_to(np.asarray(a.h, float), (3,))
        faces = [lo[d] + h3[d] * np.arange(int(round((hi[d] - lo[d]) / h3[d])) + 1) for d in range(3)]
    patches = []
    if a.wall_h or a.fin_h or a.engine_h or a.plume_h:
        from rocketcfd.patches import wall_patches
        a.implicit = not a.explicit
        pbox = tuple(int(v) for v in a.patch_box.split(","))
        if a.wall_h:
            hw = tuple(float(v) for v in a.wall_h.split(","))
            hw = hw * 3 if len(hw) == 1 else hw
            print(f"hull shell patches ({hw} m cells within {a.wall_shell} m of the hull):")
            patches += wall_patches(faces, hw, a.wall_shell, sdf=sdf, box=pbox)
        if a.engine_h:
            R_eng = P.radius + 1.0
            reg = ((-2.0, -R_eng, -R_eng), (P.skirt_bottom_x + 1.0, R_eng, R_eng))
            print(f"engine-end patches ({a.engine_h} m cells within {a.engine_shell} m of the walls, x < {reg[1][0]:.1f} m):")
            patches += wall_patches(faces, a.engine_h, a.engine_shell, sdf=sdf, region=reg, box=(64, 64, 64))
        if a.plume_h:
            Rp = P.radius + 1.5
            reg = ((-a.plume_length, -Rp, -Rp), (0.5, Rp, Rp))
            print(f"plume block patches ({a.plume_h} m cells, x {reg[0][0]:.0f} .. {reg[1][0]:.1f} m, radius {Rp:.1f} m):")
            patches += wall_patches(faces, a.plume_h, 0.0, sdf=lambda X, Y, Z: np.zeros_like(X) - 1e-9, region=reg,
                                    box=(64, 64, 64), inner=1e9)
        if a.fin_h:
            if thin_sdf is None:
                raise SystemExit("--fin-h needs --fins parametric")
            print(f"grid fin patches ({a.fin_h} m cells within {a.fin_shell} m of the fins):")
            patches += wall_patches(faces, a.fin_h, a.fin_shell, thin_sdf=thin_sdf, thin_box=thin_box,
                                    box=(64, 64, 64))
    if not a.uniform or patches:
        if a.dry_run:
            n = np.prod([len(f) - 1 for f in faces]) + sum(int(np.prod([len(f) - 1 for f in pf])) for pf in patches)
            try:
                import warp  # noqa: F401
                per, what = (175, "Warp kernels + SST") if a.turbulent else (115, "Warp kernels")
                if a.implicit:
                    per, what = per + 25, what + " + implicit"
            except ImportError:
                per, what = 500, "PyTorch path"
            if patches:
                print(f"base grid + {len(patches)} patches = {n/1e6:.2f} M cells")
            print(f"estimated GPU memory ~{n * per / 2**30:.1f} GB ({what}, ~{per} B/cell); "
                  f"setup needs ~{n * 120 / 2**30:.0f} GB of system RAM")
            return None
    jets = None
    if a.engines != "none":
        from rocketcfd.vehicles.super_heavy import engine_positions
        n_on = {"centre": 3, "inner": 13, "all": 33}[a.engines]
        jets = [dict(center=(float(xe), float(y), float(z)), direction=(-1.0, 0.0, 0.0),
                     radius=0.95 * P.bell_exit_radius, mach=a.exit_mach, p=a.exit_p, T=a.exit_T)
                for y, z, xe in engine_positions(P)[:n_on]]
        ve = a.exit_mach * np.sqrt(1.4 * 287.05 * a.exit_T)
        print(f"{n_on} engines firing: exit Mach {a.exit_mach}, {a.exit_p/1e3:.0f} kPa, {a.exit_T:.0f} K "
              f"-> {ve:.0f} m/s (forces below include the exit-plane pressure, not the momentum thrust)")
    visc = dict(model="sutherland") if (a.turbulent or a.les) else None
    turb = dict(model="sst") if a.turbulent else (dict(model="les") if a.les else None)
    if patches:
        from rocketcfd.patches import PatchGrid
        runner = PatchGrid(faces, patches, fs, sdf=sdf, cfl=0.7, thin_sdf=thin_sdf, thin_box=thin_box,
                           viscous=visc, turbulence=turb, jets=jets)
        s = runner.base
    else:
        s = runner = Solver(lo, hi, a.h, fs, sdf=sdf, cfl=0.7, dtype=dtype, faces=faces, thin_sdf=thin_sdf,
                            thin_box=thin_box, viscous=visc, turbulence=turb, jets=jets)
    if jets:
        for g_ in (runner.grids if patches else [runner]):
            g_.jet_throttle = a.throttle
    if s.n_ib_fallback:
        print(f"note: {s.n_ib_fallback} ghost cells used nearest-fluid fallback (thin features)")
    if a.restart:
        from rocketcfd import restart
        restart.load(runner, a.restart)

    # steps for the requested number of flow-throughs (estimated from the freestream)
    cs = c
    hmin = min(float(g.h.min()) for g in runner.grids) if (patches and a.time_accurate) else s.h.min()
    dt_est = s.cfl / ((V + cs) / hmin + 2 * (cs + V * np.sin(al)) / hmin)
    nsteps = a.steps or int(a.flowthroughs * (s.hi[0] - s.lo[0]) / V / dt_est)
    if not a.steps:
        nsteps = int(np.ceil(nsteps / 100) * 100)
    if a.implicit:
        nsteps = a.steps or a.iters
        print(f"implicit solver: up to {nsteps} iterations at CFL {a.cfl_max:g}, stopping early when converged")
    else:
        print(f"running up to {nsteps} steps ({a.flowthroughs:g} flow-throughs), stopping early when converged")

    hist = []
    full = []
    # regions for the force breakdown (x ranges in the body frame)
    xf0 = P.fin_x_center - 0.8
    PARTS = {"aft end": (0, -5.0, P.skirt_bottom_x + 0.6),
             "hull": (0, P.skirt_bottom_x + 0.6, xf0),
             "fin band": (0, xf0, P.tank_top_x - 0.05),      # fins, pods and the hull beside them
             "top": (0, P.tank_top_x - 0.05, 1e9)}
    FIN_PARTS = []
    if a.fins != "none":
        from rocketcfd.vehicles.super_heavy import fin_selectors
        fsel = fin_selectors(P)
        FIN_PARTS = list(fsel)
        PARTS.update(fsel)
    KEYS = ("CD", "CL", "CN", "CS", "C_roll", "C_restore")
    min_steps = int(a.min_flowthroughs / a.flowthroughs * nsteps) if not a.implicit else min(600, nsteps)

    def report(sv):
        ld = sv.loads(**REFERENCE, parts=PARTS)
        hist.append((sv.step_count, sv.t, ld["CD"], ld["CN"], ld["C_restore"]))
        full.append(dict({k: float(ld[k]) for k in KEYS},
                         **{f"{k}[{n}]": v[k] for n, v in ld["parts"].items() for k in ("CD", "CN", "C_restore")}))
        print(f"      CD {ld['CD']:.4f}  CL {ld['CL']:+.4f}  CN {ld['CN']:+.4f}  CS {ld['CS']:+.4f}  "
              f"C_restore {ld['C_restore']:+.5f}  C_roll {ld['C_roll']:+.5f}", flush=True)
        if a.tol > 0 and sv.step_count >= min_steps:
            vals = [h[2:] for h in hist]
            tolv = (a.tol * max(abs(hist[-1][2]), 1e-6), a.tol, a.tol)   # CD relative; CN, C_restore absolute
            if settled(vals, tolv):
                print(f"      converged: loads steady to {a.tol:g} and not drifting", flush=True)
                return True
        return False

    if a.profile:
        from rocketcfd.profiling import profile_runner
        profile_runner(runner, a.profile, loads=lambda: runner.loads(**REFERENCE, parts=PARTS))
        return None

    t0 = time.time()
    chunk = max(100, int(round(nsteps / 40 / 50)) * 50)       # ~40 load checks over the max length
    chunk = min(chunk, nsteps)
    if a.implicit:
        chunk = 50
    hooks = []
    if a.snapshots:
        from rocketcfd.snapshots import SnapshotRecorder
        rec = SnapshotRecorder(s, a.out, q, p, ((-1.5, -6.6, -6.6), (L + 0.6, 6.6, 6.6)),
                               volume_stride=a.snap_volume,
                               patches=(runner.grids[1:] if patches else None))
        rec.record(0, 0.0)
        hooks.append((max(1, nsteps // a.snapshots), rec.record))
        print(f"recording a snapshot every {hooks[-1][0]} steps into {a.out}/snapshots")
    prb = None
    if a.probes or a.probe_points:
        from rocketcfd.probes import ProbeRecorder
        lee = -np.array([np.cos(tw), np.sin(tw)])           # leeward direction in the y-z plane
        Rb = P.radius
        pts = {"wake 3 m behind the top": (L + 3.0, 0.0, 0.0), "wake 10 m behind the top": (L + 10.0, 0.0, 0.0),
               "leeward hull, mid": (40.0, *(lee * (Rb + 2.0))), "leeward hull, near top": (62.0, *(lee * (Rb + 2.0))),
               "behind rudder fin": (P.fin_x_center + 1.5, 0.0, Rb + 3.0),
               "behind catch fin 90": (P.fin_x_center + 1.5, -(Rb + 3.0), 0.0),
               "engine end, centre": (-1.5, 0.0, 0.0), "skirt edge, leeward": (1.5, *(lee * (Rb + 0.8)))}
        for k_, txt in enumerate(t_ for t_ in a.probe_points.split(";") if t_.strip()):
            pts[f"user {k_ + 1}"] = tuple(float(v) for v in txt.split(","))
        prb = ProbeRecorder(runner, list(pts.values()), a.out, names=list(pts), every=a.probe_every)
        hooks.append((a.probe_every, prb.record))
        print(f"probes: {len(pts)} points every {a.probe_every} steps -> {a.out}/probes.npz"
              + (f" ({len(prb.missing)} outside the grid)" if len(prb.missing) else ""))
    snap = None
    if hooks:
        def _hook(step, t_):
            for every, fn in hooks:
                if step % every == 0:
                    fn(step, t_)
        import math
        per = 0
        for e_, _ in hooks:
            per = math.gcd(per, int(e_))
        snap = (max(per, 1), _hook)
    if patches and a.explicit:
        runner.run_explicit(nsteps, chunk=chunk, report=report, first_order_steps=400,
                            time_accurate=a.time_accurate, snapshot=snap)
    elif patches:
        runner.run(nsteps, chunk=chunk, first_order_steps=200, report=report, snapshot=snap,
                   cfl_max=a.cfl_max, cfl_ramp=300)
    elif a.implicit:
        s.run(nsteps, chunk=chunk, first_order_steps=200, report=report, steady=True, snapshot=snap,
              implicit=True, cfl_max=a.cfl_max, cfl_ramp=300)
    else:
        s.run(nsteps, chunk=chunk, first_order_steps=400, report=report, steady=not a.time_accurate,
              snapshot=snap)
    wall = time.time() - t0
    if prb is not None:
        prb.save()

    ld = runner.loads(**REFERENCE, parts=PARTS)
    sa = np.sin(al)
    # average over the last quarter of the load checks (at least 4): unsteady wakes make a
    # single snapshot misleading; the standard deviation shows how unsteady it was
    tail = full[-max(4, len(full) // 4):]
    avg = {k: float(np.mean([t_[k] for t_ in tail])) for k in tail[0]}
    std = {k: float(np.std([t_[k] for t_ in tail])) for k in tail[0]}
    ref = catchsim_reference(a.mach, a.aoa)
    res = dict(
        condition=dict(alt=a.alt, mach=a.mach, aoa_from_engines_first=a.aoa, V=V, q=q, p=p, rho=rho, T=T),
        grid=dict(h=a.h, h_far=None if a.uniform else (a.h_far or 4 * a.h), h_fins=a.h_fins or None,
                  uniform=a.uniform, growth=a.growth, n=list(s.n), cells=int(np.prod(s.n)),
                  steps=s.step_count, wall_s=wall, solver="implicit" if a.implicit else "explicit",
                  device=str(s.device), gpu=torch.cuda.get_device_name(s.device) if s.device.type == "cuda" else None),
        fins=dict(model=a.fins, scale=list(P.fin_scale), deflection=list(a.fin_deflection)),
        condition_windward=a.windward,
        coefficients={k: float(ld[k]) for k in ("CD", "CL", "CN", "CS", "CX", "CY", "CZ",
                                                  "Cl", "Cm", "Cn", "C_roll", "C_restore")},
        averaged=avg, averaged_std=std, averaged_over_checks=len(tail),
        catchsim_equivalent=dict(
            drag_coefficient=avg["CD"],
            cn_alpha=avg["CN"] / sa if sa > 1e-6 else None,
            cm_alpha=avg["C_restore"] / sa if sa > 1e-6 else None,   # >0: restoring (engines-first stable)
        ),
        catchsim_current=ref,
        reference=dict(S_ref=REFERENCE["S_ref"], L_ref=REFERENCE["L_ref"], x_ref=list(REFERENCE["x_ref"])),
        history=hist,
    )
    with open(f"{a.out}/loads.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    f = runner.fields(0) if patches else s.fields()
    if patches:
        runner.write_patches_vtm(f"{a.out}/flow_patches.vtm")
    title = (f"Super Heavy V3, engines-first descent: alt {a.alt/1e3:.1f} km, M {a.mach}, "
             f"{a.aoa} deg off engines-first, windward {a.windward:g} deg, fins={a.fins}"
             f"{'' if which == 'all' else ', chines=' + which}, h={a.h} m{' uniform' if a.uniform else ' near body'}"
             f"{', SST turbulent' if a.turbulent else ', inviscid'}")
    plot_slices(f, f"{a.out}/slice_xz.png", title=title + "  [x-z plane: rudder fin +z]", p_inf=p, normal_axis=1)
    plot_slices(f, f"{a.out}/slice_xy.png", title=title + "  [x-y plane: catch fins]", p_inf=p, normal_axis=2)
    write_vtk(f, f"{a.out}/flow.vtk")
    from rocketcfd import restart as _rs
    _rs.save(runner, f"{a.out}/restart.npz")

    # cross-section stations and 3-D surface pressure
    from rocketcfd.visual3d import plot_stations, surface_pressure, write_vtk_surface, render_surface, \
        fin_outline_polys
    stations = [(0.6, "engine bells"), (2.2, "bells at the thrust plate"), (12.0, "lower hull, all 4 chines"),
                (32.0, "taller chines only"), (50.0, "upper hull"), (P.fin_x_center, "grid fins"),
                (70.0, "top (truss / dome)")]
    plot_stations(f, f"{a.out}/stations.png", stations, p, q,
                  title=title + "\nCp and Mach around the booster (looking from the engines toward the top)",
                  crossflow=(fs["v"], fs["w"]), fin_sdf=thin_sdf)
    region = ((-1.5, -6.6, -6.6), (L + 0.6, 6.6, 6.6))
    grids = runner.grids if patches else None
    xyz, tris, cp_s = surface_pressure(s, q, region, grids=grids)
    if len(tris) > 400_000:
        xyz, tris, cp_s = surface_pressure(s, q, region, stride=2, grids=grids)
    extra_s, extra_v = {}, {}
    if a.turbulent:
        from rocketcfd.visual3d import surface_friction
        hw = min(float(np.min(g.h)) for g in runner.grids) if patches else a.h
        cf_s, shear = surface_friction(s, xyz, tris, q, offset=1.5 * hw if patches else a.h, grids=grids)
        extra_s["Cf"], extra_v["wall_shear"] = cf_s, shear
    write_vtk_surface(f"{a.out}/surface_cp.vtk", xyz, tris, cp_s, scalars=extra_s, vectors=extra_v)
    fin_polys = []
    if a.fins != "none":
        from rocketcfd.vehicles.grid_fin import GridFinParams
        fin_polys = fin_outline_polys(P, GridFinParams())
    tw_vec = (np.cos(tw), np.sin(tw))
    render_surface(f"{a.out}/surface_cp.png", xyz, tris, cp_s, title=title, fin_polys=fin_polys, windward=tw_vec)
    if a.turbulent:
        render_surface(f"{a.out}/skin_friction.png", xyz, tris, cf_s, title=title + "  (skin friction)",
                       fin_polys=fin_polys, windward=tw_vec, label="Cf  (skin friction coefficient)",
                       cmap_name="magma", diverging=False)
        try:
            from rocketcfd.visual3d import render_friction_lines
            from rocketcfd.viewer import fin_meshes
            render_friction_lines(f"{a.out}/friction_lines.png", f"{a.out}/surface_cp.vtk",
                                  fin_meshes(res) if a.fins != "none" else [], windward=tw_vec, title=title)
        except Exception as e:                    # noqa: BLE001  (pyvista is optional)
            print(f"friction lines skipped: {type(e).__name__}: {e}")
    from rocketcfd.visual3d import plot_summary
    plot_summary(f"{a.out}/summary.png", title, hist, avg, std, PARTS, FIN_PARTS, ref, int(np.prod(s.n)),
                 s.step_count, wall / 60)

    ce = res["catchsim_equivalent"]
    print("\n=== result ===")
    print(f"cells {np.prod(s.n)/1e6:.2f} M, {s.step_count} steps, {wall/60:.1f} min")
    print(f"averaged over the last {len(tail)} load checks (± = unsteadiness, 1 std):")
    print(f"  CD {avg['CD']:.4f} ± {std['CD']:.4f}   CN {avg['CN']:+.4f} ± {std['CN']:.4f}   "
          f"C_restore {avg['C_restore']:+.5f} ± {std['C_restore']:.5f}")
    for n in [k for k in PARTS if k not in FIN_PARTS]:
        print(f"    {n:9s}: CD {avg[f'CD[{n}]']:+.4f} ± {std[f'CD[{n}]']:.4f}   CN {avg[f'CN[{n}]']:+.4f}   "
              f"C_restore {avg[f'C_restore[{n}]']:+.5f}")
    fin_names = FIN_PARTS
    if fin_names:
        print("  per fin (body axes, x toward the top; the fins are also inside 'fin band'):")
        for n in fin_names:
            print(f"    {n:13s}: CX {ld['parts'][n]['CX']:+.4f}  CY {ld['parts'][n]['CY']:+.4f}  "
                  f"CZ {ld['parts'][n]['CZ']:+.4f}  C_restore {ld['parts'][n]['C_restore']:+.5f}")
    print(f"drag coefficient        CFD {ce['drag_coefficient']:.3f}   CatchSim body {ref['cd_body']:.3f}, "
          f"with fin factor {ref['cd_with_fins']:.3f}")
    if ce["cn_alpha"] is not None:
        print(f"normal-force slope      CFD {ce['cn_alpha']:.3f}   CatchSim {ref['cn_alpha']}")
        print(f"restoring-moment slope  CFD {ce['cm_alpha']:+.4f}  CatchSim {ref['cm_alpha']}  "
              f"({'restoring' if ce['cm_alpha'] > 0 else 'DESTABILISING'})")
    return res


if __name__ == "__main__":
    main()
