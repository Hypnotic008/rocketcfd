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
    ap.add_argument("--time-accurate", action="store_true",
                    help="global time step, all cells every step (off: local time steps + active tiles)")
    ap.add_argument("--fins", default="cad_filled", choices=["none", "cad", "cad_filled"])
    ap.add_argument("--fin-deflection", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--flowthroughs", type=float, default=4.0, help="maximum run length")
    ap.add_argument("--min-flowthroughs", type=float, default=1.0, help="never stop before this")
    ap.add_argument("--tol", type=float, default=5e-4,
                    help="stop when CD, CN, C_restore change less than this over the last 3 checks (0 = off)")
    ap.add_argument("--steps", type=int, default=0, help="override step count")
    ap.add_argument("--out", default="results/sh_descent")
    ap.add_argument("--x64", action="store_true", help="double precision")
    a = ap.parse_args(argv)

    import torch
    from rocketcfd import Solver, atmosphere
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
    sdf = lambda X, Y, Z: super_heavy_sdf(X, Y, Z, P)

    L = P.length
    lo = (-15.5, -22.0, -22.0)
    hi = (L + 18.0, 22.0, 22.0)
    dtype = torch.float64 if a.x64 else torch.float32
    faces = None
    if not a.uniform:
        from rocketcfd.grid import Zone, stretched_faces, describe
        r_out = P.radius + 1.35 + 0.5                     # chines reach 5.85 m; outer bells 5.05 m
        zones = [Zone((-1.5, -r_out, -r_out), (L + 1.0, r_out, r_out), a.h, "vehicle")]
        if a.fins != "none" and a.h_fins and a.h_fins < a.h:
            r_fin = P.radius + 4.2 + 0.4
            zones.append(Zone((P.fin_x_center - 0.8, -r_fin, -r_fin), (P.fin_x_center + 0.8, r_fin, r_fin),
                              a.h_fins, "grid fins"))
        faces = stretched_faces(lo, hi, a.h_far or 4 * a.h, zones, growth=a.growth)
        print(describe(faces))
        if a.dry_run:
            n = np.prod([len(f) - 1 for f in faces])
            try:
                import warp  # noqa: F401
                per, what = 115, "Warp kernels"
            except ImportError:
                per, what = 500, "PyTorch path"
            print(f"estimated GPU memory ~{n * per / 2**30:.1f} GB ({what}, ~{per} B/cell); "
                  f"setup needs ~{n * 120 / 2**30:.0f} GB of system RAM")
            return None
    s = Solver(lo, hi, a.h, fs, sdf=sdf, cfl=0.7, dtype=dtype, faces=faces)
    if s.n_ib_fallback:
        print(f"note: {s.n_ib_fallback} ghost cells used nearest-fluid fallback (thin features)")

    # steps for the requested number of flow-throughs (estimated from the freestream)
    cs = c
    hmin = s.h.min()
    dt_est = s.cfl / ((V + cs) / hmin + 2 * (cs + V * np.sin(al)) / hmin)
    nsteps = a.steps or int(a.flowthroughs * (s.hi[0] - s.lo[0]) / V / dt_est)
    nsteps = int(np.ceil(nsteps / 100) * 100)
    print(f"running up to {nsteps} steps ({a.flowthroughs:g} flow-throughs), stopping early when converged")

    hist = []
    full = []
    # regions for the force breakdown (x ranges in the body frame)
    PARTS = {"aft end": (0, -5.0, P.skirt_bottom_x + 0.6),
             "hull": (0, P.skirt_bottom_x + 0.6, P.tank_top_x - 0.05),
             "top": (0, P.tank_top_x - 0.05, 1e9)}
    KEYS = ("CD", "CL", "CN", "CS", "C_roll", "C_restore")
    min_steps = int(a.min_flowthroughs / a.flowthroughs * nsteps)

    def report(sv):
        ld = sv.loads(**REFERENCE, parts=PARTS)
        hist.append((sv.step_count, sv.t, ld["CD"], ld["CN"], ld["C_restore"]))
        full.append(dict({k: float(ld[k]) for k in KEYS},
                         **{f"{k}[{n}]": v[k] for n, v in ld["parts"].items() for k in ("CD", "CN", "C_restore")}))
        print(f"      CD {ld['CD']:.4f}  CL {ld['CL']:+.4f}  CN {ld['CN']:+.4f}  CS {ld['CS']:+.4f}  "
              f"C_restore {ld['C_restore']:+.5f}  C_roll {ld['C_roll']:+.5f}", flush=True)
        if a.tol > 0 and sv.step_count >= min_steps and len(hist) >= 4:
            recent = np.array([h[2:] for h in hist[-4:]])
            change = np.abs(recent - recent[-1]).max(0)
            change[0] /= max(abs(recent[-1, 0]), 1e-6)          # CD relative, CN / C_restore absolute
            if np.all(change < a.tol):
                print(f"      converged: loads changed < {a.tol:g} over the last 3 checks", flush=True)
                return True
        return False

    t0 = time.time()
    chunk = max(100, int(round(nsteps / 40 / 50)) * 50)       # ~40 load checks over the max length
    s.run(nsteps, chunk=chunk, first_order_steps=400, report=report, steady=not a.time_accurate)
    wall = time.time() - t0

    ld = s.loads(**REFERENCE, parts=PARTS)
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
                  steps=s.step_count, wall_s=wall,
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

    f = s.fields()
    title = (f"Super Heavy V3, engines-first descent: alt {a.alt/1e3:.1f} km, M {a.mach}, "
             f"{a.aoa} deg off engines-first, fins={a.fins}, h={a.h} m{' uniform' if a.uniform else ' near body'}")
    plot_slices(f, f"{a.out}/slice_xz.png", title=title + "  [x-z plane: rudder fin +z]", p_inf=p, normal_axis=1)
    plot_slices(f, f"{a.out}/slice_xy.png", title=title + "  [x-y plane: catch fins]", p_inf=p, normal_axis=2)
    write_vtk(f, f"{a.out}/flow.vtk")
    s.save(f"{a.out}/restart.npz")

    ce = res["catchsim_equivalent"]
    print("\n=== result ===")
    print(f"cells {np.prod(s.n)/1e6:.2f} M, {s.step_count} steps, {wall/60:.1f} min")
    print(f"averaged over the last {len(tail)} load checks (± = unsteadiness, 1 std):")
    print(f"  CD {avg['CD']:.4f} ± {std['CD']:.4f}   CN {avg['CN']:+.4f} ± {std['CN']:.4f}   "
          f"C_restore {avg['C_restore']:+.5f} ± {std['C_restore']:.5f}")
    for n in PARTS:
        print(f"    {n:8s}: CD {avg[f'CD[{n}]']:+.4f} ± {std[f'CD[{n}]']:.4f}   CN {avg[f'CN[{n}]']:+.4f}   "
              f"C_restore {avg[f'C_restore[{n}]']:+.5f}")
    print(f"drag coefficient        CFD {ce['drag_coefficient']:.3f}   CatchSim body {ref['cd_body']:.3f}, "
          f"with fin factor {ref['cd_with_fins']:.3f}")
    if ce["cn_alpha"] is not None:
        print(f"normal-force slope      CFD {ce['cn_alpha']:.3f}   CatchSim {ref['cn_alpha']}")
        print(f"restoring-moment slope  CFD {ce['cm_alpha']:+.4f}  CatchSim {ref['cm_alpha']}  "
              f"({'restoring' if ce['cm_alpha'] > 0 else 'DESTABILISING'})")
    return res


if __name__ == "__main__":
    main()
