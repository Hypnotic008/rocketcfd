"""Starship re-entry: the ship falls belly-first at high angle of attack (~60-70 deg), flaps out.
Steady inviscid solution: bow shock, shock layer, wake; normal / axial force and pitching moment.

Geometry: rocketcfd.vehicles.starship (approximate public dimensions), ship alone, aft end at x = 0,
nose toward +x, the belly (windward side) facing +z.

  python cases/starship_reentry.py --dry-run
  python cases/starship_reentry.py --mach 8 --alt 55000 --aoa 65 --h 0.4 --out results/reentry
  python -m rocketcfd.cinematic results/reentry --mode orbit --style mach

Caution: one perfect gas (gamma 1.4). Above Mach ~6 real air dissociates; shock-layer temperatures
here are far too high and the shock sits closer to the body in reality. Shock shapes, pressure
distribution and moments are first-order right; temperatures are not.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mach", type=float, default=8.0)
    ap.add_argument("--alt", type=float, default=55000.0)
    ap.add_argument("--aoa", type=float, default=65.0, help="angle of attack [deg] (belly into the wind)")
    ap.add_argument("--h", type=float, default=0.5, help="cell size near the ship [m]")
    ap.add_argument("--growth", type=float, default=1.12)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--cfl-max", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/reentry")
    a = ap.parse_args(argv)

    from rocketcfd import Solver, atmosphere
    from rocketcfd.grid import Zone, stretched_faces, describe
    from rocketcfd.vehicles.starship import StarshipParams, starship_sdf
    from rocketcfd.convergence import settled
    from rocketcfd.output import write_vtk

    os.makedirs(a.out, exist_ok=True)
    T, p, rho, c = atmosphere(a.alt)
    V = a.mach * c
    al = np.radians(a.aoa)
    fs = dict(rho=rho, u=-V * np.cos(al), v=0.0, w=-V * np.sin(al), p=p)   # air from the nose and the belly (+z)
    q = 0.5 * rho * V * V
    T0 = T * (1 + 0.2 * a.mach ** 2)
    print(f"alt {a.alt/1e3:.0f} km  M {a.mach}  V {V:.0f} m/s  q {q/1e3:.2f} kPa  aoa {a.aoa} deg  "
          f"(perfect-gas stagnation temperature {T0:.0f} K)")
    S = StarshipParams(x_aft=0.0)
    sdf = lambda X, Y, Z: starship_sdf(X, Y, Z, S)
    L = S.length
    R = S.radius
    lo = (-30.0, -30.0, -40.0)
    hi = (L + 20.0, 30.0, 25.0)
    zones = [Zone((-1.0, -R - 4.5, -R - 1.5), (L + 1.0, R + 4.5, R + 3.0), a.h, "ship")]
    faces = stretched_faces(lo, hi, 4 * a.h, zones, growth=a.growth)
    print(describe(faces))
    if a.dry_run:
        n = np.prod([len(f) - 1 for f in faces])
        print(f"{n/1e6:.2f} M cells, ~{n * 140 / 2**30:.1f} GB GPU memory")
        return None
    s = Solver(None, None, None, fs, faces=faces, sdf=sdf, cfl=0.6)
    REF = dict(S_ref=np.pi * R * R, L_ref=L, x_ref=(0.45 * L, 0.0, 0.0))     # CN, CA: body axes; CN > 0 toward the lee
    hist = []

    def report(sv):
        ld = sv.loads(**REF)
        hist.append((sv.step_count, ld["CX"], ld["CZ"], ld["Cm"]))
        print(f"      CA {-ld['CX']:+.4f}  CN {-ld['CZ']:+.4f}  Cm {ld['Cm']:+.5f}  (x_ref {REF['x_ref'][0]:.1f} m)",
              flush=True)
        if a.tol > 0 and len(hist) > 10:
            return settled([h_[1:] for h_ in hist], a.tol)
        return False

    t0 = time.time()
    s.run(a.iters, chunk=50, first_order_steps=300, report=report, steady=True, implicit=True,
          cfl_max=a.cfl_max, cfl_ramp=500)
    ld = s.loads(**REF)
    f = s.fields()
    write_vtk(f, f"{a.out}/flow.vtk")
    from rocketcfd.visual3d import surface_pressure, write_vtk_surface
    xyz, tris, cp_s = surface_pressure(s, q, ((-2.0, -R - 5, -R - 5), (L + 1.0, R + 5, R + 5)))
    write_vtk_surface(f"{a.out}/surface_cp.vtk", xyz, tris, cp_s)
    from rocketcfd import restart
    restart.save(s, f"{a.out}/restart.npz")
    res = dict(condition=dict(alt=a.alt, mach=a.mach, aoa=a.aoa, V=V, q=q, p=p, rho=rho, T=T),
               grid=dict(h=a.h, cells=int(np.prod(s.n)), steps=s.step_count, wall_s=time.time() - t0, solver="implicit"),
               coefficients={k: float(ld[k]) for k in ("CX", "CY", "CZ", "Cl", "Cm", "Cn", "CD", "CL")},
               reference=dict(S_ref=REF["S_ref"], L_ref=L, x_ref=list(REF["x_ref"])),
               history=hist)
    with open(f"{a.out}/loads.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"=== result ===\n  CA {-ld['CX']:+.4f}  CN {-ld['CZ']:+.4f} (toward the lee side)  Cm {ld['Cm']:+.5f}  "
          f"CD {ld['CD']:.4f}  CL {ld['CL']:+.4f}   (S_ref = pi R^2 = {REF['S_ref']:.1f} m^2, L_ref {L:.0f} m)")
    print(f"wrote {a.out}")
    return res


if __name__ == "__main__":
    main()
