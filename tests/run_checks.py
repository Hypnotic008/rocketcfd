"""One command to check a machine (or a code change): runs the quick verification cases and
prints PASS / FAIL against the results recorded in the README.

  python tests/run_checks.py            # all (a few minutes on a GPU, ~1 h on a laptop CPU)
  python tests/run_checks.py vortex patch_blasius
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def check_vortex():
    import vortex_order as v
    e = {lim: [v.run(n, lim) for n in (32, 64)] for lim in ("vanleer", "koren")}
    order_k = np.log2(e["koren"][0] / e["koren"][1])
    ok = order_k > 2.4 and e["koren"][1] < e["vanleer"][1]
    return ok, f"isentropic vortex: Koren order {order_k:.2f} (>2.4), L1 {e['koren'][1]:.2e} vs van Leer {e['vanleer'][1]:.2e}"


def check_patch_blasius():
    import patch_blasius as pb
    single, patch = pb.run("single", 3000, 20.0), pb.run("patch", 3000, 20.0)
    d = patch["drag_f"] / single["drag_f"] - 1
    return abs(d) < 0.01, f"Blasius plate: patch friction drag {patch['drag_f']:.5f} vs single grid {single['drag_f']:.5f} ({100*d:+.2f} %, need <1 %)"


def check_patch_sphere():
    import patch_sphere as ps
    single, patch, coarse = ps.run("single"), ps.run("patch"), ps.run("coarse")
    d = patch["CD"] / single["CD"] - 1
    closed = 1 - (patch["CD"] - single["CD"]) / (coarse["CD"] - single["CD"])
    ok = abs(d) < 0.015 and abs(patch["CD"] - single["CD"]) < abs(coarse["CD"] - single["CD"])
    return ok, (f"Mach 3 sphere: patch CD {patch['CD']:.4f}, fine {single['CD']:.4f}, coarse {coarse['CD']:.4f}: "
                f"{100*d:+.2f} % from fine (<1.5 %), closes {100*closed:.0f} % of the gap")


def check_low_mach():
    import low_mach_cylinder as lm
    cd0, _, cpr0, _, rms0 = lm.run(0.1, False)
    cd1, _, cpr1, _, rms1 = lm.run(0.1, True)
    ok = abs(cd1) < abs(cd0) and abs(cpr1 - 1.0) < abs(cpr0 - 1.0) and abs(cd1) < 0.03
    return ok, (f"Mach 0.1 cylinder (exact: CD 0, rear Cp 1): without the low-Mach correction CD {cd0:+.4f}, rear Cp "
                f"{cpr0:.3f}, rms Cp error {rms0:.3f}; with it (default for subsonic flow) CD {cd1:+.4f}, rear Cp "
                f"{cpr1:.3f}, rms {rms1:.3f}")


def check_shocks():
    import limiter_shocks as ls
    e = ls.sod("koren")
    so, ps, _ = ls.cylinder("koren")
    # 25 cells per radius: the standoff is ~+12 % here (+7 % at 50 cells per radius)
    return e < 0.003 and abs(so) < 0.15 and abs(ps) < 0.03, (f"Sod L1(rho) {e:.5f} (<0.003); Mach 4 cylinder (25 cells "
                                                             f"per radius) standoff {100*so:+.1f} % (<15 %), stagnation "
                                                             f"pressure {100*ps:+.1f} % (<3 %)")


def check_backends():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "compare_backends.py")],
                       capture_output=True, text=True)
    return "FAIL" not in r.stdout and r.returncode == 0, "Warp kernels vs PyTorch reference: " + \
        (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:])


def check_carbuncle():
    import carbuncle as cb
    so, ps, asym, _ = cb.run(8.0)
    return abs(so) < 0.05 and abs(ps) < 0.03 and asym < 0.05, (f"Mach 8 cylinder: standoff {100*so:+.1f} %, "
                                                              f"stagnation pressure {100*ps:+.1f} %, shock asymmetry {asym:.3f} m")


def check_patch_turbulent():
    import patch_turbulent_plate as pt
    from rocketcfd import Solver
    from rocketcfd.patches import PatchGrid
    single = Solver(None, None, None, pt.FS, faces=pt.faces(0.0016), **pt.KW)
    single.run(2000, chunk=500, first_order_steps=200, steady=True, implicit=True, cfl_ramp=200)
    bf = pt.faces(0.0064)
    yf = np.round(np.arange(-0.0336, 0.0336 + 1e-9, 0.0016), 10)
    xf = bf[0][(bf[0] >= -0.1 - 1e-9) & (bf[0] <= 1.1 + 1e-9)]
    pg = PatchGrid(bf, [[xf, yf, bf[2]]], pt.FS, **pt.KW)
    pg.verbose = False
    pg.run(2000, chunk=500, first_order_steps=200, cfl_ramp=200)
    a, b = pt.drag(single), pt.drag(pg)
    return abs(b / a - 1) < 0.02, f"turbulent plate with a patch: friction drag {b:.5f} vs single grid {a:.5f}"


def check_split_residual():
    import split_residual as sr
    worst = max(sr.compare(kw) for _, kw in sr.cases())
    return worst < 1e-4, f"split face/divergence residual vs fused kernel: worst relative difference {worst:.1e} (<1e-4)"


CHECKS = dict(backends=check_backends, split_residual=check_split_residual, vortex=check_vortex, shocks=check_shocks, low_mach=check_low_mach,
              patch_blasius=check_patch_blasius, patch_sphere=check_patch_sphere, carbuncle=check_carbuncle,
              patch_turbulent=check_patch_turbulent)

if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    results = []
    for n in names:
        t0 = time.time()
        try:
            ok, msg = CHECKS[n]()
        except Exception as e:                       # noqa: BLE001
            ok, msg = False, f"error: {type(e).__name__}: {e}"
        results.append((n, ok, msg, time.time() - t0))
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: {msg}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"\n{sum(r[1] for r in results)}/{len(results)} checks passed")
