"""Steady-state accelerations must not change the converged answer.

Runs the same 3-D sphere case (Mach 3, stretched grid) three ways until the loads settle:
  baseline   global time step, every cell every step
  local      per-cell time steps
  tiles       global time step, active-tile updating
  local+tiles per-cell time steps and active-tile updating
  implicit[:CFL]  backward-Euler DP-LUR pseudo-time stepping (work counts each implicit
              iteration as one step; one costs about 2 explicit steps of wall time)
and compares the converged loads and the work (cell updates, in full-step units).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import Zone, stretched_faces


def run(mode, faces, tol=2e-4, max_steps=6000):
    s = Solver(None, None, None, dict(rho=1.4, u=3.0, v=0.0, w=0.3, p=1.0), faces=faces,
               sdf=lambda X, Y, Z: np.sqrt(X**2 + Y**2 + Z**2) - 0.4, verbose=False, backend="warp")
    ref = dict(S_ref=np.pi * 0.16, L_ref=0.8, x_ref=(0.0, 0.0, 0.0))
    hist = []

    def report(sv):
        ld = sv.loads(**ref)
        hist.append((ld["CD"], ld["CN"]))
        if len(hist) >= 4:
            r = np.array(hist[-4:])
            return bool(np.all(np.abs(r - r[-1]).max(0) < tol))
        return False

    t0 = time.time()
    if mode.startswith("implicit"):
        cfl_max = float(mode.split(":")[1]) if ":" in mode else 20.0
        s.run(max_steps, chunk=20, first_order_steps=100, report=report, steady=True, implicit=True,
              cfl_max=cfl_max, cfl_ramp=100)
    else:
        s.run(max_steps, chunk=100, first_order_steps=200, report=report,
              steady=(mode != "baseline"), active_tiles=("tiles" in mode), local_dt=("local" in mode))
    return dict(steps=s.step_count, work=s.work_full_steps, wall=time.time() - t0,
                CD=hist[-1][0], CN=hist[-1][1], cells=int(np.prod(s.n)))


if __name__ == "__main__":
    faces = stretched_faces((-1.6, -1.6, -1.6), (2.4, 1.6, 1.6), 0.16,
                            [Zone((-0.55, -0.55, -0.55), (0.6, 0.55, 0.55), 0.04)], growth=1.12)
    modes = sys.argv[1:] or ["baseline", "tiles", "local", "local+tiles"]
    out = {m: run(m, faces) for m in modes}
    b = out.get("baseline", next(iter(out.values())))
    print(f"{'mode':12s} {'steps':>6s} {'work':>7s} {'wall s':>7s} {'CD':>8s} {'CN':>8s}  vs baseline")
    for m, r in out.items():
        print(f"{m:12s} {r['steps']:6d} {r['work']:7.0f} {r['wall']:7.1f} {r['CD']:8.4f} {r['CN']:8.4f}  "
              f"work x{b['work']/r['work']:.2f}, dCD {r['CD']-b['CD']:+.4f}, dCN {r['CN']-b['CN']:+.4f}")
    print(f"cells {b['cells']}")
