"""Engine-jet boundary condition check: a sonic, underexpanded jet into still air. The first
Mach disk sits at x / D = 0.67 sqrt(p0 / pa) (Crist, Sherman & Glass, AIAA J. 4, 1966; also
Ashkenas & Sherman), p0 = jet stagnation pressure, pa = ambient pressure.

  python tests/underexpanded_jet.py              # p_exit / p_ambient = 4, D = 1 m, 12 cells across
  python tests/underexpanded_jet.py --h 0.05 --ratio 6
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver

G = 1.4


def run(h=0.08, ratio=4.0, steps=3000, out=None, verbose=True):
    pa, Ta, R = 1e5, 288.0, 287.05
    fs = dict(rho=pa / (R * Ta), u=0.0, v=0.0, w=0.0, p=pa)
    D = 1.0
    pe = ratio * pa
    Te = 300.0
    p0 = pe * (1 + 0.5 * (G - 1)) ** (G / (G - 1))
    # nozzle: a thick tube ending at x = 0, bore radius D/2 (the exit), outer radius 0.75
    def nozzle(X, Y, Z):
        r = np.sqrt(Y * Y + Z * Z)
        tube = np.maximum(np.maximum(r - 0.75, -3.0 - X), X)
        return tube
    L = 6.5
    s = Solver((-1.6, -2.6, -2.6), (L, 2.6, 2.6), h, fs, sdf=nozzle, verbose=False, cfl=0.6,
               jets=[dict(center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), radius=0.5 * D, mach=1.0, p=pe, T=Te)])
    if verbose:
        print(f"{np.prod(s.n)/1e6:.2f} M cells, {s.jet_cells} jet cells, p0/pa = {p0/pa:.2f}", flush=True)
    s.run(steps, chunk=500, first_order_steps=300, steady=True,
          report=(lambda sv: print(f"  step {sv.step_count}", flush=True) or False) if verbose else None)
    f = s.fields()
    x = f["x"]
    j = np.argmin(np.abs(f["y"]))
    k = np.argmin(np.abs(f["z"]))
    m = f["mach"][:, j, k]
    # first Mach disk: the strongest drop in Mach along the axis downstream of the exit
    sel = x > 0.2
    dm = np.diff(m[sel])
    i = int(np.argmin(dm))
    xm = 0.5 * (x[sel][i] + x[sel][i + 1])
    ref = 0.67 * np.sqrt(p0 / pa) * D
    res = dict(h=h, ratio=ratio, p0_over_pa=p0 / pa, x_mach_disk=float(xm), x_ref=float(ref),
               err=float(xm / ref - 1), mach_max=float(m[sel].max()))
    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6.5))
        Xg, Yg = np.meshgrid(x, f["y"], indexing="ij")
        cs = ax[0].contourf(Xg, Yg, f["mach"][:, :, k], 40, cmap="turbo")
        ax[0].axvline(xm, color="w", ls="--", lw=1)
        ax[0].axvline(ref, color="k", ls=":", lw=1.5)
        ax[0].set_aspect("equal")
        fig.colorbar(cs, ax=ax[0], label="Mach")
        ax[0].set_title(f"Sonic jet, p_e/p_a = {ratio:g}: Mach disk at x/D = {xm:.2f} (empirical {ref:.2f})")
        ax[1].plot(x, m, "k-")
        ax[1].axvline(ref, color="C3", ls=":", label="0.67 sqrt(p0/pa)")
        ax[1].set_xlabel("x / D")
        ax[1].set_ylabel("Mach on the axis")
        ax[1].legend()
        fig.tight_layout()
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f"underexpanded_jet_r{ratio:g}_h{h:g}.png"), dpi=120)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.08)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="results/jet")
    a = ap.parse_args()
    r = run(a.h, a.ratio, a.steps, a.out)
    print(json.dumps(r, indent=1))
    print(f"Mach disk x/D = {r['x_mach_disk']:.2f}, empirical {r['x_ref']:.2f} ({100*r['err']:+.1f} %)")
