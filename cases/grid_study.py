"""Grid-convergence study for the Super Heavy descent case.

Runs super_heavy_descent.py at a sequence of near-body cell sizes, then estimates the
grid-converged value of each coefficient (Richardson extrapolation on the three finest
grids) and its uncertainty (grid convergence index, GCI, safety factor 1.25).

  py cases\\grid_study.py                        (body only, 0.5 -> 0.125 m)
  py cases\\grid_study.py --h 0.4 0.28 0.2 0.14 --fins cad_filled
"""
import sys, os, json, argparse, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = (("CD", "drag CD"), ("CN", "normal force CN"), ("C_restore", "restoring moment"))


def richardson(hs, fs):
    """Observed order, extrapolated value and GCI from the three finest grids."""
    (h1, f1), (h2, f2), (h3, f3) = sorted(zip(hs, fs))[:3]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return None, None, None          # oscillatory or exact: no clean estimate
    # fixed-point iteration for the observed order with non-constant refinement ratio
    p = 2.0
    for _ in range(50):
        q = np.log((r21 ** p - np.sign(e32 / e21)) / (r32 ** p - np.sign(e32 / e21)))
        p = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
    p = float(np.clip(p, 0.5, 4.0))
    f_ext = f1 + (f1 - f2) / (r21 ** p - 1)
    gci = 1.25 * abs(e21) / (r21 ** p - 1)
    return p, f_ext, gci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, nargs="+", default=[0.5, 0.35, 0.25, 0.18, 0.125])
    ap.add_argument("--fins", default="none")
    ap.add_argument("--alt", type=float, default=10745.0)
    ap.add_argument("--mach", type=float, default=2.82)
    ap.add_argument("--aoa", type=float, default=5.0)
    ap.add_argument("--windward", type=float, default=180.0)
    ap.add_argument("--out", default="results/grid_study")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for h in sorted(a.h, reverse=True):
        out = f"{a.out}/h{h:g}_fins-{a.fins}"
        if not a.plot_only and not os.path.exists(f"{out}/loads.json"):
            cmd = [sys.executable, f"{HERE}/super_heavy_descent.py", "--h", str(h), "--fins", a.fins,
                   "--alt", str(a.alt), "--mach", str(a.mach), "--aoa", str(a.aoa),
                   "--windward", str(a.windward), "--out", out]
            print(f"=== h = {h} m", flush=True)
            t0 = time.time()
            subprocess.run(cmd, check=True)
            print(f"=== h = {h} m done in {(time.time()-t0)/60:.1f} min\n", flush=True)
        if os.path.exists(f"{out}/loads.json"):
            r = json.load(open(f"{out}/loads.json"))
            rows.append(dict(h=h, cells=r["grid"]["cells"], wall_s=r["grid"]["wall_s"],
                             **{k: r["coefficients"][k] for k, _ in KEYS}))

    if len(rows) < 2:
        print("need at least two finished grids")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array([r["h"] for r in rows])
    summary = {}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (k, label) in zip(axs, KEYS):
        fs = np.array([r[k] for r in rows])
        ax.plot(hs, fs, "o-", color="C0", label="CFD")
        p, f_ext, gci = richardson(hs, fs) if len(rows) >= 3 else (None, None, None)
        if f_ext is not None:
            ax.axhline(f_ext, color="C3", ls="--", lw=1, label=f"extrapolated {f_ext:.4f} (order {p:.2f})")
            ax.axhspan(f_ext - gci, f_ext + gci, color="C3", alpha=0.12, label=f"GCI ±{gci:.4f}")
        summary[k] = dict(values=fs.tolist(), extrapolated=f_ext, order=p, gci=gci)
        ax.set_xlabel("near-body cell size h [m]")
        ax.set_title(label)
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.suptitle(f"Grid convergence: Super Heavy V3, M {a.mach}, {a.alt/1e3:.1f} km, "
                 f"{a.aoa} deg off engines-first, fins={a.fins}")
    fig.savefig(f"{a.out}/grid_study_fins-{a.fins}.png", dpi=130)
    json.dump(dict(rows=rows, summary=summary), open(f"{a.out}/grid_study_fins-{a.fins}.json", "w"),
              indent=2, default=float)

    print(f"\n{'h [m]':>7} {'cells':>8} {'time':>7} " + " ".join(f"{k:>10}" for k, _ in KEYS))
    for r in rows:
        print(f"{r['h']:7.3f} {r['cells']/1e6:7.2f}M {r['wall_s']/60:6.1f}m " +
              " ".join(f"{r[k]:10.4f}" for k, _ in KEYS))
    for k, _ in KEYS:
        sm = summary[k]
        if sm["extrapolated"] is not None:
            print(f"{k:>10}: grid-converged {sm['extrapolated']:.4f} ± {sm['gci']:.4f} (observed order {sm['order']:.2f})")
        else:
            print(f"{k:>10}: not yet in the asymptotic range (values not monotonic); add a finer grid")


if __name__ == "__main__":
    main()
