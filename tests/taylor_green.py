"""Taylor-Green vortex at Re = 1600, Mach 0.1 (the standard LES / high-order benchmark, e.g.
the International Workshop on High-Order CFD Methods; DNS: Brachet et al. 1983, van Rees et
al. 2011). The vortex breaks down into turbulence; the kinetic-energy dissipation rate
eps = -dEk/dt peaks at ~0.0127 near t = 9 (DNS, 512^3 spectral).

On a coarse grid the scheme's own dissipation is part of eps; a good LES setup gets the
peak height and time close to the DNS. Compares implicit LES (no model) with the Vreman
subgrid model.

  python tests/taylor_green.py              # N = 32 (CPU: a few minutes)
  python tests/taylor_green.py 64 --t 12    # GPU
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver

G = 1.4
RE = 1600.0
MACH = 0.1
# DNS reference points: the dissipation peak (van Rees et al. 2011, Brachet et al. 1983) and the
# exact laminar initial rate eps(0) = nu <du_i/dx_j du_i/dx_j> = 0.75 / Re
DNS_PEAK = (9.0, 0.0127)
EPS0 = 0.75 / RE


def run(n, sgs, t_end=10.0, cfl=0.6, verbose=True, limiter="koren"):
    L = np.pi
    h = 2 * L / n
    p0 = 1.0 / (G * MACH ** 2)
    fs = dict(rho=1.0, u=0.0, v=0.0, w=0.0, p=p0)
    turb = dict(model="les", sgs=sgs, wall_model=False)
    s = Solver((-L, -L, -L), (L, L, L), h, fs, bcs={(d, k): "periodic" for d in range(3) for k in (0, 1)},
               cfl=cfl, verbose=False, viscous=dict(model="constant", mu=1.0 / RE), turbulence=turb,
               limiter=limiter)
    X, Y, Z = np.meshgrid(*s.axes, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    p = p0 + (np.cos(2 * X) + np.cos(2 * Y)) * (np.cos(2 * Z) + 2.0) / 16.0
    rho = p / p0                                   # isothermal start
    s.set_primitive(np.stack([rho, u, v, 0 * u, p]))
    ts, ek = [], []

    def energy():
        f = s.fields()
        return 0.5 * float(np.mean(f["rho"] * (f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2)))

    ts.append(0.0)
    ek.append(energy())
    nxt = 0.25
    while s.t < t_end - 1e-9:
        s.run(10, chunk=10, first_order_steps=0)
        if s.t >= nxt - 1e-9:
            ts.append(s.t)
            ek.append(energy())
            nxt += 0.25
            if verbose and abs(s.t - round(s.t)) < 0.13:
                print(f"  [{sgs:7s} N={n}] t {s.t:5.2f}  Ek {ek[-1]:.5f}", flush=True)
    ts, ek = np.array(ts), np.array(ek)
    eps = -np.gradient(ek, ts)
    return ts, ek, eps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=32)
    ap.add_argument("--t", type=float, default=10.0)
    ap.add_argument("--out", default="results/taylor_green")
    ap.add_argument("--limiter", default="koren", help="reconstruction: koren (MUSCL) or weno5z")
    ap.add_argument("--sgs", nargs="+", default=["none", "vreman"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for sgs in a.sgs:
        ts, ek, eps = run(a.n, sgs, a.t, limiter=a.limiter)
        i = int(np.argmax(eps))
        res[sgs] = dict(t=ts.tolist(), Ek=ek.tolist(), eps=eps.tolist(), peak=float(eps[i]), t_peak=float(ts[i]))
        print(f"{sgs:7s} N={a.n}: peak dissipation {eps[i]:.5f} at t = {ts[i]:.2f}  (DNS 0.0127 at ~9; exact eps(0) {EPS0:.5f}, here {eps[0]:.5f})", flush=True)
    tag = "" if a.limiter == "koren" else f"_{a.limiter}"
    with open(f"{a.out}/tgv_N{a.n}{tag}.json", "w") as fh:
        json.dump(res, fh)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*DNS_PEAK, "k*", ms=14, label="DNS peak (van Rees et al. 2011)")
    ax.plot(0, EPS0, "ko", label="exact initial rate")
    for sgs, st in (("none", "C0--"), ("vreman", "C1-")):
        if sgs not in res:
            continue
        ax.plot(res[sgs]["t"], res[sgs]["eps"], st, label=f"N={a.n}, " + ("implicit LES" if sgs == "none" else "Vreman"))
    ax.set_xlabel("t"); ax.set_ylabel("-dEk/dt"); ax.set_title(f"Taylor-Green vortex, Re 1600, M 0.1, {a.n}^3")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/tgv_N{a.n}{tag}.png", dpi=130)
    print(f"wrote {a.out}/tgv_N{a.n}{tag}.png")
