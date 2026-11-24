"""Aero database sweep: one case per GPU, in parallel.

Runs super_heavy_descent.py over a Mach x angle-of-attack matrix (default: points
along CatchSim's nominal descent) with CUDA_VISIBLE_DEVICES pinned per process,
then gathers every loads.json into one CSV that quatsim/aero.py can interpolate.

  python cases/aero_sweep.py --gpus 12 --h 0.1 --fins cad_filled
"""
import sys, os, json, argparse, subprocess, itertools, time, csv

HERE = os.path.dirname(os.path.abspath(__file__))

# (altitude m, Mach) samples along CatchSim's nominal coast and landing-burn approach
TRAJECTORY = [(47000, 3.95), (28700, 4.18), (20000, 3.8), (15000, 3.3), (10745, 2.82),
              (7000, 2.0), (5000, 1.5), (3500, 1.2), (2700, 1.08), (2000, 0.9)]
AOAS = [0.0, 5.0, 10.0, 20.0, 30.0]
# which side meets the crossflow (dimensions.md azimuth): rudder side, catch fin, raceway side
WINDWARD = [0.0, 90.0, 180.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--fins", default="cad_filled")
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cases = [(alt, m, aoa, ww) for (alt, m), aoa, ww in itertools.product(TRAJECTORY, AOAS, WINDWARD)
             if not (aoa == 0.0 and ww != 0.0)]
    os.makedirs(a.out, exist_ok=True)
    queue = list(cases)
    running = {}  # gpu -> (proc, case)
    while queue or running:
        for gpu in range(a.gpus):
            if gpu not in running and queue:
                alt, m, aoa, ww = queue.pop(0)
                out = f"{a.out}/alt{alt/1e3:.1f}_M{m:.2f}_a{aoa:.0f}_w{ww:.0f}"
                if os.path.exists(f"{out}/loads.json"):
                    continue                      # already done: the sweep can be resumed
                cmd = [sys.executable, f"{HERE}/super_heavy_descent.py", "--alt", str(alt), "--mach", str(m),
                       "--aoa", str(aoa), "--windward", str(ww), "--h", str(a.h), "--fins", a.fins,
                       "--out", out]
                print(f"[gpu {gpu}] {' '.join(cmd[2:])}", flush=True)
                if a.dry_run:
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false")
                os.makedirs(out, exist_ok=True)
                log = open(f"{out}/log.txt", "w")
                running[gpu] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out)
        if a.dry_run:
            break
        for gpu, (proc, out) in list(running.items()):
            if proc.poll() is not None:
                print(f"[gpu {gpu}] finished {out} (exit {proc.returncode})", flush=True)
                del running[gpu]
        time.sleep(5)

    rows = []
    for d in sorted(os.listdir(a.out)):
        fpath = f"{a.out}/{d}/loads.json"
        if os.path.exists(fpath):
            r = json.load(open(fpath))
            c, e = r["condition"], r["catchsim_equivalent"]
            co = r["coefficients"]
            rows.append(dict(alt=c["alt"], mach=c["mach"], aoa=c["aoa_from_engines_first"],
                             windward=r.get("condition_windward"), CD=co["CD"], CL=co["CL"], CN=co["CN"],
                             CS=co["CS"], C_restore=co["C_restore"], C_roll=co["C_roll"],
                             cn_alpha=e["cn_alpha"], cm_alpha=e["cm_alpha"]))
    if rows:
        with open(f"{a.out}/aero_database.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}/aero_database.csv ({len(rows)} cases)")


if __name__ == "__main__":
    main()
