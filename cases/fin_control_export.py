"""Fin control derivatives for CatchSim from deflected aero sweeps.

Run the undeflected sweep and one sweep per fin with only that fin deflected (same Mach,
aoa and roll lists; roll 0 or 180 if the catch-fin-270 derivatives are to come from mirror
symmetry), then combine:

  python cases/aero_database.py --aoa 0 8 --roll 0 --out results/db_d0
  python cases/aero_database.py --aoa 0 8 --roll 0 --fin-deflection 10 0 0 --out results/db_rudder10
  python cases/aero_database.py --aoa 0 8 --roll 0 --fin-deflection 0 10 0 --out results/db_catch10
  python cases/fin_control_export.py --base results/db_d0 --rudder results/db_rudder10:10 \\
      --catch90 results/db_catch10:10 --out results/fin_control.json

integrations/catchsim/aero_cfd.py: CFDFinControl reads the result.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rocketcfd.catchsim_export import build_fin_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="undeflected sweep directory")
    ap.add_argument("--rudder", help="dir:deflection_deg")
    ap.add_argument("--catch90", help="dir:deflection_deg")
    ap.add_argument("--catch270", help="dir:deflection_deg (else from mirror symmetry of catch90)")
    ap.add_argument("--out", default="results/fin_control.json")
    a = ap.parse_args()
    runs = {}
    for name, v in (("rudder fin", a.rudder), ("catch fin 90", a.catch90), ("catch fin 270", a.catch270)):
        if v:
            d, deg = v.rsplit(":", 1)
            runs[name] = (os.path.join(d, "cases.csv"), float(deg))
    out = build_fin_control(os.path.join(a.base, "cases.csv"), runs)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["fins"].items():
        print(f"{k}: {len(v)} conditions")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
