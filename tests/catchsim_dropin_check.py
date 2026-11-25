"""Consistency check of the CatchSim drop-in (integrations/catchsim/aero_cfd.py) against the
sweep that produced its table: for every computed case, rebuild the relative wind in body axes,
ask the drop-in for force and moment, and compare with the case's body-axis coefficients.
Also checks the mirror rule (roll r vs 360 - r) and moment transfer to another centre of mass.

  python tests/catchsim_dropin_check.py results/aero_db
"""
import sys, os, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "integrations", "catchsim"))
import numpy as np
import aero_cfd as A

out = sys.argv[1] if len(sys.argv) > 1 else "results/aero_db"
db = A.CFDAeroDatabase(f"{out}/aero_db.json")
worst = 0.0
for r in csv.DictReader(open(f"{out}/cases.csv")):
    m, al, roll = float(r["mach"]), float(r["aoa"]), float(r["roll"])
    tw = np.radians(90.0 + roll)
    a = np.radians(al)
    wind = np.array([np.cos(a), -np.sin(a) * np.cos(tw), -np.sin(a) * np.sin(tw)])   # as in the sweep
    F, M = A.body_loads(db, wind, m, 1.0)
    F /= db.S
    M /= db.S * db.L
    ref = np.array([float(r[k]) for k in ("CX", "CY", "CZ", "Cl", "Cm", "Cn")])
    err = np.abs(np.concatenate([F, M]) - ref).max()
    worst = max(worst, err)
    print(f"M {m} aoa {al:4.1f} roll {roll:5.1f}: max |drop-in - CFD| = {err:.2e}")
# mirror: roll r and 360 - r give mirrored loads (y -> -y): F_y, M_x, M_z change sign
for roll in (30.0, 90.0, 150.0):
    tw1, tw2 = np.radians(90.0 + roll), np.radians(90.0 + 360.0 - roll)
    a = np.radians(8.0)
    w1 = np.array([np.cos(a), -np.sin(a) * np.cos(tw1), -np.sin(a) * np.sin(tw1)])
    w2 = np.array([np.cos(a), -np.sin(a) * np.cos(tw2), -np.sin(a) * np.sin(tw2)])
    F1, M1 = A.body_loads(db, w1, db.mach[0], 1.0)
    F2, M2 = A.body_loads(db, w2, db.mach[0], 1.0)
    e = max(np.abs(F2 - F1 * [1, -1, 1]).max(), np.abs(M2 - M1 * [-1, 1, -1]).max())
    print(f"mirror roll {roll} / {360 - roll}: {e:.2e}")
    worst = max(worst, e / (db.S * db.L))
# moment transfer
F, M0 = A.body_loads(db, [np.cos(0.1), 0.0, -np.sin(0.1)], db.mach[0], 1.0)
_, M1 = A.body_loads(db, [np.cos(0.1), 0.0, -np.sin(0.1)], db.mach[0], 1.0, x_cg=db.x_ref + 10.0)
print(f"moment transfer check: {np.abs(M1 - (M0 + np.cross([-10.0, 0, 0], F))).max():.2e}")
print("PASS" if worst < 1e-6 else f"FAIL (worst {worst:.2e})")
