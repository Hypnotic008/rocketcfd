"""Turn an aero sweep (cases/aero_database.py -> cases.csv) into the table CatchSim reads.

Coefficients are stored in the crossflow frame of each case, which varies smoothly with
roll angle and so interpolates well (body-axis components rotate with the crossflow and
do not):

    ex     body axis, from the engines toward the top (CatchSim's nose axis)
    cross  unit crossflow direction: the part of the relative wind normal to ex, pointing
           from the windward to the leeward side
    side   ex x cross

    force  / (q S) = CX ex + CN cross + CS side
    moment / (q S L) about x_ref = Cl ex + C_cross cross + C_restore side

C_restore > 0 turns the engines back into the wind (engines-first is then stable). CD is the
drag along the relative wind. Fins: C_hinge[name] = hinge moment / (q S_fin c_fin) about each
fin's outward radial axis.

The V3 layout is mirror-symmetric about the plane through the rudder fin and the raceway
(y -> -y in rocketcfd axes), so cases cover roll 0-180 deg and roll r > 180 maps to 360 - r with
CS, Cl, C_cross and the hinge moments changing sign and the two catch fins swapping.
"""
import csv
import numpy as np

TABLE_KEYS = ("CX", "CN", "CS", "Cl", "C_cross", "C_restore", "CD")
MIRROR_SIGN = dict(CX=1, CN=1, CS=-1, Cl=-1, C_cross=-1, C_restore=1, CD=1)


def crossflow_frame(roll_deg):
    """cross and side unit vectors (rocketcfd body axes) for a windward azimuth `roll_deg`
    (dimensions.md convention, as in cases/super_heavy_descent.py)."""
    tw = np.radians(90.0 + roll_deg)
    cross = np.array([0.0, -np.cos(tw), -np.sin(tw)])
    side = np.cross([1.0, 0.0, 0.0], cross)
    return cross, side


def frame_coefficients(row, roll_deg):
    """Crossflow-frame coefficients of one case from its body-axis coefficients."""
    F = np.array([row["CX"], row["CY"], row["CZ"]])
    M = np.array([row["Cl"], row["Cm"], row["Cn"]])
    cross, side = crossflow_frame(roll_deg)
    return dict(CX=F[0], CN=F @ cross, CS=F @ side, Cl=M[0], C_cross=M @ cross, C_restore=M @ side,
                CD=row["CD"])


def build_database(csv_path, reference, fin_names=("rudder fin", "catch fin 90", "catch fin 270")):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) and k != "converged" else v) for k, v in r.items()})
    mach = sorted({r["mach"] for r in rows})
    aoa = sorted({r["aoa"] for r in rows})
    roll = sorted({r["roll"] for r in rows if r["aoa"] > 0} or {0.0})
    shape = (len(mach), len(aoa), len(roll))
    hinge_keys = [f"C_hinge[{n}]" for n in fin_names]
    tab = {k: np.full(shape, np.nan) for k in TABLE_KEYS + tuple(hinge_keys)}
    conv = np.zeros(shape, bool)
    for r in rows:
        i, j = mach.index(r["mach"]), aoa.index(r["aoa"])
        rolls = roll if r["aoa"] == 0.0 else [r["roll"]]        # zero incidence: same flow at every roll
        for rr in rolls:
            if rr not in roll:
                continue
            k = roll.index(rr)
            fc = frame_coefficients(r, rr)
            for key in TABLE_KEYS:
                tab[key][i, j, k] = fc[key]
            for hk in hinge_keys:
                if hk in r and r[hk] not in ("", None):
                    tab[hk][i, j, k] = r[hk]
            conv[i, j, k] = str(r.get("converged", "True")) in ("True", "1", "1.0")
    # fill holes along the angle-of-attack axis (nearest computed incidence at that Mach / roll)
    for key, T in tab.items():
        for i in range(shape[0]):
            for k in range(shape[2]):
                col = T[i, :, k]
                ok = np.isfinite(col)
                if ok.any() and not ok.all():
                    T[i, :, k] = np.interp(aoa, np.array(aoa)[ok], col[ok])
    return dict(
        description="Super Heavy V3 aerodynamic coefficients from rocketcfd (see rocketcfd/catchsim_export.py "
                    "for axes and signs). Inviscid unless noted.",
        reference=dict(S_ref=reference["S_ref"], L_ref=reference["L_ref"], x_ref=list(reference["x_ref"]),
                       x_from="aft end (engine plane), toward the top"),
        fins=dict(names=list(fin_names), S_fin=11.26, c_fin=0.5,
                  mirror_swap={"catch fin 90": "catch fin 270", "catch fin 270": "catch fin 90"}),
        mach=mach, aoa_deg=aoa, roll_deg=roll,
        tables={k: np.where(np.isfinite(v), v, None).tolist() for k, v in tab.items()},
        converged=conv.tolist(),
        mirror_sign=MIRROR_SIGN,
    )
