"""Super Heavy V3 (Block 3) outer mold line, unpowered, for aerodynamics.

Dimensions follow the photo-measured V3 notes in spacecraft-models/starship
(dimensions.md, "Super Heavy V3 in detail"); tags there say which numbers SpaceX
published and which are estimated from photos. None of this is SpaceX data, and every
number is a parameter.

Body frame
  +x along the vehicle axis, x = 0 at the OUTER-ring nozzle exit plane (the datum of
  dimensions.md), x = 72.3 m at the top of the hot-staging truss.
  Azimuth here = 90 deg + azimuth in dimensions.md, so the rudder grid fin (their 0 deg)
  is on +z and the two catch fins (their 90 / 270 deg) are on -y / +y. This matches
  CatchSim's renderer (third fin on body +Z).

Modelled: hull, tapered thrust plate, 33 exposed Raptor 3 bells (outer ring overhangs
the hull), 4 chines, raceway, 3 grid fins from CatchSim's CAD (scaled per axis),
fin pods, forward dome inside the hot-staging truss.
Not modelled (too small for the grids we run, or unknown): truss struts, conduit band,
engine valve housings, catch hardpoints, fin teeth details beyond the CAD.
"""
from dataclasses import dataclass
import os
import numpy as np
from .. import geometry as g

DEFAULT_FIN_CAD = os.environ.get(
    "ROCKETCFD_FIN_CAD", os.path.join(os.path.dirname(__file__), "..", "data", "gridfin.i16"))
_FIN_CACHE = {}


def load_fin_cad(path, res, filled):
    """Grid fin from CatchSim's baked STEP mesh (int16 millimetres, non-indexed triangles).
    CAD axes: x along the body axis (centred on the lattice depth, 0.75 m), y around the
    hull (6 m), z radial with the root adapter at z = 0 and the tip at 7 m."""
    key = (os.path.abspath(path), res, filled)
    if key not in _FIN_CACHE:
        tris = np.fromfile(path, dtype="<i2").reshape(-1, 3, 3).astype(float) * 0.001
        sdf = g.MeshSDF(tris, res=res)
        _FIN_CACHE[key] = sdf.filled() if filled else sdf
    return _FIN_CACHE[key]


@dataclass
class SuperHeavyParams:
    radius: float = 4.5
    length: float = 72.3                  # outer nozzle exits to top of hot-staging truss
    # aft end
    skirt_bottom_x: float = 2.75          # skirt edge = outer edge of the thrust plate
    plate_center_x: float = 2.5           # plate tapers down toward the middle
    # Raptor 3 bells: 1.3 m exits, ~2.9 m from exit to top
    bell_exit_radius: float = 0.65
    bell_top_radius: float = 0.32
    bell_length: float = 2.0
    powerhead_radius: float = 0.40
    # (count, pitch radius, exit x, azimuths in dimensions.md convention [deg])
    centre: tuple = (3, 1.1, -0.5, (0.0, 108.0, 252.0))
    inner: tuple = (10, 2.5, -0.2, None)  # 18 + 36 k
    outer: tuple = (20, 4.4, 0.0, None)   # 9 + 18 k
    # chines: (azimuth deg in dimensions.md convention, x_bottom, x_top)
    chines: tuple = ((160.0, 4.95, 36.8), (200.0, 4.95, 36.8),     # taller pair, raceway side
                     (59.0, 4.95, 28.8), (301.0, 4.95, 28.8))      # shorter pair, rudder side
    chine_standoff: float = 1.35          # outer face 5.85 m from the axis
    chine_width: float = 0.7              # NOT measured: estimate
    raceway: tuple = (178.0, 36.0, 68.0, 0.7, 0.3)  # azimuth, x0, x1, width, height (estimate)
    # hot-staging truss: open struts (not modelled) around the forward dome
    tank_top_x: float = 68.3
    dome_rise: float = 2.9                # dome top ~71.2 m
    # grid fins (dimensions.md: 3.5 m wide, ~4.2 m reach from the hull, ~0.5 m deep,
    # mid-depth at x ~ 65.35 m). Azimuths in dimensions.md convention: rudder 0, catch 90/270.
    fin_azimuths: tuple = (0.0, 90.0, 270.0)
    fin_x_center: float = 65.35
    fin_root_x_offset: float = -0.05      # CAD root sits just inside the hull
    fin_model: str = "cad"                # 'cad' | 'cad_filled' | 'none'
    fin_cad_path: str = DEFAULT_FIN_CAD
    # CAD is 0.75 (axial) x 6.0 (around) x 7.0 (radial) m; scale to 0.5 x 3.5 x 4.2 m
    fin_scale: tuple = (0.5 / 0.75, 3.5 / 6.0, 4.2 / 7.0)
    fin_cad_res: float = 0.02
    fin_deflection_deg: tuple = (0.0, 0.0, 0.0)   # rotation about each fin's radial axis
    fin_pod_radius: float = 0.6           # round bearing pod on the tank wall
    fin_pod_height: float = 0.25


def _az(their_deg):
    """dimensions.md azimuth -> body-frame angle (rudder on +z)."""
    return np.radians(90.0 + their_deg)


def _ring_azimuths(group, kind):
    count, _, _, az = group
    if az is not None:
        return np.array(az, float)
    start = {"inner": 18.0, "outer": 9.0}[kind]
    return start + (360.0 / count) * np.arange(count)


def engine_positions(P: SuperHeavyParams):
    """(y, z, exit_x) of all 33 nozzle centres."""
    out = []
    for kind, grp in (("centre", P.centre), ("inner", P.inner), ("outer", P.outer)):
        for a in _ring_azimuths(grp, kind):
            th = _az(a)
            out.append((grp[1] * np.cos(th), grp[1] * np.sin(th), grp[2]))
    return np.array(out)


def super_heavy_sdf(X, Y, Z, P: SuperHeavyParams = SuperHeavyParams()):
    R = P.radius
    rho = np.sqrt(Y * Y + Z * Z)
    parts = []
    # hull: skirt edge to top of the tank wall
    parts.append(g.cylinder_x(X, Y, Z, R, P.skirt_bottom_x, P.tank_top_x))
    # tapered thrust plate: lowest at the centre, meets the skirt edge at the hull
    slope = (P.skirt_bottom_x - P.plate_center_x) / R
    plate = (P.plate_center_x + slope * rho - X) / np.sqrt(1 + slope ** 2)
    parts.append(g.intersect(plate, g.cylinder_x(X, Y, Z, R, P.plate_center_x - 0.5, P.skirt_bottom_x + 0.5)))
    # 33 Raptor 3 bells + powerheads up to the plate
    for y, z, xe in engine_positions(P):
        xt = xe + P.bell_length
        parts.append(g.frustum_x(X, Y, Z, P.bell_exit_radius, P.bell_top_radius, xe, xt, y, z))
        parts.append(g.cylinder_x(X, Y, Z, P.powerhead_radius, xt - 0.05, P.skirt_bottom_x + 0.1, y, z))
    # chines
    for az, x0, x1 in P.chines:
        th = _az(az)
        rc = R + 0.5 * P.chine_standoff - 0.15
        parts.append(g.radial_box(X, Y, Z, th, 0.5 * (x0 + x1), rc, 0.5 * (x1 - x0),
                                  0.5 * P.chine_standoff + 0.15, 0.5 * P.chine_width))
    # raceway
    az, x0, x1, w, hgt = P.raceway
    parts.append(g.radial_box(X, Y, Z, _az(az), 0.5 * (x0 + x1), R + 0.5 * hgt - 0.1,
                              0.5 * (x1 - x0), 0.5 * hgt + 0.1, 0.5 * w))
    # forward dome inside the truss (half ellipsoid)
    a = P.dome_rise
    ell = (np.sqrt(((X - P.tank_top_x) / a) ** 2 + (rho / R) ** 2) - 1.0) * min(a, R)
    parts.append(g.intersect(ell, P.tank_top_x - 0.01 - X))
    # grid fins and pods
    if P.fin_model != "none":
        for k, az in enumerate(P.fin_azimuths):
            th = _az(az)
            parts.append(_grid_fin_cad(X, Y, Z, P, th, np.radians(P.fin_deflection_deg[k])))
            r, t = g.radial_frame(Y, Z, th)
            parts.append(g.box_local(X - P.fin_x_center, r - (R + 0.5 * P.fin_pod_height - 0.1), t,
                                     P.fin_pod_radius, 0.5 * P.fin_pod_height + 0.1, P.fin_pod_radius))
    return g.union(*parts)


def _grid_fin_cad(X, Y, Z, P, theta, delta):
    sdf = load_fin_cad(P.fin_cad_path, P.fin_cad_res, P.fin_model == "cad_filled")
    sx, sy, sz = P.fin_scale
    r, t = g.radial_frame(Y, Z, theta)
    a = X - P.fin_x_center
    cd, sd = np.cos(delta), np.sin(delta)
    a2, t2 = a * cd + t * sd, -a * sd + t * cd
    root = P.radius + P.fin_root_x_offset
    near = (np.abs(a) < 0.75 * sx + 1.0) & (r > root - 1.0) & (r < root + 7.0 * sz + 1.0) \
        & (np.abs(t) < 3.0 * sy + 1.0)
    out = np.full(np.shape(X), 1.0)
    # non-uniform scale: distance scaled by the smallest factor stays a valid lower bound
    out[near] = min(sx, sy, sz) * sdf(a2[near] / sx, t2[near] / sy, (r[near] - root) / sz)
    return out


def v3_params(**kw):
    return SuperHeavyParams(**kw)


# CatchSim reference quantities (quatsim/aero.py restoring_moment) so coefficients compare
# directly. x_ref = centre of mass at 40 % of length from the aft end (CatchSim viz3d COM_FRAC).
REFERENCE = dict(S_ref=63.6, L_ref=72.3, x_ref=(0.40 * 72.3, 0.0, 0.0))
