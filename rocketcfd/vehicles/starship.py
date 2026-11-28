"""Starship (upper stage) and the vented hot-staging interstage, for the hot-staging case.

Approximate public dimensions (not measured; good enough for plume / interstage flow pictures
and first-order loads): 9 m diameter, ~52 m tall, tangent-ogive nose, two aft and two forward
flaps, aft skirt around 3 sea-level Raptors (inner) and 3 vacuum Raptors (outer).

Frame: the booster's (rocketcfd.vehicles.super_heavy): +x along the stack from the booster's
engines (x = 0) toward the ship's nose. The ship's aft end sits on the hot-stage ring top.
"""
from dataclasses import dataclass

import numpy as np

from .. import geometry as g


@dataclass
class StarshipParams:
    radius: float = 4.5
    x_aft: float = 72.3                # ship aft end = top of the booster's hot-stage ring
    skirt_height: float = 2.2          # aft skirt (open at the bottom) around the engines
    barrel_length: float = 33.0        # aft end to the start of the nose
    nose_length: float = 19.0          # tangent ogive
    # engines: (count, pitch radius, exit radius, bell length, exit x above x_aft, azimuth offset deg)
    sea_level: tuple = (3, 1.05, 0.65, 2.0, 0.3, 0.0)
    vacuum: tuple = (3, 3.0, 1.15, 3.0, -0.3, 60.0)    # pitch radius 3.0: the bell rims clear the truss
    # flaps: (x0, x1 above x_aft, radial reach from the hull, thickness)
    aft_flaps: tuple = (2.5, 11.5, 3.6, 0.35)
    fwd_flaps: tuple = (35.0, 41.5, 2.4, 0.3)
    flap_azimuths: tuple = (90.0, 270.0)     # body-frame angles of the flap pairs (deg)
    # hot-stage ring (part of the booster): vented shell between the tank top and x_aft
    ring_x0: float = 68.3
    ring_bands: float = 0.5            # solid band height at the bottom and top of the ring
    ring_struts: int = 20
    ring_strut_width: float = 0.6
    ring_thickness: float = 0.2
    # V3 hot-stage truss (photos: a zig-zag of round tubes between two rings, the forward dome
    # inside it): `truss_nodes` tube joints on the lower ring, as many on the upper ring half a
    # pitch round, each lower joint the apex of a V. Estimates from pictures, not drawings.
    ring_style: str = "v3_truss"       # 'v3_truss' | 'vented' (the earlier straight-strut ring)
    truss_nodes: int = 20
    truss_tube_radius: float = 0.12
    truss_band: float = 0.35           # height of the lower and upper rings
    truss_band_thickness: float = 0.15
    # forward-dome protection plates (photos): six rounded plates in a flower round a
    # concentric-ring disc at the apex. Three inner plates start at the disc and widen outward
    # (rounded trapezoids) under the ship's sea-level engines; three outer plates fill the
    # wedge-shaped gaps between them, starting further out (rounded V's pointing inward, bent over
    # the dome), under the RVacs. Plan-view sizes [m] (estimates from photos): apex disc radius;
    # inner plate inner / outer radius and inner / outer width; outer plate inner / outer radius;
    # corner rounding radius; gap between plates; standoff above the dome
    plate_apex_r: float = 0.9
    plate_inner: tuple = (1.0, 3.1, 1.0, 2.5)
    plate_outer: tuple = (1.85, 3.7)
    plate_round: float = 0.5
    plate_gap: float = 0.12
    plate_standoff: float = 0.05
    apex_disc: bool = True
    # engine cant [deg, outward = the exhaust pointing away from the axis], about each engine's
    # gimbal point at the top of its bell; and the separation: the ship moved this far up off
    # the hot stage [m] (the truss stays with the booster)
    sl_cant: float = 0.0
    vac_cant: float = 0.0
    sep_gap: float = 0.0

    @property
    def length(self):
        return self.barrel_length + self.nose_length

    @property
    def x_ship_aft(self):
        """The ship's aft end (x_aft is the top of the booster's hot stage; they part by sep_gap)."""
        return self.x_aft + self.sep_gap

    @property
    def x_nose(self):
        return self.x_ship_aft + self.length


def engine_frames(P: StarshipParams):
    """The six ship engines: [dict(kind, exit=(x, y, z) exit centre, dir = exhaust direction,
    exit_r, top_r, length (bell, exit to gimbal point), pivot)]. Cant (P.sl_cant, P.vac_cant)
    swings each bell about its gimbal point at the top, outward for positive angles."""
    xs = P.x_ship_aft
    out = []
    for kind, (n, pr, re_, bl, dx, az0), cant in (("sea level", P.sea_level, P.sl_cant), ("vacuum", P.vacuum, P.vac_cant)):
        for k in range(n):
            a = np.radians(az0 + 360.0 * k / n)
            xe = xs + dx
            top_x = min(xe + bl, xs + P.skirt_height + 0.05)
            L = top_x - xe
            pivot = np.array([top_x, pr * np.cos(a), pr * np.sin(a)])
            c = np.radians(cant)
            d = np.array([-np.cos(c), np.sin(c) * np.cos(a), np.sin(c) * np.sin(a)])
            out.append(dict(kind=kind, exit=tuple(pivot + L * d), dir=tuple(d), exit_r=re_,
                            top_r=0.32 if kind == "sea level" else 0.45, length=L, pivot=tuple(pivot)))
    return out


def engines(P: StarshipParams):
    """[(kind, y, z, exit_x, exit_radius, bell_length)] of the six ship engines (exit centres)."""
    return [(e["kind"], e["exit"][1], e["exit"][2], e["exit"][0], e["exit_r"], P.sea_level[3] if e["kind"] == "sea level"
             else P.vacuum[3]) for e in engine_frames(P)]


def ogive_radius(xx, R, L):
    """Tangent ogive of base radius R and length L; xx measured from the base (0..L)."""
    rho = (R * R + L * L) / (2.0 * R)
    return np.sqrt(np.maximum(rho * rho - np.clip(xx, 0.0, L) ** 2, 0.0)) + R - rho


def starship_sdf(X, Y, Z, P: StarshipParams = StarshipParams()):
    R = P.radius
    rr = np.sqrt(Y * Y + Z * Z)
    xs = P.x_ship_aft
    parts = []
    # tank body from the aft bulkhead (top of the skirt) to the nose base
    parts.append(g.cylinder_x(X, Y, Z, R, xs + P.skirt_height, xs + P.barrel_length))
    # nose: tangent ogive (approximate distance: radial gap scaled by the local slope)
    xx = X - (xs + P.barrel_length)
    rn = ogive_radius(xx, R, P.nose_length)
    nose = (rr - rn) * 0.9
    nose = np.maximum(nose, -xx)                      # cut below the nose base
    nose = np.maximum(nose, xx - P.nose_length)      # and above the tip
    parts.append(nose)
    # aft skirt: open shell around the engines
    t = 0.15
    shell = np.abs(rr - (R - t / 2)) - t / 2
    parts.append(np.maximum(shell, np.abs(X - (xs + P.skirt_height / 2)) - P.skirt_height / 2))
    # engines: bells hanging from the aft bulkhead
    for e in engine_frames(P):
        ax_ = -np.asarray(e["dir"])                    # exit -> gimbal point
        if ax_[0] > 0.999999:
            parts.append(g.frustum_x(X, Y, Z, e["exit_r"], e["top_r"], e["exit"][0], e["exit"][0] + e["length"],
                                     e["exit"][1], e["exit"][2]))
        else:
            parts.append(g.frustum_axis(X, Y, Z, e["exit"], ax_, e["exit_r"], e["top_r"], e["length"]))
    # flaps (radial plates)
    for (x0, x1, reach, th), r_at in ((P.aft_flaps, None), (P.fwd_flaps, "nose")):
        for az in P.flap_azimuths:
            th_ = np.radians(az)
            xc = xs + 0.5 * (x0 + x1)
            rbase = R if r_at is None else float(ogive_radius(0.5 * (x0 + x1) - P.barrel_length, R, P.nose_length))
            parts.append(g.radial_box(X, Y, Z, th_, xc, rbase + 0.5 * reach - 0.2, 0.5 * (x1 - x0),
                                      0.5 * reach + 0.2, 0.5 * th))
    return g.union(*parts)


def truss_tubes(P: StarshipParams = StarshipParams()):
    """[(p0, p1)] end points of the V3 truss tubes (lower joint -> upper joint)."""
    R = P.radius - P.truss_tube_radius - 0.03       # the tubes form the outer skin, as in the photos
    x0 = P.ring_x0 + P.truss_band
    x1 = P.x_aft - P.truss_band
    n = P.truss_nodes
    out = []
    for k in range(n):
        tb = 2 * np.pi * k / n
        lo = (x0, R * np.cos(tb), R * np.sin(tb))
        for s_ in (+1, -1):                       # the two legs of the V
            tt = tb + s_ * np.pi / n
            out.append((lo, (x1, R * np.cos(tt), R * np.sin(tt))))
    return out


def hot_stage_truss_sdf(X, Y, Z, P: StarshipParams = StarshipParams()):
    """V3 hot stage: lower and upper rings joined by a zig-zag of round tubes, open between."""
    R = P.radius
    rr = np.sqrt(Y * Y + Z * Z)
    t = P.truss_band_thickness
    shell = np.abs(rr - (R - t / 2)) - t / 2
    b = P.truss_band
    x0, x1 = P.ring_x0, P.x_aft
    parts = [np.maximum(shell, np.abs(X - (x0 + b / 2)) - b / 2),
             np.maximum(shell, np.abs(X - (x1 - b / 2)) - b / 2)]
    for p0, p1 in truss_tubes(P):
        parts.append(g.capsule(X, Y, Z, p0, p1, P.truss_tube_radius))
    return g.union(*parts)


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _convex_sdf_2d(Y, Z, V):
    """Signed distance to a convex polygon (vertices V, counter-clockwise) in the plane."""
    n = len(V)
    inside = None
    out2 = None
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        e = b - a
        L = np.hypot(*e)
        nx, nz = e[1] / L, -e[0] / L                       # outward normal of a CCW polygon
        dl = (Y - a[0]) * nx + (Z - a[1]) * nz             # signed distance to the edge's line
        inside = dl if inside is None else np.maximum(inside, dl)
        t = np.clip(((Y - a[0]) * e[0] + (Z - a[1]) * e[1]) / (L * L), 0.0, 1.0)
        d2 = (Y - a[0] - t * e[0]) ** 2 + (Z - a[1] - t * e[1]) ** 2
        out2 = d2 if out2 is None else np.minimum(out2, d2)
    return np.where(inside > 0, np.sqrt(out2), inside)


def _ccw(V):
    V = np.asarray(V, float)
    area = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
    return V if area > 0 else V[::-1]


def _rot(az):
    return np.array([[np.cos(az), -np.sin(az)], [np.sin(az), np.cos(az)]])


def _inner_plate_poly(az, P):
    """Corners (y, z) of an inner plate before rounding: a trapezoid along azimuth az."""
    r0, r1, w0, w1 = P.plate_inner
    # straight sides, outer end an arc of radius r1 (about the axis)
    a1 = np.arcsin(min(0.99, (w1 / 2) / r1))
    arc = [[r1 * np.cos(t), r1 * np.sin(t)] for t in np.linspace(-a1, a1, 7)]
    loc = np.array([[r0, -w0 / 2]] + arc + [[r0, w0 / 2]])      # (radial, tangential)
    return _ccw(loc @ _rot(az).T)


def _outer_plate_poly(az, P):
    """Corners of an outer plate before rounding: the gap between the two neighbouring inner
    plates (their facing sides moved over by the plate gap), from radius ra to rb."""
    ra, rb = P.plate_outer
    R = _rot(az)
    sides = []
    for sgn in (+1, -1):                                   # neighbours at az + 60 deg and az - 60 deg
        T = _inner_plate_poly(az + sgn * np.pi / 3, P) @ R  # into the outer plate's frame (radial, tangential)
        # the neighbour's side facing this plate: the edge whose midpoint is nearest the axis u
        best = None
        for i in range(len(T)):
            a, b = T[i], T[(i + 1) % len(T)]
            m = 0.5 * (a + b)
            if abs(np.hypot(*a) - np.hypot(*b)) < 1e-6:
                continue                                   # (inner / outer ends, both corners at one radius: skip)
            key = abs(m[1])
            if best is None or key < best[0]:
                best = (key, a, b)
        _, a, b = best
        if np.hypot(*b) < np.hypot(*a):
            a, b = b, a                                    # along the side, outward
        e = (b - a) / np.hypot(*(b - a))
        nrm = np.array([-e[1], e[0]])
        if nrm @ (np.array([0.5 * (ra + rb), 0.0]) - a) < 0:
            nrm = -nrm                                     # toward this plate
        sides.append((a + P.plate_gap * nrm, e))
    # each side from u = ra out to where it meets the circle of radius rb; the outer end an arc
    inner, outer = [], []
    for p0, e in sides:
        inner.append(p0 + (ra - p0[0]) / e[0] * e)
        # |p0 + t e| = rb, the root beyond the inner end
        bq = 2 * (p0 @ e)
        cq = p0 @ p0 - rb * rb
        t = (-bq + np.sqrt(bq * bq - 4 * cq)) / 2
        outer.append(p0 + t * e)
    t0, t1 = sorted(np.arctan2(q[1], q[0]) for q in outer)
    arc = [[rb * np.cos(t), rb * np.sin(t)] for t in np.linspace(t0, t1, 9)]
    lo_in = min(inner, key=lambda q: q[1])
    hi_in = max(inner, key=lambda q: q[1])
    loc = np.array([lo_in] + arc + [hi_in])
    return _ccw(loc @ R.T)


_ROUND_CACHE = {}


def _rounded_field(V, r, h=0.01):
    """Plan-view signed distance of the polygon V with every corner rounded to radius r (a
    morphological opening on a 1 cm raster: erode by r, dilate by r), as an interpolator."""
    key = (V.tobytes(), r, h)
    if key in _ROUND_CACHE:
        return _ROUND_CACHE[key]
    from scipy.ndimage import distance_transform_edt as edt
    from scipy.interpolate import RegularGridInterpolator
    lo = V.min(0) - r - 0.3
    hi = V.max(0) + r + 0.3
    ay = np.arange(lo[0], hi[0] + h, h)
    az = np.arange(lo[1], hi[1] + h, h)
    Y, Z = np.meshgrid(ay, az, indexing="ij")
    inside = _convex_sdf_2d(Y, Z, V) < 0
    core = edt(inside) * h > r                               # eroded
    opened = edt(~core) * h <= r                             # dilated back: corners rounded
    sd = np.where(opened, -edt(opened) * h, edt(~opened) * h)
    f = RegularGridInterpolator((ay, az), sd, bounds_error=False, fill_value=None)
    _ROUND_CACHE[key] = (f, lo, hi)
    return _ROUND_CACHE[key]


def _rounded(Y, Z, V, r):
    f, lo, hi = _rounded_field(V, r)
    Yc = np.clip(Y, lo[0], hi[0])
    Zc = np.clip(Z, lo[1], hi[1])
    d = f(np.stack([np.ravel(Yc), np.ravel(Zc)], -1)).reshape(np.shape(Y))
    # beyond the raster: add the distance to it (a bound, enough for the plates' thin layer)
    return d + np.hypot(Y - Yc, Z - Zc)


def dome_plate_footprints(P: StarshipParams = StarshipParams()):
    """[(name, f(Y, Z) -> plan-view signed distance)] of the protection plates; 'apex disc' first
    (if P.apex_disc), then 'inner plate 1..3' (under the sea-level engines) and 'outer plate 1..3'."""
    out = []
    if P.apex_disc:
        out.append(("apex disc", lambda Y, Z: np.sqrt(Y * Y + Z * Z) - P.plate_apex_r))
    az_sl = np.radians(P.sea_level[5])
    rr = P.plate_round
    for k in range(3):
        V = _inner_plate_poly(az_sl + 2 * np.pi * k / 3, P)
        out.append((f"inner plate {k + 1}", (lambda V_: lambda Y, Z: _rounded(Y, Z, V_, rr))(V)))
    for k in range(3):
        V = _outer_plate_poly(az_sl + np.pi / 3 + 2 * np.pi * k / 3, P)
        out.append((f"outer plate {k + 1}", (lambda V_: lambda Y, Z: _rounded(Y, Z, V_, rr))(V)))
    return out


def dome_plate_outlines(P: StarshipParams = StarshipParams(), h=0.01):
    """{name: [(N, 2) arrays of (y, z)]} plan-view outlines of the plates (for drawings)."""
    from skimage.measure import find_contours
    ax = np.arange(-4.6, 4.6 + h / 2, h)
    Y, Z = np.meshgrid(ax, ax, indexing="ij")
    out = {}
    for name, f in dome_plate_footprints(P):
        cs = find_contours(f(Y, Z), 0.0)
        out[name] = [np.column_stack([np.interp(c[:, 0], np.arange(len(ax)), ax),
                                      np.interp(c[:, 1], np.arange(len(ax)), ax)]) for c in cs if len(c) > 20]
    return out


def dome_plate_centroids(P: StarshipParams = StarshipParams(), h=0.05):
    """{name: (y, z)} area centroid of each plate's plan view."""
    ax = np.arange(-4.6, 4.6 + h / 2, h)
    Y, Z = np.meshgrid(ax, ax, indexing="ij")
    out = {}
    for name, f in dome_plate_footprints(P):
        m = f(Y, Z) < 0
        out[name] = (float(Y[m].mean()), float(Z[m].mean())) if m.any() else (0.0, 0.0)
    return out


def dome_distance(X, Y, Z, tank_top_x=68.3, dome_rise=2.9, R=4.5):
    """Approximate signed distance to the booster's forward dome (half ellipsoid)."""
    rho = np.sqrt(Y * Y + Z * Z)
    return (np.sqrt(((X - tank_top_x) / dome_rise) ** 2 + (rho / R) ** 2) - 1.0) * min(dome_rise, R)


def dome_plates_sdf(X, Y, Z, P: StarshipParams = StarshipParams(), tank_top_x=68.3, dome_rise=2.9, which=None):
    """The protection plates as thin caps standing `plate_standoff` proud of the dome. which:
    index into dome_plate_footprints (or None for all)."""
    standoff = P.plate_standoff
    d = dome_distance(X, Y, Z, tank_top_x, dome_rise, P.radius)
    layer = np.maximum(d - standoff, -d - 0.02)
    layer = np.maximum(layer, tank_top_x - X)
    parts = []
    for k, (name, f) in enumerate(dome_plate_footprints(P)):
        if which is not None and k != which:
            continue
        parts.append(np.maximum(layer, f(Y, Z)))
    return g.union(*parts)


def interstage_sdf(X, Y, Z, P: StarshipParams = StarshipParams(), tank_top_x=68.3, dome_rise=2.9):
    """The hot stage as built for P.ring_style: truss (+ dome plates) or the vented ring."""
    if P.ring_style == "vented":
        return hot_stage_ring_sdf(X, Y, Z, P)
    return np.minimum(hot_stage_truss_sdf(X, Y, Z, P), dome_plates_sdf(X, Y, Z, P, tank_top_x, dome_rise))


def hot_stage_ring_sdf(X, Y, Z, P: StarshipParams = StarshipParams()):
    """Vented interstage on top of the booster: solid bands at the bottom and top, struts between."""
    R = P.radius
    rr = np.sqrt(Y * Y + Z * Z)
    t = P.ring_thickness
    shell = np.abs(rr - (R - t / 2)) - t / 2
    x0, x1 = P.ring_x0, P.x_aft
    b = P.ring_bands
    bottom = np.maximum(shell, np.abs(X - (x0 + b / 2)) - b / 2)
    top = np.maximum(shell, np.abs(X - (x1 - b / 2)) - b / 2)
    parts = [bottom, top]
    for k in range(P.ring_struts):
        th = 2 * np.pi * (k + 0.5) / P.ring_struts
        parts.append(g.radial_box(X, Y, Z, th, 0.5 * (x0 + x1), R - t / 2, 0.5 * (x1 - x0), t / 2,
                                  0.5 * P.ring_strut_width))
    return g.union(*parts)
