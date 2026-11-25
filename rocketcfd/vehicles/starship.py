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
    vacuum: tuple = (3, 3.15, 1.15, 3.0, -0.3, 60.0)
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

    @property
    def length(self):
        return self.barrel_length + self.nose_length

    @property
    def x_nose(self):
        return self.x_aft + self.length


def engines(P: StarshipParams):
    """[(kind, y, z, exit_x, exit_radius)] of the six ship engines."""
    out = []
    for kind, (n, pr, re_, bl, dx, az0) in (("sea level", P.sea_level), ("vacuum", P.vacuum)):
        for k in range(n):
            a = np.radians(az0 + 360.0 * k / n)
            out.append((kind, pr * np.cos(a), pr * np.sin(a), P.x_aft + dx, re_, bl))
    return out


def ogive_radius(xx, R, L):
    """Tangent ogive of base radius R and length L; xx measured from the base (0..L)."""
    rho = (R * R + L * L) / (2.0 * R)
    return np.sqrt(np.maximum(rho * rho - np.clip(xx, 0.0, L) ** 2, 0.0)) + R - rho


def starship_sdf(X, Y, Z, P: StarshipParams = StarshipParams()):
    R = P.radius
    rr = np.sqrt(Y * Y + Z * Z)
    xs = P.x_aft
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
    for kind, y, z, xe, re_, bl in engines(P):
        top_r = 0.32 if kind == "sea level" else 0.45
        parts.append(g.frustum_x(X, Y, Z, re_, top_r, xe, min(xe + bl, xs + P.skirt_height + 0.05), y, z))
    # flaps (radial plates)
    for (x0, x1, reach, th), r_at in ((P.aft_flaps, None), (P.fwd_flaps, "nose")):
        for az in P.flap_azimuths:
            th_ = np.radians(az)
            xc = xs + 0.5 * (x0 + x1)
            rbase = R if r_at is None else float(ogive_radius(0.5 * (x0 + x1) - P.barrel_length, R, P.nose_length))
            parts.append(g.radial_box(X, Y, Z, th_, xc, rbase + 0.5 * reach - 0.2, 0.5 * (x1 - x0),
                                      0.5 * reach + 0.2, 0.5 * th))
    return g.union(*parts)


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
