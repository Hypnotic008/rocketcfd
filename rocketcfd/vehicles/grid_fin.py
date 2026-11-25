"""Parametric Super Heavy V3 grid fin as an exact signed distance function.

Built from the photo-measured description in spacecraft-models/starship dimensions.md
("Grid fins"), with every number a parameter:

  * outline: nearly square body, 45 deg tapered root to a narrow root on the centreline,
    outer corners cut at 45 deg (an octagon seen from above)
  * lattice: thin steel plates set diagonally to the edges (diamond cells), 8 diamonds
    across the width -> plates ~0.31 m apart; the 45 deg root and corner edges run along
    lattice plates, as on the real fin
  * teeth: only on the face toward the engines (upstream in an engines-first descent).
    Each plate's edge rises to a sharp point where plates cross and dips in a deep rounded
    notch between, reaching nearly halfway through the fin. Edge plates are notched the same
    way; the cross plate has V notches. The other face is flat.
  * solid edge plates along the sides, cut corners and outer end; a solid cross plate
    where the sides turn in toward the root; a solid triangular attachment area at the root
  * plate thickness ~1 cm: a ~3 t steel fin with ~39 m^2 of lattice plate (one side) gives
    ~1 cm. Not published; it is a parameter.

Local frame (same as the CatchSim CAD): x through the fin along the booster axis (the
toothed face is at x = -depth/2, toward the engines), y around the hull (width), z radial
from the hull surface (z = 0) outward. The hinge axis is the z axis.
"""
from dataclasses import dataclass
import numpy as np

S2 = np.sqrt(2.0)


@dataclass
class GridFinParams:
    width: float = 3.5            # around the hull (measured ~3.5 m)
    depth: float = 0.5            # through the fin, along the flow (measured ~0.5 m)
    diamonds: int = 8             # diamonds across the width (counted in photos)
    plate: float = 0.010          # lattice plate thickness (mass-based estimate)
    edge_plate: float = 0.016     # side / outer / cross plate thickness (estimate)
    neck: float = 0.30            # hull surface to the root edge (reach 4.2 m = neck + 3.9 m fin)
    root_half: float = 0.475      # half-width of the narrow root edge
    rows_outer: int = 6           # cross plate to outer end, in node-row spacings (= width / diamonds)
    chamfer_rows: int = 2         # size of the 45 deg cut outer corners, in node-row spacings
    teeth: bool = True
    tooth_depth: float = 0.45     # notch depth as a fraction of the fin depth ("nearly halfway")
    tooth_land: float = 0.006     # width of the tooth tips (a zero-width tip is not a valid solid)
    neck_width: float = 0.6
    neck_depth: float = 0.4

    # derived layout -------------------------------------------------------
    @property
    def s(self):                   # node-row spacing along y and z (= diamond diagonal)
        return self.width / self.diamonds

    @property
    def p(self):                   # perpendicular spacing between parallel plates
        return self.s / S2

    @property
    def w(self):
        return 0.5 * self.width

    @property
    def z_root(self):
        return self.neck

    @property
    def z_cross(self):
        return self.neck + (self.w - self.root_half)

    @property
    def z_tip(self):
        return self.z_cross + self.rows_outer * self.s

    @property
    def chamfer(self):
        return self.chamfer_rows * self.s

    def summary(self):
        return (f"grid fin: {self.width:.2f} m wide x {self.z_tip - self.z_root:.2f} m root-to-tip "
                f"(reach {self.z_tip:.2f} m) x {self.depth:.2f} m deep; plates {1000*self.plate:.0f} mm, "
                f"{self.p:.3f} m apart; teeth {'on' if self.teeth else 'off'}"
                f" ({self.tooth_depth*self.depth:.2f} m deep)")


def _ellipse_notch(along, xin, pitch_half, depth, land=0.0):
    """SDF (negative inside) of semi-elliptical notches centred between crossings.
    along: coordinate along the plate edge, already shifted so crossings are at multiples
    of 2*pitch_half; xin: distance into the fin from the toothed face. Notch half-width is
    pitch_half - land/2, leaving a tooth tip of width `land` at each crossing."""
    c = 2.0 * pitch_half * (np.floor(along / (2.0 * pitch_half)) + 0.5)
    half = pitch_half - 0.5 * land
    r = np.sqrt(((along - c) / half) ** 2 + (xin / depth) ** 2)
    return (r - 1.0) * min(half, depth)


def _v_notch(along, xin, pitch_half, depth, land=0.0):
    c = 2.0 * pitch_half * (np.floor(along / (2.0 * pitch_half)) + 0.5)
    half = pitch_half - 0.5 * land
    return (np.abs(along - c) / half + xin / depth - 1.0) * (half * depth / np.hypot(half, depth))


def _outline(P, y, z):
    """Convex outline of the fin in (y, z): SDF, negative inside."""
    w, c, r0 = P.w, P.chamfer, P.root_half
    ay = np.abs(y)
    d = P.z_root - z                                                   # root edge
    d = np.maximum(d, ((ay - r0) - (z - P.z_root)) / S2)               # tapered root edges
    d = np.maximum(d, ay - w)                                          # sides
    d = np.maximum(d, ((ay - w) + (z - (P.z_tip - c))) / S2)           # cut outer corners
    d = np.maximum(d, z - P.z_tip)                                     # outer end
    return d


def grid_fin_sdf(X, Y, Z, P: GridFinParams = GridFinParams()):
    """SDF of the fin in its local frame (see module docstring)."""
    hd = 0.5 * P.depth
    xin = X + hd                                   # depth measured from the toothed face
    in_depth = np.abs(X) - hd
    bdepth = P.tooth_depth * P.depth
    zp = Z - P.z_cross                             # lattice phase: node rows at multiples of s
    u = (Y + zp) / S2                              # plates of family A at u = k p
    v = (-Y + zp) / S2                             # plates of family B at v = m p
    p, s = P.p, P.s
    outline = _outline(P, Y, Z)

    # lattice plates, notched between crossings on the toothed face
    dA = np.abs(u - p * np.round(u / p)) - 0.5 * P.plate
    dB = np.abs(v - p * np.round(v / p)) - 0.5 * P.plate
    if P.teeth:
        dA = np.maximum(dA, -_ellipse_notch(v, xin, 0.5 * p, bdepth, P.tooth_land))
        dB = np.maximum(dB, -_ellipse_notch(u, xin, 0.5 * p, bdepth, P.tooth_land))
    lattice = np.maximum(np.minimum(dA, dB), outline)

    # edge plates: a band just inside the outline, notched on the toothed face
    te = P.edge_plate
    band = np.abs(outline + 0.5 * te) - 0.5 * te
    if P.teeth:
        ay = np.abs(Y)
        on_side = (ay > P.w - 2 * te) & (Z < P.z_tip - P.chamfer)
        on_end = Z > P.z_tip - 2 * te
        # sides: contacts every s along z; outer end: contacts every s along y;
        # cut corners run along lattice plates (family A on +y, family B on -y): contacts every p
        along = np.where(on_side, zp, np.where(on_end, Y + 0.5 * s * (P.diamonds % 2), np.where(Y > 0, v, u)))
        pitch_half = np.where(on_side | on_end, 0.5 * s, 0.5 * p)
        c = 2.0 * pitch_half * (np.floor(along / (2.0 * pitch_half)) + 0.5)
        half = pitch_half - 0.5 * P.tooth_land
        r = np.sqrt(((along - c) / half) ** 2 + (xin / bdepth) ** 2)
        notch = (r - 1.0) * np.minimum(half, bdepth)
        root_edge = Z < P.z_cross                   # the tapered root edges are smooth plates
        band = np.where(root_edge, band, np.maximum(band, -notch))
    edges = np.maximum(band, outline)

    # cross plate (V notches) and the solid root triangle
    cross = np.maximum(np.abs(Z - P.z_cross) - 0.5 * te, np.abs(Y) - P.w)
    if P.teeth:
        cross = np.maximum(cross, -_v_notch(Y + 0.5 * s * (P.diamonds % 2), xin, 0.5 * s, bdepth, P.tooth_land))
    h_tri = P.z_cross - P.z_root
    nrm = np.hypot(h_tri, P.root_half)
    tri = np.maximum(P.z_root - Z, ((np.abs(Y) - P.root_half) * h_tri + (Z - P.z_root) * P.root_half) / nrm)
    tri = np.maximum(tri, Z - P.z_cross)

    body = np.minimum(np.minimum(lattice, edges), np.minimum(cross, tri))
    body = np.maximum(body, in_depth)
    # neck / shaft from the hull surface to the root edge
    neck = np.maximum.reduce([np.abs(X) - 0.5 * P.neck_depth, np.abs(Y) - 0.5 * P.neck_width,
                              -0.05 - Z, Z - (P.z_root + 0.05)])
    return np.minimum(body, neck)


def grid_fin_box(P: GridFinParams = GridFinParams(), pad=0.02):
    """Bounding box (lo, hi) of the fin in its local frame."""
    return ((-0.5 * P.depth - pad, -P.w - pad, -0.05 - pad),
            (0.5 * P.depth + pad, P.w + pad, P.z_tip + pad))


def planform_area(P: GridFinParams = GridFinParams()):
    """Outline area in the (y, z) plane (reference area for fin coefficients)."""
    w, r0, c = P.w, P.root_half, P.chamfer
    h_tri = P.z_cross - P.z_root
    return (2 * w) * (P.z_tip - P.z_cross) - c * c + (w + r0) * h_tri


def cascade_slope(mach, P: "GridFinParams" = None):
    """Supersonic linear-theory normal-force slope of the fin (per rad, on planform_area), for
    incidence about the hinge (z) axis: every plate is a thin supersonic airfoil, lift slope
    4 / beta on its own area, beta = sqrt(M^2 - 1). The diagonal lattice plates see the
    crossflow component normal to them (alpha / sqrt 2) and contribute that force's projection
    (1 / sqrt 2): 2 / beta per unit plate area; the two side edge plates face the crossflow
    fully: 4 / beta. Lattice plate length per unit planform = 2 / p (two families, spacing p).

    Interference between plates inside a cell (Mach waves reaching the neighbouring plates
    within the fin depth) and the lower dynamic pressure of a partly choked lattice are
    neglected, so this is an upper bound, approached at high supersonic Mach (Burkhalter's
    grid-fin theory adds those effects)."""
    P = P or GridFinParams()
    mach = np.asarray(mach, float)
    beta = np.sqrt(np.maximum(mach ** 2 - 1.0, 1e-6))
    S = planform_area(P)
    lattice = 2.0 * S / P.p * P.depth                      # one side, m^2
    sides = 2.0 * (P.z_tip - P.chamfer - P.z_cross) * P.depth
    return (2.0 * lattice + 4.0 * sides) / (S * beta)
