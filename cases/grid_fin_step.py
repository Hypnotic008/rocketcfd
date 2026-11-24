"""Build the parametric V3 grid fin as an exact CAD solid (STEP) with CadQuery.

Same parameters as rocketcfd/vehicles/grid_fin.py (the CFD geometry), so the CAD file
and the simulated fin are the same object. Each plate is drawn as a 2-D profile in its
own plane with the teeth already in the outline (smooth spline notches), extruded to the
plate thickness, trimmed to the fin outline and fused: no notch-cutting booleans.

  pip install cadquery
  python cases/grid_fin_step.py                 # results/grid_fin_v3.step (+ a light STL)
  python cases/grid_fin_step.py --no-teeth

Frame (metres): x through the fin (toothed face at x = -depth/2, toward the engines),
y around the hull, z out from the hull surface.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cadquery as cq
from rocketcfd.vehicles.grid_fin import GridFinParams, planform_area

S2 = np.sqrt(2.0)


def upstream_plane(origin, xDir, normal):
    """Sketch plane whose local y axis is global +x, so that local y = -depth/2 (where the
    notches are drawn) is always the toothed face toward the engines. The plate is extruded
    symmetrically, so flipping the normal does not move it."""
    xd = np.asarray(xDir, float)
    nm = np.asarray(normal, float)
    if np.cross(nm, xd)[0] < 0:
        nm = -nm
    return cq.Plane(origin=tuple(origin), xDir=tuple(xd), normal=tuple(nm))


def notched_plate(plane, s0, s1, depth, thick, pitch=None, notch_depth=0.0, shape="round", phase=0.0, land=0.0):
    """Plate in `plane` (local x = along the edge, local y = global x through the fin),
    spanning along = [s0, s1], with notches on the local y = -depth/2 edge centred at
    phase + (j + 1/2) * pitch, half-width pitch / 2."""
    hd = 0.5 * depth
    wp = cq.Workplane(plane).moveTo(s0, hd).lineTo(s1, hd).lineTo(s1, -hd)
    if pitch is None or notch_depth <= 0:
        wp = wp.lineTo(s0, -hd)
    else:
        a = 0.5 * pitch - 0.5 * land          # leaves a tooth tip of width `land` at each crossing
        # notch centres inside [s0, s1], walked from s1 back to s0
        j1 = int(np.floor((s1 - phase) / pitch - 0.5))
        j0 = int(np.ceil((s0 - phase) / pitch - 0.5))
        cur = s1
        for j in range(j1, j0 - 1, -1):
            c = phase + (j + 0.5) * pitch
            if c + a > s1 - 1e-9 or c - a < s0 + 1e-9:
                continue
            if cur > c + a + 1e-9:
                wp = wp.lineTo(c + a, -hd)
            if shape == "round":
                th = np.linspace(0.0, np.pi, 17)[1:]
                pts = [(c + a * np.cos(t), -hd + notch_depth * np.sin(t)) for t in th]
                pts[-1] = (c - a, -hd)
                wp = wp.spline(pts, includeCurrent=True)
            else:                                    # V notch
                wp = wp.lineTo(c, -hd + notch_depth).lineTo(c - a, -hd)
            cur = c - a
        if cur > s0 + 1e-9:
            wp = wp.lineTo(s0, -hd)
    return wp.close().extrude(0.5 * thick, both=True)


def outline_prism(P):
    w, r0, c = P.w, P.root_half, P.chamfer
    pts = [(-r0, P.z_root), (r0, P.z_root), (w, P.z_cross), (w, P.z_tip - c), (w - c, P.z_tip),
           (-(w - c), P.z_tip), (-w, P.z_tip - c), (-w, P.z_cross)]
    # sketch in the global y-z plane, extruded along x through the depth
    plane = cq.Plane(origin=(-0.5 * P.depth, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))
    return cq.Workplane(plane).polyline(pts).close().extrude(P.depth), pts


def build(P: GridFinParams):
    t0 = time.time()
    d, p, s, w = P.depth, P.p, P.s, P.w
    nd = P.tooth_depth * d if P.teeth else 0.0
    outline, verts = outline_prism(P)
    zc = P.z_cross
    U = [(y + (z - zc)) / S2 for y, z in verts]
    V = [(-y + (z - zc)) / S2 for y, z in verts]
    parts = []

    def plane_A(k):            # plate u = k p; along = v; normal +u
        o = (0.0, k * p / S2, zc + k * p / S2)
        return upstream_plane(o, (0, -1 / S2, 1 / S2), (0, 1 / S2, 1 / S2))

    def plane_B(m):            # plate v = m p; along = u; normal +v
        o = (0.0, -m * p / S2, zc + m * p / S2)
        return upstream_plane(o, (0, 1 / S2, 1 / S2), (0, -1 / S2, 1 / S2))

    pad = 2 * p
    land = P.tooth_land
    # lattice plates that lie on an edge (cut corners, root tapers) are left to the edge
    # plates: two solids sharing a face are a classic source of invalid CAD
    kc = round((w + P.z_tip - zc - P.chamfer) / S2 / p)
    mt = round(w / S2 / p)
    skip_A, skip_B = {kc, -mt}, {kc, -mt}
    for k in range(int(np.floor(min(U) / p)), int(np.ceil(max(U) / p)) + 1):
        if k in skip_A:
            continue
        s0, s1 = p * np.floor((min(V) - pad) / p), p * np.ceil((max(V) + pad) / p)
        parts.append(notched_plate(plane_A(k), s0, s1, d, P.plate, p, nd, land=land))
    for m in range(int(np.floor(min(V) / p)), int(np.ceil(max(V) / p)) + 1):
        if m in skip_B:
            continue
        s0, s1 = p * np.floor((min(U) - pad) / p), p * np.ceil((max(U) + pad) / p)
        parts.append(notched_plate(plane_B(m), s0, s1, d, P.plate, p, nd, land=land))

    te = P.edge_plate
    # sides (along z, notches every s from the cross plate), outer end (along y), cross plate (V notches)
    for sgn in (1, -1):
        pl = upstream_plane((0, sgn * (w - 0.5 * te), zc), (0, 0, 1), (0, sgn, 0))
        parts.append(notched_plate(pl, -s, P.z_tip - zc + s, d, te, s, nd, land=land))
    pl = upstream_plane((0, 0, P.z_tip - 0.5 * te), (0, 1, 0), (0, 0, -1))
    parts.append(notched_plate(pl, -w - s, w + s, d, te, s, nd, phase=-w, land=land))
    pl = upstream_plane((0, 0, zc), (0, 1, 0), (0, 0, -1))
    parts.append(notched_plate(pl, -w, w, d, te, s, nd, shape="V", phase=-w, land=land))
    # cut corners: along lattice plates (A on +y, B on -y), notched every p; root tapers: smooth
    for sgn in (1, -1):
        if sgn > 0:
            o = (0.0, (kc * p - 0.5 * te) / S2, zc + (kc * p - 0.5 * te) / S2)
            pl = upstream_plane(o, (0, -1 / S2, 1 / S2), (0, 1 / S2, 1 / S2))
        else:
            o = (0.0, -(kc * p - 0.5 * te) / S2, zc + (kc * p - 0.5 * te) / S2)
            pl = upstream_plane(o, (0, 1 / S2, 1 / S2), (0, -1 / S2, 1 / S2))
        parts.append(notched_plate(pl, -kc * p, kc * p, d, te, p, nd, land=land))
    # root tapers lie on plates v = -mt p (+y side) and u = -mt p (-y side)
    for sgn in (1, -1):
        off = -mt * p + 0.5 * te
        if sgn > 0:
            pl = upstream_plane((0.0, -off / S2, zc + off / S2), (0, 1 / S2, 1 / S2), (0, -1 / S2, 1 / S2))
        else:
            pl = upstream_plane((0.0, off / S2, zc + off / S2), (0, -1 / S2, 1 / S2), (0, 1 / S2, 1 / S2))
        parts.append(notched_plate(pl, -3 * w, 3 * w, d, te))
    # solid root triangle
    tri_plane = cq.Plane(origin=(-0.5 * d, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))
    tri = cq.Workplane(tri_plane).polyline([(-P.root_half, P.z_root), (P.root_half, P.z_root), (0.0, zc)]).close().extrude(d)

    print(f"  {len(parts)} plates drawn ({time.time()-t0:.1f} s); trimming to the outline and fusing ...", flush=True)
    body = None
    out_solid = outline.val()
    for k, part in enumerate(parts):
        piece = part.val().intersect(out_solid)
        body = piece if body is None else body.fuse(piece)
    body = body.fuse(tri.val())
    neck = cq.Workplane("XY").box(P.neck_depth, P.neck_width, P.z_root + 0.1, centered=(True, True, False)) \
        .translate((0, 0, -0.05)).val()
    body = body.fuse(neck).clean()
    if not body.isValid():
        body = body.fix()
    print(f"  fused ({time.time()-t0:.1f} s), valid solid: {body.isValid()}, "
          f"{len(body.Solids())} solid(s), {len(body.Faces())} faces", flush=True)
    return body


def check_against_sdf(body, P, slab=0.004):
    """Cross-section area of the CAD solid vs the CFD distance function at several depths.
    Teeth on the upstream face mean much less material near x = -depth/2; the CAD and the
    CFD geometry should agree slice by slice."""
    from rocketcfd.vehicles.grid_fin import grid_fin_sdf
    y = np.linspace(-P.w - 0.05, P.w + 0.05, 1401)
    z = np.linspace(P.z_root, P.z_tip + 0.02, 1601)       # fin body only (above the neck)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    cell = (y[1] - y[0]) * (z[1] - z[0])
    print("  slice x [m]   CAD area [m^2]   CFD-geometry area [m^2]   (toothed face at x = %.2f)" % (-P.depth / 2))
    ok = True
    for x0 in (-0.22, -0.15, -0.05, 0.05, 0.15, 0.22):
        box = cq.Workplane("XY").box(slab, 2 * P.w + 0.4, P.z_tip - P.z_root + 0.04, centered=(True, True, False)) \
            .translate((x0, 0, P.z_root)).val()
        a_cad = body.intersect(box).Volume() / slab
        a_sdf = (grid_fin_sdf(np.full_like(Y, x0), Y, Z, P) < 0).sum() * cell
        ok &= abs(a_cad - a_sdf) < 0.1 * max(a_sdf, 0.05)
        print(f"  {x0:+10.2f}   {a_cad:14.3f}   {a_sdf:22.3f}")
    print("  CAD matches the CFD geometry" if ok else "  MISMATCH between CAD and CFD geometry")


if __name__ == "__main__":
    P = GridFinParams(teeth="--no-teeth" not in sys.argv)
    print(P.summary(), f"| planform {planform_area(P):.2f} m^2")
    body = build(P)
    os.makedirs("results", exist_ok=True)
    tag = "" if P.teeth else "_noteeth"
    step = f"results/grid_fin_v3{tag}.step"
    cq.exporters.export(body, step)
    stl = f"results/grid_fin_v3{tag}_light.stl"
    cq.exporters.export(body, stl, tolerance=0.002, angularTolerance=0.2)
    print(f"{step}: {os.path.getsize(step)/1e6:.1f} MB   {stl}: {os.path.getsize(stl)/1e6:.1f} MB   "
          f"volume {body.Volume():.3f} m^3 (steel: {7850*body.Volume()/1000:.2f} t)")
    check_against_sdf(body, P)

