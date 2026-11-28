"""Signed-distance-function (SDF) geometry kernel.

Convention: phi < 0 inside solid, phi > 0 in fluid, |phi| ~ distance to surface.
All functions are vectorised over numpy arrays X, Y, Z (meters). The vehicle axis is +x.
Unions/subtractions of exact SDFs give exact distances outside the solid and a
conservative bound inside, which is all the immersed-boundary method needs.
"""
import numpy as np


def union(*phis):
    out = phis[0]
    for p in phis[1:]:
        out = np.minimum(out, p)
    return out


def subtract(a, b):
    """Solid a with solid b removed."""
    return np.maximum(a, -b)


def intersect(a, b):
    return np.maximum(a, b)


def _combine(d_rad, d_ax):
    outside = np.sqrt(np.maximum(d_rad, 0.0) ** 2 + np.maximum(d_ax, 0.0) ** 2)
    inside = np.minimum(np.maximum(d_rad, d_ax), 0.0)
    return outside + inside


def cylinder_x(X, Y, Z, r, x0, x1, yc=0.0, zc=0.0):
    """Capped cylinder along x from x0 to x1."""
    rho = np.sqrt((Y - yc) ** 2 + (Z - zc) ** 2)
    xm, xh = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    return _combine(rho - r, np.abs(X - xm) - xh)


def frustum_x(X, Y, Z, r0, r1, x0, x1, yc=0.0, zc=0.0):
    """Capped cone frustum along x, radius r0 at x0 and r1 at x1 (approximate SDF)."""
    rho = np.sqrt((Y - yc) ** 2 + (Z - zc) ** 2)
    t = np.clip((X - x0) / (x1 - x0), 0.0, 1.0)
    r = r0 + (r1 - r0) * t
    cosa = (x1 - x0) / np.hypot(x1 - x0, r1 - r0)
    xm, xh = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    return _combine((rho - r) * cosa, np.abs(X - xm) - xh)


def sphere(X, Y, Z, r, c=(0.0, 0.0, 0.0)):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def box_local(a, b, c, ha, hb, hc):
    """Axis-aligned box in local coordinates (a, b, c) with half sizes."""
    q = [np.abs(a) - ha, np.abs(b) - hb, np.abs(c) - hc]
    outside = np.sqrt(sum(np.maximum(qi, 0.0) ** 2 for qi in q))
    inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)
    return outside + inside


def radial_frame(Y, Z, theta):
    """Local radial (r) and tangential (t) coordinates for azimuth theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return Y * c + Z * s, -Y * s + Z * c


def radial_box(X, Y, Z, theta, xc, rc, ha, hr, ht):
    """Box centred at axial station xc, radial distance rc, at azimuth theta."""
    r, t = radial_frame(Y, Z, theta)
    return box_local(X - xc, r - rc, t, ha, hr, ht)


def lattice_holes(u, v, cell, web, angle=np.pi / 4):
    """Infinite-depth square lattice of holes in the (u, v) plane (rotated by `angle`).
    Returns SDF of the hole region (negative inside holes)."""
    c, s = np.cos(angle), np.sin(angle)
    a, b = u * c + v * s, -u * s + v * c
    qa = np.mod(a + 0.5 * cell, cell) - 0.5 * cell
    qb = np.mod(b + 0.5 * cell, cell) - 0.5 * cell
    h = 0.5 * (cell - web)
    return box_local(qa, qb, np.zeros_like(qa), h, h, 1e9)


class MeshSDF:
    """Signed distance from a closed triangle mesh, tabulated on a local grid.

    tris: (n, 3, 3) triangle vertices. Inside/outside by ray parity along local x,
    distance by Euclidean distance transforms on the voxel grid (accurate to ~res).
    """

    def __init__(self, tris, res=0.01, pad=0.1):
        from scipy import ndimage
        tris = np.asarray(tris, float)
        self.lo = tris.reshape(-1, 3).min(0) - pad
        hi = tris.reshape(-1, 3).max(0) + pad
        self.res = res
        n = np.ceil((hi - self.lo) / res).astype(int) + 1
        ax = [self.lo[d] + res * np.arange(n[d]) for d in range(3)]
        # ray parity along x for every (y, z) column
        hits_j, hits_k, hits_x = [], [], []
        for t in tris:
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
            ymin, ymax = min(y0, y1, y2), max(y0, y1, y2)
            zmin, zmax = min(z0, z1, z2), max(z0, z1, z2)
            j0 = int(np.ceil((ymin - self.lo[1]) / res)); j1 = int(np.floor((ymax - self.lo[1]) / res))
            k0 = int(np.ceil((zmin - self.lo[2]) / res)); k1 = int(np.floor((zmax - self.lo[2]) / res))
            if j1 < j0 or k1 < k0:
                continue
            Yg, Zg = np.meshgrid(ax[1][j0:j1 + 1], ax[2][k0:k1 + 1], indexing="ij")
            det = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
            if abs(det) < 1e-14:
                continue
            # tiny offset avoids double counting rays that hit shared edges exactly
            yy, zz = Yg + 1.234e-7, Zg + 2.345e-7
            l1 = ((yy - y0) * (z2 - z0) - (y2 - y0) * (zz - z0)) / det
            l2 = ((y1 - y0) * (zz - z0) - (yy - y0) * (z1 - z0)) / det
            inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            if not inside.any():
                continue
            xh = x0 + l1 * (x1 - x0) + l2 * (x2 - x0)
            jj, kk = np.nonzero(inside)
            hits_j.append(jj + j0); hits_k.append(kk + k0); hits_x.append(xh[inside])
        hj, hk, hx = map(np.concatenate, (hits_j, hits_k, hits_x))
        # count crossings to the left of each voxel centre: parity -> inside
        solid = np.zeros(n, bool)
        order = np.lexsort((hx, hk, hj))
        hj, hk, hx = hj[order], hk[order], hx[order]
        keys = hj * n[2] + hk
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for seg in np.split(np.arange(len(keys)), bounds):
            j, k = hj[seg[0]], hk[seg[0]]
            xs = hx[seg]
            cnt = np.searchsorted(xs, ax[0])
            solid[:, j, k] = (cnt % 2) == 1
        d_out = ndimage.distance_transform_edt(~solid) * res
        d_in = ndimage.distance_transform_edt(solid) * res
        self.solid = solid
        self.phi = np.where(solid, -d_in + 0.5 * res, d_out - 0.5 * res)
        self.hi = self.lo + res * (n - 1)
        self.solid_fraction = solid.mean()

    def filled(self):
        """Copy with every through-hole along local x closed (outline extruded across the
        mesh's x extent): a solid plate with the same planform."""
        from scipy import ndimage
        out = object.__new__(MeshSDF)
        out.lo, out.hi, out.res = self.lo, self.hi, self.res
        cols = self.solid.any(axis=0)
        cols = ndimage.binary_fill_holes(cols)
        xs = np.flatnonzero(self.solid.any(axis=(1, 2)))
        solid = np.zeros_like(self.solid)
        solid[xs.min():xs.max() + 1] = cols[None]
        d_out = ndimage.distance_transform_edt(~solid) * self.res
        d_in = ndimage.distance_transform_edt(solid) * self.res
        out.solid = solid
        out.phi = np.where(solid, -d_in + 0.5 * self.res, d_out - 0.5 * self.res)
        out.solid_fraction = solid.mean()
        return out

    def __call__(self, x, y, z):
        """Evaluate in mesh coordinates. Outside the table: distance to its bounding box."""
        from scipy.ndimage import map_coordinates
        pts = np.stack([x, y, z])
        f = [(pts[d] - self.lo[d]) / self.res for d in range(3)]
        inside_box = np.all([(pts[d] >= self.lo[d]) & (pts[d] <= self.hi[d]) for d in range(3)], axis=0)
        out = np.empty(np.shape(x))
        if inside_box.any():
            out[inside_box] = map_coordinates(self.phi, [fi[inside_box] for fi in f], order=1)
        if (~inside_box).any():
            c, hh = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
            p = [pts[d][~inside_box] - c[d] for d in range(3)]
            out[~inside_box] = box_local(p[0], p[1], p[2], *hh) + self.phi.max()
        return out


def capsule(X, Y, Z, p0, p1, r):
    """Round tube of radius r between points p0 and p1 (exact distance)."""
    p0 = np.asarray(p0, float)
    d = np.asarray(p1, float) - p0
    L2 = float(d @ d)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1] + (Z - p0[2]) * d[2]) / max(L2, 1e-12), 0.0, 1.0)
    return np.sqrt((X - p0[0] - t * d[0]) ** 2 + (Y - p0[1] - t * d[1]) ** 2 + (Z - p0[2] - t * d[2]) ** 2) - r
