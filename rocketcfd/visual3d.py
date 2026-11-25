"""Visual output for whole-vehicle runs: cross-section stations, 3-D surface pressure, VTK."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_stations(f, path, stations, p_inf, q_inf, title="", half=11.0, crossflow=None, fin_sdf=None,
                  labels=None):
    """Pressure coefficient and Mach on y-z cross-sections at body stations.
    stations: list of (x, label). crossflow: (vy, vz) of the freestream, drawn as an arrow
    from the windward side. fin_sdf(X, Y, Z): thin parts drawn as outlines (they are walls
    between cells, so they do not show as solid cells)."""
    y, z = f["y"], f["z"]
    jy = (y > -half) & (y < half)
    kz = (z > -half) & (z < half)
    yy, zz = y[jy], z[kz]
    n = len(stations)
    fig, axs = plt.subplots(n, 2, figsize=(11, 4.6 * n), constrained_layout=True)
    axs = np.atleast_2d(axs)
    cp_all = []
    for x0, _ in stations:
        i = int(np.argmin(np.abs(f["x"] - x0)))
        cp_all.append(((f["p"][i][np.ix_(jy, kz)] - p_inf) / q_inf)[~f["solid"][i][np.ix_(jy, kz)]])
    cmax = max(0.2, float(np.percentile(np.concatenate(cp_all), 99.5)))
    cmin = min(-0.2, float(np.percentile(np.concatenate(cp_all), 0.5)))
    for row, (x0, lab) in enumerate(stations):
        i = int(np.argmin(np.abs(f["x"] - x0)))
        solid = f["solid"][i][np.ix_(jy, kz)]
        cp = np.ma.masked_where(solid, (f["p"][i][np.ix_(jy, kz)] - p_inf) / q_inf)
        mach = np.ma.masked_where(solid, f["mach"][i][np.ix_(jy, kz)])
        for col, (data, cmap, clab, norm) in enumerate(((cp, "RdBu_r", "Cp", TwoSlopeNorm(0.0, cmin, cmax)),
                                                         (mach, "turbo", "Mach", None))):
            ax = axs[row, col]
            im = ax.pcolormesh(yy, zz, data.T, cmap=cmap, norm=norm, shading="auto")
            ax.contour(yy, zz, f["phi"][i][np.ix_(jy, kz)].T, levels=[0.0], colors="k", linewidths=1.0)
            if fin_sdf is not None:
                Yf, Zf = np.meshgrid(np.linspace(-half, half, 1101), np.linspace(-half, half, 1101), indexing="ij")
                dfin = fin_sdf(np.full_like(Yf, f["x"][i]), Yf, Zf)
                if (dfin < 0).any():
                    ax.contour(Yf[:, 0], Zf[0], dfin.T, levels=[0.0], colors="k", linewidths=0.6)
            ax.set_aspect("equal")
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_xlabel("y [m]  (catch fins at +/-y)")
            ax.set_ylabel("z [m]  (rudder fin +z, tower side -z)")
            ax.set_title(f"x = {f['x'][i]:.1f} m: {lab}  ({clab})")
            fig.colorbar(im, ax=ax, shrink=0.8, label=clab)
            if crossflow is not None and np.hypot(*crossflow) > 1e-9:
                d = np.array(crossflow) / np.hypot(*crossflow)
                ax.annotate("", xy=-d * 6.2, xytext=-d * 10.0,
                            arrowprops=dict(arrowstyle="->", lw=2.0, color="k"))
                ax.text(*(-d * 10.4), "crossflow", ha="center", va="center", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def surface_pressure(solver, q_inf, region, stride=1):
    """Body surface (phi = 0 of the hull) with Cp at each vertex.
    region: ((x0, y0, z0), (x1, y1, z1)) to triangulate. Returns (vertices, faces, cp)."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import map_coordinates
    ax = solver.axes
    rng = [np.flatnonzero((ax[d] >= region[0][d]) & (ax[d] <= region[1][d])) for d in range(3)]
    sl = tuple(slice(r[0], r[-1] + 1, stride) for r in rng)
    phi = solver.phi[sl]
    v, fcs, _, _ = marching_cubes(phi, 0.0)
    idx = v * stride + np.array([r[0] for r in rng])            # fractional padded indices
    W = solver.primitive()
    p = map_coordinates(W[4], idx.T, order=1)
    xyz = np.stack([np.interp(idx[:, d], np.arange(len(ax[d])), ax[d]) for d in range(3)], 1)
    cp = (p - solver.W_inf[4]) / q_inf
    return xyz, fcs, cp


def write_vtk_surface(path, xyz, faces, cp):
    """Legacy binary VTK polydata with Cp, for ParaView."""
    with open(path, "wb") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nrocketcfd surface Cp\nBINARY\nDATASET POLYDATA\n"
                 f"POINTS {len(xyz)} float\n".encode())
        fh.write(xyz.astype(">f4").tobytes())
        fh.write(f"\nPOLYGONS {len(faces)} {4 * len(faces)}\n".encode())
        conn = np.hstack([np.full((len(faces), 1), 3), faces]).astype(">i4")
        fh.write(conn.tobytes())
        fh.write(f"\nPOINT_DATA {len(xyz)}\nSCALARS Cp float 1\nLOOKUP_TABLE default\n".encode())
        fh.write(cp.astype(">f4").tobytes())


def fin_outline_polys(P, fin):
    """Fin outlines (octagons) in body coordinates, for drawing the fins on 3-D views."""
    from .vehicles.super_heavy import _az
    w, r0, c = fin.w, fin.root_half, fin.chamfer
    yz = [(-r0, fin.z_root), (r0, fin.z_root), (w, fin.z_cross), (w, fin.z_tip - c), (w - c, fin.z_tip),
          (-(w - c), fin.z_tip), (-w, fin.z_tip - c), (-w, fin.z_cross)]
    polys = []
    for az in P.fin_azimuths:
        th = _az(az)
        for xo in (-0.5 * fin.depth, 0.5 * fin.depth):
            pts = []
            for t, zr in yz:
                r = P.radius + zr
                pts.append((P.fin_x_center + xo, r * np.cos(th) - t * np.sin(th), r * np.sin(th) + t * np.cos(th)))
            polys.append(pts)
    return polys


def render_surface(path, xyz, faces, cp, title="", fin_polys=(), views=None, clim=None, windward=(0.0, 1.0)):
    """windward: unit (y, z) direction from the axis toward the side the crossflow comes from.
    The scene is rotated about the booster axis so that side faces the first view's camera."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    t = np.asarray(windward, float) / (np.hypot(*windward) + 1e-30)
    perp = np.array([-t[1], t[0]])

    def rot(pts):
        pts = np.asarray(pts, float)
        out = pts.copy()
        out[..., 1] = -(pts[..., 1] * t[0] + pts[..., 2] * t[1])
        out[..., 2] = pts[..., 1] * perp[0] + pts[..., 2] * perp[1]
        return out
    xyz = rot(xyz)
    fin_polys = [rot(pl) for pl in fin_polys]
    tri = xyz[faces]
    cpf = cp[faces].mean(1)
    lo, hi = clim if clim else (float(np.percentile(cpf, 1)), float(np.percentile(cpf, 99)))
    norm = TwoSlopeNorm(0.0, min(lo, -0.05), max(hi, 0.05))
    cmap = plt.get_cmap("RdBu_r")
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    cen = tri.mean(1)
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    # (title, elev, azim, x-range or None, grid slot)
    views = views or [("windward side (facing the crossflow)", 6, -90, None, (0, slice(0, 2))),
                      ("leeward side", 6, 90, None, (1, slice(0, 2))),
                      ("engine end", 18, -140, (xmin - 0.5, xmin + 9.0), (2, 0)),
                      ("grid fins and top", 18, -40, (xmax - 11.0, xmax + 0.5), (2, 1))]
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 0.55, 1.4])
    for vt, el, az, xr, slot in views:
        if xr is None and el < 10:
            # full-length side view: flat projection (painter's algorithm) so the 72 m booster
            # fills the panel; camera on the -y (az -90) or +y (az 90) side of the rotated frame
            a = fig.add_subplot(gs[slot[0], slot[1]])
            sgn = -1.0 if az < 0 else 1.0
            facing = (nrm[:, 1] * sgn) > 0
            depth = -sgn * cen[:, 1]
            order = np.argsort(depth)[::-1]
            order = order[facing[order] | True]
            shade = 0.55 + 0.45 * np.abs(nrm[order, 1])
            col = cmap(norm(cpf[order]))
            col[:, :3] *= shade[:, None]
            from matplotlib.collections import PolyCollection
            xs = tri[order][:, :, 0] if sgn < 0 else -tri[order][:, :, 0]
            a.add_collection(PolyCollection(np.stack([xs, tri[order][:, :, 2]], -1), facecolors=col,
                                            edgecolors="none"))
            for pl in fin_polys:
                pl = np.asarray(pl)
                a.fill(pl[:, 0] if sgn < 0 else -pl[:, 0], pl[:, 2], color=(0.25, 0.25, 0.27), alpha=0.9, lw=0)
            allx = np.concatenate([xs.ravel()] + [np.asarray(pl)[:, 0] * (1 if sgn < 0 else -1) for pl in fin_polys])
            a.set_xlim(allx.min() - 1, allx.max() + 1)
            a.set_ylim(-10.5, 10.5)
            a.set_aspect("equal")
            a.set_axis_off()
            a.set_title(vt + ("   (engines on the left)" if sgn < 0 else "   (engines on the right)"))
            continue
        a = fig.add_subplot(gs[slot[0], slot[1]], projection="3d")
        m = np.ones(len(tri), bool) if xr is None else (cen[:, 0] >= xr[0]) & (cen[:, 0] <= xr[1])
        eye = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                        np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
        shade = 0.55 + 0.45 * np.abs(nrm[m] @ eye)
        col = cmap(norm(cpf[m]))
        col[:, :3] *= shade[:, None]
        a.add_collection3d(Poly3DCollection(tri[m], facecolors=col, edgecolor="none"))
        fp = [pl for pl in fin_polys if xr is None or (xr[0] <= np.mean(np.asarray(pl)[:, 0]) <= xr[1])]
        if fp:
            a.add_collection3d(Poly3DCollection(fp, facecolors=(0.25, 0.25, 0.27, 0.85), edgecolor="k",
                                                linewidths=0.3))
        pts = tri[m].reshape(-1, 3)
        if fp:
            pts = np.vstack([pts, np.concatenate([np.asarray(pl) for pl in fp])])
        mn, mx = pts.min(0), pts.max(0)
        for f_, a_, b_ in zip((a.set_xlim, a.set_ylim, a.set_zlim), mn, mx):
            f_(a_, b_)
        a.set_box_aspect(tuple((mx - mn) / (mx - mn).max()), zoom=1.25 if xr is None else 1.0)
        a.view_init(elev=el, azim=az)
        a.set_axis_off()
        a.set_title(vt)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, label="Cp  (pressure coefficient)", location="bottom", pad=0.03)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
