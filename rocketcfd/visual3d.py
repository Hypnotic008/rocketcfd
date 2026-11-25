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


def write_vtk_surface(path, xyz, faces, cp, scalars=None, vectors=None):
    """Legacy binary VTK polydata with Cp (and optional extra point scalars / vectors)."""
    with open(path, "wb") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nrocketcfd surface Cp\nBINARY\nDATASET POLYDATA\n"
                 f"POINTS {len(xyz)} float\n".encode())
        fh.write(xyz.astype(">f4").tobytes())
        fh.write(f"\nPOLYGONS {len(faces)} {4 * len(faces)}\n".encode())
        conn = np.hstack([np.full((len(faces), 1), 3), faces]).astype(">i4")
        fh.write(conn.tobytes())
        fh.write(f"\nPOINT_DATA {len(xyz)}\nSCALARS Cp float 1\nLOOKUP_TABLE default\n".encode())
        fh.write(cp.astype(">f4").tobytes())
        for name, val in (scalars or {}).items():
            fh.write(f"\nSCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            fh.write(np.asarray(val).astype(">f4").tobytes())
        for name, val in (vectors or {}).items():
            fh.write(f"\nVECTORS {name} float\n".encode())
            fh.write(np.asarray(val).astype(">f4").tobytes())


def vertex_normals(xyz, faces):
    fn = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]])
    vn = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30)


def surface_friction(solver, xyz, faces, q_inf, offset):
    """Wall shear on the body surface from the law of the wall (same as the solver's wall
    function): velocity sampled `offset` metres out along the surface normal. Returns
    (Cf magnitude, wall-shear vector [Pa]) per vertex."""
    from scipy.ndimage import map_coordinates
    from .wallfunction import spalding_utau
    n = vertex_normals(xyz, faces)
    # marching cubes on phi gives normals pointing toward increasing phi (into the fluid)
    pts = xyz + offset * n
    ax = solver.axes
    idx = np.stack([np.interp(pts[:, d], ax[d], np.arange(len(ax[d]))) for d in range(3)])
    W = solver.primitive()
    rho = map_coordinates(W[0], idx, order=1)
    vel = np.stack([map_coordinates(W[1 + c], idx, order=1) for c in range(3)], 1)
    p = map_coordinates(W[4], idx, order=1)
    v = solver.viscous
    T = p / (rho * v["R"])
    mu = v["mu"] if v["model"] == "constant" else v["mu"] * (T / v["T0"]) ** 1.5 * (v["T0"] + v["S"]) / (T + v["S"])
    ut = vel - (vel * n).sum(1, keepdims=True) * n
    um = np.linalg.norm(ut, axis=1)
    utau = spalding_utau(um, offset, mu / rho)
    tau = rho * utau ** 2
    shear = tau[:, None] * ut / np.maximum(um, 1e-12)[:, None]
    return tau / q_inf, shear


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


def render_surface(path, xyz, faces, cp, title="", fin_polys=(), views=None, clim=None, windward=(0.0, 1.0),
                   label="Cp  (pressure coefficient)", cmap_name="RdBu_r", diverging=True):
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
    if diverging:
        norm = TwoSlopeNorm(0.0, min(lo, -0.05), max(hi, 0.05))
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(max(lo, 0.0), hi)
    cmap = plt.get_cmap(cmap_name)
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
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, label=label, location="bottom", pad=0.03)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(path, title, hist, avg, std, parts, fin_names, ref, cells, steps, minutes):
    """One-page run summary: load history, where the loads come from, per-fin contributions,
    and the key numbers next to CatchSim's."""
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    h = np.array([(r[0], r[2], r[3], r[4]) for r in hist])
    ax = fig.add_subplot(gs[0, 0:2])
    if len(h):
        ax.plot(h[:, 0], h[:, 1], label="CD")
        ax.plot(h[:, 0], h[:, 2], label="CN")
        ax2 = ax.twinx()
        ax2.plot(h[:, 0], h[:, 3], color="C3", label="restoring moment")
        ax2.axhline(0.0, color="C3", lw=0.6, ls=":")
        ax2.set_ylabel("restoring moment", color="C3")
        ax.set_xlabel("step")
        ax.set_ylabel("CD, CN")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    ax.set_title("how the loads settled during the run")
    names = [n for n in parts if n not in fin_names]
    for col, key, lab in ((0, "CD", "drag CD"), (1, "CN", "normal force CN"), (2, "C_restore", "restoring moment")):
        a = fig.add_subplot(gs[1, col])
        vals = [avg.get(f"{key}[{n}]", 0.0) for n in names]
        colors = ["C2" if v >= 0 else "C3" for v in vals] if key == "C_restore" else ["C0"] * len(vals)
        a.barh(names, vals, color=colors)
        a.axvline(0.0, color="k", lw=0.6)
        a.set_title(f"{lab} by region  (total {avg[key]:+.4f})")
        for y_, v in enumerate(vals):
            a.text(v, y_, f" {v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    a = fig.add_subplot(gs[0, 2])
    a.axis("off")
    lines = [f"cells {cells/1e6:.1f} M   steps {steps}   {minutes:.0f} min", "",
             f"CD          {avg['CD']:.3f} +/- {std['CD']:.3f}",
             f"CN (lift)   {avg['CN']:.3f} +/- {std['CN']:.3f}",
             f"restoring   {avg['C_restore']:+.4f} +/- {std['C_restore']:.4f}",
             "   " + ("restoring: engines-first stable" if avg["C_restore"] > 0 else "destabilising"), ""]
    if fin_names:
        lines.append("per fin, restoring moment:")
        for n in fin_names:
            lines.append(f"   {n:14s} {avg.get(f'C_restore[{n}]', 0.0):+.4f}")
        lines.append("")
    if ref:
        lines += ["CatchSim (quatsim/aero.py) now:", f"   drag (with fin factor) {ref['cd_with_fins']:.2f}",
                  f"   cn_alpha {ref['cn_alpha']}   cm_alpha {ref['cm_alpha']}"]
    a.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_friction_lines(path, surface_vtk, fin_meshes=(), windward=(0.0, 1.0), title=""):
    """Skin-friction magnitude on the hull with friction lines (surface streamlines of the wall
    shear). Lines converging mark separation, diverging lines attachment. Needs pyvista."""
    import pyvista as pv
    surf = pv.read(surface_vtk)
    if "wall_shear" not in surf.point_data:
        return False
    rng = np.random.default_rng(0)
    seeds = pv.PolyData(surf.points[rng.choice(surf.n_points, min(260, surf.n_points), replace=False)])
    try:
        lines = surf.streamlines_from_source(seeds, vectors="wall_shear", surface_streamlines=True,
                                             integration_direction="both", max_steps=700,
                                             initial_step_length=0.5, step_unit="cl")
    except Exception:                       # noqa: BLE001
        lines = None
    t = np.asarray(windward, float) / (np.hypot(*windward) + 1e-30)
    cf = np.asarray(surf.point_data["Cf"])
    lim = float(np.percentile(cf, 99))
    pl = pv.Plotter(off_screen=True, shape=(2, 1), window_size=(2000, 1100), border=False)
    for row, (side, lab) in enumerate(((1.0, "windward side"), (-1.0, "leeward side"))):
        pl.subplot(row, 0)
        pl.set_background("white")
        pl.add_mesh(surf, scalars="Cf", cmap="magma", clim=(0, lim), show_scalar_bar=(row == 0),
                    scalar_bar_args=dict(title="Cf (skin friction)", vertical=True))
        if lines is not None and lines.n_points:
            pl.add_mesh(lines, color="#3fb4ff", line_width=1.0, opacity=0.85)
        for m in fin_meshes:
            pl.add_mesh(m, color="#3b3d42")
        cam = np.array([36.0, 0.0, 0.0]) + side * 95.0 * np.array([0.0, t[0], t[1]])
        up = np.array([0.0, -t[1], t[0]])
        pl.camera_position = [tuple(cam), (36.0, 0.0, 0.0), tuple(up)]
        pl.camera.zoom(1.9)
        pl.add_text(f"{lab}: skin friction and friction lines", font_size=12, color="black")
    pl.screenshot(path)
    pl.close()
    return True
