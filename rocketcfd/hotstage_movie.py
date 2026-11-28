"""Hot-staging film: the ship's Raptors lighting on the booster's hot stage.

Left, 3-D: the booster's forward dome, each protection plate (white rims), the V3 truss and the
ship's aft end with its six engines, coloured by the gas temperature (or pressure) next to the
surface; the plume volume-rendered by gas temperature (exhaust-corrected, rocketcfd.hotstage).
Camera 'cutaway' (default): the near half of the truss, the ship's skirt and the plume removed, so
the plumes are seen hitting the dome and the plates; 'outside': the whole stack, the plume
venting between the truss tubes; 'orbit': once round the outside.
Right: gas temperature and pressure on the pitch plane through the axis, from the finest grid.

  python -m rocketcfd.cinematic results/hot_staging --mode hotstage
  python -m rocketcfd.cinematic results/hot_staging --mode hotstage --quantity pressure --camera outside
"""
import os
import glob
import numpy as np
import pyvista as pv

from . import hotstage as HS

C_BG = "#07080b"


def _snaps(folder):
    return sorted(glob.glob(os.path.join(folder, "snapshots", "snap_*.npz")))


def _gas_T(rho, dp, u, v, w, p_inf, th):
    p = np.asarray(dp, np.float64) + p_inf
    rho = np.asarray(rho, np.float64)
    Tm = p / (np.maximum(rho, 1e-9) * HS.R_AIR)
    T = HS.exhaust_temperature(Tm, u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2 + w.astype(np.float64) ** 2, **th)
    return np.where(rho >= th["rho_min"], T, th["T_inf"]), p


class _Field:
    """Gas temperature in the volume, from the base snapshot volume and the recorded patches."""

    def __init__(self, mesh, th):
        self.mesh, self.th = mesh, th
        self.np_ = int(mesh.get("n_patches", 0))

    def levels(self, snap):
        from .cinematic import _fill_solid
        m, th = self.mesh, self.th
        p_inf = float(m["p_inf"])
        out = []
        if "vol_rho" in snap:
            T, p = _gas_T(snap["vol_rho"], snap["vol_dp"], snap["vol_u"], snap["vol_v"], snap["vol_w"], p_inf, th)
            sol = m["vol_solid"]
            out.append(((m["vx"], m["vy"], m["vz"]), dict(T=_fill_solid(T, sol), p=_fill_solid(p, sol))))
        for k in range(self.np_):
            if f"p{k}_rho" not in snap:
                continue
            T, p = _gas_T(snap[f"p{k}_rho"], snap[f"p{k}_dp"], snap[f"p{k}_u"], snap[f"p{k}_v"], snap[f"p{k}_w"], p_inf, th)
            sol = m[f"p{k}_solid"]
            out.append(((m[f"p{k}_x"], m[f"p{k}_y"], m[f"p{k}_z"]), dict(T=_fill_solid(T, sol), p=_fill_solid(p, sol))))
        return out


def _plane(mesh, snap, th, xr, zr, h=0.03):
    """Gas temperature and pressure on the x-z plane (y = 0) on a uniform image, finest grid wins."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    xs = np.arange(xr[0], xr[1], h)
    zs = np.arange(zr[0], zr[1], h)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    P = np.column_stack([X.ravel(), Z.ravel()])
    T_img = np.full(X.size, np.nan)
    p_img = np.full(X.size, np.nan)
    sol_img = np.zeros(X.size, bool)
    p_inf = float(mesh["p_inf"])

    def put(a_, b_, rho, pp, V, solid, hcell):
        pp = np.asarray(pp, np.float64)
        rho = np.asarray(rho, np.float64)
        V = np.asarray(V, np.float64)
        T = HS.exhaust_temperature(pp / (np.maximum(rho, 1e-9) * HS.R_AIR), V * V, **th)
        T = np.where(rho >= th["rho_min"], T, th["T_inf"])
        lo = np.array([a_[0], b_[0]]) - 0.51 * hcell
        hi = np.array([a_[-1], b_[-1]]) + 0.51 * hcell
        sel = np.all((P >= lo) & (P <= hi), axis=1)
        if not sel.any():
            return
        for arr, img in ((T, T_img), (pp, p_img)):
            img[sel] = RGI((a_, b_), arr, bounds_error=False, fill_value=None)(P[sel])
        sol_img[sel] = RGI((a_, b_), solid.astype(float), method="nearest", bounds_error=False, fill_value=None)(P[sel]) > 0.5

    if "p_xz" in snap:
        put(mesh["x"], mesh["z"], snap["rho_xz"], snap["p_xz"], snap["V_xz"], mesh["solid_xz"], 0.0)
    n = int(mesh.get("n_plane_patches", 0))
    order = []
    for k in range(n):
        if f"pp{k}_xz_p" in snap:
            a_ = mesh[f"pp{k}_xz_a"]
            order.append((-float(np.min(np.diff(a_))) if len(a_) > 1 else 0.0, k))
    for _, k in sorted(order):                                  # coarse first, finest last
        a_, b_ = mesh[f"pp{k}_xz_a"], mesh[f"pp{k}_xz_b"]
        put(a_, b_, snap[f"pp{k}_xz_rho"], snap[f"pp{k}_xz_p"], snap[f"pp{k}_xz_V"], mesh[f"pp{k}_xz_solid"],
            float(np.min(np.diff(a_))))
    sh = X.shape
    return xs, zs, T_img.reshape(sh), p_img.reshape(sh), sol_img.reshape(sh)


def _static_meshes(S, PB):
    """Booster tank below the dome and the ship's tank above its skirt, plain steel."""
    tank = pv.Cylinder(center=(PB.tank_top_x - 6.0, 0, 0), direction=(1, 0, 0), radius=S.radius, height=12.0,
                       resolution=160, capping=False)
    ship = pv.Cylinder(center=(S.x_ship_aft + S.skirt_height + 6.0, 0, 0), direction=(1, 0, 0), radius=S.radius,
                       height=12.0, resolution=160, capping=False)
    aft_bulkhead = pv.Disc(center=(S.x_ship_aft + S.skirt_height, 0, 0), normal=(1, 0, 0), inner=0.0, outer=S.radius, c_res=160)
    return [tank, ship, aft_bulkhead]


def _plate_rims(S, PB):
    """The protection plates' edges as thin white tubes on the dome (they read as separate plates)."""
    from .vehicles.starship import dome_plate_outlines
    rims = []
    for cs in dome_plate_outlines(S, h=0.02).values():
        for c in cs:
            c = c[:: max(1, len(c) // 300)]
            y, z = c[:, 0], c[:, 1]
            rho = np.hypot(y, z) / S.radius
            x = PB.tank_top_x + PB.dome_rise * np.sqrt(np.clip(1 - rho ** 2, 0, 1)) + S.plate_standoff + 0.01
            pts = np.column_stack([x, y, z])
            line = pv.lines_from_points(np.vstack([pts, pts[:1]]))
            rims.append(line.tube(radius=0.03))
    return rims


def render_hotstage(folder, out, quantity="temperature", camera="cutaway", width=1920, height=1080, fps=30,
                    tween=3, start=0, still_frame=None, title=None, progress=None, volume=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .vehicles.starship import StarshipParams
    from .vehicles.super_heavy import v3_params
    from .cinematic import _FinestGrid
    files = _snaps(folder)[start:]
    if not files:
        raise SystemExit(f"no snapshots in {folder}/snapshots (run cases/hot_staging.py with --snapshots)")
    mesh = dict(np.load(os.path.join(folder, "snapshots", "mesh.npz")))
    hm = HS.load_mesh(os.path.join(folder, "snapshots")) if os.path.exists(
        os.path.join(folder, "snapshots", "hotstage_mesh.npz")) else HS.load_mesh(folder)
    th = {k: float(hm[k]) for k in ("h0_inf", "h0_jet", "cp_exh", "dh_exh", "rho_min", "T_inf")}
    import json
    meta = {}
    if os.path.exists(os.path.join(folder, "loads.json")):
        with open(os.path.join(folder, "loads.json")) as fh:
            meta = json.load(fh)
    ring = meta.get("grid", {}).get("ring", "v3")
    geo = meta.get("geometry", {})
    S = StarshipParams(ring_style="v3_truss" if ring == "v3" else "vented", sl_cant=float(geo.get("sl_cant", 0.0)),
                       vac_cant=float(geo.get("vac_cant", 0.0)), sep_gap=float(geo.get("gap", 0.0)))
    PB = v3_params(fin_model="none")
    field = _Field(mesh, th)

    # surfaces (fixed): one PolyData per part
    parts = []
    for name, sl, f in HS.part_slices(hm):
        v = hm["v"][sl]
        if len(f) == 0:
            continue
        pd = pv.PolyData(v.astype(np.float64), np.column_stack([np.full(len(f), 3), f]).ravel())
        parts.append((name, sl, pd))
    rims = _plate_rims(S, PB) if ring == "v3" else []
    statics = _static_meshes(S, PB)
    cut = camera == "cutaway"
    if cut:                                   # remove the near half (y < 0, camera side)
        def keep_far(pd):
            return pd.clip(normal=(0, 1, 0), origin=(0, 0.0, 0), invert=False) if pd.n_points else pd   # keeps y > 0
        parts = [(n, sl, keep_far(pd) if n in ("truss", "ship aft") else pd) for n, sl, pd in parts]
        statics = [keep_far(s_) for s_ in statics]
    # per-part vertex map after clipping (clip keeps 'vtkOriginalPointIds' only when asked: map by nearest)
    from scipy.spatial import cKDTree
    maps = []
    for name, sl, pd in parts:
        src = hm["v"][sl]
        maps.append(cKDTree(src).query(np.asarray(pd.points))[1])

    if quantity == "temperature":
        qkey, cmap, clim, unit, log = "hs_T", "inferno", [300.0, float(hm.get("T_c", 3600.0))], "K", False
        qlab = "Gas temperature at the wall"
    else:
        qkey, cmap, clim, unit, log = "hs_p", "viridis", [1.0, 2000.0], "kPa", True
        qlab = "Pressure at the wall"

    # framing
    xc = 0.5 * (PB.tank_top_x + S.x_ship_aft + 1.0)
    focus = np.array([xc, 0.0, 0.0])
    box = (PB.tank_top_x - 1.5, S.x_ship_aft + S.skirt_height + 1.0, -6.5, 6.5, -6.5, 6.5)
    W3 = int(width * 2 / 3)
    WP = width - W3

    def cam(sfrac):
        if camera == "orbit":
            a = np.radians(-100 + 200 * sfrac)
            pos = focus + np.array([3.5, 24 * np.sin(a), 24 * np.cos(a) * 0.55 + 6])
        elif cut:
            pos = focus + np.array([9.0, -25.0, 9.0])
        else:
            pos = focus + np.array([6.0, -30.0, 11.0])
        return [tuple(pos), tuple(focus), (1, 0, 0)]

    pl = pv.Plotter(off_screen=True, window_size=(W3, height))
    pl.set_background(C_BG)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:                                     # noqa: BLE001
        pass
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(focus[0] + 30, -30, 30), focal_point=tuple(focus), intensity=0.85))
    pl.add_light(pv.Light(position=(focus[0] - 20, 30, -10), focal_point=tuple(focus), intensity=0.35, color="#bcd4ff"))
    pl.add_light(pv.Light(light_type="headlight", intensity=0.25))

    # time axis: snapshots with tweening
    snaps_cache = {}

    def snap(i):
        if i not in snaps_cache:
            if len(snaps_cache) > 3:
                snaps_cache.pop(min(snaps_cache))
            snaps_cache[i] = dict(np.load(files[i]))
        return snaps_cache[i]

    def blend(a_, b_, f_):
        return a_ if f_ == 0 else (1 - f_) * a_.astype(np.float32) + f_ * b_.astype(np.float32)

    n = len(files)
    frames = [(i, k / tween) for i in range(n - 1) for k in range(tween)] + [(n - 1, 0.0)]
    if still_frame is not None:
        frames = [frames[min(still_frame, len(frames) - 1)]]
    t_ign = float(meta.get("jets", {}).get("ignition", 0.02))

    writer = None
    if still_frame is None:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=None, bitrate=None,
                                    ffmpeg_params=["-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p"])
    xs_r, zs_r = (PB.tank_top_x - 2.0, S.x_ship_aft + S.skirt_height + 2.5), (-7.0, 7.0)
    peak = None
    for fi, (i, fr) in enumerate(frames):
        A = snap(i)
        B = snap(min(i + 1, n - 1)) if fr else A
        t = (1 - fr) * float(A["t"]) + fr * float(B["t"])
        qv = blend(A[qkey], B[qkey], fr).astype(np.float64)
        Tv = blend(A["hs_T"], B["hs_T"], fr).astype(np.float64)
        pv_ = blend(A["hs_p"], B["hs_p"], fr).astype(np.float64)
        if quantity != "temperature":
            qv = qv / 1e3
        pl.clear_actors()
        for (name, sl, pd), mp in zip(parts, maps):
            vals = qv[sl][mp]
            # no hot gas next to the wall (thin gas, or outside the plume): plain steel
            pd.point_data["q"] = np.where(np.isfinite(vals), np.maximum(vals, clim[0]), np.nan)
            pl.add_mesh(pd, scalars="q", cmap=cmap, clim=clim, log_scale=log, smooth_shading=True,
                        nan_color="#7d838b", specular=0.35 if name.startswith("plate") else 0.2, specular_power=25,
                        show_scalar_bar=False, render=False)
        for r_ in rims:
            pl.add_mesh(r_, color="#f2f2f2", smooth_shading=True, render=False)
        for s_ in statics:
            pl.add_mesh(s_, color="#8e949c", metallic=0.6, roughness=0.45, pbr=True, smooth_shading=True, render=False)
        if volume:
            levs_a = field.levels(A)
            levs = levs_a if not fr else [((ax_), {k: blend(fa[k], fb[k], fr) for k in fa})
                                          for (ax_, fa), (_, fb) in zip(levs_a, field.levels(B))]
            if levs:
                hv = 0.09
                dims = [int((box[2 * d + 1] - box[2 * d]) / hv) + 1 for d in range(3)]
                if cut:
                    dims[1] = int((box[3] - 0.0) / hv) + 1
                img = pv.ImageData(dimensions=dims, spacing=(hv, hv, hv),
                                   origin=(box[0], 0.0 if cut else box[2], box[4]))
                Tg = _FinestGrid(levs)("T", np.asarray(img.points))
                img.point_data["T"] = np.clip(np.nan_to_num(Tg, nan=th["T_inf"]), 200, 4000).astype(np.float32)
                pl.add_volume(img, scalars="T", cmap="inferno", clim=[600.0, 3800.0],
                              opacity=[0.0, 0.0, 0.015, 0.05, 0.11, 0.2], shade=False, mapper="fixed_point",
                              show_scalar_bar=False, render=False)
        pl.camera_position = cam(fi / max(len(frames) - 1, 1))
        pl.camera.view_angle = 30.0
        img3 = pl.screenshot(return_img=True)

        # ---- right column: the pitch plane
        xs, zs, Ti, Pi, Si = _plane(mesh, A, th, xs_r, zs_r)
        dpi = 100
        fig = plt.figure(figsize=(WP / dpi, height / dpi), dpi=dpi, facecolor=C_BG)
        for row, (arr, cm, lab, norm) in enumerate(((Ti, "inferno", "Gas temperature [K]", matplotlib.colors.Normalize(300, 3800)),
                                                    (Pi / 1e3, "viridis", "Pressure [kPa]", matplotlib.colors.LogNorm(0.1, 2000)))):
            ax = fig.add_axes([0.13, 0.55 - 0.47 * row + 0.03, 0.70, 0.38])
            a2 = np.where(Si, np.nan, arr)
            im = ax.imshow(a2.T, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap=cm, norm=norm,
                           interpolation="bilinear", aspect="equal")
            ax.imshow(np.where(Si, 1.0, np.nan).T, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]],
                      cmap=matplotlib.colors.ListedColormap(["#9aa0a8"]), aspect="equal")
            ax.set_facecolor(C_BG)
            for s_ in ax.spines.values():
                s_.set_color("#30343a")
            ax.tick_params(colors="#9c9c9c", labelsize=8)
            ax.set_xlabel("x [m]", color="#9c9c9c", fontsize=8)
            ax.set_ylabel("z [m]", color="#9c9c9c", fontsize=8)
            ax.set_title(lab + " — pitch plane", color="#efefef", fontsize=10, loc="left")
            cax = fig.add_axes([0.86, 0.55 - 0.47 * row + 0.03, 0.03, 0.38])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(colors="#9c9c9c", labelsize=7)
            cb.outline.set_edgecolor("#30343a")
        fig.canvas.draw()
        img2 = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :W3] = img3[:height, :W3, :3]
        frame[:img2.shape[0], W3:W3 + img2.shape[1]] = img2[:height, :WP]
        # overlays on the 3-D view (matplotlib text on the composited frame)
        thr = min(1.0, t / max(t_ign, 1e-9))
        names = hm["names"]
        def part_stat(prefix, arr, fn):
            vals = [arr[sl] for nm, sl, _ in HS.part_slices(hm) if nm.startswith(prefix)]
            v = np.concatenate(vals) if vals else np.array([np.nan])
            v = v[np.isfinite(v)]
            return fn(v) if v.size else np.nan
        Tp = part_stat("plate", Tv, lambda v: np.percentile(v, 99.5))
        Pp = part_stat("plate", pv_, lambda v: np.percentile(v, 99.5)) / 1e3
        Td = part_stat("dome", Tv, lambda v: np.percentile(v, 99.5))
        Tt = part_stat("truss", Tv, lambda v: np.percentile(v, 99.5))
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.figimage(frame, 0, 0, origin="upper")
        txt = title or ("Hot staging — Starship's 6 Raptor 3s lighting on the V3 hot stage"
                        + (f", sea-level engines canted {S.sl_cant:g}° out" if S.sl_cant else "")
                        + (f", {S.sep_gap:g} m apart" if S.sep_gap else ""))
        fig.text(0.012, 0.955, txt, color="#efefef", fontsize=15, fontweight="bold")
        fig.text(0.012, 0.92, f"t = {t * 1e3:6.1f} ms after ignition command   ship engines {100 * thr:3.0f} %   "
                               f"Mach {meta.get('condition', {}).get('mach', 4.5):.1f}, "
                               f"{meta.get('condition', {}).get('alt', 65e3) / 1e3:.0f} km", color="#c8c8c8", fontsize=11,
                 family="monospace")
        fig.text(0.012, 0.06, f"protection plates: {Tp:5.0f} K  {Pp:6.0f} kPa   bare dome: {Td:5.0f} K   truss: {Tt:5.0f} K"
                              "   (99.5th percentile, gas next to the wall)", color="#c8c8c8", fontsize=10.5, family="monospace")
        fig.text(0.012, 0.03, f"surfaces: {qlab.lower()} [{unit}]   plume: gas temperature (volume)   "
                              f"white rims: dome protection plates" + ("   (cutaway: near half removed)" if cut else ""),
                 color="#8c8c8c", fontsize=9.5)
        # colour bar for the surfaces
        cax = fig.add_axes([0.62, 0.11, 0.012, 0.3])
        nm_ = matplotlib.colors.LogNorm(*clim) if log else matplotlib.colors.Normalize(*clim)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=nm_, cmap=cmap), cax=cax)
        cb.ax.tick_params(colors="#c8c8c8", labelsize=8)
        cb.set_label(f"{qlab} [{unit}]", color="#c8c8c8", fontsize=8.5)
        fig.canvas.draw()
        out_img = np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3].copy()
        plt.close(fig)
        if writer is None:
            import imageio.v2 as imageio
            imageio.imwrite(out, out_img)
        else:
            writer.append_data(out_img)
        if progress:
            progress(fi + 1, len(frames))
    if writer is not None:
        writer.close()
    pl.close()
    return out
