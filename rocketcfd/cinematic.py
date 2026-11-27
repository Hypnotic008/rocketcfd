"""Cinematic renders and movies in the style of NASA's LAVA / Fun3D launch-vehicle videos:
black background, metallic body (or surface pressure), vortex surfaces (Q-criterion) coloured
by Mach number, a greyscale numerical-schlieren plane, soft lighting, slow camera moves.

Works on
  * a steady result (flow.vtk): camera orbit / fly-by around the frozen flow;
  * a run recorded with --snapshots N --snap-volume S: the flow evolving in time.

  python -m rocketcfd.cinematic results/glide_patch --mode orbit
  python -m rocketcfd.cinematic results/les_wake --mode evolve --style lava
  python -m rocketcfd.cinematic results/x --mode still --style schlieren --width 3840 --height 2160

Styles: lava (vortices coloured by Mach + schlieren plane), schlieren (schlieren plane only,
shadowgraph look), mach (Mach plane + vortices), pressure (surface Cp + Mach plane).
"""
import argparse
import glob
import json
import os

import numpy as np
import pyvista as pv

from . import derived
from .flowdata import FlowData

STYLES = ("lava", "schlieren", "mach", "pressure", "plume")


# ---------------------------------------------------------------- data
def _snap_grid(mesh, snap):
    x, y, z = mesh["vx"], mesh["vy"], mesh["vz"]
    g = pv.RectilinearGrid(x, y, z)
    p_inf = float(mesh["p_inf"])
    f = dict(rho=snap["vol_rho"].astype(np.float32), u=snap["vol_u"].astype(np.float32),
             v=snap["vol_v"].astype(np.float32), w=snap["vol_w"].astype(np.float32),
             p=snap["vol_dp"].astype(np.float32) + p_inf)
    solid = mesh["vol_solid"]
    for k in f:
        if k in ("u", "v", "w"):
            f[k] = np.where(solid, 0.0, f[k])
    gam = float(mesh.get("gamma", 1.4))
    f["mach"] = np.sqrt(f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2) / np.sqrt(gam * np.maximum(f["p"], 1e-6) / np.maximum(f["rho"], 1e-9))
    f["phi"] = np.where(solid, -1.0, 1.0).astype(np.float32)
    for k, a in f.items():
        g.point_data[k] = np.ascontiguousarray(a.transpose(2, 1, 0)).reshape(-1)
    g.point_data["velocity"] = np.column_stack([g.point_data["u"], g.point_data["v"], g.point_data["w"]])
    return g


def load_steady(folder):
    meta = {}
    mp = os.path.join(folder, "loads.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            meta = json.load(fh)
    c = meta.get("condition") or {}
    fs = dict(rho=c["rho"], p=c["p"], u=c["V"], v=0.0, w=0.0) if {"rho", "p", "V"} <= set(c) else None
    g = FlowData(folder, fs)                   # base grid + refinement patches (finest wins)
    surf = None
    sp = os.path.join(folder, "surface_cp.vtk")
    if os.path.exists(sp):
        surf = pv.read(sp)
    elif "phi" in g.base.point_data:
        surf = g.base.contour([0.0], scalars="phi")
    return g, surf, fs


def _surface_from_mesh(mesh, cp=None):
    xyz, faces = mesh["xyz"], mesh["faces"]
    f = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    s = pv.PolyData(np.asarray(xyz, float), f)
    if cp is not None:
        s.point_data["Cp"] = np.asarray(cp, np.float32)
    return s


# ---------------------------------------------------------------- scene
def auto_q(g, frac=0.985):
    q = g.values("Q") if isinstance(g, FlowData) else np.asarray(g.point_data["Q"])
    pos = q[q > 0]
    return float(np.quantile(pos, frac)) if pos.size else 0.0


def build(pl, flow, surf, style, q_level=None, plane_y=None, mach_range=None, cp_range=(-1.0, 1.0), fs=None,
          plane_axis=1, t_range=None):
    """Add the actors for one frame. Returns the Q level used. flow: a FlowData or one grid."""
    fd = flow if isinstance(flow, FlowData) else FlowData.from_grid(flow, fs)
    for n in (["schlieren", "Q"] if style != "pressure" else ["schlieren"]):
        fd.ensure(n)
    mrange = mach_range or [0.0, float(np.nanpercentile(fd.values("mach"), 99.5))]
    b = fd.bounds()
    # the plane: through the axis (or at plane_y), cut by the body
    normal = [0, 0, 0]
    normal[plane_axis] = 1
    origin = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    origin[plane_axis] = 0.0 if plane_y is None else plane_y
    sl = fd.slice(normal, origin)
    if surf is not None and surf.n_points:          # frame the plane around the vehicle
        sb = surf.bounds
        L = max(sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4])
        box = [sb[0] - 0.25 * L, sb[1] + 0.45 * L, sb[2] - 0.35 * L, sb[3] + 0.35 * L, sb[4] - 0.35 * L, sb[5] + 0.35 * L]
        box[2 * plane_axis], box[2 * plane_axis + 1] = b[2 * plane_axis], b[2 * plane_axis + 1]
        sl = sl.clip_box(box, invert=False)
    if "phi" in sl.point_data and np.any(np.asarray(sl.point_data["phi"]) < 0):
        sl = sl.threshold(0.0, scalars="phi")
    if style in ("lava", "schlieren", "plume"):
        pl.add_mesh(sl, scalars="schlieren", cmap="gray", clim=[0.05, 1.0], show_scalar_bar=False,
                    ambient=0.55, diffuse=0.45, specular=0.0, opacity=0.92 if style != "schlieren" else 1.0)
    else:
        pl.add_mesh(sl, scalars="mach", cmap="turbo", clim=mrange, show_scalar_bar=True,
                    scalar_bar_args=dict(title="Mach", color="white", vertical=True, position_x=0.92,
                                         position_y=0.2, height=0.6, width=0.03, fmt="%.2f"))
    ql = q_level
    if style in ("lava", "mach"):
        ql = q_level if q_level is not None else auto_q(fd)
        if ql > 0:
            iso = fd.contour(ql, "Q")
            if iso.n_points:
                if "mach" not in iso.point_data:
                    iso = fd.sample_onto(iso)
                iso = iso.smooth(n_iter=20) if iso.n_points < 3_000_000 else iso
                pl.add_mesh(iso, scalars="mach", cmap="turbo", clim=mrange, smooth_shading=True,
                            specular=0.5, specular_power=25, show_scalar_bar=(style == "lava"),
                            scalar_bar_args=dict(title="Mach", color="white", vertical=True, position_x=0.92,
                                                 position_y=0.2, height=0.6, width=0.03, fmt="%.2f"))
    if style == "plume":
        # hot exhaust: glowing temperature volume, and a hot-gas surface coloured by Mach
        fd.ensure("T")
        T_inf = (fs["p"] / (fs["rho"] * derived.R_AIR)) if fs else float(np.nanpercentile(fd.values("T"), 5))
        T_lo, T_hi = t_range if t_range is not None else plume_t_range(fd, T_inf)
        if T_hi > 1.5 * T_inf and T_hi > T_lo:
            box = [b[0], b[1], b[2], b[3], b[4], b[5]]
            n = 170
            ext = np.array([box[1] - box[0], box[3] - box[2], box[5] - box[4]])
            h = ext.max() / n
            dims = np.maximum((ext / h).astype(int) + 1, 2)
            img = pv.ImageData(dimensions=dims, spacing=(h, h, h), origin=(box[0], box[2], box[4]))
            img = fd.sample_onto(img)
            a_ = np.asarray(img.point_data["T"], float)
            a_[~np.isfinite(a_)] = T_inf
            img.point_data["T"] = np.clip(a_, T_inf, None)
            # glowing exhaust: transparent below T_lo, brighter and more opaque toward T_hi
            pl.add_volume(img, scalars="T", cmap="inferno", clim=[T_lo, T_hi],
                          opacity=[0.0, 0.03, 0.08, 0.16, 0.28, 0.45, 0.65], show_scalar_bar=False)
            # two nested hot-gas surfaces: a translucent outer flame and a bright core
            for frac, op in ((0.2, 0.3), (0.65, 0.85)):
                hot = fd.contour(T_lo + frac * (T_hi - T_lo), "T")
                if hot.n_points:
                    if "T" not in hot.point_data:
                        hot = fd.sample_onto(hot)
                    pl.add_mesh(hot, scalars="T", cmap="inferno", clim=[T_lo - 0.3 * (T_hi - T_lo), T_hi],
                                opacity=op, smooth_shading=True, ambient=0.45, specular=0.2,
                                show_scalar_bar=(frac > 0.5),
                                scalar_bar_args=dict(title="T [K]", color="white", vertical=True, position_x=0.92,
                                                     position_y=0.2, height=0.6, width=0.03, fmt="%.0f"))
    if surf is not None and surf.n_points:
        if style == "pressure" and "Cp" in surf.point_data:
            pl.add_mesh(surf, scalars="Cp", cmap="coolwarm", clim=list(cp_range), smooth_shading=True,
                        scalar_bar_args=dict(title="Cp", color="white", vertical=True, position_x=0.04,
                                             position_y=0.2, height=0.6, width=0.03))
        else:
            pl.add_mesh(surf, color="#d6dae0", ambient=0.22, diffuse=0.75, specular=0.55, specular_power=30,
                        smooth_shading=True)
    return ql


def plume_t_range(fd, T_inf):
    """Temperature range for the plume style: from the hot exhaust's upper end."""
    fd.ensure("T")
    Tv = fd.values("T")
    Tv = Tv[np.isfinite(Tv)]
    return max(1.5 * T_inf, float(np.percentile(Tv, 90))), float(np.percentile(Tv, 99.7))


def focus_bounds(fd, surf, style, fs=None):
    """Bounds the camera frames: the vehicle, plus the hot exhaust for the plume style."""
    bb = list(surf.bounds if surf is not None else fd.bounds())
    if style == "plume":
        T_inf = (fs["p"] / (fs["rho"] * derived.R_AIR)) if fs else None
        if T_inf is None:
            fd.ensure("T")
            T_inf = float(np.nanpercentile(fd.values("T"), 5))
        lo_T, _ = plume_t_range(fd, T_inf)
        pts = []
        for v in fd.visible():
            if "T" in v.point_data:
                m = np.asarray(v.point_data["T"]) > lo_T
                if m.any():
                    pts.append(np.asarray(v.points)[m])
        if pts:
            P = np.concatenate(pts)
            lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
            for d in range(3):
                bb[2 * d] = min(bb[2 * d], lo[d])
                bb[2 * d + 1] = max(bb[2 * d + 1], hi[d])
    return tuple(bb)


def setup_plotter(width, height):
    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background("black")
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:                       # noqa: BLE001
        pass
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(-60, -80, 60), focal_point=(36, 0, 0), intensity=0.9))
    pl.add_light(pv.Light(position=(120, 60, 40), focal_point=(36, 0, 0), intensity=0.5, color="#bcd4ff"))
    pl.add_light(pv.Light(light_type="headlight", intensity=0.25))
    return pl


def orbit_cameras(n, bb, elev=0.28, radius=1.25, start_deg=-120.0, sweep_deg=240.0):
    c = np.array([(bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, (bb[4] + bb[5]) / 2])
    L = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    cams = []
    for i in range(n):
        a = np.radians(start_deg + sweep_deg * i / max(n - 1, 1))
        pos = c + radius * L * np.array([np.sin(a) * 0.55, -np.cos(a), elev])
        cams.append([tuple(pos), tuple(c), (0, 0, 1)])
    return cams


def title_text(pl, text):
    if text:
        pl.add_text(text, position="upper_left", font_size=11, color="#c8ccd4")


# ---------------------------------------------------------------- products
def render_still(folder, out, style="lava", width=1920, height=1080, q=None, title=None, camera=None):
    flow, surf, fs = load_steady(folder)
    pl = setup_plotter(width, height)
    build(pl, flow, surf, style, q_level=q, fs=fs)
    bb = focus_bounds(flow, surf, style, fs)
    pl.camera_position = camera or orbit_cameras(1, bb, start_deg=-35, sweep_deg=0)[0]
    title_text(pl, title)
    pl.screenshot(out)
    pl.close()
    return out


def render_orbit(folder, out, style="lava", width=1920, height=1080, frames=240, fps=30, q=None, title=None):
    flow, surf, fs = load_steady(folder)
    pl = setup_plotter(width, height)
    build(pl, flow, surf, style, q_level=q, fs=fs)
    title_text(pl, title)
    bb = focus_bounds(flow, surf, style, fs)
    pl.open_movie(out, framerate=fps, quality=8)
    for cam in orbit_cameras(frames, bb):
        pl.camera_position = cam
        pl.write_frame()
    pl.close()
    return out


def render_evolve(folder, out, style="lava", width=1920, height=1080, fps=30, q=None, title=None, every=1,
                  orbit_deg=40.0, progress=None):
    sd = os.path.join(folder, "snapshots")
    mesh = dict(np.load(os.path.join(sd, "mesh.npz")))
    if "vx" not in mesh:
        raise SystemExit("these snapshots have no 3-D fields: rerun with --snapshots N --snap-volume 2")
    files = sorted(glob.glob(os.path.join(sd, "snap_*.npz")))[::every]
    if not files:
        raise SystemExit("no snapshots found")
    last = _snap_grid(mesh, np.load(files[-1]))
    derived.add_to_grid(last, ["Q"])
    ql = q if q is not None else auto_q(last)               # one level for the whole movie
    mrange = [0.0, float(np.nanpercentile(np.asarray(last.point_data["mach"]), 99.5))]
    trange = None
    if style == "plume":                                     # one temperature range for the whole movie
        fd_last = FlowData.from_grid(last)
        T_inf = float(mesh["p_inf"]) / (float(mesh.get("rho_inf", 1.0)) * derived.R_AIR)
        trange = plume_t_range(fd_last, T_inf)
    pl = setup_plotter(width, height)
    bb = (float(mesh["xyz"][:, 0].min()), float(mesh["xyz"][:, 0].max()), float(mesh["xyz"][:, 1].min()),
          float(mesh["xyz"][:, 1].max()), float(mesh["xyz"][:, 2].min()), float(mesh["xyz"][:, 2].max()))
    cams = orbit_cameras(len(files), bb, start_deg=-60.0, sweep_deg=orbit_deg)
    pl.open_movie(out, framerate=fps, quality=8)
    for i, fpath in enumerate(files):
        snap = np.load(fpath)
        g = _snap_grid(mesh, snap)
        surf = _surface_from_mesh(mesh, snap["cp"].astype(np.float32) if "cp" in snap else None)
        pl.clear_actors()
        build(pl, g, surf, style, q_level=ql, mach_range=mrange, t_range=trange)
        title_text(pl, (title or "") + f"   t = {float(snap['t']):.3f} s")
        pl.camera_position = cams[i]
        pl.write_frame()
        if progress:
            progress(i + 1, len(files))
    pl.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--mode", choices=["still", "orbit", "evolve", "schlieren", "flow", "flow-orbit"], default="still",
                    help="schlieren: 2-D side-on schlieren film of the snapshots (no model); flow: the 3-D model "
                         "with tracer streaks through the recorded flow; flow-orbit: the same on a steady result")
    ap.add_argument("--plane", choices=["xz", "xy"], default="xz", help="schlieren: plane through the axis")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("X0", "X1"),
                    help="schlieren: zoom to this axial window [m] (x from the engine end)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("Y0", "Y1"),
                    help="schlieren: zoom to this cross window [m] (recorded out to +/-15 m)")
    ap.add_argument("--camera", choices=["three-quarter", "fins", "flyby"], default="three-quarter",
                    help="flow: framing (flyby: wide, then a push in along the hull to the fins)")
    ap.add_argument("--no-vortices", action="store_true", help="flow: no vortex surfaces")
    ap.add_argument("--surface", choices=["cp", "metal"], default="cp", help="flow: vehicle coloured by Cp, or steel")
    ap.add_argument("--flow-plane", choices=["schlieren", "mach", "none"], default="schlieren",
                    help="flow: backdrop plane through the axis")
    ap.add_argument("--no-shocks", action="store_true", help="flow: no shock surfaces")
    ap.add_argument("--trail", type=int, default=14, help="flow: streak length in frames")
    ap.add_argument("--no-streaks", action="store_true", help="flow: no tracer streaks")
    ap.add_argument("--tween", type=int, default=1,
                    help="flow: frames per snapshot interval, fields blended between snapshots (smoother motion)")
    ap.add_argument("--volume", action="store_true",
                    help="flow: volume-render the density jump (3-D schlieren smoke) instead of shock sheets")
    ap.add_argument("--oil-flow", action="store_true",
                    help="flow: surface oil-flow lines on the hull (near-wall flow, separation lines)")
    ap.add_argument("--start", type=int, default=0,
                    help="flow / schlieren: skip this many snapshots at the start (a start-up transient)")
    ap.add_argument("--refine", type=int, default=None,
                    help="flow: cut the shock / vortex surfaces from the flow resampled this many times finer "
                         "(default: 2 on volumes recorded coarser than 0.45 m, else 1)")
    ap.add_argument("--still-frame", type=int, default=None, help="flow: write only this frame, as a PNG")
    ap.add_argument("--style", choices=STYLES, default="lava")
    ap.add_argument("--out", default=None, help="output file (default: in the results folder)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=240, help="orbit: number of frames")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=1, help="evolve: use every n-th snapshot")
    ap.add_argument("--q", type=float, default=None, help="Q-criterion level (default: automatic)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    ext = "png" if a.mode == "still" else "mp4"
    tag = a.style if a.mode in ("still", "orbit", "evolve") else (a.plane if a.mode == "schlieren" else a.camera)
    out = a.out or os.path.join(a.folder, f"cinematic_{a.mode}_{tag}.{ext}")
    prog = lambda i, n: print(f"frame {i}/{n}", flush=True)
    if a.mode == "schlieren":
        if (a.xlim or a.ylim) and not a.out:
            out = out[:-4] + "_zoom.mp4"
        render_schlieren2d(a.folder, out, a.plane, a.width, a.height, a.fps, title=a.title, progress=prog,
                           xlim=a.xlim, ylim=a.ylim, start=a.start)
    elif a.mode in ("flow", "flow-orbit"):
        if a.still_frame is not None and not a.out:
            out = out[:-4] + f"_f{a.still_frame}.png"
        render_flow(a.folder, out, a.width, a.height, a.fps, a.q, a.title, a.frames,
                    "evolve" if a.mode == "flow" else "orbit", prog, not a.no_vortices, a.camera,
                    surface=a.surface, plane=a.flow_plane, shocks=not a.no_shocks, trail=a.trail,
                    still_frame=a.still_frame, streaks=not a.no_streaks, refine=a.refine, start=a.start,
                    oil=a.oil_flow, volume=a.volume, tween=a.tween)
    elif a.mode == "still":
        render_still(a.folder, out, a.style, a.width, a.height, a.q, a.title)
    elif a.mode == "orbit":
        render_orbit(a.folder, out, a.style, a.width, a.height, a.frames, a.fps, a.q, a.title)
    else:
        render_evolve(a.folder, out, a.style, a.width, a.height, a.fps, a.q, a.title, a.every,
                      progress=lambda i, n: print(f"frame {i}/{n}", flush=True))
    print(f"wrote {out}")




# ---------------------------------------------------------------- vehicle model for movies
def vehicle_meshes(meta=None, h=0.06):
    """High-resolution booster body (cached STL, made once from the geometry) plus the grid fins at
    their deflections (rocketcfd.viewer.fin_meshes). Returns (body, [fins])."""
    from .viewer import fin_meshes
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"_body_v3_h{h:g}.stl")
    if not os.path.exists(path):
        from .vehicles.super_heavy import super_heavy_sdf, v3_params
        from .output import write_stl_chunked
        P = v3_params(fin_model="none")
        print(f"building the body model ({h} m, once) -> {path}", flush=True)
        write_stl_chunked(lambda X, Y, Z: super_heavy_sdf(X, Y, Z, P), (-0.8, -6.2, -6.2), (P.length + 0.2, 6.2, 6.2), h, path)
    body = pv.read(path).smooth(n_iter=15)
    fins = fin_meshes(meta or {"fins": {"model": "parametric"}})
    return body, fins


def add_vehicle(pl, body, fins, color="#c9ced6"):
    pl.add_mesh(body, color=color, ambient=0.22, diffuse=0.75, specular=0.5, specular_power=30, smooth_shading=True)
    for f in fins:
        pl.add_mesh(f, color="#9aa2ad", ambient=0.25, diffuse=0.7, specular=0.3)


# ---------------------------------------------------------------- 2-D schlieren movie (no model)
def _schlieren_layers(mesh, plane, snap0):
    """Every grid that has data on the plane through the axis, coarsest first:
    (name, a axis, b axis, solid 2-D, cell size, getter(snap) -> field 2-D, kind)."""
    b_key = "y" if plane == "xy" else "z"
    dens = f"rho_{plane}" in snap0
    layers = []
    # base grid: full-resolution plane (density on new runs, Mach on older ones)
    a_ax, b_ax = mesh["x"].astype(float), mesh[b_key].astype(float)
    key = f"rho_{plane}" if dens else f"mach_{plane}"
    h0 = float(min(np.min(np.diff(a_ax)), np.min(np.diff(b_ax))))
    layers.append(("base", a_ax, b_ax, mesh[f"solid_{plane}"].astype(bool), h0,
                   (lambda sn, k=key: sn[k].astype(np.float32).astype(float))))
    n_pp = int(mesh.get("n_plane_patches", 0))
    if n_pp and dens:
        for k in range(n_pp):
            if f"pp{k}_{plane}_a" not in mesh:
                continue
            a_, b_ = mesh[f"pp{k}_{plane}_a"].astype(float), mesh[f"pp{k}_{plane}_b"].astype(float)
            h = float(min(np.min(np.diff(a_)), np.min(np.diff(b_))))
            layers.append((f"pp{k}", a_, b_, mesh[f"pp{k}_{plane}_solid"].astype(bool), h,
                           (lambda sn, k=k: sn[f"pp{k}_{plane}_rho"].astype(np.float32).astype(float))))
    else:
        # older runs: slice the recorded patch volumes (strided) where they reach the plane
        gam = float(mesh.get("gamma", 1.4))
        p_inf = float(mesh["p_inf"])
        for k in range(int(mesh.get("n_patches", 0))):
            c = mesh[f"p{k}_{'z' if plane == 'xy' else 'y'}"].astype(float)
            if len(c) < 2:
                continue
            j = int(np.argmin(np.abs(c)))
            if abs(c[j]) > 0.5 * float(np.max(np.diff(c))) + 1e-9:
                continue
            a_, b_ = mesh[f"p{k}_x"].astype(float), mesh[f"p{k}_{b_key}"].astype(float)
            take = (lambda A, j=j: A[:, :, j]) if plane == "xy" else (lambda A, j=j: A[:, j, :])
            h = float(min(np.min(np.diff(a_)), np.min(np.diff(b_))))

            def get(sn, k=k, take=take):
                r = take(sn[f"p{k}_rho"]).astype(float)
                if dens:
                    return r
                u, v, w = (take(sn[f"p{k}_{c_}"]).astype(float) for c_ in "uvw")
                pr = np.maximum(take(sn[f"p{k}_dp"]).astype(float) + p_inf, 1e-6)
                return np.sqrt(u * u + v * v + w * w) / np.sqrt(gam * pr / np.maximum(r, 1e-9))
            layers.append((f"p{k}", a_, b_, take(mesh[f"p{k}_solid"]).astype(bool), h, get))
    layers.sort(key=lambda L: -L[4])
    return layers, ("density" if dens else "Mach number")


def render_schlieren2d(folder, out, plane="xz", width=1920, height=1080, fps=30, k=5.0, title=None,
                       progress=None, xlim=None, ylim=None, start=0):
    """Side-on numerical schlieren of the recorded snapshots, like a wind-tunnel schlieren film:
    gradient magnitude on the plane through the axis (dark = steep), the vehicle a black silhouette.

    Composites every grid on the plane, finest on top: the base grid at full resolution plus the
    refinement patches (e.g. 4 cm around the grid fins). Runs recorded before full-resolution
    density planes existed use the full-resolution Mach planes instead (shocks look the same)."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    sd = os.path.join(folder, "snapshots")
    mesh = dict(np.load(os.path.join(sd, "mesh.npz")))
    files = sorted(glob.glob(os.path.join(sd, "snap_*.npz")))[start:]
    if not files:
        raise SystemExit(f"no snapshots in {sd}")
    layers, kind = _schlieren_layers(mesh, plane, np.load(files[0]))
    base = layers[0]
    a0, a1, b0, b1 = base[1][0], base[1][-1], base[2][0], base[2][-1]
    if xlim:                                                    # zoom: crop to this window [m]
        a0, a1 = max(a0, xlim[0]), min(a1, xlim[1])
    if ylim:
        b0, b1 = max(b0, ylim[0]), min(b1, ylim[1])
    hp = max((a1 - a0) / width, (b1 - b0) / height)            # metres per pixel, fit the frame
    A = a0 + hp * (np.arange(int((a1 - a0) / hp)) + 0.5)
    B = b0 + hp * (np.arange(int((b1 - b0) / hp)) + 0.5)
    print(f"schlieren ({kind} gradient, plane {plane}): {len(layers)} grids, finest "
          f"{min(L[4] for L in layers) * 100:.0f} cm, {hp * 100:.1f} cm per pixel")
    # per layer: the pixel rectangle it covers, and the nearest-fluid fill for its solid cells
    prep = []
    solid_img = np.zeros((len(A), len(B)))
    for name, a_, b_, sol, h, get in layers:
        h = round(h, 4)
        ia = slice(np.searchsorted(A, a_[0]), np.searchsorted(A, a_[-1], "right"))
        ib = slice(np.searchsorted(B, b_[0]), np.searchsorted(B, b_[-1], "right"))
        if ia.stop <= ia.start or ib.stop <= ib.start:
            continue
        AA, BB = np.meshgrid(A[ia], B[ib], indexing="ij")
        pts = np.stack([AA.ravel(), BB.ravel()], -1)
        fill = distance_transform_edt(sol, return_distances=False, return_indices=True) if sol.any() and (~sol).any() else None
        solid_img[ia, ib] = RegularGridInterpolator((a_, b_), sol.astype(float))(pts).reshape(AA.shape)
        prep.append((h, a_, b_, sol, get, fill, ia, ib, pts, AA.shape))
    # refinement levels (patches of one cell size), coarsest first. Each level is blended in over a
    # few of its own cells from the edge of its covered area, and the thin gaps between
    # neighbouring patches (interiors only are recorded) are filled from the nearest patch pixel,
    # so no patch outlines show
    from scipy.ndimage import binary_closing
    levels = []
    for h in sorted({P[0] for P in prep}, reverse=True):
        members = [P for P in prep if P[0] == h]
        cov = np.zeros((len(A), len(B)), bool)
        for P in members:
            cov[P[6], P[7]] = True
        if h == prep[0][0]:
            levels.append((members, None, None, None))
            continue
        gap = max(2, int(np.ceil(2.5 * h / hp)))
        closed = binary_closing(cov, iterations=gap) | cov
        near = distance_transform_edt(~cov, return_distances=False, return_indices=True)
        dist = distance_transform_edt(closed) * hp
        wgt = np.clip(dist / (3.0 * h), 0.0, 1.0)
        levels.append((members, cov, closed, (near, wgt)))

    def grad_image(snap):
        g_img = np.zeros((len(A), len(B)))
        for members, cov, closed, aux in levels:
            g_lev = g_img if cov is None else np.zeros_like(g_img)
            for h, a_, b_, sol, get, fill, ia, ib, pts, shp in members:
                f = get(snap)
                if fill is not None:
                    f = f[tuple(fill)]                            # walls: no step into solid values
                ga, gb = np.gradient(f, a_, b_)
                g_lev[ia, ib] = RegularGridInterpolator((a_, b_), np.hypot(ga, gb))(pts).reshape(shp)
            if cov is not None:
                near, wgt = aux
                g_lev = g_lev[tuple(near)]                         # gaps between patches
                g_img = np.where(closed, wgt * g_lev + (1.0 - wgt) * g_img, g_img)
        return g_img

    body = solid_img > 0.5
    ref = float(np.percentile(grad_image(np.load(files[-1]))[~body], 99.0)) or 1.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, height // 50))
    except OSError:
        font = ImageFont.load_default()
    ox, oy = (width - len(A)) // 2, (height - len(B)) // 2
    writer = iio.get_writer(out, fps=fps, codec="libx264", quality=9, macro_block_size=8)
    for i, fpath in enumerate(files):
        snap = np.load(fpath)
        img = np.exp(-k * np.minimum(grad_image(snap) / ref, 1.0))
        img *= np.clip(1.0 - (solid_img - 0.35) / 0.3, 0.0, 1.0)  # anti-aliased black silhouette
        frame = np.zeros((height, width), np.uint8)
        frame[oy:oy + len(B), ox:ox + len(A)] = (255 * img.T[::-1]).astype(np.uint8)
        im = Image.fromarray(frame).convert("RGB")
        txt = (title + "   " if title else "") + f"t = {float(snap['t']) * 1e3:.1f} ms"
        ImageDraw.Draw(im).text((width // 100, height // 60), txt, fill=(255, 255, 255), font=font,
                                stroke_width=max(2, height // 400), stroke_fill=(0, 0, 0))
        writer.append_data(np.asarray(im))
        if progress:
            progress(i + 1, len(files))
    writer.close()
    return out


# ---------------------------------------------------------------- particles through the flow
class Tracers:
    """Massless tracer particles emitted from a ring around the vehicle upstream and carried by the
    (possibly changing) velocity field; drawn as short streaks coloured by Mach."""

    def __init__(self, axes, bb, flow_dir, n_per_frame=110, tail=5, r_in=3.6, r_out=9.5, seed=1, target=None,
                 upstream=None, max_age=None):
        self.ax = axes
        self.rng = np.random.default_rng(seed)
        self.n = n_per_frame
        self.tail = tail
        self.dir = np.asarray(flow_dir, float) / np.linalg.norm(flow_dir)
        c = np.array([(bb[0] + bb[1]) / 2, 0.0, 0.0])
        L = bb[1] - bb[0]
        # emitter: a ring perpendicular to the flow, `upstream` metres before `target` along the flow
        # (default: just ahead of the vehicle), so the particles arrive where the camera looks
        if target is None:
            self.origin = c - self.dir * (0.5 * L + 4.0)
        else:
            self.origin = np.asarray(target, float) - self.dir * upstream
        tmp = np.array([0.0, 1.0, 0.0]) if abs(self.dir[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        self.e1 = np.cross(self.dir, tmp); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.dir, self.e1)
        self.r_in, self.r_out = r_in, r_out
        self.paths = np.zeros((0, tail, 3))
        self.age = np.zeros(0, int)
        self.max_age = max_age
        self.lo = np.array([a[0] for a in axes]); self.hi = np.array([a[-1] for a in axes])

    def emit(self):
        r = np.sqrt(self.rng.uniform(self.r_in ** 2, self.r_out ** 2, self.n))     # a ring around the hull
        th = self.rng.uniform(0, 2 * np.pi, self.n)
        p = self.origin + (r * np.cos(th))[:, None] * self.e1 + (r * np.sin(th))[:, None] * self.e2
        p += self.dir * self.rng.uniform(0.0, 2.0, self.n)[:, None]
        new = np.repeat(p[:, None, :], self.tail, 1)
        self.paths = np.concatenate([self.paths, new])
        self.age = np.concatenate([self.age, np.zeros(self.n, int)])

    def step(self, vel_interp, solid_interp, dt, substeps=4):
        self.emit()
        P = self.paths[:, -1].copy()
        h = dt / substeps
        for _ in range(substeps):
            v1 = vel_interp(np.clip(P, self.lo, self.hi))
            v2 = vel_interp(np.clip(P + 0.5 * h * v1, self.lo, self.hi))
            P = P + h * v2
        self.paths = np.concatenate([self.paths[:, 1:], P[:, None]], 1)
        self.age += 1
        inside = np.all((P > self.lo) & (P < self.hi), axis=1)
        ok = inside & (solid_interp(np.clip(P, self.lo, self.hi)) < 0.5)
        if self.max_age:
            ok &= self.age < self.max_age
        self.paths = self.paths[ok]
        self.age = self.age[ok]

    def polydata(self, mach_interp):
        n, t, _ = self.paths.shape
        if n == 0:
            return None
        pts = self.paths.reshape(-1, 3)
        lines = np.hstack([np.full((n, 1), t), np.arange(n * t).reshape(n, t)]).ravel()
        poly = pv.PolyData(pts, lines=lines)
        poly.point_data["mach"] = mach_interp(np.clip(pts, self.lo, self.hi))
        return poly


def _interps(x, y, z, u, v, w, mach, solid, plevels=()):
    if plevels:
        fg = _FinestGrid([((x, y, z), dict(u=u, v=v, w=w, mach=mach, solid=solid.astype(float)))]
                         + [(ax_, {k: f_[k] for k in ("u", "v", "w", "mach", "solid")}) for ax_, f_ in plevels])
        vel = lambda P: np.stack([fg("u", P), fg("v", P), fg("w", P)], -1)
        return vel, (lambda P: fg("mach", P)), (lambda P: fg("solid", P))
    from scipy.interpolate import RegularGridInterpolator as RGI
    ax = (x, y, z)
    iu, iv, iw = (RGI(ax, a, bounds_error=False, fill_value=0.0) for a in (u, v, w))
    vel = lambda P: np.stack([iu(P), iv(P), iw(P)], -1)
    im = RGI(ax, mach, bounds_error=False, fill_value=0.0)
    isd = RGI(ax, solid.astype(float), bounds_error=False, fill_value=0.0)
    return vel, im, isd


class _FinestGrid:
    """Point sampling over the base grid and its refinement patches: each point takes its value
    from the finest grid that contains it (patches applied coarse to fine)."""

    def __init__(self, levels):
        from scipy.interpolate import RegularGridInterpolator as RGI
        self.levels = []
        order = sorted(levels, key=lambda L: -min(float(np.min(np.diff(a))) for a in L[0]))
        for axes, fields in order:
            ints = {k: RGI(axes, a, bounds_error=False, fill_value=0.0) for k, a in fields.items()}
            lo = np.array([a[0] for a in axes]); hi = np.array([a[-1] for a in axes])
            self.levels.append((lo, hi, ints))

    def __call__(self, name, P):
        P = np.asarray(P, float)
        out = self.levels[0][2][name](P)
        for lo, hi, ints in self.levels[1:]:
            inside = np.all((P > lo) & (P < hi), axis=1)
            if inside.any():
                out[inside] = ints[name](P[inside])
        return out


def _patch_level(x, y, z, u, v, w, rho, p, solid, gamma=1.4):
    """One refinement patch as a _FinestGrid level (velocity, Mach, solid, wall-filled pressure)."""
    u, v, w = (np.where(solid, 0.0, a) for a in (u, v, w))
    mach = np.sqrt(u * u + v * v + w * w) / np.sqrt(gamma * np.maximum(p, 1e-6) / np.maximum(rho, 1e-9))
    return ((x, y, z), dict(u=u, v=v, w=w, mach=mach, solid=solid.astype(float), pf=_fill_solid(p, solid),
                            rho=_fill_solid(rho, solid)))


def _refine_box(x, y, z, fields, box, factor=2):
    """Crop the fields to `box` and resample them `factor` times finer (cubic): the iso-surfaces
    cut from the result follow the data smoothly instead of in cell-sized facets."""
    from scipy.ndimage import zoom
    sl = []
    for a_, lo, hi in ((x, box[0], box[1]), (y, box[2], box[3]), (z, box[4], box[5])):
        i0 = max(int(np.searchsorted(a_, lo)) - 2, 0)
        i1 = min(int(np.searchsorted(a_, hi)) + 2, len(a_))
        sl.append(slice(i0, i1))
    sl = tuple(sl)
    axes = [a_[s_] for a_, s_ in zip((x, y, z), sl)]
    if factor <= 1:
        return axes, {k: a[sl] for k, a in fields.items()}
    new_axes = [zoom(a_, factor, order=1) for a_ in axes]
    out = {}
    for k, a in fields.items():
        out[k] = zoom(a[sl].astype(np.float32), factor, order=1 if k == "solid" else 3, mode="nearest")
    return new_axes, out


def _patch_boxes(plevels):
    return [(np.array([a[0] for a in ax_]), np.array([a[-1] for a in ax_])) for ax_, _ in plevels]


def _mask_boxes(axes, arr, boxes, value):
    """Set `arr` to `value` inside each box (where a finer patch draws its own surfaces)."""
    for lo, hi in boxes:
        ii = [slice(int(np.searchsorted(a_, lo[d], "right")), int(np.searchsorted(a_, hi[d], "left")))
              for d, a_ in enumerate(axes)]
        arr[tuple(ii)] = value
    return arr


def _rgrid(x, y, z, **fields):
    g = pv.RectilinearGrid(x, y, z)
    for nm, a in fields.items():
        g.point_data[nm] = np.ascontiguousarray(np.asarray(a, np.float32).transpose(2, 1, 0)).reshape(-1)
    return g


def _fill_solid(a, solid):
    """Values inside the body replaced by the nearest fluid value (so a surface sample is the wall value)."""
    if not solid.any():
        return a
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(solid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _shock_sensor(rho, x, y, z, solid, wall_cells=0):
    """Density jump per cell, |grad rho| h / rho: ~0.1-0.4 inside a captured shock, small elsewhere.
    wall_cells: also zero it this many cells from the body (the stair-stepped immersed wall reads
    as a jump there)."""
    gx, gy, gz = np.gradient(rho, x, y, z)
    hx, hy, hz = (np.gradient(c) for c in (x, y, z))
    h = np.maximum.reduce(np.meshgrid(hx, hy, hz, indexing="ij"))
    s = np.sqrt(gx * gx + gy * gy + gz * gz) * h / np.maximum(rho, 1e-9)
    if wall_cells > 0 and solid.any():
        from scipy.ndimage import binary_dilation
        s[binary_dilation(solid, iterations=wall_cells)] = 0.0
    s[solid] = 0.0
    return s


def _big_pieces(surf, frac=0.04, keep=6):
    """The largest connected pieces of an iso-surface (each at least `frac` of the biggest)."""
    if surf.n_points == 0:
        return surf
    con = surf.connectivity("all")
    rid = np.asarray(con.point_data["RegionId"])
    counts = np.bincount(rid)
    order = np.argsort(counts)[::-1][:keep]
    good = [r for r in order if counts[r] >= frac * counts[order[0]] and counts[r] > 50]
    if not good:
        return pv.PolyData()
    return con.extract_points(np.isin(rid, good), adjacent_cells=False).extract_surface()


class _Actors(list):
    """(mesh, add_mesh kwargs) pairs, built once and re-added to the plotter every frame."""

    def append_mesh(self, mesh, tag=None, **kw):
        # smooth shading: point normals computed once here (VTK shades with them), instead of
        # PyVista recomputing them every time the mesh is re-added
        if kw.pop("smooth_shading", False) and isinstance(mesh, pv.PolyData) and mesh.n_points:
            mesh = mesh.compute_normals(cell_normals=False, split_vertices=False, auto_orient_normals=False)
        kw["render"] = False
        self.append((mesh, kw, tag))


def _vehicle_actors(body, fins, color="#c9ced6"):
    acts = _Actors()
    acts.append_mesh(body, color=color, ambient=0.22, diffuse=0.75, specular=0.5, specular_power=30,
                     smooth_shading=True)
    for f in fins:
        acts.append_mesh(f, color="#9aa2ad", ambient=0.25, diffuse=0.7, specular=0.3)
    return acts


def _pitch_plane_normal(vdir):
    """Normal of the plane holding the vehicle axis and the crossflow (the plane of the shock pattern)."""
    c = np.array([0.0, vdir[1], vdir[2]])
    if np.linalg.norm(c) < 1e-3 * max(np.linalg.norm(vdir), 1e-9):
        return np.array([0.0, 1.0, 0.0])
    n = np.cross([1.0, 0.0, 0.0], c / np.linalg.norm(c))
    return n / np.linalg.norm(n)


def _streak_poly(tr, mach_interp, fade=1.6, m_inf=None, dm=0.3):
    """Tracer paths as polylines with a per-point opacity that fades toward the tail (comet streaks)
    and, given the freestream Mach, toward undisturbed flow: streaks are bright where the flow has
    been slowed or accelerated (shock layers, the fins, the wake) and faint in the freestream."""
    poly = tr.polydata(mach_interp)
    if poly is None:
        return None
    n, t = tr.paths.shape[:2]
    a = np.tile(0.05 + 0.95 * (np.arange(t) / max(t - 1, 1)) ** fade, n)
    if m_inf is not None:
        dev = np.abs(np.asarray(poly.point_data["mach"]) - m_inf) / dm
        a = a * (0.18 + 0.82 * np.clip(dev, 0.0, 1.0))
    poly.point_data["alpha"] = a.astype(np.float32)
    return poly


class _OilFlow:
    """Surface oil-flow lines: the velocity sampled a short distance off the wall along the
    surface normal, projected onto the surface, and integrated ON the surface from a fixed set of
    seed points (the same seeds every frame, so the pattern evolves rather than flickers)."""

    def __init__(self, body, n_seeds=2600, offset=0.45, seed=3):
        # body: a coarse (~0.25 m) hull surface; integrating on the 0.06 m display mesh is ~10x slower
        b = body.triangulate().clean()
        self.surf = b.compute_normals(cell_normals=False, split_vertices=False, auto_orient_normals=True)
        self.pts = np.asarray(self.surf.points)
        self.nrm = np.asarray(self.surf.point_data["Normals"])
        self.offset = offset
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.surf.n_points, size=min(n_seeds, self.surf.n_points), replace=False)
        self.source = pv.PolyData(self.pts[idx])

    def lines(self, vel, v_ref):
        V = vel(self.pts + self.offset * self.nrm)
        vt = V - np.sum(V * self.nrm, 1, keepdims=True) * self.nrm
        sp = np.linalg.norm(vt, axis=1)
        m = self.surf.copy()
        m.point_data["vt"] = vt
        m.point_data["speed"] = sp / max(v_ref, 1e-9)
        m.set_active_vectors("vt")
        try:
            sl = m.streamlines_from_source(self.source, vectors="vt", surface_streamlines=True,
                                           integration_direction="both", initial_step_length=0.3,
                                           step_unit="l", max_step_length=0.5, max_steps=300, max_length=9.0,
                                           terminal_speed=1e-3 * max(v_ref, 1e-9), compute_vorticity=False)
        except Exception:                   # noqa: BLE001
            return None
        if sl.n_points == 0:
            return None
        if "Normals" in sl.point_data:      # lift off the surface so the lines are not z-fought
            sl.points = np.asarray(sl.points) + 0.04 * np.asarray(sl.point_data["Normals"])
        return sl


def render_flow(folder, out, width=1920, height=1080, fps=30, q=None, title=None, frames=360, mode="evolve",
                progress=None, show_vortices=True, camera="three-quarter", surface="cp", plane="schlieren",
                shocks=True, trail=14, still_frame=None, streaks=True, refine=None, start=0, oil=False,
                volume=False, tween=1):
    """The full 3-D model (body + grid fins) with the flow around it, LAVA-style:

      * tracer streaks through the hull boundary layer and the fins, coloured by Mach, with fading
        comet tails and a soft glow;
      * the hull and fins coloured by surface pressure (Cp) sampled from the flow (surface='cp'),
        or polished steel (surface='metal');
      * a numerical-schlieren (or Mach) plane through the axis in the pitch plane behind the
        vehicle, carrying the shock pattern;
      * translucent shock surfaces (density jump per cell) and vortex surfaces (Q-criterion);
      * a graded background, depth-peeled transparency and a slow camera move.

    mode 'evolve': the recorded snapshots (particles carried by the changing flow); 'orbit': the
    steady flow.vtk with a slow camera move. still_frame=k: write frame k only, as a PNG (out).
    tween=N: N frames per snapshot interval, the flow fields blended linearly between snapshots
    (smoother motion when snapshots are far apart in time; N times the render time).
    volume=True: volume rendering of the density jump per cell (numerical schlieren in 3-D: the
    shocks and expansion fans as glowing smoke, the LAVA density-gradient look) in place of the
    glass shock sheets.
    oil=True: surface oil-flow lines on the hull (streamlines of the near-wall velocity
    integrated on the surface, as in a wind-tunnel oil-flow test: separation and attachment
    lines, the flow around the chines). streaks=False: no tracer streaks. refine: the shock and vortex
    surfaces are cut from the flow resampled this many times finer (cubic), and from every
    refinement patch at its own resolution, so they follow the data instead of cell facets."""
    meta = {}
    mp = os.path.join(folder, "loads.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            meta = json.load(fh)
    body, fins = vehicle_meshes(meta)
    bb = body.bounds
    cond = meta.get("condition") or {}
    if mode == "evolve":
        sd = os.path.join(folder, "snapshots")
        mesh = dict(np.load(os.path.join(sd, "mesh.npz")))
        if "vx" not in mesh:
            raise SystemExit("needs 3-D snapshots: run with --snapshots N --snap-volume 1")
        files = sorted(glob.glob(os.path.join(sd, "snap_*.npz")))[start:]
        x, y, z = (mesh[k].astype(float) for k in ("vx", "vy", "vz"))
        solid = mesh["vol_solid"].astype(bool)
        p_inf = float(mesh["p_inf"])

        n_p = int(mesh.get("n_patches", 0))
        gam = float(mesh.get("gamma", 1.4))

        def frame_fields(i):
            s = np.load(files[i])
            u, v, w = (s[k].astype(np.float32).astype(float) for k in ("vol_u", "vol_v", "vol_w"))
            u[solid], v[solid], w[solid] = 0, 0, 0
            rho = s["vol_rho"].astype(np.float32).astype(float)
            p = s["vol_dp"].astype(np.float32).astype(float) + p_inf
            return u, v, w, s["mach_vol"].astype(np.float32).astype(float), rho, p, float(s["t"])

        def patch_levels(i):
            if not n_p:
                return []
            s = np.load(files[i])
            out = []
            for k in range(n_p):
                if f"p{k}_u" not in s:
                    continue
                g = lambda n: s[f"p{k}_{n}"].astype(np.float32).astype(float)
                out.append(_patch_level(*(mesh[f"p{k}_{a}"].astype(float) for a in "xyz"), g("u"), g("v"), g("w"),
                                        g("rho"), g("dp") + p_inf, mesh[f"p{k}_solid"].astype(bool), gam))
            return out
        nfr = len(files)
    else:
        fd = load_steady(folder)[0]
        f, x, y, z = derived.grid_arrays(fd.base)
        solid = f["phi"] < 0 if "phi" in f else np.zeros_like(f["rho"], bool)
        u, v, w = (np.where(solid, 0.0, f[c]) for c in "uvw")
        rho, p = f["rho"].astype(float), f["p"].astype(float)
        a_snd = np.sqrt(1.4 * np.maximum(p, 1e-6) / np.maximum(rho, 1e-9))
        mach = f.get("mach", np.sqrt(u * u + v * v + w * w) / a_snd)
        p_inf = float(cond.get("p", np.median(p[~solid])))
        frame_fields = lambda i: (u, v, w, mach, rho, p, 0.0)
        steady_patches = []
        for gp in fd.patches:
            fp, px, py, pz = derived.grid_arrays(gp)
            if not all(k in fp for k in ("u", "v", "w", "rho", "p")):
                continue
            ps = fp["phi"] < 0 if "phi" in fp else np.zeros_like(fp["rho"], bool)
            steady_patches.append(_patch_level(px, py, pz, fp["u"], fp["v"], fp["w"], fp["rho"].astype(float),
                                               fp["p"].astype(float), ps))
        patch_levels = lambda i: steady_patches
        nfr = frames
    u0, v0, w0, m0, rho0, p0, _ = frame_fields(nfr - 1)
    if refine is None:                    # auto: resample coarse (strided) volumes to ~0.3 m
        h_rec = float(np.median(np.diff(x)))
        refine = 2 if h_rec > 0.45 else 1
    fluid = ~solid
    vmean = np.array([np.mean(a[fluid]) for a in (u0, v0, w0)])
    Vinf = float(np.linalg.norm(vmean))
    vdir = vmean / max(Vinf, 1e-9)
    rho_inf = float(cond.get("rho", np.median(rho0[fluid])))
    m_inf = float(cond.get("mach", np.median(m0[fluid])))
    q_inf = 0.5 * rho_inf * float(cond.get("V", Vinf)) ** 2
    # colour ranges from the flow near the vehicle (the far field would make everything one colour)
    X3 = np.meshgrid(x, y, z, indexing="ij")
    near = fluid & (np.hypot(X3[1], X3[2]) < 12.0) & (X3[0] > bb[0] - 5) & (X3[0] < bb[1] + 10)
    # streak colours: the spread of Mach in the layer the particles fly through (near freestream
    # everywhere else), so the slowed and accelerated regions stand out
    band = fluid & (np.hypot(X3[1], X3[2]) < 10.5) & (X3[0] > bb[0] - 2) & (X3[0] < bb[1] + 4)
    mrange = [float(np.percentile(m0[band], 4)), float(np.percentile(m0[band], 99))]
    from .vehicles.super_heavy import SuperHeavyParams
    SP = SuperHeavyParams()
    R_hull = SP.radius
    close = camera == "fins"
    if close:
        tgt, up_d = (SP.fin_x_center, 0.0, 0.0), 14.0
    else:
        tgt, up_d = (bb[0], 0.0, 0.0), 8.0
    L_v = bb[1] - bb[0]
    tr = Tracers((x, y, z), bb, vmean, n_per_frame=80 if close else 40, tail=trail,
                 r_in=R_hull + 0.25, r_out=R_hull + (5.5 if close else 4.5), target=tgt, upstream=up_d,
                 max_age=int((up_d + 26.0) / 1.2) if close else int((up_d + L_v + 12.0) / 1.2))
    dt_draw = 1.2 / max(Vinf, 1e-6)             # particles move ~1.2 m per frame on screen
    # one Q level and one shock level for the whole movie
    g_last = _rgrid(x, y, z, u=u0, v=v0, w=w0, rho=np.ones_like(u0), p=np.ones_like(u0))
    g_last.point_data["velocity"] = np.column_stack([g_last.point_data[c] for c in "uvw"])
    ql = None
    if show_vortices:
        derived.add_to_grid(g_last, ["Q"])
        ql = q if q is not None else auto_q(g_last, frac=0.99)
    s_lvl = 0.12
    # framing region for the plane and the shock surfaces
    L = bb[1] - bb[0]
    c = np.array([(bb[0] + bb[1]) / 2, 0.0, 0.0])
    nrm = _pitch_plane_normal(vdir)
    box = [bb[0] - 0.22 * L, bb[1] + 0.35 * L, -0.42 * L, 0.42 * L, -0.42 * L, 0.42 * L]

    fill_idx = None
    if solid.any():
        from scipy.ndimage import distance_transform_edt
        fill_idx = tuple(distance_transform_edt(solid, return_distances=False, return_indices=True))

    def surface_cp(p_field, plevels=()):
        pf = p_field[fill_idx] if fill_idx is not None else p_field
        fg = _FinestGrid([((x, y, z), dict(pf=pf))] + [(ax_, dict(pf=f_["pf"])) for ax_, f_ in plevels])
        out_m = []
        for m in [body] + list(fins):
            mm = m.copy()
            mm.point_data["Cp"] = ((fg("pf", np.asarray(mm.points)) - p_inf) / q_inf).astype(np.float32)
            out_m.append(mm)
        return out_m
    if surface == "cp":
        cps = np.concatenate([np.asarray(m.point_data["Cp"]) for m in surface_cp(p0, patch_levels(nfr - 1))])
        cp_hi = float(max(abs(np.percentile(cps, 5)), abs(np.percentile(cps, 95)), 0.05))
        cp_range = [-cp_hi, cp_hi]

    oilf = _OilFlow(vehicle_meshes(meta, h=0.25)[0]) if oil else None
    pl = setup_plotter(width, height)
    try:
        pl.set_background("#02040a", top="#121c30")
        pl.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
    except Exception:                       # noqa: BLE001
        pass
    if still_frame is None:
        pl.open_movie(out, framerate=fps, quality=8)
    # warm up the tracers so the first frame is already full of streaks
    u, v, w, m, rho, p, _ = frame_fields(0)
    vel, im, isd = _interps(x, y, z, u, v, w, m, solid, patch_levels(0))
    for _ in range(int(((26.0 if close else L) + up_d) / 1.2) + trail + 5 if streaks else 0):
        tr.step(vel, isd, dt_draw)
    bar = dict(color="white", vertical=True, height=0.55, width=0.022, position_y=0.22, fmt="%.1f",
               title_font_size=14, label_font_size=12)
    def field_layers(u, v, w, m, rho, p, plev=()):
        acts = _Actors()
        # vehicle
        if surface == "cp":
            ms = surface_cp(p, plev)
            for k, mm in enumerate(ms):
                acts.append_mesh(mm, scalars="Cp", cmap="coolwarm", clim=cp_range, smooth_shading=(k == 0),
                            ambient=0.25, diffuse=0.7, specular=0.35, specular_power=25,
                            show_scalar_bar=(k == 0), scalar_bar_args=dict(title="Cp", position_x=0.035, **bar))
        else:
            acts.extend(_vehicle_actors(body, fins))
        # backdrop plane through the axis (pitch plane), cut away around the camera side
        sens = _shock_sensor(rho, x, y, z, solid, wall_cells=2)
        from scipy.ndimage import gaussian_filter
        g = _rgrid(x, y, z, mach=m, phi=np.where(solid, -1.0, 1.0), sens=sens,
                   sch=np.exp(-14.0 * gaussian_filter(sens, 0.8)))      # plane: one-cell smoothing
        if plane != "none":
            sl = g.slice(normal=tuple(nrm), origin=tuple(c)).clip_box(box, invert=False)
            if sl.n_points:
                sl = sl.threshold(0.0, scalars="phi")
            if sl.n_points:
                if plane == "schlieren":       # shadowgraph glow: shocks light on a dark plane
                    acts.append_mesh(sl, scalars="sch", cmap="bone_r", clim=[0.1, 1.0], opacity=0.85,
                                show_scalar_bar=False, ambient=0.6, diffuse=0.4, specular=0.0)
                else:
                    acts.append_mesh(sl, scalars="mach", cmap="turbo", clim=mrange, opacity=0.6,
                                show_scalar_bar=False, ambient=0.6, diffuse=0.4, specular=0.0)
        # shock and vortex surfaces: from the base flow resampled finer, plus every patch at its own
        # resolution (the base draws nothing inside a patch)
        boxes = _patch_boxes(plev)
        grids_ = []
        ax_r, f_r = _refine_box(x, y, z, dict(u=u, v=v, w=w, mach=m, rho=_fill_solid(rho, solid),
                                              solid=solid.astype(float)), box, refine)
        grids_.append((ax_r, f_r, True))
        for ax_p, f_p in plev:
            grids_.append((list(ax_p), dict(u=f_p["u"], v=f_p["v"], w=f_p["w"], mach=f_p["mach"], rho=f_p["rho"],
                                            solid=f_p["solid"]), False))
        sh_all, vx_all, vol_levels = [], [], []
        for ax_, f_, is_base in grids_:
            sol_ = f_["solid"] > 0.5
            if volume:
                sv = _shock_sensor(f_["rho"], *ax_, sol_, wall_cells=2 * (refine if is_base else 1))
                if is_base:
                    sv *= max(refine, 1)
                vol_levels.append((tuple(ax_), dict(sens=sv)))
            if shocks and not volume:
                sn = _shock_sensor(f_["rho"], *ax_, sol_, wall_cells=2 * (refine if is_base else 1))
                if is_base:
                    sn *= max(refine, 1)           # per cell of the recorded grid, not the resampled one
                    _mask_boxes(ax_, sn, boxes, 0.0)
                gs = _rgrid(*ax_, sens=sn)
                sh = gs.contour([s_lvl], scalars="sens")
                if sh.n_points:
                    sh_all.append(sh)
            if show_vortices and ql:
                gq = _rgrid(*ax_, u=f_["u"], v=f_["v"], w=f_["w"], mach=f_["mach"],
                            rho=np.ones_like(f_["u"]), p=np.ones_like(f_["u"]))
                gq.point_data["velocity"] = np.column_stack([gq.point_data[cc] for cc in "uvw"])
                derived.add_to_grid(gq, ["Q"])
                if is_base and boxes:
                    qa = np.asarray(gq.point_data["Q"]).reshape(tuple(len(a_) for a_ in ax_[::-1])).transpose(2, 1, 0).copy()
                    _mask_boxes(ax_, qa, boxes, -1.0)
                    gq.point_data["Q"] = np.ascontiguousarray(qa.transpose(2, 1, 0)).reshape(-1)
                iso = gq.contour([ql], scalars="Q")
                if iso.n_points:
                    vx_all.append(iso)
        if vol_levels:
            # uniform voxels over the framing box, each from the finest grid there
            hv = max(0.25, (box[1] - box[0]) / 360.0)
            dims = [int((box[2 * d + 1] - box[2 * d]) / hv) + 1 for d in range(3)]
            img = pv.ImageData(dimensions=dims, spacing=(hv, hv, hv), origin=(box[0], box[2], box[4]))
            fgv = _FinestGrid(vol_levels)
            sv = fgv("sens", np.asarray(img.points))
            img.point_data["sens"] = np.clip(sv, 0.0, 0.5).astype(np.float32)
            acts.append((img, dict(scalars="sens", cmap="bone", clim=[0.03, 0.35],
                                   opacity=[0.0, 0.0, 0.015, 0.05, 0.13, 0.26], shade=False,
                                   mapper="fixed_point",     # CPU ray casting: renders headless too
                                   show_scalar_bar=False, render=False), "volume"))
        if sh_all:
            sh = _big_pieces(pv.merge(sh_all).clip_box(box, invert=False), keep=12)
            if sh.n_points:
                acts.append_mesh(sh.smooth_taubin(n_iter=30, pass_band=0.08), tag="shock", color="#d8ecff",
                                 opacity=0.16, smooth_shading=True, specular=0.6, specular_power=40, ambient=0.3)
        if vx_all:
            iso = pv.merge(vx_all)
            if iso.n_points:
                acts.append_mesh(iso.smooth_taubin(n_iter=25, pass_band=0.1), scalars="mach", cmap="turbo",
                                 clim=mrange, opacity=0.45, smooth_shading=True, specular=0.4,
                                 show_scalar_bar=not streaks,
                                 scalar_bar_args=dict(title="Mach", position_x=0.94, **bar))
        return acts

    layer_cache = {}
    frames_to_draw = range(nfr) if still_frame is None else [min(still_frame, nfr - 1)]
    if mode == "evolve" and tween > 1 and still_frame is None and nfr > 1:
        sched = [(i_, f_) for i_ in range(nfr - 1) for f_ in np.arange(tween) / tween] + [(nfr - 1, 0.0)]
    else:
        sched = [(i_, 0.0) for i_ in frames_to_draw]
    loaded = {}

    def fields_at(i_, f_):
        """Snapshot i_ blended a fraction f_ toward i_ + 1 (fields and patch levels)."""
        def get(j):
            if j not in loaded:
                if len(loaded) >= 3:
                    loaded.pop(min(loaded))
                loaded[j] = (frame_fields(j), patch_levels(j))
            return loaded[j]
        a_, la = get(i_)
        if f_ <= 0.0:
            return a_, la
        b_, lb = get(i_ + 1)
        mix = tuple((1 - f_) * x_ + f_ * y_ for x_, y_ in zip(a_, b_))
        lev = [(axa, {k_: (1 - f_) * fa[k_] + f_ * fb[k_] for k_ in fa}) for (axa, fa), (_, fb) in zip(la, lb)]
        return mix, lev

    last_i = -1
    for n_fr, (i, frac) in enumerate(sched):
        (u, v, w, m, rho, p, t), plev = fields_at(i, frac)
        s_pos = (i + frac) / max(nfr - 1, 1)
        if streaks:
            vel, im, isd = _interps(x, y, z, u, v, w, m, solid, plev)
            for _ in range(max(1, i - last_i) if still_frame is not None else 1):
                tr.step(vel, isd, dt_draw)
        last_i = i
        # camera: slow drift around a three-quarter view (or a close-up of the fins)
        a = np.radians(-35 + 30 * s_pos)
        ctr_f = np.array([SP.fin_x_center - 2.0, 0.0, 2.0])
        pos_f = ctr_f + 26.0 * np.array([-0.55 * np.cos(a), -np.cos(a) * 0.6 + np.sin(a) * 0.4, 0.45])
        ctr_w, pos_w = c, c + 1.35 * L * np.array([np.sin(a) * 0.5, -np.cos(a), 0.3])
        shock_fade = 0.4 if close else 1.0
        if close:
            ctr, pos = ctr_f, pos_f
        elif camera == "flyby":
            # wide three-quarter view, then a push in along the hull to the fins
            k = np.clip((s_pos - 0.15) / 0.7, 0.0, 1.0)
            k = k * k * (3 - 2 * k)
            shock_fade = 1.0 - 0.85 * k
            ctr = (1 - k) * ctr_w + k * ctr_f
            pos = (1 - k) * pos_w + k * pos_f
        else:
            ctr, pos = ctr_w, pos_w
        pl.clear_actors()
        key = (i, frac) if mode == "evolve" else 0
        if layer_cache.get("key") != key:
            layer_cache.clear()
            layer_cache.update(key=key, actors=field_layers(u, v, w, m, rho, p, plev))
        for mesh_, kw, tag in layer_cache["actors"]:
            if tag == "volume":
                pl.add_volume(mesh_, **kw)
                continue
            if tag == "shock":
                kw = dict(kw, opacity=kw["opacity"] * shock_fade)
            pl.add_mesh(mesh_, **kw)
        if oilf is not None:
            if layer_cache.get("oil_key") != key:
                vel_o = _interps(x, y, z, u, v, w, m, solid, plev)[0]
                layer_cache["oil"] = oilf.lines(vel_o, Vinf)
                layer_cache["oil_key"] = key
            ol = layer_cache.get("oil")
            if ol is not None:
                pl.add_mesh(ol, scalars="speed", cmap="bone", clim=[0.0, 0.9], line_width=1.3, opacity=0.85,
                            lighting=False, show_scalar_bar=False, render=False)
        # streaks: soft wide glow under a bright fading core
        poly = _streak_poly(tr, im, m_inf=m_inf) if streaks else None
        if poly is not None:
            lw = 2.2 if close else 1.4
            pl.add_mesh(poly, scalars="mach", cmap="turbo", clim=mrange, line_width=lw * 3.2, opacity=0.08,
                        show_scalar_bar=False, lighting=False, render=False)
            pl.add_mesh(poly, scalars="mach", cmap="turbo", clim=mrange, line_width=lw, opacity="alpha",
                        lighting=False, show_scalar_bar=True, render=False,
                        scalar_bar_args=dict(title="Mach", position_x=0.94, **bar))
        txt = (title + "   " if title else "")
        if mode == "evolve":
            txt += f"t = {t * 1e3:.1f} ms"
        if txt:
            pl.add_text(txt, position="upper_left", font_size=11, color="#c8ccd4")
        pl.camera_position = [tuple(pos), tuple(ctr), (0, 0, 1)]
        if still_frame is not None:
            pl.screenshot(out)
        else:
            pl.write_frame()
        if progress:
            progress(n_fr + 1, len(sched))
    pl.close()
    return out


if __name__ == "__main__":
    main()
