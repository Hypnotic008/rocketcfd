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

STYLES = ("lava", "schlieren", "mach", "pressure")


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
          plane_axis=1):
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
    if style in ("lava", "schlieren"):
        pl.add_mesh(sl, scalars="schlieren", cmap="gray", clim=[0.05, 1.0], show_scalar_bar=False,
                    ambient=0.55, diffuse=0.45, specular=0.0, opacity=0.92 if style == "lava" else 1.0)
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
    if surf is not None and surf.n_points:
        if style == "pressure" and "Cp" in surf.point_data:
            pl.add_mesh(surf, scalars="Cp", cmap="coolwarm", clim=list(cp_range), smooth_shading=True,
                        scalar_bar_args=dict(title="Cp", color="white", vertical=True, position_x=0.04,
                                             position_y=0.2, height=0.6, width=0.03))
        else:
            pl.add_mesh(surf, color="#d6dae0", ambient=0.22, diffuse=0.75, specular=0.55, specular_power=30,
                        smooth_shading=True)
    return ql


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
    bb = surf.bounds if surf is not None else flow.bounds()
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
    bb = surf.bounds if surf is not None else flow.bounds()
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
        build(pl, g, surf, style, q_level=ql, mach_range=mrange)
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
    ap.add_argument("--mode", choices=["still", "orbit", "evolve"], default="still")
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
    out = a.out or os.path.join(a.folder, f"cinematic_{a.mode}_{a.style}.{ext}")
    if a.mode == "still":
        render_still(a.folder, out, a.style, a.width, a.height, a.q, a.title)
    elif a.mode == "orbit":
        render_orbit(a.folder, out, a.style, a.width, a.height, a.frames, a.fps, a.q, a.title)
    else:
        render_evolve(a.folder, out, a.style, a.width, a.height, a.fps, a.q, a.title, a.every,
                      progress=lambda i, n: print(f"frame {i}/{n}", flush=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
