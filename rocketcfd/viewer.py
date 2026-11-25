"""3-D viewer and animation recorder for rocketcfd booster results (PyVista / VTK).

Shows the real geometry at all times (booster hull coloured by surface pressure, the
toothed grid fins as solid parts), streamlines coloured by Mach with particles flowing
along them, and a cross-section slice that can be dragged or swept along the booster.

  py -m pip install pyvista imageio-ffmpeg
  py -m rocketcfd.viewer results\\glide_rudder                 interactive window
  py -m rocketcfd.viewer results\\glide_rudder --movie orbit   MP4: orbit with flowing particles
  py -m rocketcfd.viewer results\\glide_rudder --movie sweep   MP4: slice sweeping engines -> top
  py -m rocketcfd.viewer results\\glide_rudder --movie both

Interactive keys: space = pause/play particles, o = orbit once, k = sweep the slice,
s = streamlines on/off, f = fins on/off, h = hull transparent/opaque, r = reset view.
The slider moves the slice along the booster. Mouse: rotate / pan (shift) / zoom (wheel).

Frame: x along the booster from the engines (x = 0) to the top (x = 72.3 m); rudder fin on +z,
tower side on -z, catch fins on +/-y.
"""
import argparse
import json
import os
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- loading
def load_flow(path, stride=1):
    g = pv.read(path)
    if stride > 1 and isinstance(g, pv.RectilinearGrid):
        nx, ny, nz = g.dimensions
        x, y, z = g.x[::stride], g.y[::stride], g.z[::stride]
        h = pv.RectilinearGrid(x, y, z)
        for name in g.point_data:
            a = np.asarray(g.point_data[name])
            shp = (nz, ny, nx) + a.shape[1:]
            h.point_data[name] = a.reshape(shp)[::stride, ::stride, ::stride].reshape((-1,) + a.shape[1:])
        g = h
    if "phi" in g.point_data and "velocity" in g.point_data:
        v = np.array(g.point_data["velocity"])
        v[np.asarray(g.point_data["phi"]) < 0] = 0.0          # streamlines must not run into the hull
        g.point_data["velocity"] = v
    if "mach" in g.point_data:
        g.set_active_scalars("mach")
    return g


def fin_meshes(meta, fin_stl=None):
    """The fitted grid fins as solid meshes in body coordinates."""
    from .vehicles.super_heavy import SuperHeavyParams, _az
    from .vehicles.grid_fin import GridFinParams
    model = meta.get("fins", {}).get("model", "none")
    if model == "none":
        return []
    P = SuperHeavyParams()
    defl = meta.get("fins", {}).get("deflection") or [0.0] * len(P.fin_azimuths)
    stl = fin_stl or os.path.join(HERE, "..", "results", "grid_fin_v3_light.stl")
    if model == "parametric" and os.path.exists(stl):
        base = pv.read(stl)
    else:                                   # fall back: triangulate the fin distance function
        from .vehicles.grid_fin import grid_fin_sdf, grid_fin_box
        from .output import write_stl_chunked
        tmp = os.path.join(os.path.dirname(stl), "_fin_tmp.stl")
        GP = GridFinParams()
        lo, hi = grid_fin_box(GP, pad=0.03)
        write_stl_chunked(lambda X, Y, Z: grid_fin_sdf(X, Y, Z, GP), lo, hi, 0.008, tmp)
        base = pv.read(tmp)
    out = []
    for k, az in enumerate(P.fin_azimuths):
        th = _az(az)
        pts = np.asarray(base.points, float)
        a, t, zr = pts[:, 0], pts[:, 1], pts[:, 2]
        d = np.radians(defl[k] if k < len(defl) else 0.0)
        a2, t2 = a * np.cos(d) - t * np.sin(d), a * np.sin(d) + t * np.cos(d)
        r = P.radius + zr
        m = base.copy()
        m.points = np.stack([P.fin_x_center + a2, r * np.cos(th) - t2 * np.sin(th),
                             r * np.sin(th) + t2 * np.cos(th)], 1)
        out.append(m)
    return out


def freestream_dir(meta):
    from .vehicles.super_heavy import _az
    al = np.radians(meta["condition"]["aoa_from_engines_first"])
    tw = _az(meta.get("condition_windward", 180.0))
    d = np.array([np.cos(al), -np.sin(al) * np.cos(tw), -np.sin(al) * np.sin(tw)])
    return d / np.linalg.norm(d), tw


def streamlines(flow, meta, n=16, half=6.0):
    """Streamlines from a rake upstream of the engine end, across the whole vehicle."""
    d, _ = freestream_dir(meta)
    e1 = np.cross(d, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    c0 = np.array([-0.5, 0.0, 0.0]) - 5.0 * d
    s = np.linspace(-half, half, n)
    S1, S2 = np.meshgrid(s, s)
    seeds = c0 + S1.ravel()[:, None] * e1 + S2.ravel()[:, None] * e2
    keep = np.hypot(S1, S2).ravel() < half
    src = pv.PolyData(seeds[keep])
    kw = dict(vectors="velocity", integration_direction="forward", initial_step_length=0.2,
              step_unit="cl", max_steps=6000)
    try:
        sl = flow.streamlines_from_source(src, max_length=160.0, **kw)
    except TypeError:
        sl = flow.streamlines_from_source(src, max_time=160.0, **kw)
    return sl


class Particles:
    """Particles moving along streamlines at the flow speed (time from IntegrationTime)."""

    def __init__(self, sl, per_line=6):
        self.lines = []
        t_all = np.asarray(sl.point_data["IntegrationTime"])
        conn = sl.lines
        i = 0
        while i < len(conn):
            n = conn[i]
            ids = conn[i + 1:i + 1 + n]
            i += n + 1
            if n < 4:
                continue
            t = t_all[ids]
            if t[-1] <= t[0]:
                continue
            self.lines.append((np.asarray(sl.points[ids]), t - t[0]))
        self.T = np.median([t[-1] for _, t in self.lines]) if self.lines else 1.0
        self.per_line = per_line
        self.cloud = pv.PolyData(self.positions(0.0))

    def positions(self, phase):
        pts = []
        for P, t in self.lines:
            q = (phase + np.arange(self.per_line) / self.per_line) % 1.0 * self.T
            q = q[q <= t[-1]]
            if q.size:
                pts.append(np.stack([np.interp(q, t, P[:, k]) for k in range(3)], 1))
        return np.concatenate(pts) if pts else np.zeros((1, 3))

    def update(self, phase):
        self.cloud.points = self.positions(phase)


# ---------------------------------------------------------------- scene
def build_scene(pl, folder, meta, flow, args):
    surf = pv.read(os.path.join(folder, "surface_cp.vtk"))
    cp = np.asarray(surf.point_data["Cp"])
    lim = float(np.percentile(np.abs(cp), 99))
    actors = {}
    actors["hull"] = pl.add_mesh(surf, scalars="Cp", cmap="RdBu_r", clim=(-lim, lim), smooth_shading=True,
                                 scalar_bar_args=dict(title="Cp (surface pressure)", vertical=True,
                                                      position_x=0.90, position_y=0.55, height=0.35))
    fins = fin_meshes(meta, args.fin_stl)
    actors["fins"] = [pl.add_mesh(m, color="#3b3d42", pbr=True, metallic=0.7, roughness=0.35) for m in fins]
    scene = dict(actors=actors, particles=None, slice_actor=None)
    if flow is not None:
        print("tracing streamlines ...", flush=True)
        sl = streamlines(flow, meta, n=args.seeds)
        if sl.n_points:
            tubes = sl.tube(radius=0.035)
            m = np.asarray(flow.point_data["mach"])
            scene["mach_range"] = (0.0, float(np.percentile(m[np.asarray(flow.point_data["phi"]) > 0], 99.9)))
            actors["stream"] = pl.add_mesh(tubes, scalars="mach", cmap="turbo", opacity=0.35,
                                           clim=scene["mach_range"],
                                           scalar_bar_args=dict(title="Mach", vertical=True,
                                                                position_x=0.90, position_y=0.08, height=0.35))
            parts = Particles(sl)
            actors["particles"] = pl.add_mesh(parts.cloud, color="white", point_size=7,
                                              render_points_as_spheres=True)
            scene["particles"] = parts
        scene["flow"] = flow
    return scene


def set_slice(pl, scene, x0):
    flow = scene.get("flow")
    if flow is None:
        return
    if scene["slice_actor"] is not None:
        pl.remove_actor(scene["slice_actor"])
    sl = flow.slice(normal="x", origin=(x0, 0.0, 0.0))
    if sl.n_points:
        sl = sl.clip_box([x0 - 1.0, x0 + 1.0, -12.0, 12.0, -12.0, 12.0], invert=False)
    if sl.n_points and "phi" in sl.point_data:
        sl = sl.threshold(0.0, scalars="phi")              # drop the inside of the hull
    if "mach_range" not in scene:
        m = np.asarray(flow.point_data["mach"])
        scene["mach_range"] = (0.0, float(np.percentile(m[np.asarray(flow.point_data["phi"]) > 0], 99.9)))
    scene["slice_actor"] = pl.add_mesh(sl, scalars="mach", cmap="turbo", clim=scene["mach_range"], opacity=0.9,
                                       show_scalar_bar=False) if sl.n_points else None


def camera_orbit(n, center=(36.0, 0.0, 0.0), radius=85.0, start=0.0, lead=-22.0):
    """Orbit around the booster axis; 'up' turns with the camera so the booster stays level
    on screen. lead < 0 puts the camera toward the engine end, looking slightly up the body."""
    out = []
    for k in range(n):
        a = start + 2 * np.pi * k / n
        c, s_ = np.cos(a), np.sin(a)
        pos = (center[0] + lead, radius * c, radius * s_)
        out.append((pos, center, (0.0, -s_, c)))
    return out


def default_view(pl, meta):
    _, tw = freestream_dir(meta)
    w = np.array([0.0, np.cos(tw), np.sin(tw)])           # windward side direction
    side = np.cross([1.0, 0.0, 0.0], w)
    pos = np.array([30.0, 0.0, 0.0]) + 70 * (0.8 * w + 0.6 * side) + np.array([-25.0, 0, 0])
    pl.camera_position = [tuple(pos), (36.0, 0.0, 0.0), tuple(-side if abs(side[2]) < 0.9 else w)]


# ---------------------------------------------------------------- modes
def interactive(folder, meta, flow, args):
    pl = pv.Plotter(window_size=(1600, 900), title=f"rocketcfd - {os.path.basename(os.path.normpath(folder))}")
    pl.set_background("#0d1117", top="#2d3440")
    scene = build_scene(pl, folder, meta, flow, args)
    pl.add_text(title_of(meta), font_size=10, color="white")
    default_view(pl, meta)
    state = dict(phase=0.0, play=True, hull_opaque=True)
    if flow is not None:
        set_slice(pl, scene, 2.0)
        pl.add_slider_widget(lambda x: set_slice(pl, scene, x), [-1.0, 72.0], value=2.0,
                             title="slice x [m]", pointa=(0.05, 0.08), pointb=(0.45, 0.08), color="white")

    def tick(step):
        if scene["particles"] is not None and state["play"]:
            state["phase"] = (state["phase"] + 0.004) % 1.0
            scene["particles"].update(state["phase"])
            pl.render()

    def toggle(name):
        acts = scene["actors"].get(name)
        for a_ in (acts if isinstance(acts, list) else [acts]):
            if a_ is not None:
                a_.SetVisibility(not a_.GetVisibility())
        pl.render()

    def hull_alpha():
        state["hull_opaque"] = not state["hull_opaque"]
        scene["actors"]["hull"].GetProperty().SetOpacity(1.0 if state["hull_opaque"] else 0.35)
        pl.render()

    def orbit():
        _, tw = freestream_dir(meta)
        for pos in camera_orbit(180, start=tw - np.radians(35)):
            pl.camera_position = pos
            tick(0)
            pl.render()

    def sweep():
        for x0 in np.linspace(-1.0, 72.0, 120):
            set_slice(pl, scene, x0)
            tick(0)
            pl.render()

    pl.add_key_event("space", lambda: state.update(play=not state["play"]))
    pl.add_key_event("s", lambda: (toggle("stream"), toggle("particles")))
    pl.add_key_event("f", lambda: toggle("fins"))
    pl.add_key_event("h", hull_alpha)
    pl.add_key_event("o", orbit)
    pl.add_key_event("k", sweep)
    pl.add_key_event("r", lambda: default_view(pl, meta))
    pl.add_timer_event(max_steps=10 ** 7, duration=33, callback=tick)
    pl.show()


def movie(folder, meta, flow, args):
    out = os.path.join(folder, f"animation_{args.movie}.mp4")
    pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
    pl.set_background("#0d1117", top="#2d3440")
    scene = build_scene(pl, folder, meta, flow, args)
    pl.add_text(title_of(meta), font_size=10, color="white")
    pl.open_movie(out, framerate=30, quality=8)
    parts = scene["particles"]
    phase = 0.0

    def step_particles():
        nonlocal phase
        if parts is not None:
            phase = (phase + 0.004) % 1.0
            parts.update(phase)

    if args.movie in ("orbit", "both"):
        _, tw = freestream_dir(meta)
        for pos in camera_orbit(args.frames, start=tw - np.radians(35)):
            pl.camera_position = pos
            step_particles()
            pl.write_frame()
    if args.movie in ("sweep", "both") and flow is not None:
        default_view(pl, meta)
        for x0 in np.linspace(-1.0, 72.0, args.frames):
            set_slice(pl, scene, x0)
            step_particles()
            pl.write_frame()
    pl.close()
    print(f"wrote {out}")


def title_of(meta):
    c = meta["condition"]
    w = meta.get("condition_windward", 180.0)
    side = {0.0: "rudder side", 180.0: "tower side", 90.0: "catch fin", 270.0: "catch fin"}.get(float(w), f"{w:g} deg")
    return (f"Super Heavy V3   M {c['mach']}   {c['alt']/1000:.1f} km   {c['aoa_from_engines_first']:g} deg off "
            f"engines-first   crossflow on the {side}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", help="a results folder (loads.json, surface_cp.vtk, flow.vtk)")
    ap.add_argument("--movie", choices=["orbit", "sweep", "both"], help="record an MP4 instead of opening a window")
    ap.add_argument("--frames", type=int, default=360, help="frames per movie segment (30 fps)")
    ap.add_argument("--stride", type=int, default=1, help="read every n-th flow point (faster, less memory)")
    ap.add_argument("--seeds", type=int, default=16, help="streamline rake size (n x n)")
    ap.add_argument("--no-flow", action="store_true", help="surface only (skip the 3-D flow file)")
    ap.add_argument("--fin-stl", default=None)
    args = ap.parse_args(argv)
    folder = args.folder
    meta = json.load(open(os.path.join(folder, "loads.json")))
    flow = None
    fp = os.path.join(folder, "flow.vtk")
    if not args.no_flow and os.path.exists(fp):
        print(f"reading {fp} (stride {args.stride}) ...", flush=True)
        flow = load_flow(fp, args.stride)
    if args.movie:
        movie(folder, meta, flow, args)
    else:
        interactive(folder, meta, flow, args)


if __name__ == "__main__":
    main()
