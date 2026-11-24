"""Export the parametric V3 grid fin as a watertight STL and render it.

  python cases/grid_fin_geometry.py               (4 mm voxels, ~1-2 min)
  python cases/grid_fin_geometry.py 0.003 --no-teeth
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rocketcfd.vehicles.grid_fin import GridFinParams, grid_fin_sdf, grid_fin_box, planform_area
from rocketcfd.output import write_stl_chunked

args = [a for a in sys.argv[1:] if not a.startswith("--")]
h = float(args[0]) if args else 0.004
P = GridFinParams(teeth="--no-teeth" not in sys.argv)
out = "results"
os.makedirs(out, exist_ok=True)
print(P.summary(), f"| planform {planform_area(P):.2f} m^2")
lo, hi = grid_fin_box(P, pad=2 * h)
t0 = time.time()
name = f"{out}/grid_fin_v3{'' if P.teeth else '_noteeth'}.stl"
tri = write_stl_chunked(lambda X, Y, Z: grid_fin_sdf(X, Y, Z, P), lo, hi, h, name)
print(f"{name}: {len(tri)} triangles, {time.time()-t0:.0f} s  (units: metres; x = through the fin, "
      f"toothed face at x = -{P.depth/2:g}; y = around the hull; z = out from the hull)")


def shade(t, light):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return 0.3 + 0.7 * np.abs(n @ light)


fig = plt.figure(figsize=(16, 8))
views = ((1, "whole fin, toothed face (toward the engines)", None, (-60, 25)),
         (2, "close-up of the teeth", ((-0.3, 0.3), (-0.2, 0.75), (2.0, 2.9)), (-55, 30)))
light = np.array([-0.6, -0.4, 0.7]) / np.linalg.norm([-0.6, -0.4, 0.7])
for k, title, win, (az, el) in views:
    ax = fig.add_subplot(1, 2, k, projection="3d")
    t = tri
    if win is not None:
        c = t.mean(1)
        m = np.all([(c[:, d] > win[d][0]) & (c[:, d] < win[d][1]) for d in range(3)], axis=0)
        t = t[m]
    else:
        t = t[:: max(1, len(t) // 400000)]
    ax.add_collection3d(Poly3DCollection(t, facecolors=plt.cm.Greys(0.25 + 0.6 * shade(t, light)), edgecolor="none"))
    mn, mx = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
    for setl, a_, b_ in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mn, mx):
        setl(a_, b_)
    ax.set_box_aspect(tuple(np.maximum(mx - mn, 1e-3) / (mx - mn).max()))
    ax.view_init(elev=el, azim=az)
    ax.set_title(title)
    ax.set_xlabel("x through fin")
    ax.set_ylabel("y around hull")
    ax.set_zlabel("z from hull")
fig.tight_layout()
fig.savefig(f"{out}/grid_fin_v3_render.png", dpi=120)
print(f"{out}/grid_fin_v3_render.png")
