"""Render the parametric Super Heavy V3 geometry (with the CAD grid fins) and export STLs."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd.vehicles.super_heavy import super_heavy_sdf, v3_params
from rocketcfd.output import write_stl, render_mesh

out = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(out, exist_ok=True)
P = v3_params()
sdf = lambda X, Y, Z: super_heavy_sdf(X, Y, Z, P)
t0 = time.time()
v, f = write_stl(sdf, (-1.0, -10.5, -10.5), (73.0, 10.5, 10.5), 0.15, f"{out}/super_heavy_v3.stl")
print(len(f), "triangles", f"{time.time()-t0:.0f} s", flush=True)
render_mesh(v, f, f"{out}/super_heavy_v3_side.png", "Super Heavy V3 (dimensions from photo-measured estimates)",
            elev=12, azim=-70)
c = v[f].mean(1)
render_mesh(v, f[c[:, 0] < 4], f"{out}/super_heavy_v3_aft.png",
            "Aft end: 33 exposed Raptor 3 bells on a tapered thrust plate", elev=-40, azim=-20)
# fine close-up of the top with the lattice resolved
# fine close-up of the +z grid fin with the lattice resolved
xf = P.fin_x_center
v2, f2 = write_stl(sdf, (xf - 2.5, -3.0, 2.5), (xf + 2.5, 3.0, 10.0), 0.025, f"{out}/super_heavy_v3_fin.stl")
render_mesh(v2, f2, f"{out}/super_heavy_v3_fin.png",
            "Grid fin: CatchSim CAD scaled to 3.5 m x 4.2 m x 0.5 m (lattice resolved)", elev=25, azim=-35)
print("done", f"{time.time()-t0:.0f} s")
