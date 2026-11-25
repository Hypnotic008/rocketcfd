"""Local refinement check with an immersed (thick) body: Mach 3 sphere, inviscid.

  single : stretched grid with h_fine in a box around the sphere
  coarse : the same with 2 h_fine there
  patch  : the coarse grid plus wall_patches of h_fine cells within `shell` of the surface

The patch run should reproduce `single` (drag and side force) while the coarse run differs.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import Zone, stretched_faces
from rocketcfd.patches import PatchGrid, wall_patches

FS = dict(rho=1.4, u=3.0, v=0.0, w=0.3, p=1.0)
R0 = 0.4
SDF = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - R0
REF = dict(S_ref=np.pi * R0 ** 2, L_ref=2 * R0, x_ref=(0.0, 0.0, 0.0))
H = 0.04


def faces(h_near):
    return stretched_faces((-1.6, -1.6, -1.6), (2.4, 1.6, 1.6), 0.16,
                           [Zone((-0.56, -0.56, -0.56), (0.64, 0.56, 0.56), h_near)], growth=1.12)


def run(mode, iters=600, shell=0.24):
    t0 = time.time()
    kw = dict(sdf=SDF, verbose=False)
    if mode == "patch":
        bf = faces(2 * H)
        pf = wall_patches(bf, H, shell, sdf=SDF, box=(12, 12, 12), verbose=True)
        obj = PatchGrid(bf, pf, FS, **kw)
        obj.verbose = False
        obj.run(iters, chunk=100, first_order_steps=100, cfl_max=20.0, cfl_ramp=100)
        cells = sum(int(np.prod(g.n)) for g in obj.grids)
    else:
        obj = Solver(None, None, None, FS, faces=faces(H if mode == "single" else 2 * H), **kw)
        obj.run(iters, chunk=100, first_order_steps=100, steady=True, implicit=True, cfl_max=20.0, cfl_ramp=100)
        cells = int(np.prod(obj.n))
    ld = obj.loads(**REF)
    return dict(CD=ld["CD"], CN=ld["CN"], cells=cells, wall=time.time() - t0)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["coarse", "patch", "single"]:
        r = run(m)
        print(f"{m:7s} cells {r['cells']:8d}  CD {r['CD']:.4f}  CN {r['CN']:+.4f}  {r['wall']:.0f} s", flush=True)
