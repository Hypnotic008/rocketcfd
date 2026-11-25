"""Local refinement with turbulence: flat plate, Re 1e7, k-omega SST + wall function, implicit.
A patch of fine cells around the plate on a coarse base grid against one fine grid.

  python tests/patch_turbulent_plate.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces
from rocketcfd.patches import PatchGrid

T, P = 288.15, 101325.0
R, G = 287.05, 1.4
RHO = P / (R * T)
U = 0.4 * np.sqrt(G * R * T)
RE = 1e7
MU = RHO * U / RE
FS = dict(rho=RHO, u=U, v=0.0, w=0.0, p=P)
HX = 0.02


def plate(X, Y, Z):
    a = np.abs(X - 0.5) - 0.5
    b = np.abs(Y) - 5e-5
    return np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b, 0) ** 2) + np.minimum(np.maximum(a, b), 0)


def faces(hy):
    return [axis_faces(-0.3, 4.0, 6 * HX, [(-0.05, 1.05, HX)], 1.08),
            axis_faces(-0.6, 0.6, 0.05, [(-0.03, 0.03, hy)], 1.08), np.array([0.0, HX])]


KW = dict(thin_sdf=plate, bcs={(2, 0): "periodic", (2, 1): "periodic"},
          viscous=dict(model="constant", mu=MU, R=R, Pr=0.72), turbulence=dict(model="sst"), verbose=False)


def drag(obj):
    return obj.loads(S_ref=HX, L_ref=1.0)["F_friction"][0] / (0.5 * RHO * U ** 2 * HX)


if __name__ == "__main__":
    Rex = RHO * U / MU
    white = 2 * 0.455 / np.log10(Rex) ** 2.58          # Schlichting integral, both sides
    print(f"reference friction drag (both sides, fully turbulent, Prandtl-Schlichting): {white:.5f}")
    for mode in sys.argv[1:] or ["single", "patch"]:
        t0 = time.time()
        if mode == "single":
            obj = Solver(None, None, None, FS, faces=faces(0.0016), **KW)
            obj.run(3000, chunk=500, first_order_steps=200, steady=True, implicit=True, cfl_ramp=200)
        else:
            bf = faces(0.0064)
            yf = np.round(np.arange(-0.0336, 0.0336 + 1e-9, 0.0016), 10)
            xf = bf[0][(bf[0] >= -0.1 - 1e-9) & (bf[0] <= 1.1 + 1e-9)]
            obj = PatchGrid(bf, [[xf, yf, bf[2]]], FS, **KW)
            obj.verbose = False
            obj.run(3000, chunk=500, first_order_steps=200, cfl_ramp=200)
        cells = sum(int(np.prod(g.n)) for g in obj.grids) if mode == "patch" else int(np.prod(obj.n))
        print(f"{mode:6s} cells {cells:6d}  friction drag {drag(obj):.5f}  {time.time()-t0:.0f} s", flush=True)
