"""Local refinement check: laminar flat plate (Blasius) with the boundary layer resolved
by a patch instead of by the whole grid.

  single : one stretched grid with hy (fine) across a +/-2 cm band around the plate
  coarse : the same with 4x taller cells next to the plate (boundary layer under-resolved)
  patch  : the coarse grid plus a patch of fine cells around the plate

The patch run should reproduce `single` (Cf and the friction drag from Solver.loads),
and differ clearly from `coarse`. All runs use the implicit solver.
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
RE = 1e5
MU = RHO * U / RE
FS = dict(rho=RHO, u=U, v=0.0, w=0.0, p=P)
THICK = 1e-4


def plate(X, Y, Z):
    a = np.abs(X - 0.5) - 0.5
    b = np.abs(Y) - THICK / 2
    return np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b, 0) ** 2) + np.minimum(np.maximum(a, b), 0)


def base_faces(hy, hx=0.02, x_out=4.0, y_half=0.6):
    return [axis_faces(-0.3, x_out, 6 * hx, [(-0.05, 1.05, hx)], 1.08),
            axis_faces(-y_half, y_half, 12 * 0.0016 * 5, [(-0.02, 0.02, hy)], 1.08),
            np.array([0.0, hx])]


KW = dict(thin_sdf=plate, bcs={(2, 0): "periodic", (2, 1): "periodic"},
          viscous=dict(model="constant", mu=MU, R=R, Pr=0.72), verbose=False)


def cf_from(f):
    x, y = f["x"], f["y"]
    ju = np.flatnonzero(y > 0)[0]
    jl = np.flatnonzero(y < 0)[-1]
    sel = (x > 0.2) & (x < 0.8)
    tau = MU * 0.5 * (f["u"][sel, ju, 0] / y[ju] + f["u"][sel, jl, 0] / (-y[jl]))
    Cf = tau / (0.5 * RHO * U ** 2)
    return np.mean(Cf / (0.664 / np.sqrt(RHO * U * x[sel] / MU)) - 1)


def drag(obj, span):
    ld = obj.loads(S_ref=span * 1.0, L_ref=1.0)
    return ld["F_friction"][0] / (0.5 * RHO * U ** 2 * span), ld["CD"]


def run(mode, iters, cfl):
    t0 = time.time()
    if mode == "patch":
        bf = base_faces(0.0064)
        yf = np.round(np.arange(-0.0224, 0.0224 + 1e-9, 0.0016), 10)
        xf = bf[0][(bf[0] >= -0.1 - 1e-9) & (bf[0] <= 1.1 + 1e-9)]
        obj = PatchGrid(bf, [[xf, yf, bf[2]]], FS, **KW)
        obj.verbose = False
        obj.run(iters, chunk=250, first_order_steps=200, cfl_max=cfl, cfl_ramp=200)
        f = obj.fields(1)
    else:
        obj = Solver(None, None, None, FS, faces=base_faces(0.0016 if mode == "single" else 0.0064), **KW)
        obj.run(iters, chunk=250, first_order_steps=200, steady=True, implicit=True, cfl_max=cfl, cfl_ramp=200)
        f = obj.fields()
    cd_f, cd = drag(obj, 0.02)
    cells = sum(int(np.prod(g.n)) for g in obj.grids) if mode == "patch" else int(np.prod(obj.n))
    return dict(cf=cf_from(f), drag_f=cd_f, cd=cd, cells=cells, wall=time.time() - t0)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    modes = sys.argv[2:] or ["coarse", "patch", "single"]
    # Blasius friction drag of both sides of a plate, per unit span: 2 * 1.328 / sqrt(Re_L)
    print(f"Blasius: Cf on 0.2<x<0.8 matches 0.664/sqrt(Re_x); friction drag coefficient (both sides) "
          f"{2*1.328/np.sqrt(RE):.5f}")
    for m in modes:
        r = run(m, iters, 20.0)
        print(f"{m:7s} cells {r['cells']:7d}  Cf error {100*r['cf']:+6.2f} %  friction drag {r['drag_f']:.5f}  "
              f"(CD incl. pressure {r['cd']:.5f})  {r['wall']:.0f} s", flush=True)
