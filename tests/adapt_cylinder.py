"""Solution-adaptive refinement check: Mach 4 cylinder (2-D, inviscid).

  fine   : one grid with h_fine around the body and shock
  coarse : 4 h_fine there
  adapt  : the coarse solution, then patches of h_fine on cells with a density jump > 5 %
           per cell (bow shock, expansion) and within 0.1 D of the wall, re-solved

Compares drag and bow-shock standoff (Billig correlation).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import Solver
from rocketcfd.grid import axis_faces
from rocketcfd.patches import PatchGrid, feature_cells, cell_patches, init_from

G, M = 1.4, 4.0
FS = dict(rho=G, u=M, v=0.0, w=0.0, p=1.0)
SDF = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.5
KW = dict(sdf=SDF, bcs={(2, 0): "periodic", (2, 1): "periodic"}, verbose=False)
HF = 0.01


def faces(h):
    return [axis_faces(-2.0, 1.5, 8 * HF * 4, [(-1.2, 0.6, h)], 1.08),
            axis_faces(-2.5, 2.5, 8 * HF * 4, [(-1.4, 1.4, h)], 1.08), np.array([0.0, HF])]


def standoff(grids):
    """Shock position on the stagnation line from the finest grid at each point."""
    xs = np.linspace(-1.2, -0.51, 700)
    best = np.full(xs.size, np.inf)
    p = np.zeros(xs.size)
    for g in grids:
        f = g.fields()
        j = np.argmin(np.abs(f["y"]))
        if abs(f["y"][j]) > 0.6 * g.h[1]:
            continue
        inside = (xs > g.lo[0]) & (xs < g.hi[0])
        vol = np.prod(g.h)
        use = inside & (vol < best)
        p[use] = np.interp(xs[use], f["x"], f["p"][:, j, 0])
        best[use] = vol
    i = np.flatnonzero(p > 0.5 * (1.0 + p.max()))[0]
    billig = 0.5 * 0.386 * np.exp(4.67 / M ** 2)
    return (-0.5 - xs[i]) / billig - 1


def cd(obj):
    return obj.loads(S_ref=1.0 * HF, L_ref=1.0)["CD"]


if __name__ == "__main__":
    t0 = time.time()
    fine = Solver(None, None, None, FS, faces=faces(HF), **KW)
    fine.run(1500, chunk=500, first_order_steps=100, steady=True, implicit=True, cfl_ramp=100)
    print(f"fine   cells {int(np.prod(fine.n)):7d}  CD {cd(fine):.4f}  standoff {100*standoff([fine]):+.1f} %  "
          f"{time.time()-t0:.0f} s", flush=True)
    t0 = time.time()
    coarse = Solver(None, None, None, FS, faces=faces(4 * HF), **KW)
    coarse.run(1000, chunk=500, first_order_steps=100, steady=True, implicit=True, cfl_ramp=100)
    print(f"coarse cells {int(np.prod(coarse.n)):7d}  CD {cd(coarse):.4f}  standoff {100*standoff([coarse]):+.1f} %  "
          f"{time.time()-t0:.0f} s", flush=True)
    t0 = time.time()
    cen, half = feature_cells(coarse, threshold=0.05, wall_shell=0.1)
    pf = cell_patches(faces(4 * HF), (HF, HF, HF), cen, half, box=(64, 64, 1))
    pg = PatchGrid(faces(4 * HF), pf, FS, **KW)
    pg.verbose = False
    init_from(pg, coarse)
    pg.run(1000, chunk=500, first_order_steps=0, cfl_start=5.0, cfl_ramp=100)
    cells = sum(int(np.prod(g.n)) for g in pg.grids)
    print(f"adapt  cells {cells:7d}  CD {cd(pg):.4f}  standoff {100*standoff(pg.grids):+.1f} %  "
          f"({len(pf)} patches)  {time.time()-t0:.0f} s", flush=True)
