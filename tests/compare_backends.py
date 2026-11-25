"""Check that the fused Warp kernels reproduce the PyTorch reference path."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from rocketcfd import Solver


def case(name):
    if name == "sod":
        N = 200
        bcs = {(0, 0): "extrapolate", (0, 1): "extrapolate", (1, 0): "periodic", (1, 1): "periodic",
               (2, 0): "periodic", (2, 1): "periodic"}
        kw = dict(lo=(0, 0, 0), hi=(1, 1 / N, 1 / N), h=1 / N, freestream=dict(rho=1, u=0, v=0, w=0, p=1),
                  bcs=bcs, cfl=0.5)

        def init(s):
            x = s.axes[0]
            W = np.zeros((5,) + s.phi.shape)
            left = (x < 0.5)[:, None, None]
            W[0] = np.where(left, 1.0, 0.125)
            W[4] = np.where(left, 1.0, 0.1)
            s.set_primitive(W)
        return kw, init, 60
    if name == "cylinder":
        h = 0.05
        kw = dict(lo=(-3, -4.5, 0), hi=(3.5, 4.5, h), h=h, freestream=dict(rho=1.4, u=4.0, v=0, w=0, p=1.0),
                  sdf=lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 1.0, bcs={(2, 0): "periodic", (2, 1): "periodic"})
        return kw, None, 150
    if name == "sphere3d":
        from rocketcfd.grid import Zone, stretched_faces
        faces = stretched_faces((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 0.12, [Zone((-0.6,) * 3, (0.6,) * 3, 0.05)], 1.15)
        kw = dict(lo=None, hi=None, h=None, faces=faces, freestream=dict(rho=1.4, u=3.0, v=0.2, w=0.4, p=1.0),
                  sdf=lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 0.4)
        return kw, None, 80


def main():
    worst = 0.0
    for name in ("sod", "cylinder", "sphere3d"):
        kw, init, steps = case(name)
        out = {}
        for be in ("torch", "warp"):
            s = Solver(**kw, backend=be, verbose=False, low_mach=False)   # the torch path has no low-Mach fix
            if init:
                init(s)
            t0 = time.time()
            s.run(steps, chunk=steps, first_order_steps=steps // 3)
            out[be] = (s.U.cpu().numpy().astype(np.float64), s.t, time.time() - t0, s.history[-1][2])
        Ut, Uw = out["torch"][0], out["warp"][0]
        scale = np.abs(Ut).reshape(5, -1).max(1)[:, None, None, None] + 1e-30
        err = np.max(np.abs(Ut - Uw) / scale)
        worst = max(worst, err)
        print(f"{name:9s} steps {steps:4d}  max rel diff {err:.2e}  t {out['torch'][1]:.6f} vs {out['warp'][1]:.6f}  "
              f"res {out['torch'][3]:.4e} vs {out['warp'][3]:.4e}  time torch {out['torch'][2]:.1f}s warp {out['warp'][2]:.1f}s")
    print("PASS" if worst < 1e-3 else "FAIL", f"(worst {worst:.2e})")


if __name__ == "__main__":
    main()
