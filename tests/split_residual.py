"""The split residual (face-flux kernels + divergence, the default) against the fused
one-thread-per-cell residual_kernel, on states that exercise every branch: immersed-boundary
ghost cells, thin walls with plate normals, laminar no-slip walls, the k-omega wall function,
LES, low-Mach correction, shock switch, 2-D (inactive direction) and stretched grids.

  python tests/split_residual.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from rocketcfd import Solver


def sphere(X, Y, Z):
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 0.5


def plate(X, Y, Z):          # inclined thin plate
    a = np.radians(20.0)
    n = (-np.sin(a), np.cos(a), 0.0)
    d = X * n[0] + Y * n[1]
    along = X * np.cos(a) + Y * np.sin(a)
    return np.maximum(np.abs(d) - 0.004, np.abs(along) - 0.5)


def cases():
    fs = dict(rho=1.0, u=600.0, v=0.0, w=0.0, p=1e5)
    slow = dict(rho=1.2, u=30.0, v=0.0, w=0.0, p=1e5)
    yield "sphere M1.7 IB (Euler)", dict(lo=(-1.5, -1, -1), hi=(1.5, 1, 1), h=0.06, freestream=fs, sdf=sphere)
    yield "thin plate (Euler, normals)", dict(lo=(-1, -0.6, -0.12), hi=(1, 0.6, 0.12), h=0.04, freestream=fs,
                                              thin_sdf=plate, thin_box=((-0.6, -0.3, -0.12), (0.6, 0.3, 0.12)))
    yield "sphere laminar no-slip, low Mach", dict(lo=(-1.5, -1, -1), hi=(1.5, 1, 1), h=0.06, freestream=slow,
                                                   sdf=sphere, viscous=dict(model="constant", mu=0.02))
    yield "sphere SST wall function", dict(lo=(-1.5, -1, -1), hi=(1.5, 1, 1), h=0.06, freestream=fs, sdf=sphere,
                                           viscous=dict(model="sutherland"), turbulence=dict(model="sst"))
    yield "plate LES wall model", dict(lo=(-1, -0.6, -0.12), hi=(1, 0.6, 0.12), h=0.04, freestream=slow,
                                       thin_sdf=plate, thin_box=((-0.6, -0.3, -0.12), (0.6, 0.3, 0.12)),
                                       viscous=dict(model="sutherland"), turbulence=dict(model="les", sgs="vreman"))
    yield "2-D cylinder, stretched", dict(faces=[np.cumsum(np.r_[-1.5, np.full(40, 0.04) * 1.02 ** np.arange(40)]),
                                                 np.linspace(-1, 1, 41), np.array([0.0, 0.05])],
                                          lo=None, hi=None, h=None, freestream=fs,
                                          sdf=lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - 0.3,
                                          bcs={(0, 0): "freestream", (0, 1): "extrapolate", (1, 0): "freestream",
                                               (1, 1): "freestream", (2, 0): "periodic", (2, 1): "periodic"})


def compare(kw):
    kw = dict(kw)
    lo, hi, h, fs = kw.pop("lo"), kw.pop("hi"), kw.pop("h"), kw.pop("freestream")
    s = Solver(lo, hi, h, fs, verbose=False, **kw)
    s.run(30, chunk=30, first_order_steps=10, steady=True)          # a developing, non-uniform state
    out = []
    with torch.no_grad():
        for split in (False, True):
            s.split_residual = split
            s._wprim("U")
            s._wresidual(1.0)
            out.append(s.R_buf.clone())
    m = s.mask_f.bool()
    a, b = out[0][:, m], out[1][:, m]
    scale = a.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return float(((a - b).abs() / scale).max())


if __name__ == "__main__":
    worst = 0.0
    for name, kw in cases():
        d = compare(kw)
        worst = max(worst, d)
        print(f"{name:36s} max |split - fused| / max|R| = {d:.2e}", flush=True)
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
