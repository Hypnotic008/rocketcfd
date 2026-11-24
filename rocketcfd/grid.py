"""Stretched Cartesian grids with refinement zones ("bodies of influence").

Each zone is a box with a target cell size. Along each axis the cell size is the
smallest target of any zone covering that coordinate, and away from the zones it grows
by at most `growth` per cell (like an inflation-layer growth ratio) up to `base`.

This is a tensor-product grid: a zone refines the whole slab it spans along each axis,
not only its own box. True local (octree / patch) refinement is the next step; slabs
are simple, conservative and robust, and already put most cells where they matter.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Zone:
    lo: tuple          # (x, y, z) box corner
    hi: tuple
    h: float           # target cell size inside the box
    name: str = ""


def axis_faces(lo, hi, base, intervals, growth=1.15):
    """Face coordinates along one axis.

    intervals: list of (a, b, h) target sizes on [a, b].
    Cell size at x is min(base, min_k [h_k + (growth - 1) * dist(x, [a_k, b_k])]),
    so neighbouring cells differ by at most a factor `growth`.
    """
    def size(x):
        s = base
        for a, b, h in intervals:
            d = max(a - x, 0.0, x - b)
            s = min(s, h + (growth - 1.0) * d)
        return s

    faces = [lo]
    x = lo
    while x < hi - 1e-9:
        s = size(x)
        s = size(x + 0.5 * s)          # midpoint evaluation keeps growth symmetric
        x = x + s
        faces.append(x)
    faces = np.array(faces)
    # land exactly on hi: stretch the cells proportionally
    faces = lo + (faces - lo) * (hi - lo) / (faces[-1] - lo)
    return faces


def stretched_faces(lo, hi, base, zones=(), growth=1.15):
    """Face coordinates for the three axes."""
    out = []
    for d in range(3):
        iv = [(z.lo[d], z.hi[d], z.h) for z in zones]
        out.append(axis_faces(lo[d], hi[d], base, iv, growth))
    return out


def describe(faces):
    n = [len(f) - 1 for f in faces]
    w = [np.diff(f) for f in faces]
    lines = [f"{n[0]} x {n[1]} x {n[2]} = {np.prod(n)/1e6:.2f} M cells"]
    for d, name in enumerate("xyz"):
        g = np.max(np.maximum(w[d][1:] / w[d][:-1], w[d][:-1] / w[d][1:])) if n[d] > 1 else 1.0
        lines.append(f"  {name}: {n[d]} cells, size {w[d].min():.3f} .. {w[d].max():.3f} m, "
                     f"max growth {g:.3f}")
    return "\n".join(lines)
