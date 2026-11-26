"""Quick checks of the flow-movie helpers in rocketcfd.cinematic (no solver run, a few seconds).

  python tests/cinematic_helpers.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from rocketcfd.cinematic import _fill_solid, _pitch_plane_normal, _shock_sensor


def main():
    # nearest-fluid fill: a solid block takes the values of the fluid next to it
    a = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    solid = np.zeros_like(a, bool)
    solid[2, :, :] = True
    f = _fill_solid(a, solid)
    assert np.all(f[~solid] == a[~solid])
    assert np.all((f[2] == a[1]) | (f[2] == a[3])), "solid cells must copy a neighbouring fluid value"

    # shock sensor: a 2:1 density step resolved over ~2 cells reads ~0.3-0.5, smooth flow ~0
    x = np.linspace(0, 10, 101); y = z = np.linspace(0, 1, 6)
    X = np.meshgrid(x, y, z, indexing="ij")[0]
    rho = 1.0 + 1.0 / (1.0 + np.exp(-(X - 5.0) / 0.08))
    s = _shock_sensor(rho, x, y, z, np.zeros_like(rho, bool))
    assert 0.2 < s.max() < 0.6, s.max()
    smooth = 1.0 + 0.1 * X / 10.0
    assert _shock_sensor(smooth, x, y, z, np.zeros_like(rho, bool)).max() < 0.01

    # pitch plane: flow along +x with crossflow in z -> plane xz (normal +-y); pure axial -> y
    n = _pitch_plane_normal(np.array([0.99, 0.0, 0.14]))
    assert abs(abs(n[1]) - 1.0) < 1e-9
    n = _pitch_plane_normal(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(n, [0, 1, 0])
    n = _pitch_plane_normal(np.array([0.9, 0.3, 0.0]))
    assert abs(abs(n[2]) - 1.0) < 1e-9
    print("cinematic helpers: OK")


if __name__ == "__main__":
    main()
