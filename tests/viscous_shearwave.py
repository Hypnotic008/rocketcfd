"""Viscous terms: a sinusoidal shear wave in a periodic box decays as exp(-nu k^2 t).
Measured effective viscosity / set viscosity should be 1 (0.999 on 64 cells)."""
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from rocketcfd import Solver
R, g = 287.05, 1.4
T, p = 288.15, 101325.0
rho = p / (R * T); c = np.sqrt(g * R * T)
Ly = 1.0; N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
U0 = float(sys.argv[2]) * c if len(sys.argv) > 2 else 0.05 * c
mu = 2.0                                   # nu = mu / rho ~ 1.6 m^2/s: decays in ~ (Ly/2pi)^2/nu
nu = mu / rho
k = 2 * np.pi / Ly
for mach_x in (0.0, 0.4):
    fs = dict(rho=rho, u=mach_x * c, v=0.0, w=0.0, p=p)
    s = Solver((0, 0, 0), (1.0 / N, Ly, 1.0 / N), 1.0 / N, fs,
               bcs={(0, 0): "periodic", (0, 1): "periodic", (1, 0): "periodic", (1, 1): "periodic",
                    (2, 0): "periodic", (2, 1): "periodic"},
               viscous=dict(model="constant", mu=mu, R=R, Pr=0.72), verbose=False, cfl=0.5)
    y = s.axes[1]
    W = np.zeros((5,) + s.phi.shape)
    W[0] = rho; W[4] = p
    W[1] = mach_x * c + U0 * np.sin(k * y)[None, :, None]
    s.set_primitive(W)
    t_end = 0.3 / (nu * k * k)
    while s.t < t_end:
        s.run(50, chunk=50, first_order_steps=0)
    f = s.fields()
    amp = np.sqrt(2) * np.sqrt(np.mean((f["u"][0, :, 0] - mach_x * c) ** 2))
    nu_eff = -np.log(amp / U0) / (k * k * s.t)
    print(f"N {N}  background Mach {mach_x}:  nu_eff / nu = {nu_eff/nu:.4f}  ({s.step_count} steps)")
