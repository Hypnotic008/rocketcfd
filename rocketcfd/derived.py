"""Derived flow quantities for visualisation, on rectilinear grids (numpy or a PyVista
RectilinearGrid / ImageData from flow.vtk).

  schlieren  numerical schlieren: exp(-k |grad rho| / max|grad rho|) (dark = strong gradient),
             the shadowgraph-like look of NASA's LAVA / Fun3D movies (Quirk 1994 mapping)
  Q          Q-criterion 0.5 (|Omega|^2 - |S|^2): vortex cores where Q > 0
  vorticity  |curl u|
  T          temperature p / (rho R)
  Cp         pressure coefficient (p - p_inf) / q_inf
  p0_ratio   local total pressure over freestream total pressure (shock / viscous losses)
  lambda2    second eigenvalue of S^2 + Omega^2 (Jeong & Hussain 1995): vortex cores where < 0
"""
import numpy as np

R_AIR = 287.05

DESCRIPTIONS = {
    "mach": "Mach number",
    "p": "static pressure [Pa]",
    "rho": "density [kg/m^3]",
    "T": "temperature [K]",
    "Cp": "pressure coefficient",
    "schlieren": "numerical schlieren (density gradient, dark = shock / shear layer)",
    "grad_rho": "density gradient magnitude [kg/m^4]",
    "vorticity": "vorticity magnitude [1/s]",
    "Q": "Q-criterion [1/s^2] (> 0: vortex core)",
    "lambda2": "lambda-2 [1/s^2] (< 0: vortex core)",
    "p0_ratio": "total pressure / freestream total pressure",
    "speed": "velocity magnitude [m/s]",
    "mut_ratio": "eddy / molecular viscosity",
}


def _grad(a, x, y, z):
    """Gradient of a (nx, ny, nz) array on (possibly stretched) rectilinear axes."""
    out = []
    for d, c in enumerate((x, y, z)):
        if len(c) < 2:
            out.append(np.zeros_like(a))
        else:
            out.append(np.gradient(a, c, axis=d))
    return out


def velocity_gradient(u, v, w, x, y, z):
    """J[i][j] = d u_i / d x_j, each (nx, ny, nz)."""
    return [_grad(c, x, y, z) for c in (u, v, w)]


def q_criterion(J):
    s2 = 0.0
    o2 = 0.0
    for i in range(3):
        for j in range(3):
            s = 0.5 * (J[i][j] + J[j][i])
            o = 0.5 * (J[i][j] - J[j][i])
            s2 = s2 + s * s
            o2 = o2 + o * o
    return 0.5 * (o2 - s2)


def lambda2(J):
    A = np.empty(J[0][0].shape + (3, 3))
    for i in range(3):
        for j in range(3):
            A[..., i, j] = J[i][j]
    S = 0.5 * (A + np.swapaxes(A, -1, -2))
    O = 0.5 * (A - np.swapaxes(A, -1, -2))
    M = S @ S + O @ O
    return np.linalg.eigvalsh(M)[..., 1]


def vorticity(J):
    wx = J[2][1] - J[1][2]
    wy = J[0][2] - J[2][0]
    wz = J[1][0] - J[0][1]
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def schlieren(rho, x, y, z, k=10.0, solid=None):
    g = _grad(rho, x, y, z)
    gm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    if solid is not None:
        gm = np.where(solid, 0.0, gm)
    ref = np.percentile(gm[gm > 0], 99.0) if np.any(gm > 0) else 1.0
    return np.exp(-k * np.minimum(gm / max(ref, 1e-30), 1.0)), gm


def compute(fields, x, y, z, names, freestream=None, gamma=1.4, R=R_AIR):
    """fields: dict of (nx, ny, nz) arrays with rho, p, u, v, w (and optionally mach, phi, mut).
    Returns {name: array} for the requested derived names."""
    out = {}
    rho, p = fields["rho"], fields["p"]
    u, v, w = fields["u"], fields["v"], fields["w"]
    solid = fields["phi"] < 0 if "phi" in fields else None
    need_J = any(n in names for n in ("Q", "vorticity", "lambda2"))
    J = velocity_gradient(u, v, w, x, y, z) if need_J else None
    fs = freestream or {}
    for n in names:
        if n == "T":
            out[n] = p / (rho * R)
        elif n == "speed":
            out[n] = np.sqrt(u * u + v * v + w * w)
        elif n == "mach" and "mach" not in fields:
            out[n] = np.sqrt(u * u + v * v + w * w) / np.sqrt(gamma * p / rho)
        elif n in ("schlieren", "grad_rho"):
            sch, gm = schlieren(rho, x, y, z, solid=solid)
            out["schlieren"], out["grad_rho"] = sch, gm
        elif n == "Q":
            out[n] = q_criterion(J)
        elif n == "lambda2":
            out[n] = lambda2(J)
        elif n == "vorticity":
            out[n] = vorticity(J)
        elif n == "Cp":
            if fs:
                q = 0.5 * fs["rho"] * (fs["u"] ** 2 + fs["v"] ** 2 + fs["w"] ** 2)
                out[n] = (p - fs["p"]) / max(q, 1e-30)
        elif n == "p0_ratio":
            M2 = (u * u + v * v + w * w) / (gamma * p / rho)
            p0 = p * (1 + 0.5 * (gamma - 1) * M2) ** (gamma / (gamma - 1))
            if fs:
                M2i = (fs["u"] ** 2 + fs["v"] ** 2 + fs["w"] ** 2) / (gamma * fs["p"] / fs["rho"])
                out[n] = p0 / (fs["p"] * (1 + 0.5 * (gamma - 1) * M2i) ** (gamma / (gamma - 1)))
        elif n == "mut_ratio" and "mut" in fields:
            T = p / (rho * R)
            mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
            out[n] = fields["mut"] / mu
    if solid is not None:
        for n in out:
            if n != "schlieren":
                out[n] = np.where(solid, np.nan if n not in ("Q", "lambda2") else 0.0, out[n])
    return out


def grid_arrays(g):
    """(fields, x, y, z) as (nx, ny, nz) arrays from a PyVista RectilinearGrid / ImageData."""
    import pyvista as pv
    nx, ny, nz = g.dimensions
    if isinstance(g, pv.RectilinearGrid):
        x, y, z = np.asarray(g.x), np.asarray(g.y), np.asarray(g.z)
    else:
        o, sp = g.origin, g.spacing
        x, y, z = (o[d] + sp[d] * np.arange(n) for d, n in enumerate((nx, ny, nz)))
    f = {}
    for name in g.point_data:
        a = np.asarray(g.point_data[name])
        if a.ndim == 2 and a.shape[1] == 3:
            for c, cn in enumerate("uvw"):
                f[cn] = a[:, c].reshape((nz, ny, nx)).transpose(2, 1, 0)
        else:
            f[name] = a.reshape((nz, ny, nx)).transpose(2, 1, 0)
    return f, x, y, z


def add_to_grid(g, names, freestream=None):
    """Compute derived fields and add them to the PyVista grid's point data (in place)."""
    f, x, y, z = grid_arrays(g)
    out = compute(f, x, y, z, names, freestream)
    for n, a in out.items():
        g.point_data[n] = np.ascontiguousarray(a.transpose(2, 1, 0)).reshape(-1).astype(np.float32)
    return list(out)
