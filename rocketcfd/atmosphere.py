"""U.S. Standard Atmosphere 1976, 0-86 km geometric altitude."""
import numpy as np

R_AIR = 287.0531  # J/(kg K)
G0 = 9.80665
R_EARTH = 6356766.0

# geopotential base heights [m], lapse rates [K/m], base temperatures [K], base pressures [Pa]
_HB = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
_LB = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]
_TB = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
_PB = [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420]


def atmosphere(z):
    """Return (T [K], p [Pa], rho [kg/m^3], a [m/s]) at geometric altitude z [m]."""
    if not 0.0 <= z <= 86000.0:
        raise ValueError("US76 model here covers 0-86 km")
    h = R_EARTH * z / (R_EARTH + z)
    i = max(k for k in range(7) if _HB[k] <= h)
    L, Tb, Pb, hb = _LB[i], _TB[i], _PB[i], _HB[i]
    T = Tb + L * (h - hb)
    if L == 0.0:
        p = Pb * np.exp(-G0 * (h - hb) / (R_AIR * Tb))
    else:
        p = Pb * (Tb / T) ** (G0 / (R_AIR * L))
    rho = p / (R_AIR * T)
    a = np.sqrt(1.4 * R_AIR * T)
    return T, p, rho, a
