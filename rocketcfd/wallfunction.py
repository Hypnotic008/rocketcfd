"""Law of the wall (Spalding) on the host side, matching the solver's wall function."""
import numpy as np

KAPPA, B = 0.41, 5.0


def spalding_utau(u, y, nu):
    """Friction velocity from Spalding's law, vectorised (u, y, nu arrays or scalars)."""
    u = np.maximum(np.asarray(u, float), 1e-12)
    y = np.asarray(y, float)
    nu = np.asarray(nu, float)
    ut = np.sqrt(nu * u / y)
    lg = y * ut / nu > 11
    for _ in range(8):
        ut = np.where(lg, KAPPA * u / np.log(np.maximum(9.0 * y * ut / nu, 1.0001)), ut)
    for _ in range(12):
        up = u / ut
        kp = KAPPA * up
        e = np.exp(-KAPPA * B)
        f = up + e * (np.exp(kp) - 1 - kp - 0.5 * kp ** 2 - kp ** 3 / 6) - y * ut / nu
        dfdup = 1 + e * KAPPA * (np.exp(kp) - 1 - kp - 0.5 * kp ** 2)
        df = dfdup * (-up / ut) - y / nu
        ut = np.maximum(ut - f / df, 0.1 * ut)
    return ut
