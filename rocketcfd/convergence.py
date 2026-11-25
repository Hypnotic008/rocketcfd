"""Stopping test for steady runs.

Requiring the last few load checks to agree is not enough: a subsonic case can creep
toward its answer so slowly that consecutive checks agree while the result is still
several percent away. The test therefore also fits a straight line through a longer
window of checks and requires the projected drift over the next `horizon` checks to be
below the tolerance as well.
"""
import numpy as np


def settled(history, tol, window=4, trend_window=10, horizon=10):
    """history: list of tuples of monitored values, one per load check.
    tol: scalar or per-value tolerances (absolute)."""
    h = np.asarray(history, float)
    if h.ndim != 2 or len(h) < max(window, trend_window):
        return False
    tol = np.broadcast_to(np.asarray(tol, float), (h.shape[1],))
    recent = h[-window:]
    if np.any(np.abs(recent - recent[-1]).max(0) >= tol):
        return False
    y = h[-trend_window:]
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]        # change per check
    return bool(np.all(np.abs(slope) * horizon < tol))
