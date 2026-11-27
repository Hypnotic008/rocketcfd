"""rocketcfd.adapt: averaged-criterion patches stay within the cell budget, cover where the
criterion is largest, and skip what an existing, finer patch already refines.

  python tests/adapt_budget.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd.adapt import adapted_patches, goal_weight


def synthetic():
    f = [np.linspace(0, 40, 81), np.linspace(-10, 10, 41), np.linspace(-10, 10, 41)]   # 0.5 m cells
    X = np.meshgrid(*[0.5 * (a[1:] + a[:-1]) for a in f], indexing="ij")
    # a "shock" sheet at x = 10 and a "wake" blob at (30, 0, 0)
    C = 50 * np.exp(-((X[0] - 10) / 0.8) ** 2) + 30 * np.exp(-((X[0] - 30) ** 2 + X[1] ** 2 + X[2] ** 2) / 8.0)
    return dict(C=C, faces=f, fluid=np.ones(C.shape, bool), samples=10), f, X


def main():
    crit, f, X = synthetic()
    ok = True
    for budget in (2e5, 6e5):
        pf, thr = adapted_patches(crit, f, 0.25, budget, verbose=False)
        n = sum(int(np.prod([len(a) - 1 for a in p])) for p in pf)
        inside = lambda x, y, z: any(p[0][0] <= x <= p[0][-1] and p[1][0] <= y <= p[1][-1] and p[2][0] <= z <= p[2][-1]
                                     for p in pf)
        cov = inside(10.0, 0.0, 0.0)
        print(f"budget {budget:.0e}: {len(pf)} patches, {n} cells (<= budget: {n <= budget}), "
              f"shock sheet covered: {cov}")
        ok &= n <= budget and cov and n > 0.3 * budget
    # an existing finer patch over the shock: nothing new there
    existing = [[np.linspace(8, 12, 41), np.linspace(-10, 10, 201), np.linspace(-10, 10, 201)]]
    pf, _ = adapted_patches(crit, f, 0.25, 2e5, existing=existing, verbose=False)
    overlap = any(p[0][0] < 11.9 and p[0][-1] > 8.1 and p[1][0] < 9 and p[1][-1] > -9 for p in pf)
    print(f"existing finer patch over the shock: new patches overlapping it: {overlap}")
    ok &= not overlap
    # goal weighting pulls the budget toward a body at x = 30
    sdf = lambda x, y, z: np.sqrt((x - 30) ** 2 + y ** 2 + z ** 2) - 1.0
    pfw, _ = adapted_patches(crit, f, 0.25, 2e5, weight=goal_weight(crit, sdf), verbose=False)
    wake = any(p[0][0] <= 30 <= p[0][-1] for p in pfw)
    print(f"loads-weighted: the region at the body (x = 30) refined: {wake}")
    ok &= wake
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
