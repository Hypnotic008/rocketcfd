"""Where does a step's time go? Times explicit steps, implicit iterations and the load
integration on an already-built Solver or PatchGrid, with a per-kernel breakdown.

On CUDA the kernel times come from Warp's native activity timer (GPU time, including the
cached/recorded launches); wall time minus GPU-busy time is Python + launch overhead. On the
CPU every launch is synchronous, so kernels are timed directly.

  python cases/super_heavy_descent.py <same options as the run> --profile 20
"""
import time
from collections import defaultdict

import numpy as np
import torch


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


class _CpuKernelTimer:
    """Wraps Solver._L and warp.launch to time each kernel (CPU device: launches are synchronous)."""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def __enter__(self):
        import warp as wp
        from rocketcfd.solver import Solver
        self._wp, self._S = wp, Solver
        self._L0, self._launch0 = Solver._L, wp.launch
        timer = self

        def L(s, kernel, dim, inputs):
            t0 = time.perf_counter()
            r = timer._L0(s, kernel, dim, inputs)
            timer.t[kernel.key] += time.perf_counter() - t0
            timer.n[kernel.key] += 1
            return r

        def launch(kernel, *a, **k):
            t0 = time.perf_counter()
            r = timer._launch0(kernel, *a, **k)
            if not k.get("record_cmd"):
                timer.t[kernel.key] += time.perf_counter() - t0
                timer.n[kernel.key] += 1
            return r

        Solver._L, wp.launch = L, launch
        return self

    def __exit__(self, *exc):
        self._S._L, self._wp.launch = self._L0, self._launch0

    def results(self):
        return {k: (1e3 * self.t[k], self.n[k]) for k in self.t}


def _kernel_breakdown(dev, fn):
    """Run fn() and return ({kernel: (ms, launches)}, wall_s)."""
    if dev.type == "cuda":
        import warp as wp
        wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        wall = time.perf_counter() - t0
        res = wp.timing_end()
        out = defaultdict(lambda: [0.0, 0])
        for r in res:
            name = r.name.split("_cuda_kernel")[0]
            out[name][0] += r.elapsed
            out[name][1] += 1
        return {k: tuple(v) for k, v in out.items()}, wall
    with _CpuKernelTimer() as tm:
        t0 = time.perf_counter()
        fn()
        wall = time.perf_counter() - t0
    return tm.results(), wall


def _print_breakdown(title, ker, wall, per, unit):
    busy = sum(v[0] for v in ker.values()) / 1e3
    print(f"\n{title}: {wall / per:.3f} s per {unit}  (kernels busy {busy / per:.3f} s = "
          f"{100 * busy / max(wall, 1e-12):.0f} % of wall time; the rest is Python/launch overhead and torch ops)")
    for k, (ms, n) in sorted(ker.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"    {ms / per:9.2f} ms/{unit}  {n / per:7.1f} launches/{unit}  {k}")


def profile_runner(runner, n=20, loads=None, warm=30):
    """runner: Solver or PatchGrid (already built). loads: callable doing one load integration."""
    patched = hasattr(runner, "grids")
    base = runner.base if patched else runner
    grids = runner.grids if patched else [runner]
    dev = base.device
    ncell = sum(int(np.prod(g.n)) for g in grids)
    print(f"\n=== profile: {len(grids)} grid(s), {ncell / 1e6:.2f} M cells, device {dev}"
          f"{' (' + torch.cuda.get_device_name(dev) + ')' if dev.type == 'cuda' else ''} ===")
    if patched:
        runner.verbose = False

    def explicit(k, first=0):
        if patched:
            runner.run_explicit(k, chunk=k, first_order_steps=first)
        else:
            runner.run(k, chunk=k, first_order_steps=first, steady=True)

    def implicit(k, first=0):
        kw = dict(chunk=k, first_order_steps=first, cfl_start=1.0, cfl_max=1.0, cfl_ramp=1)
        if patched:
            runner.run(k, explicit_finish=0, **kw)
        else:
            runner.run(k, steady=True, implicit=True, explicit_finish=0, **kw)

    # warm-up: compiles both orders and both solvers, moves the flow off the impulsive start
    t0 = time.perf_counter()
    explicit(warm, first=warm)
    explicit(3)
    implicit(3)
    _sync(dev)
    print(f"warm-up ({warm + 6} steps, includes kernel compilation): {time.perf_counter() - t0:.1f} s")

    out = {}
    for name, fn, unit in (("explicit SSP-RK2 local-time step", explicit, "step"),
                           ("implicit iteration (red-black DP-LUR, 8 sweeps)", implicit, "iter")):
        _sync(dev)
        t0 = time.perf_counter()
        fn(n)
        _sync(dev)
        wall = time.perf_counter() - t0
        ker, wall_k = _kernel_breakdown(dev, lambda: fn(max(2, n // 4)))
        _print_breakdown(name, ker, wall_k, max(2, n // 4), unit)
        out[unit] = wall / n
        print(f"    timed run of {n}: {wall / n:.3f} s per {unit} = {ncell / (wall / n) / 1e6:.1f} Mcell-{unit}s/s")
    if loads is not None:
        t0 = time.perf_counter()
        loads()
        _sync(dev)
        t1 = time.perf_counter()
        loads()
        _sync(dev)
        t2 = time.perf_counter()
        out["loads"] = t2 - t1
        print(f"\nload integration: first call {t1 - t0:.2f} s (builds the triangulated surface), then {t2 - t1:.2f} s")
    print(f"\nsummary: explicit {out['step']:.3f} s/step, implicit {out['iter']:.3f} s/iter"
          + (f", loads {out['loads']:.2f} s per check" if "loads" in out else ""))
    return out
