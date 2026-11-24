"""Speed and memory benchmark: PyTorch reference vs fused Warp kernels.

  py cases\\benchmark.py            (Windows)
  python cases/benchmark.py        (Linux)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from rocketcfd import Solver


def bench(backend, n, steps=60):
    h = 2.0 / n
    s = Solver((-1, -1, -1), (1, 1, 1), h, dict(rho=1.4, u=3.0, v=0.0, w=0.3, p=1.0),
               sdf=lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 0.4, verbose=False, backend=backend)
    s.run(10, chunk=10)                      # warm-up: kernel compilation, allocator
    if s.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    s.run(steps, chunk=steps)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    cells = np.prod(s.n)
    mem = torch.cuda.max_memory_allocated() / cells if s.device.type == "cuda" else float("nan")
    return cells * steps / dt / 1e6, mem


if __name__ == "__main__":
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    sizes = (64, 128, 160) if torch.cuda.is_available() else (48, 64)
    for n in sizes:
        row = [f"{n}^3 = {n**3/1e6:5.2f} M cells:"]
        for be in ("torch", "warp"):
            try:
                rate, mem = bench(be, n)
                row.append(f"{be} {rate:7.1f} Mcell-steps/s ({mem:.0f} B/cell)")
            except Exception as e:  # noqa: BLE001
                row.append(f"{be} failed: {type(e).__name__}: {e}")
        print("   ".join(row), flush=True)
