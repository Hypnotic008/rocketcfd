# rocketcfd

A compressible-flow solver for launch-vehicle aerodynamics, built first for
Super Heavy's unpowered, engines-first descent. Its first job is to replace the
guessed aero coefficients in CatchSim (`quatsim/aero.py`).

## Status: milestone 1 (inviscid aero loads)

| piece | state |
|---|---|
| 3-D compressible Euler, finite volume, HLLC + HLLE shock fix, MUSCL, SSP-RK2 | done |
| Cartesian grid + ghost-cell immersed boundary (geometry from signed distance functions; no meshing) | done |
| CPU and NVIDIA GPU from the same code (PyTorch; native Windows and Linux) | done, verified on an RTX 4050 Laptop GPU |
| Fused NVIDIA Warp kernels (one pass per cell per stage); PyTorch path kept as the reference (`tests/compare_backends.py`) | done, matches the reference to 5e-5 |
| Stretched grids with refinement zones and growth-rate limit; symmetric about the finest zone | done |
| U.S. Standard Atmosphere 1976 | done |
| Super Heavy V3 geometry: hull, tapered thrust plate, 33 exposed Raptor 3 bells (outer ring overhanging the hull), 4 chines (taller pair on the raceway side), raceway, fin pods, forward dome | done |
| Grid fins from CatchSim's STEP-derived mesh (`rocketcfd/data/gridfin.i16`), scaled per axis to 3.5 m wide x 4.2 m reach x 0.5 m deep | done |
| Crossflow from any side (`--windward`): lift, side force, rolling moment for the gliding entry | done |
| Thin walls (blocked faces) for parts thinner than a cell, e.g. grid fin lattice plates; separate thin-wall loads | done |
| Pressure loads -> CD, CN, Cm in CatchSim's reference frame | done |
| One-case-per-GPU sweep that writes an aero database CSV | done |
| Output: slice plots, ParaView VTK, STL of the geometry | done |

### Validation

| case | reference | result |
|---|---|---|
| Sod shock tube | exact Riemann solution | mean density error 0.16 % (400 cells) |
| Cylinder, Mach 4: stagnation pressure | Rayleigh pitot formula | -1.0 % |
| Cylinder, Mach 4: bow-shock standoff | Billig correlation | +6.9 % (40 cells / radius) |
| Cylinder, Mach 4: surface pressure | modified Newtonian | matches near stagnation, above it further round (expected) |
| Cylinder, Mach 4, stretched grid (2.8x fewer cells) | uniform-grid result | standoff within 1 %, stagnation pressure -0.5 % |
| 1 mm flat plate, Mach 2, 5 deg, plate along the grid (thin walls) | shock-expansion theory | face pressures exact to 4 digits; CN -1.7 % |
| Same plate tilted across the grid (staircase walls) | shock-expansion theory | CN +6.1 % (100 cells/chord), -2.5 % (200 cells/chord) |
| Warp kernels vs PyTorch reference (Sod, cylinder, 3-D sphere) | identical numerics | max difference 2e-4 (float32) |

## Running

Windows or Linux, NVIDIA GPU (no WSL needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Then:

```bash
python cases/validate_sod.py
python cases/validate_cylinder.py 4
python cases/validate_flat_plate.py            # thin walls (add --inclined for staircase walls)
python cases/super_heavy_geometry.py                         # STL + renders
python cases/super_heavy_descent.py --h 0.1 --fins cad_filled
python cases/aero_sweep.py --gpus 1 --h 0.3                  # descent x AoA x windward side
python cases/benchmark.py                                    # speed: PyTorch vs Warp kernels
python tests/compare_backends.py                             # Warp kernels vs PyTorch reference
```

### Sizing for the GPU

Measured on an RTX 4050 Laptop GPU: about 0.46 KB per cell with the PyTorch path, so 6 GB holds ~11 M cells.
On a 6 GB laptop RTX 4050, that's about 7 M cells, or h ~ 0.3 m for the full Super Heavy domain
(106 x 44 x 44 m). Halving h needs 8x the memory and ~16x the time. Local refinement
(roadmap 2) and a lower-memory flux loop are how finer grids will fit.


## Limits of this milestone (read before using numbers)

* **Inviscid.** No boundary layers or skin friction. Pressure loads on the forebody, fins
  and engine end are what Euler does well. The wake and base region behind the
  vehicle (the top, in descent) separate in reality, so base pressure here is unreliable.
* **Grid fin lattice.** The scaled lattice plates are a few cm thick and need ~2 cm cells to resolve. Until local
  refinement exists, `cad_filled` treats each fin as a solid plate of the same outline.
  That overstates fin drag, especially supersonically, where real grid fins pass most of the flow.
  The transonic choking that CatchSim's notes call for needs the resolved lattice.
* **Uniform grid, one GPU per case.** The sweep runs cases one after another on a single GPU,
  or one per GPU on a multi-GPU machine.
* **Geometry is estimated.** Dimensions follow the photo-measured V3 notes in
  [spacecraft-models/starship `dimensions.md`](https://github.com/spacecraft-models/starship/blob/main/dimensions.md),
  which tag each number as published, reported or estimated. Chine width is a guess.
  Truss struts, conduit band and valve housings are not modelled.

## Roadmap

1. GPU run on the target machine; grid-convergence study (h = 0.6, 0.45, 0.3 m);
   cut memory per cell (~650 B now) so finer grids fit in 6 GB.
2. Local refinement around the fins -> resolved lattice, fin-alone runs for
   Cn(Mach, alpha, deflection) and hinge moments, including transonic choking.
3. Navier-Stokes + wall model (skin friction, base flow); transonic descent.
4. Engine plume inflow boundary (CEA exit conditions), multi-species mixing:
   landing-burn plume interaction, then hot staging.
5. High-temperature air (two-temperature, 5/11 species) for Starship entry.
6. Multiphase (Lagrangian droplets) for the deluge / liftoff plume.
