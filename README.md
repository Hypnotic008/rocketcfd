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
| Same plate tilted across the grid (staircase walls) | shock-expansion theory | CN +10.8 % (100 cells/chord), +7.5 % (200 cells/chord): first-order convergence |
| Warp kernels vs PyTorch reference (Sod, cylinder, 3-D sphere) | identical numerics | max difference 2e-4 (float32) |
| Viscous shear wave, periodic box (`tests/viscous_shearwave.py`) | exact decay exp(-nu k^2 t) | effective / set viscosity 0.999 (64 cells), 0.987 (16) |
| Laminar flat plate, Mach 0.4, Re 1e5 (`cases/validate_blasius.py`) | Blasius | skin friction -0.5 % (0.2 < x < 0.8 m), velocity profile within 0.7 % of U |
| Turbulent flat plate, SST + wall function, Re 1e7 (`cases/validate_turbulent_plate.py`) | White / Schlichting Cf, log law | Cf -6.3 % from White (on Schlichting); first-cell y+ 143; profile on the log law |
| Same at Re 1e9 (Super Heavy scale, `--re 1e9`) | White Cf, log law | Cf -3.1 % from White with the first cell at y+ 10,700; profile on the log law |

## Results so far

### Super Heavy V3 body without fins: grid-convergence study

Mach 2.82 at 10.7 km (CatchSim's peak dynamic pressure on descent), 5 deg off engines-first,
crossflow onto the raceway side. Run on an RTX 4050 Laptop GPU; the finest grid (20.8 M cells,
0.125 m near the body) took 16 minutes. Coefficients use CatchSim's references
(S = 63.6 m^2, L = 72.3 m, moments about the centre of mass at 0.40 L from the engines).

| near-body cell size | 0.5 m | 0.35 m | 0.25 m | 0.18 m | 0.125 m |
|---|---|---|---|---|---|
| cells | 0.48 M | 1.22 M | 3.01 M | 7.46 M | 20.8 M |
| CD | 2.218 | 2.158 | 2.265 | 2.064 | 2.108 |
| CN | 0.207 | 0.188 | 0.208 | 0.216 | 0.219 |
| restoring moment | -0.050 | -0.041 | -0.050 | -0.056 | -0.055 |

Breakdown on the finest grid (averaged over the end of the run; the flow is steady to 0.1 %):

| region | CD | CN | restoring moment |
|---|---|---|---|
| engine end (33 bells, thrust plate) | 1.929 | 0.057 | -0.021 |
| hull and chines | 0.034 | 0.162 | -0.034 |
| top (dome, wake) | 0.146 | 0.000 | +0.000 |

Best estimates and how they compare with CatchSim (`quatsim/aero.py`):

| | CFD | CatchSim |
|---|---|---|
| drag CD (body) | 2.11 +/- 0.10 | 1.285 (body guess); 2.57 total after the flight-calibrated x2 fin factor |
| normal-force slope (per rad) | 2.5 +/- 0.1 | 2.0 |
| restoring-moment slope (per rad) | -0.63 +/- 0.03: the bare body is unstable engines-first | +0.35 (whole vehicle) |

* Normal force and restoring moment converge (finest two grids within 1-2 %).
* Drag scatters by about +/- 5 % because almost all of it acts on the engine end, where the
  outer nozzles are only ~4 cm apart: each grid opens or closes those gaps differently. Resolving
  them needs a ~6 cm engine-end zone.
* The body supplies ~2.1 of CatchSim's flight-calibrated 2.57; fins and viscous drag share the rest.
* For CatchSim's +0.35 stability, the grid fins must contribute about +1.0: the fin sweep
  (`cases/grid_fin.py`) is the next result.

### V3 grid fin, isolated: Mach sweep

Parametric V3 fin (`rocketcfd/vehicles/grid_fin.py`: 3.5 m x 3.9 m, 0.5 m deep, 10 mm plates
0.31 m apart, teeth on the face toward the engines), 2.5 cm cells (5.7 M cells), inviscid,
Mach swept downward with warm starts like the real descent. Slope of the normal force per
radian, on the 11.26 m^2 outline area:

| Mach | 0.6 | 0.8 | 0.9 | 1.0 | 1.1 | 1.3 | 1.6 | 2.0 | 2.8 | 4.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| dCN/dalpha | 1.93 | 1.49 | 1.55 | 1.56 | 1.62 | 1.45 | 1.40 | 2.01 | ~1.9* | 1.40 |
| relative to Mach 0.6 | 1.00 | 0.77 | 0.80 | 0.81 | 0.84 | 0.75 | 0.73 | 1.04 | ~0.99 | 0.73 |

\* From 5 and 10 deg only: the 0 deg case at Mach 2.8 kept an asymmetric, partly choked lattice
from the previous (10 deg) case. The sweep now starts every incidence from the alpha-0 state.

* At Mach 4 the lattice is **started**: oblique shocks at each plate edge, supersonic flow through
  the cells, no shock ahead of the fin. Axial force ~0.18 (the old 4.5 cm square-edged CAD fin
  stayed choked at 2.8 with ~0.45).
* Between Mach ~1.6 and ~0.8 the lattice is **choked**: a detached shock stands ahead of the fin
  and the effectiveness drops ~25 %. The fin restarts near Mach 2.
* Compared with CatchSim's `effectiveness()` (dip to 0.55 at Mach 1.1, 0.9 above Mach 2): the dip
  is shallower and wider, and the supersonic level varies (1.04 at Mach 2, 0.73 at Mach 4).
* Per fin at Mach 2.8: ~21 x q N per radian (CatchSim: ~4.3 x q). With the body's -0.63, the two catch
  fins (+0.34) leave the restoring-moment slope at about -0.29 without body upwash on the fins:
  roughly neutral to mildly unstable engines-first at peak dynamic pressure, against CatchSim's
  +0.35. An installed (fins-on) booster run is needed to settle it.

### Whole booster with installed grid fins: gliding attitude

V3 with the three parametric fins (thin walls), Mach 2.82 at 10.7 km, 15 deg off
engines-first, crossflow arriving on the rudder side. 34.2 M cells (0.25 m along the body,
6 cm through the fins), 65 min on an RTX 4050 Laptop GPU. Averaged over the last 10 load checks.

| region | CD | CN | restoring moment |
|---|---|---|---|
| engine end | 2.865 | 0.176 | -0.066 |
| hull + chines | 0.336 | 1.231 | -0.034 |
| fin band (fins, pods, hull beside them) | 0.177 | 0.206 | +0.109 |
| top | 0.154 | 0.003 | +0.002 |
| **total** | **3.53** | **1.62** | **+0.011 +/- 0.002** |

Per fin (restoring moment): catch fins +0.0245 each, rudder fin +0.010.

* With the fins installed the booster is **barely stable** engines-first at this attitude: the
  engine end and hull (-0.100) are just outweighed by the fin section (+0.109). CatchSim assumes
  +0.35 (as a slope: CFD +0.04 per sin(alpha)).
* The catch fins carry most of the pitch stability; the rudder fin sits in the windward flow.
* The tower side is a low-energy, vortical leeward region from x ~ 12 m to the top: a fin there
  would see much less dynamic pressure. The taller (tower-side) chines shed vortices there.
* Glide lift-to-drag ~0.2 (CL ~0.73 from CN and CA); the hull carries 3/4 of the normal force.

Same condition with the **tower side** into the wind (`--windward 180`):

| | rudder side windward | tower side windward |
|---|---|---|
| CN (lift) | 1.62 | 1.50 |
| CD | 3.53 | 3.52 |
| restoring moment | **+0.011 (stable)** | **-0.059 (unstable)** |
| rudder fin | +0.010 | -0.002 (in the leeward wake) |
| each catch fin | +0.0245 | +0.0248 |
| fin band | +0.109 | +0.054 |
| engine end | -0.066 | -0.088 |

* Stability depends on **roll attitude**: stable with the rudder side windward, unstable with
  the tower side windward. CatchSim's axisymmetric aero model cannot represent this.
* The rudder fin only works on the windward side; in the leeward wake it does nothing.
* The catch fins (at +/-y, symmetric to the crossflow either way) give the same moment in both
  runs, a consistency check on the comparison.
* With the rudder side windward the hull beside the fins adds ~+0.05 (rudder fin / pod raising
  the pressure on the hull ahead of them); with the tower side windward it does not.

Rudder side windward **without chines** (`--no-chines`):

| | with chines | no chines | chines' effect |
|---|---|---|---|
| CN (lift) | 1.616 | 1.482 | +0.134 (+9 %) |
| CD | 3.532 | 3.483 | +0.05 |
| restoring moment | +0.011 | +0.085 | **-0.074** |
| hull restoring moment | -0.034 | +0.035 | -0.069 |

* The chines are lift devices (+9 % normal force) that sit mostly between the engines and the
  centre of mass, i.e. on the leading half in an engines-first descent, so they cost almost all
  of the static margin. The fins and the engine end are unchanged (consistency check).
* Open item: the hot-staging truss is not modelled. In supersonic descent it cannot affect
  anything upstream of it; its own crossflow load at 15 deg is estimated at a restoring moment of
  roughly +0.01 (comparable to the margin with chines) and needs viscous flow to be meaningful.

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


## Viewing results in 3-D

```bash
py -m pip install pyvista imageio-ffmpeg
py -m rocketcfd.viewer results\glide_rudder                  # interactive window
py -m rocketcfd.viewer results\glide_rudder --movie orbit    # MP4: orbit with flowing particles
py -m rocketcfd.viewer results\glide_rudder --movie sweep    # MP4: slice sweeping engines -> top
py -m rocketcfd.viewer results\glide_rudder --stride 2       # quarter of the memory, faster to load
```

The real geometry is always shown: the hull coloured by surface pressure and the toothed grid
fins as solid parts. Streamlines are coloured by Mach with particles moving at the local flow
speed; a slider (or `k`) moves a Mach slice along the booster. Keys: space pause, `o` orbit,
`k` sweep, `s` streamlines, `f` fins, `h` see-through hull, `r` reset view.

Animations of the flow developing: run a case with `--snapshots 150` (add `--time-accurate` for
a physically time-accurate movie), then

```bash
py -m rocketcfd.viewer results\glide_rudder --movie evolve
```

plays the saved frames: surface Cp on the hull and Mach on the rudder/tower and catch-fin planes.

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
