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
| Laminar Navier-Stokes; k-omega SST with a Spalding wall function; skin friction included in the loads | done |
| Implicit steady solver: backward Euler in pseudo time, matrix-free Rusanov Jacobian, 8 Jacobi sweeps (DP-LUR), local pseudo time steps; k-omega advanced the same way (`--implicit`) | done |
| Dual time stepping (BDF2 outer steps, implicit inner iterations) for unsteady flows: the time step follows the physics, not the smallest cell | done |
| Local refinement: patches of finer cells (anisotropic allowed) coupled to a coarse base grid; generator for shells around walls and boxes around the grid fins (`--wall-h`, `--fin-h`) | done |
| Large-eddy simulation: implicit LES (default) or Vreman subgrid model, equilibrium wall model or wall-resolved (`--les`) | done, Taylor-Green vortex checked |
| Desktop app: set up, queue and watch runs, 3-D results, plots, animations (`python -m rocketcfd.app`) | done |
| Cinematic renders and movies (vortex surfaces coloured by Mach, schlieren, LAVA-style) (`python -m rocketcfd.cinematic`) | done |

### What changed in the latest round (Nov 25, evening)

* **Desktop app** (`python -m rocketcfd.app`, or double-click `rocketcfd_app.bat`): see
  [Desktop app](#desktop-app) below.
* **GPU speed**: the residual is now computed as face-flux kernels (one direction each, compiled
  with the direction fixed) plus a divergence kernel. Compiled for the RTX 4050 (sm_89) the old
  fused kernel needed 174 registers per thread and a 232-byte spill frame (one 256-thread block
  per SM, ~17 % occupancy); the new inviscid kernels need 52-63 registers and no spills (viscous
  80-91). Each face flux is also computed once instead of twice. CPU: 1.9x faster inviscid
  residual. The implicit sweeps read cached spectral radii (96 -> 70 registers, identical results). Results identical to the fused kernel (inviscid bit for bit, viscous to 2.5e-5;
  `tests/split_residual.py`). `--profile 20` measures it on your GPU.
* **Patch layout**: candidate patches are tiled on base-grid cells, so snapping no longer pushes
  neighbouring patches into each other, and under anisotropic cells the depth kept inside a wall
  follows the wall normal. Inviscid production setup 6.11 M -> 5.50 M cells; the viscous setup with a hull
  shell (0.25 x 0.03 x 0.03 m) 39 M -> 22 M cells (now fits 6 GB). Mach 3 sphere with patches:
  +0.43 % from an all-fine grid (+0.81 % with the old layout) at the same patch cell count.
* **LES** (`--les`, `turbulence=dict(model="les")`): time-accurate, implicit LES by default (the
  upwind scheme's dissipation is the subgrid model), Vreman model optional, equilibrium wall
  model. Patch runs advance all grids with one global time step (`run_explicit(time_accurate=True)`).
  Taylor-Green vortex, Re 1600, 32^3: dissipation peak 0.0117 at t 7.0 (DNS 0.0127 at t 9; a
  coarse grid peaks early); Vreman over-dissipates here (peak at t 5.5).
* **Visuals**: `rocketcfd.derived` (schlieren, Q, lambda-2, vorticity, T, Cp, total-pressure ratio),
  `rocketcfd.cinematic` (stills, orbits, time evolutions in the NASA LAVA style), 3-D snapshots
  now store the full state (rho, u, v, w, p) for movies.

### Earlier round (solver work, Nov 25)

* **Implicit steady solver** (backward Euler in pseudo time, matrix-free Jacobian, red-black
  Gauss-Seidel relaxation): same answers as the explicit solver in far fewer iterations
  (sphere 2.5-4x less wall time on CPU; most gain on thin near-wall cells). Turbulent runs finish
  with explicit steps (see limits).
* **Dual time stepping** for unsteady flow (vortex shedding): the time step follows the physics.
* **Local refinement patches**: fine cells only where needed (engine end, grid fins, boundary-
  layer shells, shocks), snapped to the base grid; forebody loads within 0.03 % of an all-fine grid.
  The inviscid booster setup with 6 cm engine end and fins is 6.1 M cells instead of 34 M.
* **Numerics**: Koren (kappa = 1/3) reconstruction (third order where smooth), low-Mach correction
  for subsonic freestreams, plate normals on thin walls, true-normal wall function, loads on a
  triangulated surface (exact for any cell shape).
* **Robustness / memory**: thin-wall setup 13 GB -> 0.3 GB for a grid fin; NaN in viscous patch runs
  fixed; launches cached; divergence retry in the database sweep.
* **CatchSim**: `cases/aero_database.py` (Mach x aoa x roll sweep with warm starts) ->
  `aero_db.json` -> `integrations/catchsim/aero_cfd.py` (force/moment tables with the mirror rule,
  quasi-steady pitch damping, fin control derivatives and hinge moments).
* **Checks**: `tests/run_checks.py` runs the verification cases with PASS / FAIL.

### Next runs on the GPU (in this order)

```bash
git pull
pip install -r requirements.txt                              # adds PySide6 + pyvistaqt for the app
python tests/run_checks.py                                   # ~minutes: confirms this machine matches
# 0. how much faster is the split residual on this GPU? (compare with 0.73 s/step, 0.74 s/iter before)
python cases/super_heavy_descent.py --h 0.3 --fins parametric --fin-h 0.06 --engine-h 0.06 --aoa 15 --windward 0 --profile 20
# LES movie of the booster wake (no patches: 1.8 M cells; ~1 flow-through per few minutes)
python cases/super_heavy_descent.py --h 0.3 --fins parametric --aoa 15 --windward 0 --les --flowthroughs 3 --snapshots 200 --snap-volume 1 --out results/les_wake   # ~1.2 GB of frames
python -m rocketcfd.cinematic results/les_wake --mode evolve --style lava
# viscous RANS with a boundary-layer shell (22 M cells, ~4.1 GB)
python cases/super_heavy_descent.py --h 0.3 --fins parametric --fin-h 0.06 --engine-h 0.06 --turbulent --wall-h 0.25,0.03,0.03 --wall-shell 0.3 --patch-box 32,48,48 --aoa 15 --windward 0 --out results/glide_rans
# 1. inviscid glide case with local refinement, to compare with the slab-refined result
#    (CD 3.532, CN 1.616, C_restore +0.011 at 34 M cells)
python cases/super_heavy_descent.py --h 0.3 --fins parametric --fin-h 0.06 --engine-h 0.06 --aoa 15 --windward 0 --out results/glide_patch
# 2. the same, finer (grid convergence of the patch setup)
python cases/super_heavy_descent.py --h 0.25 --fins parametric --fin-h 0.04 --engine-h 0.04 --aoa 15 --windward 0 --out results/glide_patch_fine
# 3. a quick aero database, then the full one (overnight) -> aero_db.json for CatchSim
python cases/aero_database.py --mach 2.8 2.0 1.3 0.9 --aoa 0 8 16 --roll 0 180 --out results/db_quick
python cases/aero_database.py --out results/aero_db
# 4. isolated grid fin Mach sweep with the current numerics (the fin table below predates them);
#    prints the supersonic cascade-theory slope next to each result
python cases/grid_fin.py --implicit --h 0.025 --out results/grid_fin_v2
# 5. viscous: does SST separate from a smooth cylinder once the boundary layer is resolved?
python cases/validate_cylinder_turbulent.py --dual --patch-h 0.001 --shell 0.06
```

### Validation

| case | reference | result |
|---|---|---|
| Sod shock tube | exact Riemann solution | mean density error 0.16 % (400 cells) |
| Cylinder, Mach 4: stagnation pressure | Rayleigh pitot formula | -1.0 % |
| Cylinder, Mach 4: bow-shock standoff | Billig correlation | +6.9 % (40 cells / radius) |
| Cylinder, Mach 4: surface pressure | modified Newtonian | matches near stagnation, above it further round (expected) |
| Cylinder, Mach 4, stretched grid (2.8x fewer cells) | uniform-grid result | standoff within 1 %, stagnation pressure -0.5 % |
| 1 mm flat plate, Mach 2, 5 deg, plate along the grid (thin walls) | shock-expansion theory | face pressures exact to 4 digits; CN -1.7 % |
| Same plate tilted across the grid (staircase of blocked faces) | shock-expansion theory | converged, CN averaged over the end of the run. Plate normal in the wall flux (default; used where within ~70 deg of the face axis): -3.6 % / -5.4 % at 100 / 200 cells per chord; axis-mirror staircase (old): +7.7 % / -2.6 %. The normals help on coarse grids only; both end a few per cent low |
| NACA 0012, Mach 0.8, 1.25 deg (`tests/naca0012.py`) | Euler references (AGARD AR-211) CL 0.35-0.36, CD 0.022-0.023 | shock at x/c 0.63 (ref ~0.62); CL 0.288, CD 0.018 at 250 cells/chord: thin sharp trailing edges lose lift with ghost-cell walls (both surfaces share the ghost cells there) |
| Warp kernels vs PyTorch reference (Sod, cylinder, 3-D sphere) | identical numerics | max difference 2e-4 (float32) |
| Viscous shear wave, periodic box (`tests/viscous_shearwave.py`) | exact decay exp(-nu k^2 t) | effective / set viscosity 0.999 (64 cells), 0.987 (16) |
| Laminar flat plate, Mach 0.4, Re 1e5 (`cases/validate_blasius.py`) | Blasius | skin friction -0.5 % (0.2 < x < 0.8 m), velocity profile within 0.7 % of U |
| Turbulent flat plate, SST + wall function, Re 1e7 (`cases/validate_turbulent_plate.py`) | White / Schlichting Cf, log law | Cf -6.3 % from White (on Schlichting); first-cell y+ 143; profile on the log law |
| Same at Re 1e9 (Super Heavy scale, `--re 1e9`) | White Cf, log law | Cf -3.1 % from White with the first cell at y+ 10,700; profile on the log law |
| Isentropic vortex, one period (`tests/vortex_order.py`) | exact (initial field) | Koren (default): order 2.8, L1 1.2e-4 at 128^2; van Leer: order 2.3, 3.1e-4 |
| Sod / Mach 4 cylinder with Koren (`tests/limiter_shocks.py`) | exact / Rayleigh pitot | Sod L1 0.00144 (van Leer 0.00172); stagnation pressure +0.3 % (van Leer -1.0 %) |
| Inviscid cylinder, Mach 0.1 (`tests/low_mach_cylinder.py`) | potential flow: CD 0, rear Cp 1 | low-Mach correction (default for subsonic freestreams): CD +0.010, rear Cp 1.07; without: CD -0.037, rear Cp 0.69. (A run with the briefly leaky ghost cells had suggested otherwise; it was the walls.) |
| Implicit vs explicit, Mach 3 sphere (`tests/steady_accel.py`) | explicit local time step | CD 0.9625 vs 0.9627, 480 vs 2000 steps, 2.5x less wall time (CPU) |
| Implicit vs explicit, laminar plate (2x coarse grid) | explicit | same Cf (-7.15 %); needs 8 Jacobi sweeps (2 diverge, 4 stall) |
| Red-black Gauss-Seidel vs Jacobi relaxation | same steady state | Mach 3 sphere CD 0.9591 vs 0.9596 in 89 s vs 140 s; plate: same Cf after the same iterations (default now) |
| Super Heavy bare body, 0.5 m grid, M 2.82, 5 deg (CPU) | earlier explicit van Leer run | implicit + Koren: CD 2.208, CN 0.205, C_restore -0.051 vs 2.218, 0.207, -0.050; with the ghost-cell fix: 2.202, 0.199, -0.0515 |
| Isolated V3 grid fin, Mach 4, 4 cm cells (CPU, implicit) | supersonic cascade theory 1.83 /rad (upper bound) | slope 1.11 /rad with the current numerics, 1.09 with the old (van Leer, axis-mirror walls); 2.5 cm cells gave 1.40 earlier: the fin is resolution-limited, not numerics-limited, at these cell sizes |
| Ghost-cell fix, Mach 4 cylinder forebody / Mach 3 sphere | before the fix | forebody drag 1.2361 vs 1.2355 (unchanged); sphere CD 0.981 vs 0.959, the difference is in the base / wake region |
| Solution-adaptive patches, Mach 4 cylinder (`tests/adapt_cylinder.py`) | fine grid, Billig | bow-shock standoff +7.0 % (fine grid +7.4 %, coarse +13.9 %) |
| Local refinement with SST turbulence, flat plate Re 1e7 (`tests/patch_turbulent_plate.py`) | one fine grid | friction drag 0.00581 vs 0.00581 (fully turbulent Prandtl-Schlichting 0.00601) |
| Mach 8 / 12 cylinders (`tests/carbuncle.py`): carbuncle check | Billig, Rayleigh pitot | M 8: standoff +1.2 %, stagnation pressure -0.7 %, symmetric shock; M 12: +5.3 %, -6.1 %, no carbuncle (the shock switch to HLLE works; very high Mach loses some stagnation pressure) |
| Wall patches, Mach 4 cylinder forebody drag (4x finer cells within 0.1 / 0.4 of the wall) | one fine grid (1.2355) | 1.2231 (-1.0 %) / 1.2467 (+0.9 %); with patch edges snapped to base-grid faces (default): 1.2351 (-0.03 %); coarse alone 1.1526 (-6.7 %). The inviscid wake behind the body stays grid-dependent |
| Local refinement, laminar plate (`tests/patch_blasius.py`) | one fine grid | friction drag 0.00899 vs 0.00896 (coarse grid alone 0.00967), 68 % of the cells |
| Local refinement, Mach 3 sphere (`tests/patch_sphere.py`) | one fine grid | CD 0.9666 vs 0.9569 (coarse alone 1.0114), 45 % of the cells |
| Laminar cylinder Re 100, dual time stepping | St 0.164, Cd 1.33 | St 0.160 (25 cells/D, 50 steps per period, 15 inner iterations); with 40 steps and 12 inner iterations St 0.137 and Cd 1.336, the same with a refinement patch: use the defaults (80 / 20) |
| Laminar cylinder Re 40, steady (immersed no-slip wall) | Dennis & Chang: Cd 1.52 | Cd 1.486 / 1.561 at D/25 / D/50 (tangential velocity extrapolated through the wall, triangulated loads; before: 1.464 / 1.372, moving away) |
| Flat-nosed body, Mach 3, front face exactly on a cell face / between (`flat-face` test) | the same body, other grids | CD 1.61 / 1.62 (loads: triangulated surface); an intermediate ghost-cell change that scaled the normal velocity let the flow through the face (CD 0.03) and was reverted |
| Turbulent plate Re 1e7, implicit (`--implicit`) | White Cf | +2.6 % (explicit, converged: -6.2 %): two different converged SST states, see limits |
| CatchSim drop-in vs the sweep it reads (`tests/catchsim_dropin_check.py`) | the computed cases | identical to 1e-15; mirror rule and centre-of-mass transfer exact |

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

> **Drag values in this section are wrong (found Nov 25).** These runs refined the fins with
> whole slabs of fine cells, which left the engine end with long, flat cells (0.25 m along the
> axis, 6 cm across). The old smoothed-band force integration mis-weights a flat wall facing the
> long side of such cells: the engine-end axial force came out ~1.5x too large (CD_aft 2.87 is above
> what stagnation pressure on the whole frontal area allows, ~2.2). Loads now use a triangulated
> surface (`Solver.loads(method="tri")`), which does not depend on cell shape; on a flat-faced test
> body the band gave -90 % .. +150 % depending on where the face fell between cell centres, the
> triangulated surface 1.45-1.50 every time. CN and C_restore come mostly from the hull sides, where
> those cells were fine, but treat them as unverified until these cases are rerun with patches.

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

### Implicit solver and local refinement

Steady runs converge in far fewer iterations with the implicit solver, most of all when
the grid has thin cells next to walls:

```bash
python cases/validate_blasius.py --implicit
python cases/validate_turbulent_plate.py --implicit
python cases/super_heavy_descent.py --h 0.25 --h-fins 0.07 --fins parametric --aoa 15 --windward 0 --implicit
```

Local refinement puts fine cells only where they are needed instead of refining whole
slabs of the grid. Patches can be anisotropic: around the hull, cells only 3 cm across the
wall but 25 cm along the booster resolve the boundary layer at a fraction of the cost.
Patches imply the implicit solver:

```bash
# inviscid production setting: 0.3 m body grid, 6 cm patches on the engine end and the fins
# (6.1 M cells, ~0.8 GB): the engine-end patches resolve the ~4 cm gaps between the outer
# nozzles that made drag scatter +/- 5 % between grids
python cases/super_heavy_descent.py --h 0.3 --fins parametric --fin-h 0.06 --engine-h 0.06 --aoa 15 --windward 0
# grid fins resolved by 3 cm patches around each fin (no fin slab), coarse base grid
python cases/super_heavy_descent.py --h 0.3 --fins parametric --fin-h 0.03 --aoa 15 --windward 0
# plus a boundary-layer shell around the hull, viscous
python cases/super_heavy_descent.py --h 0.4 --fins parametric --fin-h 0.03 --wall-h 0.25,0.03,0.03 \
    --wall-shell 0.4 --aoa 15 --windward 0 --turbulent
# add --dry-run to see the patch count, cells and memory first
```

Unsteady flows use dual time stepping (`Solver.run_unsteady`, `PatchGrid.run_unsteady`), e.g.
the cylinder with its boundary layer resolved by a 1 mm patch shell:

```bash
python cases/validate_cylinder_turbulent.py --dual --patch-h 0.001 --shell 0.06
python cases/validate_cylinder_turbulent.py --laminar --re 100 --mach 0.2 --h 0.02 --dual   # laminar shedding check
```

Numerics defaults (changed in this round, see the validation table): Koren (kappa = 1/3) MUSCL
reconstruction instead of van Leer; thin walls use the real plate normal; immersed-wall ghost cells
use the triangulated surface for loads. The Thornber low-Mach correction is on for subsonic
freestreams (`low_mach="auto"`). Older results in this README used van Leer and the axis-
mirror thin walls, so reruns will differ by about the size of their stated uncertainty.

### Aero database for CatchSim

`cases/aero_database.py` sweeps Mach x angle of attack x roll attitude in one process: the
grid is built once (roll turns the crossflow, not the vehicle), every case warm-starts from a
converged neighbour, the fins are resolved by local refinement patches and the implicit solver
runs to converged loads. Each case records the body-axis forces and moments, the breakdown by
region and every fin's forces and hinge moment, appended to `cases.csv` (a stopped sweep
resumes). At the end it writes `aero_db.json`.

```bash
python cases/aero_database.py --dry-run                     # grid, patches, memory, case list
python cases/aero_database.py                               # default: 10 Mach x 6 aoa x 3 roll (GPU, hours)
python cases/aero_database.py --mach 2.8 1.6 1.1 0.8 --aoa 0 8 16 --roll 0 180 --out results/db_quick
```

`integrations/catchsim/aero_cfd.py` is a drop-in for CatchSim (copy it into `quatsim/`):
`CFDAeroModel` replaces `AeroModel` (full CFD force at the actual angle of attack and roll,
grid fins included), `cfd_restoring_moment` replaces `restoring_moment` (full CFD moment about
the centre of mass, plus CatchSim's pitch damping). It reads `aero_db.json`; the V3 layout's
mirror symmetry fills roll angles 180-360. `tests/catchsim_dropin_check.py <out dir>` checks it
reproduces every computed case.

Pitch damping: the table also carries a quasi-steady `cm_q` per Mach (each region of the vehicle
sees its incidence change by q l / V when pitching: cm_q = 2 sum CN_a (l / L)^2, CatchSim's
normalisation), used by `cfd_restoring_moment` unless one is given. On the coarse test grid it
comes out ~0.5-0.7 against CatchSim's guessed 12; check it on the GPU database before relying on it
(lag terms are not included).

Fin control: run the sweep again with one fin deflected (`--fin-deflection 0 10 0` for a catch fin,
`10 0 0` for the rudder) and combine with `cases/fin_control_export.py`; `CFDFinControl` in the
drop-in gives control forces, moments and hinge moments per fin deflection (the other catch fin
follows from the mirror symmetry).

### Checking a machine

```bash
python tests/run_checks.py      # quick verification cases with PASS / FAIL
```

### Sizing for the GPU

Warp kernels: about 115 B per cell inviscid explicit, +25 B implicit, ~200 B with SST (`--dry-run`
prints the estimate), so a 6 GB card holds ~40 M inviscid cells. The inviscid booster setup with
6 cm patches is 5.5 M cells (~0.7 GB); with a 3 cm boundary-layer shell for RANS, 22 M (~4 GB).
`--profile 20` times explicit steps, implicit iterations and the load integration on your GPU
with a per-kernel breakdown.


## Desktop app

**Install on Windows (one click):** download or clone the repository, then double-click
`packaging\install_windows.bat`. It makes a private Python environment (`.venv`), installs CUDA
PyTorch and the requirements, checks that the GPU is found, and puts a **rocketcfd** shortcut on
the desktop. Run it again to update after a `git pull`.

Or in an existing Python environment:

```bash
pip install -r requirements.txt
python -m rocketcfd.app            # Windows: double-click rocketcfd_app.bat
```

| page | what it does |
|---|---|
| Setup | case templates (booster descent, aero database, grid fin, validation cases) as grouped forms with an explanation on every field; presets (quick look, glide with fins, viscous RANS, LES movie); grid-size and GPU-memory estimate; speed measurement; save / load setups; shows the equivalent command line |
| Run | queue of runs executed one after another; live residual and force plots; log; progress and worst-case time left |
| 3-D results | body coloured by Cp (or metallic), cutting plane in any direction with its own field and contour lines, velocity arrows, iso-surfaces (Q-criterion vortices, sonic surface, ...), streamline tubes, volume rendering (shocks as glowing sheets), backgrounds, camera presets, screenshots; **Cinematic look** button |
| Plots | convergence history; force breakdown by part with unsteadiness whiskers; filled contour plots of any field on any plane with in-plane streamlines; line probes (CSV export); aero database curves; compare runs (table + chart of the coefficients of any ticked runs); PNG / SVG / PDF |
| Animations | play back a run's snapshots; render movies (cinematic evolve / orbit / still, schlieren, streamline particles, slice sweep); built-in player |

Fields available everywhere: Mach, p, rho, T, Cp, numerical schlieren, Q-criterion, lambda-2,
vorticity, total-pressure ratio, speed (computed on demand from the saved flow).

## Cinematic renders

```bash
python -m rocketcfd.cinematic results/glide_patch --mode still                 # 1920x1080 PNG
python -m rocketcfd.cinematic results/glide_patch --mode orbit --style lava    # MP4 camera orbit
python -m rocketcfd.cinematic results/les_wake --mode evolve                   # flow evolving in time
```

Styles: `lava` (vortex surfaces coloured by Mach over a greyscale schlieren plane, black
background), `schlieren`, `mach`, `pressure`. `evolve` needs a run with `--snapshots N
--snap-volume 1` (or 2 for half the resolution and an eighth of the disk space).

## Engine plumes

`--engines centre|inner|all` fires 3, 13 or 33 engines (landing / boostback burns, supersonic
retropropulsion, plume pictures): the solid cells just inside each nozzle exit hold the exhaust
state (`Solver(jets=[...])`), so the plume is part of the flow. Single perfect gas: the exhaust is
"air" at `--exit-T` chosen so the exit velocity is right (Mach 3.5 at 2200 K -> ~3300 m/s, Raptor-like;
`--exit-p` 80 kPa). The forces include the exit-plane pressure but not the momentum thrust.
Check: `tests/underexpanded_jet.py` (Mach-disk position of a sonic jet vs the Crist-Sherman-Glass
correlation).

```bash
python cases/super_heavy_descent.py --mach 0.4 --alt 2000 --aoa 0 --h 0.3 --fins parametric --engines centre --implicit --out results/landing_burn
python -m rocketcfd.cinematic results/landing_burn --mode orbit --style mach
```

## Probes: time histories and spectra

`--probes` records rho, u, v, w, p every step at points in the wake, beside the hull, behind the
fins and at the engine end (`--probe-points "x,y,z; ..."` adds more) into `probes.npz`; the app's
Plots -> Probes shows the history and its spectrum (peak frequency, Strouhal number). Use with
`--les` for buffeting and wake-pressure fluctuations.

## Viewing results in 3-D

Runs with local refinement also write `flow_patches.vtm` (every patch grid with a `covered` mask);
the app and `rocketcfd.cinematic` draw each region from the finest grid there.

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

What a booster run writes (in its `--out` folder):

| file | contents |
|---|---|
| `summary.png` | load history, drag / normal force / restoring moment by region, per-fin contributions, key numbers vs CatchSim |
| `stations.png` | Cp and Mach on seven cross-sections from the engine bells to the top |
| `surface_cp.png` | surface pressure: windward / leeward views, engine-end and fin close-ups |
| `skin_friction.png`, `friction_lines.png` | (`--turbulent`) skin friction and friction lines; converging lines mark separation |
| `slice_xz.png`, `slice_xy.png` | Mach and pressure on the two planes through the axis |
| `loads.json` | every number, including the history |
| `surface_cp.vtk`, `flow.vtk` | for the viewer / ParaView (surface Cp, Cf and wall shear; 3-D field incl. eddy viscosity) |
| `snapshots/` | (`--snapshots N`) frames for `viewer --movie evolve` |

The viewer adds Q-criterion vortex surfaces (`v`) and friction lines (`l`).

## Limits of this milestone (read before using numbers)

* **Patch coupling is not conservative** (overlapping-grid style: interpolated ghost layers,
  averaged covered cells). Forebody loads match a single fine grid to ~1 % when the patch shell is
  at least a few fine cells thick; keep patch edges out of shocks where possible. Laminar skin
  friction on curved immersed walls: total drag now converges (Re 40 cylinder 1.585 / 1.527 vs 1.52),
  but the split between friction (0.56 / 0.44) and pressure (1.03 / 1.09) still moves with the grid.
* **Turbulent runs: the implicit k-omega path does not land on the steady state.** On the flat
  plate the implicit solver settles at Cf -1 % (Koren; +2.6 % with van Leer) from White, and the
  explicit solver continued from that state moves it to -6.3 %, where explicit runs from scratch
  also end. So the explicit result is the steady state and the implicit one is not. Diagnosed
  further (Nov 25): the implicit residual stalls about two orders down (0.3 from 14 at CFL 20, 0.17
  at CFL 5) and where it settles depends on the CFL number, so it is an iteration that does not
  converge, not a second steady state; the largest residuals sit in the wall-function cells and the
  near wake behind the trailing edge. Neither an implicit wall-shear diagonal (`wall_stiffness`) nor
  a wall-cell CFL cap (`cfl_wall`) removes the stall. `Solver.run(implicit=True)` therefore
  finishes turbulent runs with explicit local-time-step steps (`explicit_finish`, default 3x the
  implicit iterations, stopping early on the same convergence test). Inviscid and laminar implicit
  runs agree with explicit ones.
* **LES is for unsteady pictures and trends, not yet for numbers.** Implicit LES relies on the
  upwind scheme's dissipation; on the Taylor-Green vortex at 32^3 the dissipation peak comes early
  (t 7.0 vs 9.0). The wall model takes the first cell's velocity (no matching point further out:
  expect the log-layer mismatch described by Bose & Park 2018 in the wall shear). Inflow is
  laminar (no synthetic turbulence).
* **Plumes are one gas.** The exhaust is air with an adjusted temperature: the plume's shape and
  shocks are right to first order, its mixing and afterburning are not.
* **Experimental grid-fin data not yet used.** The wind-tunnel reports with isolated grid-fin normal
  force vs Mach (e.g. Washington & Miller 1993; DTIC ADA383925 / ADA388765; the Orion LAS grid-fin
  tests on NTRS) could not be downloaded from the development container. Validating the fin model
  against one of them is the next accuracy step for the fins.

* **Turbulence needs a resolved boundary layer for smooth-body separation.** SST + wall function
  gets skin friction right (flat plate within 3 % at Re 1e9) but a cylinder at Re 1e7 with
  1-2 cells across its ~1 cm boundary layer stays attached (Cd 0.05 instead of 0.5-0.75, no
  shedding). Separation from smooth surfaces needs ~10+ cells across the boundary layer, i.e.
  near-wall refinement and implicit time stepping (roadmap); separation at sharp edges (chines,
  fin plates) is fixed by the geometry and is captured.
* **Inviscid by default.** No boundary layers or skin friction unless `--turbulent`. Pressure loads on the forebody, fins
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
