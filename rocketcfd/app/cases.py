"""Case templates for the app: each maps friendly, grouped form fields onto a case script's
command-line options, with presets for typical uses."""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Field:
    key: str                      # command-line flag without the leading dashes
    label: str
    kind: str                     # float | int | bool | choice | text | floats
    default: Any
    help: str = ""
    group: str = "General"
    choices: Optional[list] = None
    lo: float = -1e12
    hi: float = 1e12
    step: float = 0.1
    decimals: int = 3
    advanced: bool = False
    positional: bool = False      # passed as a bare argument, not --key value
    always: bool = False          # always written to the command line


@dataclass
class Case:
    name: str
    script: str                   # path relative to the repository, or "-m module"
    description: str
    fields: list
    presets: dict = field(default_factory=dict)
    out_key: str = "out"          # field holding the results folder
    dry_run: bool = False         # supports --dry-run (grid size / memory estimate)
    profile: bool = False         # supports --profile N
    gpu_minutes: str = ""


def _sh_fields():
    G1, G2, G3, G4, G5, G6 = ("Flight condition", "Vehicle", "Grid", "Local refinement", "Solver", "Output")
    return [
        Field("mach", "Mach number", "float", 2.82, "Freestream Mach number.", G1, lo=0.05, hi=12, step=0.1),
        Field("alt", "Altitude [m]", "float", 10745.0, "Sets the air density, pressure and temperature (US standard "
              "atmosphere). Inviscid results depend on Mach only.", G1, lo=0, hi=80000, step=500, decimals=0),
        Field("aoa", "Angle of attack [deg]", "float", 5.0, "Degrees away from engines-first (0 = falling "
              "straight down, engines into the wind).", G1, lo=0, hi=90, step=1, decimals=1),
        Field("windward", "Windward side [deg]", "float", 180.0, "Which side faces the crossflow: 0 = rudder fin, "
              "90 / 270 = catch fins, 180 = raceway / tall chines.", G1, lo=0, hi=359, step=15, decimals=0),
        Field("fins", "Grid fins", "choice", "none", "parametric = the V3 grid-fin model as thin walls. Needs "
              "fine cells near the fins (Fin patch cell size).", G2, choices=["none", "parametric", "cad", "cad_filled"]),
        Field("fin-deflection", "Fin deflections [deg]", "floats", "0 0 0", "Three numbers: rudder fin, catch fin "
              "90, catch fin 270.", G2, advanced=True),
        Field("chines", "Chines", "choice", "all", "Which chines to model.", G2, choices=["all", "tall", "short", "none"]),
        Field("engines", "Engines firing", "choice", "none", "Rocket plumes from the nozzle exits (landing / boostback "
              "burn, retropropulsion): centre = 3, inner = 13, all = 33.", G2, choices=["none", "centre", "inner", "all"]),
        Field("exit-mach", "Nozzle exit Mach", "float", 3.5, "", G2, lo=1.0, hi=8.0, step=0.1, advanced=True),
        Field("exit-p", "Nozzle exit pressure [Pa]", "float", 80e3, "", G2, lo=1e3, hi=1e6, step=5e3, decimals=0,
              advanced=True),
        Field("exit-T", "Exit temperature [K]", "float", 2200.0, "Single-gas model: chosen so the exit velocity is "
              "Raptor-like (~3300 m/s).", G2, lo=200, hi=4000, step=50, decimals=0, advanced=True),
        Field("throttle", "Throttle", "float", 1.0, "0..1", G2, lo=0.0, hi=1.0, step=0.05, advanced=True),
        Field("plume-h", "Plume block cell size [m]", "float", 0.0, "Refine a block in front of the engines "
              "(where the plume goes) to this cell size (0 = off).", G2, lo=0.0, hi=1.0, step=0.05),
        Field("plume-length", "Plume block length [m]", "float", 30.0, "", G2, lo=2, hi=200, step=5, advanced=True),
        Field("upstream", "Domain ahead of the engines [m]", "float", 15.5, "Make it longer than the plume "
              "(e.g. 45 m with engines firing).", G3, lo=5, hi=300, step=5, advanced=True),
        Field("h", "Cell size near vehicle [m]", "float", 0.5, "Base grid cell size around the booster. 0.3 is a good "
              "balance; 0.5 for quick looks.", G3, lo=0.02, hi=2, step=0.05),
        Field("h-far", "Farfield cell size [m]", "float", 0.0, "0 = 4 x the near-vehicle size.", G3, lo=0, hi=10,
              advanced=True),
        Field("growth", "Max stretching ratio", "float", 1.12, "Largest size ratio between neighbouring cells.", G3,
              lo=1.0, hi=1.5, step=0.01, advanced=True),
        Field("fin-h", "Fin patch cell size [m]", "float", 0.0, "Refinement patches around the grid fins (0 = off). "
              "0.06 resolves the lattice; turns the implicit solver on.", G4, lo=0, hi=0.5, step=0.01),
        Field("fin-shell", "Fin patch thickness [m]", "float", 0.3, "How far around the fins to refine.", G4,
              lo=0.05, hi=2, advanced=True),
        Field("engine-h", "Engine-end patch cell size [m]", "float", 0.0, "Refinement around the engine bells, thrust "
              "plate and skirt (0 = off). 0.06 resolves the gaps between the nozzles.", G4, lo=0, hi=0.5, step=0.01),
        Field("engine-shell", "Engine patch thickness [m]", "float", 0.5, "", G4, lo=0.05, hi=2, advanced=True),
        Field("wall-h", "Hull shell patch cells hx,hy,hz [m]", "text", "", "Viscous runs: fine cells across the "
              "hull wall, e.g. 0.25,0.03,0.03 (blank = off).", G4, advanced=True),
        Field("wall-shell", "Hull shell thickness [m]", "float", 0.45, "", G4, lo=0.05, hi=3, advanced=True),
        Field("patch-box", "Max patch size (cells x,y,z)", "text", "32,96,96", "Smaller boxes hug thin shells "
              "tighter (fewer cells) but make more patches.", G4, advanced=True),
        Field("turbulent", "Viscous turbulent (RANS k-omega SST)", "bool", False, "Viscous air with the SST model and "
              "a wall function. Off = inviscid (Euler).", G5),
        Field("les", "Unsteady LES (wall-modelled)", "bool", False, "Time-accurate large-eddy simulation with the "
              "Vreman subgrid model and an equilibrium wall model: for unsteady wakes and movies. Use with "
              "Snapshots.", G5),
        Field("scheme", "Reconstruction scheme", "choice", "koren", "koren: third-order MUSCL (default). weno5z: "
              "fifth-order WENO-Z as in NASA LAVA, less numerical dissipation: sharper vortices and wakes in "
              "LES. Time-accurate runs only (steady / implicit solves use koren).", G5,
              choices=["koren", "weno5z"]),
        Field("adapt-from", "Refine from LES (folder)", "text", "", "An earlier LES run's results folder: add "
              "refinement patches where its time-averaged error criterion is largest (Mozaffari et al. 2022).",
              G5, advanced=True),
        Field("adapt-h", "Adapted patch cell size [m]", "float", 0.15, "", G5, lo=0.02, hi=1.0, step=0.01,
              advanced=True),
        Field("adapt-cells", "Adapted patch cell budget", "float", 3e6, "Number of new cells the adapted patches "
              "may use (sets the refinement threshold, as in LAVA).", G5, lo=1e4, hi=5e7, step=5e5, advanced=True),
        Field("implicit", "Implicit solver", "bool", False, "Implicit steady solver: far fewer iterations. "
              "On automatically with refinement patches.", G5),
        Field("explicit", "Force explicit (with patches)", "bool", False, "Explicit local time steps on the patch "
              "grids instead of the implicit solver.", G5, advanced=True),
        Field("cfl-max", "Implicit CFL", "float", 0.0, "0 = automatic (20; 5 with engines firing)", G5, lo=0, hi=200,
              step=5, advanced=True),
        Field("iters", "Max implicit iterations", "int", 4000, "", G5, lo=10, hi=100000, step=500),
        Field("flowthroughs", "Max flow-throughs (explicit)", "float", 4.0, "Run length for the explicit solver.",
              G5, lo=0.1, hi=100, step=1, advanced=True),
        Field("tol", "Convergence tolerance", "float", 5e-4, "Stop when CD, CN, C_restore change less than this "
              "(0 = never stop early).", G5, lo=0, hi=0.1, step=1e-4, decimals=5),
        Field("time-accurate", "Time-accurate", "bool", False, "Global time step: physically correct unsteady flow "
              "(movies). Slower to converge.", G5, advanced=True),
        Field("snapshots", "Snapshots for movies", "int", 0, "Save this many frames during the run (0 = none).",
              G6, lo=0, hi=5000, step=50),
        Field("snap-volume", "3-D snapshot stride", "int", 0, "Also save a coarse 3-D field every n-th cell "
              "(0 = planes only). Needed for 3-D LAVA-style movies; 2 is a good value.", G6, lo=0, hi=8, step=1),
        Field("probes", "Probe points", "bool", False, "Record pressure and velocity every step at points in the "
              "wake, beside the hull, behind the fins and at the engine end (Plots -> Probes: history and "
              "spectrum). For unsteady / LES runs.", G6),
        Field("probe-points", "Extra probe points", "text", "", 'e.g. "75,0,0; 80,0,3" (x,y,z in m)', G6, advanced=True),
        Field("restart", "Start from run (folder)", "text", "", "A previous run's results folder: same grid = "
              "continue it; other grid or physics = interpolated start (e.g. LES or RANS from a converged "
              "inviscid run, which skips most of the start-up).", G5),
        Field("x64", "Double precision", "bool", False, "", G6, advanced=True),
        Field("out", "Results folder", "text", "results/sh_descent", "", G6, always=True),
    ]


SUPER_HEAVY = Case(
    "Super Heavy descent - one flight condition",
    "cases/super_heavy_descent.py",
    "Unpowered engines-first descent of Super Heavy V3 (hull, chines, 33 engine bells, grid fins). "
    "Writes forces and moments (CatchSim coefficients), 3-D flow, surface pressure and images.",
    _sh_fields(),
    presets={
        "Quick look (body only, coarse)": dict(h=0.5, fins="none", implicit=True, iters=1500),
        "Glide with fins (inviscid, patches)": dict(h=0.3, fins="parametric", **{"fin-h": 0.06, "engine-h": 0.06},
                                                   aoa=15.0, windward=0.0, iters=4000),
        "Peak-q, viscous RANS (22 M cells, ~4 GB)": dict(h=0.3, fins="parametric", turbulent=True, aoa=15.0, windward=0.0,
                                                        **{"fin-h": 0.06, "engine-h": 0.06, "wall-h": "0.25,0.03,0.03",
                                                           "wall-shell": 0.3, "patch-box": "32,48,48"}),
        "Landing burn plume (3 engines, subsonic)": dict(mach=0.4, alt=2000.0, aoa=0.0, h=0.3, engines="centre",
                                                         implicit=True, iters=3000, fins="parametric",
                                                         **{"engine-h": 0.1, "plume-h": 0.15, "upstream": 45.0}),
        "Movie: unsteady LES wake": dict(h=0.3, fins="parametric", les=True, **{"time-accurate": True,
                                         "snapshots": 200, "snap-volume": 1, "fin-h": 0.0, "probes": True}, aoa=15.0,
                                         windward=0.0),
    },
    dry_run=True, profile=True, gpu_minutes="10-60 min on a laptop GPU")


AERO_DB = Case(
    "Aero database - Mach x AoA x roll sweep (CatchSim)",
    "cases/aero_database.py",
    "Runs every combination of Mach, angle of attack and roll, warm-starting each from the last, and "
    "writes aero_db.json for CatchSim. Resumable: re-running skips finished cases.",
    [
        Field("mach", "Mach numbers", "floats", "4.0 3.2 2.6 2.0 1.6 1.3 1.1 0.95 0.8 0.6", "Space-separated list.",
              "Sweep", always=True),
        Field("aoa", "Angles of attack [deg]", "floats", "0 4 8 12 16 20", "", "Sweep", always=True),
        Field("roll", "Roll angles [deg]", "floats", "0 90 180", "", "Sweep", always=True),
        Field("alt", "Altitude [m]", "float", 10745.0, "", "Sweep", lo=0, hi=80000, step=500, decimals=0),
        Field("h", "Cell size near vehicle [m]", "float", 0.3, "", "Grid", lo=0.05, hi=2, step=0.05),
        Field("fins", "Grid fins", "choice", "parametric", "", "Grid", choices=["none", "parametric"]),
        Field("fin-h", "Fin patch cell size [m]", "float", 0.06, "", "Grid", lo=0, hi=0.5, step=0.01),
        Field("engine-h", "Engine-end patch cell size [m]", "float", 0.06, "", "Grid", lo=0, hi=0.5, step=0.01),
        Field("turbulent", "Viscous turbulent (RANS)", "bool", False, "", "Solver"),
        Field("explicit", "Explicit solver", "bool", False, "", "Solver", advanced=True),
        Field("max-iters", "Max iterations per case", "int", 1500, "", "Solver", lo=50, hi=50000, step=100),
        Field("tol", "Convergence tolerance", "float", 5e-4, "", "Solver", lo=0, hi=0.1, decimals=5),
        Field("out", "Results folder", "text", "results/aero_db", "", "Output", always=True),
    ],
    presets={
        "Quick database (12 cases)": dict(mach="2.8 2.0 1.3 0.9", aoa="0 8 16", roll="0 180"),
        "Full database (180 cases)": {},
    },
    dry_run=True, gpu_minutes="hours: about 10-20 min per case")


GRID_FIN = Case(
    "Isolated grid fin - Mach x alpha sweep",
    "cases/grid_fin.py",
    "One V3 grid fin in a wind tunnel: normal force slope, drag and hinge moment versus Mach. "
    "Compares with cascade theory.",
    [
        Field("mach", "Mach numbers", "floats", "0.6 0.8 0.9 1.0 1.1 1.3 1.6 2.0 2.8 4.0", "", "Sweep", always=True),
        Field("alpha", "Angles [deg]", "floats", "0 5 10", "", "Sweep", always=True),
        Field("h", "Cell size around the fin [m]", "float", 0.025, "", "Grid", lo=0.005, hi=0.2, step=0.005),
        Field("implicit", "Implicit solver", "bool", True, "", "Solver"),
        Field("out", "Results folder", "text", "results/grid_fin", "", "Output", always=True),
    ],
    presets={"Supersonic only": dict(mach="1.6 2.0 2.8 4.0")}, dry_run=True)


CYLINDER = Case(
    "Validation - cylinder vortex shedding (unsteady)",
    "cases/validate_cylinder_turbulent.py",
    "Flow past a circular cylinder: drag and Strouhal number against experiment. Re 100 laminar is "
    "the classic von Karman street - a good first unsteady case and a nice movie.",
    [
        Field("re", "Reynolds number", "float", 1e7, "", "Flow", lo=10, hi=1e9, decimals=0, step=100),
        Field("mach", "Mach number", "float", 0.3, "", "Flow", lo=0.05, hi=0.8),
        Field("laminar", "Laminar (no turbulence model)", "bool", False, "Use for Re up to ~180.", "Flow"),
        Field("h", "Cell size near cylinder [m]", "float", 0.008, "Diameter is 1 m.", "Grid", lo=0.001, hi=0.1,
              step=0.001, decimals=4),
        Field("dual", "Dual time stepping", "bool", False, "Implicit inner iterations: larger time steps.", "Solver"),
        Field("periods", "Shedding periods", "float", 25.0, "", "Solver", lo=1, hi=500, step=5),
        Field("out", "Results folder", "text", "results", "", "Output", always=True),
    ],
    presets={"Re 100 von Karman street": dict(re=100.0, mach=0.2, laminar=True, h=0.02, dual=True)})


TGV = Case(
    "Validation - Taylor-Green vortex (LES)",
    "tests/taylor_green.py",
    "Decaying turbulence in a periodic box at Re 1600: the standard benchmark for the LES mode "
    "(dissipation rate vs the DNS peak).",
    [
        Field("n", "Cells per side", "int", 32, "", "Grid", lo=16, hi=256, step=16, positional=True, always=True),
        Field("t", "End time", "float", 10.0, "", "Solver", lo=1, hi=20, step=1),
        Field("out", "Results folder", "text", "results/taylor_green", "", "Output", always=True),
    ])


PLATE = Case(
    "Validation - inclined flat plate (thin wall)",
    "cases/validate_flat_plate.py",
    "Supersonic flat plate at incidence against shock-expansion theory: checks the thin-wall model "
    "used for the grid fins.",
    [
        Field("mach", "Mach number", "float", 2.0, "", "Flow", lo=1.1, hi=8),
        Field("alpha", "Angle [deg]", "float", 5.0, "", "Flow", lo=0, hi=30),
        Field("h", "Cell size [m]", "float", 0.01, "", "Grid", lo=0.001, hi=0.1, decimals=4),
        Field("inclined", "Plate inclined to the grid", "bool", False, "", "Grid"),
        Field("implicit", "Implicit solver", "bool", True, "", "Solver"),
    ])


HOT_STAGING = Case(
    "Hot staging - Starship lights on the booster",
    "cases/hot_staging.py",
    "Starship's engines ignite while it sits on the booster's vented interstage: the plumes hit the "
    "booster's dome and blow out through the vents. Time-accurate, with an ignition ramp; made for "
    "movies (snapshots) and pressure histories (probes). Approximate public dimensions.",
    [
        Field("mach", "Mach number", "float", 4.5, "", "Flight condition", lo=0.5, hi=10, step=0.5),
        Field("alt", "Altitude [m]", "float", 65000.0, "", "Flight condition", lo=0, hi=90000, step=1000, decimals=0),
        Field("aoa", "Angle of attack [deg]", "float", 0.0, "", "Flight condition", lo=0, hi=20, step=1),
        Field("ship-engines", "Ship engines", "choice", "all", "", "Engines", choices=["all", "sea-level", "none"]),
        Field("booster-engines", "Booster engines still firing", "choice", "centre", "", "Engines",
              choices=["centre", "inner", "none"]),
        Field("ignition", "Ignition ramp [s]", "float", 0.02, "", "Engines", lo=0.0, hi=1.0, step=0.005, decimals=3),
        Field("h", "Cell size near the stack [m]", "float", 0.5, "", "Grid", lo=0.05, hi=2, step=0.05),
        Field("ring-h", "Interstage refinement [m]", "float", 0.0, "Cell size in a block around the hot-stage ring "
              "(0 = off), e.g. 0.1.", "Grid", lo=0.0, hi=0.5, step=0.02),
        Field("time", "Physical time [s]", "float", 0.15, "", "Solver", lo=0.001, hi=5, step=0.05),
        Field("cfl", "CFL", "float", 0.6, "", "Solver", lo=0.1, hi=0.9, step=0.05, advanced=True),
        Field("snapshots", "Snapshots for movies", "int", 0, "", "Output", lo=0, hi=5000, step=50),
        Field("snap-volume", "3-D snapshot stride", "int", 0, "", "Output", lo=0, hi=8),
        Field("probes", "Probe points", "bool", False, "", "Output"),
        Field("out", "Results folder", "text", "results/hot_staging", "", "Output", always=True),
    ],
    presets={"Quick test (coarse, 50 ms)": dict(h=1.0, time=0.05, snapshots=20, **{"snap-volume": 1}, probes=True),
             "Movie (fine interstage, 0.25 s)": dict(h=0.4, time=0.25, snapshots=150, probes=True,
                                                     **{"ring-h": 0.1, "snap-volume": 1})},
    dry_run=True, gpu_minutes="minutes (quick test) to ~1-2 h (movie)")


REENTRY = Case(
    "Starship re-entry - belly first at high angle of attack",
    "cases/starship_reentry.py",
    "Starship falling belly-first (~65 deg angle of attack), flaps out: bow shock, shock layer and wake; "
    "normal / axial force and pitching moment. One perfect gas: shock-layer temperatures above Mach ~6 "
    "are too high.",
    [
        Field("mach", "Mach number", "float", 8.0, "", "Flight condition", lo=1.5, hi=30, step=1),
        Field("alt", "Altitude [m]", "float", 55000.0, "", "Flight condition", lo=0, hi=90000, step=1000, decimals=0),
        Field("aoa", "Angle of attack [deg]", "float", 65.0, "", "Flight condition", lo=0, hi=90, step=5),
        Field("h", "Cell size near the ship [m]", "float", 0.5, "", "Grid", lo=0.05, hi=2, step=0.05),
        Field("iters", "Max implicit iterations", "int", 3000, "", "Solver", lo=50, hi=50000, step=500),
        Field("cfl-max", "Implicit CFL", "float", 10.0, "", "Solver", lo=1, hi=100, step=5, advanced=True),
        Field("out", "Results folder", "text", "results/reentry", "", "Output", always=True),
    ],
    presets={"Quick look (1 m cells)": dict(h=1.0, iters=800), "Standard (0.3 m)": dict(h=0.3)},
    dry_run=True, gpu_minutes="5-30 min")


CASES = [SUPER_HEAVY, AERO_DB, HOT_STAGING, REENTRY, GRID_FIN, CYLINDER, TGV, PLATE]


def command_line(case, values):
    """Arguments (list of str) for the case script from {key: value}."""
    args = []
    for f in case.fields:
        v = values.get(f.key, f.default)
        if f.kind == "bool":
            if v:
                args.append("--" + f.key)
            continue
        if not f.always and v == f.default:
            continue
        if f.kind == "text" and str(v).strip() == "" and not f.always:
            continue
        if f.kind == "floats":
            vals = str(v).replace(",", " ").split()
            args += vals if f.positional else ["--" + f.key] + vals
            continue
        sv = str(v)
        args += [sv] if f.positional else ["--" + f.key, sv]
    return args
