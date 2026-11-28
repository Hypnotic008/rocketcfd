"""Raptor 3 engines as plume inflow conditions: one-dimensional ideal nozzle flow of methane /
oxygen combustion products, sized to the rated thrust.

The CFD does not simulate the combustion chamber or the nozzle. Each engine is an inflow at its
nozzle exit disk (rocketcfd.solver `jets`): the cells just inside the exit hold the exhaust
state (density, velocity, pressure), so the plume leaves through the exit plane. That state
comes from the chamber conditions through an isentropic expansion to the exit area:

  chamber      p_c, T_c (adiabatic flame temperature of CH4/LOX at the mixture ratio)
  throat       A* from the thrust: F = A* p_c C_F(eps, gamma, p_amb / p_c), eps = A_e / A*
  exit         M_e from the area ratio, then p_e, T_e, V_e, rho_e

Exhaust gas: mostly H2O, CO2, CO and H2, mean molar mass ~22.6 g/mol. The ideal-nozzle formulas
use one isentropic exponent; with frozen flow (gamma ~1.20) they give ~7 % less Isp than Raptor,
because the real exhaust keeps recombining (releasing heat) as it expands. gamma = 1.13 is the
effective exponent that reproduces the published figures: Raptor 3 vacuum 381 s in vacuum
(published 380 s), Raptor 3 sea level 342 s at sea level (published ~350 s). Methalox burns essentially soot-free: there are no condensed particles in the plume (as
in solid motors' alumina), so the plume is a gas with that molar mass.

The solver carries one gas, air (gamma 1.4). The exit state is imposed with the exhaust's
density, velocity and pressure, so mass flow, momentum flux (thrust) and exit pressure are
Raptor's; downstream the gas expands and compresses with gamma 1.4, and temperatures are
recovered with rocketcfd.hotstage.exhaust_temperature.

Published figures (approximate, SpaceX 2024): Raptor 3 sea level 280 tf at sea level, Raptor 3
vacuum ~306 tf, chamber pressure ~350 bar, exit diameters ~1.3 m and ~2.3 m. Chamber and
injector sizes below are estimates from the throat area and a typical contraction ratio; the
injector itself (hundreds of coaxial elements, millimetre scale) is far below any grid here.

  python -m rocketcfd.engines            the table for both engines at 65 km and at sea level
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

R_UNIV = 8314.46
G0 = 9.80665


@dataclass
class RaptorSpec:
    name: str
    thrust: float                 # rated thrust [N]
    thrust_at: float              # ambient pressure of the rating [Pa] (101325: sea level, 0: vacuum)
    exit_diameter: float          # [m]
    p_c: float = 350e5            # chamber pressure [Pa]
    T_c: float = 3600.0           # chamber temperature [K]
    gamma: float = 1.13           # effective (shifting equilibrium), fitted to the published Isp
    molar_mass: float = 22.6      # [kg/kmol]
    mixture_ratio: float = 3.6    # O/F by mass
    contraction: float = 3.5      # chamber / throat area (estimate)

    @property
    def R(self):
        return R_UNIV / self.molar_mass


RAPTOR3_SL = RaptorSpec("Raptor 3 sea level", 280e3 * G0, 101325.0, 1.30)
RAPTOR3_VAC = RaptorSpec("Raptor 3 vacuum", 306e3 * G0, 0.0, 2.30)


def _area_ratio(M, g):
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M * M)) ** ((g + 1) / (2 * (g - 1)))


def _expand(eps, g):
    """Supersonic exit Mach number and p_e / p_c for area ratio eps."""
    Me = brentq(lambda M: _area_ratio(M, g) - eps, 1.0001, 30.0)
    return Me, (1 + 0.5 * (g - 1) * Me * Me) ** (-g / (g - 1))


def _thrust_coeff(eps, g, pe_pc, pa_pc):
    k = np.sqrt(2 * g * g / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe_pc ** ((g - 1) / g)))
    return k + (pe_pc - pa_pc) * eps


def design(spec: RaptorSpec):
    """Throat sized so the engine makes its rated thrust at its rating pressure."""
    g = spec.gamma
    Ae = np.pi * spec.exit_diameter ** 2 / 4

    def resid(At):
        eps = Ae / At
        _, pe_pc = _expand(eps, g)
        return At * spec.p_c * _thrust_coeff(eps, g, pe_pc, spec.thrust_at / spec.p_c) - spec.thrust

    At = brentq(resid, Ae / 400, Ae / 2)
    return At, Ae


def exit_state(spec: RaptorSpec, p_amb=0.0, throttle=1.0):
    """Everything about the engine at ambient pressure p_amb (chamber pressure scaled by throttle)."""
    g, R = spec.gamma, spec.R
    At, Ae = design(spec)
    pc = spec.p_c * throttle
    eps = Ae / At
    Me, pe_pc = _expand(eps, g)
    pe = pe_pc * pc
    Te = spec.T_c / (1 + 0.5 * (g - 1) * Me * Me)
    ae = np.sqrt(g * R * Te)
    Ve = Me * ae
    rho_e = pe / (R * Te)
    mdot = rho_e * Ve * Ae
    cstar = np.sqrt(g * R * spec.T_c) / g / (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    F = mdot * Ve + (pe - p_amb) * Ae
    return dict(name=spec.name, propellant=f"CH4 / LOX, O/F {spec.mixture_ratio}", p_c=pc, T_c=spec.T_c,
                gamma=g, molar_mass=spec.molar_mass, R=R, throat_diameter=2 * np.sqrt(At / np.pi),
                chamber_diameter=2 * np.sqrt(spec.contraction * At / np.pi), exit_diameter=spec.exit_diameter,
                area_ratio=eps, M_e=Me, p_e=pe, T_e=Te, V_e=Ve, rho_e=rho_e, mdot=mdot, cstar=cstar,
                thrust=F, isp=F / (mdot * G0), p_amb=p_amb, momentum_flux=rho_e * Ve * Ve,
                underexpansion=pe / max(p_amb, 1e-9))


def jet(spec: RaptorSpec, center, direction=(-1.0, 0.0, 0.0), p_amb=0.0, gamma_solver=1.4, radius_frac=1.0):
    """A solver `jets` entry with Raptor's exit density, velocity and pressure. The solver's jet
    state is rho = p / (R T), u = mach sqrt(gamma_solver R T): with R and T of the exhaust and
    mach = V_e / sqrt(gamma_solver R T_e) it imposes exactly (rho_e, V_e, p_e)."""
    e = exit_state(spec, p_amb)
    m_solver = e["V_e"] / np.sqrt(gamma_solver * e["R"] * e["T_e"])
    return dict(center=tuple(float(c) for c in center), direction=tuple(direction),
                radius=radius_frac * 0.5 * e["exit_diameter"], area=float(np.pi * e["exit_diameter"] ** 2 / 4), mach=float(m_solver), p=float(e["p_e"]),
                T=float(e["T_e"]), R=float(e["R"]), lip=True, engine=e)


def table(p_amb=10.0):
    rows = [exit_state(s, p_amb) for s in (RAPTOR3_SL, RAPTOR3_VAC)]
    fmt = [("propellant", "{}", 1), ("chamber pressure [bar]", "{:.0f}", "p_c", 1e-5),
           ("chamber temperature [K]", "{:.0f}", "T_c", 1), ("exhaust molar mass [g/mol]", "{:.1f}", "molar_mass", 1),
           ("exhaust gamma", "{:.2f}", "gamma", 1), ("chamber (injector face) diameter [m]", "{:.2f}", "chamber_diameter", 1),
           ("throat diameter [m]", "{:.3f}", "throat_diameter", 1), ("exit diameter [m]", "{:.2f}", "exit_diameter", 1),
           ("area ratio", "{:.0f}", "area_ratio", 1), ("exit Mach", "{:.2f}", "M_e", 1),
           ("exit pressure [kPa]", "{:.1f}", "p_e", 1e-3), ("exit temperature [K]", "{:.0f}", "T_e", 1),
           ("exit velocity [m/s]", "{:.0f}", "V_e", 1), ("exit density [kg/m3]", "{:.3f}", "rho_e", 1),
           ("mass flow [kg/s]", "{:.0f}", "mdot", 1), ("thrust here [kN]", "{:.0f}", "thrust", 1e-3),
           ("Isp here [s]", "{:.0f}", "isp", 1)]
    print(f"Raptor 3 at ambient {p_amb:g} Pa (ideal nozzle)")
    print(f"{'':38s}" + "".join(f"{r['name']:>22s}" for r in rows))
    for item in fmt:
        if item[0] == "propellant":
            print(f"{'propellant':38s}" + "".join(f"{r['propellant']:>22s}" for r in rows))
            continue
        lab, f, key, sc = item
        print(f"{lab:38s}" + "".join(f"{f.format(r[key] * sc):>22s}" for r in rows))
    return rows


if __name__ == "__main__":
    table(10.0)
    print()
    table(101325.0)
