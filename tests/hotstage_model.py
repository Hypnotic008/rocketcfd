"""Hot-staging model checks (no CFD): Raptor 3 nozzles make their rated thrust and published Isp;
the solver jet state carries the exhaust's density, velocity and pressure; the exhaust
temperature mapping is exact at the nozzle exit and at stagnation; the plates sit on the dome and
the RVac bells clear the truss.

  python tests/hotstage_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rocketcfd import engines as EN, hotstage as HS
from rocketcfd.vehicles.starship import (StarshipParams, engines as ship_engines, hot_stage_truss_sdf,
                                         dome_plate_footprints, dome_plate_centroids, dome_distance)


def main():
    ok = True
    G0 = EN.G0
    for spec, pa, isp_pub in ((EN.RAPTOR3_SL, 101325.0, 350.0), (EN.RAPTOR3_VAC, 0.0, 380.0)):
        e = EN.exit_state(spec, pa)
        thr = abs(e["thrust"] / spec.thrust - 1)
        isp = abs(e["isp"] / isp_pub - 1)
        print(f"{spec.name}: thrust {e['thrust']/G0/1e3:.0f} tf (rated {spec.thrust/G0/1e3:.0f}), Isp {e['isp']:.0f} s "
              f"(published {isp_pub:.0f}), V_e {e['V_e']:.0f} m/s, mdot {e['mdot']:.0f} kg/s")
        ok &= thr < 1e-6 and isp < 0.04
        j = EN.jet(spec, (0, 0, 0), p_amb=pa)
        rho = j["p"] / (j["R"] * j["T"])
        V = j["mach"] * np.sqrt(1.4 * j["R"] * j["T"])
        ok &= abs(rho / e["rho_e"] - 1) < 1e-9 and abs(V / e["V_e"] - 1) < 1e-9
    e = EN.exit_state(EN.RAPTOR3_SL, 10.0)
    th = HS.thermo_constants(230.0, 1378.0, e)
    # the model's state at the exit and at stagnation (same total enthalpy, V = 0)
    Tm_exit = e["p_e"] / (e["rho_e"] * HS.R_AIR)
    T_exit = HS.exhaust_temperature(np.array(Tm_exit), np.array(e["V_e"] ** 2), **th)
    h0 = th["h0_jet"]
    T_stag = HS.exhaust_temperature(np.array(h0 / HS.CP_AIR), np.array(0.0), **th)
    T_air = HS.exhaust_temperature(np.array(230.0), np.array(1378.0 ** 2), **th)
    print(f"exhaust temperature: exit {float(T_exit):.0f} K (T_e {e['T_e']:.0f}), stagnation {float(T_stag):.0f} K "
          f"(T_c {e['T_c']:.0f}), freestream air {float(T_air):.0f} K (230)")
    ok &= abs(T_exit / e["T_e"] - 1) < 1e-6 and abs(T_stag / e["T_c"] - 1) < 0.02 and abs(T_air - 230) < 1e-6
    S = StarshipParams()
    # plates: seven parts (apex disc + 3 + 3), no overlaps, each on the dome surface
    h = 0.02
    ax = np.arange(-4.6, 4.6, h)
    Y, Z = np.meshgrid(ax, ax, indexing="ij")
    fps = dome_plate_footprints(S)
    cover = sum((f(Y, Z) < 0).astype(int) for _, f in fps)
    names = [n for n, _ in fps]
    d = []
    for n, (cy, cz) in dome_plate_centroids(S).items():
        x = 68.3 + 2.9 * np.sqrt(max(0.0, 1 - (np.hypot(cy, cz) / S.radius) ** 2))
        d.append(float(dome_distance(np.array(x), np.array(cy), np.array(cz))))
    print(f"plates: {names}; overlapping: {int((cover > 1).sum())} cells; centroids on the dome: max |d| {max(map(abs, d)):.3f} m")
    ok &= len(fps) == 7 and int((cover > 1).sum()) == 0 and max(map(abs, d)) < 0.03
    from rocketcfd.vehicles.starship import engine_frames
    best = 9.0
    for e in engine_frames(S):
        a = np.linspace(0, 2 * np.pi, 400)
        y, z = e["exit"][1] + e["exit_r"] * np.cos(a), e["exit"][2] + e["exit_r"] * np.sin(a)
        best = min(best, float(hot_stage_truss_sdf(np.full(400, e["exit"][0]), y, z, S).min()))
    lowest = min(e["exit"][0] for e in engine_frames(S))
    print(f"engine exits: lowest at x {lowest:.2f}, the skirt's edge at {S.x_aft:.2f} (inside: {lowest > S.x_aft}); "
          f"bell rims to the truss {best:.3f} m clear")
    ok &= best > 0.03 and lowest > S.x_aft
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
