"""Hot staging: gas temperature and pressure on the booster's forward dome, its protection
plates, the V3 truss and the ship's aft end, during the ship's engine start.

  * part_meshes: surface meshes of each component, from its own SDF, so the dome, every
    protection plate, the truss and the ship's aft end are separate surfaces (separate colours
    and separate statistics).
  * WallSampler: the gas next to each surface vertex (one fine cell out along the normal),
    gathered on the GPU from the finest grid there; per-vertex pressure and temperature into
    every snapshot, a running peak, and per-part time histories.
  * exhaust_temperature: the solver carries one gas (gamma 1.4). The engines impose Raptor's
    exit density, velocity and pressure (rocketcfd.engines), so mass flow and thrust are right,
    but the gas then stores its internal energy as gamma-1.4 gas: where the plume stops on the
    dome, p / (rho R) of the model reads ~5000+ K, where the real exhaust (cp ~3.2 kJ/kg/K
    effective) reaches its chamber temperature, ~3600 K. Temperature is recovered from the
    enthalpy instead: the exhaust fraction f = (h0 - h0_inf) / (h0_jet - h0_inf) from the total
    enthalpy (conserved through mixing and shocks in adiabatic flow), the real static enthalpy
    h = h_model + f dh (dh: the exhaust's internal energy the gamma-1.4 model leaves out at the
    exit), T = h / (f cp_exh + (1 - f) cp_air). Exact at the nozzle exit (T_e) and where the
    plume stops (T_c); approximate between: constant cp, no dissociation of shocked air, no wall
    heat transfer. Gas temperature next to the wall, not the wall's own temperature.
  * Near vacuum (65 km: 10 Pa, 1.6e-4 kg/m^3) p / rho is noise in the almost empty cells around
    the plume, and the thin air the starting plume shocks really is heated to ~8000 K but carries
    no heat. Temperatures count only where the gas density exceeds rho_min (default 5e-3 kg/m^3,
    ~30x ambient); peaks are 99.5th percentiles over each part.

  python -m rocketcfd.hotstage report results/hot_staging        figures + summary of a run
"""
import os
import glob
import json
import numpy as np

NG = 2
R_AIR = 287.05
CP_AIR = 3.5 * R_AIR       # gamma 1.4, consistent with the solver
T_CHAMBER = 3600.0          # Raptor (methalox, ~O/F 3.6) chamber temperature [K], approximate


# ---------------------------------------------------------------- temperature
def thermo_constants(T_inf, V_inf, engine, gamma_solver=1.4, rho_min=5e-3):
    """Constants for exhaust_temperature. engine: rocketcfd.engines.exit_state(...) of the
    engine whose plume dominates (the ship's sea-level Raptors)."""
    h0_inf = CP_AIR * T_inf + 0.5 * V_inf ** 2
    Re, Te, Ve, ge = engine["R"], engine["T_e"], engine["V_e"], engine["gamma"]
    k = gamma_solver / (gamma_solver - 1)
    h_model = k * Re * Te                          # the model's static enthalpy p / rho * g/(g-1) at the exit
    cp_exh = ge / (ge - 1) * Re
    return dict(h0_inf=float(h0_inf), h0_jet=float(h_model + 0.5 * Ve * Ve), cp_exh=float(cp_exh),
                dh_exh=float(cp_exh * Te - h_model), rho_min=float(rho_min), T_c=float(engine["T_c"]), T_inf=float(T_inf))


def exhaust_temperature(T_model, V2, h0_inf, h0_jet, cp_exh, dh_exh, **_):
    """Gas temperature of the air / exhaust mixture (numpy or torch). T_model = p / (rho R_air)."""
    h = CP_AIR * T_model
    f = ((h + 0.5 * V2 - h0_inf) / (h0_jet - h0_inf)).clip(0.0, 1.0)
    return (h + f * dh_exh) / (f * cp_exh + (1.0 - f) * CP_AIR)


# ---------------------------------------------------------------- geometry
def _march(sdf, lo, hi, h):
    from skimage.measure import marching_cubes
    ax = [np.arange(lo[d], hi[d] + 0.5 * h, h) for d in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    F = sdf(X, Y, Z)
    if F.min() >= 0 or F.max() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    v, f, _, _ = marching_cubes(F, 0.0, spacing=(h, h, h))
    return v + np.asarray(lo, float), f.astype(np.int64)


def _normals(sdf, v, eps=0.01):
    n = np.zeros_like(v)
    for d in range(3):
        a, b = v.copy(), v.copy()
        a[:, d] += eps
        b[:, d] -= eps
        n[:, d] = sdf(a[:, 0], a[:, 1], a[:, 2]) - sdf(b[:, 0], b[:, 1], b[:, 2])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def vertex_areas(v, f):
    a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    out = np.zeros(len(v))
    for k in range(3):
        np.add.at(out, f[:, k], a / 3.0)
    return out


def part_meshes(S, PB, h=0.04, verbose=True):
    """{name: dict(v, f, sdf)} for the dome, each protection plate, the truss and the ship's aft
    end. S: StarshipParams, PB: SuperHeavyParams (dome position)."""
    from .vehicles import starship as st
    from .vehicles.super_heavy import super_heavy_sdf
    x0, rise, R = PB.tank_top_x, PB.dome_rise, S.radius
    out = {}
    dome = lambda X, Y, Z: np.maximum(st.dome_distance(X, Y, Z, x0, rise, R), x0 + 0.02 - X)
    v, f = _march(dome, (x0 - 0.1, -R - 0.1, -R - 0.1), (x0 + rise + 0.1, R + 0.1, R + 0.1), h)
    out["dome"] = dict(v=v, f=f, sdf=lambda X, Y, Z: st.dome_distance(X, Y, Z, x0, rise, R))
    if S.ring_style != "vented":
        for k, (cy, cz, rp) in enumerate(st.dome_plate_list(S)):
            sd = (lambda kk: lambda X, Y, Z: st.dome_plates_sdf(X, Y, Z, S, x0, rise, which=kk))(k)
            lo = (x0 + 0.5, cy - rp - 0.06, cz - rp - 0.06)
            hi = (x0 + rise + 0.2, cy + rp + 0.06, cz + rp + 0.06)
            v, f = _march(sd, lo, hi, 0.5 * h)
            out[f"plate {k}"] = dict(v=v, f=f, sdf=sd)
        tr = lambda X, Y, Z: st.hot_stage_truss_sdf(X, Y, Z, S)
    else:
        tr = lambda X, Y, Z: st.hot_stage_ring_sdf(X, Y, Z, S)
    v, f = _march(tr, (S.ring_x0 - 0.1, -R - 0.1, -R - 0.1), (S.x_aft + 0.1, R + 0.1, R + 0.1), h)
    out["truss"] = dict(v=v, f=f, sdf=tr)
    ship = lambda X, Y, Z: np.maximum(st.starship_sdf(X, Y, Z, S), X - (S.x_aft + S.skirt_height + 0.6))
    v, f = _march(ship, (S.x_aft - 0.6, -R - 0.1, -R - 0.1), (S.x_aft + S.skirt_height + 0.7, R + 0.1, R + 0.1), h)
    out["ship aft"] = dict(v=v, f=f, sdf=lambda X, Y, Z: st.starship_sdf(X, Y, Z, S))
    if verbose:
        print("hot-stage surfaces: " + ", ".join(f"{k} {len(p['v'])}" for k, p in out.items()) + " vertices")
    return out


def dome_mask(v, S):
    """Dome vertices not under a protection plate (the bare dome between the plates)."""
    from .vehicles.starship import dome_plate_list
    if S.ring_style == "vented":
        return np.ones(len(v), bool)
    ok = np.ones(len(v), bool)
    for cy, cz, rp in dome_plate_list(S):
        ok &= np.hypot(v[:, 1] - cy, v[:, 2] - cz) > rp
    return ok


# ---------------------------------------------------------------- sampling
class WallSampler:
    """Gas pressure and temperature next to each part's vertices, every record()."""

    def __init__(self, grids, parts, thermo, S, folder=None, offset_cells=0.8):
        import torch
        self.thermo = thermo
        self.names = list(parts)
        v = np.concatenate([parts[k]["v"] for k in self.names])
        n = np.concatenate([_normals(parts[k]["sdf"], parts[k]["v"]) for k in self.names])
        self.split = np.cumsum([0] + [len(parts[k]["v"]) for k in self.names])
        faces = [parts[k]["f"] for k in self.names]
        self.areas = np.concatenate([vertex_areas(parts[k]["v"], parts[k]["f"]) for k in self.names])
        self.weight = self.areas.copy()
        i_d = self.names.index("dome")
        a, b = self.split[i_d], self.split[i_d + 1]
        self.weight[a:b] *= dome_mask(v[a:b], S)                 # dome stats: the bare dome only
        order = sorted(range(len(grids)), key=lambda i: float(min(np.min(np.diff(ax)) for ax in grids[i].axes)))
        owner = np.full(len(v), -1)
        self.gather = []                  # (grid, corners tensor, weights tensor, point index tensor)
        for gi in order:
            g = grids[gi]
            ax = g.axes
            hmin = float(min(np.min(np.diff(a_)) for a_ in ax))
            free = np.flatnonzero(owner < 0)
            if free.size == 0:
                break
            P = v[free] + offset_cells * hmin * n[free]
            lo = np.array([a_[NG] for a_ in ax]); hi = np.array([a_[-NG - 1] for a_ in ax])
            ins = np.all((P >= lo) & (P <= hi), axis=1)
            if not ins.any():
                continue
            P, idx = P[ins], free[ins]
            fi = np.stack([np.interp(P[:, d], ax[d], np.arange(len(ax[d]))) for d in range(3)], 1)
            i0 = np.floor(fi).astype(np.int64)
            fr = fi - i0
            shape = g.phi.shape
            fluid = np.asarray(g.fluid).reshape(-1)
            cs, ws = [], []
            for da in (0, 1):
                for db in (0, 1):
                    for dc in (0, 1):
                        ii = np.minimum(i0[:, 0] + da, shape[0] - 1)
                        jj = np.minimum(i0[:, 1] + db, shape[1] - 1)
                        kk = np.minimum(i0[:, 2] + dc, shape[2] - 1)
                        c = np.ravel_multi_index((ii, jj, kk), shape)
                        w = ((fr[:, 0] if da else 1 - fr[:, 0]) * (fr[:, 1] if db else 1 - fr[:, 1])
                             * (fr[:, 2] if dc else 1 - fr[:, 2]))
                        cs.append(c)
                        ws.append(w * fluid[c])             # only fluid cells: no ghost / solid states
            W = np.stack(ws, 1)
            sw = W.sum(1)
            good = sw > 0.05
            if not good.any():
                continue
            W = W[good] / sw[good, None]
            C = np.stack(cs, 1)[good]
            idx = idx[good]
            owner[idx] = gi
            dev = g.U.device
            self.gather.append((g, torch.as_tensor(C, device=dev), torch.as_tensor(W, dtype=g.dtype, device=dev),
                                torch.as_tensor(idx, device=dev), hmin))
        self.valid = owner >= 0
        self.n = len(v)
        self.peak_p = np.full(self.n, np.nan)
        self.peak_T = np.full(self.n, np.nan)
        self.history = []
        h_used = {}
        for g, _, _, idx, hmin in self.gather:
            h_used[f"{hmin:.3f}"] = h_used.get(f"{hmin:.3f}", 0) + int(idx.numel())
        print(f"wall sampler: {int(self.valid.sum())} of {self.n} surface points in fluid "
              f"(by grid spacing: {', '.join(f'{k} m: {c}' for k, c in sorted(h_used.items()))})")
        self.mesh = dict(names=np.array(self.names), split=self.split, v=v.astype(np.float32), n=n.astype(np.float32),
                         faces_split=np.cumsum([0] + [len(f_) for f_ in faces]),
                         f=np.concatenate(faces).astype(np.int32), valid=self.valid, areas=self.areas.astype(np.float32),
                         weight=self.weight.astype(np.float32), **{k: np.float64(x) for k, x in thermo.items()})
        if folder:
            os.makedirs(folder, exist_ok=True)
            np.savez_compressed(os.path.join(folder, "hotstage_mesh.npz"), **self.mesh)

    def sample(self):
        import torch
        p = np.full(self.n, np.nan, np.float32)
        T = np.full(self.n, np.nan, np.float32)
        th = self.thermo
        with torch.no_grad():
            for g, C, Wt, idx, _ in self.gather:
                if g.backend == "warp":
                    g._wprim("U")
                    W = g.W_buf
                else:
                    W = g._prim(g.U)
                gat = [(W[c].reshape(-1)[C] * Wt).sum(1) for c in range(5)]
                rho, u, v_, w, pp = gat
                Tm = pp / (rho.clamp_min(1e-9) * R_AIR)
                Tc = exhaust_temperature(Tm, u * u + v_ * v_ + w * w, **th)
                Tc = torch.where(rho >= th["rho_min"], Tc, torch.full_like(Tc, float("nan")))
                ii = idx.cpu().numpy()
                p[ii] = pp.float().cpu().numpy()
                T[ii] = Tc.float().cpu().numpy()
        return p, T

    def record(self, step, t):
        """Sample, update peaks and the per-part history; returns the snapshot fields."""
        p, T = self.sample()
        with np.errstate(invalid="ignore"):
            self.peak_p = np.fmax(self.peak_p, p)
            self.peak_T = np.fmax(self.peak_T, T)
        row = dict(step=int(step), t=float(t))
        for k, name in enumerate(self.names + ["plates"]):
            if name == "plates":
                sel = np.zeros(self.n, bool)
                for j, nm in enumerate(self.names):
                    if nm.startswith("plate"):
                        sel[self.split[j]:self.split[j + 1]] = True
            else:
                sel = np.zeros(self.n, bool)
                sel[self.split[k]:self.split[k + 1]] = True
            sel &= np.isfinite(p) & (self.weight > 0)
            if not sel.any():
                continue
            w = self.weight[sel]
            r = dict(p_max=float(np.percentile(p[sel], 99.5)), p_mean=float((p[sel] * w).sum() / w.sum()))
            st = sel & np.isfinite(T)
            if st.any():
                wt = self.weight[st]
                # mean over the whole part: thin-gas points count at the ambient temperature
                Tf = np.where(np.isfinite(T), T, self.thermo.get("T_inf", 230.0))[sel]
                r.update(T_max=float(np.percentile(T[st], 99.5)), T_mean=float((Tf * w).sum() / w.sum()))
            row[name] = r
        self.history.append(row)
        return dict(hs_p=p, hs_T=T.astype(np.float16))

    def save(self, folder, meta=None):
        np.savez_compressed(os.path.join(folder, "hotstage_peaks.npz"), peak_p=self.peak_p, peak_T=self.peak_T, **self.mesh)
        with open(os.path.join(folder, "hotstage.json"), "w") as fh:
            json.dump(dict(meta=meta or {}, parts=self.names, history=self.history), fh, indent=1, default=float)
        self.write_vtk(folder)

    def write_vtk(self, folder):
        """hotstage_<part>.vtk surfaces with the peak pressure and temperature (ParaView)."""
        m = self.mesh
        for k, name in enumerate(self.names):
            a, b = m["split"][k], m["split"][k + 1]
            fa, fb = m["faces_split"][k], m["faces_split"][k + 1]
            v, f = m["v"][a:b], m["f"][fa:fb]
            fn = os.path.join(folder, "hotstage_" + name.replace(" ", "_") + ".vtk")
            with open(fn, "w") as fh:
                fh.write(f"# vtk DataFile Version 3.0\nhot staging {name}\nASCII\nDATASET POLYDATA\nPOINTS {len(v)} float\n")
                np.savetxt(fh, v, fmt="%.4f")
                fh.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
                np.savetxt(fh, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
                fh.write(f"POINT_DATA {len(v)}\n")
                for nm, arr in (("peak_pressure_Pa", self.peak_p[a:b]), ("peak_gas_temperature_K", self.peak_T[a:b])):
                    fh.write(f"SCALARS {nm} float 1\nLOOKUP_TABLE default\n")
                    np.savetxt(fh, np.nan_to_num(arr, nan=0.0), fmt="%.5g")


def load_mesh(folder):
    for fn in ("hotstage_peaks.npz", "snapshots/hotstage_mesh.npz", "hotstage_mesh.npz"):
        path = os.path.join(folder, fn)
        if os.path.exists(path):
            d = dict(np.load(path, allow_pickle=False))
            d["names"] = [str(x) for x in d["names"]]
            return d
    raise FileNotFoundError(f"no hot-stage surface data in {folder}")


def part_slices(m):
    """[(name, vertex slice, face array with vertex indices local to the part)]."""
    out = []
    for k, name in enumerate(m["names"]):
        a, b = int(m["split"][k]), int(m["split"][k + 1])
        fa, fb = int(m["faces_split"][k]), int(m["faces_split"][k + 1])
        out.append((name, slice(a, b), m["f"][fa:fb]))
    return out


# ---------------------------------------------------------------- report
def _p995(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 99.5)) if a.size else float("nan")


def report(folder, out=None):
    """Time histories and peak maps of pressure and gas temperature on the hot-stage parts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = out or os.path.join(folder, "hotstage_report")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(folder, "hotstage.json")) as fh:
        H = json.load(fh)
    hist = H["history"]
    m = load_mesh(folder)
    pk = np.load(os.path.join(folder, "hotstage_peaks.npz"))
    peak_p, peak_T = pk["peak_p"], pk["peak_T"]
    t = np.array([r["t"] for r in hist]) * 1e3

    # ---- style (dataviz skill: one restrained system, categorical slots in fixed order)
    INK, MUTED, GRID, BG = "#191919", "#666666", "#e8e8e8", "#ffffff"
    CAT = ["#257de0", "#f47230", "#00b894", "#f2a600"]          # reference categorical slots 1-4, fixed order
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": BG,
                         "axes.facecolor": BG, "legend.frameon": False, "axes.titleweight": "bold",
                         "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titlelocation": "left"})
    series = [("dome", "Bare dome (between plates)"), ("plates", "Protection plates"), ("truss", "Truss"),
              ("ship aft", "Ship aft end")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, (q, scale, unit, lab) in enumerate((("p", 1e-3, "kPa", "Pressure"), ("T", 1.0, "K", "Gas temperature"))):
        for row, stat in enumerate(("max", "mean")):
            ax = axs[row, col]
            for i, (key, label) in enumerate(series):
                y = np.array([r.get(key, {}).get(f"{q}_{stat}", np.nan) for r in hist]) * scale
                if np.all(np.isnan(y)):
                    continue
                ax.plot(t, y, color=CAT[i], lw=1.8, label=label)
                if np.isfinite(y[-1]):
                    ax.annotate(label, (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=MUTED,
                                fontsize=8.5, va="center")
            ax.set_title(f"{lab}, {'peak' if stat == 'max' else 'area-weighted mean'} [{unit}]")
            if row == 1:
                ax.set_xlabel("time since ship engine start [ms]")
            ax.set_xlim(t[0], t[0] + (t[-1] - t[0]) * 1.3)
    axs[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Hot staging: loads on the booster's forward dome, plates, truss and the ship's aft end",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hotstage_history.png"), dpi=160)
    plt.close(fig)

    # full thrust: per-vertex time mean over the snapshots after the ignition ramp (the start-up
    # shock sets the whole-run peaks; the steady impingement is what the plates live with)
    t_ign = float(H.get("meta", {}).get("ignition", 0.0))
    snaps = sorted(glob.glob(os.path.join(folder, "snapshots", "snap_*.npz")))
    acc_p, acc_T, nT, n_s = None, None, None, 0
    for fn in snaps:
        with np.load(fn) as d:
            if "hs_p" not in d or float(d["t"]) < t_ign:
                continue
            p_, T_ = d["hs_p"].astype(np.float64), d["hs_T"].astype(np.float64)
        if acc_p is None:
            acc_p, acc_T, nT = np.zeros_like(p_), np.zeros_like(T_), np.zeros_like(T_)
        acc_p += np.nan_to_num(p_)
        okT = np.isfinite(T_)
        acc_T[okT] += T_[okT]
        nT += okT
        n_s += 1
    mean_p = acc_p / n_s if n_s else None
    mean_T = np.where(nT > 0, acc_T / np.maximum(nT, 1), np.nan) if n_s else None

    # per plate: peak over the run, and the full-thrust mean
    names = m["names"]
    plates = [(i, n) for i, n in enumerate(names) if n.startswith("plate")]
    rows = []
    for i, n in plates + [(names.index("dome"), "dome")]:
        sl = slice(int(m["split"][i]), int(m["split"][i + 1]))
        w = m["weight"][sl] > 0
        ok = w & np.isfinite(peak_p[sl])
        if not ok.any():
            continue
        lab = ("centre plate" if n == "plate 0" else n) if n != "dome" else "bare dome"
        mp_ = _p995(mean_p[sl][ok]) if mean_p is not None else np.nan
        mT_ = _p995(mean_T[sl][ok]) if mean_T is not None else np.nan
        rows.append((lab, _p995(peak_p[sl][ok]), _p995(peak_T[sl][ok]), mp_, mT_))
    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(11, 0.5 * len(rows) + 1.8))
        lab = [r[0] for r in rows]
        yy = np.arange(len(rows))[::-1].astype(float)
        both = mean_p is not None
        for ax, k, sc, unit, title in ((axs[0], 1, 1e-3, "kPa", "Pressure"), (axs[1], 2, 1.0, "K", "Gas temperature")):
            series = [(k, "highest during the run (start-up included)", CAT[0])]
            if both:
                series.append((k + 2, f"full thrust, time mean ({n_s} snapshots)", CAT[1]))
            hb = 0.36 if both else 0.62
            vmax = 0.0
            for si, (col, sname, colr) in enumerate(series):
                vals = np.array([r[col] for r in rows]) * sc
                off = (0.5 - si) * (hb + 0.04) if both else 0.0
                ax.barh(yy + off, vals, color=colr, height=hb, label=sname)
                for y_, v_ in zip(yy + off, vals):
                    if np.isfinite(v_):
                        ax.text(v_, y_, f" {v_:,.0f}", va="center", fontsize=8, color=INK)
                vmax = max(vmax, np.nanmax(vals))
            ax.set_yticks(yy, lab)
            ax.set_title(f"{title} [{unit}], 99.5th percentile over each plate")
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, vmax * 1.2)
        h_, l_ = axs[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.legend(h_, l_, loc="upper left", ncol=2, fontsize=8.5, bbox_to_anchor=(0.01, 1.0))
        fig.savefig(os.path.join(out, "hotstage_plates.png"), dpi=160)
        plt.close(fig)

    # top-down peak maps of the dome and plates (looking down the axis from the ship)
    sel = [i for i, n in enumerate(names) if n == "dome" or n.startswith("plate")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, arr, sc, cmap, unit, title in ((axs[0], peak_p, 1e-3, "viridis", "kPa", "Peak pressure"),
                                          (axs[1], peak_T, 1.0, "inferno", "K", "Peak gas temperature")):
        lo_, hi_ = np.inf, -np.inf
        for i in sel:
            sl = slice(int(m["split"][i]), int(m["split"][i + 1]))
            a_ = arr[sl] * sc
            if np.isfinite(a_).any():
                lo_, hi_ = min(lo_, np.nanpercentile(a_, 1)), max(hi_, np.nanpercentile(a_, 99.5))
        for i in sel:
            sl = slice(int(m["split"][i]), int(m["split"][i + 1]))
            fa, fb = int(m["faces_split"][i]), int(m["faces_split"][i + 1])
            v, f = m["v"][sl], m["f"][fa:fb]
            up = m["n"][sl][:, 0] > 0.0                      # upward-facing (toward the ship)
            keep = up[f].all(1)
            if not keep.any():
                continue
            val = np.nan_to_num(arr[sl] * sc, nan=lo_)
            tri = ax.tripcolor(v[:, 1], v[:, 2], f[keep], val, cmap=cmap, vmin=lo_, vmax=hi_, shading="gouraud",
                               zorder=2 if names[i].startswith("plate") else 1)
        from .vehicles.starship import dome_plate_list, StarshipParams
        for cy, cz, rp in dome_plate_list(StarshipParams()):
            ax.add_patch(plt.Circle((cy, cz), rp, fill=False, color="white", lw=0.9, zorder=3))
        ax.set_aspect("equal")
        ax.set_title(f"{title} [{unit}] — dome and plates, from above")
        ax.set_xlabel("y [m]")
        ax.grid(False)
        fig.colorbar(tri, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hotstage_dome_map.png"), dpi=160)
    plt.close(fig)

    # summary
    summ = {}
    for i, n in enumerate(names):
        sl = slice(int(m["split"][i]), int(m["split"][i + 1]))
        ok = (m["weight"][sl] > 0) & np.isfinite(peak_p[sl])
        if ok.any():
            summ[n] = dict(peak_p_kPa=round(_p995(peak_p[sl][ok]) / 1e3, 1),
                           peak_T_K=round(_p995(peak_T[sl][ok])), points=int(ok.sum()))
            if mean_p is not None:
                summ[n].update(full_thrust_p_kPa=round(_p995(mean_p[sl][ok]) / 1e3, 1),
                               full_thrust_T_K=round(_p995(mean_T[sl][ok])) if np.isfinite(_p995(mean_T[sl][ok])) else None)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    for n, s in summ.items():
        ft = (f"   full thrust: {s['full_thrust_p_kPa']:8.1f} kPa, {s['full_thrust_T_K'] or 0:5d} K"
              if "full_thrust_p_kPa" in s else "")
        print(f"  {n:12s} peak p {s['peak_p_kPa']:9.1f} kPa   peak gas T {s['peak_T_K']:6d} K{ft}")
    print(f"wrote {out}: hotstage_history.png, hotstage_plates.png, hotstage_dome_map.png, summary.json")
    return summ


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["report"])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    report(a.folder, a.out)


if __name__ == "__main__":
    main()
