"""Plots page: publication-style 2-D plots of a results folder - convergence history, force
breakdown by part, filled contour plots of any field on any plane, line probes, and the aero
database curves (coefficients vs angle of attack / Mach)."""
import csv
import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from .common import RESULTS, mpl_canvas, read_json, result_folders, style_axes

FIELDS = ["mach", "Cp", "p", "rho", "T", "schlieren", "vorticity", "Q", "p0_ratio", "speed"]


class PlotsPage(QtWidgets.QWidget):
    def __init__(self, results_page=None):
        super().__init__()
        self.results_page = results_page        # shares the loaded 3-D data
        self.folder = None
        self.meta = {}
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Plots")
        t.setObjectName("title")
        top.addWidget(t)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Results folder:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(320)
        self.combo.activated.connect(lambda *_: self.load(self.combo.currentData()))
        top.addWidget(self.combo)
        b = QtWidgets.QPushButton("Refresh")
        b.clicked.connect(self.refresh)
        top.addWidget(b)
        top.addStretch(1)
        self.dark = QtWidgets.QCheckBox("dark plots")
        self.dark.setChecked(True)
        self.dark.toggled.connect(lambda *_: self.redraw())
        top.addWidget(self.dark)
        b = QtWidgets.QPushButton("Save plot...")
        b.clicked.connect(self.save)
        top.addWidget(b)
        lay.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.canvases = {}
        for name, builder in (("Convergence", self._tab_conv), ("Forces by part", self._tab_parts),
                              ("Contour plot", self._tab_contour), ("Line probe", self._tab_line),
                              ("Aero database", self._tab_db)):
            w = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(w)
            ctl = builder()
            cw, fig, canvas = mpl_canvas()
            self.canvases[name] = (fig, canvas)
            if ctl is not None:
                box = QtWidgets.QWidget()
                box.setLayout(ctl)
                box.setMaximumWidth(300)
                hl.addWidget(box)
            hl.addWidget(cw, 1)
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(lambda *_: self.redraw())
        self.refresh()

    # ------------------------------------------------------------------ controls
    def _tab_conv(self):
        return None

    def _tab_parts(self):
        lay = QtWidgets.QFormLayout()
        self.part_coef = QtWidgets.QComboBox()
        self.part_coef.addItems(["CD", "CN", "C_restore"])
        self.part_coef.currentTextChanged.connect(lambda *_: self.redraw())
        lay.addRow("Coefficient", self.part_coef)
        return lay

    def _tab_contour(self):
        lay = QtWidgets.QFormLayout()
        self.c_field = QtWidgets.QComboBox()
        self.c_field.addItems(FIELDS)
        self.c_field.setEditable(True)
        lay.addRow("Field", self.c_field)
        self.c_axis = QtWidgets.QComboBox()
        self.c_axis.addItems(["x-z plane (normal y)", "x-y plane (normal z)", "y-z plane (normal x)"])
        lay.addRow("Plane", self.c_axis)
        self.c_pos = QtWidgets.QDoubleSpinBox()
        self.c_pos.setRange(-1e6, 1e6)
        self.c_pos.setDecimals(3)
        lay.addRow("Plane position [m]", self.c_pos)
        self.c_levels = QtWidgets.QSpinBox()
        self.c_levels.setRange(4, 200)
        self.c_levels.setValue(40)
        lay.addRow("Colour levels", self.c_levels)
        self.c_lines = QtWidgets.QSpinBox()
        self.c_lines.setRange(0, 100)
        self.c_lines.setValue(12)
        lay.addRow("Contour lines", self.c_lines)
        self.c_cmap = QtWidgets.QComboBox()
        self.c_cmap.addItems(["turbo", "viridis", "plasma", "inferno", "coolwarm", "RdBu_r", "jet", "gray", "bone"])
        lay.addRow("Colour map", self.c_cmap)
        self.c_zoom = QtWidgets.QDoubleSpinBox()
        self.c_zoom.setRange(0.05, 1.0)
        self.c_zoom.setSingleStep(0.05)
        self.c_zoom.setValue(1.0)
        self.c_zoom.setToolTip("Fraction of the domain shown around the body")
        lay.addRow("Zoom", self.c_zoom)
        self.c_stream = QtWidgets.QCheckBox("in-plane streamlines")
        lay.addRow(self.c_stream)
        b = QtWidgets.QPushButton("Draw")
        b.setObjectName("primary")
        b.clicked.connect(self.redraw)
        lay.addRow(b)
        return lay

    def _tab_line(self):
        lay = QtWidgets.QFormLayout()
        self.l_field = QtWidgets.QComboBox()
        self.l_field.addItems(FIELDS)
        self.l_field.setEditable(True)
        lay.addRow("Field", self.l_field)
        self.l_a = QtWidgets.QLineEdit("-10 0 0")
        self.l_b = QtWidgets.QLineEdit("90 0 0")
        lay.addRow("From x y z [m]", self.l_a)
        lay.addRow("To x y z [m]", self.l_b)
        self.l_n = QtWidgets.QSpinBox()
        self.l_n.setRange(10, 20000)
        self.l_n.setValue(800)
        lay.addRow("Points", self.l_n)
        b = QtWidgets.QPushButton("Draw")
        b.setObjectName("primary")
        b.clicked.connect(self.redraw)
        lay.addRow(b)
        self.l_export = QtWidgets.QPushButton("Export CSV...")
        self.l_export.clicked.connect(self._export_line)
        lay.addRow(self.l_export)
        return lay

    def _tab_db(self):
        lay = QtWidgets.QFormLayout()
        self.db_y = QtWidgets.QComboBox()
        self.db_y.setEditable(True)
        lay.addRow("Coefficient", self.db_y)
        self.db_x = QtWidgets.QComboBox()
        self.db_x.addItems(["aoa", "mach"])
        lay.addRow("Horizontal axis", self.db_x)
        self.db_roll = QtWidgets.QComboBox()
        lay.addRow("Roll", self.db_roll)
        for w in (self.db_y, self.db_x, self.db_roll):
            w.currentTextChanged.connect(lambda *_: self.redraw())
        return lay

    # ------------------------------------------------------------------ data
    def refresh(self):
        cur = self.combo.currentData()
        self.combo.clear()
        for p in result_folders():
            self.combo.addItem(os.path.relpath(p, RESULTS), p)
        if cur:
            i = self.combo.findData(cur)
            if i >= 0:
                self.combo.setCurrentIndex(i)
        if self.combo.count() and self.folder is None:
            self.load(self.combo.currentData())

    def load(self, folder):
        if not folder:
            return
        self.folder = folder
        i = self.combo.findData(folder)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self.meta = read_json(os.path.join(folder, "loads.json")) or {}
        self.rows = []
        cp = os.path.join(folder, "cases.csv")
        if os.path.exists(cp):
            with open(cp) as fh:
                for r in csv.DictReader(fh):
                    rr = {}
                    for k, v in r.items():
                        try:
                            rr[k] = float(v)
                        except (TypeError, ValueError):
                            rr[k] = v
                    self.rows.append(rr)
        if self.rows:
            num = [k for k, v in self.rows[0].items() if isinstance(v, float) and k not in ("mach", "aoa", "roll")]
            self.db_y.blockSignals(True)
            self.db_y.clear()
            self.db_y.addItems(num)
            for pref in ("CN", "C_restore", "CD"):
                if pref in num:
                    self.db_y.setCurrentText(pref)
                    break
            self.db_y.blockSignals(False)
            self.db_roll.blockSignals(True)
            self.db_roll.clear()
            self.db_roll.addItems([f"{r:g}" for r in sorted({r.get("roll", 0.0) for r in self.rows})])
            self.db_roll.blockSignals(False)
        self.redraw()

    def _flow(self):
        rp = self.results_page
        if rp is not None and rp.scene is not None and rp.scene.folder == self.folder and rp.scene.flow is not None:
            return rp.scene
        from .results_page import Scene
        if not os.path.exists(os.path.join(self.folder, "flow.vtk")):
            return None
        sc = Scene(self.folder)
        if rp is not None:
            rp.scene_cache = sc
        return sc

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        name = self.tabs.tabText(self.tabs.currentIndex())
        fig, canvas = self.canvases[name]
        dark = self.dark.isChecked()
        fig.clear()
        fig.set_facecolor("#1e2228" if dark else "white")
        try:
            getattr(self, "_draw_" + {"Convergence": "conv", "Forces by part": "parts", "Contour plot": "contour",
                                      "Line probe": "line", "Aero database": "db"}[name])(fig, dark)
        except Exception as e:                  # noqa: BLE001
            ax = fig.add_subplot(111)
            style_axes(ax, dark)
            ax.text(0.5, 0.5, f"cannot draw: {e}", ha="center", va="center", transform=ax.transAxes,
                    color="#d8dee9" if dark else "black")
        canvas.draw_idle()

    def _msg(self, fig, dark, text):
        ax = fig.add_subplot(111)
        style_axes(ax, dark)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#d8dee9" if dark else "black")
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_conv(self, fig, dark):
        h = self.meta.get("history")
        if not h:
            return self._msg(fig, dark, "No load history (loads.json) in this folder.")
        h = np.asarray(h, float)
        names = ["CD", "CN", "C_restore"]
        for i, n in enumerate(names):
            ax = style_axes(fig.add_subplot(1, 3, i + 1), dark)
            ax.plot(h[:, 0], h[:, 2 + i], "-o", ms=3, color=("#ebcb8b", "#a3be8c", "#b48ead")[i])
            av = (self.meta.get("averaged") or {}).get(n)
            if av is not None:
                ax.axhline(av, color="#bf616a", ls="--", lw=1, label=f"average {av:.4f}")
                ax.legend(fontsize=8)
            ax.set_title(n)
            ax.set_xlabel("iteration / step")
        fig.tight_layout()

    def _draw_parts(self, fig, dark):
        av = self.meta.get("averaged") or {}
        k = self.part_coef.currentText()
        parts = {key[len(k) + 1:-1]: v for key, v in av.items() if key.startswith(k + "[")}
        if not parts:
            return self._msg(fig, dark, "No per-part breakdown in loads.json.")
        sd = self.meta.get("averaged_std") or {}
        ax = style_axes(fig.add_subplot(111), dark)
        names = list(parts)
        vals = [parts[n] for n in names]
        errs = [sd.get(f"{k}[{n}]", 0.0) for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=errs, color="#5e81ac", ecolor="#d8dee9" if dark else "black", capsize=3)
        ax.set_yticks(y, names)
        ax.set_xlabel(f"{k} contribution (bars: average, whiskers: unsteadiness)")
        total = av.get(k)
        if total is not None:
            ax.set_title(f"{k} total {total:.4f}")
        fig.tight_layout()

    def _plane(self, sc, field):
        """2-D arrays (a, b, values) of the field on the chosen plane, nearest grid plane."""
        from .. import derived
        f, x, y, z = derived.grid_arrays(sc.flow)
        if field not in f:
            sc.ensure(field)
            f, x, y, z = derived.grid_arrays(sc.flow)
        ax_i = self.c_axis.currentIndex()
        d = (1, 2, 0)[ax_i]
        coords = (x, y, z)
        k = int(np.argmin(np.abs(coords[d] - self.c_pos.value())))
        take = lambda a: np.take(a, k, axis=d)
        inpl = [e for e in range(3) if e != d]
        return coords[inpl[0]], coords[inpl[1]], take(f[field]), {c: take(f[c]) for c in ("u", "v", "w", "phi") if c in f}, inpl

    def _draw_contour(self, fig, dark):
        if not self.folder:
            return self._msg(fig, dark, "Choose a results folder.")
        sc = self._flow()
        if sc is None:
            return self._msg(fig, dark, "No flow.vtk in this folder.")
        field = self.c_field.currentText()
        a, b, v, extra, inpl = self._plane(sc, field)
        ax = style_axes(fig.add_subplot(111), dark)
        ax.grid(False)
        v = np.ma.masked_invalid(v)
        if "phi" in extra:
            v = np.ma.masked_where(extra["phi"] < 0, v)
        lo, hi = np.nanpercentile(v.compressed(), [0.5, 99.5]) if v.count() else (0, 1)
        if field == "schlieren":
            lo, hi = 0.0, 1.0
        lev = np.linspace(lo, hi if hi > lo else lo + 1, self.c_levels.value())
        cs = ax.contourf(a, b, v.T, levels=lev, cmap=self.c_cmap.currentText(), extend="both")
        if self.c_lines.value():
            ax.contour(a, b, v.T, levels=np.linspace(lo, hi, self.c_lines.value() + 2)[1:-1], colors="k",
                       linewidths=0.5, alpha=0.6)
        if "phi" in extra:
            ax.contourf(a, b, extra["phi"].T, levels=[-1e9, 0], colors=["#3b4252"])
            ax.contour(a, b, extra["phi"].T, levels=[0], colors=["#d8dee9" if dark else "black"], linewidths=0.8)
        if self.c_stream.isChecked():
            comp = "uvw"
            ua, ub = extra[comp[inpl[0]]], extra[comp[inpl[1]]]
            ag, bg = np.linspace(a[0], a[-1], len(a)), np.linspace(b[0], b[-1], len(b))
            from scipy.interpolate import RegularGridInterpolator
            A, B = np.meshgrid(ag, bg, indexing="ij")
            ui = RegularGridInterpolator((a, b), ua)(np.stack([A, B], -1))
            vi = RegularGridInterpolator((a, b), ub)(np.stack([A, B], -1))
            ax.streamplot(ag, bg, ui.T, vi.T, color="white", linewidth=0.5, density=1.6, arrowsize=0.6)
        bb = sc.body_bounds()
        z = self.c_zoom.value()
        if z < 1.0:
            ca, cb = (bb[2 * inpl[0]] + bb[2 * inpl[0] + 1]) / 2, (bb[2 * inpl[1]] + bb[2 * inpl[1] + 1]) / 2
            ha, hb = z * (a[-1] - a[0]) / 2, z * (b[-1] - b[0]) / 2
            ax.set_xlim(ca - ha, ca + ha)
            ax.set_ylim(cb - hb, cb + hb)
        ax.set_aspect("equal")
        cb_ = fig.colorbar(cs, ax=ax, shrink=0.85)
        cb_.set_label(field, color="#d8dee9" if dark else "black")
        if dark:
            cb_.ax.tick_params(colors="#c8ccd4")
        ax.set_xlabel("xyz"[inpl[0]] + " [m]")
        ax.set_ylabel("xyz"[inpl[1]] + " [m]")
        ax.set_title(f"{field} on {'xyz'[inpl[0]]}-{'xyz'[inpl[1]]} plane at {'xyz'[(1, 2, 0)[self.c_axis.currentIndex()]]} = "
                     f"{self.c_pos.value():g} m")
        fig.tight_layout()

    def _line_data(self):
        sc = self._flow()
        if sc is None:
            return None
        field = self.l_field.currentText()
        sc.ensure(field)
        a = [float(t) for t in self.l_a.text().split()]
        b = [float(t) for t in self.l_b.text().split()]
        line = sc.flow.sample_over_line(a, b, resolution=self.l_n.value() - 1)
        dist = np.linalg.norm(np.asarray(line.points) - np.asarray(a), axis=1)
        val = np.asarray(line.point_data[field], float)
        if "vtkValidPointMask" in line.point_data:
            val[np.asarray(line.point_data["vtkValidPointMask"]) == 0] = np.nan
        if "phi" in line.point_data:
            val[np.asarray(line.point_data["phi"]) < 0] = np.nan
        return dist, val, field

    def _draw_line(self, fig, dark):
        if not self.folder:
            return self._msg(fig, dark, "Choose a results folder.")
        d = self._line_data()
        if d is None:
            return self._msg(fig, dark, "No flow.vtk in this folder.")
        dist, val, field = d
        ax = style_axes(fig.add_subplot(111), dark)
        ax.plot(dist, val, color="#88c0d0", lw=1.4)
        ax.set_xlabel("distance along the line [m]")
        ax.set_ylabel(field)
        ax.set_title(f"{field} from ({self.l_a.text()}) to ({self.l_b.text()})")
        fig.tight_layout()

    def _export_line(self):
        d = self._line_data()
        if d is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export line", os.path.join(self.folder, "line.csv"),
                                                        "CSV (*.csv)")
        if path:
            np.savetxt(path, np.column_stack(d[:2]), delimiter=",", header=f"distance_m,{d[2]}", comments="")

    def _draw_db(self, fig, dark):
        if not self.rows:
            return self._msg(fig, dark, "No cases.csv here: choose an aero database results folder.")
        yk = self.db_y.currentText()
        xk = self.db_x.currentText()
        gk = "mach" if xk == "aoa" else "aoa"
        roll = float(self.db_roll.currentText() or 0)
        ax = style_axes(fig.add_subplot(111), dark)
        groups = sorted({r[gk] for r in self.rows})
        import matplotlib.cm as cm
        for i, gv in enumerate(groups):
            pts = sorted((r[xk], r[yk]) for r in self.rows if r[gk] == gv and isinstance(r.get(yk), float)
                         and (abs(r.get("roll", 0) - roll) < 1e-6 or r.get("aoa", 1) == 0))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", ms=4, color=cm.turbo(i / max(len(groups) - 1, 1)), label=f"{gk} {gv:g}")
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(f"{yk} at roll {roll:g} deg")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()

    def save(self):
        name = self.tabs.tabText(self.tabs.currentIndex())
        fig, _ = self.canvases[name]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", os.path.join(
            self.folder or RESULTS, name.replace(" ", "_").lower() + ".png"), "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
