"""3-D results page: an interactive VTK view of a results folder with layers that can each be
switched on and styled: body surface (Cp), a cutting plane with contours, iso-surfaces
(Q-criterion vortices, sonic surface, ...), streamlines, volume rendering, and a one-click
"cinematic" look (dark background, schlieren plane, vortex surfaces coloured by Mach)."""
import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import derived
from .common import RESULTS, hint, read_json, result_folders

CMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet", "gray", "bone",
         "cividis", "Spectral_r"]
DERIVED = ["schlieren", "grad_rho", "Q", "vorticity", "T", "Cp", "p0_ratio", "speed", "lambda2"]


def _freestream(meta):
    c = (meta or {}).get("condition") or {}
    if "rho" in c and "V" in c and "p" in c:
        return dict(rho=c["rho"], p=c["p"], u=c["V"], v=0.0, w=0.0)
    return None


class Scene:
    """Data + PyVista actors for one results folder."""

    def __init__(self, folder):
        import pyvista as pv
        self.pv = pv
        self.folder = folder
        self.meta = read_json(os.path.join(folder, "loads.json")) or {}
        self.surface = None
        from ..flowdata import FlowData
        self.fd = FlowData(folder, _freestream(self.meta))       # base grid + refinement patches
        self.flow = self.fd.base
        self.patches = self.fd.patches
        spath = os.path.join(folder, "surface_cp.vtk")
        if os.path.exists(spath):
            self.surface = pv.read(spath)
        elif self.flow is not None and "phi" in self.flow.point_data:
            try:
                self.surface = self.flow.contour([0.0], scalars="phi")
            except Exception:                   # noqa: BLE001
                self.surface = None
        self.fs = _freestream(self.meta)

    def fields(self):
        if self.flow is None:
            return []
        names = [n for n in self.flow.point_data if n not in ("velocity",)]
        return names + [d for d in DERIVED if d not in names]

    def ensure(self, name):
        """Make sure the field exists on the flow grid (computing derived fields on demand)."""
        if self.flow is None:
            return False
        if name in self.flow.point_data and all(name in g.point_data for g in self.patches):
            return True
        if name in DERIVED or name in self.flow.point_data:
            return self.fd.ensure(name)
        return False

    def bounds(self):
        if self.flow is not None:
            return self.flow.bounds
        if self.surface is not None:
            return self.surface.bounds
        return (0, 1, 0, 1, 0, 1)

    def body_bounds(self):
        if self.surface is not None and self.surface.n_points:
            return self.surface.bounds
        return self.bounds()


class ResultsPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.scene = None
        self.actors = {}
        self.plotter = None
        lay = QtWidgets.QHBoxLayout(self)

        # ---- left: folder + summary
        left = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("3-D results")
        t.setObjectName("title")
        left.addWidget(t)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QComboBox()
        self.folder.setMinimumWidth(260)
        self.folder.activated.connect(lambda *_: self.load(self.folder.currentData()))
        row.addWidget(self.folder, 1)
        b = QtWidgets.QPushButton("Browse...")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        left.addLayout(row)
        b = QtWidgets.QPushButton("Refresh list")
        b.clicked.connect(self.refresh_folders)
        left.addWidget(b)
        self.summary = QtWidgets.QTextBrowser()
        self.summary.setMaximumWidth(330)
        left.addWidget(self.summary, 1)
        lay.addLayout(left, 0)

        # ---- centre: 3-D view
        self.view_host = QtWidgets.QVBoxLayout()
        try:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
            self.view_host.addWidget(self.plotter.interactor, 1)
            self.plotter.set_background("#14171c", top="#2b3240")
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception as e:                  # noqa: BLE001
            lab = QtWidgets.QLabel(f"3-D view unavailable: {e}\n(install pyvistaqt and a working OpenGL driver)")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            self.view_host.addWidget(lab, 1)
        cams = QtWidgets.QHBoxLayout()
        for name in ("Side", "Top", "Engine end", "Top end", "Isometric", "Fit"):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self.camera(n))
            cams.addWidget(b)
        cams.addStretch(1)
        b = QtWidgets.QPushButton("Screenshot...")
        b.clicked.connect(self.screenshot)
        cams.addWidget(b)
        self.view_host.addLayout(cams)
        lay.addLayout(self.view_host, 1)

        # ---- right: layer controls
        ctl = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(ctl)
        b = QtWidgets.QPushButton("Cinematic look (LAVA style)")
        b.setObjectName("primary")
        b.setToolTip("Dark background, grey body, schlieren plane, vortex surfaces coloured by Mach.")
        b.clicked.connect(self.cinematic)
        cl.addWidget(b)

        g = QtWidgets.QGroupBox("Colour field")
        f = QtWidgets.QFormLayout(g)
        self.field = QtWidgets.QComboBox()
        self.field.currentTextChanged.connect(lambda *_: self._range_auto())
        f.addRow("Field", self.field)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        self.cmap.currentTextChanged.connect(lambda *_: self.rebuild())
        f.addRow("Colour map", self.cmap)
        self.auto = QtWidgets.QCheckBox("automatic range")
        self.auto.setChecked(True)
        self.auto.toggled.connect(lambda *_: self._range_auto())
        f.addRow(self.auto)
        self.vmin = QtWidgets.QDoubleSpinBox()
        self.vmax = QtWidgets.QDoubleSpinBox()
        for w in (self.vmin, self.vmax):
            w.setRange(-1e12, 1e12)
            w.setDecimals(4)
            w.editingFinished.connect(self._range_manual)
        f.addRow("min", self.vmin)
        f.addRow("max", self.vmax)
        self.bar = QtWidgets.QCheckBox("colour bar")
        self.bar.setChecked(True)
        self.bar.toggled.connect(lambda *_: self.rebuild())
        f.addRow(self.bar)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Body")
        f = QtWidgets.QFormLayout(g)
        self.body_on = QtWidgets.QCheckBox("show")
        self.body_on.setChecked(True)
        self.body_on.toggled.connect(lambda *_: self.rebuild("body"))
        f.addRow(self.body_on)
        self.body_col = QtWidgets.QComboBox()
        self.body_col.addItems(["surface pressure Cp", "metallic grey", "same as colour field"])
        self.body_col.currentTextChanged.connect(lambda *_: self.rebuild("body"))
        f.addRow("Colour", self.body_col)
        self.body_op = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.body_op.setRange(5, 100)
        self.body_op.setValue(100)
        self.body_op.valueChanged.connect(lambda *_: self.rebuild("body"))
        f.addRow("Opacity", self.body_op)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Cutting plane")
        f = QtWidgets.QFormLayout(g)
        self.sl_on = QtWidgets.QCheckBox("show")
        self.sl_on.setChecked(True)
        self.sl_on.toggled.connect(lambda *_: self.rebuild("slice"))
        f.addRow(self.sl_on)
        self.sl_field = QtWidgets.QComboBox()
        self.sl_field.setToolTip("Field shown on the plane (default: the colour field)")
        self.sl_field.currentTextChanged.connect(lambda *_: self.rebuild("slice"))
        f.addRow("Plane field", self.sl_field)
        self.sl_ax = QtWidgets.QComboBox()
        self.sl_ax.addItems(["y (side view plane x-z)", "z (plane x-y)", "x (cross-section)"])
        self.sl_ax.currentIndexChanged.connect(lambda *_: self.rebuild("slice"))
        f.addRow("Normal", self.sl_ax)
        self.sl_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_pos.setRange(0, 1000)
        self.sl_pos.setValue(500)
        self.sl_pos.valueChanged.connect(lambda *_: self._slice_later())
        f.addRow("Position", self.sl_pos)
        self.sl_lines = QtWidgets.QSpinBox()
        self.sl_lines.setRange(0, 60)
        self.sl_lines.setValue(0)
        self.sl_lines.setToolTip("Number of contour lines drawn on the plane (0 = none)")
        self.sl_lines.valueChanged.connect(lambda *_: self.rebuild("slice"))
        f.addRow("Contour lines", self.sl_lines)
        self.sl_vec = QtWidgets.QCheckBox("velocity arrows")
        self.sl_vec.toggled.connect(lambda *_: self.rebuild("slice"))
        f.addRow(self.sl_vec)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Iso-surface")
        f = QtWidgets.QFormLayout(g)
        self.iso_on = QtWidgets.QCheckBox("show")
        self.iso_on.toggled.connect(lambda *_: self.rebuild("iso"))
        f.addRow(self.iso_on)
        self.iso_f = QtWidgets.QComboBox()
        self.iso_f.currentTextChanged.connect(lambda *_: self._iso_default())
        f.addRow("Field", self.iso_f)
        self.iso_v = QtWidgets.QDoubleSpinBox()
        self.iso_v.setRange(-1e12, 1e12)
        self.iso_v.setDecimals(4)
        self.iso_v.editingFinished.connect(lambda: self.rebuild("iso"))
        f.addRow("Value", self.iso_v)
        self.iso_col = QtWidgets.QCheckBox("colour by colour field")
        self.iso_col.setChecked(True)
        self.iso_col.toggled.connect(lambda *_: self.rebuild("iso"))
        f.addRow(self.iso_col)
        self.iso_op = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_op.setRange(5, 100)
        self.iso_op.setValue(85)
        self.iso_op.valueChanged.connect(lambda *_: self.rebuild("iso"))
        f.addRow("Opacity", self.iso_op)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Streamlines")
        f = QtWidgets.QFormLayout(g)
        self.st_on = QtWidgets.QCheckBox("show")
        self.st_on.toggled.connect(lambda *_: self.rebuild("stream"))
        f.addRow(self.st_on)
        self.st_n = QtWidgets.QSpinBox()
        self.st_n.setRange(2, 60)
        self.st_n.setValue(9)
        self.st_n.valueChanged.connect(lambda *_: self.rebuild("stream"))
        f.addRow("Seeds per side", self.st_n)
        self.st_tube = QtWidgets.QCheckBox("tubes")
        self.st_tube.setChecked(True)
        self.st_tube.toggled.connect(lambda *_: self.rebuild("stream"))
        f.addRow(self.st_tube)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Volume rendering")
        f = QtWidgets.QFormLayout(g)
        self.vol_on = QtWidgets.QCheckBox("show")
        self.vol_on.toggled.connect(lambda *_: self.rebuild("volume"))
        f.addRow(self.vol_on)
        self.vol_f = QtWidgets.QComboBox()
        self.vol_f.setToolTip("grad_rho shows shocks as glowing sheets; vorticity shows wakes and vortices")
        self.vol_f.currentTextChanged.connect(lambda *_: self.rebuild("volume"))
        f.addRow("Field", self.vol_f)
        self.vol_op = QtWidgets.QComboBox()
        self.vol_op.addItems(["sigmoid", "linear", "geom", "sigmoid_r", "linear_r"])
        self.vol_op.currentTextChanged.connect(lambda *_: self.rebuild("volume"))
        f.addRow("Opacity curve", self.vol_op)
        cl.addWidget(g)

        g = QtWidgets.QGroupBox("Background")
        f = QtWidgets.QFormLayout(g)
        self.bg = QtWidgets.QComboBox()
        self.bg.addItems(["dark gradient", "black", "white", "light gradient"])
        self.bg.currentTextChanged.connect(self._background)
        f.addRow(self.bg)
        cl.addWidget(g)
        cl.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(ctl)
        sc.setMinimumWidth(310)
        lay.addWidget(sc, 0)

        self._slice_timer = QtCore.QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.timeout.connect(lambda: self.rebuild("slice"))
        self.refresh_folders()

    # ------------------------------------------------------------------ data
    def refresh_folders(self):
        cur = self.folder.currentData()
        self.folder.clear()
        for p in result_folders():
            self.folder.addItem(os.path.relpath(p, RESULTS), p)
        if cur:
            i = self.folder.findData(cur)
            if i >= 0:
                self.folder.setCurrentIndex(i)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Results folder", RESULTS)
        if d:
            self.load(d)

    def load(self, folder):
        if not folder:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.scene = Scene(folder)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        i = self.folder.findData(folder)
        if i < 0:
            self.folder.addItem(os.path.relpath(folder, RESULTS) if folder.startswith(RESULTS) else folder, folder)
            i = self.folder.count() - 1
        self.folder.setCurrentIndex(i)
        flds = self.scene.fields()
        for combo in (self.field, self.iso_f, self.sl_field, self.vol_f):
            combo.blockSignals(True)
            combo.clear()
            if combo is self.sl_field:
                combo.addItem("(colour field)")
            combo.addItems(flds + (["grad_rho"] if combo is self.vol_f and "grad_rho" not in flds else []))
            combo.blockSignals(False)
        self.vol_f.setCurrentText("grad_rho")
        if "mach" in flds:
            self.field.setCurrentText("mach")
        if "Q" in flds:
            self.iso_f.setCurrentText("Q")
        self._summary()
        self._range_auto(rebuild=False)
        self._iso_default(rebuild=False)
        self.rebuild()
        self.camera("Isometric")

    def _summary(self):
        m = self.scene.meta
        s = self.scene
        html = f"<h3>{os.path.basename(s.folder)}</h3>"
        c = m.get("condition") or {}
        if c:
            html += "<b>Condition</b><br>" + "<br>".join(
                f"{k}: {v:.4g}" for k, v in c.items() if isinstance(v, (int, float)))
        av = m.get("averaged") or m.get("coefficients") or {}
        if av:
            html += "<br><br><b>Coefficients</b>" + (" (averaged)" if "averaged" in m else "") + "<table>"
            for k, v in av.items():
                if "[" not in k:
                    sd = (m.get("averaged_std") or {}).get(k)
                    html += f"<tr><td>{k}</td><td align=right>{v:+.4f}</td><td>{'&plusmn;%.4f' % sd if sd else ''}</td></tr>"
            html += "</table>"
        gr = m.get("grid") or {}
        if gr:
            html += "<br><b>Run</b><br>" + "<br>".join(
                f"{k}: {gr[k]}" for k in ("solver", "cells", "steps", "wall_s", "gpu") if k in gr)
        files = sorted(os.listdir(s.folder))
        html += "<br><br><b>Files</b><br>" + "<br>".join(files[:40])
        if s.flow is None:
            html += "<br><br><i>No flow.vtk here: only the surface / summary can be shown.</i>"
        self.summary.setHtml(html)

    # ------------------------------------------------------------------ drawing
    def _range_auto(self, rebuild=True):
        if self.scene is None or not self.auto.isChecked():
            if rebuild:
                self.rebuild()
            return
        name = self.field.currentText()
        if self.scene.ensure(name):
            a = np.asarray(self.scene.flow.point_data[name])
            a = a[np.isfinite(a)]
            if a.size:
                lo, hi = np.percentile(a, [0.5, 99.5])
                if name == "schlieren":
                    lo, hi = 0.0, 1.0
                self.vmin.setValue(float(lo))
                self.vmax.setValue(float(hi if hi > lo else lo + 1))
        if rebuild:
            self.rebuild()

    def _range_manual(self):
        self.auto.setChecked(False)
        self.rebuild()

    def _iso_default(self, rebuild=True):
        if self.scene is None:
            return
        name = self.iso_f.currentText()
        defaults = dict(mach=1.0, Q=None, vorticity=None, lambda2=None, Cp=0.0, T=None)
        if self.scene.ensure(name):
            a = np.asarray(self.scene.flow.point_data[name])
            a = a[np.isfinite(a)]
            v = defaults.get(name)
            if v is None and a.size:
                if name == "Q":
                    v = float(np.percentile(a[a > 0], 97)) if np.any(a > 0) else 0.0
                elif name == "lambda2":
                    v = float(np.percentile(a[a < 0], 3)) if np.any(a < 0) else 0.0
                else:
                    v = float(np.percentile(a, 95))
            self.iso_v.setValue(v or 0.0)
        if rebuild and self.iso_on.isChecked():
            self.rebuild("iso")

    def _slice_later(self):
        self._slice_timer.start(120)

    def _clim(self):
        return [self.vmin.value(), self.vmax.value()]

    def _remove(self, key):
        for a in self.actors.pop(key, []):
            self.plotter.remove_actor(a, render=False)

    def rebuild(self, which=None):
        if self.plotter is None or self.scene is None:
            return
        layers = [which] if which else ["body", "slice", "iso", "stream", "volume"]
        for k in layers:
            self._remove(k)
            try:
                getattr(self, "_draw_" + k)()
            except Exception as e:              # noqa: BLE001
                print(f"[results] {k} layer: {e}")
        self.plotter.render()

    def _sargs(self, title):
        return dict(title=title, color="#d8dee9" if self.bg.currentText() in ("dark gradient", "black") else "black",
                    vertical=True, position_x=0.9, position_y=0.15, height=0.7, width=0.04,
                    title_font_size=12, label_font_size=10, fmt="%.3g")

    def _draw_body(self):
        s = self.scene
        if not self.body_on.isChecked() or s.surface is None or s.surface.n_points == 0:
            return
        op = self.body_op.value() / 100.0
        mode = self.body_col.currentText()
        acts = []
        if mode.startswith("surface") and "Cp" in s.surface.point_data:
            acts.append(self.plotter.add_mesh(s.surface, scalars="Cp", cmap="coolwarm", opacity=op, smooth_shading=True,
                                              show_scalar_bar=self.bar.isChecked(),
                                              scalar_bar_args=self._sargs("Cp (surface)"), render=False))
        elif mode.startswith("same") and s.flow is not None:
            name = self.field.currentText()
            body = s.fd.sample_onto(s.surface) if name not in s.surface.point_data else s.surface
            acts.append(self.plotter.add_mesh(body, scalars=name, cmap=self.cmap.currentText(), clim=self._clim(),
                                              opacity=op, smooth_shading=True, show_scalar_bar=False, render=False))
        else:
            acts.append(self.plotter.add_mesh(s.surface, color="#b8bec8", metallic=0.6, roughness=0.35, pbr=True,
                                              opacity=op, smooth_shading=True, render=False))
        self.actors["body"] = acts

    def _slice_geom(self):
        s = self.scene
        b = s.bounds()
        ax = self.sl_ax.currentIndex()            # 0: normal y, 1: normal z, 2: normal x
        d = (1, 2, 0)[ax]
        frac = self.sl_pos.value() / 1000.0
        bb = s.body_bounds()
        if d == 0:
            lo, hi = bb[0] - 0.1 * (bb[1] - bb[0]), bb[1] + 0.1 * (bb[1] - bb[0])
        else:
            lo, hi = b[2 * d], b[2 * d + 1]
        pos = lo + frac * (hi - lo)
        normal = [0, 0, 0]
        normal[d] = 1
        origin = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        origin[d] = pos
        return normal, origin

    def _field_style(self, name):
        """(cmap, clim) for a field: the colour-field settings, or its own automatic range."""
        if name == self.field.currentText():
            return self.cmap.currentText(), self._clim()
        if name == "schlieren":
            return "gray", [0.05, 1.0]
        a = np.asarray(self.scene.flow.point_data[name])
        a = a[np.isfinite(a)]
        lo, hi = (np.percentile(a, [0.5, 99.5]) if a.size else (0.0, 1.0))
        return self.cmap.currentText(), [float(lo), float(hi if hi > lo else lo + 1)]

    def _draw_slice(self):
        s = self.scene
        if not self.sl_on.isChecked() or s.flow is None:
            return
        name = self.sl_field.currentText()
        if name in ("", "(colour field)"):
            name = self.field.currentText()
        if not s.ensure(name):
            return
        cmap, clim = self._field_style(name)
        normal, origin = self._slice_geom()
        sl = s.fd.slice(normal, origin)       # finest grid in each region
        if "phi" in sl.point_data:
            sl = sl.threshold(0.0, scalars="phi") if np.any(np.asarray(sl.point_data["phi"]) < 0) else sl
        acts = [self.plotter.add_mesh(sl, scalars=name, cmap=cmap, clim=clim,
                                      show_scalar_bar=self.bar.isChecked() and name != "schlieren",
                                      scalar_bar_args=self._sargs(name), nan_opacity=0.0, render=False)]
        n = self.sl_lines.value()
        if n > 0:
            lines = sl.contour(np.linspace(*clim, n + 2)[1:-1], scalars=name)
            if lines.n_points:
                acts.append(self.plotter.add_mesh(lines, color="black" if name != "schlieren" else "white",
                                                  line_width=1.2, render=False))
        if self.sl_vec.isChecked() and "velocity" in sl.point_data:
            sub = sl.extract_points(np.arange(0, sl.n_points, max(1, sl.n_points // 1500)))
            vmax = float(np.max(np.linalg.norm(np.asarray(sub.point_data["velocity"]), axis=1))) or 1.0
            bb = s.body_bounds()
            scale = 0.04 * (bb[1] - bb[0]) / vmax
            arrows = sub.glyph(orient="velocity", scale=False, factor=scale * vmax)
            acts.append(self.plotter.add_mesh(arrows, color="white", render=False))
        self.actors["slice"] = acts

    def _draw_iso(self):
        s = self.scene
        if not self.iso_on.isChecked() or s.flow is None:
            return
        name = self.iso_f.currentText()
        if not s.ensure(name):
            return
        iso = s.fd.contour(self.iso_v.value(), name)
        if iso.n_points == 0:
            return
        if self.iso_col.isChecked():
            col = self.field.currentText()
            s.ensure(col)
            if col not in iso.point_data:
                iso = s.fd.sample_onto(iso)
            act = self.plotter.add_mesh(iso, scalars=col, cmap=self.cmap.currentText(), clim=self._clim(),
                                        opacity=self.iso_op.value() / 100.0, smooth_shading=True,
                                        show_scalar_bar=False, render=False)
        else:
            act = self.plotter.add_mesh(iso, color="#88c0d0", opacity=self.iso_op.value() / 100.0,
                                        smooth_shading=True, render=False)
        self.actors["iso"] = [act]

    def _draw_stream(self):
        s = self.scene
        if not self.st_on.isChecked() or s.flow is None or "velocity" not in s.flow.point_data:
            return
        name = self.field.currentText()
        s.ensure(name)
        bb = s.body_bounds()
        b = s.bounds()
        n = self.st_n.value()
        # a rake upstream of the body, perpendicular to the mean flow direction
        vel = np.asarray(s.flow.point_data["velocity"])
        vmean = vel[np.linalg.norm(vel, axis=1) > 0].mean(0)
        d = int(np.argmax(np.abs(vmean)))
        up = bb[2 * d] - 0.05 * (bb[2 * d + 1] - bb[2 * d]) if vmean[d] > 0 else bb[2 * d + 1] + 0.05 * (bb[2 * d + 1] - bb[2 * d])
        up = min(max(up, b[2 * d] + 1e-3), b[2 * d + 1] - 1e-3)
        others = [e for e in range(3) if e != d]
        span = [(bb[2 * e] - 0.3 * (bb[2 * e + 1] - bb[2 * e]), bb[2 * e + 1] + 0.3 * (bb[2 * e + 1] - bb[2 * e])) for e in others]
        g1, g2 = np.meshgrid(np.linspace(*span[0], n), np.linspace(*span[1], n), indexing="ij")
        pts = np.zeros((n * n, 3))
        pts[:, d] = up
        pts[:, others[0]] = g1.ravel()
        pts[:, others[1]] = g2.ravel()
        seeds = self.scene.pv.PolyData(pts)
        sl = s.flow.streamlines_from_source(seeds, vectors="velocity", max_length=4 * (b[1] - b[0]),
                                            integration_direction="both", max_steps=4000)
        if sl.n_points == 0:
            return
        geo = sl.tube(radius=0.0012 * max(bb[1] - bb[0], bb[3] - bb[2])) if self.st_tube.isChecked() else sl
        act = self.plotter.add_mesh(geo, scalars=name if name in geo.point_data else None, cmap=self.cmap.currentText(),
                                    clim=self._clim(), show_scalar_bar=False, smooth_shading=True, render=False)
        self.actors["stream"] = [act]

    def _draw_volume(self):
        s = self.scene
        if not self.vol_on.isChecked() or s.flow is None:
            return
        name = self.vol_f.currentText() or "grad_rho"
        if not s.ensure(name):
            return
        cmap, clim = self._field_style(name)
        if name in ("grad_rho", "vorticity", "Q"):
            a_ = np.asarray(s.flow.point_data[name])
            a_ = a_[np.isfinite(a_) & (a_ > 0)]
            clim = [0.0, float(np.percentile(a_, 99.5)) if a_.size else 1.0]
            cmap = "inferno"
        pv = s.pv
        b = s.bounds()
        n = 160
        ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        h = ext.max() / n
        dims = np.maximum((ext / h).astype(int) + 1, 2)
        img = pv.ImageData(dimensions=dims, spacing=(h, h, h), origin=(b[0], b[2], b[4]))
        img = s.fd.sample_onto(img)
        a = np.asarray(img.point_data[name], float)
        a[~np.isfinite(a)] = clim[0]
        img.point_data[name] = a
        act = self.plotter.add_volume(img, scalars=name, cmap=cmap, clim=clim,
                                      opacity=self.vol_op.currentText(), show_scalar_bar=False)
        self.actors["volume"] = [act]

    # ------------------------------------------------------------------ view
    def _background(self, name):
        if self.plotter is None:
            return
        if name == "dark gradient":
            self.plotter.set_background("#14171c", top="#2b3240")
        elif name == "black":
            self.plotter.set_background("black")
        elif name == "white":
            self.plotter.set_background("white")
        else:
            self.plotter.set_background("#ffffff", top="#c9d6e3")
        self.rebuild()

    def camera(self, name):
        if self.plotter is None or self.scene is None:
            return
        bb = self.scene.body_bounds()
        c = np.array([(bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, (bb[4] + bb[5]) / 2])
        L = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4], 1e-3)
        pos = {"Side": c + [0, -1.6 * L, 0], "Top": c + [0, 0, 1.6 * L], "Engine end": c + [-1.4 * L, 0.25 * L, 0.2 * L],
               "Top end": c + [1.4 * L, 0.25 * L, 0.2 * L], "Isometric": c + [-0.6 * L, -1.0 * L, 0.55 * L]}.get(name)
        if pos is None:
            self.plotter.reset_camera()
        else:
            up = (0, 1, 0) if name == "Top" else (0, 0, 1)
            self.plotter.camera_position = [tuple(pos), tuple(c), up]
        self.plotter.render()

    def screenshot(self):
        if self.plotter is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", os.path.join(
            self.scene.folder if self.scene else RESULTS, "view.png"), "PNG (*.png)")
        if path:
            self.plotter.screenshot(path, scale=2)

    def cinematic(self):
        """One-click LAVA-like look."""
        if self.scene is None or self.scene.flow is None:
            return
        self.bg.setCurrentText("black")
        self.body_col.setCurrentText("metallic grey")
        self.field.setCurrentText("mach")
        self.cmap.setCurrentText("turbo")
        self.sl_on.setChecked(True)
        self.sl_field.setCurrentText("schlieren")
        self.sl_ax.setCurrentIndex(0)
        self.sl_pos.setValue(500)
        self.iso_f.setCurrentText("Q")
        self._iso_default(rebuild=False)
        self.iso_col.setChecked(True)
        self.iso_on.setChecked(True)
        self.auto.setChecked(True)
        self._range_auto(rebuild=False)
        self.rebuild()
        self.camera("Isometric")
