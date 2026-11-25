"""Setup page: pick a case template, fill in a grouped form (with presets and tooltips), check
the grid size, then run or queue it."""
import json
import os
import shlex
import sys

from PySide6 import QtCore, QtWidgets

from .cases import CASES, command_line
from .common import ROOT, hint


class SetupPage(QtWidgets.QWidget):
    run_requested = QtCore.Signal(object)       # dict(case, args, title, out)
    queue_requested = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.case = None
        self.widgets = {}
        self.rows = {}
        outer = QtWidgets.QHBoxLayout(self)

        # left: case templates
        left = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("New simulation")
        t.setObjectName("title")
        left.addWidget(t)
        left.addWidget(hint("Choose what to simulate. Hover over any field for an explanation."))
        self.case_list = QtWidgets.QListWidget()
        for c in CASES:
            self.case_list.addItem(c.name)
        self.case_list.setMinimumWidth(300)
        self.case_list.currentRowChanged.connect(self._select_case)
        left.addWidget(self.case_list, 1)
        self.desc = QtWidgets.QLabel()
        self.desc.setWordWrap(True)
        self.desc.setObjectName("hint")
        left.addWidget(self.desc)
        outer.addLayout(left, 0)

        # right: form
        right = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset = QtWidgets.QComboBox()
        self.preset.setMinimumWidth(280)
        top.addWidget(self.preset)
        b = QtWidgets.QPushButton("Apply preset")
        b.clicked.connect(self._apply_preset)
        top.addWidget(b)
        top.addStretch(1)
        self.adv = QtWidgets.QCheckBox("Show advanced options")
        self.adv.toggled.connect(self._show_advanced)
        top.addWidget(self.adv)
        right.addLayout(top)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        right.addWidget(self.scroll, 1)

        cmd_row = QtWidgets.QHBoxLayout()
        cmd_row.addWidget(QtWidgets.QLabel("Command:"))
        self.cmd = QtWidgets.QLineEdit()
        self.cmd.setReadOnly(True)
        cmd_row.addWidget(self.cmd, 1)
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.cmd.text()))
        cmd_row.addWidget(copy)
        right.addLayout(cmd_row)
        extra_row = QtWidgets.QHBoxLayout()
        extra_row.addWidget(QtWidgets.QLabel("Extra arguments:"))
        self.extra = QtWidgets.QLineEdit()
        self.extra.setPlaceholderText("optional, passed as-is, e.g. --growth 1.1")
        self.extra.textChanged.connect(self._update_cmd)
        extra_row.addWidget(self.extra, 1)
        right.addLayout(extra_row)

        btns = QtWidgets.QHBoxLayout()
        self.b_est = QtWidgets.QPushButton("Estimate grid size && memory")
        self.b_est.setToolTip("Builds the grid only (seconds) and reports cells and GPU memory.")
        self.b_est.clicked.connect(lambda: self._quick("--dry-run"))
        btns.addWidget(self.b_est)
        self.b_prof = QtWidgets.QPushButton("Measure speed")
        self.b_prof.setToolTip("Builds the case and times 20 steps of each solver (a few minutes).")
        self.b_prof.clicked.connect(lambda: self._quick("--profile", "20"))
        btns.addWidget(self.b_prof)
        b = QtWidgets.QPushButton("Save setup...")
        b.clicked.connect(self._save)
        btns.addWidget(b)
        b = QtWidgets.QPushButton("Load setup...")
        b.clicked.connect(self._load)
        btns.addWidget(b)
        btns.addStretch(1)
        b = QtWidgets.QPushButton("Add to queue")
        b.clicked.connect(lambda: self.queue_requested.emit(self.job()))
        btns.addWidget(b)
        b = QtWidgets.QPushButton("Run now")
        b.setObjectName("primary")
        b.clicked.connect(lambda: self.run_requested.emit(self.job()))
        btns.addWidget(b)
        right.addLayout(btns)
        outer.addLayout(right, 1)
        self.case_list.setCurrentRow(0)

    # ------------------------------------------------------------------ form
    def _select_case(self, row):
        self.case = CASES[row]
        c = self.case
        self.desc.setText(c.description + (f"\n\nTypical run time: {c.gpu_minutes}." if c.gpu_minutes else ""))
        self.preset.clear()
        self.preset.addItem("(defaults)")
        for k in c.presets:
            self.preset.addItem(k)
        form_host = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(form_host)
        self.widgets, self.rows = {}, {}
        groups = {}
        for f in c.fields:
            if f.group not in groups:
                gb = QtWidgets.QGroupBox(f.group)
                fl = QtWidgets.QFormLayout(gb)
                fl.setLabelAlignment(QtCore.Qt.AlignRight)
                groups[f.group] = fl
                lay.addWidget(gb)
            w = self._make_widget(f)
            lab = QtWidgets.QLabel(f.label)
            lab.setToolTip(f.help)
            w.setToolTip(f.help)
            groups[f.group].addRow(lab, w)
            self.widgets[f.key] = (f, w)
            self.rows[f.key] = (lab, w)
        lay.addStretch(1)
        self.scroll.setWidget(form_host)
        self.b_est.setEnabled(c.dry_run)
        self.b_prof.setEnabled(c.profile)
        self._show_advanced(self.adv.isChecked())
        self._update_cmd()

    def _make_widget(self, f):
        if f.kind == "float":
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(f.lo, f.hi)
            w.setDecimals(f.decimals)
            w.setSingleStep(f.step)
            w.setValue(float(f.default))
            w.valueChanged.connect(self._update_cmd)
        elif f.kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(max(f.lo, -2**31)), int(min(f.hi, 2**31 - 1)))
            w.setSingleStep(int(max(f.step, 1)))
            w.setValue(int(f.default))
            w.valueChanged.connect(self._update_cmd)
        elif f.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(f.default))
            w.toggled.connect(self._update_cmd)
        elif f.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems(f.choices)
            w.setCurrentText(str(f.default))
            w.currentTextChanged.connect(self._update_cmd)
        else:
            w = QtWidgets.QLineEdit(str(f.default))
            w.textChanged.connect(self._update_cmd)
        w.setMinimumWidth(160)
        return w

    def _show_advanced(self, on):
        for key, (lab, w) in self.rows.items():
            adv = self.widgets[key][0].advanced
            lab.setVisible(on or not adv)
            w.setVisible(on or not adv)

    def values(self):
        out = {}
        for key, (f, w) in self.widgets.items():
            if f.kind == "float":
                out[key] = round(w.value(), f.decimals)
            elif f.kind == "int":
                out[key] = w.value()
            elif f.kind == "bool":
                out[key] = w.isChecked()
            elif f.kind == "choice":
                out[key] = w.currentText()
            else:
                out[key] = w.text()
        return out

    def set_values(self, vals):
        for key, v in vals.items():
            if key not in self.widgets:
                continue
            f, w = self.widgets[key]
            if f.kind == "float":
                w.setValue(float(v))
            elif f.kind == "int":
                w.setValue(int(v))
            elif f.kind == "bool":
                w.setChecked(bool(v))
            elif f.kind == "choice":
                w.setCurrentText(str(v))
            else:
                w.setText(str(v))
        self._update_cmd()

    def _apply_preset(self):
        name = self.preset.currentText()
        defaults = {f.key: f.default for f in self.case.fields}
        self.set_values(defaults)
        if name in self.case.presets:
            self.set_values(self.case.presets[name])

    def args(self):
        a = command_line(self.case, self.values())
        extra = self.extra.text().strip()
        if extra:
            a += shlex.split(extra)
        return a

    def _update_cmd(self, *_):
        if self.case is None or not self.widgets:
            return
        self.cmd.setText("python " + self.case.script + " " + " ".join(shlex.quote(x) for x in self.args()))

    def job(self):
        v = self.values()
        out = v.get(self.case.out_key, "")
        return dict(case=self.case.name, script=self.case.script, args=self.args(),
                    title=f"{self.case.name.split(' - ')[0]}: " + self._short(v), out=os.path.join(ROOT, out) if out else None)

    def _short(self, v):
        bits = []
        for k in ("mach", "aoa", "alt", "re", "n"):
            if k in v and self.widgets[k][0].kind != "floats":
                bits.append(f"{k} {v[k]}")
        return ", ".join(bits) or "run"

    # ------------------------------------------------------------------ quick checks
    def _quick(self, *flag):
        cmd = [sys.executable, "-u", os.path.join(ROOT, self.case.script)] + self.args() + list(flag)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Grid estimate" if flag[0] == "--dry-run" else "Speed measurement")
        dlg.resize(820, 520)
        lay = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        lay.addWidget(txt)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        proc = QtCore.QProcess(dlg)
        proc.setWorkingDirectory(ROOT)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: txt.appendPlainText(bytes(proc.readAllStandardOutput()).decode(errors="replace").rstrip()))
        proc.finished.connect(lambda *_: txt.appendPlainText("\n[done]"))
        proc.start(cmd[0], cmd[1:])
        dlg.finished.connect(lambda *_: proc.kill())
        dlg.show()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", os.path.join(ROOT, "setup.json"),
                                                        "Setup (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump(dict(case=self.case.name, values=self.values(), extra=self.extra.text()), fh, indent=2)

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load setup", ROOT, "Setup (*.json)")
        if not path:
            return
        with open(path) as fh:
            d = json.load(fh)
        names = [c.name for c in CASES]
        if d.get("case") in names:
            self.case_list.setCurrentRow(names.index(d["case"]))
        self.set_values(d.get("values", {}))
        self.extra.setText(d.get("extra", ""))
