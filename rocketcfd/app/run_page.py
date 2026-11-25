"""Run page: a queue of simulations run one after another, with live convergence plots, the
solver log, progress and a time estimate."""
import os
import re
import sys
import time

from PySide6 import QtCore, QtWidgets

from .common import ROOT, hint, mpl_canvas, style_axes

RE_MAX = re.compile(r"up to (\d+) (?:iterations|steps)")
RE_RES = re.compile(r"^(?:iter|step)\s+(\d+).*?res\(rho\) = ([0-9.eE+-]+)")
RE_TGV = re.compile(r"t\s+([\d.]+)\s+Ek\s+([\d.]+)")
COEF_KEYS = ("CD", "CL", "CN", "CS", "C_restore", "C_roll", "CY", "Cm", "Cn")


def parse_coefficients(line):
    toks = line.split()
    out = {}
    for i in range(len(toks) - 1):
        if toks[i] in COEF_KEYS:
            try:
                out[toks[i]] = float(toks[i + 1])
            except ValueError:
                pass
    return out


class Job:
    def __init__(self, d):
        self.__dict__.update(d)
        self.status = "queued"
        self.log = []
        self.res = []          # (iteration, residual)
        self.coef = []         # (iteration, {name: value})
        self.series = []       # generic (x, y) e.g. TGV energy
        self.max_it = None
        self.t0 = None
        self.t1 = None


class RunPage(QtWidgets.QWidget):
    finished = QtCore.Signal(object)          # Job

    def __init__(self):
        super().__init__()
        self.jobs = []
        self.current = None
        self.proc = None
        lay = QtWidgets.QHBoxLayout(self)

        # left: queue
        left = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("Runs")
        t.setObjectName("title")
        left.addWidget(t)
        left.addWidget(hint("Runs execute one after another. Leave the app open; you can browse results "
                            "of finished runs while the next one computes."))
        self.queue = QtWidgets.QListWidget()
        self.queue.setMinimumWidth(320)
        self.queue.currentRowChanged.connect(self._show_job)
        left.addWidget(self.queue, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_start = QtWidgets.QPushButton("Start queue")
        self.b_start.setObjectName("primary")
        self.b_start.clicked.connect(self.start_next)
        row.addWidget(self.b_start)
        self.b_stop = QtWidgets.QPushButton("Stop current")
        self.b_stop.setObjectName("danger")
        self.b_stop.clicked.connect(self.stop)
        row.addWidget(self.b_stop)
        left.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self._remove)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Clear finished")
        b.clicked.connect(self._clear_finished)
        row.addWidget(b)
        left.addLayout(row)
        self.b_open = QtWidgets.QPushButton("Open results of selected run")
        self.b_open.clicked.connect(self._open_results)
        left.addWidget(self.b_open)
        lay.addLayout(left, 0)

        # right: status, plots, log
        right = QtWidgets.QVBoxLayout()
        self.status = QtWidgets.QLabel("No run selected.")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        right.addWidget(self.progress)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        pw, self.fig, self.canvas = mpl_canvas()
        split.addWidget(pw)
        self.logw = QtWidgets.QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(20000)
        self.logw.setStyleSheet("font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 9pt;")
        split.addWidget(self.logw)
        split.setSizes([520, 260])
        right.addWidget(split, 1)
        lay.addLayout(right, 1)
        self.ax_res = self.fig.add_subplot(1, 2, 1)
        self.ax_coef = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9, wspace=0.28)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1500)
        self._dirty = True
        self._draw()

    # ------------------------------------------------------------------ queue
    def add(self, d, start=False):
        j = Job(d)
        self.jobs.append(j)
        self.queue.addItem(self._label(j))
        self.queue.setCurrentRow(len(self.jobs) - 1)
        if start and self.proc is None:
            self.start_next()

    def _label(self, j):
        icon = dict(queued="[ ]", running="[>]", done="[ok]", failed="[x]", stopped="[-]")[j.status]
        return f"{icon} {j.title}"

    def _relabel(self):
        for i, j in enumerate(self.jobs):
            self.queue.item(i).setText(self._label(j))

    def start_next(self):
        if self.proc is not None:
            return
        nxt = next((j for j in self.jobs if j.status == "queued"), None)
        if nxt is None:
            return
        self.current = nxt
        nxt.status = "running"
        nxt.t0 = time.time()
        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(ROOT)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("MPLBACKEND", "Agg")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        script = nxt.script
        args = (["-u", "-m", script[3:].strip()] if script.startswith("-m ") else ["-u", os.path.join(ROOT, script)])
        self.proc.start(sys.executable, args + list(nxt.args))
        self._relabel()
        self.queue.setCurrentRow(self.jobs.index(nxt))

    def stop(self):
        if self.proc is not None:
            self.current.status = "stopped"
            self.proc.kill()

    def _done(self, code, _status):
        j = self.current
        j.t1 = time.time()
        if j.status != "stopped":
            j.status = "done" if code == 0 else "failed"
        self.proc = None
        self._relabel()
        self._dirty = True
        self.finished.emit(j)
        self.start_next()

    def _remove(self):
        i = self.queue.currentRow()
        if i < 0 or self.jobs[i].status == "running":
            return
        self.jobs.pop(i)
        self.queue.takeItem(i)

    def _clear_finished(self):
        for i in reversed(range(len(self.jobs))):
            if self.jobs[i].status in ("done", "failed", "stopped"):
                self.jobs.pop(i)
                self.queue.takeItem(i)

    def _open_results(self):
        i = self.queue.currentRow()
        if i >= 0:
            self.finished.emit(self.jobs[i])

    # ------------------------------------------------------------------ output
    def _read(self):
        j = self.current
        txt = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in txt.splitlines():
            j.log.append(line)
            self._parse(j, line)
            if self.jobs.index(j) == self.queue.currentRow():
                self.logw.appendPlainText(line)
        self._dirty = True

    def _parse(self, j, line):
        m = RE_MAX.search(line)
        if m:
            j.max_it = int(m.group(1))
        m = RE_RES.search(line)
        if m:
            j.res.append((int(m.group(1)), float(m.group(2))))
            return
        if line.startswith(" ") and "CD" in line:
            c = parse_coefficients(line)
            if c:
                j.coef.append((j.res[-1][0] if j.res else len(j.coef), c))
            return
        m = RE_TGV.search(line)
        if m:
            j.series.append((float(m.group(1)), float(m.group(2))))

    def _show_job(self, row):
        if row < 0 or row >= len(self.jobs):
            return
        j = self.jobs[row]
        self.logw.setPlainText("\n".join(j.log[-20000:]))
        self.logw.moveCursor(self.logw.textCursor().End)
        self._dirty = True
        self._refresh()

    def _refresh(self):
        row = self.queue.currentRow()
        if row < 0 or row >= len(self.jobs):
            return
        j = self.jobs[row]
        el = ((j.t1 or time.time()) - j.t0) if j.t0 else 0.0
        it = j.res[-1][0] if j.res else 0
        txt = f"<b>{j.title}</b><br>status: {j.status}"
        if j.t0:
            txt += f" &nbsp;|&nbsp; elapsed {el/60:.1f} min"
        if j.max_it:
            self.progress.setRange(0, j.max_it)
            self.progress.setValue(min(it, j.max_it))
            if j.status == "running" and it > 0:
                eta = el / it * (j.max_it - it)
                txt += (f" &nbsp;|&nbsp; {it} of at most {j.max_it} &nbsp;|&nbsp; worst case {eta/60:.0f} min left "
                        f"(stops earlier if the forces settle)")
        else:
            self.progress.setRange(0, 0 if j.status == "running" else 1)
            self.progress.setValue(1 if j.status != "running" else 0)
        if j.coef:
            last = j.coef[-1][1]
            txt += "<br>" + " &nbsp; ".join(f"{k} <b>{v:+.4f}</b>" for k, v in last.items())
        txt += f"<br><span style='color:#8a93a3'>results: {j.out or '-'}</span>"
        self.status.setText(txt)
        if self._dirty:
            self._draw(j)
            self._dirty = False

    def _draw(self, j=None):
        for ax in (self.ax_res, self.ax_coef):
            ax.clear()
            style_axes(ax)
        self.ax_res.set_title("Residual (should fall)")
        self.ax_res.set_xlabel("iteration / step")
        self.ax_coef.set_title("Forces (should flatten out)")
        self.ax_coef.set_xlabel("iteration / step")
        if j is not None and j.res:
            it, r = zip(*j.res)
            self.ax_res.semilogy(it, r, color="#88c0d0", lw=1.2)
        if j is not None and j.coef:
            keys = [k for k in ("CD", "CN", "C_restore", "CL") if k in j.coef[-1][1]]
            x = [c[0] for c in j.coef]
            for k, col in zip(keys, ("#ebcb8b", "#a3be8c", "#b48ead", "#d08770")):
                self.ax_coef.plot(x, [c[1].get(k, float("nan")) for c in j.coef], "-o", ms=3, color=col, label=k)
            self.ax_coef.legend(facecolor="#2e3440", edgecolor="#4c566a", labelcolor="#d8dee9", fontsize=8)
        elif j is not None and j.series:
            x, y = zip(*j.series)
            self.ax_coef.plot(x, y, "-o", ms=3, color="#ebcb8b")
            self.ax_coef.set_title("Kinetic energy")
            self.ax_coef.set_xlabel("time")
        self.canvas.draw_idle()
