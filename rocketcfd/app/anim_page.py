"""Animations page: play back the snapshots a run recorded (Mach on the symmetry planes,
surface pressure), make movies (cinematic LAVA-style 3-D, orbits, slice sweeps) and watch them
in the built-in player."""
import glob
import os
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from .common import ROOT, RESULTS, hint, mpl_canvas, new_process, open_path, python_exe, result_folders, style_axes

MOVIES = [
    ("Cinematic: flow evolving (LAVA style, needs 3-D snapshots)", "-m rocketcfd.cinematic", ["--mode", "evolve", "--style", "lava"]),
    ("Cinematic: orbit around the steady flow (LAVA style)", "-m rocketcfd.cinematic", ["--mode", "orbit", "--style", "lava"]),
    ("Cinematic: orbit, schlieren only", "-m rocketcfd.cinematic", ["--mode", "orbit", "--style", "schlieren"]),
    ("Cinematic: orbit, Mach plane + vortices", "-m rocketcfd.cinematic", ["--mode", "orbit", "--style", "mach"]),
    ("Cinematic: orbit, surface pressure", "-m rocketcfd.cinematic", ["--mode", "orbit", "--style", "pressure"]),
    ("Cinematic: orbit, engine plumes (temperature)", "-m rocketcfd.cinematic", ["--mode", "orbit", "--style", "plume"]),
    ("Cinematic: still image (4K)", "-m rocketcfd.cinematic", ["--mode", "still", "--width", "3840", "--height", "2160"]),
    ("Streamline particles: orbit", "-m rocketcfd.viewer", ["--movie", "orbit"]),
    ("Slice sweeping along the booster", "-m rocketcfd.viewer", ["--movie", "sweep"]),
    ("Planes evolving (2-D snapshots)", "-m rocketcfd.viewer", ["--movie", "evolve"]),
]


class AnimPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.folder = None
        self.mesh = None
        self.files = []
        self.proc = None
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Animations")
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
        lay.addLayout(top)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)

        # --- snapshot player
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(hint("Frames recorded during a run with 'Snapshots for movies' > 0. Mach number on the two "
                         "planes through the booster axis."))
        cw, self.fig, self.canvas = mpl_canvas()
        v.addWidget(cw, 1)
        row = QtWidgets.QHBoxLayout()
        self.play_b = QtWidgets.QPushButton("Play")
        self.play_b.clicked.connect(self._toggle)
        row.addWidget(self.play_b)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._frame)
        row.addWidget(self.slider, 1)
        self.flabel = QtWidgets.QLabel("-")
        row.addWidget(self.flabel)
        row.addWidget(QtWidgets.QLabel("fps"))
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(12)
        row.addWidget(self.fps)
        self.vmax = QtWidgets.QDoubleSpinBox()
        self.vmax.setRange(0.1, 30)
        self.vmax.setValue(3.0)
        self.vmax.setPrefix("Mach max ")
        self.vmax.valueChanged.connect(lambda *_: self._frame(self.slider.value()))
        row.addWidget(self.vmax)
        v.addLayout(row)
        tabs.addTab(w, "Snapshot player")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next)

        # --- movie maker
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for name, *_ in MOVIES:
            self.kind.addItem(name)
        form.addRow("Movie", self.kind)
        self.res = QtWidgets.QComboBox()
        self.res.addItems(["1920x1080", "1280x720", "2560x1440", "3840x2160"])
        form.addRow("Resolution", self.res)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(10, 3000)
        self.frames.setValue(240)
        form.addRow("Frames (orbit)", self.frames)
        self.mfps = QtWidgets.QSpinBox()
        self.mfps.setRange(5, 60)
        self.mfps.setValue(30)
        form.addRow("Frames per second", self.mfps)
        self.mtitle = QtWidgets.QLineEdit()
        self.mtitle.setPlaceholderText("optional caption")
        form.addRow("Caption", self.mtitle)
        v.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Render movie")
        b.setObjectName("primary")
        b.clicked.connect(self._render)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Cancel")
        b.clicked.connect(self._cancel)
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.mlog = QtWidgets.QPlainTextEdit()
        self.mlog.setReadOnly(True)
        self.mlog.setMaximumHeight(160)
        v.addWidget(self.mlog)
        v.addWidget(QtWidgets.QLabel("Movies and images in this folder:"))
        self.movies = QtWidgets.QListWidget()
        self.movies.itemDoubleClicked.connect(lambda it: self._play_movie(it.data(QtCore.Qt.UserRole)))
        v.addWidget(self.movies, 1)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Play selected")
        b.clicked.connect(lambda: self.movies.currentItem() and self._play_movie(
            self.movies.currentItem().data(QtCore.Qt.UserRole)))
        row.addWidget(b)
        b = QtWidgets.QPushButton("Open folder")
        b.clicked.connect(lambda: self.folder and open_path(self.folder))
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        tabs.addTab(w, "Make movies")

        # --- player
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.player = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
            self.video = QVideoWidget()
            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.video)
            v.addWidget(self.video, 1)
            row = QtWidgets.QHBoxLayout()
            for name, fn in (("Play", self.player.play), ("Pause", self.player.pause), ("Stop", self.player.stop)):
                b = QtWidgets.QPushButton(name)
                b.clicked.connect(fn)
                row.addWidget(b)
            self.loop = QtWidgets.QCheckBox("loop")
            self.loop.setChecked(True)
            row.addWidget(self.loop)
            row.addStretch(1)
            v.addLayout(row)
            self.player.mediaStatusChanged.connect(self._media_status)
        except Exception as e:                  # noqa: BLE001
            v.addWidget(QtWidgets.QLabel(f"Built-in player unavailable ({e}); movies open in your default player."))
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.hide()
        v.addWidget(self.image, 1)
        self.tabs = tabs
        tabs.addTab(w, "Player")
        self.refresh()

    # ------------------------------------------------------------------ folders
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
        sd = os.path.join(folder, "snapshots")
        self.files = sorted(glob.glob(os.path.join(sd, "snap_*.npz")))
        self.mesh = dict(np.load(os.path.join(sd, "mesh.npz"))) if os.path.exists(os.path.join(sd, "mesh.npz")) else None
        self.slider.setRange(0, max(len(self.files) - 1, 0))
        if self.files:                    # colour range from the last frame
            last = np.load(self.files[-1])
            m = np.asarray(last["mach_xz"], np.float32)
            m = m[np.isfinite(m)]
            if m.size:
                self.vmax.blockSignals(True)
                self.vmax.setValue(float(np.round(np.percentile(m, 99.5), 1)) or 1.0)
                self.vmax.blockSignals(False)
        self.slider.setValue(0)
        self._frame(0)
        self._list_movies()

    def _list_movies(self):
        self.movies.clear()
        if not self.folder:
            return
        for f in sorted(glob.glob(os.path.join(self.folder, "*.mp4")) + glob.glob(os.path.join(self.folder, "*.png")),
                        key=os.path.getmtime, reverse=True):
            it = QtWidgets.QListWidgetItem(os.path.basename(f))
            it.setData(QtCore.Qt.UserRole, f)
            self.movies.addItem(it)

    # ------------------------------------------------------------------ snapshot player
    def _toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_b.setText("Play")
        elif self.files:
            self.timer.start(int(1000 / self.fps.value()))
            self.play_b.setText("Pause")

    def _next(self):
        if not self.files:
            return
        self.slider.setValue((self.slider.value() + 1) % len(self.files))

    def _frame(self, i):
        self.fig.clear()
        if not self.files or self.mesh is None:
            ax = style_axes(self.fig.add_subplot(111))
            ax.text(0.5, 0.5, "No snapshots in this folder.\nRun with 'Snapshots for movies' > 0.", ha="center",
                    va="center", color="#d8dee9", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw_idle()
            self.flabel.setText("-")
            return
        s = np.load(self.files[i])
        m = self.mesh
        x = m["x"]
        for k, (key, other, solid, lab) in enumerate((("mach_xz", m["z"], m["solid_xz"], "z"),
                                                       ("mach_xy", m["y"], m["solid_xy"], "y"))):
            ax = style_axes(self.fig.add_subplot(2, 1, k + 1))
            a = np.ma.masked_where(solid, s[key].astype(np.float32))
            im = ax.pcolormesh(x, other, a.T, cmap="turbo", vmin=0, vmax=self.vmax.value(), shading="auto")
            ax.set_aspect("equal")
            ax.set_ylabel(lab + " [m]")
            if k == 0:
                ax.set_title(f"Mach   t = {float(s['t']):.4f} s   step {int(s['step'])}")
        ax.set_xlabel("x [m]")
        cb = self.fig.colorbar(im, ax=self.fig.axes, shrink=0.8)
        cb.ax.tick_params(colors="#c8ccd4")
        self.canvas.draw_idle()
        self.flabel.setText(f"{i + 1} / {len(self.files)}")

    # ------------------------------------------------------------------ movies
    def _render(self):
        if not self.folder or self.proc is not None:
            return
        name, mod, extra = MOVIES[self.kind.currentIndex()]
        w, h = self.res.currentText().split("x")
        args = ["-u", "-m", mod.split()[1], self.folder] + list(extra)
        if mod.endswith("cinematic"):
            if "--width" not in extra:
                args += ["--width", w, "--height", h]
            args += ["--frames", str(self.frames.value()), "--fps", str(self.mfps.value())]
            if self.mtitle.text().strip():
                args += ["--title", self.mtitle.text().strip()]
        else:
            args += ["--fps", str(self.mfps.value()), "--frames", str(self.frames.value())]
        self.mlog.clear()
        self.mlog.appendPlainText("python " + " ".join(args))
        self.proc = new_process(self)
        self.proc.readyReadStandardOutput.connect(
            lambda: self.mlog.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self.proc.finished.connect(self._rendered)
        self.proc.start(python_exe(), args)

    def _cancel(self):
        if self.proc is not None:
            self.proc.kill()

    def _rendered(self, code, _):
        self.mlog.appendPlainText("[finished]" if code == 0 else f"[failed, exit code {code}]")
        self.proc = None
        self._list_movies()

    def _play_movie(self, path):
        self.tabs.setCurrentIndex(2)
        if path.endswith(".png"):
            from PySide6 import QtGui
            if self.player is not None:
                self.player.stop()
                self.video.hide()
            pm = QtGui.QPixmap(path)
            self.image.setPixmap(pm.scaled(self.image.size() if self.image.width() > 50 else QtCore.QSize(1280, 720),
                                           QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.image.show()
            return
        self.image.hide()
        if self.player is None:
            open_path(path)
            return
        self.video.show()
        self.player.setSource(QtCore.QUrl.fromLocalFile(path))
        self.player.play()

    def _media_status(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        if st == QMediaPlayer.EndOfMedia and self.loop.isChecked():
            self.player.setPosition(0)
            self.player.play()
