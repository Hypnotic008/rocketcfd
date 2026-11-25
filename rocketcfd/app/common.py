"""Shared helpers for the rocketcfd desktop app."""
import json
import os

from PySide6 import QtCore, QtGui, QtWidgets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")

DARK_QSS = """
QWidget { background: #1e2228; color: #d8dee9; font-size: 10pt; }
QMainWindow::separator { background: #2b3038; width: 3px; }
QGroupBox { border: 1px solid #39404b; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #88c0d0; font-weight: bold; }
QPushButton { background: #2f3540; border: 1px solid #434c5e; border-radius: 5px; padding: 5px 12px; }
QPushButton:hover { background: #3b4252; }
QPushButton:pressed { background: #4c566a; }
QPushButton:disabled { color: #6b7280; }
QPushButton#primary { background: #2e6f9e; border-color: #3b88c0; color: white; font-weight: bold; }
QPushButton#primary:hover { background: #3b88c0; }
QPushButton#danger { background: #8f3a3a; border-color: #b04545; color: white; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget, QTableWidget, QTextBrowser {
    background: #262b33; border: 1px solid #3b4250; border-radius: 4px; padding: 3px; selection-background-color: #3b6e95; }
QComboBox QAbstractItemView { background: #262b33; }
QTabBar::tab { background: #262b33; padding: 6px 14px; border: 1px solid #39404b; border-bottom: none;
               border-top-left-radius: 5px; border-top-right-radius: 5px; }
QTabBar::tab:selected { background: #2f3540; color: #88c0d0; }
QListWidget#nav { background: #171a1f; border: none; font-size: 11pt; }
QListWidget#nav::item { padding: 12px 10px; }
QListWidget#nav::item:selected { background: #2e3440; color: #88c0d0; border-left: 3px solid #88c0d0; }
QProgressBar { border: 1px solid #3b4250; border-radius: 4px; text-align: center; background: #262b33; }
QProgressBar::chunk { background: #3b88c0; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #3b4250; border-radius: 2px; }
QSlider::handle:horizontal { background: #88c0d0; width: 14px; margin: -6px 0; border-radius: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #6b7486; border-radius: 3px; background: #262b33; }
QCheckBox::indicator:checked { background: #3b88c0; border-color: #88c0d0; }
QToolTip { background: #2e3440; color: #eceff4; border: 1px solid #4c566a; }
QHeaderView::section { background: #2b3038; padding: 4px; border: none; }
QScrollArea { border: none; }
QLabel#hint { color: #8a93a3; font-size: 9pt; }
QLabel#title { font-size: 15pt; font-weight: bold; color: #eceff4; }
"""


def settings():
    return QtCore.QSettings("rocketcfd", "rocketcfd-app")


def result_folders():
    """Results folders that hold something the app can show, newest first."""
    out = []
    if not os.path.isdir(RESULTS):
        return out
    for root, dirs, files in os.walk(RESULTS):
        depth = root[len(RESULTS):].count(os.sep)
        if depth > 2:
            dirs[:] = []
            continue
        if any(f in files for f in ("flow.vtk", "flow_patches.vtm", "surface_cp.vtk", "loads.json", "cases.csv")) or \
                os.path.isdir(os.path.join(root, "snapshots")):
            out.append(root)
    out.sort(key=lambda p: -os.path.getmtime(p))
    return out


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:                              # noqa: BLE001
        return None


def hint(text):
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


def mpl_canvas(dark=True):
    """A matplotlib figure + canvas + toolbar in a widget. Returns (widget, figure, canvas)."""
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 4), dpi=100, facecolor="#1e2228" if dark else "white")
    canvas = FigureCanvasQTAgg(fig)
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(NavigationToolbar2QT(canvas, w))
    lay.addWidget(canvas, 1)
    return w, fig, canvas


def style_axes(ax, dark=True):
    if not dark:
        return ax
    ax.set_facecolor("#232830")
    for s in ax.spines.values():
        s.set_color("#5c6370")
    ax.tick_params(colors="#c8ccd4")
    ax.xaxis.label.set_color("#d8dee9")
    ax.yaxis.label.set_color("#d8dee9")
    ax.title.set_color("#eceff4")
    ax.grid(True, color="#3b4250", lw=0.6, alpha=0.8)
    return ax


def open_path(path):
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


def python_exe():
    """The console Python next to the app's interpreter (the desktop shortcut starts the app
    with pythonw.exe, whose children would have no usable stdout)."""
    import sys
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def new_process(parent):
    """A QProcess in the project folder, output merged, unbuffered, no console window on Windows."""
    proc = QtCore.QProcess(parent)
    proc.setWorkingDirectory(ROOT)
    proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
    env = QtCore.QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("MPLBACKEND", "Agg")
    proc.setProcessEnvironment(env)
    import sys
    if sys.platform == "win32":
        try:
            def no_window(args):
                args.flags |= 0x08000000          # CREATE_NO_WINDOW
            proc.setCreateProcessArgumentsModifier(no_window)
        except Exception:                          # noqa: BLE001
            pass
    return proc
