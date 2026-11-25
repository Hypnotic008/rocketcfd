"""rocketcfd desktop app: set up, run and look at simulations.

  python -m rocketcfd.app            (or double-click rocketcfd_app.bat on Windows)
"""
import os
import sys


def main():
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6 import QtCore, QtGui, QtWidgets
    from .common import DARK_QSS, ROOT, hint, settings

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("rocketcfd")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)

    from .setup_page import SetupPage
    from .run_page import RunPage
    from .results_page import ResultsPage
    from .plots_page import PlotsPage
    from .anim_page import AnimPage
    from .help_page import HelpPage

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("rocketcfd - compressible CFD for launch vehicles")
    central = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(central)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    nav = QtWidgets.QListWidget()
    nav.setObjectName("nav")
    nav.setFixedWidth(170)
    stack = QtWidgets.QStackedWidget()
    setup, run, results = SetupPage(), RunPage(), ResultsPage()
    plots, anim, helpp = PlotsPage(results), AnimPage(), HelpPage()
    pages = [("Setup", setup), ("Run", run), ("3-D results", results), ("Plots", plots), ("Animations", anim),
             ("Help", helpp)]
    for name, w in pages:
        nav.addItem(name)
        wrap = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(wrap)
        wl.setContentsMargins(12, 10, 12, 10)
        wl.addWidget(w)
        stack.addWidget(wrap)
    nav.currentRowChanged.connect(stack.setCurrentIndex)
    lay.addWidget(nav)
    lay.addWidget(stack, 1)
    win.setCentralWidget(central)

    def go(name):
        nav.setCurrentRow([p[0] for p in pages].index(name))

    setup.run_requested.connect(lambda job: (run.add(job, start=True), go("Run")))
    setup.queue_requested.connect(lambda job: (run.add(job), go("Run")))

    def finished(job):
        for p in (results, plots, anim):
            p.refresh_folders() if hasattr(p, "refresh_folders") else p.refresh()
        if job.out and os.path.isdir(job.out) and job.status == "done":
            win.statusBar().showMessage(f"Finished: {job.title}  ->  {job.out}", 15000)

    run.finished.connect(finished)

    def open_results(job):
        if job.out and os.path.isdir(job.out):
            results.load(job.out)
            plots.load(job.out)
            anim.load(job.out)
            go("3-D results")

    run.b_open.clicked.disconnect()
    run.b_open.clicked.connect(lambda: run.queue.currentRow() >= 0 and open_results(run.jobs[run.queue.currentRow()]))

    # device line in the status bar
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (no CUDA PyTorch)"
    except Exception:                           # noqa: BLE001
        dev = "PyTorch not installed"
    win.statusBar().showMessage(f"Compute device: {dev}   |   project: {ROOT}")

    st = settings()
    geo = st.value("geometry")
    if geo is not None:
        win.restoreGeometry(geo)
    else:
        win.resize(1600, 980)
    app.aboutToQuit.connect(lambda: st.setValue("geometry", win.saveGeometry()))
    nav.setCurrentRow(0)
    win.show()
    if os.environ.get("ROCKETCFD_APP_SCREENSHOTS"):
        return app, win, nav, dict(pages)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
