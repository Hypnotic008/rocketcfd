"""Help page: a short guide for users with intermediate CFD experience."""
from PySide6 import QtWidgets

GUIDE = """
<h2>Getting started</h2>
<ol>
<li><b>Setup</b>: pick a case on the left, choose a <i>preset</i> and press <b>Apply preset</b>, adjust the
fields (hover over any field for an explanation), then <b>Estimate grid size &amp; memory</b> to check it fits
your GPU (6 GB holds about 40 million cells inviscid). <b>Run now</b> starts it; <b>Add to queue</b> lines up
several runs to compute one after another (overnight sweeps).</li>
<li><b>Run</b>: watch the residual fall (left plot) and the forces flatten out (right plot). The run stops by
itself once the forces are steady. The time estimate is the worst case.</li>
<li><b>3-D results</b>: choose the results folder. Switch layers on and off on the right: body surface coloured by
surface pressure, a cutting plane (drag the <i>Position</i> slider), iso-surfaces, streamlines and volume
rendering. <b>Cinematic look</b> sets up the NASA-style picture in one click. Mouse: left-drag rotates,
shift+drag pans, wheel zooms.</li>
<li><b>Plots</b>: convergence history, the force breakdown by part (with unsteadiness whiskers), filled
contour plots of any field on any plane (with in-plane streamlines), line probes (export to CSV) and the aero
database curves. <b>Save plot</b> writes PNG, SVG or PDF.</li>
<li><b>Animations</b>: play back the frames recorded during a run, and render movies: cinematic LAVA-style
orbits and time evolutions, schlieren, slice sweeps. Finished movies play in the built-in player.</li>
</ol>

<h2>Which solver settings?</h2>
<table border=1 cellpadding=5 cellspacing=0>
<tr><th>Goal</th><th>Settings</th></tr>
<tr><td>Forces and moments (CatchSim), fast</td><td>inviscid, implicit solver, cell size 0.3 m, fin and engine
patches 0.06 m</td></tr>
<tr><td>Friction drag, boundary layers</td><td>Viscous turbulent (RANS): hull shell patches 0.25,0.03,0.03</td></tr>
<tr><td>Unsteady wake, movies</td><td>Unsteady LES + Snapshots 300 + 3-D snapshot stride 2 (no patches for a
quick movie)</td></tr>
<tr><td>Many conditions</td><td>Aero database case (resumable)</td></tr>
</table>

<h2>Fields you can show</h2>
<ul>
<li><b>mach, p, rho</b>: from the solver. <b>T</b> temperature, <b>Cp</b> pressure coefficient,
<b>p0_ratio</b> total-pressure ratio (shows shock losses), <b>speed</b>.</li>
<li><b>schlieren</b>: numerical schlieren, exp(-k |grad rho|): shocks and shear layers dark, as in a wind-tunnel
schlieren photograph.</li>
<li><b>Q</b>: Q-criterion; iso-surfaces of positive Q show vortex cores (the chine and fin vortices).
<b>lambda2</b> is an alternative (negative = vortex). <b>vorticity</b>: |curl u|.</li>
</ul>

<h2>Is my result trustworthy?</h2>
<ul>
<li>The residual should fall by 2-3 orders and the forces should stop changing. Whiskers in <i>Forces by part</i>
show how unsteady each part is.</li>
<li>Repeat with a finer grid (cell size x 0.7): if the forces change by less than a few per cent the grid is
fine enough.</li>
<li>The validation cases (cylinder, flat plate, Taylor-Green) show how the solver compares with theory and
experiment.</li>
</ul>

<h2>Command line</h2>
<p>Everything in the app is a command you can also run in a terminal; the Setup page shows it under
<i>Command</i> (with a Copy button). Movies: <tt>python -m rocketcfd.cinematic results/&lt;folder&gt; --mode orbit</tt>.</p>
"""


class HelpPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QTextBrowser()
        t.setHtml(GUIDE)
        t.setOpenExternalLinks(True)
        lay.addWidget(t)
