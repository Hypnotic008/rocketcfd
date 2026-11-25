"""A results folder's 3-D flow for viewers: the base grid (flow.vtk) plus, for runs with local
refinement, the patch grids (flow_patches.vtm). Every region is drawn from the finest grid
there: cells marked 'covered' (owned by a finer grid) are left out of slices and iso-surfaces.
Derived fields (rocketcfd.derived) are computed on demand on every grid."""
import os

import numpy as np
import pyvista as pv

from . import derived


class FlowData:
    def __init__(self, folder, freestream=None):
        self.folder = folder
        self.fs = freestream
        self.base = None
        self.patches = []
        p = os.path.join(folder, "flow.vtk")
        if os.path.exists(p):
            self.base = self._prep(pv.read(p))
        q = os.path.join(folder, "flow_patches.vtm")
        if os.path.exists(q):
            mb = pv.read(q)
            self.patches = [self._prep(mb[i]) for i in range(mb.n_blocks) if mb[i] is not None and mb[i].n_points]
        self._vis = None

    @classmethod
    def from_grid(cls, g, freestream=None):
        fd = cls.__new__(cls)
        fd.folder, fd.fs, fd.base, fd.patches, fd._vis = None, freestream, cls._prep(g), [], None
        return fd

    def values(self, name):
        """All values of a field over the visible cells of every grid (for colour ranges, levels)."""
        self.ensure(name)
        return np.concatenate([np.asarray(v.point_data[name]).ravel() for v in self.visible() if name in v.point_data])

    @staticmethod
    def _prep(g):
        if "velocity" in g.point_data and "phi" in g.point_data:
            v = np.array(g.point_data["velocity"])
            v[np.asarray(g.point_data["phi"]) < 0] = 0.0          # streamlines must not run into the hull
            g.point_data["velocity"] = v
        return g

    @property
    def grids(self):
        return ([self.base] if self.base is not None else []) + self.patches

    def ensure(self, name):
        if name == "schlieren" and self.patches:
            return self._schlieren()
        ok = False
        for g in self.grids:
            if name not in g.point_data:
                if name in derived.DESCRIPTIONS or name in ("schlieren", "grad_rho", "Q", "lambda2", "vorticity", "T",
                                                            "Cp", "p0_ratio", "speed", "mach"):
                    derived.add_to_grid(g, [name], freestream=self.fs)
                    self._vis = None
            ok = ok or name in g.point_data
        return ok

    def _schlieren(self, k=10.0):
        """Schlieren with one gradient scale for all grids (per-grid scaling would show the patch edges)."""
        if all("schlieren" in g.point_data for g in self.grids):
            return True
        for g in self.grids:
            if "grad_rho" not in g.point_data:
                derived.add_to_grid(g, ["grad_rho"], freestream=self.fs)
        allg = np.concatenate([np.asarray(g.point_data["grad_rho"]).ravel() for g in self.grids])
        allg = allg[np.isfinite(allg) & (allg > 0)]
        ref = float(np.percentile(allg, 99.0)) if allg.size else 1.0
        for g in self.grids:
            gm = np.nan_to_num(np.asarray(g.point_data["grad_rho"], float))
            g.point_data["schlieren"] = np.exp(-k * np.minimum(gm / max(ref, 1e-30), 1.0)).astype(np.float32)
        self._vis = None
        return True

    @staticmethod
    def _faces(c):
        c = np.asarray(c, float)
        if len(c) < 2:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        mid = 0.5 * (c[1:] + c[:-1])
        return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])

    def _cells(self, g):
        """The grid's real cells (faces rebuilt from the centres) with the values as cell data."""
        if isinstance(g, pv.RectilinearGrid):
            fx, fy, fz = self._faces(g.x), self._faces(g.y), self._faces(g.z)
        else:                                    # ImageData
            nx, ny, nz = g.dimensions
            o, sp = g.origin, g.spacing
            fx, fy, fz = (self._faces(o[d] + sp[d] * np.arange(n)) for d, n in enumerate((nx, ny, nz)))
        v = pv.RectilinearGrid(fx, fy, fz)
        for n in g.point_data:
            v.cell_data[n] = np.asarray(g.point_data[n])
        return v

    def visible(self):
        """Each grid without its covered cells, as cells (so base grid and patches tile exactly,
        with no gaps or overlaps), smoothed back to point data for slicing and iso-surfaces.
        Cached until a new field is added."""
        if self._vis is None:
            out = []
            for g in self.grids:
                if "covered" in g.point_data and np.any(np.asarray(g.point_data["covered"]) > 0.5):
                    v = self._cells(g).threshold(0.5, scalars="covered", invert=True, preference="cell")
                    if v.n_cells:
                        out.append(v.cell_data_to_point_data())
                elif self.patches:
                    out.append(self._cells(g).cell_data_to_point_data())
                else:
                    out.append(g)                # single grid: the original point grid
            self._vis = out
        return self._vis

    def _merge(self, pieces):
        pieces = [p for p in pieces if p is not None and p.n_points]
        if not pieces:
            return pv.PolyData()
        m = pieces[0] if len(pieces) == 1 else pieces[0].merge(pieces[1:])
        return m.extract_surface() if not isinstance(m, pv.PolyData) else m

    def slice(self, normal, origin):
        return self._merge([v.slice(normal=normal, origin=origin) for v in self.visible()])

    def contour(self, value, scalars):
        return self._merge([v.contour([value], scalars=scalars) for v in self.visible()])

    def sample_onto(self, mesh):
        """Field values at the points of `mesh`, finest grid wins."""
        out = mesh.sample(self.base) if self.base is not None else mesh
        for g in self.patches:
            s = mesh.sample(g)
            ok = np.asarray(s.point_data["vtkValidPointMask"]) > 0
            if "covered" in s.point_data:
                ok &= np.asarray(s.point_data["covered"]) < 0.5
            if not ok.any():
                continue
            for n in s.point_data:
                if n in out.point_data and n != "vtkValidPointMask":
                    a = np.array(out.point_data[n])
                    a[ok] = np.asarray(s.point_data[n])[ok]
                    out.point_data[n] = a
        return out

    def bounds(self):
        return self.base.bounds if self.base is not None else (0, 1, 0, 1, 0, 1)

    def field_names(self):
        if self.base is None:
            return []
        return [n for n in self.base.point_data if n not in ("velocity", "covered")]
