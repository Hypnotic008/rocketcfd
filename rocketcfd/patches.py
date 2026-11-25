"""Local refinement with patches (composite grids) for the implicit steady solver.

A coarse base grid covers the whole domain; patches are boxes of finer cells (each its own
stretched-grid Solver, so a patch can be fine across a wall and coarse along it) placed
where resolution is needed -- typically a shell around the walls that holds the boundary
layer. Each pseudo-time iteration:

  1. every grid takes one implicit (DP-LUR) step; when a grid converts its state to
     primitive variables, the ghost layers on its patch sides are filled by trilinear
     interpolation from the finest other grid at each ghost-cell centre (the base, or a
     neighbouring patch) -- `_gather`;
  2. coarse cells that lie entirely inside a finer grid are not updated by their own grid
     but overwritten with the volume average of the fine cells inside them -- `_restrict`
     (finest grids first, so nested patches work).

Surface loads are integrated on each grid over the cells it owns (the finest grid at each
cell centre) and summed. The coupling is not exactly conservative at patch boundaries, so
put patch edges in smooth flow (outside the boundary layer, away from shocks where possible).

All grids' state arrays live in shared flat buffers (one per field), so the interpolation and
averaging are single gathers. Use `wall_patches` to generate a shell of patches around the
walls of a body, and `PatchGrid.run` (implicit) to solve.
"""
import time
import numpy as np
import torch

from .solver import Solver, NG, default_device


# ---------------------------------------------------------------- patch generation
def _lattice(lo, hi, h):
    n = int(np.ceil((hi - lo) / h - 1e-9))
    f = lo + h * np.arange(n + 1)
    f[-1] = min(f[-1], hi)
    if n >= 2 and f[-1] - f[-2] < 0.3 * h:              # avoid a sliver cell at the end
        f = np.delete(f, -2)
    return f


def wall_patches(base_faces, h, shell, sdf=None, thin_sdf=None, box=(32, 64, 64), margin=2,
                 sample=2, inner=None, thin_box=None, verbose=True):
    """Patches of cells of size h = (hx, hy, hz) covering the fluid within `shell` metres of
    the walls of `sdf` (and of `thin_sdf`, for thin parts such as grid fins).

    The domain is tiled into candidate boxes of `box` patch cells; boxes the walls do not come
    near are skipped using the distance bound of the SDF; each remaining box is shrunk to the
    bounding box of its flagged cells plus `margin` cells (sampled every `sample` cells).
    `inner` (default: three patch cells) also keeps cells this far inside thick walls, so the
    patch's immersed-boundary ghost cells and their interpolation stencils are in the patch.
    thin_sdf may be large (not a distance) away from the thin parts if `thin_box` -- their
    bounding box -- is given. Returns a list of [xf, yf, zf] face arrays.
    """
    base_faces = [np.asarray(f, float) for f in base_faces]
    lo = np.array([f[0] for f in base_faces])
    hi = np.array([f[-1] for f in base_faces])
    h = np.broadcast_to(np.asarray(h, float), (3,))
    box = np.broadcast_to(np.asarray(box, int), (3,))
    if inner is None:
        inner = 3.0 * float(h.max())
    fine = [_lattice(lo[d], hi[d], h[d]) for d in range(3)]
    nf = [len(f) - 1 for f in fine]
    starts = [np.arange(0, nf[d], box[d]) for d in range(3)]
    out = []
    t0 = time.time()
    n_cand = 0
    for i0 in starts[0]:
        for j0 in starts[1]:
            for k0 in starts[2]:
                i1, j1, k1 = min(i0 + box[0], nf[0]), min(j0 + box[1], nf[1]), min(k0 + box[2], nf[2])
                b0 = np.array([fine[0][i0], fine[1][j0], fine[2][k0]])
                b1 = np.array([fine[0][i1], fine[1][j1], fine[2][k1]])
                c = 0.5 * (b0 + b1)
                rad = 0.5 * np.linalg.norm(b1 - b0)
                near = False
                if sdf is not None:
                    ph = float(np.asarray(sdf(c[0:1], c[1:2], c[2:3])).ravel()[0])
                    near |= -rad - inner <= ph <= rad + shell
                if thin_sdf is not None:
                    if thin_box is not None:
                        tb0 = np.asarray(thin_box[0]) - shell
                        tb1 = np.asarray(thin_box[1]) + shell
                        near |= bool(np.all(b1 >= tb0) and np.all(b0 <= tb1))
                    else:
                        ph = float(np.asarray(thin_sdf(c[0:1], c[1:2], c[2:3])).ravel()[0])
                        near |= abs(ph) <= rad + shell
                if not near:
                    continue
                n_cand += 1
                # sample the box's cell centres (every `sample` cells, plus the last)
                idx = []
                for d, (a0, a1) in enumerate(((i0, i1), (j0, j1), (k0, k1))):
                    ii = np.arange(a0, a1, sample)
                    if ii[-1] != a1 - 1:
                        ii = np.append(ii, a1 - 1)
                    idx.append(ii)
                cen = [0.5 * (fine[d][idx[d]] + fine[d][idx[d] + 1]) for d in range(3)]
                X, Y, Z = np.meshgrid(*cen, indexing="ij")
                flag = np.zeros(X.shape, bool)
                if sdf is not None:
                    ph = sdf(X, Y, Z)
                    flag |= (ph > -inner) & (ph < shell)
                if thin_sdf is not None:
                    ph = thin_sdf(X, Y, Z)
                    flag |= np.abs(ph) < shell
                if not flag.any():
                    continue
                rng = []
                for d in range(3):
                    ax = tuple(e for e in range(3) if e != d)
                    on = np.flatnonzero(flag.any(axis=ax))
                    a = max(idx[d][on[0]] - margin - (sample - 1), (i0, j0, k0)[d])
                    b = min(idx[d][on[-1]] + 1 + margin + (sample - 1), (i1, j1, k1)[d])
                    rng.append((a, b))
                out.append([fine[d][rng[d][0]:rng[d][1] + 1].copy() for d in range(3)])
    if verbose:
        cells = sum(int(np.prod([len(f) - 1 for f in p])) for p in out)
        print(f"wall_patches: {len(out)} patches from {n_cand} candidate boxes, {cells/1e6:.2f} M cells, "
              f"cell size {tuple(float(v) for v in h)} m, shell {shell} m ({time.time()-t0:.1f} s)")
    return out


# ---------------------------------------------------------------- composite grid
def _gather_kernels():
    import warp as wp

    @wp.kernel
    def gather_kernel(F: wp.array(dtype=wp.float32), ncomp: int, dst: wp.array(dtype=wp.int32),
                      dstr: int, src: wp.array2d(dtype=wp.int32), sstr: wp.array(dtype=wp.int32),
                      w: wp.array2d(dtype=wp.float32)):
        t = wp.tid()
        for c in range(ncomp):
            acc = float(0.0)
            for k in range(8):
                acc = acc + w[t, k] * F[src[t, k] + c * sstr[t]]
            F[dst[t] + c * dstr] = acc

    @wp.kernel
    def restrict_kernel(F: wp.array(dtype=wp.float32), ncomp: int, dst: wp.array(dtype=wp.int32), dstr: int,
                        start: wp.array(dtype=wp.int32), src: wp.array(dtype=wp.int32), sstr: int,
                        w: wp.array(dtype=wp.float32)):
        t = wp.tid()
        for c in range(ncomp):
            acc = float(0.0)
            for e in range(start[t], start[t + 1]):
                acc = acc + w[e] * F[src[e] + c * sstr]
            F[dst[t] + c * dstr] = acc

    return gather_kernel, restrict_kernel


class PatchGrid:
    """A base grid plus refinement patches, solved together by the implicit solver.

    base_faces  : [xf, yf, zf] of the base grid (see rocketcfd.grid.stretched_faces)
    patch_faces : list of [xf, yf, zf], e.g. from `wall_patches`; patches finer than the base
    freestream, **kw : as for Solver (sdf, thin_sdf, viscous, turbulence, bcs, ...);
                  bcs apply to the base, and to patch sides that lie on the domain boundary
    """

    def __init__(self, base_faces, patch_faces, freestream, verbose=True, device=None, **kw):
        self.verbose = verbose
        faces_all = [[np.asarray(f, float) for f in base_faces]] + \
                    [[np.asarray(f, float) for f in p] for p in patch_faces]
        self.n_grids = len(faces_all)
        shapes = [tuple(len(f) - 1 + 2 * NG for f in fc) for fc in faces_all]
        sizes = np.array([int(np.prod(sh)) for sh in shapes], np.int64)
        co = np.concatenate([[0], np.cumsum(sizes)])
        if 5 * co[-1] >= 2 ** 31:
            raise ValueError(f"{co[-1]/1e6:.0f} M cells (with ghost layers) is beyond the 32-bit index range")
        self.sizes, self.co = sizes, co[:-1]
        self.device = torch.device(device) if device is not None else default_device()
        dtype = kw.get("dtype", torch.float32)
        self.bufs = {}
        N = int(co[-1])

        def make_alloc(g):
            def alloc(name, shape):
                nc = shape[0] if len(shape) == 4 else 1
                if name not in self.bufs:
                    self.bufs[name] = torch.empty(nc * N, dtype=dtype, device=self.device)
                a = int(nc * co[g])
                return self.bufs[name][a:a + nc * int(sizes[g])].view(shape)
            return alloc

        t0 = time.time()
        base_bcs = kw.pop("bcs", "auto")
        self.base = Solver(None, None, None, freestream, faces=faces_all[0], alloc=make_alloc(0),
                           verbose=verbose, device=self.device, bcs=base_bcs, **kw)
        self.grids = [self.base]
        lo, hi = self.base.lo, self.base.hi
        self._on_bnd = [None]
        for g, fc in enumerate(faces_all[1:], start=1):
            on = {}
            for d in range(3):
                for side in (0, 1):
                    edge = fc[d][0] if side == 0 else fc[d][-1]
                    on[(d, side)] = abs(edge - (lo[d] if side == 0 else hi[d])) < 1e-9 * max(1.0, abs(edge))
            self._on_bnd.append(on)
            self.grids.append(Solver(None, None, None, freestream, faces=fc, alloc=make_alloc(g),
                                     verbose=False, device=self.device, bcs=self._patch_bcs(g), **kw))
        # fineness: cell volume of each grid's smallest cells
        self.vol = np.array([float(np.prod(s.h)) for s in self.grids])
        # fineness rank (0 = finest); equal cell sizes are ordered by grid index (later = finer),
        # so overlapping patches of the same size never both own a region
        order = sorted(range(self.n_grids), key=lambda g: (self.vol[g], -g))
        self.rank = np.empty(self.n_grids, int)
        self.rank[order] = np.arange(self.n_grids)
        self._build_coupling()
        if verbose:
            cells = [int(np.prod(s.n)) for s in self.grids]
            print(f"PatchGrid: base {cells[0]/1e6:.2f} M cells + {len(cells)-1} patches "
                  f"{sum(cells[1:])/1e6:.2f} M cells; {self.n_covered} covered coarse cells; "
                  f"setup {time.time()-t0:.1f} s")

    def _patch_bcs(self, g):
        """Patch sides on the domain boundary take the base grid's condition, the rest are interfaces."""
        return {k: (self.base.bcs[k] if on else "interface") for k, on in self._on_bnd[g].items()}

    def set_freestream(self, freestream, keep_state=False):
        """New flight condition on all grids (same grid and geometry); keep_state warm-starts."""
        self.base.set_freestream(freestream, keep_state)
        for g in range(1, self.n_grids):
            s = self.grids[g]
            s._bcs_arg = self._patch_bcs(g)
            s.set_freestream(freestream, keep_state)
        self.history.clear()

    def state(self):
        """Copy of the solution (all grids), for set_state."""
        return {k: v.clone() for k, v in self.bufs.items() if k in ("U", "K", "O")}

    def set_state(self, st):
        for k, v in st.items():
            self.bufs[k].copy_(v)
        for s in self.grids:
            if s.backend == "warp":
                s.U1.copy_(s.U)

    # ------------------------------------------------------------ coupling data
    def _local_vol(self, s, X):
        """Cell volume of grid s at points X (3, N) (inf outside its interior)."""
        v = np.ones(X.shape[1])
        inside = np.ones(X.shape[1], bool)
        for d in range(3):
            f = s.faces[d]
            inside &= (X[d] >= f[0]) & (X[d] < f[-1])
            k = np.clip(np.searchsorted(f, X[d], side="right") - 1, 0, len(f) - 2)
            v *= f[k + 1] - f[k]
        v[~inside] = np.inf
        return v

    def _valid(self, s):
        """Cells of s whose state is meaningful for interpolation: fluid or IB ghost."""
        if not hasattr(s, "_valid_flat"):
            v = s.fluid.ravel().copy()
            if s.ib is not None:
                v[s.ib["g"].cpu().numpy()] = True
            s._valid_flat = v
        return s._valid_flat

    def _stencil(self, q, X):
        """Trilinear stencil (N, 8) of padded flat indices and weights in grid q's interior cells."""
        shape = tuple(m + 2 * NG for m in q.n)
        i0, fr = [], []
        for d in range(3):
            c = q.axes[d][NG:NG + q.n[d]]
            if q.n[d] == 1:
                i0.append(np.zeros(X.shape[1], np.int64))
                fr.append(np.zeros(X.shape[1]))
                continue
            k = np.clip(np.searchsorted(c, X[d]) - 1, 0, q.n[d] - 2)
            i0.append(k)
            fr.append(np.clip((X[d] - c[k]) / (c[k + 1] - c[k]), 0.0, 1.0))
        valid = self._valid(q)
        idx = np.zeros((X.shape[1], 8), np.int64)
        w = np.zeros((X.shape[1], 8))
        m = 0
        for a in (0, 1):
            for b in (0, 1):
                for cc in (0, 1):
                    ii = np.minimum(i0[0] + a, q.n[0] - 1) + NG
                    jj = np.minimum(i0[1] + b, q.n[1] - 1) + NG
                    kk = np.minimum(i0[2] + cc, q.n[2] - 1) + NG
                    flat = np.ravel_multi_index((ii, jj, kk), shape)
                    idx[:, m] = flat
                    w[:, m] = ((fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1])
                               * (fr[2] if cc else 1 - fr[2])) * valid[flat]
                    m += 1
        tot = w.sum(1)
        bad = tot < 1e-6
        w[bad] = 0.0
        w[bad, 0] = 1.0
        tot[bad] = 1.0
        return idx, w / tot[:, None]

    def _build_coupling(self):
        import warp as wp
        self._gk, self._rk = _gather_kernels()
        self._wp = wp
        grids, co, sizes = self.grids, self.co, self.sizes
        self.gather = [None] * len(grids)
        # --- interface ghost cells of each patch
        for p in range(1, len(grids)):
            s = grids[p]
            shape = tuple(m + 2 * NG for m in s.n)
            interior = np.zeros(shape, bool)
            interior[NG:-NG, NG:-NG, NG:-NG] = True
            gh = np.flatnonzero(~interior)
            ijk = np.unravel_index(gh, shape)
            X = np.stack([s.axes[d][ijk[d]] for d in range(3)])
            best = np.full(gh.size, -1)
            bv = np.full(gh.size, np.inf)
            for q in range(len(grids)):
                if q == p:
                    continue
                v = self._local_vol(grids[q], X) * (1.0 + 1e-9 * self.rank[q])
                better = v < bv
                best[better], bv[better] = q, v[better]
            keep = best >= 0
            dst, s1, s5, sstr, ws = [], [], [], [], []
            for q in np.unique(best[keep]):
                sel = np.flatnonzero(best == q)
                idx, w = self._stencil(grids[q], X[:, sel])
                dst.append(gh[sel])
                s1.append(co[q] + idx)               # 1-component buffers (K, O): co + cell
                s5.append(5 * co[q] + idx)           # 5-component buffers: 5 co + c S + cell
                sstr.append(np.full(sel.size, sizes[q]))
                ws.append(w)
            if not dst:
                continue
            dst = np.concatenate(dst)
            dev = self.device
            as_i = lambda a: torch.as_tensor(np.concatenate(a).astype(np.int32), device=dev)
            self.gather[p] = dict(
                n=dst.size,
                d1=torch.as_tensor((co[p] + dst).astype(np.int32), device=dev),
                d5=torch.as_tensor((5 * co[p] + dst).astype(np.int32), device=dev),
                s1=as_i(s1), s5=as_i(s5), sstr=as_i(sstr),
                w=torch.as_tensor(np.concatenate(ws).astype(np.float32), device=dev))
            s._iface = self._make_hook(p)
        # --- covered coarse cells and their fine-cell averages; ownership for the loads
        self.restrict = []
        self.n_covered = 0
        owned = []                                      # interior cells only: ghost layers belong to others
        for s in grids:
            o = np.zeros(tuple(m + 2 * NG for m in s.n), bool)
            o[NG:-NG, NG:-NG, NG:-NG] = True
            owned.append(o)
        order = np.argsort(self.rank)                    # finest first
        for c in order:
            fine = grids[c]
            for g in range(len(grids)):
                if g == c or self.rank[g] <= self.rank[c]:
                    continue
                coarse = grids[g]
                ent = self._restriction(coarse, fine)
                if ent is None:
                    continue
                cov, dcell, start, scell, w = ent
                # a coarse cell is covered by the finest grid that contains it: drop cells
                # already covered by a finer one (processed earlier)
                self.restrict.append(dict(g=g, c=c, cov=cov, dcell=dcell, start=start, scell=scell, w=w))
            # ownership: cells (of coarser grids) whose centre is inside this finer grid's interior
            for g in range(len(grids)):
                if g == c or self.rank[g] <= self.rank[c]:
                    continue
                s = grids[g]
                inside = np.ones(owned[g].shape, bool)
                for d in range(3):
                    a = s.axes[d]
                    m_ = (a >= fine.faces[d][0]) & (a < fine.faces[d][-1])
                    sh = [1, 1, 1]
                    sh[d] = -1
                    inside &= m_.reshape(sh)
                owned[g] &= ~inside
        # a coarse cell restricted from several fine grids keeps the finest (first) one
        seen = [set() for _ in grids]
        dev = self.device
        packed = []
        for r in self.restrict:
            g = r["g"]
            keep = np.array([cell not in seen[g] for cell in r["dcell"]], bool)
            seen[g].update(r["dcell"][keep].tolist())
            if not keep.any():
                continue
            st, sc, ww = r["start"], r["scell"], r["w"]
            counts = np.diff(st)[keep]
            sel = np.concatenate([np.arange(st[i], st[i + 1]) for i in np.flatnonzero(keep)])
            start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
            packed.append(dict(
                g=g, c=r["c"], n=int(keep.sum()), cells=r["dcell"][keep],
                d1=torch.as_tensor((co[g] + r["dcell"][keep]).astype(np.int32), device=dev),
                d5=torch.as_tensor((5 * co[g] + r["dcell"][keep]).astype(np.int32), device=dev),
                start=torch.as_tensor(start, device=dev),
                s1=torch.as_tensor((co[r["c"]] + sc[sel]).astype(np.int32), device=dev),
                s5=torch.as_tensor((5 * co[r["c"]] + sc[sel]).astype(np.int32), device=dev),
                w=torch.as_tensor(ww[sel].astype(np.float32), device=dev)))
        self.restrict = packed
        # covered cells are not updated by their own grid
        for r in self.restrict:
            s = grids[r["g"]]
            ijk = np.unravel_index(r["cells"], tuple(m + 2 * NG for m in s.n))
            ii = tuple(torch.as_tensor(v - NG, device=dev) for v in ijk)
            s.update_mask[ii] = False
            s.mask_f[ii] = 0.0
            self.n_covered += r["n"]
        for s, o in zip(grids, owned):
            s.n_fluid = max(int(s.update_mask.sum()), 1)
            s.owned = o.astype(float)

    def _restriction(self, coarse, fine):
        """Coarse interior cells entirely inside fine's interior box, with the normalised
        volume weights of the fine fluid cells whose centres lie inside each of them."""
        shape_c = tuple(m + 2 * NG for m in coarse.n)
        shape_f = tuple(m + 2 * NG for m in fine.n)
        tol = 1e-9
        per = []
        for d in range(3):
            fc = coarse.faces[d]
            ff = fine.faces[d]
            full = (fc[:-1] >= ff[0] - tol * max(1, abs(ff[0]))) & (fc[1:] <= ff[-1] + tol * max(1, abs(ff[-1])))
            if coarse.n[d] == 1 and fine.n[d] == 1:
                full = np.array([True])
            per.append(full)
        if not all(p.any() for p in per):
            return None
        # fine interior cells -> coarse cell containing the centre
        fcen = [fine.axes[d][NG:NG + fine.n[d]] for d in range(3)]
        cidx = [np.clip(np.searchsorted(coarse.faces[d], fcen[d], side="right") - 1, 0, coarse.n[d] - 1)
                for d in range(3)]
        okd = [per[d][cidx[d]] for d in range(3)]
        if not all(o.any() for o in okd):
            return None
        sel = [np.flatnonzero(okd[d]) for d in range(3)]
        I, J, K = np.meshgrid(*sel, indexing="ij")
        I, J, K = I.ravel(), J.ravel(), K.ravel()
        fflat = np.ravel_multi_index((I + NG, J + NG, K + NG), shape_f)
        vol = (np.diff(fine.faces[0])[I] * np.diff(fine.faces[1])[J] * np.diff(fine.faces[2])[K])
        vol = vol * fine.fluid.ravel()[fflat]
        cflat = np.ravel_multi_index((cidx[0][I] + NG, cidx[1][J] + NG, cidx[2][K] + NG), shape_c)
        keep = vol > 0
        cflat, fflat, vol = cflat[keep], fflat[keep], vol[keep]
        if cflat.size == 0:
            return None
        o = np.argsort(cflat, kind="stable")
        cflat, fflat, vol = cflat[o], fflat[o], vol[o]
        cells, start = np.unique(cflat, return_index=True)
        start = np.append(start, cflat.size)
        tot = np.add.reduceat(vol, start[:-1])
        w = vol / np.repeat(tot, np.diff(start))
        # only coarse cells that are fluid in the coarse grid's own view need values
        return True, cells, start, fflat, w

    def _warp_views(self, D):
        """Warp views of the torch index / weight tensors of a gather or restriction entry (cached)."""
        if "_wv" not in D:
            D["_wv"] = {k: self._wp.from_torch(v) for k, v in D.items() if isinstance(v, torch.Tensor)}
        return D["_wv"]

    def _make_hook(self, p):
        def hook(s):
            G = self.gather[p]
            v = self._warp_views(G)
            wp = self._wp
            dev = s._wdev
            st = s._wp_stream
            s._L(self._gk, G["n"], [self._wbuf("W"), 5, v["d5"], int(self.sizes[p]), v["s5"], v["sstr"], v["w"]])
            if s.turbulence:
                if "zs" not in G:
                    G["zs"] = torch.zeros_like(G["sstr"])
                    G.pop("_wv", None)
                    v = self._warp_views(G)
                for name in ("K", "O"):
                    s._L(self._gk, G["n"], [self._wbuf(name), 1, v["d1"], 0, v["s1"], v["zs"], v["w"]])
        return hook

    def _wbuf(self, name):
        if not hasattr(self, "_wbufs"):
            self._wbufs = {}
        if name not in self._wbufs:
            self._wbufs[name] = self._wp.from_torch(self.bufs[name])
        return self._wbufs[name]

    def _restrict(self):
        wp = self._wp
        s0 = self.base
        for r in self.restrict:
            g, c = r["g"], r["c"]
            v = self._warp_views(r)
            s0._L(self._rk, r["n"], [self._wbuf("U"), 5, v["d5"], int(self.sizes[g]), v["start"], v["s5"],
                                    int(self.sizes[c]), v["w"]])
            if s0.turbulence:
                for name in ("K", "O"):
                    s0._L(self._rk, r["n"], [self._wbuf(name), 1, v["d1"], 0, v["start"], v["s1"], 0, v["w"]])

    # ------------------------------------------------------------ solve
    def run(self, iters, chunk=50, first_order_steps=200, report=None, cfl_max=20.0, cfl_start=1.0,
            cfl_ramp=300, sweeps=8, snapshot=None):
        """Implicit pseudo-time iterations on all grids together (see Solver.run(implicit=True))."""
        t0 = time.time()
        done = 0
        order_by_size = list(np.argsort(-self.rank))     # coarse to fine: patches see this iteration's base
        ncell = sum(int(np.prod(s.n)) for s in self.grids)
        start_it = self.base.step_count
        with torch.no_grad():
            while done < iters:
                k = min(chunk, iters - done)
                for kk in range(k):
                    it = self.base.step_count + kk
                    order = 0.0 if it < first_order_steps else 1.0
                    ramp = min(1.0, (it - start_it) / max(cfl_ramp, 1))
                    cfl = cfl_start * (cfl_max / cfl_start) ** ramp
                    want = kk == k - 1
                    rs = []
                    for g in order_by_size:
                        r = self.grids[g]._step_implicit(order, cfl, sweeps, want_res=want)
                        rs.append((self.grids[g].n_fluid, r))
                    self._restrict()
                    if snapshot is not None and (it + 1) % snapshot[0] == 0:
                        snapshot[1](it + 1, self.t)
                done += k
                for s in self.grids:
                    s.step_count += k
                tot = sum(n for n, _ in rs)
                res = float(torch.sqrt(sum(n * r * r for n, r in rs) / tot))
                self.history.append((self.base.step_count, res))
                if not np.isfinite(res):
                    raise FloatingPointError(f"diverged at iteration {self.base.step_count}")
                if self.verbose:
                    el = time.time() - t0
                    print(f"iter {self.base.step_count:6d}  CFL {cfl:6.1f}  res(rho) = {res/self.base.fs['rho']:.3e} 1/s"
                          f"  [{ncell*done/el/1e6:.1f} Mcell-iters/s]", flush=True)
                if report and report(self):
                    break
        return self

    def run_unsteady(self, nsteps, dt, inner=30, inner_tol=1e-2, cfl=20.0, sweeps=8, order=1.0,
                     report=None, every=10, snapshot=None, bdf=2):
        """Dual time stepping on all grids together (see Solver.run_unsteady)."""
        t0 = time.time()
        order_by_size = list(np.argsort(-self.rank))
        ncell = sum(int(np.prod(s.n)) for s in self.grids)
        iters = 0
        with torch.no_grad():
            for n in range(nsteps):
                duals = [g._dual_shift(bdf, dt) for g in self.grids]
                r0 = None
                for m in range(inner):
                    rs = []
                    for gi in order_by_size:
                        g = self.grids[gi]
                        rs.append((g.n_fluid, g._step_implicit(order, cfl, sweeps, want_res=True, dual=duals[gi])))
                    self._restrict()
                    tot = sum(nn for nn, _ in rs)
                    r = float(torch.sqrt(sum(nn * rr * rr for nn, rr in rs) / tot))
                    iters += 1
                    if not np.isfinite(r):
                        raise FloatingPointError(f"diverged at physical step {self.step_count}, inner {m}")
                    r0 = r if r0 is None else r0
                    if m > 0 and r < inner_tol * r0:
                        break
                for g in self.grids:
                    g.step_count += 1
                    g.t += dt
                self.last_inner = (m + 1, r / max(r0, 1e-30))
                if snapshot is not None and self.step_count % snapshot[0] == 0:
                    snapshot[1](self.step_count, self.t)
                if (n + 1) % every == 0 or n == nsteps - 1:
                    self.history.append((self.step_count, r0))
                    if self.verbose:
                        el = time.time() - t0
                        print(f"step {self.step_count:6d}  t = {self.t:.4e} s  inner {m+1:3d} "
                              f"(res drop {r/max(r0,1e-30):.1e})  {iters/(n+1):.1f} inner/step  "
                              f"[{ncell*iters/el/1e6:.1f} Mcell-iters/s]", flush=True)
                    if report and report(self):
                        break
        return self

    @property
    def t(self):
        return self.base.t

    @property
    def history(self):
        if not hasattr(self, "_hist"):
            self._hist = []
        return self._hist

    @property
    def step_count(self):
        return self.base.step_count

    def loads(self, S_ref, L_ref, x_ref=(0.0, 0.0, 0.0), band=1.5, parts=None, friction=True):
        """Surface loads summed over the grids, each over the cells it owns (all coefficients are
        linear in the forces, so the sum of the per-grid coefficients is the total)."""
        tot = None
        for s in self.grids:
            ld = s.loads(S_ref, L_ref, x_ref, band=band, parts=parts, friction=friction)
            if tot is None:
                tot = ld
                continue
            for k, v in ld.items():
                if k == "parts":
                    for pn, pv in v.items():
                        for kk in pv:
                            tot["parts"][pn][kk] += pv[kk]
                elif k != "q_inf":
                    tot[k] = tot[k] + v
        return tot

    def fields(self, grid=0):
        """Fields of one grid (0 = base, whose covered cells hold the averaged patch solution)."""
        return self.grids[grid].fields()


# ---------------------------------------------------------------- solution-adaptive patches
def feature_cells(solver, threshold=0.05, wall_shell=0.0, field="rho"):
    """Cells of `solver` (interior) to refine: relative jump of `field` across a cell above
    `threshold` (shocks, expansion fans, shear layers), and (optionally) fluid cells within
    `wall_shell` metres of a wall. Returns (centres (3, N), half widths (3, N))."""
    from .solver import NG
    W = solver.primitive()
    q = W[{"rho": 0, "p": 4}[field]]
    jump = np.zeros_like(q)
    for d in range(3):
        if solver.n[d] < 2:
            continue
        g = np.gradient(q, solver.axes[d], axis=d)
        w = solver.widths[d].reshape([-1 if e == d else 1 for e in range(3)])
        jump = np.maximum(jump, np.abs(g) * w / np.maximum(q, 1e-30))
    flag = (jump > threshold) & solver.fluid
    if wall_shell > 0:
        flag |= solver.fluid & (solver.phi < wall_shell)
    inner = np.zeros(flag.shape, bool)
    inner[NG:-NG, NG:-NG, NG:-NG] = True
    idx = np.nonzero(flag & inner)
    cen = np.stack([solver.axes[d][idx[d]] for d in range(3)])
    half = np.stack([0.5 * solver.widths[d][idx[d]] for d in range(3)])
    return cen, half


def cell_patches(base_faces, h, cen, half, box=(48, 48, 48), margin=2, verbose=True):
    """Patches of cells of size h = (hx, hy, hz) covering the boxes cen +- half (e.g. from
    feature_cells), clustered on the same lattice and tiling as wall_patches."""
    base_faces = [np.asarray(f, float) for f in base_faces]
    lo = np.array([f[0] for f in base_faces])
    hi = np.array([f[-1] for f in base_faces])
    h = np.broadcast_to(np.asarray(h, float), (3,))
    box = np.broadcast_to(np.asarray(box, int), (3,))
    fine = [_lattice(lo[d], hi[d], h[d]) for d in range(3)]
    nf = [len(f) - 1 for f in fine]
    if cen.shape[1] == 0:
        return []
    # fine-cell index range of every flagged coarse cell
    a = [np.clip(np.searchsorted(fine[d], cen[d] - half[d], side="right") - 1 - margin, 0, nf[d] - 1)
         for d in range(3)]
    b = [np.clip(np.searchsorted(fine[d], cen[d] + half[d], side="left") + margin, 1, nf[d]) for d in range(3)]
    tile = [np.minimum(a[d] // box[d], (nf[d] - 1) // box[d]) for d in range(3)]
    out = []
    keys = np.stack(tile, 1)
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    a = [v[order] for v in a]
    b = [v[order] for v in b]
    starts = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    for grp in np.split(np.arange(keys.shape[0]), starts):
        t = keys[grp[0]]
        rng = []
        for d in range(3):
            t0, t1 = t[d] * box[d], min((t[d] + 1) * box[d], nf[d])
            lo_i = max(int(a[d][grp].min()), t0)
            hi_i = min(int(b[d][grp].max()), t1)
            if nf[d] == 1:
                lo_i, hi_i = 0, 1
            rng.append((lo_i, max(hi_i, lo_i + 1)))
        out.append([fine[d][rng[d][0]:rng[d][1] + 1].copy() for d in range(3)])
    if verbose:
        cells = sum(int(np.prod([len(f) - 1 for f in p])) for p in out)
        print(f"cell_patches: {len(out)} patches, {cells/1e6:.2f} M cells, cell size {tuple(float(v) for v in h)} m")
    return out


def init_from(new, old):
    """Initial state of every grid of `new` (PatchGrid or Solver) interpolated from the base
    grid of `old` (trilinear in primitive variables; k, omega too when turbulent)."""
    from scipy.ndimage import map_coordinates
    src = old.base if hasattr(old, "base") else old
    W = src.primitive()
    Kt = Ot = None
    if src.turbulence:
        Kt = src.K_buf.cpu().numpy().astype(np.float64)
        Ot = src.O_buf.cpu().numpy().astype(np.float64)
    grids = new.grids if hasattr(new, "grids") else [new]
    for g in grids:
        X = np.meshgrid(*g.axes, indexing="ij")
        idx = [np.interp(X[d], src.axes[d], np.arange(len(src.axes[d]))) for d in range(3)]
        Wg = np.stack([map_coordinates(W[c], idx, order=1, mode="nearest") for c in range(5)])
        g.set_primitive(Wg)
        if Kt is not None and g.turbulence:
            g.K_buf.copy_(torch.as_tensor(map_coordinates(Kt, idx, order=1, mode="nearest"), dtype=g.dtype))
            g.O_buf.copy_(torch.as_tensor(map_coordinates(Ot, idx, order=1, mode="nearest"), dtype=g.dtype))
