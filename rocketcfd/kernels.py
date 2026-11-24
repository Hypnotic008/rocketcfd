"""Fused NVIDIA Warp kernels for the Euler solver (float32).

One thread per cell does the whole residual (MUSCL reconstruction, HLLC/HLLE fluxes in
all three directions, shock switch), so each cell's stencil is read once instead of the
~1000 full-array passes the PyTorch reference path makes per step. Warp compiles the
same kernels for CUDA (Windows and Linux, no Visual Studio needed) and for the CPU.

The numerics are identical to rocketcfd/gas.py + Solver._residual (the PyTorch
reference); tests/compare_backends.py checks the two against each other.
"""
import warp as wp

wp.config.quiet = True
vec5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.func
def load(W: wp.array4d(dtype=wp.float32), i: int, j: int, k: int):
    return vec5(W[0, i, j, k], W[1, i, j, k], W[2, i, j, k], W[3, i, j, k], W[4, i, j, k])


@wp.func
def p2c(q: vec5, g: float):
    ke = 0.5 * q[0] * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return vec5(q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3], q[4] / (g - 1.0) + ke)


@wp.func
def pflux(q: vec5, u: vec5, d: int):
    un = q[1 + d]
    f = u * un
    f[1 + d] = f[1 + d] + q[4]
    f[4] = f[4] + q[4] * un
    return f


@wp.func
def ustar(q: vec5, u: vec5, S: float, m: float, Ss: float, d: int):
    r = m / (S - Ss + 1.0e-30)
    out = vec5(r, r * q[1], r * q[2], r * q[3], 0.0)
    out[1 + d] = r * Ss
    un = q[1 + d]
    out[4] = r * (u[4] / q[0] + (Ss - un) * (Ss + q[4] / (q[0] * (S - un) + 1.0e-30)))
    return out


@wp.func
def riemann(qL: vec5, qR: vec5, d: int, g: float, hlle: int):
    uL = p2c(qL, g)
    uR = p2c(qR, g)
    fL = pflux(qL, uL, d)
    fR = pflux(qR, uR, d)
    rL = qL[0]
    rR = qR[0]
    unL = qL[1 + d]
    unR = qR[1 + d]
    pL = qL[4]
    pR = qR[4]
    cL = wp.sqrt(g * pL / rL)
    cR = wp.sqrt(g * pR / rR)
    sL = wp.sqrt(rL)
    sR = wp.sqrt(rR)
    wt = 1.0 / (sL + sR)
    g1 = g / (g - 1.0)
    HL = g1 * pL / rL + 0.5 * (qL[1] * qL[1] + qL[2] * qL[2] + qL[3] * qL[3])
    HR = g1 * pR / rR + 0.5 * (qR[1] * qR[1] + qR[2] * qR[2] + qR[3] * qR[3])
    ux = (sL * qL[1] + sR * qR[1]) * wt
    uy = (sL * qL[2] + sR * qR[2]) * wt
    uz = (sL * qL[3] + sR * qR[3]) * wt
    utd = (sL * unL + sR * unR) * wt
    Ht = (sL * HL + sR * HR) * wt
    ct = wp.sqrt(wp.max((g - 1.0) * (Ht - 0.5 * (ux * ux + uy * uy + uz * uz)), 1.0e-12))
    SL = wp.min(unL - cL, utd - ct)
    SR = wp.max(unR + cR, utd + ct)
    if hlle != 0:
        a = wp.min(SL, 0.0)
        b = wp.max(SR, 0.0)
        return (fL * b - fR * a + (uR - uL) * (a * b)) / (b - a + 1.0e-30)
    if SL >= 0.0:
        return fL
    if SR <= 0.0:
        return fR
    mL = rL * (SL - unL)
    mR = rR * (SR - unR)
    Ss = (pR - pL + unL * mL - unR * mR) / (mL - mR + 1.0e-30)
    if Ss >= 0.0:
        return fL + (ustar(qL, uL, SL, mL, Ss, d) - uL) * SL
    return fR + (ustar(qR, uR, SR, mR, Ss, d) - uR) * SR


@wp.func
def lim(a: float, b: float, kind: int):
    ab = a * b
    if ab <= 0.0:
        return 0.0
    if kind == 1:
        return wp.sign(a) * wp.min(wp.abs(a), wp.abs(b))
    return 2.0 * ab / (a + b)


@wp.func
def face_flux(qm: vec5, qa: vec5, qb: vec5, qp: vec5, d: int, g: float, order: float,
              kind: int, hlle: int, zl: float, zr: float):
    """Flux at the face between qa and qb. zl / zr = 0 switch off the slope of qa / qb
    (used when the outer stencil cell lies across a thin wall)."""
    qL = qa
    qR = qb
    if order > 0.0:
        for c in range(5):
            dC = qb[c] - qa[c]
            qL[c] = qa[c] + 0.5 * order * zl * lim(qa[c] - qm[c], dC, kind)
            qR[c] = qb[c] - 0.5 * order * zr * lim(dC, qp[c] - qb[c], kind)
        if qL[0] <= 0.0 or qL[4] <= 0.0 or qR[0] <= 0.0 or qR[4] <= 0.0:
            qL = qa
            qR = qb
    return riemann(qL, qR, d, g, hlle)


@wp.func
def wall_flux(q: vec5, d: int, g: float, own_left: int):
    """Flux through a blocked (thin-wall) face: Riemann problem against the mirror state.
    Zero mass, energy and tangential-momentum flux; the normal momentum carries the wall pressure."""
    m = q
    m[1 + d] = -q[1 + d]
    if own_left != 0:
        return riemann(q, m, d, g, 0)
    return riemann(m, q, d, g, 0)


@wp.kernel
def prim_kernel(U: wp.array4d(dtype=wp.float32), g: float, rmin: float, pmin: float,
                W: wp.array4d(dtype=wp.float32)):
    i, j, k = wp.tid()
    r = wp.max(U[0, i, j, k], rmin)
    u = U[1, i, j, k] / r
    v = U[2, i, j, k] / r
    w = U[3, i, j, k] / r
    p = (g - 1.0) * (U[4, i, j, k] - 0.5 * r * (u * u + v * v + w * w))
    W[0, i, j, k] = r
    W[1, i, j, k] = u
    W[2, i, j, k] = v
    W[3, i, j, k] = w
    W[4, i, j, k] = wp.max(p, pmin)


@wp.kernel
def ib_kernel(W: wp.array4d(dtype=wp.float32), ghost: wp.array(dtype=wp.vec3i),
              corner: wp.array2d(dtype=wp.vec3i), weight: wp.array2d(dtype=wp.float32),
              normal: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    q = vec5()
    for m in range(8):
        c = corner[t, m]
        wm = weight[t, m]
        if wm != 0.0:
            q = q + load(W, c[0], c[1], c[2]) * wm
    n = normal[t]
    vel = wp.vec3(q[1], q[2], q[3])
    vel = vel - 2.0 * wp.dot(vel, n) * n
    gc = ghost[t]
    W[0, gc[0], gc[1], gc[2]] = q[0]
    W[1, gc[0], gc[1], gc[2]] = vel[0]
    W[2, gc[0], gc[1], gc[2]] = vel[1]
    W[3, gc[0], gc[1], gc[2]] = vel[2]
    W[4, gc[0], gc[1], gc[2]] = q[4]


@wp.kernel
def shock_kernel(W: wp.array4d(dtype=wp.float32), active: wp.vec3i, thr: float,
                 S: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    nx = W.shape[1]
    ny = W.shape[2]
    nz = W.shape[3]
    s = 0.0
    if active[0] != 0 and i > 0 and i < nx - 1:
        pp = W[4, i + 1, j, k]
        pm = W[4, i - 1, j, k]
        s = wp.max(s, wp.abs(pp - pm) / wp.min(pp, pm))
    if active[1] != 0 and j > 0 and j < ny - 1:
        pp = W[4, i, j + 1, k]
        pm = W[4, i, j - 1, k]
        s = wp.max(s, wp.abs(pp - pm) / wp.min(pp, pm))
    if active[2] != 0 and k > 0 and k < nz - 1:
        pp = W[4, i, j, k + 1]
        pm = W[4, i, j, k - 1]
        s = wp.max(s, wp.abs(pp - pm) / wp.min(pp, pm))
    flag = int(0)
    if s > thr:
        flag = 1
    S[i, j, k] = flag


@wp.func
def cell_residual(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                  ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                  iz: wp.array(dtype=wp.float32), active: wp.vec3i, g: float, order: float,
                  kind: int, blk: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    """Residual -div(F) of interior cell (i, j, k). blk[i,j,k] bit d set = the face between
    cell (i,j,k) and its +d neighbour is a thin wall."""
    I = i + 2
    J = j + 2
    K = k + 2
    acc = vec5()
    for d in range(3):
        if active[d] != 0:
            di = int(0)
            dj = int(0)
            dk = int(0)
            inv = float(0.0)
            if d == 0:
                di = 1
                inv = ix[i]
            elif d == 1:
                dj = 1
                inv = iy[j]
            else:
                dk = 1
                inv = iz[k]
            qm2 = load(W, I - 2 * di, J - 2 * dj, K - 2 * dk)
            qm1 = load(W, I - di, J - dj, K - dk)
            q0 = load(W, I, J, K)
            qp1 = load(W, I + di, J + dj, K + dk)
            qp2 = load(W, I + 2 * di, J + 2 * dj, K + 2 * dk)
            s0 = S[I, J, K]
            hl_l = wp.max(S[I - di, J - dj, K - dk], s0)
            hl_r = wp.max(s0, S[I + di, J + dj, K + dk])
            bLL = (blk[I - 2 * di, J - 2 * dj, K - 2 * dk] >> d) & 1
            bL = (blk[I - di, J - dj, K - dk] >> d) & 1
            bR = (blk[I, J, K] >> d) & 1
            bRR = (blk[I + di, J + dj, K + dk] >> d) & 1
            fl = vec5()
            fr = vec5()
            if bL != 0:
                fl = wall_flux(q0, d, g, 0)
            else:
                fl = face_flux(qm2, qm1, q0, qp1, d, g, order, kind, hl_l,
                               float(1 - bLL), float(1 - bR))
            if bR != 0:
                fr = wall_flux(q0, d, g, 1)
            else:
                fr = face_flux(qm1, q0, qp1, qp2, d, g, order, kind, hl_r,
                               float(1 - bL), float(1 - bRR))
            acc = acc - (fr - fl) * inv
    return acc


@wp.kernel
def residual_kernel(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                    ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                    iz: wp.array(dtype=wp.float32), active: wp.vec3i, g: float, order: float,
                    kind: int, blk: wp.array3d(dtype=wp.int32), R: wp.array4d(dtype=wp.float32)):
    i, j, k = wp.tid()
    acc = cell_residual(W, S, ix, iy, iz, active, g, order, kind, blk, i, j, k)
    for c in range(5):
        R[c, i, j, k] = acc[c]


TILE = 8


@wp.func
def tile_cell(tiles: wp.array(dtype=wp.vec3i), t: int):
    """Cell (i, j, k) handled by thread t of an active-tile launch (8x8x8 cells per tile)."""
    tl = tiles[t // 512]
    loc = t % 512
    return wp.vec3i(tl[0] * 8 + loc // 64, tl[1] * 8 + (loc // 8) % 8, tl[2] * 8 + loc % 8)


@wp.kernel
def residual_tiles_kernel(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                          ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                          iz: wp.array(dtype=wp.float32), active: wp.vec3i, g: float, order: float,
                          kind: int, blk: wp.array3d(dtype=wp.int32), tiles: wp.array(dtype=wp.vec3i),
                          R: wp.array4d(dtype=wp.float32)):
    c3 = tile_cell(tiles, wp.tid())
    i = c3[0]
    j = c3[1]
    k = c3[2]
    if i >= R.shape[1] or j >= R.shape[2] or k >= R.shape[3]:
        return
    acc = cell_residual(W, S, ix, iy, iz, active, g, order, kind, blk, i, j, k)
    for c in range(5):
        R[c, i, j, k] = acc[c]


@wp.kernel
def activity_kernel(R: wp.array4d(dtype=wp.float32), W: wp.array4d(dtype=wp.float32),
                    dtl: wp.array3d(dtype=wp.float32), mask: wp.array3d(dtype=wp.float32),
                    act: wp.array3d(dtype=wp.float32)):
    """Per-tile maximum relative density change per step: |dt * R_rho / rho|."""
    i, j, k = wp.tid()
    if mask[i, j, k] > 0.5:
        v = wp.abs(dtl[i, j, k] * R[0, i, j, k] / W[0, i + 2, j + 2, k + 2])
        wp.atomic_max(act, i // 8, j // 8, k // 8, v)


@wp.kernel
def rate_kernel(W: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                mask: wp.array3d(dtype=wp.float32), g: float, rate: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    c = wp.sqrt(g * W[4, I, J, K] / W[0, I, J, K])
    r = float(0.0)
    if active[0] != 0:
        r = r + (wp.abs(W[1, I, J, K]) + c) * ix[i]
    if active[1] != 0:
        r = r + (wp.abs(W[2, I, J, K]) + c) * iy[j]
    if active[2] != 0:
        r = r + (wp.abs(W[3, I, J, K]) + c) * iz[k]
    rate[i, j, k] = r * mask[i, j, k]


@wp.func
def rk_cell(U0: wp.array4d(dtype=wp.float32), Ua: wp.array4d(dtype=wp.float32),
            R: wp.array4d(dtype=wp.float32), dt: wp.array(dtype=wp.float32),
            dtl: wp.array3d(dtype=wp.float32), local: int, mask: wp.array3d(dtype=wp.float32),
            a0: float, a1: float, out: wp.array4d(dtype=wp.float32), i: int, j: int, k: int):
    I = i + 2
    J = j + 2
    K = k + 2
    m = mask[i, j, k]
    h = dt[0]
    if local != 0:
        h = dtl[i, j, k]
    for c in range(5):
        u0 = U0[c, I, J, K]
        if m > 0.5:
            out[c, I, J, K] = a0 * u0 + a1 * (Ua[c, I, J, K] + h * R[c, i, j, k])
        else:
            out[c, I, J, K] = u0


@wp.kernel
def rk_kernel(U0: wp.array4d(dtype=wp.float32), Ua: wp.array4d(dtype=wp.float32),
              R: wp.array4d(dtype=wp.float32), dt: wp.array(dtype=wp.float32),
              dtl: wp.array3d(dtype=wp.float32), local: int,
              mask: wp.array3d(dtype=wp.float32), a0: float, a1: float,
              out: wp.array4d(dtype=wp.float32)):
    """out = a0 * U0 + a1 * (Ua + dt * R) on fluid cells, U0 elsewhere (interior only).
    dt is the global step, or the per-cell step dtl when local != 0 (steady runs)."""
    i, j, k = wp.tid()
    rk_cell(U0, Ua, R, dt, dtl, local, mask, a0, a1, out, i, j, k)


@wp.kernel
def rk_tiles_kernel(U0: wp.array4d(dtype=wp.float32), Ua: wp.array4d(dtype=wp.float32),
                    R: wp.array4d(dtype=wp.float32), dt: wp.array(dtype=wp.float32),
                    dtl: wp.array3d(dtype=wp.float32), local: int,
                    mask: wp.array3d(dtype=wp.float32), a0: float, a1: float,
                    tiles: wp.array(dtype=wp.vec3i), out: wp.array4d(dtype=wp.float32)):
    c3 = tile_cell(tiles, wp.tid())
    if c3[0] >= R.shape[1] or c3[1] >= R.shape[2] or c3[2] >= R.shape[3]:
        return
    rk_cell(U0, Ua, R, dt, dtl, local, mask, a0, a1, out, c3[0], c3[1], c3[2])
