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
    """Limited slope from the two one-sided differences. kind bits 0-1: 0 van Leer, 1 minmod,
    2 Koren (kappa = 1/3, third-order upwind-biased where smooth); for Koren, a is the
    difference away from the face and b the one across it."""
    ab = a * b
    if ab <= 0.0:
        return 0.0
    k = kind & 3
    if k == 1:
        return wp.sign(a) * wp.min(wp.abs(a), wp.abs(b))
    if k == 2:
        aa = wp.abs(a)
        bb = wp.abs(b)
        return wp.sign(a) * wp.min(wp.min(2.0 * bb, (aa + 2.0 * bb) / 3.0), 2.0 * aa)
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
            qR[c] = qb[c] - 0.5 * order * zr * lim(qp[c] - qb[c], dC, kind)
        if qL[0] <= 0.0 or qL[4] <= 0.0 or qR[0] <= 0.0 or qR[4] <= 0.0:
            qL = qa
            qR = qb
        if (kind & 4) != 0 and hlle == 0:
            # low-Mach correction (Thornber et al., J. Comput. Phys. 227, 2008): scale the
            # reconstructed velocity jump by the local Mach number, so the upwind dissipation
            # falls with Mach number instead of swamping slow flow (stagnation regions, wakes).
            # Off where the shock switch is on.
            cL = wp.sqrt(g * qL[4] / qL[0])
            cR = wp.sqrt(g * qR[4] / qR[0])
            mL = wp.sqrt(qL[1] * qL[1] + qL[2] * qL[2] + qL[3] * qL[3]) / cL
            mR = wp.sqrt(qR[1] * qR[1] + qR[2] * qR[2] + qR[3] * qR[3]) / cR
            z = wp.min(1.0, wp.max(mL, mR))
            for c in range(1, 4):
                avg = 0.5 * (qL[c] + qR[c])
                jmp = 0.5 * z * (qL[c] - qR[c])
                qL[c] = avg + jmp
                qR[c] = avg - jmp
    return riemann(qL, qR, d, g, hlle)


@wp.func
def weno5z(a: float, b: float, c: float, d: float, e: float):
    """Fifth-order WENO-Z (Borges et al. 2008) value at the face between c and d from the
    upwind-biased stencil a b c | d e (the scheme of NASA LAVA's Cartesian solver; Kiris et al.,
    Aerosp. Sci. Technol. 55, 2016). Uniform-grid weights."""
    b0 = 13.0 / 12.0 * (a - 2.0 * b + c) * (a - 2.0 * b + c) + 0.25 * (a - 4.0 * b + 3.0 * c) * (a - 4.0 * b + 3.0 * c)
    b1 = 13.0 / 12.0 * (b - 2.0 * c + d) * (b - 2.0 * c + d) + 0.25 * (b - d) * (b - d)
    b2 = 13.0 / 12.0 * (c - 2.0 * d + e) * (c - 2.0 * d + e) + 0.25 * (3.0 * c - 4.0 * d + e) * (3.0 * c - 4.0 * d + e)
    t5 = wp.abs(b0 - b2)
    eps = 1.0e-12 * (c * c + d * d) + 1.0e-30
    a0 = 0.1 * (1.0 + t5 / (b0 + eps))
    a1 = 0.6 * (1.0 + t5 / (b1 + eps))
    a2 = 0.3 * (1.0 + t5 / (b2 + eps))
    p0 = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0
    p1 = (-b + 5.0 * c + 2.0 * d) / 6.0
    p2 = (2.0 * c + 5.0 * d - e) / 6.0
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)


@wp.func
def weno_face_flux(q2: vec5, qm: vec5, qa: vec5, qb: vec5, qp: vec5, q3: vec5, d: int, g: float,
                   kind: int, hlle: int):
    """Flux at the face between qa and qb with WENO5-Z reconstruction of the primitive variables
    from the six cells q2 qm qa | qb qp q3. Falls back to MUSCL if a state comes out non-physical."""
    qL = qa
    qR = qb
    for c in range(5):
        qL[c] = weno5z(q2[c], qm[c], qa[c], qb[c], qp[c])
        qR[c] = weno5z(q3[c], qp[c], qb[c], qa[c], qm[c])
    if qL[0] <= 0.0 or qL[4] <= 0.0 or qR[0] <= 0.0 or qR[4] <= 0.0:
        return face_flux(qm, qa, qb, qp, d, g, 1.0, kind, hlle, 1.0, 1.0)
    if (kind & 4) != 0 and hlle == 0:
        cL = wp.sqrt(g * qL[4] / qL[0])
        cR = wp.sqrt(g * qR[4] / qR[0])
        mL = wp.sqrt(qL[1] * qL[1] + qL[2] * qL[2] + qL[3] * qL[3]) / cL
        mR = wp.sqrt(qR[1] * qR[1] + qR[2] * qR[2] + qR[3] * qR[3]) / cR
        z = wp.min(1.0, wp.max(mL, mR))
        for c in range(1, 4):
            avg = 0.5 * (qL[c] + qR[c])
            jmp = 0.5 * z * (qL[c] - qR[c])
            qL[c] = avg + jmp
            qR[c] = avg - jmp
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


@wp.func
def blk_normal(b: int):
    """Plate normal packed into bits 8-31 of a blocked-face word (3 x 8 bits, 128 = 0);
    zero vector if none was stored."""
    if ((b >> 8) & 16777215) == 0:
        return wp.vec3(0.0, 0.0, 0.0)
    return wp.vec3(float((b >> 8) & 255) - 128.0, float((b >> 16) & 255) - 128.0,
                   float((b >> 24) & 255) - 128.0) / 127.0


@wp.func
def wall_flux_n(q: vec5, d: int, g: float, own_left: int, b: int):
    """Blocked-face flux using the real plate normal n (from bits 8-31 of b): the wall pressure
    comes from the velocity normal to the plate, u.n, instead of the axis component, so a
    plate lying obliquely across the grid (a staircase of blocked faces) does not see the
    flow sliding along it as an impact. Pressure only, along the face axis; summed over a
    staircase of faces this gives p * A * n. Falls back to the axis mirror without a normal."""
    if ((b >> (3 + d)) & 1) == 0:
        return wall_flux(q, d, g, own_left)       # no reliable plate normal for this face
    n = blk_normal(b)
    ln = wp.length(n)
    if ln < 0.5:
        return wall_flux(q, d, g, own_left)
    n = n / ln
    if (own_left != 0 and n[d] < 0.0) or (own_left == 0 and n[d] > 0.0):
        n = -n
    un = q[1] * n[0] + q[2] * n[1] + q[3] * n[2]       # toward the wall
    m = vec5(q[0], 0.0, 0.0, 0.0, q[4])
    if own_left != 0:
        m[1 + d] = un
    else:
        m[1 + d] = -un
    return wall_flux(m, d, g, own_left)


# ---------------------------------------------------------------- viscous terms
@wp.struct
class Visc:
    mode: int                      # 0 inviscid, 1 constant viscosity, 2 Sutherland
    mu0: float
    T0: float
    S: float
    R: float                       # gas constant (T = p / (rho R))
    Pr: float
    cx: wp.array(dtype=wp.float32)  # padded cell-centre coordinates
    cy: wp.array(dtype=wp.float32)
    cz: wp.array(dtype=wp.float32)
    turb: int                      # 1: k-omega SST eddy viscosity + wall function
    mut: wp.array3d(dtype=wp.float32)    # eddy viscosity (padded)
    dist: wp.array3d(dtype=wp.float32)   # wall distance (padded; large far from walls)
    fluid: wp.array3d(dtype=wp.float32)  # 1 in fluid cells (padded)


# SST constants (Menter 2003) and log law
BSTAR = wp.constant(0.09)
KAPPA = wp.constant(0.41)
A1 = wp.constant(0.31)
SK1 = wp.constant(0.85)
SK2 = wp.constant(1.0)
SW1 = wp.constant(0.5)
SW2 = wp.constant(0.856)
BETA1 = wp.constant(0.075)
BETA2 = wp.constant(0.0828)
GAM1 = wp.constant(0.5532)
GAM2 = wp.constant(0.4404)
PRT = wp.constant(0.9)
BLOG = wp.constant(5.0)


@wp.func
def qcomp(W: wp.array4d(dtype=wp.float32), i: int, j: int, k: int, c: int, R: float):
    """c = 0, 1, 2: velocity components; c = 3: temperature."""
    if c < 3:
        return W[1 + c, i, j, k]
    return W[4, i, j, k] / (W[0, i, j, k] * R)


@wp.func
def coord(vp: Visc, e: int, idx: int):
    if e == 0:
        return vp.cx[idx]
    if e == 1:
        return vp.cy[idx]
    return vp.cz[idx]


@wp.func
def cdiff(W: wp.array4d(dtype=wp.float32), vp: Visc, i: int, j: int, k: int, c: int, e: int):
    """Central derivative of component c along e at cell (i, j, k) (padded indices)."""
    di = int(0)
    dj = int(0)
    dk = int(0)
    idx = int(0)
    if e == 0:
        di = 1
        idx = i
    elif e == 1:
        dj = 1
        idx = j
    else:
        dk = 1
        idx = k
    num = qcomp(W, i + di, j + dj, k + dk, c, vp.R) - qcomp(W, i - di, j - dj, k - dk, c, vp.R)
    return num / (coord(vp, e, idx + 1) - coord(vp, e, idx - 1))


@wp.func
def face_deriv(W: wp.array4d(dtype=wp.float32), vp: Visc, ia: int, ja: int, ka: int,
               ib: int, jb: int, kb: int, d: int, c: int, e: int):
    """Derivative of component c along e on the face between cells A and B (B = A + e_d)."""
    if e == d:
        ida = ia
        idb = ib
        if d == 1:
            ida = ja
            idb = jb
        elif d == 2:
            ida = ka
            idb = kb
        return (qcomp(W, ib, jb, kb, c, vp.R) - qcomp(W, ia, ja, ka, c, vp.R)) / \
            (coord(vp, d, idb) - coord(vp, d, ida))
    return 0.5 * (cdiff(W, vp, ia, ja, ka, c, e) + cdiff(W, vp, ib, jb, kb, c, e))


@wp.func
def mu_of(vp: Visc, T: float):
    if vp.mode == 1:
        return vp.mu0
    return vp.mu0 * wp.pow(T / vp.T0, 1.5) * (vp.T0 + vp.S) / (T + vp.S)


@wp.func
def visc_flux(W: wp.array4d(dtype=wp.float32), vp: Visc, g: float, ia: int, ja: int, ka: int,
              ib: int, jb: int, kb: int, d: int):
    """Viscous flux (stress and heat conduction) through the face between A and B = A + e_d."""
    mul = 0.5 * (mu_of(vp, qcomp(W, ia, ja, ka, 3, vp.R)) + mu_of(vp, qcomp(W, ib, jb, kb, 3, vp.R)))
    mt = float(0.0)
    if vp.turb != 0:
        mt = 0.5 * (vp.mut[ia, ja, ka] + vp.mut[ib, jb, kb])
    mu = mul + mt
    kcond = g * vp.R / (g - 1.0) * (mul / vp.Pr + mt / PRT)
    div = face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, 0, 0) + face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, 1, 1) \
        + face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, 2, 2)
    t = wp.vec3()
    for j in range(3):
        tj = mu * (face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, j, d) + face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, d, j))
        if j == d:
            tj = tj - (2.0 / 3.0) * mu * div
        t[j] = tj
    uf = wp.vec3(0.5 * (W[1, ia, ja, ka] + W[1, ib, jb, kb]), 0.5 * (W[2, ia, ja, ka] + W[2, ib, jb, kb]),
                 0.5 * (W[3, ia, ja, ka] + W[3, ib, jb, kb]))
    en = wp.dot(uf, t) + kcond * face_deriv(W, vp, ia, ja, ka, ib, jb, kb, d, 3, d)
    return vec5(0.0, t[0], t[1], t[2], en)


@wp.func
def wall_visc(W: wp.array4d(dtype=wp.float32), vp: Visc, i: int, j: int, k: int, d: int, wall_high: int):
    """Shear on a thin-wall (blocked) face: no-slip, adiabatic; the wall is half a cell away."""
    mu = mu_of(vp, qcomp(W, i, j, k, 3, vp.R))
    idx = i
    if d == 1:
        idx = j
    elif d == 2:
        idx = k
    half = 0.25 * (coord(vp, d, idx + 1) - coord(vp, d, idx - 1))
    sgn = 1.0
    if wall_high != 0:
        sgn = -1.0
    t = wp.vec3()
    for c in range(3):
        if c != d:
            t[c] = sgn * mu * W[1 + c, i, j, k] / half
    return vec5(0.0, t[0], t[1], t[2], 0.0)


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
              normal: wp.array(dtype=wp.vec3), ratio: wp.array(dtype=wp.float32), noslip: int):
    """Immersed-boundary ghost cells: image-point state with the wall-normal velocity mirrored
    (the flux across the fluid / ghost face then carries no mass: the wall is watertight) and,
    on no-slip walls, the tangential velocity extrapolated linearly through zero at the wall:
    -ratio x image value (ratio = ghost / image distance from the wall, 1 for an exact mirror
    point). Scaling the normal component too made walls leak where a ghost sits on the wall."""
    t = wp.tid()
    q = vec5()
    for m in range(8):
        c = corner[t, m]
        wm = weight[t, m]
        if wm != 0.0:
            q = q + load(W, c[0], c[1], c[2]) * wm
    n = normal[t]
    r = ratio[t]
    vel = wp.vec3(q[1], q[2], q[3])
    vn = wp.dot(vel, n)
    if noslip != 0:
        vel = -r * (vel - vn * n) - vn * n           # no-slip: zero velocity at the wall
    else:
        vel = vel - 2.0 * vn * n                     # slip: mirror the normal component
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
def spalding_utau(u: float, y: float, nu: float):
    """Friction velocity from Spalding's law of the wall (valid from the viscous sublayer
    through the log layer): y+ = u+ + exp(-kB) [exp(k u+) - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6]."""
    if u <= 1.0e-12:
        return 0.0
    ut = wp.sqrt(nu * u / y)                      # viscous-sublayer guess
    if y * ut / nu > 11.0:                        # log-law guess, fixed-point
        for it in range(8):
            ut = KAPPA * u / wp.log(wp.max(9.0 * y * ut / nu, 1.0001))
    for it in range(12):                          # Newton on g(ut) = spalding(u/ut) - y ut / nu
        up = u / ut
        kp = KAPPA * up
        e = wp.exp(-KAPPA * BLOG)
        f = up + e * (wp.exp(kp) - 1.0 - kp - 0.5 * kp * kp - kp * kp * kp / 6.0) - y * ut / nu
        dfdup = 1.0 + e * KAPPA * (wp.exp(kp) - 1.0 - kp - 0.5 * kp * kp)
        df = dfdup * (-up / ut) - y / nu
        ut = wp.max(ut - f / df, 0.1 * ut)
    return ut


@wp.func
def wallfn_flux(W: wp.array4d(dtype=wp.float32), vp: Visc, i: int, j: int, k: int, d: int, wall_high: int):
    """Wall shear from the law of the wall on a face between fluid cell (i,j,k) and a wall
    (immersed-boundary ghost cell or thin plate) across the face normal e_d.

    The wall's own normal n comes from the gradient of the wall-distance field; the shear acts
    along the velocity tangential to that wall, and each face of a staircase carries the share
    |n . e_d| of it, so the faces of a stepped wall add up to tau * A along the real wall (an
    axis-aligned wall gives n = e_d and the plain formula)."""
    rho = W[0, i, j, k]
    nu = mu_of(vp, qcomp(W, i, j, k, 3, vp.R)) / rho
    gn = wp.vec3()
    for e in range(3):
        di = int(0)
        dj = int(0)
        dk = int(0)
        idx = i
        if e == 0:
            di = 1
        elif e == 1:
            dj = 1
            idx = j
        else:
            dk = 1
            idx = k
        span = coord(vp, e, idx + 1) - coord(vp, e, idx - 1)
        if span > 0.0:
            gn[e] = (vp.dist[i + di, j + dj, k + dk] - vp.dist[i - di, j - dj, k - dk]) / span
    gl = wp.length(gn)
    n = wp.vec3()
    if gl > 0.5 and gl < 2.0:
        n = gn / gl
    else:
        n[d] = 1.0
    u = wp.vec3(W[1, i, j, k], W[2, i, j, k], W[3, i, j, k])
    ut = u - wp.dot(u, n) * n
    ut_mag = wp.length(ut)
    y = wp.max(vp.dist[i, j, k], 1.0e-6)
    tau = float(0.0)
    if ut_mag > 1.0e-12:
        utau = spalding_utau(ut_mag, y, nu)
        tau = rho * utau * utau / ut_mag
    sgn = 1.0
    if wall_high != 0:
        sgn = -1.0
    share = wp.abs(n[d])
    t = sgn * tau * share * ut
    return vec5(0.0, t[0], t[1], t[2], 0.0)


@wp.func
def cell_residual(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                  ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                  iz: wp.array(dtype=wp.float32), active: wp.vec3i, g: float, order: float,
                  kind: int, blk: wp.array3d(dtype=wp.int32), vp: Visc, i: int, j: int, k: int):
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
                fl = wall_flux_n(q0, d, g, 0, blk[I - di, J - dj, K - dk])
            else:
                fl = face_flux(qm2, qm1, q0, qp1, d, g, order, kind, hl_l,
                               float(1 - bLL), float(1 - bR))
            if bR != 0:
                fr = wall_flux_n(q0, d, g, 1, blk[I, J, K])
            else:
                fr = face_flux(qm1, q0, qp1, qp2, d, g, order, kind, hl_r,
                               float(1 - bL), float(1 - bRR))
            acc = acc - (fr - fl) * inv
            if vp.mode != 0:
                vl = vec5()
                vr = vec5()
                wl = bL
                wr = bR
                if vp.turb == 1:
                    if vp.fluid[I - di, J - dj, K - dk] < 0.5:
                        wl = 1
                    if vp.fluid[I + di, J + dj, K + dk] < 0.5:
                        wr = 1
                if wl != 0:
                    if vp.turb == 1:
                        vl = wallfn_flux(W, vp, I, J, K, d, 0)
                    else:
                        vl = wall_visc(W, vp, I, J, K, d, 0)
                else:
                    vl = visc_flux(W, vp, g, I - di, J - dj, K - dk, I, J, K, d)
                if wr != 0:
                    if vp.turb == 1:
                        vr = wallfn_flux(W, vp, I, J, K, d, 1)
                    else:
                        vr = wall_visc(W, vp, I, J, K, d, 1)
                else:
                    vr = visc_flux(W, vp, g, I, J, K, I + di, J + dj, K + dk, d)
                acc = acc + (vr - vl) * inv
    return acc


@wp.kernel
def residual_kernel(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                    ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                    iz: wp.array(dtype=wp.float32), active: wp.vec3i, g: float, order: float,
                    kind: int, blk: wp.array3d(dtype=wp.int32), vp: Visc, R: wp.array4d(dtype=wp.float32)):
    i, j, k = wp.tid()
    acc = cell_residual(W, S, ix, iy, iz, active, g, order, kind, blk, vp, i, j, k)
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
                          kind: int, blk: wp.array3d(dtype=wp.int32), vp: Visc,
                          tiles: wp.array(dtype=wp.vec3i), R: wp.array4d(dtype=wp.float32)):
    c3 = tile_cell(tiles, wp.tid())
    i = c3[0]
    j = c3[1]
    k = c3[2]
    if i >= R.shape[1] or j >= R.shape[2] or k >= R.shape[3]:
        return
    acc = cell_residual(W, S, ix, iy, iz, active, g, order, kind, blk, vp, i, j, k)
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
                mask: wp.array3d(dtype=wp.float32), g: float, vp: Visc, rate: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    c = wp.sqrt(g * W[4, I, J, K] / W[0, I, J, K])
    r = float(0.0)
    s2 = float(0.0)
    if active[0] != 0:
        r = r + (wp.abs(W[1, I, J, K]) + c) * ix[i]
        s2 = s2 + ix[i] * ix[i]
    if active[1] != 0:
        r = r + (wp.abs(W[2, I, J, K]) + c) * iy[j]
        s2 = s2 + iy[j] * iy[j]
    if active[2] != 0:
        r = r + (wp.abs(W[3, I, J, K]) + c) * iz[k]
        s2 = s2 + iz[k] * iz[k]
    if vp.mode != 0:                 # explicit viscous stability limit
        mu = mu_of(vp, W[4, I, J, K] / (W[0, I, J, K] * vp.R))
        if vp.turb != 0:
            mu = mu + vp.mut[I, J, K]
        r = r + 2.0 * mu / W[0, I, J, K] * wp.max(4.0 / 3.0, g / vp.Pr) * s2
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


# ---------------------------------------------------------------- k-omega SST transport
@wp.func
def strain2(W: wp.array4d(dtype=wp.float32), vp: Visc, i: int, j: int, k: int):
    """S^2 = 2 S_ij S_ij from central velocity derivatives at cell (i, j, k)."""
    gm = wp.mat33()
    for a in range(3):
        for b in range(3):
            gm[a, b] = cdiff(W, vp, i, j, k, a, b)
    s2 = float(0.0)
    for a in range(3):
        for b in range(3):
            sab = 0.5 * (gm[a, b] + gm[b, a])
            s2 = s2 + 2.0 * sab * sab
    return s2


@wp.kernel
def mut_kernel(W: wp.array4d(dtype=wp.float32), Kt: wp.array3d(dtype=wp.float32),
               Om: wp.array3d(dtype=wp.float32), vp: Visc, out: wp.array3d(dtype=wp.float32)):
    """SST eddy viscosity mu_t = rho a1 k / max(a1 omega, S F2) on padded cells."""
    i, j, k = wp.tid()
    if i < 1 or j < 1 or k < 1 or i > W.shape[1] - 2 or j > W.shape[2] - 2 or k > W.shape[3] - 2:
        out[i, j, k] = 0.0
        return
    if vp.fluid[i, j, k] < 0.5:
        out[i, j, k] = 0.0
        return
    rho = W[0, i, j, k]
    kk = wp.max(Kt[i, j, k] / rho, 0.0)
    om = wp.max(Om[i, j, k] / rho, 1.0e-12)
    nu = mu_of(vp, qcomp(W, i, j, k, 3, vp.R)) / rho
    d = wp.max(vp.dist[i, j, k], 1.0e-6)
    arg2 = wp.max(2.0 * wp.sqrt(kk) / (BSTAR * om * d), 500.0 * nu / (d * d * om))
    f2 = wp.tanh(arg2 * arg2)
    s = wp.sqrt(strain2(W, vp, i, j, k))
    out[i, j, k] = rho * A1 * kk / wp.max(A1 * om, s * f2)


@wp.kernel
def vreman_kernel(W: wp.array4d(dtype=wp.float32), vp: Visc, cv: float, out: wp.array3d(dtype=wp.float32)):
    """Vreman (2004) subgrid eddy viscosity mu_t = rho c sqrt(B_beta / (a_ij a_ij)) with
    a_ij = du_j/dx_i, beta_ij = sum_m D_m^2 a_mi a_mj (D_m: local cell width along m). It vanishes
    in pure shear near walls and in laminar regions without damping functions. c = 0.07 (~2.5 Cs^2)."""
    i, j, k = wp.tid()
    if i < 1 or j < 1 or k < 1 or i > W.shape[1] - 2 or j > W.shape[2] - 2 or k > W.shape[3] - 2:
        out[i, j, k] = 0.0
        return
    if vp.fluid[i, j, k] < 0.5:
        out[i, j, k] = 0.0
        return
    a = wp.mat33()
    for m in range(3):
        for n in range(3):
            a[m, n] = cdiff(W, vp, i, j, k, n, m)          # du_n / dx_m
    dl = wp.vec3(0.5 * (vp.cx[i + 1] - vp.cx[i - 1]), 0.5 * (vp.cy[j + 1] - vp.cy[j - 1]),
                 0.5 * (vp.cz[k + 1] - vp.cz[k - 1]))
    aa = float(0.0)
    for m in range(3):
        for n in range(3):
            aa = aa + a[m, n] * a[m, n]
    b = wp.mat33()
    for p in range(3):
        for q in range(3):
            s = float(0.0)
            for m in range(3):
                s = s + dl[m] * dl[m] * a[m, p] * a[m, q]
            b[p, q] = s
    bb = b[0, 0] * b[1, 1] - b[0, 1] * b[0, 1] + b[0, 0] * b[2, 2] - b[0, 2] * b[0, 2] \
        + b[1, 1] * b[2, 2] - b[1, 2] * b[1, 2]
    nut = float(0.0)
    if aa > 1.0e-20 and bb > 0.0:
        nut = cv * wp.sqrt(bb / aa)
    out[i, j, k] = W[0, i, j, k] * nut


@wp.func
def tval(T: wp.array3d(dtype=wp.float32), W: wp.array4d(dtype=wp.float32), i: int, j: int, k: int, floor: float):
    return wp.max(T[i, j, k] / W[0, i, j, k], floor)


@wp.kernel
def turb_residual_kernel(W: wp.array4d(dtype=wp.float32), Kt: wp.array3d(dtype=wp.float32),
                         Om: wp.array3d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                         iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                         blk: wp.array3d(dtype=wp.int32), vp: Visc,
                         Rk: wp.array3d(dtype=wp.float32), Rw: wp.array3d(dtype=wp.float32),
                         Dk: wp.array3d(dtype=wp.float32), Dw: wp.array3d(dtype=wp.float32),
                         utau: wp.array3d(dtype=wp.float32)):
    """Explicit k / omega residuals (first-order upwind convection, SST-blended diffusion,
    production, cross diffusion), the implicit destruction coefficients, and the friction
    velocity in cells next to a wall (-1 elsewhere)."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    if vp.fluid[I, J, K] < 0.5:
        Rk[i, j, k] = 0.0
        Rw[i, j, k] = 0.0
        Dk[i, j, k] = 0.0
        Dw[i, j, k] = 0.0
        utau[i, j, k] = -1.0
        return
    rho = W[0, I, J, K]
    mul = mu_of(vp, qcomp(W, I, J, K, 3, vp.R))
    nu = mul / rho
    kk = tval(Kt, W, I, J, K, 0.0)
    om = tval(Om, W, I, J, K, 1.0e-12)
    mt = vp.mut[I, J, K]
    d = wp.max(vp.dist[I, J, K], 1.0e-6)
    near = int(0)
    gk = wp.vec3()
    gw = wp.vec3()
    kmv = wp.vec3()
    kpv = wp.vec3()
    wmv = wp.vec3()
    wpv = wp.vec3()
    okm_v = wp.vec3i()
    okp_v = wp.vec3i()
    # pass 1: neighbour values (a wall or blocked face repeats the own value) and gradients
    for e in range(3):
        if active[e] != 0:
            di = int(0)
            dj = int(0)
            dk = int(0)
            idx = int(0)
            if e == 0:
                di = 1
                idx = I
            elif e == 1:
                dj = 1
                idx = J
            else:
                dk = 1
                idx = K
            okm = (((blk[I - di, J - dj, K - dk] >> e) & 1) == 0) and vp.fluid[I - di, J - dj, K - dk] > 0.5
            okp = (((blk[I, J, K] >> e) & 1) == 0) and vp.fluid[I + di, J + dj, K + dk] > 0.5
            if not okm or not okp:
                near = 1
            kmv[e] = kk
            wmv[e] = om
            kpv[e] = kk
            wpv[e] = om
            if okm:
                okm_v[e] = 1
                kmv[e] = tval(Kt, W, I - di, J - dj, K - dk, 0.0)
                wmv[e] = tval(Om, W, I - di, J - dj, K - dk, 1.0e-12)
            if okp:
                okp_v[e] = 1
                kpv[e] = tval(Kt, W, I + di, J + dj, K + dk, 0.0)
                wpv[e] = tval(Om, W, I + di, J + dj, K + dk, 1.0e-12)
            span = coord(vp, e, idx + 1) - coord(vp, e, idx - 1)
            gk[e] = (kpv[e] - kmv[e]) / span
            gw[e] = (wpv[e] - wmv[e]) / span
    cd = 2.0 * rho * SW2 / om * wp.dot(gk, gw)
    cdkw = wp.max(cd, 1.0e-10)
    arg1 = wp.min(wp.max(wp.sqrt(kk) / (BSTAR * om * d), 500.0 * nu / (d * d * om)),
                  4.0 * rho * SW2 * kk / (cdkw * d * d))
    f1 = wp.tanh(arg1 * arg1 * arg1 * arg1)
    sk = f1 * SK1 + (1.0 - f1) * SK2
    sw = f1 * SW1 + (1.0 - f1) * SW2
    beta = f1 * BETA1 + (1.0 - f1) * BETA2
    gam = f1 * GAM1 + (1.0 - f1) * GAM2
    # pass 2: face fluxes (first-order upwind convection + blended diffusion); none through walls
    ak = float(0.0)
    aw = float(0.0)
    for e in range(3):
        if active[e] != 0:
            di = int(0)
            dj = int(0)
            dk = int(0)
            inv = float(0.0)
            idx = int(0)
            if e == 0:
                di = 1
                inv = ix[i]
                idx = I
            elif e == 1:
                dj = 1
                inv = iy[j]
                idx = J
            else:
                dk = 1
                inv = iz[k]
                idx = K
            for side in range(2):
                ok = okm_v[e]
                ia = I - di
                ja = J - dj
                ka = K - dk
                ib = I
                jb = J
                kb = K
                kA = kmv[e]
                wA = wmv[e]
                kB = kk
                wB = om
                if side == 1:
                    ok = okp_v[e]
                    ia = I
                    ja = J
                    ka = K
                    ib = I + di
                    jb = J + dj
                    kb = K + dk
                    kA = kk
                    wA = om
                    kB = kpv[e]
                    wB = wpv[e]
                if ok != 0:
                    mf = 0.5 * (W[0, ia, ja, ka] * W[1 + e, ia, ja, ka] + W[0, ib, jb, kb] * W[1 + e, ib, jb, kb])
                    ku = kB
                    wu = wB
                    if mf > 0.0:
                        ku = kA
                        wu = wA
                    dn = coord(vp, e, idx + side) - coord(vp, e, idx + side - 1)
                    mtf = 0.5 * (vp.mut[ia, ja, ka] + vp.mut[ib, jb, kb])
                    fk = mf * ku - (mul + sk * mtf) * (kB - kA) / dn
                    fw = mf * wu - (mul + sw * mtf) * (wB - wA) / dn
                    sgn = -1.0
                    if side == 0:
                        sgn = 1.0
                    ak = ak + sgn * fk * inv
                    aw = aw + sgn * fw * inv
    s2 = strain2(W, vp, I, J, K)
    praw = mt * s2
    pk = wp.min(praw, 10.0 * BSTAR * rho * kk * om)
    ratio = float(1.0)
    if praw > 1.0e-30:
        ratio = pk / praw
    pw = gam * rho * s2 * ratio
    Rk[i, j, k] = ak + pk
    Rw[i, j, k] = aw + pw + (1.0 - f1) * cd
    Dk[i, j, k] = BSTAR * om
    Dw[i, j, k] = beta * om
    ut = float(-1.0)
    if near != 0:
        vel = wp.sqrt(W[1, I, J, K] * W[1, I, J, K] + W[2, I, J, K] * W[2, I, J, K] + W[3, I, J, K] * W[3, I, J, K])
        ut = spalding_utau(vel, d, nu)
    utau[i, j, k] = ut


@wp.kernel
def turb_update_kernel(K0: wp.array3d(dtype=wp.float32), O0: wp.array3d(dtype=wp.float32),
                       Ka: wp.array3d(dtype=wp.float32), Oa: wp.array3d(dtype=wp.float32),
                       Rk: wp.array3d(dtype=wp.float32), Rw: wp.array3d(dtype=wp.float32),
                       Dk: wp.array3d(dtype=wp.float32), Dw: wp.array3d(dtype=wp.float32),
                       dt: wp.array(dtype=wp.float32), dtl: wp.array3d(dtype=wp.float32), local: int,
                       vp: Visc, W: wp.array4d(dtype=wp.float32), utau: wp.array3d(dtype=wp.float32),
                       a0: float, a1: float, kmin: float, wmin: float,
                       outK: wp.array3d(dtype=wp.float32), outO: wp.array3d(dtype=wp.float32)):
    """Point-implicit RK stage for rho k and rho omega; near-wall cells take omega (and, above
    y+ = 11, k) from the wall function."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    if vp.fluid[I, J, K] < 0.5:
        outK[I, J, K] = K0[I, J, K]
        outO[I, J, K] = O0[I, J, K]
        return
    h = dt[0]
    if local != 0:
        h = dtl[i, j, k]
    ks = (Ka[I, J, K] + h * Rk[i, j, k]) / (1.0 + h * Dk[i, j, k])
    ws = (Oa[I, J, K] + h * Rw[i, j, k]) / (1.0 + h * Dw[i, j, k])
    kn = a0 * K0[I, J, K] + a1 * ks
    wn = a0 * O0[I, J, K] + a1 * ws
    rho = W[0, I, J, K]
    kn = wp.max(kn, rho * kmin)
    wn = wp.max(wn, rho * wmin)
    ut = utau[i, j, k]
    if ut >= 0.0:
        y = wp.max(vp.dist[I, J, K], 1.0e-6)
        nu = mu_of(vp, qcomp(W, I, J, K, 3, vp.R)) / rho
        wv = 6.0 * nu / (BETA1 * y * y)
        wl = ut / (wp.sqrt(BSTAR) * KAPPA * y)
        wn = rho * wp.sqrt(wv * wv + wl * wl)
        if y * ut / nu > 11.0:
            kn = rho * ut * ut / wp.sqrt(BSTAR)
    outK[I, J, K] = kn
    outO[I, J, K] = wn


# ---------------------------------------------------------------- implicit (DP-LUR) steady solver
@wp.func
def cons_at(U: wp.array4d(dtype=wp.float32), i: int, j: int, k: int):
    return vec5(U[0, i, j, k], U[1, i, j, k], U[2, i, j, k], U[3, i, j, k], U[4, i, j, k])


@wp.func
def c2p(u: vec5, g: float):
    r = wp.max(u[0], 1.0e-12)
    vx = u[1] / r
    vy = u[2] / r
    vz = u[3] / r
    p = wp.max((g - 1.0) * (u[4] - 0.5 * r * (vx * vx + vy * vy + vz * vz)), 1.0e-12)
    return vec5(r, vx, vy, vz, p)


@wp.func
def flux_of_cons(u: vec5, d: int, g: float):
    return pflux(c2p(u, g), u, d)


@wp.func
def lam_speed(W: wp.array4d(dtype=wp.float32), vp: Visc, g: float, I: int, J: int, K: int, d: int, inv: float):
    """Spectral radius along d as a speed: |u_d| + c, plus 2 nu_eff max(4/3, g/Pr) / dx for
    the viscous terms (so that lam * inv is the per-cell rate)."""
    c = wp.sqrt(g * W[4, I, J, K] / W[0, I, J, K])
    lam = wp.abs(W[1 + d, I, J, K]) + c
    if vp.mode != 0:
        mu = mu_of(vp, W[4, I, J, K] / (W[0, I, J, K] * vp.R))
        if vp.turb != 0:
            mu = mu + vp.mut[I, J, K]
        lam = lam + 2.0 * mu / W[0, I, J, K] * wp.max(4.0 / 3.0, g / vp.Pr) * inv
    return lam


@wp.func
def dir_info(ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32),
             d: int, i: int, j: int, k: int):
    """(di, dj, dk), 1/dx at the cell, position along d."""
    if d == 0:
        return wp.vec3i(1, 0, 0), ix[i], i
    if d == 1:
        return wp.vec3i(0, 1, 0), iy[j], j
    return wp.vec3i(0, 0, 1), iz[k], k


@wp.func
def inv_at(ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32),
           d: int, p: int):
    if d == 0:
        return ix[p]
    if d == 1:
        return iy[p]
    return iz[p]


@wp.kernel
def implicit_sweep_kernel(U: wp.array4d(dtype=wp.float32), W: wp.array4d(dtype=wp.float32),
                          R: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                          iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                          blk: wp.array3d(dtype=wp.int32), mask: wp.array3d(dtype=wp.float32),
                          wst: wp.array3d(dtype=wp.float32), vp: Visc,
                          g: float, cfl: float, ddiag: float, first: int, color: int,
                          dUo: wp.array4d(dtype=wp.float32), dUn: wp.array4d(dtype=wp.float32)):
    """One Jacobi sweep of the backward-Euler system (1/dt + dF/dU) dU = R with a matrix-free,
    first-order Rusanov Jacobian (data-parallel LU relaxation, Wright, Candler & Bose 1996):

        D dU_i = R_i - sum_d 1/dx [ 0.5 (dF(i+1) - lam(i+1) dU(i+1)) - 0.5 (dF(i-1) + lam(i-1) dU(i-1)) ]

    with D = (1 + 1/cfl) sum_d lam_d / dx_d, i.e. the local pseudo time step dt = cfl / sum(lam/dx).
    dF(n) = F(U_n + dU_n) - F(U_n) is the exact flux change. Boundary, wall and thin-wall
    neighbours are treated explicitly (no coupling). first = 1 starts from dU = 0.
    ddiag adds a0 / dt_phys to the diagonal in dual time stepping (0 for steady runs).
    color = 0 / 1: red-black Gauss-Seidel half sweep in place (dUo is dUn; only cells with
    (i + j + k) % 2 == color are updated, from their neighbours of the other colour); -1: Jacobi."""
    i, j, k = wp.tid()
    if color >= 0 and (i + j + k) % 2 != color:
        return
    I = i + 2
    J = j + 2
    K = k + 2
    if mask[i, j, k] < 0.5:
        for c in range(5):
            dUn[c, I, J, K] = 0.0
        return
    s = float(0.0)
    off = vec5()
    for d in range(3):
        if active[d] != 0:
            e, inv, pos = dir_info(ix, iy, iz, d, i, j, k)
            s = s + lam_speed(W, vp, g, I, J, K, d, inv) * inv
            if first == 0:
                for side in range(2):
                    sg = 1
                    if side == 1:
                        sg = -1
                    p = pos + sg
                    if p >= 0 and p < mask.shape[d]:
                        ni = i + sg * e[0]
                        nj = j + sg * e[1]
                        nk = k + sg * e[2]
                        bi = I
                        bj = J
                        bk = K
                        if sg < 0:
                            bi = I - e[0]
                            bj = J - e[1]
                            bk = K - e[2]
                        if ((blk[bi, bj, bk] >> d) & 1) == 0 and mask[ni, nj, nk] > 0.5:
                            du = cons_at(dUo, ni + 2, nj + 2, nk + 2)
                            un = cons_at(U, ni + 2, nj + 2, nk + 2)
                            dF = flux_of_cons(un + du, d, g) - flux_of_cons(un, d, g)
                            ln = lam_speed(W, vp, g, ni + 2, nj + 2, nk + 2, d, inv_at(ix, iy, iz, d, p))
                            off = off + (float(sg) * dF - du * ln) * (0.5 * inv)
    dg = s / cfl + s + ddiag + wst[i, j, k]
    for c in range(5):
        dUn[c, I, J, K] = (R[c, i, j, k] - off[c]) / dg


@wp.kernel
def lam_kernel(W: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
               iz: wp.array(dtype=wp.float32), active: wp.vec3i, vp: Visc, g: float,
               lam: wp.array4d(dtype=wp.float32)):
    """Spectral radius (as a speed, lam_speed) of every interior cell along each direction, for
    implicit_sweep_lam_kernel."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    lam[0, i, j, k] = 0.0
    lam[1, i, j, k] = 0.0
    lam[2, i, j, k] = 0.0
    if active[0] != 0:
        lam[0, i, j, k] = lam_speed(W, vp, g, I, J, K, 0, ix[i])
    if active[1] != 0:
        lam[1, i, j, k] = lam_speed(W, vp, g, I, J, K, 1, iy[j])
    if active[2] != 0:
        lam[2, i, j, k] = lam_speed(W, vp, g, I, J, K, 2, iz[k])


@wp.kernel
def implicit_sweep_lam_kernel(U: wp.array4d(dtype=wp.float32), lam: wp.array4d(dtype=wp.float32),
                          R: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                          iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                          blk: wp.array3d(dtype=wp.int32), mask: wp.array3d(dtype=wp.float32),
                          wst: wp.array3d(dtype=wp.float32),
                          g: float, cfl: float, ddiag: float, first: int, color: int,
                          dUo: wp.array4d(dtype=wp.float32), dUn: wp.array4d(dtype=wp.float32)):
    """implicit_sweep_kernel with the spectral radii lam[d] (lam_kernel) read from a cache made
    once per iteration instead of recomputed in every sweep (same arithmetic).

    One Jacobi sweep of the backward-Euler system (1/dt + dF/dU) dU = R with a matrix-free,
    first-order Rusanov Jacobian (data-parallel LU relaxation, Wright, Candler & Bose 1996):

        D dU_i = R_i - sum_d 1/dx [ 0.5 (dF(i+1) - lam(i+1) dU(i+1)) - 0.5 (dF(i-1) + lam(i-1) dU(i-1)) ]

    with D = (1 + 1/cfl) sum_d lam_d / dx_d, i.e. the local pseudo time step dt = cfl / sum(lam/dx).
    dF(n) = F(U_n + dU_n) - F(U_n) is the exact flux change. Boundary, wall and thin-wall
    neighbours are treated explicitly (no coupling). first = 1 starts from dU = 0.
    ddiag adds a0 / dt_phys to the diagonal in dual time stepping (0 for steady runs).
    color = 0 / 1: red-black Gauss-Seidel half sweep in place (dUo is dUn; only cells with
    (i + j + k) % 2 == color are updated, from their neighbours of the other colour); -1: Jacobi."""
    i, j, k = wp.tid()
    if color >= 0 and (i + j + k) % 2 != color:
        return
    I = i + 2
    J = j + 2
    K = k + 2
    if mask[i, j, k] < 0.5:
        for c in range(5):
            dUn[c, I, J, K] = 0.0
        return
    s = float(0.0)
    off = vec5()
    for d in range(3):
        if active[d] != 0:
            e, inv, pos = dir_info(ix, iy, iz, d, i, j, k)
            s = s + lam[d, i, j, k] * inv
            if first == 0:
                for side in range(2):
                    sg = 1
                    if side == 1:
                        sg = -1
                    p = pos + sg
                    if p >= 0 and p < mask.shape[d]:
                        ni = i + sg * e[0]
                        nj = j + sg * e[1]
                        nk = k + sg * e[2]
                        bi = I
                        bj = J
                        bk = K
                        if sg < 0:
                            bi = I - e[0]
                            bj = J - e[1]
                            bk = K - e[2]
                        if ((blk[bi, bj, bk] >> d) & 1) == 0 and mask[ni, nj, nk] > 0.5:
                            du = cons_at(dUo, ni + 2, nj + 2, nk + 2)
                            un = cons_at(U, ni + 2, nj + 2, nk + 2)
                            dF = flux_of_cons(un + du, d, g) - flux_of_cons(un, d, g)
                            ln = lam[d, ni, nj, nk]
                            off = off + (float(sg) * dF - du * ln) * (0.5 * inv)
    dg = s / cfl + s + ddiag + wst[i, j, k]
    for c in range(5):
        dUn[c, I, J, K] = (R[c, i, j, k] - off[c]) / dg


@wp.kernel
def implicit_apply_kernel(U: wp.array4d(dtype=wp.float32), dU: wp.array4d(dtype=wp.float32),
                          mask: wp.array3d(dtype=wp.float32), g: float, maxrel: float):
    """U += dU on fluid cells, scaled down where density or pressure would change by more than
    maxrel (keeps large pseudo time steps positive)."""
    i, j, k = wp.tid()
    if mask[i, j, k] < 0.5:
        return
    I = i + 2
    J = j + 2
    K = k + 2
    u0 = cons_at(U, I, J, K)
    du = cons_at(dU, I, J, K)
    q0 = c2p(u0, g)
    u1 = u0 + du
    r1 = u1[0]
    p1 = (g - 1.0) * (u1[4] - 0.5 * (u1[1] * u1[1] + u1[2] * u1[2] + u1[3] * u1[3]) / wp.max(r1, 1.0e-30))
    m = wp.max(wp.abs(r1 - q0[0]) / q0[0], wp.abs(p1 - q0[4]) / q0[4])
    f = float(1.0)
    if m > maxrel or r1 <= 0.0 or p1 <= 0.0:
        f = maxrel / wp.max(m, maxrel)
    for c in range(5):
        U[c, I, J, K] = u0[c] + f * du[c]


@wp.kernel
def turb_rate_kernel(W: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                     iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                     vp: Visc, cfl: float, dtl: wp.array3d(dtype=wp.float32)):
    """Local explicit-stable step for the k / omega transport (upwind convection at |u| plus
    diffusion with nu + nu_t), used for the turbulence update in implicit runs."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    rho = W[0, I, J, K]
    nu = (mu_of(vp, W[4, I, J, K] / (rho * vp.R)) + vp.mut[I, J, K]) / rho
    r = float(0.0)
    for d in range(3):
        if active[d] != 0:
            e, inv, pos = dir_info(ix, iy, iz, d, i, j, k)
            r = r + wp.abs(W[1 + d, I, J, K]) * inv + 2.0 * nu * inv * inv
    dtl[i, j, k] = cfl / wp.max(r, 1.0e-30)


# ---------------------------------------------------------------- dual time stepping (BDF1 / BDF2)
@wp.kernel
def dual_time_kernel(U: wp.array4d(dtype=wp.float32), Un: wp.array4d(dtype=wp.float32),
                     Um: wp.array4d(dtype=wp.float32), R: wp.array4d(dtype=wp.float32),
                     mask: wp.array3d(dtype=wp.float32), a0: float, a1: float, a2: float, idt: float):
    """Unsteady residual R* = R - (a0 U + a1 U^n + a2 U^(n-1)) / dt for the inner pseudo-time
    iterations (BDF2: 1.5, -2, 0.5; BDF1: 1, -1, 0)."""
    i, j, k = wp.tid()
    if mask[i, j, k] < 0.5:
        return
    I = i + 2
    J = j + 2
    K = k + 2
    for c in range(5):
        R[c, i, j, k] = R[c, i, j, k] - (a0 * U[c, I, J, K] + a1 * Un[c, I, J, K] + a2 * Um[c, I, J, K]) * idt


@wp.kernel
def dual_time_turb_kernel(Kn: wp.array3d(dtype=wp.float32), On: wp.array3d(dtype=wp.float32),
                          Km: wp.array3d(dtype=wp.float32), Om: wp.array3d(dtype=wp.float32),
                          Rk: wp.array3d(dtype=wp.float32), Rw: wp.array3d(dtype=wp.float32),
                          Dk: wp.array3d(dtype=wp.float32), Dw: wp.array3d(dtype=wp.float32),
                          a0: float, a1: float, a2: float, idt: float):
    """Physical-time terms for the point-implicit k / omega update: the a0 part joins the
    implicit destruction coefficient, the old time levels the explicit residual."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    Rk[i, j, k] = Rk[i, j, k] - (a1 * Kn[I, J, K] + a2 * Km[I, J, K]) * idt
    Rw[i, j, k] = Rw[i, j, k] - (a1 * On[I, J, K] + a2 * Om[I, J, K]) * idt
    Dk[i, j, k] = Dk[i, j, k] + a0 * idt
    Dw[i, j, k] = Dw[i, j, k] + a0 * idt


# ---------------------------------------------------------------- implicit k-omega (Jacobi sweeps)
@wp.kernel
def turb_sweep_kernel(W: wp.array4d(dtype=wp.float32), Kt: wp.array3d(dtype=wp.float32),
                      Om: wp.array3d(dtype=wp.float32), Rk: wp.array3d(dtype=wp.float32),
                      Rw: wp.array3d(dtype=wp.float32), Dk: wp.array3d(dtype=wp.float32),
                      Dw: wp.array3d(dtype=wp.float32), dtl: wp.array3d(dtype=wp.float32), vp: Visc,
                      ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                      iz: wp.array(dtype=wp.float32), active: wp.vec3i, blk: wp.array3d(dtype=wp.int32),
                      first: int, dKo: wp.array3d(dtype=wp.float32), dOo: wp.array3d(dtype=wp.float32),
                      dKn: wp.array3d(dtype=wp.float32), dOn: wp.array3d(dtype=wp.float32)):
    """Jacobi sweep for the backward-Euler k / omega update in pseudo time (step dtl):
    (1/h + D + a_P / rho) d(rho k)_i = R_i - D (rho k)_i + sum_nb c_nb d(rho k)_nb / rho_nb
    with the first-order upwind convection + diffusion coefficients of turb_residual_kernel
    (sigma taken as 1 in the operator) and the destruction coefficient D. Arrays are padded."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    if vp.fluid[I, J, K] < 0.5:
        dKn[I, J, K] = 0.0
        dOn[I, J, K] = 0.0
        return
    rho = W[0, I, J, K]
    mul = mu_of(vp, qcomp(W, I, J, K, 3, vp.R))
    mt = vp.mut[I, J, K]
    aP = float(0.0)
    sk = float(0.0)
    sw = float(0.0)
    for e in range(3):
        if active[e] != 0:
            d3, inv, pos = dir_info(ix, iy, iz, e, i, j, k)
            idx = pos + 2
            for side in range(2):
                sg = -1
                if side == 1:
                    sg = 1
                NI = I + sg * d3[0]
                NJ = J + sg * d3[1]
                NK = K + sg * d3[2]
                bi = I
                bj = J
                bk = K
                if side == 0:
                    bi = NI
                    bj = NJ
                    bk = NK
                if ((blk[bi, bj, bk] >> e) & 1) == 0 and vp.fluid[NI, NJ, NK] > 0.5:
                    mf = 0.5 * (W[0, I, J, K] * W[1 + e, I, J, K] + W[0, NI, NJ, NK] * W[1 + e, NI, NJ, NK])
                    dn = coord(vp, e, idx + side) - coord(vp, e, idx + side - 1)
                    gd = (mul + 0.5 * (mt + vp.mut[NI, NJ, NK])) / dn
                    outf = wp.max(mf, 0.0)
                    inf = wp.max(-mf, 0.0)
                    if side == 0:
                        outf = wp.max(-mf, 0.0)
                        inf = wp.max(mf, 0.0)
                    aP = aP + (outf + gd) * inv
                    p = pos + sg
                    if first == 0 and p >= 0 and p < dtl.shape[e]:
                        c = (inf + gd) * inv / W[0, NI, NJ, NK]
                        sk = sk + c * dKo[NI, NJ, NK]
                        sw = sw + c * dOo[NI, NJ, NK]
    h = dtl[i, j, k]
    rk = Rk[i, j, k] - Dk[i, j, k] * Kt[I, J, K]
    rw = Rw[i, j, k] - Dw[i, j, k] * Om[I, J, K]
    dKn[I, J, K] = (rk + sk) / (1.0 / h + Dk[i, j, k] + aP / rho)
    dOn[I, J, K] = (rw + sw) / (1.0 / h + Dw[i, j, k] + aP / rho)


@wp.kernel
def turb_apply_kernel(Kt: wp.array3d(dtype=wp.float32), Om: wp.array3d(dtype=wp.float32),
                      dK: wp.array3d(dtype=wp.float32), dO: wp.array3d(dtype=wp.float32),
                      W: wp.array4d(dtype=wp.float32), vp: Visc, utau: wp.array3d(dtype=wp.float32),
                      kmin: float, wmin: float):
    """rho k, rho omega += update (floored; near-wall cells from the wall function, as in
    turb_update_kernel)."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    if vp.fluid[I, J, K] < 0.5:
        return
    rho = W[0, I, J, K]
    kn = wp.max(Kt[I, J, K] + dK[I, J, K], rho * kmin)
    wn = wp.max(Om[I, J, K] + dO[I, J, K], rho * wmin)
    ut = utau[i, j, k]
    if ut >= 0.0:
        y = wp.max(vp.dist[I, J, K], 1.0e-6)
        nu = mu_of(vp, qcomp(W, I, J, K, 3, vp.R)) / rho
        wv = 6.0 * nu / (BETA1 * y * y)
        wl = ut / (wp.sqrt(BSTAR) * KAPPA * y)
        wn = rho * wp.sqrt(wv * wv + wl * wl)
        if y * ut / nu > 11.0:
            kn = rho * ut * ut / wp.sqrt(BSTAR)
    Kt[I, J, K] = kn
    Om[I, J, K] = wn


@wp.kernel
def wall_stiff_kernel(W: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32),
                      iy: wp.array(dtype=wp.float32), iz: wp.array(dtype=wp.float32), active: wp.vec3i,
                      blk: wp.array3d(dtype=wp.int32), vp: Visc, out: wp.array3d(dtype=wp.float32)):
    """Implicit diagonal of the wall-function shear (turbulent walls) per unit time: the law-of-
    the-wall stress grows roughly as u^2, so d(shear force / volume) / d(rho u) ~ 2 tau / (rho u_t)
    per wall face, times the face's share |n . e_d| and 1/dx. Without it the explicit wall shear
    makes near-wall cells overshoot at large pseudo time steps."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    acc = float(0.0)
    if vp.fluid[I, J, K] > 0.5:
        rho = W[0, I, J, K]
        for d in range(3):
            if active[d] != 0:
                e, inv, pos = dir_info(ix, iy, iz, d, i, j, k)
                for side in range(2):
                    wall = int(0)
                    if side == 0:
                        if ((blk[I - e[0], J - e[1], K - e[2]] >> d) & 1) != 0 or vp.fluid[I - e[0], J - e[1], K - e[2]] < 0.5:
                            wall = 1
                    else:
                        if ((blk[I, J, K] >> d) & 1) != 0 or vp.fluid[I + e[0], J + e[1], K + e[2]] < 0.5:
                            wall = 1
                    if wall != 0:
                        f = wallfn_flux(W, vp, I, J, K, d, side)
                        tm = wp.sqrt(f[1] * f[1] + f[2] * f[2] + f[3] * f[3])
                        ut = wp.sqrt(W[1, I, J, K] * W[1, I, J, K] + W[2, I, J, K] * W[2, I, J, K]
                                     + W[3, I, J, K] * W[3, I, J, K])
                        if ut > 1.0e-6:
                            acc = acc + 2.0 * tm / (rho * ut) * inv
    out[i, j, k] = acc


# ---------------------------------------------------------------- split residual (faces, then divergence)
# The fused residual_kernel computes every face flux twice (once from each side) and, with all
# three directions, walls and viscous terms in one thread, needs ~174 registers plus a 232-byte
# spill frame on sm_89: one 256-thread block per SM (~17 % occupancy) on an RTX 40xx. Here each
# face flux is computed once, by a kernel specialised for one direction (compile-time constant,
# so no dynamically indexed vectors) and for inviscid / viscous flow; a light divergence kernel
# then adds the wall fluxes (thin-wall faces, wall function) and writes R. Same arithmetic as
# residual_kernel (tests/split_residual.py compares the two).

def _make_face_kernel(D: int, VISC: int, WENO: int = 0):
    DI = 1 if D == 0 else 0
    DJ = 1 if D == 1 else 0
    DK = 1 if D == 2 else 0

    @wp.kernel
    def face_kernel(W: wp.array4d(dtype=wp.float32), S: wp.array3d(dtype=wp.int32),
                    blk: wp.array3d(dtype=wp.int32), g: float, order: float, kind: int, vp: Visc,
                    F: wp.array4d(dtype=wp.float32)):
        i, j, k = wp.tid()
        # left cell of the face (padded indices); the right cell is L + e_D
        I = i + 2 - wp.static(DI)
        J = j + 2 - wp.static(DJ)
        K = k + 2 - wp.static(DK)
        f = vec5()
        if ((blk[I, J, K] >> wp.static(D)) & 1) == 0:
            qm = load(W, I - wp.static(DI), J - wp.static(DJ), K - wp.static(DK))
            qa = load(W, I, J, K)
            qb = load(W, I + wp.static(DI), J + wp.static(DJ), K + wp.static(DK))
            qp = load(W, I + wp.static(2 * DI), J + wp.static(2 * DJ), K + wp.static(2 * DK))
            bLL = (blk[I - wp.static(DI), J - wp.static(DJ), K - wp.static(DK)] >> wp.static(D)) & 1
            bRR = (blk[I + wp.static(DI), J + wp.static(DJ), K + wp.static(DK)] >> wp.static(D)) & 1
            hl = wp.max(S[I, J, K], S[I + wp.static(DI), J + wp.static(DJ), K + wp.static(DK)])
            # WENO5-Z (kind bit 3) where its six-cell stencil fits inside the padded array and
            # crosses no thin wall; MUSCL elsewhere (domain / patch edges, next to fin plates)
            use_w = int(0)
            if wp.static(WENO != 0) and order > 0.0 and bLL == 0 and bRR == 0:
                ia = I * wp.static(DI) + J * wp.static(DJ) + K * wp.static(DK)
                if ia >= 2 and ia + 3 < W.shape[wp.static(D + 1)]:
                    b2L = (blk[I - wp.static(2 * DI), J - wp.static(2 * DJ), K - wp.static(2 * DK)] >> wp.static(D)) & 1
                    b2R = (blk[I + wp.static(2 * DI), J + wp.static(2 * DJ), K + wp.static(2 * DK)] >> wp.static(D)) & 1
                    if b2L == 0 and b2R == 0:
                        use_w = 1
                        # all six stencil cells in the fluid (no immersed-boundary ghost or solid
                        # cell: there the mirrored states made WENO's stagnation pressure 3.6 %
                        # low on the Mach 4 cylinder); the mask exists when vp.fluid is full size
                        if vp.fluid.shape[0] > 1:
                            for o in range(-2, 4):
                                if vp.fluid[I + o * wp.static(DI), J + o * wp.static(DJ), K + o * wp.static(DK)] < 0.5:
                                    use_w = 0
            if wp.static(WENO != 0):
                if use_w != 0:
                    q2 = load(W, I - wp.static(2 * DI), J - wp.static(2 * DJ), K - wp.static(2 * DK))
                    q3 = load(W, I + wp.static(3 * DI), J + wp.static(3 * DJ), K + wp.static(3 * DK))
                    f = weno_face_flux(q2, qm, qa, qb, qp, q3, wp.static(D), g, kind, hl)
                else:
                    f = face_flux(qm, qa, qb, qp, wp.static(D), g, order, kind, hl, float(1 - bLL), float(1 - bRR))
            else:
                f = face_flux(qm, qa, qb, qp, wp.static(D), g, order, kind, hl, float(1 - bLL), float(1 - bRR))
            if wp.static(VISC != 0):
                wall = int(0)
                if vp.turb == 1:
                    if vp.fluid[I, J, K] < 0.5 or vp.fluid[I + wp.static(DI), J + wp.static(DJ), K + wp.static(DK)] < 0.5:
                        wall = 1
                if wall == 0:
                    f = f - visc_flux(W, vp, g, I, J, K, I + wp.static(DI), J + wp.static(DJ), K + wp.static(DK),
                                      wp.static(D))
        for c in range(5):
            F[c, I, J, K] = f[c]

    return face_kernel


@wp.func
def div_dir(W: wp.array4d(dtype=wp.float32), F: wp.array4d(dtype=wp.float32), blk: wp.array3d(dtype=wp.int32),
            vp: Visc, g: float, q0: vec5, inv: float, I: int, J: int, K: int, di: int, dj: int, dk: int, d: int,
            visc: int):
    bL = (blk[I - di, J - dj, K - dk] >> d) & 1
    bR = (blk[I, J, K] >> d) & 1
    fl = load(F, I - di, J - dj, K - dk)
    fr = load(F, I, J, K)
    if bL != 0:
        fl = wall_flux_n(q0, d, g, 0, blk[I - di, J - dj, K - dk])
    if bR != 0:
        fr = wall_flux_n(q0, d, g, 1, blk[I, J, K])
    acc = (fl - fr) * inv
    if visc != 0:
        wl = bL
        wr = bR
        if vp.turb == 1:
            if vp.fluid[I - di, J - dj, K - dk] < 0.5:
                wl = 1
            if vp.fluid[I + di, J + dj, K + dk] < 0.5:
                wr = 1
        if wl != 0:
            vl = vec5()
            if vp.turb == 1:
                vl = wallfn_flux(W, vp, I, J, K, d, 0)
            else:
                vl = wall_visc(W, vp, I, J, K, d, 0)
            acc = acc - vl * inv
        if wr != 0:
            vr = vec5()
            if vp.turb == 1:
                vr = wallfn_flux(W, vp, I, J, K, d, 1)
            else:
                vr = wall_visc(W, vp, I, J, K, d, 1)
            acc = acc + vr * inv
    return acc


def _make_div_kernel(D: int, VISC: int):
    DI = 1 if D == 0 else 0
    DJ = 1 if D == 1 else 0
    DK = 1 if D == 2 else 0

    @wp.kernel
    def div_kernel(W: wp.array4d(dtype=wp.float32), inv: wp.array(dtype=wp.float32), g: float,
                   blk: wp.array3d(dtype=wp.int32), vp: Visc, F: wp.array4d(dtype=wp.float32), first: int,
                   R: wp.array4d(dtype=wp.float32)):
        """R (+)= this direction's flux difference (first: overwrite)."""
        i, j, k = wp.tid()
        I = i + 2
        J = j + 2
        K = k + 2
        q0 = load(W, I, J, K)
        ii = i
        if wp.static(D == 1):
            ii = j
        if wp.static(D == 2):
            ii = k
        acc = div_dir(W, F, blk, vp, g, q0, inv[ii], I, J, K, wp.static(DI), wp.static(DJ), wp.static(DK),
                      wp.static(D), wp.static(VISC))
        if first != 0:
            for c in range(5):
                R[c, i, j, k] = acc[c]
        else:
            for c in range(5):
                R[c, i, j, k] = R[c, i, j, k] + acc[c]

    return div_kernel


FACE_KERNELS = [[_make_face_kernel(d, v) for d in range(3)] for v in (0, 1)]
# WENO5-Z variants, compiled separately so the default kernels keep their register budget
FACE_KERNELS_WENO = [[_make_face_kernel(d, v, 1) for d in range(3)] for v in (0, 1)]
DIV_KERNELS = [[_make_div_kernel(d, v) for d in range(3)] for v in (0, 1)]


# ---------------------------------------------------------------- engine jets
@wp.kernel
def jet_kernel(W: wp.array4d(dtype=wp.float32), cells: wp.array(dtype=wp.vec3i), state: wp.array(dtype=vec5),
               group: wp.array(dtype=wp.int32), ambient: vec5, throttle: wp.array(dtype=wp.float32)):
    """Nozzle-exit inflow: the solid cells just inside each nozzle exit hold the exhaust state
    (overriding the immersed-boundary mirror), so the jet flows out through the exit plane.
    throttle[jet] 0..1 blends from the ambient state (ignition / shutdown transients)."""
    t = wp.tid()
    c = cells[t]
    q = ambient + (state[t] - ambient) * throttle[group[t]]
    for m in range(5):
        W[m, c[0], c[1], c[2]] = q[m]


@wp.kernel
def wall_cfl_kernel(lam: wp.array4d(dtype=wp.float32), ix: wp.array(dtype=wp.float32), iy: wp.array(dtype=wp.float32),
                    iz: wp.array(dtype=wp.float32), active: wp.vec3i, blk: wp.array3d(dtype=wp.int32), vp: Visc,
                    cfl: float, cfl_wall: float, out: wp.array3d(dtype=wp.float32)):
    """Extra implicit diagonal that caps the pseudo-time CFL number at cfl_wall in cells next to a
    wall-function wall (thin-wall face or immersed solid neighbour). The law-of-the-wall shear is
    explicit in the implicit operator; at large pseudo time steps these cells oscillate and the
    run stalls. Pseudo time only: the converged state is unchanged."""
    i, j, k = wp.tid()
    I = i + 2
    J = j + 2
    K = k + 2
    out[i, j, k] = 0.0
    if cfl <= cfl_wall or vp.fluid[I, J, K] < 0.5:
        return
    wall = int(0)
    s = float(0.0)
    for d in range(3):
        if active[d] != 0:
            e, inv, pos = dir_info(ix, iy, iz, d, i, j, k)
            s = s + lam[d, i, j, k] * inv
            if ((blk[I - e[0], J - e[1], K - e[2]] >> d) & 1) != 0 or ((blk[I, J, K] >> d) & 1) != 0:
                wall = 1
            if vp.fluid[I - e[0], J - e[1], K - e[2]] < 0.5 or vp.fluid[I + e[0], J + e[1], K + e[2]] < 0.5:
                wall = 1
    if wall != 0:
        out[i, j, k] = s * (1.0 / cfl_wall - 1.0 / cfl)
