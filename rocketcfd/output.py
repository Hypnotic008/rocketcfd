"""Plots, VTK (ParaView) and STL export."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_slices(f, path, title="", p_inf=None, normal_axis=2, extra=None, at=None, window=None):
    """Mach and pressure on the plane normal to `normal_axis` (0=x, 1=y, 2=z), through the
    middle of the domain or at coordinate `at`. window = ((a0, a1), (b0, b1)) crops the view."""
    ax_names = "xyz"
    k = f["solid"].shape[normal_axis] // 2 if at is None else int(np.argmin(np.abs(f[ax_names[normal_axis]] - at)))
    take = lambda a: np.take(a, k, axis=normal_axis)
    inplane = [d for d in range(3) if d != normal_axis]
    c0, c1 = f[ax_names[inplane[0]]], f[ax_names[inplane[1]]]
    solid = take(f["solid"])
    mach = np.ma.masked_where(solid, take(f["mach"]))
    p = take(f["p"]) / (p_inf if p_inf else 1.0)
    p = np.ma.masked_where(solid, p)
    aspect = (c0[-1] - c0[0]) / (c1[-1] - c1[0])
    fig, axs = plt.subplots(2, 1, figsize=(12, 2 * 6 / aspect + 1.5), constrained_layout=True)
    for axp, data, lab, cmap in ((axs[0], mach, "Mach", "turbo"),
                                 (axs[1], p, "p / p_inf" if p_inf else "p [Pa]", "viridis")):
        im = axp.pcolormesh(c0, c1, data.T, cmap=cmap, shading="auto")
        axp.contour(c0, c1, take(f["phi"]).T, levels=[0.0], colors="k", linewidths=0.8)
        axp.set_aspect("equal")
        axp.set_xlabel(f"{ax_names[inplane[0]]} [m]")
        axp.set_ylabel(f"{ax_names[inplane[1]]} [m]")
        fig.colorbar(im, ax=axp, label=lab, shrink=0.9)
    if window is not None:
        for axp in axs:
            axp.set_xlim(*window[0])
            axp.set_ylim(*window[1])
    axs[0].set_title(title)
    if extra:
        extra(axs)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_vtk(f, path):
    """Legacy binary VTK rectilinear grid (handles stretched grids), for ParaView/VisIt."""
    x, y, z = f["x"], f["y"], f["z"]
    nx, ny, nz = len(x), len(y), len(z)
    with open(path, "wb") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nrocketcfd\nBINARY\nDATASET RECTILINEAR_GRID\n"
                 f"DIMENSIONS {nx} {ny} {nz}\n".encode())
        for name, c in (("X", x), ("Y", y), ("Z", z)):
            fh.write(f"{name}_COORDINATES {len(c)} float\n".encode())
            fh.write(np.asarray(c).astype(">f4").tobytes())
            fh.write(b"\n")
        fh.write(f"POINT_DATA {nx*ny*nz}\n".encode())
        for name in ("rho", "p", "mach", "phi"):
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            fh.write(np.ascontiguousarray(f[name].transpose(2, 1, 0)).astype(">f4").tobytes())
            fh.write(b"\n")
        fh.write(b"VECTORS velocity float\n")
        vel = np.stack([f["u"], f["v"], f["w"]], axis=-1).transpose(2, 1, 0, 3)
        fh.write(np.ascontiguousarray(vel).astype(">f4").tobytes())


def write_stl(sdf, lo, hi, h, path):
    """Triangulate the phi = 0 surface with marching cubes and write binary STL."""
    from skimage.measure import marching_cubes
    axes = [np.arange(lo[d], hi[d] + h, h) for d in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    phi = sdf(X, Y, Z)
    verts, faces, _, _ = marching_cubes(phi, 0.0, spacing=(h, h, h))
    verts += np.array([a[0] for a in axes])
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(b"rocketcfd".ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())
    return verts, faces


def render_mesh(verts, faces, path, title="", elev=15, azim=-60):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(nrm @ light), 0, 1)
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(tri, facecolors=plt.cm.Greys(0.25 + 0.6 * shade), edgecolor="none")
    ax.add_collection3d(pc)
    used = tri.reshape(-1, 3)
    mn, mx = used.min(0), used.max(0)
    ext = np.maximum(mx - mn, 1e-3)
    for setlim, lo_, hi_ in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mn, mx):
        setlim(lo_, hi_)
    ax.set_box_aspect(tuple(ext / ext.max()))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
