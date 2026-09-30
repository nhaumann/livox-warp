"""A prior map on the GPU for nearest-point queries: a dense 5 cm grid over the map's extent.

Per cell the grid holds
  - `start`: the map points sorted by cell, as a prefix array (points of cell c are start[c] .. start[c+1]),
    so the exact nearest map point within one cell (5 cm) of a query is a scan of the 27 cells around it;
  - `nn`: the map point nearest to the cell centre, jump-flooded from the points over a band of ~0.8 m, so
    queries farther from the map still get a nearest point (within ~1 cm at 10 cm, ~2 cm farther out);
  - `df`: that distance in cm as a byte (255 = beyond the band): the distance field the global search scores
    hypotheses with.
A 5M-point scan of a 17 x 21 x 9 m building is 35M cells, ~450 MB.

Used by the global localisation (localize.py: the distance field scores hypotheses, the nearest points feed
the ICP) and by the viewer's Changes colouring (PriorGrid.distances: every visible point's distance to the map).
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from . import gpu

CELL = 0.05  # grid cell (m)
IDX_BITS = 23  # map point index bits in a seed key; the distance to the cell centre (mm) sits above them
IDX_SHIFT = wp.constant(IDX_BITS)
IDX_MASK = wp.constant((1 << IDX_BITS) - 1)
MAX_POINTS = 1 << IDX_BITS
EMPTY = wp.constant(2**31 - 1)  # an unclaimed cell: above every seed key (prepare() keeps indices below MAX_POINTS - 1)
JFA_STEPS = (8, 4, 2, 1, 1)  # cells: a band of ~0.8 m around every surface
PLANAR = wp.constant(0.5)  # planarity above which a map point's normal is trusted as a surface normal


@wp.struct
class Grid:
    origin: wp.vec3
    cell: float
    inv_cell: float
    nx: int
    ny: int
    nz: int


@wp.func
def grid_flat(g: Grid, ix: int, iy: int, iz: int) -> int:
    if ix < 0 or iy < 0 or iz < 0 or ix >= g.nx or iy >= g.ny or iz >= g.nz:
        return -1
    return (iz * g.ny + iy) * g.nx + ix


@wp.func
def grid_at(g: Grid, w: wp.vec3) -> int:
    return grid_flat(g, int(wp.floor((w[0] - g.origin[0]) * g.inv_cell)),
                     int(wp.floor((w[1] - g.origin[1]) * g.inv_cell)),
                     int(wp.floor((w[2] - g.origin[2]) * g.inv_cell)))


@wp.func
def cell_centre(g: Grid, c: int) -> wp.vec3:
    ix = c % g.nx
    r = c / g.nx
    iy = r % g.ny
    iz = r / g.ny
    return g.origin + wp.vec3((float(ix) + 0.5) * g.cell, (float(iy) + 0.5) * g.cell, (float(iz) + 0.5) * g.cell)


@wp.func
def nearest_point(g: Grid, start: wp.array(dtype=wp.int32), nn: wp.array(dtype=wp.int32), xyz: wp.array(dtype=wp.vec3),
                  w: wp.vec3) -> int:
    """The map point nearest to w: exact when it lies within one cell (5 cm; every such point is in the 27 cells
    around w), otherwise the closest of those cells' jump-flooded seeds (within ~1 cm at 10 cm), or -1 beyond
    the band."""
    ix = int(wp.floor((w[0] - g.origin[0]) * g.inv_cell))
    iy = int(wp.floor((w[1] - g.origin[1]) * g.inv_cell))
    iz = int(wp.floor((w[2] - g.origin[2]) * g.inv_cell))
    best = int(-1)
    bd = g.cell * g.cell
    seed = int(-1)
    sd = float(1.0e30)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c = grid_flat(g, ix + dx, iy + dy, iz + dz)
                if c >= 0:
                    for k in range(start[c], start[c + 1]):
                        d = w - xyz[k]
                        d2 = wp.dot(d, d)
                        if d2 < bd:
                            bd = d2
                            best = k
                    if best < 0:
                        s = nn[c]
                        if s >= 0:
                            d = w - xyz[s]
                            d2 = wp.dot(d, d)
                            if d2 < sd:
                                sd = d2
                                seed = s
    if best >= 0:
        return best
    return seed


@wp.kernel
def k_seed(xyz: wp.array(dtype=wp.vec3), g: Grid, nn: wp.array(dtype=wp.int32)):
    """Each map point claims its cell; the one nearest the cell centre wins (key = mm << IDX_SHIFT | index)."""
    i = wp.tid()
    p = xyz[i]
    c = grid_at(g, p)
    if c < 0:
        return
    dmm = wp.min(int(wp.length(p - cell_centre(g, c)) * 1000.0), 255)
    wp.atomic_min(nn, c, (dmm << IDX_SHIFT) | i)


@wp.kernel
def k_unpack(nn: wp.array(dtype=wp.int32)):
    """Keep only the point index of each winning seed key; unclaimed cells become -1."""
    c = wp.tid()
    v = nn[c]
    if v == EMPTY:
        nn[c] = -1
    else:
        nn[c] = v & IDX_MASK


@wp.kernel
def k_jfa(nn: wp.array(dtype=wp.int32), xyz: wp.array(dtype=wp.vec3), g: Grid, step: int):
    """Jump flooding: take the nearest of the seeds held `step` cells away in the 26 directions (in place: a
    neighbour read mid-update is still a valid seed, just one found sooner)."""
    c = wp.tid()
    p = cell_centre(g, c)
    ix = c % g.nx
    r = c / g.nx
    iy = r % g.ny
    iz = r / g.ny
    best = nn[c]
    bd = float(1.0e30)
    if best >= 0:
        d = p - xyz[best]
        bd = wp.dot(d, d)
    changed = int(0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                j = grid_flat(g, ix + dx * step, iy + dy * step, iz + dz * step)
                if j >= 0 and j != c:
                    s = nn[j]
                    if s >= 0 and s != best:
                        d = p - xyz[s]
                        d2 = wp.dot(d, d)
                        if d2 < bd:
                            bd = d2
                            best = s
                            changed = 1
    if changed != 0:
        nn[c] = best


@wp.kernel
def k_df(nn: wp.array(dtype=wp.int32), xyz: wp.array(dtype=wp.vec3), g: Grid, df: wp.array(dtype=wp.uint8)):
    """The distance field: each cell centre's distance to its seed in cm, 255 beyond the band."""
    c = wp.tid()
    s = nn[c]
    if s < 0:
        df[c] = wp.uint8(255)
        return
    d = wp.length(cell_centre(g, c) - xyz[s]) * 100.0
    df[c] = wp.uint8(wp.min(int(d + 0.5), 255))


@wp.kernel
def k_changes_T(
    pts: wp.array(dtype=wp.vec3),  # the viewer's world frame
    n: int,
    T: wp.mat44,  # world -> map
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32),
):
    """Distance of every point (placed in the map by T) to the nearest prior-map point; 1e3 when nothing is
    mapped within the grid's band (~0.8 m)."""
    i = wp.tid()
    if i >= n:
        return
    w = wp.transform_point(T, pts[i])
    s = nearest_point(g, start, nn, xyz, w)
    d = float(1.0e3)
    if s >= 0:
        d = wp.length(w - xyz[s])
    out[i] = d


def horizontal_levels(xyz: np.ndarray, normal: np.ndarray, planarity: np.ndarray, bin_m: float = 0.02):
    """Histogram of the heights of a map's horizontal surfaces (points with a near-vertical, trusted normal)."""
    nz = np.abs(normal[:, 2].astype(np.float32))
    sel = (nz > 0.97) & (planarity.astype(np.float32) > PLANAR)
    lo, hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    edges = np.arange(lo - 0.5, hi + 0.5 + bin_m, bin_m)
    h, _ = np.histogram(xyz[sel, 2], bins=edges)
    return h.astype(np.float32), float(edges[0]), bin_m


class PriorGrid:
    """A prior map on the GPU: points (sorted by cell), unit normals, planarity, and the grid (see the module).

    Construction only allocates, on the calling thread's current stream (construct it on the thread that owns the
    default stream, or inside a WarpWorker scope()); prepare() is the host work and build() the GPU work, on an
    explicit stream. After build() nothing writes the arrays again, so any stream may read them.
    """

    @staticmethod
    def prepare(host: dict, cell: float = CELL, pad: float = 0.6) -> dict:
        """Host preparation, a few seconds for millions of points (run it off the render thread): float32 arrays
        sorted by grid cell, unit normals, the grid geometry and cell prefix, horizontal levels."""
        if host.get("_prepared"):
            return host
        xyz = np.asarray(host["xyz"], dtype=np.float32)
        n = len(xyz)
        if n >= MAX_POINTS:
            raise ValueError(f"prior map has {n:,} points; the grid seeds hold at most {MAX_POINTS - 1:,}")
        nrm = np.asarray(host["normal"], dtype=np.float32)
        planar = np.asarray(host["planarity"] if "planarity" in host else np.ones(n), dtype=np.float32)
        lo, hi = xyz.min(0), xyz.max(0)
        origin = (lo - pad).astype(np.float32)
        dims = np.ceil((hi + pad - origin) / cell).astype(np.int64)
        ijk = np.floor((xyz - origin) / cell).astype(np.int64)
        key = (ijk[:, 2] * dims[1] + ijk[:, 1]) * dims[0] + ijk[:, 0]
        order = np.argsort(key, kind="stable")
        ncell = int(np.prod(dims))
        counts = np.bincount(key, minlength=ncell)
        start = np.zeros(ncell + 1, np.int32)
        np.cumsum(counts, out=start[1:])
        xyz = np.ascontiguousarray(xyz[order])
        nrm = nrm[order]
        nrm = np.ascontiguousarray(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6), dtype=np.float32)
        planar = np.ascontiguousarray(planar[order])
        return {"_prepared": True, "xyz": xyz, "normal": nrm, "planarity": planar, "lo": lo, "hi": hi,
                "origin": origin, "cell": float(cell), "dims": dims, "start": start,
                "levels": horizontal_levels(xyz, nrm, planar)}

    def __init__(self, host: dict, device=None):
        self.device = wp.get_device(device)
        d = self.device
        host = self.prepare(host)
        self._host = host
        self.n = len(host["xyz"])
        self.lo, self.hi = host["lo"], host["hi"]
        self.dims = host["dims"]
        self.ncell = int(np.prod(self.dims))
        g = Grid()
        g.origin = wp.vec3(*host["origin"])
        g.cell, g.inv_cell = host["cell"], 1.0 / host["cell"]
        g.nx, g.ny, g.nz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])
        self.g = g
        self.levels = host["levels"]
        self.xyz = wp.empty(self.n, dtype=wp.vec3, device=d)
        self.nrm = wp.empty(self.n, dtype=wp.vec3, device=d)
        self.planar = wp.empty(self.n, dtype=wp.float32, device=d)
        self.start = wp.empty(self.ncell + 1, dtype=wp.int32, device=d)
        self.nn = wp.empty(self.ncell, dtype=wp.int32, device=d)
        self.df = wp.empty(self.ncell, dtype=wp.uint8, device=d)
        self.ready = False
        self.build_s = 0.0

    def gpu_bytes(self) -> int:
        return self.n * (12 + 12 + 4) + self.ncell * (4 + 4 + 1)

    def build(self, stream):
        """Upload the points and build the grid, every op on `stream`; returns when it is done."""
        t0 = time.perf_counter()
        d = self.device
        h = self._host

        def up(dst, src, dtype):
            wp.copy(dst, wp.array(src, dtype=dtype, device="cpu", copy=False), stream=stream)

        up(self.xyz, h["xyz"], wp.vec3)
        up(self.nrm, h["normal"], wp.vec3)
        up(self.planar, h["planarity"], wp.float32)
        up(self.start, h["start"], wp.int32)
        wp.launch(gpu.k_fill_i32, dim=self.ncell, inputs=[self.nn, EMPTY], device=d, stream=stream)
        wp.launch(k_seed, dim=self.n, inputs=[self.xyz, self.g, self.nn], device=d, stream=stream)
        wp.launch(k_unpack, dim=self.ncell, inputs=[self.nn], device=d, stream=stream)
        for step in JFA_STEPS:
            wp.launch(k_jfa, dim=self.ncell, inputs=[self.nn, self.xyz, self.g, int(step)], device=d, stream=stream)
        wp.launch(k_df, dim=self.ncell, inputs=[self.nn, self.xyz, self.g, self.df], device=d, stream=stream)
        if stream is not None:
            wp.synchronize_stream(stream)
        else:
            wp.synchronize_device(d)
        self.build_s = time.perf_counter() - t0
        self.ready = True

    def distances(self, pts, n: int, T_map_from_pts: np.ndarray, out, stream):
        """Distance to the map of the first n points of `pts` (given in a frame that T maps into the map) into
        `out`, on `stream`. Read-only on the grid, so any stream may call it once build() has returned."""
        if n <= 0:
            return
        T = wp.mat44(*np.asarray(T_map_from_pts, dtype=np.float32).flatten())
        wp.launch(k_changes_T, dim=n, inputs=[pts, n, T, self.g, self.start, self.nn, self.xyz, out],
                  device=self.device, stream=stream)
