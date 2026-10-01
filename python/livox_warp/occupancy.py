"""A differentiable occupancy field learned from LiDAR rays, warm-started from the prior map.

Each cell of a dense grid holds a logit theta; along a ray the density is sigma(x) = softplus(trilinear(theta, x))
per cell length, so softplus = 1 is one unit of optical depth per cell. A ray passes a stretch [p, q] untouched
with probability exp(-tau(p, q)), tau the integral of sigma (Beer-Lambert), and a return at range r, measured
with an error of about a, has the negative log-likelihood

    tau(r0, r - a)  -  log(1 - exp(-tau(r - a, r + a)))

free space up to the return, and the ray stopping around it. The window a is measured along the surface's
normal, not the ray: a ray grazing a floor travels a long way just above it, so along the ray the window is
a / cos(incidence) (the scan's normal at the return; capped), which keeps a grazing ray's free stretch a clear
of the surface it ends on. The viewer's carving (gpu.py) settles the same
question with rules: a ray through a voxel argues against it, a return in it argues for it, a grazing ray
counts less. Here the balance is a likelihood, trilinear across cells, and wp.Tape gives its gradient for every
cell at once:

  - A dual-return firing is two sub-rays of one beam footprint, and the second echo passed where the first one
    stopped: the two pull the density there in opposite directions and it settles in between. That partial
    occupancy is what an edge or a railing is to a beam a few millimetres wide.
  - A firing that returned nothing (its direction from rosette.py) argues for free space along it, weakly (a
    black surface or a window returns nothing too) and only out to `miss_range`.
  - The prior: theta starts at the scan's occupancy and is pulled back toward it (`prior_weight`), so what no
    ray sees stays as the scan has it. The field then says what changed (classify()): cells the scan had that
    rays now pass through (REMOVED), cells it lacked that returns now stop in (ADDED), and cells returns landed
    in that rays later passed through (TRANSIENT: people, doors). change_mask() hands that to walkfit.py, which
    leaves those points out of the trajectory fit.
  - What "changed" means needs more than the field: a scan surface is REMOVED only where rays pierce it (cross its
    plane at a scan point, the viewer's carving test) and none echo, not where they merely skim the cell above a
    floor; and a new surface is ADDED only where one of the scan's stations had a line of sight to it (through the
    scan's own distance field). A surface no station could see is UNSCANNED: new to the scan, not to the place.

The quadrature has one thread per (ray, sample): each adds its slice of tau to the ray's two sums with an atomic,
so no loop carries a differentiable value and the gradient is exact (see dmath.py). The free stretch is sampled
at stratified random offsets (an unbiased estimate that moves every step), the return window at fixed points.
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

from . import dmath
from .mapgrid import PLANAR, Grid, grid_at, nearest_point

# cell codes (classify())
UNOBSERVED = wp.constant(0)
FREE = wp.constant(1)
SURFACE = wp.constant(2)
ADDED = wp.constant(3)
REMOVED = wp.constant(4)
TRANSIENT = wp.constant(5)
UNCERTAIN = wp.constant(6)
UNSCANNED = wp.constant(7)
CODE_NAMES = ("unobserved", "free", "surface", "added", "removed", "transient", "uncertain", "unscanned")
EXCLUDE_POINT = wp.constant(1)  # change_mask bit: a return landing in this cell is not on the scan
EXCLUDE_MAP = wp.constant(2)  # change_mask bit: scan points in this cell are gone
SURF_LOGIT = 2.5  # softplus 2.6: a scan surface stops 93% of the rays crossing one cell
FREE_LOGIT = -9.0  # softplus 1.2e-4: ten metres of free space let 97.5% of the rays through
THETA_MIN = wp.constant(-14.0)
THETA_MAX = wp.constant(8.0)


@wp.struct
class Volume:
    origin: wp.vec3  # corner of cell (0, 0, 0)
    cell: float
    inv_cell: float
    nx: int
    ny: int
    nz: int


@wp.struct
class ChangeMask:
    """EXCLUDE_* bits per cell of a field's volume (on = 0: no mask; the arrays are then placeholders)."""

    vol: Volume
    bits: wp.array(dtype=wp.uint8)
    on: int


@wp.func
def vol_flat(v: Volume, ix: int, iy: int, iz: int) -> int:
    if ix < 0 or iy < 0 or iz < 0 or ix >= v.nx or iy >= v.ny or iz >= v.nz:
        return -1
    return (iz * v.ny + iy) * v.nx + ix


@wp.func
def vol_cell(v: Volume, x: wp.vec3) -> int:
    g = (x - v.origin) * v.inv_cell
    return vol_flat(v, int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2])))


@wp.func
def vol_centre(v: Volume, c: int) -> wp.vec3:
    ix = c % v.nx
    r = c / v.nx
    iy = r % v.ny
    iz = r / v.ny
    return v.origin + wp.vec3((float(ix) + 0.5) * v.cell, (float(iy) + 0.5) * v.cell, (float(iz) + 0.5) * v.cell)


@wp.func
def mask_bits(m: ChangeMask, x: wp.vec3) -> int:
    if m.on == 0:
        return 0
    c = vol_cell(m.vol, x)
    if c < 0:
        return 0
    return int(m.bits[c])


@wp.func
def logit_at(theta: wp.array(dtype=float), v: Volume, x: wp.vec3, outside: float) -> float:
    """Trilinear interpolation of the cell-centred logits; `outside` beyond the volume."""
    g = (x - v.origin) * v.inv_cell - wp.vec3(0.5, 0.5, 0.5)
    fx = wp.floor(g[0])
    fy = wp.floor(g[1])
    fz = wp.floor(g[2])
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    ux = g[0] - fx
    uy = g[1] - fy
    uz = g[2] - fz
    acc = float(0.0)
    for dz in range(2):
        wz = wp.where(dz == 0, 1.0 - uz, uz)
        for dy in range(2):
            wy = wp.where(dy == 0, 1.0 - uy, uy)
            for dx in range(2):
                wx = wp.where(dx == 0, 1.0 - ux, ux)
                c = vol_flat(v, ix + dx, iy + dy, iz + dz)
                val = outside
                if c >= 0:
                    val = theta[c]
                acc += wx * wy * wz * val
    return acc


@wp.func
def rand01(seed: wp.uint32) -> float:
    h = seed * wp.uint32(747796405) + wp.uint32(2891336453)
    h = ((h >> ((h >> wp.uint32(28)) + wp.uint32(4))) ^ h) * wp.uint32(277803737)
    h = (h >> wp.uint32(22)) ^ h
    return float(h) / 4294967295.0


@wp.kernel
def k_warm(
    v: Volume,
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    thickness: float,
    surf: float,
    free: float,
    theta0: wp.array(dtype=float),
    theta: wp.array(dtype=float),
    dist_cm: wp.array(dtype=wp.uint8),
):
    """The scan's occupancy (a cell whose centre has a scan point within `thickness` is surface) and each cell's
    distance to the scan in cm (255: farther than the map grid's band, ~0.8 m)."""
    c = wp.tid()
    p = vol_centre(v, c)
    s = nearest_point(g, start, nn, xyz, p)
    val = free
    dcm = int(255)
    if s >= 0:
        dd = wp.length(p - xyz[s])
        dcm = wp.min(int(dd * 100.0 + 0.5), 255)
        if dd < thickness:
            val = surf
    theta0[c] = val
    theta[c] = val
    dist_cm[c] = wp.uint8(dcm)


@wp.kernel
def k_samples(
    theta: wp.array(dtype=float),
    v: Volume,
    outside: float,
    o: wp.array(dtype=wp.vec3),
    d: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=float),
    ids: wp.array(dtype=wp.int32),
    s_free: int,
    s_hit: int,
    r0: float,
    half: wp.array(dtype=float),
    miss_range: float,
    seed: int,
    tau_free: wp.array(dtype=float),
    tau_hit: wp.array(dtype=float),
):
    """Sample j of batch ray b adds sigma * step to the ray's free or return-window optical depth."""
    b, j = wp.tid()
    ray = ids[b]
    rr = r[ray]
    hw = half[ray]
    if j < s_free:
        hi = miss_range
        if rr >= 0.0:
            hi = rr - hw
        if hi <= r0:
            return
        h = (hi - r0) / float(s_free)
        jitter = rand01(wp.uint32(seed) * wp.uint32(2654435761) ^ wp.uint32(b * (s_free + s_hit) + j))
        s = r0 + (float(j) + jitter) * h
        sig = dmath.softplus(logit_at(theta, v, o[ray] + d[ray] * s, outside)) * v.inv_cell
        wp.atomic_add(tau_free, b, sig * h)
    else:
        if rr < 0.0:
            return
        h = 2.0 * hw / float(s_hit)
        s = rr - hw + (float(j - s_free) + 0.5) * h
        sig = dmath.softplus(logit_at(theta, v, o[ray] + d[ray] * s, outside)) * v.inv_cell
        wp.atomic_add(tau_hit, b, sig * h)


@wp.kernel
def k_windows(
    o: wp.array(dtype=wp.vec3),
    d: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=float),
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    planar: wp.array(dtype=float),
    a: float,
    max_half: float,
    half: wp.array(dtype=float),
):
    """Each return's window half-width along its ray: a / cos(incidence) against the scan's surface at the return
    (planar scan points within 10 cm), a where the scan has nothing there."""
    i = wp.tid()
    h = a
    rr = r[i]
    if rr >= 0.0:
        p = o[i] + d[i] * rr
        s = nearest_point(g, start, nn, xyz, p)
        if s >= 0:
            nv = nrm[s]
            ln = wp.length(nv)
            if wp.length(p - xyz[s]) < 0.1 and planar[s] > PLANAR and ln > 0.5:
                cs = wp.abs(wp.dot(d[i], nv)) / ln
                h = wp.min(a / wp.max(cs, 1.0e-3), max_half)
    half[i] = h


@wp.kernel
def k_ray_loss(
    r: wp.array(dtype=float),
    w: wp.array(dtype=float),
    ids: wp.array(dtype=wp.int32),
    tau_free: wp.array(dtype=float),
    tau_hit: wp.array(dtype=float),
    scale: float,
    loss: wp.array(dtype=float),
):
    """Each batch ray's negative log-likelihood (module docstring), scaled to estimate the sum over all rays."""
    b = wp.tid()
    ray = ids[b]
    nll = tau_free[b]
    if r[ray] >= 0.0:
        nll = nll - wp.log(1.0 - wp.exp(-tau_hit[b]) + 1.0e-4)
    wp.atomic_add(loss, 0, w[ray] * nll * scale)


@wp.kernel
def k_adam_theta(
    theta: wp.array(dtype=float),
    g: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    theta0: wp.array(dtype=float),
    prior_weight: float,
    lr: float,
    b1: float,
    b2: float,
    c1: float,
    c2: float,
):
    """Adam on the logits with the prior's gradient 2 lambda (theta - theta0) added here, not through the tape."""
    c = wp.tid()
    gi = g[c] + 2.0 * prior_weight * (theta[c] - theta0[c])
    if gi == 0.0 and m[c] == 0.0:
        return  # untouched and at the prior: nothing to do (most of a building's cells, most steps)
    mi = b1 * m[c] + (1.0 - b1) * gi
    vi = b2 * v[c] + (1.0 - b2) * gi * gi
    m[c] = mi
    v[c] = vi
    theta[c] = wp.clamp(theta[c] - lr * (mi / c1) / (wp.sqrt(vi / c2) + 1.0e-12), THETA_MIN, THETA_MAX)


@wp.kernel
def k_observe(
    o: wp.array(dtype=wp.vec3),
    d: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=float),
    v: Volume,
    r0: float,
    half: wp.array(dtype=float),
    dist_cm: wp.array(dtype=wp.uint8),
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    planar: wp.array(dtype=float),
    pierce_r: float,
    passes: wp.array(dtype=wp.int32),
    hits: wp.array(dtype=wp.int32),
    pierces: wp.array(dtype=wp.int32),
):
    """Per cell: returns' free stretches that crossed it, returns that landed in it, and free stretches that pierced
    a scan surface in it: crossed the plane of a planar scan point within pierce_r of the point (or passed within
    half that of a point with no plane). Not differentiated; misses do not count as observations."""
    i = wp.tid()
    rr = r[i]
    if rr < 0.0:
        return
    step = 0.5 * v.cell
    n = wp.min(int((rr - half[i] - r0) / step), 8000)
    last = int(-1)
    last_p = int(-1)
    prev = o[i] + d[i] * r0
    for k in range(n):
        x = o[i] + d[i] * (r0 + (float(k) + 0.5) * step)
        c = vol_cell(v, x)
        if c >= 0 and c != last:
            wp.atomic_add(passes, c, 1)
            last = c
        if c >= 0 and int(dist_cm[c]) <= 5:
            s = nearest_point(g, start, nn, xyz, x)
            if s >= 0:
                q = xyz[s]
                nv = nrm[s]
                ln = wp.length(nv)
                pierced = int(0)
                if planar[s] > PLANAR and ln > 0.5:
                    un = nv / ln
                    d0 = wp.dot(un, prev - q)
                    d1 = wp.dot(un, x - q)
                    if d0 * d1 <= 0.0 and d0 != d1:
                        cp = prev + (x - prev) * (d0 / (d0 - d1))
                        if wp.length(cp - q) < pierce_r:
                            pierced = 1
                else:
                    seg = x - prev
                    tt = wp.clamp(wp.dot(q - prev, seg) / wp.max(wp.dot(seg, seg), 1.0e-12), 0.0, 1.0)
                    if wp.length(prev + seg * tt - q) < 0.5 * pierce_r:
                        pierced = 1
                if pierced != 0:
                    pc = vol_cell(v, q)
                    if pc >= 0 and pc != last_p:
                        wp.atomic_add(pierces, pc, 1)
                        last_p = pc
        prev = x
    c = vol_cell(v, o[i] + d[i] * rr)
    if c >= 0:
        wp.atomic_add(hits, c, 1)


@wp.kernel
def k_station_view(
    cands: wp.array(dtype=wp.int32),
    v: Volume,
    g: Grid,
    df: wp.array(dtype=wp.uint8),
    stations: wp.array(dtype=wp.vec3),
    n_st: int,
    block_cm: int,
    codes: wp.array(dtype=wp.uint8),
):
    """A candidate ADDED cell that no station of the scan could see (every line of sight crosses a scan surface,
    found in the scan's own distance field, before reaching it) becomes UNSCANNED."""
    i = wp.tid()
    c = cands[i]
    p = vol_centre(v, c)
    seen = int(0)
    for s in range(n_st):
        if seen == 0:
            o = stations[s]
            dv = p - o
            ln = wp.length(dv)
            if ln > 0.2:
                u = dv / ln
                step = 0.5 * g.cell
                n = int((ln - 2.0 * g.cell) / step)
                blocked = int(0)
                for k in range(n):
                    if blocked == 0:
                        gc = grid_at(g, o + u * ((float(k) + 0.5) * step))
                        if gc >= 0:
                            if int(df[gc]) <= block_cm:
                                blocked = 1
                if blocked == 0:
                    seen = 1
    if seen == 0:
        codes[c] = wp.uint8(UNSCANNED)


@wp.kernel
def k_classify(
    theta: wp.array(dtype=float),
    theta0: wp.array(dtype=float),
    dist_cm: wp.array(dtype=wp.uint8),
    passes: wp.array(dtype=wp.int32),
    hits: wp.array(dtype=wp.int32),
    pierces: wp.array(dtype=wp.int32),
    occ_hi: float,
    occ_lo: float,
    min_obs: int,
    min_hits: int,
    min_pierce: int,
    added_cm: int,
    echo_fraction: float,
    codes: wp.array(dtype=wp.uint8),
):
    """ADDED: occupied now, returns landing in it, and farther than added_cm from every scan point (a surface the
    scan has, met a cell off, is not new); OccupancyField.classify then checks that the scan could have seen it.
    REMOVED: the scan's surface, now free, pierced by min_pierce rays and echoing almost none of them (a thin rail
    that rays slip past still echoes some). TRANSIENT: free now, but returns landed in it."""
    c = wp.tid()
    occ = 1.0 - wp.exp(-dmath.softplus(theta[c]))  # the chance a ray crossing the cell stops in it
    prior = theta0[c] > 0.0
    np_ = passes[c]
    nh = hits[c]
    npi = pierces[c]
    code = int(UNCERTAIN)
    if np_ + nh < min_obs:
        code = int(UNOBSERVED)
    elif occ >= occ_hi:
        code = int(SURFACE)
        if not prior and nh >= min_hits and int(dist_cm[c]) > added_cm:
            code = int(ADDED)
    elif occ <= occ_lo:
        code = int(FREE)
        if prior and npi >= min_pierce and float(nh) <= echo_fraction * float(npi):
            code = int(REMOVED)
        elif nh >= min_hits and not prior:
            code = int(TRANSIENT)
    codes[c] = wp.uint8(code)


@wp.kernel
def k_mask(codes: wp.array(dtype=wp.uint8), v: Volume, dilate: int, bits: wp.array(dtype=wp.uint8)):
    """EXCLUDE_POINT where a return must not be matched to the scan (added, transient, uncertain, unscanned, and
    the cells around them within `dilate`), EXCLUDE_MAP where the scan's points are gone (removed)."""
    c = wp.tid()
    ix = c % v.nx
    r = c / v.nx
    iy = r % v.ny
    iz = r / v.ny
    out = int(0)
    if int(codes[c]) == int(REMOVED):
        out = int(EXCLUDE_MAP)
    for dz in range(-dilate, dilate + 1):
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                j = vol_flat(v, ix + dx, iy + dy, iz + dz)
                if j >= 0:
                    k = int(codes[j])
                    if k == int(ADDED) or k == int(TRANSIENT) or k == int(UNCERTAIN) or k == int(UNSCANNED):
                        out = out | int(EXCLUDE_POINT)
    bits[c] = wp.uint8(out)


@wp.kernel
def k_occ_at(theta: wp.array(dtype=float), v: Volume, outside: float, pts: wp.array(dtype=wp.vec3),
             out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = 1.0 - wp.exp(-dmath.softplus(logit_at(theta, v, pts[i], outside)))


def ray_bounds(o: np.ndarray, d: np.ndarray, r: np.ndarray, pad: float = 0.3):
    """(lo, hi) of every origin and return of a ray set, padded."""
    hit = r >= 0
    ends = o[hit] + d[hit] * r[hit, None]
    pts = np.concatenate([o, ends]) if len(ends) else o
    return pts.min(0) - pad, pts.max(0) + pad


class OccupancyField:
    """The field over the box lo..hi (world metres) at `cell` metres; see the module docstring."""

    def __init__(self, lo, hi, cell: float = 0.05, device=None, max_cells: int = 160_000_000):
        self.device = wp.get_device(device)
        d = self.device
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        dims = np.ceil((hi - lo) / cell).astype(np.int64)
        if int(np.prod(dims)) > max_cells:
            raise ValueError(f"{dims.tolist()} cells of {cell * 100:g} cm exceed {max_cells:,}: use a coarser cell")
        self.lo, self.cell, self.dims = lo, float(cell), dims
        self.n = int(np.prod(dims))
        v = Volume()
        v.origin = wp.vec3(*lo.astype(np.float32))
        v.cell, v.inv_cell = float(cell), 1.0 / float(cell)
        v.nx, v.ny, v.nz = int(dims[0]), int(dims[1]), int(dims[2])
        self.vol = v
        self.theta = wp.full(self.n, FREE_LOGIT, dtype=float, device=d, requires_grad=True)
        self.theta0 = wp.full(self.n, FREE_LOGIT, dtype=float, device=d)
        self.m = wp.zeros(self.n, dtype=float, device=d)
        self.v = wp.zeros(self.n, dtype=float, device=d)
        self.passes = wp.zeros(self.n, dtype=wp.int32, device=d)
        self.hits = wp.zeros(self.n, dtype=wp.int32, device=d)
        self.pierces = wp.zeros(self.n, dtype=wp.int32, device=d)
        self.codes = wp.zeros(self.n, dtype=wp.uint8, device=d)
        self.dist_cm = wp.full(self.n, 255, dtype=wp.uint8, device=d)  # distance to the scan (warm_start)
        self.outside = FREE_LOGIT
        self.adam_t = 0
        self.n_rays = 0
        self.grid = None  # the scan (warm_start): its normals set each return's window
        self.stations = None  # the scan's station positions (warm_start), for what the scan could see
        self.half = None
        self._half_key = None
        self.history = []  # (epoch, mean negative log-likelihood per ray)

    @classmethod
    def around(cls, o, d, r, cell: float = 0.05, pad: float = 0.3, device=None, clip=None, **kw) -> OccupancyField:
        """A field over the box of a ray set's origins and returns (host arrays), within clip = (lo, hi) if given
        (the prior map's extent: returns through a window can land hundreds of metres out)."""
        lo, hi = ray_bounds(np.asarray(o), np.asarray(d), np.asarray(r), pad)
        if clip is not None:
            lo, hi = np.maximum(lo, clip[0]), np.minimum(hi, clip[1])
        return cls(lo, hi, cell, device, **kw)

    def gpu_bytes(self) -> int:
        return self.n * (4 * 6 + 2)

    # ---- inputs -------------------------------------------------------------------------------------

    def warm_start(self, grid=None, stations=None, surf: float = SURF_LOGIT, free: float = FREE_LOGIT,
                   thickness: float | None = None):
        """theta and the prior theta0 from a mapgrid.PriorGrid (the scan), or all free without one. A cell is surface
        when a scan point lies within half a cell of its centre: one cell thick, so the free cell above a floor is
        not called surface (and then removed by every ray grazing the floor). stations: the scan's scanner positions
        (prior map key "stations"), without which a surface new to the scan cannot be told from one it never saw."""
        self.grid = grid
        self.stations = None
        if stations is not None and len(stations):
            self.stations = wp.array(np.asarray(stations, np.float32).reshape(-1, 3), dtype=wp.vec3,
                                     device=self.device)
        self._half_key = None
        if grid is None:
            self.theta.fill_(free)
            self.theta0.fill_(free)
            return
        thick = 0.5 * self.cell if thickness is None else thickness
        wp.launch(k_warm, dim=self.n, device=self.device,
                  inputs=[self.vol, grid.g, grid.start, grid.nn, grid.xyz, thick, surf, free, self.theta0, self.theta,
                          self.dist_cm])

    def set_rays(self, o, d, r, w=None):
        """Rays as device or host arrays: origins and unit directions (world), range (< 0: no return), weight."""
        def dev(a, dtype):
            return a if isinstance(a, wp.array) else wp.array(np.ascontiguousarray(a), dtype=dtype, device=self.device)

        self.o, self.d = dev(o, wp.vec3), dev(d, wp.vec3)
        self.r = dev(r, float)
        self.n_rays = self.r.shape[0]
        self.w = dev(np.ones(self.n_rays, np.float32) if w is None else w, float)
        self._half_key = None

    def _windows(self, noise: float, max_half: float = 1.0) -> wp.array:
        """The rays' return-window half-widths (k_windows), for this noise; cached until the rays change."""
        key = (noise, max_half)
        if self._half_key == key:
            return self.half
        a = 2.0 * noise
        self.half = wp.full(self.n_rays, a, dtype=float, device=self.device)
        g = self.grid
        if g is not None:
            wp.launch(k_windows, dim=self.n_rays, device=self.device,
                      inputs=[self.o, self.d, self.r, g.g, g.start, g.nn, g.xyz, g.nrm, g.planar, a, max_half,
                              self.half])
        self._half_key = key
        return self.half

    # ---- training -----------------------------------------------------------------------------------

    def _forward(self, ids, B, s_free, s_hit, r0, half, miss_range, seed, scale, tau_free, tau_hit, loss):
        wp.launch(k_samples, dim=(B, s_free + s_hit), device=self.device,
                  inputs=[self.theta, self.vol, self.outside, self.o, self.d, self.r, ids, s_free, s_hit, r0, half,
                          miss_range, seed, tau_free, tau_hit])
        wp.launch(k_ray_loss, dim=B, device=self.device,
                  inputs=[self.r, self.w, ids, tau_free, tau_hit, scale, loss])

    def loss_and_grad(self, ids: np.ndarray, s_free: int = 40, s_hit: int = 8, r0: float = 0.3, noise: float = 0.03,
                      miss_range: float = 8.0, seed: int = 0):
        """The batch loss (unscaled) and d loss / d theta for the rays `ids`: for gradient checks."""
        d = self.device
        B = len(ids)
        ids_d = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=d)
        tau_free = wp.zeros(B, dtype=float, device=d, requires_grad=True)
        tau_hit = wp.zeros(B, dtype=float, device=d, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        half = self._windows(noise)
        tape = wp.Tape()
        with tape:
            self._forward(ids_d, B, s_free, s_hit, r0, half, miss_range, seed, 1.0, tau_free, tau_hit, loss)
        tape.backward(loss=loss)
        out = float(loss.numpy()[0]), self.theta.grad.numpy().copy()
        tape.zero()
        return out

    def train(self, epochs: int = 6, batch: int = 1 << 17, lr=(0.3, 0.02), prior_weight: float = 1e-3,
              s_free: int = 40, s_hit: int = 8, noise: float = 0.03, r0: float = 0.3, miss_range: float = 8.0,
              min_steps: int = 300, seed: int = 0, log=None):
        """Adam over the logits: every ray once per epoch in random batches. noise is the range error the return
        window spans twice (+-2 sigma). A logit climbs from free to surface in some 40 steps of lr, so a small ray
        set is cut into smaller batches until the run has at least min_steps."""
        say = log or (lambda s: None)
        d = self.device
        n = self.n_rays
        if n == 0:
            raise RuntimeError("no rays: set_rays() first")
        half = self._windows(noise)
        rng = np.random.default_rng(seed)
        B_max = max(1, min(batch, n, int(math.ceil(n * epochs / min_steps))))
        ids = wp.empty(B_max, dtype=wp.int32, device=d)
        tau_free = wp.zeros(B_max, dtype=float, device=d, requires_grad=True)
        tau_hit = wp.zeros(B_max, dtype=float, device=d, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        steps = epochs * int(math.ceil(n / B_max))
        b1, b2 = 0.9, 0.99
        step = 0
        t0 = time.perf_counter()
        for ep in range(epochs):
            perm = rng.permutation(n).astype(np.int32)
            total = 0.0
            for a in range(0, n, B_max):
                chunk = perm[a:a + B_max]
                B = len(chunk)
                wp.copy(ids, wp.array(chunk, dtype=wp.int32, device="cpu"), count=B)
                tau_free.zero_()
                tau_hit.zero_()
                loss.zero_()
                tape = wp.Tape()
                with tape:
                    self._forward(ids, B, s_free, s_hit, r0, half, miss_range, seed * 7919 + step, n / B,
                                  tau_free, tau_hit, loss)
                tape.backward(loss=loss)
                self.adam_t += 1
                lr_t = dmath.cosine_lr(lr[0], lr[1], step, steps)
                wp.launch(k_adam_theta, dim=self.n, device=d,
                          inputs=[self.theta, self.theta.grad, self.m, self.v, self.theta0, prior_weight, lr_t,
                                  b1, b2, 1.0 - b1**self.adam_t, 1.0 - b2**self.adam_t])
                tape.zero()
                total += float(loss.numpy()[0]) * B / n
                step += 1
            self.history.append((ep, total / n))
            say(f"occupancy: epoch {ep + 1}/{epochs}, {total / n:.4f} nats per ray "
                f"({time.perf_counter() - t0:.1f} s)")

    # ---- results ------------------------------------------------------------------------------------

    def visibility(self, r0: float = 0.3, noise: float = 0.03, pierce_r: float = 0.03):
        """passes, hits and pierces per cell (k_observe)."""
        self.passes.zero_()
        self.hits.zero_()
        self.pierces.zero_()
        g = self.grid or _empty_grid(self.device)
        wp.launch(k_observe, dim=self.n_rays, device=self.device,
                  inputs=[self.o, self.d, self.r, self.vol, r0, self._windows(noise), self.dist_cm, g.g, g.start,
                          g.nn, g.xyz, g.nrm, g.planar, pierce_r, self.passes, self.hits, self.pierces])

    def classify(self, occ_hi: float = 0.5, occ_lo: float = 0.2, min_obs: int = 3, min_hits: int = 2,
                 min_pierce: int = 3, added_m: float = 0.10, echo_fraction: float = 0.05, block_cm: int = 3) -> dict:
        """Codes per cell (UNOBSERVED .. UNSCANNED, see k_classify and k_station_view) into self.codes; returns the
        count of each."""
        self.visibility()
        wp.launch(k_classify, dim=self.n, device=self.device,
                  inputs=[self.theta, self.theta0, self.dist_cm, self.passes, self.hits, self.pierces, occ_hi, occ_lo,
                          min_obs, min_hits, min_pierce, int(round(added_m * 100)), echo_fraction, self.codes])
        if self.stations is not None and self.grid is not None:
            cands = np.flatnonzero(self.codes.numpy() == ADDED).astype(np.int32)
            if len(cands):
                g = self.grid
                wp.launch(k_station_view, dim=len(cands), device=self.device,
                          inputs=[wp.array(cands, dtype=wp.int32, device=self.device), self.vol, g.g, g.df,
                                  self.stations, self.stations.shape[0], int(block_cm), self.codes])
        counts = np.bincount(self.codes.numpy(), minlength=len(CODE_NAMES))
        return {name: int(c) for name, c in zip(CODE_NAMES, counts)}

    def change_mask(self, dilate: int = 1) -> ChangeMask:
        """The classification as walkfit.py's mask (classify() first)."""
        bits = wp.zeros(self.n, dtype=wp.uint8, device=self.device)
        wp.launch(k_mask, dim=self.n, device=self.device, inputs=[self.codes, self.vol, int(dilate), bits])
        m = ChangeMask()
        m.vol, m.bits, m.on = self.vol, bits, 1
        return m

    def centres(self, code: int) -> np.ndarray:
        """World centres (n, 3) of the cells with this code."""
        idx = np.flatnonzero(self.codes.numpy() == code)
        nx, ny = int(self.dims[0]), int(self.dims[1])
        ijk = np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)], 1)
        return self.lo + (ijk + 0.5) * self.cell

    def occupancy_at(self, pts: np.ndarray) -> np.ndarray:
        """The chance a ray crossing one cell length stops there, at world points (n, 3)."""
        p = wp.array(np.asarray(pts, np.float32), dtype=wp.vec3, device=self.device)
        out = wp.zeros(len(pts), dtype=float, device=self.device)
        wp.launch(k_occ_at, dim=len(pts), device=self.device, inputs=[self.theta, self.vol, self.outside, p, out])
        return out.numpy()

    def save(self, path: str):
        np.savez_compressed(path, lo=self.lo, cell=self.cell, dims=self.dims,
                            theta=self.theta.numpy().astype(np.float16), codes=self.codes.numpy())


class _EmptyGrid:
    """A map grid with nothing in it: the kernels that take one find no scan point anywhere."""

    def __init__(self, device):
        g = Grid()
        g.origin, g.cell, g.inv_cell, g.nx, g.ny, g.nz = wp.vec3(0.0, 0.0, 0.0), 1.0, 1.0, 1, 1, 1
        self.g = g
        self.start = wp.zeros(2, dtype=wp.int32, device=device)
        self.nn = wp.full(1, -1, dtype=wp.int32, device=device)
        self.xyz = wp.zeros(1, dtype=wp.vec3, device=device)
        self.nrm = wp.zeros(1, dtype=wp.vec3, device=device)
        self.planar = wp.zeros(1, dtype=float, device=device)
        self.df = wp.full(1, 255, dtype=wp.uint8, device=device)


_empty_grids = {}


def _empty_grid(device):
    key = str(device)
    if key not in _empty_grids:
        _empty_grids[key] = _EmptyGrid(device)
    return _empty_grids[key]


def no_mask(device) -> ChangeMask:
    """A ChangeMask that excludes nothing."""
    m = ChangeMask()
    v = Volume()
    v.origin, v.cell, v.inv_cell, v.nx, v.ny, v.nz = wp.vec3(0.0, 0.0, 0.0), 1.0, 1.0, 1, 1, 1
    m.vol, m.bits, m.on = v, wp.zeros(1, dtype=wp.uint8, device=device), 0
    return m
