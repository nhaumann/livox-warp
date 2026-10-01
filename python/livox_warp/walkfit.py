"""Continuous-time trajectory of a whole recording, fitted to a prior map (the building scan) with Warp's autodiff.

The odometry registers 0.1 s frames one after another and drifts; the drift correction (prior_session.py) moves
its world frame twice a second. Neither gives a trajectory to measure them by, and a frame-by-frame registration
to the scan fails exactly where the SLAM does. This fits one trajectory to every point of a recording at once:

  - The pose is continuous in time: a base trajectory (at first the odometry, placed in the map by the global
    localisation) times a correction, a uniform cubic B-spline of rotation vectors and translations with knots
    every `knot_dt`. Each point is placed by the pose at its own firing time: no deskewing, no frame boundary.
  - Each point is pulled onto the scan along the scan's normal (point-to-plane, Geman-McClure with its scale
    annealed from decimetres to two centimetres). Its partner on the scan is found again every few steps through
    the map grid (mapgrid.py), outside the tape: the gradient flows through the point's pose, not the search.
  - A prior on the correction's first and second differences carries it through stretches the scan does not
    constrain (a flat wall filling the view, or roll about the optical axis when the view holds no structure
    across it): there it follows the odometry's own shape. Adam alone would not: it normalises every step, so
    a direction with nothing but noise in its gradient random-walks at full step size. Its epsilon is therefore
    set per stage from the gradients' typical size, which keeps such steps as small as their gradients.
  - Calibration shared by the whole recording, optional: a range offset of the second and later returns relative
    to the first (the second echoes sit a few centimetres short; the first return's own offset is not observable
    here: for a 38 deg cone, one offset on every range is nearly a move along the optical axis, which the
    trajectory absorbs, so it stays at zero), and an angular distortion that is a function of the prism phases
    from rosette.py (sum_k d_k exp(i (m_k theta_a + n_k theta_b)) added to the direction's angles). A wedge or
    encoder error shows up there at hundreds of hertz, where no trajectory can absorb it.
  - Points the occupancy field (occupancy.py) puts on things the scan lacks, or on things that have moved since,
    are left out, and so are scan points it finds removed (set_change_mask).

wp.Tape differentiates three kernels (the points, the smoothness prior, the calibration prior) and Adam
(dmath.Adam) takes the steps; after each stage the correction is folded into the base and starts again at zero.
Every differentiated parameter is a plain kernel argument (dp, dphi, bias, dist), never a struct member. A
recording is fitted in overlapping windows, each started from the last one's correction carried forward along
the odometry and re-anchored by a registration (localize.py) if it starts far off; then the whole is polished
together, with the calibration free.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from . import dmath
from .mapgrid import PLANAR, Grid, nearest_point
from .occupancy import EXCLUDE_MAP, EXCLUDE_POINT, ChangeMask, mask_bits, no_mask
from .rosette import MAX_HARMONICS

MAX_HARM = wp.constant(MAX_HARMONICS)
R3 = wp.constant(0.03)  # the fit fractions reported: points within these distances of the scan
R5 = wp.constant(0.05)
R10 = wp.constant(0.10)
EVAL_STRIDE = wp.constant(5)  # counts per bin in k_eval: points, within 3 / 5 / 10 cm, left out by the change mask
# re-anchoring starts: map-frame offsets (x, y, z m, heading deg) around the carried-forward pose
WIDE_STARTS = tuple((dx, dy, 0.0, 0.0) for dx in (-0.3, 0.0, 0.3) for dy in (-0.3, 0.0, 0.3) if dx or dy) + (
    (0.0, 0.0, 0.0, 5.0), (0.0, 0.0, 0.0, -5.0))


@dataclass
class Stage:
    sigma: float  # Geman-McClure scale (m)
    max_dist: float  # association radius (m)
    iters: int
    lr_pos: float  # Adam step at the stage start (m); it decays to a tenth along a cosine
    lr_rot: float  # rad


@dataclass
class WalkFitConfig:
    knot_dt: float = 0.1
    stages: tuple = (Stage(0.20, 0.60, 80, 8e-3, 1.2e-3), Stage(0.08, 0.25, 80, 3e-3, 5e-4),
                     Stage(0.04, 0.12, 100, 1.2e-3, 2e-4), Stage(0.02, 0.06, 120, 5e-4, 8e-5))
    polish: tuple = (Stage(0.03, 0.10, 120, 6e-4, 1e-4), Stage(0.02, 0.06, 200, 3e-4, 5e-5))
    window: float = 8.0  # s
    step: float = 4.0  # s
    assoc_every: int = 10
    noise: float = 0.02  # m, the point noise the data term is scaled by
    smooth_pos: float = 0.01  # m: one sigma of the position correction's second difference per knot
    smooth_rot: float = 0.002  # rad
    steady_pos: float = 0.005  # m: one sigma of the position correction's change per knot (first difference)
    steady_rot: float = 0.001  # rad
    adam_floor: float = 0.05  # Adam's epsilon, as a fraction of the stage's typical gradient
    pts_per_knot: int = 1500
    min_range: float = 0.5
    max_range: float = 40.0
    fit_bias: bool = True
    fit_distortion: bool = True
    lr_bias: float = 5e-4  # m
    lr_dist: float = 2e-5  # rad
    bias_sigma: float = 0.05  # m: a weak prior keeps the calibration near zero where the data says nothing
    dist_sigma: float = 0.005  # rad
    reanchor_fit: float = 0.25  # a window whose points have less than this fraction within 10 cm is re-anchored
    seed: int = 0


class _Points:
    """Points on the GPU: sensor-frame xyz, time since the knots' t0, return index and prism phases, plus the
    segment, base pose and (for the fit set) the scan partner of each, refreshed by the solver."""

    def __init__(self, xyz, echo, t_rel, phase, device, assoc: bool):
        n = len(t_rel)
        self.n = n
        self.x = wp.array(np.ascontiguousarray(xyz, dtype=np.float32), dtype=wp.vec3, device=device)
        self.echo = wp.array(np.ascontiguousarray(echo, dtype=np.int32), dtype=wp.int32, device=device)
        self.t = wp.array(np.ascontiguousarray(t_rel, dtype=np.float32), dtype=float, device=device)
        self.phase = wp.array(np.ascontiguousarray(phase, dtype=np.float32), dtype=wp.vec2, device=device)
        self.k = wp.zeros(n, dtype=wp.int32, device=device)
        self.u = wp.zeros(n, dtype=float, device=device)
        self.bq = wp.zeros(n, dtype=wp.quat, device=device)
        self.bp = wp.zeros(n, dtype=wp.vec3, device=device)
        if assoc:
            self.q = wp.zeros(n, dtype=wp.vec3, device=device)
            self.nrm = wp.zeros(n, dtype=wp.vec3, device=device)
            self.w = wp.zeros(n, dtype=float, device=device)


@wp.struct
class Harmonics:
    """The prism harmonics the angular distortion is a function of (constant during a fit)."""

    mn: wp.array(dtype=wp.vec2i)  # MAX_HARM (m, n)
    on: wp.array(dtype=float)  # 1 for a harmonic in use
    use: int  # 0: no distortion model at all


@wp.kernel
def k_base(t: wp.array(dtype=float), inv_dt: float, n_knots: int, kq: wp.array(dtype=wp.quat),
           kp: wp.array(dtype=wp.vec3), out_k: wp.array(dtype=wp.int32), out_u: wp.array(dtype=float),
           out_q: wp.array(dtype=wp.quat), out_p: wp.array(dtype=wp.vec3)):
    """Each point's spline segment and base pose (slerp / lerp between the base knots)."""
    i = wp.tid()
    s = t[i] * inv_dt
    k = wp.clamp(int(wp.floor(s)), 0, n_knots - 2)
    u = wp.clamp(s - float(k), 0.0, 1.0)
    out_k[i] = k
    out_u[i] = u
    out_q[i] = dmath.quat_interp(kq[k], kq[k + 1], u)
    out_p[i] = kp[k] + (kp[k + 1] - kp[k]) * u


@wp.func
def correction(k: int, u: float, n: int, dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3)):
    """The B-spline correction at segment k, fraction u: (rotation, translation)."""
    B = dmath.bspline4(u)
    a = wp.vec3()
    r = wp.vec3()
    for j in range(4):
        idx = wp.clamp(k - 1 + j, 0, n - 1)
        a += B[j] * dp[idx]
        r += B[j] * dphi[idx]
    return dmath.so3_exp(r), a


@wp.func
def sensor_point(x: wp.vec3, echo: int, phase: wp.vec2, hm: Harmonics, dist: wp.array(dtype=float),
                 bias: wp.array(dtype=float)) -> wp.vec3:
    """A measured point with the calibration applied: the range offset of its return and the angular distortion
    at its prism phases."""
    r = wp.length(x)
    dirn = x / r
    if hm.use != 0 and x[0] > 0.05:
        u = wp.atan2(x[1], x[0])
        v = wp.atan2(x[2], x[0])
        du = float(0.0)
        dv = float(0.0)
        for k in range(MAX_HARM):
            h = hm.mn[k]
            ang = float(h[0]) * phase[0] + float(h[1]) * phase[1]
            cs = wp.cos(ang)
            sn = wp.sin(ang)
            a = dist[2 * k]
            b = dist[2 * k + 1]
            du += hm.on[k] * (a * cs - b * sn)
            dv += hm.on[k] * (a * sn + b * cs)
        dirn = wp.normalize(wp.vec3(1.0, wp.tan(u + du), wp.tan(v + dv)))
    return dirn * (r + bias[wp.min(echo, 2)])


@wp.func
def world_point(x: wp.vec3, echo: int, phase: wp.vec2, k: int, u: float, bq: wp.quat, bp: wp.vec3, n: int,
                dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3), hm: Harmonics,
                dist: wp.array(dtype=float), bias: wp.array(dtype=float)) -> wp.vec3:
    Rc, tc = correction(k, u, n, dp, dphi)
    return (Rc * wp.quat_to_matrix(bq)) * sensor_point(x, echo, phase, hm, dist, bias) + bp + tc


@wp.kernel
def k_loss(x: wp.array(dtype=wp.vec3), echo: wp.array(dtype=wp.int32), phase: wp.array(dtype=wp.vec2),
           ks: wp.array(dtype=wp.int32), us: wp.array(dtype=float), bq: wp.array(dtype=wp.quat),
           bp: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
           wgt: wp.array(dtype=float), n_knots: int, dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3),
           hm: Harmonics, dist: wp.array(dtype=float), bias: wp.array(dtype=float), sigma: float,
           inv_noise2: float, loss: wp.array(dtype=float)):
    """Point-to-plane residual of every associated point, Geman-McClure, in units of the point noise."""
    i = wp.tid()
    w8 = wgt[i]
    if w8 == 0.0:
        return
    w = world_point(x[i], echo[i], phase[i], ks[i], us[i], bq[i], bp[i], n_knots, dp, dphi, hm, dist, bias)
    res = wp.dot(nrm[i], w - q[i])
    wp.atomic_add(loss, 0, w8 * dmath.geman_mcclure(res, sigma) * inv_noise2)


@wp.kernel
def k_smooth(dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3), inv_sp2: float, inv_sr2: float,
             inv_vp2: float, inv_vr2: float, loss: wp.array(dtype=float)):
    """Second and first differences of the correction (knot c = j + 1 of j = 0 .. n - 3)."""
    c = wp.tid() + 1
    a = dp[c - 1] - 2.0 * dp[c] + dp[c + 1]
    b = dphi[c - 1] - 2.0 * dphi[c] + dphi[c + 1]
    va = dp[c] - dp[c - 1]
    vb = dphi[c] - dphi[c - 1]
    wp.atomic_add(loss, 0, wp.dot(a, a) * inv_sp2 + wp.dot(b, b) * inv_sr2 + wp.dot(va, va) * inv_vp2
                  + wp.dot(vb, vb) * inv_vr2)


@wp.kernel
def k_calib_prior(hm: Harmonics, dist: wp.array(dtype=float), bias: wp.array(dtype=float), inv_b2: float,
                  inv_d2: float, loss: wp.array(dtype=float)):
    """A weak pull of the calibration toward zero (thread j: harmonic j, and range offset j for j < 3)."""
    j = wp.tid()
    if j < 3:
        wp.atomic_add(loss, 0, bias[j] * bias[j] * inv_b2)
    a = dist[2 * j]
    b = dist[2 * j + 1]
    wp.atomic_add(loss, 0, hm.on[j] * (a * a + b * b) * inv_d2)


@wp.kernel
def k_associate(x: wp.array(dtype=wp.vec3), echo: wp.array(dtype=wp.int32), phase: wp.array(dtype=wp.vec2),
                t: wp.array(dtype=float), ks: wp.array(dtype=wp.int32), us: wp.array(dtype=float),
                bq: wp.array(dtype=wp.quat), bp: wp.array(dtype=wp.vec3), n_knots: int,
                dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3), hm: Harmonics,
                dist: wp.array(dtype=float), bias: wp.array(dtype=float), g: Grid,
                start: wp.array(dtype=wp.int32), nn: wp.array(dtype=wp.int32), mxyz: wp.array(dtype=wp.vec3),
                mnrm: wp.array(dtype=wp.vec3), mplanar: wp.array(dtype=float), mask: ChangeMask, t_lo: float,
                t_hi: float, max_dist: float, out_q: wp.array(dtype=wp.vec3), out_n: wp.array(dtype=wp.vec3),
                out_w: wp.array(dtype=float)):
    """Each point's partner on the scan at the current pose: the nearest scan point within max_dist and its normal
    (the direction to it where the scan is not planar). Points outside [t_lo, t_hi] or on changes get weight 0."""
    i = wp.tid()
    out_w[i] = 0.0
    ti = t[i]
    if ti < t_lo or ti > t_hi:
        return
    w = world_point(x[i], echo[i], phase[i], ks[i], us[i], bq[i], bp[i], n_knots, dp, dphi, hm, dist, bias)
    if (mask_bits(mask, w) & EXCLUDE_POINT) != 0:
        return
    s = nearest_point(g, start, nn, mxyz, w)
    if s < 0:
        return
    qv = mxyz[s]
    dv = w - qv
    dd = wp.length(dv)
    if dd > max_dist:
        return
    if (mask_bits(mask, qv) & EXCLUDE_MAP) != 0:
        return
    nv = mnrm[s]
    if mplanar[s] < PLANAR or wp.length(nv) < 0.5:
        if dd < 1.0e-6:
            return
        nv = dv / dd
    out_q[i] = qv
    out_n[i] = nv
    out_w[i] = 1.0


@wp.kernel
def k_eval(x: wp.array(dtype=wp.vec3), echo: wp.array(dtype=wp.int32), phase: wp.array(dtype=wp.vec2),
           t: wp.array(dtype=float), ks: wp.array(dtype=wp.int32), us: wp.array(dtype=float),
           bq: wp.array(dtype=wp.quat), bp: wp.array(dtype=wp.vec3), n_knots: int, dp: wp.array(dtype=wp.vec3),
           dphi: wp.array(dtype=wp.vec3), hm: Harmonics, dist: wp.array(dtype=float), bias: wp.array(dtype=float),
           g: Grid, start: wp.array(dtype=wp.int32), nn: wp.array(dtype=wp.int32), mxyz: wp.array(dtype=wp.vec3),
           mask: ChangeMask, t_lo: float, inv_bin: float, n_bins: int, counts: wp.array(dtype=wp.int32)):
    """Per time bin from t_lo: points, how many lie within 3 / 5 / 10 cm of the scan, how many the mask left out."""
    i = wp.tid()
    bf = (t[i] - t_lo) * inv_bin
    if bf < 0.0 or bf >= float(n_bins):
        return
    b = int(bf) * EVAL_STRIDE
    w = world_point(x[i], echo[i], phase[i], ks[i], us[i], bq[i], bp[i], n_knots, dp, dphi, hm, dist, bias)
    if (mask_bits(mask, w) & EXCLUDE_POINT) != 0:
        wp.atomic_add(counts, b + 4, 1)
        return
    wp.atomic_add(counts, b, 1)
    s = nearest_point(g, start, nn, mxyz, w)
    if s < 0:
        return
    d = wp.length(w - mxyz[s])
    if d < R3:
        wp.atomic_add(counts, b + 1, 1)
    if d < R5:
        wp.atomic_add(counts, b + 2, 1)
    if d < R10:
        wp.atomic_add(counts, b + 3, 1)


@wp.kernel
def k_rays(x: wp.array(dtype=wp.vec3), echo: wp.array(dtype=wp.int32), phase: wp.array(dtype=wp.vec2),
           ks: wp.array(dtype=wp.int32), us: wp.array(dtype=float), bq: wp.array(dtype=wp.quat),
           bp: wp.array(dtype=wp.vec3), n_knots: int, dp: wp.array(dtype=wp.vec3), dphi: wp.array(dtype=wp.vec3),
           hm: Harmonics, dist: wp.array(dtype=float), bias: wp.array(dtype=float), miss: int, offset: int,
           out_o: wp.array(dtype=wp.vec3), out_d: wp.array(dtype=wp.vec3), out_r: wp.array(dtype=float),
           out_x: wp.array(dtype=wp.vec3)):
    """World rays: the sensor position, the calibrated direction and range (-1 for a firing that returned
    nothing, whose x is then its unit direction), and the world point."""
    i = wp.tid()
    Rc, tc = correction(ks[i], us[i], n_knots, dp, dphi)
    R = Rc * wp.quat_to_matrix(bq[i])
    o = bp[i] + tc
    j = offset + i
    out_o[j] = o
    if miss != 0:
        out_d[j] = wp.normalize(R * x[i])
        out_r[j] = -1.0
        out_x[j] = o
        return
    p = sensor_point(x[i], echo[i], phase[i], hm, dist, bias)
    rr = wp.length(p)
    out_d[j] = R * (p / rr)
    out_r[j] = rr
    out_x[j] = o + R * p


def _subsample_per_knot(knot: np.ndarray, cap: int, rng) -> np.ndarray:
    """Indices keeping at most `cap` random entries per knot interval, in their original order."""
    order = rng.permutation(len(knot))
    by_knot = order[np.argsort(knot[order], kind="stable")]
    ks = knot[by_knot]
    first = np.searchsorted(ks, ks, side="left")
    rank = np.arange(len(ks)) - first
    return np.sort(by_knot[rank < cap])


class WalkFit:
    """The fit of one recording; see the module docstring. Points: sensor-frame xyz, attr (return index in bits
    16-23), float64 times. init: times and 4x4 sensor-to-map poses of a starting trajectory. grid: the scan's
    mapgrid.PriorGrid; localizer: a localize.GlobalLocalizer on it, for re-anchoring (optional)."""

    def __init__(self, grid, xyz, attr, t, init_times, init_poses, rosette=None, config: WalkFitConfig | None = None,
                 localizer=None, device=None, log=None):
        self.cfg = cfg = config or WalkFitConfig()
        self.device = d = wp.get_device(device)
        self.grid = grid
        self.localizer = localizer
        self.say = log or (lambda s: None)
        self.rosette = rosette
        t = np.asarray(t, dtype=np.float64)
        xyz = np.asarray(xyz, dtype=np.float32)
        echo = ((np.asarray(attr, dtype=np.uint32) >> 16) & 0xFF).astype(np.int32)
        rng = np.random.default_rng(cfg.seed)
        self.dt = cfg.knot_dt
        self.t_first, self.t_last = float(t.min()), float(t.max())
        self.t0 = self.t_first - self.dt
        self.n_knots = int(math.ceil((self.t_last - self.t0) / self.dt)) + 3
        self.knot_t = self.t0 + np.arange(self.n_knots) * self.dt
        init = dmath.interpolate_poses(np.asarray(init_times, np.float64), np.asarray(init_poses, np.float64),
                                       self.knot_t)
        self.init_R, self.init_p = init[:, :3, :3].copy(), init[:, :3, 3].copy()
        self.base_R, self.base_p = self.init_R.copy(), self.init_p.copy()
        self.kq = self.kp = None

        # parameters (requires_grad: the tape fills their .grad)
        self.dp = wp.zeros(self.n_knots, dtype=wp.vec3, device=d, requires_grad=True)
        self.dphi = wp.zeros(self.n_knots, dtype=wp.vec3, device=d, requires_grad=True)
        self.bias = wp.zeros(3, dtype=float, device=d, requires_grad=True)
        self.dist = wp.zeros(2 * MAX_HARMONICS, dtype=float, device=d, requires_grad=True)
        mn = np.zeros((MAX_HARMONICS, 2), np.int32)
        on = np.zeros(MAX_HARMONICS, np.float32)
        if rosette is not None:
            hs = [h for h in rosette.harmonics.tolist() if h != [0, 0]][:MAX_HARMONICS]
            mn[:len(hs)] = hs
            on[:len(hs)] = 1.0
        self.harmonics = mn[on > 0]
        hm = Harmonics()
        hm.mn = wp.array(mn, dtype=wp.vec2i, device=d)
        hm.on = wp.array(on, dtype=float, device=d)
        hm.use = int(rosette is not None and cfg.fit_distortion)
        self.hm = hm
        self.adam = {name: dmath.Adam(arr) for name, arr in
                     (("dp", self.dp), ("dphi", self.dphi), ("bias", self.bias), ("dist", self.dist))}
        self.knot_mask = wp.ones(self.n_knots, dtype=float, device=d)
        self.bias_mask = wp.array(np.array([0.0, 1.0, 1.0], np.float32), dtype=float, device=d)  # first return fixed
        self.loss = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        self.mask = no_mask(d)
        self.history = []  # per window: dict
        self._upload_base()

        rng_ok = np.linalg.norm(xyz, axis=1)
        idx = np.flatnonzero((rng_ok > cfg.min_range) & (rng_ok < cfg.max_range))
        knot = np.floor((t[idx] - self.t0) / self.dt).astype(np.int64)
        idx = idx[_subsample_per_knot(knot, cfg.pts_per_knot, rng)]
        self.fit = self._points(xyz[idx], echo[idx], t[idx], assoc=True)

    # ---- plumbing -----------------------------------------------------------------------------------

    def _points(self, xyz, echo, t, assoc: bool = False) -> _Points:
        t = np.asarray(t, dtype=np.float64)
        phase = self.rosette.phases_mod(t) if self.rosette is not None else np.zeros((len(t), 2), np.float32)
        p = _Points(xyz, echo, t - self.t0, phase, self.device, assoc)
        self._refresh(p)
        return p

    def _upload_base(self):
        q = Rotation.from_matrix(self.base_R).as_quat().astype(np.float32)  # x, y, z, w: Warp's order too
        self.kq = wp.array(q, dtype=wp.quat, device=self.device)
        self.kp = wp.array(self.base_p.astype(np.float32), dtype=wp.vec3, device=self.device)
        if hasattr(self, "fit"):
            self._refresh(self.fit)

    def _refresh(self, p: _Points):
        wp.launch(k_base, dim=p.n, device=self.device,
                  inputs=[p.t, 1.0 / self.dt, self.n_knots, self.kq, self.kp, p.k, p.u, p.bq, p.bp])

    def _pose_args(self, p: _Points):
        """The per-point pose and calibration inputs the kernels share, in their order."""
        return [p.k, p.u, p.bq, p.bp, self.n_knots, self.dp, self.dphi, self.hm, self.dist, self.bias]

    def fold(self):
        """Fold the correction into the base trajectory and start it again from zero."""
        dp = self.dp.numpy().astype(np.float64)
        dphi = self.dphi.numpy().astype(np.float64)
        n = self.n_knots
        j = np.arange(n)
        lo, hi = np.maximum(j - 1, 0), np.minimum(j + 1, n - 1)
        c_p = (dp[lo] + 4.0 * dp[j] + dp[hi]) / 6.0  # the B-spline at each knot (u = 0)
        c_r = (dphi[lo] + 4.0 * dphi[j] + dphi[hi]) / 6.0
        self.base_R = Rotation.from_rotvec(c_r).as_matrix() @ self.base_R
        self.base_p = self.base_p + c_p
        self.dp.zero_()
        self.dphi.zero_()
        self._upload_base()

    def set_change_mask(self, mask: ChangeMask | None):
        """occupancy.OccupancyField.change_mask(), or None for no mask."""
        self.mask = mask if mask is not None else no_mask(self.device)

    def poses(self, times) -> np.ndarray:
        """Sensor-to-map poses (n, 4, 4) at times (s) from the base trajectory (between fits, the correction is 0)."""
        T = np.tile(np.eye(4), (self.n_knots, 1, 1))
        T[:, :3, :3], T[:, :3, 3] = self.base_R, self.base_p
        return dmath.interpolate_poses(self.knot_t, T, times)

    def calibration(self) -> dict:
        b = self.bias.numpy()
        out = {"range_offset_m": {"first": float(b[0]), "second": float(b[1]), "later": float(b[2])}}
        if self.hm.use:
            dist = self.dist.numpy().reshape(-1, 2)
            out["distortion_mrad"] = {f"{m},{n}": [float(dist[k, 0]) * 1e3, float(dist[k, 1]) * 1e3]
                                      for k, (m, n) in enumerate(self.harmonics)}
        return out

    # ---- the solver ---------------------------------------------------------------------------------

    def associate(self, max_dist: float, t_lo: float, t_hi: float):
        p, g = self.fit, self.grid
        wp.launch(k_associate, dim=p.n, device=self.device,
                  inputs=[p.x, p.echo, p.phase, p.t] + self._pose_args(p)
                  + [g.g, g.start, g.nn, g.xyz, g.nrm, g.planar, self.mask, float(t_lo - self.t0),
                     float(t_hi - self.t0), float(max_dist), p.q, p.nrm, p.w])

    def objective(self, sigma: float, calib: bool) -> wp.Tape:
        """The loss from the current associations on a fresh tape, already run backward: the parameters' .grad
        hold its gradient until tape.zero()."""
        p, cfg = self.fit, self.cfg
        self.loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(k_loss, dim=p.n, device=self.device,
                      inputs=[p.x, p.echo, p.phase, p.k, p.u, p.bq, p.bp, p.q, p.nrm, p.w, self.n_knots, self.dp,
                              self.dphi, self.hm, self.dist, self.bias, float(sigma), 1.0 / cfg.noise**2, self.loss])
            wp.launch(k_smooth, dim=self.n_knots - 2, device=self.device,
                      inputs=[self.dp, self.dphi, 1.0 / cfg.smooth_pos**2, 1.0 / cfg.smooth_rot**2,
                              1.0 / cfg.steady_pos**2, 1.0 / cfg.steady_rot**2, self.loss])
            if calib:
                wp.launch(k_calib_prior, dim=MAX_HARMONICS, device=self.device,
                          inputs=[self.hm, self.dist, self.bias, 1.0 / cfg.bias_sigma**2, 1.0 / cfg.dist_sigma**2,
                                  self.loss])
        tape.backward(loss=self.loss)
        return tape

    def _set_window(self, lo: float, hi: float):
        m = ((self.knot_t >= lo - 2 * self.dt) & (self.knot_t <= hi + 2 * self.dt)).astype(np.float32)
        self.knot_mask = wp.array(m, dtype=float, device=self.device)

    def _run_stages(self, lo: float, hi: float, stages, calib: bool):
        cfg = self.cfg
        self._set_window(lo, hi)
        for st in stages:
            for a in self.adam.values():
                a.reset()
            for it in range(st.iters):
                if it % cfg.assoc_every == 0:
                    self.associate(st.max_dist, lo, hi)
                tape = self.objective(st.sigma, calib)
                if it == 0:
                    self._set_floors()
                self.adam["dp"].step(dmath.cosine_lr(st.lr_pos, 0.1 * st.lr_pos, it, st.iters), self.knot_mask)
                self.adam["dphi"].step(dmath.cosine_lr(st.lr_rot, 0.1 * st.lr_rot, it, st.iters), self.knot_mask)
                if calib and cfg.fit_bias:
                    self.adam["bias"].step(dmath.cosine_lr(cfg.lr_bias, 0.1 * cfg.lr_bias, it, st.iters),
                                           self.bias_mask)
                if calib and self.hm.use:
                    self.adam["dist"].step(dmath.cosine_lr(cfg.lr_dist, 0.1 * cfg.lr_dist, it, st.iters))
                tape.zero()
            self.fold()

    def _set_floors(self):
        """Adam's epsilon for the knots: a fraction of the typical gradient over the free knots (module docstring)."""
        free = self.knot_mask.numpy() > 0
        for name, arr in (("dp", self.dp), ("dphi", self.dphi)):
            g = np.linalg.norm(arr.grad.numpy()[free], axis=1)
            g = g[g > 0]
            self.adam[name].eps = self.cfg.adam_floor * float(np.sqrt(np.mean(g**2))) if len(g) else 1e-12

    def fit_fractions(self, lo: float, hi: float, bin_s: float | None = None, points: _Points | None = None):
        """(n_bins, 5) counts per time bin of [lo, hi): points, within 3 / 5 / 10 cm, masked (k_eval)."""
        p, g = points or self.fit, self.grid
        bin_s = bin_s or (hi - lo)
        n_bins = max(1, int(math.ceil((hi - lo) / bin_s - 1e-9)))
        counts = wp.zeros(n_bins * EVAL_STRIDE, dtype=wp.int32, device=self.device)
        wp.launch(k_eval, dim=p.n, device=self.device,
                  inputs=[p.x, p.echo, p.phase, p.t] + self._pose_args(p)
                  + [g.g, g.start, g.nn, g.xyz, self.mask, float(lo - self.t0), 1.0 / bin_s, n_bins, counts])
        return counts.numpy().reshape(n_bins, EVAL_STRIDE)

    def _fit10(self, lo: float, hi: float) -> float:
        c = self.fit_fractions(lo, hi)[0]
        return float(c[3]) / max(int(c[0]), 1)

    def _ray_arrays(self, n: int):
        d = self.device
        return (wp.zeros(n, dtype=wp.vec3, device=d), wp.zeros(n, dtype=wp.vec3, device=d),
                wp.zeros(n, dtype=float, device=d), wp.zeros(n, dtype=wp.vec3, device=d))

    def _launch_rays(self, p: _Points, miss: int, offset: int, out):
        wp.launch(k_rays, dim=p.n, device=self.device,
                  inputs=[p.x, p.echo, p.phase] + self._pose_args(p) + [miss, offset, *out])

    def world_points(self, lo: float, hi: float, points: _Points | None = None) -> np.ndarray:
        """World positions (host) of the points fired in [lo, hi] (s)."""
        p = points or self.fit
        out = self._ray_arrays(p.n)
        self._launch_rays(p, 0, 0, out)
        tt = p.t.numpy() + self.t0
        return out[3].numpy()[(tt >= lo) & (tt <= hi)]

    def _apply(self, C: np.ndarray, from_t: float):
        """Left-multiply the base knots from from_t on by the map-frame correction C."""
        sel = self.knot_t >= from_t
        self.base_R[sel] = C[:3, :3] @ self.base_R[sel]
        self.base_p[sel] = self.base_p[sel] @ C[:3, :3].T + C[:3, 3]
        self._upload_base()

    def _carry_forward(self, fitted_until: float):
        """Knots after the last well-fitted one follow the initial trajectory, moved by that knot's correction."""
        j = int(np.searchsorted(self.knot_t, fitted_until - 2 * self.dt, side="right")) - 1
        j = int(np.clip(j, 0, self.n_knots - 1))
        T_fit = np.eye(4)
        T_fit[:3, :3], T_fit[:3, 3] = self.base_R[j], self.base_p[j]
        T_init = np.eye(4)
        T_init[:3, :3], T_init[:3, 3] = self.init_R[j], self.init_p[j]
        C = T_fit @ np.linalg.inv(T_init)
        sel = np.arange(self.n_knots) > j
        self.base_R[sel] = C[:3, :3] @ self.init_R[sel]
        self.base_p[sel] = self.init_p[sel] @ C[:3, :3].T + C[:3, 3]
        self._upload_base()

    def _reanchor(self, lo: float, hi: float, fit10: float) -> str:
        """A registration of the window's points against the scan (localize.py), wide starts first, then a global
        search; applied from the window on if it fits clearly better."""
        pts = self.world_points(lo, hi)
        if len(pts) < 500:
            return "too few points to re-anchor"
        T, fit, _ = self.localizer.refine(pts, np.eye(4), schedule=((0.40, 8), (0.20, 8), (0.10, 8), (0.05, 10)),
                                          starts=WIDE_STARTS)
        how = "local"
        if fit[2] < max(fit10 + 0.1, self.cfg.reanchor_fit):
            Tg, info = self.localizer.localize(pts, log=lambda s: None)
            if info["unique"] and info["inliers10"] > fit[2]:
                T, fit, how = Tg, np.array([info["inliers"], info["inliers5"], info["inliers10"]]), "global"
        if fit[2] >= fit10 + 0.1:
            self._apply(T, lo - 2 * self.dt)
            return f"re-anchored ({how}): 10 cm fit {fit10:.2f} -> {fit[2]:.2f}"
        return f"re-anchoring found nothing better (10 cm fit {fit10:.2f})"

    def run(self) -> list:
        """Windows, then the polish; returns the per-window history."""
        cfg = self.cfg
        t_a, t_b = self.t_first, self.t_last
        starts = np.arange(t_a, max(t_b - cfg.window, t_a) + 1e-9, cfg.step)
        windows = [(float(s), float(min(s + cfg.window, t_b))) for s in starts]
        if windows[-1][1] < t_b:
            windows.append((max(t_a, t_b - cfg.window), t_b))
        fitted_until = None
        t_start = time.perf_counter()
        for lo, hi in windows:
            if fitted_until is not None:
                self._carry_forward(fitted_until)
            before = self._fit10(lo, hi)
            note = ""
            if before < cfg.reanchor_fit and self.localizer is not None:
                note = self._reanchor(lo, hi, before)
            self._run_stages(lo, hi, cfg.stages, calib=False)
            c = self.fit_fractions(lo, hi)[0]
            row = {"lo": lo, "hi": hi, "fit10_before": before, "fit3": c[1] / max(c[0], 1),
                   "fit10": c[3] / max(c[0], 1), "masked": c[4] / max(c[0] + c[4], 1), "note": note}
            self.history.append(row)
            self.say(f"walkfit: {lo - t_a:5.1f}-{hi - t_a:5.1f} s  10 cm fit {before:.2f} -> {row['fit10']:.2f}, "
                     f"3 cm {row['fit3']:.2f}" + (f"  ({note})" if note else ""))
            fitted_until = hi
        calib = cfg.fit_bias or bool(self.hm.use)
        self._run_stages(t_a, t_b, cfg.polish, calib=calib)
        c = self.fit_fractions(t_a, t_b)[0]
        self.say(f"walkfit: polished {t_b - t_a:.1f} s: within 3 cm {c[1] / max(c[0], 1):.2f}, 10 cm "
                 f"{c[3] / max(c[0], 1):.2f}; {self.calibration()} ({time.perf_counter() - t_start:.0f} s)")
        return self.history

    # ---- what the other solvers take ------------------------------------------------------------------

    def rays(self, xyz, attr, t, miss_t=None, miss_dir=None, miss_weight: float = 0.2, min_fit: float = 0.0,
             frame_dt: float = 0.1):
        """World rays of points (sensor-frame xyz, attr, float64 t) and, optionally, of firings that returned nothing
        (times, sensor-frame unit directions): device arrays (origin, direction, range, weight) for occupancy.py.
        Rays of a frame (frame_dt) with less than min_fit of its points within 3 cm of the scan weigh nothing: where
        the fit could not place the sensor, its rays cross floors and walls that are there, and the occupancy would
        read them as removed."""
        echo = ((np.asarray(attr, dtype=np.uint32) >> 16) & 0xFF).astype(np.int32)
        pts = self._points(xyz, echo, t)
        n_p = pts.n
        n_m = 0 if miss_t is None else len(miss_t)
        out = self._ray_arrays(n_p + n_m)
        self._launch_rays(pts, 0, 0, out)
        w = np.ones(n_p + n_m, np.float32)
        if n_m:
            mp = self._points(np.asarray(miss_dir, np.float32), np.zeros(n_m, np.int32), miss_t)
            self._launch_rays(mp, 1, n_p, out)
            w[n_p:] = miss_weight
        if min_fit > 0.0:
            k0 = int(math.floor(self.t_first / frame_dt))
            k1 = int(math.floor(self.t_last / frame_dt))
            c = self.fit_fractions(k0 * frame_dt, (k1 + 1) * frame_dt, frame_dt)
            poor = c[:, 1] < min_fit * np.maximum(c[:, 0], 1)

            def frame(tt):
                return np.clip(np.floor(np.asarray(tt, np.float64) / frame_dt).astype(np.int64) - k0, 0, len(c) - 1)

            w[:n_p][poor[frame(t)]] = 0.0
            if n_m:
                w[n_p:][poor[frame(miss_t)]] = 0.0
        return out[0], out[1], out[2], wp.array(w, dtype=float, device=self.device)

    def frame_reference(self, frame_dt: float = 0.1):
        """Per frame of frame_dt (the odometry's frames, from floor(first time / frame_dt)): the pose at mid-frame
        and the fraction of its points within 3 cm of the scan. The format benchmarks/ reads (ref, fit, k0)."""
        k0 = int(math.floor(self.t_first / frame_dt))
        k1 = int(math.floor(self.t_last / frame_dt))
        ref = self.poses((np.arange(k0, k1 + 1) + 0.5) * frame_dt)
        c = self.fit_fractions(k0 * frame_dt, (k1 + 1) * frame_dt, frame_dt)
        fit = c[:, 1] / np.maximum(c[:, 0], 1)
        return ref, fit.astype(np.float32), k0
