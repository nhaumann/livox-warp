"""LiDAR-only odometry on the GPU, for sensors without an IMU (the Mid-40).

Points are cut into frames of `frame_dt` seconds by their timestamps. Each completed frame is
registered against a registration map: a voxel hash that keeps, per voxel, the sum and
outer-product sum of its points *relative to the voxel centre*, so mean and covariance stay
exact in float32 no matter how far the map extends.

Registration is scan-to-map ICP in the VGICP / KISS-ICP spirit:
  - the frame is thinned to one point per half-voxel,
  - each point looks at the voxels around it (5x5x5 on the first iterations, then 3x3x3) and
    takes the closest voxel mean,
  - the residual to that mean is weighted per eigen-direction of the voxel covariance: the
    tight direction of a surface (its normal) at full inverse-variance weight, loose directions
    (along the surface) at a small fraction of it, so planar voxels act as point-to-plane
    constraints; sparse voxels, and voxels whose points form a line or a small blob, act as weak
    point-to-point constraints only. A voxel seen by one or two passes of the rosette at long
    range holds a line of points whose "normal" is arbitrary; trusting it as a surface (or as an
    edge with two tight directions) pulls every point of the next frame that lands near it by
    up to a voxel, and at the leading edge of a turn those pulls all point back toward what the
    map has already seen: a systematic bias on every frame,
  - Gauss-Newton on a left-multiplied twist (rho, theta) with a Geman-McClure kernel in
    Mahalanobis units (influence falls to zero for far residuals, unlike Cauchy) and
    Levenberg-Marquardt damping,
  - degenerate directions: the information of the surface-normal residuals alone is also
    accumulated and examined in sensor coordinates (rotation scaled by the mean range). Directions
    whose information is below `degen_thr` (a flat wall seen by a narrow cone: sliding along it,
    rolling about the optical axis) are held at the motion prediction instead of following the
    weak tangential pulls, and a small prior (`prior_info`) pulls every direction toward the
    prediction; left to the tangential pulls alone, those directions wander within a single frame.
The 6x6 normal equations are accumulated on the GPU (64 partial copies in float64 to spread the
atomics) and solved on the host; each iteration costs one tiny readback.

Prediction: constant velocity, from an exponential average (`velocity_smooth`) of the per-frame
body motion. A direction held at the prediction keeps the velocity it had, so the prediction
must not come from the single last frame: one poorly constrained frame would otherwise become
the velocity of every following frame and the drift would grow without bound.

Deskew: a 0.1 s frame is not a snapshot. Turning at 30 deg/s smears a wall 10 m away by half a
metre within one frame, so every point is first moved to where it would have been seen at
mid-frame, using the sensor's motion over the previous frame (constant velocity). After the
frame is registered its own motion is known, and if it differs the frame is deskewed again and
re-registered from the solved pose.

The first `warmup` frames are inserted without registration (hold the sensor still for a
moment), because one frame of a non-repetitive scan is too sparse a map to register against.
A gap in the data longer than `max_gap` frames (paused replay, dropped stream) restarts frame
counting with zero velocity but keeps the map and the pose.

Poses are sensor-to-world, in the frame of the pose handed to reset(): the first frame is placed
there (the mount pose, normally) and everything after it is registered relative to it.

Streams: every launch and copy names `self.stream` explicitly, readbacks go through pinned host
buffers on that stream, and clears are kernels, so the class never touches Warp's global
per-device current stream after construction. That is what lets slam_worker.OdomWorker run it
on its own thread and private stream (the warp_worker discipline) while the render thread keeps
using the default stream. Constructed without a stream it uses the device's current stream, which
keeps the single-threaded tests ordered with the Pipeline's own launches.
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from . import gpu

vec6 = wp.types.vector(length=6, dtype=wp.float32)
ACC_SLOTS = 64  # partial accumulators per pass, so a frame's atomics spread over 64 copies
# layout of one accumulator, ACC_STRIDE float64 entries (localize.py shares it)
ACC_H = 0  # 21 entries: upper triangle of H = sum w J^T M J
ACC_B = 21  # 6 entries: b = sum w J^T M r
ACC_COUNT = 27  # matched points
ACC_ERR = 28  # sum w r^T M r, the weighted error; diagnostic only, the solver never reads it back
ACC_RMS = 29  # sum w rn^2, rn the residual along the tightest voxel direction (a metric rms)
ACC_WEIGHT = 30  # sum w
ACC_RANGE = 31  # sum of the matched points' sensor range
ACC_HG = 32  # 21 entries: upper triangle of the H of the surface-normal residuals alone
ACC_STRIDE = 64
REG_CAP = wp.constant(4096)  # points a registration voxel accumulates before it stops changing

# registration schedule
WIDE_ITERS = 3  # iterations that search the 5x5x5 voxels around a point before narrowing to 3x3x3
KAPPA_START, KAPPA_DECAY, KAPPA_MIN = 6.0, 0.7, 2.0  # robust-kernel scale in Mahalanobis units, per iteration
STEP_TOL = 1e-4  # a step below this (m and rad) ends the iterations
TIGHT_FRACTION = 0.6  # a voxel direction narrower than this fraction of a uniform fill is a surface normal
WEAK_COND = 1e-3  # a normalised H whose min/max eigenvalue ratio is below this is reported weak
# frame bookkeeping
REDESKEW_ROT = 2e-3  # rad: a solved motion this far from the deskew prediction deskews and registers again
REDESKEW_TRANS = 5e-3  # m: the same, for translation
MAX_FRAME_JUMP = 1.5  # m: a frame-to-frame translation this large is a failed registration, not motion


@wp.kernel
def k_frame_append(
    sx: wp.array(dtype=wp.vec3),
    sa: wp.array(dtype=wp.uint32),
    st: wp.array(dtype=wp.float32),
    inv_dt: float,
    k_min: int,
    k_open: int,
    cap: int,
    count: wp.array(dtype=wp.int32),
    f_xyz: wp.array(dtype=wp.vec3),
    f_attr: wp.array(dtype=wp.uint32),
    f_t: wp.array(dtype=wp.float32),
    f_frame: wp.array(dtype=wp.int32),
):
    """Copy the staged points with k_min <= frame <= k_open into the open frame's raw buffer."""
    i = wp.tid()
    k = int(wp.floor(wp.max(st[i], 0.0) * inv_dt))
    if k > k_open or k < k_min:
        return
    j = wp.atomic_add(count, 0, 1)
    if j >= cap:
        return
    f_xyz[j] = sx[i]
    f_attr[j] = sa[i]
    f_t[j] = st[i]
    f_frame[j] = k_open  # the frame it is deskewed, registered and posed with, even when it came in late


@wp.kernel
def k_ds_insert(
    xyz: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=wp.uint32),
    n: int,
    inv_v: float,
    min_range: float,
    max_range: float,
    ret_mask: int,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3),
):
    """Keep the first point to claim each thinning voxel (sensor frame), appended in arbitrary order;
    drop noise-tagged returns and echoes whose return index is not in ret_mask."""
    i = wp.tid()
    if i >= n:
        return
    p = xyz[i]
    r = wp.length(p)
    if r < min_range or r > max_range:
        return
    if gpu.tag_of(attr[i]) != 0 or ((1 << gpu.ret_of(attr[i])) & ret_mask) == 0:
        return
    key = gpu.voxel_key(p, inv_v)
    h = gpu.slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1):
            out[wp.atomic_add(count, 0, 1)] = p
            return
        if prev == key:
            return


@wp.kernel
def k_deskew(
    raw: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=wp.float32),
    m: int,
    t_mid: float,
    inv_dt: float,
    rho: wp.vec3,
    theta: wp.vec3,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= m:
        return
    s = wp.clamp((t[i] - t_mid) * inv_dt, -1.0, 1.0)
    out[i] = gpu.deskew(raw[i], s, rho, theta)


@wp.func
def voxel_center(ix: int, iy: int, iz: int, voxel: float) -> wp.vec3:
    return wp.vec3((float(ix) + 0.5) * voxel, (float(iy) + 0.5) * voxel, (float(iz) + 0.5) * voxel)


@wp.kernel
def k_reg_insert(
    xyz: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=wp.uint32),
    n: int,
    T: wp.mat44,
    inv_v: float,
    voxel: float,
    max_range: float,
    ret_mask: int,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    rsum: wp.array(dtype=wp.vec3),
    router: wp.array(dtype=wp.mat33),
    rcnt: wp.array(dtype=wp.int32),
    stats: wp.array(dtype=wp.int32),
):
    """Add a frame's points, posed by T, to the registration map: per voxel the count and the sum and
    outer-product sum of the points relative to the voxel centre. stats[0] counts the voxels claimed,
    stats[1] the points that found no slot within the probe window (a full table)."""
    i = wp.tid()
    if i >= n:
        return
    p = xyz[i]
    if wp.length(p) > max_range or gpu.tag_of(attr[i]) != 0 or ((1 << gpu.ret_of(attr[i])) & ret_mask) == 0:
        return
    w = wp.transform_point(T, p)
    ix = int(wp.floor(w[0] * inv_v))
    iy = int(wp.floor(w[1] * inv_v))
    iz = int(wp.floor(w[2] * inv_v))
    key = gpu.voxel_key_ijk(ix, iy, iz)
    r = w - voxel_center(ix, iy, iz, voxel)
    h = gpu.slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            if prev == wp.int64(-1):
                wp.atomic_add(stats, 0, 1)
            if rcnt[s] < REG_CAP:
                wp.atomic_add(rsum, s, r)
                wp.atomic_add(router, s, wp.outer(r, r))
                wp.atomic_add(rcnt, s, 1)
            return
    wp.atomic_add(stats, 1, 1)


@wp.func
def accumulate(
    acc: wp.array(dtype=wp.float64),
    slot: int,
    J0: vec6,
    J1: vec6,
    J2: vec6,
    M: wp.mat33,
    r: wp.vec3,
    wgt: float,
):
    """H += w J^T M J (upper triangle), b += w J^T M r for a 3-vector residual r with precision M,
    plus the count, weighted error and weight sum; the ACC_* layout from `slot`."""
    Mr = M * r
    idx = int(ACC_H)
    for p in range(6):
        for q in range(p, 6):
            v = M[0, 0] * J0[p] * J0[q] + M[0, 1] * J0[p] * J1[q] + M[0, 2] * J0[p] * J2[q]
            v += M[1, 0] * J1[p] * J0[q] + M[1, 1] * J1[p] * J1[q] + M[1, 2] * J1[p] * J2[q]
            v += M[2, 0] * J2[p] * J0[q] + M[2, 1] * J2[p] * J1[q] + M[2, 2] * J2[p] * J2[q]
            wp.atomic_add(acc, slot + idx, wp.float64(wgt * v))
            idx += 1
    for p in range(6):
        wp.atomic_add(acc, slot + ACC_B + p, wp.float64(wgt * (Mr[0] * J0[p] + Mr[1] * J1[p] + Mr[2] * J2[p])))
    wp.atomic_add(acc, slot + ACC_COUNT, wp.float64(1.0))
    wp.atomic_add(acc, slot + ACC_ERR, wp.float64(wgt * wp.dot(r, Mr)))
    wp.atomic_add(acc, slot + ACC_WEIGHT, wp.float64(wgt))


@wp.func
def accumulate_h(acc: wp.array(dtype=wp.float64), slot: int, J0: vec6, J1: vec6, J2: vec6, M: wp.mat33, wgt: float):
    """H += w J^T M J (upper triangle) only, at `slot`."""
    idx = int(0)
    for p in range(6):
        for q in range(p, 6):
            v = M[0, 0] * J0[p] * J0[q] + M[0, 1] * J0[p] * J1[q] + M[0, 2] * J0[p] * J2[q]
            v += M[1, 0] * J1[p] * J0[q] + M[1, 1] * J1[p] * J1[q] + M[1, 2] * J1[p] * J2[q]
            v += M[2, 0] * J2[p] * J0[q] + M[2, 1] * J2[p] * J1[q] + M[2, 2] * J2[p] * J2[q]
            wp.atomic_add(acc, slot + idx, wp.float64(wgt * v))
            idx += 1


@wp.kernel
def k_icp(
    ds: wp.array(dtype=wp.vec3),
    m: int,
    T: wp.mat44,
    inv_v: float,
    voxel: float,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    rsum: wp.array(dtype=wp.vec3),
    router: wp.array(dtype=wp.mat33),
    rcnt: wp.array(dtype=wp.int32),
    reach: int,
    max_d2: float,
    noise2: float,
    tight2: float,
    alpha: float,
    kappa: float,
    acc: wp.array(dtype=wp.float64),
):
    """One Gauss-Newton accumulation pass: each thinned point, posed by T, takes the closest voxel mean
    within `reach` voxels and adds its weighted residual to the accumulator (the ACC_* layout): H, b,
    count, weighted error, and the H of the surface-normal residuals alone (what the geometry really
    constrains)."""
    i = wp.tid()
    if i >= m:
        return
    p = ds[i]
    w = wp.transform_point(T, p)
    slot = (i & (ACC_SLOTS - 1)) * ACC_STRIDE
    J0 = vec6(1.0, 0.0, 0.0, 0.0, w[2], -w[1])
    J1 = vec6(0.0, 1.0, 0.0, -w[2], 0.0, w[0])
    J2 = vec6(0.0, 0.0, 1.0, w[1], -w[0], 0.0)
    bx = int(wp.floor(w[0] * inv_v))
    by = int(wp.floor(w[1] * inv_v))
    bz = int(wp.floor(w[2] * inv_v))
    best_d2 = max_d2
    best_s = int(-1)
    best_mu = wp.vec3()
    best_c = int(0)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                jx = bx + dx
                jy = by + dy
                jz = bz + dz
                s = gpu.find_slot(gpu.voxel_key_ijk(jx, jy, jz), keys, mask)
                if s >= 0:
                    c = rcnt[s]
                    if c > 0:
                        mu = voxel_center(jx, jy, jz, voxel) + rsum[s] / float(c)
                        dd = w - mu
                        d2 = wp.dot(dd, dd)
                        if d2 < best_d2:
                            best_d2 = d2
                            best_s = s
                            best_mu = mu
                            best_c = c
    if best_s < 0:
        return
    r = w - best_mu
    cf = float(best_c)
    M = wp.mat33(0.0)
    Mg = wp.mat33(0.0)  # the tight part of M only
    rn = float(0.0)  # residual along the tightest direction, for a metric rms
    mrel = rsum[best_s] / cf
    cov = router[best_s] / cf - wp.outer(mrel, mrel)
    Q, ev = wp.eig3(cov)
    nt = int(0)  # tight directions: 1 = a surface, 2 = a line of points, 3 = a small blob
    for a in range(3):
        if ev[a] < tight2:
            nt += 1
    if best_c >= 5 and nt <= 1:
        kmin = int(0)
        if ev[1] < ev[kmin]:
            kmin = 1
        if ev[2] < ev[kmin]:
            kmin = 2
        for a in range(3):
            q = wp.vec3(Q[0, a], Q[1, a], Q[2, a])
            e_a = wp.max(ev[a], 0.0) + noise2
            wa = 1.0 / e_a
            if ev[a] >= tight2:
                wa = alpha / e_a  # a direction the surface extends along: keep only a weak pull
            else:
                Mg += wa * wp.outer(q, q)
            M += wa * wp.outer(q, q)
            if a == kmin:
                rn = wp.dot(r, q)
    else:
        # too few points for a shape, or a line / blob whose normal is not trustworthy: a weak isotropic pull
        wa = alpha / (0.25 * voxel * voxel + noise2)
        M = wp.mat33(wa, 0.0, 0.0, 0.0, wa, 0.0, 0.0, 0.0, wa)
        rn = wp.length(r)
    d2 = wp.dot(r, M * r)
    wgt = 1.0 / (1.0 + d2 / (kappa * kappa))
    wgt = wgt * wgt  # Geman-McClure: the influence of a far residual falls to zero instead of staying flat
    wp.atomic_add(acc, slot + ACC_RMS, wp.float64(wgt * rn * rn))
    accumulate(acc, slot, J0, J1, J2, M, r, wgt)
    accumulate_h(acc, slot + ACC_HG, J0, J1, J2, Mg, wgt)
    wp.atomic_add(acc, slot + ACC_RANGE, wp.float64(wp.length(p)))


@wp.kernel
def k_reduce(acc: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    j = wp.tid()
    s = wp.float64(0.0)
    for k in range(ACC_SLOTS):
        s += acc[k * ACC_STRIDE + j]
    out[j] = s


_TRIU6 = np.triu_indices(6)  # row-major upper triangle, the order accumulate() writes


def _sym6(upper: np.ndarray) -> np.ndarray:
    """The symmetric 6x6 matrix whose upper triangle (21 entries, row-major) is `upper`."""
    H = np.zeros((6, 6))
    H[_TRIU6] = upper
    return H + np.triu(H, 1).T


def _twist_between(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    """Left twist (rho, theta) with exp(twist) T_from ~= T_to (small-motion approximation)."""
    M = T_to @ np.linalg.inv(T_from)
    theta = Rotation.from_matrix(M[:3, :3]).as_rotvec()
    return np.concatenate([M[:3, 3], theta])


def _apply_twist(T: np.ndarray, delta: np.ndarray) -> np.ndarray:
    Rz = Rotation.from_rotvec(delta[3:]).as_matrix()
    out = T.copy()
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = Rz @ T[:3, 3] + delta[:3]
    return out


def _body_twist(d: np.ndarray) -> np.ndarray:
    """(rho, theta) of a small relative motion d (4x4), first order: what deskew() expects."""
    return np.concatenate([d[:3, 3], Rotation.from_matrix(d[:3, :3]).as_rotvec()]).astype(np.float32)


def _twist_matrix(xi: np.ndarray) -> np.ndarray:
    """Inverse of _body_twist: the relative motion (4x4) of a first-order twist (rho, theta)."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(np.asarray(xi[3:], dtype=np.float64)).as_matrix()
    out[:3, 3] = xi[:3]
    return out


def _sensor_basis(T: np.ndarray, L: float) -> np.ndarray:
    """J with left twist = J @ s, where s = (sensor translation in sensor axes, rotation about the sensor in sensor
    axes times L). In s, a unit of rotation moves a point at range L as far as a unit of translation does."""
    c = T[:3, 3]
    J = np.eye(6)
    J[:3, 3:] = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])  # rotate about the sensor
    B = np.zeros((6, 6))
    B[:3, :3] = B[3:, 3:] = T[:3, :3]
    return J @ B @ np.diag([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])


def _constrained_step(A: np.ndarray, rhs: np.ndarray, Hg: np.ndarray, T: np.ndarray, L: float, thr: float,
                      to_pred: np.ndarray):
    """Solve A delta = rhs inside the directions the geometry constrains; hold the others at the prediction.

    Hg is the information of the surface-normal residuals alone. In sensor coordinates with rotation scaled by L,
    directions whose information is below thr are frozen: their part of the step is whatever brings the pose back
    onto the prediction (to_pred, the left twist from T to it), and the rest is the least-squares step in the
    remaining directions. Returns (delta, (eigenvalues, eigenvectors, kept mask))."""
    J = _sensor_basis(T, L)
    lam, V = np.linalg.eigh(J.T @ Hg @ J)
    good = lam >= thr
    if good.all():
        return np.linalg.solve(A, rhs), (lam, V, good)
    Vd = V[:, ~good]
    d0 = Vd @ (Vd.T @ np.linalg.solve(J, to_pred))
    if good.any():
        Vg = V[:, good]
        As = J.T @ A @ J
        d0 = d0 + Vg @ np.linalg.solve(Vg.T @ As @ Vg, Vg.T @ (J.T @ rhs - As @ d0))
    return J @ d0, (lam, V, good)


class Odometry:
    """Frame cutting, the registration map and scan-to-map ICP on one stream; see the module docstring."""

    def __init__(self, pipe: gpu.Pipeline, device=None, reg_slots: int = 1 << 21, frame_cap: int = 1 << 18,
                 stream=None):
        self.pipe = pipe
        self.device = wp.get_device(device)
        d = self.device
        # every launch/copy/readback of this class goes on this stream (see the module docstring)
        self.stream = stream if stream is not None else (wp.get_stream(d) if d.is_cuda else None)
        assert reg_slots & (reg_slots - 1) == 0
        self.reg_cap = reg_slots
        self.reg_keys = wp.full(reg_slots, -1, dtype=wp.int64, device=d)
        self.reg_sum = wp.zeros(reg_slots, dtype=wp.vec3, device=d)
        self.reg_outer = wp.zeros(reg_slots, dtype=wp.mat33, device=d)
        self.reg_cnt = wp.zeros(reg_slots, dtype=wp.int32, device=d)
        self.reg_stats = wp.zeros(2, dtype=wp.int32, device=d)  # voxels claimed, points that found no slot

        self.frame_cap = frame_cap
        self.f_raw = wp.zeros(frame_cap, dtype=wp.vec3, device=d)  # as measured
        self.f_xyz = wp.zeros(frame_cap, dtype=wp.vec3, device=d)  # deskewed to mid-frame
        self.f_attr = wp.zeros(frame_cap, dtype=wp.uint32, device=d)
        self.f_t = wp.zeros(frame_cap, dtype=wp.float32, device=d)
        self.f_frame = wp.zeros(frame_cap, dtype=wp.int32, device=d)
        self.f_count = wp.zeros(1, dtype=wp.int32, device=d)

        # the thinning table holds at most one key per frame point and is probed with a power-of-two mask
        self.ds_cap = 1 << (frame_cap - 1).bit_length()
        self.ds_keys = wp.full(self.ds_cap, -1, dtype=wp.int64, device=d)
        self.ds_xyz = wp.zeros(frame_cap, dtype=wp.vec3, device=d)
        self.ds_count = wp.zeros(1, dtype=wp.int32, device=d)
        self.acc = wp.zeros(ACC_SLOTS * ACC_STRIDE, dtype=wp.float64, device=d)
        self.acc_out = wp.zeros(ACC_STRIDE, dtype=wp.float64, device=d)
        # pinned host mirrors for readbacks on self.stream
        pin = d.is_cuda
        self._h_i32 = wp.zeros(2, dtype=wp.int32, device="cpu", pinned=pin)
        self._h_acc = wp.zeros(ACC_STRIDE, dtype=wp.float64, device="cpu", pinned=pin)
        # numpy views taken once, here: array.numpy() enters a ScopedStream even for host arrays
        self._h_i32_np = self._h_i32.numpy()
        self._h_acc_np = self._h_acc.numpy()

        # tuning
        self.frame_dt = 0.1
        self.reg_voxel = 0.15
        self.max_iter = 16
        self.warmup = 8  # frames inserted without registration while the map fills in
        self.max_gap = 20  # frames of missing data that restart frame counting
        self.min_points = 200
        self.min_corr = 60
        self.min_range = 0.3
        self.max_range = 60.0
        self.noise = 0.02  # sensor range noise (1 sigma, m) added to every voxel covariance
        self.ret_mask = 0b1111  # return indices (bit k = return k) used for registration and the map
        self.alpha = 0.05  # weight of loose (along-surface) directions relative to tight ones
        self.damping = 1.0e-3  # Levenberg-Marquardt: fraction of the diagonal added to itself
        self.velocity_smooth = 0.85  # prediction from an exponential average of the per-frame motion (0 = last frame)
        # directions whose surface-normal information (1/m^2 in sensor coordinates, rotation times the mean range)
        # is below this are held at the prediction; 3000 ~ 1.8 cm 1-sigma at the `noise` model (0 = off)
        self.degen_thr = 3000.0
        self.prior_info = 2000.0  # same units: a pull toward the prediction in every direction (0 = off)
        # keyframes: after warm-up a frame goes into the map only once the sensor has moved this far from the
        # last frame that did. A standing sensor then registers against a fixed map instead of one that follows
        # its own estimate (which lets small errors random-walk), and people walking past are not baked in.
        self.kf_dist = 0.05
        self.kf_angle = 1.0  # degrees
        self.deskew = True
        self.redeskew = True  # deskew again with the frame's own solved motion and re-register
        self.epoch = 0  # pose epoch of the pose-table entries this instance writes (set per batch by OdomWorker)

        self.reset(np.eye(4, dtype=np.float64))

    # ---- stream helpers --------------------------------------------------------------------

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device, stream=self.stream)

    def synchronize(self):
        """Wait for everything queued on self.stream (nothing to wait for on the CPU)."""
        if self.stream is not None:
            wp.synchronize_stream(self.stream)

    def _read_i32(self, arr, n: int = 1):
        """First n int32 of a device array, read on self.stream (waits for the work queued before it)."""
        wp.copy(self._h_i32, arr, count=n, stream=self.stream)
        self.synchronize()
        return self._h_i32_np[:n].copy()

    def _zero_i32(self, arr):
        self._launch(gpu.k_fill_i32, arr.shape[0], [arr, 0])

    # ---- state -----------------------------------------------------------------------------

    @staticmethod
    def empty_stats() -> dict:
        """The statistics refreshed by every finalised frame.

        cond: min/max eigenvalue of H normalised by its diagonal, an inverse condition number (1 = every direction
        equally constrained, 0 = a direction the scan does not constrain at all).
        map_overflow: points that found no slot in the registration map within the probe window (map full)."""
        return {"frames": 0, "iters": 0, "corr": 0, "ds": 0, "rms": 0.0, "cond": 1.0, "ms": 0.0,
                "weak": False, "map_voxels": 0, "map_full": 0.0, "map_overflow": 0, "skipped": 0, "warm": True,
                "speed": 0.0, "turn": 0.0, "gaps": 0, "degen": 0, "keyframes": 0}

    def reset(self, T0: np.ndarray):
        """Forget the map and the trajectory; the next frame is posed at T0."""
        self._launch(gpu.k_fill_i64, self.reg_cap, [self.reg_keys, wp.int64(-1)])
        self._launch(gpu.k_zero_vec3, self.reg_cap, [self.reg_sum])
        self._launch(gpu.k_zero_mat33, self.reg_cap, [self.reg_outer])
        self._zero_i32(self.reg_cnt)
        self._zero_i32(self.reg_stats)
        self.T = np.array(T0, dtype=np.float64)
        self.T_prev = None
        self.delta = np.eye(4)
        self.xi = np.zeros(6, np.float32)  # body-frame motion over one frame (rho, theta)
        self.xi_s = np.zeros(6)  # its exponential average (velocity_smooth), what the prediction uses
        self.k_open = None
        self.k_pred = -1
        self.map_frames = 0
        self.T_key = None  # pose of the last frame inserted into the map
        self.traj = []  # (frame id, sensor time at mid-frame, 4x4 pose)
        self.stats = self.empty_stats()
        self._zero_i32(self.f_count)

    def restart_clock(self):
        """The point timestamps jumped: keep the map and pose, restart frame counting at rest."""
        self.k_open = None
        self.k_pred = -1
        self.T_prev = None
        self.delta = np.eye(4)
        self.xi = np.zeros(6, np.float32)
        self.xi_s = np.zeros(6)
        self._zero_i32(self.f_count)

    def predict(self) -> np.ndarray:
        return self.T @ self.delta

    # ---- per batch -------------------------------------------------------------------------

    def push(self, sx, sa, st, n: int, t_min: float, t_max: float, on_frame):
        """Feed one staged batch (points, attributes, timestamps). on_frame(k, T, f_xyz, f_attr, f_t, f_frame, m)
        runs per completed frame, with the frame's points deskewed to mid-frame."""
        if n <= 0:
            return
        inv_dt = 1.0 / self.frame_dt
        k_lo = int(math.floor(max(t_min, 0.0) * inv_dt))
        k_hi = int(math.floor(max(t_max, 0.0) * inv_dt))
        k_prev = self.k_pred  # highest frame id that already has a pose entry (-1: none this clock)
        if self.k_open is not None and k_hi - self.k_open > self.max_gap:
            # data went missing: close what we have and restart counting from this batch, at rest
            if int(self._read_i32(self.f_count)[0]) > 0:
                self._finalize(self.k_open, on_frame)
            self.restart_clock()
            self.stats["gaps"] += 1
        if self.k_open is None:
            # a batch that itself spans a gap starts at its newest frame; the stale part is dropped
            self.k_open = k_lo if k_hi - k_lo <= self.max_gap else k_hi
            self._zero_i32(self.f_count)
            self.pipe.set_pose(self.k_open, self.T, self.xi, self.epoch)
            # ids this batch (or batches the worker dropped) skipped over still have points in the live
            # ring: hold them at the current pose instead of leaving their slots unposed
            lo = (k_prev + 1) if 0 <= k_prev < self.k_open else k_lo
            if lo < self.k_open:
                self.pipe.set_pose_range(lo, self.k_open, self.T, None, self.epoch)
            self.k_pred = self.k_open
        # every frame id in this batch needs a pose entry now, so its live points render somewhere sensible
        for k in range(max(self.k_pred + 1, self.k_open + 1), k_hi + 1):
            self.pipe.set_pose(k, self.predict(), self.xi, self.epoch)
            self.k_pred = k
        k_min = self.k_open - 1  # a slightly late point joins the open frame; older ones are dropped
        while True:
            self._launch(k_frame_append, n,
                         [sx, sa, st, inv_dt, k_min, self.k_open, self.frame_cap, self.f_count,
                          self.f_raw, self.f_attr, self.f_t, self.f_frame])
            if k_hi <= self.k_open:
                break
            self._finalize(self.k_open, on_frame)
            self.k_open += 1
            k_min = self.k_open  # points of the frame just closed must not be appended twice
            self._zero_i32(self.f_count)
            self.pipe.set_pose(self.k_open, self.predict(), self.xi, self.epoch)
            self.k_pred = max(self.k_pred, self.k_open)

    # ---- per frame -------------------------------------------------------------------------

    def _finalize(self, k: int, on_frame):
        """Close frame k: deskew and register it, grow the map, update the velocity, publish the pose."""
        t0 = time.perf_counter()
        m = min(int(self._read_i32(self.f_count)[0]), self.frame_cap)
        warm = self.map_frames < self.warmup
        t_mid = (k + 0.5) * self.frame_dt
        T, xi, ok = self._estimate(m, t_mid, warm)
        if ok and (warm or self._is_keyframe(T)):
            self._insert(m, T)
        if self.T_prev is not None and not warm:
            self._update_velocity(np.linalg.inv(self.T_prev) @ T)
        self.T_prev = self.T = T
        self.pipe.set_pose(k, T, xi, self.epoch)
        self.traj.append((k, t_mid, T.copy()))
        self._update_stats(t0)  # before on_frame, which snapshots the stats
        on_frame(k, T, self.f_xyz, self.f_attr, self.f_t, self.f_frame, m)

    def _estimate(self, m: int, t_mid: float, warm: bool):
        """Deskew the open frame (m points, mid-frame time t_mid) and register it. Returns (T, xi, ok): the pose,
        the body twist the points were deskewed with, and whether the pose is trusted (during warm-up the sensor
        is assumed to hold still; a frame too sparse to register keeps the prediction)."""
        xi = self.xi if (self.deskew and not warm) else np.zeros(6, np.float32)
        if m > 0:
            self._deskew(m, t_mid, xi)
        T = self.T if warm else self.predict()
        if m < self.min_points:
            self.stats["skipped"] += 1
            return T, xi, False
        if warm:
            return T, xi, True  # the map is still filling in: assume the sensor holds still
        T, ok = self._register(m, T)
        if ok and self.deskew and self.redeskew and self.T_prev is not None:
            xi_new = _body_twist(np.linalg.inv(self.T_prev) @ T)
            if (np.linalg.norm(xi_new[3:] - xi[3:]) > REDESKEW_ROT
                    or np.linalg.norm(xi_new[:3] - xi[:3]) > REDESKEW_TRANS):
                xi = xi_new
                self._deskew(m, t_mid, xi)
                T, ok = self._register(m, T)
        return T, xi, ok

    def _update_velocity(self, d: np.ndarray):
        """Fold the frame's motion d (relative to the previous frame) into the constant-velocity prediction."""
        if np.linalg.norm(d[:3, 3]) < MAX_FRAME_JUMP:
            self.delta = d
            self.xi = _body_twist(d)
            if self.velocity_smooth > 0.0:
                a = self.velocity_smooth
                self.xi_s = a * self.xi_s + (1.0 - a) * self.xi.astype(np.float64)
                self.delta = _twist_matrix(self.xi_s)
        else:  # a jump this large in one frame is a failure, not motion: predict rest from here
            self.delta = np.eye(4)
            self.xi = np.zeros(6, np.float32)
            self.xi_s = np.zeros(6)

    def _update_stats(self, t0: float):
        """Refresh the per-frame stats: map fill, speed and turn rate, and the time since t0."""
        nvox, overflow = (int(v) for v in self._read_i32(self.reg_stats, 2))
        self.stats["frames"] += 1
        self.stats["warm"] = self.map_frames < self.warmup
        self.stats["map_voxels"], self.stats["map_full"] = nvox, nvox / self.reg_cap
        self.stats["map_overflow"] = overflow
        self.stats["speed"] = float(np.linalg.norm(self.xi[:3]) / self.frame_dt)
        self.stats["turn"] = float(math.degrees(np.linalg.norm(self.xi[3:])) / self.frame_dt)
        self.stats["ms"] = (time.perf_counter() - t0) * 1e3

    def _deskew(self, m: int, t_mid: float, xi: np.ndarray):
        self._launch(k_deskew, m,
                     [self.f_raw, self.f_t, m, float(t_mid), 1.0 / self.frame_dt,
                      wp.vec3(*xi[:3]), wp.vec3(*xi[3:]), self.f_xyz])

    def _is_keyframe(self, T: np.ndarray) -> bool:
        if self.T_key is None:
            return True
        d = np.linalg.inv(self.T_key) @ T
        ang = math.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec()))
        return np.linalg.norm(d[:3, 3]) >= self.kf_dist or ang >= self.kf_angle

    def _insert(self, m: int, T: np.ndarray):
        """Add the frame, posed by T, to the registration map as a keyframe."""
        self._launch(k_reg_insert, m,
                     [self.f_xyz, self.f_attr, m, wp.mat44(*T.astype(np.float32).flatten()), 1.0 / self.reg_voxel,
                      self.reg_voxel, self.max_range, int(self.ret_mask), self.reg_cap - 1,
                      self.reg_keys, self.reg_sum, self.reg_outer, self.reg_cnt, self.reg_stats])
        self.map_frames += 1
        self.T_key = T.copy()
        self.stats["keyframes"] += 1

    # ---- registration ----------------------------------------------------------------------

    def _downsample(self, m: int) -> int:
        self._launch(gpu.k_fill_i64, self.ds_cap, [self.ds_keys, wp.int64(-1)])
        self._zero_i32(self.ds_count)
        self._launch(k_ds_insert, m,
                     [self.f_xyz, self.f_attr, m, 2.0 / self.reg_voxel, self.min_range, self.max_range,
                      int(self.ret_mask), self.ds_cap - 1, self.ds_keys, self.ds_count, self.ds_xyz])
        return int(self._read_i32(self.ds_count)[0])

    def _accumulate(self, n_ds: int, T: np.ndarray, reach: int, max_d2: float, noise2: float, tight2: float,
                    kappa: float) -> np.ndarray:
        """One k_icp pass over the thinned frame at pose T; returns the reduced accumulator (the ACC_* layout)."""
        self._launch(gpu.k_fill_f64, self.acc.shape[0], [self.acc, wp.float64(0.0)])
        self._launch(k_icp, n_ds,
                     [self.ds_xyz, n_ds, wp.mat44(*T.astype(np.float32).flatten()), 1.0 / self.reg_voxel,
                      self.reg_voxel, self.reg_cap - 1, self.reg_keys, self.reg_sum, self.reg_outer, self.reg_cnt,
                      reach, max_d2, noise2, tight2, self.alpha, kappa, self.acc])
        self._launch(k_reduce, ACC_STRIDE, [self.acc, self.acc_out])
        wp.copy(self._h_acc, self.acc_out, stream=self.stream)
        self.synchronize()
        return self._h_acc_np.copy()

    def _register(self, m: int, T_pred: np.ndarray):
        """Scan-to-map ICP of the deskewed frame, starting from T_pred. Returns (T, ok)."""
        n_ds = self._downsample(m)
        self.stats["ds"] = n_ds
        if n_ds < self.min_corr:
            return T_pred, False
        T = T_pred.copy()
        voxel = self.reg_voxel
        noise2 = self.noise * self.noise
        tight2 = (TIGHT_FRACTION * voxel / math.sqrt(12.0)) ** 2  # the variance of a uniform fill is voxel^2 / 12
        iters, corr, rms, cond, n_degen = 0, 0, 0.0, 1.0, 0
        H = np.eye(6)
        for it in range(self.max_iter):
            reach = 2 if it < WIDE_ITERS else 1  # wide search first, then local refinement
            max_d2 = ((reach + 0.5) * voxel) ** 2
            kappa = max(KAPPA_MIN, KAPPA_START * (KAPPA_DECAY ** it))  # Geman-McClure scale, shrinking
            v = self._accumulate(n_ds, T, reach, max_d2, noise2, tight2, kappa)
            corr = int(v[ACC_COUNT])
            if corr < self.min_corr:
                break
            H, b, Hg = _sym6(v[ACC_H:ACC_H + 21]), v[ACC_B:ACC_B + 6], _sym6(v[ACC_HG:ACC_HG + 21])
            L = max(v[ACC_RANGE] / corr, 1.0)  # mean range of the matched points
            rms = math.sqrt(max(v[ACC_RMS], 0.0) / max(v[ACC_WEIGHT], 1e-9))  # metric, along surface normals
            dg = np.diag(H)
            A = H + np.diag(1e-9 * dg.mean() + self.damping * dg)  # Levenberg-Marquardt
            rhs = -b
            xi = _twist_between(T, T_pred)  # the left twist from T back onto the prediction
            if self.prior_info > 0.0:
                # a fixed information per metre toward the prediction: negligible where the scan pins the pose
                Ji = np.linalg.inv(_sensor_basis(T, L))
                P = self.prior_info * (Ji.T @ Ji)
                A = A + P
                rhs = rhs + P @ xi
            try:
                if self.degen_thr > 0.0:
                    delta, (_, _, kept) = _constrained_step(A, rhs, Hg, T, L, self.degen_thr, xi)
                    n_degen = int((~kept).sum())
                else:
                    delta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                break
            T = _apply_twist(T, delta)
            iters = it + 1
            if np.linalg.norm(delta[:3]) < STEP_TOL and np.linalg.norm(delta[3:]) < STEP_TOL:
                break
        if corr >= self.min_corr:
            dg = np.sqrt(np.maximum(np.diag(H), 1e-12))
            ev = np.linalg.eigvalsh(H / np.outer(dg, dg))  # H normalised by its diagonal
            cond = float(max(ev[0], 0.0) / max(ev[-1], 1e-12))
        self.stats.update(iters=iters, corr=corr, rms=rms, cond=cond, weak=cond < WEAK_COND or n_degen > 0,
                          degen=n_degen)
        return T, corr >= self.min_corr
