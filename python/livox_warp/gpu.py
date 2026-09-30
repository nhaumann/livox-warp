"""Warp GPU pipeline for live LiDAR points.

Every per-point step after the socket runs as a Warp kernel: ring-buffer ingest, age / range /
reflectivity / noise-tag / return / crop filters, the per-frame pose (mount, or the odometry pose
of the frame the point belongs to), stream compaction, a voxel-hash integration map with
free-space carving, hash-grid denoising, PCA normals and curvature, colormapping, and the final
write into the OpenGL vertex buffers through CUDA-GL interop (no host round trip).

Poses: every ring point carries a frame id (floor(t / frame_dt)). A pose table, indexed by frame
id modulo POSE_SLOTS, maps a frame to its sensor-to-world matrix at mid-frame, plus the sensor's motion over that
frame (a body-frame twist) so each point can be deskewed to mid-frame by its own timestamp.
Without odometry every point is frame 0, the table holds the viewer mount pose and no motion;
with odometry each frame's entry is first the predicted pose and then the registered one, so
live points snap into place as soon as their frame is solved. Pose edits are collected on the
host and uploaded in one scatter before any kernel reads the table. Each slot also records which
frame (unmasked id) and which epoch it was written for; the epoch changes whenever the live clock
restarts or the map is cleared. With odometry on, a live point is drawn only once its own slot
holds its own frame's pose for the current epoch, so a frame the worker has not posed yet is
simply not drawn for a render frame instead of landing at the origin or at a stale pose.
"""

from __future__ import annotations

import threading
import time
from typing import Literal

import numpy as np
import warp as wp

# Color modes: the index into COLOR_MODES is what k_shade switches on.
COLOR_MODES = [
    "Reflectivity",
    "Height",
    "Range",
    "Age",
    "Curvature",
    "Normal",
    "Lit",
    "Return",
    "Noise tag",
    "Device",
    "Ground",
    "Clusters",
    "Speed",
    "Motion",
    "Changes",
    "Solid",
]
(MODE_REFLECTIVITY, MODE_HEIGHT, MODE_RANGE, MODE_AGE, MODE_CURVATURE, MODE_NORMAL, MODE_LIT, MODE_RETURN,
 MODE_TAG, MODE_DEVICE, MODE_GROUND, MODE_CLUSTERS, MODE_SPEED, MODE_MOTION, MODE_CHANGES, MODE_SOLID) = range(16)
SCALAR_MODES = {MODE_REFLECTIVITY, MODE_HEIGHT, MODE_RANGE, MODE_AGE, MODE_CURVATURE, MODE_LIT, MODE_GROUND, MODE_SPEED}
NEEDS_NORMALS = {MODE_CURVATURE, MODE_NORMAL, MODE_LIT}

MAP_COUNT_CAP = wp.constant(4096)  # stop adding to a voxel's float32 sums past this (hits keep counting)
POSE_SLOTS = 1 << 16
POSE_MASK = wp.constant(POSE_SLOTS - 1)
MAX_CLUSTERS = 4096
# Voxel keys (voxel_key_ijk): three VOXEL_KEY_BITS-bit fields in an int64, each an axis index plus VOXEL_KEY_BIAS
VOXEL_KEY_BITS = 21
VOXEL_KEY_BIAS = 1 << (VOXEL_KEY_BITS - 1)
VOXEL_KEY_MASK = (1 << VOXEL_KEY_BITS) - 1
# 0x9E3779B97F4A7C15 as a signed 64-bit constant (a literal that large would be emitted as int32 first)
HASH_MULTIPLIER = wp.constant(wp.int64(-7046029254386353131))

# Free-space carving (k_carve): where a ray stops short of its return, and what counts as piercing a voxel.
CARVE_MIN_RANGE = wp.constant(0.05)  # returns closer than this (m) are not traced
CARVE_MARGIN_VOXELS = wp.constant(3.0)  # stop at least this many voxels before the return ...
CARVE_MARGIN_FRACTION = wp.constant(0.03)  # ... and at least this fraction of the range
CARVE_GRAZING_VOXELS = wp.constant(1.5)  # a surface layer is this thick (voxels) along its normal ...
CARVE_MIN_COSINE = wp.constant(0.05)  # ... divided by at least this cosine of incidence
CARVE_PIERCE_PLANE = wp.constant(0.6)  # crossing a surface voxel counts within this (voxels) of its centroid
CARVE_PIERCE_POINT = wp.constant(0.5)  # crossing any other voxel counts within this (voxels) of its centroid


@wp.struct
class View:
    now: float
    persist: float  # seconds a point stays visible; <= 0 keeps everything
    min_range: float
    max_range: float
    min_refl: int
    noise_mask: int  # bit k keeps points whose spatial-noise confidence (tag bits 0-1) == k
    ret_mask: int  # bit k keeps return k
    crop_on: int
    crop_lo: wp.vec3
    crop_hi: wp.vec3
    inv_frame_dt: float  # > 0: deskew live points by the frame twist table (odometry on)
    epoch: int  # current pose epoch: with odometry on, only slots written for it are trusted
    # integration-map history, for the Motion label and ghost carving
    inv_voxel: float
    map_mask: int
    dyn_on: int  # label live points by map history (needs a hash probe per point)
    dyn_settle: float  # seconds a voxel must have existed before its points count as static
    carve_on: int  # hide map voxels that rays have passed through more than they were hit
    carve_ratio: float
    carve_min: int
    carve_stale: float  # a voxel hit within this many seconds is never a ghost, whatever passed through it


@wp.struct
class Shade:
    mode: int
    lo: float
    hi: float
    solid: wp.vec3
    now: float
    sensor: wp.vec3
    eye: wp.vec3
    denoise: int
    min_nbrs: int
    has_normals: int
    has_gnd: int
    has_cid: int
    has_dyn: int
    has_chg: int  # chg holds each point's distance to a prior map (Changes mode)
    chg_near: float  # below: matches the prior map (green)
    chg_far: float  # above: new or moved (red); in between amber
    hide_gnd: int
    hide_dyn: int
    hide_static: int


@wp.func
def refl_of(a: wp.uint32) -> int:
    return int(a & wp.uint32(255))


@wp.func
def tag_of(a: wp.uint32) -> int:
    return int((a >> wp.uint32(8)) & wp.uint32(255))


@wp.func
def ret_of(a: wp.uint32) -> int:
    return int((a >> wp.uint32(16)) & wp.uint32(255))


@wp.func
def dev_of(a: wp.uint32) -> int:
    return int(a >> wp.uint32(24))


@wp.func
def in_crop(w: wp.vec3, v: View) -> int:
    if v.crop_on == 0:
        return 1
    if w[0] < v.crop_lo[0] or w[1] < v.crop_lo[1] or w[2] < v.crop_lo[2]:
        return 0
    if w[0] > v.crop_hi[0] or w[1] > v.crop_hi[1] or w[2] > v.crop_hi[2]:
        return 0
    return 1


@wp.func
def keep_raw(p: wp.vec3, a: wp.uint32, v: View) -> int:
    """Filters that act on the raw return (sensor frame)."""
    r = wp.length(p)
    if r < v.min_range or r > v.max_range:
        return 0
    if refl_of(a) < v.min_refl:
        return 0
    if ((v.noise_mask >> (tag_of(a) & 3)) & 1) == 0:
        return 0
    if ((v.ret_mask >> (ret_of(a) & 3)) & 1) == 0:
        return 0
    return 1


@wp.func
def turbo(x: float) -> wp.vec3:
    """Google's polynomial fit of the Turbo colormap."""
    x = wp.clamp(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return wp.vec3(wp.clamp(r, 0.0, 1.0), wp.clamp(g, 0.0, 1.0), wp.clamp(b, 0.0, 1.0))


@wp.func
def palette(k: int) -> wp.vec3:
    if k == 0:
        return wp.vec3(0.90, 0.90, 0.90)
    if k == 1:
        return wp.vec3(1.00, 0.25, 0.20)
    if k == 2:
        return wp.vec3(1.00, 0.60, 0.10)
    if k == 3:
        return wp.vec3(1.00, 0.95, 0.20)
    if k == 4:
        return wp.vec3(0.25, 0.70, 1.00)
    if k == 5:
        return wp.vec3(0.45, 1.00, 0.45)
    return wp.vec3(0.85, 0.35, 1.00)


@wp.func
def hue_color(k: int) -> wp.vec3:
    """A distinct, saturated color per integer id (golden-ratio hue walk)."""
    x = float(k) * 0.6180339887 + 0.11
    h = (x - wp.floor(x)) * 6.0
    i = int(wp.floor(h))
    f = h - float(i)
    s = 0.72
    p = 1.0 - s
    q = 1.0 - s * f
    t = 1.0 - s * (1.0 - f)
    i = i % 6
    if i == 0:
        return wp.vec3(1.0, t, p)
    if i == 1:
        return wp.vec3(q, 1.0, p)
    if i == 2:
        return wp.vec3(p, 1.0, t)
    if i == 3:
        return wp.vec3(p, q, 1.0)
    if i == 4:
        return wp.vec3(t, p, 1.0)
    return wp.vec3(1.0, p, q)


@wp.func
def pack_rgba(c: wp.vec3, a: float) -> wp.uint32:
    r = wp.uint32(wp.clamp(c[0], 0.0, 1.0) * 255.0 + 0.5)
    g = wp.uint32(wp.clamp(c[1], 0.0, 1.0) * 255.0 + 0.5)
    b = wp.uint32(wp.clamp(c[2], 0.0, 1.0) * 255.0 + 0.5)
    al = wp.uint32(wp.clamp(a, 0.0, 1.0) * 255.0 + 0.5)
    return r | (g << wp.uint32(8)) | (b << wp.uint32(16)) | (al << wp.uint32(24))


@wp.func
def voxel_key_ijk(ix: int, iy: int, iz: int) -> wp.int64:
    """Pack three voxel indices into one int64 key: VOXEL_KEY_BITS per axis, each biased to be positive."""
    bias = wp.int64(VOXEL_KEY_BIAS)
    m = wp.int64(VOXEL_KEY_MASK)
    x = (wp.int64(ix) + bias) & m
    y = (wp.int64(iy) + bias) & m
    z = (wp.int64(iz) + bias) & m
    return x | (y << wp.int64(VOXEL_KEY_BITS)) | (z << wp.int64(2 * VOXEL_KEY_BITS))


@wp.func
def voxel_key(w: wp.vec3, inv: float) -> wp.int64:
    return voxel_key_ijk(int(wp.floor(w[0] * inv)), int(wp.floor(w[1] * inv)), int(wp.floor(w[2] * inv)))


@wp.func
def slot_hash(k: wp.int64) -> int:
    h = k * HASH_MULTIPLIER
    h = h ^ (h >> wp.int64(29))
    return int(h & wp.int64(2147483647))


@wp.func
def find_slot(key: wp.int64, keys: wp.array(dtype=wp.int64), mask: int) -> int:
    """Slot holding `key` in an open-addressed table, or -1."""
    h = slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        k = keys[s]
        if k == key:
            return s
        if k == wp.int64(-1):
            return -1
    return -1


@wp.func
def rotvec_rotate(r: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Rotate p by the rotation vector r (axis * angle)."""
    a = wp.length(r)
    if a < 1.0e-9:
        return p
    return wp.quat_rotate(wp.quat_from_axis_angle(r / a, a), p)


@wp.func
def deskew(p: wp.vec3, s: float, rho: wp.vec3, theta: wp.vec3) -> wp.vec3:
    """A point measured a fraction s of a frame after mid-frame -> the mid-frame sensor frame.

    (rho, theta) is the sensor's body-frame motion over one frame; first-order SE(3) interpolation.
    """
    return rotvec_rotate(theta * s, p) + rho * s


@wp.func
def transient(c: int, miss: int, v: View) -> int:
    """A map voxel that rays have passed through more often than returns have landed in it."""
    if miss < v.carve_min:
        return 0
    return wp.where(float(miss) > v.carve_ratio * float(c), 1, 0)


@wp.func
def ghost(c: int, miss: int, last: float, v: View) -> int:
    """A transient voxel that nothing lands in any more: something that moved away."""
    if v.now - last < v.carve_stale:
        return 0
    return transient(c, miss, v)


# --------------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------------


@wp.kernel
def k_pose_scatter(
    idx: wp.array(dtype=wp.int32),
    mats: wp.array(dtype=wp.mat44),
    rho: wp.array(dtype=wp.vec3),
    theta: wp.array(dtype=wp.vec3),
    poses: wp.array(dtype=wp.mat44),
    tw_rho: wp.array(dtype=wp.vec3),
    tw_theta: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.int32),
    epochs: wp.array(dtype=wp.int32),
    owner_frame: wp.array(dtype=wp.int32),
    owner_epoch: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    j = idx[i]
    poses[j] = mats[i]
    tw_rho[j] = rho[i]
    tw_theta[j] = theta[i]
    owner_frame[j] = frames[i]
    owner_epoch[j] = epochs[i]


@wp.kernel
def k_rebase_clock(keys: wp.array(dtype=wp.int64), first: wp.array(dtype=wp.float32),
                   last: wp.array(dtype=wp.float32), shift: float):
    """Move the map's first/last-seen times onto a restarted clock (new = old - shift)."""
    s = wp.tid()
    if keys[s] != wp.int64(-1):
        first[s] = first[s] - shift
        last[s] = last[s] - shift


@wp.kernel
def k_frame_of(t: wp.array(dtype=wp.float32), inv_dt: float, out: wp.array(dtype=wp.int32)):
    """Frame id of each staged point: floor(t / frame_dt), unmasked (readers mask it into the pose
    table and compare the full id with the slot's owner); 0 when inv_dt is 0."""
    i = wp.tid()
    if inv_dt <= 0.0:
        out[i] = 0
        return
    out[i] = int(wp.floor(wp.max(t[i], 0.0) * inv_dt))


@wp.kernel
def k_ingest(
    src_xyz: wp.array(dtype=wp.vec3),
    src_attr: wp.array(dtype=wp.uint32),
    src_t: wp.array(dtype=wp.float32),
    src_frame: wp.array(dtype=wp.int32),
    offset: int,
    head: int,
    cap: int,
    ring_xyz: wp.array(dtype=wp.vec3),
    ring_attr: wp.array(dtype=wp.uint32),
    ring_t: wp.array(dtype=wp.float32),
    ring_frame: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    s = (head + i) % cap
    ring_xyz[s] = src_xyz[offset + i]
    ring_attr[s] = src_attr[offset + i]
    ring_t[s] = src_t[offset + i]
    ring_frame[s] = src_frame[offset + i]


@wp.kernel
def k_map_insert(
    src_xyz: wp.array(dtype=wp.vec3),
    src_attr: wp.array(dtype=wp.uint32),
    src_t: wp.array(dtype=wp.float32),
    src_frame: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.mat44),
    v: View,
    inv_voxel: float,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    last: wp.array(dtype=wp.float32),
    first: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = src_xyz[i]
    a = src_attr[i]
    if keep_raw(p, a, v) == 0:
        return
    w = wp.transform_point(poses[src_frame[i] & POSE_MASK], p)
    key = voxel_key(w, inv_voxel)
    h = slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            if prev == wp.int64(-1):
                wp.atomic_add(stats, 0, 1)
            # hits keep counting (carving compares them with misses); only the float32 sums stop,
            # and exactly MAP_COUNT_CAP of them are added because atomic_add returns unique old values
            old = wp.atomic_add(cnt, s, 1)
            if old < MAP_COUNT_CAP:
                wp.atomic_add(acc, s, wp.vec4(w[0], w[1], w[2], float(refl_of(a))))
            wp.atomic_max(last, s, src_t[i])
            wp.atomic_min(first, s, src_t[i])
            return
    wp.atomic_add(stats, 1, 1)  # table full along this probe chain


@wp.func
def map_centroid(s: int, acc: wp.array(dtype=wp.vec4), cnt: wp.array(dtype=wp.int32)) -> wp.vec3:
    av = acc[s]
    m = 1.0 / float(wp.min(cnt[s], MAP_COUNT_CAP))
    return wp.vec3(av[0] * m, av[1] * m, av[2] * m)


@wp.func
def local_plane(
    ix: int,
    iy: int,
    iz: int,
    origin: wp.vec3,
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    mask: int,
) -> wp.vec4:
    """Normal of the surface through map voxel (ix, iy, iz), from a PCA of the centroids of the occupied
    voxels in its 3x3x3 neighbourhood (taken relative to `origin`, a point nearby, for float32 precision).
    w = 2 when they form a surface (smallest spread well below the middle one), w = 1 when at least 4
    are occupied but they do not (an edge, a line or a blob: the normal is the least-spread direction),
    w = 0 when fewer are."""
    cs = wp.vec3()
    cc = wp.mat33()
    cn = float(0.0)
    for gx in range(-1, 2):
        for gy in range(-1, 2):
            for gz in range(-1, 2):
                sn = find_slot(voxel_key_ijk(ix + gx, iy + gy, iz + gz), keys, mask)
                if sn >= 0:
                    if cnt[sn] > 0:
                        q = map_centroid(sn, acc, cnt) - origin
                        cs += q
                        cc += wp.outer(q, q)
                        cn += 1.0
    if cn < 4.0:
        return wp.vec4(0.0, 0.0, 1.0, 0.0)
    mu = cs / cn
    Q, ev = wp.eig3(cc / cn - wp.outer(mu, mu))
    k = int(0)
    if ev[1] < ev[k]:
        k = 1
    if ev[2] < ev[k]:
        k = 2
    j = (k + 1) % 3
    j2 = (k + 2) % 3
    mid = wp.min(ev[j], ev[j2])
    ok = float(1.0)
    if ev[k] < 0.25 * mid:
        ok = 2.0
    return wp.vec4(Q[0, k], Q[1, k], Q[2, k], ok)


@wp.kernel
def k_carve(
    xyz: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=wp.uint32),
    frames: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.mat44),
    n: int,
    every: int,
    inv_voxel: float,
    mask: int,
    max_range: float,
    max_steps: int,
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    miss: wp.array(dtype=wp.int32),
):
    """Walk each return's ray through the integration map and count the occupied voxels it pierces.

    A voxel a ray passes through should be empty: many misses against few hits marks a ghost
    (something that moved away). Two guards keep static surfaces from being carved:
    - the walk stops a margin short of the return, and the margin grows as the ray grazes the
      surface it lands on (its normal comes from the voxel centroids around the endpoint), since a
      grazing ray runs inside that surface's own voxel layer for voxel / sin(angle) metres;
    - a crossed voxel counts as a miss only if the ray crosses the surface patch inside it: where the
      voxel's neighbourhood forms a surface, the ray must cross that plane within 0.6 voxel of the
      voxel's centroid (a ray skimming over a floor never does); elsewhere it must pass within half a
      voxel of the centroid, not merely through a corner of the cube.
    Only every `every`-th return is traced, so each miss counts `every` times: misses and hits then
    stay on the same scale whatever the sampling.
    """
    i = wp.tid() * every
    if i >= n:
        return
    if tag_of(attr[i]) != 0:
        return
    T = poses[frames[i] & POSE_MASK]
    o = wp.vec3(T[0, 3], T[1, 3], T[2, 3])
    w = wp.transform_point(T, xyz[i])
    d = w - o
    L = wp.length(d)
    if L < CARVE_MIN_RANGE or L > max_range:
        return
    d = d / L
    voxel = 1.0 / inv_voxel
    margin = wp.max(CARVE_MARGIN_VOXELS * voxel, CARVE_MARGIN_FRACTION * L)
    # grazing incidence: the landing surface's normal from the voxel centroids around the endpoint
    pe = local_plane(int(wp.floor(w[0] * inv_voxel)), int(wp.floor(w[1] * inv_voxel)),
                     int(wp.floor(w[2] * inv_voxel)), w, keys, acc, cnt, mask)
    if pe[3] > 0.5:
        cosine = wp.abs(pe[0] * d[0] + pe[1] * d[1] + pe[2] * d[2])
        margin = wp.max(margin, CARVE_GRAZING_VOXELS * voxel / wp.max(cosine, CARVE_MIN_COSINE))
    stop = (L - margin) * inv_voxel  # in voxel units
    if stop <= 1.0:
        return
    pos = o * inv_voxel
    ix = int(wp.floor(pos[0]))
    iy = int(wp.floor(pos[1]))
    iz = int(wp.floor(pos[2]))
    sx = int(1)
    sy = int(1)
    sz = int(1)
    tmx = float(1.0e30)
    tmy = float(1.0e30)
    tmz = float(1.0e30)
    tdx = float(1.0e30)
    tdy = float(1.0e30)
    tdz = float(1.0e30)
    if d[0] > 1.0e-9:
        tmx = (float(ix + 1) - pos[0]) / d[0]
        tdx = 1.0 / d[0]
    elif d[0] < -1.0e-9:
        sx = -1
        tmx = (pos[0] - float(ix)) / (-d[0])
        tdx = -1.0 / d[0]
    if d[1] > 1.0e-9:
        tmy = (float(iy + 1) - pos[1]) / d[1]
        tdy = 1.0 / d[1]
    elif d[1] < -1.0e-9:
        sy = -1
        tmy = (pos[1] - float(iy)) / (-d[1])
        tdy = -1.0 / d[1]
    if d[2] > 1.0e-9:
        tmz = (float(iz + 1) - pos[2]) / d[2]
        tdz = 1.0 / d[2]
    elif d[2] < -1.0e-9:
        sz = -1
        tmz = (pos[2] - float(iz)) / (-d[2])
        tdz = -1.0 / d[2]
    for step in range(max_steps):
        t = float(0.0)
        if tmx < tmy and tmx < tmz:
            t = tmx
            ix += sx
            tmx += tdx
        elif tmy < tmz:
            t = tmy
            iy += sy
            tmy += tdy
        else:
            t = tmz
            iz += sz
            tmz += tdz
        if t >= stop:
            return
        s = find_slot(voxel_key_ijk(ix, iy, iz), keys, mask)
        if s >= 0:
            if cnt[s] > 0:
                cen = map_centroid(s, acc, cnt)
                rel = cen - o
                pierced = int(0)
                pl = local_plane(ix, iy, iz, cen, keys, acc, cnt, mask)
                if pl[3] > 1.5:
                    nn = wp.vec3(pl[0], pl[1], pl[2])
                    dn = wp.dot(d, nn)
                    if wp.abs(dn) > 1.0e-4:
                        x = d * (wp.dot(rel, nn) / dn) - rel  # plane crossing, relative to the centroid
                        if wp.length(x) < CARVE_PIERCE_PLANE * voxel:
                            pierced = 1
                else:
                    perp = rel - wp.dot(rel, d) * d
                    if wp.length(perp) < CARVE_PIERCE_POINT * voxel:
                        pierced = 1
                if pierced != 0:
                    wp.atomic_add(miss, s, every)


@wp.func
def settled_near(
    ix: int,
    iy: int,
    iz: int,
    v: View,
    map_keys: wp.array(dtype=wp.int64),
    map_cnt: wp.array(dtype=wp.int32),
    map_first: wp.array(dtype=wp.float32),
    map_miss: wp.array(dtype=wp.int32),
) -> int:
    """1 when map voxel (ix, iy, iz) or one of its 6 neighbours has existed for dyn_settle seconds and is
    not a ghost. The neighbours matter for the Mid-40: its non-repetitive pattern keeps filling in
    first-time voxels on static surfaces (and samples a far surface too sparsely for every fine voxel
    to be old), and those fill-ins sit next to old voxels of the same surface."""
    for nb in range(7):
        jx = ix
        jy = iy
        jz = iz
        if nb == 1:
            jx = ix - 1
        elif nb == 2:
            jx = ix + 1
        elif nb == 3:
            jy = iy - 1
        elif nb == 4:
            jy = iy + 1
        elif nb == 5:
            jz = iz - 1
        elif nb == 6:
            jz = iz + 1
        s = find_slot(voxel_key_ijk(jx, jy, jz), map_keys, v.map_mask)
        if s >= 0:
            age = v.now - map_first[s]
            if age < 0.0:
                age = 1.0e9  # clock restarted: history predates the new clock, treat as old
            if age >= v.dyn_settle and transient(map_cnt[s], map_miss[s], v) == 0:
                return 1
    return 0


@wp.kernel
def k_classify_live(
    ring_xyz: wp.array(dtype=wp.vec3),
    ring_attr: wp.array(dtype=wp.uint32),
    ring_t: wp.array(dtype=wp.float32),
    ring_frame: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.mat44),
    tw_rho: wp.array(dtype=wp.vec3),
    tw_theta: wp.array(dtype=wp.vec3),
    owner_frame: wp.array(dtype=wp.int32),
    owner_epoch: wp.array(dtype=wp.int32),
    filled: int,
    v: View,
    map_keys: wp.array(dtype=wp.int64),
    map_cnt: wp.array(dtype=wp.int32),
    map_first: wp.array(dtype=wp.float32),
    map_last: wp.array(dtype=wp.float32),
    map_miss: wp.array(dtype=wp.int32),
    flag: wp.array(dtype=wp.int32),
    out_xyz: wp.array(dtype=wp.vec3),
    out_attr: wp.array(dtype=wp.uint32),
    out_t: wp.array(dtype=wp.float32),
    out_dyn: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    flag[i] = 0
    if i >= filled:
        return
    t = ring_t[i]
    if v.persist > 0.0:
        if v.now - t > v.persist:
            return
    p = ring_xyz[i]
    a = ring_attr[i]
    if keep_raw(p, a, v) == 0:
        return
    fid = ring_frame[i]
    f = fid & POSE_MASK
    if v.inv_frame_dt > 0.0:
        # odometry on: draw only once this slot holds this frame's pose for the current epoch
        if owner_frame[f] != fid or owner_epoch[f] != v.epoch:
            return
        x = t * v.inv_frame_dt
        frac = x - wp.floor(x) - 0.5  # the point's time from mid-frame, in frames: [-0.5, 0.5)
        p = deskew(p, frac, tw_rho[f], tw_theta[f])
    w = wp.transform_point(poses[f], p)
    if in_crop(w, v) == 0:
        return
    flag[i] = 1
    out_xyz[i] = w
    out_attr[i] = a
    out_t[i] = t
    dyn = int(0)
    if v.dyn_on != 0:
        # a point is static when the map voxel it lands in, or one of its 6 neighbours, is old and no ghost
        ix = int(wp.floor(w[0] * v.inv_voxel))
        iy = int(wp.floor(w[1] * v.inv_voxel))
        iz = int(wp.floor(w[2] * v.inv_voxel))
        dyn = 1 - settled_near(ix, iy, iz, v, map_keys, map_cnt, map_first, map_miss)
    out_dyn[i] = dyn


@wp.kernel
def k_classify_map(
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    last: wp.array(dtype=wp.float32),
    first: wp.array(dtype=wp.float32),
    miss: wp.array(dtype=wp.int32),
    v: View,
    min_count: int,
    flag: wp.array(dtype=wp.int32),
    out_xyz: wp.array(dtype=wp.vec3),
    out_attr: wp.array(dtype=wp.uint32),
    out_t: wp.array(dtype=wp.float32),
    out_dyn: wp.array(dtype=wp.int32),
):
    s = wp.tid()
    flag[s] = 0
    if keys[s] == wp.int64(-1):
        return
    c = cnt[s]
    if c < min_count:
        return
    if v.persist > 0.0:
        if v.now - last[s] > v.persist:
            return
    g = ghost(c, miss[s], last[s], v)
    if v.carve_on != 0 and g != 0:
        return
    a = acc[s]
    inv = 1.0 / float(wp.min(c, MAP_COUNT_CAP))
    w = wp.vec3(a[0] * inv, a[1] * inv, a[2] * inv)
    if in_crop(w, v) == 0:
        return
    refl = wp.min(int(a[3] * inv + 0.5), 255)
    flag[s] = 1
    out_xyz[s] = w
    # mean reflectivity only: tag, return and device carry no meaning for a merged voxel
    out_attr[s] = wp.uint32(refl)
    out_t[s] = last[s]
    # moving: rays pass through it (transient), or it is new and so is everything around it (a
    # first-time voxel next to an old one is the scan pattern filling in a known surface)
    dyn = int(0)
    if transient(c, miss[s], v) != 0:
        dyn = 1
    elif settled_near(int(wp.floor(w[0] * v.inv_voxel)), int(wp.floor(w[1] * v.inv_voxel)),
                      int(wp.floor(w[2] * v.inv_voxel)), v, keys, cnt, first, miss) == 0:
        dyn = 1
    out_dyn[s] = dyn


@wp.kernel
def k_scatter(
    flag: wp.array(dtype=wp.int32),
    offs: wp.array(dtype=wp.int32),
    in_xyz: wp.array(dtype=wp.vec3),
    in_attr: wp.array(dtype=wp.uint32),
    in_t: wp.array(dtype=wp.float32),
    in_dyn: wp.array(dtype=wp.int32),
    out_xyz: wp.array(dtype=wp.vec3),
    out_attr: wp.array(dtype=wp.uint32),
    out_t: wp.array(dtype=wp.float32),
    out_dyn: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if flag[i] != 0:
        j = offs[i]
        out_xyz[j] = in_xyz[i]
        out_attr[j] = in_attr[i]
        out_t[j] = in_t[i]
        out_dyn[j] = in_dyn[i]


@wp.kernel
def k_total(flag: wp.array(dtype=wp.int32), offs: wp.array(dtype=wp.int32), n: int, out: wp.array(dtype=wp.int32)):
    out[0] = offs[n - 1] + flag[n - 1]


# Neighborhood analysis runs on a per-frame voxel summary of the visible points, not on the raw
# points: an accumulated Livox cloud puts ~30,000 points inside a 10 cm radius, which made a raw
# hash-grid pass cost a second per frame. Voxels a third of the radius wide leave ~30 weighted
# neighbors per query, with the same answers (counts are weighted by points per voxel).


@wp.kernel
def k_nb_insert(
    xyz: wp.array(dtype=wp.vec3),
    inv_h: float,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    slot_of_pt: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    w = xyz[i]
    key = voxel_key(w, inv_h)
    h = slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            wp.atomic_add(acc, s, wp.vec4(w[0], w[1], w[2], 1.0))
            slot_of_pt[i] = s
            return
    slot_of_pt[i] = -1


@wp.kernel
def k_nb_flag(keys: wp.array(dtype=wp.int64), flag: wp.array(dtype=wp.int32)):
    s = wp.tid()
    flag[s] = wp.where(keys[s] != wp.int64(-1), 1, 0)


@wp.kernel
def k_nb_compact(
    flag: wp.array(dtype=wp.int32),
    offs: wp.array(dtype=wp.int32),
    acc: wp.array(dtype=wp.vec4),
    vidx: wp.array(dtype=wp.int32),
    v_xyz: wp.array(dtype=wp.vec3),
    v_w: wp.array(dtype=wp.float32),
):
    s = wp.tid()
    if flag[s] != 0:
        j = offs[s]
        a = acc[s]
        v_xyz[j] = wp.vec3(a[0], a[1], a[2]) / a[3]
        v_w[j] = a[3]
        vidx[s] = j


@wp.kernel
def k_nb_voxel(
    grid: wp.uint64,
    v_xyz: wp.array(dtype=wp.vec3),
    v_w: wp.array(dtype=wp.float32),
    radius: float,
    want_normals: int,
    sensor: wp.vec3,
    out_cnt: wp.array(dtype=wp.float32),
    out_n: wp.array(dtype=wp.vec3),
    out_curv: wp.array(dtype=wp.float32),
):
    """Point-weighted neighbor count, and optionally a PCA normal + curvature, per voxel."""
    i = wp.tid()
    p = v_xyz[i]
    r2 = radius * radius
    q = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    wsum = float(0.0)
    s1 = wp.vec3()
    s2 = wp.mat33()
    while wp.hash_grid_query_next(q, j):
        d = v_xyz[j] - p
        if wp.dot(d, d) <= r2:
            w = v_w[j]
            wsum += w
            s1 += w * d
            s2 += w * wp.outer(d, d)
    out_cnt[i] = wsum
    if want_normals == 0:
        return
    to_sensor = wp.normalize(sensor - p)
    inv = 1.0 / wp.max(wsum, 1.0)
    mu = s1 * inv
    cov = s2 * inv - wp.outer(mu, mu)
    Q, ev = wp.eig3(cov)
    k = int(0)
    if ev[1] < ev[k]:
        k = 1
    if ev[2] < ev[k]:
        k = 2
    n = wp.normalize(wp.vec3(Q[0, k], Q[1, k], Q[2, k]))
    if wp.dot(n, to_sensor) < 0.0:
        n = -n
    tr = ev[0] + ev[1] + ev[2]
    if tr <= 1.0e-12:
        n = to_sensor  # a lone voxel: no surface to fit
    out_n[i] = n
    out_curv[i] = wp.max(ev[k], 0.0) / wp.max(tr, 1.0e-12)


@wp.kernel
def k_nb_gather(
    slot_of_pt: wp.array(dtype=wp.int32),
    vidx: wp.array(dtype=wp.int32),
    v_cnt: wp.array(dtype=wp.float32),
    v_n: wp.array(dtype=wp.vec3),
    v_curv: wp.array(dtype=wp.float32),
    want_normals: int,
    out_cnt: wp.array(dtype=wp.int32),
    out_n: wp.array(dtype=wp.vec3),
    out_curv: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = slot_of_pt[i]
    if s < 0:
        out_cnt[i] = 1 << 30  # table overflow: never hide the point
        out_n[i] = wp.vec3(0.0, 0.0, 1.0)
        out_curv[i] = 0.0
        return
    v = vidx[s]
    out_cnt[i] = int(v_cnt[v] + 0.5)
    if want_normals != 0:
        out_n[i] = v_n[v]
        out_curv[i] = v_curv[v]


@wp.kernel
def k_shade(
    xyz: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=wp.uint32),
    t: wp.array(dtype=wp.float32),
    nrm: wp.array(dtype=wp.vec3),
    curv: wp.array(dtype=wp.float32),
    nbr: wp.array(dtype=wp.int32),
    gnd: wp.array(dtype=wp.int32),
    hag: wp.array(dtype=wp.float32),
    cid: wp.array(dtype=wp.int32),
    cl_vel: wp.array(dtype=wp.vec3),
    cl_tid: wp.array(dtype=wp.int32),
    dyn: wp.array(dtype=wp.int32),
    chg: wp.array(dtype=wp.float32),
    s: Shade,
    out_pos: wp.array(dtype=wp.vec3),
    out_col: wp.array(dtype=wp.uint32),
    out_nrm: wp.array(dtype=wp.vec3),
    out_scalar: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = xyz[i]
    a = attr[i]
    out_pos[i] = p
    if s.has_normals != 0:
        out_nrm[i] = nrm[i]
    else:
        out_nrm[i] = wp.vec3(0.0, 0.0, 1.0)
    alpha = 1.0
    if s.denoise != 0:
        if nbr[i] < s.min_nbrs:
            alpha = 0.0
    is_gnd = int(0)
    if s.has_gnd != 0:
        is_gnd = gnd[i]
    is_dyn = int(0)
    if s.has_dyn != 0:
        is_dyn = dyn[i]
    c = int(-1)
    if s.has_cid != 0:
        c = cid[i]
    if s.hide_gnd != 0 and is_gnd != 0:
        alpha = 0.0
    if s.hide_dyn != 0 and is_dyn != 0:
        alpha = 0.0
    if s.hide_static != 0 and is_dyn == 0:
        alpha = 0.0
    val = float(0.0)
    col = s.solid
    m = s.mode
    span = wp.max(s.hi - s.lo, 1.0e-6)
    if m == 0:
        val = float(refl_of(a))
    elif m == 1:
        val = p[2]
    elif m == 2:
        val = wp.length(p - s.sensor)
    elif m == 3:
        val = s.now - t[i]
    elif m == 4:
        val = curv[i]
    elif m == 6:
        val = float(refl_of(a))
    elif m == 10:
        val = hag[i]
    elif m == 12:
        if c >= 0:
            val = wp.length(cl_vel[wp.min(c, MAX_CLUSTERS - 1)])
    elif m == 14:
        if s.has_chg != 0:
            val = chg[i]
    if m == 0 or m == 1 or m == 2 or m == 3 or m == 4:
        col = turbo((val - s.lo) / span)
    elif m == 5:
        n = nrm[i]
        col = wp.vec3(wp.abs(n[0]), wp.abs(n[1]), wp.abs(n[2]))
    elif m == 6:
        base = turbo((val - s.lo) / span)
        lambert = wp.abs(wp.dot(nrm[i], wp.normalize(s.eye - p)))
        col = base * (0.25 + 0.75 * lambert)
    elif m == 7:
        col = palette(ret_of(a) & 3)
    elif m == 8:
        col = palette(tag_of(a) & 3)
    elif m == 9:
        col = palette(4 + (dev_of(a) % 3))
    elif m == 10:
        if is_gnd != 0:
            col = wp.vec3(0.42, 0.38, 0.34)
        else:
            col = turbo((val - s.lo) / span)
    elif m == 11:
        if c < 0:
            col = wp.vec3(0.30, 0.31, 0.34)
        else:
            tid = cl_tid[wp.min(c, MAX_CLUSTERS - 1)]
            col = hue_color(wp.where(tid >= 0, tid, c + 1000))
    elif m == 12:
        if c < 0:
            col = wp.vec3(0.30, 0.31, 0.34)
        else:
            col = turbo((val - s.lo) / span)
    elif m == 13:
        if is_dyn != 0:
            col = wp.vec3(1.0, 0.32, 0.22)
        else:
            col = wp.vec3(0.52, 0.55, 0.60)
    elif m == 14:
        if s.has_chg == 0:
            col = wp.vec3(0.52, 0.55, 0.60)  # not localised in a prior map yet
        elif val < s.chg_near:
            col = wp.vec3(0.30, 0.85, 0.40)  # where the prior map has a surface
        elif val < s.chg_far:
            col = wp.vec3(1.00, 0.72, 0.18)  # close to one: slightly moved, or the edge of something new
        else:
            col = wp.vec3(1.00, 0.24, 0.20)  # nothing in the prior map here: new or moved
    out_col[i] = pack_rgba(col, alpha)
    out_scalar[i] = val


@wp.kernel
def k_gather_stride(src: wp.array(dtype=wp.float32), stride: int, n: int, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    j = i * stride
    if j < n:
        out[i] = src[j]


@wp.kernel
def k_gather_z(src: wp.array(dtype=wp.vec3), stride: int, n: int, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    j = i * stride
    if j < n:
        out[i] = src[j][2]


@wp.kernel
def k_gather_xyz(src: wp.array(dtype=wp.vec3), stride: int, n: int, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    j = i * stride
    if j < n:
        out[i] = src[j]


# Fills and clears as kernels: unlike array.fill_() / zero_() they run on whatever stream the caller
# launches them on, which is what lets other threads keep their own streams (see odom.py).


@wp.kernel
def k_fill_i32(a: wp.array(dtype=wp.int32), v: int):
    a[wp.tid()] = v


@wp.kernel
def k_fill_i64(a: wp.array(dtype=wp.int64), v: wp.int64):
    a[wp.tid()] = v


@wp.kernel
def k_fill_f64(a: wp.array(dtype=wp.float64), v: wp.float64):
    a[wp.tid()] = v


@wp.kernel
def k_zero_vec3(a: wp.array(dtype=wp.vec3)):
    a[wp.tid()] = wp.vec3()


@wp.kernel
def k_zero_mat33(a: wp.array(dtype=wp.mat33)):
    a[wp.tid()] = wp.mat33(0.0)


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


class HostMirror:
    """A pinned host buffer for reading small device arrays on one explicit stream.

    array.numpy() copies on the legacy default stream, which waits for every other blocking stream,
    so a render-thread readback would stall behind the odometry worker's kernels. This copies on the
    given stream and waits for that stream only. The numpy view is taken once, at construction.
    """

    def __init__(self, device, dtype, n: int):
        self.device = wp.get_device(device)
        self.host = wp.zeros(n, dtype=dtype, device="cpu", pinned=self.device.is_cuda)
        self.np = self.host.numpy()

    def read(self, arr, count: int, stream=None, sync: bool = True):
        """Copy the first `count` elements; returns a numpy view valid until the next read."""
        count = min(count, self.host.shape[0])
        if count <= 0:
            return self.np[:0]
        if not self.device.is_cuda:
            wp.copy(self.host, arr, count=count)
            return self.np[:count]
        wp.copy(self.host, arr, count=count, stream=stream)
        if sync:
            wp.synchronize_stream(stream)
        return self.np[:count]


def mount_matrix(roll: float, pitch: float, yaw: float, x: float, y: float, z: float) -> np.ndarray:
    """Livox extrinsic convention: rotate about x (roll), then y (pitch), then z (yaw); degrees, metres."""
    r, p, w = np.radians([roll, pitch, yaw])
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(w), np.sin(w)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = (x, y, z)
    return m.astype(np.float32)


class _LaunchTimer:
    """GPU time of the launches issued between start() and stop(), readable once the stream has synced.

    Launches return before the kernel runs, so a host clock around them would time the enqueue, not the
    work. On the CPU backend launches are synchronous and the host clock is right.
    """

    def __init__(self, device, stream):
        self.stream = stream if device.is_cuda else None
        self.ms = 0.0
        self._t0 = 0.0
        self._pending = False
        if self.stream is not None:
            self._ev0 = wp.Event(device, enable_timing=True)
            self._ev1 = wp.Event(device, enable_timing=True)

    def start(self):
        if self.stream is not None:
            wp.record_event(self._ev0, self.stream)
        else:
            self._t0 = time.perf_counter()

    def stop(self):
        if self.stream is not None:
            wp.record_event(self._ev1, self.stream)
            self._pending = True
        else:
            self.ms = (time.perf_counter() - self._t0) * 1e3

    def read(self) -> float:
        """The last measured time in ms (waits for the end event if the stream has not passed it yet)."""
        if self._pending:
            self.ms = float(wp.get_event_elapsed_time(self._ev0, self._ev1))
            self._pending = False
        return self.ms


class Pipeline:
    """GPU state for one viewer: live ring buffer, pose table, integration map, per-frame work arrays.

    Threading: every launch goes to the calling thread's current stream, so a Pipeline is driven from
    one thread (the viewer's render thread). The exceptions are set_pose and set_pose_range, which only
    stage host-side edits under a lock and may be called from the odometry worker; flush_poses, called
    by the render thread, uploads them.
    """

    def __init__(self, ring_capacity: int = 1 << 23, map_capacity: int = 1 << 23, device=None,
                 nb_capacity: int = 1 << 22):
        """ring_capacity: live points kept; map_capacity and nb_capacity: hash-table slots for the integration
        map and for the per-frame voxel summary of the neighbourhood analysis, both powers of two."""
        self.device = wp.get_device(device)
        d = self.device
        # the render thread's stream: the device's current stream at construction (the default stream)
        self.stream = wp.get_stream(d) if d.is_cuda else None
        self._h_total = HostMirror(d, wp.int32, 1)
        self._h_stats = HostMirror(d, wp.int32, 2)
        self.cap = int(ring_capacity)
        self.map_cap = int(map_capacity)
        for name, cap in (("map_capacity", self.map_cap), ("nb_capacity", int(nb_capacity))):
            if cap <= 0 or cap & (cap - 1):
                raise ValueError(f"{name} must be a power of two, got {cap}")
        self.work_cap = max(self.cap, self.map_cap)

        self.ring_xyz = wp.zeros(self.cap, dtype=wp.vec3, device=d)
        self.ring_attr = wp.zeros(self.cap, dtype=wp.uint32, device=d)
        self.ring_t = wp.zeros(self.cap, dtype=wp.float32, device=d)
        self.ring_frame = wp.zeros(self.cap, dtype=wp.int32, device=d)
        self.head = 0
        self.filled = 0

        # frame id -> sensor-to-world pose at mid-frame, and the sensor's body-frame motion over the frame
        self.poses = wp.zeros(POSE_SLOTS, dtype=wp.mat44, device=d)
        self.tw_rho = wp.zeros(POSE_SLOTS, dtype=wp.vec3, device=d)
        self.tw_theta = wp.zeros(POSE_SLOTS, dtype=wp.vec3, device=d)
        self._pose_host = np.tile(np.eye(4, dtype=np.float32), (POSE_SLOTS, 1, 1))
        self._tw_host = np.zeros((POSE_SLOTS, 6), np.float32)
        self._pose_dirty = set()
        self._pose_lock = threading.Lock()  # set_pose may come from the odometry worker thread
        self.owner_frame = wp.full(POSE_SLOTS, -1, dtype=wp.int32, device=d)
        self.owner_epoch = wp.full(POSE_SLOTS, -1, dtype=wp.int32, device=d)
        self._own_host = np.full((POSE_SLOTS, 2), -1, np.int32)  # (unmasked frame, epoch) per slot
        self.epoch = 0  # render-side pose epoch: bumped on clock restarts and map clears
        self._pin_ev = wp.Event(d) if d.is_cuda else None  # last async copy out of the pinned staging
        self._pin_ev_used = False
        self.set_pose(0, np.eye(4, dtype=np.float32))
        self.flush_poses()

        self.map_keys = wp.full(self.map_cap, -1, dtype=wp.int64, device=d)
        self.map_acc = wp.zeros(self.map_cap, dtype=wp.vec4, device=d)
        self.map_cnt = wp.zeros(self.map_cap, dtype=wp.int32, device=d)
        self.map_last = wp.zeros(self.map_cap, dtype=wp.float32, device=d)
        self.map_first = wp.full(self.map_cap, 1.0e30, dtype=wp.float32, device=d)
        self.map_miss = wp.zeros(self.map_cap, dtype=wp.int32, device=d)
        self.map_stats = wp.zeros(2, dtype=wp.int32, device=d)  # occupied, dropped

        n = self.work_cap
        self.flag = wp.zeros(n, dtype=wp.int32, device=d)
        self.offs = wp.zeros(n, dtype=wp.int32, device=d)
        self.w_xyz = wp.zeros(n, dtype=wp.vec3, device=d)
        self.w_attr = wp.zeros(n, dtype=wp.uint32, device=d)
        self.w_t = wp.zeros(n, dtype=wp.float32, device=d)
        self.w_dyn = wp.zeros(n, dtype=wp.int32, device=d)
        self.c_xyz = wp.zeros(n, dtype=wp.vec3, device=d)
        self.c_attr = wp.zeros(n, dtype=wp.uint32, device=d)
        self.c_t = wp.zeros(n, dtype=wp.float32, device=d)
        self.c_dyn = wp.zeros(n, dtype=wp.int32, device=d)
        self.nrm = wp.zeros(n, dtype=wp.vec3, device=d)
        self.curv = wp.zeros(n, dtype=wp.float32, device=d)
        self.nbr = wp.zeros(n, dtype=wp.int32, device=d)
        self.gnd = wp.zeros(n, dtype=wp.int32, device=d)  # 1 = ground (perception.py)
        self.hag = wp.zeros(n, dtype=wp.float32, device=d)  # height above local ground
        self.chg = wp.zeros(n, dtype=wp.float32, device=d)  # distance to a prior map (Changes mode)
        self.cid = wp.full(n, -1, dtype=wp.int32, device=d)  # cluster index or -1
        self.cl_vel = wp.zeros(MAX_CLUSTERS, dtype=wp.vec3, device=d)  # per cluster: tracked velocity
        self.cl_tid = wp.full(MAX_CLUSTERS, -1, dtype=wp.int32, device=d)  # per cluster: track id
        self.scalar = wp.zeros(n, dtype=wp.float32, device=d)
        # strided samples of the visible set, for percentiles on the host
        sample_n = 1 << 16
        self.sample = wp.zeros(sample_n, dtype=wp.float32, device=d)
        self._sample_xyz = wp.zeros(sample_n, dtype=wp.vec3, device=d)
        self._h_sample = HostMirror(d, wp.float32, sample_n)
        self._h_sample_xyz = HostMirror(d, wp.vec3, sample_n)
        self.total = wp.zeros(1, dtype=wp.int32, device=d)
        self.grid = wp.HashGrid(128, 128, 128, device=d)

        # per-frame voxel summary for neighborhood analysis
        self.nb_cap = int(nb_capacity)
        nb = self.nb_cap
        self.nb_keys = wp.full(nb, -1, dtype=wp.int64, device=d)
        self.nb_acc = wp.zeros(nb, dtype=wp.vec4, device=d)
        self.nb_flag = wp.zeros(nb, dtype=wp.int32, device=d)
        self.nb_offs = wp.zeros(nb, dtype=wp.int32, device=d)
        self.nb_vidx = wp.zeros(nb, dtype=wp.int32, device=d)
        self.v_xyz = wp.zeros(nb, dtype=wp.vec3, device=d)
        self.v_w = wp.zeros(nb, dtype=wp.float32, device=d)
        self.v_cnt = wp.zeros(nb, dtype=wp.float32, device=d)
        self.v_nrm = wp.zeros(nb, dtype=wp.vec3, device=d)
        self.v_curv = wp.zeros(nb, dtype=wp.float32, device=d)
        self.slot_of_pt = wp.zeros(n, dtype=wp.int32, device=d)
        self.nb_voxels = 0

        self._stage_cap = 0
        self._stage_xyz = self._stage_attr = self._stage_t = self._stage_frame = None
        self._pin_xyz = self._pin_attr = self._pin_t = None

        self.count = 0
        self.normals_valid = False
        self.map_occupied = 0
        self.map_dropped = 0
        # ms per step: build and neighbors are host times (each contains a sync); ingest, carve and shade are
        # the GPU time of their last launch, read after build's sync
        self.ms = {"ingest": 0.0, "build": 0.0, "neighbors": 0.0, "shade": 0.0, "carve": 0.0}
        self._timers = {name: _LaunchTimer(d, self.stream) for name in ("ingest", "carve", "shade")}

    # ---- memory --------------------------------------------------------------------------

    def gpu_bytes(self) -> int:
        """Device memory held by this pipeline's arrays (the neighbour hash grid is not counted)."""
        return sum(a.capacity for a in vars(self).values() if isinstance(a, wp.array) and a.device.is_cuda)

    def _ensure_stage(self, n: int):
        if n <= self._stage_cap:
            return
        cap = max(n, 1 << 16, self._stage_cap * 2)
        d = self.device
        self._stage_xyz = wp.empty(cap, dtype=wp.vec3, device=d)
        self._stage_attr = wp.empty(cap, dtype=wp.uint32, device=d)
        self._stage_t = wp.empty(cap, dtype=wp.float32, device=d)
        self._stage_frame = wp.empty(cap, dtype=wp.int32, device=d)
        pin = d.is_cuda
        self._pin_xyz = wp.empty(cap, dtype=wp.vec3, device="cpu", pinned=pin)
        self._pin_attr = wp.empty(cap, dtype=wp.uint32, device="cpu", pinned=pin)
        self._pin_t = wp.empty(cap, dtype=wp.float32, device="cpu", pinned=pin)
        self._stage_cap = cap

    def clear_live(self):
        self.head = 0
        self.filled = 0

    def clear_map(self):
        self.map_keys.fill_(-1)
        self.map_acc.zero_()
        self.map_cnt.zero_()
        self.map_last.zero_()
        self.map_first.fill_(1.0e30)
        self.map_miss.zero_()
        self.map_stats.zero_()
        self.map_occupied = self.map_dropped = 0

    # ---- poses ---------------------------------------------------------------------------

    def set_pose(self, frame: int, mat: np.ndarray, twist=None, epoch: int = 0):
        """Sensor-to-world matrix (row-major 4x4) and body-frame motion (rho, theta) for one frame id,
        written for `epoch`. Takes effect at the next flush_poses(), which every reader calls first.
        """
        j = frame & (POSE_SLOTS - 1)
        m = np.asarray(mat, dtype=np.float32)
        tw = np.zeros(6, np.float32) if twist is None else np.asarray(twist, dtype=np.float32)
        with self._pose_lock:
            self._pose_host[j] = m
            self._tw_host[j] = tw
            self._own_host[j] = (frame, epoch)
            self._pose_dirty.add(j)

    def set_pose_range(self, k0: int, k1: int, mat: np.ndarray, twist=None, epoch: int = 0):
        """The same pose for frames k0 .. k1-1 (vectorised: a gap can span thousands of ids)."""
        if k1 <= k0:
            return
        k0 = max(k0, k1 - POSE_SLOTS)
        ks = np.arange(k0, k1, dtype=np.int64)
        js = (ks & (POSE_SLOTS - 1)).astype(np.int64)
        m = np.asarray(mat, dtype=np.float32)
        tw = np.zeros(6, np.float32) if twist is None else np.asarray(twist, dtype=np.float32)
        with self._pose_lock:
            self._pose_host[js] = m
            self._tw_host[js] = tw
            self._own_host[js, 0] = ks.astype(np.int32)
            self._own_host[js, 1] = epoch
            self._pose_dirty.update(js.tolist())

    def flush_poses(self):
        """Upload every pose edited since the last flush in one scatter."""
        with self._pose_lock:
            if not self._pose_dirty:
                return
            idx = np.fromiter(self._pose_dirty, dtype=np.int32, count=len(self._pose_dirty))
            self._pose_dirty.clear()
            mats = self._pose_host[idx].copy()
            tws = self._tw_host[idx].copy()
            own = self._own_host[idx].copy()
        d = self.device
        # wp.array() from pageable host memory has consumed the data when it returns: no staging races
        w_idx = wp.array(idx, dtype=wp.int32, device=d)
        w_mat = wp.array(mats, dtype=wp.mat44, device=d)
        w_rho = wp.array(np.ascontiguousarray(tws[:, :3]), dtype=wp.vec3, device=d)
        w_th = wp.array(np.ascontiguousarray(tws[:, 3:]), dtype=wp.vec3, device=d)
        w_fr = wp.array(np.ascontiguousarray(own[:, 0]), dtype=wp.int32, device=d)
        w_ep = wp.array(np.ascontiguousarray(own[:, 1]), dtype=wp.int32, device=d)
        wp.launch(k_pose_scatter, dim=len(idx),
                  inputs=[w_idx, w_mat, w_rho, w_th, self.poses, self.tw_rho, self.tw_theta, w_fr, w_ep,
                          self.owner_frame, self.owner_epoch], device=d)

    def rebase_clock(self, shift: float):
        """The live clock restarted (replay loop, LiDAR reboot): move the map's first/last-seen times
        onto the new clock, new = old - shift, so persistence, Motion labels and ghosts keep working."""
        wp.launch(k_rebase_clock, dim=self.map_cap, inputs=[self.map_keys, self.map_first, self.map_last, float(shift)],
                  device=self.device)

    # ---- per batch -----------------------------------------------------------------------

    def stage(self, xyz, attr, t, n: int, inv_frame_dt: float = 0.0):
        """Put a batch on the GPU (or accept Warp arrays) and compute its frame ids.

        Returns (xyz, attr, t, frame) Warp arrays valid until the next stage() call.
        """
        if isinstance(xyz, np.ndarray) and self._pin_ev_used:
            # the previous batch's async copy may still be reading the pinned buffers we are about to
            # rewrite (or free, if they grow): wait for exactly that copy
            wp.synchronize_event(self._pin_ev)
        self._ensure_stage(n)
        if isinstance(xyz, np.ndarray):
            self._pin_xyz.numpy()[:n] = xyz.reshape(-1, 3)[:n]
            self._pin_attr.numpy()[:n] = attr[:n]
            self._pin_t.numpy()[:n] = t[:n]
            wp.copy(self._stage_xyz, self._pin_xyz, count=n)
            wp.copy(self._stage_attr, self._pin_attr, count=n)
            wp.copy(self._stage_t, self._pin_t, count=n)
            if self._pin_ev is not None:
                wp.record_event(self._pin_ev)
                self._pin_ev_used = True
            sx, sa, st = self._stage_xyz, self._stage_attr, self._stage_t
        else:
            sx, sa, st = xyz, attr, t
        sf = self._stage_frame
        wp.launch(k_frame_of, dim=n, inputs=[st, float(inv_frame_dt), sf], device=self.device)
        return sx, sa, st, sf

    def ingest_ring(self, sx, sa, st, sf, n: int):
        """Append staged points to the live ring."""
        if n <= 0:
            return
        timer = self._timers["ingest"]
        timer.start()
        keep = min(n, self.cap)
        wp.launch(
            k_ingest,
            dim=keep,
            inputs=[sx, sa, st, sf, n - keep, self.head, self.cap,
                    self.ring_xyz, self.ring_attr, self.ring_t, self.ring_frame],
            device=self.device,
        )
        timer.stop()
        self.head = (self.head + keep) % self.cap
        self.filled = min(self.filled + keep, self.cap)

    def map_insert(self, sx, sa, st, sf, n: int, view: View, voxel: float):
        """Accumulate staged points into the integration map through their frames' poses."""
        if n <= 0:
            return
        self.flush_poses()
        wp.launch(
            k_map_insert,
            dim=n,
            inputs=[
                sx, sa, st, sf, self.poses, view, 1.0 / max(voxel, 1e-4), self.map_cap - 1,
                self.map_keys, self.map_acc, self.map_cnt, self.map_last, self.map_first, self.map_stats,
            ],
            device=self.device,
        )

    def carve(self, sx, sa, sf, n: int, voxel: float, every: int = 2, max_range: float = 40.0,
              max_steps: int = 4096):
        """Free-space carving: count map voxels each ray passes through (see k_carve)."""
        if n <= 0:
            return
        self.flush_poses()
        timer = self._timers["carve"]
        timer.start()
        every = max(1, int(every))
        wp.launch(
            k_carve,
            dim=(n + every - 1) // every,
            inputs=[sx, sa, sf, self.poses, n, every, 1.0 / max(voxel, 1e-4), self.map_cap - 1,
                    float(max_range), int(max_steps), self.map_keys, self.map_acc, self.map_cnt, self.map_miss],
            device=self.device,
        )
        timer.stop()

    def ingest(self, xyz, attr, t, n: int, view: View, map_on: bool, voxel: float):
        """Stage, ring and map in one call, without odometry. The viewer runs the steps itself so it can
        hand the staged batch to the odometry worker; this is for scripts and tests."""
        if n <= 0:
            return
        sx, sa, st, sf = self.stage(xyz, attr, t, n, 0.0)
        self.ingest_ring(sx, sa, st, sf, n)
        if map_on:
            self.map_insert(sx, sa, st, sf, n, view, voxel)

    # ---- per frame -----------------------------------------------------------------------

    def build(self, view: View, map_on: bool, min_count: int, neighbors: Literal["", "count", "normals"],
              radius: float, sensor) -> int:
        """Filter + transform + compact the visible set into c_*; optionally neighbor stats.

        neighbors: "" (none), "count" (denoise only) or "normals" (normals + curvature + count).
        """
        t0 = time.perf_counter()
        d = self.device
        self.flush_poses()
        if map_on:
            n = self.map_cap
            wp.launch(
                k_classify_map,
                dim=n,
                inputs=[self.map_keys, self.map_acc, self.map_cnt, self.map_last, self.map_first, self.map_miss,
                        view, int(min_count), self.flag, self.w_xyz, self.w_attr, self.w_t, self.w_dyn],
                device=d,
            )
        else:
            n = self.cap
            wp.launch(
                k_classify_live,
                dim=n,
                inputs=[self.ring_xyz, self.ring_attr, self.ring_t, self.ring_frame, self.poses,
                        self.tw_rho, self.tw_theta, self.owner_frame, self.owner_epoch, self.filled, view,
                        self.map_keys, self.map_cnt, self.map_first, self.map_last, self.map_miss,
                        self.flag, self.w_xyz, self.w_attr, self.w_t, self.w_dyn],
                device=d,
            )
        flag, offs = self.flag[:n], self.offs[:n]
        wp.utils.array_scan(flag, offs, inclusive=False)
        wp.launch(k_total, dim=1, inputs=[flag, offs, n, self.total], device=d)
        wp.launch(
            k_scatter,
            dim=n,
            inputs=[flag, offs, self.w_xyz, self.w_attr, self.w_t, self.w_dyn,
                    self.c_xyz, self.c_attr, self.c_t, self.c_dyn],
            device=d,
        )
        if map_on:
            self._h_stats.read(self.map_stats, 2, self.stream, sync=False)
        self.count = int(self._h_total.read(self.total, 1, self.stream)[0])  # the frame's first host sync
        if map_on:
            self.map_occupied, self.map_dropped = (int(x) for x in self._h_stats.np[:2])  # covered by that sync
        for name, timer in self._timers.items():
            self.ms[name] = timer.read()
        self.ms["build"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        self.normals_valid = False
        if neighbors and self.count > 0:
            self.neighborhoods(float(radius), neighbors == "normals", sensor)
            self.normals_valid = neighbors == "normals"
        self.ms["neighbors"] = (time.perf_counter() - t1) * 1e3
        return self.count

    def load_points(self, xyz: np.ndarray):
        """Make an arbitrary world-frame point set the visible set, as build() would: for scripts that want
        neighborhoods() or shade() on their own points."""
        n = min(len(xyz), self.work_cap)
        src = wp.array(np.ascontiguousarray(xyz[:n], dtype=np.float32), dtype=wp.vec3, device=self.device)
        wp.copy(self.c_xyz, src, count=n)
        self.count = n
        self.normals_valid = False

    def neighborhoods(self, radius: float, want_normals: bool, sensor):
        """Fill nbr (and nrm/curv) for the visible points via a voxel summary a third of `radius` wide.

        build() calls this when asked for neighbors; on its own it serves points placed by load_points().
        """
        d, n, nb = self.device, self.count, self.nb_cap
        self.nb_keys.fill_(-1)
        self.nb_acc.zero_()
        wp.launch(k_nb_insert, dim=n,
                  inputs=[self.c_xyz, 3.0 / radius, nb - 1, self.nb_keys, self.nb_acc, self.slot_of_pt], device=d)
        wp.launch(k_nb_flag, dim=nb, inputs=[self.nb_keys, self.nb_flag], device=d)
        wp.utils.array_scan(self.nb_flag, self.nb_offs, inclusive=False)
        wp.launch(k_total, dim=1, inputs=[self.nb_flag, self.nb_offs, nb, self.total], device=d)
        wp.launch(k_nb_compact, dim=nb,
                  inputs=[self.nb_flag, self.nb_offs, self.nb_acc, self.nb_vidx, self.v_xyz, self.v_w], device=d)
        m = int(self._h_total.read(self.total, 1, self.stream)[0])
        self.nb_voxels = m
        if m == 0:
            return
        vox = self.v_xyz[:m]
        self.grid.build(vox, radius)
        wp.launch(k_nb_voxel, dim=m,
                  inputs=[self.grid.id, vox, self.v_w, radius, int(want_normals), wp.vec3(*sensor),
                          self.v_cnt, self.v_nrm, self.v_curv], device=d)
        wp.launch(k_nb_gather, dim=n,
                  inputs=[self.slot_of_pt, self.nb_vidx, self.v_cnt, self.v_nrm, self.v_curv, int(want_normals),
                          self.nbr, self.nrm, self.curv], device=d)

    def shade(self, shade: Shade, out_pos: wp.array, out_col: wp.array, out_nrm: wp.array):
        """Color the compacted points straight into the (mapped) GL vertex buffers."""
        if self.count == 0:
            return
        timer = self._timers["shade"]
        timer.start()
        wp.launch(
            k_shade,
            dim=self.count,
            inputs=[self.c_xyz, self.c_attr, self.c_t, self.nrm, self.curv, self.nbr,
                    self.gnd, self.hag, self.cid, self.cl_vel, self.cl_tid, self.c_dyn, self.chg, shade],
            outputs=[out_pos, out_col, out_nrm, self.scalar],
            device=self.device,
        )
        timer.stop()

    def _sample_stride(self):
        """(stride, count) that spreads the sample buffer over the visible points."""
        m = self.sample.shape[0]
        stride = max(1, self.count // m)
        return stride, min(m, (self.count + stride - 1) // stride)

    def _sample(self, kernel, src, lo_pct, hi_pct):
        if self.count == 0:
            return None
        stride, k = self._sample_stride()
        wp.launch(kernel, dim=k, inputs=[src, stride, self.count, self.sample], device=self.device)
        vals = self._h_sample.read(self.sample, k, self.stream)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        return float(lo), float(hi)

    def scalar_percentiles(self, lo_pct: float = 2.0, hi_pct: float = 98.0):
        """Robust range of the last shaded scalar, from a strided GPU sample."""
        return self._sample(k_gather_stride, self.scalar, lo_pct, hi_pct)

    def floor_z(self) -> float | None:
        """A low percentile of visible z: where the floor usually is."""
        r = self._sample(k_gather_z, self.c_xyz, 1.0, 99.0)
        return None if r is None else r[0]

    def bounds(self, lo_pct: float = 2.0, hi_pct: float = 98.0):
        """Robust (lo, hi) corners of the visible set, from a strided GPU sample."""
        if self.count == 0:
            return None
        stride, k = self._sample_stride()
        wp.launch(k_gather_xyz, dim=k, inputs=[self.c_xyz, stride, self.count, self._sample_xyz], device=self.device)
        pts = self._h_sample_xyz.read(self._sample_xyz, k, self.stream)
        return np.percentile(pts, lo_pct, axis=0), np.percentile(pts, hi_pct, axis=0)

    def export(self):
        """Visible points (world frame) and their attrs, on the host."""
        n = self.count
        return self.c_xyz.numpy()[:n].copy(), self.c_attr.numpy()[:n].copy(), self.c_t.numpy()[:n].copy()

    def export_labels(self):
        """Ground flag and cluster index of the visible points, on the host."""
        n = self.count
        return self.gnd.numpy()[:n].copy(), self.cid.numpy()[:n].copy()

