"""Point sources for the viewer and the tests: a live sensor, an LVXR replay or a simulated Mid-40.

Every source looks the same to its consumer: poll() returns the points that arrived since the last
call as (xyz, attr, t, n) and sets last_t0 / last_t1 to the batch's time span; `attr` packs the
reflectivity (bits 0-7), the noise tag (8-15), the return index (16-23) and the device index
(24-31); label, stats(), status() and close() describe and end the source. LiveSource and
ReplaySource wrap the Rust protocol stack and hand back numpy views of its buffers. SimSource
ray-casts the Mid-40's rosette pattern against an analytic room on the GPU and hands back Warp
arrays; its step() takes no wall clock, so a test run is reproducible.

The simulated room is a box holding a crate, a shelf, two pillars, a table and a retro-reflective
sign, plus a person-sized sphere walking left-right in front of the far wall: the moving object the
perception tests track. The "office" scene is the same room furnished as an open-plan office
(desks, chairs, shelving, doorways with rooms behind them, window recesses, columns and ceiling
fixtures). A scene is a list of primitives collected by SceneBuilder and uploaded as a SimScene.
The sensor can stand still, walk a figure-eight or turn in place, and knows its own pose at any
time (pose_at), which is what the odometry tests compare against.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import warp as wp

from . import _native
from .rosette import RosetteParams, rosette_dir_p

# Livox device type codes as the protocol reports them (the viewer keys its Mid-40 UI on MID40).
MID40 = 1
DEV_UNKNOWN = 255


def clean_returns(xyz: np.ndarray, attr: np.ndarray, min_range: float = 0.3) -> np.ndarray:
    """Mask of the returns worth keeping: noise tag 0 (a confident return) and farther than min_range.

    Very close returns are the sensor's own housing or dust on the window.
    """
    return (((attr >> 8) & 0xFF) == 0) & (np.linalg.norm(xyz, axis=1) > min_range)


class _NativeSource:
    """poll() for the sources the Rust stack feeds: numpy views over its buffers plus the batch's time span."""

    def __init__(self):
        self.last_t0 = 0.0  # time span (sensor seconds) of the batch the last poll() returned
        self.last_t1 = 0.0

    def _drain(self) -> tuple[bytes, bytes, bytes, int]:
        raise NotImplementedError

    def poll(self):
        xyz, attr, t, n = self._drain()
        if not n:
            return None
        t = np.frombuffer(t, np.float32)
        self.last_t0, self.last_t1 = float(t.min()), float(t.max())
        return np.frombuffer(xyz, np.float32).reshape(-1, 3), np.frombuffer(attr, np.uint32), t, n


class LiveSource(_NativeSource):
    """A sensor on the network, driven by the Rust protocol stack (_native.Device)."""

    kind = "live"

    def __init__(self, lidar_ip: str, host_ip: str, dev_type: int = DEV_UNKNOWN, start: bool = True):
        super().__init__()
        self.dev = _native.Device(lidar_ip, host_ip, dev_type)
        self.lidar_ip, self.host_ip, self.dev_type = lidar_ip, host_ip, dev_type
        self.sampling = False
        self.recording = None  # path of the LVXR file being written, or None
        if start:
            self.start()

    @property
    def label(self) -> str:
        return f"{_native.device_type_name(self.dev_type)} {self.lidar_ip} via {self.host_ip}"

    def start(self):
        self.dev.start_sampling()
        self.sampling = True

    def stop(self):
        self.dev.stop_sampling()
        self.sampling = False

    def _drain(self):
        return self.dev.drain()

    def start_recording(self, path: str) -> None:
        """Write every raw packet to `path`, an LVXR file that ReplaySource plays back."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.dev.start_recording(path)
        self.recording = path

    def stop_recording(self) -> int:
        """Close the recording; returns the number of packets written (0 when none was running)."""
        n = self.dev.stop_recording()
        self.recording = None
        return int(n or 0)

    def stats(self) -> dict:
        return self.dev.stats()

    def status(self) -> dict:
        return self.dev.status()

    def close(self):
        try:
            if self.recording:
                self.stop_recording()
            if self.sampling:
                self.stop()
        except RuntimeError:
            pass
        self.dev.disconnect()


class ReplaySource(_NativeSource):
    """An LVXR recording played back at `speed` times real time by the Rust stack (_native.Replay)."""

    kind = "replay"

    def __init__(self, path: str, speed: float = 1.0, looped: bool = True):
        super().__init__()
        self.path = path
        self.replay = _native.Replay(path, speed, looped)
        self.looped = looped
        self.dev_type = self.replay.dev_type

    @property
    def label(self) -> str:
        return (f"replay {os.path.basename(self.path)} ({_native.device_type_name(self.dev_type)}, "
                f"{self.replay.stream_profile})")

    def _drain(self):
        return self.replay.drain()

    def read_all(self):
        """Replay the whole recording (looped=False) and return every point as (xyz, attr, t), time-ordered.

        The replay thread's last batch is in the sink before finished() turns true, so finished() is read
        before each drain and the loop ends after the drain that follows it."""
        if self.looped:
            raise ValueError("read_all needs a ReplaySource with looped=False")
        xs, attrs, ts = [], [], []
        while True:
            done = self.replay.finished()
            b = self.poll()
            if b is not None:
                x, a, t, n = b
                xs.append(x[:n].copy())
                attrs.append(a[:n].copy())
                ts.append(t[:n].copy())
            if done:
                break
            time.sleep(0.002)  # poll often: the replay drops what its sink cannot hold between drains
        if not xs:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), np.zeros(0, np.float32)
        x, a, t = np.concatenate(xs), np.concatenate(attrs), np.concatenate(ts)
        order = np.argsort(t, kind="stable")
        return x[order], a[order], t[order]

    def stats(self) -> dict:
        s = self.replay.stats()
        s["loops"] = self.replay.loops()
        return s

    def status(self):
        return None

    def close(self):
        self.replay = None


# --------------------------------------------------------------------------------------------
# Simulated Mid-40: the rosette scan pattern ray-cast against an analytic room, entirely on the GPU
# --------------------------------------------------------------------------------------------

SIM_MOTIONS = ["static", "walk", "turn"]  # how the sensor moves; a kernel gets the index
SIM_SCENES = ["box", "office"]

# The room (world metres): its inside walls, floor and ceiling, and the crate every scene holds,
# whose top the perception tests measure. Tuples for the scene builder and the tests, Warp
# constants for the kernels.
SIM_ROOM_LO = (-2.0, -4.0, -1.3)
SIM_ROOM_HI = (14.0, 4.5, 2.6)
SIM_FLOOR_Z = SIM_ROOM_LO[2]
SIM_CRATE_LO = (4.0, -1.8, SIM_FLOOR_Z)
SIM_CRATE_HI = (5.2, -0.6, -0.2)
ROOM_LO = wp.constant(wp.vec3(*SIM_ROOM_LO))
ROOM_HI = wp.constant(wp.vec3(*SIM_ROOM_HI))

# The moving sensor holds still for HOLD seconds first, as a person starting odometry would, then
# walks a figure-eight (one lap per 2 pi / OMEGA seconds) or turns in place.
HOLD = wp.constant(1.5)
OMEGA = wp.constant(2.0 * math.pi / 40.0)

# The person-sized sphere walking left-right in front of the far wall: x, y amplitude and angular
# rate, centre height, radius and reflectivity. Nothing stands between it and the sensor.
PERSON_X = wp.constant(12.0)
PERSON_AMP = wp.constant(2.5)
PERSON_RATE = wp.constant(0.6)
PERSON_Z = wp.constant(-0.85)
PERSON_R = wp.constant(0.45)
PERSON_REFL = wp.constant(90.0)

_OFFICE_LAYOUT_SEED = 20260930  # any fixed value; the layout must be reproducible


@wp.func
def slab(o: wp.vec3, inv_d: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    """Entry distance into an AABB, or 1e9 on a miss."""
    t1 = wp.cw_mul(lo - o, inv_d)
    t2 = wp.cw_mul(hi - o, inv_d)
    tmin = wp.max(wp.max(wp.min(t1[0], t2[0]), wp.min(t1[1], t2[1])), wp.min(t1[2], t2[2]))
    tmax = wp.min(wp.min(wp.max(t1[0], t2[0]), wp.max(t1[1], t2[1])), wp.max(t1[2], t2[2]))
    if tmax >= wp.max(tmin, 0.0) and tmin > 0.0:
        return tmin
    return 1.0e9


@wp.func
def room_exit(o: wp.vec3, d: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec2:
    """Distance to the inside wall of a box the ray starts in, and which wall (0..5)."""
    best = float(1.0e9)
    wall = float(0.0)
    for k in range(3):
        if d[k] > 1.0e-6:
            t = (hi[k] - o[k]) / d[k]
            if t < best:
                best = t
                wall = float(2 * k + 1)
        if d[k] < -1.0e-6:
            t = (lo[k] - o[k]) / d[k]
            if t < best:
                best = t
                wall = float(2 * k)
    return wp.vec2(best, wall)


@wp.func
def sphere_hit(o: wp.vec3, d: wp.vec3, c: wp.vec3, r: float) -> float:
    """Entry distance into a sphere, or 1e9 on a miss."""
    oc = o - c
    b = wp.dot(oc, d)
    disc = b * b - (wp.dot(oc, oc) - r * r)
    if disc > 0.0:
        ts = -b - wp.sqrt(disc)
        if ts > 0.0:
            return ts
    return 1.0e9


@wp.func
def rand01(seed: wp.uint32) -> float:
    h = seed * wp.uint32(747796405) + wp.uint32(2891336453)
    h = ((h >> ((h >> wp.uint32(28)) + wp.uint32(4))) ^ h) * wp.uint32(277803737)
    h = (h >> wp.uint32(22)) ^ h
    return float(h) / 4294967295.0


@wp.func
def sim_pose(t: float, motion: int) -> wp.transform:
    """Sensor pose at time t for a SIM_MOTIONS index. Must match SimSource.pose_at."""
    if motion == 0:
        return wp.transform_identity()
    u = wp.max(t - HOLD, 0.0) * OMEGA
    p = wp.vec3(3.0, 0.5, 0.0)
    yaw = float(0.0)
    pitch = float(0.0)
    roll = float(0.0)
    if motion == 1:
        p = wp.vec3(5.0 + 3.0 * wp.sin(u), 0.5 + 1.5 * wp.sin(2.0 * u), 0.15 * wp.sin(0.7 * u))
        yaw = wp.atan2(3.0 * wp.cos(2.0 * u), 3.0 * wp.cos(u))
        pitch = 0.05 * wp.sin(1.3 * u)
        roll = 0.04 * wp.sin(0.9 * u)
    else:
        yaw = 0.7 * wp.sin(0.5 * u)
        pitch = 0.03 * wp.sin(1.1 * u)
    q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
               wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch),
                      wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), roll)))
    return wp.transform(p, q)


@wp.func
def person_centre(t: float) -> wp.vec3:
    """Centre of the walking sphere at time t. Must match SimSource.person_at."""
    return wp.vec3(PERSON_X, PERSON_AMP * wp.sin(PERSON_RATE * t), PERSON_Z)


@wp.func
def rosette_dir(t: float) -> wp.vec3:
    """Firing direction (sensor frame) at time t: two counter-rotating prisms sweep a 38.4 deg circle."""
    half = wp.radians(19.2)
    w1 = 2.0 * wp.pi * 153.7
    w2 = 2.0 * wp.pi * 97.3
    u = half * 0.5 * (wp.cos(w1 * t) + wp.cos(w2 * t))
    v = half * 0.5 * (wp.sin(w1 * t) - wp.sin(w2 * t))
    return wp.normalize(wp.vec3(1.0, wp.tan(u), wp.tan(v)))


@wp.func
def pack_return(r: float, refl: float, seed: wp.uint32):
    """The noise model: range jitter, reflectivity jitter, and 2 % of returns pulled short and flagged
    with noise tag 1 to exercise the filters. Returns the measured range and the packed attr word."""
    rng = r + (rand01(seed) - 0.5) * 0.02
    tag = int(0)
    if rand01(seed ^ wp.uint32(0x9E3779B9)) < 0.02:
        rng = rng * (0.3 + 0.6 * rand01(seed ^ wp.uint32(0x85EBCA6B)))
        tag = 1
    attr = wp.uint32(int(wp.min(refl + rand01(seed + wp.uint32(7)) * 6.0, 255.0))) | (wp.uint32(tag) << wp.uint32(8))
    return rng, attr


@wp.struct
class SimScene:
    """A scene's static primitives as device arrays (SceneBuilder.upload), ray-cast by scene_cast."""

    box_c: wp.array(dtype=wp.vec3)  # boxes rotated about z: centre
    box_h: wp.array(dtype=wp.vec3)  # half extents in the box frame
    box_r: wp.array(dtype=wp.vec3)  # cos(yaw), sin(yaw), reflectivity
    n_box: int
    cyl_p: wp.array(dtype=wp.vec4)  # axis-aligned cylinders: centre on the two other axes, radius, reflectivity
    cyl_e: wp.array(dtype=wp.vec3)  # extent along the axis (lo, hi), axis (0, 1, 2)
    n_cyl: int
    sph: wp.array(dtype=wp.vec4)  # static spheres: centre, radius
    sph_refl: wp.array(dtype=float)  # their reflectivity
    n_sph: int
    rec_lo: wp.array(dtype=wp.vec3)  # recesses: boxes beyond a wall (or beyond another recess) that rays
    rec_hi: wp.array(dtype=wp.vec3)  # passing through the wall aperture continue into
    rec_refl: wp.array(dtype=float)  # reflectivity of a recess's inside
    n_rec: int


@wp.func
def obb_hit(o: wp.vec3, d: wp.vec3, c: wp.vec3, h: wp.vec3, cs: float, sn: float) -> float:
    """Entry distance into a box rotated by yaw (cs, sn) about z around c, or 1e9."""
    q = o - c
    ol = wp.vec3(cs * q[0] + sn * q[1], -sn * q[0] + cs * q[1], q[2])
    dl = wp.vec3(cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1], d[2])
    inv = wp.vec3(1.0 / (dl[0] + 1.0e-9), 1.0 / (dl[1] + 1.0e-9), 1.0 / (dl[2] + 1.0e-9))
    return slab(ol, inv, -h, h)


@wp.func
def cyl_hit(o: wp.vec3, d: wp.vec3, p: wp.vec4, e: wp.vec3) -> float:
    """Entry distance into an axis-aligned capped cylinder, or 1e9."""
    axis = int(e[2])
    a0 = (axis + 1) % 3
    a1 = (axis + 2) % 3
    oa = o[a0] - p[0]
    ob = o[a1] - p[1]
    da = d[a0]
    db = d[a1]
    r2 = p[2] * p[2]
    best = float(1.0e9)
    A = da * da + db * db
    if A > 1.0e-12:
        B = oa * da + ob * db
        disc = B * B - A * (oa * oa + ob * ob - r2)
        if disc > 0.0:
            t = (-B - wp.sqrt(disc)) / A
            if t > 0.0:
                z = o[axis] + t * d[axis]
                if z >= e[0] and z <= e[1]:
                    best = t
    dz = d[axis]
    if wp.abs(dz) > 1.0e-9:
        for k in range(2):
            zc = e[0]
            if k == 1:
                zc = e[1]
            t = (zc - o[axis]) / dz
            if t > 0.0 and t < best:
                pa = oa + t * da
                pb = ob + t * db
                if pa * pa + pb * pb <= r2:
                    best = t
    return best


@wp.func
def box_exit(o: wp.vec3, d: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    """Distance at which a ray starting inside an AABB leaves it."""
    inv = wp.vec3(1.0 / (d[0] + 1.0e-9), 1.0 / (d[1] + 1.0e-9), 1.0 / (d[2] + 1.0e-9))
    t1 = wp.cw_mul(lo - o, inv)
    t2 = wp.cw_mul(hi - o, inv)
    return wp.min(wp.min(wp.max(t1[0], t2[0]), wp.max(t1[1], t2[1])), wp.max(t1[2], t2[2]))


@wp.func
def inside(p: wp.vec3, lo: wp.vec3, hi: wp.vec3, eps: float) -> int:
    if p[0] < lo[0] - eps or p[1] < lo[1] - eps or p[2] < lo[2] - eps:
        return 0
    if p[0] > hi[0] + eps or p[1] > hi[1] + eps or p[2] > hi[2] + eps:
        return 0
    return 1


@wp.func
def scene_cast(o: wp.vec3, d: wp.vec3, t: float, sc: SimScene) -> wp.vec2:
    """Distance and reflectivity of the first return along a ray from o in direction d at time t."""
    hit = room_exit(o, d, ROOM_LO, ROOM_HI)
    best = float(hit[0])
    refl = float(18.0 + 6.0 * hit[1])
    # doorways, windows and alcoves: a ray leaving the room through an opening continues into the
    # recess behind it (two passes: a door opening, then the corridor behind it)
    for rep in range(2):
        p = o + d * best
        for j in range(sc.n_rec):
            lo = sc.rec_lo[j]
            hi = sc.rec_hi[j]
            if inside(p, lo, hi, 1.0e-4) == 1:
                te = box_exit(o, d, lo, hi)
                if te > best + 1.0e-5:
                    best = te
                    refl = sc.rec_refl[j]
                    p = o + d * best
    for j in range(sc.n_box):
        r = sc.box_r[j]
        tb = obb_hit(o, d, sc.box_c[j], sc.box_h[j], r[0], r[1])
        if tb < best:
            best = tb
            refl = r[2]
    for j in range(sc.n_cyl):
        tb = cyl_hit(o, d, sc.cyl_p[j], sc.cyl_e[j])
        if tb < best:
            best = tb
            refl = sc.cyl_p[j][3]
    for j in range(sc.n_sph):
        s = sc.sph[j]
        tb = sphere_hit(o, d, wp.vec3(s[0], s[1], s[2]), s[3])
        if tb < best:
            best = tb
            refl = sc.sph_refl[j]
    tb = sphere_hit(o, d, person_centre(t), PERSON_R)
    if tb < best:
        best = tb
        refl = PERSON_REFL
    return wp.vec2(best, refl)


@wp.kernel
def k_sim_scan(
    t0: float,
    dt: float,
    first: int,
    motion: int,
    sc: SimScene,
    out_xyz: wp.array(dtype=wp.vec3),
    out_attr: wp.array(dtype=wp.uint32),
    out_t: wp.array(dtype=wp.float32),
):
    """Point i fires at t0 + i * dt along the rosette, from the moving sensor into the scene. `first` is
    point 0's index in the whole run and seeds the noise, so a run is reproducible."""
    i = wp.tid()
    t = t0 + float(i) * dt
    d_s = rosette_dir(t)
    pose = sim_pose(t, motion)
    d = wp.quat_rotate(wp.transform_get_rotation(pose), d_s)
    hit = scene_cast(wp.transform_get_translation(pose), d, t, sc)
    r, attr = pack_return(hit[0], hit[1], wp.uint32(first + i))
    out_xyz[i] = d_s * r
    out_attr[i] = attr
    out_t[i] = t


@wp.kernel
def k_sim_scan_rosette(
    t0: float,
    dt: float,
    first: int,
    motion: int,
    sc: SimScene,
    ros: RosetteParams,
    out_xyz: wp.array(dtype=wp.vec3),
    out_attr: wp.array(dtype=wp.uint32),
    out_t: wp.array(dtype=wp.float32),
):
    """k_sim_scan with a fitted scan pattern (rosette.RosetteModel) in place of the built-in one; the pattern's
    phases at t0 come in `ros`, computed on the host in float64."""
    i = wp.tid()
    t = t0 + float(i) * dt
    d_s = rosette_dir_p(float(i) * dt, ros)
    pose = sim_pose(t, motion)
    d = wp.quat_rotate(wp.transform_get_rotation(pose), d_s)
    hit = scene_cast(wp.transform_get_translation(pose), d, t, sc)
    r, attr = pack_return(hit[0], hit[1], wp.uint32(first + i))
    out_xyz[i] = d_s * r
    out_attr[i] = attr
    out_t[i] = t


@wp.kernel
def k_tls_scan(
    stations: wp.array(dtype=wp.vec3),
    n_az: int,
    n_el: int,
    el_lo: float,
    step: float,
    t: float,
    noise: float,
    sc: SimScene,
    out: wp.array(dtype=wp.vec3),
):
    """One ray per (station, azimuth, elevation) of a terrestrial scanner's spherical grid, into the scene at
    time t (where the walking sphere stands then); range noise uniform within +-noise."""
    s, a, e = wp.tid()
    az = float(a) * step
    el = el_lo + float(e) * step
    d = wp.vec3(wp.cos(el) * wp.cos(az), wp.cos(el) * wp.sin(az), wp.sin(el))
    o = stations[s]
    hit = scene_cast(o, d, t, sc)
    seed = wp.uint32((s * n_az + a) * n_el + e)
    r = hit[0] + (rand01(seed) - 0.5) * 2.0 * noise
    out[(s * n_az + a) * n_el + e] = o + d * r


# --------------------------------------------------------------------------------------------
# Scenes: the room's contents as lists of primitives
# --------------------------------------------------------------------------------------------


class SceneBuilder:
    """Collects a scene's primitives; heights h are above the floor (world z = SIM_FLOOR_Z + h)."""

    def __init__(self):
        self.boxes = []  # (cx, cy, cz, hx, hy, hz, yaw, refl)
        self.cyls = []  # (c0, c1, r, refl, lo, hi, axis)
        self.sphs = []  # (cx, cy, cz, r, refl)
        self.recs = []  # (lo xyz, hi xyz, refl), world z

    def aabb(self, lo, hi, refl):
        """An axis-aligned box between the world corners lo and hi."""
        self.boxes.append(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2,
                           (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2, 0.0, refl))

    def box(self, x0, x1, y0, y1, h0, h1, refl):
        """An axis-aligned box over x0..x1, y0..y1 between heights h0 and h1."""
        self.aabb((x0, y0, SIM_FLOOR_Z + h0), (x1, y1, SIM_FLOOR_Z + h1), refl)

    def obox(self, cx, cy, lx, ly, h0, h1, yaw, refl):
        """A box of size lx (along its facing yaw) by ly, centred at (cx, cy)."""
        self.boxes.append((cx, cy, SIM_FLOOR_Z + (h0 + h1) / 2, lx / 2, ly / 2, (h1 - h0) / 2, yaw, refl))

    @staticmethod
    def local(cx, cy, yaw, fx, fy):
        """World position of the point (fx, fy) in the frame at (cx, cy) facing yaw."""
        c, s = math.cos(yaw), math.sin(yaw)
        return cx + c * fx - s * fy, cy + s * fx + c * fy

    def vcyl(self, x, y, r, h0, h1, refl):
        """A vertical cylinder between heights h0 and h1."""
        self.cyls.append((x, y, r, refl, SIM_FLOOR_Z + h0, SIM_FLOOR_Z + h1, 2))

    def hcyl(self, axis, a, b, r, lo, hi, refl):
        """Horizontal cylinder along x (axis 0: a=y, b=z) or y (axis 1: a=z, b=x); world coordinates."""
        self.cyls.append((a, b, r, refl, lo, hi, axis))

    def sphere(self, x, y, h, r, refl):
        self.sphs.append((x, y, SIM_FLOOR_Z + h, r, refl))

    def recess(self, x0, x1, y0, y1, h0, h1, refl):
        """A space beyond a wall that rays through the wall's aperture continue into."""
        self.recs.append(((x0, y0, SIM_FLOOR_Z + h0), (x1, y1, SIM_FLOOR_Z + h1), refl))

    # ---- furniture ------------------------------------------------------------------------

    def desk(self, cx, cy, yaw, rng, width=1.6, depth=0.8, monitor=True, tower=True):
        """Office desk; yaw is the direction a seated person faces (toward the monitor)."""
        def L(fx, fy):
            return self.local(cx, cy, yaw, fx, fy)
        top = rng.uniform(35, 60)
        self.obox(cx, cy, depth, width, 0.72, 0.75, yaw, top)
        for sgn in (-1, 1):
            x, y = L(0.0, sgn * (width / 2 - 0.015))
            self.obox(x, y, depth - 0.04, 0.03, 0.0, 0.72, yaw, top - 5)
        x, y = L(depth / 2 - 0.03, 0.0)
        self.obox(x, y, 0.02, width - 0.06, 0.32, 0.72, yaw, top - 5)
        if monitor:
            off = rng.uniform(-0.2, 0.2)
            x, y = L(depth / 2 - 0.2, off)
            self.obox(x, y, 0.04, 0.56, 0.86, 1.2, yaw + rng.uniform(-0.25, 0.25), 9.0)
            self.vcyl(x, y, 0.035, 0.75, 0.86, 60.0)
            x, y = L(-depth / 2 + 0.2, off)
            self.obox(x, y, 0.15, 0.45, 0.75, 0.78, yaw, 12.0)  # keyboard
        if tower:
            x, y = L(0.0, width / 2 - 0.25)
            self.obox(x, y, 0.45, 0.2, 0.01, 0.43, yaw, 14.0)

    def chair(self, cx, cy, yaw, refl=12.0):
        """Office chair facing yaw (seat, backrest, gas post, star base as a disc)."""
        self.obox(cx, cy, 0.48, 0.48, 0.42, 0.5, yaw, refl)
        x, y = self.local(cx, cy, yaw, -0.24, 0.0)
        self.obox(x, y, 0.06, 0.46, 0.55, 1.02, yaw, refl)
        self.vcyl(cx, cy, 0.03, 0.1, 0.42, 70.0)
        self.vcyl(cx, cy, 0.3, 0.04, 0.1, 25.0)

    def bookshelf(self, cx, cy, yaw, rng, width=0.9, depth=0.35, height=2.0):
        """Open bookshelf backed against a wall; yaw is its facing (the wall is behind)."""
        def L(fx, fy):
            return self.local(cx, cy, yaw, fx, fy)
        wood = rng.uniform(40, 55)
        x, y = L(-depth / 2 + 0.01, 0.0)
        self.obox(x, y, 0.02, width, 0.0, height, yaw, wood)
        for sgn in (-1, 1):
            x, y = L(0.0, sgn * (width / 2 - 0.01))
            self.obox(x, y, depth, 0.02, 0.0, height, yaw, wood)
        levels = np.linspace(0.05, height - 0.02, 6)
        for k, hz in enumerate(levels):
            self.obox(cx, cy, depth, width - 0.04, hz, hz + 0.02, yaw, wood)
            if k == len(levels) - 1:
                break
            fy = -width / 2 + 0.03
            while True:
                w = rng.uniform(0.08, 0.3)
                if fy + w > width / 2 - 0.03:
                    break
                if rng.uniform() < 0.8:  # some gaps
                    dep = rng.uniform(0.18, depth - 0.04)
                    ht = min(rng.uniform(0.18, 0.34), levels[k + 1] - hz - 0.04)
                    x, y = L(-depth / 2 + 0.02 + dep / 2, fy + w / 2)
                    self.obox(x, y, dep, w, hz + 0.02, hz + 0.02 + ht, yaw, rng.uniform(15, 90))
                fy += w + rng.uniform(0.0, 0.05)

    def window(self, wall, lo, hi, h0, h1, depth, refl=70.0):
        """Window in a wall: recess to the pane/blinds, a sill ledge, and a mullion."""
        m = 0.5 * (lo + hi)
        if wall == "-y":
            y = SIM_ROOM_LO[1]
            self.recess(lo, hi, y - depth, y + 1e-3, h0, h1, refl)
            self.box(lo - 0.05, hi + 0.05, y, y + 0.06, h0 - 0.03, h0, 65.0)
            self.box(m - 0.03, m + 0.03, y - depth, y - depth + 0.08, h0, h1, 60.0)
        elif wall == "-x":
            x = SIM_ROOM_LO[0]
            self.recess(x - depth, x + 1e-3, lo, hi, h0, h1, refl)
            self.box(x, x + 0.06, lo - 0.05, hi + 0.05, h0 - 0.03, h0, 65.0)
            self.box(x - depth, x - depth + 0.08, m - 0.03, m + 0.03, h0, h1, 60.0)

    def upload(self, device) -> SimScene:
        """The primitives as device arrays."""
        b = np.array(self.boxes, np.float32).reshape(-1, 8)
        c = np.array(self.cyls, np.float32).reshape(-1, 7)
        s = np.array(self.sphs, np.float32).reshape(-1, 5)
        r_lo = np.array([r[0] for r in self.recs], np.float32).reshape(-1, 3)
        r_hi = np.array([r[1] for r in self.recs], np.float32).reshape(-1, 3)
        r_refl = np.array([r[2] for r in self.recs], np.float32)

        def arr(x, dtype):
            return wp.array(x, dtype=dtype, device=device)

        sc = SimScene()
        sc.box_c, sc.box_h, sc.n_box = arr(b[:, 0:3], wp.vec3), arr(b[:, 3:6], wp.vec3), len(b)
        sc.box_r = arr(np.stack([np.cos(b[:, 6]), np.sin(b[:, 6]), b[:, 7]], 1), wp.vec3)
        sc.cyl_p, sc.cyl_e, sc.n_cyl = arr(c[:, 0:4], wp.vec4), arr(c[:, 4:7], wp.vec3), len(c)
        sc.sph, sc.sph_refl, sc.n_sph = arr(s[:, 0:4], wp.vec4), arr(s[:, 4], float), len(s)
        sc.rec_lo, sc.rec_hi, sc.rec_refl = arr(r_lo, wp.vec3), arr(r_hi, wp.vec3), arr(r_refl, float)
        sc.n_rec = len(r_lo)
        return sc


def box_scene() -> SceneBuilder:
    """The bare room: a crate, a shelf, two pillars, a table and a retro-reflective sign."""
    S = SceneBuilder()
    H = SIM_ROOM_HI[2] - SIM_FLOOR_Z  # floor to ceiling
    S.aabb(SIM_CRATE_LO, SIM_CRATE_HI, 55.0)  # crate
    S.box(9.5, 10.1, 2.6, 4.4, 0.0, 2.9, 30.0)  # shelf
    S.box(2.3, 2.7, -2.7, -2.3, 0.0, H, 40.0)  # pillar
    S.box(10.8, 11.2, 3.3, 3.7, 0.0, H, 40.0)  # pillar
    S.box(8.0, 9.2, -3.3, -2.1, 0.0, 0.75, 48.0)  # table
    S.box(13.8, 13.99, -0.6, 0.6, 1.7, 2.3, 220.0)  # retro-reflective sign
    return S


def office_scene() -> SceneBuilder:
    """The box room furnished as an open-plan office / lab.

    Everything reaching below the ceiling stays at least 0.9 m (horizontally) from the walk's
    full figure-eight (x 2..8, y -1..2), the turn position (3, 0.5) and the static position
    (0, 0), and out of the walking sphere's lane (x 11.55..12.45, |y| < 2.95).
    """
    rng = np.random.default_rng(_OFFICE_LAYOUT_SEED)
    S = box_scene()
    H = SIM_ROOM_HI[2] - SIM_FLOOR_Z  # floor to ceiling

    # --- south wall (y = -4): windows over a row of desks, pilasters between them, radiators
    for xc in (0.8, 5.0, 9.6):
        S.window("-y", xc - 0.8, xc + 0.8, 0.9, 2.4, 0.25)
        S.box(xc - 0.6, xc + 0.6, -4.0, -3.9, 0.15, 0.75, 60.0)  # radiator
    for xc in (2.75, 7.25, 11.75):
        S.box(xc - 0.2, xc + 0.2, -4.0, -3.75, 0.0, H, 32.0)  # pilaster
    for x0, x1 in ((-1.7, -0.1), (0.1, 1.7), (3.2, 4.8), (5.0, 6.6), (9.8, 11.4), (12.2, 13.8)):
        xc = 0.5 * (x0 + x1)
        S.desk(xc, -3.45, -math.pi / 2, rng)
        S.chair(xc + rng.uniform(-0.25, 0.25), -2.72 + rng.uniform(-0.05, 0.1),
                -math.pi / 2 + rng.uniform(-0.6, 0.6))
    for xc in (0.0, 4.9):  # cubicle partitions between desk pairs
        S.box(xc - 0.03, xc + 0.03, -3.95, -2.45, 0.0, 1.3, 28.0)
    S.vcyl(2.0, -3.6, 0.15, 0.0, 0.4, 35.0)  # bins
    S.vcyl(6.85, -3.65, 0.15, 0.0, 0.4, 35.0)

    # --- north wall (y = 4.5): a doorway into a corridor, bookshelves, whiteboard, cabinets, alcove
    S.recess(0.6, 1.6, 4.5 - 1e-3, 4.65, 0.0, 2.1, 35.0)  # door opening (wall thickness)
    S.recess(-2.0, 8.0, 4.65, 6.45, 0.0, 2.7, 30.0)  # corridor behind it
    S.box(0.52, 0.6, 4.47, 4.5, 0.0, 2.18, 70.0)  # door frame trim
    S.box(1.6, 1.68, 4.47, 4.5, 0.0, 2.18, 70.0)
    S.box(0.52, 1.68, 4.47, 4.5, 2.1, 2.18, 70.0)
    S.obox(1.71, 4.06, 0.9, 0.04, 0.01, 2.08, -math.pi / 2 + 0.25, 45.0)  # door leaf, open into the room
    S.box(-1.2, 7.5, 6.0, 6.45, 0.0, 0.9, 40.0)  # corridor: a bench/cabinet row on its far side
    for xc in (2.85, 3.8, 4.75):
        S.bookshelf(xc, 4.5 - 0.175, -math.pi / 2, rng)
    S.box(5.8, 7.6, 4.48, 4.5, 0.9, 2.1, 90.0)  # whiteboard
    S.box(5.8, 7.6, 4.42, 4.5, 0.87, 0.9, 60.0)  # its marker tray
    S.box(7.8, 8.3, 3.9, 4.5, 0.0, 1.3, 55.0)  # filing cabinets
    S.box(8.35, 8.85, 3.9, 4.5, 0.0, 1.3, 55.0)
    S.box(8.9, 9.45, 3.75, 4.35, 0.0, 0.45, 42.0)  # cardboard boxes
    S.obox(9.15, 4.05, 0.45, 0.4, 0.45, 0.8, 0.3, 40.0)
    S.box(-1.95, -1.1, 3.95, 4.5, 0.0, 1.3, 50.0)  # cabinet in the corner
    S.recess(12.0, 13.6, 4.5 - 1e-3, 5.1, 0.0, H, 26.0)  # kitchenette alcove
    S.box(12.02, 13.58, 4.5, 5.1, 0.0, 0.9, 58.0)  # counter
    S.box(12.02, 13.58, 4.75, 5.1, 1.5, 2.2, 52.0)  # wall cupboards
    S.box(12.2, 12.5, 4.6, 4.9, 0.9, 1.25, 20.0)  # coffee machine

    # --- east wall (x = 14): a doorway into a side room, tall cabinets, a low sideboard
    S.recess(14.0 - 1e-3, 14.15, -3.2, -2.2, 0.0, 2.1, 35.0)
    S.recess(14.15, 17.5, -4.0, -0.5, 0.0, 2.7, 28.0)
    S.box(14.8, 15.6, -3.9, -3.3, 0.0, 0.75, 45.0)  # something in the side room
    S.box(13.97, 14.0, -3.28, -3.2, 0.0, 2.18, 70.0)  # frame trim
    S.box(13.97, 14.0, -2.2, -2.12, 0.0, 2.18, 70.0)
    S.box(13.4, 14.0, 2.8, 4.3, 0.0, 2.0, 50.0)  # tall cabinets
    S.box(13.5, 14.0, -1.8, -1.0, 0.0, 0.9, 45.0)  # sideboard
    S.box(13.97, 14.0, 1.0, 2.6, 1.2, 2.4, 85.0)  # projection screen / board

    # --- west wall (x = -2): a window, coat rack, plant, boxes
    S.window("-x", -1.0, 1.0, 0.9, 2.4, 0.25)
    S.vcyl(-1.6, 3.2, 0.025, 0.0, 1.8, 60.0)  # coat rack pole
    S.vcyl(-1.6, 3.2, 0.25, 0.0, 0.03, 60.0)  # its base
    S.obox(-1.6, 3.35, 0.25, 0.45, 0.8, 1.7, 0.4, 22.0)  # a coat
    S.vcyl(-1.55, -3.5, 0.22, 0.0, 0.45, 38.0)  # plant pot
    S.sphere(-1.55, -3.5, 1.0, 0.45, 25.0)  # foliage
    S.box(-1.95, -1.35, -2.6, -2.0, 0.0, 0.5, 42.0)  # cardboard boxes
    S.obox(-1.62, -2.3, 0.5, 0.45, 0.5, 0.88, 0.2, 40.0)

    # --- free-standing: a structural column, a desk island, a round meeting table
    S.vcyl(6.75, 0.5, 0.25, 0.0, H, 45.0)
    for yc in (-0.75, 1.05):
        S.desk(9.95, yc, math.pi, rng)  # the island's users sit on its east side, facing west
    for yc in (-0.75, 1.05):
        S.chair(10.72, yc + rng.uniform(-0.2, 0.2), math.pi + rng.uniform(-0.5, 0.5))
    S.vcyl(-0.5, 2.9, 0.5, 0.72, 0.75, 55.0)  # round meeting table top
    S.vcyl(-0.5, 2.9, 0.05, 0.0, 0.72, 60.0)
    for a in (0.3, 2.2, 4.0):
        S.chair(-0.5 + 0.8 * math.cos(a), 2.9 + 0.8 * math.sin(a), a + math.pi, 30.0)

    # --- ceiling: light fixtures, a duct, a beam, a cable tray
    for xc in (0.0, 3.0, 6.0, 9.0, 12.0):
        for yc in (-2.5, 0.5, 3.0):
            S.box(xc - 0.6, xc + 0.6, yc - 0.15, yc + 0.15, H - 0.14, H - 0.08, 150.0)
            S.box(xc - 0.02, xc + 0.02, yc - 0.01, yc + 0.01, H - 0.08, H, 60.0)
    S.hcyl(0, -1.9, 2.2, 0.2, -2.0, 14.0, 75.0)  # round duct along the room
    S.box(4.35, 4.65, -4.0, 4.5, H - 0.4, H, 35.0)  # beam across the room
    S.box(-2.0, 14.0, 2.4, 2.7, H - 0.3, H - 0.25, 65.0)  # cable tray
    return S


class SimSource:
    """A Mid-40 simulated on the GPU: `rate` points per second scanned through a scene from a moving sensor.

    poll() emits the points the wall clock says are due (the viewer); step(n) emits the next n points
    regardless of it (the tests). Both launch the scan on the device's current stream and hand back
    Warp arrays that stay valid until the next call.
    """

    kind = "sim"
    dev_type = MID40

    def __init__(self, rate: float = 100_000.0, device=None, motion: str = "static", scene: str = "box",
                 geometry: SceneBuilder | None = None, rosette=None):
        """geometry: a scene of your own instead of `scene`'s (the tests build variants of one);
        rosette: a rosette.RosetteModel to scan with instead of the built-in pattern (a fitted one makes the
        simulator scan like that unit)."""
        self.device = wp.get_device(device)
        self.rate = rate
        self.motion = SIM_MOTIONS.index(motion)
        self.scene = SIM_SCENES.index(scene)
        self.geometry = geometry or (office_scene() if scene == "office" else box_scene())
        self._scene = self.geometry.upload(self.device)
        self.rosette = rosette
        self._rosette_cache = {}
        self.t = 0.0
        self.last_t0 = 0.0  # time span of the batch the last poll() / step() returned
        self.last_t1 = 0.0
        self.emitted = 0
        self.wall = time.perf_counter()
        self.cap = 0
        self._grow(1 << 16)

    @property
    def label(self) -> str:
        return f"simulated Mid-40 (Warp, {SIM_MOTIONS[self.motion]}, {SIM_SCENES[self.scene]})"

    def _grow(self, n: int):
        self.cap = max(n, self.cap * 2)
        self.xyz = wp.empty(self.cap, dtype=wp.vec3, device=self.device)
        self.attr = wp.empty(self.cap, dtype=wp.uint32, device=self.device)
        self.tt = wp.empty(self.cap, dtype=wp.float32, device=self.device)

    def step(self, n: int):
        """Generate the next n points regardless of wall time (tests)."""
        if n <= 0:
            return None
        if n > self.cap:
            self._grow(n)
        dt = 1.0 / self.rate
        self.last_t0 = self.t
        if self.rosette is None:
            wp.launch(k_sim_scan, dim=n, device=self.device,
                      inputs=[self.t, dt, self.emitted, self.motion, self._scene, self.xyz, self.attr, self.tt])
        else:
            ros = self.rosette.gpu_params(self.t, self.t + n * dt, self.device, self._rosette_cache)
            wp.launch(k_sim_scan_rosette, dim=n, device=self.device,
                      inputs=[self.t, dt, self.emitted, self.motion, self._scene, ros, self.xyz, self.attr, self.tt])
        self.t += n * dt
        self.last_t1 = self.t - dt
        self.emitted += n
        return self.xyz, self.attr, self.tt, n

    def poll(self):
        now = time.perf_counter()
        n = int((now - self.wall) * self.rate)
        if n <= 0:
            return None
        n = min(n, int(self.rate * 0.25))  # after a stall, don't burst more than 250 ms
        self.wall = now
        return self.step(n)

    def pose_at(self, t: float) -> np.ndarray:
        """Ground-truth sensor-to-world pose at time t (same formulas as sim_pose)."""
        T = np.eye(4)
        if self.motion == 0:
            return T
        u = max(t - HOLD, 0.0) * OMEGA
        if self.motion == 1:
            p = np.array([5.0 + 3.0 * math.sin(u), 0.5 + 1.5 * math.sin(2.0 * u), 0.15 * math.sin(0.7 * u)])
            yaw = math.atan2(3.0 * math.cos(2.0 * u), 3.0 * math.cos(u))
            pitch = 0.05 * math.sin(1.3 * u)
            roll = 0.04 * math.sin(0.9 * u)
        else:
            p = np.array([3.0, 0.5, 0.0])
            yaw = 0.7 * math.sin(0.5 * u)
            pitch = 0.03 * math.sin(1.1 * u)
            roll = 0.0
        cx, sx, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        cz, sz = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T[:3, :3] = rz @ ry @ rx
        T[:3, 3] = p
        return T

    @staticmethod
    def person_at(t: float) -> np.ndarray:
        """Centre of the walking sphere at time t (same formula as person_centre)."""
        return np.array([PERSON_X, PERSON_AMP * math.sin(PERSON_RATE * t), PERSON_Z])

    def stats(self) -> dict:
        # the keys of _native.Device.stats() the viewer reads; a simulation has no packets to count
        return {"points": self.emitted, "packets": 0, "lost_packets": 0, "bad_packets": 0, "data_type": 0,
                "problems": []}

    def status(self):
        return None

    def close(self):
        pass


def tls_scan(geometry: SceneBuilder, stations, t: float = 0.0, step_deg: float = 0.2,
             elevation_deg: tuple = (-60.0, 89.0), noise: float = 0.002, device=None) -> np.ndarray:
    """A terrestrial scanner's capture of a scene: a spherical grid of rays (step_deg apart, over the elevation
    range) from each station, at time t (the walking sphere stands where it is then). World points (n, 3); rays
    that left every surface are dropped. With prior_map.from_points this is a simulated prior map."""
    d = wp.get_device(device)
    sc = geometry.upload(d)
    st = np.asarray(stations, dtype=np.float32).reshape(-1, 3)
    n_az = int(round(360.0 / step_deg))
    n_el = int((elevation_deg[1] - elevation_deg[0]) / step_deg) + 1
    out = wp.empty(len(st) * n_az * n_el, dtype=wp.vec3, device=d)
    wp.launch(k_tls_scan, dim=(len(st), n_az, n_el), device=d,
              inputs=[wp.array(st, dtype=wp.vec3, device=d), n_az, n_el, math.radians(elevation_deg[0]),
                      math.radians(step_deg), float(t), float(noise), sc, out])
    pts = out.numpy()
    return pts[np.all(np.isfinite(pts), axis=1) & (np.abs(pts).max(axis=1) < 1.0e3)]
