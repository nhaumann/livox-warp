"""Firing-level reader for LVXR recordings: every firing with its exact time, the empty ones included.

The live path (the Rust Sink, see crates/livox/src/net.rs) drops "no return" records and rounds point times to
float32 seconds. The solvers want both: the scan-pattern fit (rosette.py) needs firing times exact to the
nanosecond, and the occupancy field (occupancy.py) reads a firing that returned nothing as free space along its
direction. This decodes the recording's datagrams (record.rs: `LVXR2\\n`, dev_type, profile, then
`host_ns:u64 len:u16 datagram`) like point.rs, on the Sink's time base, so `t` here is the viewer's `t` in
float64: seconds since the first packet, on the LiDAR's clock.

Data types: cartesian and spherical, plain, extended, dual and triple (the IMU packets are skipped). The
Mid-40 dual-return firmware (profile 1) sends plain records in pairs, two returns of one firing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

HEADER_LEN = 18  # a data packet's header: version, slot, id, rsvd, status u32, timestamp type, data type, timestamp
CLOCK_JUMP_NS = 1_000_000_000  # net.rs: a gap this long is the clock moving, not lost packets
PROFILE_STANDARD, PROFILE_MID40_DUAL = 0, 1
MID40_DUAL_RECORDS = 100  # point.rs MID40_RECORDS_PER_PACKET: 50 firings, two returns each
# data type -> (bytes per record, returns per record, spherical)
LAYOUTS = {0: (13, 1, False), 1: (9, 1, True), 2: (14, 1, False), 3: (10, 1, True), 4: (28, 2, False),
           5: (16, 2, True), 7: (42, 3, False), 8: (22, 3, True)}
FAST_DEVICES = {2, 3, 7}  # Tele-15, Horizon, Avia: 240k samples/s; the rest 100k


@dataclass
class Firings:
    """Every firing of a recording, in time order. R is the most returns any firing carries."""

    t: np.ndarray  # (n,) float64 s, the viewer's time base
    xyz: np.ndarray  # (n, R, 3) float32 m, sensor frame; zero where a return slot is empty
    refl: np.ndarray  # (n, R) uint8
    tag: np.ndarray  # (n, R) uint8 (the extended types' noise bits; 0 for the plain Mid-40 types)
    valid: np.ndarray  # (n, R) bool: a return was reported
    direction: np.ndarray  # (n, 3) float32 unit firing direction where the data gives one (spherical types,
    #                        or any return), NaN where it does not (an empty cartesian firing)
    dev_type: int
    profile: int
    interval: float  # s between firings

    @property
    def returns(self) -> int:
        return self.xyz.shape[1]

    @property
    def misses(self) -> np.ndarray:
        """Firings that returned nothing."""
        return ~self.valid.any(axis=1)

    def head(self, seconds: float) -> Firings:
        """The firings of the first `seconds`."""
        k = self.t < self.t[0] + seconds
        return replace(self, t=self.t[k], xyz=self.xyz[k], refl=self.refl[k], tag=self.tag[k], valid=self.valid[k],
                       direction=self.direction[k])

    def points(self, clean: bool = True, min_range: float = 0.3):
        """The returns as flat arrays (xyz, attr, t float64, firing index), time-ordered; attr packs reflectivity,
        tag << 8 and return << 16 as the live path does. clean: only confident returns beyond min_range (the
        rule of sources.clean_returns)."""
        f, r = np.nonzero(self.valid)
        xyz = self.xyz[f, r]
        tag = self.tag[f, r].astype(np.uint32)
        attr = self.refl[f, r].astype(np.uint32) | (tag << 8) | (r.astype(np.uint32) << 16)
        keep = np.ones(len(f), bool)
        if clean:
            keep = (tag == 0) & (np.linalg.norm(xyz, axis=1) > min_range)
        order = np.argsort(self.t[f[keep]], kind="stable")
        sel = np.flatnonzero(keep)[order]
        return xyz[sel], attr[sel], self.t[f[sel]], f[sel]


def _packets(data: bytes):
    """(dev_type, profile, datagram offsets, datagram lengths) of a whole recording file."""
    if data[:6] == b"LVXR1\n":
        dev, profile, off = data[6], PROFILE_STANDARD, 7
    elif data[:6] == b"LVXR2\n":
        dev, profile, off = data[6], data[7], 8
    else:
        raise ValueError("not an LVXR recording")
    offs, lens = [], []
    n = len(data)
    while off + 10 <= n:
        ln = data[off + 8] | (data[off + 9] << 8)
        if off + 10 + ln > n:
            break  # a truncated last datagram (the recorder was killed mid-write)
        offs.append(off + 10)
        lens.append(ln)
        off += 10 + ln
    return int(dev), int(profile), np.array(offs, np.int64), np.array(lens, np.int64)


def _u(buf: np.ndarray, at: np.ndarray, nbytes: int) -> np.ndarray:
    """Little-endian unsigned integers of nbytes at byte offsets `at` (any shape)."""
    v = np.zeros(at.shape, np.uint64)
    for j in range(nbytes):
        v |= buf[at + j].astype(np.uint64) << np.uint64(8 * j)
    return v


def _timestamps_ns(buf: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """DataHeader::timestamp_ns: types 0/1/4 are ns already; type 3 is GPS (hour, then us in the hour)."""
    ts = _u(buf, offs + 10, 8)
    gps = buf[offs + 8] == 3
    if gps.any():
        hour = buf[offs[gps] + 13].astype(np.uint64)
        us = _u(buf, offs[gps] + 14, 4)
        ts[gps] = (hour * np.uint64(3_600_000_000) + us) * np.uint64(1000)
    return ts


def _time_base(ts: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Packet times relative to the Sink's base (ns, int64): reset when the clock goes backwards, shifted over
    a forward jump of a second or more so time stays continuous (net.rs Sink::ingest)."""
    out = np.empty(len(ts), np.int64)
    base = None
    expected = None
    for i in range(len(ts)):
        t = int(ts[i])
        span = int(spans[i])
        if base is None or t < base:
            base = t
            expected = None
        if expected is not None and span > 0 and t > expected + span // 2:
            gap = t - expected
            if gap >= CLOCK_JUMP_NS:
                base += gap
        expected = t + span
        out[i] = t - base
    return out


def _decode(rec: np.ndarray, dtype: int):
    """Records (n_pk, n_rec, size) uint8 -> xyz (n_pk, n_rec, R, 3), refl, tag (n_pk, n_rec, R), and the
    spherical direction (n_pk, n_rec, 3) or None."""
    size, R, sph = LAYOUTS[dtype]
    sh = rec.shape[:2]

    def i32(o):
        return rec[..., o:o + 4].copy().view("<i4")[..., 0]

    def u32(o):
        return rec[..., o:o + 4].copy().view("<u4")[..., 0]

    def u16(o):
        return rec[..., o:o + 2].copy().view("<u2")[..., 0]

    xyz = np.zeros(sh + (R, 3), np.float32)
    refl = np.zeros(sh + (R,), np.uint8)
    tag = np.zeros(sh + (R,), np.uint8)
    direction = None

    def spherical(depth_mm, theta, phi):
        th = np.radians(theta.astype(np.float32) * 0.01)
        ph = np.radians(phi.astype(np.float32) * 0.01)
        unit = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).astype(np.float32)
        return unit, unit * (depth_mm.astype(np.float32) * 1e-3)[..., None]

    if dtype in (0, 2):
        xyz[..., 0, :] = np.stack([i32(0), i32(4), i32(8)], -1).astype(np.float32) * 1e-3
        refl[..., 0] = rec[..., 12]
        if dtype == 2:
            tag[..., 0] = rec[..., 13]
    elif dtype in (1, 3):
        direction, xyz[..., 0, :] = spherical(u32(0), u16(4), u16(6))
        refl[..., 0] = rec[..., 8]
        if dtype == 3:
            tag[..., 0] = rec[..., 9]
    elif dtype in (4, 7):
        for k in range(R):
            b = 14 * k
            xyz[..., k, :] = np.stack([i32(b), i32(b + 4), i32(b + 8)], -1).astype(np.float32) * 1e-3
            refl[..., k] = rec[..., b + 12]
            tag[..., k] = rec[..., b + 13]
    else:  # 5, 8: one direction, R (depth, reflectivity, tag)
        theta, phi = u16(0), u16(2)
        for k in range(R):
            b = 4 + 6 * k
            direction, xyz[..., k, :] = spherical(u32(b), theta, phi)
            refl[..., k] = rec[..., b + 4]
            tag[..., k] = rec[..., b + 5]
    return xyz, refl, tag, direction


def read_firings(path: str | os.PathLike) -> Firings:
    """Every firing of an LVXR recording (see the module docstring)."""
    with open(path, "rb") as f:
        data = f.read()
    dev, profile, offs, lens = _packets(data)
    buf = np.frombuffer(data, np.uint8)
    interval = 4_167 if dev in FAST_DEVICES else 10_000
    dual = profile == PROFILE_MID40_DUAL
    ok = lens >= HEADER_LEN
    offs, lens = offs[ok], lens[ok]
    dtypes = buf[offs + 9].astype(np.int64)
    point = np.isin(dtypes, list(LAYOUTS))
    offs, lens, dtypes = offs[point], lens[point], dtypes[point]
    sizes = np.array([LAYOUTS[int(d)][0] for d in dtypes], np.int64) if len(dtypes) else np.zeros(0, np.int64)
    records = (lens - HEADER_LEN) // np.maximum(sizes, 1)
    # the time base sees every point packet, malformed ones too (as the Sink does), before any is rejected
    firings_per = records // 2 if dual else records
    t_pk = _time_base(_timestamps_ns(buf, offs), firings_per * interval)
    whole = (lens - HEADER_LEN) == records * sizes
    if dual:  # point.rs: dual packets are exactly MID40_DUAL_RECORDS plain records
        whole &= np.isin(dtypes, [0, 1]) & (records == MID40_DUAL_RECORDS)
    offs, lens, dtypes, t_pk = offs[whole], lens[whole], dtypes[whole], t_pk[whole]

    R_max = max([LAYOUTS[int(d)][1] for d in np.unique(dtypes)] + [2 if dual else 1])
    parts = []
    for d in np.unique(dtypes):
        size, R, sph = LAYOUTS[int(d)]
        for ln in np.unique(lens[dtypes == d]):
            sel = np.flatnonzero((dtypes == d) & (lens == ln))
            n_rec = int((ln - HEADER_LEN) // size)
            body = b"".join(data[o + HEADER_LEN:o + HEADER_LEN + n_rec * size] for o in offs[sel].tolist())
            rec = np.frombuffer(body, np.uint8).reshape(len(sel), n_rec, size)
            xyz, refl, tag, direction = _decode(rec, int(d))
            if dual:
                n_f = n_rec // 2
                xyz = xyz.reshape(len(sel), n_f, 2, 3)
                refl = refl.reshape(len(sel), n_f, 2)
                tag = tag.reshape(len(sel), n_f, 2)
                if direction is not None:
                    direction = direction.reshape(len(sel), n_f, 2, 3)[:, :, 0]
            else:
                n_f = n_rec
            t = (t_pk[sel, None] + np.arange(n_f)[None, :] * interval).astype(np.float64) * 1e-9
            parts.append((t.ravel(), xyz.reshape(-1, xyz.shape[-2], 3), refl.reshape(-1, refl.shape[-1]),
                          tag.reshape(-1, tag.shape[-1]),
                          None if direction is None else direction.reshape(-1, 3)))
    if not parts:
        raise ValueError(f"{path}: no point packets")
    n = sum(len(p[0]) for p in parts)
    t = np.empty(n)
    xyz = np.zeros((n, R_max, 3), np.float32)
    refl = np.zeros((n, R_max), np.uint8)
    tag = np.zeros((n, R_max), np.uint8)
    direction = np.full((n, 3), np.nan, np.float32)
    a = 0
    for pt, px, pr, pg, pd in parts:
        b = a + len(pt)
        R = px.shape[1]
        t[a:b], xyz[a:b, :R], refl[a:b, :R], tag[a:b, :R] = pt, px, pr, pg
        if pd is not None:
            direction[a:b] = pd
        a = b
    valid = np.any(xyz != 0.0, axis=2)
    # cartesian firings with a return: their direction is the return's
    need = np.isnan(direction[:, 0]) & valid.any(axis=1)
    if need.any():
        first = np.argmax(valid[need], axis=1)
        v = xyz[np.flatnonzero(need), first]
        direction[need] = v / np.linalg.norm(v, axis=1, keepdims=True)
    order = np.argsort(t, kind="stable")
    return Firings(t=t[order], xyz=xyz[order], refl=refl[order], tag=tag[order], valid=valid[order],
                   direction=direction[order], dev_type=dev, profile=profile, interval=interval * 1e-9)
