"""Prior maps: a previously captured scan of the area (E57), cached as a compact .npz.

    python -m livox_warp.prior_map convert scan.e57 --voxel 0.02      # -> maps/scan_20mm.npz
    python -m livox_warp.prior_map stations maps/scan_20mm.npz [--e57 scan.e57]   # add the scanner stations

The module provides
  - convert(): reads every scan of the E57 in its world pose, voxel-downsamples on the GPU
    (voxel_downsample(): sums relative to each voxel's centre, so float32 stays exact at any extent;
    mean colour and intensity are kept) and adds a PCA normal and a planarity per point from the
    neighbours within `normal_radius` (normals());
  - load(): the cache as a dict of its arrays (xyz, normal, planarity, rgb, intensity, count, voxel,
    normal_radius, source, stations), in a second instead of the minutes and many GB a full E57 read takes;
  - stations: each scan's scanner position, from the E57's scan headers, or estimated from the point density
    when the E57 is one merged scan (estimate_stations). occupancy.py needs them to tell a surface that is new
    from one the scan never saw; add them to an older cache with the `stations` command.

GPU queries against a loaded map (nearest point, distance field) are mapgrid.PriorGrid's.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import warp as wp

from . import gpu

# The key layout of gpu.voxel_key_ijk: three KEY_BITS-bit fields, each an axis index biased by KEY_BIAS.
# gpu.py spells the layout inline; these must match it.
KEY_BITS = gpu.VOXEL_KEY_BITS
KEY_BIAS = gpu.VOXEL_KEY_BIAS
KEY_MASK = gpu.VOXEL_KEY_MASK
KEY_SHIFT_Z = 2 * KEY_BITS
MAX_PROBES = 128  # open-addressing probes before a point counts as dropped
NORMALS_GRID = 256  # wp.HashGrid cells per axis for the normal estimation (16.8M cells: a few points each)


@wp.func
def voxel_key_unpack(key: wp.int64) -> wp.vec3i:
    """The (ix, iy, iz) that gpu.voxel_key_ijk packed into key."""
    m = wp.int64(KEY_MASK)
    bias = wp.int64(KEY_BIAS)
    ix = int((key & m) - bias)
    iy = int(((key >> wp.int64(KEY_BITS)) & m) - bias)
    iz = int(((key >> wp.int64(KEY_SHIFT_Z)) & m) - bias)
    return wp.vec3i(ix, iy, iz)


@wp.func
def voxel_centre(ix: int, iy: int, iz: int, voxel: float) -> wp.vec3:
    return wp.vec3((float(ix) + 0.5) * voxel, (float(iy) + 0.5) * voxel, (float(iz) + 0.5) * voxel)


@wp.kernel
def k_vox_insert(
    xyz: wp.array(dtype=wp.vec3),
    inten: wp.array(dtype=wp.float32),
    rgb: wp.array(dtype=wp.vec3),
    inv_v: float,
    voxel: float,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    rsum: wp.array(dtype=wp.vec3),
    csum: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    stats: wp.array(dtype=wp.int32),  # voxels claimed, points dropped
):
    """Add each point to its voxel's sums in an open-addressed table keyed by gpu.voxel_key_ijk."""
    i = wp.tid()
    p = xyz[i]
    ix = int(wp.floor(p[0] * inv_v))
    iy = int(wp.floor(p[1] * inv_v))
    iz = int(wp.floor(p[2] * inv_v))
    key = gpu.voxel_key_ijk(ix, iy, iz)
    center = voxel_centre(ix, iy, iz, voxel)
    h = gpu.slot_hash(key) & mask
    for probe in range(MAX_PROBES):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            if prev == wp.int64(-1):
                wp.atomic_add(stats, 0, 1)
            wp.atomic_add(rsum, s, p - center)
            c = rgb[i]
            wp.atomic_add(csum, s, wp.vec4(c[0], c[1], c[2], inten[i]))
            wp.atomic_add(cnt, s, 1)
            return
    wp.atomic_add(stats, 1, 1)


@wp.kernel
def k_vox_flag(keys: wp.array(dtype=wp.int64), flag: wp.array(dtype=wp.int32)):
    s = wp.tid()
    flag[s] = wp.where(keys[s] != wp.int64(-1), 1, 0)


@wp.kernel
def k_vox_emit(
    flag: wp.array(dtype=wp.int32),
    offs: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    rsum: wp.array(dtype=wp.vec3),
    csum: wp.array(dtype=wp.vec4),
    cnt: wp.array(dtype=wp.int32),
    voxel: float,
    out_xyz: wp.array(dtype=wp.vec3),
    out_col: wp.array(dtype=wp.vec4),
    out_cnt: wp.array(dtype=wp.int32),
):
    """Compact the claimed slots into dense arrays of mean position, colour + intensity and count."""
    s = wp.tid()
    if flag[s] == 0:
        return
    j = offs[s]
    ijk = voxel_key_unpack(keys[s])
    center = voxel_centre(ijk[0], ijk[1], ijk[2], voxel)
    c = float(cnt[s])
    out_xyz[j] = center + rsum[s] / c
    out_col[j] = csum[s] / c
    out_cnt[j] = cnt[s]


@wp.kernel
def k_normals(
    grid: wp.uint64,
    xyz: wp.array(dtype=wp.vec3),
    radius: float,
    out_n: wp.array(dtype=wp.vec3),
    out_planar: wp.array(dtype=wp.float32),
):
    """PCA normal and planarity ((l1 - l0) / l2, 1 = perfectly flat) from the neighbours within radius."""
    i = wp.tid()
    p = xyz[i]
    q = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    n = float(0.0)
    s1 = wp.vec3()
    s2 = wp.mat33()
    r2 = radius * radius
    while wp.hash_grid_query_next(q, j):
        d = xyz[j] - p
        if wp.dot(d, d) <= r2:
            n += 1.0
            s1 += d
            s2 += wp.outer(d, d)
    if n < 4.0:
        out_n[i] = wp.vec3(0.0, 0.0, 0.0)
        out_planar[i] = 0.0
        return
    mu = s1 / n
    Q, ev = wp.eig3(s2 / n - wp.outer(mu, mu))
    lo = int(0)
    hi = int(0)
    for a in range(1, 3):
        if ev[a] < ev[lo]:
            lo = a
        if ev[a] > ev[hi]:
            hi = a
    mid = 3 - lo - hi
    if lo == hi:
        mid = lo
    out_n[i] = wp.normalize(wp.vec3(Q[0, lo], Q[1, lo], Q[2, lo]))
    out_planar[i] = (ev[mid] - ev[lo]) / wp.max(ev[hi], 1.0e-12)


def voxel_downsample(xyz: np.ndarray, intensity: np.ndarray, rgb: np.ndarray, voxel: float, device=None,
                     chunk: int = 1 << 25, slots_log2: int = 27):
    """Mean position / colour / intensity per voxel of a (huge) cloud, on the GPU in chunks.
    Returns (xyz, rgb, intensity, count) per voxel, in the table's slot order."""
    d = wp.get_device(device)
    cap = 1 << slots_log2
    keys = wp.full(cap, -1, dtype=wp.int64, device=d)
    rsum = wp.zeros(cap, dtype=wp.vec3, device=d)
    csum = wp.zeros(cap, dtype=wp.vec4, device=d)
    cnt = wp.zeros(cap, dtype=wp.int32, device=d)
    stats = wp.zeros(2, dtype=wp.int32, device=d)
    n = len(xyz)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        px = wp.array(np.ascontiguousarray(xyz[a:b], dtype=np.float32), dtype=wp.vec3, device=d)
        pi = wp.array(np.ascontiguousarray(intensity[a:b], dtype=np.float32), dtype=wp.float32, device=d)
        pc = wp.array(np.ascontiguousarray(rgb[a:b], dtype=np.float32), dtype=wp.vec3, device=d)
        wp.launch(k_vox_insert, dim=b - a,
                  inputs=[px, pi, pc, 1.0 / voxel, voxel, cap - 1, keys, rsum, csum, cnt, stats], device=d)
        del px, pi, pc
    st = stats.numpy()
    if st[1]:
        raise RuntimeError(f"voxel table full: {st[1]} points dropped; raise slots_log2")
    flag = wp.zeros(cap, dtype=wp.int32, device=d)
    offs = wp.zeros(cap, dtype=wp.int32, device=d)
    wp.launch(k_vox_flag, dim=cap, inputs=[keys, flag], device=d)
    wp.utils.array_scan(flag, offs, inclusive=False)
    m = int(st[0])
    out_xyz = wp.zeros(m, dtype=wp.vec3, device=d)
    out_col = wp.zeros(m, dtype=wp.vec4, device=d)
    out_cnt = wp.zeros(m, dtype=wp.int32, device=d)
    wp.launch(k_vox_emit, dim=cap, inputs=[flag, offs, keys, rsum, csum, cnt, voxel, out_xyz, out_col, out_cnt],
              device=d)
    col = out_col.numpy()
    return out_xyz.numpy(), col[:, :3], col[:, 3], out_cnt.numpy()


def normals(xyz: np.ndarray, radius: float, device=None):
    """PCA normal and planarity of every point from its neighbours within `radius`, on the GPU.

    planarity = (l1 - l0) / l2 of the neighbourhood's covariance eigenvalues: 1 for a flat patch, 0 for a line
    or clutter. Points with fewer than 4 neighbours get a zero normal and planarity. The normals' sign is
    arbitrary (PCA has none)."""
    d = wp.get_device(device)
    pts = wp.array(xyz, dtype=wp.vec3, device=d)
    grid = wp.HashGrid(NORMALS_GRID, NORMALS_GRID, NORMALS_GRID, device=d)
    grid.build(pts, radius)
    n = wp.zeros(len(xyz), dtype=wp.vec3, device=d)
    pl = wp.zeros(len(xyz), dtype=wp.float32, device=d)
    wp.launch(k_normals, dim=len(xyz), inputs=[grid.id, pts, radius, n, pl], device=d)
    return n.numpy(), pl.numpy()


def convert(e57_path: str, out_path: str, voxel: float = 0.02, normal_radius: float = 0.08, device=None, log=print):
    """E57 -> the .npz cache: every scan in its world pose, voxel-downsampled, with normals and planarity.
    Colour is scaled to 8 bits; `source` records the E57's file name."""
    import pye57

    t0 = time.time()
    e = pye57.E57(e57_path)
    stations = e57_stations(e)
    xs, ins, cs = [], [], []
    total = 0
    for i in range(e.scan_count):
        h = e.get_header(i)
        f = h.point_fields
        d = e.read_scan(i, ignore_missing_fields=True, colors="colorRed" in f, intensity="intensity" in f,
                        transform=True)
        n = len(d["cartesianX"])
        total += n
        xs.append(np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1).astype(np.float32))
        ins.append(np.asarray(d.get("intensity", np.zeros(n)), dtype=np.float32))
        if "colorRed" in d:
            cs.append(np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], 1).astype(np.float32))
        else:
            cs.append(np.zeros((n, 3), np.float32))
        del d
    xyz = np.concatenate(xs)
    inten = np.concatenate(ins)
    rgb = np.concatenate(cs)
    del xs, ins, cs
    log(f"read {total:,} points in {time.time() - t0:.0f} s")
    t1 = time.time()
    vx, vc, vi, vn = voxel_downsample(xyz, inten, rgb, voxel, device)
    del xyz, inten, rgb
    log(f"{len(vx):,} voxels of {voxel * 100:g} cm in {time.time() - t1:.0f} s")
    t2 = time.time()
    nrm, planar = normals(vx, normal_radius, device)
    log(f"normals (radius {normal_radius * 100:g} cm) in {time.time() - t2:.0f} s")
    cmax = max(float(vc.max()), 1.0)
    rgb8 = np.clip(vc * (255.0 / cmax if cmax > 255.0 else 1.0), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, xyz=vx.astype(np.float32), normal=nrm.astype(np.float16),
                        planarity=planar.astype(np.float16), rgb=rgb8, intensity=vi.astype(np.float32),
                        count=np.minimum(vn, 65535).astype(np.uint16), voxel=np.float32(voxel),
                        normal_radius=np.float32(normal_radius), source=os.path.basename(e57_path), stations=stations)
    log(f"wrote {out_path} ({os.path.getsize(out_path) / 2**20:.0f} MiB) in {time.time() - t0:.0f} s total")


def e57_stations(e) -> np.ndarray:
    """(n_scans, 3) scanner positions from the scan headers of a pye57.E57 (their poses' translations)."""
    out = []
    for i in range(e.scan_count):
        tr = getattr(e.get_header(i), "translation", None)
        if tr is not None:
            out.append(np.asarray(tr, dtype=np.float64).reshape(3))
    return np.array(out, np.float32).reshape(-1, 3)


def estimate_stations(pm: dict, height: float = 1.5, cell: float = 0.1, smooth: float = 0.25, min_sep: float = 2.0,
                      rel: float = 0.1, column: float = 1.5) -> np.ndarray:
    """Where a terrestrial scanner stood, from a cache whose E57 kept no station poses (one merged scan).

    A scanner samples a fixed angular grid, so the raw points per voxel (the cache's `count`) fall off as
    cos(incidence) / distance^2: on the floor beneath a station and the ceiling above it they peak. The mean count
    per horizontal voxel in 10 cm columns, smoothed, has a local maximum over every station (at least min_sep
    apart, above `rel` of the strongest). Its height: the densest horizontal level within `column` metres is the
    floor or the ceiling, whichever is nearer; with surfaces 1.8-4 m below it, it is the ceiling and the lowest of
    them the floor (a desk top is not); the station stood `height` above the floor, inside the room. (The column
    is wide because a scanner cannot see the cone below its tripod: no floor lies near the point beneath it.)"""
    from scipy import ndimage

    xyz = np.asarray(pm["xyz"], np.float64)
    nz = np.abs(np.asarray(pm["normal"], np.float64)[:, 2])
    pl = np.asarray(pm.get("planarity", np.ones(len(xyz))), np.float32)
    cnt = np.asarray(pm["count"], np.float64)
    horiz = (nz > 0.97) & (pl > 0.5)
    p, c = xyz[horiz], cnt[horiz]
    if len(p) < 100:
        return np.zeros((0, 3), np.float32)
    lo = p[:, :2].min(0)
    ij = np.floor((p[:, :2] - lo) / cell).astype(np.int64)
    dims = ij.max(0) + 1
    total = np.zeros(dims)
    area = np.zeros(dims)
    np.add.at(total, (ij[:, 0], ij[:, 1]), c)
    np.add.at(area, (ij[:, 0], ij[:, 1]), 1.0)
    s = smooth / cell
    dens = ndimage.gaussian_filter(total, s) / np.maximum(ndimage.gaussian_filter(area, s), 1e-6)
    dens[ndimage.gaussian_filter(area, s) < 0.3] = 0.0  # no horizontal surface around: no station to place
    size = int(2 * round(min_sep / cell / 2) + 1)
    peaks = (dens == ndimage.maximum_filter(dens, size=size)) & (dens > rel * dens.max())
    out = []
    for i, j in zip(*np.nonzero(peaks)):
        centre = lo + (np.array([i, j]) + 0.5) * cell
        col = np.linalg.norm(p[:, :2] - centre, axis=1) < column
        if col.sum() < 10:
            continue
        zs, cs = p[col, 2], c[col]
        bins = np.floor(zs / 0.05).astype(np.int64)
        levels = {}
        for b in np.unique(bins):
            m = bins == b
            levels[b] = (float(np.median(zs[m])), float(cs[m].mean()), int(m.sum()))
        ordered = [v for v in sorted(levels.values(), key=lambda v: v[0]) if v[2] >= 20]  # surfaces, not clutter
        if not ordered:
            continue
        nearest = max(ordered, key=lambda v: v[1])  # the densest: the floor or the ceiling, whichever is nearer
        below = [v for v in ordered if 1.8 <= nearest[0] - v[0] <= 4.0]
        floor = below[0][0] if below else nearest[0]  # under a ceiling, the lowest surface (not a desk) is the floor
        ceiling = min([v[0] for v in ordered if v[0] - floor >= 1.8] or [floor + 2.5])
        out.append((centre[0], centre[1], min(floor + height, ceiling - 0.2)))
    return np.array(out, np.float32).reshape(-1, 3)


def add_stations(e57_path: str | None, npz_path: str, log=print):
    """Write scanner stations into an existing cache: from the E57's scan headers (seconds, not minutes), or, with
    no E57 or one whose scans carry no poses (one merged scan), estimated from the cache (estimate_stations)."""
    z = dict(np.load(npz_path))
    st = np.zeros((0, 3), np.float32)
    if e57_path:
        import pye57

        st = e57_stations(pye57.E57(e57_path))
        st = st[np.linalg.norm(st, axis=1) > 0]  # a merged scan's single header sits at the origin
    how = f"from {os.path.basename(e57_path)}" if len(st) else "estimated from the scan's point density"
    if not len(st):
        st = estimate_stations(z)
    z["stations"] = st
    np.savez_compressed(npz_path, **z)
    log(f"{npz_path}: {len(st)} stations {how}")


def load(path: str) -> dict:
    """A cache written by convert(), as a dict of its arrays."""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def from_points(xyz: np.ndarray, voxel: float = 0.02, normal_radius: float = 0.08, device=None,
                slots_log2: int = 24, stations=None) -> dict:
    """A prior map from a world point cloud, the way convert() makes one from an E57 (voxel means, normals,
    planarity, the stations if given; no colour). The tests build simulated prior maps with it from
    sources.tls_scan."""
    n = len(xyz)
    vx, _, _, vn = voxel_downsample(np.asarray(xyz, dtype=np.float32), np.zeros(n, np.float32),
                                    np.zeros((n, 3), np.float32), voxel, device, slots_log2=slots_log2)
    nrm, planar = normals(vx, normal_radius, device)
    return {"xyz": vx.astype(np.float32), "normal": nrm.astype(np.float32), "planarity": planar.astype(np.float32),
            "count": vn, "voxel": np.float32(voxel), "normal_radius": np.float32(normal_radius),
            "stations": np.zeros((0, 3), np.float32) if stations is None else np.asarray(stations, np.float32)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m livox_warp.prior_map")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="E57 -> downsampled .npz cache with normals")
    c.add_argument("e57")
    c.add_argument("--voxel", type=float, default=0.02, help="voxel size in m (default: %(default)s)")
    c.add_argument("--normal-radius", type=float, default=0.08, help="neighbourhood radius in m for the normals")
    c.add_argument("--out", default=None, help="output .npz (default: maps/<e57 name>_<voxel mm>mm.npz)")
    st = sub.add_parser("stations", help="add scanner stations to an existing .npz cache: from the E57's scan "
                                         "headers, or estimated from the cache when it has none")
    st.add_argument("npz")
    st.add_argument("--e57", help="the E57 the cache was made from (its scan poses); without one, or with a "
                                  "merged scan, the stations are estimated")
    a = p.parse_args(argv)
    if a.cmd == "stations":
        add_stations(a.e57, a.npz)
        return
    wp.config.quiet = True
    wp.init()
    if a.cmd == "convert":
        out = a.out or os.path.join("maps", os.path.splitext(os.path.basename(a.e57))[0].lower()
                                    + f"_{int(round(a.voxel * 1000))}mm.npz")
        convert(a.e57, out, a.voxel, a.normal_radius)


if __name__ == "__main__":
    main()
