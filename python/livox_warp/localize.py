"""Global localisation of a LiDAR scan in a prior map, on the GPU, in about a second.

A narrow cone (the Mid-40's 38 deg) sees one room of a whole building: too little for feature matching,
which has nothing but planes to describe there. This searches instead, coarse to fine:

1. "up" in the sensor frame comes from the scan itself: every near-horizontal plane (floor, ceiling,
   landing) is found, and one normal is fitted to all of them together, which is more accurate than any
   single plane. Both signs are candidates (the scanner may be upside down), and so is a second
   horizontal family if there is one; candidates are ranked by their coarse scores.
2. The sensor height is not searched blindly: the heights of the scan's horizontal surfaces (relative
   to the sensor, gravity-aligned) are correlated with the heights of the map's horizontal surfaces.
   The best few offsets are the only sensor heights tried: a plausible height above a mapped floor.
3. The map lives in a dense 5 cm grid that holds, per cell, the map point nearest to the cell centre
   (jump flooding on the GPU) and that distance in cm as a byte (a distance field). Every position on a
   25 cm grid at those heights, in free space, and every heading in 2 deg steps is scored by how close
   the scan's points land to the map (soft count within 25 cm), then the best ~100 distinct hypotheses
   are re-scored on a finer local grid within 12 cm.
4. The best 16 are refined together with a batched point-to-plane ICP on the GPU (correspondence: the
   nearest map point through the grid, exact within one cell; the 6x6 solve on the host), and the
   winner is the one with the most scan points within 3 cm of the map. Its margin over the best answer
   somewhere else (> 1 m or > 15 deg away) says how unique it is. Once one gravity candidate gives a
   clear, unique answer the others are not refined (early exit).

Everything the search launches or copies names its stream explicitly, and it never allocates after
construction, so it can run on a warp_worker thread while the render thread uses the default stream
(see prior_worker.py). Construct PriorGrid and GlobalLocalizer on the thread that owns the default
stream (or inside a WarpWorker scope()), and order the first use of their buffers after that.

usage: python -m livox_warp.localize MAP.npz RECORDING.lvxr [...] [--secs S] [--out POSES.npz]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from . import gpu, prior_map
from .mapgrid import CELL, PLANAR, Grid, PriorGrid, grid_at, nearest_point
from .odom import ACC_B, ACC_COUNT, ACC_H, ACC_SLOTS, ACC_STRIDE, accumulate, vec6

EVAL_RADII = (0.03, 0.05, 0.10)  # a pose's fit: the fractions of the scan within these distances (m) of the map
N_EVAL = len(EVAL_RADII)
_TRIU = np.triu_indices(6)  # the order accumulate() stores the upper triangle of the 6x6 normal matrix in


# ---- search ---------------------------------------------------------------------------------


@wp.kernel
def k_score_grid(
    pts: wp.array(dtype=wp.vec3),  # scan points, gravity-aligned sensor frame (z up)
    n_pts: int,
    pos: wp.array(dtype=wp.vec3),  # candidate sensor positions
    n_yaw: int,
    yaw0: float,
    dyaw: float,
    g: Grid,
    df: wp.array(dtype=wp.uint8),
    tau_cm: float,
    free_cm: float,
    out: wp.array(dtype=wp.float32),
):
    """Soft count of scan points within tau of the map, for every (position, heading); -1 where the sensor would
    sit inside a mapped surface or outside the grid."""
    h = wp.tid()
    ip = h / n_yaw
    iyaw = h - ip * n_yaw
    c = pos[ip]
    cc = grid_at(g, c)
    if cc < 0:
        out[h] = -1.0
        return
    if float(df[cc]) < free_cm:
        out[h] = -1.0
        return
    yaw = yaw0 + float(iyaw) * dyaw
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    inv_t2 = 1.0 / (tau_cm * tau_cm)
    s = float(0.0)
    for k in range(n_pts):
        p = pts[k]
        w = wp.vec3(c[0] + cy * p[0] - sy * p[1], c[1] + sy * p[0] + cy * p[1], c[2] + p[2])
        j = grid_at(g, w)
        if j >= 0:
            d = float(df[j])
            if d < tau_cm:
                s += 1.0 - d * d * inv_t2
    out[h] = s


@wp.kernel
def k_score_list(
    pts: wp.array(dtype=wp.vec3),
    n_pts: int,
    hyp: wp.array(dtype=wp.vec4),  # x, y, z, yaw
    g: Grid,
    df: wp.array(dtype=wp.uint8),
    tau_cm: float,
    out: wp.array(dtype=wp.float32),
):
    """The soft count of k_score_grid for a list of (x, y, z, yaw) hypotheses."""
    h = wp.tid()
    q = hyp[h]
    cy = wp.cos(q[3])
    sy = wp.sin(q[3])
    inv_t2 = 1.0 / (tau_cm * tau_cm)
    s = float(0.0)
    for k in range(n_pts):
        p = pts[k]
        w = wp.vec3(q[0] + cy * p[0] - sy * p[1], q[1] + sy * p[0] + cy * p[1], q[2] + p[2])
        j = grid_at(g, w)
        if j >= 0:
            d = float(df[j])
            if d < tau_cm:
                s += 1.0 - d * d * inv_t2
    out[h] = s


@wp.kernel
def k_icp_batch(
    pts: wp.array(dtype=wp.vec3),  # scan, sensor frame
    n_pts: int,
    T: wp.array(dtype=wp.mat44),  # one pose per hypothesis
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    planar: wp.array(dtype=wp.float32),
    max_d2: float,
    c2: float,
    alpha: float,
    acc: wp.array(dtype=wp.float64),
):
    """One Gauss-Newton accumulation of point-to-plane ICP for every hypothesis at once."""
    tid = wp.tid()
    h = tid / n_pts
    i = tid - h * n_pts
    w = wp.transform_point(T[h], pts[i])
    s = nearest_point(g, start, nn, xyz, w)
    if s < 0:
        return
    q = xyz[s]
    r = w - q
    if wp.dot(r, r) > max_d2:
        return
    n = nrm[s]
    M = wp.mat33(alpha, 0.0, 0.0, 0.0, alpha, 0.0, 0.0, 0.0, alpha)
    if planar[s] > PLANAR and wp.length(n) > 0.5:
        M = wp.outer(n, n) * (1.0 - alpha) + M  # the normal at full weight, along the surface only a little
    else:
        M = M * 4.0  # an edge or clutter: a weak point-to-point pull
    d2 = wp.dot(r, M * r)
    wgt = 1.0 / (1.0 + d2 / c2)
    wgt = wgt * wgt  # Geman-McClure
    J0 = vec6(1.0, 0.0, 0.0, 0.0, w[2], -w[1])
    J1 = vec6(0.0, 1.0, 0.0, -w[2], 0.0, w[0])
    J2 = vec6(0.0, 0.0, 1.0, w[1], -w[0], 0.0)
    slot = (h * ACC_SLOTS + (i & (ACC_SLOTS - 1))) * ACC_STRIDE
    accumulate(acc, slot, J0, J1, J2, M, r, wgt)


@wp.kernel
def k_reduce_batch(acc: wp.array(dtype=wp.float64), n_hyp: int, out: wp.array(dtype=wp.float64)):
    """Sum the ACC_SLOTS partial accumulators of every hypothesis into one."""
    tid = wp.tid()
    h = tid / ACC_STRIDE
    j = tid - h * ACC_STRIDE
    s = wp.float64(0.0)
    for k in range(ACC_SLOTS):
        s += acc[(h * ACC_SLOTS + k) * ACC_STRIDE + j]
    out[tid] = s


@wp.kernel
def k_eval_batch(
    pts: wp.array(dtype=wp.vec3),
    n_pts: int,
    T: wp.array(dtype=wp.mat44),
    g: Grid,
    start: wp.array(dtype=wp.int32),
    nn: wp.array(dtype=wp.int32),
    xyz: wp.array(dtype=wp.vec3),
    radii: wp.vec3,  # EVAL_RADII
    counts: wp.array(dtype=wp.int32),  # N_EVAL per hypothesis: the scan points within each radius
):
    """Count the scan points within radii[k] of their nearest map point (exact within one cell), per hypothesis."""
    tid = wp.tid()
    h = tid / n_pts
    i = tid - h * n_pts
    w = wp.transform_point(T[h], pts[i])
    s = nearest_point(g, start, nn, xyz, w)
    if s < 0:
        return
    d = wp.length(w - xyz[s])
    for k in range(N_EVAL):
        if d < radii[k]:
            wp.atomic_add(counts, h * N_EVAL + k, 1)


# ---- host side ------------------------------------------------------------------------------


def _rot_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _twist_apply(T: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a Gauss-Newton step (translation, rotation vector), left-multiplied in the map frame."""
    Rz = Rotation.from_rotvec(delta[3:]).as_matrix()
    out = T.copy()
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = Rz @ T[:3, 3] + delta[:3]
    return out


def pose_difference(A: np.ndarray, B: np.ndarray):
    """(translation m, rotation deg) between two 4x4 poses."""
    d = np.linalg.inv(A) @ B
    ang = math.degrees(math.acos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), ang


class ScanPrep:
    """Host preparation of a scan in the sensor frame: 5 cm cloud, normals, horizontal planes -> up candidates."""

    def __init__(self, xyz: np.ndarray, voxel: float = 0.05, max_tilt_deg: float = 45.0):
        import open3d as o3d

        try:
            o3d.utility.random.seed(0)  # repeatable RANSAC
        except AttributeError:
            pass
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
        pcd = pcd.voxel_down_sample(voxel)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        self.pts = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
        self.planes = []  # (inliers, unit normal, offset, inlier indices)
        rest = pcd
        idx = np.arange(len(self.pts))
        for _ in range(8):
            if len(rest.points) < 300:
                break
            model, inl = rest.segment_plane(0.03, 3, 1000)
            if len(inl) < 150:
                break
            n = np.array(model[:3], dtype=np.float64)
            k = np.linalg.norm(n)
            self.planes.append((len(inl), n / k, float(model[3]) / k, idx[np.asarray(inl)]))
            keep = np.ones(len(rest.points), bool)
            keep[np.asarray(inl)] = False
            idx = idx[keep]
            rest = rest.select_by_index(inl, invert=True)
        self.max_tilt = max_tilt_deg

    def up_candidates(self):
        """[(up unit vector, evidence)] ranked by plane support; each family gives both signs.

        One normal is fitted to all planes of a family together (the smallest eigenvector of the sum of their
        scatter matrices about their own centroids): several floors, landings and ceilings agree better than the
        largest one alone."""
        cos_tilt = math.cos(math.radians(self.max_tilt))
        fams = []  # [support, [plane indices], seed normal]
        for i, (cnt, n, _, _) in enumerate(self.planes):
            if abs(n[2]) < cos_tilt:
                continue  # a wall for a sensor that is roughly level (or upside down)
            for f in fams:
                if abs(float(n @ f[2])) > math.cos(math.radians(10.0)):
                    f[0] += cnt
                    f[1].append(i)
                    break
            else:
                fams.append([cnt, [i], n])
        out = []
        for support, members, n0 in sorted(fams, key=lambda f: -f[0])[:2]:
            S = np.zeros((3, 3))
            for i in members:
                P = self.pts[self.planes[i][3]]
                P = P - P.mean(0)
                S += P.T @ P
            w, V = np.linalg.eigh(S)
            n = V[:, 0]
            if n @ n0 < 0:
                n = -n
            out += [(n, support), (-n, support)]
        if not out:  # no floor or ceiling in view: assume a level sensor, either way up
            out = [(np.array([0.0, 0.0, 1.0]), 0), (np.array([0.0, 0.0, -1.0]), 0)]
        return out

    def height_profile(self, up: np.ndarray, lo: float, n_bins: int, bin_m: float):
        """Histogram of heights (along up) of the scan's horizontal points, relative to the sensor."""
        horiz = np.abs(self.normals @ up) > 0.97
        z = self.pts[horiz] @ up
        h, _ = np.histogram(z, bins=n_bins, range=(lo, lo + n_bins * bin_m))
        return h.astype(np.float32)


class GlobalLocalizer:
    """The search. All scratch buffers are allocated here (calling thread); localize() and refine() only launch
    and copy with an explicit stream, so they may run on a worker thread."""

    PTS_CAP = 1 << 17  # scan points after 5 cm thinning
    SCORE_CAP = 1 << 23  # (position, heading) hypotheses per coarse launch
    HYP_CAP = 1 << 17  # listed hypotheses per launch
    ICP_CAP = 16  # poses refined or evaluated together

    def __init__(self, grid: PriorGrid, device=None):
        self.grid = grid
        self.device = wp.get_device(device) if device is not None else grid.device
        d = self.device
        pin = d.is_cuda
        self.d_pts = wp.empty(self.PTS_CAP, dtype=wp.vec3, device=d)
        self.d_coarse = wp.empty(4096, dtype=wp.vec3, device=d)
        self.d_mid = wp.empty(8192, dtype=wp.vec3, device=d)
        self.d_pos = wp.empty(self.SCORE_CAP // 64, dtype=wp.vec3, device=d)
        self.d_score = wp.empty(self.SCORE_CAP, dtype=wp.float32, device=d)
        self.d_hyp = wp.empty(self.HYP_CAP, dtype=wp.vec4, device=d)
        self.d_T = wp.empty(self.ICP_CAP, dtype=wp.mat44, device=d)
        self.d_acc = wp.empty(self.ICP_CAP * ACC_SLOTS * ACC_STRIDE, dtype=wp.float64, device=d)
        self.d_acc_out = wp.empty(self.ICP_CAP * ACC_STRIDE, dtype=wp.float64, device=d)
        self.d_counts = wp.empty(self.ICP_CAP * N_EVAL, dtype=wp.int32, device=d)
        # pinned host mirrors; numpy views taken once, here (array.numpy() enters a ScopedStream)
        self.h_pts = wp.empty(self.PTS_CAP, dtype=wp.vec3, device="cpu", pinned=pin)
        self.h_pos = wp.empty(self.SCORE_CAP // 64, dtype=wp.vec3, device="cpu", pinned=pin)
        self.h_score = wp.empty(self.SCORE_CAP, dtype=wp.float32, device="cpu", pinned=pin)
        self.h_hyp = wp.empty(self.HYP_CAP, dtype=wp.vec4, device="cpu", pinned=pin)
        self.h_T = wp.empty(self.ICP_CAP, dtype=wp.mat44, device="cpu", pinned=pin)
        self.h_acc = wp.empty(self.ICP_CAP * ACC_STRIDE, dtype=wp.float64, device="cpu", pinned=pin)
        self.h_counts = wp.empty(self.ICP_CAP * N_EVAL, dtype=wp.int32, device="cpu", pinned=pin)
        self.np_pts = self.h_pts.numpy()
        self.np_pos = self.h_pos.numpy()
        self.np_score = self.h_score.numpy()
        self.np_hyp = self.h_hyp.numpy()
        self.np_T = self.h_T.numpy()
        self.np_acc = self.h_acc.numpy()
        self.np_counts = self.h_counts.numpy()
        self.seed = 0  # every search and refinement draws its subsets from a fresh generator with this seed
        # tuning
        self.xy_step = 0.25
        self.yaw_step_deg = 2.0
        self.n_heights = 8  # sensor heights tried per gravity candidate
        self.tau_coarse = 25.0  # cm
        self.tau_mid = 12.0
        self.free_cm = 10.0  # the sensor sits at least this far from any mapped surface
        self.n_coarse_pts = 1024
        self.n_mid_pts = 2048
        self.keep_coarse = 128  # coarse local maxima re-scored on the finer grid (per gravity candidate)
        self.n_icp = self.ICP_CAP  # medium winners refined by ICP (per gravity candidate)
        # unique: inliers (3 cm) at least accept_inliers, and the best answer elsewhere at least accept_gap lower and
        # below accept_ratio of it. A wrong pose that still puts the floor and one wall in place scores ~0.4.
        self.accept_inliers = 0.45
        self.accept_gap = 0.15
        self.accept_ratio = 0.75

    # ---- stream helpers ----------------------------------------------------------------------

    def _launch(self, stream, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device, stream=stream)

    def _sync(self, stream):
        if stream is not None:
            wp.synchronize_stream(stream)
        else:
            wp.synchronize_device(self.device)

    def _upload(self, stream, dst, host, host_np, data):
        n = len(data)
        host_np[:n] = data
        wp.copy(dst, host, count=n, stream=stream)
        self._sync(stream)  # the pinned staging buffer is reused by the next upload

    def _download(self, stream, src, host, host_np, n):
        wp.copy(host, src, count=n, stream=stream)
        self._sync(stream)
        return host_np[:n].copy()

    # ---- stages --------------------------------------------------------------------------------

    def _heights(self, prep: ScanPrep, up: np.ndarray):
        """Sensor heights where the scan's horizontal surfaces line up with the map's, best first."""
        h_map, lo, bin_m = self.grid.levels
        span = 6.0  # scan heights within +-6 m of the sensor
        nb = int(round(2 * span / bin_m))
        h_scan = prep.height_profile(up, -span, nb, bin_m)
        if h_scan.sum() < 50:
            return None
        # smooth both by +-2 bins so a 2-4 cm disagreement still overlaps
        k = np.array([0.25, 0.5, 1.0, 0.5, 0.25], np.float32)
        hm = np.convolve(np.sqrt(h_map), k, mode="same")
        hs = np.convolve(np.sqrt(h_scan), k, mode="same")
        corr = np.correlate(hm, hs, mode="full")  # corr[j] ~ shift of (j - (nb - 1)) bins
        shifts = np.arange(len(corr)) - (nb - 1)
        # sensor z = map height of scan bin 0 - (-span) ... scan bin b sits at -span + b*bin; map bin m at lo + m*bin
        z_sensor = lo + shifts * bin_m + span
        ok = (z_sensor >= self.grid.lo[2] - 0.5) & (z_sensor <= self.grid.hi[2] + 0.5)
        corr = np.where(ok, corr, -1.0)
        out = []
        for i in np.argsort(-corr):
            if corr[i] <= 0 or len(out) >= self.n_heights:
                break
            z = float(z_sensor[i])
            if all(abs(z - o) > 0.08 for o in out):
                out.append(z)
        return out

    def _coarse(self, stream, pts_g: np.ndarray, heights):
        """Every free-space position on the xy grid at the given heights x every heading."""
        g = self.grid
        gx = np.arange(g.lo[0], g.hi[0] + 1e-6, self.xy_step)
        gy = np.arange(g.lo[1], g.hi[1] + 1e-6, self.xy_step)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        pos = np.concatenate([np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1) for z in heights])
        pos = pos.astype(np.float32)
        n_yaw = int(round(360.0 / self.yaw_step_deg))
        dyaw = math.radians(self.yaw_step_deg)
        m = len(pts_g)
        self._upload(stream, self.d_coarse, self.h_pts, self.np_pts, pts_g.astype(np.float32))
        per = max(1, min(self.d_pos.shape[0], self.SCORE_CAP // n_yaw))
        scores = []
        for a in range(0, len(pos), per):
            b = min(len(pos), a + per)
            self._upload(stream, self.d_pos, self.h_pos, self.np_pos, pos[a:b])
            self._launch(stream, k_score_grid, (b - a) * n_yaw,
                         [self.d_coarse, m, self.d_pos, n_yaw, 0.0, dyaw, g.g, g.df, self.tau_coarse, self.free_cm,
                          self.d_score])
            scores.append(self._download(stream, self.d_score, self.h_score, self.np_score, (b - a) * n_yaw))
        sc = np.concatenate(scores)
        return pos, (len(heights), len(gx), len(gy), n_yaw), dyaw, sc

    def _local_maxima(self, sc: np.ndarray, shape, keep: int) -> np.ndarray:
        """Flat indices of the best `keep` coarse hypotheses that beat all 26 grid neighbours in (x, y, heading)
        at their height. Local maxima rather than a suppression radius: a false mode a grid step or two from the
        true pose can outscore it at this resolution, and a radius around the false mode would swallow the true
        one; keeping every local maximum leaves the finer stages to tell them apart."""
        nz, nx, ny, nyaw = shape
        pool = self._top(sc)
        pool = pool[sc[pool] > 0]
        iyaw = pool % nyaw
        ip = pool // nyaw
        iy = ip % ny
        ix = (ip // ny) % nx
        iz = ip // (ny * nx)
        best = np.ones(len(pool), bool)
        mine = sc[pool]
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oa in (-1, 0, 1):
                    if ox == oy == oa == 0:
                        continue
                    jx, jy = ix + ox, iy + oy
                    ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                    ja = (iyaw + oa) % nyaw
                    j = (((iz * nx + np.clip(jx, 0, nx - 1)) * ny + np.clip(jy, 0, ny - 1)) * nyaw + ja)
                    best &= ~ok | (sc[j] <= mine)
        return pool[best][:keep]

    @staticmethod
    def _top(scores: np.ndarray, pool: int = 20000) -> np.ndarray:
        """Indices of the `pool` best scores, best first."""
        pool = min(pool, len(scores))
        idx = np.argpartition(-scores, pool - 1)[:pool] if pool < len(scores) else np.arange(len(scores))
        return idx[np.argsort(-scores[idx])]

    @staticmethod
    def _nms(hyps: np.ndarray, scores: np.ndarray, keep: int, dist: float, ang_deg: float):
        """Greedy non-maximum suppression over (x, y, z, yaw) hypotheses; indices of the survivors, best first."""
        idx = GlobalLocalizer._top(scores)
        chosen = []
        ang = math.radians(ang_deg)
        for i in idx:
            if scores[i] <= 0:
                break
            h = hyps[i]
            ok = True
            for j in chosen:
                o = hyps[j]
                dyaw = abs((h[3] - o[3] + math.pi) % (2 * math.pi) - math.pi)
                if np.linalg.norm(h[:3] - o[:3]) < dist and dyaw < ang:
                    ok = False
                    break
            if ok:
                chosen.append(i)
                if len(chosen) >= keep:
                    break
        return np.array(chosen, dtype=np.int64)

    def _score_list(self, stream, pts_dev, m: int, hyps: np.ndarray, tau: float) -> np.ndarray:
        out = []
        for a in range(0, len(hyps), self.HYP_CAP):
            b = min(len(hyps), a + self.HYP_CAP)
            self._upload(stream, self.d_hyp, self.h_hyp, self.np_hyp, hyps[a:b].astype(np.float32))
            self._launch(stream, k_score_list, b - a,
                         [pts_dev, m, self.d_hyp, self.grid.g, self.grid.df, tau, self.d_score])
            out.append(self._download(stream, self.d_score, self.h_score, self.np_score, b - a))
        return np.concatenate(out)

    def _icp(self, stream, Ts: list, n_pts: int, schedule=((0.40, 8), (0.20, 8), (0.10, 8), (0.05, 10))):
        """Point-to-plane ICP of every pose in Ts at once against the map; the scan is already in d_pts.
        schedule: (correspondence distance m, iterations) stages, coarse to fine."""
        g = self.grid
        Ts = [T.copy() for T in Ts]
        k = len(Ts)
        alpha = 0.05
        for dmax, iters in schedule:
            c2 = (dmax * 0.5) ** 2
            for _ in range(iters):
                self.np_T[:k] = np.array([T.astype(np.float32) for T in Ts])
                wp.copy(self.d_T, self.h_T, count=k, stream=stream)
                self._launch(stream, gpu.k_fill_f64, self.d_acc.shape[0], [self.d_acc, wp.float64(0.0)])
                self._launch(stream, k_icp_batch, k * n_pts,
                             [self.d_pts, n_pts, self.d_T, g.g, g.start, g.nn, g.xyz, g.nrm, g.planar, dmax * dmax, c2,
                              alpha, self.d_acc])
                self._launch(stream, k_reduce_batch, k * ACC_STRIDE, [self.d_acc, k, self.d_acc_out])
                v = self._download(stream, self.d_acc_out, self.h_acc, self.np_acc, k * ACC_STRIDE)
                v = v.reshape(k, ACC_STRIDE)
                moved = 0.0
                for h in range(k):
                    if v[h, ACC_COUNT] < 30:
                        continue
                    H = np.zeros((6, 6))
                    H[_TRIU] = v[h, ACC_H:ACC_H + 21]
                    H += np.triu(H, 1).T
                    b = v[h, ACC_B:ACC_B + 6]
                    A = H + np.diag(1e-3 * np.diag(H) + 1e-9)
                    try:
                        delta = np.linalg.solve(A, -b)
                    except np.linalg.LinAlgError:
                        continue
                    Ts[h] = _twist_apply(Ts[h], delta)
                    moved = max(moved, float(np.linalg.norm(delta[:3])), float(np.linalg.norm(delta[3:])) * 5.0)
                if moved < 2e-4:
                    break
        return Ts

    def _evaluate(self, stream, Ts: list, n_pts: int) -> np.ndarray:
        """Fraction of the scan within each of EVAL_RADII of the map, per pose (at most ICP_CAP poses)."""
        g = self.grid
        k = len(Ts)
        self.np_T[:k] = np.array([T.astype(np.float32) for T in Ts])
        wp.copy(self.d_T, self.h_T, count=k, stream=stream)
        self._launch(stream, gpu.k_fill_i32, k * N_EVAL, [self.d_counts, 0])
        self._launch(stream, k_eval_batch, k * n_pts,
                     [self.d_pts, n_pts, self.d_T, g.g, g.start, g.nn, g.xyz, wp.vec3(*EVAL_RADII), self.d_counts])
        c = self._download(stream, self.d_counts, self.h_counts, self.np_counts, k * N_EVAL).reshape(k, N_EVAL)
        return c / max(n_pts, 1)

    # ---- tracking: refine a known pose -----------------------------------------------------------

    def refine(self, xyz: np.ndarray, T0: np.ndarray, stream=None, schedule=((0.30, 6), (0.12, 8), (0.05, 8)),
               starts=(), switch_margin: float = 0.05):
        """Point-to-plane ICP of a point set against the map, starting from T0 (which maps the points' frame into
        the map). The pose-tracking step: no search, a few milliseconds. Returns (T, fit, info); fit = the fraction
        of points within each of EVAL_RADII of the map at T.

        starts: extra map-frame offsets (x, y, z in m, optionally a heading in deg about the vertical through
        the points' centre) to also start from, all refined in one batch. Repetitive structure (a staircase)
        has near-equal minima a period apart, and an ICP started in the wrong one stays there. An offset start
        wins only if its 3 cm fit beats the plain result by switch_margin."""
        rng = np.random.default_rng(self.seed)  # the same points always give the same answer
        pts = np.asarray(xyz, dtype=np.float32)
        key = np.floor(pts / 0.05).astype(np.int64)  # one point per 5 cm voxel, like the search's scan
        _, first = np.unique(key, axis=0, return_index=True)
        pts = pts[np.sort(first)]
        if len(pts) > self.PTS_CAP:
            pts = pts[rng.choice(len(pts), self.PTS_CAP, replace=False)]
        n = len(pts)
        if n < 200:
            raise RuntimeError(f"too few points to refine ({n} after 5 cm thinning)")
        self._upload(stream, self.d_pts, self.h_pts, self.np_pts, pts)
        T0 = np.asarray(T0, dtype=np.float64)
        inits = [T0]
        c_map = T0[:3, :3] @ pts.mean(axis=0).astype(np.float64) + T0[:3, 3]  # the points' centre, in the map
        # the batch of ICP_CAP holds T0, the offset starts and, at evaluation, the unrefined T0 as it was
        for off in list(starts)[: self.ICP_CAP - 2]:
            Ti = T0.copy()
            off = np.asarray(off, dtype=np.float64)
            if len(off) > 3 and off[3] != 0.0:
                Rz = _rotz(math.radians(off[3]))  # a heading offset about the vertical through the points' centre
                Ti[:3, :3] = Rz @ Ti[:3, :3]
                Ti[:3, 3] = Rz @ (Ti[:3, 3] - c_map) + c_map
            Ti[:3, 3] += off[:3]
            inits.append(Ti)
        Ts = self._icp(stream, inits, n, schedule)
        fits = self._evaluate(stream, Ts + [T0], n)  # the refined poses, and T0 itself as it was
        fit_start = fits[-1]
        fits = fits[:-1]
        best = 0
        for i in range(1, len(Ts)):
            if fits[i, 0] > fits[best, 0] + (switch_margin if best == 0 else 0.0):
                best = i
        return Ts[best], fits[best], {"switched": best != 0, "plain_fit": float(fits[0, 0]), "n": n,
                                      "fit_start": float(fit_start[0])}

    # ---- the search -------------------------------------------------------------------------

    def localize(self, xyz: np.ndarray, stream=None, up_prior: np.ndarray | None = None, log=print,
                 progress=None):
        """Sensor-to-map pose (4x4) of a scan given in the sensor frame, plus diagnostics.

        up_prior: gravity "up" in the sensor frame if known (e.g. from a mount pose); it only orders the gravity
        candidates. progress(text) is called between stages."""
        t0 = time.perf_counter()
        say = progress or (lambda s: None)
        tm = {}
        rng = np.random.default_rng(self.seed)  # the same scan always gives the same answer
        prep = ScanPrep(xyz)
        pts = prep.pts
        if len(pts) > self.PTS_CAP:
            pts = pts[rng.choice(len(pts), self.PTS_CAP, replace=False)]
        n_pts = len(pts)
        if n_pts < 200:
            raise RuntimeError(f"scan too sparse to localise ({n_pts} points after 5 cm thinning)")
        self._upload(stream, self.d_pts, self.h_pts, self.np_pts, pts.astype(np.float32))
        # coarse / medium subsets: spread evenly over the scan (one per 15 cm voxel first)
        key = np.floor(pts / 0.15).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        spread = pts[first]
        sub_c = spread[rng.choice(len(spread), min(self.n_coarse_pts, len(spread)), replace=False)]
        sub_m = pts[rng.choice(n_pts, min(self.n_mid_pts, n_pts), replace=False)]
        ups = prep.up_candidates()
        if up_prior is not None:
            upp = np.asarray(up_prior, float) / np.linalg.norm(up_prior)
            ups.sort(key=lambda u: -float(u[0] @ upp))
        tm["prep"] = time.perf_counter() - t0
        # coarse for every gravity candidate, then refine the candidates in order of their best coarse score
        cands = []
        t1 = time.perf_counter()
        n_hyp_total = 0
        for up, support in ups:
            say(f"coarse search, up {np.round(up, 2)}")
            Rg = _rot_to(up, np.array([0.0, 0.0, 1.0]))  # sensor -> gravity-aligned sensor frame
            heights = self._heights(prep, up)
            if not heights:
                heights = list(np.arange(self.grid.lo[2], self.grid.hi[2] + 1e-6, self.xy_step))
            pc = sub_c @ Rg.T
            pos, shape, dyaw, sc = self._coarse(stream, pc, heights)
            n_yaw = shape[3]
            n_hyp_total += len(sc)
            best = self._local_maxima(sc, shape, self.keep_coarse)
            hyps = np.concatenate([pos[best // n_yaw], ((best % n_yaw) * dyaw)[:, None]], 1)
            cands.append({"up": up, "Rg": Rg, "heights": heights, "coarse": hyps, "coarse_score": sc[best],
                          "best_coarse": float(sc[best[0]]) / len(pc) if len(best) else 0.0,
                          "support": support})
        tm["coarse"] = time.perf_counter() - t1
        cands.sort(key=lambda c: -c["best_coarse"])
        # medium: a finer local grid around each coarse winner, tighter tolerance, more points
        results = []  # (inliers3, inliers5, inliers10, T, candidate index, medium score)
        tm["mid"] = tm["icp"] = 0.0
        refined = 0
        for ci, c in enumerate(cands):
            if not len(c["coarse"]):
                continue
            say(f"refining gravity candidate {ci + 1}/{len(cands)}")
            tm_a = time.perf_counter()
            Rg = c["Rg"]
            pm = sub_m @ Rg.T
            self._upload(stream, self.d_mid, self.h_pts, self.np_pts, pm.astype(np.float32))
            # +-1 coarse step: two modes a step apart share one coarse local maximum, and the window around it must
            # reach both
            off = np.arange(-self.xy_step, self.xy_step + 1e-6, self.xy_step / 4)
            offz = np.array([-0.04, 0.0, 0.04])
            offy = np.radians(np.arange(-self.yaw_step_deg, self.yaw_step_deg + 1e-6, 0.5))
            A, B, Cz, D = np.meshgrid(off, off, offz, offy, indexing="ij")
            local = np.stack([A.ravel(), B.ravel(), Cz.ravel(), D.ravel()], 1)
            hyps = (c["coarse"][:, None, :] + local[None, :, :]).reshape(-1, 4)
            ms = self._score_list(stream, self.d_mid, len(pm), hyps, self.tau_mid)
            # a small suppression radius: a false mode a few grid steps from the truth must not hide it (ICP and
            # the exact inlier count tell them apart, the 12 cm score only nearly does)
            best = self._nms(hyps, ms, self.n_icp, 0.1, 1.5)
            tm["mid"] += time.perf_counter() - tm_a
            tm_b = time.perf_counter()
            Ts = []
            for i in best:
                x, y, z, yaw = hyps[i]
                T = np.eye(4)
                T[:3, :3] = _rotz(yaw) @ Rg
                T[:3, 3] = (x, y, z)
                Ts.append(T)
            Ts = self._icp(stream, Ts, n_pts)
            fr = self._evaluate(stream, Ts, n_pts)
            for j, i in enumerate(best):
                results.append((float(fr[j, 0]), float(fr[j, 1]), float(fr[j, 2]), Ts[j], ci, float(ms[i]) / len(pm)))
            tm["icp"] += time.perf_counter() - tm_b
            refined += 1
            # early exit: this candidate's best answer is good and unique
            results.sort(key=lambda r: -r[0])
            top = results[0]
            alt = self._runner_up(results)
            if self._unique(top[0], alt):
                break
        results.sort(key=lambda r: -r[0])
        if not results:
            raise RuntimeError("no hypothesis survived (is the scan inside the map?)")
        top = results[0]
        alt = self._runner_up(results)
        tm["total"] = time.perf_counter() - t0
        info = {"inliers": top[0], "inliers5": top[1], "inliers10": top[2], "runner_up": alt,
                "unique": self._unique(top[0], alt),
                "up": cands[top[4]]["up"], "heights": cands[top[4]]["heights"], "gravity_candidates": len(cands),
                "refined_candidates": refined, "hypotheses": n_hyp_total, "scan_points": n_pts, "times": tm}
        log(f"  {tm['total']:.2f} s (prep {tm['prep']:.2f}, coarse {tm['coarse']:.2f} for {n_hyp_total / 1e6:.1f}M "
            f"hypotheses, medium {tm['mid']:.2f}, ICP {tm['icp']:.2f}; {refined}/{len(cands)} gravity candidates "
            f"refined): inliers {top[0]:.3f} within 3 cm (runner-up elsewhere {alt:.3f})")
        return top[3], info

    def _unique(self, best: float, alt: float) -> bool:
        return best >= self.accept_inliers and best - alt >= self.accept_gap and alt <= self.accept_ratio * best

    @staticmethod
    def _runner_up(results) -> float:
        best = results[0][3]
        for r in results[1:]:
            dt, ang = pose_difference(best, r[3])
            if dt > 1.0 or ang > 15.0:
                return r[0]
        return 0.0


class Localizer:
    """Single-threaded convenience wrapper (the CLI, tools and tests): builds the grid and searches on one stream."""

    def __init__(self, map_xyz: np.ndarray, map_normal: np.ndarray, map_planarity: np.ndarray | None = None,
                 device=None, stream=None):
        self.device = wp.get_device(device)
        host = {"xyz": map_xyz, "normal": map_normal}
        if map_planarity is not None:
            host["planarity"] = map_planarity
        self.grid = PriorGrid(host, self.device)
        self.loc = GlobalLocalizer(self.grid, self.device)
        self.stream = stream if stream is not None else (wp.get_stream(self.device) if self.device.is_cuda else None)
        wp.synchronize_device(self.device)
        self.grid.build(self.stream)

    def search(self, xyz: np.ndarray, log=print, up_prior=None):
        """GlobalLocalizer.localize on this stream: the pose of a sensor-frame scan, and diagnostics."""
        return self.loc.localize(xyz, self.stream, up_prior=up_prior, log=log)

    def refine(self, xyz: np.ndarray, T0: np.ndarray, **kw):
        """GlobalLocalizer.refine on this stream: ICP of a point set from the known pose T0."""
        return self.loc.refine(xyz, T0, self.stream, **kw)


# ---- command line ---------------------------------------------------------------------------


def load_scan(path: str, secs: float | None = None) -> np.ndarray:
    """All clean points of a recording (sensor frame; noise-tagged and very close returns dropped), or those of
    its first `secs` seconds."""
    from .sources import ReplaySource, clean_returns  # the native module: only the CLI needs it

    src = ReplaySource(path, speed=200.0, looped=False)
    try:
        xyz, attr, t = src.read_all()
    finally:
        src.close()
    keep = clean_returns(xyz, attr)
    xyz, t = xyz[keep], t[keep]
    if len(xyz) == 0:
        raise RuntimeError(f"{path}: no clean points")
    if secs is not None:
        xyz = xyz[t - t.min() < secs]
    return xyz


def pose_key(recording: str) -> str:
    """The npz key of a recording's pose: its file name with dots replaced (npz keys are identifiers)."""
    return os.path.basename(recording).replace(".", "_")


def save_poses(path: str, poses: dict):
    """Write poses (pose_key -> 4x4) to an npz, keeping the poses of recordings already in it."""
    if os.path.exists(path):
        old = dict(np.load(path))
        old.update(poses)
        poses = old
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **poses)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m livox_warp.localize",
                                description="Localise recordings in a prior map and save the sensor poses.")
    p.add_argument("map", help="prior map .npz (python -m livox_warp.prior_map convert)")
    p.add_argument("recordings", nargs="+", help=".lvxr recordings, each localised from its points")
    p.add_argument("--secs", type=float, default=None, help="use only the first S seconds of each recording")
    p.add_argument("--out", default=os.path.join("maps", "recording_poses.npz"),
                   help="poses npz to write, relative to the current directory (default: %(default)s); "
                        "poses of other recordings already in it are kept")
    a = p.parse_args(argv)
    wp.config.quiet = True
    wp.init()
    t0 = time.perf_counter()
    pm = prior_map.load(a.map)
    loc = Localizer(pm["xyz"], pm["normal"], pm.get("planarity"))
    print(f"{a.map}: {len(pm['xyz']):,} points, grid {loc.grid.dims.tolist()} of {CELL * 100:g} cm built in "
          f"{loc.grid.build_s:.2f} s ({time.perf_counter() - t0:.1f} s with loading)")
    poses = {}
    for path in a.recordings:
        xyz = load_scan(path, a.secs)
        print(f"{os.path.basename(path)}: {len(xyz):,} points" + (f" (first {a.secs:g} s)" if a.secs else ""))
        T, info = loc.search(xyz)
        print(f"  sensor at {T[:3, 3].round(3)}, sensor z in map {T[:3, 2].round(3)}, forward {T[:3, 0].round(3)}"
              + ("" if info["unique"] else "  (NOT unique)"))
        poses[pose_key(path)] = T
    save_poses(a.out, poses)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
