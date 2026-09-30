"""Scene understanding on the visible points, all on the GPU: ground, clusters, tracks.

Ground: a grid-based segmentation in the spirit of Patchwork / GSeg3D. Points fall into 2D cells
(toroidal 2048 x 2048 grid so any world extent works); each cell records its lowest z and how many
points support that minimum. A point is ground when it sits within `thresh` of the lowest
supported minimum in the 5x5 cells around it, allowing `slope_step` of rise per cell so slopes and
curbs stay ground while table tops and crate lids do not. With normals available, steep surfaces
near ground level (the foot of a wall) are excluded.

Clusters: Euclidean connected components of the non-ground points, on a voxel summary (one node
per `voxel`), linked when closer than `connect`. Labels converge by min-label propagation with root
hooking and pointer jumping, iterated until nothing changes. Each cluster's count, centroid and box
are read back for the tracker.

Tracks: a nearest-neighbour tracker on the host keeps cluster identity across frames and
estimates velocity over sensor time, which the shader paints as per-point speed (the software
stand-in for the per-point velocity that FMCW LiDARs measure).
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from . import gpu

GRID = 2048  # ground cells per axis (toroidal)


@wp.func
def cell_of(x: float, y: float, inv_cell: float) -> int:
    cx = int(wp.floor(x * inv_cell)) & (GRID - 1)
    cy = int(wp.floor(y * inv_cell)) & (GRID - 1)
    return cy * GRID + cx


@wp.kernel
def k_gnd_min(xyz: wp.array(dtype=wp.vec3), inv_cell: float, cell_min: wp.array(dtype=wp.float32)):
    i = wp.tid()
    p = xyz[i]
    wp.atomic_min(cell_min, cell_of(p[0], p[1], inv_cell), p[2])


@wp.kernel
def k_gnd_support(
    xyz: wp.array(dtype=wp.vec3),
    inv_cell: float,
    thick: float,
    cell_min: wp.array(dtype=wp.float32),
    cell_sup: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = xyz[i]
    c = cell_of(p[0], p[1], inv_cell)
    if p[2] - cell_min[c] < thick:
        wp.atomic_add(cell_sup, c, 1)


@wp.kernel
def k_gnd_label(
    xyz: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    use_nrm: int,
    inv_cell: float,
    cell_min: wp.array(dtype=wp.float32),
    cell_sup: wp.array(dtype=wp.int32),
    min_sup: int,
    thresh: float,
    slope_step: float,
    nz_min: float,
    out_gnd: wp.array(dtype=wp.int32),
    out_hag: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = xyz[i]
    cx = int(wp.floor(p[0] * inv_cell))
    cy = int(wp.floor(p[1] * inv_cell))
    g = float(1.0e30)
    g_floor = float(1.0e30)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c = ((cy + dy) & (GRID - 1)) * GRID + ((cx + dx) & (GRID - 1))
            if cell_sup[c] >= min_sup:
                cand = cell_min[c] + slope_step * float(wp.max(wp.abs(dx), wp.abs(dy)))
                if cand < g:
                    g = cand
                    g_floor = cell_min[c]
    if g > 1.0e29:
        out_gnd[i] = 0
        out_hag[i] = 0.0
        return
    h = p[2] - g
    is_gnd = int(0)
    if h < thresh:
        is_gnd = 1
        # a steep normal within the ground band is the foot of a wall, not ground; but a point right at
        # ground level keeps its label even if its normal tilts where the floor meets something
        if use_nrm != 0 and h > 0.2 * thresh:
            nv = nrm[i]
            if wp.abs(nv[2]) < nz_min:
                is_gnd = 0
    out_gnd[i] = is_gnd
    # Height above ground. Ground points sit on the slope-allowed estimate. Everything else is measured
    # from the lowest supported ground it stands over, without the slope allowance, so a crate top
    # reads its real height instead of one lowered by slope_step per cell of the crate's footprint.
    if is_gnd != 0:
        out_hag[i] = h
    else:
        out_hag[i] = p[2] - g_floor


@wp.kernel
def k_cl_insert(
    xyz: wp.array(dtype=wp.vec3),
    gnd: wp.array(dtype=wp.int32),
    use_gnd: int,
    inv_v: float,
    mask: int,
    keys: wp.array(dtype=wp.int64),
    acc: wp.array(dtype=wp.vec4),
    slot_of_pt: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    slot_of_pt[i] = -1
    if use_gnd != 0:
        if gnd[i] != 0:
            return
    w = xyz[i]
    key = gpu.voxel_key(w, inv_v)
    h = gpu.slot_hash(key) & mask
    for probe in range(64):
        s = (h + probe) & mask
        prev = wp.atomic_cas(keys, s, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            wp.atomic_add(acc, s, wp.vec4(w[0], w[1], w[2], 1.0))
            slot_of_pt[i] = s
            return


@wp.kernel
def k_cc_init(label: wp.array(dtype=wp.int32)):
    i = wp.tid()
    label[i] = i


@wp.kernel
def k_cc_step(
    grid: wp.uint64,
    v_xyz: wp.array(dtype=wp.vec3),
    radius: float,
    label: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = v_xyz[i]
    r2 = radius * radius
    q = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    m = label[i]
    while wp.hash_grid_query_next(q, j):
        d = v_xyz[j] - p
        if wp.dot(d, d) <= r2:
            lj = label[j]
            if lj < m:
                m = lj
    li = label[i]
    if m < li:
        # hook the ancestor i points at as well, so a whole tree merges at once (Shiloach-Vishkin style)
        wp.atomic_min(label, li, m)
        wp.atomic_min(label, i, m)
        changed[0] = 1


@wp.kernel
def k_cc_jump(label: wp.array(dtype=wp.int32)):
    i = wp.tid()
    label[i] = label[label[i]]


@wp.kernel
def k_cc_reset(
    st_cnt: wp.array(dtype=wp.float32),
    st_sum: wp.array(dtype=wp.vec3),
    st_lo: wp.array(dtype=wp.vec3),
    st_hi: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    st_cnt[i] = 0.0
    st_sum[i] = wp.vec3()
    st_lo[i] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    st_hi[i] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)


@wp.kernel
def k_cc_stats(
    label: wp.array(dtype=wp.int32),
    v_xyz: wp.array(dtype=wp.vec3),
    v_w: wp.array(dtype=wp.float32),
    st_cnt: wp.array(dtype=wp.float32),
    st_sum: wp.array(dtype=wp.vec3),
    st_lo: wp.array(dtype=wp.vec3),
    st_hi: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    r = label[i]
    p = v_xyz[i]
    w = v_w[i]
    wp.atomic_add(st_cnt, r, w)
    wp.atomic_add(st_sum, r, p * w)
    wp.atomic_min(st_lo, r, p)
    wp.atomic_max(st_hi, r, p)


@wp.kernel
def k_cc_flag(
    label: wp.array(dtype=wp.int32),
    st_cnt: wp.array(dtype=wp.float32),
    min_pts: float,
    flag: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    f = int(0)
    if label[i] == i:
        if st_cnt[i] >= min_pts:
            f = 1
    flag[i] = f


@wp.kernel
def k_cc_compact(
    flag: wp.array(dtype=wp.int32),
    offs: wp.array(dtype=wp.int32),
    st_cnt: wp.array(dtype=wp.float32),
    st_sum: wp.array(dtype=wp.vec3),
    st_lo: wp.array(dtype=wp.vec3),
    st_hi: wp.array(dtype=wp.vec3),
    max_out: int,
    dense: wp.array(dtype=wp.int32),
    out_cnt: wp.array(dtype=wp.float32),
    out_cen: wp.array(dtype=wp.vec3),
    out_lo: wp.array(dtype=wp.vec3),
    out_hi: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dense[i] = -1
    if flag[i] != 0:
        j = offs[i]
        if j < max_out:
            dense[i] = j
            out_cnt[j] = st_cnt[i]
            out_cen[j] = st_sum[i] / st_cnt[i]
            out_lo[j] = st_lo[i]
            out_hi[j] = st_hi[i]


@wp.kernel
def k_cc_gather(
    slot_of_pt: wp.array(dtype=wp.int32),
    vidx: wp.array(dtype=wp.int32),
    label: wp.array(dtype=wp.int32),
    dense: wp.array(dtype=wp.int32),
    out_cid: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    s = slot_of_pt[i]
    if s < 0:
        out_cid[i] = -1
        return
    out_cid[i] = dense[label[vidx[s]]]


class Tracker:
    """Greedy nearest-centroid tracker that runs on sensor time, not on render frames.

    Association runs on every call (the dense cluster ids change with every clustering pass): every
    track is predicted to `now` from its last matched position and velocity, and the closest
    prediction-cluster pairs are matched first, as long as their distance is within the gate
    `gate_base` + `gate_vel` * (the distance the track would have moved since it was last seen)
    + `gate_ext` * (the cluster's box diagonal), all in metres: a fast track may have moved further
    than predicted, and a big cluster's centroid jumps as its visible part changes. A track keeps
    centroid samples at least `baseline` seconds of sensor time apart over the last `window`
    seconds, and its velocity is the least-squares slope through them. The slope is reported as zero
    unless it exceeds `k_sig` times its own standard error: a static object's centroid wanders by a
    few cm as the non-repetitive scan pattern covers it, and that must not read as motion. The
    prediction starts from the last matched position (never from repeated extrapolation), tracks are
    dropped `max_age` seconds after they were last seen and confirmed once followed for `confirm_s`.
    So the output is the same whether the viewer calls update() at 10 Hz or 144 Hz."""

    def __init__(self, baseline: float = 0.1, window: float = 0.8, k_sig: float = 2.0, max_age: float = 0.8,
                 confirm_s: float = 0.3, gate_base: float = 0.75, gate_vel: float = 0.5, gate_ext: float = 0.25):
        self.tracks = {}
        self.next_id = 0
        self.last_t = None
        self.baseline = baseline
        self.window = window
        self.k_sig = k_sig
        self.max_age = max_age
        self.confirm_s = confirm_s
        self.gate_base = gate_base
        self.gate_vel = gate_vel
        self.gate_ext = gate_ext

    def _velocity(self, hist):
        """Least-squares slope of the centroid samples, zero when it is not significant."""
        if len(hist) < 2:
            return np.zeros(3)
        t = np.array([h[0] for h in hist])
        p = np.array([h[1] for h in hist])
        tc = t - t.mean()
        den = float((tc ** 2).sum())
        if den <= 0.0:
            return np.zeros(3)
        pc = p - p.mean(axis=0)
        v = (tc[:, None] * pc).sum(axis=0) / den
        if len(hist) >= 3:
            res = pc - tc[:, None] * v
            se = np.sqrt((res ** 2).sum(axis=0) / (len(hist) - 2) / den)  # per-axis standard error
            if np.linalg.norm(v) < self.k_sig * np.linalg.norm(se):
                return np.zeros(3)
        return v

    def reset(self):
        self.tracks.clear()
        self.last_t = None

    def _measure(self, tr: dict, c: dict, now: float):
        """Fold a matched cluster into its track."""
        hist = tr["hist"]
        if now - hist[-1][0] >= self.baseline - 1e-3:
            hist.append((now, c["cen"].copy()))
            while hist[0][0] < now - self.window - 1e-6:
                hist.pop(0)
            tr["vel"] = self._velocity(hist)
            tr["n_meas"] += 1
        tr["pos"], tr["t"], tr["seen"] = c["cen"].copy(), now, now
        tr["lo"], tr["hi"], tr["cnt"] = c["lo"].copy(), c["hi"].copy(), c["cnt"]

    def _spawn(self, c: dict, now: float) -> int:
        """A new track on an unmatched cluster; returns its id."""
        t = self.next_id
        self.next_id += 1
        self.tracks[t] = {"pos": c["cen"].copy(), "vel": np.zeros(3), "lo": c["lo"].copy(), "hi": c["hi"].copy(),
                          "cnt": c["cnt"], "t": now, "seen": now, "born": now, "hist": [(now, c["cen"].copy())],
                          "n_meas": 0}
        return t

    def update(self, clusters: list, now: float):
        """Associate the clusters (dicts with cen, lo, hi, cnt) with the tracks.

        Returns per cluster its track id (int32 array) and velocity ((n, 3) array)."""
        if self.last_t is not None and now < self.last_t - 1e-6:
            self.reset()  # the sensor clock restarted (replay loop)
        self.last_t = now
        ids = list(self.tracks.keys())
        tid_of = np.full(len(clusters), -1, np.int32)
        vel_of = np.zeros((len(clusters), 3))
        if ids and clusters:
            dts = np.array([max(now - self.tracks[t]["t"], 0.0) for t in ids])
            vel = np.array([self.tracks[t]["vel"] for t in ids])
            pred = np.array([self.tracks[t]["pos"] for t in ids]) + vel * dts[:, None]
            cen = np.array([c["cen"] for c in clusters])
            ext = np.array([np.linalg.norm(c["hi"] - c["lo"]) for c in clusters])
            dist = np.linalg.norm(pred[:, None, :] - cen[None, :, :], axis=2)
            gate = self.gate_base + (self.gate_vel * np.linalg.norm(vel, axis=1) * dts)[:, None] + self.gate_ext * ext
            used_t, used_c = set(), set()
            for flat in np.argsort(dist, axis=None, kind="stable"):  # closest pairs first
                a, b = divmod(int(flat), len(clusters))
                if a in used_t or b in used_c or dist[a, b] > gate[a, b]:
                    continue
                used_t.add(a)
                used_c.add(b)
                tr = self.tracks[ids[a]]
                self._measure(tr, clusters[b], now)
                tid_of[b] = ids[a]
                vel_of[b] = tr["vel"]
        for t in [t for t, tr in self.tracks.items() if now - tr["seen"] > self.max_age]:
            del self.tracks[t]
        for b, c in enumerate(clusters):
            if tid_of[b] < 0:
                tid_of[b] = self._spawn(c, now)
        return tid_of, vel_of

    def confirmed(self):
        """Tracks seen in the latest update that have been followed for at least confirm_s."""
        return [(t, tr) for t, tr in self.tracks.items()
                if tr["seen"] == self.last_t and tr["n_meas"] >= 1 and tr["seen"] - tr["born"] >= self.confirm_s - 1e-3]


class Perception:
    """Ground labels, clusters and tracks for the pipeline's visible points (pipe.c_xyz[:pipe.count]).

    ground() writes pipe.gnd / pipe.hag, cluster() writes pipe.cid and reads the cluster summaries
    back, track() uploads pipe.cl_vel / pipe.cl_tid; the shader reads all five.

    Render thread only. The kernel launches and the fill_() / zero_() calls go to the device's
    current stream, while the readbacks and uploads are issued explicitly on `self.stream`, the
    stream that was current when this object was built. Those are the same stream only on the
    thread that built it, and that is what orders the copies after the kernels; the odometry worker
    runs on its own stream and must not call in here.
    """

    def __init__(self, pipe: gpu.Pipeline, device=None, cluster_slots: int = 1 << 20):
        self.pipe = pipe
        self.device = wp.get_device(device)
        d = self.device
        n = pipe.work_cap
        self.cell_min = wp.zeros(GRID * GRID, dtype=wp.float32, device=d)
        self.cell_sup = wp.zeros(GRID * GRID, dtype=wp.int32, device=d)

        assert cluster_slots & (cluster_slots - 1) == 0
        self.cl_cap = cluster_slots
        c = cluster_slots
        self.cl_keys = wp.full(c, -1, dtype=wp.int64, device=d)
        self.cl_acc = wp.zeros(c, dtype=wp.vec4, device=d)
        self.cl_flag = wp.zeros(c, dtype=wp.int32, device=d)
        self.cl_offs = wp.zeros(c, dtype=wp.int32, device=d)
        self.cl_vidx = wp.zeros(c, dtype=wp.int32, device=d)
        self.cv_xyz = wp.zeros(c, dtype=wp.vec3, device=d)
        self.cv_w = wp.zeros(c, dtype=wp.float32, device=d)
        self.label = wp.zeros(c, dtype=wp.int32, device=d)
        self.changed = wp.zeros(1, dtype=wp.int32, device=d)
        self.st_cnt = wp.zeros(c, dtype=wp.float32, device=d)
        self.st_sum = wp.zeros(c, dtype=wp.vec3, device=d)
        self.st_lo = wp.zeros(c, dtype=wp.vec3, device=d)
        self.st_hi = wp.zeros(c, dtype=wp.vec3, device=d)
        self.dense = wp.zeros(c, dtype=wp.int32, device=d)
        self.cl_slot_of_pt = wp.zeros(n, dtype=wp.int32, device=d)
        mc = gpu.MAX_CLUSTERS
        self.out_cnt = wp.zeros(mc, dtype=wp.float32, device=d)
        self.out_cen = wp.zeros(mc, dtype=wp.vec3, device=d)
        self.out_lo = wp.zeros(mc, dtype=wp.vec3, device=d)
        self.out_hi = wp.zeros(mc, dtype=wp.vec3, device=d)
        self.total = wp.zeros(1, dtype=wp.int32, device=d)
        self.grid = wp.HashGrid(128, 128, 128, device=d)
        self.stream = wp.get_stream(d) if d.is_cuda else None  # render thread's stream
        # staging for the per-cluster uploads; numpy views taken once (array.numpy() enters a ScopedStream)
        self._vel_stage = wp.zeros(mc, dtype=wp.vec3, device="cpu", pinned=d.is_cuda)
        self._tid_stage = wp.zeros(mc, dtype=wp.int32, device="cpu", pinned=d.is_cuda)
        self._vel_np = self._vel_stage.numpy()
        self._tid_np = self._tid_stage.numpy()
        self._stage_ev = wp.Event(d) if d.is_cuda else None  # the last upload out of the staging
        self._h_i32 = gpu.HostMirror(d, wp.int32, 1)
        self._h_cnt = gpu.HostMirror(d, wp.float32, mc)
        self._h_cen = gpu.HostMirror(d, wp.vec3, mc)
        self._h_lo = gpu.HostMirror(d, wp.vec3, mc)
        self._h_hi = gpu.HostMirror(d, wp.vec3, mc)
        self.tracker = Tracker()
        self.clusters = []
        self.cc_iters = 0
        self.cc_max_passes = 256
        self.cc_converged = True
        self.ms = {"ground": 0.0, "clusters": 0.0}

    # ---- ground ------------------------------------------------------------------------------

    def ground(self, *, cell: float = 0.5, thick: float = 0.15, min_sup: int = 3, thresh: float = 0.2,
               slope_step: float = 0.3, use_normals: bool = False, nz_min: float = 0.7):
        """Label pipe.gnd / pipe.hag for the visible points.

        cell: grid cell size (m); thick: band above a cell's lowest point that counts as its support;
        min_sup: supporting points a cell needs to be a ground candidate; thresh: height above the
        local ground estimate up to which a point is ground; slope_step: rise allowed per cell of
        distance; use_normals / nz_min: with normals, a point whose |normal z| is below nz_min is the
        foot of a wall rather than ground.
        """
        n = self.pipe.count
        if n == 0:
            return
        t0 = time.perf_counter()
        d = self.device
        inv = 1.0 / max(cell, 0.05)
        self.cell_min.fill_(1.0e30)
        self.cell_sup.zero_()
        xyz = self.pipe.c_xyz
        wp.launch(k_gnd_min, dim=n, inputs=[xyz, inv, self.cell_min], device=d)
        wp.launch(k_gnd_support, dim=n, inputs=[xyz, inv, thick, self.cell_min, self.cell_sup], device=d)
        wp.launch(k_gnd_label, dim=n,
                  inputs=[xyz, self.pipe.nrm, int(use_normals and self.pipe.normals_valid), inv,
                          self.cell_min, self.cell_sup, int(min_sup), thresh, slope_step, nz_min,
                          self.pipe.gnd, self.pipe.hag], device=d)
        self.ms["ground"] = (time.perf_counter() - t0) * 1e3

    # ---- clusters ----------------------------------------------------------------------------

    def cluster(self, voxel: float = 0.1, connect: float = 0.35, min_pts: int = 30, use_ground: bool = True):
        """Connected components of the (non-ground) visible points; fills pipe.cid, returns the cluster list."""
        n = self.pipe.count
        self.clusters = []
        if n == 0:
            return self.clusters
        t0 = time.perf_counter()
        m = self._voxelize(n, voxel, use_ground)
        if m == 0:
            self.pipe.cid.fill_(-1)
            return self.clusters
        self._label(m, connect)
        k = self._compact(n, m, min_pts)
        if k:
            s = self.stream
            self._h_cnt.read(self.out_cnt, k, s, sync=False)
            self._h_cen.read(self.out_cen, k, s, sync=False)
            self._h_lo.read(self.out_lo, k, s, sync=False)
            self._h_hi.read(self.out_hi, k, s, sync=True)
            cnt, cen, lo, hi = self._h_cnt.np[:k], self._h_cen.np[:k], self._h_lo.np[:k], self._h_hi.np[:k]
            self.clusters = [{"cnt": float(cnt[j]), "cen": cen[j].astype(np.float64), "lo": lo[j].astype(np.float64),
                              "hi": hi[j].astype(np.float64)} for j in range(k)]
        self.ms["clusters"] = (time.perf_counter() - t0) * 1e3
        return self.clusters

    def _voxelize(self, n: int, voxel: float, use_ground: bool) -> int:
        """Bin the n (non-ground) points into `voxel` cells, one node per occupied cell; returns the node count."""
        d, c = self.device, self.cl_cap
        self.cl_keys.fill_(-1)
        self.cl_acc.zero_()
        wp.launch(k_cl_insert, dim=n,
                  inputs=[self.pipe.c_xyz, self.pipe.gnd, int(use_ground), 1.0 / max(voxel, 0.01), c - 1,
                          self.cl_keys, self.cl_acc, self.cl_slot_of_pt], device=d)
        wp.launch(gpu.k_nb_flag, dim=c, inputs=[self.cl_keys, self.cl_flag], device=d)
        wp.utils.array_scan(self.cl_flag, self.cl_offs, inclusive=False)
        wp.launch(gpu.k_total, dim=1, inputs=[self.cl_flag, self.cl_offs, c, self.total], device=d)
        wp.launch(gpu.k_nb_compact, dim=c,
                  inputs=[self.cl_flag, self.cl_offs, self.cl_acc, self.cl_vidx, self.cv_xyz, self.cv_w], device=d)
        return int(self._h_i32.read(self.total, 1, self.stream)[0])

    def _label(self, m: int, connect: float):
        """Min-label propagation over the m nodes, linking those within `connect`, until nothing changes."""
        d = self.device
        vox, label = self.cv_xyz[:m], self.label[:m]
        self.grid.build(vox, connect)
        wp.launch(k_cc_init, dim=m, inputs=[label], device=d)
        self.cc_iters = 0
        self.cc_converged = False
        while self.cc_iters < self.cc_max_passes:
            self.changed.zero_()
            for _ in range(4):
                wp.launch(k_cc_step, dim=m, inputs=[self.grid.id, vox, connect, label, self.changed], device=d)
                wp.launch(k_cc_jump, dim=m, inputs=[label], device=d)
                wp.launch(k_cc_jump, dim=m, inputs=[label], device=d)
                self.cc_iters += 1
            if int(self._h_i32.read(self.changed, 1, self.stream)[0]) == 0:
                self.cc_converged = True
                break
        if not self.cc_converged:
            # safety cap hit: at least point every label at a root, so no voxel's stats land on a non-root
            for _ in range(max(1, int(m).bit_length())):
                wp.launch(k_cc_jump, dim=m, inputs=[label], device=d)

    def _compact(self, n: int, m: int, min_pts: int) -> int:
        """Per-root statistics, dense ids for the roots with at least min_pts points, and pipe.cid for the n
        points; returns the number of clusters (at most gpu.MAX_CLUSTERS)."""
        d = self.device
        vox, label = self.cv_xyz[:m], self.label[:m]
        wp.launch(k_cc_reset, dim=m, inputs=[self.st_cnt, self.st_sum, self.st_lo, self.st_hi], device=d)
        wp.launch(k_cc_stats, dim=m,
                  inputs=[label, vox, self.cv_w, self.st_cnt, self.st_sum, self.st_lo, self.st_hi], device=d)
        flag, offs = self.cl_flag[:m], self.cl_offs[:m]
        wp.launch(k_cc_flag, dim=m, inputs=[label, self.st_cnt, float(min_pts), flag], device=d)
        wp.utils.array_scan(flag, offs, inclusive=False)
        wp.launch(gpu.k_total, dim=1, inputs=[flag, offs, m, self.total], device=d)
        wp.launch(k_cc_compact, dim=m,
                  inputs=[flag, offs, self.st_cnt, self.st_sum, self.st_lo, self.st_hi, gpu.MAX_CLUSTERS,
                          self.dense, self.out_cnt, self.out_cen, self.out_lo, self.out_hi], device=d)
        wp.launch(k_cc_gather, dim=n,
                  inputs=[self.cl_slot_of_pt, self.cl_vidx, label, self.dense, self.pipe.cid], device=d)
        return min(int(self._h_i32.read(self.total, 1, self.stream)[0]), gpu.MAX_CLUSTERS)

    # ---- tracks ------------------------------------------------------------------------------

    def track(self, now: float):
        """Associate the last clusters with tracks; upload per-cluster track id and velocity for shading."""
        tid_of, vel_of = self.tracker.update(self.clusters, now)
        k = min(len(self.clusters), gpu.MAX_CLUSTERS)
        if self._stage_ev is not None:
            wp.synchronize_event(self._stage_ev)  # the previous upload may still be reading the staging
        self._vel_np[:k], self._vel_np[k:] = vel_of[:k], 0.0
        self._tid_np[:k], self._tid_np[k:] = tid_of[:k], -1
        wp.copy(self.pipe.cl_vel, self._vel_stage, stream=self.stream)
        wp.copy(self.pipe.cl_tid, self._tid_stage, stream=self.stream)
        if self._stage_ev is not None:
            wp.record_event(self._stage_ev, self.stream)
        return self.tracker.confirmed()
