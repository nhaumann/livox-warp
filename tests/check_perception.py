"""Ground segmentation, clustering, tracking, motion labels and free-space carving on the simulated Mid-40.

The static simulation looks down a room: the floor at SIM_FLOOR_Z, a crate (SIM_CRATE_LO..SIM_CRATE_HI), a
shelf, pillars, a table and a person-sized sphere walking left-right (SimSource.person_at). Ground must be
the floor only, the crate top must read its real height above the floor, the sphere must come out as one
tracked cluster with the right speed, its points must be labelled moving, and after it has walked on the
map voxels it left behind must be carved, while the floor (seen at grazing incidence) stays. Also checked:
the tracker at render rate reports static objects as static, carving with every other return traced
behaves the same, and connected components converge on a 60 m wall.

With a static recording, its saved pose and the prior map present (tests/README.md), carving is also scored
on real data: everything in such a recording is static, so map voxels that lie on the prior scan (surfaces
seen at grazing incidence among them) must not be carved.
"""

import sys

import common
import numpy as np
import warp as wp
from livox_warp import gpu, prior_map
from livox_warp.perception import Perception
from livox_warp.sources import SIM_CRATE_HI, SIM_CRATE_LO, SIM_FLOOR_Z, SimSource
from scipy.spatial import cKDTree

VOXEL = 0.03
PERSIST = 0.3
FLOOR_TOP = SIM_FLOOR_Z + 0.1  # points below this lie on the floor
INSET = 0.1  # kept clear of an object's edges when selecting the points on one of its faces


def view(pipe, now, persist=PERSIST, dyn=True, carve=True, settle=2.5, voxel=VOXEL):
    """The viewer's defaults for motion labels and carving; the simulated noise tags are dropped.
    settle = 2.5 s: the sphere re-crosses its own path within about 2 s."""
    return common.view(now, pipe, persist=persist, voxel=voxel, noise_mask=0b0001, dyn_on=int(dyn),
                       dyn_settle=settle, carve_on=int(carve), carve_ratio=2.0, carve_min=3, carve_stale=1.0)


def feed(pipe, sim, seconds, carve=True, every=1, step=0.1):
    for _ in range(int(round(seconds / step))):
        xyz, attr, t, n = sim.step(int(sim.rate * step))
        sx, sa, st, sf = pipe.stage(xyz, attr, t, n, 0.0)
        pipe.ingest_ring(sx, sa, st, sf, n)
        pipe.map_insert(sx, sa, st, sf, n, view(pipe, sim.t), VOXEL)
        if carve:
            pipe.carve(sx, sa, sf, n, VOXEL, every=every)


def build_live(pipe, now):
    return pipe.build(view(pipe, now), map_on=False, min_count=1, neighbors="normals", radius=0.15,
                      sensor=(0, 0, 0))


def build_map(pipe, now, carve, voxel=VOXEL):
    """The map as the viewer shows it, with settle = 0 so that 'moving' comes from carving alone, not from
    the rule that voxels which appeared in the last few seconds are new (at 3 cm the rosette keeps filling
    in floor voxels)."""
    v = view(pipe, now, persist=0.0, carve=carve, settle=0.0, voxel=voxel)
    return pipe.build(v, map_on=True, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))


def perceive(perc, now):
    perc.ground(cell=0.5, thick=0.15, min_sup=3, thresh=0.2, slope_step=0.3, use_normals=True)
    perc.cluster(voxel=0.1, connect=0.35, min_pts=40, use_ground=True)
    return perc.track(now)


def person_speed(sim, t, h=1e-3):
    """The sphere's true speed at t, from its path."""
    return float(np.linalg.norm(sim.person_at(t + h) - sim.person_at(t - h)) / (2 * h))


def window_centre(sim, t, n=64):
    """Mean true sphere centre over the persist window ending at t: what a cluster centroid follows."""
    return np.mean([sim.person_at(t - PERSIST * (k + 0.5) / n) for k in range(n)], axis=0)


def far_wall_x(m, depth):
    """Where the wall facing the sensor begins: the deepest `depth` metres of the set, judged from a high
    percentile of x so that the odd noisy return past the wall does not move it."""
    return float(np.percentile(m[:, 0], 99.0)) - depth


def carve_check(pipe, sim, label):
    """Ghosts where the sphere started are hidden; the floor (grazing incidence) and the far wall are not.

    A carved ghost needs rays through it, and at 12 m the Mid-40 sends about 1.6 rays per second through
    each 3 cm voxel's surface patch, so after ~3 s about half to three quarters are gone."""
    p0 = sim.person_at(0.1)
    cnt_all = build_map(pipe, sim.t, carve=False)
    m_all = pipe.c_xyz.numpy()[:cnt_all]
    cnt_c = build_map(pipe, sim.t, carve=True)
    m_c = pipe.c_xyz.numpy()[:cnt_c]
    dyn_c = pipe.c_dyn.numpy()[:cnt_c].astype(bool)
    x_far = far_wall_x(m_all, 0.3)

    def ghost(m):
        return np.linalg.norm(m - p0, axis=1) < 0.4

    def wall(m):
        return (m[:, 0] > x_far) & (np.abs(m[:, 1]) < 2.0)

    def floor(m):
        # the floor 2-11 m ahead, where the rays meet it at grazing incidence
        return (m[:, 2] < FLOOR_TOP) & (m[:, 0] > 2.0) & (m[:, 0] < 11.0)

    g0, g1 = ghost(m_all).sum(), ghost(m_c).sum()
    w0, w1 = wall(m_all).sum(), wall(m_c).sum()
    f0, f1 = floor(m_all).sum(), floor(m_c).sum()
    f_moving = dyn_c[floor(m_c)].mean() if f1 else 1.0
    print(f"carving ({label}): map {cnt_all:,} voxels, {cnt_c:,} after hiding ghosts; where the sphere started "
          f"{g0} -> {g1}; far wall kept {w1 / max(w0, 1):.1%}; floor at grazing incidence kept "
          f"{f1 / max(f0, 1):.1%}, labelled moving {f_moving:.1%} ({pipe.ms['carve']:.2f} ms/batch)")
    return bool(g0 > 50 and g1 < 0.5 * g0 and w1 / max(w0, 1) > 0.97 and f0 > 1000 and f1 / max(f0, 1) > 0.99
                and f_moving < 0.02)


def carve_real(dev, rec, T, prior):
    """Carving on a static recording placed in the prior map by its saved pose T: of the map voxels within
    4 cm of the scan, hardly any may be labelled moving or hidden, at either voxel size."""
    x, a, t = common.load_recording(rec)
    P = prior_map.load(str(prior))["xyz"]
    tree = cKDTree(P[np.linalg.norm(P - T[:3, 3], axis=1) < 30.0])
    ok = True
    for rv in (0.02, 0.05):
        pr = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 21, device=dev)
        for i in range(0, len(t), 5000):
            xs, as_, ts = x[i:i + 5000], a[i:i + 5000], t[i:i + 5000]
            n = len(ts)
            sx, sa, st, sf = pr.stage(xs, as_, ts, n, 0.0)
            pr.map_insert(sx, sa, st, sf, n, view(pr, float(ts.max()), voxel=rv), rv)
            pr.carve(sx, sa, sf, n, rv, every=2)
        now_r = float(t.max())
        n0 = build_map(pr, now_r, carve=False, voxel=rv)
        m0 = pr.c_xyz.numpy()[:n0]
        d0 = pr.c_dyn.numpy()[:n0].astype(bool)
        n1 = build_map(pr, now_r, carve=True, voxel=rv)
        kept = {tuple(q) for q in np.round(pr.c_xyz.numpy()[:n1], 5)}
        hidden = np.array([tuple(q) not in kept for q in np.round(m0, 5)])
        dist, _ = tree.query(m0 @ T[:3, :3].T + T[:3, 3], distance_upper_bound=1.0)
        surf, off = dist < 0.04, dist > 0.15
        print(f"carving on {rec.name} ({rv * 100:.0f} cm voxels): {n0:,} voxels, {surf.mean():.0%} on the prior "
              f"scan; of those {d0[surf].mean():.1%} labelled moving, {hidden[surf].mean():.1%} hidden. "
              f"Off the scan (edge noise, new objects): {hidden[off].mean():.0%} hidden")
        ok &= bool(surf.mean() > 0.5 and d0[surf].mean() < 0.03 and hidden[surf].mean() < 0.02)
        del pr
    return ok


def main():
    dev = common.device()
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 21, device=dev)
    perc = Perception(pipe, dev)
    sim = SimSource(device=dev, motion="static")
    out_pos = wp.zeros(pipe.work_cap, dtype=wp.vec3, device=dev)
    out_col = wp.zeros(pipe.work_cap, dtype=wp.uint32, device=dev)
    out_nrm = wp.zeros(pipe.work_cap, dtype=wp.vec3, device=dev)

    ok = True
    feed(pipe, sim, 3.0)  # 3 s: enough history for voxels to settle
    now = sim.t
    cnt = build_live(pipe, now)
    xyz = pipe.c_xyz.numpy()[:cnt]

    # --- ground -------------------------------------------------------------------------------
    perc.ground(cell=0.5, thick=0.15, min_sup=3, thresh=0.2, slope_step=0.3, use_normals=True)  # module compile
    perc.ground(cell=0.5, thick=0.15, min_sup=3, thresh=0.2, slope_step=0.3, use_normals=True)
    gnd = pipe.gnd.numpy()[:cnt].astype(bool)
    hag = pipe.hag.numpy()[:cnt]
    floor = xyz[:, 2] < FLOOR_TOP
    lo, hi = np.asarray(SIM_CRATE_LO, dtype=float), np.asarray(SIM_CRATE_HI, dtype=float)
    over_crate = np.all(xyz[:, :2] > lo[:2] + INSET, axis=1) & np.all(xyz[:, :2] < hi[:2] - INSET, axis=1)
    crate_top = over_crate & (xyz[:, 2] > hi[2] - INSET)
    crate_height = hi[2] - SIM_FLOOR_Z
    walls = (xyz[:, 2] > SIM_FLOOR_Z + 0.4) & ~crate_top
    hag_top = float(np.median(hag[crate_top]))
    print(f"ground: {gnd.sum():,} of {cnt:,} points; floor recall {gnd[floor].mean():.1%}, "
          f"crate-top labelled ground {gnd[crate_top].mean():.1%}, walls/objects labelled ground "
          f"{gnd[walls].mean():.1%}, {perc.ms['ground']:.2f} ms")
    print(f"        height above ground on the crate top: median {hag_top:.2f} m (true {crate_height:.2f}); "
          f"on the floor: median {np.median(np.abs(hag[floor])):.3f} m")
    ok &= gnd[floor].mean() > 0.95 and gnd[crate_top].mean() < 0.05 and gnd[walls].mean() < 0.02
    ok &= abs(hag_top - crate_height) < 0.1 and np.median(np.abs(hag[floor])) < 0.05

    # --- clusters and tracking ----------------------------------------------------------------
    clusters = perc.cluster(voxel=0.1, connect=0.35, min_pts=40, use_ground=True)
    perc.track(now)
    times = [now]
    cid = pipe.cid.numpy()[:cnt]
    near = np.linalg.norm(xyz - sim.person_at(now - 0.05), axis=1) < 0.6
    ids, counts = np.unique(cid[near & (cid >= 0)], return_counts=True)
    in_ids = dict(zip(ids.tolist(), counts.tolist()))
    print(f"clusters: {len(clusters)} found in {perc.ms['clusters']:.2f} ms ({perc.cc_iters} label passes, "
          f"converged {perc.cc_converged}); sphere points fall in cluster ids {in_ids}")
    ok &= len(clusters) >= 4 and len(ids) >= 1 and counts.max() / max(near.sum(), 1) > 0.8 and perc.cc_converged

    # a few more sensor frames so the tracker has a velocity
    for _ in range(5):
        feed(pipe, sim, 0.1)
        now = sim.t
        cnt = build_live(pipe, now)
        tracks = perceive(perc, now)
        times.append(now)
    xyz = pipe.c_xyz.numpy()[:cnt]
    cid = pipe.cid.numpy()[:cnt]
    near = np.linalg.norm(xyz - sim.person_at(now - 0.05), axis=1) < 0.6
    ids = np.unique(cid[near & (cid >= 0)])
    vel = pipe.cl_vel.numpy()
    speeds = [float(np.linalg.norm(vel[i])) for i in ids]

    # Independent reference for the tracker's documented estimator: the least-squares slope through the
    # centroid samples (>= 0.1 s apart, the last 0.8 s), here through the true window centres.
    assert (perc.tracker.baseline, perc.tracker.window) == (0.1, 0.8), "update the reference below"
    ts = [times[0]]
    for t in times[1:]:
        if t - ts[-1] >= 0.1 - 1e-3:
            ts.append(t)
    ts = np.array([t for t in ts if t >= ts[-1] - 0.8 - 1e-6])
    cs = np.array([window_centre(sim, t) for t in ts])
    tcen = ts - ts.mean()
    v_ref = (tcen[:, None] * (cs - cs.mean(axis=0))).sum(axis=0) / (tcen ** 2).sum()
    s_ref = float(np.linalg.norm(v_ref))
    s_now = person_speed(sim, now)
    print(f"tracking: {len(tracks)} confirmed tracks; sphere speed estimate {[round(s, 3) for s in speeds]} m/s vs "
          f"{s_ref:.3f} m/s expected from the true motion (instantaneous {s_now:.2f} m/s)")
    ok &= len(speeds) >= 1 and s_ref > 0.3 and abs(max(speeds) - s_ref) < 0.25 * s_ref

    # --- shading with labels (exercise every mode) --------------------------------------------
    for mode in (gpu.MODE_GROUND, gpu.MODE_CLUSTERS, gpu.MODE_SPEED, gpu.MODE_MOTION):
        s = common.shade(now, mode, hi=2.0, has_normals=1, has_gnd=1, has_cid=1, has_dyn=1)
        pipe.shade(s, out_pos, out_col, out_nrm)
    wp.synchronize()
    dyn = pipe.c_dyn.numpy()[:cnt].astype(bool)
    wall_pts = (xyz[:, 0] > far_wall_x(xyz, 0.5)) & (np.abs(xyz[:, 1]) > 3.0)  # off the sphere's lane
    print(f"motion: sphere points labelled moving {dyn[near].mean():.1%}, "
          f"far-wall points labelled moving {dyn[wall_pts].mean():.1%}")
    ok &= dyn[near].mean() > 0.7 and dyn[wall_pts].mean() < 0.1

    # --- free-space carving -------------------------------------------------------------------
    # the sphere has walked away from where it started, so those map voxels should be ghosts by now
    ok &= carve_check(pipe, sim, "every return")

    # --- tracker at render rate ---------------------------------------------------------------
    # The viewer clusters and tracks on every render frame, with only ~7 ms of new sensor data each time.
    # Static objects must still read as static (no velocity arrows, which start at 0.15 m/s).
    static_speeds, sphere_speeds = [], []
    for i in range(int(1.5 * 144)):
        feed(pipe, sim, 1.0 / 144, step=1.0 / 144, every=2)
        now = sim.t
        build_live(pipe, now)
        tracks = perceive(perc, now)
        if i < 72:  # let the tracks settle for 0.5 s
            continue
        for _, tr in tracks:
            d = np.linalg.norm(tr["pos"][:2] - sim.person_at(now)[:2])
            (sphere_speeds if d < 0.5 else static_speeds).append(float(np.linalg.norm(tr["vel"])))
    st = np.array(static_speeds)
    arrows = float((st > 0.15).mean()) if len(st) else 1.0
    print(f"tracker at 144 Hz: {len(st)} static track-frames, speed median {np.median(st):.3f} m/s, "
          f"p90 {np.percentile(st, 90):.3f}, max {st.max():.3f}; {arrows:.1%} would get a velocity arrow; "
          f"sphere median {np.median(sphere_speeds) if sphere_speeds else float('nan'):.2f} m/s")
    ok &= len(st) > 300 and np.percentile(st, 90) < 0.1 and arrows < 0.03 and len(sphere_speeds) > 50
    ok &= np.median(sphere_speeds) > 0.3

    # --- carving with every other return traced (the viewer default) --------------------------
    pipe2 = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 21, device=dev)
    sim2 = SimSource(device=dev, motion="static")
    feed(pipe2, sim2, 3.5, every=2)
    ok &= carve_check(pipe2, sim2, "every 2nd return")
    del pipe2

    # --- connected components converge on long structures -------------------------------------
    wall_grid = np.meshgrid(np.arange(-30, 30, 0.05), np.arange(-1.5, 1.5, 0.05))
    for label, pts, vox, conn in (
        ("60 x 3 m wall", np.stack(wall_grid, -1).reshape(-1, 2), 0.1, 0.35),
        ("60 m rail", np.stack([np.arange(-30, 30, 0.02), np.zeros(3000)], -1), 0.05, 0.1),
    ):
        p3 = np.zeros((len(pts), 3), np.float32)
        p3[:, 0], p3[:, 1], p3[:, 2] = 20.0, pts[:, 0], pts[:, 1]
        pipe.load_points(p3)
        cl = perc.cluster(voxel=vox, connect=conn, min_pts=10, use_ground=False)
        print(f"components on a {label}: {len(cl)} cluster(s) after {perc.cc_iters} label passes, "
              f"converged {perc.cc_converged}")
        ok &= len(cl) == 1 and perc.cc_converged

    # --- carving on real data, scored against the prior scan ----------------------------------
    poses = common.saved_poses()
    rec, prior = common.static_recording(poses), common.prior_map()
    if rec is not None and prior is not None and common.pose_key(rec) in poses:
        ok &= carve_real(dev, rec, poses[common.pose_key(rec)], prior)
    else:
        print("carving on real data: skipped (needs a static recording, its saved pose and the prior map)")

    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
