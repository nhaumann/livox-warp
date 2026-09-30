"""LiDAR-only odometry against the simulated Mid-40's ground-truth trajectory (no wall clock, no data).

Feeds the walking / turning / static simulations through Pipeline.stage -> Odometry.push exactly as the
app does, then compares every frame's registered pose with the sensor's true pose at the frame midpoint,
both expressed relative to the first frame. Position and rotation are both gated for every motion, the
ground truth must actually move (the turn spans a real angle, the walk real distance), and registration
must have run: a tracker that never registers would otherwise pass the static and turn cases by standing
still.

LIVOX_SIM_SCENE picks the simulated scene: box (default, the bare room) or office (the same room furnished
with desks, chairs, shelving, doorways, window recesses, columns and ceiling fixtures). LIVOX_TEST_SCALE
shortens every run (for the CPU backend).
"""

import os
import sys
import time

import common
import numpy as np
import warp as wp
from livox_warp import gpu
from livox_warp.odom import Odometry
from livox_warp.sources import SimSource

SCENE = os.environ.get("LIVOX_SIM_SCENE", "box")
MAP_VOXEL = 0.05


def run(dev, motion: str, seconds: float, batch: int = 2000, frame_dt: float = 0.1):
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 20, device=dev)
    odom = Odometry(pipe, dev)
    odom.frame_dt = frame_dt
    odom.reset(np.eye(4))
    sim = SimSource(device=dev, motion=motion, scene=SCENE)
    inserted = []
    stats = []

    def on_frame(k, T, fx, fa, ft, ff, m):
        pipe.map_insert(fx, fa, ft, ff, m, common.view(sim.t, pipe, voxel=MAP_VOXEL), MAP_VOXEL)
        inserted.append(m)
        stats.append(dict(odom.stats))

    t0 = time.perf_counter()
    n_batches = int(seconds * sim.rate / batch)
    for _ in range(n_batches):
        xyz, attr, t, n = sim.step(batch)
        sx, sa, st, sf = pipe.stage(xyz, attr, t, n, 1.0 / frame_dt)
        odom.push(sx, sa, st, n, sim.last_t0, sim.last_t1, on_frame)
        pipe.ingest_ring(sx, sa, st, sf, n)
    wp.synchronize()
    wall = time.perf_counter() - t0

    traj = odom.traj
    assert len(traj) >= 2, "no frames registered"
    T_first_gt = sim.pose_at(traj[0][1])
    err_t, err_r, span_t, span_r = [], [], [], []
    for k, tf, T in traj:
        gt = np.linalg.inv(T_first_gt) @ sim.pose_at(tf)
        err_t.append(np.linalg.norm(T[:3, 3] - gt[:3, 3]))
        err_r.append(common.rot_angle(T[:3, :3].T @ gt[:3, :3]))
        span_t.append(np.linalg.norm(gt[:3, 3]))
        span_r.append(common.rot_angle(gt[:3, :3]))
    err_t, err_r = np.array(err_t), np.array(err_r)
    # registration evidence after the warm-up: iterations ran and found enough correspondences
    post = stats[odom.warmup + 2:]
    reg = np.mean([s_["iters"] > 0 and s_["corr"] >= odom.min_corr for s_ in post]) if post else 0.0
    dist = 0.0
    for a, b in zip(traj[:-1], traj[1:]):
        dist += np.linalg.norm(sim.pose_at(b[1])[:3, 3] - sim.pose_at(a[1])[:3, 3])
    st = odom.stats
    print(f"{motion:7s}: {len(traj)} frames over {seconds:.0f} s ({dist:.1f} m travelled), "
          f"{wall * 1e3 / len(traj):.1f} ms wall per frame, mean {np.mean(inserted):.0f} pts/frame")
    print(f"         position error rmse {np.sqrt(np.mean(err_t**2)) * 100:.1f} cm, max {err_t.max() * 100:.1f} cm, "
          f"final {err_t[-1] * 100:.1f} cm; rotation rmse {np.sqrt(np.mean(err_r**2)):.2f} deg, "
          f"max {err_r.max():.2f} deg")
    print(f"         last frame: {st['iters']} iters, {st['corr']}/{st['ds']} correspondences, "
          f"rms {st['rms'] * 100:.1f} cm, conditioning {st['cond']:.1e}, {st['map_voxels']:,} map voxels, "
          f"{st['skipped']} skipped")
    print(f"         ground truth spans {max(span_t) * 100:.0f} cm and {max(span_r):.0f} deg; "
          f"{reg:.0%} of frames after warm-up registered")
    return dict(rmse=np.sqrt(np.mean(err_t**2)), final=err_t[-1], dist=dist, rot_rmse=np.sqrt(np.mean(err_r**2)),
                rot_max=err_r.max(), span_t=max(span_t), span_r=max(span_r), reg=reg)


def main():
    dev = common.device()
    print(f"scene: {SCENE}")
    ok = True
    scale = float(os.environ.get("LIVOX_TEST_SCALE", "1.0"))  # shorter runs for the CPU backend
    r = run(dev, "static", 6.0 * scale)
    ok &= r["reg"] > 0.9 and r["rmse"] < 0.02 and r["rot_max"] < 0.2
    r = run(dev, "turn", 15.0 * scale)
    ok &= r["reg"] > 0.9 and r["span_r"] > 20.0 * min(scale, 1.0) and r["rmse"] < 0.10 and r["rot_rmse"] < 0.5
    r = run(dev, "walk", 20.0 * scale)
    ok &= r["reg"] > 0.9 and r["span_t"] > 1.0 * min(scale, 1.0) and r["rmse"] < 0.15
    ok &= r["final"] < 0.02 * max(r["dist"], 1.0) + 0.15 and r["rot_rmse"] < 2.0
    print("OK" if ok else "FAIL: odometry error above tolerance")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
