"""Odometry on the warp_worker thread, exactly as the viewer drives it, under render-stream load (no data).

Checks:
1. accuracy: the static, turn and walk simulations track ground truth in position and rotation as
   well as the inline odometry, registration ran on nearly every frame (a worker that never
   registers would pass static and turn by standing still), the worker returned as many frames as
   the inline run and dropped nothing;
2. snapshot integrity (own-or-drop): every frame snapshot handed back to the render thread holds
   only that frame's points (frame ids k or k-1, times inside the frame), so a snapshot torn by the
   next frame's writes would be caught. All checks happen after the run, never mid-loop, so the
   detector itself cannot serialize the race (the lesson from warp_worker's ownership tests);
3. decoupling: render-loop time per iteration with the worker vs the old inline odometry. Gated on
   the walk, where inline registration costs most (inline p99 ~18-24 ms, worker ~3-4 ms): the worker's
   p99 must be under half the inline one. The turn is only printed: its inline cost is small enough
   (p99 7-12 ms) that OS scheduling noise decides the comparison.

A negative control runs the same loop with the worker handing back BORROWED views of its frame
buffers instead of owned snapshots, and must show torn snapshots: if it does not, the detector is
broken, not the handoff safe.

LIVOX_TEST_SPEEDUP: simulated seconds per wall second (default 2).
"""

import os
import sys
import time

import common
import numpy as np
import warp as wp
from livox_warp import gpu
from livox_warp.odom import Odometry
from livox_warp.slam_worker import FrameResult, OdomWorker
from livox_warp.sources import SimSource

FRAME_DT = 0.1
MAP_VOXEL = 0.05
SPEEDUP = float(os.environ.get("LIVOX_TEST_SPEEDUP", "2.0"))
# (motion, seconds, position rmse limit cm, rotation rmse limit deg)
CASES = (("static", 6.0, 2.0, 0.1), ("turn", 15.0, 2.0, 0.3), ("walk", 20.0, 15.0, 2.0))


@wp.kernel
def render_load(a: wp.array(dtype=float), iters: int):
    """Stand-in for the viewer's per-frame GPU work (shade, neighbours, draw) on the render stream."""
    i = wp.tid()
    x = a[i]
    for k in range(iters):
        x = wp.sin(x) * 0.5 + 1.0
    a[i] = x


def pose_err(traj, sim):
    T0 = sim.pose_at(traj[0][0])
    et, er = [], []
    for tf, T in traj:
        gt = np.linalg.inv(T0) @ sim.pose_at(tf)
        et.append(np.linalg.norm(T[:3, 3] - gt[:3, 3]))
        er.append(common.rot_angle(T[:3, :3].T @ gt[:3, :3]))
    return np.sqrt(np.mean(np.square(et))), np.sqrt(np.mean(np.square(er)))


class Borrowed:
    """What the worker would hand over without own(): a view of a buffer it keeps rewriting."""

    def __init__(self, a):
        self.array = a

    def wait(self, stream):
        pass

    def release(self):
        pass


def borrowing_on_frame(worker):
    """The negative control's frame hook. It deliberately reaches into the worker's implementation
    (worker._gen, worker.q_out and the _on_frame it replaces) to hand back views of the frame buffers
    instead of owned snapshots: there is no supported way to do that, and there must not be."""

    def on_frame(k, T, fx, fa, ft, ff, m):
        o = worker.odom
        worker.q_out.put(FrameResult(worker._gen, k, np.array(T), np.array(o.xi), m,
                                     [Borrowed(a) for a in (fx, fa, ft, ff)], dict(o.stats)))
    return on_frame


def run(dev, motion, seconds, threaded, load_iters=400, borrow=False):
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 20, device=dev)
    load = wp.zeros(1 << 18, dtype=float, device=dev)
    sim = SimSource(device=dev, motion=motion)
    traj, loop_ms, torn, checked, stats = [], [], [], 0, []
    held = []  # (k, m, snapshot arrays) checked after the run

    def view(now):
        return common.view(now, pipe, voxel=MAP_VOXEL, inv_frame_dt=1.0 / FRAME_DT)

    if threaded:
        worker = OdomWorker(pipe, dev)
        if borrow:
            worker._on_frame = borrowing_on_frame(worker)
        worker.configure(frame_dt=FRAME_DT)
        worker.reset(np.eye(4))
        odom = worker.odom  # its tuning (warm-up, min_corr) judges the registration evidence below
    else:
        odom = Odometry(pipe, dev)
        odom.frame_dt = FRAME_DT
        odom.reset(np.eye(4))

        def on_frame(k, T, fx, fa, ft, ff, m):
            pipe.map_insert(fx, fa, ft, ff, m, view(sim.t), MAP_VOXEL)
            traj.append(((k + 0.5) * FRAME_DT, T.copy()))
            stats.append(dict(odom.stats))

    loop_dt = 1.0 / 60.0
    per_loop = int(sim.rate * loop_dt * SPEEDUP)
    n_loops = int(seconds * sim.rate / per_loop)
    t_next = time.perf_counter()
    for _ in range(n_loops):
        t0 = time.perf_counter()
        xyz, attr, t, n = sim.step(per_loop)
        sx, sa, st, sf = pipe.stage(xyz, attr, t, n, 1.0 / FRAME_DT)
        pipe.ingest_ring(sx, sa, st, sf, n)
        if threaded:
            worker.submit(sx, sa, st, n, sim.last_t0, sim.last_t1)
            for r in worker.poll():
                traj.append(((r.k + 0.5) * FRAME_DT, r.T))
                stats.append(r.stats)
                if r.snaps:
                    fx, fa, ft, ff = r.arrays()
                    pipe.map_insert(fx, fa, ft, ff, r.m, view(sim.t), MAP_VOXEL)
                    # keep an independent device copy of the snapshot (on the render stream, after the
                    # consumer-side wait poll() recorded) to verify after the run, without syncing now
                    held.append((r.k, r.m, [wp.clone(a) for a in (ff, ft)]))
        else:
            odom.push(sx, sa, st, n, sim.last_t0, sim.last_t1, on_frame)
        wp.launch(render_load, dim=load.shape[0], inputs=[load, load_iters])
        pipe.build(view(sim.t), map_on=False, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))
        if threaded:
            worker.release_consumed()
        loop_ms.append((time.perf_counter() - t0) * 1e3)
        t_next += loop_dt
        time.sleep(max(0.0, t_next - time.perf_counter()))
    if threaded:
        t_end = time.perf_counter() + 5.0
        while worker.status()["pending"] and time.perf_counter() < t_end:
            time.sleep(0.01)
        for r in worker.poll():
            traj.append(((r.k + 0.5) * FRAME_DT, r.T))
            stats.append(r.stats)
        worker.release_consumed()
        status = worker.status()
        worker.close()
    wp.synchronize_device(dev)
    mask = gpu.POSE_SLOTS - 1
    for k, m, (ff, ft) in held:
        fr = ff.numpy()[:m]
        tt = ft.numpy()[:m]
        checked += 1
        ok_ids = np.isin(fr, [k & mask, (k - 1) & mask])
        ok_t = (tt >= (k - 1) * FRAME_DT - 1e-4) & (tt < (k + 1) * FRAME_DT + 1e-4)
        if not (ok_ids.all() and ok_t.all()):
            torn.append((k, int((~ok_ids).sum()), int((~ok_t).sum())))
    rmse_t, rmse_r = pose_err(traj, sim)
    lm = np.array(loop_ms)
    # after the warm-up: iterations ran and found enough correspondences
    post = stats[odom.warmup + 2:]
    reg = float(np.mean([s["iters"] > 0 and s["corr"] >= odom.min_corr for s in post])) if post else 0.0
    out = {"frames": len(traj), "rmse_cm": rmse_t * 100, "rot_deg": rmse_r, "reg": reg,
           "loop_p50": np.percentile(lm, 50), "loop_p99": np.percentile(lm, 99), "loop_max": lm.max(),
           "checked": checked, "torn": torn}
    if threaded:
        out["drops"] = status["drops"]
        out["busy_ms"] = status["busy_ms"]
    return out


def main():
    dev = common.device()
    ok = True
    for motion, secs, tol, tol_r in CASES:
        res = {}
        for threaded in (False, True):
            r = res[threaded] = run(dev, motion, secs, threaded)
            tag = "worker" if threaded else "inline"
            extra = ""
            if threaded:
                extra = (f"; {r['checked']} snapshots checked, {len(r['torn'])} torn; drops {r['drops']}; "
                         f"worker {r['busy_ms']:.1f} ms/batch")
            print(f"{motion:6s} {tag:6s}: {r['frames']} frames, rmse {r['rmse_cm']:.1f} cm, "
                  f"rot {r['rot_deg']:.2f} deg, {r['reg']:.0%} registered; render loop p50 {r['loop_p50']:.2f} ms "
                  f"p99 {r['loop_p99']:.2f} ms max {r['loop_max']:.2f} ms{extra}")
            ok &= r["rmse_cm"] < tol and r["rot_deg"] < tol_r and r["reg"] > 0.9
            if threaded:
                ok &= not r["torn"] and r["checked"] > 0 and not any(r["drops"].values())
                if r["torn"]:
                    print("   torn snapshots (frame, bad ids, bad times):", r["torn"][:10])
        inline, worker = res[False], res[True]
        ok &= worker["frames"] >= 0.97 * inline["frames"]
        if motion != "static":
            decoupled = worker["loop_p99"] < 0.5 * inline["loop_p99"]
            verdict = ("OK" if decoupled else "NOT decoupled") if motion == "walk" else "not gated"
            print(f"         decoupling: render loop p99 {worker['loop_p99']:.2f} ms with the worker vs "
                  f"{inline['loop_p99']:.2f} ms inline -> {verdict}")
            if motion == "walk":
                ok &= decoupled
    r = run(dev, "turn", 6.0, True, borrow=True)
    print(f"negative control (borrowed views, no own()): {len(r['torn'])} of {r['checked']} snapshots torn "
          f"(must be > 0, or the detector cannot see the race)")
    ok &= len(r["torn"]) > 0
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
