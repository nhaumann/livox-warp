"""Localisation in a prior map on real recordings, driven exactly as the viewer does.

Needs the prior map and a recording of the scanner standing still inside it (with its saved pose, if
maps/recording_poses.npz has one); step 3 also needs a handheld walk that starts at rest inside it and the
walk's frame-by-frame reference (maps/walk_reference.npz). See tests/README.md; skips what is missing.

1. Global localisation of the static recording from its first 1.5 s: within 2 cm / 0.3 deg of the saved
   pose (the exhaustive search of python -m livox_warp.localize), in a few seconds.
2. Static session (no odometry): PriorSession localises from the live points, then tracks; the pose stays
   put, and the Changes distances of the live window split into matching (< 3 cm), near (3-10 cm) and new
   (> 10 cm: whatever was added since the scan).
3. The walk with odometry on its worker thread and the session correcting its drift: the poses in the map
   are compared with the reference where it is solid (over 80% of its points within 3 cm of the scan, no
   implausible step) and before its trusted_until time, past which the reference itself is unreliable.
   The session must not lose track before that time.
"""

import sys
import time

import common
import numpy as np
from livox_warp import gpu, localize, prior_map
from livox_warp.prior_session import PriorSession
from livox_warp.slam_worker import OdomWorker
from livox_warp.sources import clean_returns

FRAME_DT = 0.1


def quiet(*_):
    """A log sink for the session and the localizer."""


def step_localise(dev, map_path, x, a, t, T_saved):
    pm = prior_map.load(str(map_path))
    L = localize.Localizer(pm["xyz"], pm["normal"], pm.get("planarity"), device=dev)
    first = (t - t.min() < 1.5) & clean_returns(x, a)
    L.search(x[first][:20000], log=quiet)  # warm-up: compile, open3d
    t0 = time.perf_counter()
    T_loc, info = L.search(x[first], log=quiet)
    secs = time.perf_counter() - t0
    msg = f"1. localisation from the first 1.5 s: {secs:.1f} s, inliers {info['inliers']:.2f}, unique {info['unique']}"
    ok = bool(info["unique"]) and secs < 5.0
    if T_saved is not None:
        dt, dr = common.pose_diff(T_saved, T_loc)
        msg += f"; vs the saved pose {dt * 100:.1f} cm / {dr:.2f} deg"
        ok &= dt < 0.02 and dr < 0.3
    print(msg)
    return ok


def step_static_session(dev, map_path, x, a, t, T_saved):
    session = PriorSession(str(map_path), dev, log=quiet)
    common.wait_ready(session)
    Ts, first = [], []

    def record(now, sess):
        if sess.T_MW is not None:
            if not first:
                first.append(now - float(t.min()))
            Ts.append(sess.T_MW.copy())

    common.drive(session, x, a, t, on_batch=record)
    if not Ts:
        print("2. static session: never localised")
        session.close()
        return False
    t_first = first[0]
    drift = max(common.pose_diff(Ts[0], T)[0] for T in Ts) * 100
    rot = max(common.pose_diff(Ts[0], T)[1] for T in Ts)
    # Changes: the live window of the last second, distances to the prior
    pipe = gpu.Pipeline(ring_capacity=1 << 20, map_capacity=1 << 18, device=dev)
    last = t > t.max() - 1.0
    n = int(last.sum())
    sx, sa, st, sf = pipe.stage(x[last], a[last], t[last], n, 0.0)
    pipe.ingest_ring(sx, sa, st, sf, n)
    v = common.view(float(t.max()), pipe, voxel=0.02, noise_mask=0b0001)
    cnt = pipe.build(v, map_on=False, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))
    session.grid.distances(pipe.c_xyz, cnt, session.T_MW, pipe.chg, None)
    d = pipe.chg.numpy()[:cnt]
    g, am, r = np.mean(d < 0.03), np.mean((d >= 0.03) & (d < 0.10)), np.mean(d >= 0.10)
    print(f"2. static session: localised after {t_first:.1f} s of data, then {session.corrections} corrections "
          f"({session.rejected} rejected), pose moved at most {drift:.2f} cm / {rot:.3f} deg, "
          f"last fit {session.fit:.2f}")
    print(f"   Changes on the last second ({cnt:,} points): {g:.0%} match the scan, {am:.0%} within 10 cm, {r:.0%} new")
    ok = t_first < 4.0 and drift < 2.0 and rot < 0.3 and g > 0.7 and 0.02 < r < 0.2
    if T_saved is not None:
        dt, dr = common.pose_diff(T_saved, session.T_MW)
        print(f"   final pose vs the saved pose: {dt * 100:.1f} cm / {dr:.2f} deg")
        ok &= dt < 0.03 and dr < 0.5
    session.close()
    return bool(ok)


def step_walk(dev, map_path, walk, ref_path):
    x, a, t = common.load_recording(walk)
    ref, trusted, trusted_until = common.load_walk_reference(ref_path, FRAME_DT)
    k0 = int(np.floor(t.min() / FRAME_DT))
    session = PriorSession(str(map_path), dev, log=quiet)
    common.wait_ready(session)
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 18, device=dev)
    worker = OdomWorker(pipe, dev)
    worker.configure(frame_dt=FRAME_DT)
    worker.reset(np.eye(4))
    rows, states = [], []

    def on_frame(r, sess):
        if sess.T_MW is not None:
            rows.append((r.k, sess.T_MW @ r.T))
        states.append(((r.k - k0) * FRAME_DT, sess.state))

    common.drive(session, x, a, t, worker, pipe, on_frame=on_frame, frame_dt=FRAME_DT)
    worker.close()
    et, tt = [], []
    for k, T in rows:
        f = k - k0
        if 0 <= f < len(ref) and trusted[f]:
            et.append(np.linalg.norm(ref[f, :3, 3] - T[:3, 3]) * 100)
            tt.append(f * FRAME_DT)
    et, tt = np.array(et), np.array(tt)
    lost_before = [s for s in states if s[1] == "lost" and s[0] < trusted_until]
    first_loc = min(tt) if len(tt) else float("nan")
    until = f"{trusted_until:.0f} s" if np.isfinite(trusted_until) else "the end"
    print(f"3. walk: localised by {first_loc:.1f} s, {session.corrections} corrections ({session.rejected} rejected); "
          f"vs the reference on {len(et)} trusted frames before {until}: median {np.median(et):.1f} cm, "
          f"p90 {np.percentile(et, 90):.1f} cm; lost before {until}: {len(lost_before)} frames; "
          f"state at the end: {session.state}")
    ok = len(et) > 300 and np.median(et) < 3.0 and np.percentile(et, 90) < 6.0 and not lost_before
    session.close()
    return bool(ok)


def main():
    map_path = common.prior_map()
    poses = common.saved_poses()
    rec = common.static_recording(poses)
    if map_path is None or rec is None:
        common.skip("needs the prior map and a recording of the scanner standing still inside it",
                    "LIVOX_PRIOR_MAP", "LIVOX_STATIC_RECORDINGS")
    dev = common.device()
    T_saved = poses.get(common.pose_key(rec))
    x, a, t = common.load_recording(rec)
    ok = step_localise(dev, map_path, x, a, t, T_saved)
    ok &= step_static_session(dev, map_path, x, a, t, T_saved)
    walk, ref = common.walk_recording(), common.walk_reference()
    if walk is not None and ref is not None:
        ok &= step_walk(dev, map_path, walk, ref)
    else:
        print("3. walk recording or its reference missing: skipped")
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
