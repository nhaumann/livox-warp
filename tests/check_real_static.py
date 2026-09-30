"""Odometry on real recordings of a STATIC sensor: the pose must not wander.

Needs recordings of the scanner standing still (LIVOX_STATIC_RECORDINGS, or every *static*.lvxr under
$LIVOX_DATA_DIR/recordings); skips when there are none. Each is replayed to numpy through the Rust replay
(fast) and fed to Odometry offline in time order, so runs are quick. Ground truth is "no motion": every
frame's pose should stay at the first frame's. The jitter that remains is mostly roll about the optical
axis, the direction a 38 deg circular field of view constrains least.

The GPU odometry is not bit-for-bit deterministic (atomic accumulation order), so the same input gives a
slightly different drift every run. Each recording therefore runs --runs times (default 5) and is judged on
its worst run against limits above that spread (final within 1.5 cm / 0.75 deg, never beyond 3 cm / 1 deg)
and on its median run against the tighter target (final within 1 cm / 0.5 deg). Registration must have run:
for a static sensor an odometry that never registers is perfect, so at least 90% of the frames after the
warm-up need iterations and enough correspondences.

usage: python tests/check_real_static.py [recording.lvxr ...] [--set attr=value ...] [--repeat N] [--runs N]

--repeat N plays each recording N times back to back (timestamps continued) as one longer static session:
the same scans again, so it tests whether the estimate random-walks, not new noise.
"""

import argparse
import sys
from pathlib import Path

import common
import numpy as np
from livox_warp import gpu
from livox_warp.odom import Odometry

JUDGED = ("max_cm", "max_deg", "final_cm", "final_deg")


def repeat(data, n):
    x, a, t = data
    if n <= 1:
        return data
    span = float(t.max() - t.min()) + 0.01
    return (np.concatenate([x] * n), np.concatenate([a] * n),
            np.concatenate([t + np.float32(k * span) for k in range(n)]))


def run(data, dev, overrides, batch=1700):
    x, a, t = data
    pipe = gpu.Pipeline(ring_capacity=1 << 20, map_capacity=1 << 18, device=dev)
    od = Odometry(pipe, dev)
    for k, v in overrides.items():
        setattr(od, k, v)
    od.reset(np.eye(4))
    stats = []

    def on_frame(k, T, fx, fa, ft, ff, m):
        stats.append(dict(od.stats))

    for i in range(0, len(t), batch):
        xs, as_, ts = x[i:i + batch], a[i:i + batch], t[i:i + batch]
        n = len(ts)
        sx, sa, st, _ = pipe.stage(xs, as_, ts, n, 1.0 / od.frame_dt)
        od.push(sx, sa, st, n, float(ts.min()), float(ts.max()), on_frame)
    poses = [T for _, _, T in od.traj][od.warmup:]
    tr = np.array([np.linalg.norm(T[:3, 3]) for T in poses]) * 100
    rot = np.array([common.rot_angle(T[:3, :3]) for T in poses])
    post = stats[od.warmup + 2:]
    rms = np.array([s["rms"] for s in post]) * 100
    ms = np.array([s["ms"] for s in stats])
    reg = float(np.mean([s["iters"] > 0 and s["corr"] >= od.min_corr for s in post])) if post else 0.0
    return {"frames": len(od.traj), "max_cm": tr.max(), "final_cm": tr[-1], "max_deg": rot.max(),
            "final_deg": rot[-1], "rms_cm": float(np.median(rms)) if len(rms) else 0.0,
            "ms": float(np.median(ms)), "reg": reg}


def check(path, dev, overrides, reps, runs):
    """Run one recording `runs` times; True if its worst and median runs are within the limits."""
    data = repeat(common.load_recording(path), reps)
    rs = [run(data, dev, overrides) for _ in range(runs)]
    worst = {k: max(r[k] for r in rs) for k in JUDGED}
    med = {k: float(np.median([r[k] for r in rs])) for k in JUDGED}
    reg = min(r["reg"] for r in rs)
    r = rs[0]
    print(f"{path.name}: {r['frames']} frames x {runs} runs, fit rms {r['rms_cm']:.2f} cm, "
          f"{r['ms']:.1f} ms/frame, {reg:.0%} of frames registered (worst run)")
    print(f"    drift max  {med['max_cm']:.2f} cm / {med['max_deg']:.3f} deg median, "
          f"{worst['max_cm']:.2f} cm / {worst['max_deg']:.3f} deg worst")
    print(f"    final      {med['final_cm']:.2f} cm / {med['final_deg']:.3f} deg median, "
          f"{worst['final_cm']:.2f} cm / {worst['final_deg']:.3f} deg worst")
    ok = reg > 0.9 and r["rms_cm"] > 0.0
    ok &= worst["max_cm"] < 3.0 and worst["max_deg"] < 1.0 and worst["final_cm"] < 1.5 and worst["final_deg"] < 0.75
    ok &= med["final_cm"] < 1.0 and med["final_deg"] < 0.5
    return bool(ok)


def parse_override(text):
    """attr=value for --set: an integer stays an integer, anything else becomes a float."""
    key, value = text.split("=", 1)
    return key, int(value) if value.lstrip("-").isdigit() else float(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recordings", nargs="*", type=Path,
                    help="recordings of the scanner standing still (default: LIVOX_STATIC_RECORDINGS)")
    ap.add_argument("--set", action="append", default=[], type=parse_override, metavar="ATTR=VALUE",
                    help="override an Odometry tuning attribute, e.g. --set reg_voxel=0.2")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="play each recording N times back to back")
    ap.add_argument("--runs", type=int, default=5, metavar="N", help="runs per recording (default 5)")
    args = ap.parse_args()
    paths = args.recordings or common.static_recordings()
    if not paths:
        common.skip("no recording of the scanner standing still", "LIVOX_STATIC_RECORDINGS")
    dev = common.device()
    overrides = dict(args.set)
    ok = True
    for p in paths:
        if not p.exists():
            print(f"{p.name}: not found, skipped")
            continue
        ok &= check(p, dev, overrides, args.repeat, args.runs)
    print("OK" if ok else "FAIL: a static sensor drifted or jittered beyond the limits, or registration did not run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
