"""Benchmark the SLAM on a real recording against a prior scan of the place.

usage: python benchmarks/bench_prior_map.py [recording.lvxr] [--map prior.npz] [--reference walk_reference.npz]
                                            [--out bench.npz]

The recording (default: the walk recording, see tests/README.md) must start with the scanner standing still
for ~1.5 s inside the mapped area, for the initial localisation. The SLAM runs exactly as in the viewer
(odometry on its worker thread, the prior-map session correcting drift), and it is scored against the scan
itself, not against another trajectory estimate: a frame-by-frame scan registration (the obvious reference)
fails exactly where the SLAM is hard (fast swings, a flat wall filling the view), and then it is the
reference that is wrong. Per half second:

  free fit       fraction of the last second's points within 10 cm of the scan, placed by the free
                 odometry and the initial alignment only: 0.6-0.9 while it holds, falling as it drifts
  drift          how far the prior-map session has had to move the odometry's world to keep it on the
                 scan (cm / deg): the odometry's accumulated drift
  fit            the same fraction after that correction: low means lost, or a place the scan lacks
  state          the session's state (tracking, lost, ...)

With --reference (ref (N, 4, 4) per frame, fit (N,) and trusted_until in seconds), the corrected and the
free poses are also compared with it wherever it is solid (fit > 0.8, no implausible step) and before
trusted_until.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import common  # noqa: E402
from livox_warp import gpu, localize, prior_map  # noqa: E402
from livox_warp.prior_session import PriorSession, deskew  # noqa: E402
from livox_warp.slam_worker import OdomWorker  # noqa: E402
from livox_warp.sources import clean_returns  # noqa: E402


def quiet(*_):
    """A log sink for the session."""


def run_slam(dev, map_path, x, a, t, frame_dt):
    """The viewer's SLAM over the recording: {frame id: (T_W, xi, T_MW or None, state)} and the session."""
    session = PriorSession(str(map_path), dev, log=quiet)
    session.frame_dt = frame_dt
    common.wait_ready(session)
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 18, device=dev)
    worker = OdomWorker(pipe, dev)
    worker.configure(frame_dt=frame_dt)
    worker.reset(np.eye(4))
    frames = {}

    def on_frame(r, sess):
        frames[r.k] = (r.T.copy(), np.array(r.xi, dtype=np.float64),
                       None if sess.T_MW is None else sess.T_MW.copy(), sess.state)

    common.drive(session, x, a, t, worker, pipe, on_frame=on_frame, frame_dt=frame_dt)
    worker.close()
    session.close()  # its copy of the map goes before the scoring localizer builds another
    return frames, session


def score_windows(dev, map_path, frames, x, a, t, first, frame_dt):
    """Per half second: (time, free fit, corrected fit, drift cm, drift deg, state)."""
    pm = prior_map.load(str(map_path))
    L = localize.Localizer(pm["xyz"], pm["normal"], pm.get("planarity"), device=dev)
    ks = sorted(frames)
    T_MW0 = frames[first][2]
    k0 = int(np.floor(t.min() / frame_dt))
    kf = np.floor(t / frame_dt).astype(np.int64)
    order = np.argsort(kf, kind="stable")
    kf_sorted = kf[order]
    rng = np.random.default_rng(0)
    rows = []
    step = max(1, int(round(0.5 / frame_dt)))
    win = max(1, int(round(1.0 / frame_dt)))
    for j in range(first + win, ks[-1] + 1, step):
        pts = []
        for k in range(j - win + 1, j + 1):
            if k not in frames:
                continue
            lo, hi = np.searchsorted(kf_sorted, [k, k + 1])
            idx = order[lo:hi]
            if len(idx) > 3000:
                idx = rng.choice(idx, 3000, replace=False)
            idx = idx[clean_returns(x[idx], a[idx])]
            T_W, xi, _, _ = frames[k]
            s = t[idx] / frame_dt - k - 0.5
            p = deskew(x[idx].astype(np.float64), s, xi)
            pts.append(p @ T_W[:3, :3].T + T_W[:3, 3])
        if not pts or j not in frames:
            continue
        sub = np.concatenate(pts)
        if len(sub) < 500:
            continue
        T_MW = frames[j][2]
        _, fit_free, _ = L.refine(sub, T_MW0, schedule=())
        fit_corr = L.refine(sub, T_MW, schedule=())[1] if T_MW is not None else np.zeros(3)
        d = np.linalg.inv(T_MW0) @ T_MW if T_MW is not None else np.eye(4)
        rows.append(((j - k0) * frame_dt, float(fit_free[2]), float(fit_corr[0]),
                     np.linalg.norm(d[:3, 3]) * 100, common.rot_angle(d[:3, :3]), frames[j][3]))
    return rows


def reference_errors(ref_path, frames, k0, first, frame_dt):
    """{time: (corrected error cm, free error cm)} on the frames where the reference is trusted."""
    ref, trusted, _ = common.load_walk_reference(ref_path, frame_dt)
    T_MW0 = frames[first][2]
    errors = {}
    for k in sorted(frames):
        f = k - k0
        T_W, _, T_MW, _ = frames[k]
        if 0 <= f < len(ref) and trusted[f] and T_MW is not None:
            errors[f * frame_dt] = (np.linalg.norm(ref[f, :3, 3] - (T_MW @ T_W)[:3, 3]) * 100,
                                    np.linalg.norm(ref[f, :3, 3] - (T_MW0 @ T_W)[:3, 3]) * 100)
    return errors


def report(rows, ref_err, lost_fit):
    header = "\n  window   free fit   drift (cm / deg)   fit    state"
    if ref_err is not None:
        header += "        vs reference: corrected / free (median cm)"
    print(header)
    tmax = rows[-1][0] if rows else 0.0
    for lo in np.arange(0.0, tmax + 5.0, 5.0):
        r = [q for q in rows if lo <= q[0] < lo + 5.0]
        if not r:
            continue
        states = sorted({q[5] for q in r})
        line = (f"  {lo:4.0f}-{lo + 5:<4.0f} {np.median([q[1] for q in r]):6.2f}    {np.max([q[3] for q in r]):6.1f} / "
                f"{np.max([q[4] for q in r]):5.2f}     {np.median([q[2] for q in r]):5.2f}  {','.join(states)}")
        if ref_err is not None:
            e = [v for tt_, v in ref_err.items() if lo <= tt_ < lo + 5.0]
            line += (f"   {np.median([v[0] for v in e]):6.1f} / {np.median([v[1] for v in e]):6.1f} ({len(e)} frames)"
                     if e else "   (reference not solid)")
        print(line)
    held = [q for q in rows if q[2] >= lost_fit]
    # the stretch from the start until the fit first fails: there the drift figures are trustworthy
    until = next((q[0] for q in rows if q[2] < lost_fit), None)
    early = [q for q in rows if until is None or q[0] < until]
    print(f"\n  on the scan (fit >= {lost_fit}) for {len(held) / max(len(rows), 1):.0%} of the half seconds"
          + (f"; first lost at {until:.1f} s" if until is not None else "; never lost"))
    if early:
        print(f"  until then the free odometry drifted at most {max(q[3] for q in early):.1f} cm / "
              f"{max(q[4] for q in early):.2f} deg from the scan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", nargs="?", type=Path, default=common.walk_recording(),
                    help="a walk that starts at rest in the mapped area (default: LIVOX_WALK_RECORDING)")
    ap.add_argument("--map", type=Path, default=common.prior_map(), help="the prior map (default: LIVOX_PRIOR_MAP)")
    ap.add_argument("--reference", type=Path, help="frame-by-frame reference to compare with (walk_reference.npz)")
    ap.add_argument("--out", type=Path, help="save the poses and the per-window scores here (.npz)")
    ap.add_argument("--frame-dt", type=float, default=0.1)
    args = ap.parse_args()
    if args.recording is None or args.map is None:
        common.skip("needs a recording and the prior map", "LIVOX_WALK_RECORDING", "LIVOX_PRIOR_MAP")
    dev = common.device()

    t_start = time.perf_counter()
    x, a, t = common.load_recording(args.recording)
    print(f"{args.recording.name}: {len(t):,} points over {t.max() - t.min():.1f} s")
    frames, session = run_slam(dev, args.map, x, a, t, args.frame_dt)
    ks = sorted(frames)
    first = next((k for k in ks if frames[k][2] is not None), None)
    if first is None:
        sys.exit("never localised: does the recording start with the scanner standing still in the mapped area?")
    k0 = int(np.floor(t.min() / args.frame_dt))
    print(f"localised {(first - k0) * args.frame_dt:.1f} s in; {session.corrections} drift corrections, "
          f"{session.rejected} rejected")

    rows = score_windows(dev, args.map, frames, x, a, t, first, args.frame_dt)
    ref_err = reference_errors(args.reference, frames, k0, first, args.frame_dt) if args.reference else None
    report(rows, ref_err, session.lost_fit)
    if args.out:
        nan_pose = np.full((4, 4), np.nan)
        np.savez(args.out, k=np.array(ks), T_W=np.array([frames[k][0] for k in ks]),
                 T_MW=np.array([frames[k][2] if frames[k][2] is not None else nan_pose for k in ks]),
                 windows=np.array([q[:5] for q in rows]))
        print(f"  saved {args.out}")
    print(f"  ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
