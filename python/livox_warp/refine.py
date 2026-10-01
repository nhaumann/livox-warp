"""Refine a recording against a prior map: scan pattern, continuous-time trajectory, occupancy and what changed.

    python -m livox_warp.refine RECORDING.lvxr --map maps/prior_20mm.npz --out out/walk

The three solvers feed one another (rosette.py, walkfit.py, occupancy.py):
  1. every firing of the recording (lvxr.py), and the scan pattern fitted to them: the prism phases of every
     point, and the directions of the firings that returned nothing;
  2. a starting trajectory: the odometry over the recording, placed in the map by a global localisation: of the
     first 1.5 s at rest, or, where that view is ambiguous, of the odometry's own map over the first 5, 10, 20
     or 40 s (a bigger piece of the building is far less ambiguous, and over that long the odometry holds);
  3. the trajectory fitted to the scan;
  4. the occupancy field from those rays and the empty firings, warm-started from the scan; what changed;
  5. the trajectory fitted again with the changes left out and the calibration free (a range offset per return,
     the angular distortion over the prism phases), from where step 3 left it;
  6. the occupancy field again, from the final rays.

Writes to --out: rosette.npz; trajectory.npz (the knots, and per 0.1 s frame ref / fit / k0 / trusted_until: the
reference benchmarks/bench_prior_map.py --reference reads); calibration.json; occupancy.npz; changes.ply (added
red, removed blue, transient amber); report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import warp as wp

from . import dmath, export, gpu, localize, lvxr, occupancy, prior_map, rosette, walkfit
from .odom import Odometry

FINE_STAGES = (walkfit.Stage(0.06, 0.15, 60, 2e-3, 3e-4), walkfit.Stage(0.03, 0.08, 80, 8e-4, 1.2e-4),
               walkfit.Stage(0.02, 0.06, 100, 4e-4, 6e-5))
CHANGE_COLOURS = {occupancy.ADDED: (255, 64, 48), occupancy.REMOVED: (64, 128, 255),
                  occupancy.TRANSIENT: (255, 190, 60)}
MAX_MISSES = 2_000_000
# a frame whose points fit the scan less than this (within 3 cm) adds no rays to the occupancy: the fit did not place
# it, and its rays would cross the floor and walls that are there and read them as removed
RAY_MIN_FIT = 0.5


def odometry_trajectory(xyz, attr, t, device, frame_dt: float = 0.1, batch: int = 2000):
    """The LiDAR-only odometry over a recording's points, as the viewer runs it: (mid-frame times, 4x4 poses in its
    world, which is the first frame's sensor frame)."""
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 18, device=device)
    odom = Odometry(pipe, device)
    odom.frame_dt = frame_dt
    odom.reset(np.eye(4))
    t32 = np.asarray(t, dtype=np.float32)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    attr = np.ascontiguousarray(attr, dtype=np.uint32)

    def on_frame(*_):
        pass

    for a in range(0, len(t32), batch):
        ts = t32[a:a + batch]
        n = len(ts)
        sx, sa, st, _ = pipe.stage(xyz[a:a + batch], attr[a:a + batch], ts, n, 1.0 / frame_dt)
        odom.push(sx, sa, st, n, float(ts.min()), float(ts.max()), on_frame)
    odom.synchronize()
    if len(odom.traj) < 2:
        raise RuntimeError("the odometry produced no trajectory")
    return np.array([tm for _, tm, _ in odom.traj]), np.array([T for _, _, T in odom.traj])


def place_in_map(loc, xyz, t, times, poses_w, spans=(5.0, 10.0, 20.0, 40.0), max_points: int = 400_000):
    """T_MW, the odometry's world in the map, with the search's diagnostics and the seconds it took: the first 1.5 s
    at rest (the odometry's world is the first frame), else the odometry's own map over growing spans."""
    T, info = loc.search(xyz[t < t[0] + 1.5], log=lambda s: None)
    if info["unique"]:
        return T, info, 1.5
    tried = [f"1.5 s: {info['inliers']:.2f} vs {info['runner_up']:.2f}"]
    for span in spans:
        idx = np.flatnonzero(t < t[0] + span)
        idx = idx[::max(1, len(idx) // max_points)]
        Tp = dmath.interpolate_poses(times, poses_w, t[idx])
        w = np.einsum("nij,nj->ni", Tp[:, :3, :3], xyz[idx]) + Tp[:, :3, 3]
        T, info = loc.search(w.astype(np.float32), log=lambda s: None)
        if info["unique"]:
            return T, info, span
        tried.append(f"{span:g} s: {info['inliers']:.2f} vs {info['runner_up']:.2f}")
        if span >= t[-1] - t[0]:
            break
    raise RuntimeError("the recording does not localise uniquely in the map (inliers vs the best elsewhere: "
                       + "; ".join(tried) + "): does it start inside the mapped area?")


def miss_rays(fir: lvxr.Firings, ros: rosette.RosetteModel, cap: int = MAX_MISSES, seed: int = 0):
    """Times and sensor-frame directions of the firings that returned nothing: the data's own direction where it has
    one (spherical streams), else the scan pattern's."""
    idx = np.flatnonzero(fir.misses)
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    d = fir.direction[idx].astype(np.float64)
    need = ~np.isfinite(d[:, 0])
    if need.any():
        d[need] = ros.direction(fir.t[idx[need]])
    return fir.t[idx], d.astype(np.float32)


def refine(recording: str, map_path: str, out_dir: str, device=None, seconds: float | None = None,
           frame_dt: float = 0.1, cell: float = 0.05, epochs: int = 6, log=print) -> dict:
    """Everything the module docstring lists; returns the report (also written to out_dir/report.json)."""
    t_all = time.perf_counter()
    dev = wp.get_device(device)
    os.makedirs(out_dir, exist_ok=True)
    report = {"recording": os.path.basename(recording), "map": os.path.basename(map_path)}

    def stamp(name, t0):
        report.setdefault("seconds", {})[name] = round(time.perf_counter() - t0, 1)

    # 1. firings and the scan pattern
    t0 = time.perf_counter()
    fir = lvxr.read_firings(recording)
    if seconds:
        fir = fir.head(seconds)
    ros = rosette.fit_firings(fir, log=log)
    ros.save(os.path.join(out_dir, "rosette.npz"))
    xyz, attr, t, _ = fir.points()
    miss_t, miss_dir = miss_rays(fir, ros)
    report["firings"] = {"count": int(len(fir.t)), "returns": int(len(t)), "misses": int(fir.misses.sum()),
                         "span_s": float(fir.t[-1] - fir.t[0])}
    report["rosette"] = ros.stats
    log(f"{report['recording']}: {len(fir.t):,} firings over {report['firings']['span_s']:.1f} s, {len(t):,} clean "
        f"returns, {report['firings']['misses']:,} empty firings")
    stamp("firings_and_rosette", t0)

    # 2. the starting trajectory
    t0 = time.perf_counter()
    pm = prior_map.load(map_path)
    loc = localize.Localizer(pm["xyz"], pm["normal"], pm.get("planarity"), device=dev)
    stations = pm.get("stations")
    if stations is None or not len(stations):  # a merged E57 keeps no scan poses: find them in the point density
        stations = prior_map.estimate_stations(pm)
        log(f"the prior map has no scanner stations; {len(stations)} estimated from its point density "
            "(python -m livox_warp.prior_map stations map.npz saves them)")
    report["stations"] = int(len(stations))
    del pm
    times, poses_w = odometry_trajectory(xyz, attr, t, dev, frame_dt)
    T_MW, info, span = place_in_map(loc, xyz, t, times, poses_w)
    init = T_MW[None] @ poses_w
    report["placed_from_s"] = span
    log(f"odometry: {len(times)} frames; placed in the map from the first {span:g} s (inliers {info['inliers']:.2f} "
        f"within 3 cm, runner-up elsewhere {info['runner_up']:.2f})")
    stamp("initial_trajectory", t0)

    # 3. the trajectory against the scan
    t0 = time.perf_counter()
    cfg_a = walkfit.WalkFitConfig(fit_bias=False, fit_distortion=False)
    fit_a = walkfit.WalkFit(loc.grid, xyz, attr, t, times, init, rosette=ros, config=cfg_a, localizer=loc.loc,
                            device=dev, log=log)
    report["walkfit_first"] = fit_a.run()
    stamp("walkfit_first", t0)

    # 4. occupancy from those rays; what changed
    t0 = time.perf_counter()
    o, d, r, w = fit_a.rays(xyz, attr, t, miss_t, miss_dir, min_fit=RAY_MIN_FIT, frame_dt=frame_dt)
    report["rays_left_out_first"] = float(np.mean(w.numpy() == 0.0))
    field = occupancy.OccupancyField.around(o.numpy(), d.numpy(), r.numpy(), cell, device=dev,
                                            clip=(loc.grid.lo - 0.5, loc.grid.hi + 0.5))
    field.warm_start(loc.grid, stations)
    field.set_rays(o, d, r, w)
    field.train(epochs=epochs, log=log)
    report["changes_first"] = field.classify()
    log(f"occupancy: {report['changes_first']}")
    stamp("occupancy_first", t0)

    # 5. the trajectory again, changes left out, calibration free
    t0 = time.perf_counter()
    cfg_b = replace(walkfit.WalkFitConfig(), stages=FINE_STAGES)
    fit_b = walkfit.WalkFit(loc.grid, xyz, attr, t, fit_a.knot_t, fit_a.poses(fit_a.knot_t), rosette=ros,
                            config=cfg_b, localizer=loc.loc, device=dev, log=log)
    fit_b.set_change_mask(field.change_mask())
    report["walkfit_second"] = fit_b.run()
    report["calibration"] = fit_b.calibration()
    stamp("walkfit_second", t0)

    # 6. occupancy from the final rays (warm-started again from the scan, so the first pass's errors do not stay)
    t0 = time.perf_counter()
    o, d, r, w = fit_b.rays(xyz, attr, t, miss_t, miss_dir, min_fit=RAY_MIN_FIT, frame_dt=frame_dt)
    report["rays_left_out"] = float(np.mean(w.numpy() == 0.0))
    field.warm_start(loc.grid, stations)
    field.m.zero_()
    field.v.zero_()
    field.adam_t = 0
    field.set_rays(o, d, r, w)
    field.train(epochs=epochs, log=log)
    report["changes"] = field.classify()
    log(f"occupancy: {report['changes']} ({report['rays_left_out']:.0%} of the rays left out: frames the fit did not "
        f"place)")
    stamp("occupancy_final", t0)

    # outputs
    ref, fit, k0 = fit_b.frame_reference(frame_dt)
    np.savez(os.path.join(out_dir, "trajectory.npz"), knot_t=fit_b.knot_t, knot_R=fit_b.base_R, knot_p=fit_b.base_p,
             ref=ref, fit=fit, k0=k0, frame_dt=frame_dt, trusted_until=len(ref) * frame_dt)
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"walkfit": report["calibration"], "rosette": {"omega": ros.omega.tolist(), **ros.stats}}, f,
                  indent=2)
    field.save(os.path.join(out_dir, "occupancy.npz"))
    pts, rgb = [], []
    for code, col in CHANGE_COLOURS.items():
        c = field.centres(code)
        pts.append(c)
        rgb.append(np.tile(np.array(col, np.uint8), (len(c), 1)))
    pts = np.concatenate(pts).astype(np.float32)
    export.write_ply(os.path.join(out_dir, "changes.ply"), pts, np.concatenate(rgb), np.zeros(len(pts), np.uint8))
    report["frames"] = {"count": int(len(ref)), "fit3_median": float(np.median(fit)),
                        "fit3_above_0.8": float(np.mean(fit > 0.8))}
    report["seconds"]["total"] = round(time.perf_counter() - t_all, 1)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"done in {report['seconds']['total']:.0f} s: {len(ref)} frames, median 3 cm fit "
        f"{report['frames']['fit3_median']:.2f}; written to {out_dir}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m livox_warp.refine", description=__doc__.split("\n")[0])
    p.add_argument("recording", help="an .lvxr recording made inside the mapped area")
    p.add_argument("--map", required=True, help="the prior map .npz (python -m livox_warp.prior_map convert)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--seconds", type=float, help="use only the first S seconds of the recording")
    p.add_argument("--cell", type=float, default=0.05, help="occupancy cell (m, default %(default)s)")
    p.add_argument("--epochs", type=int, default=6, help="occupancy training epochs (default %(default)s)")
    p.add_argument("--device", help="Warp device (default: cuda:0 if available)")
    a = p.parse_args(argv)
    wp.config.quiet = True
    wp.init()
    dev = a.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    refine(a.recording, a.map, a.out, dev, a.seconds, cell=a.cell, epochs=a.epochs)


if __name__ == "__main__":
    main()
