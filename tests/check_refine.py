"""The three solvers chained as refine.py chains them: on the simulator against ground truth, and on a real walk.

Simulator (the office as it is now, a prior map of it as it was):
  - the scan pattern fitted from the walk's own directions recovers the simulator's prism rates;
  - the trajectory fitted from a 30 cm / 3 deg drift (no mask, no calibration);
  - the occupancy field from those rays marks the changes, and its mask leaves out most points on the added box;
  - the trajectory fitted again with the mask and the calibration free is no worse, and the calibration of a
    sensor that has none stays at zero.
Real (the walk recording and the prior map, see tests/README.md): refine.refine() end to end, written to
$LIVOX_DATA_DIR/out/refine; its frames must sit on the scan.
"""

import common
import numpy as np
from livox_warp import dmath, localize, occupancy, refine, rosette, walkfit

SECONDS = 20.0


def simulated(dev):
    pm = common.sim_prior_map(dev)
    loc = localize.Localizer(pm["xyz"], pm["normal"], pm["planarity"], device=dev)
    sim, x, a, t = common.sim_walk(dev, SECONDS)

    ros = rosette.fit(t, rosette.measured_angles(x), log=print)
    truth = rosette.RosetteModel.sim_default().freq_hz
    off = np.abs(ros.freq_hz - truth)
    print(f"rosette: {ros.freq_hz[0]:+.4f} / {ros.freq_hz[1]:+.4f} Hz from the walk (truth {truth[0]:+.1f} / "
          f"{truth[1]:+.1f}), residual {ros.stats['rms_mrad']:.2f} mrad")
    assert np.all(off < 1e-3), off

    times = np.arange(0.05, SECONDS, 0.1)
    gt = np.array([sim.pose_at(float(s)) for s in times])
    init = common.drifted(gt, times, 0.30, 3.0)
    cfg_a = walkfit.WalkFitConfig(fit_bias=False, fit_distortion=False)
    fit_a = walkfit.WalkFit(loc.grid, x, a, t, times, init, rosette=ros, config=cfg_a, localizer=loc.loc, device=dev)
    fit_a.run()
    e_a = common.traj_errors(fit_a.poses(times), gt)

    o, d, r, w = fit_a.rays(x, a, t)
    field = occupancy.OccupancyField.around(o.numpy(), d.numpy(), r.numpy(), 0.05, device=dev)
    field.warm_start(loc.grid, pm["stations"])
    field.set_rays(o, d, r, w)
    field.train(epochs=6)
    print(f"occupancy from the first fit: {field.classify()}")
    mask = field.change_mask()
    bits = mask.bits.numpy()
    # the walk's returns on the added box (placed by the truth): how many does the mask leave out?
    Tp = dmath.interpolate_poses(times, gt, t)
    world = np.einsum("nij,nj->ni", Tp[:, :3, :3], x) + Tp[:, :3, 3]
    on_box = common.box_distance(world, common.ADDED_BOX) < 0.02
    ijk = np.floor((world[on_box] - field.lo) / field.cell).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < field.dims), axis=1)
    flat = (ijk[:, 2] * field.dims[1] + ijk[:, 1]) * field.dims[0] + ijk[:, 0]
    masked = np.zeros(len(ijk), bool)
    masked[inside] = (bits[flat[inside]] & occupancy.EXCLUDE_POINT) != 0
    print(f"mask: leaves out {masked.mean():.0%} of the {on_box.sum():,} returns on the added box")
    assert masked.mean() > 0.7, masked.mean()

    fit_b = walkfit.WalkFit(loc.grid, x, a, t, fit_a.knot_t, fit_a.poses(fit_a.knot_t), rosette=ros,
                            config=walkfit.WalkFitConfig(stages=refine.FINE_STAGES), localizer=loc.loc, device=dev)
    fit_b.set_change_mask(mask)
    fit_b.run()
    e_b = common.traj_errors(fit_b.poses(times), gt)
    cal = fit_b.calibration()
    worst_dist = max(abs(complex(*v)) for v in cal["distortion_mrad"].values())
    print(f"trajectory: {e_a[0] * 100:.2f} cm / {e_a[1]:.3f} deg rms after the first fit, {e_b[0] * 100:.2f} cm / "
          f"{e_b[1]:.3f} deg after the masked, calibrated one; range offsets "
          f"{cal['range_offset_m']['first'] * 100:+.2f} cm, distortion at most {worst_dist:.2f} mrad (truth 0)")
    assert e_b[0] <= e_a[0] * 1.1 + 0.002 and e_b[0] < 0.015, (e_a, e_b)
    assert abs(cal["range_offset_m"]["first"]) < 0.005 and worst_dist < 0.3, cal


def real(dev):
    walk, prior = common.walk_recording(), common.prior_map()
    if walk is None or prior is None:
        print("(no walk recording and prior map: the real part did not run; set LIVOX_DATA_DIR / "
              "LIVOX_WALK_RECORDING / LIVOX_PRIOR_MAP)")
        return
    out = common.DATA_DIR / "out" / "refine"
    rep = refine.refine(str(walk), str(prior), str(out), dev)
    f = rep["frames"]
    print(f"real walk: {f['count']} frames, median {f['fit3_median']:.2f} of each frame's points within 3 cm of the "
          f"scan, {f['fit3_above_0.8']:.0%} of the frames above 0.8; calibration {rep['calibration']}")
    assert f["fit3_median"] > 0.5, f


def main():
    dev = common.device()
    simulated(dev)
    real(dev)


if __name__ == "__main__":
    main()
