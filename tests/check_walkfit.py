"""The continuous-time trajectory fit (walkfit.py) on the simulated office, against ground truth.

The prior map is a simulated terrestrial scan of the office as it was (common.sim_prior_map), the walk the
simulated Mid-40 through the office as it is now (a box added, one removed, a person walking).

1. The tape's gradient of the fit's loss matches central differences, for the knots, the range offsets and the
   angular distortion: the adjoint kernels Warp generates are what the solver steps on.
2. A 20 s walk started from a trajectory drifting smoothly up to 30 cm and 3 deg away from the truth: the fit
   brings it back within 1.5 cm / 0.15 deg rms.
3. Calibration: in the same walk half the returns are labelled second returns and measured 4 cm long, and every
   direction carries a 1.5 mrad error at the (1, 1) prism harmonic. The fit recovers both (the second returns'
   offset relative to the first, which stays at zero by design) and leaves the other harmonics near zero.
"""

import math

import common
import numpy as np
import warp as wp
from livox_warp import localize, rosette, walkfit

SECONDS = 20.0


def gt_poses(sim, times):
    return np.array([sim.pose_at(float(ti)) for ti in times])


def pattern():
    """The simulator's built-in pattern, with every default harmonic available to the distortion model."""
    base = rosette.RosetteModel.sim_default()
    h = np.array(rosette.DEFAULT_HARMONICS)
    coef = np.zeros(len(h), np.complex128)
    coef[:2] = base.coef
    return rosette.RosetteModel(omega=base.omega, harmonics=h, coef=coef)


def set_array(arr: wp.array, values: np.ndarray):
    wp.copy(arr, wp.array(values.astype(np.float32), dtype=arr.dtype, device=arr.device))


def gradient_check(loc, sim, x, a, t, dev):
    sel = (t >= 1.0) & (t < 4.0)
    times = np.arange(0.85, 4.2, 0.1)
    init = common.drifted(gt_poses(sim, times), times, 0.02, 0.2)
    cfg = walkfit.WalkFitConfig(pts_per_knot=300)
    wf = walkfit.WalkFit(loc.grid, x[sel], a[sel], t[sel], times, init, rosette=pattern(), config=cfg, device=dev)
    rng = np.random.default_rng(1)
    set_array(wf.dp, rng.normal(0, 0.01, (wf.n_knots, 3)))
    set_array(wf.dphi, rng.normal(0, 0.002, (wf.n_knots, 3)))
    set_array(wf.bias, np.array([0.01, -0.02, 0.0]))
    set_array(wf.dist, rng.normal(0, 0.001, 2 * rosette.MAX_HARMONICS))
    wf.associate(0.3, 1.0, 4.0)

    def loss():
        tape = wf.objective(0.05, calib=True)
        v = float(wf.loss.numpy()[0])
        tape.zero()
        return v

    tape = wf.objective(0.05, calib=True)
    grads = {"dp": wf.dp.grad.numpy().copy(), "dphi": wf.dphi.grad.numpy().copy(),
             "bias": wf.bias.grad.numpy().copy(), "dist": wf.dist.grad.numpy().copy()}
    tape.zero()
    # the loss is a float32 sum of atomics: its last digits change from one evaluation to the next, so only an
    # entry whose step moves the loss well past that floor can be judged by finite differences
    l0 = [loss() for _ in range(4)]
    floor = max(np.ptp(l0), 1e-6 * abs(l0[0]))
    arrays = {"dp": wf.dp, "dphi": wf.dphi, "bias": wf.bias, "dist": wf.dist}
    eps = {"dp": 2e-3, "dphi": 2e-4, "bias": 2e-3, "dist": 2e-4}
    worst = 0.0
    judged = 0
    for name, arr in arrays.items():
        g = grads[name].reshape(-1)
        base = arr.numpy().copy()
        flat = base.reshape(-1)
        top = [i for i in np.argsort(-np.abs(g))[:3] if abs(g[i]) * eps[name] > 100 * floor]
        assert top, f"no entry of {name} moves the loss past its rounding floor"
        for idx in top:  # the entries the loss depends on most
            judged += 1
            vals = []
            for sgn in (1, -1):
                v = flat.copy()
                v[idx] += sgn * eps[name]
                set_array(arr, v.reshape(base.shape))
                vals.append(loss())
            set_array(arr, base)
            fd = (vals[0] - vals[1]) / (2 * eps[name])
            rel = abs(fd - g[idx]) / max(abs(fd), abs(g[idx]), 1e-9)
            worst = max(worst, rel)
            assert rel < 0.05, f"{name}[{idx}]: tape {g[idx]:.4g}, finite differences {fd:.4g}"
    print(f"gradients: tape vs central differences on {judged} entries of the knots, range offsets and distortion: "
          f"worst {worst:.2%} apart (loss rounding floor {floor:.2g} of {abs(l0[0]):.3g})")


def drift(loc, sim, x, a, t, dev):
    times = np.arange(0.05, SECONDS, 0.1)
    gt = gt_poses(sim, times)
    init = common.drifted(gt, times, 0.30, 3.0)
    cfg = walkfit.WalkFitConfig(fit_bias=False, fit_distortion=False)
    wf = walkfit.WalkFit(loc.grid, x, a, t, times, init, config=cfg, localizer=loc.loc, device=dev, log=print)
    wf.run()
    e0 = common.traj_errors(init, gt)
    e1 = common.traj_errors(wf.poses(times), gt)
    print(f"drift: started {e0[0] * 100:.1f} cm / {e0[1]:.2f} deg rms off the truth, fitted to "
          f"{e1[0] * 100:.2f} cm / {e1[1]:.3f} deg")
    assert e1[0] < 0.015 and e1[1] < 0.15, e1


def calibration(loc, sim, x, a, t, dev):
    rng = np.random.default_rng(2)
    model = pattern()
    second = rng.random(len(t)) < 0.5
    a2 = (a & ~np.uint32(0xFF << 16)) | (second.astype(np.uint32) << 16)
    r = np.linalg.norm(x, axis=1)
    x2 = x * ((r + 0.04 * second) / r)[:, None]
    d_true = 1.5e-3 * np.exp(0.5j)
    th = model.phases(t)
    z = rosette.measured_angles(x2) + d_true * np.exp(1j * (th[:, 0] + th[:, 1]))
    dirs = np.stack([np.ones(len(z)), np.tan(z.real), np.tan(z.imag)], 1)
    x2 = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * np.linalg.norm(x2, axis=1)[:, None]).astype(np.float32)

    times = np.arange(0.05, SECONDS, 0.1)
    gt = gt_poses(sim, times)
    init = common.drifted(gt, times, 0.05, 0.5)
    wf = walkfit.WalkFit(loc.grid, x2, a2, t, times, init, rosette=model, localizer=loc.loc, device=dev, log=print)
    wf.run()
    cal = wf.calibration()
    b = cal["range_offset_m"]
    dist = cal["distortion_mrad"]
    d11 = complex(*dist["1,1"]) / 1e3
    others = max(abs(complex(*v)) for k, v in dist.items() if k != "1,1") / 1e3
    e = common.traj_errors(wf.poses(times), gt)
    print(f"calibration: second returns {b['second'] * 100:+.2f} cm from the first (truth -4); (1, 1) distortion "
          f"{abs(d11) * 1e3:.2f} mrad at {math.degrees(np.angle(d11)):.0f} deg (truth {abs(d_true) * 1e3:.2f} mrad at "
          f"{math.degrees(np.angle(-d_true)):.0f} deg), the others at most {others * 1e3:.2f} mrad; trajectory "
          f"{e[0] * 100:.2f} cm / {e[1]:.3f} deg rms")
    print("             " + ", ".join(f"({k}) {abs(complex(*v)):.2f}" for k, v in dist.items()) + " mrad")
    assert b["first"] == 0.0 and abs(b["second"] + 0.04) < 0.005, b
    assert abs(d11 + d_true) < 0.3e-3, d11
    assert others < 0.3e-3, others
    assert e[0] < 0.01 and e[1] < 0.15, e


def main():
    dev = common.device()
    pm = common.sim_prior_map(dev)
    loc = localize.Localizer(pm["xyz"], pm["normal"], pm["planarity"], device=dev)
    print(f"prior map: {len(pm['xyz']):,} points from the simulated terrestrial scan")
    sim, x, a, t = common.sim_walk(dev, SECONDS)
    print(f"walk: {len(t):,} clean returns over {SECONDS:.0f} s")
    gradient_check(loc, sim, x, a, t, dev)
    drift(loc, sim, x, a, t, dev)
    calibration(loc, sim, x, a, t, dev)


if __name__ == "__main__":
    main()
