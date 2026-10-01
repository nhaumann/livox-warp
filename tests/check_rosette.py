"""The scan-pattern fit (rosette.py), the simulator scanning with it, and the firing-level reader (lvxr.py).

1. Synthetic firings from a known pattern with what a real stream has: a third and a fourth harmonic, a slow phase
   wander on both prisms, angular noise, dropped packets, empty firings and gross outliers. The fit must find the
   rates to 1e-3 Hz (prism a the faster one), reproduce the pattern to 0.1 mrad (a quarter of the noise), and leave
   a residual at the noise level.
2. SimSource(rosette=model) fires where the model says (the float64 phases handed to the kernel per batch).
3. With a recording (LIVOX_RECORDING, else any): the reader's returns match the native replay's point for point
   (positions and float32 times), and the unit's real pattern is fitted and reported.
"""

import math
import sys

import common
import numpy as np
from livox_warp import lvxr, rosette
from livox_warp.sources import SimSource

NOISE = 3e-4  # rad, per angle


def true_model():
    m = rosette.RosetteModel(omega=2 * math.pi * np.array([153.71, -97.28]),
                             harmonics=np.array(rosette.DEFAULT_HARMONICS),
                             coef=np.zeros(len(rosette.DEFAULT_HARMONICS), np.complex128))
    m.coef[0] = 0.1676
    m.coef[1] = 0.1676 * np.exp(0.7j)
    m.coef[2] = 0.0020 * np.exp(1.1j)  # (1, 1)
    m.coef[6] = 0.0008 * np.exp(-0.4j)  # (2, -1)
    return m


def wander(t):
    return np.stack([0.004 * np.sin(2 * math.pi * 0.3 * t), 0.003 * np.sin(2 * math.pi * 0.17 * t + 1.0)], 1)


def synthetic():
    rng = np.random.default_rng(3)
    model = true_model()
    seconds = 20.0
    t = np.arange(int(seconds / rosette.LATTICE_DT)) * rosette.LATTICE_DT
    packet = (np.arange(len(t)) // 100)
    dropped = rng.random(packet.max() + 1) < 0.02  # whole packets lost
    keep = ~dropped[packet] & (rng.random(len(t)) > 0.05)  # and 5 % empty firings
    t = t[keep]
    th = model.phases(t) + wander(t)
    z = np.exp(1j * (th @ model.harmonics.T)) @ model.coef
    z = z + NOISE * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    bad = rng.random(len(t)) < 0.005
    z[bad] += 0.02 * np.exp(2j * math.pi * rng.random(bad.sum()))
    d = np.stack([np.ones(len(z)), np.tan(z.real), np.tan(z.imag)], 1)
    z_meas = rosette.measured_angles(d / np.linalg.norm(d, axis=1, keepdims=True))

    fit = rosette.fit(t, z_meas, log=print)
    df = np.abs(fit.freq_hz - model.freq_hz)
    tt = np.linspace(0.5, seconds - 0.5, 20000)
    th_true = model.phases(tt) + wander(tt)
    z_true = np.exp(1j * (th_true @ model.harmonics.T)) @ model.coef
    err = np.abs(fit.angles(tt) - z_true) * 1000
    expect = NOISE * math.sqrt(2) * 1000
    print(f"synthetic: rates off by {df[0]:.1e} / {df[1]:.1e} Hz; pattern within {np.percentile(err, 99):.3f} mrad "
          f"(p99) of the truth; residual {fit.stats['rms_mrad']:.3f} mrad (noise {expect:.3f}), "
          f"{fit.stats['outliers']:.2%} outliers")
    assert np.all(df < 1e-3), f"rates off by {df} Hz"
    assert np.percentile(err, 99) < 0.1, f"pattern off by {np.percentile(err, 99):.3f} mrad"
    assert fit.stats["rms_mrad"] < 1.2 * expect, fit.stats["rms_mrad"]
    return fit


def simulator(model, dev):
    sim = SimSource(device=dev, motion="static", scene="box", rosette=model)
    sim.t = 37.0  # far from zero: the phases must not lose precision
    x, _, t, n = sim.step(20_000)  # a batch's phases are linear in time: the wander bends within ~0.25 s
    x, t = x.numpy()[:n], t.numpy()[:n]
    t64 = 37.0 + np.arange(n) / sim.rate
    got = rosette.measured_angles(x)
    want = model.angles(t64)
    err = np.abs(got - want) * 1000
    print(f"simulator: fires within {err.max():.3f} mrad of the model at t = 37 s")
    assert err.max() < 0.05, err.max()


def recording(path):
    fir = lvxr.read_firings(path)
    xyz, attr, t64, _ = fir.points(clean=False)
    nat_x, nat_a, nat_t = common.load_recording(path)
    t32 = t64.astype(np.float32)
    i = np.searchsorted(t32, nat_t)
    ok = np.zeros(len(nat_t), bool)
    for off in (0, 1, 2):  # dual returns share a time: try the neighbours too
        j = np.minimum(i + off, len(t32) - 1)
        ok |= (t32[j] == nat_t) & np.all(np.abs(xyz[j] - nat_x) < 1e-5, axis=1) & (attr[j] == nat_a)
    print(f"{path.name}: {len(fir.t):,} firings, {len(t64):,} returns, {fir.misses.mean():.1%} empty; the native "
          f"replay's {len(nat_t):,} points: {ok.mean():.4%} found here unchanged")
    assert ok.mean() > 0.999, ok.mean()
    model = rosette.fit_firings(fir, log=print)
    s = model.stats
    print(f"  pattern: {s['freq_hz'][0]:+.4f} Hz and {s['freq_hz'][1]:+.4f} Hz, residual {s['rms_mrad']:.2f} mrad, "
          f"phase wander {s['wander_mrad_pp'][0]:.1f} / {s['wander_mrad_pp'][1]:.1f} mrad peak to peak")
    print("  harmonics (deg): " + ", ".join(f"{k}: {v:.3f}" for k, v in s["harmonics_deg"].items()))
    assert s["rms_mrad"] < 3.0, "the pattern does not explain the directions"


def main():
    dev = common.device()
    fit = synthetic()
    simulator(fit, dev)
    rec = common.any_recording()
    if rec is None:
        print("(no recording: the reader and real-pattern part did not run; set LIVOX_DATA_DIR / LIVOX_RECORDING)")
        return
    recording(rec)


if __name__ == "__main__":
    sys.exit(main())
