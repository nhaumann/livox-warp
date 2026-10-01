"""The Mid-40's scan pattern: a two-prism (Risley) rosette fitted to a recording's own firing directions.

The Mid-40 steers its beam with two wedge prisms spinning at different rates. In the angle domain
z = u + i v (u = atan(y / x), v = atan(z / x) of the firing direction) a thin-prism pair draws
z(t) = c_a exp(i theta_a(t)) + c_b exp(i theta_b(t)), and real prisms add small terms at combinations of the two
angles. The model here is

    z(t) = sum_k c_k exp(i (m_k theta_a(t) + n_k theta_b(t))),    theta_j(t) = omega_j (t - t0) + wander_j(t)

with complex c_k over a set of harmonics (m_k, n_k), the prism rates omega_a and omega_b (signed: the prisms may
turn opposite ways), and a slow phase wander per prism (motor speed control is not perfect), piecewise linear
between windows. sources.rosette_dir is this model with two harmonics and no wander (RosetteModel.sim_default).

fit() needs nothing but firing times and directions:
  1. the two rates from the two strongest peaks of z's spectrum: the firings sit on a 10 us lattice, so an FFT of
     z on that lattice works straight through dropped packets and empty firings;
  2. Gauss-Newton on the rates over the first seconds (growing spans), the coefficients solved linearly at
     every step (variable projection), outliers down-weighted;
  3. harmonic pursuit: the strongest line left in the residual's spectrum is matched to an integer combination
     m f_a + n f_b and added, until no line stands out (a real Mid-40 needs conjugate terms such as (-2, -1): its
     prisms are not thin, and the angle domain is not analytic in their phases);
  4. the phase wander of each prism, tracked window by window (50 ms) through the whole recording from the
     start, each window started from the last: the motors' speed control hunts at a few hertz and drifts by
     radians over a minute, which no constant rate follows; then the coefficients over everything, and the
     tracking once more.
Everything is float64: a prism's phase after a minute is ~6e4 rad, and float32 would lose milliradians.

What the fit gives the other solvers:
  - the prism phases of every firing: walkfit.py models the sensor's angular distortion as a function of them;
  - the direction of firings that returned nothing (cartesian data carries none): occupancy.py reads free space
    along them;
  - a simulator that scans with this unit's pattern (sources.SimSource(rosette=...));
  - the residual itself: how repeatable the pattern and the timestamps are, in milliradians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

MAX_HARMONICS = 8  # walkfit's distortion model unrolls a loop of this length (it takes the pattern's first ones)
BASE_HARMONICS = ((1, 0), (0, 1), (0, 0))  # the two prisms and the pattern's centre (boresight offset)
DEFAULT_HARMONICS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, -1), (-1, 2))  # a fixed set, for tests
PURSUIT_ORDER = 3  # |m|, |n| at most this in the harmonics the pursuit may add
PURSUIT_MAX = 24  # harmonics in all
PURSUIT_SNR = 200.0  # a residual line this many times the median spectral power is a harmonic worth adding
LATTICE_DT = 1e-5  # s between Mid-40 firings
SHORT_SPAN = 6.0  # s: the rates and harmonics are fitted over this much; the phase tracker follows the rest
SPANS = (2.0, SHORT_SPAN)  # Gauss-Newton time spans (s), each fitted from the last one's rates


@dataclass
class RosetteModel:
    omega: np.ndarray  # (2,) rad/s, signed
    harmonics: np.ndarray  # (K, 2) int: (m, n) of each term
    coef: np.ndarray  # (K,) complex128, rad
    t0: float = 0.0  # phase origin (s)
    wander_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (W,) window centres (s)
    wander: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (W, 2) rad
    stats: dict = field(default_factory=dict)

    @classmethod
    def sim_default(cls) -> RosetteModel:
        """The pattern sources.rosette_dir draws: 153.7 Hz and 97.3 Hz the other way, 19.2 deg half angle."""
        half = math.radians(19.2)
        return cls(omega=np.array([2 * math.pi * 153.7, -2 * math.pi * 97.3]),
                   harmonics=np.array([[1, 0], [0, 1]]), coef=np.array([half / 2, half / 2], np.complex128))

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2 * math.pi)

    def phases(self, t) -> np.ndarray:
        """(n, 2) prism phases theta_a, theta_b at times t (float64 s)."""
        t = np.asarray(t, dtype=np.float64)
        th = (t - self.t0)[:, None] * self.omega[None, :]
        if len(self.wander_t):
            th[:, 0] += np.interp(t, self.wander_t, self.wander[:, 0])
            th[:, 1] += np.interp(t, self.wander_t, self.wander[:, 1])
        return th

    def phases_mod(self, t) -> np.ndarray:
        """The phases reduced to [0, 2 pi), as float32 for the GPU (enough there: the harmonics are small
        multiples of them)."""
        return np.mod(self.phases(t), 2 * math.pi).astype(np.float32)

    def basis(self, t) -> np.ndarray:
        """(n, K) exp(i (m theta_a + n theta_b)) per harmonic."""
        return np.exp(1j * (self.phases(t) @ self.harmonics.T.astype(np.float64)))

    def angles(self, t) -> np.ndarray:
        """Modelled z = u + i v at times t."""
        return self.basis(t) @ self.coef

    def direction(self, t) -> np.ndarray:
        """(n, 3) unit firing directions (sensor frame) at times t."""
        z = self.angles(t)
        d = np.stack([np.ones(len(z)), np.tan(z.real), np.tan(z.imag)], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    # ---- persistence ------------------------------------------------------------------------------

    def save(self, path: str):
        np.savez(path, omega=self.omega, harmonics=self.harmonics, coef=self.coef, t0=self.t0,
                 wander_t=self.wander_t, wander=self.wander)

    @classmethod
    def load(cls, path: str) -> RosetteModel:
        z = np.load(path)
        return cls(omega=z["omega"], harmonics=z["harmonics"], coef=z["coef"], t0=float(z["t0"]),
                   wander_t=z["wander_t"], wander=z["wander"])

    # ---- the GPU side (the simulator) ---------------------------------------------------------------

    def gpu_params(self, t_start: float, t_end: float, device, cache: dict | None = None) -> RosetteParams:
        """RosetteParams for firings from t_start to t_end: the phases at t_start and the rates over the span,
        computed here in float64, so the kernel's float32 never sees a large phase."""
        cache = {} if cache is None else cache
        if "harm" not in cache:
            cache["harm"] = wp.array(self.harmonics.astype(np.int32), dtype=wp.vec2i, device=device)
            cache["coef"] = wp.array(np.stack([self.coef.real, self.coef.imag], 1).astype(np.float32),
                                     dtype=wp.vec2, device=device)
        th = self.phases(np.array([t_start, max(t_end, t_start + 1e-6)]))
        p = RosetteParams()
        p.phase0 = wp.vec2(*np.mod(th[0], 2 * math.pi).astype(np.float32))
        p.rate = wp.vec2(*((th[1] - th[0]) / max(t_end - t_start, 1e-6)).astype(np.float32))
        p.harm, p.coef, p.n_harm = cache["harm"], cache["coef"], cache["harm"].shape[0]
        return p


@wp.struct
class RosetteParams:
    """A RosetteModel for one batch of firings (gpu_params): phases at the batch start, rates over it."""

    phase0: wp.vec2
    rate: wp.vec2
    harm: wp.array(dtype=wp.vec2i)
    coef: wp.array(dtype=wp.vec2)
    n_harm: int


@wp.func
def rosette_dir_p(dt: float, p: RosetteParams) -> wp.vec3:
    """Firing direction (sensor frame) dt seconds after the batch start of p."""
    ta = p.phase0[0] + p.rate[0] * dt
    tb = p.phase0[1] + p.rate[1] * dt
    z = wp.vec2()
    for k in range(p.n_harm):
        h = p.harm[k]
        ang = float(h[0]) * ta + float(h[1]) * tb
        c = p.coef[k]
        cs = wp.cos(ang)
        sn = wp.sin(ang)
        z += wp.vec2(c[0] * cs - c[1] * sn, c[0] * sn + c[1] * cs)
    return wp.normalize(wp.vec3(1.0, wp.tan(z[0]), wp.tan(z[1])))


# ---- fitting ----------------------------------------------------------------------------------------


def measured_angles(direction: np.ndarray) -> np.ndarray:
    """z = u + i v of firing directions (n, 3), any length."""
    d = np.asarray(direction, dtype=np.float64)
    return np.arctan2(d[:, 1], d[:, 0]) + 1j * np.arctan2(d[:, 2], d[:, 0])


def _peaks(t: np.ndarray, z: np.ndarray, seconds: float, lattice: float, min_hz: float = 2.0):
    """The two strongest spectral lines of z (signed Hz) from the first `seconds` of data on the firing lattice."""
    seg = t < t[0] + seconds
    idx = np.round((t[seg] - t[0]) / lattice).astype(np.int64)
    n = int(idx.max()) + 1
    N = 1 << (2 * n - 1).bit_length()  # zero padding to twice the span: finer bins to interpolate between
    x = np.zeros(N, np.complex128)
    x[idx] = z[seg] - z[seg].mean()
    P = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(N, lattice)
    P[np.abs(f) < min_hz] = 0.0
    out = []
    df = f[1] - f[0]
    for _ in range(2):
        i = int(np.argmax(P))
        a, b, c = np.log(P[(i - 1) % N] + 1e-300), np.log(P[i] + 1e-300), np.log(P[(i + 1) % N] + 1e-300)
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0  # parabolic interpolation of the log power
        out.append((f[i] + frac * df, P[i]))
        width = max(4, int(4 * len(x) / max(n, 1)))  # the main lobe of a line, in padded bins
        P[max(0, i - width):i + width + 1] = 0.0
        if i - width < 0:
            P[(i - width) % N:] = 0.0
        if i + width >= N:
            P[:(i + width) % N + 1] = 0.0
    out.sort(key=lambda p: -p[1])
    return np.array([out[0][0], out[1][0]])


def _robust_weights(r: np.ndarray) -> np.ndarray:
    """Cauchy weights at three robust sigmas of |r|."""
    a = np.abs(r)
    s = 1.4826 * float(np.median(a)) + 1e-12
    return 1.0 / (1.0 + (a / (3.0 * s)) ** 2)


def _solve_coef(E: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(E * sw, z * sw[:, 0], rcond=None)[0]


def _gauss_newton(model: RosetteModel, t, z, iters: int = 6):
    """Rates by Gauss-Newton with the coefficients re-solved at every step; returns the robust weights."""
    H = model.harmonics.astype(np.float64)
    w = np.ones(len(t))
    for _ in range(iters):
        E = model.basis(t)
        model.coef = _solve_coef(E, z, w)
        r = z - E @ model.coef
        w = _robust_weights(r)
        dt = t - model.t0
        G = E * model.coef[None, :]
        J = np.stack([(1j * G * H[None, :, 0]).sum(1) * dt, (1j * G * H[None, :, 1]).sum(1) * dt], 1)
        A = np.real(J.conj().T @ (J * w[:, None]))
        b = np.real(J.conj().T @ (r * w))
        step = np.linalg.solve(A + 1e-9 * np.trace(A) * np.eye(2), b)
        model.omega = model.omega + step
        if np.all(np.abs(step) * max(float(dt.max()), 1e-3) < 1e-7):
            break
    E = model.basis(t)
    model.coef = _solve_coef(E, z, w)
    return w


def _track(model: RosetteModel, t, z, window: float, iters: int = 3):
    """The two prisms' phase wander, window by window from the start: each window is fitted (Gauss-Newton on two
    offsets, coefficients fixed) from the last one's offsets carried on at their recent rate. A phase that drifts
    by radians over a minute is followed that way, where one fit of all windows at once would need a good guess
    for each. Returns the robust weights of the final residual."""
    H = model.harmonics.astype(np.float64)
    lo, hi = float(t[0]), float(t[-1])
    nw = max(1, int(math.ceil((hi - lo) / window)))
    bounds = np.searchsorted(t, lo + window * np.arange(nw + 1))
    bounds[-1] = len(t)
    centres = lo + (np.arange(nw) + 0.5) * window
    model.wander_t, model.wander = np.zeros(0), np.zeros((0, 2))
    th0 = model.phases(t)
    delta = np.zeros((nw, 2))
    cur = np.zeros(2)
    rate = np.zeros(2)
    w_all = np.ones(len(t))
    for j in range(nw):
        a, b = bounds[j], bounds[j + 1]
        d = cur + rate * window
        if b - a >= 30:
            th, zz = th0[a:b], z[a:b]
            for _ in range(iters):
                G = np.exp(1j * ((th + d) @ H.T)) * model.coef[None, :]
                r = zz - G.sum(1)
                w = _robust_weights(r)
                Ja = (1j * G * H[None, :, 0]).sum(1)
                Jb = (1j * G * H[None, :, 1]).sum(1)
                A = np.array([[np.sum(w * np.abs(Ja) ** 2), np.sum(w * np.real(Ja.conj() * Jb))],
                              [0.0, np.sum(w * np.abs(Jb) ** 2)]])
                A[1, 0] = A[0, 1]
                rhs = np.array([np.sum(w * np.real(Ja.conj() * r)), np.sum(w * np.real(Jb.conj() * r))])
                if np.linalg.det(A) <= 1e-12 * A[0, 0] * A[1, 1]:
                    break
                d = d + np.linalg.solve(A, rhs)
            w_all[a:b] = w
            if j:
                rate = 0.5 * rate + 0.5 * (d - cur) / window
        delta[j] = d
        cur = d
    model.wander_t, model.wander = centres, delta
    return w_all


def _rms_mrad(r: np.ndarray, w: np.ndarray) -> float:
    inl = w > 0.5
    return float(np.sqrt(np.mean(np.abs(r[inl]) ** 2)) * 1000) if inl.any() else float("nan")


def _pursuit(model: RosetteModel, t, z, w, lattice: float, say) -> list:
    """Add the harmonics the residual's spectrum asks for (module docstring, step 3); returns those added."""
    span = min(4.0, float(t[-1] - t[0]))
    seg = t < t[0] + span
    idx = np.round((t[seg] - t[0]) / lattice).astype(np.int64)
    n = int(idx.max()) + 1
    N = 1 << (2 * n - 1).bit_length()
    added = []
    while len(model.harmonics) < PURSUIT_MAX:
        fa, fb = model.freq_hz
        have = [tuple(h) for h in model.harmonics.tolist()]
        f_have = np.array([m * fa + k * fb for m, k in have])
        cand = [(m, k) for m in range(-PURSUIT_ORDER, PURSUIT_ORDER + 1)
                for k in range(-PURSUIT_ORDER, PURSUIT_ORDER + 1)
                if (m, k) not in have and np.min(np.abs(f_have - (m * fa + k * fb))) > 3.0]  # not a wander sideband
        if not cand:
            break
        r = (z - model.angles(t)) * w
        x = np.zeros(N, np.complex128)
        x[idx] = r[seg]
        P = np.abs(np.fft.fft(x)) ** 2
        floor = float(np.median(P)) + 1e-300
        f_c = np.array([m * fa + k * fb for m, k in cand])
        b = np.round(f_c * N * lattice).astype(np.int64)
        power = np.max([P[(b + o) % N] for o in range(-2, 3)], axis=0)
        j = int(np.argmax(power))
        if power[j] < PURSUIT_SNR * floor:
            break
        model.harmonics = np.vstack([model.harmonics, np.array(cand[j])[None]])
        model.coef = _solve_coef(model.basis(t), z, w)
        added.append((cand[j], float(power[j] / floor)))
    if added:
        say("rosette: harmonics added from the residual: "
            + ", ".join(f"({m},{k}) x{snr:.0f}" for (m, k), snr in added))
    return added


def fit(t: np.ndarray, z: np.ndarray, harmonics=None, lattice: float = LATTICE_DT, wander_window: float = 0.05,
        max_points: int = 1_500_000, seed: int = 0, log=None) -> RosetteModel:
    """A RosetteModel from firing times t (float64 s) and measured angles z (measured_angles()). harmonics: a fixed
    set of (m, n), or None to start from BASE_HARMONICS and let the residual add what it needs."""
    say = log or (lambda s: None)
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.complex128)
    order = np.argsort(t, kind="stable")
    t, z = t[order], z[order]
    f_ab = _peaks(t, z, min(2.0, float(t[-1] - t[0])), lattice)
    f_ab = f_ab[np.argsort(-np.abs(f_ab))]  # prism a is the faster one: a fixed meaning for the harmonics
    say(f"rosette: spectral lines at {f_ab[0]:+.3f} Hz and {f_ab[1]:+.3f} Hz")
    harm = np.array(BASE_HARMONICS if harmonics is None else harmonics, np.int64).reshape(-1, 2)
    model = RosetteModel(omega=2 * math.pi * f_ab, harmonics=harm, coef=np.zeros(len(harm), np.complex128),
                         t0=float(t[0]))
    if len(t) > max_points:  # spread evenly in time, so every span keeps its share
        keep = np.sort(np.random.default_rng(seed).choice(len(t), max_points, replace=False))
        t_fit, z_fit = t[keep], z[keep]
    else:
        t_fit, z_fit = t, z

    def rates(spans):
        w = np.ones(len(t_fit))
        for span in spans:
            sel = t_fit < t_fit[0] + span
            w = np.ones(len(t_fit))
            w[sel] = _gauss_newton(model, t_fit[sel], z_fit[sel])
            if sel.all():
                break
        return w

    w = rates(SPANS)
    short = t_fit < t_fit[0] + SHORT_SPAN
    rms0 = _rms_mrad(z_fit[short] - model.angles(t_fit[short]), w[short])  # the rates alone, before any tracking
    w = _track(model, t_fit, z_fit, wander_window)
    model.coef = _solve_coef(model.basis(t_fit), z_fit, w)
    if harmonics is None:  # on the tracked phases the residual is down to what the harmonics leave
        _pursuit(model, t_fit[short], z_fit[short], w[short], lattice, say)
    for _ in range(2):  # track, then the coefficients over everything with the tracked phases, then track again
        w = _track(model, t_fit, z_fit, wander_window)
        model.coef = _solve_coef(model.basis(t_fit), z_fit, w)
    # the mean rate the tracked phases kept is the prisms' rate: fold it in, leaving the wander as the deviation
    dt = model.wander_t - model.t0
    for j in range(2):
        slope = float(np.polyfit(dt, model.wander[:, j], 1)[0]) if len(dt) > 2 else 0.0
        model.omega[j] += slope
        model.wander[:, j] -= slope * dt
    r = z_fit - model.angles(t_fit)
    w = _robust_weights(r)
    rms = _rms_mrad(r, w)
    sec = np.floor(t_fit - t_fit[0]).astype(np.int64)
    per_s = np.sqrt(np.bincount(sec, np.abs(r) ** 2 * (w > 0.5)) / np.maximum(np.bincount(sec, w > 0.5), 1)) * 1000
    model.stats = {"points": int(len(t_fit)), "span_s": float(t_fit[-1] - t_fit[0]), "freq_hz": model.freq_hz.tolist(),
                   "rms_mrad_untracked": rms0, "rms_mrad": rms, "outliers": float(np.mean(w <= 0.5)),
                   "wander_mrad_pp": (np.ptp(model.wander, axis=0) * 1000).tolist() if len(model.wander) else [0, 0],
                   "harmonics_deg": {f"{m},{n}": float(np.degrees(abs(c)))
                                     for (m, n), c in zip(model.harmonics.tolist(), model.coef)},
                   "rms_mrad_per_s": per_s.tolist()}
    say(f"rosette: {model.freq_hz[0]:+.4f} Hz / {model.freq_hz[1]:+.4f} Hz, {len(model.harmonics)} harmonics, "
        f"residual {rms:.2f} mrad ({rms0:.2f} on the first seconds untracked), "
        f"{model.stats['outliers']:.1%} outliers")
    return model


def fit_firings(firings, **kw) -> RosetteModel:
    """fit() over a recording's firings (lvxr.Firings) that have a direction."""
    d = firings.direction
    ok = np.isfinite(d[:, 0]) & (d[:, 0] > 0.5)
    return fit(firings.t[ok], measured_angles(d[ok]), lattice=firings.interval, **kw)
