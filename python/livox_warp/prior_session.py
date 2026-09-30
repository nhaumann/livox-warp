"""Localisation in a prior map, kept up to date while the viewer runs (render thread; host-side logic).

The viewer's world frame W stays what it is (the mount pose, or the odometry's first frame), so the
live ring, the integration map and the pose table never need rewriting. This class estimates T_MW,
world -> prior map, and the viewer places the prior map (drawing, Changes distances) with it:

1. Localise: a scan of the last `loc_s` seconds in the current sensor frame goes to the prior-map
   worker's global search (about a second). With odometry on, the points are placed by the frame poses
   (a moving sensor is fine); without it the sensor must stand still. T_MW = T_map_sensor @ inv(T_W_sensor).
2. Track: every `track_every` seconds the last `track_window` seconds of points, in W, are registered
   to the prior from the current T_MW (multi-start point-to-plane ICP, milliseconds on the worker). A
   result moves T_MW if it fits (>= min_fit of the points within 3 cm), moves T_MW by less than
   max_jump / max_rot, and is either small where the scan already fits well (the odometry's slow drift:
   < small_jump / small_rot at a fit of routine_fit or more) or fits clearly better than T_MW as it was
   (by min_gain). The last condition matters in a degenerate view: facing one wall, a registration can
   slide along it with no change in fit, and accepting those slides would walk the pose away while the
   odometry itself holds. An accepted correction is also applied no faster than rate_t / rate_rot,
   unless the fit rises strongly (strong_gain).
3. Lost: `lost_after` poor fits in a row mark the pose lost. First the last pose is refined from a wide
   set of nearby starts (STARTS_WIDE: shifts, heights and headings; the last `loc_s` seconds of points):
   most losses are a fast swing or a moment of featureless view, and the scanner is still near where it
   was. Only if that does not fit well does the global search run (every `relocalise_every` s until it
   gives a unique answer near the last good alignment); a global search alone would compete every
   look-alike place in the building.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .prior_worker import PriorWorker
from .sources import clean_returns

STATES = ("loading", "allocating", "building", "ready", "localising", "tracking", "ambiguous", "lost", "error")
"""Every value PriorSession.state takes; the transitions are in the class docstring."""


def _print_log(text: str, level: str):
    print(f"[{level}] {text}")


def _angle_deg(R: np.ndarray) -> float:
    return math.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def deskew(p: np.ndarray, s: np.ndarray, xi: np.ndarray | None) -> np.ndarray:
    """gpu.deskew on the host: points measured a fraction s of a frame after mid-frame -> the mid-frame frame.
    xi = (rho, theta) is the sensor's body motion over one frame; None or zero leaves the points alone."""
    if xi is None or not np.any(xi):
        return p
    rv = s[:, None] * np.asarray(xi[3:], dtype=np.float64)[None, :]
    return Rotation.from_rotvec(rv).apply(p) + s[:, None] * np.asarray(xi[:3], dtype=np.float64)[None, :]


@dataclass(frozen=True, eq=False)
class _Job:
    """What a worker job was started from; it comes back with the job's result."""

    kind: str  # "loc" (global search), "track" (drift correction) or "reacquire" (wide local search when lost)
    T: np.ndarray  # "loc": T_ref, the sensor pose in W the scan is relative to; else T_MW as it was when queued
    t: float  # wall time when queued
    p_sensor: np.ndarray | None = None  # "track": the sensor position in the map, the pivot of a limited step


@dataclass(frozen=True, eq=False)
class TrackStep:
    """One tracking result as it was judged (PriorSession.history)."""

    t: float
    fit: float  # the fit of the alignment kept
    fit_new: float  # the fit of the registration's answer
    gain: float  # fit_new minus the fit of T_MW as it was
    jump: float  # m the answer moved T_MW's sensor position
    rot: float  # deg
    accepted: bool
    reached: bool  # False when the rate limit let T_MW move only part of the way
    odom_note: str  # the viewer's odometry stats at the time


class PriorSession:
    """The session's state machine (`state`, one of STATES):

        loading -> allocating -> building -> ready     the worker loads and builds the map (error at any step)
        ready | ambiguous -> localising -> tracking    a unique global answer
                                        -> ambiguous   no unique answer and no alignment yet
                                                       (accept_ambiguous() takes the answer anyway)
        tracking -> lost                               lost_after poor fits in a row; T_MW is kept
        lost -> tracking                               re-acquired near the last pose, a good track, or a unique
                                                       global answer near the last good alignment
        any -> ready                                   forget()

    While "lost", the local re-acquisition and then the global search run without changing the state.
    """

    STARTS = ((0.2, 0, 0), (-0.2, 0, 0), (0, 0.2, 0), (0, -0.2, 0), (0, 0, 0.2), (0, 0, -0.2))
    # re-acquiring near the last pose after a loss: shifts, heights and headings (x, y, z m, heading deg); as many
    # as one refine batch takes
    STARTS_WIDE = ((0.35, 0, 0), (-0.35, 0, 0), (0, 0.35, 0), (0, -0.35, 0), (0.7, 0, 0), (-0.7, 0, 0),
                   (0, 0.7, 0), (0, -0.7, 0), (0, 0, 0.25), (0, 0, -0.25), (0, 0, 0, 10.0), (0, 0, 0, -10.0),
                   (0, 0, 0, 20.0), (0, 0, 0, -20.0))

    def __init__(self, path: str, device=None, log=None):
        """log(text, level) with level "info", "ok" or "warn"; the default prints."""
        self.path = path
        self.log = log or _print_log
        self.worker = PriorWorker(path, device)
        self.T_MW = None  # world -> prior map; None until localised
        self._state = "loading"
        self.detail = ""
        self.fit = 0.0  # last tracking fit (fraction of points within 3 cm)
        self.corrections = 0
        self.rejected = 0
        self.reacquired = 0
        self.poor = 0  # poor fits in a row
        self.auto = True  # localise as soon as there is data
        self.track_on = True
        self.frame_dt = 0.1
        # tuning (see the module docstring)
        self.keep_s = 3.0
        self.loc_s = 2.0
        self.track_every = 0.5
        self.track_window = 1.0
        self.pts_per_frame = 4000
        self.min_fit = 0.35
        self.lost_fit = 0.3
        self.lost_after = 3
        self.max_jump = 0.5
        self.max_rot = 5.0
        # routine drift: corrections this small, where the scan already fits well, need no fit gain (at a high fit
        # there is none left to show); the degenerate slides come with mediocre fits
        self.small_jump = 0.10
        self.small_rot = 2.0
        self.routine_fit = 0.6
        self.min_gain = 0.03  # anything else must raise the 3 cm fit by this much
        # and the alignment moves toward an accepted result at most this fast (about the sensor): faster than the
        # odometry drifts, too slow for chance fit gains in a degenerate view to walk the pose away
        self.rate_t = 0.05  # m/s
        self.rate_rot = 1.0  # deg/s
        self.strong_gain = 0.15  # unless the fit rises this much, to at least routine_fit: then all at once
        # after a loss, a new global answer must be near the last good alignment: the scanner cannot jump, and a
        # building has look-alike places (mirror-image rooms) that a search can pick
        self.reloc_near_m = 1.0
        self.reloc_near_mps = 0.5
        self.reloc_near_deg = 20.0
        self.reloc_near_dps = 10.0
        self.relocalise_every = 2.0
        self.reacquire_fit = 0.45  # a wide local search can land in a wrong basin: ask for a clearly good fit
        self.history = collections.deque(maxlen=2000)  # TrackStep per tracking result
        self.odom_note = ""  # the viewer's latest odometry stats, recorded with every TrackStep
        self._batches = collections.deque()  # (t_min, t_max, xyz float32, t float32) of recent clean points
        self._frames = collections.OrderedDict()  # frame id -> (T_W sensor pose at mid-frame, xi)
        self._want_loc = False
        self._next_track = 0.0
        self._next_loc = 0.0
        self._rng = np.random.default_rng(0)
        # judged about the current alignment; _reset_tracking() clears them
        self._last_step = None  # time of the last accepted correction
        self._last_good = None  # (time, T_MW) while the fit was good
        self._pending_pose = None  # (T_map_sensor, T_ref) of the last ambiguous localisation
        self._reacquire = False  # lost, and the local re-acquisition has not been tried yet

    # ---- state ---------------------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str):
        if value not in STATES:
            raise ValueError(f"unknown prior-map state {value!r}")
        self._state = value

    @property
    def ready(self) -> bool:
        """The map is built: localise and track jobs are accepted."""
        return self.worker.ready

    @property
    def pending(self) -> bool:
        """A localise or track job is queued or running."""
        return self.worker.pending > 0

    @property
    def draw(self):
        """(xyz float32, rgba uint8) of the map thinned for drawing, once loaded; None before."""
        return self.worker.draw

    @property
    def grid(self):
        """The map's PriorGrid once built (for Changes distances), else None."""
        return self.worker.grid if self.worker.ready else None

    @property
    def localised(self) -> bool:
        return self.T_MW is not None

    @property
    def has_pending_pose(self) -> bool:
        """A non-unique localisation is waiting for accept_ambiguous()."""
        return self._pending_pose is not None

    def status(self) -> dict:
        ws = self.worker.status()
        return {"state": self.state, "detail": self.detail, "fit": self.fit, "corrections": self.corrections,
                "rejected": self.rejected, "reacquired": self.reacquired, "points": ws["points"], "mib": ws["mib"],
                "busy": ws["busy"], "load_s": ws["load_s"], "build_s": ws["build_s"]}

    # ---- inputs (render thread) ----------------------------------------------------------------

    def add_points(self, xyz: np.ndarray, attr: np.ndarray, t: np.ndarray, n: int):
        """Raw sensor-frame points as they arrive (copied; noise-tagged and very close returns dropped)."""
        if n <= 0 or self.worker.state == "error":
            return
        x = np.asarray(xyz[:n], dtype=np.float32)
        tt = np.asarray(t[:n], dtype=np.float32)
        keep = clean_returns(x, np.asarray(attr[:n]))
        if not keep.any():
            return
        x, tt = x[keep].copy(), tt[keep].copy()
        self._batches.append((float(tt.min()), float(tt.max()), x, tt))
        t_end = float(tt.max())
        while self._batches and self._batches[0][1] < t_end - self.keep_s:
            self._batches.popleft()

    def add_frame(self, k: int, T_W: np.ndarray, xi: np.ndarray | None):
        """An odometry frame: its mid-frame sensor pose in W and its body motion (for deskewing)."""
        self._frames[k] = (np.array(T_W, dtype=np.float64), None if xi is None else np.array(xi, dtype=np.float64))
        while len(self._frames) > 64:
            self._frames.popitem(last=False)

    def clock_restarted(self):
        """The timestamps jumped (replay loop): the buffered points and frame ids belong to the old clock."""
        self._batches.clear()
        self._frames.clear()

    def world_reset(self, T_W_old_now: np.ndarray | None, T_W_new_now: np.ndarray):
        """W was redefined (odometry reset, mount change) with the sensor where it is: keep the map pose."""
        self._frames.clear()
        if self.T_MW is not None and T_W_old_now is not None:
            fix = T_W_old_now @ np.linalg.inv(T_W_new_now)
            self.T_MW = self.T_MW @ fix
            if self._last_good is not None:
                self._last_good = (self._last_good[0], self._last_good[1] @ fix)

    def localise_now(self):
        self._want_loc = True

    def forget(self):
        """Drop the alignment: the next search may land anywhere."""
        self.T_MW = None
        self._reset_tracking()
        if self.worker.ready:
            self.state = "ready"
        self.detail = ""

    def accept_ambiguous(self):
        """Use the last non-unique localisation anyway (the user can tell from the Changes colours)."""
        if self._pending_pose is None:
            return
        T_ms, T_ref = self._pending_pose
        self._reset_tracking()
        self.T_MW = T_ms @ np.linalg.inv(T_ref)
        self.state, self.detail = "tracking", "accepted an ambiguous pose"

    def close(self):
        self.worker.close()

    # ---- per frame (render thread) -------------------------------------------------------------

    def tick(self, now_wall: float, T_W_now: np.ndarray, odom: bool):
        """Advance: allocate/build the map, hand out jobs, take results. T_W_now: the sensor pose in W now."""
        w = self.worker
        w.tick()
        if w.state == "error":
            self.state, self.detail = "error", w.error
            return
        if not w.ready:
            self.state, self.detail = w.state, w.busy
            return
        if self.state in ("loading", "allocating", "building"):
            self.state = "ready"
            self.log(f"prior map ready: {w.grid.n:,} points, grid {w.grid.gpu_bytes() / 2**20:.0f} MiB, "
                     f"built in {w.build_s:.2f} s", "ok")
        for item in w.poll():
            self._result(item)
        if w.pending:
            return
        if self._reacquire and self.T_MW is not None and not self._want_loc:
            self._reacquire = False
            sub = self._world_points(self.loc_s, odom, T_W_now)
            if sub is not None and w.track(sub, self.T_MW, starts=self.STARTS_WIDE, switch_margin=0.0,
                                           ctx=_Job("reacquire", self.T_MW.copy(), now_wall)):
                self.detail = "lost: searching near the last pose"
                return
        due = now_wall >= self._next_loc
        want_loc = self._want_loc or (self.auto and self.T_MW is None and due) or (self.state == "lost" and due)
        if want_loc:
            scan, T_ref = self._scan_sensor_frame(odom, T_W_now)
            if scan is not None:
                up = None
                if self.T_MW is not None:  # the map's up in the sensor frame, to order the gravity candidates
                    up = np.linalg.inv(self.T_MW @ T_ref)[:3, :3] @ np.array([0.0, 0.0, 1.0])
                if w.localize(scan, up_prior=up, ctx=_Job("loc", T_ref, now_wall)):
                    self._want_loc = False
                    self._next_loc = now_wall + self.relocalise_every
                    if self.state != "lost":
                        self.state = "localising"
            return
        if self.T_MW is not None and self.track_on and now_wall >= self._next_track:
            sub = self._world_points(self.track_window, odom, T_W_now)
            if sub is not None:
                p_sensor = (self.T_MW @ T_W_now)[:3, 3].copy()
                w.track(sub, self.T_MW, starts=self.STARTS, ctx=_Job("track", self.T_MW.copy(), now_wall, p_sensor))
                self._next_track = now_wall + self.track_every

    # ---- point sets ----------------------------------------------------------------------------

    def _points_since(self, seconds: float):
        if not self._batches:
            return None, None
        t_end = self._batches[-1][1]
        sel = [b for b in self._batches if b[1] >= t_end - seconds]
        x = np.concatenate([b[2] for b in sel])
        t = np.concatenate([b[3] for b in sel])
        keep = t >= t_end - seconds
        return x[keep], t[keep]

    def _world_points(self, seconds: float, odom: bool, T_W_now: np.ndarray):
        """The recent points in W: placed by their frames' poses with odometry, by the fixed pose without."""
        x, t = self._points_since(seconds)
        if x is None or len(x) < 500:
            return None
        if not odom:
            if len(x) > 60000:
                x = x[self._rng.choice(len(x), 60000, replace=False)]
            return x.astype(np.float64) @ T_W_now[:3, :3].T + T_W_now[:3, 3]
        k = np.floor(t / self.frame_dt).astype(np.int64)
        out = []
        for kk in np.unique(k):
            f = self._frames.get(int(kk))
            if f is None:
                continue
            T, xi = f
            sel = np.nonzero(k == kk)[0]
            if len(sel) > self.pts_per_frame:
                sel = self._rng.choice(sel, self.pts_per_frame, replace=False)
            s = t[sel] / self.frame_dt - kk - 0.5
            p = deskew(x[sel].astype(np.float64), s, xi)
            out.append(p @ T[:3, :3].T + T[:3, 3])
        if not out:
            return None
        pts = np.concatenate(out)
        return pts if len(pts) >= 500 else None

    def _scan_sensor_frame(self, odom: bool, T_W_now: np.ndarray):
        """A scan for the global search in the current sensor frame, and the W pose it is relative to."""
        if odom and self._frames:
            T_ref = next(reversed(self._frames.values()))[0]
        else:
            T_ref = T_W_now
        pw = self._world_points(self.loc_s, odom, T_ref)
        if pw is None:
            return None, None
        span = self._batches[-1][1] - self._batches[0][0] if self._batches else 0.0
        if span < min(self.loc_s, self.keep_s) * 0.7:
            return None, None  # not enough history yet
        ps = (pw - T_ref[:3, 3]) @ T_ref[:3, :3]
        return ps.astype(np.float32), T_ref

    # ---- results -------------------------------------------------------------------------------

    def _reset_tracking(self):
        """Forget everything judged about the old alignment (a new one is set, or none)."""
        self.fit = 0.0
        self.poor = 0
        self._last_step = None
        self._last_good = None
        self._pending_pose = None
        self._reacquire = False

    def _result(self, item):
        kind = item[0]
        if kind == "error":
            self.detail = item[1]
            if self.state == "localising":
                self.state = "ready"
        elif kind == "pose":
            self._on_pose(*item[1:])
        elif item[4].kind == "reacquire":
            self._on_reacquire(*item[1:])
        else:
            self._on_track(*item[1:])

    def _on_pose(self, T_ms: np.ndarray, info: dict, job: _Job):
        """A global search's answer: T_map_sensor for the scan relative to job.T (= T_ref)."""
        txt = (f"inliers {info.get('inliers', 0):.2f}, runner-up {info.get('runner_up', 0):.2f}, "
               f"{info.get('seconds', 0):.1f} s")
        if bool(info.get("unique")):
            if self._last_good is not None and not self._near_last_good(T_ms, job):
                return
            self._reset_tracking()
            self.T_MW = T_ms @ np.linalg.inv(job.T)
            self.state, self.detail = "tracking", f"localised: {txt}"
            self.log(f"localised in the prior map ({txt})", "ok")
        else:
            self.state = "lost" if self.T_MW is not None else "ambiguous"
            self.detail = f"no unique answer ({txt})"
            self._pending_pose = (T_ms, job.T)
            self.log(f"prior map: no unique pose ({txt}); retrying", "warn")

    def _near_last_good(self, T_ms: np.ndarray, job: _Job) -> bool:
        """Is a new global answer where the last good alignment (allowing for the time since) says the sensor is?"""
        t_good, T_good = self._last_good
        el = max(0.0, job.t - t_good)
        T_cand = T_ms @ np.linalg.inv(job.T)
        dist = float(np.linalg.norm(T_ms[:3, 3] - (T_good @ job.T)[:3, 3]))
        dang = _angle_deg((T_cand @ np.linalg.inv(T_good))[:3, :3])
        near_m = self.reloc_near_m + self.reloc_near_mps * el
        near_deg = self.reloc_near_deg + self.reloc_near_dps * el
        if dist <= near_m and dang <= near_deg:
            return True
        self.detail = (f"a match {dist:.1f} m / {dang:.0f} deg from the last good pose was ignored "
                       f"(Forget pose to search anywhere)")
        self.log(f"prior map: ignored a match {dist:.1f} m / {dang:.0f} deg from the last good pose; retrying", "warn")
        return False

    def _on_reacquire(self, T_new: np.ndarray, fit: np.ndarray, info: dict, job: _Job):
        """The wide local search after a loss: back on track if it fits clearly, else the global search is due."""
        if self.T_MW is None or not np.allclose(job.T, self.T_MW) or self.state != "lost":
            return
        self.fit = float(fit[0])
        if self.fit >= self.reacquire_fit:
            d = np.linalg.inv(self.T_MW) @ T_new
            self.T_MW = T_new
            self.reacquired += 1
            self.state, self.poor = "tracking", 0
            self.detail = f"re-acquired near the last pose: fit {self.fit:.2f}"
            self.log(f"prior map: re-acquired near the last pose (fit {self.fit:.2f}, moved "
                     f"{np.linalg.norm(d[:3, 3]) * 100:.0f} cm / {_angle_deg(d[:3, :3]):.1f} deg)", "ok")
        else:
            self.detail = f"not near the last pose (best fit {self.fit:.2f}): searching the whole map"
            self.log(f"prior map: not near the last pose (best fit {self.fit:.2f}); searching the whole map", "warn")
            self._next_loc = 0.0

    def _on_track(self, T_new: np.ndarray, fit: np.ndarray, info: dict, job: _Job):
        """A drift-correction result: gate it (see the module docstring), step toward it, judge the pose kept."""
        if self.T_MW is None or not np.allclose(job.T, self.T_MW):
            return  # the pose was replaced (localisation, forget) while this ran
        fit_new = float(fit[0])
        fit_kept = float(info.get("fit_start", 0.0))  # the fit of T_MW as it is
        d = np.linalg.inv(self.T_MW) @ T_new
        jump, rot = float(np.linalg.norm(d[:3, 3])), _angle_deg(d[:3, :3])
        small = jump < self.small_jump and rot < self.small_rot and fit_new >= self.routine_fit
        gain = fit_new - fit_kept
        accepted = (fit_new >= self.min_fit and jump < self.max_jump and rot < self.max_rot
                    and (small or gain >= self.min_gain))
        reached = True
        if accepted:
            strong = gain >= self.strong_gain and fit_new >= self.routine_fit
            dt = self.track_every  # the time the correction covers: since the last accepted one, at most 2 s
            if self._last_step is not None:
                dt = min(max(job.t - self._last_step, self.track_every), 2.0)
            self.T_MW, reached = self._step_toward(T_new, job.p_sensor, dt, strong)
            self._last_step = job.t
            self.corrections += 1
            if reached:
                fit_kept = fit_new
        else:
            self.rejected += 1
        # judge the pose that is kept: a good fit somewhere the gates refused says nothing about it
        self.fit = fit_kept
        if self.fit >= self.min_fit and self.state != "lost":
            self._last_good = (job.t, self.T_MW.copy())
        self.history.append(TrackStep(job.t, self.fit, fit_new, gain, jump, rot, accepted, reached, self.odom_note))
        self.poor = 0 if self.fit >= self.lost_fit else self.poor + 1
        if self.poor >= self.lost_after and self.state != "lost":
            last = list(self.history)[-self.lost_after:]
            steps = "; ".join(f"fit {h.fit_new:.2f}, {h.jump * 100:.0f} cm / {h.rot:.1f} deg, "
                              f"{'accepted' if h.accepted else 'rejected'}" for h in last)
            self.state, self.detail = "lost", f"fit {self.fit:.2f}: searching near the last pose"
            self.log(f"prior map: tracking lost, searching near the last pose (last steps: {steps})", "warn")
            self._reacquire = True
            self._next_loc = 0.0
        elif self.state != "lost":
            self.state = "tracking"
            self.detail = f"fit {self.fit:.2f}, {self.corrections} corrections"
        elif accepted and self.fit >= self.reacquire_fit:
            self.state, self.poor = "tracking", 0  # back on the scan by itself
            self.detail = f"back on the scan: fit {self.fit:.2f}"
            self.log(f"prior map: back on the scan (fit {self.fit:.2f})", "ok")

    def _step_toward(self, T_new: np.ndarray, p_map: np.ndarray | None, dt: float, full: bool):
        """Move T_MW toward T_new, the rotation about the sensor position p_map, at most rate_t * dt and
        rate_rot * dt unless full (or there is no pivot). Returns (the new T_MW, whether it reached T_new)."""
        if full or p_map is None:
            return T_new, True
        D = T_new @ np.linalg.inv(self.T_MW)  # map-frame change
        p = np.asarray(p_map, dtype=np.float64)
        u = D[:3, :3] @ p + D[:3, 3] - p  # how far it moves the sensor
        rv = Rotation.from_matrix(D[:3, :3]).as_rotvec()
        reached = True
        a, amax = np.linalg.norm(rv), math.radians(self.rate_rot) * dt
        if a > amax:
            rv, reached = rv * (amax / a), False
        n, nmax = np.linalg.norm(u), self.rate_t * dt
        if n > nmax:
            u, reached = u * (nmax / n), False
        R = Rotation.from_rotvec(rv).as_matrix()
        D2 = np.eye(4)
        D2[:3, :3] = R
        D2[:3, 3] = p - R @ p + u  # rotate about the sensor, then move it by u
        return D2 @ self.T_MW, reached
