"""Shared scaffolding for the scripts under tests/, benchmarks/ and tools/.

Importing this module puts the package (python/) on sys.path and initialises Warp quietly; nothing else
happens at import. No data is assumed: every recording and map comes from the environment, with the layout
below as the default, and a script whose data is absent calls skip() and exits 0.

    $LIVOX_DATA_DIR/                default: the repository root
        recordings/*.lvxr           recordings made with the viewer or the CLI; a name containing "static"
                                    (the scanner standing still) or "walk" (a handheld walk) is what
                                    static_recordings() and walk_recording() pick by default
        maps/prior_20mm.npz         the prior map: python -m livox_warp.prior_map convert scan.e57
        maps/recording_poses.npz    one 4x4 sensor-to-map pose per recording, keyed by pose_key():
                                    python -m livox_warp.localize
        maps/walk_reference.npz     a frame-by-frame reference for the walk: ref (N, 4, 4), fit (N,) and
                                    trusted_until (s), written by whoever generates it

Environment: LIVOX_DATA_DIR, LIVOX_PRIOR_MAP, LIVOX_STATIC_RECORDINGS (os.pathsep-separated paths),
LIVOX_WALK_RECORDING, LIVOX_RECORDING and LIVOX_DEVICE (default cuda:0); tests/README.md has the details.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from livox_warp import gpu  # noqa: E402
from livox_warp.sources import ReplaySource  # noqa: E402

wp.config.quiet = True
wp.init()

DATA_DIR = Path(os.environ.get("LIVOX_DATA_DIR", ROOT))
PRIOR_MAP_NAME = "prior_20mm.npz"
POSES_NAME = "recording_poses.npz"
WALK_REFERENCE_NAME = "walk_reference.npz"


# ---- data files ----------------------------------------------------------------------------------


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def _from_env(name: str) -> Path | None:
    """The path $name points at, or None (with a note) if the variable is set but the file is not there."""
    p = Path(os.environ[name])
    if not p.exists():
        print(f"{name}={p}: not found")
    return _existing(p)


def _recordings(substring: str = "") -> list[Path]:
    return sorted(p for p in (DATA_DIR / "recordings").glob("*.lvxr") if substring in p.name)


def recording(name: str) -> Path | None:
    """$LIVOX_DATA_DIR/recordings/<name>, if it exists."""
    return _existing(DATA_DIR / "recordings" / name)


def static_recordings() -> list[Path]:
    """Recordings of the scanner standing still: LIVOX_STATIC_RECORDINGS, else every *static*.lvxr."""
    env = os.environ.get("LIVOX_STATIC_RECORDINGS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return _recordings("static")


def walk_recording() -> Path | None:
    """A handheld walk starting at rest in the mapped area: LIVOX_WALK_RECORDING, else the first *walk*.lvxr."""
    if os.environ.get("LIVOX_WALK_RECORDING"):
        return _from_env("LIVOX_WALK_RECORDING")
    walks = _recordings("walk")
    return walks[0] if walks else None


def any_recording() -> Path | None:
    """Any recording at all: LIVOX_RECORDING, else the first *.lvxr."""
    if os.environ.get("LIVOX_RECORDING"):
        return _from_env("LIVOX_RECORDING")
    recs = _recordings()
    return recs[0] if recs else None


def prior_map() -> Path | None:
    """The prior map: LIVOX_PRIOR_MAP, else $LIVOX_DATA_DIR/maps/prior_20mm.npz."""
    if os.environ.get("LIVOX_PRIOR_MAP"):
        return _from_env("LIVOX_PRIOR_MAP")
    return _existing(DATA_DIR / "maps" / PRIOR_MAP_NAME)


def poses_file() -> Path | None:
    """$LIVOX_DATA_DIR/maps/recording_poses.npz, if it exists."""
    return _existing(DATA_DIR / "maps" / POSES_NAME)


def walk_reference() -> Path | None:
    """$LIVOX_DATA_DIR/maps/walk_reference.npz, if it exists."""
    return _existing(DATA_DIR / "maps" / WALK_REFERENCE_NAME)


def pose_key(path) -> str:
    """The key under which maps/recording_poses.npz stores a recording's pose: its file name, dots replaced."""
    return Path(path).name.replace(".", "_")


def saved_poses() -> dict:
    """The saved sensor-to-map poses keyed by pose_key(); empty when the file is absent."""
    p = poses_file()
    return dict(np.load(p)) if p is not None else {}


def static_recording(poses: dict | None = None) -> Path | None:
    """The first static recording, preferring one that has a saved pose."""
    recs = static_recordings()
    poses = saved_poses() if poses is None else poses
    return next((r for r in recs if pose_key(r) in poses), recs[0] if recs else None)


def load_walk_reference(path, frame_dt: float = 0.1):
    """The walk reference: (ref (N, 4, 4) per frame, trusted (N,) bool, trusted_until s).

    A frame is trusted where the reference is solid (over 80% of its points within 3 cm of the scan and no
    implausible step to either neighbour) and before trusted_until, past which whoever generated the file
    found the reference itself unreliable."""
    R = np.load(path)
    ref, fit = R["ref"], R["fit"]
    trusted_until = float(R["trusted_until"]) if "trusted_until" in R else np.inf
    step = np.r_[0.0, np.linalg.norm(np.diff(ref[:, :3, 3], axis=0), axis=1)]
    trusted = (fit > 0.8) & (step < 0.3) & (np.r_[step[1:], 0.0] < 0.3)
    trusted &= np.arange(len(ref)) * frame_dt < trusted_until
    return ref, trusted, trusted_until


def skip(reason: str, *env: str):
    """Print a SKIP line and exit 0: the data this script needs is not there.

    env: the variables that would point at it, listed after LIVOX_DATA_DIR."""
    print(f"SKIP: {reason} (set {' / '.join(('LIVOX_DATA_DIR',) + env)})")
    sys.exit(0)


# ---- GPU ------------------------------------------------------------------------------------------


def device():
    """The Warp device the scripts run on: LIVOX_DEVICE, default cuda:0."""
    return wp.get_device(os.environ.get("LIVOX_DEVICE", "cuda:0"))


def view(now: float, pipe: gpu.Pipeline, persist: float = 1.0, voxel: float = 0.05, **overrides) -> gpu.View:
    """A View with the viewer's permissive defaults: everything in range, no crop, the map at `voxel` metres
    (its mask from the pipeline). Any other View field is set by keyword: noise_mask=0b0001, dyn_on=1, ..."""
    v = gpu.View()
    v.now, v.persist = now, persist
    v.min_range, v.max_range, v.min_refl = 0.1, 500.0, 0
    v.noise_mask, v.ret_mask, v.crop_on = 0b1111, 0b1111, 0
    v.crop_lo, v.crop_hi = wp.vec3(-1e9, -1e9, -1e9), wp.vec3(1e9, 1e9, 1e9)
    v.inv_voxel, v.map_mask = 1.0 / voxel, pipe.map_cap - 1
    for name, value in overrides.items():
        setattr(v, name, value)
    return v


def shade(now: float, mode: int, **overrides) -> gpu.Shade:
    """A Shade for colour `mode` with the sensor at the origin and a fixed eye; other fields by keyword."""
    s = gpu.Shade()
    s.mode, s.lo, s.hi, s.solid, s.now = mode, 0.0, 150.0, wp.vec3(1, 1, 1), now
    s.sensor, s.eye = wp.vec3(0, 0, 0), wp.vec3(-5, 0, 2)
    for name, value in overrides.items():
        setattr(s, name, value)
    return s


# ---- geometry -------------------------------------------------------------------------------------


def rot_angle(R: np.ndarray) -> float:
    """The angle of a 3x3 rotation, degrees."""
    return math.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def pose_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(metres, degrees) from pose A to pose B (4x4)."""
    d = np.linalg.inv(A) @ B
    return float(np.linalg.norm(d[:3, 3])), rot_angle(d[:3, :3])


# ---- recordings and sessions ----------------------------------------------------------------------


def load_recording(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a recording to numpy: (xyz, attr, t) in time order, in the sensor frame."""
    src = ReplaySource(str(path), speed=200.0, looped=False)
    xs, ats, ts = [], [], []
    idle = 0
    while idle < 40:
        time.sleep(0.02)
        b = src.poll()
        if b is None:
            idle += 1 if src.replay.finished() else 0
            continue
        idle = 0
        xyz, attr, t, n = b
        xs.append(xyz[:n].copy())
        ats.append(attr[:n].copy())
        ts.append(t[:n].copy())
    src.close()
    x, a, t = np.concatenate(xs), np.concatenate(ats), np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return x[order], a[order], t[order]


def wait_ready(session, timeout: float = 300.0):
    """Tick a PriorSession until its map is loaded and its grid built; raise if that fails."""
    deadline = time.perf_counter() + timeout
    while not session.ready:
        if session.state == "error":
            raise RuntimeError(f"prior map: {session.detail}")
        if time.perf_counter() > deadline:
            raise TimeoutError("the prior map did not become ready")
        session.tick(0.0, np.eye(4), False)
        time.sleep(0.01)


def drive(session, x, a, t, worker=None, pipe=None, on_frame=None, on_batch=None, batch: int = 2000,
          frame_dt: float = 0.1, wait: float = 5.0) -> np.ndarray:
    """Feed a recording through a PriorSession, and through an OdomWorker on `pipe` if given, the way the
    viewer's frame loop does. The session schedules on sensor time; after each batch its job is waited for
    (at most `wait` s: live, the worker keeps up). on_frame(result, session) runs per odometry frame with
    the worker's FrameResult, on_batch(now, session) after each batch. Returns the last odometry pose."""
    odom = worker is not None
    T_now = np.eye(4)
    for i in range(0, len(t), batch):
        xs, as_, ts = x[i:i + batch], a[i:i + batch], t[i:i + batch]
        n = len(ts)
        session.add_points(xs, as_, ts, n)
        if odom:
            sx, sa, st, _ = pipe.stage(xs, as_, ts, n, 1.0 / frame_dt)
            worker.submit(sx, sa, st, n, float(ts.min()), float(ts.max()))
            for r in worker.poll():
                session.add_frame(r.k, r.T, r.xi)
                T_now = r.T
                if on_frame is not None:
                    on_frame(r, session)
            worker.release_consumed()
        now = float(ts.max())
        session.tick(now, T_now, odom)
        deadline = time.perf_counter() + wait
        while session.pending and time.perf_counter() < deadline:
            time.sleep(0.002)
            session.tick(now, T_now, odom)
        if on_batch is not None:
            on_batch(now, session)
    return T_now
