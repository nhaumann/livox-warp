"""LiDAR odometry on its own thread and private CUDA stream (the warp_worker discipline).

Registration host-syncs on every Gauss-Newton iteration (the 6x6 solve runs on the host). On the
render thread those syncs stall drawing, and drawing's GPU work delays registration. Here the two
run side by side:

    render thread (default stream)                      odometry thread (private stream "odom")
    stage a batch for the ring ──own()──> snapshot ────> append to frame, deskew, register (ICP)
                               stream "odom-in"                          │
    map insert + carving <──── pooled snapshot <──own()── deskewed frame points
                               stream "odom-out"                         │
    pose table (host, locked) <──────────────────────── set_pose(k, T, twist)

Rules followed (warp_worker.context R1-R6):
- each worker owns one private stream, created on the main thread before the thread starts, at
  normal priority: drawing is interactive, a frame of odometry has 100 ms;
- the odometry thread launches, copies and reads back only with explicit stream=, clears with
  kernels and never allocates, so it never touches Warp's global per-device current stream;
- a batch crosses into the worker as an owned snapshot (the staging buffers are rewritten next
  frame), a finished frame crosses back as a pooled owned snapshot; a full queue or an exhausted
  pool drops that item and counts it (own-or-drop). Pose results are small host data and always
  get through;
- every kernel is compiled on the main thread before the thread starts (warmup).

Results carry a generation number: reset() and restart_clock() bump it, drop the batches still
queued for the old generation, and anything produced for an older generation is dropped on arrival
instead of landing in a freshly cleared map. Each batch also carries the render thread's pose epoch,
which the odometry writes into the pose table with every pose (see gpu.Pipeline).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import warp as wp
from warp_worker import WarpWorker, warmup

from . import gpu
from .odom import Odometry


@dataclass
class FrameResult:
    gen: int
    k: int  # frame id
    T: np.ndarray  # sensor-to-world at mid-frame
    xi: np.ndarray  # body-frame motion over the frame (rho, theta)
    m: int  # points in the frame
    snaps: list | None  # owned (xyz, attr, t, frame) of the deskewed frame, or None if dropped
    stats: dict = field(default_factory=dict)

    def arrays(self):
        return [s.array for s in self.snaps]

    def release(self):
        for s in self.snaps or ():
            s.release()
        self.snaps = None


# items of the input queue, render thread -> odometry thread


@dataclass
class _Batch:
    gen: int  # the render-side generation it was submitted under
    epoch: int  # the render thread's pose epoch, written with every pose of this batch
    snaps: list  # owned (xyz, attr, t) snapshots of the staged points
    n: int
    t_min: float
    t_max: float


@dataclass
class _Reset:
    gen: int
    T0: np.ndarray


@dataclass
class _Restart:
    gen: int


@dataclass
class _Config:
    params: dict


class _Stop:
    """Ends the odometry thread."""


class OdomWorker:
    """Runs an Odometry on its own thread: the render thread submits batches and polls FrameResults."""

    def __init__(self, pipe: gpu.Pipeline, device=None, max_batches: int = 120, out_slots: int = 8,
                 frame_cap: int = 1 << 18):
        self.pipe = pipe
        self.device = wp.get_device(device)
        dev = str(self.device)
        # the render thread's stream: everything the Pipeline launches without stream= goes here
        self.render_stream = wp.get_stream(self.device)
        self.w_in = WarpWorker("odom-in", device=dev, priority=0, sync=False)
        self.w = WarpWorker("odom", device=dev, priority=0, sync=True)
        self.w_out = WarpWorker("odom-out", device=dev, priority=0, sync=False)
        for w in (self.w_in, self.w, self.w_out):
            _ = w.stream  # create the streams here, on the main thread (R1)
        self.odom = Odometry(pipe, self.device, frame_cap=frame_cap, stream=self.w.stream)
        o = self.odom
        self.pools = [self.w_out.make_pool((o.frame_cap,), dt, slots=out_slots)
                      for dt in (wp.vec3, wp.uint32, wp.float32, wp.int32)]
        warmup(dev)  # R5: compile every registered kernel before any thread uses one
        wp.synchronize_device(self.device)

        self.max_batches = max_batches
        self.q_in: queue.Queue = queue.Queue()
        self.q_out: queue.Queue = queue.Queue()
        self.gen = 0  # render-side generation
        self._worker_gen = 0  # the generation the odometry thread is working on (written by that thread only)
        self._pending = 0  # batches queued and not yet processed
        self._lock = threading.Lock()
        self.drops = {"input_full": 0, "input_own": 0, "output_pool": 0, "stale": 0, "error": 0}
        self.processed = 0  # batches the odometry thread finished
        self.frames = 0  # frame results the render thread polled
        self.last_error = ""
        self.busy_ms = 0.0  # EMA of worker time per batch
        self._consumed: list[FrameResult] = []  # snapshots the render thread read this frame
        self._thread = threading.Thread(target=self._run, name="livox-odometry", daemon=True)
        self._stop = False
        self._thread.start()

    @property
    def _gen(self) -> int:
        """The odometry thread's generation; tests/check_slam_worker.py's borrowing on_frame hook reads it."""
        return self._worker_gen

    # ---- render thread -------------------------------------------------------------------

    def submit(self, sx, sa, st, n: int, t_min: float, t_max: float, epoch: int = 0) -> bool:
        """Hand one staged batch (points, attributes, timestamps) to the worker as owned snapshots.
        False = dropped (counted)."""
        if n <= 0:
            return True
        with self._lock:
            if self._pending >= self.max_batches:
                self.drops["input_full"] += 1
                return False
        snaps = []
        for a in (sx, sa, st):
            s = self.w_in.own(a[:n], produced_on=self.render_stream)
            if s is None:
                break
            # producer side of the async contract: the render stream's next write to the staging
            # buffers waits for this copy to finish reading them
            s.release_source_on(self.render_stream)
            snaps.append(s)
        if len(snaps) < 3:
            with self._lock:
                self.drops["input_own"] += 1
            return False
        with self._lock:
            self._pending += 1
        self.q_in.put(_Batch(self.gen, epoch, snaps, n, t_min, t_max))
        return True

    def _new_generation(self):
        """Bump the generation and drop every batch still queued for the old one (control items keep
        their order). Dropping a batch here frees its snapshots on the render thread, which is safe:
        submit() made the render stream wait for each snapshot's copy (release_source_on)."""
        self.gen += 1
        keep = []
        while True:
            try:
                it = self.q_in.get_nowait()
            except queue.Empty:
                break
            if isinstance(it, _Batch):
                with self._lock:
                    self._pending -= 1
                    self.drops["stale"] += 1
            else:
                keep.append(it)
        for it in keep:
            self.q_in.put(it)
        self._drain_out()

    def reset(self, T0: np.ndarray):
        """Forget map and trajectory; the next frame is posed at T0. Batches and results of the old
        generation are discarded."""
        self._new_generation()
        self.q_in.put(_Reset(self.gen, np.array(T0, dtype=np.float64)))

    def restart_clock(self):
        """The live clock restarted: keep the map and pose, restart frame counting at rest; batches and
        results of the old clock are discarded."""
        self._new_generation()
        self.q_in.put(_Restart(self.gen))

    def configure(self, **params):
        """Set Odometry tuning attributes (frame_dt, reg_voxel, max_iter, ...) between batches. A name that is
        not a scalar attribute of the Odometry raises AttributeError here, before anything is queued."""
        for name in params:
            if not isinstance(getattr(self.odom, name, None), (int, float, str)):
                raise AttributeError(f"Odometry has no tuning attribute {name!r}")
        self.q_in.put(_Config(dict(params)))

    def poll(self) -> list[FrameResult]:
        """Finished frames for the current generation. Each result's snapshots are ready to read on
        the render stream; call release_consumed() after the render stream has passed its next sync."""
        out = []
        while True:
            try:
                r = self.q_out.get_nowait()
            except queue.Empty:
                break
            if r.gen != self.gen:
                r.release()
                with self._lock:
                    self.drops["stale"] += 1
                continue
            if r.snaps:
                for s in r.snaps:
                    s.wait(self.render_stream)  # consumer side: render kernels wait for the copies
            self.frames += 1
            self._consumed.append(r)
            out.append(r)
        return out

    def release_consumed(self):
        """Return this frame's snapshot slots to their pools. Call only after the render stream has
        been synchronized past the kernels that read them (Pipeline.build does that every frame)."""
        for r in self._consumed:
            r.release()
        self._consumed.clear()

    def _drain_out(self):
        while True:
            try:
                self.q_out.get_nowait().release()
            except queue.Empty:
                break

    def status(self) -> dict:
        with self._lock:
            return {"pending": self._pending, "processed": self.processed, "frames": self.frames,
                    "drops": dict(self.drops), "busy_ms": round(self.busy_ms, 2), "last_error": self.last_error,
                    "pool_free": min(p.free_slots for p in self.pools),
                    "workers": [w.status() for w in (self.w_in, self.w, self.w_out)]}

    def close(self):
        if self._stop:
            return
        self._stop = True
        self._new_generation()  # drop the queued batches so the stop is not behind them
        self.q_in.put(_Stop())
        self._thread.join(timeout=5.0)
        self._drain_out()
        self.release_consumed()
        if self._thread.is_alive():
            print("OdomWorker.close: the odometry thread did not stop; leaving its streams open", file=sys.stderr)
            return
        for w in (self.w_in, self.w, self.w_out):
            w.close()

    # ---- odometry thread -----------------------------------------------------------------

    def _run(self):
        while True:
            item = self.q_in.get()
            if isinstance(item, _Stop):
                break
            try:
                if isinstance(item, _Reset):
                    self._worker_gen = item.gen
                    self.odom.reset(item.T0)
                    self.odom.synchronize()
                elif isinstance(item, _Restart):
                    self._worker_gen = item.gen
                    self.odom.restart_clock()
                elif isinstance(item, _Config):
                    for name, value in item.params.items():
                        setattr(self.odom, name, value)
                elif isinstance(item, _Batch):
                    self._batch(item)
            except Exception as e:  # noqa: BLE001 - one bad batch must not kill the worker
                with self._lock:
                    self.drops["error"] += 1
                    self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()

    def _batch(self, item: _Batch):
        try:
            if item.gen != self._worker_gen:
                with self._lock:
                    self.drops["stale"] += 1
                return
            t0 = time.perf_counter()
            s = self.w.stream
            for sn in item.snaps:
                sn.wait(s)  # consumer side: our kernels wait for the input copies
            sx, sa, st = (sn.array for sn in item.snaps)
            self.odom.epoch = item.epoch
            self.odom.push(sx, sa, st, item.n, item.t_min, item.t_max, self._on_frame)
            # our reads of the snapshots are finished before we drop them (they are freed on release)
            wp.synchronize_stream(s)
            dt = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self.processed += 1
                self.busy_ms = dt if self.processed == 1 else 0.9 * self.busy_ms + 0.1 * dt
        finally:
            with self._lock:
                self._pending -= 1

    def _on_frame(self, k, T, fx, fa, ft, ff, m):
        """Runs on the odometry thread when a frame's pose is final: snapshot its points for the render thread."""
        snaps = []
        for arr, pool in zip((fx, fa, ft, ff), self.pools):
            sn = self.w_out.own(arr, produced_on=self.w.stream, pool=pool)
            if sn is None:
                for x in snaps:
                    x.release()
                snaps = None
                with self._lock:
                    self.drops["output_pool"] += 1
                break
            # producer side: the next frame's writes to fx..ff wait for this copy
            sn.release_source_on(self.w.stream)
            snaps.append(sn)
        o = self.odom
        self.q_out.put(FrameResult(self._worker_gen, k, np.array(T, dtype=np.float64),
                                   np.array(o.xi, dtype=np.float32), m, snaps, dict(o.stats)))
