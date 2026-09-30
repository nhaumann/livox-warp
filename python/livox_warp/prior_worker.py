"""A prior map on its own thread and private CUDA stream: load, build the GPU grid, localise, track.

    load thread (host only)        render thread                      prior thread (stream "prior")
    np.load, thin for drawing,  ─> tick(): allocate the device  ────> upload, jump-flood the grid,
    sort by cell                    buffers on the default stream,     warm open3d  -> state "ready"
                                    record an event                     │
    collected scan (host numpy) ──> localize(xyz) ───────────────────> coarse-to-fine search + ICP
    recent submap (host numpy)  ──> track(xyz, T0) ──────────────────> multi-start point-to-plane ICP
                                    poll() <── results ─────────────────┘

Rules followed (warp_worker.context R1-R6):
- the prior thread owns one private stream, created on the main thread (R1). WarpWorker is only that
  stream's factory here: the thread runs its own job queue, not WarpWorker's submit/own machinery;
- the prior thread launches and copies only with explicit stream=, and never allocates: every device
  and pinned buffer (PriorGrid, GlobalLocalizer) is allocated by tick() on the render thread, which owns
  the default stream, and the prior stream waits on an event recorded after those allocations (R3). So
  the prior thread never changes Warp's global current stream, and needs no scope();
- scans cross into the thread as host numpy arrays, results come back as host data (R4);
- the localisation kernels are compiled on the main thread, in tick(), before the thread uses them (R5).

Once "ready", the grid's device arrays are never written again, so any stream may read them (the render
thread's Changes colouring).
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback

import numpy as np
import warp as wp
from warp_worker import WarpWorker

from . import localize, mapgrid, prior_map
from .mapgrid import PriorGrid


def draw_subset(host: dict, max_points: int = 900_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A thinned copy of the map for drawing: xyz (float32) and RGBA8 (the scan's colour dimmed, or a height
    ramp when it has none). A fixed-seed random subset, so the thinning is spatially uniform whatever order the
    points are stored in."""
    xyz = host["xyz"]
    n = len(xyz)
    sel = slice(None) if n <= max_points else np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    p = np.ascontiguousarray(xyz[sel], dtype=np.float32)
    rgb = host.get("rgb")
    if rgb is not None and len(rgb) == n and float(np.mean(rgb)) > 1.0:
        c = rgb[sel].astype(np.float32)
    else:
        z = p[:, 2]
        t = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
        c = np.stack([150 + 60 * t, 155 + 40 * t, 170 - 20 * t], 1)
    rgba = np.empty((len(p), 4), np.uint8)
    rgba[:, :3] = np.clip(c * 0.5, 0, 255).astype(np.uint8)  # dimmed: context behind the live points
    rgba[:, 3] = 255
    return p, rgba


class PriorWorker:
    """The map's life cycle: state loading -> allocating -> building -> ready, or error at any step; then
    localize() and track() queue jobs whose results poll() hands back."""

    def __init__(self, path: str, device=None):
        self.path = path
        self.device = wp.get_device(device)
        self.render_stream = wp.get_stream(self.device) if self.device.is_cuda else None
        # WarpWorker is the cleanest way to get a private stream created on the main thread (R1); nothing else
        # of it is used. Reading .stream here creates it.
        self.w = WarpWorker("prior", device=str(self.device), priority=0, sync=True)
        self.stream = self.w.stream
        self.state = "loading"
        self.error = ""
        self.host = None
        self.draw = None  # (xyz, rgba) thinned for drawing
        self.grid: PriorGrid | None = None
        self.loc: localize.GlobalLocalizer | None = None
        self.busy = ""  # what the thread is doing now (for the UI)
        self.pending = 0  # jobs queued or running (localize / track)
        self.load_s = 0.0
        self.build_s = 0.0
        self._alloc_event = None
        self._q: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._thread = None
        threading.Thread(target=self._load, name="prior-load", daemon=True).start()

    # ---- load thread (host only, no Warp) ----------------------------------------------------

    def _load(self):
        t0 = time.perf_counter()
        try:
            host = prior_map.load(self.path)
            for k in ("xyz", "normal"):
                if k not in host:
                    raise ValueError(f"{os.path.basename(self.path)} has no '{k}' array")
            self.draw = draw_subset(host)
            self.host = PriorGrid.prepare(host)
            self.load_s = time.perf_counter() - t0
            self.state = "allocating"
        except Exception as e:  # noqa: BLE001 - shown in the UI
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    # ---- render thread -------------------------------------------------------------------------

    def tick(self):
        """Call once per frame on the render thread: allocates the GPU buffers when the host data is in."""
        if self.state != "allocating":
            return
        try:
            d = self.device
            self.grid = PriorGrid(self.host, d)  # allocations on the default (render) stream
            self.loc = localize.GlobalLocalizer(self.grid, d)
            wp.load_module(localize, device=d)  # R5: compile the search and grid kernels here, not on the thread
            wp.load_module(mapgrid, device=d)
            if self.render_stream is not None:
                self._alloc_event = wp.Event(d)
                self.render_stream.record_event(self._alloc_event)
            self.state = "building"
            self._thread = threading.Thread(target=self._run, name="prior-map", daemon=True)
            self._thread.start()
            self._q.put(("build",))
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            traceback.print_exc()

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def localize(self, xyz: np.ndarray, up_prior=None, ctx=None) -> bool:
        """Queue a localisation of a scan (sensor frame, host array). False if the map is not ready."""
        if not self.ready:
            return False
        self.pending += 1
        self._q.put(("localize", np.ascontiguousarray(xyz, dtype=np.float32), up_prior, ctx))
        return True

    def track(self, xyz: np.ndarray, T0: np.ndarray, starts=(), ctx=None, switch_margin: float = 0.05) -> bool:
        """Queue a refinement of pose T0 (maps the points' frame into the map) from a host point set."""
        if not self.ready:
            return False
        self.pending += 1
        self._q.put(("track", np.ascontiguousarray(xyz, dtype=np.float32), np.array(T0, dtype=np.float64),
                     tuple(starts), ctx, float(switch_margin)))
        return True

    def poll(self) -> list:
        """Finished work: [("pose", T, info, ctx) | ("track", T, fit, info, ctx) | ("error", text, ctx)]."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def status(self) -> dict:
        g = self.grid
        return {"state": self.state, "error": self.error, "busy": self.busy, "load_s": self.load_s,
                "build_s": self.build_s, "points": 0 if g is None else g.n,
                "mib": 0.0 if g is None else g.gpu_bytes() / 2**20}

    def close(self):
        self._q.put(("stop",))
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.w.close()

    # ---- prior thread --------------------------------------------------------------------------

    def _run(self):
        s = self.stream
        while True:
            item = self._q.get()
            if item[0] == "stop":
                return
            try:
                if item[0] == "build":
                    self.busy = "building the map grid"
                    if self._alloc_event is not None and s is not None:
                        s.wait_event(self._alloc_event)  # the buffers were allocated on the render stream
                    t0 = time.perf_counter()
                    self.grid.build(s)
                    self.build_s = time.perf_counter() - t0
                    localize.ScanPrep(np.random.default_rng(0).normal(size=(2000, 3)))  # warm open3d (~1 s)
                    self.state = "ready"
                elif item[0] == "localize":
                    _, xyz, up_prior, ctx = item
                    self.busy = "localising"
                    t0 = time.perf_counter()
                    T, info = self.loc.localize(xyz, s, up_prior=up_prior, log=lambda m: None,
                                                progress=lambda m: setattr(self, "busy", m))
                    info["scan_raw_points"] = len(xyz)
                    info["seconds"] = time.perf_counter() - t0
                    self._out.put(("pose", T, info, ctx))
                elif item[0] == "track":
                    _, xyz, T0, starts, ctx, margin = item
                    self.busy = "tracking"
                    T, fit, info = self.loc.refine(xyz, T0, s, starts=starts, switch_margin=margin)
                    self._out.put(("track", T, fit, info, ctx))
            except Exception as e:  # noqa: BLE001 - one failure must not kill the worker
                msg = f"{type(e).__name__}: {e}"
                if item[0] == "build":
                    traceback.print_exc()
                    self.error = msg
                    self.state = "error"
                else:
                    self._out.put(("error", msg, item[-1]))
            finally:
                if item[0] in ("localize", "track"):
                    self.pending -= 1
                self.busy = ""
