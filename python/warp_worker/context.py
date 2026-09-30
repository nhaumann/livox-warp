"""warp_worker.context — Warp CUDA context / stream model + collision-safe helpers.

This module is the single source of truth for how this codebase shares the GPU
across many host threads without Warp state collisions. Read this before using
streams from more than one thread.

================================================================================
HOW WARP CONTEXTS & STREAMS WORK  (from the NVIDIA Warp concurrency/allocator docs)
================================================================================
1. ONE CUDA CONTEXT PER DEVICE. Warp's runtime creates/uses a single primary CUDA
   context per device. EVERY Warp stream and array on a device shares that one
   context. There is no per-worker context, so workers cannot collide at the
   context level — provided they all stay on the same device handle and nobody
   creates extra contexts. (Multi-GPU = one context per device; partition workers
   by device.)

2. THE "CURRENT STREAM" IS PER-DEVICE *GLOBAL* STATE — NOT THREAD-LOCAL.
   At init Warp makes a default stream per device and stores it as the device's
   "current stream" (a single attribute, `device.stream`). Every kernel launch and
   memory op issued WITHOUT an explicit `stream=` goes on that current stream.
   `wp.set_stream()` / `wp.get_stream()` / `wp.ScopedStream()` read & WRITE that one
   shared attribute. => If two host threads enter `wp.ScopedStream` (or call
   set_stream) on the same device concurrently, they stomp each other's current
   stream and kernels land on the wrong stream. ScopedStream also `sync_enter=True`
   by default, adding a host stall. THIS is the collision to avoid.

3. STREAM-ORDERED MEMORY POOL ALLOCATOR. `wp.zeros/empty/clone/full` allocate from
   a pool ordered on the CURRENT stream *at allocation time*. So an allocation made
   while another thread's stream is current is ordered on the WRONG stream, and
   using it on your stream is a use-before-alloc race under contention.

4. SAFE, STATE-FREE PRIMITIVES (use these across threads):
   - `wp.launch(kernel, dim, inputs, stream=s)`   — runs on s, touches no global state
   - `wp.copy(dst, src, stream=s)`                 — same
   - `stream.record_event(e)` / `stream.wait_event(e)` / `stream.wait_stream(o)` —
     cross-stream ordering on the device (no host stall)
   - `wp.synchronize_stream(s)` / `wp.synchronize_event(e)` — host waits (releases GIL)

================================================================================
COLLISION-SAFE RULES enforced by this package
================================================================================
R1. Each worker owns ONE private stream (high priority), created once, reused.
R2. ALL kernel launches use explicit `stream=worker.stream`. Never ScopedStream /
    set_stream from a worker thread for launches.
R3. The ONLY ops that lack a `stream=` and therefore read the global current stream
    — chiefly memory allocation (`wp.zeros/empty/clone`), and any builder/op that
    relies on the current stream — are funneled through `scoped_stream()` /
    `on_stream()`, which hold a single PROCESS-WIDE lock while they briefly set the
    current stream. Held for microseconds (a submit), so GPU work still overlaps;
    only the submission is serialized.
R4. Cross-thread buffer handoff goes through `OwnedFrame` (a materialized copy) or a
    CUDA event — never a borrowed view of a buffer another thread will overwrite.
R5. Pre-warm kernels with `warmup()` on the main thread before spawning workers, so
    first-time compilation never contends across threads.
R6. One context per device: do not create extra contexts; keep each worker on the
    same `DEVICE` handle (or partition by device for multi-GPU).
"""
from __future__ import annotations

import threading
import contextlib

DEVICE = "cuda:0"

# The one process-wide lock that serializes every mutation of Warp's global
# per-device current stream (allocations + any current-stream-dependent op). See R3.
GLOBAL_WARP_LOCK = threading.Lock()


def cuda_available() -> bool:
    try:
        import warp as wp
        return wp.is_cuda_available()
    except Exception:
        return False


def make_stream(priority: int = -1, device: str = DEVICE):
    """A private NVIDIA Warp stream (priority -1 = high), or None without CUDA.
    Reused for the lifetime of a worker; never recreated per call (R1)."""
    try:
        import warp as wp
    except Exception:
        return None
    last = None
    for kwargs in ({"device": device, "priority": priority}, {"device": device}, {}):
        try:
            return wp.Stream(**kwargs)
        except Exception as e:
            last = e
    if last is not None:
        try:
            print(f"[warp_worker] wp.Stream create failed ({device}): {last}; "
                  f"falling back to the default stream")
        except Exception:
            pass
    return None


@contextlib.contextmanager
def scoped_stream(stream):
    """Run the enclosed CURRENT-STREAM-DEPENDENT ops (memory allocation, or any op
    that has no `stream=` parameter) with `stream` as the device's current stream,
    under the global lock so threads don't stomp each other's current stream (R3).
    Use this ONLY for ops that lack a `stream=` parameter — for kernel launches pass
    `stream=` explicitly instead.

    `sync_enter=False` so we don't pay ScopedStream's default host stall; the worker
    orders its own work via the stream / own() / events.
    """
    if stream is None:
        yield
        return
    import warp as wp
    with GLOBAL_WARP_LOCK:
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            yield


def on_stream(stream, fn):
    """Run a CURRENT-STREAM-DEPENDENT callable `fn()` with `stream` as the device's
    current stream, under the global lock (R3). Use for any op that lacks a `stream=`
    parameter — e.g. an allocation `lambda: wp.zeros(...)`, or some library builder
    that implicitly uses the current stream. Returns whatever `fn()` returns."""
    with scoped_stream(stream):
        return fn()


def sync(stream=None):
    """Host-wait for a stream (or the whole device if stream is None). Releases the
    GIL while blocked, so other worker threads keep running."""
    import warp as wp
    if stream is None:
        wp.synchronize()
    else:
        wp.synchronize_stream(stream)


def warmup(device: str = DEVICE):
    """Force-compile/load all registered Warp kernels on the calling (main) thread
    BEFORE spawning worker threads, so first-time compilation never contends across
    threads (R5). Safe no-op without CUDA."""
    try:
        import warp as wp
        wp.force_load(device=device)
    except Exception:
        pass
