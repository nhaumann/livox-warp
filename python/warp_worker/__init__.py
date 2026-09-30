"""warp_worker — a generic, collision-safe GPU worker template for NVIDIA Warp.

Construct once, delegate work on an isolated private stream, hand buffers across
threads with own-or-drop, monitor + kill live. All Warp global-state collisions
(the per-device current stream, stream-ordered allocations, and any op lacking a
`stream=` parameter) are funneled through a single process lock; everything else
uses explicit `stream=`.

    from warp_worker import WarpWorker, warmup, list_workers, close_worker

    warmup()                                   # pre-compile kernels (main thread)
    w = WarpWorker("stage")                     # private high-prio stream
    with w.scope():                             # current-stream-dependent op, e.g. alloc (locked)
        a = wp.zeros(n, dtype=float)
    wp.launch(k, dim, inputs=[...], stream=w.stream)
    snap = w.own(reused_buffer)                 # independent, materialized snapshot
    ...                                         # hand snap to other worker threads
    w.close()

See `warp_worker.context` for the documented Warp context/stream model and rules.
"""
from .context import (
    DEVICE, GLOBAL_WARP_LOCK, make_stream, scoped_stream, on_stream,
    sync, warmup, cuda_available,
)
from .worker import (
    OwnedBuffer, SnapshotPool, OwnedFrame, FramePool, WarpWorker, list_workers, get_worker, close_worker, close_all_workers,
)

__all__ = [
    "WarpWorker", "OwnedFrame", "warmup", "make_stream", "scoped_stream",
    "on_stream", "sync", "cuda_available", "DEVICE", "GLOBAL_WARP_LOCK",
    "list_workers", "get_worker", "close_worker", "close_all_workers",
]
