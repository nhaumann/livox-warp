"""warp_worker.worker — an isolated, monitorable GPU work delegate.

A WarpWorker owns ONE private Warp stream and gives you the collision-safe
primitives to (a) hand a reused GPU buffer across a thread boundary without
tearing (`own`), (b) run an op on the private stream (`submit` / explicit
`stream=`), and (c) run a current-stream-dependent op (e.g. a memory
allocation) without racing other threads' current stream (`scope()`). Every
worker auto-registers so the whole fleet can be listed, status-checked, and
closed at runtime.

Domain-agnostic: the same contract covers simulation stages, solvers,
encoders, async readbacks, retained previews — any multithreaded pipeline
where a reused GPU buffer crosses a thread boundary.

THE CORE CONTRACT (own-or-drop): a producer never hands a borrowed view of a
reused buffer across a thread boundary — it hands an `OwnedBuffer` (an
independent snapshot with a known readiness state) or returns None and the
caller drops that work item.

Generic over buffer type: `own()` accepts a Warp array OR a torch CUDA
tensor.
"""
from __future__ import annotations

import time
import threading
import weakref

from .context import make_stream, scoped_stream, sync as _sync


# ── unit that crosses a producer→consumer boundary ──────────────────────────
class OwnedBuffer:
    """An INDEPENDENT GPU buffer (Warp array or torch tensor) safe to read across a
    thread boundary, plus an optional readiness event.

    THE ASYNC CONTRACT (sync=False) has TWO sides, both on the same event:

        producer finished writing source
                  |
        snapshot stream copies source -> owned slot     (own() orders this
                  |                                      via produced_on=)
        copy completed  ==  self.event
                  |-- consumer may read .array          -> wait(consumer_stream)
                  '-- producer may REUSE the source     -> release_source_on(
                                                              producer_stream)

    A consumer-side wait alone does NOT make source reuse safe: the producer
    must order its next write on the copy-completion event too, or the
    snapshot can capture a MIXTURE of generations of the source. Under
    sync=True both sides are host-synchronized inside own() and no
    event discipline is needed.

    `release()` returns a pooled slot (idempotent; no-op if not pooled)."""
    __slots__ = ("array", "event", "_pool", "_slot")

    def __init__(self, array, event=None, *, pool=None, slot=None):
        self.array = array          # the independent buffer (wp.array or torch.Tensor)
        self.event = event
        self._pool = pool
        self._slot = slot

    # alias so torch-style callers can use `.tensor`
    @property
    def tensor(self):
        return self.array

    def wait(self, stream) -> None:
        """CONSUMER side: make `stream` wait until the snapshot contents are
        ready to read (no-op under sync=True: already materialized)."""
        if self.event is not None and stream is not None:
            try:
                stream.wait_event(self.event)
            except Exception:
                pass

    def release_source_on(self, producer_stream) -> None:
        """PRODUCER side: order `producer_stream`'s NEXT write to the source
        buffer after the snapshot copy has finished reading it. Required
        before reusing the source under sync=False; no-op under sync=True."""
        if self.event is not None and producer_stream is not None:
            try:
                producer_stream.wait_event(self.event)
            except Exception:
                pass

    def release(self) -> None:
        p, s = self._pool, self._slot
        self._pool = None
        if p is not None and s is not None:
            p._release_slot(s)


# ── registry (enumerate + kill live) ────────────────────────────────────────
_REGISTRY = weakref.WeakValueDictionary()
_REG_LOCK = threading.Lock()
_REG_SEQ = [0]


def _register(w) -> str:
    with _REG_LOCK:
        _REG_SEQ[0] += 1
        wid = f"{getattr(w, 'name', 'worker')}#{_REG_SEQ[0]}"
    _REGISTRY[wid] = w
    return wid


def list_workers() -> list:
    out = []
    for wid, w in list(_REGISTRY.items()):
        try:
            st = dict(w.status())
        except Exception as e:
            st = {"error": repr(e)}
        st["id"] = wid
        out.append(st)
    return sorted(out, key=lambda s: s.get("id", ""))


def get_worker(wid: str):
    return _REGISTRY.get(wid)


def close_worker(wid: str) -> bool:
    w = _REGISTRY.get(wid)
    if w is None:
        return False
    try:
        w.close()
        return True
    except Exception:
        return False


def close_all_workers(prefix: str = "") -> int:
    n = 0
    for wid, w in list(_REGISTRY.items()):
        if prefix and not wid.startswith(prefix):
            continue
        try:
            w.close(); n += 1
        except Exception:
            pass
    return n


def _is_torch(x) -> bool:
    return hasattr(x, "clone") and hasattr(x, "is_cuda")


def _is_warp(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "device") \
        and not _is_torch(x)


# ── bounded snapshot pool ───────────────────────────────────────────────────
class SnapshotPool:
    """Bounded, PREALLOCATED snapshot slots -- own-or-drop as a complete
    backpressure policy.

    Slots are allocated ONCE on the worker's stream (no per-item allocation
    on the hot path; Warp's allocator is stream-ordered and allocation is
    not free at high rates). `own(view, pool=p)` copies into a free slot or
    returns None (counted as drops["pool_exhausted"]);
    `OwnedBuffer.release()` returns the slot.

    Reuse ordering: the worker's stream serializes successive copies into a
    recycled slot. A consumer reading on ANOTHER stream must call
    frame.wait(consumer_stream) before reading and release() only when done
    -- release is the consumer's statement that the slot may be overwritten.
    """

    def __init__(self, worker, shape, dtype, slots: int = 4, device=None):
        import warp as wp
        self.worker = worker
        self.shape = tuple(shape) if hasattr(shape, "__len__") else (shape,)
        self.dtype = dtype
        dev = device or worker.device
        with worker.scope():
            self._slots = [wp.empty(shape=self.shape, dtype=dtype, device=dev)
                           for _ in range(int(slots))]
        self._free = list(range(int(slots)))
        self._out = set()
        self._lock = threading.Lock()
        self.exhausted = 0

    @property
    def n_slots(self) -> int:
        return len(self._slots)

    @property
    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    def _acquire(self):
        with self._lock:
            if not self._free:
                self.exhausted += 1
                return None, None
            i = self._free.pop()
            self._out.add(i)
            return i, self._slots[i]

    def _release_slot(self, i) -> None:
        with self._lock:
            if i in self._out:          # idempotent; double-release is a no-op
                self._out.discard(i)
                self._free.append(i)


# ── the worker ──────────────────────────────────────────────────────────────
class WarpWorker:
    """Reusable GPU delegate on a private high-priority Warp stream.

        w = WarpWorker("stage")
        with w.scope():                 # current-stream-dependent ops (e.g. allocation)
            a = wp.zeros(n, dtype=float)
        wp.launch(k, dim, inputs=[...], stream=w.stream)   # launches: explicit stream=
        owned = w.own(some_reused_buffer)                  # safe cross-thread snapshot
        st = w.status();  w.close()
    """

    def __init__(self, name: str, *, device: str = "cuda:0",
                 priority: int = -1, sync: bool = True):
        self.name = name
        self.device = device
        self._priority = int(priority)   # REQUESTED priority (scheduling hint,
        self._sync = bool(sync)          # not a guarantee; support varies)
        self._stream = None               # lazy
        self._closed = False
        self._lock = threading.Lock()
        self.processed = 0
        self.drops = {"none_input": 0, "not_buffer": 0, "error": 0,
                      "closed": 0, "pool_exhausted": 0}
        self.last_error = ""
        self._lat_ms_ema = 0.0
        self._reg_id = _register(self)

    @property
    def stream(self):
        """The private Warp stream (lazy, high priority). Launch kernels on it with
        `stream=w.stream`; never via ScopedStream from a worker thread (see context)."""
        if self._stream is None and not self._closed:
            self._stream = make_stream(self._priority, self.device)
        return self._stream

    # back-compat alias (older call sites use .wp_stream)
    @property
    def wp_stream(self):
        return self.stream

    def scope(self):
        """Context manager that makes this worker's stream the current stream UNDER
        THE GLOBAL LOCK, for ops with no `stream=` parameter (e.g. allocations)."""
        return scoped_stream(self.stream)

    def synchronize(self):
        _sync(self.stream)

    def _tick(self, t0):
        dt = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.processed += 1
            self._lat_ms_ema = dt if self.processed == 1 else \
                0.9 * self._lat_ms_ema + 0.1 * dt

    def make_pool(self, shape, dtype, slots: int = 4) -> "SnapshotPool":
        """Preallocate a bounded snapshot pool on this worker's stream."""
        return SnapshotPool(self, shape, dtype, slots=slots)

    def own(self, view, *, produced_on=None, pool=None):
        """Clone `view` (a reused GPU buffer) into an INDEPENDENT OwnedBuffer,
        or return None (caller drops).

        produced_on: the producer's stream (or wp.Event) when the source was
        written ASYNCHRONOUSLY -- the snapshot copy is ordered after those
        writes device-side. Without it, the caller must ensure the source is
        materialized before own() (e.g. the producer host-synced).

        sync=True (default): own() returns only after the snapshot has
        materialized -- safe for both consumer reads AND immediate source
        reuse. sync=False: returns immediately with OwnedBuffer.event; the
        consumer must wait() and the producer must release_source_on() its
        stream before overwriting the source (see OwnedBuffer docstring --
        the destination being independent does NOT make the source free)."""
        if self._closed:
            self.drops["closed"] += 1
            return None
        if view is None:
            self.drops["none_input"] += 1
            return None
        t0 = time.perf_counter()
        try:
            if _is_torch(view):
                return self._own_torch(view, t0)
            if _is_warp(view):
                return self._own_warp(view, t0, produced_on, pool)
            self.drops["not_buffer"] += 1
            return None
        except Exception as e:
            self.drops["error"] += 1
            self.last_error = repr(e)
            return None

    def _own_warp(self, view, t0, produced_on=None, pool=None):
        import warp as wp
        s = self.stream
        slot_i = None
        if pool is not None:
            slot_i, dst = pool._acquire()
            if slot_i is None:
                self.drops["pool_exhausted"] += 1
                return None
            if dst.shape != view.shape or dst.dtype != view.dtype:
                pool._release_slot(slot_i)
                self.drops["not_buffer"] += 1
                return None
        # PRODUCER-READINESS ordering: the copy must not start before the
        # producer's queued writes to the source have finished (review gap:
        # an async producer could still be writing while we read)
        if produced_on is not None and s is not None:
            try:
                if isinstance(produced_on, wp.Event):
                    s.wait_event(produced_on)
                else:                      # a stream: record + wait
                    ev = wp.Event(s.device)
                    produced_on.record_event(ev)
                    s.wait_event(ev)
            except Exception:
                pass
        if pool is None:
            # allocate the clone ordered on our stream (locked)
            with scoped_stream(s):
                dst = wp.empty(shape=view.shape, dtype=view.dtype,
                               device=view.device)
        wp.copy(dst, view, stream=s)
        event = None
        if s is not None:
            if self._sync:
                wp.synchronize_stream(s)
            else:
                event = wp.Event(s.device)
                s.record_event(event)
        else:
            wp.synchronize()
        self._tick(t0)
        return OwnedBuffer(dst, event, pool=pool, slot=slot_i)

    def _own_torch(self, view, t0):
        import torch
        owned = view.clone()
        event = None
        if owned.is_cuda:
            if self._sync:
                torch.cuda.current_stream(owned.device).synchronize()
            else:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(owned.device))
        self._tick(t0)
        return OwnedBuffer(owned, event)

    def submit(self, work):
        """Run `work(stream)` (which MUST return a fresh buffer) on the private
        stream; returns its output as an OwnedFrame, or None."""
        if self._closed:
            self.drops["closed"] += 1
            return None
        t0 = time.perf_counter()
        try:
            out = work(self.stream)
            if out is None:
                self.drops["error"] += 1
                return None
            if self._sync:
                _sync(self.stream)
            self._tick(t0)
            return OwnedBuffer(out)
        except Exception as e:
            self.drops["error"] += 1
            self.last_error = repr(e)
            return None

    def status(self) -> dict:
        with self._lock:
            return {
                "name": self.name, "running": not self._closed,
                "processed": self.processed, "drops": dict(self.drops),
                "latency_ms_ema": round(self._lat_ms_ema, 3),
                "last_error": self.last_error, "sync": self._sync,
            }

    def close(self) -> None:
        """Drain + free the stream. Idempotent. After close, own()/submit()
        drop (counted) so a closed worker degrades only its own stage, never
        crashing the pipeline. Closing does not preempt queued kernels; CUDA
        completes in-flight work first."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._stream is not None:
                _sync(self._stream)
        except Exception:
            pass
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ── back-compat aliases (pre-0.2 names) ─────────────────────────────────────
OwnedFrame = OwnedBuffer
FramePool = SnapshotPool
