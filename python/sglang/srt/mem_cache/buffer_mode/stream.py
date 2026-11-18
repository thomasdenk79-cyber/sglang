"""Streamed restore of buffer-only storage hits larger than the host arena.

Buffer-only HiCache stages a storage hit in one host bounce, so a hit larger
than the host arena was declined and recomputed. With
``SGLANG_HICACHE_BUFFER_STREAM=1`` such a hit instead allocates its FULL
device rows up front and a small pinned ring of ``inflight`` chunk slots.
The prefetch IO worker reads chunk ``k + 1`` from storage into its ring slot
(through the backend's parallel read workers) while chunk ``k`` is copied
host-to-device on a dedicated stream; a slot is reused only after the H2D of
the chunk it last held completed. KV-derived sidecars (e.g. the QSA indexer)
ride the KV ring slots and device rows. Trailing state pools (Mamba) are
fetched by the regular sidecar read after the last KV chunk, and loaded at
admission like any staged prefetch.

The restore is all-or-nothing: until every chunk has landed on the device
the operation reports zero completed tokens, so a short read, a deadline or
an abort discards the device rows and the request recomputes.

Copies are paced by a per-scheduler-iteration H2D quota
(``SGLANG_HICACHE_BUFFER_STREAM_H2D_MB_PER_ITER``): stream priority only
orders SM work, not DMA, and the host link is shared with the transfers of
every forward.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def stream_enabled() -> bool:
    return envs.SGLANG_HICACHE_BUFFER_STREAM.get()


def stream_chunk_tokens(page_size: int) -> int:
    raw = envs.SGLANG_HICACHE_BUFFER_STREAM_CHUNK_TOKENS.get()
    return max(page_size, raw - raw % page_size)


def stream_inflight() -> int:
    return max(1, envs.SGLANG_HICACHE_BUFFER_STREAM_INFLIGHT.get())


@dataclass(frozen=True)
class StreamChunk:
    """Tokens ``[start, end)`` of the hit, staged in ring slot ``slot``."""

    index: int
    start: int
    end: int
    slot: int

    @property
    def tokens(self) -> int:
        return self.end - self.start


def plan_stream_chunks(
    hit_tokens: int, chunk_tokens: int, page_size: int, inflight: int
) -> list[StreamChunk]:
    """Page-aligned chunks covering the hit, cycling over the ring slots."""
    if hit_tokens <= 0 or hit_tokens % page_size:
        raise ValueError(f"hit {hit_tokens} is not a positive multiple of {page_size}")
    chunk = max(page_size, chunk_tokens - chunk_tokens % page_size)
    num_chunks = -(-hit_tokens // chunk)
    slots = max(1, min(inflight, num_chunks))
    return [
        StreamChunk(i, i * chunk, min(hit_tokens, (i + 1) * chunk), i % slots)
        for i in range(num_chunks)
    ]


def ring_tokens(chunks: Sequence[StreamChunk]) -> int:
    """Host ring size: one full-size chunk per slot."""
    return (max(c.slot for c in chunks) + 1) * chunks[0].tokens


def ring_slice(chunk: StreamChunk, slot_tokens: int) -> slice:
    base = chunk.slot * slot_tokens
    return slice(base, base + chunk.tokens)


def chunk_batches(
    chunk: StreamChunk, page_size: int, batch_pages: int
) -> list[tuple[int, int]]:
    """The chunk's storage batches as hit page ranges ``[start, end)``."""
    first, last = chunk.start // page_size, chunk.end // page_size
    return [(p, min(last, p + batch_pages)) for p in range(first, last, batch_pages)]


def split_span(num_tokens: int, piece_tokens: int, page_size: int) -> list[int]:
    """Page-aligned piece lengths, each at most ``piece_tokens``, covering a
    span in order (an unaligned tail stays in the last piece)."""
    piece = max(page_size, piece_tokens - piece_tokens % page_size)
    lengths = []
    remaining = num_tokens
    while remaining > piece:
        lengths.append(piece)
        remaining -= piece
    if remaining > 0:
        lengths.append(remaining)
    return lengths


@dataclass
class StreamStats:
    tokens: int
    chunks: int
    inflight: int
    bytes_per_token: int = 0
    read_s: float = 0.0
    h2d_s: float = 0.0
    slot_wait_s: float = 0.0
    grant_wait_s: float = 0.0
    wall_s: float = 0.0
    ok: bool = False
    # Scheduler iteration time while this restore ran vs the idle baseline:
    # a proxy for the decode TPOT the restore costs other requests.
    step_s_sum: float = 0.0
    steps: int = 0
    baseline_step_s: float = 0.0

    def overlap_ratio(self) -> float:
        """Share of the shorter stage hidden behind the other: 1 is a fully
        pipelined restore, 0 a serial one."""
        shorter = min(self.read_s, self.h2d_s)
        if shorter <= 0:
            return 0.0
        hidden = self.read_s + self.h2d_s - self.wall_s
        return max(0.0, min(1.0, hidden / shorter))

    def _rate(self, seconds: float) -> str:
        if seconds <= 0 or self.bytes_per_token <= 0:
            return "n/a"
        return f"{self.tokens * self.bytes_per_token / seconds / 1e9:.2f}"

    def format(self, rid: Any) -> str:
        return (
            f"HiCache buffer stream restore req={rid} ok={self.ok} "
            f"tokens={self.tokens} chunks={self.chunks} inflight={self.inflight} "
            f"GB={self.tokens * self.bytes_per_token / 1e9:.2f} "
            f"read_s={self.read_s:.3f} ssd_GBps={self._rate(self.read_s)} "
            f"h2d_s={self.h2d_s:.3f} h2d_GBps={self._rate(self.h2d_s)} "
            f"slot_wait_s={self.slot_wait_s:.3f} "
            f"grant_wait_s={self.grant_wait_s:.3f} wall_s={self.wall_s:.3f} "
            f"overlap={self.overlap_ratio():.2f} "
            f"step_ms={self.mean_step_s() * 1e3:.2f} "
            f"step_ms_baseline={self.baseline_step_s * 1e3:.2f}"
        )

    def mean_step_s(self) -> float:
        return self.step_s_sum / self.steps if self.steps else 0.0


def h2d_quota_tokens(bytes_per_token: int, page_size: int) -> int:
    """Per-iteration H2D quota in page-aligned tokens (0 = unthrottled)."""
    quota_mb = envs.SGLANG_HICACHE_BUFFER_STREAM_H2D_MB_PER_ITER.get()
    if quota_mb <= 0 or bytes_per_token <= 0:
        return 0
    tokens = (quota_mb << 20) // bytes_per_token
    return max(page_size, tokens - tokens % page_size)


def split_piece_ranges(start: int, end: int, piece_tokens: int) -> list[tuple[int, int]]:
    """``[start, end)`` in pieces of at most ``piece_tokens`` (0 = one piece)."""
    if piece_tokens <= 0:
        return [(start, end)]
    return [(s, min(end, s + piece_tokens)) for s in range(start, end, piece_tokens)]


class StreamGate:
    """Scheduler-to-IO-worker handshake of one streamed restore.

    Rows: the scheduler reserves a chunk's unpublished device rows (evicting
    cold tree nodes, never touching locked ones) and grants them; the IO
    worker takes the grant before it queues that chunk's H2D.

    Credit: each scheduler iteration deposits the H2D quota (tokens, capped
    at one iteration's quota so idle iterations do not bank a burst); the IO
    worker spends it piece by piece before queueing each piece's copy, so
    the restore never takes more than the quota of the host link per
    forward."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._grants: dict[int, Any] = {}
        self._closed = False
        self._credit = 0
        # Chunks the IO worker has taken (read by the scheduler; int under GIL).
        self.taken = 0

    def grant(self, index: int, grant: Any) -> None:
        with self._cv:
            self._grants[index] = grant
            self._cv.notify_all()

    def deposit(self, tokens: int, cap: int) -> None:
        with self._cv:
            self._credit = min(cap, self._credit + tokens)
            self._cv.notify_all()

    def spend(
        self, tokens: int, cancelled: Callable[[], bool], poll_s: float = 0.02
    ) -> bool:
        """Block until ``tokens`` of credit are available and take them."""
        with self._cv:
            while self._credit < tokens:
                if self._closed or cancelled():
                    return False
                self._cv.wait(poll_s)
            self._credit -= tokens
            return True

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def wait(
        self, index: int, cancelled: Callable[[], bool], poll_s: float = 0.02
    ) -> Optional[Any]:
        with self._cv:
            while True:
                if index in self._grants:
                    self.taken += 1
                    return self._grants.pop(index)
                if self._closed or cancelled():
                    return None
                self._cv.wait(poll_s)


def drive_stream(
    chunks: Sequence[StreamChunk],
    *,
    read_chunk: Callable[[StreamChunk], bool],
    acquire: Callable[[StreamChunk], Optional[Any]],
    submit_h2d: Callable[[StreamChunk, Any], tuple[Any, bool]],
    wait_h2d: Callable[[Any], None],
    h2d_seconds: Callable[[Any], float],
    stats: StreamStats,
    clock: Callable[[], float] = time.perf_counter,
) -> bool:
    """Read each chunk into its ring slot, take its device grant and queue
    its H2D, reusing a slot only after the H2D of its previous chunk
    completed. Storage reads of chunk k+1 overlap the H2D of chunk k.
    ``submit_h2d`` returns ``(handle, ok)``; a not-ok submission may have
    queued part of the chunk, which ``handle`` still covers. Every queued
    H2D is waited for before returning, success or not: the ring and device
    rows are released by the caller afterwards."""
    slots = max(c.slot for c in chunks) + 1 if chunks else 0
    handles: dict[int, Any] = {}
    waited: set[int] = set()
    ok = True
    start = clock()
    try:
        for chunk in chunks:
            prev = chunk.index - slots
            if prev in handles and prev not in waited:
                t0 = clock()
                wait_h2d(handles[prev])
                waited.add(prev)
                stats.slot_wait_s += clock() - t0
            t0 = clock()
            ok = read_chunk(chunk)
            stats.read_s += clock() - t0
            if not ok:
                break
            t0 = clock()
            grant = acquire(chunk)
            stats.grant_wait_s += clock() - t0
            if grant is None:
                ok = False
                break
            handles[chunk.index], ok = submit_h2d(chunk, grant)
            if not ok:
                break
    except BaseException:
        ok = False
        raise
    finally:
        for index, handle in handles.items():
            if index not in waited:
                wait_h2d(handle)
        stats.h2d_s = sum(h2d_seconds(h) for h in handles.values())
        stats.wall_s = clock() - start
        stats.ok = ok and len(handles) == len(chunks)
    return stats.ok


@dataclass
class BufferStreamPlan:
    """Scheduler-owned state of one streamed restore, attached to its
    prefetch operation as ``buffer_stream``. Granted device rows are
    reserved but unpublished: they belong to the plan until ``staged``
    hands them to the staged prefetch, which publishes them at admission."""

    request: Any
    chunks: list[StreamChunk]
    slot_tokens: int
    ring_host_indices: torch.Tensor
    ring_units: int
    engine: Any
    stats: StreamStats
    gate: StreamGate = field(default_factory=StreamGate)
    device_parts: dict = field(default_factory=dict)
    next_grant: int = 0
    # Per-iteration H2D quota and copy piece size in tokens (0 = unthrottled).
    quota_tokens: int = 0
    staged: bool = False
    finished: bool = False
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def tokens(self) -> int:
        return self.chunks[-1].end

    def granted_tokens(self) -> int:
        return sum(len(rows) for rows in self.device_parts.values())

    def device_rows(self) -> torch.Tensor:
        """All chunks' device rows in hit order; valid once every chunk is granted."""
        assert len(self.device_parts) == len(self.chunks)
        return torch.cat([self.device_parts[c.index] for c in self.chunks])

    def take_device_parts(self) -> list:
        parts = list(self.device_parts.values())
        self.device_parts = {}
        return parts


def _completion_seconds(completion) -> float:
    if not getattr(completion, "timing_enabled", False):
        return 0.0
    try:
        return completion.start_event.elapsed_time(completion.finish_event) / 1000.0
    except Exception:
        return 0.0


def resolve_stream_device(device, engine) -> torch.device:
    """Return ``device`` with an explicit index (``set_device`` rejects bare ``cuda``).

    The H2D stream was created on the scheduler thread's current device, so its
    index is authoritative; this worker thread's own current device defaults to 0.
    """
    device = torch.device(device)
    if device.type != "cuda" or device.index is not None:
        return device
    stream_device = getattr(
        getattr(engine, "host_to_device_stream", None), "device", None
    )
    if isinstance(stream_device, torch.device) and stream_device.index is not None:
        return stream_device
    return torch.device("cuda", torch.cuda.current_device())


def streamed_page_transfer(cc, operation) -> int:
    """IO-worker body of a streamed prefetch; returns the completed KV pages
    (all or none). Puts exactly one progress ack per storage batch, like the
    regular transfer, so the cross-rank ack reduction stays paired; every
    ack but the last carries zero completed tokens."""
    from sglang.srt.managers.cache_controller import PrefetchAck
    from sglang.srt.mem_cache.hicache_storage import (
        STORAGE_BATCH_SIZE,
        HiCacheStorageExtraInfo,
        PoolName,
        PoolTransfer,
    )

    plan: BufferStreamPlan = operation.buffer_stream
    page_size = cc.page_size
    hashes = operation.hash_value
    total_pages = len(hashes)
    if total_pages * page_size != plan.tokens:
        raise RuntimeError(
            f"streamed prefetch span changed: {total_pages} pages vs {plan.tokens}"
        )
    kv_derived = [
        t
        for t in operation.pool_transfers or ()
        if t.indices_from_pool == PoolName.KV
    ]
    planned_acks = sum(
        len(chunk_batches(c, page_size, STORAGE_BATCH_SIZE)) for c in plan.chunks
    )
    sent = [0]
    prefix_keys = [operation.prefix_keys]

    def put_ack(completed_tokens: int) -> None:
        cc.prefetch_sync_queue.put(
            PrefetchAck(
                rid=operation.request_id,
                completed_tokens=completed_tokens,
                operation=operation,
            )
        )
        sent[0] += 1

    def read_chunk(chunk: StreamChunk) -> bool:
        ring = plan.ring_host_indices[ring_slice(chunk, plan.slot_tokens)]
        first_page = chunk.start // page_size
        for start, end in chunk_batches(chunk, page_size, STORAGE_BATCH_SIZE):
            if operation.is_terminated() or cc.storage_stop_event.is_set():
                return False
            batch_hashes = hashes[start:end]
            offset = (start - first_page) * page_size
            batch_host = ring[offset : offset + (end - start) * page_size]
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys[0])
            try:
                hit_pages = cc._page_transfer_kv_batch(
                    operation, batch_hashes, batch_host, extra_info, kv_derived
                )
            except Exception:
                logger.exception(
                    "HiCache streamed prefetch read failed for request %s",
                    operation.request_id,
                )
                hit_pages = 0
            if prefix_keys[0] is not None:
                prefix_keys[0] = prefix_keys[0] + batch_hashes
            if sent[0] < planned_acks - 1:
                put_ack(0)
            if hit_pages != len(batch_hashes):
                return False
        return True

    def cancelled() -> bool:
        return operation.is_terminated() or cc.storage_stop_event.is_set()

    def acquire(chunk: StreamChunk):
        return plan.gate.wait(chunk.index, cancelled)

    def submit_piece(ring, device):
        op = SimpleNamespace(
            host_indices=ring,
            device_indices=device,
            pool_transfers=[
                PoolTransfer(
                    name=t.name,
                    host_indices=ring,
                    device_indices=device,
                    indices_from_pool=PoolName.KV,
                )
                for t in kv_derived
            ]
            or None,
        )
        host_indices, device_indices, pool_transfers = cc._move_op_indices(op)
        transfers = cc._l2_load_transfers(host_indices, device_indices, pool_transfers)
        return plan.engine.submit_host_to_device(
            transfers, transfer_layer_id_max=cc.transfer_layer_id_max
        )

    def submit_h2d(chunk: StreamChunk, grant):
        device, fence = grant
        ring = plan.ring_host_indices[ring_slice(chunk, plan.slot_tokens)]
        if fence is not None:
            # The rows were freed by forwards queued before the grant: copy
            # after them (the forward launched next runs concurrently).
            plan.engine.host_to_device_stream.wait_event(fence)
        completions = []
        for start, end in split_piece_ranges(0, chunk.tokens, plan.quota_tokens):
            # Spend this iteration's quota right before the copy, so it is
            # queued just after the scheduler launched the next forward.
            if plan.quota_tokens and not plan.gate.spend(end - start, cancelled):
                return completions, False
            completions.append(submit_piece(ring[start:end], device[start:end]))
        return completions, True

    def wait_h2d(completions) -> None:
        for completion in completions:
            completion.finish_event.synchronize()

    ok = False
    device = resolve_stream_device(cc.device, plan.engine)
    try:
        if device.type == "cuda":
            # The current device is per thread; the worker's default is 0.
            torch.cuda.set_device(device)
        ok = drive_stream(
            plan.chunks,
            read_chunk=read_chunk,
            acquire=acquire,
            submit_h2d=submit_h2d,
            wait_h2d=wait_h2d,
            h2d_seconds=lambda cs: sum(_completion_seconds(c) for c in cs),
            stats=plan.stats,
        )
    finally:
        while sent[0] < planned_acks - 1:
            put_ack(0)
        put_ack(plan.tokens if ok else 0)
    return total_pages if ok else 0
