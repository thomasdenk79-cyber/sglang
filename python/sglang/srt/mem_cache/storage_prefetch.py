from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import msgspec

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)

# OpenAI error.type of a request whose confirmed storage hit missed its deadline;
# carried explicitly from the scheduler abort to the HTTP and SSE error objects.
RESTORE_TIMEOUT_ERR_TYPE = "restore_timeout"


@dataclass
class _StoragePrefetchRetry:
    immediate: bool
    storage_hit_end: Optional[int] = None
    due_step: Optional[int] = None


class StoragePrefetchRetries:
    """New L3 attempts only (parked IO and retained staging retry at their owner);
    step deadlines keep TP ranks in lockstep, retries follow queue order, and a
    request past its re-issue budget is admitted with what the device holds."""

    def __init__(self):
        self._pending: dict[str, _StoragePrefetchRetry] = {}
        self._step = 0
        self.promises = RestorePromises()

    def poll_miss(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(False, storage_hit_end)

    def refetch(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(True, storage_hit_end)

    def cancel(self, req_id: str) -> None:
        self._pending.pop(req_id, None)

    def is_pending(self, req_id: str) -> bool:
        return req_id in self._pending

    def clear(self) -> None:
        self._pending.clear()

    def pop_ready(
        self,
        waiting_queue: list[Req],
        interval: int,
        max_attempts: int,
        *,
        uncapped_immediate: bool = False,
    ) -> list[tuple[Req, Optional[int]]]:
        self._step += 1
        if not waiting_queue or not self._pending:
            return []
        # A speculative miss must never delay the queue head.
        head_id = waiting_queue[0].rid
        head_retry = self._pending.get(head_id)
        if head_retry is not None and not head_retry.immediate:
            self.cancel(head_id)
        if not any(
            retry.due_step is None or retry.due_step <= self._step
            for retry in self._pending.values()
        ):
            return []

        ready = []
        for req in waiting_queue:
            retry = self._pending.get(req.rid)
            if retry is None:
                continue
            if req.storage_prefetch_retry_attempts >= max_attempts and not (
                uncapped_immediate and retry.immediate
            ):
                self.cancel(req.rid)
                continue
            if not retry.immediate:
                if interval <= 0:
                    self.cancel(req.rid)
                    continue
                if retry.due_step is None:
                    retry.due_step = self._step + interval
                if retry.due_step > self._step:
                    continue
            self.cancel(req.rid)
            ready.append((req, retry.storage_hit_end))
        return ready


def restore_wait_s(missing_tokens: int, min_wait_s: float) -> float:
    """Restore deadline of a confirmed hit of ``missing_tokens`` storage tokens."""
    return max(min_wait_s, 1.0 + 0.25 * math.ceil(missing_tokens / 1000))


class RestoreState(enum.Enum):
    PROMISED = "promised"
    STAGING = "staging"
    STAGED = "staged"
    LOADING = "loading"
    ADMITTED = "admitted"
    TERMINAL = "terminal"


class RestorePromise(msgspec.Struct, kw_only=True):
    """One confirmed storage hit of one request generation. Span and deadline
    are fixed at t0 (the first confirmed hit); retries, polls, prefix shrinks,
    staging releases and requeues before admission keep them."""

    rid: str
    generation: int
    promised_end: int
    required_tokens: int
    t0: float
    deadline: float
    state: RestoreState = RestoreState.PROMISED
    # Rank-agreed verdict of the tick vote; set once, never cleared.
    expired: bool = False
    # Gate-issued re-reads after the span was lost with nothing in flight.
    rearms: int = 0
    operation_ids: list[int] = []

    @property
    def span_start(self) -> int:
        return self.promised_end - self.required_tokens


class RestorePromises:
    """Live restore promises by rid, in creation order. Creation, admission,
    expiry and abandonment all happen at rank-agreed points, so the live set
    (and the vote over it) is identical on every rank."""

    def __init__(self):
        self._live: dict[str, RestorePromise] = {}
        self._operation_generation: dict[int, int] = {}
        self._next_generation = 0
        self._timeouts: list[RestorePromise] = []

    def __len__(self) -> int:
        return len(self._live)

    def get(self, rid: str) -> Optional[RestorePromise]:
        return self._live.get(rid)

    def pending(self, rid: str) -> Optional[RestorePromise]:
        """The promise still owed a restore: live and not expired."""
        promise = self._live.get(rid)
        return None if promise is None or promise.expired else promise

    def confirm(
        self,
        *,
        rid: str,
        operation_id: int,
        promised_end: int,
        missing_tokens: int,
        now: float,
        min_wait_s: float,
    ) -> RestorePromise:
        """Create the promise at the first confirmed hit; a later hit of the
        same generation only binds its operation."""
        promise = self._live.get(rid)
        if promise is None:
            promise = RestorePromise(
                rid=rid,
                generation=self._next_generation,
                promised_end=promised_end,
                required_tokens=missing_tokens,
                t0=now,
                deadline=now + restore_wait_s(missing_tokens, min_wait_s),
            )
            self._next_generation += 1
            self._live[rid] = promise
        promise.operation_ids.append(operation_id)
        self._operation_generation[operation_id] = promise.generation
        return promise

    def advance(self, *, operation_id: int, state: RestoreState) -> bool:
        """Record progress of one operation; a completion of a retired
        generation (or an unbound operation) is ignored."""
        generation = self._operation_generation.get(operation_id)
        if generation is None:
            return False
        promise = next(
            (p for p in self._live.values() if p.generation == generation), None
        )
        if promise is None or promise.expired:
            return False
        promise.state = state
        return True

    def reset_to_promised(self, rid: str) -> None:
        promise = self.pending(rid)
        if promise is not None:
            promise.state = RestoreState.PROMISED

    def expiry_votes(self, now: float) -> list[int]:
        """Local verdicts in live order; already expired promises vote 0."""
        return [
            int(not p.expired and now > p.deadline) for p in self._live.values()
        ]

    def apply_verdicts(self, verdicts: list[int], now: float) -> list[RestorePromise]:
        expired = []
        for promise, verdict in zip(list(self._live.values()), verdicts):
            if verdict and not promise.expired:
                promise.expired = True
                promise.state = RestoreState.TERMINAL
                self._log_outcome(
                    promise, "restore_timeout", restored_tokens=0, now=now
                )
                expired.append(promise)
        self._timeouts.extend(expired)
        return expired

    def take_timeouts(self) -> list[RestorePromise]:
        timeouts, self._timeouts = self._timeouts, []
        return timeouts

    def admit(self, rid: str, *, covered_end: int, now: float) -> None:
        """The request entered running: retire its promise as restored."""
        promise = self._live.get(rid)
        if promise is None or promise.expired:
            return
        promise.state = RestoreState.ADMITTED
        restored = max(0, min(covered_end, promise.promised_end) - promise.span_start)
        self._log_outcome(promise, "restored", restored_tokens=restored, now=now)
        self._retire(rid)

    def drop(self, rid: str) -> None:
        """The request ended (abort, timeout abort); its generation retires."""
        if rid in self._live:
            self._retire(rid)

    def digest_keys(self) -> Iterator[str]:
        for promise in self._live.values():
            yield f"{promise.rid}\0{promise.generation}\0{int(promise.expired)}"

    def _retire(self, rid: str) -> None:
        promise = self._live.pop(rid)
        for operation_id in promise.operation_ids:
            self._operation_generation.pop(operation_id, None)

    @staticmethod
    def _log_outcome(
        promise: RestorePromise, outcome: str, *, restored_tokens: int, now: float
    ) -> None:
        # Prompt-free by contract: ids and counters only.
        logger.info(
            "HiCache restore outcome req=%s generation=%d outcome=%s wait_ms=%d "
            "restored_tokens=%d required_tokens=%d",
            promise.rid,
            promise.generation,
            outcome,
            max(0, int((now - promise.t0) * 1000)),
            restored_tokens,
            promise.required_tokens,
        )


class WindowLoadBackPlan(msgspec.Struct, frozen=True):
    """Logical coverage and the missing subranges that need new device rows."""

    start: int
    end: int
    missing_ranges: tuple[tuple[int, int], ...]

    @property
    def num_tokens(self) -> int:
        return sum(end - start for start, end in self.missing_ranges)


@dataclass(frozen=True)
class StagedPrefetchPlan:
    """One admission pass, computed before promoting the joint component match."""

    operation_id: int
    key: RadixKey
    device_prefix_len: int
    full_tokens: int
    windows: dict[ComponentType, WindowLoadBackPlan]
    mamba_slots: int = 0
    # Each window slot's sub-pool, the name its host hit goes by outside the tree.
    window_sub_pools: dict[ComponentType, str] = field(default_factory=dict)

    def window_tokens(self, component_type: ComponentType) -> int:
        window = self.windows.get(component_type)
        return window.num_tokens if window is not None else 0

    def window_host_hit_lengths(self) -> dict[str, int]:
        """Host hit tokens per window sub-pool."""
        return {
            self.window_sub_pools[ct]: window.num_tokens
            for ct, window in self.windows.items()
        }
