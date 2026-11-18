"""Dynamic extension of a reduced speculative draft vocabulary.

A static hot-token map (``--speculative-token-map``) cannot know topic words such
as identifiers that mostly appear in the prompt. This appends ``num_slots``
dynamic rows after the static rows of the sliced draft head, scored with the
target LM head rows and shared by all running requests. See
``dynamic_draft_vocab.md`` next to this module for the policies and env knobs.

Row layout (``hot_token_id`` / ``head``)::

    [protected static | ring (FIFO, tier A) | pool (scored, tier B) | sink]

* ``fifo`` (default): ring only; distinct new prompt/output tokens are inserted
  in arrival order and the oldest is evicted, a reuse does not refresh it.
* ``lru``: pool only; every staged occurrence refreshes the token's recency.
* ``lfu``: pool only; exponentially decayed frequency, output tokens weigh more.
* ``tiered``: ring fed by prompt and output tokens plus a "proven" pool fed only
  by output (accepted) tokens. An output token in the ring is promoted by
  swapping it with the pool victim, which gets the second chance in the ring.

With ``adaptive_static`` only the first ``static_core`` ids of the static map
stay protected; the others start in the pool as if last used ``static_horizon``
tokens in the future, so they are demoted once unused for that long.

Columns always hold unique target ids: empty slots are prefilled with distinct
absent tokens, so ``hot_token_id`` stays a valid draft-to-target map for
rejection sampling. ``hot_token_id`` and ``head`` are views of buffers that are
only updated in place with fixed shapes, so captured CUDA graphs see the current
vocabulary; ``commit`` only launches device ops (no host sync).

Tokens are staged with ``stage`` and inserted by ``commit``. The worker commits at
one point per decode round, after the round's draft probabilities were scattered
to the target vocab and before the draft extend produces the next ones, so every
draft distribution is mapped with the vocabulary it was computed with.

Adapted from the V3 dynamic draft vocabulary of
https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000 (MIT).
"""

import atexit
import hashlib
import logging
import math
import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)

_STATIC = -3
_ABSENT = -1

POLICIES = ("fifo", "lru", "tiered", "lfu")
SCORE_MODES = ("lru", "lfu")
STATE_VERSION = 1
# Score of prefilled placeholder pool slots: always the first victims.
_EMPTY = -1e300
# Largest exponent of a decayed-LFU increment before the scores are rescaled.
_MAX_EXP = 600.0

# Counter slots. Lane categories 0..4 come first so one scatter_add fills them.
_C_IGNORED, _C_MISS, _C_HIT_STATIC, _C_HIT_RING, _C_HIT_POOL = range(5)
_C_RING_INS, _C_POOL_INS, _C_PROMOTE, _C_EVICT, _C_DEMOTE, _C_COMMITS = range(5, 11)
# Gauges (latest value, not cumulative).
_C_ACTIVE, _C_EXPIRED = 11, 12
# Expired pool slots sort below live ones (by last use) in the victim key.
_EXPIRED_OFFSET = 1e15
_NUM_COUNTERS = 16


@dataclass
class DynamicVocabConfig:
    policy: str = "fifo"
    # Tier B ("proven") pool size for ``tiered``.
    proven_slots: int = 0
    # Pool score for ``tiered``: lru or decayed lfu (``lru``/``lfu`` imply it).
    proven_score: str = "lru"
    half_life: float = 65536.0
    output_weight: float = 4.0
    commit_every: int = 1
    adaptive_static: bool = False
    static_core: int = 8192
    # Optional ranked id list (torch.save'd list/tensor or JSON list / {"ids"});
    # orders the static map before the core is cut. Default: map file order.
    static_rank: str = ""
    static_horizon: float = 1048576.0
    state_dir: str = ""
    profile: str = "default"
    save_interval: float = 300.0
    metrics: bool = False
    # Pool tokens without an output use for ttl_s seconds are evicted first
    # (0 disables); at least min_rows rows always stay live.
    ttl_s: float = 0.0
    min_rows: int = 0

    @classmethod
    def from_env(cls) -> "DynamicVocabConfig":
        from sglang.srt.environ import envs

        return cls(
            policy=envs.SGLANG_SPEC_DYNAMIC_VOCAB_POLICY.get(),
            proven_slots=envs.SGLANG_SPEC_DYNAMIC_VOCAB_PROVEN_SLOTS.get(),
            proven_score=envs.SGLANG_SPEC_DYNAMIC_VOCAB_PROVEN_SCORE.get(),
            half_life=envs.SGLANG_SPEC_DYNAMIC_VOCAB_HALF_LIFE.get(),
            output_weight=envs.SGLANG_SPEC_DYNAMIC_VOCAB_OUTPUT_WEIGHT.get(),
            commit_every=envs.SGLANG_SPEC_DYNAMIC_VOCAB_COMMIT_EVERY.get(),
            adaptive_static=envs.SGLANG_SPEC_DYNAMIC_VOCAB_ADAPTIVE_STATIC.get(),
            static_core=envs.SGLANG_SPEC_DYNAMIC_VOCAB_STATIC_CORE.get(),
            static_rank=envs.SGLANG_SPEC_DYNAMIC_VOCAB_STATIC_RANK.get(),
            static_horizon=envs.SGLANG_SPEC_DYNAMIC_VOCAB_STATIC_HORIZON.get(),
            state_dir=envs.SGLANG_SPEC_DYNAMIC_VOCAB_STATE_DIR.get(),
            profile=envs.SGLANG_SPEC_DYNAMIC_VOCAB_PROFILE.get(),
            save_interval=envs.SGLANG_SPEC_DYNAMIC_VOCAB_SAVE_INTERVAL.get(),
            metrics=envs.SGLANG_SPEC_DYNAMIC_VOCAB_METRICS.get(),
            ttl_s=envs.SGLANG_SPEC_DYNAMIC_VOCAB_TTL_S.get(),
            min_rows=envs.SGLANG_SPEC_DYNAMIC_VOCAB_MIN_ROWS.get(),
        )

    @property
    def score_mode(self) -> str:
        return {"lru": "lru", "lfu": "lfu"}.get(self.policy, self.proven_score)


def token_map_hash(hot_token_id: torch.Tensor) -> str:
    ids = hot_token_id.detach().to("cpu", torch.int64).contiguous().numpy()
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


class DynamicDraftVocab:
    """Static hot rows followed by ``num_slots`` dynamic rows."""

    def __init__(
        self,
        hot_token_id: torch.Tensor,
        target_head: torch.Tensor,
        num_slots: int,
        vocab_size: int,
        max_observe: int = 0,
        config: Optional[DynamicVocabConfig] = None,
    ) -> None:
        cfg = config or DynamicVocabConfig()
        if cfg.policy not in POLICIES:
            raise ValueError(f"Unknown dynamic vocab policy {cfg.policy!r}.")
        if cfg.score_mode not in SCORE_MODES:
            raise ValueError(f"Unknown dynamic vocab score {cfg.proven_score!r}.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if cfg.policy == "tiered" and cfg.proven_slots <= 0:
            raise ValueError("The tiered policy needs proven_slots > 0.")
        if cfg.adaptive_static and cfg.policy == "fifo":
            raise ValueError("adaptive_static needs a scored policy, not fifo.")
        if cfg.ttl_s > 0 and cfg.policy == "fifo":
            raise ValueError("ttl_s needs a scored policy, not fifo.")
        if cfg.commit_every <= 0 or cfg.half_life <= 0:
            raise ValueError("commit_every and half_life must be positive.")
        if target_head.dim() != 2 or target_head.shape[0] < vocab_size:
            raise ValueError(
                "The dynamic draft vocabulary needs the full, unsharded target "
                f"LM head; got {tuple(target_head.shape)} for vocab {vocab_size}."
            )
        if not target_head.is_floating_point():
            raise ValueError(
                "The dynamic draft vocabulary needs an unquantized target LM head; "
                f"got {target_head.dtype}."
            )
        target_head = target_head.detach()
        device = target_head.device
        hot = hot_token_id.detach().to("cpu", torch.int64).flatten()
        num_static = hot.numel()
        if num_static == 0 or int(hot.min()) < 0 or int(hot.max()) >= vocab_size:
            raise ValueError("hot_token_id must hold ids within the target vocab.")
        if torch.unique(hot).numel() != num_static:
            raise ValueError("hot_token_id must hold unique ids.")
        map_hash = token_map_hash(hot)
        if cfg.static_rank:
            hot = rank_static_map(hot, load_rank_file(cfg.static_rank))
        core = num_static
        if cfg.adaptive_static:
            # 0 makes the whole map a seed that ages out like any token.
            core = max(0, min(cfg.static_core, num_static))
        num_ring = num_slots if cfg.policy in ("fifo", "tiered") else 0
        num_pool = {"fifo": 0, "tiered": cfg.proven_slots}.get(cfg.policy, num_slots)
        num_pool += num_static - core
        total = num_ring + num_pool
        if total - (num_static - core) > vocab_size - num_static:
            raise ValueError(
                f"num_slots={total - (num_static - core)} exceeds the "
                f"{vocab_size - num_static} tokens outside the static map."
            )

        self.config = cfg
        self.policy = cfg.policy
        self.vocab_size = vocab_size
        self.num_static = num_static
        self.num_hot = core
        self.num_ring = num_ring
        self.num_pool = num_pool
        self.num_slots = total
        self.num_rows = core + total
        self.max_observe = max_observe if max_observe > 0 else 16 * total
        self.map_hash = map_hash
        self._lam = math.log(2.0) / cfg.half_life
        self._clock = 0.0  # staged-token time of the scores
        self._tbase = 0.0  # time at which decayed-LFU scores are unscaled
        self._stamp = 0
        self._commit_calls = 0

        layout = self._initial_layout(hot, core, self._load_state())
        ring_tok, pool_tok, pool_score, pool_origin, pool_age, cursor = layout

        slot_tok = torch.cat((ring_tok, pool_tok))
        # Index vocab_size is a sink for masked lanes and always reads static.
        slot_of = torch.full((vocab_size + 1,), _ABSENT, dtype=torch.int64)
        slot_of[hot[:core]] = _STATIC
        slot_of[vocab_size] = _STATIC
        slot_of[slot_tok] = torch.arange(total)
        self.slot_of = slot_of.to(device)

        # Row num_rows is a sink for masked lanes, outside the views.
        ids = torch.empty(self.num_rows + 1, dtype=torch.int64)
        ids[:core] = hot[:core]
        ids[core : self.num_rows] = slot_tok
        ids[-1] = ids[0]
        self._ids = ids.to(device)
        self._rows = target_head[self._ids]
        self._target_head = target_head
        self._cursor = torch.tensor([cursor], dtype=torch.int64, device=device)
        # Last arrival stamp per token; stamps grow across calls, so no reset.
        self._last_seen = torch.full(
            (vocab_size + 1,), -1, dtype=torch.int64, device=device
        )
        # Pool scores (index num_pool is a sink) and static-origin flags.
        self._score = torch.cat((pool_score, torch.zeros(1, dtype=torch.float64)))
        self._score = self._score.to(device)
        self._origin = torch.cat((pool_origin, torch.zeros(1, dtype=torch.bool)))
        self._origin = self._origin.to(device)
        # Monotonic seconds of the last output use per pool slot (TTL expiry).
        self._t0 = time.monotonic()
        self._last_out = torch.cat((-pool_age, torch.zeros(1, dtype=torch.float64)))
        self._last_out = self._last_out.to(device)
        self._expired_mask = torch.zeros(num_pool + 1, dtype=torch.bool, device=device)
        self._expired_until = -math.inf
        if num_pool and cfg.score_mode == "lfu":
            self._acc = torch.zeros(vocab_size + 1, dtype=torch.float64, device=device)
        self._counters = torch.zeros(_NUM_COUNTERS, dtype=torch.int64, device=device)
        self._metrics_buf = None
        self._metrics_event = None
        self._metrics_last = torch.zeros(_NUM_COUNTERS, dtype=torch.int64)
        self._pending: Optional[torch.Tensor] = None
        self._pending_out: Optional[torch.Tensor] = None
        self._init_persistence(device)

    # ------------------------------------------------------------------ views

    @property
    def hot_token_id(self) -> torch.Tensor:
        return self._ids[: self.num_rows]

    @property
    def head(self) -> torch.Tensor:
        return self._rows[: self.num_rows]

    # ----------------------------------------------------------------- update

    @torch.no_grad()
    def stage(
        self,
        token_ids: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        output: bool = False,
    ) -> None:
        """Queue tokens for the next ``commit``; ``valid`` masks lanes out.

        ``output`` marks generated (accepted) tokens; they feed the proven tier
        and the coverage metrics. Only the most recent ``max_observe`` staged
        tokens are kept, which bounds the commit cost.
        """
        ids = token_ids.reshape(-1).to(device=self._ids.device, dtype=torch.int64)
        if valid is not None:
            ids = torch.where(valid.reshape(-1).to(ids.device), ids, self.vocab_size)
        out = torch.full(ids.shape, output, dtype=torch.bool, device=ids.device)
        if self._pending is not None:
            ids = torch.cat((self._pending, ids))
            out = torch.cat((self._pending_out, out))
        elif ids.data_ptr() == token_ids.data_ptr():
            # The caller's buffer may be reused before the commit.
            ids = ids.clone()
        self._pending = ids[-self.max_observe :]
        self._pending_out = out[-self.max_observe :]

    @torch.no_grad()
    def commit(self) -> None:
        """Insert the staged tokens (every ``commit_every`` calls)."""
        if self._pending is None:
            return
        self._commit_calls += 1
        if self._commit_calls % self.config.commit_every:
            return
        pending, out = self._pending, self._pending_out
        self._pending = self._pending_out = None
        self.observe(pending, out)

    @torch.no_grad()
    def observe(
        self, input_ids: torch.Tensor, output: Union[bool, torch.Tensor] = False
    ) -> None:
        """Insert or refresh tokens according to the policy."""
        ids = input_ids.reshape(-1)[-self.max_observe :]
        n = ids.numel()
        if n == 0:
            return
        sink = self.vocab_size
        ids = ids.to(torch.int64)
        tok = torch.where((ids >= 0) & (ids < sink), ids, sink)
        if isinstance(output, torch.Tensor):
            out = output.reshape(-1)[-n:].to(tok.device)
        else:
            out = torch.full((n,), bool(output), dtype=torch.bool, device=tok.device)
        if self.config.metrics:
            self._count_hits(tok, out)
        if self.num_pool:
            eligible = out if self.policy == "tiered" else None
            self._pool_update(tok, out, eligible)
        if self.num_ring:
            self._ring_insert(tok)
        self._clock += n
        if self._state_path is not None:
            self._maybe_save()

    def _ring_insert(self, tok: torch.Tensor) -> None:
        """FIFO insertion of distinct absent tokens into the ring slots."""
        n = tok.numel()
        sink = self.vocab_size
        new = self.slot_of[tok] == _ABSENT

        # The last occurrence of each distinct new token is its lane.
        stamp = torch.arange(self._stamp, self._stamp + n, device=tok.device)
        self._stamp += n
        self._last_seen.scatter_reduce_(
            0, torch.where(new, tok, sink), stamp, reduce="amax"
        )
        win = new & (self._last_seen[tok] == stamp)

        # Ring slots in arrival order; keep the most recent num_ring tokens.
        order = torch.cumsum(win, 0) - 1
        drop = (win.sum() - self.num_ring).clamp(min=0)
        win = win & (order >= drop)
        pos = torch.where(
            win, (self._cursor + order - drop) % self.num_ring, self.num_ring
        )
        row = torch.where(win, self.num_hot + pos, self.num_rows)

        prev = self._ids[row]
        self.slot_of.index_put_(
            (torch.where(win, prev, sink),),
            torch.where(win, _ABSENT, _STATIC),
        )
        self.slot_of.index_put_(
            (torch.where(win, tok, sink),),
            torch.where(win, pos, _STATIC),
        )
        src = torch.where(win, tok, self._ids[-1])
        self._ids.index_copy_(0, row, src)
        if n <= self.num_ring:
            self._rows.index_copy_(0, row, self._target_head[src])
        else:
            # Bound long prefill chunks to one ring-sized gather.
            ring = slice(self.num_hot, self.num_hot + self.num_ring)
            self._rows[ring] = self._target_head[self._ids[ring]]
        inserted = win.sum()
        self._cursor.copy_((self._cursor + inserted) % self.num_ring)
        if self.config.metrics:
            self._counters[_C_RING_INS] += inserted
        if self._ring_real is not None:
            self._ring_real.index_fill_(0, pos, True)

    def _lane_scores(self, n: int, out: torch.Tensor) -> torch.Tensor:
        t = torch.arange(n, dtype=torch.float64, device=out.device) + self._clock
        if self.config.score_mode == "lru":
            return t
        if (self._clock + n - self._tbase) * self._lam > _MAX_EXP:
            scale = math.exp(-(self._clock - self._tbase) * self._lam)
            live = self._score > 0
            self._score.copy_(torch.where(live, self._score * scale, self._score))
            self._tbase = self._clock
        weight = torch.where(out, self.config.output_weight, 1.0)
        return weight * torch.exp((t - self._tbase) * self._lam)

    def _pool_update(
        self, tok: torch.Tensor, out: torch.Tensor, eligible: Optional[torch.Tensor]
    ) -> None:
        """Refresh pool tokens and admit distinct candidates over the victims.

        Candidates are absent tokens, plus ring tokens for ``tiered`` (promotion).
        The k-th candidate in arrival order replaces the k-th lowest-scored pool
        slot if it scores higher; a promoted ring token swaps with its victim.
        """
        n = tok.numel()
        sink = self.vocab_size
        p0, num_pool = self.num_ring, self.num_pool
        lru = self.config.score_mode == "lru"
        slot = self.slot_of[tok]
        lane_ok = tok != sink
        if eligible is not None:
            lane_ok = lane_ok & eligible
        val = self._lane_scores(n, out)

        # Refresh tokens already in the pool.
        pidx = torch.where(lane_ok & (slot >= p0), slot - p0, num_pool)
        ttl = self.config.ttl_s > 0
        if ttl:
            now = self._now()
            used = torch.where(out, pidx, num_pool)
            self._last_out.index_fill_(0, used, now)
            expired = self._expired(now)
            expired.index_fill_(0, used, False)
        if lru:
            self._score.scatter_reduce_(0, pidx, val, reduce="amax")
        else:
            self._score.scatter_add_(0, pidx, val)

        # Distinct candidates at their last occurrence.
        cand = lane_ok & ((slot == _ABSENT) | ((slot >= 0) & (slot < p0)))
        ctok = torch.where(cand, tok, sink)
        stamp = torch.arange(self._stamp, self._stamp + n, device=tok.device)
        self._stamp += n
        self._last_seen.scatter_reduce_(0, ctok, stamp, reduce="amax")
        win = cand & (self._last_seen[tok] == stamp)
        if lru:
            cval = val
        else:
            self._acc.index_add_(0, ctok, val)
            cval = self._acc[tok]
            self._acc.index_fill_(0, ctok, 0.0)

        order = torch.cumsum(win, 0) - 1
        drop = (win.sum() - num_pool).clamp(min=0)
        win = win & (order >= drop)
        k = min(n, num_pool)
        key = self._score[:num_pool]
        if ttl:
            key = torch.where(
                expired[:num_pool], self._last_out[:num_pool] - _EXPIRED_OFFSET, key
            )
        victims = torch.topk(key, k, largest=False).indices
        vic = victims[(order - drop).clamp(0, k - 1)]
        beats = cval > self._score[vic]
        if ttl:
            beats = beats | expired[vic]
        admit = win & beats
        v = torch.where(admit, vic, num_pool)
        vslot = p0 + v  # slot num_slots maps to the sink row
        vrow = self.num_hot + vslot
        vtok = self._ids[vrow]
        from_ring = admit & (slot >= 0)
        if self.config.metrics:
            vscore = self._score[v]
            self._counters[_C_POOL_INS] += admit.sum()
            self._counters[_C_PROMOTE] += from_ring.sum()
            self._counters[_C_EVICT] += (admit & (vscore > _EMPTY)).sum()
            self._counters[_C_DEMOTE] += (admit & self._origin[v]).sum()

        # The victim leaves (or swaps into the promoted token's ring slot).
        drop_vic = admit & ~from_ring
        self.slot_of.index_put_(
            (torch.where(drop_vic, vtok, sink),),
            torch.where(drop_vic, _ABSENT, _STATIC),
        )
        self.slot_of.index_put_(
            (torch.where(from_ring, vtok, sink),),
            torch.where(from_ring, slot, _STATIC),
        )
        self.slot_of.index_put_(
            (torch.where(admit, tok, sink),),
            torch.where(admit, vslot, _STATIC),
        )
        filler = self._ids[-1]
        ring_row = torch.where(from_ring, self.num_hot + slot, self.num_rows)
        pool_src = torch.where(admit, tok, filler)
        ring_src = torch.where(from_ring, vtok, filler)
        self._ids.index_copy_(0, vrow, pool_src)
        self._ids.index_copy_(0, ring_row, ring_src)
        if n <= self.num_slots:
            self._rows.index_copy_(0, vrow, self._target_head[pool_src])
            self._rows.index_copy_(0, ring_row, self._target_head[ring_src])
        else:
            dyn = slice(self.num_hot, self.num_rows)
            self._rows[dyn] = self._target_head[self._ids[dyn]]
        self._score.index_copy_(0, v, cval)
        self._origin.index_fill_(0, v, False)
        if ttl:
            self._last_out.index_fill_(0, v, now)
            expired.index_fill_(0, v, False)
        if self._ring_real is not None:
            self._ring_real.index_fill_(
                0, torch.where(from_ring, slot, self.num_ring), True
            )

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def _expired(self, now: float) -> torch.Tensor:
        """Pool slots past the TTL (plus a sink), keeping ``min_rows`` live rows.

        Placeholder slots count as dead already; if too many tokens expired, the
        most recently used of them stay live. The mask is recomputed at most
        every ``ttl_s / 16`` (<= 1 s); callers clear the bits of used slots.
        """
        if now < self._expired_until:
            return self._expired_mask
        self._expired_until = now + min(1.0, self.config.ttl_s / 16)
        num_pool = self.num_pool
        last = self._last_out[:num_pool]
        real = self._score[:num_pool] > _EMPTY
        expired = real & (now - last > self.config.ttl_s)
        budget = self.num_rows - self.config.min_rows
        dead = (~real).sum()
        if budget < num_pool:
            # Expiring every expired slot may cut the live rows below the floor.
            allowed = (budget - dead).clamp(min=0)
            order = torch.argsort(torch.where(expired, last, math.inf))
            rank = torch.empty_like(order)
            rank[order] = torch.arange(num_pool, device=order.device)
            expired = expired & (rank < allowed)
        if self.config.metrics:
            live = self.num_rows - dead - expired.sum()
            self._counters[_C_ACTIVE] = live
            self._counters[_C_EXPIRED] = expired.sum()
        self._expired_mask[:num_pool] = expired
        return self._expired_mask

    # ---------------------------------------------------------------- metrics

    def _count_hits(self, tok: torch.Tensor, out: torch.Tensor) -> None:
        """Classify output lanes by where the vocab held the token (pre-update)."""
        s = self.slot_of[tok]
        lane = out & (tok != self.vocab_size)
        pslot = (s - self.num_ring).clamp(0, self.num_pool)
        cat = torch.where(
            s == _ABSENT,
            _C_MISS,
            torch.where(
                s == _STATIC,
                _C_HIT_STATIC,
                torch.where(
                    s < self.num_ring,
                    _C_HIT_RING,
                    torch.where(self._origin[pslot], _C_HIT_STATIC, _C_HIT_POOL),
                ),
            ),
        )
        cat = torch.where(lane, cat, _C_IGNORED)
        self._counters.scatter_add_(0, cat, torch.ones_like(cat))
        self._counters[_C_COMMITS] += 1

    def counters(self) -> Dict[str, int]:
        """Cumulative counters (synchronizes; for tests and offline use)."""
        return self._counters_dict(self._counters.cpu())

    @staticmethod
    def _counters_dict(c: torch.Tensor) -> Dict[str, int]:
        c = c.tolist()
        return {
            "out": c[_C_MISS] + c[_C_HIT_STATIC] + c[_C_HIT_RING] + c[_C_HIT_POOL],
            "miss": c[_C_MISS],
            "hit_static": c[_C_HIT_STATIC],
            "hit_ring": c[_C_HIT_RING],
            "hit_pool": c[_C_HIT_POOL],
            "ring_insert": c[_C_RING_INS],
            "pool_insert": c[_C_POOL_INS],
            "promote": c[_C_PROMOTE],
            "evict": c[_C_EVICT],
            "static_demote": c[_C_DEMOTE],
            "commits": c[_C_COMMITS],
            "active": c[_C_ACTIVE],
            "expired": c[_C_EXPIRED],
        }

    def metrics_delta(self) -> Optional[Dict[str, int]]:
        """Counters since the previous call, without a host sync.

        On CUDA this returns the snapshot the previous call started copying
        (None until one has landed) and starts the next asynchronous copy, so
        the numbers lag by one call.
        """
        if not self.config.metrics:
            return None
        if not self._counters.is_cuda:
            return self._delta(self._counters.clone())
        result = None
        if self._metrics_buf is not None and self._metrics_event.query():
            result = self._delta(self._metrics_buf.clone())
            self._metrics_buf = None
        if self._metrics_buf is None:
            buf = torch.empty(_NUM_COUNTERS, dtype=torch.int64, pin_memory=True)
            buf.copy_(self._counters, non_blocking=True)
            self._metrics_buf = buf
            self._metrics_event = torch.cuda.Event()
            self._metrics_event.record()
        return result

    def _delta(self, cur: torch.Tensor) -> Dict[str, int]:
        delta = cur - self._metrics_last
        delta[_C_ACTIVE : _C_EXPIRED + 1] = cur[_C_ACTIVE : _C_EXPIRED + 1]
        self._metrics_last = cur
        return self._counters_dict(delta)

    def log_suffix(self) -> str:
        """``draft vocab: cov=..`` for the decode log line, or '' (no sync)."""
        d = self.metrics_delta()
        if d is None or d["out"] == 0:
            return ""
        self.last_coverage = 1.0 - d["miss"] / d["out"]
        pct = lambda x: f"{100.0 * x / d['out']:.1f}%"  # noqa: E731
        msg = f"draft vocab: cov={pct(d['out'] - d['miss'])} static={pct(d['hit_static'])}"
        if self.num_ring:
            msg += f" A={pct(d['hit_ring'])}"
        if self.num_pool:
            msg += f" B={pct(d['hit_pool'])}"
        msg += f" ins={d['ring_insert'] + d['pool_insert']}"
        if self.policy == "tiered":
            msg += f" promo={d['promote']}"
        if self.num_pool:
            msg += f" evict={d['evict']}"
        if self.num_static > self.num_hot:
            msg += f" demote={d['static_demote']}"
        if self.config.ttl_s > 0:
            msg += f" active={d['active']} expired={d['expired']}"
        return msg

    last_coverage: float = -1.0

    # ------------------------------------------------------------ persistence

    def _initial_layout(self, hot: torch.Tensor, core: int, state: Optional[dict]):
        """Slot contents on CPU: ring tokens, pool tokens/scores/origin, cursor."""
        cfg = self.config
        num_ring, num_pool = self.num_ring, self.num_pool
        present = torch.zeros(self.vocab_size + 1, dtype=torch.bool)
        present[hot[:core]] = True
        demotable = hot[core:]
        is_demotable = torch.zeros_like(present)
        is_demotable[demotable] = True

        pool_tok = torch.empty(0, dtype=torch.int64)
        pool_score = torch.empty(0, dtype=torch.float64)
        pool_origin = torch.empty(0, dtype=torch.bool)
        pool_age = torch.empty(0, dtype=torch.float64)
        ring_tok = torch.empty(0, dtype=torch.int64)
        if state is not None and num_pool:
            tok, score, origin = (
                state["pool_ids"],
                state["pool_scores"],
                state["pool_origin"],
            )
            if state["score_mode"] != cfg.score_mode:
                # Keep the learned order with neutral scores of this mode.
                rank = torch.arange(1, tok.numel() + 1, dtype=torch.float64)
                score = -rank if cfg.score_mode == "lru" else 1.0 / rank
            keep = _first_unique(tok, present)[:num_pool]
            pool_tok, pool_score = tok[keep], score[keep]
            pool_origin = origin[keep] & is_demotable[pool_tok]
            pool_age = state["pool_out_age"][keep]
            present[pool_tok] = True
        if state is not None and num_ring:
            newest_first = state["ring_ids"].flip(0)
            keep = _first_unique(newest_first, present)[:num_ring]
            ring_tok = newest_first[keep].flip(0)
            present[ring_tok] = True
        cursor = ring_tok.numel() % num_ring if num_ring else 0
        self._loaded_ring = ring_tok.numel()

        if num_pool > pool_tok.numel() and demotable.numel():
            extra = demotable[~present[demotable]][: num_pool - pool_tok.numel()]
            if cfg.score_mode == "lru":
                init = cfg.static_horizon
            else:
                init = cfg.output_weight * math.exp(
                    min(cfg.static_horizon * self._lam, _MAX_EXP)
                )
            pool_tok = torch.cat((pool_tok, extra))
            pool_score = torch.cat(
                (pool_score, torch.full((extra.numel(),), init, dtype=torch.float64))
            )
            pool_origin = torch.cat(
                (pool_origin, torch.ones(extra.numel(), dtype=torch.bool))
            )
            pool_age = torch.cat(
                (pool_age, torch.zeros(extra.numel(), dtype=torch.float64))
            )
            present[extra] = True

        need_ring = num_ring - ring_tok.numel()
        need_pool = num_pool - pool_tok.numel()
        fill = torch.nonzero(~present[: self.vocab_size]).flatten()
        fill = fill[: need_ring + need_pool]
        ring_tok = torch.cat((ring_tok, fill[:need_ring]))
        pool_tok = torch.cat((pool_tok, fill[need_ring:]))
        pool_score = torch.cat(
            (pool_score, torch.full((need_pool,), _EMPTY, dtype=torch.float64))
        )
        pool_origin = torch.cat((pool_origin, torch.zeros(need_pool, dtype=torch.bool)))
        pool_age = torch.cat((pool_age, torch.zeros(need_pool, dtype=torch.float64)))
        return ring_tok, pool_tok, pool_score, pool_origin, pool_age, cursor

    @property
    def state_path(self) -> Optional[str]:
        cfg = self.config
        if not cfg.state_dir:
            return None
        return os.path.join(cfg.state_dir, f"dvocab-{cfg.profile}.pt")

    def _load_state(self) -> Optional[dict]:
        path = self.state_path
        if path is None or not os.path.exists(path):
            return None
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            header = (
                state.get("version"),
                state.get("vocab_size"),
                state.get("map_hash"),
            )
            if header != (STATE_VERSION, self.vocab_size, self.map_hash):
                logger.warning(
                    f"Ignoring dynamic draft vocab state {path}: header {header} "
                    f"does not match {(STATE_VERSION, self.vocab_size, self.map_hash)}."
                )
                return None
            for key in ("ring_ids", "pool_ids"):
                t = state[key]
                if t.dtype != torch.int64 or t.dim() != 1:
                    raise ValueError(f"bad {key}")
                if t.numel() and (int(t.min()) < 0 or int(t.max()) >= self.vocab_size):
                    raise ValueError(f"{key} out of vocab")
            if state["pool_scores"].shape != state["pool_ids"].shape:
                raise ValueError("pool_scores/pool_ids mismatch")
            if state["pool_origin"].shape != state["pool_ids"].shape:
                raise ValueError("pool_origin/pool_ids mismatch")
            state["pool_scores"] = state["pool_scores"].to(torch.float64)
            state["pool_origin"] = state["pool_origin"].to(torch.bool)
            age = state.get("pool_out_age")
            if age is None or age.shape != state["pool_ids"].shape:
                age = torch.zeros(state["pool_ids"].shape, dtype=torch.float64)
            state["pool_out_age"] = age.to(torch.float64).clamp(min=0)
            return state
        except Exception as e:  # corrupt or foreign file: start cold
            logger.warning(f"Ignoring dynamic draft vocab state {path}: {e}")
            return None

    def _init_persistence(self, device: torch.device) -> None:
        self._state_path = self.state_path
        self._ring_real = None
        self._save_thread: Optional[threading.Thread] = None
        if self._state_path is None:
            return
        os.makedirs(self.config.state_dir, exist_ok=True)
        real = torch.zeros(self.num_ring + 1, dtype=torch.bool)
        real[: self._loaded_ring] = True
        self._ring_real = real.to(device)
        self._next_save = time.monotonic() + self.config.save_interval
        atexit.register(_save_at_exit, weakref.ref(self))

    def _maybe_save(self) -> None:
        now = time.monotonic()
        if now < self._next_save:
            return
        if self._save_thread is not None and self._save_thread.is_alive():
            return
        self._next_save = now + self.config.save_interval
        snap = self._snapshot(non_blocking=True)
        self._save_thread = threading.Thread(
            target=self._write_snapshot, args=(snap,), daemon=True
        )
        self._save_thread.start()

    def _snapshot(self, non_blocking: bool) -> dict:
        on_cuda = self._ids.is_cuda and non_blocking
        dyn = slice(self.num_hot, self.num_rows)

        def host(t):
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=on_cuda)
            buf.copy_(t, non_blocking=on_cuda)
            return buf

        snap = {
            "ids": host(self._ids[dyn]),
            "score": host(self._score[: self.num_pool]),
            "origin": host(self._origin[: self.num_pool]),
            "last_out": host(self._last_out[: self.num_pool]),
            "now": self._now(),
            "ring_real": host(self._ring_real[: self.num_ring]),
            "cursor": host(self._cursor),
            "clock": self._clock,
            "tbase": self._tbase,
            "event": None,
        }
        if on_cuda:
            snap["event"] = torch.cuda.Event()
            snap["event"].record()
        return snap

    def _write_snapshot(self, snap: dict) -> None:
        try:
            if snap["event"] is not None:
                snap["event"].synchronize()
            ids = snap["ids"]
            cursor = int(snap["cursor"][0]) if self.num_ring else 0
            ring = ids[: self.num_ring].roll(-cursor)
            ring = ring[snap["ring_real"].roll(-cursor)]
            score = snap["score"]
            real = score > _EMPTY
            if self.config.score_mode == "lru":
                norm = snap["clock"] - score  # age; negative while protected
                order = torch.argsort(norm[real], stable=True)
            else:
                norm = score * math.exp(-(snap["clock"] - snap["tbase"]) * self._lam)
                order = torch.argsort(-norm[real], stable=True)
            state = {
                "version": STATE_VERSION,
                "vocab_size": self.vocab_size,
                "map_hash": self.map_hash,
                "policy": self.policy,
                "score_mode": self.config.score_mode,
                "saved_at": time.time(),
                "ring_ids": ring.clone(),
                "pool_ids": ids[self.num_ring :][real][order].clone(),
                "pool_scores": norm[real][order].clone(),
                "pool_origin": snap["origin"][real][order].clone(),
                "pool_out_age": (snap["now"] - snap["last_out"])[real][order]
                .clamp(min=0)
                .clone(),
            }
            if self.config.score_mode == "lru":
                # Stored as ages; loading restarts the clock at zero.
                state["pool_scores"] = -state["pool_scores"]
            _atomic_save(state, self._state_path)
        except Exception as e:
            logger.warning(f"Saving dynamic draft vocab state failed: {e}")

    def save_state(self) -> None:
        """Write the state file synchronously (tests, shutdown)."""
        if self._state_path is not None:
            self._write_snapshot(self._snapshot(non_blocking=False))


def _save_at_exit(ref: "weakref.ref[DynamicDraftVocab]") -> None:
    dv = ref()
    if dv is None:
        return
    try:
        if dv._save_thread is not None:
            dv._save_thread.join(timeout=10)
        dv.save_state()
    except Exception:
        pass


def load_rank_file(path: str) -> torch.Tensor:
    """Ids in rank order from a torch.save'd list/tensor or a JSON list/{"ids"}."""
    if path.endswith(".json"):
        import json

        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data["ids"]
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)
    return torch.as_tensor(data, dtype=torch.int64).flatten()


def rank_static_map(hot: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Reorder ``hot`` by first position in ``rank``; unranked ids keep order last."""
    big = rank.numel()
    pos = {}
    for i, t in enumerate(rank.tolist()):
        pos.setdefault(t, i)
    keys = torch.tensor([pos.get(t, big + i) for i, t in enumerate(hot.tolist())])
    return hot[torch.argsort(keys, stable=True)]


def export_ranked_map(
    state_path: str,
    token_map: torch.Tensor,
    k: int,
    rank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-``k`` draft map: learned tokens by score, then the (ranked) static map.

    Use it to shrink ``--speculative-token-map`` to what a workload profile has
    actually used. The new map has a new hash, so start it with a new profile.
    """
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    base = token_map if rank is None else rank_static_map(token_map, rank)
    ranked = torch.cat((state["pool_ids"], state["ring_ids"].flip(0), base))
    out, seen = [], set()
    for t in ranked.tolist():
        if t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) == k:
                break
    return torch.tensor(out, dtype=torch.int64)


def _first_unique(tok: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Indices of the first occurrence of each token not already present."""
    seen = set()
    keep = []
    for i, t in enumerate(tok.tolist()):
        if not present[t] and t not in seen:
            seen.add(t)
            keep.append(i)
    return torch.tensor(keep, dtype=torch.int64)


def _atomic_save(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a top-K --speculative-token-map from a token map, an "
        "optional rank file and an optional learned dynamic-vocab state file."
    )
    parser.add_argument("--token-map", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--state", help="dvocab-<profile>.pt to rank learned tokens first"
    )
    parser.add_argument("--rank", help="ranked id list for the static map")
    args = parser.parse_args(argv)
    token_map = load_rank_file(args.token_map)
    rank = load_rank_file(args.rank) if args.rank else None
    if args.state:
        ids = export_ranked_map(args.state, token_map, args.k, rank)
    else:
        base = token_map if rank is None else rank_static_map(token_map, rank)
        ids = base[: args.k]
    torch.save(ids.tolist(), args.out)
    print(f"wrote {ids.numel()} ids to {args.out}")


if __name__ == "__main__":
    main()
