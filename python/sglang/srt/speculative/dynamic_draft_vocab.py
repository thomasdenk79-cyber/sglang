"""Dynamic extension of a reduced speculative draft vocabulary.

A static hot-token map (``--speculative-token-map``) cannot know topic words such
as identifiers that mostly appear in the prompt. This keeps a fixed-size FIFO ring
of non-static tokens (prompt and accepted tokens) scored with the target LM head
rows, appended after the static rows of the sliced draft head. The ring is shared
by all running requests.

Columns always hold unique target ids: empty ring slots are prefilled with
distinct non-static tokens, so ``hot_token_id`` stays a valid draft-to-target map
for rejection sampling. ``hot_token_id`` and ``head`` are views of buffers that
are only updated in place, so captured CUDA graphs see the current ring.

Tokens are staged with ``stage`` and inserted by ``commit``. The worker commits at
one point per decode round, after the round's draft probabilities were scattered
to the target vocab and before the draft extend produces the next ones, so every
draft distribution is mapped with the ring it was computed with.

Adapted from the V3 dynamic draft vocabulary of
https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000 (MIT).
"""

from typing import Optional

import torch

_STATIC = -3
_ABSENT = -1


class DynamicDraftVocab:
    """Static hot rows followed by ``num_slots`` FIFO ring rows."""

    def __init__(
        self,
        hot_token_id: torch.Tensor,
        target_head: torch.Tensor,
        num_slots: int,
        vocab_size: int,
        max_observe: int = 0,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
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
        hot = hot_token_id.to(device=device, dtype=torch.int64).flatten()
        num_hot = hot.numel()
        if num_hot == 0 or int(hot.min()) < 0 or int(hot.max()) >= vocab_size:
            raise ValueError("hot_token_id must hold ids within the target vocab.")
        if torch.unique(hot).numel() != num_hot:
            raise ValueError("hot_token_id must hold unique ids.")
        if num_slots > vocab_size - num_hot:
            raise ValueError(
                f"num_slots={num_slots} exceeds the {vocab_size - num_hot} "
                "tokens outside the static map."
            )

        self.vocab_size = vocab_size
        self.num_hot = num_hot
        self.num_slots = num_slots
        self.max_observe = max_observe if max_observe > 0 else 16 * num_slots

        # Index vocab_size is a sink for masked lanes and always reads static.
        slot_of = torch.full((vocab_size + 1,), _ABSENT, dtype=torch.int64)
        slot_of[hot.cpu()] = _STATIC
        slot_of[vocab_size] = _STATIC
        fill = torch.nonzero(slot_of[:vocab_size] == _ABSENT).flatten()[:num_slots]
        slot_of[fill] = torch.arange(num_slots)
        self.slot_of = slot_of.to(device)

        # Row num_hot + num_slots is a sink for masked lanes, outside the views.
        ids = torch.empty(num_hot + num_slots + 1, dtype=torch.int64, device=device)
        ids[:num_hot] = hot
        ids[num_hot : num_hot + num_slots] = fill.to(device)
        ids[-1] = ids[0]
        self._ids = ids
        self._rows = target_head[ids]
        self._target_head = target_head
        self._cursor = torch.zeros(1, dtype=torch.int64, device=device)
        # Last arrival stamp per token; stamps grow across calls, so no reset.
        self._last_seen = torch.full(
            (vocab_size + 1,), -1, dtype=torch.int64, device=device
        )
        self._stamp = 0
        self._pending: Optional[torch.Tensor] = None

    @property
    def hot_token_id(self) -> torch.Tensor:
        return self._ids[: self.num_hot + self.num_slots]

    @property
    def head(self) -> torch.Tensor:
        return self._rows[: self.num_hot + self.num_slots]

    @torch.no_grad()
    def stage(
        self, token_ids: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> None:
        """Queue tokens for the next ``commit``; ``valid`` masks lanes out.

        Only the most recent ``max_observe`` staged tokens are kept, which bounds
        the commit cost; the ring keeps at most ``num_slots`` of them anyway.
        """
        ids = token_ids.reshape(-1).to(device=self._ids.device, dtype=torch.int64)
        if valid is not None:
            ids = torch.where(valid.reshape(-1).to(ids.device), ids, self.vocab_size)
        if self._pending is not None:
            ids = torch.cat((self._pending, ids))
        elif ids.data_ptr() == token_ids.data_ptr():
            # The caller's buffer may be reused before the commit.
            ids = ids.clone()
        self._pending = ids[-self.max_observe :]

    @torch.no_grad()
    def commit(self) -> None:
        """Insert the staged tokens into the ring."""
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self.observe(pending)

    @torch.no_grad()
    def observe(self, input_ids: torch.Tensor) -> None:
        """Insert unseen tokens into the ring, evicting the oldest entries."""
        ids = input_ids.reshape(-1)[-self.max_observe :]
        n = ids.numel()
        if n == 0:
            return
        sink = self.vocab_size
        ids = ids.to(torch.int64)
        tok = torch.where((ids >= 0) & (ids < sink), ids, sink)
        new = self.slot_of[tok] == _ABSENT

        # The last occurrence of each distinct new token is its lane.
        stamp = torch.arange(self._stamp, self._stamp + n, device=ids.device)
        self._stamp += n
        self._last_seen.scatter_reduce_(
            0, torch.where(new, tok, sink), stamp, reduce="amax"
        )
        win = new & (self._last_seen[tok] == stamp)

        # Ring slots in arrival order; keep the most recent num_slots tokens.
        order = torch.cumsum(win, 0) - 1
        drop = (win.sum() - self.num_slots).clamp(min=0)
        win = win & (order >= drop)
        pos = torch.where(
            win, (self._cursor + order - drop) % self.num_slots, self.num_slots
        )
        row = self.num_hot + pos

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
        if n <= self.num_slots:
            self._rows.index_copy_(0, row, self._target_head[src])
        else:
            # Bound long prefill chunks to one ring-sized gather.
            ring = slice(self.num_hot, self.num_hot + self.num_slots)
            self._rows[ring] = self._target_head[self._ids[ring]]
        self._cursor.copy_((self._cursor + win.sum()) % self.num_slots)
