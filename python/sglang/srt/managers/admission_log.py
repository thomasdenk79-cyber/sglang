"""Per-request admission accounting for HiCache.

One line per admitted prefill request records how many prompt tokens came
from the device radix cache, from the SSD tier (HiCache) or from fresh
recompute. Emission is gated by ``SGLANG_HICACHE_ADMISSION_LOG``; the helpers
here are pure so they can be tested without a scheduler.
"""

from __future__ import annotations

from typing import Any, Dict


def admission_fields(
    prompt_len: int,
    prefix_len: int,
    device_hit: int,
    host_hit: int,
    storage_hit: int,
    attempts: int,
    wait_ms: int,
) -> Dict[str, int]:
    """Build the numeric fields of one admission record.

    ``recompute`` is what admission still has to prefill: the prompt minus
    the cached prefix, never negative.
    """
    return {
        "prompt": prompt_len,
        "device": device_hit,
        "host": host_hit,
        "storage": storage_hit,
        "recompute": max(prompt_len - prefix_len, 0),
        "attempts": attempts,
        "wait_ms": wait_ms,
    }


def format_admission(rid: Any, fields: Dict[str, int]) -> str:
    """Render one admission record as a single greppable log line."""
    return (
        f"HiCache admission req={rid}"
        f" prompt={fields['prompt']}"
        f" device={fields['device']}"
        f" host={fields['host']}"
        f" storage={fields['storage']}"
        f" recompute={fields['recompute']}"
        f" attempts={fields['attempts']}"
        f" wait_ms={fields['wait_ms']}"
    )
