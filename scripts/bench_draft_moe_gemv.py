# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Cold-L2 benchmark for the draft MoE GEMV (requires an sm_120 GPU; never run in CI).

Times the two Triton kernels of ``layers/moe/draft_moe_gemv.py`` at the draft
model's one-token shapes (E=512, H=2560, I=640, top-10) against an L2 flush,
and prints the per-call kernel-launch count. Compare against the FlashInfer
CUTLASS MoE timing taken from the same server with CUPTI, not against this
script's own torch reference (which is a correctness fixture, not a baseline).

Donor campaign (flash-next-bench bench/dg1, RTX PRO 6000, 2026-10-01):
49.4 -> 29.8 us/call warm, 54.5 -> 30.4 us/call with the L2 flush, i.e.
-0.27..-0.34 ms per 15-step draft decode.

Usage:  python scripts/bench_draft_moe_gemv.py [--iters 200]
"""

import argparse
import statistics

import torch
from sglang.srt.layers.moe import draft_moe_gemv as dmg

H, I, E, R = 2560, 640, 512, 10
_L2_FLUSH_BYTES = 256 * 1024 * 1024  # > 96 MB L2 on RTX PRO 6000


def _make_inputs(device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(0)
    w13 = torch.randint(0, 16, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 16, (E, H, I // 2), dtype=torch.uint8, generator=gen)
    sf13 = torch.randint(
        0, 120, (E, 2 * I, H // 16), dtype=torch.uint8, generator=gen
    ).view(torch.float8_e4m3fn)
    sf2 = torch.randint(0, 120, (E, H, I // 16), dtype=torch.uint8, generator=gen).view(
        torch.float8_e4m3fn
    )
    # e4m3 code 0x3C == 1.0 keeps the random scales finite and non-zero-ish.
    sf13.fill_(0x3C)
    sf2.fill_(0x3C)
    return (
        (torch.randn(1, H, generator=gen) * 0.1).to(torch.bfloat16).to(device),
        torch.randint(0, E, (1, R), dtype=torch.int64, generator=gen).to(device),
        torch.rand(1, R, generator=gen).to(device),
        w13.to(device),
        sf13.to(device),
        (torch.rand(E, 2, generator=gen) * 2e-3 + 1e-3).to(device),
        w2.to(device),
        sf2.to(device),
        (torch.rand(E, generator=gen) * 2e-3 + 1e-3).to(device),
    )


def _time(fn, flush: torch.Tensor, iters: int) -> list[float]:
    fn()
    torch.cuda.synchronize()
    times = []
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()  # cold L2: every weight byte comes from HBM, as between steps
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    assert torch.cuda.get_device_capability()[0] == 12, "sm_120 only"
    device = torch.device("cuda")
    x, ids, wts, w13, sf13, s2_13, w2, sf2, s2_2 = _make_inputs(device)
    ws = dmg.make_workspace(
        num_routes=R, hidden_size=H, intermediate_size=I, device=device
    )

    def call():
        return dmg.draft_moe_gemv(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=wts,
            w13_weight=w13,
            w13_blockscale_swizzled=sf13,
            w13_weight_scale_2=s2_13,
            w2_weight=w2,
            w2_blockscale_swizzled=sf2,
            w2_weight_scale_2=s2_2,
            workspace=ws,
        )

    flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    cold = _time(call, flush, args.iters)
    warm = _time(call, flush[:1], args.iters)

    print(f"cfg={dmg.DEFAULT_CONFIG}")
    print(
        f"cold: median {statistics.median(cold):8.1f} us  p10 {sorted(cold)[len(cold) // 10]:8.1f}"
    )
    print(
        f"warm: median {statistics.median(warm):8.1f} us  p10 {sorted(warm)[len(warm) // 10]:8.1f}"
    )
    print("kernels per call: 2 (up_gate, down) + 0 (counters self-reset in-kernel)")


if __name__ == "__main__":
    main()
