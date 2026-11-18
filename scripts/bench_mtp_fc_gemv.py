#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
"""Micro-benchmark for the MTP entry-fc GEMV (needs a GPU; run on sm_120).

Times cuBLAS (what ``nn.Linear`` dispatches to) against the sm_120 split-K
Triton skinny GEMV that ``SGLANG_MTP_FC_GEMV=1`` routes the MTP entry
projections to, at this checkpoint's real shapes: ``fc_embedding`` and
``fc_hidden`` are both [hidden_size 2560, hidden_size 2560] BF16 with no bias,
and a verify step calls each once with 1..16 rows.

    python scripts/bench_mtp_fc_gemv.py
    python scripts/bench_mtp_fc_gemv.py --iters 400 --check --graph

``--check`` compares the GEMV against an FP32 dequantized-weight reference per
shape, ``--graph`` adds a CUDA-graph replay column (the regime the draft runs
in: the whole verify step is one graph).

Donor record (docs/lab-notes/DT1_DRAFT_TAIL_2026-10-01.md section 5, RTX PRO
6000): the cuBLAS GEMM plus splitKreduce cost 12.5 us for one entry
projection, and the skinny GEMV saved 1.52 us per forward at 1 row and 1.93
at 4 rows behind an L2 flush (1.63..2.20 warm), estimated -6.5..-14 us per
draft step. The draft calls these two projections once per verify step, so a
saving here shows up once per step, not per layer.
"""

from __future__ import annotations

import argparse
import statistics

import torch

HIDDEN = 2560  # config.json: hidden_size; both fc weights are [2560, 2560]
MS = (1, 2, 4, 8, 16)
_L2_FLUSH_BYTES = 256 * 1024 * 1024


def _median(fn, flush: torch.Tensor, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        flush.zero_()
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(times)


def _median_graph(fn, iters: int) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record()
        graph.replay()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", type=str, default=",".join(str(m) for m in MS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.get_device_capability()[0] == 12, "sm_120 only"

    from sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv import (
        bf16_skinny_gemv_applies,
        bf16_skinny_linear,
    )

    torch.manual_seed(0)
    weight = (torch.randn(HIDDEN, HIDDEN, device="cuda") * 0.05).bfloat16().contiguous()
    flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    print(f"mtp fc: [N, K] = [{HIDDEN}, {HIDDEN}] BF16 bias-free, iters={args.iters}")
    print(f"{'rows':>5}  {'cuBLAS us':>10}  {'gemv us':>9}  {'speedup':>8}")
    for m in [int(v) for v in args.ms.split(",")]:
        # The draft passes hc_count x rows to fc_hidden; the row count that
        # reaches the GEMM is what the gate tests, so time it flat.
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        assert bf16_skinny_gemv_applies(x, weight), (m,)
        if args.check:
            ref = x.float() @ weight.float().t()
            got = bf16_skinny_linear(x, weight).float()
            err = (got - ref).abs().max().item()
            assert err <= 2e-2 * ref.abs().max().item(), (m, err)
        old = _median(lambda: torch.nn.functional.linear(x, weight), flush, args.iters)
        new = _median(lambda: bf16_skinny_linear(x, weight), flush, args.iters)
        extra = ""
        if args.graph:
            g_old = _median_graph(
                lambda: torch.nn.functional.linear(x, weight), args.iters
            )
            g_new = _median_graph(lambda: bf16_skinny_linear(x, weight), args.iters)
            extra = f"  graph {g_old:6.1f} -> {g_new:6.1f} us"
        print(f"{m:5d}  {old:10.2f}  {new:9.2f}  {old / new:7.2f}x{extra}")
    print("both fc projections share this shape; one call each per verify step")


if __name__ == "__main__":
    main()
