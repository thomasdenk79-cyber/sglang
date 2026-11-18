#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
"""Micro-benchmark for the MoE router GEMV (needs a GPU; run on sm_120).

Times the cuBLAS GEMM the router gate runs today (``ReplicatedLinear`` ->
``F.linear``) against the Triton GEMV the ``SGLANG_ROUTER_GEMV=1`` hook routes
to, at this checkpoint's real gate shape [512, 2560] and every decode width:

    python scripts/bench_router_gemv.py
    python scripts/bench_router_gemv.py --iters 400 --check --graph

``--check`` compares the GEMV against an FP32 reference per shape first, so a
row of timings is never printed for a kernel that is simply wrong. ``--graph``
adds a CUDA-graph replay column, which is the regime the decode path runs in.
Both paths see the same tensors; only the kernel differs.

Donor record (docs/optimizations.md B3, RTX PRO 6000): the cuBLAS GEMM plus
splitKreduce pair cost 119 us per step over 61 calls; the retuned skinny GEMV
ran the gate at 6.18 -> 4.42 us per call at M == 16, i.e. about -86 us per
step and 63 fewer kernels. The dedicated router kernel was *slower* than
cuBLAS there (7.5 vs 5.0 us), which is what the biased column of this table
prices.

The gate is one call per MoE layer per step, so the last column prices the
saving for the 48 MoE layers of one decode step.
"""

from __future__ import annotations

import argparse
import statistics

import torch

N, K = 512, 2560  # num_experts x hidden_size, config.json of the checkpoint
MS = (1, 2, 4, 8, 16)
_L2_FLUSH_BYTES = 256 * 1024 * 1024


def _time(fn, flush: torch.Tensor, iters: int) -> float:
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


def _time_graph(fn, iters: int) -> float:
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

    from sglang.srt.layers.moe import router_gemv as rg

    gen = torch.Generator(device="cpu").manual_seed(0)
    weight = (torch.randn(N, K, generator=gen) * 0.05).bfloat16().cuda().contiguous()
    bias = (torch.randn(N, generator=gen) * 0.1).bfloat16().cuda().contiguous()
    flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    print(f"router gate: N={N} K={K} bias=both iters={args.iters}")
    print(
        f"{'M':>3}  {'bias':>5}  {'cuBLAS us':>10}  {'gemv us':>9}  "
        f"{'speedup':>8}  {'per step (48 layers)':>20}"
    )
    for m in [int(v) for v in args.ms.split(",")]:
        x = (torch.randn(m, K, generator=gen) * 0.5).bfloat16().cuda().contiguous()
        assert rg.router_gemv_applies(x, weight, bias, require_cuda=True), (m,)
        for use_bias in (False, True):
            b = bias if use_bias else None
            if args.check:
                ref = rg.router_gemv_reference(x, weight, b)
                got = rg.router_gemv(x, weight, b)
                assert got.dtype == torch.float32
                err = (got - ref).abs().max().item()
                assert err <= 2e-2 * ref.abs().max().item(), (m, use_bias, err)
            old = _time(
                lambda: torch.nn.functional.linear(x, weight, b), flush, args.iters
            )
            new = _time(lambda: rg.router_gemv(x, weight, b), flush, args.iters)
            extra = ""
            if args.graph:
                g_old = _time_graph(
                    lambda: torch.nn.functional.linear(x, weight, b), args.iters
                )
                g_new = _time_graph(lambda: rg.router_gemv(x, weight, b), args.iters)
                extra = f"  graph {g_old:7.1f} -> {g_new:7.1f} us"
            print(
                f"{m:3d}  {str(use_bias):>5}  {old:10.1f}  {new:9.1f}  "
                f"{old / new:7.2f}x  {(old - new) * 48:17.1f} us{extra}"
            )


if __name__ == "__main__":
    main()
