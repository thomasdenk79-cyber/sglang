#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
"""Micro-benchmark for the MXFP8 skinny GEMV (needs a GPU; run on sm_120).

Times the W8A8 dense MXFP8 GEMM ``Fp8LinearMethod.apply`` runs today against
the Triton weight-only (W8A16) GEMV the ``SGLANG_MXFP8_SKINNY_GEMV=1`` hook
routes to, per (N, K) and per decode width M, and prints one row per pair:

    python scripts/bench_mxfp8_skinny_gemv.py
    python scripts/bench_mxfp8_skinny_gemv.py --ms 1,4,16 --reps 300 --graph
    python scripts/bench_mxfp8_skinny_gemv.py --shapes shared_down,gdn_out_proj --check

``--graph`` wraps the calls in a CUDA graph and times replays, which is the
regime the decode path actually runs in (no launch overhead, no allocator
traffic); the plain event timing includes both. ``--check`` additionally
compares the GEMV against a dequantized-weight fp32 reference for each shape.

Weight and scale bytes are this checkpoint's real MXFP8 layout, filled with
deterministic random data (``--seed``), and both paths go through the same
``Fp8LinearMethod`` instance so the old path is exactly the production one,
derived scales included.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import torch

# (name, N, K) as the model builds them; see config.json of
# /srv/llm-models/Qwen3.8-Flash-Next-FP4-Local (hidden_size 2560).
SHAPES = [
    ("gdn_in_proj_qkvz", 16384, 2560),
    ("gdn_in_proj_ba", 96, 2560),  # dequantized to bf16 at load: gate must refuse
    ("gdn_out_proj", 2560, 6144),
    ("attn_qkv_proj", 13312, 2560),
    ("attn_o_proj", 2560, 6144),
    ("attn_k_proj", 512, 2560),
    ("attn_v_proj", 512, 2560),
    ("indexer_index_qk_proj", 640, 2560),
    ("shared_gate_up_proj", 1280, 2560),
    ("shared_down_proj", 2560, 640),
]

GROUP_K = 32


def _fill(n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda", dtype=torch.float32) * 0.05).to(
        torch.float8_e4m3fn
    )
    # 2^-9 .. 2^10, the exponent range a MXFP8 quantizer lands in.
    s = torch.randint(
        118, 138, (n, k // GROUP_K), generator=g, dtype=torch.uint8, device="cuda"
    )
    return w.contiguous(), s.contiguous()


def _layer(n: int, k: int, seed: int) -> SimpleNamespace:
    """A layer shaped like the ColumnParallel/RowParallel ones the model makes."""
    from torch.nn.parameter import Parameter

    w, s = _fill(n, k, seed)
    return SimpleNamespace(
        weight=Parameter(w, requires_grad=False),
        weight_scale_inv=Parameter(s, requires_grad=False),
        input_scale=None,
        bias=None,
        logical_widths=[n],
        output_size_per_partition=n,
        input_size_per_partition=k,
        orig_dtype=torch.bfloat16,
        dequantized_bf16=False,
        deepgemm_mxfp8_input=False,
        keep_plain_weight_layout=False,
    )


def _method():
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
        use_mxfp8=True,
    )
    return Fp8LinearMethod(config)


def _time(fn, reps: int, warmup: int, graph: bool) -> float:
    """Median microseconds per call; with ``graph``, inside a CUDA graph replay."""
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        try:
            capture = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(capture, stream=stream):
                    for _ in range(reps):
                        fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                start = torch.cuda.Event(True)
                end = torch.cuda.Event(True)
                start.record()
                capture.replay()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1e3 / reps)
            return statistics.median(times)
        except Exception as e:  # a backend that cannot be captured here
            print(f"  (graph capture failed: {e}; timing eager)", file=sys.stderr)
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", default="1,2,4,8,16", help="comma-separated decode widths")
    ap.add_argument("--shapes", default="", help="comma-separated name filter")
    ap.add_argument("--reps", type=int, default=100, help="calls per timed batch")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--graph", action="store_true", help="time inside a CUDA graph replay"
    )
    ap.add_argument("--check", action="store_true", help="also check against fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="print the tile plan per row")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("this benchmark needs a CUDA GPU", file=sys.stderr)
        return 2
    cap = torch.cuda.get_device_capability()
    if cap[0] != 12:
        print(
            f"WARNING: tuned for sm_120, this is sm_{cap[0]}{cap[1]}", file=sys.stderr
        )

    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import mxfp8_skinny_gemv as gemv

    ms = [int(v) for v in args.ms.split(",") if v.strip()]
    wanted = {s.strip() for s in args.shapes.split(",") if s.strip()}
    rows = [s for s in SHAPES if not wanted or s[0] in wanted]

    method = _method()
    envs.SGLANG_MXFP8_SKINNY_GEMV_MAX_M.set(max(ms))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} "
        f"backend={method.mxfp8_dense_backend} sms={sms} "
        f"timing={'graph replay' if args.graph else 'events'}"
    )
    header = (
        f"{'shape':<22} {'N':>7} {'K':>6} {'M':>3} {'old us':>9} {'new us':>9} "
        f"{'speedup':>8} {'new GB/s':>9} {'max rel':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, n, k in rows:
        layer = _layer(n, k, args.seed)
        method.process_weights_after_loading(layer)
        gemv_able = gemv.mxfp8_skinny_gemv_applies(
            torch.zeros(1, k, dtype=torch.bfloat16, device="cuda"),
            layer.weight,
            layer.weight_scale_inv,
        )
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            with envs.SGLANG_MXFP8_SKINNY_GEMV.override(False):
                old = _time(
                    lambda layer=layer, x=x: method.apply(layer, x),
                    args.reps,
                    args.warmup,
                    args.graph,
                )
            with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
                new = _time(
                    lambda layer=layer, x=x: method.apply(layer, x),
                    args.reps,
                    args.warmup,
                    args.graph,
                )
            gbps = n * k / (new * 1e-6) / 1e9 if new > 0 else 0.0
            rel = float("nan")
            if args.check and gemv_able:
                ref = gemv.mxfp8_skinny_gemv_reference(
                    x, layer.weight, layer.weight_scale_inv
                )
                with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
                    got = method.apply(layer, x).float()
                rel = ((got - ref).abs() / ref.abs().clamp(min=1e-3)).max().item()
            plan = (
                f"  plan={gemv._plan(m, n, k, sms)}" if args.plan and gemv_able else ""
            )
            print(
                f"{name:<22} {n:>7} {k:>6} {m:>3} {old:>9.2f} {new:>9.2f} "
                f"{old / new:>7.2f}x {gbps:>9.0f} {rel:>9.4f} "
                f"[{'gemv' if gemv_able else 'w8a8'}]{plan}"
            )
        del layer
        torch.cuda.empty_cache()
    print(
        "\nroute=gemv: SGLANG_MXFP8_SKINNY_GEMV=1 routes this shape to the Triton GEMV;\n"
        "route=w8a8: the gate refused it (too small, or dequantized to bf16 at load) and\n"
        "both columns are the W8A8 GEMM.\n"
        "Weight bytes read = N*K (fp8) + N*K/32 (UE8M0); GB/s is over N*K."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
