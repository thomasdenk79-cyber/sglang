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

Every row says what *executed*, not what the gate would allow: the GEMV counts
its own launches (``mxfp8_skinny_gemv.launch_count()``), the capture happens on
a stream whose split-K scratch was materialized before the capture (a launch
whose scratch is missing mid-capture falls back to the W8A8 GEMM, and a table
row that quietly timed the fallback would be worse than no measurement), and a
row that claims ``gemv`` without a single GEMV launch makes the exit code
non-zero. ``gemv(fallback)`` in the route column means exactly that; the
``plan=`` column shows the tile the row was supposed to run.

Weight and scale bytes are this checkpoint's real MXFP8 layout, filled with
deterministic random data (``--seed``), and both paths go through the same
``Fp8LinearMethod`` instance so the old path is exactly the production one,
derived scales included.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Optional, Tuple

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
DEVICE = torch.device("cuda", 0)


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


def _capture_stream() -> torch.cuda.Stream:
    """The stream the decode runner captures on, or a fresh one.

    Capturing where ``prealloc`` already put the split-K scratch is the point:
    on any other stream the first split-K launch of a capture misses, falls back
    to the W8A8 GEMM and the graph silently replays the old path.
    """
    try:
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
        )

        stream = get_or_create_global_graph_capture_stream()
    except Exception as e:  # a build without the shared capture stream
        print(
            f"  (no shared capture stream: {e}; using a private one)", file=sys.stderr
        )
        stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        from sglang.srt.layers.quantization import mxfp8_skinny_gemv as gemv

        gemv.prealloc(DEVICE, stream)
    return stream


def _time(
    fn, reps: int, warmup: int, graph: bool, capture_stream: Optional[torch.cuda.Stream]
) -> Tuple[float, int]:
    """(median microseconds per call, GEMV launches the timed section recorded).

    In graph mode the launch count is taken across the capture, so it says what
    ended up in the graph rather than what the gate would have allowed.
    """
    import statistics

    from sglang.srt.layers.quantization import mxfp8_skinny_gemv as gemv

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        stream = capture_stream or torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                gemv.prealloc(DEVICE, stream)
                before = gemv.launch_count()
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture, stream=stream):
                    for _ in range(reps):
                        fn()
                captured = gemv.launch_count() - before
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
            return statistics.median(times), captured
        except Exception as e:  # a backend that cannot be captured here
            print(f"  (graph capture failed: {e}; timing eager)", file=sys.stderr)
    before = gemv.launch_count()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times), gemv.launch_count() - before


def _plan_note(m: int, n: int, k: int, sms: int) -> str:
    """The tile this row runs, marked if the tuned table entry was overridden."""
    from sglang.srt.layers.quantization import mxfp8_skinny_gemv as gemv

    plan = gemv._plan(m, n, k, sms)
    bucket = gemv._m_bucket(m)
    table = gemv._TUNED.get((bucket, n, k))
    demoted = table is not None and table[2] != plan[2]
    return f"  plan={plan}{' (table splits demoted)' if demoted else ''}"


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
    # The gate has to be open while the weights are processed: that is where
    # prealloc() materializes the split-K scratch of the forward, capture and alt
    # streams. The rows below then only have to bring their own capture stream.
    with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
        gemv.prealloc(DEVICE)
    capture_stream = _capture_stream() if args.graph else None
    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} "
        f"backend={method.mxfp8_dense_backend} sms={sms} "
        f"timing={'graph replay' if args.graph else 'events'}"
    )
    header = (
        f"{'shape':<22} {'N':>7} {'K':>6} {'M':>3} {'old us':>9} {'new us':>9} "
        f"{'speedup':>8} {'new GB/s':>9} {'max rel':>9} {'route':>14}"
    )
    print(header)
    print("-" * len(header))

    bad = []
    for name, n, k in rows:
        layer = _layer(n, k, args.seed)
        with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
            # Runs prealloc() for this layer's device (mxfp8_skinny_gemv
            # ._mxfp8_skinny_gemv_enabled is only true with the flag on).
            method.process_weights_after_loading(layer)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            # The gate as apply() sees it, at this exact width: above
            # MAX_TUNED_M it refuses, and that row is w8a8, not a failure.
            gemv_able = gemv.mxfp8_skinny_gemv_applies(
                x, layer.weight, layer.weight_scale_inv
            )
            with envs.SGLANG_MXFP8_SKINNY_GEMV.override(False):
                old, old_launches = _time(
                    lambda layer=layer, x=x: method.apply(layer, x),
                    args.reps,
                    args.warmup,
                    args.graph,
                    capture_stream,
                )
            with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
                new, new_launches = _time(
                    lambda layer=layer, x=x: method.apply(layer, x),
                    args.reps,
                    args.warmup,
                    args.graph,
                    capture_stream,
                )
            # What ran, from the kernel's own counter -- not from the predicate.
            if new_launches > 0 and old_launches > 0:
                bad.append((name, m, "the W8A8 column launched the GEMV"))
            route = "w8a8"
            if gemv_able:
                route = f"gemv({new_launches})" if new_launches else "gemv(fallback)"
            elif new_launches:
                bad.append((name, m, "the gate refused it but the GEMV ran"))
            if gemv_able and not new_launches:
                bad.append(
                    (
                        name,
                        m,
                        "no GEMV launch: this row timed the W8A8 fallback "
                        "(prealloc did not cover the capture stream)",
                    )
                )
            gbps = n * k / (new * 1e-6) / 1e9 if new > 0 and new_launches else 0.0
            rel = float("nan")
            if args.check and gemv_able:
                ref = gemv.mxfp8_skinny_gemv_reference(
                    x, layer.weight, layer.weight_scale_inv
                )
                with envs.SGLANG_MXFP8_SKINNY_GEMV.override(True):
                    got = method.apply(layer, x).float()
                rel = ((got - ref).abs() / ref.abs().clamp(min=1e-3)).max().item()
                if not gemv.counters_are_zero(DEVICE):
                    bad.append((name, m, "split-K counters were left non-zero"))
            plan = _plan_note(m, n, k, sms) if args.plan and gemv_able else ""
            speed = f"{old / new:>7.2f}x" if new_launches else f"{'n/a':>8}"
            print(
                f"{name:<22} {n:>7} {k:>6} {m:>3} {old:>9.2f} {new:>9.2f} "
                f"{speed} {gbps:>9.0f} {rel:>9.4f} {route:>14}{plan}"
            )
        del layer
        torch.cuda.empty_cache()
    print(
        "\nroute=gemv(n): the GEMV kernel launched n times in the timed section (the "
        "capture, in --graph mode);\nroute=gemv(fallback): the gate allowed it but no "
        "launch reached the kernel, so both columns are the\n  W8A8 GEMM and the row is "
        "not a measurement of the GEMV;\nroute=w8a8: the gate refused it (too small, or "
        "dequantized to bf16 at load).\n"
        "plan= is (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages); a table "
        "entry _fit demoted to SPLITS=1 is marked, because such an entry never runs.\n"
        "Weight bytes read = N*K (fp8) + N*K/32 (UE8M0); GB/s is over N*K."
    )
    for name, m, why in bad:
        print(f"ERROR {name} M={m}: {why}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
