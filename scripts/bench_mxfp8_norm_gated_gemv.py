#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
"""Micro-benchmark for the fused gated RMSNorm + MXFP8 GEMV (needs sm_120).

At this checkpoint's real GDN out_proj shape (K = num_v_heads 48 x head_v_dim
128 = 6144 in, N = hidden_size 2560) three columns are timed per decode width:

    norm + GEMV   the unfused pair: the production RMSNormGated kernel, then
                  the same GEMV launched with norm_fused=False.
    fused         one launch, the SGLANG_NORM_INTO_GEMV=1 path.

so the row prices the fusion alone and not the GEMV. Run it as

    python scripts/bench_mxfp8_norm_gated_gemv.py
    python scripts/bench_mxfp8_norm_gated_gemv.py --iters 300 --check
    python scripts/bench_mxfp8_norm_gated_gemv.py --ms 1,2,4,8,16 --layers 36

Donor campaign (flash-next-bench bench/h1/h1_check.py, CUPTI medians, RTX PRO
6000): -0.95 us mean, -1.06 us median per call over the unfused pair, x 36 GDN
layers = -36 us per decode step and 36 fewer kernels per step.

The M == 1 and M == 2 rows ride the tile the donor measured at M == 4 with the
cheaper broadcast reduction; nothing has measured those on a GPU, so they carry
a "*" and the A/B should treat them as unconfirmed until they win.
"""

from __future__ import annotations

import argparse
import statistics

import torch

N, K, G = 2560, 6144, 128  # out_proj: hidden_size x num_v_heads * head_v_dim
MS = (1, 2, 4, 8, 16)
EPS = 1e-6
_L2_FLUSH_BYTES = 256 * 1024 * 1024


def _median(fn, flush: torch.Tensor, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        flush.zero_()  # cold L2, as between two decode steps
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", type=str, default=",".join(str(m) for m in MS))
    ap.add_argument(
        "--activation", type=str, default="swish", choices=["swish", "sigmoid"]
    )
    ap.add_argument("--layers", type=int, default=36, help="GDN layers per step")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.get_device_capability()[0] == 12, "sm_120 only"

    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as GatedRMSNorm
    from sglang.srt.layers.quantization import mxfp8_norm_gated_gemv as fused

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)
    weight = (
        (torch.randn(N, K, generator=gen) * 0.05)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .cuda()
        .contiguous()
    )
    scale = (
        torch.randint(118, 138, (N, K // 32), dtype=torch.uint8, generator=gen)
        .cuda()
        .contiguous()
    )
    norm_weight = (
        (torch.randn(G, generator=gen) * 0.1 + 1.0).bfloat16().cuda().contiguous()
    )
    norm = (
        GatedRMSNorm(
            G,
            EPS,
            group_size=None,
            norm_before_gate=True,
            device="cuda",
            dtype=torch.bfloat16,
            activation=args.activation,
        )
        .eval()
        .to(torch.bfloat16)
    )
    norm.weight.data.copy_(norm_weight)
    flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    # Materialize the split-K scratch before any timing: an eager launch would
    # else pay its first-touch allocation inside the measured region.
    fused.prealloc(device)

    print(
        f"gdn out_proj N={N} K={K} norm group={G} act={args.activation} iters={args.iters}"
    )
    print(
        f"{'M':>3}  {'norm+gemv us':>13}  {'fused us':>10}  {'saved us':>9}  "
        f"{'per step':>10}  {'rel':>9}"
    )
    for m in [int(v) for v in args.ms.split(",")]:
        x = (torch.randn(m, K, generator=gen) * 0.4).bfloat16().cuda().contiguous()
        z = (torch.randn(m, K, generator=gen) * 1.5).bfloat16().cuda().contiguous()
        rows = m * (K // G)

        def pair():
            a = norm(x.view(rows, G), z.view(rows, G))
            return fused.fused_norm_gated_gemv(
                a.view(m, K),
                x,
                norm_weight,
                weight,
                scale,
                G,
                EPS,
                args.activation,
                norm_fused=False,
            )

        def one():
            return fused.fused_norm_gated_gemv(
                x, z, norm_weight, weight, scale, G, EPS, args.activation
            )

        assert one() is not None and pair() is not None, "no scratch or no plan"
        if args.check:
            ref = fused.fused_norm_gated_gemv_reference(
                x, z, norm_weight, weight, scale, G, EPS, args.activation
            )
            got = one().float()
            rel = (got - ref.float()).norm().item() / (ref.float().norm().item() + 1e-9)
            err = (got - ref.float()).abs().max().item()
            assert err <= 2e-2 * ref.float().abs().max().item(), (m, err)
        else:
            rel = float("nan")
        old, new = _median(pair, flush, args.iters), _median(one, flush, args.iters)
        mark = "*" if m <= 2 or m > 4 else " "
        print(
            f"{m:3d}  {old:13.2f}  {new:9.2f}{mark}  {old - new:8.2f}  "
            f"{(old - new) * args.layers:9.1f}  {rel:9.2e}"
        )
    print(f'"*" = a tile the donor never measured at this M (see the docstring).')


if __name__ == "__main__":
    main()
