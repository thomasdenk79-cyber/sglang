# p16 correctness ports (2026-11-25)

Target: isolated `sglang-p16-nvfp4` donor snapshot. Upstream source of truth:
`/srv/community-repos/sglang-upstream` (git show only). One commit per fix,
switch-gated off by default where behaviour changes. p16 does CPU checks only;
t15 (sm_120) owns GPU validation. Not taken: reverted bundle `775db052b6`,
Beam-only `a8ac86595c`, unmerged draft PRs #51051-#51068.

| Fix | Upstream | p16 status | Switch | GPU validation needed on t15 |
|---|---|---|---|---|
| NVFP4 hidden padding | `0f1509829b` | ported (adapted) | `SGLANG_P16_NVFP4_TRTPAD_HIDDEN=1` | GSM8K on hidden 2688 (Nemotron-Nano 30B-A3B NVFP4): padded ~0.925 vs unpadded ~0.405; also check per-token gated/non-gated, CUDA-graph replay and symmetric MoE output buffer. |
| GDN prefill host stalls | `03d4f49f22` | ported: host-sync clamp hunk only | none (host-sync only) | Prefill with prefix-cache tracking; assert no host sync and no checkpoint-plan drift. Full prewarm not portable (donor lacks `gdn_prefill_graph.py`, `use_cp`, host track metadata). |
| GDN in-place state pool | `ff68baae6d` | ported: int64 track indices + unspecialized num_cache_lines | none for portable hunks | Strided/envelope state pools, GDN extend + causal conv; full in-place path not portable (donor lacks `state_pool_io.py`/`PoolStaging`). |
| GDN MIS reset sync | `077acd49a1` | skipped: MIS absent in donor | n/a | Not applicable: donor `gdn_backend.py` has no MIS path (`_forward_extend_mis`, `build_gdn_mis_metadata` absent). |
| GDN MIS fused gates | `304a6016b3` | skipped MIS part; int64 QKV-split offsets ported | n/a | MIS fused gating absent in donor; only int64 QKV-split offset hardening is portable; test >2^31 column strides on GPU. |

## Skip rationale (077acd49a1)

The fix changes `_forward_extend_mis` in `gdn_backend.py`. The p16 donor has no MIS
path at all: `_forward_extend_mis`, `build_gdn_mis_metadata`, `mis_metadata` and
`test_gdn_mis_output.py` are absent (`rg` over `python/sglang/` and `test/`). The
upstream test file is therefore also not applicable. Revisit only if MIS is added.

## Skip rationale (304a6016b3)

`fused_qkv_gating_gdn_prefill`, the `_gdn_gating_values` helper and the MIS
call sites do not exist in the p16 donor (its `fused_gdn_gating` lives under
`kernels/ops/attention/fla/`, and there is no MIS backend). Only the
int64 offset hardening of `fused_qkv_split_gdn_prefill_kernel` was portable.
