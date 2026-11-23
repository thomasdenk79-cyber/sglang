# rc5 rebase conflict log (rc/qwen38-rc2 a090492091 -> upstream/main 61915e54ee)
| # | Commit | File | Resolution | Follow-up |
|---|---|---|---|---|
| 1 | 2f3dce16e0 CSF NVFP4 loader | model_loader/loader.py | keep both: upstream `safetensors_index_file` + ours `weight_name_filter` | - |
| 3 | 7d5965fd1d PLE exact pin | qwen4_exp_ple_table.py | skipped: upstreamed as f29cc397a2 (#48249), diff = comment only | - |
| 7 | 50f3315bd2 dynamic draft vocab | eagle_worker_v2.py | stage ring BEFORE upstream prefill-split early return; keep both methods | TEST: dynamic vocab under `_draft_extend_for_prefill_split` |
| 16 | f340e338ac sm120 skinny GEMV | layers/quantization/unquant.py | keep upstream hip/tlx imports + our sm120 block; upstream renamed guard `_is_gfx95` -> `_rowmajor_ok` | - |
| 20 | d9dd1b9f0f top-p split-row | test_speculative_sampling.py | keep both tests | - |
| 25 | f0729be83e post-verify pool presize | test_eagle_post_verify_graphs.py | upstream rename watermark_state->watermark; keep our decode_cuda_graph_capture_bs | - |
| 32 | 94ca1f582b write-back ordered demote | unified_radix_cache.py, environ.py, test | skipped: superseded by upstream de41ccfae0 (#50469), ours was opt-in, never live | later fork commits must not reference `_unlanded_write_acks`/`SGLANG_OPT_HICACHE_WRITE_BACK_ORDERED_DEMOTE` |
| 33 | 8974f29715 subagent host evict | environ.py, io_struct.py, schedule_batch.py, request_log.json | keep both (filler_seed + parent_session_id); dropped ORDERED_DEMOTE context line; json = upstream | REGEN request_log.json after rebase |
| 37 | 6f85196aa1 mamba host scale | hybrid_pool_assembler.py | upstream gfx950 `mamba_host_ratio()` x our `_mamba_host_ratio_scale()` | - |
| 52 | f9511985fc mamba prefill admission | schedule_policy.py | merged imports (upstream GammaAdaController + our ensure_mamba_capacity) | - |
| 53 | 7f4ceec17d JIT caches under SGLANG_CACHE_DIR | environ.py | our default + upstream SGLANG_JIT_CUDA_ARCH_POLICY | first rc5 start: expect JIT rebuild (cache key change) |
| 56 | 698a65db95 adaptive draft vocab | eagle_worker_v2.py | stage(next_token_ids, output=True) before upstream split return (same as #7) | same TEST as #7 |

Base: upstream/main 61915e54ee (2026-11-23). Runtime image: lmsysorg/sglang:nightly-dev-cu13-20261122-e96e9675 (same pyproject as main). Stack: 64 own commits from rc/qwen38-rc2 a090492091; 92 commits of an older upstream merge (<=7aaebb1cf0) dropped automatically.
