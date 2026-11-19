# PATCH-LEDGER — qwen3.8 Flash-Next RC2 upstream stack

Status: 2026-11-19 11:04 UTC · Created by operator task 140 (Akte 140) · Maintainer: operator; update
this ledger whenever a commit lands on `rc/qwen38-rc*` or a feature branch changes.

## 1. Scope and sources of truth

| Ref | Path | Role |
|---|---|---|
| Stack | `/srv/community-repos/sglang-wt-rc2` (worktree of `/srv/community-repos/sglang-upstream`), branch `rc/qwen38-rc2`, head `1ad48f14e1`, base = upstream `25c1c439` (nightly 20261117, per [O]report-flashnext-scheduler-audit.md) | Production `qwen38-rc2` serving at :30000 |
| [R] QFN register | `/srv/community-repos/qwen38-flashnext-nvfp4-kv/docs/PATCHES.md` + `patches/register.yaml` (repo head `e9ffc2d34a`, generated 2026-11-19 04:42) | Primary per-commit evidence: origin, PR id, type, tests, GPU, upstream plan. Generated data — where it contradicts [INV], [INV] wins (see findings). |
| [O] | `/srv/luni-repos/luni-ai-operator/reports/` | Operator run reports (evidence paths cited per row) |
| [INV] | [O]report-2026-11-19-sglang-upstream-patch-inventory.md | Verified upstream PR states (2026-11-19), incl. #50156 revert and split series |
| [HB] | [O]report-flashnext-hyperburst-evidenz.md | Hyperburst comparison, per-number provenance |

Feature branches on top of `rc/qwen38-rc2`: `feat/qsa-nvfp4-fp8-gather` (7 commits, pipeline run 02),
`perf/mxfp8-skinny-gemv` (8 commits, run 03), `feat/draft-vocab-head-buckets` (23 commits, run 04).
The Pennyroyal 3.1.6 overlay in [R]'s repo is no longer the runtime path; [R] stays as historical
evidence source only.

## 2. Stack overview (measured 2026-11-19 11:04 UTC)

```
git -C /srv/community-repos/sglang-wt-rc2 log --oneline --abbrev=10 25c1c439..rc/qwen38-rc2 | wc -l   # 155
```

| Origin (register) | Commits | Meaning |
|---|---|---|
| BACKPORT (merged upstream) | 104 | Nightly-sync backports, patch-id equal to upstream; disappear on rebase |
| BACKPORT-UNMERGED | 26 | Foreign commits without merged upstream PR — rebase risk, re-land watch |
| OWN | 24 | Developed in this fork (`fork-only (deferred)` in register) |
| MERGE | 1 | `6fdf52f7ad` merge of feat/hicache-buffer-stream (absent from [R], finding F3) |

Remote state of `origin` = `git@github.com:thomasdenk79-cyber/sglang.git` (private), measured 2026-11-19:

| Branch | Local tip | origin tip | Delta |
|---|---|---|---|
| `rc/qwen38-rc2` | `1ad48f14e1` | `1ad48f14e1` | identical (already mirrored) |
| `feat/qsa-nvfp4-fp8-gather` | `6bdcfa37ab` | `6bdcfa37ab` | identical |
| `perf/mxfp8-skinny-gemv` | `484b87168e` | `484b87168e` | identical |
| `feat/draft-vocab-head-buckets` | `182f38c4bb` | **(no remote ref)** | single copy on t15 — backup priority |

`upstream` = `https://github.com/sgl-project/sglang.git`; `upstream/main` tip at measurement:
`145647c55` (2026-11-19). Base `25c1c439` is an ancestor of `upstream/main` (verified
`git merge-base --is-ancestor` → rc 0), so a forward rebase is possible without grafts.

## 3. Commit groups — `25c1c439..rc/qwen38-rc2`

Column `n` = commit count (sum over the `G*` rows = 155 stack commits; the `DOK` row tracks
documentation-only commits added to the branch after delivery). `Tests` = sum of the register field
"Tests" (number of test files touched per commit). Evidence `[R] QFN-a..b` = the register rows
in `/srv/community-repos/qwen38-flashnext-nvfp4-kv/docs/PATCHES.md` (and `patches/register.yaml`)
for exactly these commits; where additional proof exists it is named with an explicit path.
Rows without extra proof have GPU state `pending`/`n/a` in the register — treated as
**unverified for runtime effect** (register fields are evidence of provenance, not of effect).

| ID | Purpose | Origin | PRs | n | Tests | Re-land watch | Evidence | Commits |
|---|---|---|---|---|---|---|---|---|
| **G00** | Merge of feat/hicache-buffer-stream into the RC2 integration branch (merge commit; absent from the QFN register — see findings F3). | n/a | - | 1 | 0 | - | [R] - | `6fdf52f7ad` |
| **G02** | HiCache host-to-device read path: zero-copy barrier-free fast_file reads, buffer stream for storage hits larger than the host arena, oversize backup splitting (rust tree core), stream-restore worker device index, optional O_DIRECT page reads. | fork-only (deferred) | - | 7 | 7 | - | [R] QFN-0141..QFN-0154; [O]report-hicache-stream-on-busy-arena.md (stream-Pfad-Kontext), [O]report-165-backport-hicache-loadback.md (#50911-Kandidat auf bp/50911, nicht im Stack) | `1ad48f14e1`, `660b99fbc5`, `48e38cb469`, `5a215cb2ed`, `c9090cb88a`, `4479f46fff`, `cece48b717` |
| **G03** | QSA NVFP4 KV cache for Qwen compressed-QSA models with HiCache sidecars: E4M3 block-scale mirror, MTP draft sidecar mirror, fail-closed host rejection pin, server_args enablement. | fork-only (deferred) | - | 5 | 7 | - | [R] QFN-0136..QFN-0140; [HB] (NVFP4-KV-Produktionsstand, Zeile HB-14) | `47045ecad3`, `af952805a7`, `071b8d4460`, `a22a95b12b`, `fec92099ea` |
| **G04** | Dynamic draft vocabulary: ring buffer, adaptive policies, coverage metrics (decode log + Prometheus), documentation, stack test init. | open (unmerged upstream) / fork-only (deferred) | - | 6 | 4 | - | [R] QFN-0099..QFN-0150 | `bc40552bbe`, `d7f4537d16`, `698a65db95`, `d1506bcbe4`, `e572adacc8`, `50f3315bd2` |
| **G05** | Warm-start speedups: consolidated JIT/torch-extension/CuTe DSL caches under SGLANG_CACHE_DIR, FlashInfer autotune replay skip, numpy CSF NVFP4 scale fast decode. | fork-only (deferred) | - | 3 | 3 | - | [R] QFN-0145..QFN-0147 | `740c65442e`, `3f3fca99de`, `7f4ceec17d` |
| **G06** | Mamba prefill slot admission fix (single commit; production hotfix for the re-prefill pathology class). | fork-only (deferred) | - | 1 | 1 | - | [R] QFN-0144; [O]report-seed-mamba-path-cap-2-setzen--re-prefill-pathologie.md (MAMBA_MAX_STATES_PER_PATH=2) | `f9511985fc` |
| **G07** | Hyperconnection gate side stream allowed on SM120 (counterpart of HB-03; GPU validation of QFN-0097 still pending). | fork-only (deferred) | - | 1 | 0 | - | [R] QFN-0097 | `5e04e7d3f6` |
| **G08** | qwen4_exp: reject incomplete NVFP4 PLE checkpoints; drop stale is_in_ci patch in weight-utils test. | fork-only (deferred) | - | 2 | 2 | - | [R] QFN-0096..QFN-0098 | `177f542b78`, `e20444cb72` |
| **G09** | HiCache backport mini-paket from upstream PRs (#43749 #44776 #46451 #47339 #49329). NOTE: #44776 and #49329 are still OPEN upstream although the register says is-upstream (finding F1, verified in [INV] L17-18). | is-upstream (merged nightly sync) | #43749, #44776, #46451, #47339, #49329 | 5 | 10 | watch #44776, #49329 (still open, [INV]) — not merged despite register label | [R] QFN-0130..QFN-0134; [INV] L17-18 (#44776/#49329 offen — Register-Korrektur F1) | `0a3dd3b87b`, `ed1a4a6920`, `28d6637e33`, `1371c8e707`, `b236c865bb` |
| **G10** | QSA/sparse-decode and MoE kernel backports: split-K fp8 sparse decode (#48296), indexer JIT decode scoring (#48250), fused NVFP4 SwiGLU MoE for SM120 (#48262), flashinfer shared-workspace release (#48258), NIXL incomplete-file rejection (#46131), QSA index-key ring overwrite fix (#46125). | is-upstream (merged nightly sync) | #46125, #46131, #48250, #48258, #48262, #48296 | 6 | 18 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0100..QFN-0105 | `5ecd6b504b`, `1fa9d253d1`, `9738457e01`, `186573083a`, `129f2dd21a`, `8ad6626ebf` |
| **G11** | Unmerged HiCache perf pack (9 commits, [Perf]/[Feat] hicache): sub-agent host-eviction ranking, ordered write-back demotion, KV/Mamba CTA load quotas, Mamba host-pool ratio scale, tracking-slot parking, stream-ordered host-only checkpoints. Part of the #50156 split lineage — re-land watch #51051–#51068. | open (unmerged upstream) | - | 9 | 17 | watch #51051–#51068 (split series of #50156; all open drafts 2026-11-19, [INV]) — drop local twin when one lands | [R] QFN-0118..QFN-0129; [INV] (PR-Status), [O]report-review-hicache-codepath.md | `6f85196aa1`, `8974f29715`, `94ca1f582b`, `3e88495145`, `fdb7c51056`, `05679afd70`, `bddd781672`, `253e02ea75`, `969ffa7820` |
| **G12** | Unmerged spec/EAGLE perf pack (9 commits): post-verify graph pool presize and key bound, single-launch verify commit conv-window/PLE rollbacks, mrope device rows, graph input copy plan, fused top-p mass, tiled GDN ReplaySSM compact commit, draft top-1 via sampler argmax, multi-CTA verify tree, draft vocab weight placeholders. | open (unmerged upstream) | - | 9 | 13 | watch #51051–#51068 (split series of #50156; all open drafts 2026-11-19, [INV]) — drop local twin when one lands | [R] QFN-0109..QFN-0117; [HB] (Vergleichstabelle HB-01..14) | `f0729be83e`, `84964e80dd`, `2f6c27b3fe`, `23840e403b`, `204b7f0ff0`, `d9dd1b9f0f`, `ad5fa6cfde`, `05ed262e0c`, `de3d9ad2cd` |
| **G13** | Unmerged kernel micro-optimizations: sm_120 skinny BF16 GEMV for MoE router and MTP fc (HB-06/HB-07), MoE singleton route pruning (HB-01), QSA qk-norm/KV-prepare/gate PDL launches (HB-02), non-specialized graph metadata kernels. | open (unmerged upstream) | - | 4 | 3 | watch #51051–#51068 (split series of #50156; all open drafts 2026-11-19, [INV]) — drop local twin when one lands | [R] QFN-0107..QFN-0128; [HB] Zeilen HB-01/HB-02/HB-06/HB-07 (Autor-Berichte, eigene Messung 644.4 tok/s in [HB]) | `0a3763cfac`, `91230fbeee`, `f340e338ac`, `ca9caafb62` |
| **G14** | Quantization and PLE offload group: NVFP4-packed QAD checkpoints, CSF-compressed expert scales, pinned PLE table at exact size, THP-backed pinned PLE host table (#51065, merged upstream). | is-upstream (merged nightly sync) / open (unmerged upstream) | #51065 | 4 | 8 | watch #51051–#51068 (split series of #50156; all open drafts 2026-11-19, [INV]) — drop local twin when one lands | [R] QFN-0093..QFN-0126; [INV] | `8a5ac1de54`, `7d5965fd1d`, `792dd865a0`, `2f3dce16e0` |
| **G16** | Upstream scheduler + GC heap-freeze package: scheduler weight-write wait (#50717), SPF prefix dedupe (#46022), encoder-replay reservation (#46063), GC idle freeze and frozen-garbage checks (#50883 #50882 #50980 #50981 #50983). Touches the scheduler.py conflict hotspot. | is-upstream (merged nightly sync) | #46022, #46063, #50717, #50882, #50883, #50980, #50981, #50983 | 8 | 16 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0013..QFN-0081; [O]report-174-backport-scheduler-minipaket.md (Titelbeleg, Rohtranskript-inhalt schwach) | `aa3ccefd38`, `d1e7987757`, `8d07d63f37`, `9eb23a3e63`, `f74df0c3d6`, `0e8c255117`, `988b043982`, `9bf6da2214` |
| **G15e** | Upstream model and runtime support package: Kimi-K3 day-0 merge, CP prefill, Inkling2 kernels, DSpark/DFLASH spec series, beam-search, KVShard Qwen3.8-Flash-Next, dsv41 Engram, tokenizer sampling masks, disaggregation fixes. | is-upstream (merged nightly sync) | #38344, #44349, #44447, #46190, #47293, #47854, #48414, #48745, #50126, #50470, #50471, #50523, #50524, #50525, #50526, #50712, #50740, #50741, #50764, #50771, #50839, #50895, #50937, #50941, #50982, #50995 | 26 | 70 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0009..QFN-0092 | `7aaebb1cf0`, `372b35034d`, `1280ff574b`, `4b9f9043fa`, `1aa45c6617`, `9c1b3eceb8`, `db12772f0c`, `c02b04fb50`, `c67db7c762`, `09185be8f9`, `4e07dc8aa1`, `6321f714dd`, `56d5f6cb71`, `b743e776ab`, `d8a6e5a9de`, `837d1cf76e`, `471ae1b32c`, `41419ca900`, `6795aed975`, `a27a59c2e7`, `a0d169e8d0`, `7fb3ccb5d2`, `0bf51f9d0b`, `00db5b72e3`, `70b34f322a`, `98c45e11c4` |
| **G15f** | Remaining upstream runtime fixes: SWA branch-point checkpoint insert (#50059), dsv4 V4.1 prefill indexer DeepGEMM fp8 MQA scoring (#50719). | is-upstream (merged nightly sync) | #50059, #50719 | 2 | 4 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0069..QFN-0080 | `90121c62a8`, `9fd8b7fef2` |
| **G15a** | Upstream AMD/ROCm package (nightly sync; not our hardware — carried for tree consistency with the merged base). | is-upstream (merged nightly sync) | #41683, #45205, #47052, #48064, #48065, #48095, #48097, #49384, #49732, #49884, #50503, #50504, #50512, #50656, #50677, #50686, #50747, #50748, #50759, #50870 | 20 | 17 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0002..QFN-0088 | `85360aaab8`, `5cefe8de82`, `703e763d5f`, `4469f9fc7f`, `eade493029`, `9158db2b97`, `7e2d9b85a3`, `f549aa8b7b`, `87f393c249`, `4009bc1be5`, `b5a4fca3a6`, `d766b7327e`, `ddb0129831`, `04f3bb4678`, `3da953fab5`, `e4d84dca71`, `6c9a1938d0`, `04af670a31`, `b097614b25`, `2adf75fec6` |
| **G15b** | Upstream NPU/Intel/MUSA package (nightly sync; not our hardware). | is-upstream (merged nightly sync) | #37570, #41496, #41521, #43305, #46029, #49289, #49353, #49481, #49999, #50097, #50117, #50486, #50676, #50725, #50744, #50925, #50969 | 17 | 53 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0011..QFN-0079 | `ba9c86bb7e`, `513e44c89c`, `cc7054723b`, `ced0383482`, `cb08ff1b9d`, `3400b40b34`, `5dd0c10e33`, `fbdad0224a`, `435ee288ed`, `e55e2eeb1f`, `22df117a99`, `eef42710c4`, `5b35f1badb`, `d08d15985b`, `4aef4d60e3`, `9e14a21940`, `2a71304ac5` |
| **G15c** | Upstream diffusion package (nightly sync; unrelated to the LLM serving path). | is-upstream (merged nightly sync) | #34514, #38766, #49840, #50638, #50640, #50867, #50868, #50960 | 8 | 15 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0001..QFN-0089 | `9d9b45d413`, `8241f8e6f6`, `2e07b70e50`, `8e737da037`, `2d8e3681dc`, `2123631e0e`, `3797ca3b1c`, `d79c53ca37` |
| **G15d** | Upstream CI/test/chore/doc package (includes suite unification #50874 — large test-inventory churn, low runtime risk). | is-upstream (merged nightly sync) | #40732, #42806, #50815, #50824, #50872, #50874, #50884, #50940, #50994, #51021, #51044 | 11 | 987 | already in upstream nightly — rebase drops by patch-id | [R] QFN-0019..QFN-0084 | `b43c4c00a6`, `31e55a5120`, `b2cc023f23`, `39af8b1211`, `b8555f91bb`, `b2fc126830`, `47dbe1702c`, `ba93265487`, `bb477b230b`, `83c2c71774`, `f534dbaeb2` |

| **DOK** | Documentation-only commits added to `rc/qwen38-rc2` after stack delivery (this ledger itself). Append each new docs commit here; the coverage check pins the stack range below, so docs commits never distort it. | OWN (docs) | - | 1 | 0 | n/a | [O]report-flashnext-rebase-entscheidung.md | `091eaa3a04` |

## 4. Feature branches on top of `rc/qwen38-rc2`

All three branch tips carry `rc/qwen38-rc2` head `1ad48f14e1` as merge-base (verified
`git merge-base`, 2026-11-19). None of their commits appear in [R] as current tips — the
register's `pending:feat/draft-vocab-head-buckets` set is stale (10 registered shas, zero
overlap with the current 23-commit tip — finding F2). Branch evidence therefore = git itself
(diffstat, test files) + [HB] pipeline runs 02–04; runtime effect is **unverified** unless
[HB] names a measurement path.

### 4.1 `feat/qsa-nvfp4-fp8-gather` (7 commits, `6bdcfa37ab`, run 02)

FP8 scratch attention over NVFP4 QSA gathers: dequantize gathers into an FP8 scratch,
FP8-vs-BF16 scratch end-to-end comparison, graph valid-count/dtype fixes, asserted
descale roundtrip, recipe constants pinned to the quantizer.
Diff vs rc2: 7 files, +1304/−20 (measured `git diff --shortstat`). Tests:
`test/registered/unit/layers/attention/test_qsa_nvfp4_fp8_gather_launch.py`,
`test/registered/unit/layers/attention/test_qsa_nvfp4_kv.py`. Upstream: local (no PR).
GPU effect: unverified (no measurement path in [HB] for this branch yet).

| Commit | Subject |
|---|---|
| `2420da2ba9` | Add SGLANG_QSA_NVFP4_SCRATCH and the E4M3-safe descale recipe helpers |
| `bfecb53b0d` | Dequantize QSA NVFP4 gathers into an FP8 scratch and allow split-K decode on it |
| `9c2b5ef568` | Test NVFP4 FP8 gather scratch fidelity and launch wiring |
| `c2ca781cc2` | Assert V-side descale roundtrip in the NVFP4 FP8 prefill pack test |
| `dd5ed40b53` | Fix graph valid counts and FA2 dtype, bound tables, log NVFP4 plan |
| `58bc1e3dc2` | Compare FP8 and BF16 scratch attention end to end |
| `6bdcfa37ab` | Pin FP4 recipe constants to the quantizer and document the scratch flag |

### 4.2 `perf/mxfp8-skinny-gemv` (8 commits, `484b87168e`, run 03)

Triton skinny GEMV for dense MXFP8 linears: kernel + apply-routing, device gate hardening,
graph-timed benchmark fix, M=2/M=8 interpreter coverage, CPU tests + GPU micro-benchmark,
scratch-counter documentation. Diff vs rc2: 5 files, +2274/−0. Tests:
`test/registered/unit/layers/test_mxfp8_skinny_gemv.py`. Upstream: local. GPU effect:
unverified (micro-benchmark exists on branch; no production measurement path).

| Commit | Subject |
|---|---|
| `3ec65ce789` | perf(mxfp8): add a Triton skinny GEMV for dense MXFP8 linears |
| `da75a9ed4e` | perf(mxfp8): route skinny MXFP8 linears to the Triton GEMV in apply |
| `bc5fe7f716` | test(mxfp8): cover the skinny GEMV on CPU and add a GPU micro-benchmark |
| `06c4c2d5b9` | test(mxfp8): cover M=2 and M=8 in the skinny GEMV interpreter check |
| `672fa14d50` | fix(mxfp8): harden the skinny GEMV gate, plan and scratch |
| `65e21d5d96` | fix(mxfp8): ask the weight device before the skinny GEMV |
| `7f0f4e1b2b` | fix(bench): time the GEMV inside CUDA graphs, not the fallback |
| `484b87168e` | docs(mxfp8): say what the skinny GEMV scratch counters guarantee |

### 4.3 `feat/draft-vocab-head-buckets` (23 commits, `182f38c4bb`, run 04)

Draft vocab head-size buckets: batch-rounded bucket sizing, LRU-ordered heads, capture
against the batch's head bucket, recency across restarts/boot, padded-shape eager binds,
compact mirror without temporary, boot gates and formatter hygiene. Diff vs rc2:
12 files, +1825/−39. Tests: `test/registered/unit/speculative/test_dynamic_draft_vocab.py`.
Upstream: local. **Single copy on t15** — no origin ref (see §2). GPU effect: unverified.

| Commit | Subject |
|---|---|
| `4816d60035` | spec: keep the draft head LRU-ordered and bucketed by batch size |
| `e5d38fe194` | spec: capture the draft graphs against the batch's head bucket |
| `985dbe7d7d` | test: cover draft vocab head-size buckets |
| `a972e6e472` | docs: document draft vocab head-size buckets |
| `41c6bbf35d` | spec: size draft vocab head buckets by rounding the batch size up |
| `e0da15352c` | spec: compact the draft vocab mirror without a temporary |
| `519e091f69` | spec: carry draft vocab bucket recency over restarts and boot |
| `c1191b40dd` | spec: bind eager draft-extend forwards to the batch bucket |
| `13e8716af9` | docs: match the dynamic draft vocab doc to the bucket contract |
| `c6cf4e8d19` | spec: keep the bucket code formatter-clean |
| `5ddd3607ce` | spec: bind the eager draft extends to the padded graph shape |
| `1ba3155071` | spec: hand the captured draft head a buffer its width fits |
| `70e73795b9` | test: give the harness draft workers the bucket hooks the runners read |
| `e70597154b` | spec: keep a damaged tok_use table from costing the warm start |
| `684b58e68d` | spec: report a bucket ladder the worker cannot use at boot |
| `44cf786931` | docs: match the vocab doc to the bucketed capture and pool sizing |
| `e431613b69` | test: cover the eager binds, harness contract and boot gates |
| `355d5518f1` | spec: keep the pool height off the buckets-off logits buffer path |
| `0ab0597d5d` | spec: keep the capture ladder off the buckets-off eager binds |
| `9bb905c23c` | test: pin the buckets-off paths to the rc2 field set |
| `4849d7b201` | spec: ask the worker for its bucket ladder in one place |
| `db598c0b88` | spec: keep the capture path off a ladderless worker |
| `182f38c4bb` | test: pin the capture path against a worker without bucket hooks |

## 5. Hyperburst comparison ledger (HB block, pre-landed from Akte 142 / Margot 2026-11-19)

Source of details: [HB] (parity table; every number carries source commit, configuration
and measurement path; projections are labelled). Paths relative to
`/srv/community-repos/hyperburst-rtxpro6000` and `/srv/community-repos/sglang-rtxpro6000`
(branch `flash-next-fast`, publication commit `d21289f` = 121-patch stack on `16e5682aad`
plus ten FlashInfer-0.6.17 patches plus source request sgl-project/sglang#30967).

| ID | Feature | HB-Quelle | Effekt (Messpfad) | rc2-Stand | Entscheidung |
|---|---|---|---|---|---|
| HB-01 | P2 Singleton-Prune tau | Patches 0100-0103 ab `d21289f` | -9.5 % W16-Schritt (`docs/optimizations.md` §E L884-960) | vorhanden (QFN-0107) | halten; Haertung laeuft in Run 06 |
| HB-02 | Triton-PDL | Patches 0067-0074 | rund 4x Startlatenz-Gewinn W4 (`docs/optimizations.md` §C13 L731-758) | teil-weise (nur QSA) | adapt ueber Run 06 (0f1b00fbe) |
| HB-03 | HC-Fusionen | Patches 0003-0077 + `b509a44` | Mischung in drei Kerneln (`docs/optimizations.md` §C8-C10) | vorhanden (QFN-0097 GPU-Bestaetigung offen) | halten; Folge: GPU-Validierung QFN-0097 |
| HB-04 | GDN-Layouts/out_proj-Retune | Patches 0018-0070 + `b509a44` | out_proj 14 auf 13.1 us (`docs/lab-notes/EXCLUSIVE_TIME_MAP.md:310`) | vorhanden (QFN-0061) | halten |
| HB-05 | NORM_INTO_GEMV | Patch 0079 | Norm in GEMV-Endphase (`docs/optimizations.md` §C11 L681-711) | fehlt | adapt ueber Run 05 (e2c0661b9) |
| HB-06 | Router-GEMV | Patch 0019 + RT1 (`f3ff967`, `b035b05`) | Router 3.8 auf 1.5-1.6 us (`docs/optimizations.md` §B3 L405-422) | teil-weise | adapt ueber Run 05 (5ddbc4809, dd2725ba4) |
| HB-07 | MTP-FC-GEMV | `899fc4e` | -0.10 bis -0.20 ms/Schritt (`docs/lab-notes/DRAFT_GLUE_SPEC.md:337-530`; HB Auslieferung AUS) | vorhanden (QFN-0108) | halten; Einschalten nur nach Messung |
| HB-08 | Draft-MoE-GEMV | Patch 0112 (`72de55f`) | 49.4 auf 29.8 us/Aufruf (`docs/lab-notes/DG1_DRAFT_MOE_GEMV.md`) | fehlt | adapt ueber Run 05 (26c45b4dd) |
| HB-09 | W8A16-GEMV | Patches 0004-0053 | 6.99 auf 5.81 ms/Schritt W16 (`docs/optimizations.md` §B1 L335-390) | teil-weise (nur BF16-Einstieg) | offen; Befund: Run 02 ist MXFP8 (QFN-0159), nicht W8A16 |
| HB-10 | v5-MTP-Head | keine - privates Gut (`docs/optimizations.md:1525`) | +1-5 % W4, +2-8 % W16 gegen v3 (`docs/reproduce.md:110`) - Projektion | fehlt | ablehnen - Quelle nicht erhaeltlich |
| HB-11 | Token-Map 49152 | `d21289f` (public_32768/49152.pt) | heisse Karte ersetzt Haefufigkeitskarte (`docs/optimizations.md` §A1 L51-79) | vorhanden (hyudryu_top49152) | halten (Groessenparitaet); Inhaltsvergleich offen |
| HB-12 | _compact_kv-Anomalie | keine - HB fasste den Grundkernel nie | 13.2 gegen 4.7 us/Aufruf (`docs/lab-notes/EXCLUSIVE_TIME_MAP_0907.md:214`) | vorhanden (eigener QSA-Pfad) | offen - Auditaufgabe im eigenen Baum |
| HB-13 | MXFP4-Experten | Idee unveraendert; naechster Schipp `089550a` | geschaetzt -372 dann -270 us (`docs/lab-notes/MOE_SMALLM_SPEC.md` §8.5) - Schaetzung | Code da, Produktion NVFP4-CSF (QFN-0093) | offen - erst GPU-Fenster mit Messung |
| HB-14 | FP8-KV gegen NVFP4-KV | Grundfork-Kernel (`launch/serve-local.sh:53`) | bitgenauer Einstapel-Speicher (`docs/optimizations.md` §C3 L483-500) | Produktion NVFP4-KV (Ziel nvfp4, Entwurf fp8_e4m3) | ablehnen - NVFP4-KV bleibt; alle HB-Zahlen inkl. 710.2 stehen auf FP8-KV-Grund |

Note to the owner (from Akte 142): the 710.2 tok/s figure at four parallel requests is
author-reported without a published measurement file (`docs/updates/2026-10-27.md:105` in the
hyperburst repo); in this ledger it counts as a projection only. Own reference measurement of
the HB stack on our machine: 644.4 tok/s — measurement path [HB], section Wirkung.

## 6. Backup plan (owner decision — the operator does not push)

Push target is `origin` = `git@github.com:thomasdenk79-cyber/sglang.git` (private).
State as measured 2026-11-19 (§2): `rc/qwen38-rc2`, `feat/qsa-nvfp4-fp8-gather` and
`perf/mxfp8-skinny-gemv` are already mirrored on origin tip-for-tip;
`feat/draft-vocab-head-buckets` is a **single copy on t15** and this ledger's own commit
lands on top of `rc/qwen38-rc2`, making the rc2 ref diverge from origin as well.
Exact commands (run by the owner from `/srv/community-repos/sglang-wt-rc2`):

```bash
cd /srv/community-repos/sglang-wt-rc2

# 1) Immutable backup refs (protects against ref moves, not against disk loss)
git push --atomic origin \
  rc/qwen38-rc2:backup/rc-qwen38-rc2-20261119 \
  feat/qsa-nvfp4-fp8-gather:backup/feat-qsa-nvfp4-fp8-gather-20261119 \
  perf/mxfp8-skinny-gemv:backup/perf-mxfp8-skinny-gemv-20261119 \
  feat/draft-vocab-head-buckets:backup/feat-draft-vocab-head-buckets-20261119

# 2) Verify the push (must list the four backup/* refs with the §2 tips)
git ls-remote --heads origin 'refs/heads/backup/*'

# 3) LKG tag anchor (push-gate friendly: never force, never rewind)
git tag lkg/rc-qwen38-rc2-20261119 rc/qwen38-rc2
git push origin refs/tags/lkg/rc-qwen38-rc2-20261119

# 4) Offline bundle (survives disk loss; prerequisites = base commit, small file)
git bundle create /srv/luni-repos/luni-ai-operator/log/run/flashnext-stack-20261119.bundle \
  rc/qwen38-rc2 feat/qsa-nvfp4-fp8-gather perf/mxfp8-skinny-gemv \
  feat/draft-vocab-head-buckets ^25c1c439
git bundle verify /srv/luni-repos/luni-ai-operator/log/run/flashnext-stack-20261119.bundle
# copy the bundle off the machine (owner channel), verify anytime with:
#   git bundle verify <pfad>   (inside a clone that contains 25c1c439)
```

Do **not** delete or force-move existing origin refs; history rewrite of these branches is
out of scope of this ledger (task boundary). Other local branches worth the same protection
(`rc/qwen38-rc3/rc4` lineage, `stack/qwen38-20261117`, `bp/50911-hicache-loadback`) exist in
the same object store; list them with `git branch` before pushing and extend step 1 if the
owner wants them archived too.

## 7. Re-base plan onto a newer upstream nightly

Preconditions (measured 2026-11-19): base `25c1c439` is an ancestor of `upstream/main`
(`145647c55`), backup per §6 executed by the owner, engine lease taken before any GPU
validation (`bin/comm.sh lock` + router check), production `qwen38-rc2` (:30000) untouched
until the rebased stack passes its test gate.

```bash
cd /srv/community-repos/sglang-wt-rc2
git fetch upstream --prune
NEW=145647c55                      # pin the chosen nightly here and log it in this ledger
git branch rc/qwen38-rc2-base-20261119 rc/qwen38-rc2        # local safety anchor

# 1) Replay the whole stack onto the new base; git rebase drops patch-equivalent upstream commits
git worktree add /srv/community-repos/sglang-wt-rebase -b rc/qwen38-rc3 $NEW
cd /srv/community-repos/sglang-wt-rebase
git rebase --onto "$NEW" 25c1c439 rc/qwen38-rc3

# 2) Audit kept vs dropped: expect G09*, G10, G15a–G15f, G16 to vanish by patch-id;
#    G11–G14 partly (upstream rework changes patch-ids), OWN stays untouched
git log --oneline --cherry-pick --right-only "$NEW"...rc/qwen38-rc2 | wc -l

# 3) Re-land watch: when a PR of the #51051–#51068 split series (successor line of the
#    reverted #50156, [INV]) merges upstream, delete its local twin from G11/G12/G13/G14
#    instead of resolving the duplicate by hand

# 4) Move the three feature branches (all stacked on the old rc2 tip):
for b in feat/qsa-nvfp4-fp8-gather perf/mxfp8-skinny-gemv feat/draft-vocab-head-buckets; do
  git rebase --onto rc/qwen38-rc3 1ad48f14e1 "$b" || break
done

# 5) Test gate before any production switch (group test files from §3/§4):
python3 -m pytest test/registered/unit/mem_cache/test_buffer_stream_restore.py \
  test/registered/unit/layers/attention/test_qsa_nvfp4_kv.py \
  test/registered/unit/layers/attention/test_qsa_nvfp4_fp8_gather_launch.py \
  test/registered/unit/layers/test_mxfp8_skinny_gemv.py \
  test/registered/unit/speculative/test_dynamic_draft_vocab.py

# 6) Re-run the coverage check of Appendix A against the rebased stack, then refresh §2/§3
```

Expected conflict hot spots — most-changed files across the 155 stack commits (measured
`git log --name-only 25c1c439..rc/qwen38-rc2 | sort | uniq -c`):

| File | × | Colliding groups |
|---|---|---|
| `python/sglang/srt/environ.py` | 28 | G02–G05, G11–G13 (flag registry — conflicts near-certain, mechanical) |
| `mem_cache/unified_radix_cache.py` | 8 | G11 vs upstream radix churn |
| `docs/docs/references/environment_variables.mdx` | 7 | with environ.py |
| `managers/scheduler.py` | 6 | G16, G11 residency |
| `mem_cache/kv_cache_configurator.py` | 5 | G03, G11 |
| `mem_cache/hicache_storage.py` | 5 | G02, G09 |
| `managers/schedule_policy.py` | 5 | G06, G16 |
| `layers/attention/qwen_sparse_attn_backend.py` | 5 | G03, G10, G13 |
| `test/registered/unit/mem_cache/test_buffer_stream_restore.py` | 5 | G02 (test follows implementation) |
| `speculative/eagle_worker_v2.py` | 4 | G04, G12, F3 |
| `models/qwen4_exp.py` | 4 | G08, G14 |
| `mem_cache/unified_cache/components/mamba.py` | 4 | G09, G11 |
| `model_executor/pool_configurator.py` / `forward_batch_info.py` | 4 | G12 |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 4 | G03 |

## 8. Findings (register discrepancies recorded during creation)

| # | Finding | Consequence |
|---|---|---|
| F1 | [R] labels `QFN-0131` (#44776) and `QFN-0132` (#49329) as `is-upstream`, but both PRs are **open, unmerged** — verified 2026-11-19 in [INV] (rows L17–18: mergeable, `blocked`, 0 human reviews). | This ledger's Upstream column is corrected; register fix queued as follow-up task in the operator queue (`TASKS/todo/*register-status-fix*`). Until regeneration, trust [INV] over [R] for unmerged-looking PRs. |
| F2 | [R] set `pending:feat/draft-vocab-head-buckets` is stale: 10 registered shas, zero overlap with the current 23-commit tip (`182f38c4bb`). | Branch truth = git + §4 tables; regenerate the register before next use. |
| F3 | Merge commit `6fdf52f7ad` is absent from [R] (register is subject-keyed and skips merges). | Coverage accounting relies on this ledger (Appendix A), not on register row count. |
| F4 | Hyperburst's 710.2 tok/s @4 parallel is author-reported (`docs/updates/2026-10-27.md:105` in the hyperburst repo), own measurement 644.4 tok/s [HB]. | Ledger §5 keeps it as projection only. |
| F5 | Akte 142 context said the rc2 branch "is not pushed (single copy on t15)"; measurement 2026-11-19 shows origin already mirrors rc2/gather/gemv tip-for-tip (§2). | Backup plan §6 focuses the urgency on dvbucket + future ref moves; the Akte note is outdated. |

## Appendix A — Coverage check (machine)

Run after every landing on `rc/*` and before every release step of §7:

```bash
cd /srv/community-repos/sglang-wt-rc2
PIN=1ad48f14e1                     # stack tip at ledger delivery; docs commits after the pin live in the DOK row
git log --format=%h --abbrev=10 25c1c439..$PIN | sort > /tmp/range.txt
grep -E '^\| \*\*[GD]' docs/PATCH-LEDGER.md | grep -oE '`[0-9a-f]{10}`' | tr -d '`' | sort > /tmp/doc.txt
comm -23 /tmp/range.txt /tmp/doc.txt | wc -l   # must be 0 — every stack commit listed
uniq -d /tmp/doc.txt                            # must be empty — each listed exactly once
# informational: any NEW non-docs commit on the branch beyond $PIN must become a G row:
git log --format='%h %s' --abbrev=10 $PIN..rc/qwen38-rc2
```

Result at creation (2026-11-19 11:04 UTC, re-checked after the §4 header fix): pinned
range=155, listed=155 in G rows (+1 DOK row), missing=0, duplicated=0; only docs commits
(`091eaa3a04` and its follow-ups) sit beyond the pin.
