# Dynamic draft vocabulary

Self-learning draft LM head for EAGLE/MTP with `--speculative-token-map`
(FR-Spec). Module: `dynamic_draft_vocab.py`; wiring: `eagle_worker_v2.py`.

## Contents
- [Row layout](#row-layout)
- [Policies](#policies)
- [Environment variables](#environment-variables)
- [Metrics](#metrics)
- [Persistence and profiles](#persistence-and-profiles)
- [Smaller static maps](#smaller-static-maps)
- [CUDA graphs and host syncs](#cuda-graphs-and-host-syncs)
- [Future: head-size buckets](#future-head-size-buckets)

## Row layout
The draft head is `target_head[ids]` in one fixed buffer:

| Rows | Content |
|---|---|
| `[0, core)` | protected static map (all of it unless `ADAPTIVE_STATIC=1`) |
| ring | tier A: FIFO of recent prompt + output tokens |
| pool | tier B: scored tokens (LRU stamp or decayed LFU) |
| last | sink row for unused slots |

Each token lives in at most one row. Draft cost scales with the **total**
row count, so the goal is a higher hit rate at the same or a smaller total.

## Policies
`SGLANG_SPEC_DYNAMIC_VOCAB_POLICY`:

- `fifo` (default): ring of `SLOTS` with distinct tokens, never refreshed. Bit-identical to the previous behaviour.
- `lru`: pool of `SLOTS`; every staged occurrence refreshes recency; the least recently used token is evicted.
- `lfu`: pool of `SLOTS` with exponentially decayed frequency (`HALF_LIFE` staged tokens); output tokens weigh `OUTPUT_WEIGHT`.
- `tiered`: ring of `SLOTS` (prompt + output), plus a proven pool of `PROVEN_SLOTS` fed only by accepted output tokens. An output token is promoted from the ring (swap) or admitted from absent. Pool scoring uses `PROVEN_SCORE` = `lru` | `lfu`.

How eviction works:
- It runs only when there are new candidates.
- For `k` candidates it does one `topk(k, largest=False)` over the pool scores.
- A candidate is admitted only if its score beats the victim's score.

`ADAPTIVE_STATIC=1` keeps only the first `STATIC_CORE` static ids protected:
- The rest seed the pool with a head start of `STATIC_HORIZON` staged tokens.
- They are demoted once they stay unused for longer than that.
- `STATIC_CORE=0` gives a fully adaptive head: it is seeded from the persisted profile, otherwise from the map, and everything can age out.

`TTL_S` expiry:
- Pool tokens without an accepted-output use for `TTL_S` seconds are evicted first.
- `MIN_ROWS` live rows are always kept.

## Environment variables
All names are prefixed `SGLANG_SPEC_DYNAMIC_VOCAB_`. Definitions: `environ.py`.

| Name | Default | Meaning |
|---|---|---|
| `SLOTS` | 0 | ring (fifo/tiered) or pool (lru/lfu) size; 0 disables |
| `POLICY` | fifo | fifo, lru, lfu or tiered |
| `PROVEN_SLOTS` | 0 | tiered proven-pool size |
| `PROVEN_SCORE` | lru | tiered pool scoring: lru or lfu |
| `HALF_LIFE` | 65536 | LFU half-life in staged tokens |
| `OUTPUT_WEIGHT` | 4.0 | LFU weight of output vs prompt tokens |
| `COMMIT_EVERY` | 1 | apply staged tokens every n-th commit |
| `ADAPTIVE_STATIC` | 0 | score the static map beyond the core |
| `STATIC_CORE` | 8192 | protected static rows (0 allowed) |
| `STATIC_RANK` | "" | rank file ordering the static map before the core cut |
| `STATIC_HORIZON` | 1048576 | head start of non-core static ids (staged tokens) |
| `TTL_S` | 0 | evict-first age in seconds (0 = off) |
| `MIN_ROWS` | 0 | live-row floor for TTL expiry |
| `STATE_DIR` | "" | persistence directory (empty = off) |
| `PROFILE` | default | state file `dvocab-<PROFILE>.pt` |
| `SAVE_INTERVAL` | 300 | seconds between background saves |
| `METRICS` | 0 | coverage counters and log suffix |

## Metrics
With `METRICS=1`, the decode log line gets a suffix like this:

```
draft vocab: cov=97.1% static=88.0% A=6.2% B=2.9% ins=310 promo=40 evict=12 demote=3 active=57344 expired=0
```

- `cov`: the share of accepted output tokens that were in the draft vocab before the update.
- `static`, `A` and `B`: hits per tier.
- The counters (`ins`, `promo`, `evict`, `demote`) are deltas since the last log line.
- Prometheus gauge: `sglang:spec_draft_vocab_coverage` (-1 when disabled).
- On CUDA the counters are copied asynchronously, so they lag one log interval.

## Persistence and profiles
The file is `STATE_DIR/dvocab-<PROFILE>.pt`.

Saving:
- A daemon thread writes it every `SAVE_INTERVAL` seconds, and once more at exit.
- The write is atomic: a temporary file followed by `os.replace`.

The header holds the version, vocab size, map hash, policy and score mode. Files whose vocab size or map hash differ are ignored (with a warning, then a cold start).

Use one profile per workload, e.g. `PROFILE=python`, `rust`, `cpp`. Mount `STATE_DIR` as a volume in containers.

## Smaller static maps
`tokenmaps/hyudryu_top49152.pt` is sorted by token id, **not** by frequency:
- Its first N ids are only a BPE-merge-order proxy.
- To cut a better core, pass a frequency-ranked list via `STATIC_RANK` or `--rank`.

The following command writes a top-K map. When `--state` is given, learned tokens go first, ordered by score:

```bash
python -m sglang.srt.speculative.dynamic_draft_vocab \
  --token-map tokenmaps/hyudryu_top49152.pt --k 32768 \
  --state /dvocab/dvocab-python.pt --out tokenmaps/python_top32768.pt
```

A new map has a new hash, so start it with a new profile.

## CUDA graphs and host syncs
The draft graphs capture the head tensor and `hot_token_id` pointers. Because of that:
- All updates are in place in fixed buffers.
- Shapes never change.
- Commit is vectorized torch with no `.item()` or host sync on the decode path.

Kernel cost per commit, roughly:
- fifo: about 25 small kernels
- lru: about 45
- tiered: about 65
- metrics: about 12 more

That is about 75–200 µs of GPU time per commit; `COMMIT_EVERY` amortizes it.

## Future: head-size buckets
This is not implemented.

The real compute saving from a smaller active set would come from capturing the draft graphs per head-size bucket and using `head[:bucket]`. It needs:
- per-bucket captures in the draft decode and draft-extend runners;
- sliced logits buffers and `hot_token_id`;
- compaction that keeps live rows as a prefix;
- bucket switches between batches, with hysteresis.
