"""CPU micro-benchmark of the HiCacheFile read path (manual, not run in CI).

Replays the streamed-restore L3 read of a QSA NVFP4 deployment on CPU: real host
pool classes (MHA page_first KV + KV-scale / indexer / draft sidecars), the real
HiCacheFile backend, and the controller's ``_page_transfer_kv_batch`` per
STORAGE_BATCH_SIZE batch, cycling through a host-page ring like the stream.

Pages come from an existing store (mounted read-only) or a synthetic store with
the same file-size mix (``--make-synthetic``). Drop the page cache on the host
(``echo 1 > /proc/sys/vm/drop_caches``) before a cold run.

Example (in the sglang container, store at /hicache):
  python scripts/playground/hicache/file_read_bench.py --store /hicache \
      --model-name /models/qwen38-qsa-nvfp4-e4m3b16-v1-dtype_float4_e2m1fn_x2
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import os
import pstats
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    QSANVFP4KVCacheMethod,
)
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    prepare_host_pool_config,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

PAGE = 64
# Per-page file sizes of the measured store (bytes).
KV_LAYERS, KV_HEADS, KV_HEAD_DIM = 12, 2, 256  # 393216 B NVFP4 KV, 2 x 24576 B scales
DRAFT_HEADS, DRAFT_HEAD_DIM = 2, 128  # 1 layer bf16 K+V: 65536 B
SIDECAR_BYTES = {PoolName.INDEXER: 49152, PoolName.DRAFT_INDEXER: 4096}
SCALE_NAMES = (PoolName.KV_K_SCALE, PoolName.KV_V_SCALE)


def _memory(layout):
    return SimpleNamespace(
        hicache_size=0,
        hicache_ratio=2,
        hicache_mem_layout=layout,
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="cache",
    )


def _group(device_pool, layout):
    with (
        patch.object(hybrid_pool_assembler, "get_memory", return_value=_memory(layout)),
        patch.object(
            hybrid_pool_assembler,
            "get_parallel",
            return_value=SimpleNamespace(
                dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
            ),
        ),
        patch.object(
            hybrid_pool_assembler, "_get_allocator_type", return_value="default"
        ),
        patch.object(
            hybrid_pool_assembler,
            "get_mha_host_pool_cls",
            return_value=partial(MHATokenToKVPoolHost, pin_memory=False),
        ),
    ):
        config = prepare_host_pool_config(
            root=device_pool,
            full_layer_mapping={i: i for i in range(device_pool.layer_num)},
            transfer_layer_id_max=device_pool.layer_num,
            transfer_page_size=PAGE,
        )
        return hybrid_pool_assembler.build_host_pool_group(config=config)


def build_pools(ring_pages: int, layout: str):
    """Real host pools whose flat pages match the store's file-size mix."""
    device_tokens = ring_pages * PAGE // 2  # hicache_ratio=2
    kv = MHATokenToKVPool(
        size=device_tokens,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=KV_HEADS,
        head_dim=KV_HEAD_DIM,
        layer_num=KV_LAYERS,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
        quant_method=QSANVFP4KVCacheMethod(
            num_layers=KV_LAYERS, device="cpu", page_size=PAGE
        ),
    )
    group = _group(kv, layout)
    pools = {name: group.get_pool(name) for name in (PoolName.KV, *SCALE_NAMES)}
    draft = MHATokenToKVPool(
        size=device_tokens,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=DRAFT_HEADS,
        head_dim=DRAFT_HEAD_DIM,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    pools[PoolName.DRAFT] = _group(draft, layout).get_pool(PoolName.KV)
    host_pages = pools[PoolName.KV].size // PAGE
    for name, nbytes in SIDECAR_BYTES.items():
        pools[name] = DeepSeekV4PagedHostPool(
            pool_name=str(name),
            device_buffers=[torch.empty(1, PAGE, nbytes // PAGE, dtype=torch.uint8)],
            item_bytes=nbytes,
            num_host_pages=host_pages,
            slot_page_size=PAGE,
            layout=layout,
            pin_memory=False,
            page_aligned_only=True,
        )
    for name, pool in pools.items():
        size = (
            pool.get_dummy_flat_data_page().numel()
            * pool.get_dummy_flat_data_page().element_size()
        )
        print(f"pool {name:14s} {type(pool).__name__:28s} page_bytes={size}")
    return pools


def make_storage(store, model_name, pools):
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )
    storage = HiCacheFile(config, file_path=store)
    storage.register_mem_pool_host(pools[PoolName.KV])
    for name, pool in pools.items():
        storage.register_mem_host_pool_v2(pool, name)
    return storage


def list_keys(storage, store, limit):
    suffix = storage.config_suffix + ".bin"
    keys = sorted(
        n[: -len(suffix)]
        for n in os.listdir(store)
        if n.endswith(suffix) and "." not in n[: -len(suffix)]
    )
    return keys[:limit]


def make_synthetic(storage, pools, pages):
    """Write ``pages`` random pages for every pool through the backend itself."""
    kv = pools[PoolName.KV]
    ring = kv.alloc(min(pages, kv.size // PAGE) * PAGE)
    g = torch.Generator().manual_seed(0)
    keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(pages)]
    span = ring.numel() // PAGE
    for start in range(0, pages, span):
        part = keys[start : start + span]
        idx = ring[: len(part) * PAGE]
        for pool in pools.values():
            buf = pool.kv_buffer
            buf.view(torch.uint8).copy_(
                torch.randint(
                    0, 256, buf.view(torch.uint8).shape, dtype=torch.uint8, generator=g
                )
            )
        storage.batch_set_v2(
            [PoolTransfer(name=n, host_indices=idx, keys=part) for n in pools]
        )
    kv.free(ring)
    return keys


def run(storage, pools, keys, batch):
    kv = pools[PoolName.KV]
    ring_pages = kv.size // PAGE
    ring = kv.alloc(ring_pages * PAGE)
    cc = object.__new__(HiCacheController)
    cc.storage_backend = storage
    cc.storage_host_pool = kv
    cc.mem_pool_host = kv
    cc.page_size = PAGE
    cc.storage_backend_type = "file"
    cc.page_get_func = cc._generic_page_get
    op = SimpleNamespace(request_id="bench", is_terminated=lambda: False)
    sidecars = [
        PoolTransfer(name=n, indices_from_pool=PoolName.KV)
        for n in pools
        if n != PoolName.KV
    ]
    hits = 0
    t0 = time.perf_counter()
    for start in range(0, len(keys), batch):
        part = keys[start : start + batch]
        off = (start % ring_pages) * PAGE
        if off + len(part) * PAGE > ring.numel():
            off = 0
        idx = ring[off : off + len(part) * PAGE]
        hits += cc._page_transfer_kv_batch(
            op, part, idx, HiCacheStorageExtraInfo(), sidecars
        )
    elapsed = time.perf_counter() - t0
    return hits, elapsed


def digest(pools):
    return " ".join(
        f"{name}:{hashlib.sha256(pools[name].kv_buffer.view(torch.uint8).numpy().tobytes()).hexdigest()[:10]}"
        for name in sorted(pools, key=str)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--model-name", default="bench-model")
    p.add_argument("--pages", type=int, default=6603)
    p.add_argument("--ring-pages", type=int, default=1024)
    p.add_argument("--batch", type=int, default=STORAGE_BATCH_SIZE)
    p.add_argument("--layout", default="page_first")
    p.add_argument("--make-synthetic", action="store_true")
    p.add_argument("--profile", default="", help="cProfile output path")
    p.add_argument(
        "--digest", action="store_true", help="hash host pools after the run"
    )
    args = p.parse_args()

    torch.set_num_threads(1)
    pools = build_pools(args.ring_pages, args.layout)
    storage = make_storage(args.store, args.model_name, pools)
    if args.make_synthetic:
        os.makedirs(args.store, exist_ok=True)
        keys = make_synthetic(storage, pools, args.pages)
        print(f"wrote {len(keys)} synthetic pages to {args.store}")
        return
    keys = list_keys(storage, args.store, args.pages)
    page_bytes = sum(
        pool.get_dummy_flat_data_page().numel()
        * pool.get_dummy_flat_data_page().element_size()
        for pool in pools.values()
    )
    print(
        f"workers={storage.read_workers} direct_io={storage.direct_io} "
        f"keys={len(keys)} layout={args.layout} env="
        + " ".join(
            f"{k}={v}"
            for k, v in sorted(os.environ.items())
            if k.startswith("SGLANG_HICACHE")
        )
    )
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    hits, elapsed = run(storage, pools, keys, args.batch)
    if prof:
        prof.disable()
        prof.dump_stats(args.profile)
        pstats.Stats(prof).sort_stats("tottime").print_stats(18)
    gb = hits * page_bytes / 1e9
    print(
        f"RESULT hits={hits}/{len(keys)} GB={gb:.2f} s={elapsed:.3f} GBps={gb / elapsed:.2f} "
        f"files_per_s={hits * len(pools) / elapsed:.0f}"
    )
    if args.digest:
        print("DIGEST", digest(pools))


if __name__ == "__main__":
    main()
