"""HiCacheFile fault classes: what the size gate catches, what it cannot, and
where identity changes must miss instead of collide.

Bug-regression / derived-property cases for the SSD tier spec
(docs/hicache-ssd-state-spec.md in the qwen38-flashnext-nvfp4-kv fork):

1. Defekter Block: committed files are raw bytes with no payload checksum, so
   a same-size corruption is served as a hit. Sentinel case: it fails the day
   a checksum layer lands without the contract being updated.
2. Halber Snapshot: a page is several sibling files published by independent
   renames. Any sidecar gap must truncate the restorable prefix; orphan
   ``.tmp`` files from interrupted writes must never hit and never count.
3. Versionskonflikt: the QSA format tag, logical dtype and namespace fields
   are the only version stamp (they live in the file name), so an identity
   change must miss cleanly and leave the old namespace untouched.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    KVCacheNamespace,
    derive_namespace,
    namespace_digest,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

PAGE = 4
PAGES = 6
SIDECAR_BYTES = 96
NVFP4_TAG_V1 = "m1-qsa-nvfp4-e4m3b16-v1-dtype_float4_e2m1fn_x2"
NVFP4_TAG_V2 = "m1-qsa-nvfp4-e4m3b16-v2-dtype_float4_e2m1fn_x2"
SIDECAR_NAMES = (PoolName.KV_K_SCALE, PoolName.KV_V_SCALE, PoolName.INDEXER)


def _mha_host(layout="layer_first", layers=2, heads=2, head_dim=8):
    device = MHATokenToKVPool(
        size=PAGE * PAGES,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=heads,
        head_dim=head_dim,
        layer_num=layers,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    return MHATokenToKVPoolHost(
        buffer_info=device.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE,
        layout=layout,
        pin_memory=False,
        device="cpu",
        allocator_type="default",
        device_capacity_tokens=device.size,
        start_layer=0,
        end_layer=layers,
    )


def _paged_host(name):
    return DeepSeekV4PagedHostPool(
        pool_name=str(name),
        device_buffers=[
            torch.empty(1, PAGE, SIDECAR_BYTES // PAGE, dtype=torch.uint8)
            for _ in range(2)
        ],
        item_bytes=SIDECAR_BYTES,
        num_host_pages=2 * PAGES,
        slot_page_size=PAGE,
        layout="layer_first",
        pin_memory=False,
        page_aligned_only=True,
    )


def _page_bytes(pool, offset):
    flat = pool.get_data_page(offset, flat=True)
    return flat.view(torch.uint8).clone().flatten()


class _StoreCase(CustomTestCase):
    """KV pool plus three NVFP4-style sidecar pools over a temp storage dir."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="hicache-faults-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.keys = [f"page-{i}" for i in range(PAGES)]

    def _pools(self):
        pools = {PoolName.KV: _mha_host()}
        for name in SIDECAR_NAMES:
            pools[name] = _paged_host(name)
        return pools

    def _storage(self, pools, model_name=NVFP4_TAG_V1, zero_copy=False, extra=None):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=model_name,
            extra_config=extra,
        )
        storage = HiCacheFile(config, file_path=str(self.dir))
        storage.register_mem_pool_host(pools[PoolName.KV])
        for name, pool in pools.items():
            storage.register_mem_host_pool_v2(pool, name)
        storage.zero_copy_read = zero_copy
        storage.read_workers = 1
        return storage

    def _file(self, storage, key, pool_name=PoolName.KV):
        stem = storage._get_component_key(key, str(pool_name))
        return self.dir / f"{stem}.bin"

    def _transfers(self, pools, idx=None):
        return [
            PoolTransfer(name=n, host_indices=idx, keys=self.keys) for n in pools
        ]

    def _write(self, pools, storage, seed):
        g = torch.Generator().manual_seed(seed)
        kv = pools[PoolName.KV]
        idx = kv.alloc(len(self.keys) * PAGE)
        expected = {}
        for name, pool in pools.items():
            dummy = pool.get_dummy_flat_data_page()
            nbytes = dummy.numel() * dummy.element_size()
            for i in range(len(self.keys)):
                raw = torch.randint(
                    0, 256, (nbytes,), dtype=torch.uint8, generator=g
                )
                pool.set_from_flat_data_page(i * PAGE, raw.view(dummy.dtype))
                expected[(name, i)] = _page_bytes(pool, i * PAGE)
        written = storage.batch_set_v2(self._transfers(pools, idx))
        self.assertTrue(all(all(ok) for ok in written.values()), written)
        kv.free(idx)
        return idx, expected

    def _read(self, pools, storage, idx, zero_copy):
        storage.zero_copy_read = zero_copy
        return storage.batch_get_v2(self._transfers(pools, idx))

    def _clear_pools(self, pools, idx):
        for pool in pools.values():
            for i in range(len(self.keys)):
                dummy = pool.get_dummy_flat_data_page()
                pool.set_from_flat_data_page(
                    int(idx[i * PAGE]), torch.zeros_like(dummy.view(torch.uint8)).view(dummy.dtype)
                )


class TestCorruptBlock(_StoreCase):
    """R1 sentinel: size-gated reads serve same-size corruption as a hit."""

    def _flip_middle_byte(self, path):
        data = bytearray(path.read_bytes())
        self.assertGreater(len(data), 0)
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))

    def test_corrupt_file_content_is_served_unverified(self):
        pools = self._pools()
        storage = self._storage(pools)
        idx, expected = self._write(pools, storage, seed=7)

        victim = self._file(storage, self.keys[2], PoolName.KV)
        scale = self._file(storage, self.keys[2], PoolName.KV_K_SCALE)
        self._flip_middle_byte(victim)
        self._flip_middle_byte(scale)

        gated = storage.batch_exists_v2(self.keys, self._transfers(pools))
        self.assertEqual(gated.kv_hit_pages, PAGES)

        for zero_copy in (False, True):
            with self.subTest(zero_copy=zero_copy):
                self._clear_pools(pools, idx)
                res = self._read(pools, storage, idx, zero_copy)
                self.assertTrue(all(all(r) for r in res.values()), res)
                self.assertFalse(
                    torch.equal(
                        _page_bytes(pools[PoolName.KV], int(idx[2 * PAGE])),
                        expected[(PoolName.KV, 2)],
                    ),
                    "corrupted KV bytes did not reach the host pool",
                )
                self.assertFalse(
                    torch.equal(
                        _page_bytes(pools[PoolName.KV_K_SCALE], int(idx[2 * PAGE])),
                        expected[(PoolName.KV_K_SCALE, 2)],
                    ),
                    "corrupted scale bytes did not reach the host pool",
                )
                self.assertTrue(
                    torch.equal(
                        _page_bytes(pools[PoolName.KV], int(idx[1 * PAGE])),
                        expected[(PoolName.KV, 1)],
                    ),
                    "the untouched neighbour page must round-trip exactly",
                )

    def test_oversized_and_short_files_are_misses_not_corruption(self):
        pools = self._pools()
        storage = self._storage(pools)
        idx, _ = self._write(pools, storage, seed=8)
        kv = self._file(storage, self.keys[1], PoolName.KV)
        kv.write_bytes(kv.read_bytes()[:-1])
        side = self._file(storage, self.keys[3], PoolName.KV_V_SCALE)
        side.write_bytes(side.read_bytes() + b"x")
        for zero_copy in (False, True):
            with self.subTest(zero_copy=zero_copy):
                res = self._read(pools, storage, idx, zero_copy)
                self.assertFalse(res[PoolName.KV][1])
                self.assertFalse(res[PoolName.KV_V_SCALE][3])


class TestHalfWrittenSnapshot(_StoreCase):
    """R4/R12: a missing sidecar sibling gates the page out of the prefix;
    exists() alone (KV component only) cannot see the gap."""

    def test_half_written_page_truncates_restorable_prefix(self):
        pools = self._pools()
        storage = self._storage(pools)
        idx, _ = self._write(pools, storage, seed=11)

        self._file(storage, self.keys[2], PoolName.KV_K_SCALE).unlink()

        # exists() spans only the KV component in the legacy scheme; the
        # sidecar co-presence gate lives in batch_exists_v2 and the fused read.
        self.assertTrue(storage.exists(self.keys[2]))
        self.assertFalse(
            storage._file_exists(
                storage._get_component_key(self.keys[2], str(PoolName.KV_K_SCALE))
            )
        )
        self.assertTrue(storage.exists(self.keys[0]))

        gated = storage.batch_exists_v2(self.keys, self._transfers(pools))
        self.assertEqual(gated.kv_hit_pages, 2)
        self.assertEqual(gated.extra_pool_hit_pages[PoolName.KV], PAGES)
        self.assertEqual(gated.extra_pool_hit_pages[PoolName.KV_K_SCALE], 2)
        self.assertEqual(gated.restorable_prefix_pages, [1, 2])

        res = self._read(pools, storage, idx, zero_copy=False)
        self.assertEqual(
            res[PoolName.KV_K_SCALE], [True, True, False, True, True, True]
        )
        self.assertTrue(all(res[PoolName.KV]))

    def test_orphan_tmp_files_never_hit_or_count(self):
        pools = self._pools()
        storage = self._storage(pools, extra={"max_size": "1G"})
        idx, _ = self._write(pools, storage, seed=13)
        ledger = storage._evictor._total_bytes
        self.assertGreater(ledger, 0)

        orphan = self.dir / ".9e587b5a1c0143c38a56433bcdb900e1.tmp"
        orphan.write_bytes(b"junk" * 64)
        empty = self.dir / ".00000000000000000000000000000000.tmp"
        empty.touch()

        restarted = self._storage(
            pools, extra={"max_size": "1G", "enable_metadata_cache": True}
        )
        self.assertEqual(restarted._evictor._total_bytes, ledger)
        self.assertFalse(restarted._file_exists("9e587b5a1c0143c38a56433bcdb900e1"))
        gated = restarted.batch_exists_v2(self.keys, self._transfers(pools))
        self.assertEqual(gated.kv_hit_pages, PAGES)

        self.assertTrue(restarted.clear())
        self.assertFalse(orphan.exists())
        self.assertFalse(empty.exists())


class TestVersionConflict(_StoreCase):
    """R5: identity lives in the key; an identity change misses, never collides."""

    def test_format_tag_change_misses_not_collides(self):
        pools = self._pools()
        old = self._storage(pools, model_name=NVFP4_TAG_V1)
        idx, expected = self._write(pools, old, seed=21)
        files_before = sorted(p.name for p in self.dir.glob("*.bin"))
        self.assertTrue(files_before)

        fresh = self._storage(pools, model_name=NVFP4_TAG_V2)
        self.assertNotEqual(fresh.config_suffix, old.config_suffix)
        gated = fresh.batch_exists_v2(self.keys, self._transfers(pools))
        self.assertEqual(gated.kv_hit_pages, 0)
        self.assertEqual(gated.extra_pool_hit_pages, {})

        self._write(pools, fresh, seed=22)
        for name in files_before:
            self.assertTrue((self.dir / name).exists())
        self.assertGreater(len(list(self.dir.glob("*.bin"))), len(files_before))

        self._clear_pools(pools, idx)
        res = self._read(pools, old, idx, zero_copy=False)
        self.assertTrue(all(all(r) for r in res.values()), res)
        self.assertTrue(
            torch.equal(
                _page_bytes(pools[PoolName.KV], int(idx[0])), expected[(PoolName.KV, 0)]
            ),
            "the old namespace must still restore its own bytes",
        )

    def test_namespace_digest_perturbations(self):
        base = derive_namespace(
            model_id="m1",
            dtype="float4_e2m1fn_x2",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=4,
            head_group=2,
        )
        d0 = namespace_digest(base)
        self.assertTrue(d0.startswith("ukv1-"))
        self.assertEqual(d0, namespace_digest(base))

        perturbations = {
            "dtype": dict(dtype="uint8"),
            "page_size": dict(page_size=32),
            "object_layout": dict(object_layout="other"),
            "heads": dict(total_kv_heads=8),
            "model_id": dict(model_id="m2"),
            "replication": dict(rank_replicated=True),
        }
        seen = {d0}
        for label, kw in perturbations.items():
            with self.subTest(field=label):
                fields = {
                    "model_id": base.model_id,
                    "dtype": base.dtype,
                    "page_size": base.page_size,
                    "rank_replicated": base.rank_replicated,
                    "total_kv_heads": base.total_kv_heads,
                    "head_group": base.head_group,
                    "object_layout": base.object_layout,
                }
                fields.update(kw)
                d = namespace_digest(KVCacheNamespace(**fields))
                self.assertNotIn(d, seen, f"{label} must change every key")
                seen.add(d)


class TestOrphanTmpGc(_StoreCase):
    """Task 216 live finding: ``.{uuid-hex}.tmp`` orphans from interrupted
    writes leak forever (the LRU scan only tracks .bin). Startup GC must
    reclaim the old ones, spare in-flight writes, and leave foreign files
    alone. One ledger line per removal."""

    OLD_TMP = ".aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp"
    FRESH_TMP = ".bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.tmp"
    DECOY = "notes.tmp"  # not a write-temp name -> must survive

    def _mk_tmp(self, name, age_s, size=256):
        path = self.dir / name
        path.write_bytes(b"x" * size)
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
        return path

    def test_startup_gc_removes_old_orphans_only(self):
        old = self._mk_tmp(self.OLD_TMP, 2 * 3600)
        fresh = self._mk_tmp(self.FRESH_TMP, 5)
        decoy = self._mk_tmp(self.DECOY, 2 * 3600)
        pools = self._pools()
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="INFO"
        ) as logs:
            self._storage(
                pools, extra={"max_size": "1G", "tmp_gc_age_seconds": 3600}
            )
        self.assertFalse(old.exists(), "old orphan .tmp must be swept at startup")
        self.assertTrue(fresh.exists(), "in-flight .tmp must survive the sweep")
        self.assertTrue(decoy.exists(), "non write-temp names must survive")
        joined = "\n".join(logs.output)
        self.assertIn("tmp-GC ledger", joined)
        self.assertIn(self.OLD_TMP, joined)
        self.assertNotIn(self.FRESH_TMP, joined)

    def test_gc_runs_while_evictor_is_inert(self):
        old = self._mk_tmp(self.OLD_TMP, 2 * 3600, size=1024)
        pools = self._pools()
        self._storage(pools)  # no cap, no watermark: eviction disabled
        self.assertFalse(old.exists(), "tmp leak must stop even without eviction")

    def test_gc_threshold_zero_disables_sweep(self):
        old = self._mk_tmp(self.OLD_TMP, 2 * 3600)
        pools = self._pools()
        self._storage(pools, extra={"max_size": "1G", "tmp_gc_age_seconds": 0})
        self.assertTrue(old.exists(), "threshold 0 disables the sweep")

    def test_min_free_watermark_refuses_writes(self):
        pools = self._pools()
        idx = pools[PoolName.KV].alloc(len(self.keys) * PAGE)
        storage = self._storage(
            pools, extra={"max_size": "1G", "min_free_space": "4096T"}
        )
        written = storage.batch_set_v2(self._transfers(pools, idx))
        self.assertTrue(all(not any(ok) for ok in written.values()), written)
        self.assertEqual(storage._evictor._total_bytes, 0)
        self.assertEqual(list(self.dir.glob("*.bin")), [])
        self.assertEqual(
            [p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")],
            [],
            "refused writes must leave no temp files",
        )
        # A normal watermark admits the same pages again.
        ok_storage = self._storage(pools, extra={"max_size": "1G"})
        written = ok_storage.batch_set_v2(self._transfers(pools, idx))
        self.assertTrue(all(all(ok) for ok in written.values()), written)
        kv = pools[PoolName.KV]
        kv.free(idx)


if __name__ == "__main__":
    unittest.main()
