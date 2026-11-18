"""HiCacheFile zero-copy read path: bit-identical to the staged path."""

import ctypes
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MiniMaxSparseKVPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 4
PAGES = 6
SIDECAR_BYTES = 96


def _mha_host(layout, layers=3, heads=2, head_dim=8, dtype=torch.bfloat16):
    device = MHATokenToKVPool(
        size=PAGE * PAGES,
        page_size=PAGE,
        dtype=dtype,
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


def _k_only_host(layout):
    pool = MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        idx_head_dim=3,
        dense_layer_ids=[0, 2],
        sparse_layer_ids=[1, 3],
        disable_value_sparse_layer_ids=[1, 3],
        device="cpu",
        start_layer=0,
        end_layer=4,
    )
    kv_host = MHATokenToKVPoolHost(
        buffer_info=pool.main_pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=pool.page_size,
        layout=layout,
        pin_memory=False,
        device="cpu",
        allocator_type="default",
        device_capacity_tokens=pool.main_pool.size,
        start_layer=0,
        end_layer=4,
    )
    return MHATokenToKOnlyPoolHost(
        buffer_info=pool.index_k_pool.get_device_pool_infos()[0].buffer_info,
        anchor_host=kv_host,
        layout=layout,
        pin_memory=False,
        device="cpu",
        allocator_type="default",
    )


def _paged_host(layout, num_host_pages):
    return DeepSeekV4PagedHostPool(
        pool_name="sidecar",
        device_buffers=[
            torch.empty(1, PAGE, SIDECAR_BYTES // PAGE, dtype=torch.uint8)
            for _ in range(2)
        ],
        item_bytes=SIDECAR_BYTES,
        num_host_pages=num_host_pages,
        slot_page_size=PAGE,
        layout=layout,
        pin_memory=False,
        page_aligned_only=True,
    )


def _host_tensors(pool):
    buf = pool.k_buffer if isinstance(pool, MHATokenToKOnlyPoolHost) else pool.kv_buffer
    return list(buf) if isinstance(buf, (list, tuple)) else [buf]


def _snapshot(pool):
    return [t.view(torch.uint8).clone() for t in _host_tensors(pool)]


def _fill(pool, seed):
    g = torch.Generator().manual_seed(seed)
    for t in _host_tensors(pool):
        raw = t.view(torch.uint8)
        raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8, generator=g))


def _zero(pool):
    for t in _host_tensors(pool):
        t.view(torch.uint8).zero_()


def _write_segments(segments, data: bytes):
    pos = 0
    for addr, nbytes in segments:
        ctypes.memmove(addr, data[pos : pos + nbytes], nbytes)
        pos += nbytes
    assert pos == len(data)


class TestFlatPageSegments(CustomTestCase):
    """Writing a flat page through the segments == set_from_flat_data_page."""

    def _check(self, pool):
        dummy = pool.get_dummy_flat_data_page()
        nbytes = dummy.numel() * dummy.element_size()
        g = torch.Generator().manual_seed(1)
        for index in (0, PAGE, 2 * PAGE):
            data = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)
            _fill(pool, index)
            pool.set_from_flat_data_page(index, data.view(dummy.dtype))
            want = _snapshot(pool)
            _fill(pool, index)
            segments = pool.flat_page_segments(index)
            self.assertIsNotNone(segments)
            self.assertEqual(sum(n for _, n in segments), nbytes)
            _write_segments(segments, data.numpy().tobytes())
            for got, exp in zip(_snapshot(pool), want, strict=True):
                self.assertTrue(torch.equal(got, exp))
        self.assertIsNone(pool.flat_page_segments(1))  # unaligned

    def test_mha_layouts(self):
        for layout in ("page_first", "page_first_direct", "page_head", "layer_first"):
            with self.subTest(layout=layout):
                self._check(_mha_host(layout))

    def test_k_only_layouts(self):
        for layout in ("page_first", "page_first_direct", "layer_first"):
            with self.subTest(layout=layout):
                self._check(_k_only_host(layout))

    def test_paged_sidecar_layouts(self):
        for layout in ("page_first", "page_first_direct", "layer_first"):
            with self.subTest(layout=layout):
                self._check(_paged_host(layout, PAGES))

    def test_subclass_with_own_setter_is_staged(self):
        class Custom(MHATokenToKVPoolHost):
            def set_from_flat_data_page(self, index, data_page):
                super().set_from_flat_data_page(index, data_page)

        pool = _mha_host("page_first")
        pool.__class__ = Custom
        self.assertIsNone(pool.flat_page_segments(0))


class _StoreCase(CustomTestCase):
    def setUp(self):
        self.dir = (
            Path(__file__).resolve().parent / f".hicache-zc-{self.id().split('.')[-1]}"
        )
        shutil.rmtree(self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _pools(self, layout):
        kv = _mha_host(layout)
        side = _paged_host(layout, kv.size // PAGE)
        return {PoolName.KV: kv, PoolName.INDEXER: side}

    def _storage(self, pools, zero_copy, workers=4):
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
            model_name="zc-model",
        )
        storage = HiCacheFile(config, file_path=str(self.dir))
        storage.register_mem_pool_host(pools[PoolName.KV])
        for name, pool in pools.items():
            storage.register_mem_host_pool_v2(pool, name)
        storage.zero_copy_read = zero_copy
        storage.read_workers = workers
        return storage

    def _write(self, layout, keys):
        pools = self._pools(layout)
        for seed, pool in enumerate(pools.values()):
            _fill(pool, 10 + seed)
        idx = pools[PoolName.KV].alloc(len(keys) * PAGE)
        storage = self._storage(pools, zero_copy=False)
        written = storage.batch_set_v2(
            [PoolTransfer(name=n, host_indices=idx, keys=keys) for n in pools]
        )
        self.assertTrue(all(all(ok) for ok in written.values()), written)
        return storage

    def _file(self, storage, key, suffix=""):
        return self.dir / f"{key}{suffix}{storage.config_suffix}.bin"

    def _read(self, layout, keys, zero_copy, workers=4):
        pools = self._pools(layout)
        for pool in pools.values():
            _zero(pool)
        kv = pools[PoolName.KV]
        kv.alloc(PAGE)  # read into shifted slots
        idx = kv.alloc(len(keys) * PAGE)
        storage = self._storage(pools, zero_copy, workers)
        if zero_copy:
            for pool in pools.values():
                self.assertIsNotNone(storage._zero_copy_segments(pool, [PAGE]))
        results = storage.batch_get_v2(
            [PoolTransfer(name=n, host_indices=idx, keys=keys) for n in pools]
        )
        return results, {n: _snapshot(p) for n, p in pools.items()}

    def _assert_same(self, a, b):
        for name in a:
            for x, y in zip(a[name], b[name], strict=True):
                self.assertTrue(torch.equal(x, y), name)


class TestZeroCopyBatchGet(_StoreCase):
    def test_bit_identical_to_staged_path(self):
        keys = [f"page-{i}" for i in range(PAGES)]
        for layout in ("page_first", "layer_first", "page_first_direct"):
            with self.subTest(layout=layout):
                shutil.rmtree(self.dir, ignore_errors=True)
                self._write(layout, keys)
                old_res, old = self._read(layout, keys, zero_copy=False)
                for workers in (1, 4):
                    new_res, new = self._read(layout, keys, True, workers)
                    self.assertEqual(new_res, old_res)
                    self.assertTrue(all(all(ok) for ok in new_res.values()))
                    self._assert_same(new, old)

    def test_bad_files_fail_only_their_page(self):
        keys = [f"page-{i}" for i in range(PAGES)]
        storage = self._write("page_first", keys)
        self._file(storage, keys[1]).unlink()
        side = self._file(storage, keys[2], ".indexer")
        side.write_bytes(side.read_bytes()[:-1])
        big = self._file(storage, keys[3])
        big.write_bytes(big.read_bytes() + b"x")
        old_res, _ = self._read("page_first", keys, zero_copy=False)
        new_res, _ = self._read("page_first", keys, zero_copy=True)
        self.assertEqual(new_res[PoolName.KV], [True, False, True, False, True, True])
        self.assertEqual(
            new_res[PoolName.INDEXER], [True, True, False, True, True, True]
        )
        self.assertEqual(new_res[PoolName.KV][:3], old_res[PoolName.KV][:3])
        self.assertEqual(new_res[PoolName.INDEXER], old_res[PoolName.INDEXER])


class TestFusedControllerGet(_StoreCase):
    def _controller(self, storage, kv):
        cc = object.__new__(HiCacheController)
        cc.storage_backend = storage
        cc.storage_host_pool = kv
        cc.mem_pool_host = kv
        cc.page_size = PAGE
        cc.storage_backend_type = "file"
        cc.page_get_func = cc._generic_page_get
        return cc

    def _hits(self, keys, zero_copy):
        pools = self._pools("page_first")
        kv = pools[PoolName.KV]
        idx = kv.alloc(len(keys) * PAGE)
        storage = self._storage(pools, zero_copy)
        cc = self._controller(storage, kv)
        self.assertEqual(cc._fused_kv_get_enabled(), zero_copy)
        op = SimpleNamespace(request_id="t", is_terminated=lambda: False)
        hits = cc._page_transfer_kv_batch(
            op,
            keys,
            idx,
            HiCacheStorageExtraInfo(),
            [PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)],
        )
        return hits, {n: _snapshot(p) for n, p in pools.items()}

    def test_fused_matches_generic(self):
        keys = [f"page-{i}" for i in range(PAGES)]
        storage = self._write("page_first", keys)
        old_hits, old = self._hits(keys, zero_copy=False)
        new_hits, new = self._hits(keys, zero_copy=True)
        self.assertEqual((old_hits, new_hits), (PAGES, PAGES))
        self._assert_same(new, old)

        self._file(storage, keys[2], ".indexer").unlink()
        self.assertEqual(self._hits(keys, False)[0], 2)
        self.assertEqual(self._hits(keys, True)[0], 2)
        self._file(storage, keys[1]).unlink()
        self.assertEqual(self._hits(keys, False)[0], 1)
        self.assertEqual(self._hits(keys, True)[0], 1)


if __name__ == "__main__":
    unittest.main()
