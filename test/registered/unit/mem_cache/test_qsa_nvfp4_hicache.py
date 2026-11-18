"""HiCache host/storage support for QSA NVFP4 KV (packed E2M1 + E4M3 scales)."""

import shutil
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    QSANVFP4KVCacheMethod,
)
from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    MambaStateBufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    prepare_host_pool_config,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import mha as pool_host_mha
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.qsa import QSAKVScalePoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 16
HEADS = 2
HEAD_DIM = 64
LAYERS = 2
SCALE_NAMES = (PoolName.KV_K_SCALE, PoolName.KV_V_SCALE)


def _nvfp4_pool(size=PAGE * 4, layers=LAYERS):
    return MHATokenToKVPool(
        size=size,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=layers,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
        quant_method=QSANVFP4KVCacheMethod(
            num_layers=layers, device="cpu", page_size=PAGE
        ),
    )


def _fp8_pool(size=PAGE * 4, layers=LAYERS):
    return MHATokenToKVPool(
        size=size,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=layers,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def _fake_transfer_kv_direct(src_layers, dst_layers, src_indices, dst_indices, page_size):
    """CPU stand-in for the sgl_kernel direct copy: row gather/scatter per layer."""
    for src, dst in zip(src_layers, dst_layers, strict=True):
        dst[dst_indices.long()] = src[src_indices.long()]


def _memory(layout="layer_first", hicache_size=0):
    return SimpleNamespace(
        hicache_size=hicache_size,
        hicache_ratio=2,
        hicache_mem_layout=layout,
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="cache",
    )


class _HostAssemblyCase(CustomTestCase):
    """CPU-only assembly: pageable host memory, no CUDA registration."""

    def setUp(self):
        self.memory = _memory()
        self.enterContext(
            patch.object(
                hybrid_pool_assembler, "get_memory", side_effect=lambda: self.memory
            )
        )
        self.enterContext(
            patch.object(
                hybrid_pool_assembler,
                "get_parallel",
                return_value=SimpleNamespace(
                    dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
                ),
            )
        )
        self.enterContext(
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            )
        )
        self.enterContext(
            patch.object(
                hybrid_pool_assembler,
                "get_mha_host_pool_cls",
                return_value=partial(MHATokenToKVPoolHost, pin_memory=False),
            )
        )

    def _group(self, pool):
        config = prepare_host_pool_config(
            root=pool,
            full_layer_mapping={i: i for i in range(pool.layer_num)},
            transfer_layer_id_max=pool.layer_num,
            transfer_page_size=PAGE,
        )
        group = hybrid_pool_assembler.build_host_pool_group(config=config)
        return config, group


class TestQSANVFP4DeviceInfos(CustomTestCase):
    def test_packed_kv_and_token_linear_scale_sidecars(self):
        pool = _nvfp4_pool()
        kv, k_scale, v_scale = pool.get_device_pool_infos()
        self.assertEqual(
            (kv.pool_name, k_scale.pool_name, v_scale.pool_name),
            (PoolName.KV, *SCALE_NAMES),
        )
        # Host rows carry packed bytes, not the logical bf16 head width.
        self.assertEqual(kv.buffer_info.head_dim, HEAD_DIM // 2)
        self.assertEqual(kv.buffer_info.dtype, torch.uint8)
        self.assertIsNone(kv.buffer_info.scales)
        for info, buffers in (
            (k_scale, pool.k_scale_buffer),
            (v_scale, pool.v_scale_buffer),
        ):
            self.assertEqual(info.indices_from_pool, PoolName.KV)
            self.assertEqual(info.buffer_info.compress_ratio, 1)
            self.assertEqual(info.buffer_info.page_bytes, PAGE * HEADS * HEAD_DIM // 16)
            self.assertTrue(
                all(a is b for a, b in zip(info.buffer_info.buffers, buffers))
            )
        self.assertEqual(
            pool.get_kv_scale_size_bytes(),
            2 * LAYERS * (pool.size + PAGE) * HEADS * HEAD_DIM // 16,
        )

    def test_plain_mha_host_still_rejects_token_scales(self):
        """Only the sidecar path moves the scales; a bare MHA host stays closed."""
        with self.assertRaisesRegex(ValueError, "KV scale layout 'token'"):
            MHATokenToKVPoolHost(
                buffer_info=_nvfp4_pool()._mha_buffer_info(),
                host_to_device_ratio=2,
                host_size=0,
                page_size=PAGE,
                layout="layer_first",
                pin_memory=False,
                device_capacity_tokens=PAGE * 4,
                start_layer=0,
                end_layer=LAYERS - 1,
            )


class TestQSANVFP4HostAssembly(_HostAssemblyCase):
    def test_group_mirrors_scales_beside_packed_kv(self):
        pool = _nvfp4_pool()
        config, group = self._group(pool)
        self.assertEqual(list(config.pool_names), [PoolName.KV, *SCALE_NAMES])
        kv_host = group.get_pool(PoolName.KV)
        self.assertEqual(kv_host.head_dim, HEAD_DIM // 2)
        self.assertEqual(kv_host.dtype, torch.uint8)
        self.assertEqual(
            kv_host.storage_format_tag, hybrid_pool_assembler.QSA_NVFP4_STORAGE_FORMAT_TAG
        )
        for name in SCALE_NAMES:
            host = group.get_pool(name)
            self.assertIsInstance(host, QSAKVScalePoolHost)
            self.assertEqual(host.num_host_pages, kv_host.page_num)
            self.assertEqual(host.get_size_per_token(), LAYERS * HEADS * HEAD_DIM // 16)

    def test_kv_only_host_path_refuses_scaled_pool(self):
        with self.assertRaisesRegex(ValueError, "NVFP4 KV block scales"):
            hybrid_pool_assembler.build_kv_host_pool(kv_pool=_nvfp4_pool(), page_size=PAGE)

    def test_unsupported_host_layout_fails_before_scale_allocation(self):
        self.memory = _memory(layout="page_head")
        config = prepare_host_pool_config(
            root=_nvfp4_pool(),
            full_layer_mapping={0: 0, 1: 1},
            transfer_layer_id_max=LAYERS,
            transfer_page_size=PAGE,
        )
        root_host = SimpleNamespace(page_size=PAGE, layout="page_head", page_num=4)
        with self.assertRaisesRegex(ValueError, "NVFP4 KV scales support"):
            hybrid_pool_assembler._build_sidecar_host_pool(
                config=config.sidecars[0], root=config.root, root_host_pool=root_host
            )

    def test_draft_sidecar_refuses_nvfp4_scales(self):
        controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(logical_size=PAGE * 8, size=PAGE * 8),
            transfer_layer_id_max=LAYERS,
            page_size=PAGE,
        )
        with self.assertRaisesRegex(NotImplementedError, "NVFP4 KV"):
            hybrid_pool_assembler.build_full_draft_pools(
                draft_kv_pool=_nvfp4_pool(layers=1),
                tree_cache=SimpleNamespace(cache_controller=controller),
            )


class TestQSANVFP4HostSizing(_HostAssemblyCase):
    """A fixed --hicache-size buys more NVFP4 tokens, scales included."""

    def _kv_tokens(self, pool, host_gb):
        self.memory = _memory(hicache_size=1)
        mamba_info = DevicePoolInfo(
            pool_name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            layer_ids=(LAYERS,),
            buffer_info=MambaStateBufferInfo(
                temporal=torch.zeros(1, 9, 4), conv=(torch.zeros(1, 9, 4),)
            ),
        )
        params = SimpleNamespace(
            page_size=PAGE,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
        )
        with (
            patch.object(
                hybrid_pool_assembler, "_split_hicache_size", return_value=(host_gb, 0)
            ),
            patch.object(
                hybrid_pool_assembler,
                "_rebalance_mamba_host_size",
                side_effect=lambda kv, mamba: (kv, mamba),
            ),
            patch.object(hybrid_pool_assembler, "_mamba_entry", MagicMock()),
            patch.object(hybrid_pool_assembler, "HostPoolGroup", lambda entries: entries),
            patch.object(hybrid_pool_assembler, "_build_cache_controller", MagicMock()),
            patch.object(
                hybrid_pool_assembler,
                "build_pool_entry",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            entries, _, _ = hybrid_pool_assembler.build_hybrid_mamba_stack(
                params=params,
                kv_pool=pool,
                mamba_pool=SimpleNamespace(get_device_pool_infos=lambda: (mamba_info,)),
                full_layer_mapping={i: i for i in range(LAYERS)},
                start_layer=0,
                load_cache_event=None,
                storage_backend=None,
            )
        hosts = {e.name: e.host_pool for e in entries if hasattr(e, "host_pool")}
        return hosts

    def test_budget_counts_scale_sidecars(self):
        host_gb = 0.002
        hosts = self._kv_tokens(_nvfp4_pool(), host_gb)
        kv_host = hosts[PoolName.KV]
        scale_bytes = sum(
            hosts[name].layer_num * hosts[name].item_bytes * hosts[name].num_host_pages
            for name in SCALE_NAMES
        )
        payload_per_token = 2 * LAYERS * HEADS * HEAD_DIM // 2
        scale_per_token = 2 * LAYERS * HEADS * HEAD_DIM // 16
        self.assertEqual(kv_host.size_per_token, payload_per_token)
        total = kv_host.size * payload_per_token + scale_bytes
        # Payload plus scales fit the budget; host pools round up by one page.
        per_token = payload_per_token + scale_per_token
        self.assertEqual(scale_bytes, kv_host.size * scale_per_token)
        self.assertLessEqual(total, host_gb * 1e9 + PAGE * per_token)
        self.assertGreater(total, host_gb * 1e9 - PAGE * per_token)
        fp8_host = self._kv_tokens(_fp8_pool(), host_gb)[PoolName.KV]
        # 0.5 B codes + 1/16 B scales per element vs 1 B FP8: 16/9 more tokens.
        self.assertAlmostEqual(kv_host.size / fp8_host.size, 16 / 9, delta=0.05)


class TestQSANVFP4RoundTrip(_HostAssemblyCase):
    """device -> host -> file -> host -> device, bit-exact, on CPU tensors.

    The sgl_kernel direct copy is CUDA-only; a row gather/scatter stands in for it,
    so the host pools' own layer_first/direct code paths run unchanged."""

    def setUp(self):
        super().setUp()
        for module in (memory_pool_host, pool_host_mha):
            self.enterContext(
                patch.object(
                    module,
                    "transfer_kv_direct",
                    _fake_transfer_kv_direct,
                    create=True,
                )
            )
        self.storage_dir = (
            Path(__file__).resolve().parent / f".hicache-nvfp4-{self.id().split('.')[-1]}"
        )
        shutil.rmtree(self.storage_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.storage_dir, ignore_errors=True)

    def _storage(self, group, model_name):
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
        )
        storage = HiCacheFile(config, file_path=str(self.storage_dir))
        storage.register_mem_pool_host(group.get_pool(PoolName.KV))
        for name in (PoolName.KV, *SCALE_NAMES):
            storage.register_mem_host_pool_v2(group.get_pool(name), name)
        return storage

    @staticmethod
    def _buffers(pool):
        return [
            *pool.k_buffer,
            *pool.v_buffer,
            *pool.k_scale_buffer,
            *pool.v_scale_buffer,
        ]

    def test_packed_codes_and_scales_round_trip_through_file(self):
        torch.manual_seed(0)
        src = _nvfp4_pool()
        for buffer in self._buffers(src):
            buffer.copy_(torch.randint(0, 256, buffer.shape, dtype=torch.uint8))
        _, group = self._group(src)
        hosts = {name: group.get_pool(name) for name in (PoolName.KV, *SCALE_NAMES)}
        kv_host = hosts[PoolName.KV]

        device_src = torch.arange(PAGE, 3 * PAGE)  # pages 1..2
        host_a = kv_host.alloc(2 * PAGE)
        for host in hosts.values():
            host.backup_from_device_all_layer(src, host_a, device_src, "direct")

        dtype_tag = storage_model_name(
            "qwen38", kv_host, torch.float4_e2m1fn_x2
        )
        storage = self._storage(group, dtype_tag)
        keys = ["page-a", "page-b"]
        transfers = [
            PoolTransfer(name=name, host_indices=host_a, keys=keys)
            for name in hosts
        ]
        written = storage.batch_set_v2(transfers)
        self.assertTrue(all(all(ok) for ok in written.values()), written)
        files = sorted(p.name for p in self.storage_dir.iterdir())
        self.assertEqual(len(files), 6, files)
        self.assertTrue(any(".kv_k_scale" in f for f in files))
        self.assertTrue(any(".kv_v_scale" in f for f in files))

        # Restore into different host and device slots of a fresh pool.
        dst = _nvfp4_pool()
        _, dst_group = self._group(dst)
        dst_hosts = {name: dst_group.get_pool(name) for name in hosts}
        dst_hosts[PoolName.KV].alloc(PAGE)  # shift host slots by one page
        host_b = dst_hosts[PoolName.KV].alloc(2 * PAGE)
        self.assertFalse(torch.equal(host_a, host_b))
        dst_storage = self._storage(dst_group, dtype_tag)
        exists = dst_storage.batch_exists_v2(
            keys,
            [PoolTransfer(name=name, keys=keys) for name in SCALE_NAMES],
        )
        self.assertEqual(exists.kv_hit_pages, 2)
        read = dst_storage.batch_get_v2(
            [PoolTransfer(name=name, host_indices=host_b, keys=keys) for name in hosts]
        )
        self.assertTrue(all(all(ok) for ok in read.values()), read)

        device_dst = torch.arange(2 * PAGE, 4 * PAGE)  # pages 2..3
        for layer in range(LAYERS):
            for host in dst_hosts.values():
                host.load_to_device_per_layer(dst, host_b, device_dst, layer, "direct")
        for got, want in zip(self._buffers(dst), self._buffers(src), strict=True):
            self.assertTrue(torch.equal(got[device_dst], want[device_src]))
            self.assertEqual(int(got[: 2 * PAGE].count_nonzero()), 0)

        # An FP8 namespace never sees the NVFP4 objects.
        fp8_name = storage_model_name("qwen38", SimpleNamespace(storage_format_tag=None), torch.float8_e4m3fn)
        self.assertNotEqual(fp8_name, dtype_tag)
        fp8_storage = self._storage(dst_group, fp8_name)
        self.assertEqual(fp8_storage.batch_exists_v2(keys, []).kv_hit_pages, 0)


class TestQSANVFP4Namespace(CustomTestCase):
    def test_format_tag_and_dtype_separate_fp8_and_nvfp4(self):
        tag = hybrid_pool_assembler.QSA_NVFP4_STORAGE_FORMAT_TAG
        nvfp4 = storage_model_name(
            "m", SimpleNamespace(storage_format_tag=tag), torch.float4_e2m1fn_x2
        )
        fp8 = storage_model_name(
            "m", SimpleNamespace(storage_format_tag=None), torch.float8_e4m3fn
        )
        self.assertIn(tag, nvfp4)
        self.assertIn("float4_e2m1fn_x2", nvfp4)
        self.assertNotIn("nvfp4", fp8)
        self.assertNotEqual(nvfp4, fp8)


if __name__ == "__main__":
    unittest.main()
