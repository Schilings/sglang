import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(BaseSWAKVPool):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        #
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        kwargs["page_size"] = page_size
        kwargs["enable_memory_saver"] = False # 不支持kv offload？
        kwargs["head_num"] = head_num
        kwargs["head_dim"] = head_dim
        kwargs["device"] = device
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )
        # 对外伪装成一个 KVCache，对内路由到两个子池。
        # 上层代码（attention kernel）只需要调 get_key_buffer(layer_id) 和 set_kv_buffer(layer, loc, k, v)，不需要知道 full/SWA 的存在。
        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        kwargs.pop("swa_head_num", None)
        kwargs.pop("swa_head_dim", None)
        kwargs.pop("swa_v_head_dim", None)
        self.full_kv_pool = token_to_kv_pool_class(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **kwargs,
        )
        # {layer_id: (index, is_swa_layer)}
        # # 例: 模型有 32 层, layer 0-15 是 full, layer 16-31 是 SWA
        # layers_mapping = {
        #     0:  (0, False),    # global layer 0 → full pool 的第 0 层
        #     1:  (1, False),    # global layer 1 → full pool 的第 1 层
        #     ...
        #     16: (0, True),     # global layer 16 → swa pool 的第 0 层
        #     17: (1, True),     # global layer 17 → swa pool 的第 1 层
        #     ...
        # }
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)

        """
        mapping 中的三个值
        mapping 值	    含义      	         场景
             0	     没有 SWA KV	        token 在 window 外，SWA 已被 free_swa() 回收
            >0	     对应的 swa slot	    正常映射
            -1	      末尾哨兵	        alloc_decode 中 last_loc=-1 需要映射到 swa last_loc=-1
        """
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool mem usage: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def register_layer_transfer_counter(self, layer_transfer_counter):
        # Wait happens at this wrapper. Inner pools must not wait again.
        self.layer_transfer_counter = layer_transfer_counter
        self.full_kv_pool.register_layer_transfer_counter(None)
        self.swa_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_state_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        """Q: 为什么 full pool 的 slot 需要翻译成 swa slot？
           A: req_to_token 存的是 full pool 的 slot 索引:
              req_to_token[req_idx] = [3, 7, 12, 25, 30, ...]
              Full 层直接用: full_kv_pool[slot]
              SWA 层需要查表: swa_slot = full_to_swa_index_mapping[slot]
                               swa_kv_pool[swa_slot]

           Q: 为什么用 (data_ptr(), numel()) 做缓存 key 而不是直接判断 tensor 相等？
           A: data_ptr()（非 untyped_storage().data_ptr()）包含 tensor 内部 offset，
              同一块 storage 的不同 view 会得到不同 key。
              同时判断 numel() 防止同一个地址重新分配了不同大小的 tensor。

           Q: 缓存的工作流程是怎样的？
           A: Forward pass 逐层执行:
                Layer 16 (第一个 SWA 层): key 未命中 → mapping[kv_indices] 查表 → 缓存
                Layer 17-31 (其余 SWA 层): 同一 loc tensor → key 命中 → 直接返回缓存
              Allocator 分配/释放后:
                invalidate_loc_cache() → 清缓存 → 下一轮 forward 重新查表

           Q: mapping 的值分别代表什么？
           A: 0  = 没有 SWA KV（token 已滑出窗口，被 free_swa() 回收）
              >0 = 对应的 swa slot（正常映射）
              -1 = 末尾哨兵（last_loc=-1 需要映射到 swa last_loc=-1）
        """
        assert self.full_to_swa_index_mapping is not None
        # -1 in kv_indices maps to -1 via the sentinel appended to the mapping.
        return self.full_to_swa_index_mapping[kv_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        # loc_info bundles the full loc and the pre-translated SWA loc.
        loc, swa_loc = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            # swa_loc is the full->SWA translation, computed once per forward by
            # the attention backend; set_kv_buffer never translates internally.
            assert swa_loc is not None
            self.swa_kv_pool.set_kv_buffer(
                None,
                swa_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def _filter_swa_cpu_copy(self, swa_kv_cpu, row_mask: torch.Tensor):
        if swa_kv_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            return swa_kv_cpu

        chunk_size = getattr(
            self.swa_kv_pool, "cpu_offloading_chunk_size", len(row_mask)
        )
        filtered = []
        for layer_chunks in swa_kv_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue

            k_cpu = torch.cat([chunk[0] for chunk in layer_chunks], dim=0)
            v_cpu = torch.cat([chunk[1] for chunk in layer_chunks], dim=0)
            k_cpu = k_cpu[row_mask]
            v_cpu = v_cpu[row_mask]

            filtered_layer = []
            for i in range(0, len(k_cpu), chunk_size):
                filtered_layer.append(
                    [k_cpu[i : i + chunk_size], v_cpu[i : i + chunk_size]]
                )
            filtered.append(filtered_layer)
        return filtered

    def get_cpu_copy(self, indices, mamba_indices=None):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            # Slot 0 is reserved as a dummy slot. Tail-only SWA allocations leave
            # the out-of-window full KV indices unmapped, so only copy mapped SWA
            # tokens and keep their positions for load_cpu_copy().
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
                swa_mask = swa_mask.cpu()
            else:
                swa_kv_cpu = None
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # Load KV cache back from CPU to both full and SWA pools
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # Load full KV cache to the new indices
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)

        # Load SWA KV cache if it exists
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            new_swa_mask = swa_indices > 0
            old_swa_mask = kv_cache_cpu.get("swa_mask")
            if old_swa_mask is not None:
                old_swa_mask = old_swa_mask.to(indices.device)
                row_mask = new_swa_mask[old_swa_mask].cpu()
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
            else:
                row_mask = new_swa_mask.cpu()
                swa_indices = swa_indices[new_swa_mask]

            if swa_indices.numel() == 0:
                return

            swa_kv_cpu = self._filter_swa_cpu_copy(swa_kv_cpu, row_mask)
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)
