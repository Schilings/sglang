"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Memory pool.

╔══════════════════════════════════════════════════════════════════════════════════════╗
║  💾 GPU KV Cache 内存池 —— 两级池设计                                                ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  📋 ReqToTokenPool          → 映射 request → token 在 KV pool 中的位置 (slot 索引)     ║
║  🗂️ TokenToKVPoolAllocator  → 管理 KV pool slot 的分配/释放                           ║
║  💾 KVCache                  → 实际存储 KV cache 的物理 tensor                         ║
║                                                                                      ║
║  └─ MHATokenToKVPool    标准 MHA (LLaMA/Qwen)                                       ║
║  └─ MLATokenToKVPool    Multi-Head Latent Attention (DeepSeek-V2/V3)                ║
║  └─ DSATokenToKVPool    Dual-Stream Attention 混合架构                                ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.triton_ops.cache_move import (
    copy_all_layer_kv_cache_tiled,
    set_kv_buffer_prefix_valid_tiled,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
# `SGLANG_AITER_KV_CACHE_LAYOUT` is only meaningful on the ROCm AITER backend
# (HIP + --enable-aiter / SGLANG_USE_AITER=1). On any other platform / backend
# the SHUFFLE 5D pool layout has no consumer kernels, so the env var is
# silently ignored and the legacy NHD layout is used.
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
            size_limit=size_limit,
        )
# SWA和MHA可能用的不同的head_num、head_dim、v_head_dim

    if _is_cpu and _cpu_has_amx_support:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
    )


class ReqToTokenPool:
    """📋 Request → Token Slot 映射池 —— 将每个 request 映射到其 KV cache token 位置。

    size:            最大并发 request 数
    max_context_len: 每个 request 的最大 context 长度
    req_to_token:    形状 [size, max_context_len] 的 slot 索引张量
    """

    enable_mamba_extra_buffer_lazy: bool = False

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        # 使用memory_saver管理kv_cache，支持快速offload和onload
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # 主要存储结构：[请求数量, 最大上下文长度]
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        # 可用槽位列表，一个请求一个槽
        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        # 那么req_to_token[ req1, 0:3 ] = [7, 67, 131]
        # 例如req1的前3个token在kv cache中的位置是7，67,131
        # 各自req对应的req_to_token_pool的行，记录的是每个token的kv index值
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].inflight_middle_chunks > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        # 使用memory_saver管理kv_cache，支持快速offload和onload
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                # 这是什么用法？
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )

                conv_state = _init_npu_conv_state(
                    conv_state[0], conv_state_shape, speculative_num_draft_tokens
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on forward stream."""
        need_size = len(indices)
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, indices] = z

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                :, src_indices
            ]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        current_platform.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        self.device = device
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.mamba_allocator.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_allocator.available_size()=}, {len(reqs)=}"
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    self._alloc_ping_pong_buffer(req)
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:
        """Return the ping-pong index holding the most recent tracked state.

        In lazy mode the valid state stays at next_track_idx (no eager swap).
        In normal mode it is at the "other" index (swapped after each track).
        """
        if self.enable_mamba_extra_buffer_lazy:
            return req.mamba_next_track_idx
        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)

    def _alloc_ping_pong_buffer(self, req: Req):
        """Allocate the ping-pong track buffer for a new request.

        Lazy mode allocates 1 slot with the second set to -1 (allocated
        on demand at track boundaries). Normal mode allocates all slots upfront.
        """
        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self.mamba_allocator.alloc(n)
        assert slots is not None, (
            "Not enough space for mamba ping pong idx, "
            "try to increase --mamba-full-memory-ratio."
        )
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.mamba_ping_pong_track_buffer = buf
        req.mamba_next_track_idx = 0

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value):
        """Update a ping-pong slot value and sync the device-side mapping.

        The req holds the authoritative buffer; this keeps the
        req_index_to_mamba_ping_pong_track_buffer_mapping in sync so that
        set_mamba_track_indices_from_reqs reads correct slot indices.
        """
        req.mamba_ping_pong_track_buffer[idx] = value
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = (
            req.mamba_ping_pong_track_buffer
        )

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Donate the tracked-state ping-pong slot to the radix cache.

        Returns the old slot index (shape [1]) for cache insertion and
        replaces it with new_slot so the request can continue tracking.
        In lazy mode the valid state is at next_track_idx; in normal mode
        it is at the "other" index.
        """
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        assert mamba_value_donated.item() != -1, (
            f"Donated mamba slot is -1: donate_idx={donate_idx}, "
            f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
            f"next_track_idx={req.mamba_next_track_idx}, "
            f"rid={req.rid}"
        )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return mamba_value_donated

    def free_mamba_cache(
        self, req: Req, mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_allocator.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            if self.enable_mamba_extra_buffer_lazy:
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[
                        mamba_ping_pong_track_buffer_to_free != -1
                    ]
                )
            self.mamba_allocator.free(mamba_ping_pong_track_buffer_to_free)
            # Match the req.mamba_pool_idx=None clear above so the next
            # alloc() doesn't see a stale ping-pong reference on the req
            # and skip allocation (which would silently reuse a freed
            # tensor on the req side while the new pool slot leaks).
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_allocator.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    ``loc`` is the full-pool write location; ``swa_loc`` is the pre-translated
    full->SWA location for hybrid SWA pools (``None`` otherwise). Bundling them
    lets a backend issue one ``set_kv_buffer`` call regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc)`` from a ``KVWriteLoc`` or a bare loc tensor."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc
    return loc_info, None


class KVCache(abc.ABC):
    """💾 KV Cache 抽象基类 —— 定义 get/set key/value buffer 的统一接口。

    👆 具体实现见 MHATokenToKVPool / MLATokenToKVPool / DSATokenToKVPool。
    """
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):
    """💾 MHA KV Cache Pool —— 标准 Multi-Head Attention 模型的 GPU KV 缓存。

    👆 字段和架构见文件顶部模块注释。
    """
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        # Optional SHUFFLE 5D ("vectorized") physical layout for K/V.
        # Selected by `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d` on the ROCm
        # AITER backend (HIP + SGLANG_USE_AITER=1). When active:
        #   K shape: (num_blocks, H, D_k // X, page, X)
        #   V shape: (num_blocks, H, page // X, D_v, X)   where X = 16 / dtype_bytes
        # aiter `mha_batch_prefill_func` consumes these 5D shapes natively and
        # aiter `pa_decode_gluon` reads SHUFFLE blocks directly during decode.
        # An explicit `kv_cache_layout=` argument always wins (e.g. SWAKVPool
        # passes "nhd" to keep its SWA sub-pool on the legacy layout); on
        # non-AITER platforms the env var is ignored and NHD is forced since
        # no consumer kernel exists for SHUFFLE 5D outside the AITER backend.
        self.kv_cache_layout = "nhd"
        if _use_aiter:
            layout = envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower()
            if layout not in ("nhd", "vectorized_5d"):
                raise ValueError(
                    f"Unsupported SGLANG_AITER_KV_CACHE_LAYOUT={layout!r}; "
                    "expected 'nhd' or 'vectorized_5d'."
                )
            self.kv_cache_layout = layout
            if layout == "vectorized_5d":
                # X is the inner vectorization width in the SHUFFLE layout,
                # determined by the STORAGE dtype (not the compute dtype) since
                # it controls how many elements fit in 16 bytes of the on-pool
                # tensor. For fp8 storage X=16, for bf16/fp16 X=8.
                self._kv_vector_x = 16 // self.store_dtype.itemsize
                assert (self.size + self.page_size) % self.page_size == 0
                assert self.page_size % self._kv_vector_x == 0, (
                    f"page_size={self.page_size} must be divisible by "
                    f"X={self._kv_vector_x} for vectorized_5d layout"
                )
                assert self.head_dim % self._kv_vector_x == 0
                assert self.v_head_dim % self._kv_vector_x == 0

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        # 如果允许kv cache offload到cpu，则kv cache数据会保存在cpu上
        # 因为memory_saver的卸载方式只是单纯把kv cache的数据全部清空，kv cache后续为空
        # 是否允许kv offload到cpu
        if enable_kv_cache_copy:
            # 给kernel warmup一下
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            self._kv_copy_config = None
            return

        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        # bytes_per_tile 应该表示 按 byte 分块，每一块的byte大小
        # stride_bytes 应该等于 self.head_num * self.head_dim
        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        # 这个是？
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            # 按 byte 分块，每一块的byte大小
            "bytes_per_tile": bytes_per_tile,
            # 按 byte 分块，一层k或一层v分成多少块
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        # 弄下假数据用来warmup
        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        # ( k的层数+v的层数, 分块数)
        copy_all_layer_kv_cache_tiled[grid](
            # k，v buffer的指针
            self.data_ptrs,
            # k，v buffer维度0的步长
            self.data_strides,
            # 复制到目标位置的indices
            # 哪些indices
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        # 使用memory_saver管理kv_cache，支持快速offload和onload
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.kv_cache_layout == "vectorized_5d":
                    total_slots = self.size + self.page_size
                    num_blocks = total_slots // self.page_size
                    x = self._kv_vector_x
                    # K: (num_blocks, H, D_k // X, page, X)
                    self.k_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.head_dim // x,
                                self.page_size,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # V: (num_blocks, H, page // X, D_v, X)
                    self.v_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.page_size // x,
                                self.v_head_dim,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    # [size, head_num, head_dim] for each layer
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (
                                self.size + self.page_size,
                                self.head_num,
                                self.v_head_dim,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        # 获取每一层的buffer的存储地址，然后一起存放到tensor
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        # 简单说就是预计算每个缓存张量的内存步长，用于优化KV缓存的传输和复制操作。
        # 计算每个张量形状除第一维外其余维度的乘积，再乘以数据类型大小
        self.data_strides = torch.tensor( 
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        # 遍历每一层
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            # 分块加载这些位置的数据
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        # 获取某一层的
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        # Catch stale slot ids here instead of as illegal-addr / silent KV
        # corruption in the store_kvcache write (gated on SGLANG_ENABLE_ASYNC_ASSERT).
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.kv_cache_layout == "vectorized_5d":
            # Late-import to keep the NHD path import-clean.
            from sglang.srt.layers.attention.utils import (
                launch_reshape_and_cache_shuffle_5d,
            )

            # The writer kernel uses key.stride(0) directly as the source
            # token stride; head/dim are assumed contiguous within each
            # token (stride(1)=head_size, stride(2)=1). Both hold for K/V
            # produced by QKV split + RoPE in upstream attention even when
            # the outer per-token stride is non-canonical, so we skip the
            # protective .contiguous() copies that would otherwise fire
            # large per-layer elementwise kernels.
            launch_reshape_and_cache_shuffle_5d(
                cache_k,
                cache_v,
                self.k_buffer[layer_id - self.start_layer],
                self.v_buffer[layer_id - self.start_layer],
                loc,
            )
            return

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            # size + page_size = real slots + the reserved padding slot (padded /
            # dummy tokens write there); valid index range is [0, size + page_size).
            size_limit=self.size + self.page_size,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )

        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} {tuple(loc_2d.shape)=}."
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)
        else:
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()

        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != self.k_buffer[0].device:
            commit_lens = commit_lens.to(
                device=self.k_buffer[0].device, non_blocking=True
            )
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)
        if commit_lens.dtype != torch.int32:
            commit_lens = commit_lens.to(torch.int32)

        if not (_is_cuda or _is_hip):
            row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            valid_mask = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
            valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                return
            self.set_kv_buffer(
                layer,
                loc_2d.reshape(-1).index_select(0, valid_idx),
                cache_k.index_select(0, valid_idx),
                cache_v.index_select(0, valid_idx),
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
            return

        _set_kv_buffer_prefix_valid_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc_2d,
            commit_lens,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            return

        # Catch stale indices here instead of as illegal-addr or silent KV corruption.
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # no-op; embedding mode has no KV cache to move
        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_v_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-FP4)")
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:
            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
                enable_kv_cache_copy=enable_kv_cache_copy,
            )
        else:
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    """💾 MLA KV Cache Pool —— Multi-Head Latent Attention 模型 (DeepSeek-V2/V3) 的 GPU KV 缓存。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 MLA 与 MHA 的本质区别：压缩 KV，单一缓冲                                         ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  MHA：每层存完整的 K [size, head_num, head_dim] + V [size, head_num, head_dim]     ║
    ║       → 两个独立 buffer，显存 ∝ 2 * head_num * head_dim                            ║
    ║                                                                                  ║
    ║  MLA：每层只存一份"低秩潜变量" latent KV，K 与 V 共享同一块压缩表示                  ║
    ║       kv_buffer[layer] : [size + page_size, 1, kv_lora_rank + qk_rope_head_dim]    ║
    ║         ├─ kv_lora_rank      (如 512)：压缩后的 c_kv，即 nope 部分                 ║
    ║         └─ qk_rope_head_dim  (如  64)：解耦出来的 RoPE 位置编码，即 rope 部分       ║
    ║       → 只有 1 个 buffer，head 维恒为 1，显存约为 MHA 的 1/10                       ║
    ║                                                                                  ║
    ║  ⚠️ 关键：get_value_buffer 只是 get_key_buffer 的前 kv_lora_rank 列切片，          ║
    ║     K/V 在物理上是同一块内存；attention 算子内部再做上投影把 latent 还原成完整 KV。 ║
    ║  ⚠️ +page_size 的"哨兵尾巴"：slot 0 区域用于承接 padding token 的 dummy 写入，      ║
    ║     使越界/补齐写入不污染真实数据（见 _create_buffers 注释）。                       ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链：谁创建我 / 谁读写我 / 谁搬运我                                           ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  ════════ ① 创建链（启动期，仅一次）════════                                         ║
    ║  ModelRunner.__init__                                                             ║
    ║    └─ ModelRunnerKVCacheMixin → MLATokenToKVPool(size, kv_lora_rank, ...)  本类    ║
    ║       └─ PagedTokenToKVPoolAllocator(kvcache=self)   ← allocator 持有本 pool       ║
    ║          └─ RadixCache / HiRadixCache(params)        ← 前缀缓存树持有 allocator    ║
    ║             └─ HiRadixCache 额外建 MLATokenToKVPoolHost(self)  ← Host 三级镜像池   ║
    ║       └─ AttentionBackend(model_runner)  ← backend 捕获 model_runner.token_to_kv_pool║
    ║                                                                                  ║
    ║  ════════ ② 调度 + 分配链（Scheduler 每轮，只给"位置"不写数据）════════               ║
    ║  Scheduler.get_next_batch_to_run                                                  ║
    ║    └─ RadixCache.match_prefix(key)   命中前缀 → 复用已有 slot（不重算不重写）        ║
    ║    └─ allocator.alloc(need_size)     未命中部分 → 分配新的 slot 索引               ║
    ║       └─ 写入 ReqToTokenPool.req_to_token[req, pos] = slot  ← 建立 req→slot 映射    ║
    ║    （此刻本 pool 对应 slot 仍是空的，要等 ③ forward 才真正写入 KV）                  ║
    ║                                                                                  ║
    ║  ════════ ③ 前向写 KV 链（每层 forward，set / ✍️）════════                          ║
    ║  model forward → RadixAttention(layer)                                            ║
    ║    └─ AttentionBackend.forward_extend / forward_decode                            ║
    ║       └─ set_mla_kv_buffer(layer, out_cache_loc, k_nope, k_rope)  ← 本类✍️         ║
    ║          调用方：flashinfer_mla / trtllm_mla / flashattention / dsa_backend / ...  ║
    ║       └─ (或) DeepseekV2AttentionMLA → get_token_to_kv_pool().set_mla_kv_buffer    ║
    ║                                                                                  ║
    ║  ════════ ④ 前向读 KV 链（每层 forward，get / 📖）════════                          ║
    ║  AttentionBackend.forward_*                                                       ║
    ║    └─ get_key_buffer(layer_id)      ← 拿整层 latent buffer（本类📖）               ║
    ║       配合 forward_batch.kv_indices（源自 ReqToTokenPool）做 paged gather          ║
    ║    └─ get_value_buffer(layer_id)    ← 同一 buffer 的前 kv_lora_rank 列切片          ║
    ║    └─ get_mla_kv_buffer(layer, loc) ← 取出并拆成 (k_nope, k_rope)，供 absorb 计算   ║
    ║                                                                                  ║
    ║  ════════ ⑤ 投机解码搬运链（spec decoding，move / 🚚）════════                       ║
    ║  EagleWorker 验证 draft tokens 通过后                                              ║
    ║    └─ spec_utils.move_accept_tokens_to_target_kvcache                             ║
    ║       └─ allocator.get_kvcache().move_kv_cache(tgt_loc, src_loc)  ← 本类🚚         ║
    ║    └─ base_spec_worker.duplicate_prefix_tail_to_draft_branches（topk>1 复制尾页）  ║
    ║                                                                                  ║
    ║  ════════ ⑥ PD 分离传输链（disaggregation，🌐）════════                             ║
    ║  Prefill / Decode BootstrapManager._init_kv_manager                               ║
    ║    └─ get_contiguous_buf_infos()  ← 暴露 (ptr, len, item) 给 RDMA/NVLink 引擎注册🌐║
    ║       MLA 只有一个 buffer，故只返回一份信息（区别于 MHA 的 K/V 两份）               ║
    ║                                                                                  ║
    ║  ════════ ⑦ CPU offload 链（KV 换出 / 换入，💿）════════                            ║
    ║  Scheduler → Req.offload_kv_cache / load_kv_cache                                 ║
    ║    └─ allocator.get_cpu_copy / load_cpu_copy                                      ║
    ║       └─ 本类 get_cpu_copy(GPU→CPU) / load_cpu_copy(CPU→GPU)  按 chunk 分块搬运💿  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        # latent KV 的两段维度：nope(压缩 c_kv) + rope(解耦位置编码)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        # use_dsa: DeepSeek Sparse Attention（DSA）模型走的特殊分支，
        # 其 indexer KV 由子类/外部稍后分配，故此处推迟 _finalize_allocation_log。
        self.use_dsa = use_dsa
        # DSA 专属：当以 fp8 直接存放 KV 且外部给定了 override 维度时，
        # 走"fp8 字节布局"的写入/读取路径（见 set_mla_kv_buffer 的 dsa 分支）。
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with dsa model, we assume the
        # override kv cache dim is correct and use it directly.
        # 单 token 在 buffer 最后一维的宽度：常规 MLA = nope + rope；
        # DSA-fp8 模式信任外部传入的 override 维度（含量化 scale 等额外字节）。
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        # 真正在 GPU 上申请 self.kv_buffer（每层一个 tensor）。
        self._create_buffers()

        # data_ptrs：每层 buffer 的裸指针，供 triton/cuda 内核按层定位（避免每次 .data_ptr()）。
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            # 常规 MLA：buffer 已就绪，立即统计显存占用并打印 "KV Cache is allocated"。
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        # memory_saver_adapter.region：把这块显存登记到 torch_memory_saver，
        # 支持运行期"挂起/释放"KV 显存（如多模型分时复用 GPU）。
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                # custom_mem_pool：PD 分离 + NVLink 场景下用专用内存池分配，
                # 使 buffer 落在可被 RDMA/NVLink 直接访问的地址空间（配合 ⑥ 传输链）。
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                # 形状 [size + page_size, 1, kv_cache_dim]：
                #   - size+page_size：真实容量 + 一页"哨兵尾巴"，承接 padding token 的 dummy 写入；
                #   - 中间 1：MLA 的 head 维恒为 1（K/V 已压缩进 latent，不再按头展开）；
                #   - kv_cache_dim：nope + rope（或 DSA-fp8 的 override 宽度）。
                # 每层一个 tensor，共 layer_num 份；store_dtype 在 fp8 时实际为 uint8。
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        # 释放 KV 显存（memory saver 挂起/销毁池时调用）。
        del self.kv_buffer

    def get_kv_size_bytes(self):
        # 汇总所有层 buffer 的字节数，供 _finalize_allocation_log 计算显存占用。
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # ⑥ PD 分离传输链：把每层 buffer 的"地址/总长/单页长"暴露给 KV 传输引擎注册。
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        # 与 MHA 不同——MLA 只有一份合并 buffer，故每项只返回一份（无需 K、V 各一份）。
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        # kv_item_lens：一"页"KV 的字节数 = 单 token 字节 × page_size（传输按页对齐）。
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        # ④ 读 KV 链入口：attention backend 拿"整层 latent buffer"，再用 kv_indices 做 paged gather。
        # 对 MLA 而言 key buffer = 完整 latent（nope + rope），value 只是它的前缀切片。
        if self.layer_transfer_counter is not None:
            # 分层流水：HiCache/PP 逐层异步搬运时，阻塞等待本层 KV 就绪再返回（避免读到半成品）。
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        # store_dtype != dtype：fp8 实际以 uint8 存储，对外按真实 dtype 重新解释（零拷贝 view）。
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        # layer_id - start_layer：把全局层号换算成本 rank（PP 切分后）持有的本地索引。
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        # MLA 的 "value" 并非独立张量，而是同一 buffer 的前 kv_lora_rank 列（即 nope/c_kv 部分）；
        # rope 部分只参与 query-key 的位置打分，不进入 value 上投影，故此处切掉。
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        # 一次性返回 (key, value)，供同时需要两者的 backend 使用（二者底层指向同一块内存）。
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # ③ 写 KV 链（合并写）：cache_k 已是拼好的完整 latent（nope+rope），直接整块写入。
        # loc 即 forward_batch.out_cache_loc —— ② 阶段 allocator 分配、写进 ReqToTokenPool 的 slot。
        loc, _ = unwrap_write_loc(loc_info)
        # 越界探针：loc 必须落在 [0, size+page_size)，防止脏索引污染 buffer（含哨兵尾巴）。
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        # 本路径不处理 DSA-fp8 的分段量化布局（那条路走 set_mla_kv_buffer 的 dsa 分支）。
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        # 与 get_* 对称：fp8 时按 store_dtype(uint8) 重解释后散点写入对应 slot 行。
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # ③ 写 KV 链（分段写，MLA 主路径）：nope 与 rope 分别传入，由 triton 内核融合写进同一 slot，
        # 省去上游 concat 的开销。调用方=各 MLA backend / DeepseekV2AttentionMLA。
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id

        # 分支一：ROCm(HIP) + DSA + fp8——使用原始 (nope|rope) 布局、无逐块 scale，
        # 在写入时顺带把 BF16/FP16 量化成 FP8（cast 与 paged 写入融合在一个内核里）。
        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        # 分支二：CUDA-DSA + fp8——nope 单独量化（带逐块 scale）、rope 保持 bf16 字节，
        # 拼成 uint8 字节布局后复用通用两段写入内核。
        elif self.dsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        # 分支三：常规 MLA（bf16/fp16，或 fp8 但非 DSA）——对齐 dtype 后由内核把
        # nope 写入前段、rope 写入后段，二者落在同一行的 [0:lora_rank] 与 [lora_rank:] 区间。
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            # fp8：再按 store_dtype(uint8) 重解释，使写入与 buffer 的物理 dtype 一致。
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # ④ 读 KV 链（分段读，set_mla_kv_buffer 的逆操作）：按 loc 把 latent 拆回 (nope, rope)，
        # 供 DeepseekV2AttentionMLA 的 absorb/MHA 还原路径使用，可顺带 cast 到目标 dtype。
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        # 预分配输出：nope 段 [n,1,kv_lora_rank]
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        # rope 段 [n,1,qk_rope_head_dim]
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        # triton 内核：从 kv_buffer 的 loc 行 gather，并切分前段→nope、后段→rope。
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate accepted-token combined MLA KV (latent + rope) per layer."""
        # ⑤ 投机解码搬运链：EagleWorker 校验通过后，把 draft 临时 slot(src) 的 KV
        # 整体（latent+rope 一并）搬到 target 正式 slot(tgt)，避免重算被接受的 token。
        size_limit = self.size + self.page_size
        # 源/目标索引都要做越界检查（搬运误索引会跨层污染 KV）。
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        # 无被接受 token：直接返回，省去逐层空操作。
        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        # 逐层把 src 行整块拷到 tgt 行（MLA 单 buffer，一次拷贝即搬完该层全部 KV）。
        for kv_cache in self.kv_buffer:
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

    def get_cpu_copy(self, indices, mamba_indices=None):
        # ⑦ CPU offload 链（换出 GPU→CPU）：Req.offload_kv_cache 经 allocator 转发至此，
        # 把指定 indices 的 KV 拷到 host 内存暂存（如长上下文请求被挤出 GPU 时）。
        # mamba_indices 仅供混合 Mamba 模型签名兼容，纯 MLA 用不到。
        current_platform.synchronize()  # 先同步，确保前向写入的 KV 已落盘到 buffer
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size  # 分块（默认 8192）以削峰 host pinned 内存与拷贝
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                # non_blocking=True：异步 D2H 拷贝，循环结束统一 synchronize 等待。
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()  # 等全部异步拷贝完成再返回，保证数据完整
        # 返回结构：List[layer][chunk] → CPU 张量，load_cpu_copy 按相同结构回灌。
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # ⑦ CPU offload 链（换入 CPU→GPU）：get_cpu_copy 的逆操作，请求重新调度时把 KV 灌回 GPU slot。
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]  # 按换出时的 [layer][chunk] 结构取回
                assert kv_cpu.shape[0] == len(chunk_indices)  # 防御：chunk 行数须与索引数一致
                # H2D 异步拷回，再散点写入对应 GPU slot。
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()  # 等回灌完成，后续 forward 才能读到正确历史 KV


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    """💾 MLA KV Cache Pool (FP4) —— 把 MLA latent KV 以 4-bit 浮点(E2M1)分块量化存储的子类。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 FP4 存储原理：双 buffer（packed 数据 + 逐块 scale）                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  E2M1 = 4 bit/值(1 符号 + 2 指数 + 1 尾数)；PyTorch 无原生 4-bit dtype，故：         ║
    ║    • store_dtype = uint8：两个 FP4 值"紧压"进 1 个字节(packed，故末维 = k//2)         ║
    ║    • 每 16 个 FP4 值共享 1 个 uint8 的 block scale(MXFP4 风格，block=16)             ║
    ║  每层分配两块显存(基类只有一块 kv_buffer)：                                            ║
    ║    ┌───────────────────────────────┬────────────────────────────────────────────┐     ║
    ║    │ kv_buffer[layer]              │ kv_scale_buffer[layer]                     │     ║
    ║    │ [m, 1, kv_cache_dim // 2]     │ [m, kv_cache_dim // 16]                    │     ║
    ║    │ ↑ packed FP4 数据(uint8)      │ ↑ 每块的 E8M0 指数 scale(uint8)             │     ║
    ║    └───────────────────────────────┴────────────────────────────────────────────┘     ║
    ║  其中 kv_cache_dim = kv_lora_rank + qk_rope_head_dim(nope 压缩段 + rope 解耦段)。    ║
    ║  写 ✍️：BF16 → BlockFP4KVQuantizeUtil.batched_quantize → (packed fp4, scales)        ║
    ║         → 两块分别散点写。                                                             ║
    ║  读 📖：get_key_buffer 把 (packed fp4, scales) → batched_dequantize → BF16 在线还原。 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与基类 MLATokenToKVPool 的关键区别                                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ┌────────────────┬──────────────────────────────┬──────────────────────────────┐  ║
    ║  │ 维度           │ MLATokenToKVPool(基类)        │ MLATokenToKVPoolFP4(本类)    │  ║
    ║  ├────────────────┼──────────────────────────────┼──────────────────────────────┤  ║
    ║  │ 物理精度       │ BF16/FP16 或 FP8(单 buffer)  │ FP4(E2M1) packed + 逐块 scale │  ║
    ║  │ buffer 数量    │ 1 个 kv_buffer              │ 2 个：kv_buffer + scale_buffer│  ║
    ║  │ 单元素位宽     │ 16/8 bit                     │ 4 bit(两值一字节)            │  ║
    ║  │ get_key_buffer │ 零拷贝 view(dtype)           │ 在线 dequant → BF16(重算)    │  ║
    ║  │ set_mla_kv_buf │ 直写 / fp8 量化融合          │ 先 batched_quantize 再两段写 │  ║
    ║  │ 显存占用       │ 1×                           │ ≈ 1/4 基类(4bit vs 16bit)    │  ║
    ║  └────────────────┴──────────────────────────────┴──────────────────────────────┘  ║
    ║  🎯 适用场景：kv_cache_dtype == torch.float4_e2m1fn_x2(--kv-cache-dtype fp4_e2m1)  ║
    ║     时由 ModelRunnerKVCacheMixin 选用本类(见 model_runner_kv_cache_mixin.py:593)，   ║
    ║     以 ~4× 显存压缩换取 attention 前在线 dequant 的算力开销，适合大上下文/显存吃紧。   ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链(仅列 FP4 专属读写环节，完整链路见基类 docstring)                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ③ 写 KV：DeepseekV2AttentionMLA._set_mla_kv_buffer (forward_mha.py:455)         ║
    ║    └─ set_mla_kv_buffer(layer, loc, k_nope, k_rope)  ← CUDA/AITER 主路径✍️        ║
    ║       ├─ BlockFP4KVQuantizeUtil.batched_quantize  (nope/rope 各量化一次)          ║
    ║       ├─ set_mla_kv_buffer_triton       → 写 kv_buffer(packed fp4 数据)          ║
    ║       └─ set_mla_kv_scale_buffer_triton → 写 kv_scale_buffer(逐块 scale)         ║
    ║  ④ 读 KV：attention backend / DeepseekV2AttentionMLA                               ║
    ║    └─ get_key_buffer(layer_id)                       ← 拿整层 latent(本类📖)     ║
    ║       └─ BlockFP4KVQuantizeUtil.batched_dequantize  (在线还原成 BF16)            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ⚠️ 已知边界(仅注释，未改任何代码)：
      1. set_mla_kv_buffer 中 `if self.dsa_kv_cache_store_fp8` 分支：本类构造时不传
         use_dsa / override_kv_cache_dim，故 dsa_kv_cache_store_fp8 恒为 False，
         该分支为沿用自基类的保留 dead path(待 FP4 支持 DSA 时再启用)。
      2. set_mla_kv_buffer 末尾 `cache_k_nope/cache_k_rope = ...view(store_dtype)` 两行
         重赋值的是入参局部变量，其后不再被使用(triton 实际用 *_fp4 变量)，
         无副作用，疑为 fp8 路径的复制残留。
      3. 本类未重写 move_kv_cache / get_cpu_copy / load_cpu_copy / get_contiguous_buf_infos，
         这些基类方法只搬运 kv_buffer，不会同步搬运 kv_scale_buffer；若启用
         spec-offload / hierarchical-cache / PD 分离传输，scale 会丢失，需上层避免或补齐。
    """

    def _create_buffers(self):
        """🧬 申请 FP4 双 buffer(packed 数据 + 逐块 scale)，每层一份。

        🔗 调用链定位(① 创建链)：
            MLATokenToKVPool.__init__  →  self._create_buffers()  ← 当前函数
            (构造期唯一一次，见 model_runner_kv_cache_mixin.py:593 的实例化)

        ⚙️ 行为：覆盖基类，建立两块显存——
            • kv_buffer       : [size+page_size, 1, kv_cache_dim//2] uint8 (packed FP4)
            • kv_scale_buffer : [size+page_size,    kv_cache_dim//16] uint8 (block scale)
          +page_size 的"哨兵尾巴"承接 padding token 的 dummy 写入(同基类约定)。
        ⚠️ store_dtype 在此显式置为 uint8(覆盖基类对 fp8 的推断)，因 FP4 无原生 dtype。
        """
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                # custom_mem_pool：PD 分离 + NVLink 场景下用专用内存池，使 buffer 落在
                # 可被 RDMA/NVLink 直接访问的地址空间(详见基类 _create_buffers 注释)。
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                # m = 真实容量 + 一页哨兵；n=1 为 MLA 的 head 维；k = nope+rope 合并宽度。
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                # FP4 块量化块大小：每 16 个 FP4 值共享 1 个 uint8 scale(E8M0 指数)。
                scale_block_size = 16
                # FP4 无原生 torch dtype，统一以 uint8 存储(packed：2 值/字节)。
                self.store_dtype = torch.uint8

                # 数据 buffer：k//2 字节(每字节装 2 个 FP4 值)，head 维=1。
                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # scale buffer：每 16 个元素 1 个 scale，故末维 = k // 16；与数据 buffer 行对齐。
                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        """🧹 释放 FP4 双 buffer 显存(memory saver 挂起 / 销毁池时调用)。"""
        # 与基类不同：FP4 多一块 scale buffer，需一并 del。
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        """📖 读取整层 latent KV——把 FP4(packed)+scale 在线 dequant 还原为 BF16。

        🔗 调用链定位(④ 前向读 KV)：
            AttentionBackend.forward_* / DeepseekV2AttentionMLA
              └─ get_key_buffer(layer_id)  ← 当前函数
                 └─ BlockFP4KVQuantizeUtil.batched_dequantize  (在线反量化)

        📥 参数：layer_id : 全局层号，会换算成 [layer_id - start_layer] 本地索引(PP 切分)。
        📤 返回：整层 latent [m, 1, kv_cache_dim] 的 BF16 还原结果(含 nope+rope)。
        ⚙️ 行为：分层流水时先 wait_until 本层就绪；store_dtype(uint8) != dtype(fp4) 时
            走 dequant 路径，否则(理论未用)直接返回原 buffer。
        ⚠️ 在线 dequant 每次读都重算，算力开销 >> 零拷贝 view(基类 fp8 路径)。
        """
        if self.layer_transfer_counter is not None:
            # 分层流水(HiCache/PP)：阻塞等本层 KV 搬运完成，避免读到半成品。
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            # FP4 主路径：取 packed 数据 + 对应 scale，在线反量化成 BF16 供 attention 使用。
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            # batched_dequantize：unpack 两值/字节 → E2M1 LUT → 乘 block scale → BF16。
            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """✍️ 合并写路径——cache_k 已是拼好的完整 latent(nope+rope)，量化后写入双 buffer。

        🔗 调用链定位(③ 前向写 KV)：
            DeepseekV2AttentionMLA._set_mla_kv_buffer(NPU 等非 CUDA/AITER 平台)
              └─ set_kv_buffer(layer, out_cache_loc, kv_a, k_pe)  ← 当前函数
                 └─ BlockFP4KVQuantizeUtil.batched_quantize  (整段一次量化)

        📥 参数：
            layer    : 当前 attention 层，用 layer.layer_id 定位 buffer
            loc_info : 写入目标 slot(= forward_batch.out_cache_loc)，可为 KVWriteLoc
            cache_k  : 已 concat 的 latent [n,1,kv_cache_dim](BF16/FP16)
            cache_v  : MLA 无独立 V，此处未用(签名兼容)
        📤 返回：无(原地写入 kv_buffer + kv_scale_buffer)。
        ⚙️ 行为：越界检查 → 整段 batched_quantize → 分别散点写 数据/scale。
        ⚠️ 与 set_mla_kv_buffer 的区别：本路径把 nope+rope 当成整体量化(共享一套 scale)，
           仅适用于上游已 concat 的场景；CUDA/AITER 主路径走 set_mla_kv_buffer 分段量化。
        """
        # loc_info may be a KVWriteLoc; MLA pools have no SWA target.
        loc, _ = unwrap_write_loc(loc_info)
        # 越界探针：loc 必须落在 [0, size+page_size)，防止脏索引污染 buffer(含哨兵尾巴)。
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA-FP4)")
        layer_id = layer.layer_id
        # FP4 池构造时不传 use_dsa/override_kv_cache_dim，dsa_kv_cache_store_fp8 恒为 False。
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            # 输入是 BF16/FP16(≠ FP4) → 整段量化成 packed FP4 + block scale。
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_fp4, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(
                cache_k
            )

        if self.store_dtype != self.dtype:
            # FP4 路径：packed 数据与 scale 都是 uint8，分别散点写入对应 slot 行。
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            # 理论保留分支：若未来 dtype==store_dtype(非 FP4)，直写原值。
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """✍️ 分段写路径(FP4 主路径)——nope / rope 分别量化，两段 triton 内核融合写。

        🔗 调用链定位(③ 前向写 KV，CUDA / AITER 主路径)：
            DeepseekV2AttentionMLA._set_mla_kv_buffer (forward_mha.py:464)
              └─ set_mla_kv_buffer(layer, out_cache_loc, kv_a, k_pe)  ← 当前函数
                 ├─ BlockFP4KVQuantizeUtil.batched_quantize  (nope、rope 各一次)
                 ├─ set_mla_kv_buffer_triton       → 写 kv_buffer(packed fp4 数据)
                 └─ set_mla_kv_scale_buffer_triton → 写 kv_scale_buffer(逐块 scale)

        📥 参数：
            layer        : 当前 attention 层，用 layer.layer_id 定位 buffer
            loc          : 写入目标 slot(= forward_batch.out_cache_loc)
            cache_k_nope : 压缩潜变量段 [n,1,kv_lora_rank](BF16/FP16)
            cache_k_rope : 解耦位置段   [n,1,qk_rope_head_dim](BF16/FP16)
        📤 返回：无(原地写入 kv_buffer + kv_scale_buffer)。
        ⚙️ 行为：越界检查 → nope/rope 分别 batched_quantize → 两段 triton 内核把
            数据与 scale 各自融合散点写进对应 buffer 行(避免上游 concat 开销)。
        ⚠️ 注意：见类 docstring 的"已知边界"——dsa 分支为 dead path；
            末尾对 cache_k_nope/cache_k_rope 的 view 重赋值无副作用。
        """
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA-FP4)"
        )
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            # ⚠️ dead path：本类 dsa_kv_cache_store_fp8 恒为 False(构造未传 use_dsa)。
            #    沿用自基类，走 FP8 的 quantize_k_cache；待 FP4 支持 DSA 时再启用。
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            # FP4 主路径
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    BlockFP4KVQuantizeUtil,
                )

                # nope、rope 分别量化：各自的 block scale 独立，精度更优；
                # 返回 (packed fp4 uint8, scale uint8)。
                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                # ⚠️ 此处对入参 cache_k_nope/cache_k_rope 的 view 重赋值后未再使用，
                #    实际写入用的是上面的 *_fp4 变量；疑为 fp8 路径复制残留，无副作用。
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            # ① 写 packed FP4 数据：nope 段写入行前段、rope 段写入行后段(同一 buffer 行)。
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            # ② 写 block scale：与数据 buffer 的行一一对应，nope/rope scale 同样分段写入。
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    """💾 DSA KV Cache Pool —— DeepSeek Sparse Attention (DeepSeek-V3.2) 的 GPU KV 缓存。

    👆 继承自 MLATokenToKVPool：完整复用 latent kv_buffer（nope+rope）的存取，
       额外再挂一份 "indexer KV 缓存" 用于稀疏 topk 选择。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 在 MLA 基础上新增的对外接口（框架通过这些方法使用本类）                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  set_index_k_scale_buffer(layer_id, loc, k, scale)  ✍️ 前向时写入 indexer 的 K+scale ║
    ║  get_index_k_with_scale_buffer(layer_id)            📖 取整层 indexer buffer          ║
    ║  get_index_k_continuous / get_index_k_scale_continuous 📖 分别连续读 K / scale        ║
    ║  get_index_k_scale_buffer(...)                      📖 融合读 (K, scale)（Triton 高效）║
    ║  —— 以下为重写父类方法，把 indexer 缓存与主 latent 一起处理 ——                            ║
    ║  move_kv_cache         🚚 投机解码搬运：latent + indexer 同步搬                       ║
    ║  get_cpu_copy/load     💿 offload/换入：latent + indexer 一起，避免 slot 复用读脏数据 ║
    ║  get_state_buf_infos   🌐 PD 分离：把 indexer 作为 StateType.DSA 随主 KV 一起传输     ║
    ║  get_kv_size_bytes     ⚙️ 显存统计：latent + indexer 两份相加                        ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与父类 MLATokenToKVPool 的关键区别（多实现类对比）                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  ┌────────────┬─────────────────────────────┬─────────────────────────────────┐ ║
    ║  │            │ MLATokenToKVPool（常规）      │ DSATokenToKVPool（稀疏）          │ ║
    ║  ├────────────┼─────────────────────────────┼─────────────────────────────────┤ ║
    ║  │ 缓存张量     │ 仅 kv_buffer（latent）       │ kv_buffer + index_k_with_scale  │ ║
    ║  │            │                             │ _buffer（indexer 的 fp8 K+scale）│ ║
    ║  │ attention  │ 对全部历史 token 做 MLA        │ indexer 先选 topk，只对 topk     │ ║
    ║  │ 范围        │                             │ 个 token 做 MLA                  │ ║
    ║  │ indexer    │ 无                           │ 1 head × 128 dim，per-page 分页  │ ║
    ║  │ 量化        │ nope/rope 见父类             │ indexer K 存 fp8 + per-block     │ ║
    ║  │            │                             │ scale（quant_block_size=128）    │ ║
    ║  │ page_size  │ 任意                         │ 固定 64（非 HIP）/ 16 或 1(HIP)  │ ║
    ║  └────────────┴─────────────────────────────┴─────────────────────────────────┘ ║
    ║                                                                                  ║
    ║  🎯 为什么这种场景用本类：                                                          ║
    ║     超长上下文下对全量 token 做 attention 算力 ∝ 序列长度，代价高。DSA 用一个轻量      ║
    ║     "indexer" 先对全部历史 token 打分、选出最相关的 topk 个（如 2048），后续 MLA       ║
    ║     attention 只在这 topk 个 token 上计算，把复杂度从 O(seq) 降到 O(topk)。           ║
    ║     代价是要额外缓存 indexer 自己的 K（与主 latent 不同的张量），故本类在父类之上       ║
    ║     多维护一份 index_k_with_scale_buffer，并让搬运/offload/传输都带上它。              ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 DSA 特有的 indexer 调用链（每层 forward）                                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  ════════ 创建链 ════════                                                          ║
    ║  ModelRunnerKVCacheMixin（is_dsa_model & use_mla_backend）                         ║
    ║    └─ DSATokenToKVPool(index_head_dim=128, kv_cache_dim=含fp8 scale 的 override)   ║
    ║       └─ HiRadixCache 时：attach_hybrid_dsa_pool_to_hiradix_cache                  ║
    ║          └─ 建 INDEXER sidecar host pool，复用 KV 的 page indices 一起换入换出       ║
    ║                                                                                  ║
    ║  ════════ 前向 indexer + 稀疏 attention 链 ════════                                 ║
    ║  forward_mla.py → Indexer(x, q_lora, positions, layer_id)   (dsa_indexer.py)       ║
    ║    └─ Indexer.forward_cuda                                                        ║
    ║       ├─ _store_index_k_cache → set_index_k_scale_buffer(...)  ✍️ 写本步 indexer K  ║
    ║       └─ _get_topk_paged(decode) / _get_topk_ragged(extend)                        ║
    ║          └─ get_index_k_scale_buffer / get_index_k_with_scale_buffer  📖 读全量历史 ║
    ║          └─ query × 历史 indexer K → logits → 选出 topk_indices                    ║
    ║    └─ topk_indices → DeepseekSparseAttnBackend.forward_*                           ║
    ║       └─ 只对 topk token 取 latent（父类 get_key_buffer）做 MLA attention           ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    # indexer K 的 fp8 量化块大小：每 128 个元素共享一个 fp32 scale。
    quant_block_size = 128
    # indexer buffer 以 uint8 存放（fp8 数据 + 打包的 fp32 scale 字节）。
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
    ):
        """🏗️ 构造 DSA KV 池：先建好父类的 latent kv_buffer，再额外建 indexer 缓存。

        🔗 调用链定位（创建链）：
            ModelRunnerKVCacheMixin（is_dsa_model & use_mla_backend）
              └─ DSATokenToKVPool(...)  ← 当前函数
                 └─ super().__init__(use_dsa=True, ...)  建 latent kv_buffer（父类）
                 └─ 自建 index_k_with_scale_buffer            indexer 专属缓存

        📥 关键参数（相对父类新增）：
            index_head_dim : indexer key 的维度，DSA 固定为 128（单 head）
            kv_cache_dim   : latent 单 token 宽度；若含 fp8 scale 会 ≠ lora+rope，
                             此时转成 override 传给父类，让父类按真实宽度建 buffer
            index_buf_size : indexer 缓存的 token 容量，默认与 size 相同
        ⚠️ 注意：父类构造时传 use_dsa=True，会推迟显存日志，等本函数建完 indexer
            buffer 后再统一 _finalize_allocation_log（含两份缓存的总量）。
        """
        # 仅当 latent 宽度被 fp8 scale 撑大、≠ (lora+rope) 时才覆写父类维度；
        # 否则传 None，让父类用默认 kv_lora_rank + qk_rope_head_dim。
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        # 先建父类的 latent kv_buffer；use_dsa=True 让父类跳过显存日志（留到本函数末尾）。
        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size  # indexer 缓存容量默认与主 KV 容量一致
        # num head == 1 and head dim == 128 for index_k in DSA
        # DSA indexer 的 key 固定为单 head、128 维（与主 latent 维度无关），写死校验。
        assert index_head_dim == 128

        # 各后端对 page_size 的硬约束（indexer 的分页 MQA 内核要求不同）：
        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                # AITER preshuffle paged MQA：page 必须 16 对齐
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                # HIP 旧路径：逐 token，page_size 只能为 1
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            # CUDA：deep_gemm paged MQA 内核固定按 64-token 页工作
            assert self.page_size == 64
        with (
            # 与父类一致：PD+NVLink 场景用专用内存池，便于 indexer 缓存也能被远程访问。
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # indexer 缓存：每层一个 [num_pages, per_page_bytes] 的 uint8 张量。
            # per_page_bytes = page_size * (head_dim + head_dim/quant_block_size * 4)
            #   ├─ page_size * head_dim          ：fp8 量化后的 indexer K 数据
            #   └─ page_size * (head_dim/128) * 4：每 128 元素一个 fp32 scale（4 字节）
            # 与主 latent kv_buffer 按 token 索引不同，这里按"页"索引（page-indexed）。
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        # latent + indexer 两份缓存均已就绪，此刻统一统计显存并打日志（父类已推迟到这里）。
        self._finalize_allocation_log(size)

    def _clear_buffers(self):
        """🧹 释放 GPU 缓存：父类只删 kv_buffer，这里要连 indexer 缓存一并删。"""
        del self.kv_buffer
        del self.index_k_with_scale_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """🚚 投机解码搬运 KV：latent 与 indexer 缓存必须"锁步"一起搬。

        Move latent KV and the DSA indexer cache (key + scale) in lockstep.

        🔗 调用链定位（⑤ 投机解码搬运）：
            spec_utils.move_accept_tokens_to_target_kvcache
            / base_spec_worker.duplicate_prefix_tail_to_draft_branches
              └─ move_kv_cache(tgt, src)  ← 当前函数
                 ├─ super().move_kv_cache(...)        搬主 latent kv_buffer（父类）
                 └─ 再搬 index_k_with_scale_buffer    搬 indexer K+scale（本类）
        ⚠️ 为什么必须一起搬：被接受 token 的稀疏 attention 依赖 indexer K 做 topk，
            若只搬 latent 而漏搬 indexer，topk 选择会读到错位/陈旧的 indexer 数据。
        """
        # 第一步：复用父类逻辑搬运主 latent kv_buffer。
        super().move_kv_cache(tgt_loc, src_loc)

        # 无被接受 token：父类已直接返回，这里同样早退，省去 indexer 的空操作。
        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        # 第二步：逐层把 indexer 缓存的 src 行整块搬到 tgt 行（与 latent 同一组索引）。
        for index_k in self.index_k_with_scale_buffer:
            index_k[tgt_loc_flat] = index_k[src_loc_flat]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """📖 取整层 indexer 缓存（raw uint8 张量，含 fp8 K + scale）。

        🔗 调用链定位：Indexer._get_topk_paged（decode）/ _store_index_k_cache
            → 拿到整块 buf 后由 deep_gemm/triton 内核自行按页解析。
        """
        # 分层流水：异步逐层搬运时阻塞等本层就绪，避免读到半成品（同父类 get_key_buffer）。
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        """📖 按 page_indices 把 indexer 的 fp8 K 收集成连续张量。

        🔗 调用链定位：Indexer._get_topk_ragged_with_cp / forward_indexer（CP/NPU 路径）。
        ⚙️ 委托 index_buf_accessor.GetK 内核完成 page → 连续 的 gather。
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        """📖 与 get_index_k_continuous 配套，单独收集 fp8 K 对应的 scale。

        🔗 调用链定位：同上（CP/NPU 路径分两次分别取 K 和 scale）。
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """📖 一次性融合读取 indexer 的 (K, scale)，extend 阶段 topk 选择的主路径。

        🔗 调用链定位：Indexer._get_topk_ragged（extend）→ 读全量历史 indexer K+scale
            → 与 query 做 MQA logits → 选 topk token。比分两次单独读 K/scale 更快。

        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        # 委托 GetKAndS 内核：按 page_indices 同时 gather 出 fp8 K 与其 scale。
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        """✍️ 写入本步 token 的 indexer K（fp8）+ scale —— DSA 前向的写入入口。

        🔗 调用链定位（前向写 indexer）：
            Indexer.forward_cuda → _store_index_k_cache
              └─ set_index_k_scale_buffer(...)  ← 当前函数（fused/aiter 内核不可用时的回退路径）
        ⚙️ 委托 SetKAndS 内核把 index_k 与 index_k_scale 打包写进 loc 指向的页内偏移；
            布局与 __init__ 中描述一致（前段 fp8 数据、后段 fp32 scale 字节）。
        """
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        """💿 换出 GPU→CPU：latent 与 indexer 缓存都要 offload，返回 dict 打包两者。

        🔗 调用链定位（⑦ CPU offload）：
            Scheduler → Req.offload_kv_cache → allocator.get_cpu_copy
              └─ get_cpu_copy(...)  ← 当前函数
                 ├─ super().get_cpu_copy(...)   换出主 latent（父类，按 token 索引）
                 └─ 再换出 index_k_with_scale_buffer（本类，按 page 索引）
        ⚠️ 见下方原注释：indexer 也必须一起 offload，否则 resume 后会读到别的请求遗留的脏数据。
        """
        # DSA keeps a page-indexed index_k_with_scale_buffer alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # set_index_k_scale_buffer, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)

        # indexer 按页存储：把 token 索引降采样并换算成 page 索引（每页取一个代表）。
        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()  # 先同步，确保前向写入的 indexer 数据已落盘
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        # 按页分块：token 级 chunk_size 换算成 page 级（至少 1 页），削峰拷贝。
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                # 异步 D2H 拷贝整页 indexer 数据（fp8 K + scale 一并）。
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()  # 等所有异步拷贝完成再返回

        # 用 dict 同时带回 latent 与 indexer，load_cpu_copy 按相同结构回灌。
        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        """💿 换入 CPU→GPU：get_cpu_copy 的逆操作，latent 与 indexer 一起灌回。

        🔗 调用链定位（⑦ CPU offload）：
            Scheduler → Req.load_kv_cache → allocator.load_cpu_copy
              └─ load_cpu_copy(dict, ...)  ← 当前函数
                 ├─ super().load_cpu_copy(dict["kv"], ...)  灌回主 latent（父类）
                 └─ 再把 dict["index_k"] 灌回 index_k_with_scale_buffer（本类）
        """
        # 先复用父类逻辑灌回主 latent kv_buffer。
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"], indices, mamba_indices=mamba_indices
        )

        # 与换出对称：token 索引换算成 page 索引，按页回灌 indexer 缓存。
        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]  # 按换出结构取回
                assert idx_cpu.shape[0] == len(chunk_page_indices)  # 防御：页数须匹配
                # 异步 H2D 拷回，再按 page 索引散点写回 GPU。
                idx_chunk = idx_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()  # 等回灌完成，后续稀疏 attention 才能读到正确 indexer K

    def get_state_buf_infos(self):
        """🌐 PD 分离传输：暴露 indexer 缓存的 (ptr, len, item) 给 KV 传输引擎。

        🔗 调用链定位：disaggregation/utils.py 检测到本方法存在 → 把 indexer 作为
            StateType.DSA 额外状态组件，随主 latent KV 一起从 prefill 端发往 decode 端。
        ⚠️ 与父类 get_contiguous_buf_infos（暴露 latent）互补：稀疏 attention 在 decode
            端同样需要 indexer K，故 indexer 也必须跨节点传输。
        """
        # 每层 indexer buffer 的裸指针 / 总字节数 / 单页字节数，供 RDMA/NVLink 注册。
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        # item_lens：一页（buf[0]）的字节数，传输按页对齐。
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        """⚙️ 显存统计：DSA 总占用 = 父类 latent 缓存 + indexer 缓存。"""
        # 先取父类的 latent kv_buffer 字节数，再累加每层 indexer 缓存。
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]
