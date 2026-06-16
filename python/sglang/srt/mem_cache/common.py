from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.triton_ops.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.srt.mem_cache.triton_ops.common import (
    get_last_loc_kernel as get_last_loc_kernel,
)
from sglang.srt.mem_cache.triton_ops.common import (
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_hip, support_triton
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # The page is guaranteed to be full except the last page.
    if page_size == 1:
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
) -> None:
    from sglang.srt.environ import envs

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)

    # Subtract an extra page_size so the eviction frontier never reaches the
    # radix tree insert boundary (page_floor(seq_len)). This keeps at least one
    # page of non-evicted SWA KV for the tree to store as a non-tombstone node,
    # preserving cache reuse in multi-turn scenarios. Without this, leaf nodes
    # may become tombstoned, causing SWA memory leak.
    # See also: _insert_helper case 3 in swa_radix_cache.py (defensive counterpart).
    if envs.SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN.get():
        evict_threshold = pre_len - sliding_window_size
    else:
        evict_threshold = pre_len - sliding_window_size - page_size
    new_swa_evicted_seqlen = max(
        req.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        req.swa_evicted_seqlen = new_swa_evicted_seqlen


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    """
    将 KV slot 索引写入 req_to_token_pool。

    写入两段:
      [0, prefix_len):   写 prefix_tensors    — match_prefix 匹配到的已有 KV
      [prefix_len, seq_len): 写 out_cache_loc  — 本次新分配的 KV

    写入后，req_to_token_pool[req_pool_idx, token_pos] = kv_slot_idx，
    后续 attention kernel 通过该映射取 KV cache。
    """
    if support_triton(get_global_server_args().attention_backend):
        # Triton 路径：GPU kernel 并行写入，通过 data_ptr 传递 prefix tensor
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            dtype=torch.uint64,
            pin_memory=is_pin_memory_available(req_to_token_pool.device),
        ).to(req_to_token_pool.device, non_blocking=True)
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        # CPU fallback：逐请求写
        pt = 0  # out_cache_loc 中的游标
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()  # 该请求在 req_to_token_pool 中的行
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            # 写入 prefix 部分：已有 KV 的 slot 索引
            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            # 写入 extend 部分：本次新分配的 KV slot 索引
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_global_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_global_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per
    # decode step (draft chain and verify block share the reservation).

    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # page_size > 1 + topk > 1 (spec v2 tree): worst-case page-aligned tree
        # footprint. Per topk branch needs ceil((last_page_len + num_steps) / page)
        # pages; the partial tail page can be up to page_size - 1, and each branch
        # gets its own (duplicated) copy -- so reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_info_v2.prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args)


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation (kv_committed_len +
    get_alloc_reserve_per_decode). The spec v2 page>1 topk>1 holey draft footprint
    can outgrow the default num_draft_tokens headroom (PR #26972).
    """
    # FIXME(lsyin): this is the temporary fix for the context length issue when
    # using speculative decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if (
        server_args.speculative_algorithm is not None
        and server_args.page_size > 1
        and (server_args.speculative_eagle_topk or 1) > 1
    ):
        extra = max(extra, get_alloc_reserve_per_decode(server_args))
    return extra


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    # 如果allocator内剩余kv indices不足，调用tree_cache.evict(num_tokens)清除
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    # 如果需要备份，在alloc前备份好，一般用于投机解码回退
    if backup_state:
        state = allocator.backup_state()
    # 简单场景: 按 token 直接分配，无需分页
    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    """
    为 extend batch 执行分页式的 KV slot 分配。

    流程:
      1. evict_from_tree_cache: 如果空闲 slot 不够，从 radix tree 驱逐节点释放空间
      2. allocator.alloc_extend: 三段式分页分配（Part1 填旧页 + Part2 完整新页 + Part3 新页前半段）

    为什么 num_tokens 要往上多估 len(seq_lens_cpu) * page_size？
      因为每个请求最后一个 page 可能半满 → 每个请求都可能多需要一整页。
    """
    allocator = tree_cache.token_to_kv_pool_allocator
    # 高估所需 token 数：假设每个请求都需要额外一整页（最后的半满页 + 新页）
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    # 驱逐 → 回收 full + swa pool 的空间（SWA 场景下同时检查两个池）
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    # 实际分配：调用 paged allocator 的三段式填充逻辑
    out_cache_loc = allocator.alloc_extend(
        prefix_lens,       # 每个请求的已有 prefix 长度
        prefix_lens_cpu,
        seq_lens,           # 每个请求 prefill 后的总长度
        seq_lens_cpu,
        last_loc,           # 每个请求前缀的最后一个 slot 索引
        extend_num_tokens,  # 所有请求的新增 token 总数
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_allocator.available_size()
        if tree_cache.supports_mamba():
            factor = (
                MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
                if req_to_token_pool.enable_mamba_extra_buffer_lazy
                else MAMBA_STATE_PER_REQ_PREFIX_CACHE
            )
        else:
            factor = MAMBA_STATE_PER_REQ_NO_CACHE
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    # req_to_token_pool内应该会处理chunked req的情况
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """为 extend batch 分配 KV cache 并写入 req_to_token_pool。

       Q: 这个函数做了什么？
       A: 四步流水线: 回收 SWA → 分配 pool 行 → 分配 KV 物理 slot → 写入。

       Q: 分配的 slot 什么时候写入 req_to_token_pool？
       A: 就在本函数末尾，write_cache_indices 将 prefix(已有KV) + extend(新分配) 一起写入。
    """
    # ── 1. 回收超出滑动窗口的 SWA token ──
    batch.maybe_evict_swa()

    # ── 2. 准备 prefix / lens 参数 ──
    # req的prefix_indices在req.init_next_round的适合会match prefix得到
    prefix_tensors = [r.prefix_indices for r in batch.reqs]       # 每个请求已有的 prefix slot

    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # 为每个请求分配 req_to_token_pool 中的一行
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # ── 3. 分配 KV pool 物理 slot ──
    if batch.tree_cache.page_size == 1:
        # 简单场景: evict->alloc, 按 token 直接分配，无需分页
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # 分页场景: last_loc 是每个请求 prefix 的最后一个 slot，用于判断旧 page 是否还有空位
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        # 内部分配: evict → alloc_extend（三段式: Part1处理chunked req的旧page剩余 + Part2完整新page + Part3新page前段）
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )

    # ── 4. 写入 req_to_token_pool ──
    # 每个请求一行，分两段:
    #   [0, prefix_len): prefix_tensors   — 已有的 KV slot（来自 radix cache）
    #   [prefix_len, seq_len): out_cache_loc — 新分配的 extend slot
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + seq_lens_gpu
    else:
        locs = seq_lens_gpu.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    """释放已完成请求的 KV cache，并将 KV 写入 radix tree 供后续请求复用。

    整体流程：
    1. Mamba 模型的特殊处理：可能先分配 mamba state 再分配 KV cache，
       此时 req_pool_idx 为 None，仅释放 mamba pool
    2. 调用 cache_finished_req 将 KV 插入 radix tree（prefix cache）
    3. 处理 overallocated KV（speculative decode / strip_thinking 可能多分配）
       将多余的 KV slot 归还到 token_to_kv_pool_allocator
    4. 释放 Mamba state（如果 cache 不管理 Mamba）
    5. 释放 req_to_token_pool 中的 slot

    Args:
        req: 已完成/待完成的请求
        tree_cache: radix tree 前缀缓存实例
        is_insert: 是否将 KV 插入 radix tree。默认 True；
                   skip_radix_cache_insert 标记的请求会跳过。
    """
    # MambaRadixCache 可能在分配 KV cache 之前先分配了 mamba state，
    # 此时 req_pool_idx 为 None，仅需释放 mamba pool
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    # 将已完成的 KV 序列写入 radix tree，供后续请求 prefix cache 命中复用
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    # cache_finished_req 内部会处理 speculative decode 的尾部裁剪和标记同步，
    # 完成后将 req_pool_idx 置为 None。如果已为 None 则无需后续释放。
    if req.req_pool_idx is None:
        return

    # 处理 overallocated KV：
    # - speculative decode：可能预分配了超过实际使用的 KV slot
    # - strip_thinking_cache：思考阶段 token 的 KV 故意标记为 overallocated（#22373）
    # start_p 到 end_p 区间即为需要归还的冗余 KV slot
    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # 非 speculative 且非 strip_thinking 场景不应有 overallocated KV
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    # page_size > 1 时按 page 对齐，整页释放
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    # 归还冗余的 KV slot 到 allocator
    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)

    # 如果 prefix cache 不管理 Mamba state，需要手动释放
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)

    # 最终释放 req_to_token_pool 中的 slot
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
