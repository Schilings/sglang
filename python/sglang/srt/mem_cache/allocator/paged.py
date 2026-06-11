"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""
Page-aligned memory pool.
"""


from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.triton_ops.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        """
        为一批请求的 extend 部分（prefill 新 token）分配 KV slot 索引。

        分配策略: 整页分配，一个 page 包含 page_size 个连续 slot。
        extend 部分的三段式填充（由 alloc_extend_kernel 执行）:

          Part 1: 填充旧 page 的剩余空间（不消耗新 page）
            当 prefix_len 不是 page_size 的倍数时，前缀最后一个 page 还有空位，
            这些 slot 之前已经整页分配给该请求，直接用 last_loc+1 接上。

            何时 prefix_len 不是 page_size 的倍数？
            ─────────────────────────────────────────
            • RadixCache 路径（有 radix prefix 匹配）：
              radix cache 存储时 key 经过了 .page_aligned(page_size) 截断，
              cache_unfinished_req 也只会缓存 page-aligned 长度的 KV indices。
              因此 match_prefix 返回的 prefix_len 一定是 page_size 的倍数 → Part 1 = 0。
              不会出现"填别的请求的 page"的问题——共享的 radix cache page 总是满的。

            • ChunkCache / Streaming Session 路径（无 radix prefix 匹配）：
              chunked prefill 第二次调度时，init_next_round_input 调用时 tree_cache=None，
              不会重新 match prefix。prefix_indices 由上一轮 cache_unfinished_req 直接设置
              （取全部 kv_indices，不做 page-aligned 截断）。
              如果上一轮 extend 没填满一个 page，prefix_len 就不是 page_size 的倍数 → Part 1 > 0，
              将拼接到自己之前分配的半满 page 后面。

            首次 prefill（无历史 prefix）的 prefix_len = 0，也是 page_size 的倍数 → Part 1 = 0。

          Part 2: 填充完整的新 page（从 free_pages 取）
            每个完整 page 的所有 slot 编号为 page_number * page_size + offset_in_page。

          Part 3: 填充新 page 的前半段（与 Part 2 最后一个 page 共享）
            seq_len 不对齐时，最后一个 page 只填前几个 slot。

        新 page 数 = ceil(seq_len/page_size) - ceil(prefix_len/page_size)
        两侧都用 ceil，差值自动排除了"半满的旧 page"（已在 prefix 的 ceil 中计入）。

        执行顺序: 先让 kernel 写好 out_indices（基于 free_pages 快照），
        再检查 free_pages 是否足够。不够则返回 None，调用方需 evict 后重试。
        """
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        # 如果 free_pages 可能不够，先排序合并碎片
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        # 调用 Triton kernel 并行计算每个请求的 out_indices
        # kernel 内部按三段式填充：Part1(旧page剩余) + Part2(完整新page) + Part3(新page前半段)
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        # 计算需要的新 page 数: ceil(seq/ps) - ceil(prefix/ps)
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        # 检查空闲 page 是否足够，不够则返回 None（调用方需 evict 后重试）
        if num_new_pages > len(self.free_pages):
            return None

        # 消费掉已使用的 page
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        """
        为一批 decode 请求各分配 1 个 KV slot 索引。

        两种情况:
          1. seq_len 没有跨越 page 边界 → 不需要新 page，直接用 last_loc + 1
          2. seq_len 跨越 page 边界 → 需要从 free_pages 取一个新 page，用其第一个 slot
        """
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        # decode kernel: 每个请求只分配 1 个 slot
        # 如果当前 page 有空位 → last_loc + 1；否则 → 新 page 的第一个 slot
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        # 计算 decode 需要的新 page 数（跨越 page 边界的请求数）
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        """释放 token-level 的 KV slot 索引。

        【为什么敢用 free_index // page_size 直接转 page 索引？】
        虽然入参是 token 级别的索引，但调用方保证了 page 对齐安全性：
        1. 树中的节点全由 RadixKey.page_aligned() 生成，长度是 page_size 的倍数，
           因此 evict 或 cache_unfinished_req 释放树节点的 value 时，indices 恰好对应完整的 page
        2. cache_protected_len 也是 page-aligned 的（来自 match_prefix 返回的树中 value 长度），
           所以 kv_indices[cache_protected_len : new_prefix_len] 切片也是完整 page
        3. 唯一例外是 unaligned tail（不满一个 page 的尾部），但它独属于一个 req，不会共享，
           所以释放整个 page 不会影响其他 req
        综上：不会出现"同一个 page 内部分在用、部分被 free"的情况。
        """
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            # token index → page index，torch.unique 去重同一 page 的多个 token
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            # free_group 模式：先攒着，等 free_group_end 时统一处理
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
