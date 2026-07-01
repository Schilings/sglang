import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_num_new_pages

_is_npu = is_npu()

if _is_npu:
    import torch_npu

    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: BaseSWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, BaseSWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        else:
            if _is_npu:
                PagedTokenToKVPoolAllocatorClass = NPUPagedTokenToKVPoolAllocator
            else:
                PagedTokenToKVPoolAllocatorClass = PagedTokenToKVPoolAllocator
            self.full_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size,
                page_size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size_swa,
                page_size,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self._kvcache = kvcache
        self.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None
        # 在alloc的时候，Full于SWA就会绑定映射关系了！
        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages
            <= self.full_attn_allocator.available_size() // self.page_size
            and num_swa_pages
            <= self.swa_attn_allocator.available_size() // self.page_size
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1
        # alloc是按page对齐的，所有请求所需的新页数之和
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None

        # 在alloc的时候，Full于SWA就会绑定映射关系了
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None
        # 在alloc的时候，Full于SWA就会绑定映射关系了
        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)

        return alloc_full_indices

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        """Allocate full KV for the whole extend and SWA KV only for the tail.

        This is used by disaggregated decode preallocation: decode receives full
        prompt KV for full-attention layers, but only the sliding-window state is
        transferred for SWA layers.
        """
        assert self.page_size > 1
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= extend_num_tokens

        num_full_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size
        if not self.new_pages_available(num_full_pages, num_swa_pages):
            return None

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_full_pages,
        )
        assert alloc_full_indices is not None

        if swa_tail_len == 0:
            return alloc_full_indices

        device = self.device
        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
        swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            swa_prefix_lens,
            swa_prefix_lens_cpu,
            swa_seq_lens,
            swa_seq_lens_cpu,
            swa_last_loc,
            swa_tail_len,
            num_new_pages=num_swa_pages,
        )
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(
            alloc_full_indices[-swa_tail_len:], alloc_swa_indices
        )
        if swa_tail_len < extend_num_tokens:
            self.full_to_swa_index_mapping[
                alloc_full_indices[:-swa_tail_len].to(torch.int64)
            ] = 0
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        """为 decode 步骤分配 FULL + SWA 各 1 个 token slot。

        与 alloc_extend 的关键区别：
        - decode 每次只分配 1 token/请求，核函数内动态判断是否跨 page 边界
          （未跨页 → last_loc+1；跨页 → 新 page 第一个 slot），无需预计算 num_new_pages
        - last_loc 用于判断上一格位置；FULL 和 SWA 的索引空间不同，需要 translate 转换
        """
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        if _is_npu:
            indices_2d = alloc_full_indices.to(torch.int64).unsqueeze(-1)
            torch_npu.npu_scatter_nd_update_(
                self.full_to_swa_index_mapping,
                indices_2d,
                alloc_swa_indices.to(torch.int64),
            )
        else:
            # 在alloc的时候，Full于SWA就会绑定映射关系了
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            # 因为传入的full的index，所以full可以直接free.
            # Free 是按 page 释放的吗？
            #   -> 是的。PagedTokenToKVPoolAllocator.free 内部做 free_index // page_size
            #      转 page 索引，再 torch.unique 去重后释放整 page。
            #      TokenToKVPoolAllocator(page_size=1) 则是逐 token 释放。
            #   -> 无论哪种，调用方传 token 级索引即可，allocator 内部自动处理粒度。
            # 为什么不会误释放同一 page 内其他还在用的 token？
            #   -> 框架三级保护：
            #      1. 树节点 key 由 RadixKey.page_aligned() 对齐，长度是 page_size 整数倍
            #      2. node.value 与 key 等长 -> slot 索引也恰好覆盖完整 page
            #      3. unaligned tail（不满整 page 的尾巴）独属于当前 req，不共享
            #      因此 node.value 中的索引要么覆盖整 page、要么是独占的 tail，
            #      不会出现同一个 page 内部分在用、部分要 free 的场景.
            self.full_attn_allocator.free(free_index)
            # 但是SWA的话需要转换一下index
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """Write full_to_swa_index_mapping[full_indices[i]] = swa_indices[i].

        Used by HiCache load-back path to rebuild the mapping after FULL and SWA device alloc.
        """
        if full_indices.numel() == 0:
            return
        assert full_indices.numel() == swa_indices.numel()
        if _is_npu:
            self.full_to_swa_index_mapping[full_indices.to(torch.int64)] = (
                swa_indices.to(torch.int64)
            )
        else:
            self.full_to_swa_index_mapping[full_indices] = swa_indices

    def free_swa(self, free_index: torch.Tensor):
        """释放 FULL 对应到 SWA 的 slot。

        SWA free 本身也是按 page 粒度（swa_attn_allocator.free 内部 // page_size），
        即使不扩展，传入 page 内部分 token 也能正确释放整 page。

        _expand_to_full_pages 的真正目的：防止 full_to_swa_index_mapping 残留。
        free_index=[5,6,7] 时 FULL page 1（tokens 4-7）整体被释放，
        但若只清理 [5,6,7] 的映射，token 4 的映射会残留为 dangling reference，
        指向已释放的 SWA page，后续 token 4 被重新分配时将错误命中。
        扩展为 [4,5,6,7] 确保整 page 的映射全量清零。
        """
        if free_index.numel() == 0:
            return

        # page_size == 1：每个 token 即一个 page，无需扩展
        # page_size > 1：扩展到整 page 的全部 token，保证映射全量清理
        if self.page_size == 1:
            mapping_indices = free_index
        else:
            mapping_indices = self._expand_to_full_pages(free_index)

        # 从映射表查出对应的 SWA slot 索引，过滤掉未映射的（> 0）
        swa_indices = self.full_to_swa_index_mapping[mapping_indices]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)       # 释放 SWA 端 slot
        self.full_to_swa_index_mapping[mapping_indices] = 0  # 清除映射

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        pages = torch.unique(indices // self.page_size)
        page_offsets = torch.arange(
            self.page_size, dtype=indices.dtype, device=indices.device
        )
        return (pages[:, None] * self.page_size + page_offsets[None, :]).reshape(-1)

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
