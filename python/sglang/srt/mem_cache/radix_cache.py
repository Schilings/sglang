from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

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
"""

"""
The radix tree data structure for managing the KV cache.
"""

import hashlib
import heapq
import logging
import sys
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.utils import get_eviction_strategy, split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence (raw ints in both modes)
        # 例如: ["A", "B", "C"]
        # 逻辑标识：从根节点到当前节点的路径所代表的 Token 序列
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        if self.is_bigram:
            n = len(self.token_ids)
            return n - 1 if n > 0 else 0
        return len(self.token_ids)

    # TODO(Jialin): vectorize with numpy without PyLong boxing
    def __iter__(self) -> Iterator:
        if self.is_bigram:
            t = self.token_ids
            for i in range(len(t) - 1):
                yield (t[i], t[i + 1])
        else:
            yield from self.token_ids

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> "RadixKey":
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple["RadixKey", Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: "RadixKey") -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )
# 因为keys是截断为page_size的倍数的，可能还有剩余未满一页的尾部

    # TODO(Jialin): replace zip with numpy to skip per-element PyLong boxing
    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids

        if self.is_bigram:
            # Walk raw tokens; L matching tokens imply L-1 matching bigrams.
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            matched = max(0, min(i - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        if page_size == 1:
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            return i

        min_len = min(len(self), len(other))
        i = 0
        while i < min_len:
            if t0[i : i + page_size] != t1[i : i + page_size]:
                break
            i += page_size
        return i

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        hasher = hashlib.sha256()
        if prior_hash:
            hasher.update(bytes.fromhex(prior_hash))
        t = self.token_ids
        if self.is_bigram:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
                hasher.update(t[j + 1].to_bytes(4, byteorder="little", signed=False))
        else:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        # 其中key需要截断为page_size的倍数 ----> 因此可以认为TreeNode中的key的长度全都是page_size的倍数！！！！
        # 每个child key为 (page size个token ids)
        # 树结构：子节点分支
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        # 例如: ["A", "B", "C"]
        # 逻辑标识：从根节点到当前节点的路径所代表的 Token 序列
        self.key: RadixKey = None
        # 例如: [10, 11, 12]
        # 物理索引：从根节点到当前节点的路径所代表的 Token 序列在 token_to_kv_pool 中的物理 Slot Indices
        self.value: Optional[torch.Tensor] = None
        # lock_ref > 0 时，该节点及其物理槽位不可被驱逐
        # 引用计数：当前有多少个正在运行的请求在使用这个节点
        self.lock_ref = 0
        # 驱逐策略元数据：LRU 时间戳
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        # 每一个Page的hash值
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        # 优先级
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        # 节点数+1
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class RadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        # 事件驱动模型
        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        # 那么req_to_token_pool内记录的这个req的kv indices就要修改
        # [req.cache_protected_len : new_prefix_len]原本这部分的kv indices已经被free，换成去复用多出来的prefix的token的kv indices
        # 原因就是prefix变长了！！！！！
        # 在batch.prepare_extend()就已经写在req_to_token_pool了，为什么再这里又再写一次
        # 7. new_indices[req.cache_protected_len :]指的是该请求的extend input部分，即除了prefix的实际计算的token们的kv indices
        # new_indices：整个req目前的token ids的indices
        # 6. 前面插入了，现在匹配，肯定是完全匹配，这是整个req目前的token ids的indices
        # 那么自己的kv_indices[req.cache_protected_len : new_prefix_len]这部分就不用了，直接复用多出来的prefix的token
        # 那么这里free的范围是 [prefix_len : new_prefix_len]
        # 且设置了req.cache_protected_len = prefix_len
        # 其中在init_next_round_input做计算准备的时候对prefix进行了匹配了，
        # 意思就是prefix变长了！！！！！
        # 意思是？在上次调度的时候，刚好有请求生成的token延长了这个req的前缀？？不是没有这种可能好像
        # 当new_prefix_len ！= prefix_len时？会有这种情况吗？
        # 但是经过一次调度执行后，这个时候insert也会匹配prefix，此时有new_prefix_len
        # 这里有个问题：一个req在chunked prefill前会进行prefix匹配，此时有prefix_len
        """
        # 更新req.cache_protected_len
        # 5. 注意这里free的范围是 [req.cache_protected_len : new_prefix_len]
        # 4. 将这些token插入到前缀树中，最后只会插入一个Node，返回匹配的总prefix长度
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result
# 当page_size大于1时，会在匹配前将长度截断为page_size的倍数

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        # last_node ： 返回匹配链中的最后一个节点
        # value ： 保存了匹配链中每个节点的value
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        # deterministic mode 不允许结束推理的请求进行insert
        if self.disable_finished_insert:
            is_insert = False

        # 1. 计算需要缓存的长度
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        # 2. 取出缓存的token ids
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        # 2. 计算完的token ids对应的kv已经存储在了kv pool中，获取其索引
        # 3. 这些token ids对应在kv pool里面的索引
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
# ！！！！！其中key需要截断为page_size的倍数 ----> 因此可以认为TreeNode中的key的长度全都是page_size的倍数！！！！
# 3. 计算key和value，节点中key是token ids，value其kv值在kv pool的index
# 其中key需要截断为page_size的倍数
# 4. 计算key和value，节点中key是token ids，value其kv值在kv pool的index

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        # 因此对于chunked req，req.fill_ids只是完成prefill计算的前半部分token
        # 截断后的fill_ids长度为prefix_len + trunc_len
        # 对于chunked req，其fill_ids会在调度时被截断：req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
        # 1. 计算完的token ids需要存储在前缀树中
        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        # req的last node可能变成了另一个
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        # 诶？怎么会有变短的情况呢？
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        # 如果是chunked 请求，那么该值为更新
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        # 1. 扫描树中所有 lock_ref == 0 的叶子节点
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        # 优先级低的排前面
        # 2. 按照 priority/last_access_time 从旧到新排序 (LRU)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # token对应的kv cache的存储？？
            # 为什么insert的时候没见到token_to_kv_pool_allocator
            # 3. 释放物理槽位并删除节点，直到凑齐足够的显存空间
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            # 4. 删除节点
            self._delete_leaf(x)

            # 5. 如果删除节点x后，其父节点也成为叶子节点，添加到删除堆内
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            # 6. 记录删除事件
            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        # 沿着匹配链把所有node的引用都-1
        # 沿着匹配链把所有node的引用都+1
        # 从该节点向根节点匹配
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time
# 如果不满一个page呢？
# 虽然RadixKey可能保存了更长的token ids，但是只取 tuple(key.token_ids[:page_size]) 用于匹配
# 如果页大小大于1，则取键中前page_size个token ID + extral_key组成元组作为普通键

        child_key = key.child_key(self.page_size)

        # 记录匹配链 得到的 value值
        value = []
        while len(key) > 0 and child_key in node.children.keys():
            # 匹配到子节点
            child = node.children[child_key]
            child.last_access_time = access_time
            # 但是不代表完全匹配，一个节点是可能存很长token id的，看匹配多长的prefix
            # 第一个page相同，就认为key相同
            prefix_len = child.key.match(key, page_size=self.page_size)
            # 则split，然后父节点就是匹配链的最后一个节点
            # 但是这里不是insert，将other_key继续匹配，匹配到某个节点发现要split
            # key分为两份prefix与other_key，如果是insert，则将other_key作为父node的child
            # 则把这个node拆分成两个prefix和suffix，作为父子关系，然后返回父node
            # 如果该请求的token id的匹配链到这里结束，但是没有完全匹配最后一个node
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                # 父node
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                # 使用RadixKey在Radix树中进行查询匹配的时候，会将前面的 匹配的prefix 弹出(本质不是弹出，只是slice)，逐个node匹配
                # RadixKey存放的是完整的序列的token ids
                key = key[prefix_len:]

                # 继续取新的page key匹配下一个node
                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        # 切割出一个父node，将该node作为去其child
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        # 切割child node的key value
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # 返回父node
        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        # 把匹配链的优先级都取 max
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0
# 如果不满一个page呢？
# 使用当前key的前page size个token ids进行匹配

        child_key = key.child_key(self.page_size)

        # 记录该请求匹配到的总prefix长度
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            # 匹配到子节点
            node = node.children[child_key]
            node.last_access_time = access_time
            # 但是不代表完全匹配，一个节点是可能存很长token id的，看匹配多长的prefix
            # 第一个page相同，就认为key相同
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            # 使用RadixKey在Radix树中进行查询匹配的时候，会将前面的 匹配的prefix 弹出(本质不是弹出，只是slice)，逐个node匹配
            # RadixKey存放的是完整的序列的token ids
            key = key[prefix_len:]
            value = value[prefix_len:]

            # key分为两份prefix与other_key，将other_key作为父node的child
            # 则把这个node拆分成两个prefix和suffix，作为父子关系，然后返回父node
            # 如果该请求的token id的匹配链到这里结束，但是没有完全匹配最后一个node
            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            # 继续取新的page key匹配下一个node
            if len(key):
                child_key = key.child_key(self.page_size)

        # 如果other_key不为空，将other_key作为父node的child
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            # 新节点的token id会分页，分页计算哈希值，然后分页通知，有多个事件
            # 需要存储这个新节点的token ids的，通过事件驱动，监听者会消费这些时间
            self._record_store_event(new_node)
        # 返回匹配到的总prefix长度
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                # 这一页的hash值
                stack.append(child)
        return total_size


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
