from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSAIndexerPoolHost,
    HostPoolGroup,
    LogicalHostPool,
    MambaPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  🔧 Hybrid Pool Assembler —— HiCache Host Pool 统一创建工厂                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  本模块是 HiCache (L1 GPU ↔ L2 Host ↔ L3 Storage) 的 Host 端资源统一构造入口。   ║
║  根据模型架构 (KV only / SWA / Mamba / MLA / DSA / DeepSeek V4) 自动选策,       ║
║  组装对应的 HostPoolGroup + HybridCacheController, 挂载回 cache 实例。           ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────┐            ║
║  │ 对外接口 (attach_*)：被各 cache 类的 __init__ / init_hicache 调用 │            ║
║  ├─────────────────────────────────────────────────────────────────┤            ║
║  │ attach_hybrid_pool_to_unified_cache()    → UnifiedRadixCache    │            ║
║  │ attach_hybrid_dsa_pool_to_hiradix_cache() → HiRadixCache (DSA/MLA)│           ║
║  │ attach_hybrid_pool_to_mamba_cache()       → HiMambaRadixCache    │            ║
║  │ register_stack_strategy()                → 下游 fork 可注入策略  │            ║
║  └─────────────────────────────────────────────────────────────────┘            ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────┐            ║
║  │ 内部 Build 函数 (build_*)：返回 (HostPoolGroup, HybridCacheController)│        ║
║  ├─────────────────────────────────────────────────────────────────┤            ║
║  │ build_kv_only_stack()         → 单池: PoolEntry(KV)               │            ║
║  │ build_hybrid_swa_stack()      → 双池: PoolEntry(KV) + PoolEntry(SWA)│         ║
║  │ build_deepseek_v4_hicache_stack() → V4 多池: KW+SWA+C4+C128+Indexer│         ║
║  │ build_hybrid_mamba_stack()    → 双池: PoolEntry(KV) + PoolEntry(MAMBA)│       ║
║  │ build_anchor_sidecar_stack()  → 通用: 锚定池 + N 个辅池            │            ║
║  └─────────────────────────────────────────────────────────────────┘            ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────┐            ║
║  │ 策略选择 (StackStrategy)：根据 kvcache 类型 + components 集合匹配   │            ║
║  ├─────────────────────────────────────────────────────────────────┤            ║
║  │ _DeepSeekV4Strategy   → DeepSeekV4TokenToKVPool + {FULL,SWA}    │            ║
║  │ _MambaStrategy        → HybridLinearKVPool + {FULL,MAMBA}       │            ║
║  │ _SwaStrategy          → SWAKVPool + {FULL,SWA}                  │            ║
║  │ _DsaStrategy          → DSATokenToKVPool + {FULL}               │            ║
║  │ _PlainKvStrategy      → 其他 KVCache + {FULL} (兜底)             │            ║
║  └─────────────────────────────────────────────────────────────────┘            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    server_args: ServerArgs,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
):
    """💾 根据 device pool 类型创建对应的 Host KV 池。

    ⚙️ 自动选类:
      use_mla=True  → MLATokenToKVPoolHost (DeepSeek-V2/V3 的 latent KV)
      use_mla=False → get_mha_host_pool_cls(kv_pool)
                      → MHATokenToKVPoolHost 或 AsymmetricMHATokenToKVPoolHost

    📥 kv_pool: 对应的 GPU 端 KV 池, Host 端的大小 = ratio × device 端大小。
    """
    kv_host_pool_cls = (
        MLATokenToKVPoolHost if use_mla else get_mha_host_pool_cls(kv_pool)
    )
    kwargs = {}
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    return kv_host_pool_cls(
        kv_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        **kwargs,
    )


def build_pool_entry(
    *,
    name: PoolName,
    host_pool: Any,
    device_pool: Any,
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
    is_anchor: bool = False,
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
    device_alloc_fn: Optional[Callable[[int], Any]] = None,
    device_free_fn: Optional[Callable[[Any], Any]] = None,
) -> PoolEntry:
    """🔧 构造单个 PoolEntry —— 一个 Host 子池的完整描述符。

    ⚙️ is_anchor=True → is_primary_index_anchor=True, 该池成为 HostPoolGroup 的锚定池。
    ⚙️ layer_mapping → _make_layer_mapper: 全局 layer_id → 本池局部 layer_id,
       DMA 时用它判断当前层是否归本池管理。

    💡 SWA/Mamba 的 device_alloc_fn / host_evict_fn 用于
       _resolve_pool_transfers_allocation 自动分配 sidecar 池空间。
    """
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
        is_primary_index_anchor=is_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
        device_alloc_fn=device_alloc_fn,
        device_free_fn=device_free_fn,
    )


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    """💾 构建单池 HiCache Stack —— 仅 FULL KV, 无辅组件。

    ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
    attach_hybrid_pool_to_unified_cache() (KV-only / MLA 模型)
      → _PlainKvStrategy.build() → build_kv_only_stack(...)

    ⚙️ HostPoolGroup 只有一个 PoolEntry(KV, is_anchor=True),
       锚定池即唯一池, alloc/free/DMA 全走锚定池路径。
    """
    transfer_layer_num = len(full_layer_mapping)
    # ① 建 Host KV 池: GPU pool → CPU mirror (MLATokenToKVPoolHost 或 MHATokenToKVPoolHost)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    # ② 单 PoolEntry: KV 是锚定池, 也是唯一池
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        )
    ]
    # ③ HostPoolGroup(单池) + HybridCacheController
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    """💾 构建 FULL + SWA 双池 HiCache Stack。

    ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
    attach_hybrid_pool_to_unified_cache() (SWA 模型)
      → _SwaStrategy.build() → build_hybrid_swa_stack(...)

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  🧬 构造的 HostPoolGroup 结构                                         ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  HostPoolGroup(entries=[                                            ║
    ║    PoolEntry(KV, is_anchor=True)                                     ║
    ║      host_pool=kv_host_pool     ← FULL KV CPU 池 (锚定池)            ║
    ║      device_pool=full_kv_pool   ← FULL KV GPU 池                     ║
    ║      layer_mapping=full_layer_mapping  ← 管理 full attention layers  ║
    ║    PoolEntry(SWA)                                                    ║
    ║      host_pool=swa_host_pool    ← SWA CPU 池 (辅池, 独立内存)         ║
    ║      device_pool=swa_kv_pool    ← SWA GPU 池                         ║
    ║      layer_mapping=swa_layer_mapping  ← 管理 SWA attention layers    ║
    ║      device_alloc_fn=swa_attn_allocator.alloc  ← SWA 独立分配器       ║
    ║  ])                                                                  ║
    ║                                                                      ║
    ║  💡 alloc/free 走锚定池 (KV), SWA 的分配通过                          ║
    ║     _resolve_pool_transfers_allocation 直接调                         ║
    ║     entry_map[SWA].host_pool.alloc().                                 ║
    ║  💡 DMA 时 backup/load 分两步: ① 锚定池 (KV) → ② 辅池 (SWA),          ║
    ║     各自从不同的 Device pool → 不同的 Host pool。                     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    # 总层数 = FULL 层 + SWA 层的并集
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    # ① 建两个独立的 Host 池: KV + SWA, 各自绑定自己的 GPU pool
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )

    # For SWA hybrid, the device alloc/free goes through the inner swa_attn_allocator
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=full_kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,                 # ← KV 是锚定池
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,    # SWA 用独立分配器
            device_free_fn=swa_attn_allocator.free,
        ),
    ]
    # ② HostPoolGroup + HybridCacheController
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def _deepseek_v4_num_host_pages(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    swa_page_size: int,
) -> tuple[int, int]:
    allocator = params.token_to_kv_pool_allocator
    device_full_size = getattr(allocator, "size_full", kvcache.size)
    device_full_pages = (device_full_size + page_size - 1) // page_size

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size

    if server_args.hicache_size > 0:
        raise ValueError(
            "DeepSeek V4 HiCache currently does not support --hicache-size; "
            "use --hicache-ratio instead."
        )
    ratio = server_args.hicache_ratio
    full_host_pages = max(int(device_full_pages * ratio), device_full_pages + 1)
    swa_host_pages = max(int(device_swa_pages * ratio), device_swa_pages + 1)
    return full_host_pages, swa_host_pages


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    # TODO(hzh0425): Support PP for deepseek v4 with hicache
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}
    swa_layer_mapping = {
        layer_id: layer_id for layer_id in range(len(kvcache.swa_kv_pool.kv_buffer))
    }

    c4_layer_mapping = {}
    c128_layer_mapping = {}
    c4_state_global_layers = []
    c128_state_global_layers = []
    for layer_id, layer_item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        if layer_item.compress_ratio == 4:
            c4_layer_mapping[layer_id] = layer_item.compress_layer_id
            c4_state_global_layers.append(layer_id)
        elif layer_item.compress_ratio == 128:
            c128_layer_mapping[layer_id] = layer_item.compress_layer_id
            c128_state_global_layers.append(layer_id)

    c4_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c4_state_global_layers)
    }
    c128_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c128_state_global_layers)
    }
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages(
        params=params,
        server_args=server_args,
        kvcache=kvcache,
        page_size=page_size,
        swa_page_size=kvcache.swa_page_size,
    )

    logical_host_pool = LogicalHostPool(num_host_pages * page_size, page_size)
    swa_host_pool = DeepSeekV4PagedHostPool(
        pool_name=str(PoolName.SWA),
        device_buffers=kvcache.swa_kv_pool.kv_buffer,
        item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,
        num_host_pages=swa_num_host_pages,
        slot_page_size=kvcache.swa_page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=logical_host_pool,
            device_pool=kvcache,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=kvcache.swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]

    if c4_layer_mapping:
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=kvcache.c4_kv_pool.kv_buffer,
            item_bytes=kvcache.c4_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),
            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            item_bytes=(
                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]
                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()
            ),
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
            state_pools=[
                kvcache.compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
            state_pools=[
                kvcache.indexer_compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_pool=c4_host_pool,
                    device_pool=kvcache.c4_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                    host_pool=c4_indexer_host_pool,
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_STATE,
                    host_pool=c4_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                    host_pool=c4_indexer_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    if c128_layer_mapping:
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=kvcache.c128_kv_pool.kv_buffer,
            item_bytes=kvcache.c128_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        # C128 state pool is intentionally not registered with hicache.
        # page_size=256 % 128 == 0, so state pool is not consumed on load.
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C128,
                    host_pool=c128_host_pool,
                    device_pool=kvcache.c128_kv_pool,
                    layer_mapping=c128_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_anchor_sidecar_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    sidecar_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    sidecar_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool)
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=sidecar_pool_name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


# ═══════════════════════════════════════════════════════════════════
# ComponentType → (cache 属性名, component 属性名) 映射表
# _apply_stack_result 通过此表把 Host pool 注入到正确的对象上
# 完整调用链:
#   _apply_stack_result → setattr(cache, attr_name, host_pool)
#                      → setattr(cache.components[ct], component_attr, host_pool)
# ═══════════════════════════════════════════════════════════════════
_COMPONENT_HOST_ATTR: dict[ComponentType, tuple[str, str]] = {
    ComponentType.FULL: ("full_kv_pool_host", "_full_kv_pool_host"),
    ComponentType.SWA: ("swa_kv_pool_host", "_swa_kv_pool_host"),
    ComponentType.MAMBA: ("mamba_pool_host", "_mamba_pool_host"),
}


@dataclass
class StackBuildResult:
    """📦 策略 build() 的统一返回结构 —— 封装 HostPoolGroup + controller + 挂载元数据。

    ╔══════════════════════════════════════════════════════════════════╗
    ║  字段                                                             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  host_pool_group         : HostPoolGroup, 注入 cache.host_pool_group
    ║  cache_controller        : HybridCacheController, 注入 cache.cache_controller
    ║  component_host_pools    : {ComponentType → host_pool},                          ║
    ║                            通过 _COMPONENT_HOST_ATTR 注入各 component             ║
    ║  sidecars                : SidecarPoolSpec[], 注册到 cache.register_sidecar_pool ║
    ║  register_req_to_token_counter: Mamba 需要额外注册 req_to_token 的 transfer counter║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    host_pool_group: HostPoolGroup
    cache_controller: HybridCacheController
    component_host_pools: dict[ComponentType, Any]
    sidecars: list[SidecarPoolSpec] = field(default_factory=list)
    # Mamba state lives in req_to_token_pool, not in kvcache, so its
    # layer_transfer_counter has to be wired separately.
    register_req_to_token_counter: bool = False
    transfer_layer_num: int = 0
    pools_desc: str = ""


class StackStrategy:
    """🎯 HiCache Stack 创建策略基类 —— 根据 kvcache 类型 + components 集合匹配。

    ╔══════════════════════════════════════════════════════════════════╗
    ║  子类实现两个方法:                                                 ║
    ║  matches(kvcache, components) → bool                             ║
    ║    isinstance 判断 kvcache 类型 + components 集合是否匹配           ║
    ║  build(cache, kvcache, params, ...) → StackBuildResult           ║
    ║    调用对应的 build_*_stack 函数, 组装 HostPoolGroup + controller  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        raise NotImplementedError

    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: Any,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        storage_backend: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        enable_storage_metrics: bool = False,
    ) -> StackBuildResult:
        raise NotImplementedError


class _DeepSeekV4Strategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        return isinstance(kvcache, DeepSeekV4TokenToKVPool) and components == {
            ComponentType.FULL,
            ComponentType.SWA,
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
            params=params,
            server_args=server_args,
            kvcache=kvcache,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        sidecars = [
            SidecarPoolSpec(
                pool_name=name,
                indices_from_pool=src,
                hit_policy=(
                    PoolHitPolicy.TRAILING_PAGES
                    if src == PoolName.SWA
                    else PoolHitPolicy.ALL_PAGES
                ),
            )
            for name, src in (
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            )
            if name in host_pool_group.entry_map
        ]
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
            },
            sidecars=sidecars,
            transfer_layer_num=kvcache.end_layer - kvcache.start_layer,
            pools_desc="KV + SWA + DeepSeekV4 sidecars",
        )


class _MambaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        return isinstance(kvcache, HybridLinearKVPool) and components == {
            ComponentType.FULL,
            ComponentType.MAMBA,
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache.full_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=kvcache.use_mla,
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            register_req_to_token_counter=True,
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping),
            pools_desc="KV + MAMBA",
        )


def _swa_layer_mappings(kvcache) -> tuple[dict[int, int], dict[int, int]]:
    """🔧 从 SWAKVPool 的 layers_mapping 拆出 FULL 层和 SWA 层的各自映射。

    📥 kvcache.layers_mapping: {global_layer_id: (local_id, is_swa)}
    📤 (full_layer_mapping, swa_layer_mapping)
       其中 full={gid: lid | not is_swa}, swa={gid: lid | is_swa}
    """
    full = {
        gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if not is_swa
    }
    swa = {gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if is_swa}
    return full, swa


class _SwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components == {ComponentType.FULL, ComponentType.SWA}
        )

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_hybrid_swa_stack(
            params=params,
            server_args=server_args,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=False,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
            },
            transfer_layer_num=len(full_layer_mapping | swa_layer_mapping),
            pools_desc="KV + SWA",
        )


class _DsaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return isinstance(kvcache, DSATokenToKVPool) and components == {
            ComponentType.FULL
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=full_kv_pool,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=full_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=storage_backend,
            use_mla=use_mla,
            override_kv_cache_dim=full_kv_pool.kv_cache_dim,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                full_kv_pool,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=[
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                ),
            ],
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV + INDEXER",
        )


class _PlainKvStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            HybridLinearKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(
            kvcache,
            (SWAKVPool, HybridLinearKVPool, DSATokenToKVPool, DeepSeekV4TokenToKVPool),
        ):
            return False
        return components == {ComponentType.FULL}

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
        host_pool_group, cache_controller = build_kv_only_stack(
            params=params,
            server_args=server_args,
            kv_pool=full_kv_pool,
            full_layer_mapping=full_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=use_mla,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV",
        )


# 策略匹配顺序: 先匹配特殊的 (DeepSeekV4 / Mamba / SWA / DSA),
# 最后 _PlainKvStrategy 作为兜底 (匹配任何单 KV pool)。
# Resolved first-to-last; _PlainKvStrategy is the catch-all fallback.
_STRATEGIES: list[StackStrategy] = [
    _DeepSeekV4Strategy(),
    _MambaStrategy(),
    _SwaStrategy(),
    _DsaStrategy(),
    _PlainKvStrategy(),
]


def register_stack_strategy(strategy: StackStrategy) -> None:
    """🔌 在策略列表头部插入新策略 —— 供下游 fork 注入自定义 (kvcache, components) 组合。

    💡 insert(0): 插入到最前端, 优先于内置策略匹配。
    """
    """Prepend a strategy so downstream forks can plug in (kvcache, components)
    combinations not in the built-in list."""
    _STRATEGIES.insert(0, strategy)


def _select_strategy(kvcache: Any, components: set[ComponentType]) -> StackStrategy:
    """🔍 按序匹配策略列表, 返回第一个 matches() 为 True 的策略。

    ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
    _PlainKvStrategy.build() _SwaStrategy.build() ... 等各子类 build()
      → _select_strategy(kvcache, components)  ← 已被废弃, attach_* 直接写 if/elif 分支

    ⚠️ 当前 attach_hybrid_pool_to_unified_cache 已经直接写了 if/elif 分支,
       策略系统仅用于各子类的策略封装, _select_strategy 用得很少。
    """
    for strategy in _STRATEGIES:
        if strategy.matches(kvcache, components):
            return strategy
    raise AssertionError(
        f"No matching HiCache strategy for kvcache={type(kvcache).__name__}, "
        f"components={sorted(c.name for c in components)}"
    )


def _apply_stack_result(
    cache: UnifiedRadixCache,
    kvcache: Any,
    params: CacheInitParams,
    result: StackBuildResult,
) -> None:
    """🔗 将 StackBuildResult 的内容挂载到 UnifiedRadixCache 实例上。

    ⚙️ 执行以下注入:
      ① cache.host_pool_group = result.host_pool_group
      ② cache.cache_controller = result.cache_controller
      ③ 对每个 ComponentType: 通过 _COMPONENT_HOST_ATTR 注入 host pool
         → cache.{attr} = host_pool
         → cache.components[ct].{component_attr} = host_pool
      ④ 注册 sidecar pools: cache.register_sidecar_pool(spec)
      ⑤ 注册 layer_transfer_counter (DMA 逐层同步用)
      ⑥ Mamba: 额外注册 req_to_token_pool 的 transfer counter
    """
    # ① 注入 HostPoolGroup 和 HybridCacheController 到 cache 实例
    cache.host_pool_group = result.host_pool_group
    cache.cache_controller = result.cache_controller

    # ② 按 ComponentType 注入 host pool 到 cache 和各 component
    for ct, host_pool in result.component_host_pools.items():
        cache_attr, component_attr = _COMPONENT_HOST_ATTR[ct]
        setattr(cache, cache_attr, host_pool)                         # cache.full_kv_pool_host = ...
        setattr(cache.components[ct], component_attr, host_pool)      # component._full_kv_pool_host = ...

    # ③ 注册 sidecar pools (如 INDEXER / DeepSeekV4 C4 等)
    for sidecar in result.sidecars:
        cache.register_sidecar_pool(sidecar)

    # ④ 注册逐层 DMA 同步计数器
    kvcache.register_layer_transfer_counter(result.cache_controller.layer_done_counter)
    # Mamba: req_to_token_pool 也需要 transfer counter (因为 mamba state 不存 kvcache)
    if result.register_req_to_token_counter:
        params.req_to_token_pool.register_layer_transfer_counter(
            result.cache_controller.layer_done_counter
        )

    logger.info(
        "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
        result.pools_desc,
        result.transfer_layer_num,
    )


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str] = None,
    storage_extra_config: Optional[dict] = None,
    storage_prefetch_threshold: int = 256,
) -> None:
    """🔗 为 UnifiedRadixCache 挂载 HiCache Host Pool stack。

    ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
    UnifiedRadixCache.init_hicache()
      → attach_hybrid_pool_to_unified_cache(cache, params, server_args)  ← 当前函数
        ├─ ① kvcache 类型检测 (SWA / Mamba / NSA / DeepSeekV4 / KV-only)
        ├─ ② 调用对应的 build_*_stack() 创建 (HostPoolGroup, HybridCacheController)
        └─ ③ 注入回 cache: host_pool_group, cache_controller, 各 component 的 host pool

    ━━━━━━━━━━━━━━ 2️⃣ SWA 分支详解 ━━━━━━━━━━━━━━
    当 kvcache 是 SWAKVPool (有 full + swa 双 GPU 池) 且 cache 有 FULL + SWA 组件:
    ┌──────────────────────────────────────────────────────────────────┐
    │  ① 拆 layer_mapping                                              │
    │     layers_mapping: {global_id: (local_id, is_swa)}              │
    │     → full_layer_mapping + swa_layer_mapping                     │
    │                                                                  │
    │  ② build_hybrid_swa_stack()                                      │
    │     → HostPoolGroup( [PoolEntry(KV, is_anchor), PoolEntry(SWA)] ) │
    │     → HybridCacheController(token_to_kv_pool_allocator, ...)     │
    │                                                                  │
    │  ③ 注入回 cache 实例                                              │
    │     cache.host_pool_group = host_pool_group                      │
    │     cache.cache_controller = cache_controller                    │
    │     cache.full_kv_pool_host  = host_pool_group.get_pool(KV)      │
    │     cache.swa_kv_pool_host   = host_pool_group.get_pool(SWA)     │
    │     cache.components[FULL]._full_kv_pool_host = ...              │
    │     cache.components[SWA]._swa_kv_pool_host   = ...              │
    │                                                                  │
    │  ④ 驱逐回调: host_swa_evict_fn / device_swa_evict_fn              │
    │     → cache.evict_host(n, ComponentType.SWA)  (驱逐 SWA Host 池)  │
    │     → cache.evict(EvictParams(swa_num_tokens=n)) (驱逐 SWA Device) │
    └──────────────────────────────────────────────────────────────────┘

    ⚠️ SWA 分支 backf_h/swa_kv_pool_host 通过 entry_map[SWA].host_pool 获取,
       是独立于 full_kv_pool_host 的另一个主机内存池。
    """
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    try:
        # ① 获取 GPU 端 KV pool, 按类型分类
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        swa_stack = isinstance(kvcache, SWAKVPool)
        mamba_stack = isinstance(kvcache, HybridLinearKVPool)
        nsa_stack = isinstance(kvcache, NSATokenToKVPool)
        deepseek_v4_stack = isinstance(kvcache, DeepSeekV4TokenToKVPool)

        # ── 阶段1: 类型校验 + 提取公共参数 ──
        if deepseek_v4_stack:
            use_mla = False
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.SWA,
            }, "DeepSeekV4TokenToKVPool requires FULL + SWA in UnifiedRadixCache."
        elif mamba_stack:
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.MAMBA,
            }, "HybridLinearKVPool currently only supports FULL + MAMBA in UnifiedRadixCache."
        elif swa_stack:
            # ═══════════ SWA 分支: kvcache=SWAKVPool, components={FULL, SWA} ═══════════
            # SWAKVPool 内部持有 full_kv_pool 和 swa_kv_pool 两个 GPU 子池,
            # 对外通过 layers_mapping 按 layer_id 路由到正确的子池
            full_kv_pool = kvcache.full_kv_pool
            use_mla = False
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.SWA,
            }, "SWAKVPool currently only supports FULL + SWA in UnifiedRadixCache."
        else:
            full_kv_pool = kvcache
            use_mla = isinstance(kvcache, MLATokenToKVPool)
            assert set(cache.components.keys()) == {
                ComponentType.FULL
            }, "Non-hybrid KV pool currently only supports FULL-only UnifiedRadixCache."

        if deepseek_v4_stack:
            host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
                params=params,
                server_args=server_args,
                kvcache=kvcache,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                device_swa_evict_fn=lambda n: cache.evict(
                    EvictParams(swa_num_tokens=n)
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.swa_kv_pool_host = host_pool_group.get_pool(PoolName.SWA)
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                cache.swa_kv_pool_host
            )
            for pool_name, indices_from_pool in (
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            ):
                if pool_name in host_pool_group.entry_map:
                    cache.register_sidecar_pool(
                        SidecarPoolSpec(
                            pool_name=pool_name,
                            indices_from_pool=indices_from_pool,
                        )
                    )
            transfer_layer_num = kvcache.end_layer - kvcache.start_layer
        elif mamba_stack:
            full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
            mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
            host_pool_group, cache_controller = build_hybrid_mamba_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                mamba_pool=params.req_to_token_pool.mamba_pool,
                full_layer_mapping=full_layer_mapping,
                mamba_layer_mapping=mamba_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
            cache.components[ComponentType.MAMBA]._mamba_pool_host = (
                cache.mamba_pool_host
            )
            params.req_to_token_pool.register_layer_transfer_counter(
                cache_controller.layer_done_counter
            )
            transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        elif swa_stack:
            # ═══════════ SWA 分支: 构建 FULL+SWA 双池 + 挂载 ═══════════

            # ① 从 SWAKVPool.layers_mapping 拆分 FULL 层和 SWA 层的各自映射
            #    layers_mapping: {global_layer_id: (local_pool_id, is_swa_bool)}
            full_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if not is_swa
            }
            swa_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if is_swa
            }
            # ② 调用 build_hybrid_swa_stack 创建 HostPoolGroup + HybridCacheController
            #    → HostPoolGroup([PoolEntry(KV, anchor), PoolEntry(SWA, sidecar)])
            #    → kv_host_pool 和 swa_host_pool 是两个独立的物理内存池
            host_pool_group, cache_controller = build_hybrid_swa_stack(
                params=params,
                server_args=server_args,
                full_kv_pool=full_kv_pool,
                swa_kv_pool=kvcache.swa_kv_pool,
                full_layer_mapping=full_layer_mapping,
                swa_layer_mapping=swa_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=False,
                # 驱逐回调: 将 cache 的 evict_host/evict 方法包装为 lambda
                # host_swa_evict_fn: 驱逐 SWA Host 池 (host_pool_group entry_map[SWA].host_pool)
                host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                # device_swa_evict_fn: 驱逐 SWA GPU 池 (swa_attn_allocator 管理的 slot)
                device_swa_evict_fn=lambda n: cache.evict(
                    EvictParams(swa_num_tokens=n)
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            # ③ 注入 HostPoolGroup + HybridCacheController 到 cache 实例
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            # ③a Full component: 注入 full_kv_pool_host
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            # ③b SWA component: 注入 swa_kv_pool_host (独立内存, 通过 entry_map[SWA] 访问)
            cache.swa_kv_pool_host = host_pool_group.get_pool(PoolName.SWA)
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                cache.swa_kv_pool_host
            )
            # 总 DMA 层数 = FULL 层数 ∪ SWA 层数
            transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
        elif nsa_stack:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_anchor_sidecar_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                sidecar_pool_name=PoolName.INDEXER,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                override_kv_cache_dim=full_kv_pool.kv_cache_dim,
                sidecar_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                    full_kv_pool,
                    kv_host_pool,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.register_sidecar_pool(
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                )
            )
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)
        else:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)

        kvcache.register_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )

        if deepseek_v4_stack:
            pools_desc = "KV + SWA + DeepSeekV4 sidecars"
        elif mamba_stack:
            pools_desc = "KV + MAMBA"
        elif swa_stack:
            pools_desc = "KV + SWA"
        elif nsa_stack:
            pools_desc = "KV + INDEXER"
        else:
            pools_desc = "KV"
        logger.info(
            "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
            pools_desc,
            transfer_layer_num,
        )
        _apply_stack_result(cache, kvcache, params, result)
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_hybrid_dsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's DSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            page_size=radix_cache.page_size,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=radix_cache.pp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_dsa_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_pool_to_mamba_cache(
    mamba_cache: HiMambaRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup (KV + Mamba) + HybridCacheController for HiMambaRadixCache.

    This entrypoint is currently intended only for HiMambaRadixCache.
    """
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=hybrid_kv.use_mla,
            host_mamba_evict_fn=mamba_cache.evict_mamba_host,
            device_mamba_evict_fn=mamba_cache.evict_mamba,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        mamba_cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
        mamba_cache.transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Attached hybrid Mamba pool stack to HiMambaRadixCache: pools=KV + MAMBA, "
            "transfer_layer_num=%s",
            mamba_cache.transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_mamba_cache failed")
        raise
