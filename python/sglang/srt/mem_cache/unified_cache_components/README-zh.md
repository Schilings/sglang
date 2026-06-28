# Unified Radix Cache（统一 Radix 缓存）

SGLang 的基于组件的、可插拔的前缀缓存框架，将 Full-attention、Sliding-Window-Attention (SWA) 和 Mamba/SSM 三种缓存统一到一棵 radix tree 中。

## 设计目标

1. **统一树结构** — 一棵 radix tree 管理所有 KV cache 类型，不再需要分别实现 `SWARadixCache`、`MambaRadixCache` 等。
2. **可插拔组件** — 每种注意力/状态类型（Full、SWA、Mamba）都是一个实现了 hook 接口的 `TreeComponent`。添加新缓存类型只需新增一个组件。
3. **组件级资源隔离** — 每个组件拥有独立的锁引用计数、可驱逐/受保护大小追踪和驱逐驱动器。辅助组件使用各自的 LRU 链表；Full 组件使用叶子集合。
4. **带优先级的级联驱逐** — 一个组件驱逐节点时，同节点上优先级 ≤ 当前组件的其他组件一起被驱逐，保持跨组件一致性。
5. **主树零特判** — 树只操作 key（逻辑层）。所有物理资源管理（分配、释放、写时复制）都由组件通过 hook 完成。

## 架构

```
┌───────────────────────────────────────────────┐
│              UnifiedRadixCache                │
│            (unified_radix_cache.py)           │
│                                               │
│  root_node ──► UnifiedTreeNode (radix tree)   │
│  components ► {ComponentType → TreeComponent} │
│  lru_lists ─► {ComponentType → UnifiedLRUList}│
└──────────┬───────────┬───────────┬────────────┘
           │           │           │
           ▼           ▼           ▼
   ┌────────────┐ ┌──────────┐ ┌─────────────┐
   │    Full    │ │   SWA    │ │    Mamba    │
   │ Component  │ │Component │ │  Component  │
   └─────┬──────┘ └────┬─────┘ └──────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
               ┌──────────────┐
               │TreeComponent │
               │    (ABC)     │
               └──────────────┘
```

## 与 Scheduler 的调用链

```
scheduler.get_next_batch_to_run()
  ├─ match_prefix(key)  → _match_prefix_helper
  │     ├─ 遍历 radix tree 节点
  │     └─ 每个节点调所有组件的 create_match_validator() 闭包
  ├─ inc_lock_ref(last_node) → acquire_component_lock() 每组件
  ├─ alloc / evict → cascade eviction
  │     ├─ Full: 驱逐叶子集合 → 级联 SWA + Mamba
  │     └─ SWA: tombstone 内部节点 / 删除叶子
  └─ run_batch(prefill/decode)

scheduler.process_batch_result()
  ├─ cache_unfinished_req() → insert + re-match + 锁交换
  └─ cache_finished_req()  → insert + dec_lock_ref + free
```

### 关键数据结构

**`UnifiedTreeNode`** — 每个节点独立存储各组件数据：

```python
node.component_data[ComponentType.FULL]   # FullComponent 数据
node.component_data[ComponentType.SWA]    # SWAComponent 数据
node.component_data[ComponentType.MAMBA]  # MambaComponent 数据
```

**`UnifiedLRUList`** — 每个辅组件一个双向链表，通过 `lru_prev[component_type]` / `lru_next[component_type]` 穿在同一批树节点上。Host LRU 使用独立的指针槽位，与 Device LRU 不冲突。支持 O(1) 插入/删除/提升，驱逐扫描 O(L)（L = 跳过被锁节点数）。Full 组件驱逐由 `evictable_device_leaves` / `evictable_host_leaves` 叶子集合驱动，不使用 Full LRU。

**`ComponentData`** — 每个节点上各组件的数据：
- `value: Tensor | None` — 设备端索引（Full 用 `TokenToKVPool`，SWA 用 `SWAKVPool`，Mamba 用 `MambaPool`）。`None` 表示 tombstone（数据已驱逐但节点结构保留）。
- `lock_ref: int` — 活跃请求对该节点组件数据的引用计数。`lock_ref > 0` 保护节点不被驱逐。
- `metadata: dict` — 组件特定状态（如 SWA 存储 `component_uuid` 用于窗口锁边界追踪）。
- `host_value: Tensor | None` — HiCache 备份后的 Host 端索引。
- `host_lock_ref: int` — 保护 Host 端数据的引用计数。

---

## 文件布局

| 文件 | 内容 |
|------|------|
| `../unified_radix_cache.py` | `UnifiedRadixCache`、`UnifiedTreeNode`、`UnifiedLRUList` |
| `tree_component.py` | `TreeComponent` ABC、`ComponentType`、`ComponentData` |
| `full_component.py` | `FullComponent` — 标准全注意力 KV cache 组件 |
| `swa_component.py` | `SWAComponent` — 滑动窗口注意力组件（tombstone/窗口追踪） |
| `mamba_component.py` | `MambaComponent` — Mamba/SSM 状态组件（写时复制） |
| `../hybrid_cache/hybrid_cache_controller.py` | `HybridCacheController` — HiCache 多池控制器 |
| `__init__.py` | 重导出所有公共类型 |

---

## 公共 API 参考

所有公共 API 都在 `UnifiedRadixCache` 上，实现 `BasePrefixCache` 接口。

**符号约定**：K = key 长度（token 数），D = 树中匹配路径深度（D ≤ K/P），P = page_size，C = 组件数（≤ 3，当作常量）。

所有树遍历操作的成本包含两部分：**O(K)** 数据操作（key 比较、tensor clone/concat）+ **O(D·C)** 组件开销（每个节点 C 个 hook）。由于 D ≤ K/P 且 C 为常量，总体 **O(K)**。

### `match_prefix(params) → MatchResult`

在树中查找最长前缀匹配。

| 项目 | 说明 |
|------|------|
| **目的** | 遍历 radix tree，找到**所有**组件验证器都通过的最长前缀 |
| **输入** | `params.key: RadixKey` — token ID + 可选的命名空间隔离 key |
| **输出** | `MatchResult(device_indices, last_device_node, last_host_node, ...)` |
| **副作用** | 更新 `last_access_time`；提升匹配路径到所有组件 LRU 的 MRU 端；可能触发 `_split_node` |
| **复杂度** | **O(K + D·C)** |

**算法详情**：
1. 对每个组件调用一次 `create_match_validator(match_device_only=...)` — 返回有状态的闭包（如 SWA 追踪累计窗口长度）。HiCache 模式下同时追踪最佳纯设备节点和最佳设备或 Host 节点。
2. 通过 `RadixKey.match()` 沿着树边走；每到一节点，调用所有验证器闭包 — 只有**所有**验证器返回 `True` 时才推进匹配边界。
3. 若匹配终止于节点中间，调用 `_split_node` → 触发每个组件的 `redistribute_on_node_split()`。
4. 后处理（`_match_post_processor`）：
   - 通过 `node_has_component_data()` 将匹配路径提升到各组件 LRU 的 MRU 端
   - 沿路径向上递减更新 `last_access_time`（parent < child）
   - 拼接匹配到的设备索引 → `torch.cat`
   - 调用各组件 `finalize_match_result()`（Mamba 执行写时复制：分配新槽位、复制 SSM 状态）

### `insert(params) → InsertResult`

将 key-value 插入树中。

| 项目 | 说明 |
|------|------|
| **目的** | 插入 token 序列 + KV 索引，复用已有前缀并释放重复 KV 槽位 |
| **输入** | `params.key`、`params.value`（KV 池索引）+ 组件特定字段 |
| **输出** | `InsertResult(prefix_len, mamba_exist)` |
| **副作用** | 创建新叶节点；更新重叠节点组件数据；释放重复 KV 索引；可能分裂节点 |
| **复杂度** | **O(K + D·C)** |

**算法详情**（`_insert_helper`）：
1. 对每个已有节点调用 `_touch_node` → 通过 `node_has_component_data()` 提升到 MRU。
2. 若 key 偏离节点中间，调用 `_split_node` → 每个组件 `redistribute_on_node_split()`。
3. 对每个重叠节点，调用 `update_component_on_insert_overlap()` — 返回 `consumed_from` 索引；树释放 `value[dup_start:consumed_from]` 作为重复池索引。
   - Full：返回 `prefix_len`（不消费，默认行为）。
   - SWA：检查重叠节点是否是 SWA 窗口边界内的 tombstone：
     - 完全在窗口内：**复活 tombstone** — 释放旧 Full value、克隆 value_slice、转换到 SWA 索引、插入 SWA LRU（返回 0 = 全部消费）。
     - 部分在窗口内：在边界处分裂节点，在窗口段上恢复 SWA（返回 `start_idx`）。
     - 完全在窗口外：返回 `prefix_len`（不消费）。
   - Mamba：返回 `prefix_len`（不消费，默认行为）。

### `evict(params) → EvictResult`

释放缓存 token 回收内存。

| 项目 | 说明 |
|------|------|
| **目的** | 每个组件从自己的 LRU 链表驱逐直到各自目标量达成 |
| **输入** | `params.num_tokens`（full）、`swa_num_tokens`（SWA）、`mamba_num`（Mamba） |
| **输出** | `EvictResult(num_tokens_evicted, swa_num_tokens_evicted, mamba_num_evicted)` |
| **副作用** | 释放池索引；从 LRU 链表移除；级联低优先级组件；向上遍历墓碑祖先 |
| **复杂度** | **O(E·H + L)** — E = 驱逐节点数，H = 墓碑链高度，L = LRU 扫描时跳过的锁定节点数 |

**级联驱逐规则**：
- **叶节点**：所有优先级 = 0 → 驱逐任意组件级联所有
- **内部节点**：Full(2) > SWA(1) > Mamba(0)
  - 驱逐 Mamba：不级联
  - 驱逐 SWA：级联到 Mamba
  - 驱逐 Full：级联到 SWA + Mamba

### `inc_lock_ref(node) → IncLockRefResult`

锁定节点路径，防止驱逐。

| 组件 | 锁策略 |
|------|--------|
| Full | **路径锁**：从 node 到 root，每个祖先 `lock_ref += 1` |
| SWA | **窗口锁**：向上累加 SWA value 长度，直到 `sliding_window_size` 填满，在边界节点记录 `component_uuid` |
| Mamba | **单节点锁**：仅 `lock_ref += 1` 于 node 自身 |

### `dec_lock_ref(node, params?) → DecLockRefResult`

解锁之前锁定的节点路径。对称于 `inc_lock_ref`。

### `cache_finished_req(req)`

请求完成后缓存 KV 入树。复杂度 **O(K)**。调用链：`prepare_for_caching_req()` → `insert()` → `dec_lock_ref()` → `cleanup_after_caching_req()`。

### `cache_unfinished_req(req)`

chunked prefill 时缓存中间结果。复杂度 **O(K)**。两次树遍历：`insert()` + `match_prefix()`，然后锁交换 `dec_lock_ref()` + `inc_lock_ref()`。

---

## TreeComponent Hook 参考

### 匹配阶段

| Hook | 作用 | 调用者 |
|------|------|--------|
| `create_match_validator()` | 返回状态化闭包，判断节点是否有效匹配边界 | `_match_prefix_helper` |
| `finalize_match_result()` | 前缀匹配完成后的后处理。Mamba：写时复制分配新槽位 | `_match_post_processor` |

### 插入阶段

| Hook | 作用 | 调用者 |
|------|------|--------|
| `update_component_on_insert_overlap()` | 处理与已有节点的重叠。SWA：可能在窗口边界内复活 tombstone | `_insert_helper` |
| `should_skip_leaf_creation()` | 否决新叶创建（当整个新叶对该组件都是 tombstone 时） | `_insert_helper` |
| `commit_insert_component_data()` | 插入遍历完成后在目标节点上最终确定组件数据 | `_insert_helper` |

### 驱逐阶段

| Hook | 作用 | 调用者 |
|------|------|--------|
| `evict_component()` | 释放节点上该组件的资源。内部驱逐做 tombstone，Host 驱逐清 `host_value` | cascade |
| `eviction_priority()` | 返回级联驱逐优先级（越高越后驱逐）。内部节点：Full(2)>SWA(1)>Mamba(0) | `_cascade_evict` |
| `drive_eviction()` | 驱动设备端驱逐直到目标量。Full 用叶子堆，SWA/Mamba 用各自 LRU | `evict` |

### 锁阶段

| Hook | 作用 | 调用者 |
|------|------|--------|
| `acquire_component_lock()` | 增加设备/Host 锁引用数 | `inc_lock_ref` |
| `release_component_lock()` | 减少设备/Host 锁引用数 | `dec_lock_ref` |

### 缓存阶段

| Hook | 作用 | 调用者 |
|------|------|--------|
| `prepare_for_caching_req()` | cache 前准备组件特定数据 | `cache_finished/unfinished_req` |
| `cleanup_after_caching_req()` | cache 后清理 | `cache_finished/unfinished_req` |

### 工具方法

| Hook | 作用 | 调用者 |
|------|------|--------|
| `build_hicache_transfers()` | 构建 HiCache 传输描述符 | HiCache 路径 |
| `node_has_component_data()` | 检查节点是否有该组件的设备/Host 数据 | 多处 |

---

## 组件行为对比

| 行为 | FullComponent | SWAComponent | MambaComponent |
|------|--------------|-------------|----------------|
| **验证器** | Full 设备数据，HiCache 时可匹配 Host 备份 | 追踪累计窗口；达到 sliding_window_size 时返回 True | Mamba 设备数据，HiCache 时可匹配 Host 备份 |
| **锁策略** | 路径锁（root → node） | 窗口锁（到窗口边界，UUID 标记） | 单节点锁 |
| **内部驱逐优先级** | 2（最后） | 1（中间） | 0（最先） |
| **分裂行为** | 复制 lock_ref 到 parent | 切分 SWA value + 复制 UUID | parent 得 None（Mamba 留在叶上） |
| **匹配后处理** | 无操作 | 无操作 | 写时复制：分配新 Mamba 槽位，复制状态 |
| **驱逐驱动** | Full 叶子集合 → 级联全部 | SWA LRU → tombstone 内部 / 级联叶 | Mamba LRU → tombstone 内部 / 级联叶 |

---

## 构建方式

当 `SGLANG_ENABLE_UNIFIED_RADIX_TREE` 启用时，`UnifiedRadixCache` 由 `mem_cache/registry.py` 直接构造。注册表在构造前设置 `params.tree_components`：

- 普通全注意力模型 → `(ComponentType.FULL,)`
- Hybrid SWA 模型 → `(ComponentType.FULL, ComponentType.SWA)`
- Hybrid SSM/Mamba 模型 → `(ComponentType.FULL, ComponentType.MAMBA)`

启用分层缓存时，注册表在构造后调用 `cache.init_hicache(server_args, params)`。
