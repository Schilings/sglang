# 注释风格完整范例

样板取自 SGLang `python/sglang/srt/mem_cache/hiradix_cache.py` 与 `memory_pool.py`。动手前对照本文件统一风格。

---

## ① 类级宏观 docstring（对外接口 + 框架交互 + 调用链图）

放在 `class X(...):` 正下方，用 `"""..."""` 包裹。覆盖：🧩 对外接口清单 → 🔗 框架如何调用这些接口 → 🧬 设计要点 → 🖼️ 图。

⚠️ **禁止把这类宏观注释写成文件顶部的 `#` 大框头注释**——哪个类的注释就放在哪个类处。box-drawing 图放在 `"""` 内部，不用 `#`。

```python
class MLATokenToKVPool(KVCache):
    """💾 MLA KV Cache Pool —— Multi-Head Latent Attention 模型 (DeepSeek-V2/V3) 的 GPU KV 缓存。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（框架通过这些方法使用本类）                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  set_mla_kv_buffer(layer, loc, k_nope, k_rope)  ✍️ 前向时写入一层 latent KV          ║
    ║  get_key_buffer(layer_id) / get_mla_kv_buffer   📖 attention 计算时读取 KV           ║
    ║  move_kv_cache(tgt, src)                         🚚 投机解码接受后搬运 KV             ║
    ║  get_contiguous_buf_infos()                      🌐 PD 分离时暴露 buffer 给传输引擎   ║
    ║  get_cpu_copy / load_cpu_copy                    💿 KV 换出 / 换入 host              ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 框架如何与本类交互（宏观调用链）                                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ════════ ① 创建链（启动期，仅一次）════════                                         ║
    ║  ModelRunner.__init__                                                             ║
    ║    └─ ModelRunnerKVCacheMixin → MLATokenToKVPool(...)  本类                        ║
    ║       └─ PagedTokenToKVPoolAllocator(kvcache=self)  ← allocator 持有本 pool         ║
    ║          └─ RadixCache / HiRadixCache(params)       ← 前缀缓存树持有 allocator      ║
    ║                                                                                  ║
    ║  ════════ ② 调度 + 分配（Scheduler 每轮，只给"位置"不写数据）════════                  ║
    ║  Scheduler.get_next_batch_to_run                                                  ║
    ║    └─ RadixCache.match_prefix(key)  命中前缀 → 复用已有 slot                        ║
    ║    └─ allocator.alloc(need)         未命中 → 分配新 slot → 写 ReqToTokenPool         ║
    ║                                                                                  ║
    ║  ════════ ③ 前向写 / ④ 前向读（每层 forward）════════                                ║
    ║  AttentionBackend.forward_*                                                       ║
    ║    └─ set_mla_kv_buffer(...)  ✍️ 写  ← flashinfer_mla / trtllm_mla / dsa ...        ║
    ║    └─ get_key_buffer(...)     📖 读  + forward_batch.kv_indices 做 paged gather     ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    🧬 设计要点：MLA 把多头 KV 压成一份低秩潜变量缓存（单 buffer，head 维=1），
       显存约为 MHA 的 1/10；K/V 共享同一块内存，attention 内部再上投影还原。
    """
```

要点：框首行 = Emoji + 标题；接口框逐条列方法 + Emoji + 一句话职责；调用链框分编号段 `════ ① xxx ════`，每步 `├─ └─ →` 标对端文件 / 函数。

---

## ② 函数级 docstring（调用链定位 + 详解）+ 行间注释

每个函数 docstring 紧随 `def`，覆盖：一句话职责 → 🔗 调用链定位 → 📥 参数 / 📤 返回 / ⚙️ 行为 / ⚠️ 注意；之后再写行间注释。

### ⭐ 调用链格式（学习 `hiradix_cache.py`）

**核心原则：从最外层调用方开始写，多跳 `→` 链到当前函数，严禁简化单跳。**

#### A. 简单函数调用链（单场景）

```
━━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
scheduler 主循环 → check_hicache_events() → loading_check()
load_back → cache_controller.load() → 入 load_queue
cache_controller.start_loading() → 逐层 DMA（load_to_device_per_layer × N）
  → 每层完成后 record finish_event → 入 ack_load_queue
loading_check 轮询 ack_load_queue → 收割每层已完成的事件 → dec_lock_ref
```

#### B. 多场景函数调用链（用 box 分列）

```python
    def write_backup(self, node: TreeNode, write_back=False) -> int:
        """将节点的 KV 从 GPU 写入 Host（GPU→Host DMA）。

        ╔══════════════════════════════════════════════════════════════════╗
        ║  write_backup 的两种触发场景                                       ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║                                                                  ║
        ║  1. write_through 模式（write_back=False）：                      ║
        ║     节点命中次数达标时自动触发，将 KV 主动写入 Host                    ║
        ║     调用链：_inc_hit_count → write_backup(node) → inc_lock_ref     ║
        ║            [DMA 飞行中] → writing_check → _finish_write_through_ack ║
        ║            → dec_lock_ref + write_backup_storage（写三层存储）         ║
        ║     前置约束：parent 必须 backuped（保证从 root 到当前节点连续备份，无空隙）║
        ║                                                                  ║
        ║  2. write_back 模式（write_back=True）：                         ║
        ║     evict 时调用，先把 KV 备份到 Host，再释放 GPU                   ║
        ║     调用链：evict → write_backup(write_back=True) → _evict_backuped  ║
        ║           evict 调用链展开：                                       ║
        ║           scheduler.run_batch()                                  ║
        ║             → prepare_for_decode/extend (schedule_batch.py)       ║
        ║               → alloc_token_slots (common.py)                     ║
        ║                 → evict_from_tree_cache(tree_cache, N) (common.py)║
        ║                   → tree_cache.evict(EvictParams(...)) ← 你在这   ║
        ║     无 parent 约束（evict 是被动触发的，不保证连续性）                 ║
        ║     写完不 inc_lock_ref（马上就要 _evict_backuped）                  ║
        ╚══════════════════════════════════════════════════════════════════╝

        写入成功后：
          node.host_value = host_indices  ← 记录 Host 端 slot 索引
          node.backuped → True
          后续可通过 load_back 从 Host 恢复到 GPU

        Host 内存不足时：
          先调 evict_host 释放其他节点的 Host KV，再重试一次
        """
```

#### C. 带 💡 设计理由的调用链

```python
    def loading_check(self):
        """🔍 轮询并收割已完成的 Host→GPU DMA 加载（load_back 的异步收尾）。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler 主循环 → check_hicache_events() → loading_check()

        ━━━━━━━━━━━━━━ 2️⃣ 为什么 load 是逐层而 write 是全层？━━━━━━━━━━━━━━
          📤 start_writing: backup_from_device_all_layer 全层一次 → 无需等计算流
          📥 load_back: 必须逐层 load_to_device_per_layer × N
             → 因为 GPU 逐层 forward 时，每层计算完马上需要该层的 KV
             → 一层加载完即可打断供计算，不用等所有层全好
        """
```

### 多跳调用链完整范例（真实 evict 函数，hiradix_cache.py:1637 风格）

```python
    def evict(self, params: EvictParams) -> EvictResult:
        """🗑️ 驱逐 GPU 端 KV 以释放显存 —— write_back / CPU offload 的核心触发点。

        ━━━━━━━━━━━━━━ 1️⃣ 完整调用链（从 scheduler 到 evict）━━━━━━━━━━━━━━
        scheduler.run_batch()
          → prepare_for_decode / prepare_for_extend       (schedule_batch.py)
            → alloc_for_decode / alloc_for_extend         (common.py)
              → alloc_token_slots / alloc_paged_token_slots_extend  (common.py:272)
                → evict_from_tree_cache(tree_cache, N)    (common.py:302)
                  → tree_cache.evict(EvictParams(num_tokens=N))   ← 你在这
        💡 scheduler 在每次 alloc KV slot 之前检查显存。若不够，
        先 evict 腾空间，再 alloc。这是 write_back / CPU offload 的核心触发点。

        ━━━━━━━━━━━━━━ 2️⃣ write_back vs write_through 驱逐行为区别 ━━━━━━━━━━━━━━
        ┌─────────────┬──────────────────────────────┬───────────────────────────────┐
        │ 策略         │ write_back                    │ write_through                  │
        ├─────────────┼──────────────────────────────┼───────────────────────────────┤
        │ 备份时机      │ evict 时被动触发              │ 命中即主动写（达到阈值后）       │
        │ evict 行为    │ write_backup → _evict_backuped│ 直接 _evict_backuped（已有备份）│
        │ DMA 同步方式  │ writing_check(write_back=True)│ writing_check(write_back=False)│
        │              │ → synchronize() 阻塞等        │ → query() 非阻塞轮询           │
        │ parent 约束  │ 无（被动驱逐，不保证连续）      │ 有（parent 必须先 backuped）    │
        └─────────────┴──────────────────────────────┴───────────────────────────────┘

        ⚙️ 行为：
        ① 遍历 device LRU 链表，找到可驱逐的叶子节点
        ② write_back 策略：先 write_backup(DMA) → _evict_backuped
           write_through 策略：直接 _evict_backuped（数据已写入 Host）
        ③ 累计驱逐的 token 数，直到满足 num_tokens 或显存够用
        ④ 返回 EvictResult（含实际驱逐 token 数、耗时等）

        📤 返回：EvictResult(evicted_tokens=N, elapsed=...)
        """
```

#### D. 数据结构的生命周期注释（类属性 / dict / 集合等）

```python
        # ═══════════════════════════════════════════════════════════════════
        # GPU→Host 写回追踪
        # key:   ack_id (node.id)
        # value: (lock_node, backup_len, publish_nodes)
        #   - lock_node:     持有读锁的节点(DMA期间防止KV被覆盖)
        #   - backup_len:    写回的token长度(用于split后恢复)
        #   - publish_nodes: DMA完成后标记为CPU_READY的节点列表
        # 完整调用链:
        #   创建: write_backup() → _track_write_through_node()
        #   更新: radix tree split → _replace_pending_write_through_node()
        #   清除: scheduler writing_check() → _finish_write_through_ack() → pop + dec_lock_ref
        self.ongoing_write_through = {}

        # ═══════════════════════════════════════════════════════════════════
        # Host→GPU 加载追踪
        # key:   ack_id (last_hit_node.id)
        # value: last_hit_node (加载任务对应的radix tree叶子节点)
        # 完整调用链:
        #   创建: scheduler → load_back() → ongoing_load_back[last_hit_node.id] = ...
        #   清除: scheduler loading_check() → pop + dec_lock_ref(释放读锁)
        self.ongoing_load_back = {}
```

要点：docstring 讲"是什么 / 在调用链何处 / 怎么用"；行间注释讲"为什么"，块首标编号（`# ③ 写 KV 链`）。调用链必须从最外层跳到当前函数，中间**至少给出 2-3 跳**，禁止 `└─ 当前函数 ← 调用方` 这种空洞单跳格式。

---

## ③ 多实现类对比（基类有多个场景化实现）

子类 docstring 要对比"主线实现"，并说明为何此场景用它。下例为 `HiRadixCache` 对 `RadixCache` 的对比（节选自 `hiradix_cache.py`）。

```python
class HiRadixCache(RadixCache):
    """分层 Radix Cache —— 在 RadixCache 基础上增加 GPU↔Host↔Storage 三级 KV 存储。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与 RadixCache 的关键区别：节点三态 vs 两态                                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  RadixCache 节点两态：                                                             ║
    ║    ┌──────────┐   evict   ┌──────────┐                                           ║
    ║    │ 在 GPU   │ ────────▶ │ 删除节点 │   value=None 即节点没了                      ║
    ║    └──────────┘           └──────────┘                                           ║
    ║  HiRadixCache 节点四态（evict 不是删除，而是"降级"，仍在树中可 load_back 恢复）：       ║
    ║    ┌──────┐ insert ┌──────────┐ evict ┌──────┐ backup ┌─────────┐                  ║
    ║    │ GPU  │ ─────▶ │ GPU+Host │ ────▶ │ Host │ ─────▶ │ Storage │                  ║
    ║    └──────┘        └──────────┘       └──────┘        └─────────┘                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🎯 为什么用本类：开启 hierarchical cache（--enable-hierarchical-cache）时，          ║
    ║     被驱逐的 KV 可降级到 Host / Storage 而非丢弃，命中后从 Host 恢复无需重算，         ║
    ║     适合长上下文 / 高前缀复用、GPU 显存吃紧的场景。                                    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
```

要点：先 🔑 列关键区别（配状态流转图 / 对比表），再 🎯 说明触发条件与适用场景。

---

## ④ 常用 ASCII 图速查

```
调用链树：   A.foo                          数据流向：   GPU ──DMA──▶ Host ──backup──▶ Storage
               └─ B.bar  ← 调用方                          ▲                          │
                  └─ C.baz                                  └────────load_back─────────┘

状态流转：   ┌──────┐  evict   ┌──────┐     对比表格：┌────────┬──────────┬──────────┐
             │ 在GPU│ ───────▶ │ 降级 │                │        │ write_through │ write_back │
             └──────┘          └──────┘                ├────────┼──────────┼──────────┤
                                                       │ 备份时机│ 命中即写  │ 驱逐才写  │
时序：  Iter N(prefill): ① match → ② alloc → ③ 写KV    └────────┴──────────┴──────────┘
        Iter N+1(decode): ① 收割 → ② alloc → run_batch
```

对齐做到大致整齐即可，信息准确性优先。
