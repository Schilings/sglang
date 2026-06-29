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

```python
    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        """✍️ 写入一层的 MLA latent KV（nope + rope 分段融合写）。

        🔗 调用链定位（③ 前向写 KV）：
            AttentionBackend.forward_extend / forward_decode
              └─ set_mla_kv_buffer(...)  ← 当前函数
                 调用方：flashinfer_mla / trtllm_mla / dsa_backend / ...

        📥 参数：
            layer        : 当前 attention 层，用 layer.layer_id 定位 buffer
            loc          : 写入目标 slot 索引（= forward_batch.out_cache_loc）
            cache_k_nope : 压缩潜变量段 [n, 1, kv_lora_rank]
            cache_k_rope : 解耦位置段   [n, 1, qk_rope_head_dim]
        📤 返回：无（原地写入 self.kv_buffer）。
        ⚙️ 行为：按平台 / dtype 分三条路径（HIP-fp8 / DSA-fp8 / 常规），均由 triton
            内核把两段融合写进同一行，省去上游 concat。
        ⚠️ 注意：loc 越界会被 maybe_detect_oob 拦截，避免污染哨兵尾巴。
        """
        # ③ 写 KV 链：先做越界探针，再按分支写入
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id
        # 分支一：ROCm(HIP) + DSA + fp8——原始 (nope|rope) 布局，写入时顺带量化成 FP8
        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            ...
```

要点：docstring 讲"是什么 / 在调用链何处 / 怎么用"；行间注释讲"为什么"，块首标编号（`# ③ 写 KV 链`）。

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
