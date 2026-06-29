---
name: code-commenter
description: 按团队约定的"图文并茂 + 调用链"富注释风格为代码添加中文注释。为重要类写宏观 docstring（对外接口清单、框架如何与之交互、宏观调用链图），为每个函数写 docstring（调用链定位 + 详解）并辅以行间注释，用 box-drawing 图与语义化 Emoji 做分层和强调，有真实可比的同类时才横向对比关键差异（无可比时不硬凑），且绝不污染原代码。当用户要求"写注释/加注释/注释这个类或函数/按团队风格注释/comment/annotate"时使用。
---

# Code Commenter（团队代码注释规范）

## Overview

为代码补充三个层次、图文并茂的中文注释：**类级宏观 docstring → 函数级 docstring → 行间注释**。注释要讲清"对外接口、框架如何与之交互、调用链、为什么这样设计"，用 box-drawing 图与 Emoji 增强可读性和层次感。最高铁律：**只加注释，绝不污染（修改）原代码**。范式参考 SGLang `hiradix_cache.py` / `memory_pool.py`。

## When To Use

- 为某个类 / 函数 / 文件写注释、补注释、按团队风格注释。
- 希望注释体现对外接口、模块间交互、调用链、设计意图。
- 仅当存在真实可比的同类（如不同硬件 / 精度的变体）时才做横向差异对比——否则不硬凑。

## Workflow

按顺序执行，**每一步都是强制要求，不可跳过**。

1. **探索（动手前必做）**：读目标代码及其基类 / 关键依赖；用 search/grep 或 code-explorer 查清——谁创建本对象、谁调用本函数、对外暴露哪些接口、与哪些模块交互（scheduler / cache / attention / spec / disagg / offload 等），记录对端的文件 + 函数 + 行号。**调用链必须基于真实代码，严禁编造。**
2. 写**类级宏观 docstring**（见下"① 类级注释"）。
3. 为每个函数写**函数级 docstring**（见下"② 函数级注释"）。
4. **⚠️ 必做：为每个函数体内部写行间注释**（见下"③ 行间注释"）。**这不是可选项 —— 注释、分支、循环、关键变量赋值处都必须有行间注释。** 执行完本步后自查：函数体内部是否每 10-20 行就有一行注释？关键 if/else 分支、循环体、数据变换语句前是否有注释？
5. **按需**补"实现差异对比"（见下"④ 多实现类对比"）：仅当本类存在真实可比的同类、且对比确有启发时才写；若类是独立实现、或"同类"差异无足轻重，**跳过此节，不要硬凑**。
6. **零污染校验**：通读 diff，确认未改任何一行可执行代码（含缩进、命名、顺序、空行结构）。
7. **验证**：对修改文件运行 read_lints，确认无新增错误；改动区域外的既有警告保持不动。

---

## ① 类级注释（重要类的宏观 docstring）

### 位置（铁律）

- **类的宏观注释就写在那个类处**——作为类 docstring，紧随 `class X(...):` 声明的**正下方**（`"""..."""`，符合 Python 规范，IDE / help() 可识别）。
- **❌ 严禁把宏观注释堆到文件顶部**写成 `#` 大框头注释块。文件头只保留原有的 license / coding / import，不新增"模块总览 / 调用链"之类的 `#` 头注释。读者不应该为了看某个类的注释而被迫上翻到文件头。
- **❌ 严禁用 `#` 写类级宏观注释**（既不写在文件头，也不写在类上方 / 类内部）。类级宏观内容一律用 `"""..."""` 三引号包裹，box-drawing 图放在 `"""` 内部。
- 若确需在 `class` 行**上方**放一段简短说明（如该类在整个文件中的定位），也必须用 `"""..."""`，**不得用 `#`**。

### 必须覆盖

1. **🧩 对外接口清单**：本类主要对外提供哪些方法 / 属性，各自一句话职责。
2. **🔗 框架如何与这些接口交互**：运行时框架（如 Scheduler / AttentionBackend）按什么顺序调用这些接口，即"宏观交互链"。
3. **🧬 设计要点 / 为什么**：核心数据结构、关键取舍。
4. **🖼️ 图**：用 box-drawing 大框 + 图（见"图文并茂"）展示上述交互流程——**图在 `"""..."""` 内部**，不是 `#` 行。

框用 box-drawing 字符 `╔ ═ ╗ ║ ╚ ╝ ╠ ╣`；框首行 = Emoji + 标题。建议至少两个框：一个"原理 / 接口"框，一个"宏观调用链"框。

## ② 函数级注释（每个函数的 docstring）

位置：函数 docstring，紧随 `def f(...):` 之下。必须覆盖：

1. **一句话职责**（带 Emoji，如 ✍️ 写 / 📖 读 / 🚚 搬运）。
2. **🔗 调用链定位**：本函数在整体调用链中的第几步、谁调用它、它又调用谁（用 `├─ └─ →` 小图）。
3. **详解**：📥 参数、📤 返回、⚙️ 行为 / 关键分支、⚠️ 注意 / 副作用 / 边界。
4. ⚠️ **docstring 之后，必须在函数体内部写入行间注释**（见下"③ 行间注释"）。 这不算"完了"，行间注释是 docstring 的延续，同样强制。

## ③ 行间注释

**这是强制要求，不是可选项。** 每个函数体内部必须加入行间注释，解释"为什么 / 意图"而非复述代码。目标是读者只看注释就能理解代码逻辑，无需逐行推敲。

### 必须注释的位置（按优先级排列）

| 优先级 | 位置 | 示例 |
|--------|------|------|
| 🔴 必注 | 关键 if/else 分支入口 | `# 分支一：整节点在窗口内 → 全恢复` |
| 🔴 必注 | 核心循环前 | `# ③ 遍历 LRU 尾端，驱逐直到满足 swa_num_tokens` |
| 🔴 必注 | 数据变换 / 类型转换 | `# Full pool 索引 → SWA pool 索引 (通过 allocator 维护的映射表)` |
| 🟡 应注 | 非显而易见的赋值 | `# inf 初始化使第一个 tombstone 就归零，保证"从根开始累计"语义` |
| 🟡 应注 | 魔法数 / 维度来源 | `# + page_size 缓冲：驱逐边界是 page 对齐的，多保留一页避免误伤` |
| 🟡 应注 | 状态变更（lock/unlock/tombstone）| `# lock_ref 1→0: 从 protected 移回 evictable` |
| 🟢 可注 | 关键 assert 含义 | `# tombstone 不应被锁：lock_ref 必须为 0` |
| 🟢 可注 | 哨兵 / 边界条件 | `# node is root → 停止向上遍历` |

### 行间注释密度要求

- 函数体每 **10-20 行代码** 至少应有一行中文注释。
- 长函数（>40 行）在关键阶段开头用编号标注：`# ① 初始化` `# ② 遍历匹配` `# ③ 后处理`
- 原有英文注释**必须保留**，中文注释**叠加补充**，不覆盖不删除。

### 禁止事项

- ❌ 禁止字面复述：`i += 1  # i 自增`
- ❌ 禁止为了凑密度而加无意义注释
- ❌ 禁止删除或覆盖原有英文注释

### 行间注释范例

```python
def evict_component(self, node, target=EvictLayer.DEVICE):
    ct = self.component_type
    cd = node.component_data[ct]
    freed = 0

    # ── Device 层驱逐：释放 SWA pool 索引 → 变 tombstone ──
    if EvictLayer.DEVICE in target and cd.value is not None:
        # 用 full_indices 而非 swa_value: 无 SWA 映射的 slot 指向同一 sentinel,
        # 直接 free swa_value 会导致 double-free
        self.cache.token_to_kv_pool_allocator.free_swa(
            node.component_data[BASE_COMPONENT_TYPE].value
        )
        freed = len(cd.value)
        self.cache.component_evictable_size_[ct] -= freed
        cd.value = None  # → tombstone (节点保留, 仅 SWA 数据清除)

    # ── Host 层驱逐 ──
    host_lru = self.cache.host_lru_lists[ct]
    if EvictLayer.HOST in target and cd.host_value is not None:
        host_freed = len(cd.host_value)
        if self._swa_kv_pool_host is not None:
            self._swa_kv_pool_host.free(cd.host_value)
        cd.host_value = None
        if host_lru.in_list(node):
            host_lru.remove_node(node)

    # ── 仅 DEVICE 层驱逐后: 若有残留 host_value, 移入 Host LRU 管理 ──
    if target is EvictLayer.DEVICE and cd.value is None and cd.host_value is not None:
        if not host_lru.in_list(node):
            host_lru.insert_mru(node)

    return freed, host_freed
```

## ④ 多实现类对比（仅当确有可比同类时才写，否则跳过）

**先判断值不值得写**：只有当本类存在一个真实可比的"主线/常规"同类（如不同硬件 / 精度 / 缓存层级的变体，二者数据结构或接口行为有**实质差异**），且读者读完对比能真正更懂本类时，才加这一节。下列情形**直接跳过，不要硬凑对比表**：

- 本类是独立实现，没有真正同层的兄弟类（继承基类 ≠ 必须对比）。
- 所谓"同类"只是签名相似，但用途 / 数据布局 / 调用链差异无足轻重，凑出来的对比表是水分。
- 本类与基类的差异已在"🧬 设计要点 / 为什么"框里讲清，无需再单列。

**确实值得对比时**，docstring 额外说明：

- **🔑 与"主线 / 常规"实现的关键区别**：数据结构、接口行为、性能特征差异（最好用对比表格）。
- **🎯 为什么这种场景用本类**：触发条件、适用场景、解决了常规实现的什么问题。

参考 `hiradix_cache.py` 中 `HiRadixCache` 对 `RadixCache` 的逐项对比框，以及 `MLATokenToKVPool` 对 `MHATokenToKVPool` 的对比。

---

## 图文并茂（宏观讲解用图）

代码注释中的"图"用 ASCII / box-drawing 绘制（纯文本，跨编辑器可见）。按需选用：

- **调用链树**：`├─ └─ →`，每节点标对端文件 / 函数。
- **数据流向图**：`→ ↑ ↓`，标注 GPU/Host/Storage 等流向。
- **状态流转图**：`┌──┐ ──evict──▶ ┌──┐`，展示对象状态迁移。
- **时序 / 分阶段图**：按 `Iter N / N+1` 或 `① 阶段 → ② 阶段` 列时间线。
- **对比表格**：`┌┬┐ ├┼┤ └┴┘`，对比分支 / 策略 / 实现差异。

对齐做到大致整齐即可，不为对齐牺牲信息准确性。

## 层次感与 Emoji 强调

- 用 Emoji 给信息**分层、分类、强调**，保持语义一致。
- 语义化词汇表：💾 存储 / 缓存 · 🧬 原理 / 结构 · 🧩 接口 / 组件 · 🔗 调用链 · 🖼️ 图 · 📥 入参 / 读入 · 📤 返回 / 写出 · 📖 读 · ✍️ 写 · 🚚 搬运 · 💿 offload · 🌐 跨节点传输 · ⚙️ 行为 · 🔑 关键区别 · 🎯 适用场景 · ⚠️ 注意 / 陷阱 · ⏳ 阻塞 · 👆 参见。可按领域扩展。

## 铁律（Hard Rules）

- **绝不污染原代码**：本 skill 只增删注释，不改任何可执行代码（逻辑、命名、顺序、缩进结构）。
- **❌ 严禁文件顶部 `#` 头注释块**：不得在文件头新增"模块总览 / 调用链 / 框架交互"之类的 `#` 大框注释。哪个类的注释就写在哪个类处（类 docstring `"""..."""`），不要让读者上翻文件头去找。
- **❌ 类级宏观注释一律用 `"""..."""`**：无论放在类 docstring 位置还是类上方，都不得用 `#`。box-drawing 图放在 `"""` 内部。`#` 仅用于函数体内的行间注释。
- 注释语言**跟随项目现有约定**（SGLang 等中文注释项目用简体中文）。
- 调用链 / 接口交互必须来自真实代码探索，不得臆造对端函数名。
- 不为对齐而牺牲框内信息的准确性。

## References

完整范例（类级宏观 docstring、函数级 docstring、多实现类对比、各类 ASCII 图）见 `references/style-examples.md`，动手前对照一次以统一风格。
