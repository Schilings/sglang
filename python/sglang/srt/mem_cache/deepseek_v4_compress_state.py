from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils import is_hip
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

_is_hip = is_hip()


@dataclasses.dataclass
class KVAndScore:
    """📦 KV + Score 的"双视图"包装：把一份连续张量 kv_score 的最后一维对半切，前半当 KV、后半当 Score。

    🧬 设计：不拆成两个 tensor，而是把 [kv | score] 拼在最后一维存进同一块内存(kv_score)，
       通过 .kv / .score 属性按 _item_size(= last_dim//2) 切片取视图（零拷贝）。
       这样压缩状态的"数据 + 打分"可整体搬运 / 散点索引 / cat，避免维护两块独立 buffer。
    🧩 主要接口：.kv / .score(视图) · from_kv_score(拼接构造) · new_empty / view / clone /
       __getitem__ / __setitem__(透传到 kv_score) · clear(kv=0, score=-inf) · cat(非末维拼接)。
    ⚠️ cat 不支持 dim=-1（末维是 kv|score 拼接维度，不能在它上面 cat）。
    """

    kv_score: torch.Tensor

    @property
    def kv(self) -> torch.Tensor:
        """📦 前半视图 = KV 部分（零拷贝切片 kv_score[..., :_item_size]）。"""
        return self.kv_score[..., : self._item_size]

    @property
    def score(self) -> torch.Tensor:
        """📦 后半视图 = Score 部分（kv_score[..., _item_size:]）。"""
        return self.kv_score[..., self._item_size :]

    @property
    def shape(self):
        """📐 透传 kv_score 的 shape。"""
        return self.kv_score.shape

    def __post_init__(self):
        """🧬 算 _item_size = last_dim // 2（KV 与 Score 各占一半）。"""
        self._item_size = self.kv_score.shape[-1] // 2

    @staticmethod
    def from_kv_score(*, kv: torch.Tensor, score: torch.Tensor) -> KVAndScore:
        """🧱 把独立的 kv 与 score 在末维 cat 成一个 KVAndScore（要求二者 shape 相同）。"""
        assert kv.shape == score.shape
        return KVAndScore(torch.cat([kv, score], dim=-1))

    def new_empty(self, new_shape) -> KVAndScore:
        """🧱 按新 shape 建空 KVAndScore（末维自动翻倍为 2*_item_size 以保 kv|score 对齐，no grad）。"""
        assert new_shape[-1] == self._item_size
        new_shape = list(new_shape)
        new_shape[-1] = 2 * self._item_size
        return KVAndScore(self.kv_score.new_empty(new_shape, requires_grad=False))

    def __getitem__(self, index) -> KVAndScore:
        """📦 索引透传，返回新的 KVAndScore 视图。"""
        return KVAndScore(self.kv_score[index])

    def __setitem__(self, index, value: KVAndScore):
        """✍️ 散点写入（写的是底层 kv_score）。"""
        self.kv_score[index] = value.kv_score

    def clear(self):
        """🧹 清零：KV=0、Score=-inf（-inf 表示"该位无打分"，供选择算子忽略）。"""
        self.kv.zero_()
        self.score.fill_(float("-inf"))

    def view(self, *args):
        """📦 reshape 视图（末维若指定非 -1，自动改成 2*_item_size 以保 kv|score 对齐）。"""
        args = list(args)
        if isinstance(args[-1], int) and args[-1] != -1:
            args[-1] = 2 * self._item_size
        return KVAndScore(self.kv_score.view(*args))

    def clone(self) -> KVAndScore:
        """📦 深拷贝。"""
        return KVAndScore(self.kv_score.clone())

    @staticmethod
    def cat(tensors: list[KVAndScore], dim: int) -> KVAndScore:
        """📦 沿非末维拼接多个 KVAndScore（末维是 kv|score 拼接维，禁止 cat）。"""
        assert dim != -1, "Concatenation along last dim is not supported."
        assert len(tensors) > 0, "At least one tensor is required for concatenation."
        item_size = tensors[0]._item_size
        for v in tensors:
            assert (
                v._item_size == item_size
            ), "All tensors must have the same item size."

        return KVAndScore(torch.cat([v.kv_score for v in tensors], dim=dim))


class CompressStatePool:
    """💾 DSV4 压缩状态池 —— 存 c4/c128 层的"压缩 KV + 打分"环形状态（供稀疏注意力压缩/恢复用）。

    由 DeepSeekV4TokenToKVPool._init_paged_compress_states 为每个 c4/c128 层建一个
    (deepseek_v4_memory_pool.py:1248/1266)；compressor / compress_hip / fused_compress_triton 读写之。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 两种模式：离线 ring vs 在线 single（由 online 开关，仅 c128 在线）                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  离线(online=False)：ring 缓冲                                                              ║
    ║    _size = (size + ring_size + 1) 向上取整到 ratio 的倍数                                    ║
    ║    last_dim = 2 * (1+overlap) * head_dim   ← KVAndScore 对半切：kv=(1+overlap)*head_dim, score 同║
    ║      • overlap = (ratio==4)：c4 与 SWA 窗口重叠，需额外存一份 overlap 区，故 (1+overlap)       ║
    ║      • c4: overlap=True  → last_dim=2*2*head_dim；c128: overlap=False → last_dim=2*1*head_dim ║
    ║  在线(online=True, 仅 c128)：ring_size=1，单状态/index                                        ║
    ║    _logical_size = size + 1 + 1；last_dim = 3 * head_dim  ← (max, sum, kv) 三件套            ║
    ║    开 MTP 时多 (online_mtp_max_draft_tokens) 份 bank：bank0=已提交状态，bank1..N=各草稿前缀    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 环形索引：SWA loc → state loc（translate_from_swa_loc_to_state_loc）                ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  每个 SWA 页对应 ring_size 个 state 槽：                                                       ║
    ║    swa_pages = swa_loc // swa_page_size                                                        ║
    ║    state_loc = swa_pages * ring_size + (swa_loc % ring_size)                                  ║
    ║    swa_loc<0 → state_loc=-1（哨兵，读出来是 clear 后的 0/-inf）                                ║
    ║  末槽 [-1] 是哨兵：每次 set 后都 clear()（kv=0, score=-inf），用作 padding/无效索引的 dummy。  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    🧩 对外接口：translate_from_swa_loc_to_state_loc / get_state_by_state_loc / set_state_by_state_loc。
    """

    def __init__(
        self,
        size: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        online: bool = False,
        swa_page_size: int = 0,
        online_mtp_max_draft_tokens: int = 0,
    ):
        """🧱 构造压缩状态池：按 online/ratio 决定 ring 或单状态布局，申请 kv_score buffer。

        🔗 调用链定位(① 创建链)：
            DeepSeekV4TokenToKVPool._init_paged_compress_states → CompressStatePool(...)
            (每 c4/c128 层一个；indexer 层用 indexer_head_dim，主 attention 层用 nope+rope)

        📥 关键参数：
            ratio       : 压缩比(4 或 128)
            ring_size   : ring 槽数(c4 非spec8/spec16；c128 非spec128/spec256；online=1)
            overlap     : 是否与 SWA 重叠(= ratio==4)，影响 last_dim 的 (1+overlap) 系数
            head_dim    : 单元维度(主=nope+rope=512；indexer=indexer_head_dim=128)
            online      : 在线压缩(仅 c128)，ring_size 必为 1
            online_mtp_max_draft_tokens : 在线 c128 + MTP 的草稿 bank 数
        ⚙️ 两条申请路径：HIP 直接 torch.empty；CUDA 走 memory_saver + custom_mem_pool(PD/NVLink)。
        """
        self.ratio = ratio
        self.ring_size = ring_size
        self.swa_page_size = swa_page_size
        self.enable_memory_saver = enable_memory_saver
        self.online_mtp_state_slot_offset = 0
        self.online_mtp_max_draft_tokens = 0

        if online:
            # 在线 c128：ring_size=1，每 index 存 (max, sum, kv) 三件套 → last_dim=3*head_dim。
            assert ring_size == 1, "online compress requires ring_size=1"
            self._logical_size = size + self.ring_size + 1
            if online_mtp_max_draft_tokens > 0:
                # Bank 0 is the committed state. Banks 1..N cache per-draft
                # prefix states for lazy commit after target verify.
                # bank0=已提交状态；bank1..N=各草稿前缀状态(待 target verify 后 lazy commit)。
                self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
                self.online_mtp_state_slot_offset = self._logical_size
            # 总槽位 = 逻辑槽 × (1 + 草稿 bank 数)。
            self._size = self._logical_size * (1 + self.online_mtp_max_draft_tokens)
            last_dim = 3 * head_dim
        else:
            # 离线 ring：_size 向上取整到 ratio 倍数(对齐压缩块)，+ring_size+1 含 ring 与哨兵。
            self._size = size + self.ring_size + 1
            self._size = (self._size + ratio - 1) // ratio * ratio
            self._logical_size = self._size
            # last_dim = 2*(1+overlap)*head_dim：KVAndScore 把它对半切成 kv | score。
            #   overlap=True(c4)：多一份 overlap 区 → (1+1)*head_dim；
            #   overlap=False(c128)：仅压缩状态 → (1+0)*head_dim。
            last_dim = 2 * (1 + overlap) * head_dim

        if _is_hip:
            # HIP：直接 torch.empty 申请。
            self.kv_score_buffer = KVAndScore(
                torch.empty((self._size, last_dim), dtype=dtype, device=device)
            )
            if not online:
                # 哨兵槽 [-1] 清零：kv=0, score=-inf（无效索引读到干净状态）。
                self.kv_score_buffer[-1].clear()
        else:
            # CUDA：走 memory_saver + custom_mem_pool(PD 分离/NVLink 场景落在可 RDMA 地址空间)。
            self.memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=enable_memory_saver
            )
            self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
                maybe_init_custom_mem_pool(device=device)
            )

            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                with (
                    torch.cuda.use_mem_pool(self.custom_mem_pool)
                    if self.custom_mem_pool
                    else nullcontext()
                ):
                    self.kv_score_buffer = KVAndScore(
                        torch.empty(
                            (self._size, last_dim),
                            dtype=dtype,
                            device=device,
                        )
                    )
                    if not online:
                        # 哨兵槽 [-1] 清零（同 HIP 路径）。
                        self.kv_score_buffer[-1].clear()

    def translate_from_swa_loc_to_state_loc(
        self, swa_loc: torch.Tensor
    ) -> torch.Tensor:
        """🔀 SWA loc → state ring loc：每 SWA 页对应 ring_size 个 state 槽。

        📥 参数：swa_loc : SWA buffer 内的 token 索引。
        📤 返回：state ring 内的索引；swa_loc<0 → -1（哨兵，读到 clear 后的 0/-inf）。
        ⚙️ 公式：swa_pages = swa_loc // swa_page_size；state_loc = swa_pages*ring_size + (swa_loc % ring_size)。
        """
        swa_pages = swa_loc // self.swa_page_size
        state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
        state_loc = torch.where(swa_loc < 0, -1, state_loc)
        return state_loc

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:
        """📖 按 state_loc 取压缩状态（返回 KVAndScore 视图，含 kv+score）。"""
        return self.kv_score_buffer[state_loc]

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):
        """✍️ 按 state_loc 写压缩状态，并清零哨兵槽 [-1]（保持 dummy 干净）。"""
        self.kv_score_buffer[state_loc] = value
        # 哨兵 [-1] 始终保持 clear(kv=0, score=-inf)，供无效/padding 索引读取。
        self.kv_score_buffer[-1].clear()
