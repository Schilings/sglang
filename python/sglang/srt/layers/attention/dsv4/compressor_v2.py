from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


CompressMetadata: TypeAlias = Union[CompressorDecodePlan, CompressorPrefillPlan]
# 注意：为向后兼容而保留的别名
FusedCompressMetadata: TypeAlias = CompressMetadata


def _use_online_compress(compress_ratio: int) -> bool:
    """Online state-pool 路径仅用于 c128。"""
    return compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


class CompressorBackendMixin:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DSV4Metadata

    # 注意：将被覆写
    def _maybe_upgrade_forward_metadata(self): ...

    def _get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        return getattr(self.forward_metadata, attr_name)

    def _get_out_loc(self, compress_ratio: int) -> torch.Tensor:
        attr_name = f"c{compress_ratio}_out_loc"
        return getattr(self.forward_metadata.core_metadata, attr_name)

    def _forward_compress_all_in_one(
        self,
        *,
        kv_score_buffer: torch.Tensor,  # offline: [pool_size, 2*(1+overlap)*head_dim]; online(c128): [pool_size, 3*head_dim]
        kv_score_input: torch.Tensor,    # [T, 2*coff*head_dim] — wkv_gate 输出（c4 的 coff=2，c128 的 coff=1）
        ape: torch.Tensor,               # [compress_ratio, coff*head_dim] — 绝对位置编码
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        kv_cache: torch.Tensor,          # 压缩 KV 分页缓存的 uint8 视图
        is_indexer: bool,
        rotate: bool,
        compress_ratio: int,             # 4 or 128
        page_size: int,
        out_loc: torch.Tensor,           # [num_compressed_tokens] — 分页缓存中的输出槽位索引
    ) -> None:
        assert compress_ratio == 4 or compress_ratio == 128
        assert rotate == is_indexer == (head_dim == 128)

        plan = self._get_paged_compress_metadata(compress_ratio)
        is_online = _use_online_compress(compress_ratio)
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)      # [pool_size, 1, 3*head_dim]
        else:
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)  # [pool_size/cr, cr, last_dim]
        kv_compressed = compress_forward(   # [num_compressed_tokens, head_dim] — 评分 & 选择后的压缩 KV
            kv_score_buffer=kv_score_buffer, # [pool_size/cr, cr, last_dim]
            kv_score_input=kv_score_input,   # [T, 2*coff*head_dim]
            ape=ape.view(-1, head_dim),      # [cr*coff, head_dim]
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            is_online=is_online,
        )
        # 注意：此处使用了一些 hack...
        compress_norm_rope_store(            # 融合：RMSNorm + RoPE + 将压缩 KV 写入分页缓存
            kv_compressed,                   # [num_compressed_tokens, head_dim]
            plan,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
            freq_cis=freqs_cis_cache,
            out_loc=out_loc,                 # [num_compressed_tokens]
            kvcache=kv_cache,                # uint8 分页缓存
            page_size=page_size,
        )

    def forward_unified(
        self,
        x: torch.Tensor,            # [T, hidden_size] — 解码器层的隐藏状态
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,      # compress_ratio ∈ {4, 128}
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return

        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", token_to_kv_pool)
        kv_score_input = compressor.compute_kv_score(x, forward_batch)  # [T, 2*coff*head_dim] — linear(x, wkv_gate)
        state_pool = compressor.get_state_pool(forward_batch)
        out_loc = self._get_out_loc(compressor.ratio)  # [num_compressed_tokens] — 输出槽位索引
        if compressor.is_in_indexer:
            kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
            page_size = token_to_kv_pool.get_index_k_page_size()
        else:
            _, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]
            assert compress_kv_pool is not None
            kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            if hasattr(compress_kv_pool, "translate_loc_to_hisparse_device"):
                # The v2 compressor writes directly into the raw C4 KV tensor.
                # HiSparse C4 therefore needs the physical C4 location here.
                out_loc = compress_kv_pool.translate_loc_to_hisparse_device(out_loc)

        self._forward_compress_all_in_one(
            kv_score_buffer=state_pool.kv_score_buffer.kv_score,  # [pool_size, last_dim] — 历史 KV+score 的环形缓冲区
            kv_score_input=kv_score_input,   # [T, 2*coff*head_dim]
            ape=compressor.ape,              # [compress_ratio, coff*head_dim]
            head_dim=compressor.head_dim,
            norm=compressor.norm,
            freqs_cis_cache=compressor.freqs_cis,
            kv_cache=kv_cache.view(dtype=torch.uint8),
            is_indexer=compressor.is_in_indexer,
            rotate=compressor.rotate,
            compress_ratio=compressor.ratio,
            page_size=page_size,
            out_loc=out_loc,                 # [num_compressed_tokens]
        )

    # 注意：为向后兼容而保留的别名
    forward_indexer_compressor = forward_unified
    forward_core_compressor = forward_unified


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> CompressMetadata:
    """构建分页压缩元数据（即计划 plan）。

    State-pool 槽位转换在 C++ planner 内部完成；
    Python 端只需传递相关张量。
    """
    if _use_online_compress(compress_ratio):
        return _create_online_paged_compressor_data(
            is_prefill=is_prefill,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            num_q_tokens=num_q_tokens,
        )

    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # NOTE: This is actually a proxy, which encounter some bug with tvm-ffi.
    # As a workaround, we use `.detach()` to get the real tensor.
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices_i64 = req_pool_indices.to(torch.int64)

    if is_prefill:
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)

        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            num_q_tokens=num_q_tokens,
            use_cuda_graph=use_prefill_cuda_graph,
        )
    else:
        return CompressorDecodePlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            seq_lens=seq_lens.to(torch.int64),
            swa_page_size=swa_page_size,
            ring_size=ring_size,
        )


def _create_online_paged_compressor_data(
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    seq_lens_cpu: Optional[List[int]],
    extend_lens_cpu: Optional[List[int]],
    use_prefill_cuda_graph: bool,
    num_q_tokens: Optional[int],
) -> CompressMetadata:
    assert not use_prefill_cuda_graph, "online c128 doesn't support cuda graph"

    swa_page_size = int(token_to_kv_pool.swa_page_size)
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices = req_pool_indices.to(torch.int64)

    if is_prefill:
        # 入口同步：在接触此 builder 的任何内容之前，捕获前一层/内核的 IMA，
        # 以免误将问题归咎于我们。
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens_planner = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)
            num_q_tokens_planner = num_q_tokens

        return CompressorPrefillPlan.generate_online(
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            num_q_tokens=int(num_q_tokens_planner),
            swa_page_size=swa_page_size,
        )
    else:
        return CompressorDecodePlan.generate_online(
            seq_lens=seq_lens.to(torch.int64),
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
        )
