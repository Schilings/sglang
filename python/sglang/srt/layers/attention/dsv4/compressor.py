from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Union

import torch
import torch.nn as nn

from sglang.jit_kernel.dsv4 import linear_bf16_fp32, triton_create_paged_compress_data
from sglang.jit_kernel.dsv4.compress_old import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.triton_kernel import act_quant
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.models.deepseek_v2 import _is_hip
from sglang.srt.utils import add_prefix

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


class CompressorBackendMixin:
    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def _maybe_upgrade_forward_metadata(self) -> None:
        pass

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,  # offline: [pool_size, 2*(1+overlap)*head_dim]; online(c128): [pool_size, 3*head_dim]
        kv_score_input: torch.Tensor,    # [T, 2*coff*head_dim] — wkv_gate 输出（c4 的 coff=2，c128 的 coff=1）
        ape: torch.Tensor,               # [compress_ratio*coff, head_dim] — 绝对位置编码（经 .view(-1, head_dim) 后）
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,             # 4 or 128
        is_paged: bool = False,
    ) -> torch.Tensor:                   # [num_compressed_tokens, head_dim] — 压缩 KV（已应用 norm+rope）
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        assert compress_ratio in (
            4,
            128,
        ), f"DSV4 supports CSA(4x) and HCA(128x) only, got {compress_ratio=}"
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)      # [pool_size, 1, 3*head_dim]
            else:
                last_dim = 2 * head_dim * coff
                assert kv_score_buffer.shape[-1] == last_dim
                kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)  # [pool_size/cr, cr, 2*coff*head_dim]
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        kv_compressed = compress_forward(   # [num_compressed_tokens, head_dim] — 评分 & 选择后的压缩 KV
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,   # [T, 2*coff*head_dim]
            ape=ape,                         # [cr*coff, head_dim]
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(    # 融合：RMSNorm + RoPE，原地修改 kv_compressed
            kv_compressed,                   # [num_compressed_tokens, head_dim]
            norm.weight,
            norm.variance_epsilon,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed  # [num_compressed_tokens, head_dim]

    def forward_core_compressor(
        self,
        x: torch.Tensor,            # [T, hidden_size] — 解码器层的隐藏状态
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,      # compress_ratio ∈ {4, 128}，is_in_indexer=False
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG 延迟升级：具体后端 (DeepseekV4AttnBackend)
        # 拥有此辅助方法。MQALayer._forward_prepare 在
        # attn_backend.forward() 之前调用我们，因此 Raw -> DSV4Metadata
        # 必须在此处完成（例如 1.6T layer 0 的 compress_ratio=128
        # 需要 cX_compress_metadata）。
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)  # [num_compressed_tokens, head_dim]
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,                    # [num_compressed_tokens]
                cache_k=new_compressed_kv,      # [num_compressed_tokens, head_dim]
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(new_compressed_kv.bfloat16())  # packed uint8
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,            # [T, hidden_size] — 解码器层的隐藏状态
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,      # compress_ratio=4，is_in_indexer=True
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG 延迟升级（原因见 forward_core_compressor）。
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)  # [num_compressed_tokens, head_dim]
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,  # [num_compressed_tokens]
                cache_k=new_compressed_kv,      # [num_compressed_tokens, head_dim]
            )
        else:
            new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                new_compressed_kv               # [num_compressed_tokens, head_dim]
            )                                   # fp8: [num_compressed_tokens, head_dim], scale: [num_compressed_tokens, head_dim/128]
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,  # [num_compressed_tokens]
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,       # [max_num_reqs, max_context_len]
    req_pool_indices: torch.Tensor,    # [batch_size]
    seq_lens: torch.Tensor,           # [batch_size]
    extend_lens: Optional[torch.Tensor] = None,   # [batch_size]（仅 prefill）
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:           # (write_loc, extra_data, plan)
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:  # [batch_size] → [batch_size] (int32)
        positions = positions.masked_fill(positions < 0, 0)
        loc = req_to_token[req_pool_indices, positions]         # [batch_size]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)  # [batch_size]
        swa_pages = swa_loc // swa_page_size                    # [batch_size]
        state_loc = swa_pages * ring_size + swa_loc % ring_size # [batch_size]
        return (state_loc // compress_ratio).to(torch.int32)    # [batch_size] — 压缩槽位索引

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        write_loc, extra_data = triton_create_paged_compress_data(  # write_loc: [num_compressed_slots]，extra_data: 可选的重叠位置
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)              # [batch_size]
        write_loc = get_raw_loc(write_positions)               # [batch_size] (int32)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)  # [batch_size]
            extra_data = write_overlap_loc.view(-1, 1)         # [batch_size, 1]
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)


class Compressor(nn.Module):
    """DeepSeek V4 NSA（原生稀疏注意力）的 KV 压缩器。

    使用学习的评分 + APE（绝对位置编码）将 ratio 个 token 的 KV 压缩为 1 个压缩 token。
    支持两种模式：
      - c4（CSA，压缩滑动注意力）：overlap=True，coff=2
      - c128（HCA，重度压缩注意力）：overlap=False，coff=1
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4   # c4 uses overlapping windows
        self.rotate = rotate
        coff = 1 + self.overlap          # coff=2 for c4 (overlap), coff=1 for c128

        # APE：绝对位置编码，形状 [ratio, coff*head_dim]
        # c4: [4, 2*head_dim]，c128: [128, head_dim]
        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        wkv_gate_dtype = torch.bfloat16
        # wkv_gate：将隐藏状态投影到 KV + score 对
        # c4:  hidden → 4*head_dim  (2*coff*head_dim = 2*2*head_dim)
        # c128: hidden → 2*head_dim (2*coff*head_dim = 2*1*head_dim)
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        if self.overlap:
            # c4 overlap: 将 APE 从 [4, 2*head_dim] → 拆分 → 拼接 → [4, 2*head_dim]
            # 从交错布局 [kv0, kv1] 重排为连续布局 [k0,k1,v0,v1]
            ape = torch.chunk(self.ape.data, 2, dim=-1)  # each: [4, head_dim]
            ape = torch.cat([ape[0], ape[1]], dim=0)       # [8, head_dim]
            self.ape.data.copy_(ape.view(self.ratio, -1))   # [4, 2*head_dim]

    # 注意：供 v2 compressor backend 使用
    def get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    # 注意：供 v2 compressor backend 使用
    def compute_kv_score(self, x: torch.Tensor, forward_batch: ForwardBatch):
        """通过 wkv_gate 将隐藏状态投影到 (KV, score) 对。

        返回：
            [T, 2*coff*head_dim] — 拼接的 KV 和 score 特征。
              c4:   [T, 4*head_dim]  (coff=2：2 个重叠窗口的 KV + 各自的 score)
              c128: [T, 2*head_dim]  (coff=1：单个窗口的 KV + score)
        """
        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)  # [T, 2*coff*head_dim]
        if nsa_use_prefill_cp(forward_batch):
            kv_score = cp_all_gather_rerange_output(
                kv_score,
                get_attention_cp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )
        return kv_score                                         # [T, 2*coff*head_dim]

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        """完整压缩流程：score → compress → norm+rope。

        返回：
            [num_compressed_tokens, head_dim] — 经 norm+rope 后的压缩 KV。
        """
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)               # [0, head_dim]

        kv_score = self.compute_kv_score(x, forward_batch)     # [T, 2*coff*head_dim]

        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4AttnBackend)
        kv_score_buffer = self.get_state_pool(forward_batch)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score  # [pool_size, 2*(1+overlap)*head_dim]
        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,                            # [T, 2*coff*head_dim]
            ape=self.ape.view(-1, self.head_dim),              # [ratio*coff, head_dim]
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )                                                       # [num_compressed_tokens, head_dim]
