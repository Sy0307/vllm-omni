# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in packed Flow for full-response and streaming inference on Hopper GPUs.

Adapted from SGLang-Omni packed_dit.py, commit
127f34b57446a3cb5588ec987da0771a3be675be (Apache-2.0):
https://github.com/sgl-project/sglang-omni
Full-context and chunk-causal attention use vLLM's FA3 binding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

FA3_PAGE_SIZE = 1


class RaggedRowAttention:
    """FA3 page metadata isolating each request and each causal query chunk."""

    def __init__(self, rows: PackedRows, *, heads: int, head_dim: int, chunk_size: int | None = None) -> None:
        self.heads, self.head_dim = heads, head_dim
        device = rows.row_ids.device
        segment_rows, segment_ends, offsets = [], [], [0]
        for row, length in enumerate(rows.lengths):
            span = chunk_size or length
            for start in range(0, length, span):
                end = min(start + span, length)
                segment_rows.append(row)
                segment_ends.append(end)
                offsets.append(offsets[-1] + end - start)
        self.cache_seqlens = torch.tensor(segment_ends, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(b - a for a, b in zip(offsets, offsets[1:]))
        starts = rows.starts_host[segment_rows].to(device)
        page = torch.arange(max(segment_ends), dtype=torch.int32, device=device)
        self.page_table = torch.where(page[None] < self.cache_seqlens[:, None], starts[:, None] + page[None], 0)


class PackedDiT:
    """Adapt a loaded DiT to compiled, variable-length inference with FA3."""

    def __init__(self, dit: torch.nn.Module) -> None:
        self.dit = dit
        self.compiled_full_forward = None

    def compile(self, dtype: torch.dtype) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Packed Flow requires FP16 or BF16")
        self.compiled_full_forward = torch.compile(
            self.forward_full,
            backend="inductor",
            dynamic=True,
            fullgraph=True,
            options={"emulate_precision_casts": True},
        )

    def row_attention(self, rows: PackedRows, *, streaming: bool) -> RaggedRowAttention:
        attn = self.dit.transformer_blocks[0].attn
        result = RaggedRowAttention(
            rows,
            heads=attn.heads,
            head_dim=attn.inner_dim // attn.heads,
            chunk_size=self.dit.static_chunk_size if streaming else None,
        )
        mark_packed_compile_metadata(rows, result)
        return result

    def forward_full(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: RaggedRowAttention,
    ) -> torch.Tensor:
        return forward_packed_tensor_geometry(
            self, x, mu, spks, cond, t, rows, attention, max_seqlen_q=attention.page_table.shape[1]
        )


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::packed_fa3",
    mutates_args=(),
    device_types="cuda",
)
def packed_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    """Alias-free FA3 boundary for the compiled PackedDiT path."""
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=page_table.shape[1],
        seqused_k=cache_seqlens,
        block_table=page_table,
        causal=False,
        fa_version=3,
    )


@packed_fa3.register_fake
def fake_packed_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::native_mish",
    mutates_args=(),
    device_types="cuda",
)
def native_mish(x: torch.Tensor) -> torch.Tensor:
    """Preserve eager CUDA Mish arithmetic across the Inductor boundary."""
    return F.mish(x)


@native_mish.register_fake
def fake_native_mish(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::native_layer_norm",
    mutates_args=(),
    device_types="cuda",
)
def native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    """Keep CUDA autocast's eager FP32 LayerNorm contract."""
    return F.layer_norm(x.float(), (normalized_size,), None, None, eps)


@native_layer_norm.register_fake
def fake_native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.float32)


@dataclass(frozen=True)
class PackedRows:
    lengths: tuple[int, ...]
    starts_host: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor

    @property
    def total(self) -> int:
        return sum(self.lengths)

    @property
    def width(self) -> int:
        return max(self.lengths)


def pack_rows(lengths: Sequence[int], device: torch.device) -> PackedRows:
    lengths = tuple(int(length) for length in lengths)
    starts_host = F.pad(torch.tensor(lengths, dtype=torch.int64).cumsum(0), (1, 0))
    starts = starts_host.to(device)
    total = int(starts_host[-1])
    row_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
        output_size=total,
    )
    positions = torch.arange(total, device=device) - starts[row_ids]
    return PackedRows(
        lengths=lengths,
        starts_host=starts_host.to(torch.int32),
        row_ids=row_ids,
        positions=positions,
    )


def gather_rows(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    """(rows, width, channels) -> (1, total, channels), each row's first
    length frames in row order."""
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows(packed: torch.Tensor, rows: PackedRows, width: int) -> torch.Tensor:
    """(1, total, channels) -> (rows, width, channels), zero past each row's
    length."""
    channels = packed.shape[2]
    flat = packed.new_zeros(len(rows.lengths) * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(len(rows.lengths), width, channels)


def mark_packed_compile_metadata(rows: PackedRows, attention: RaggedRowAttention) -> None:
    dynamo.mark_dynamic(attention.page_table, (0, 1))
    dynamo.mark_dynamic(attention.cu_seqlens_q, 0)
    dynamo.mark_dynamic(attention.cache_seqlens, 0)
    dynamo.mark_dynamic(rows.starts_host, 0)
    dynamo.mark_dynamic(rows.row_ids, 0)
    dynamo.mark_dynamic(rows.positions, 0)


def gather_rows_tensor_geometry(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows_tensor_geometry(packed: torch.Tensor, rows: PackedRows, row_count: int, width: int) -> torch.Tensor:
    channels = packed.shape[2]
    flat = packed.new_zeros(row_count * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(row_count, width, channels)


def conv_pos_embed_tensor_geometry(
    estimator: PackedDiT,
    h: torch.Tensor,
    rows: PackedRows,
    attention: RaggedRowAttention,
) -> torch.Tensor:
    row_count = rows.starts_host.shape[0] - 1
    width = attention.page_table.shape[1]
    padded = scatter_rows_tensor_geometry(h, rows, row_count, width)
    module = estimator.dit.input_embed.conv_pos_embed
    embedded = padded.permute(0, 2, 1)
    embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
    embedded = module.conv1[0](embedded)
    embedded = native_mish(embedded)
    embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
    embedded = module.conv2[0](embedded)
    embedded = native_mish(embedded)
    embedded = embedded.permute(0, 2, 1)
    return gather_rows_tensor_geometry(embedded, rows)


def rope_tensor_geometry(
    estimator: PackedDiT,
    rows: PackedRows,
    attention: RaggedRowAttention,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = attention.page_table.shape[1]
    freqs, scale = estimator.dit.rotary_embed.forward_from_seq_len(width)
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    freqs = freqs[:, rows.positions]
    return freqs.cos(), freqs.sin()


def ragged_attention_tensor_geometry(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention: RaggedRowAttention,
    max_seqlen_q: int,
) -> torch.Tensor:
    page_shape = (-1, FA3_PAGE_SIZE, attention.heads, attention.head_dim)
    output = packed_fa3(
        query[0].reshape(-1, attention.heads, attention.head_dim),
        key[0].reshape(page_shape),
        value[0].reshape(page_shape),
        attention.cache_seqlens,
        attention.page_table,
        attention.cu_seqlens_q,
        max_seqlen_q,
    )
    return output.reshape(1, -1, attention.heads * attention.head_dim)


def attend_tensor_geometry(
    attn: torch.nn.Module,
    x: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    attention: RaggedRowAttention,
    max_seqlen_q: int,
) -> torch.Tensor:
    # note (ratish): under autocast to_q, to_k and to_v would each cast the
    # float32 norm output again.
    x = x.to(attn.to_q.weight.dtype)
    query = attn.to_q(x)
    key = attn.to_k(x)
    value = attn.to_v(x)
    rotate_in_place(query, *rope)
    rotate_in_place(key, *rope)
    output = ragged_attention_tensor_geometry(query, key, value, attention, max_seqlen_q).to(query.dtype)
    return attn.to_out[1](attn.to_out[0](output))


def layer_norm_preserving_eager(layer_norm: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    normalized_size = int(layer_norm.normalized_shape[0])
    return native_layer_norm(x, normalized_size, float(layer_norm.eps))


def attn_norm_preserving_eager(
    block: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modulation = block.attn_norm.linear(block.attn_norm.silu(time_embedding))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(modulation, 6, dim=1)
    normalized = layer_norm_preserving_eager(block.attn_norm.norm, h)
    normalized = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def final_norm_preserving_eager(
    norm_out: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor,
) -> torch.Tensor:
    modulation = norm_out.linear(norm_out.silu(time_embedding))
    scale, shift = torch.chunk(modulation, 2, dim=1)
    normalized = layer_norm_preserving_eager(norm_out.norm, h)
    return normalized * (1 + scale)[:, None, :] + shift[:, None, :]


def forward_packed_tensor_geometry(
    estimator: PackedDiT,
    x: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    rows: PackedRows,
    attention: RaggedRowAttention,
    *,
    max_seqlen_q: int,
) -> torch.Tensor:
    dit = estimator.dit
    t = dit.time_embed(t)
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    h = conv_pos_embed_tensor_geometry(estimator, h, rows, attention) + h
    rope = rope_tensor_geometry(estimator, rows, attention)
    residual = h
    for block in dit.transformer_blocks:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = attn_norm_preserving_eager(block, h, t)
        h = h + gate_msa.unsqueeze(1) * attend_tensor_geometry(block.attn, norm, rope, attention, max_seqlen_q)
        ff_norm = layer_norm_preserving_eager(block.ff_norm, h)
        ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    h = final_norm_preserving_eager(dit.norm_out, h, t)
    return dit.proj_out(h)


def rotate_in_place(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """x: (1, total, heads * head_dim). Interleaved RoPE in float32 on the
    rotary dims, rounded back into x."""
    # note (ratish): the DiT rotates only the first rotary dims of the
    # flattened heads, so the rest of x is never copied.
    rotary = x[..., : cos.shape[-1]]
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)


def solve_flow_euler_packed(
    estimator: PackedDiT,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    rows: PackedRows,
    *,
    cfg_rate: float,
    streaming: bool,
) -> torch.Tensor:
    """Euler steps over a packed sequence with classifier free guidance: the
    conditional rows and their unconditional twins share one DiT call."""
    total = noise.shape[1]
    twin_rows = pack_rows(rows.lengths * 2, noise.device)
    attention = estimator.row_attention(twin_rows, streaming=streaming)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=noise.device, dtype=spks.dtype)
    forward = estimator.compiled_full_forward
    if forward is None:
        raise ValueError("Packed Flow requires a compiled packed path")
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        flow_time[:] = t
        vector_field = forward(
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            twin_rows,
            attention,
        )
        conditional = vector_field[:, :total]
        unconditional = vector_field[:, total:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
    return x.float()
