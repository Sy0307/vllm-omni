# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Paged attention for TRITON_ATTN decoders with a head-grouped decode path.

Drop-in replacement for the 2D path of vLLM's ``unified_attention`` for plain
causal GQA attention (no ALiBi, sinks, softcap, sliding window, quantized KV,
...). The prefill path reproduces ``kernel_unified_attention`` (fp32 online
softmax, bf16 P@V, same tiles). Decode rows (query length 1) take a separate
path in the same launch:

* one program handles ``DECODE_HEADS`` KV heads of a sequence, so each KV tile
  is one contiguous span of the LBNHC page (all heads of a token are adjacent)
  and the launch needs fewer, longer-lived programs (one wave on H200 for a
  128-row decode batch);
* the score / P@V products use a block-diagonal layout: query rows of head ``g``
  only see the columns of KV head ``g`` (other columns are masked to -inf, so
  they contribute exact zeros);
* sequence lookup loads the whole ``query_start_loc`` prefix as one vector
  instead of a dependent binary search.
"""

from __future__ import annotations

import os as _pdl_os

import torch
from vllm.triton_utils import tl, triton

_gdc_launch = tl.extra.cuda.gdc_launch_dependents
_gdc_wait = tl.extra.cuda.gdc_wait

# MOSS_TRITON_PDL=1 (Hopper+): launch with programmatic dependent launch. The kernel waits for its
# predecessor before any global access and lets its successor launch at once; results are unchanged.
_TRITON_PDL = _pdl_os.environ.get("MOSS_TRITON_PDL") == "1"


@triton.jit
def _online_softmax_step(S, M, L, acc, V):
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L = L * alpha + l_j
    acc = acc * alpha[:, None]
    acc += tl.dot(P.to(V.dtype), V)
    return m_j, L, acc


@triton.jit
def _prefill_block(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    scale,
    seq_idx,
    kv_head_idx,
    q_block_local_idx,
    cur_start,
    cur_batch_query_len,
    seq_len,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Same math and tiling as vLLM's kernel_unified_attention (2D, causal).
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    Q = tl.load(
        query_ptr + query_offset, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0
    )
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE
    query_abs_pos = context_len + query_pos[:, None]
    row_mask = query_mask_1[:, None] & query_mask_0[:, None]
    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)
        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(row_mask & (seq_offset[None, :] <= query_abs_pos), S, float("-inf"))
        M, L, acc = _online_softmax_step(S, M, L, acc, V)
    acc = acc / L[:, None]
    output_offset = (
        query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 + offs_d[None, :]
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


@triton.jit
def _load_kv_group(
    key_cache_ptr,
    value_cache_ptr,
    bt_ptr,
    tile_start,
    seq_len,
    col_t,
    k_head,
    offs_d,
    dim_mask,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DECODE_TILE: tl.constexpr,
):
    seq_offset = tile_start + col_t
    tile_mask = seq_offset < seq_len
    physical_block_idx = tl.load(bt_ptr + seq_offset // BLOCK_SIZE, mask=tile_mask, other=0).to(tl.int64)
    slot = (seq_offset % BLOCK_SIZE).to(tl.int64)
    k_offset = (
        physical_block_idx[None, :] * stride_k_cache_0
        + slot[None, :] * stride_k_cache_1
        + k_head[None, :] * stride_k_cache_2
        + offs_d[:, None] * stride_k_cache_3
    )
    v_offset = (
        physical_block_idx[:, None] * stride_v_cache_0
        + slot[:, None] * stride_v_cache_1
        + k_head[:, None] * stride_v_cache_2
        + offs_d[None, :] * stride_v_cache_3
    )
    K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
    V = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
    return K, V, tile_mask


@triton.jit
def _decode_group(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    scale,
    seq_idx,
    kv_head_idx,
    cur_start,
    seq_len,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    DECODE_HEADS: tl.constexpr,
    DECODE_TILE: tl.constexpr,
    DECODE_M: tl.constexpr,
):
    # One query token, DECODE_HEADS consecutive KV heads starting at kv_head_idx.
    # Columns are head-major: column c belongs to KV head c // DECODE_TILE.
    TG: tl.constexpr = DECODE_HEADS * DECODE_TILE
    offs_m = tl.arange(0, DECODE_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_c = tl.arange(0, TG)
    col_g = offs_c // DECODE_TILE
    col_t = offs_c % DECODE_TILE
    row_g = offs_m // num_queries_per_kv
    q_head = kv_head_idx * num_queries_per_kv + offs_m
    row_valid = (offs_m < DECODE_HEADS * num_queries_per_kv) & (q_head < num_query_heads)
    dim_mask = offs_d < HEAD_SIZE
    Q = tl.load(
        query_ptr + cur_start * query_stride_0 + q_head[:, None] * query_stride_1 + offs_d[None, :],
        mask=row_valid[:, None] & dim_mask[None, :],
        other=0.0,
    )
    block_table_offset = seq_idx * block_table_stride
    k_head = (kv_head_idx + col_g).to(tl.int64)
    M = tl.full([DECODE_M], float("-inf"), dtype=tl.float32)
    L = tl.full([DECODE_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([DECODE_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    num_tiles = (seq_len + DECODE_TILE - 1) // DECODE_TILE
    same_head = row_valid[:, None] & (row_g[:, None] == col_g[None, :])
    bt_ptr = block_tables_ptr + block_table_offset
    for j in range(0, num_tiles):
        K, V, tile_mask = _load_kv_group(
            key_cache_ptr,
            value_cache_ptr,
            bt_ptr,
            j * DECODE_TILE,
            seq_len,
            col_t,
            k_head,
            offs_d,
            dim_mask,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            DECODE_TILE,
        )
        S = tl.zeros(shape=(DECODE_M, TG), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(same_head & tile_mask[None, :], S, float("-inf"))
        M, L, acc = _online_softmax_step(S, M, L, acc, V)
    acc = acc / L[:, None]
    tl.store(
        output_ptr + cur_start * output_stride_0 + q_head[:, None] * output_stride_1 + offs_d[None, :],
        acc,
        mask=row_valid[:, None] & dim_mask[None, :],
    )


@triton.jit
def _paged_attention_kernel(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_seqs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_SEQS_PADDED: tl.constexpr,
    DECODE_HEADS: tl.constexpr,
    DECODE_TILE: tl.constexpr,
    DECODE_M: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    # Programs [0, num_seqs) take the first query block of sequence pid0 (no
    # lookup: decode rows start immediately); programs past num_seqs take the
    # remaining query blocks of prefill rows. Sequence s owns the extra range
    # [cu[s] // BLOCK_Q, cu[s + 1] // BLOCK_Q), which covers its
    # ceil(len / BLOCK_Q) - 1 extra blocks, so num_seqs + T // BLOCK_Q programs
    # (vLLM's grid) suffice.
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    pid0 = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    if pid0 < num_seqs:
        seq_idx = pid0
        cur_start = tl.load(query_start_len_ptr + seq_idx)
        q_block_local_idx = 0
    else:
        extra_idx = pid0 - num_seqs
        offs = tl.arange(0, MAX_SEQS_PADDED)
        vals = tl.load(query_start_len_ptr + offs, mask=offs < num_seqs, other=0x3FFFFFFF)
        seq_idx = tl.sum((vals // BLOCK_Q <= extra_idx).to(tl.int32), axis=0) - 1
        cur_start = tl.load(query_start_len_ptr + seq_idx)
        q_block_local_idx = extra_idx - cur_start // BLOCK_Q + 1
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if DECODE_HEADS > 0 and cur_batch_query_len == 1:
        if kv_head_idx * DECODE_HEADS < num_query_heads // num_queries_per_kv:
            _decode_group(
                output_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                scale,
                seq_idx,
                kv_head_idx * DECODE_HEADS,
                cur_start,
                seq_len,
                num_query_heads,
                num_queries_per_kv,
                block_table_stride,
                query_stride_0,
                query_stride_1,
                output_stride_0,
                output_stride_1,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
                DECODE_HEADS,
                DECODE_TILE,
                DECODE_M,
            )
    else:
        _prefill_block(
            output_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            scale,
            seq_idx,
            kv_head_idx,
            q_block_local_idx,
            cur_start,
            cur_batch_query_len,
            seq_len,
            num_query_heads,
            num_queries_per_kv,
            block_table_stride,
            query_stride_0,
            query_stride_1,
            output_stride_0,
            output_stride_1,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            BLOCK_Q,
            BLOCK_M,
        )


# Grouping two KV heads per decode program halves the launch to one wave for
# ~128-row decode batches on H200; below that the per-head programs are faster.
_GROUPED_DECODE_MIN_SEQS = 96
# Launch of a pure decode batch (query length 1) with grouped heads. Measured
# on H200 at 120 rows, ctx 100-300, GQA 32/8, D 128: 2 warps / 3 stages runs
# 31 us vs 38 us for the vLLM 2D kernel. Mixed batches keep Triton defaults.
_DECODE_NUM_WARPS = 2
_DECODE_NUM_STAGES = 3


def paged_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    *,
    decode_heads=None,
    decode_tile=32,
    tile_size=32,
    num_warps=None,
    num_stages=None,
):
    """Causal paged attention for any mix of query lengths (see module docstring).

    ``decode_heads=None`` picks the grouping from the (padded) batch size;
    ``0`` sends decode rows through the vLLM-equivalent prefill path.
    """
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    if decode_heads is None:
        decode_heads = 2 if num_seqs >= _GROUPED_DECODE_MIN_SEQS and num_kv_heads % 2 == 0 else 1
    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    decode_m = max(16, triton.next_power_of_2(num_queries_per_kv * max(decode_heads, 1)))
    kwargs = {}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    grid = (q.shape[0] // BLOCK_Q + num_seqs, num_kv_heads)
    _paged_attention_kernel[grid](
        out,
        q,
        k,
        v,
        block_table,
        seqused_k,
        cu_seqlens_q,
        softmax_scale,
        num_seqs,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        # One specialization for every batch up to 255 sequences (graph capture).
        MAX_SEQS_PADDED=max(256, triton.next_power_of_2(num_seqs + 1)),
        DECODE_HEADS=decode_heads,
        DECODE_TILE=decode_tile,
        DECODE_M=decode_m,
        USE_PDL=_TRITON_PDL,
        launch_pdl=_TRITON_PDL,
        **kwargs,
    )


def _supported(
    q,
    k,
    out,
    causal,
    window_size,
    softcap,
    q_descale,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    mm_prefix_range,
    rswa_prefix_lens,
    rswa_window,
    kv_quant_mode,
    chunk_lookback,
    use_td,
    mm_prefix_clamp_sliding_window,
) -> bool:
    from vllm.v1.kv_cache_interface import KVQuantMode

    head_size = q.shape[2]
    return (
        causal is True
        and window_size[0] < 0
        and not softcap
        and q_descale is None
        and alibi_slopes is None
        and output_scale is None
        and qq_bias is None
        and sinks is None
        and mm_prefix_range is None
        and (rswa_window is None or rswa_prefix_lens is None)
        and kv_quant_mode == KVQuantMode.NONE
        and chunk_lookback == -1
        and not use_td
        and not mm_prefix_clamp_sliding_window
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and out.dtype == q.dtype
        and head_size == triton.next_power_of_2(head_size)
        and q.shape[1] % k.shape[2] == 0
        and q.shape[1] // k.shape[2] <= 16
    )


def _uses_3d(
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
    max_seqlen_q,
    num_seqs,
) -> bool:
    # Mirror of unified_attention's split-KV (3D) decision; those calls stay on vLLM.
    from vllm.v1.attention.ops import triton_unified_attention as tua

    return not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or tua.is_batch_invariant
    )


@triton.jit
def _cache_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # vLLM's reshape_and_cache_kernel_flash (bf16, [blocks, slots, heads, dim] pages) with a PDL launch: it waits
    # for the RoPE kernel that produced key/value and lets the decode attention launch behind it.
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    tile_pos = tl.program_id(axis=1) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    cur_head = tile_pos // head_size
    cur_dim = tile_pos % head_size
    tgt = block_idx * block_stride + block_offset * page_stride + cur_head * head_stride + cur_dim
    mask = tile_pos < (num_heads * head_size)
    k = tl.load(key_ptr + token_idx * key_stride + tile_pos, mask=mask)
    v = tl.load(value_ptr + token_idx * value_stride + tile_pos, mask=mask)
    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def _cache_write_pdl(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale, upstream=None):
    native = kv_cache_dtype in ("auto", "bfloat16", "float16") and key_cache.dtype == key.dtype
    if not (_TRITON_PDL and native and key_cache.ndim == 4 and key.stride(-1) == 1 and key_cache.stride(-1) == 1):
        return upstream(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
    num_heads, head_size = key.shape[1], key.shape[2]
    n = num_heads * head_size
    tile = min(2048, triton.next_power_of_2(n))
    _cache_write_kernel[(slot_mapping.shape[0], triton.cdiv(n, tile))](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(1),
        num_heads=num_heads,
        head_size=head_size,
        block_size=key_cache.shape[1],
        TILE_SIZE=tile,
        USE_PDL=True,
        launch_pdl=True,
        num_warps=16,
        num_stages=10,
    )


def install_triton_attention_patch() -> bool:
    """Route TRITON_ATTN's 2D unified_attention calls through ``paged_attention``.

    Process-wide and idempotent; unsupported variants (ALiBi, sinks, quantized
    KV, sliding window, the 3D split-KV path, ...) fall through to vLLM.
    """
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.kv_cache_interface import KVQuantMode

    upstream = triton_attn.unified_attention
    if getattr(upstream, "_omni_paged_decode", False):
        return False

    def unified_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        seq_threshold_3D=None,
        num_par_softmax_segments=None,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        use_alibi_sqrt=False,
        kv_quant_mode=KVQuantMode.NONE,
        k_scale_cache=None,
        v_scale_cache=None,
        chunk_lookback=-1,
        use_td=False,
        mm_prefix_clamp_sliding_window=False,
    ):
        if _supported(
            q,
            k,
            out,
            causal,
            window_size,
            softcap,
            q_descale,
            alibi_slopes,
            output_scale,
            qq_bias,
            sinks,
            mm_prefix_range,
            rswa_prefix_lens,
            rswa_window,
            kv_quant_mode,
            chunk_lookback,
            use_td,
            mm_prefix_clamp_sliding_window,
        ) and not _uses_3d(
            seq_threshold_3D,
            num_par_softmax_segments,
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            max_seqlen_q,
            len(seqused_k),
        ):
            launch = {}
            if max_seqlen_q == 1 and len(seqused_k) >= _GROUPED_DECODE_MIN_SEQS:
                launch = {"num_warps": _DECODE_NUM_WARPS, "num_stages": _DECODE_NUM_STAGES}
            paged_attention(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, **launch)
            return
        return upstream(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size,
            block_table,
            softcap,
            q_descale,
            k_descale,
            v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            alibi_slopes=alibi_slopes,
            output_scale=output_scale,
            qq_bias=qq_bias,
            sinks=sinks,
            mm_prefix_range=mm_prefix_range,
            rswa_prefix_lens=rswa_prefix_lens,
            rswa_window=rswa_window,
            use_alibi_sqrt=use_alibi_sqrt,
            kv_quant_mode=kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=chunk_lookback,
            use_td=use_td,
            mm_prefix_clamp_sliding_window=mm_prefix_clamp_sliding_window,
        )

    setattr(unified_attention, "_omni_paged_decode", True)
    setattr(unified_attention, "__wrapped__", upstream)
    triton_attn.unified_attention = unified_attention
    if _TRITON_PDL:
        import functools

        cache_upstream = triton_attn.triton_reshape_and_cache_flash
        triton_attn.triton_reshape_and_cache_flash = functools.partial(_cache_write_pdl, upstream=cache_upstream)
    return True
