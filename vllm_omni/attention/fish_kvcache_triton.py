# ruff: noqa: N803

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on optional Triton install
    triton = None
    tl = None
    _LOAD_ERROR: Exception | None = exc
else:
    _LOAD_ERROR = None

_BLOCK_SIZE = 16
_HEAD_DIM = 128
_SMALL_PATH_MAX_SEQ_LEN = 1024
_BLOCK_N = 64


def is_available() -> bool:
    return _LOAD_ERROR is None and triton is not None and tl is not None


def load_error() -> Exception | None:
    return _LOAD_ERROR


if is_available():

    @triton.jit
    def _small_decode_kernel(
        Q,
        K,
        V,
        BLOCK_TABLE,
        SEQ_LENS,
        OUT,
        SCALE: tl.constexpr,
        MAX_SEQ_LEN: tl.constexpr,
        MAX_BLOCKS_PER_SEQ: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        KV_GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        kv_head = tl.program_id(1)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        q_heads = kv_head * KV_GROUP + offs_h
        mask_h = offs_h < KV_GROUP

        q_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        q = tl.load(Q + q_offsets, mask=mask_h[:, None], other=0.0)

        seq_len = tl.load(SEQ_LENS + batch_id)
        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        for start_n in tl.range(0, MAX_SEQ_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            mask_n = cur_n < seq_len
            logical_block = cur_n // 16
            block_offset = cur_n - logical_block * 16
            physical_block = tl.load(
                BLOCK_TABLE + batch_id * MAX_BLOCKS_PER_SEQ + logical_block,
                mask=mask_n,
                other=0,
            )

            kv_offsets = (
                (physical_block[:, None] * 16 + block_offset[:, None]) * NUM_KV_HEADS + kv_head
            ) * BLOCK_D + offs_d[None, :]
            k = tl.load(K + kv_offsets, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, -float("inf"))

            v = tl.load(V + kv_offsets, mask=mask_n[:, None], other=0.0)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        tl.store(OUT + out_offsets, acc / l_i[:, None], mask=mask_h[:, None])

    @triton.jit
    def _streaming_decode_kernel(
        Q,
        K,
        V,
        BLOCK_TABLE,
        SEQ_LENS,
        OUT,
        SCALE: tl.constexpr,
        MAX_BLOCKS_PER_SEQ: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        KV_GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        kv_head = tl.program_id(1)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        q_heads = kv_head * KV_GROUP + offs_h
        mask_h = offs_h < KV_GROUP

        q_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        q = tl.load(Q + q_offsets, mask=mask_h[:, None], other=0.0)

        seq_len = tl.load(SEQ_LENS + batch_id)
        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        start_n = 0
        while start_n < seq_len:
            cur_n = start_n + offs_n
            mask_n = cur_n < seq_len
            logical_block = cur_n // 16
            block_offset = cur_n - logical_block * 16
            physical_block = tl.load(
                BLOCK_TABLE + batch_id * MAX_BLOCKS_PER_SEQ + logical_block,
                mask=mask_n,
                other=0,
            )

            kv_offsets = (
                (physical_block[:, None] * 16 + block_offset[:, None]) * NUM_KV_HEADS + kv_head
            ) * BLOCK_D + offs_d[None, :]
            k = tl.load(K + kv_offsets, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, -float("inf"))

            v = tl.load(V + kv_offsets, mask=mask_n[:, None], other=0.0)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            start_n += BLOCK_N

        out_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        tl.store(OUT + out_offsets, acc / l_i[:, None], mask=mask_h[:, None])


def fish_decode_kvcache_attn_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    max_seq_len: int,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError(f"Fish Triton attention is unavailable: {_LOAD_ERROR!r}")
    del partial_m, partial_l, partial_acc

    batch_size, num_q_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    if head_dim != _HEAD_DIM or block_size != _BLOCK_SIZE:
        raise RuntimeError("Fish Triton attention only supports head_dim=128 and block_size=16")
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError("num_q_heads must be divisible by num_kv_heads")

    kv_group = num_q_heads // num_kv_heads
    block_h = triton.next_power_of_2(kv_group)
    max_blocks_per_seq = block_table.shape[1]

    if max_seq_len <= _SMALL_PATH_MAX_SEQ_LEN:
        _small_decode_kernel[(batch_size, num_kv_heads)](
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out,
            SCALE=float(scale),
            MAX_SEQ_LEN=int(max_seq_len),
            MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            KV_GROUP=kv_group,
            BLOCK_H=block_h,
            BLOCK_D=head_dim,
            BLOCK_N=_BLOCK_N,
            num_warps=4,
            num_stages=3,
        )
        return out

    _streaming_decode_kernel[(batch_size, num_kv_heads)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        SCALE=float(scale),
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        KV_GROUP=kv_group,
        BLOCK_H=block_h,
        BLOCK_D=head_dim,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return out
