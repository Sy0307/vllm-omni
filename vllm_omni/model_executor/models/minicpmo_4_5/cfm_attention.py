# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiled FP32 attention for the short streaming CFM windows on NVIDIA CUDA."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    q,
    k,
    v,
    mask,
    out,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qt: tl.constexpr,
    qd: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    kt: tl.constexpr,
    kd: tl.constexpr,
    vb: tl.constexpr,
    vh: tl.constexpr,
    vt: tl.constexpr,
    vd: tl.constexpr,
    mb: tl.constexpr,
    mt: tl.constexpr,
    mk: tl.constexpr,
    heads: tl.constexpr,
    nq: tl.constexpr,
    nk: tl.constexpr,
    dim: tl.constexpr,
    has_mask: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tile * block_m + tl.arange(0, block_m)
    ds = tl.arange(0, block_d)
    cols = tl.arange(0, block_n)
    queries = tl.load(
        q + b * qb + h * qh + rows[:, None] * qt + ds[None, :] * qd, (rows[:, None] < nq) & (ds[None, :] < dim), 0
    )
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    accumulator = tl.full((block_m, block_d), 0, tl.float32)
    for start in range(tl.cdiv(nk, block_n)):
        keys = start * block_n + cols
        kval = tl.load(
            k + b * kb + h * kh + keys[None, :] * kt + ds[:, None] * kd, (keys[None, :] < nk) & (ds[:, None] < dim), 0
        )
        score = tl.dot(queries, kval, input_precision="tf32x3") * scale
        valid = (rows[:, None] < nq) & (keys[None, :] < nk)
        if has_mask:
            keep = tl.load(mask + b * mb + rows[:, None] * mt + keys[None, :] * mk, valid, 0)
            valid = valid & (keep != 0)
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        # Fully masked rows produce zero, matching PyTorch SDPA.
        # Keep the running maximum at -inf until a valid tile arrives. A
        # permanent zero would underflow a later tile with very negative logits.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        correction = tl.exp(maximum - safe_max)
        prob = tl.exp(score - safe_max[:, None])
        denominator = denominator * correction + tl.sum(prob, 1)
        values = tl.load(
            v + b * vb + h * vh + keys[:, None] * vt + ds[None, :] * vd, (keys[:, None] < nk) & (ds[None, :] < dim), 0
        )
        accumulator = accumulator * correction[:, None] + tl.dot(prob, values, input_precision="tf32x3")
        maximum = new_max
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        out + ((b * nq + rows[:, None]) * heads + h) * dim + ds[None, :],
        result,
        (rows[:, None] < nq) & (ds[None, :] < dim),
    )


@torch.library.custom_op("vllm_omni::cfm_tiled_attention", mutates_args=())
def cfm_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """FP32 Q/K/V, bool [B,Q,K] mask, [B,H,Q,D] result (BQHD storage).

    Uses three TF32 products per FP32 dot; no BF16 casts or cache changes.
    Numerical equivalence is tolerance-based, not bitwise.
    """
    b, heads, nq, dim = q.shape
    nk = k.shape[2]
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("CFM tiled attention requires float32 Q/K/V")
    # Strided FP32 tiles can exceed the 99 KiB per-block limit on L4.
    # Keep the original three-stage pipeline on devices with enough SRAM.
    shared_memory = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
    num_stages = 3 if shared_memory >= 102400 else 2
    out = torch.empty((b, nq, heads, dim), device=q.device, dtype=q.dtype)
    _attention[(b, heads, triton.cdiv(nq, 32))](
        q,
        k,
        v,
        mask,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *(mask.stride() if mask is not None else (0, 0, 0)),
        heads,
        nq,
        nk,
        dim,
        mask is not None,
        dim**-0.5,
        32,
        64,
        triton.next_power_of_2(dim),
        num_warps=4,
        num_stages=num_stages,
    )
    return out.transpose(1, 2)


@cfm_attention.register_fake
def _cfm_attention_fake(q, k, v, mask=None):
    b, heads, nq, dim = q.shape
    return q.new_empty((b, nq, heads, dim)).transpose(1, 2)


@torch.library.custom_op("vllm_omni::cfm_conv_channels_last", mutates_args=())
def _conv_channels_last(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Some cuDNN IEEE kernels return NCHW even when convolution's meta kernel
    # predicts NHWC. Guarantee BLC storage at this boundary for Inductor. The
    # TF32 NHWC path needs no output copy; bias stays outside for fusion.
    out = torch.nn.functional.conv2d(x.transpose(1, 2).unsqueeze(2), weight)
    return out.squeeze(2).transpose(1, 2).contiguous()


@_conv_channels_last.register_fake
def _conv_channels_last_fake(x, weight):
    return x.new_empty((x.shape[0], x.shape[1] - weight.shape[3] + 1, weight.shape[0]))


def conv_forward_channels_last(conv, weights, x, cnn_cache):
    """Keep the two causal convolutions in BLC storage through cuDNN.

    A height-one channels-last conv2d is the original conv1d operation.
    Prepacked inference weights avoid per-step NCHW/NHWC conversions. Cache
    order and layout remain the original [B, C1+C2, 2] contract.
    """
    if cnn_cache is None:
        c1 = x.new_zeros((x.shape[0], 2, conv.in_channels))
        c2 = x.new_zeros((x.shape[0], 2, conv.out_channels))
    else:
        c1, c2 = cnn_cache.split((conv.in_channels, conv.out_channels), dim=1)
        c1, c2 = c1.transpose(1, 2), c2.transpose(1, 2)
    x1 = torch.cat((c1, x), dim=1)
    y = _conv_channels_last(x1, weights[0]) + conv.block[1].bias
    y = conv.block[4](conv.block[3](y))
    x2 = torch.cat((c2, y), dim=1)
    z = _conv_channels_last(x2, weights[1]) + conv.block[6].bias
    cache = torch.cat((x1[:, -2:].transpose(1, 2), x2[:, -2:].transpose(1, 2)), dim=1)
    return z, cache


def block_forward_optimized(block, x, time, cnn_cache, att_cache, mask, *, attention_backend, conv_weights=None):
    """CosyVoice DiT block with optional attention and convolution backends.

    The caller compiles this bound callable; original modules/methods remain
    intact for the eager fallback. Cache order stays current then history.
    """
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m, shift_c, scale_c, gate_c = block.adaLN_modulation(time).chunk(
        9, dim=-1
    )
    normalized = block.norm1(x) * (1 + scale_a) + shift_a
    if attention_backend == "tiled_fp32":
        attn = block.attn
        q = attn.q_norm(attn.to_heads(attn.to_q(normalized)))
        k = attn.k_norm(attn.to_heads(attn.to_k(normalized)))
        v = attn.to_heads(attn.to_v(normalized))
        if att_cache is not None:
            old_k, old_v = att_cache.chunk(2, dim=3)
            k = torch.cat((k, old_k), dim=2)
            v = torch.cat((v, old_v), dim=2)
        new_att = torch.cat((k, v), dim=3)
        value = cfm_attention(q, k, v, mask).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        value = attn.proj_drop(attn.proj(value))
    else:
        value, new_att = block.attn.forward_chunk(normalized, att_cache, mask)
    x = x + gate_a * value
    normalized = block.norm3(x) * (1 + scale_c) + shift_c
    if conv_weights is None:
        conv, new_cnn = block.conv.forward_chunk(normalized, cnn_cache)
    else:
        conv, new_cnn = conv_forward_channels_last(block.conv, conv_weights, normalized, cnn_cache)
    x = x + gate_c * conv
    x = x + gate_m * block.mlp(block.norm2(x) * (1 + scale_m) + shift_m)
    return x, new_cnn, new_att
