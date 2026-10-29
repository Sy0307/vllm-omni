# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text-first audio embedding addition with checkpoint dtype rounding."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _embed(
    c_ptr,
    w_ptr,
    o_ptr,
    base_ptr,
    bs0: tl.constexpr,
    bs1: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    nq: tl.constexpr,
    v: tl.constexpr,
    h: tl.constexpr,
    pad: tl.constexpr,
    bd: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * bd + tl.arange(0, bd)
    total = tl.load(base_ptr + row * bs0 + dims * bs1, dims < h, 0).to(tl.float32)
    for q in tl.static_range(nq):
        code = tl.load(c_ptr + row * cs0 + q * cs1)
        safe_code = tl.minimum(tl.maximum(code, 0), v - 1)
        value = tl.load(w_ptr + (q * v + safe_code) * h + dims, (code != pad) & (dims < h), 0)
        total = (total + value.to(tl.float32)).to(o_ptr.dtype.element_ty).to(tl.float32)
    tl.store(o_ptr + row * h + dims, total, dims < h)


@torch.library.custom_op("vllm_omni::moss_audio_embed", mutates_args=())
def audio_embed(codes: torch.Tensor, weights: torch.Tensor, pad: int, base: torch.Tensor) -> torch.Tensor:
    t, nq = codes.shape
    assert weights.is_contiguous() and weights.shape[0] == nq
    assert codes.device == weights.device and codes.device.type == "cuda"
    out = torch.empty((t, weights.shape[2]), device=codes.device, dtype=weights.dtype)
    if t:
        _embed[(t, triton.cdiv(weights.shape[2], 256))](
            codes,
            weights,
            out,
            base,
            *base.stride(),
            *codes.stride(),
            nq,
            weights.shape[1],
            weights.shape[2],
            pad,
            256,
            num_warps=4,
        )
    return out


@audio_embed.register_fake
def _(codes: torch.Tensor, weights: torch.Tensor, pad: int, base: torch.Tensor) -> torch.Tensor:
    return weights.new_empty((codes.shape[0], weights.shape[2]))
