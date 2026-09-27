# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frequency-based repetition penalty over a short device-resident window."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _window_penalty(
    logits,
    slots,
    tokens,
    total_len,
    prompt_len,
    penalty_powers,
    vocab: tl.constexpr,
    logits_stride: tl.constexpr,
    token_stride: tl.constexpr,
    window: tl.constexpr,
    block: tl.constexpr,
    history_block: tl.constexpr,
):
    row = tl.program_id(0)
    ids = tl.program_id(1) * block + tl.arange(0, block)
    slot = tl.load(slots + row).to(tl.int64)
    end = tl.load(total_len + slot)
    start = tl.maximum(end - window, tl.load(prompt_len + slot))
    offsets = tl.arange(0, history_block)
    positions = end - window + offsets
    history = tl.load(
        tokens + slot * token_stride + positions,
        (offsets < window) & (positions >= start) & (positions < end),
        other=-1,
    )
    counts = tl.sum((ids[:, None] == history[None, :]).to(tl.int32), axis=1)
    alpha = tl.load(penalty_powers + slot * (window + 1) + counts)
    value = tl.load(logits + row * logits_stride + ids, ids < vocab, other=0).to(tl.float32)
    value = tl.where(value < 0, value * alpha, tl.div_rn(value, alpha))
    tl.store(logits + row * logits_stride + ids, value, ids < vocab)


def apply_window_penalty(
    logits: torch.Tensor,
    slots: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
    penalty: torch.Tensor,
    *,
    window_size: int,
    penalty_powers: torch.Tensor | None = None,
) -> None:
    """Update FP32 logits without copying full histories or allocating a count matrix.

    Slots/lengths follow the MRv2 request-state contract. Only output tokens in
    the most recent window contribute; prompt tokens and out-of-vocabulary
    history entries never count. Calls remain ordered on the current stream.
    """
    if logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError("fused window penalty requires CUDA FP32 logits")
    if logits.stride(1) != 1 or all_token_ids.stride(1) != 1 or not 0 < window_size <= 32:
        raise ValueError("fused window penalty requires contiguous rows and a window of 1..32")
    rows, vocab = logits.shape
    if rows:
        if penalty_powers is None:
            penalty_powers = prepare_window_penalty(penalty, window_size)
        _window_penalty[(rows, triton.cdiv(vocab, 256))](
            logits,
            slots,
            all_token_ids,
            total_len,
            prompt_len,
            penalty_powers,
            vocab,
            logits.stride(0),
            all_token_ids.stride(0),
            window_size,
            256,
            triton.next_power_of_2(window_size),
            num_warps=4,
        )


def prepare_window_penalty(penalty: torch.Tensor, window_size: int) -> torch.Tensor:
    """Cache ATen's powers when request parameters change, preserving its rounding."""
    counts = torch.arange(window_size + 1, device=penalty.device, dtype=torch.float32)
    return torch.pow(penalty[:, None], counts[None, :])
