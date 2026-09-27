# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in stateless request/step Philox noise for Higgs MRV2.

The exponential distribution is unchanged; the random sequence differs from
PyTorch's CUDA generator. Slot order and peer request lengths do not affect it.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _exponential(
    out,
    seeds,
    slots,
    seq_lens,
    prompt_lens,
    num_codebooks: tl.constexpr,
    vocab_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // num_codebooks
    book = row % num_codebooks
    slot = tl.load(slots + req)
    seed = tl.load(seeds + slot).to(tl.uint64)
    step = tl.maximum(tl.load(seq_lens + req) - tl.load(prompt_lens + slot), 0)
    col = tl.arange(0, block_size)
    offset = (step * (num_codebooks * vocab_size) + book * vocab_size + col).to(tl.uint32)
    u = tl.rand(seed, offset)
    value = tl.maximum(-tl.log(1.0 - u), 1.0e-10)
    tl.store(out + row * vocab_size + col, value, col < vocab_size)


def exponential_noise(seeds, slots, seq_lens, prompt_lens, num_codebooks, vocab_size):
    # The counter is uint32. The model's bounded sequence length keeps the
    # request-step domain below wraparound; reject larger configs at setup.
    out = torch.empty((slots.numel() * num_codebooks, vocab_size), dtype=torch.float32, device=slots.device)
    _exponential[(slots.numel() * num_codebooks,)](
        out,
        seeds,
        slots,
        seq_lens,
        prompt_lens,
        num_codebooks,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        num_warps=4,
    )
    return out
