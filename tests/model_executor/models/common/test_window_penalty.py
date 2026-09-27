# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused window penalty preserves the codec sampler's scored logits."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("window", [1, 16, 32])
@pytest.mark.parametrize("cached_powers", [False, True])
def test_window_penalty_matches_reference_and_graph(batch, window, cached_powers):
    from vllm_omni.model_executor.models.common.window_penalty import apply_window_penalty, prepare_window_penalty
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _apply_codec_window_penalty_gpu,
    )

    torch.manual_seed(312)
    vocab, capacity, length = 6562, batch + 4, 4096
    tokens = torch.randint(-1, 40, (capacity, length), device="cuda", dtype=torch.int32)
    # Duplicate histories, invalid IDs, empty outputs, short histories and slot reordering.
    tokens[:, -window:] = 7
    tokens[::2, 10:20] = vocab + 1
    prompt = torch.randint(1, 30, (capacity,), device="cuda", dtype=torch.int32)
    total = prompt + torch.arange(capacity, device="cuda", dtype=torch.int32)
    total[-1] = length
    penalty = torch.linspace(1.0, 2.0, capacity, device="cuda")
    slots = torch.randperm(capacity, device="cuda")[:batch]
    logits = torch.randn(batch, vocab, device="cuda") * 10
    logits[:, :2] = torch.tensor([-torch.inf, 0.0], device="cuda")
    reference, actual = logits.clone(), logits.clone()
    args = (slots, tokens, total, prompt, penalty)
    powers = prepare_window_penalty(penalty, window) if cached_powers else None
    _apply_codec_window_penalty_gpu(reference, *args, window_size=window)
    apply_window_penalty(actual, *args, window_size=window, penalty_powers=powers)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.copy_(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_window_penalty(actual, *args, window_size=window, penalty_powers=powers)
    actual.copy_(logits)
    graph.replay()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    # A replay must read current request lengths and history, not capture-time data.
    total.copy_(prompt + 20)
    tokens[:, :64] = 7
    reference.copy_(logits)
    actual.copy_(logits)
    _apply_codec_window_penalty_gpu(reference, *args, window_size=window)
    graph.replay()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
