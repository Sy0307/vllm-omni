# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Talker's per-step path must not block the host on the GPU."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from tests.model_executor.models.minicpmo_4_5.test_talker_host_sync import _EOS, _infos, _make_talker, _states, _step
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _apply_batched_repetition_penalty
from vllm_omni.utils.device_copy import index_to_device

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def test_talker_step_does_not_synchronize_cuda(mocker) -> None:
    talker = _make_talker("cuda")
    states = _states()
    talker._request_audio_states = copy.deepcopy(states)
    infos = _infos(states)
    input_ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32, device="cuda")
    hidden = torch.randn(4, 4, device="cuda")
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0], device="cuda")
    # Warm the pinned host allocator and CUDA context outside the check.
    index_to_device([1], "cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            *_, sampled = _step(talker, infos, hidden, mocker, batched=True, input_ids=input_ids, penalties=penalties)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert sampled.reshape(-1).tolist()[1:] == [_EOS, _EOS, _EOS]
    assert talker._request_audio_states["req-live"]["recent_codes"] == [1, 2, 3]
    assert talker._request_audio_states["req-eos"]["finished"] is True


def test_talker_condition_upload_does_not_synchronize_cuda() -> None:
    talker = _make_talker("cuda")
    talker.emb_text = nn.Embedding(16, 4).cuda()
    talker.projector_semantic = nn.Linear(6, 4).cuda()
    talker._normalize = True
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    token_ids = torch.tensor([2, 3, 4])
    hidden_states = torch.randn(3, 6)
    expected = talker._build_condition_embeddings(token_ids.cuda(), hidden_states.cuda())
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            condition = talker._build_condition_embeddings(token_ids, hidden_states)
            boundary = talker._build_condition_embeddings(token_ids[:0], hidden_states[:0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(condition, expected)
    assert boundary.shape == (2, 4)


def test_codec_penalty_uploads_cpu_history_without_synchronizing_cuda() -> None:
    logits = torch.randn(3, 8, device="cuda")
    histories = [torch.tensor([1, 1, 4]), torch.empty(0, dtype=torch.long), torch.tensor([7])]
    penalties = torch.tensor([1.2, 1.05, 1.0], device="cuda")
    expected = _apply_batched_repetition_penalty(
        logits, [history.cuda() for history in histories], penalty=penalties, window_size=16
    )
    index_to_device([1], "cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = _apply_batched_repetition_penalty(logits, histories, penalty=penalties, window_size=16)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
