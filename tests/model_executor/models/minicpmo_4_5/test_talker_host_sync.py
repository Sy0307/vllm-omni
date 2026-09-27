# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 Talker per-step path: batched, host-sync-free, same results."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    ConditionalChatTTSConfig,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 8
_EOS = 7


@dataclass
class _SamplingMetadata:
    repetition_penalties: torch.Tensor
    prompt_token_ids: torch.Tensor | None = None
    no_penalties: bool = False


@dataclass
class _SamplerOutput:
    sampled_token_ids: torch.Tensor


def _make_talker(device: str = "cpu") -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = _VOCAB
    talker._codec_eos_id = _EOS
    talker._force_eos_rows = None
    talker._mask_eos_rows = None
    talker._pending_force_eos_rows = None
    talker._penalty_histories = None
    talker._request_audio_states = {}
    talker._request_condition_states = {}
    talker._deferred_cleanup_ids = set()
    talker._tts_config = ConditionalChatTTSConfig()
    generator = torch.Generator().manual_seed(0)
    talker.emb_code = nn.ModuleList([nn.Embedding(_VOCAB, 4)])
    talker.head_code = nn.ModuleList([nn.Linear(4, _VOCAB, bias=False)])
    with torch.no_grad():
        talker.emb_code[0].weight.copy_(torch.randn(_VOCAB, 4, generator=generator))
        talker.head_code[0].weight.copy_(torch.randn(_VOCAB, 4, generator=generator))
    return talker.to(device)


def _states() -> dict[str, dict[str, Any]]:
    return {
        # Mid-stream row: forwards its code, EOS still masked by min_tokens.
        "req-live": {"finished": False, "step": 2, "max_tokens": 100, "min_tokens": 5, "recent_codes": [1, 2]},
        # The previous step sampled codec EOS: forward nothing, finish, force EOS.
        "req-eos": {"finished": False, "step": 60, "max_tokens": 100, "recent_codes": [4, 4, 5]},
        # Leftover decode of an already-finished request.
        "req-done": {"finished": True, "step": 3, "max_tokens": 100},
        # Reaches its codec budget on this step.
        "req-cap": {"finished": False, "step": 8, "max_tokens": 10, "recent_codes": list(range(7)) * 3},
    }


def _infos(states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"request_id": request_id, "audio_state": dict(state), "_omni_is_prefill": False}
        for request_id, state in states.items()
    ]


def _store(infos: list[dict[str, Any]], updates: list[dict[str, Any]]) -> None:
    """What the runner's buffer update does with the CPU ``codes.audio`` delta."""
    for info, update in zip(infos, updates, strict=True):
        audio = update["codes"]["audio"]
        assert audio.device.type == "cpu" and audio.dtype == torch.long
        info["codes"] = {"audio": audio.contiguous()}


def _step(talker, infos, hidden, mocker, *, batched: bool, input_ids: torch.Tensor, penalties: torch.Tensor):
    if batched:
        returned_ids, embeds, updates = talker.preprocess_decode_batch(input_ids=input_ids, req_infos=infos)
        assert returned_ids is input_ids
    else:
        rows = [talker.preprocess(input_ids[row : row + 1], None, **info) for row, info in enumerate(infos)]
        embeds = torch.cat([row[1] for row in rows])
        updates = [row[2] for row in rows]
    _store(infos, updates)
    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=infos,
        request_token_spans=[(row, row + 1) for row in range(len(infos))],
    )
    logits = talker.compute_logits(output.text_hidden_states)
    captured: dict[str, torch.Tensor] = {}

    def _sampler(logits, sampling_metadata):
        captured["logits"] = logits.clone()
        return _SamplerOutput(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_sampler,
    )
    sampled = talker.sample(logits, _SamplingMetadata(repetition_penalties=penalties))
    return embeds, output, logits, captured["logits"], sampled.sampled_token_ids


def test_batched_decode_matches_scalar_preprocess(mocker) -> None:
    input_ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32)
    hidden = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0])
    results = {}
    talkers = {}
    for batched in (False, True):
        talker = _make_talker()
        states = _states()
        talker._request_audio_states = copy.deepcopy(states)
        results[batched] = _step(
            talker, _infos(states), hidden, mocker, batched=batched, input_ids=input_ids, penalties=penalties
        )
        talkers[batched] = talker

    # Embeddings, masked logits, penalized logits and sampled ids.
    for index in (0, 2, 3, 4):
        assert torch.equal(results[False][index], results[True][index])
    scalar_output, batched_output = results[False][1], results[True][1]
    scalar_codes = scalar_output.multimodal_outputs["codes"]["audio"]
    batched_codes = batched_output.multimodal_outputs["codes"]["audio"]
    assert [(t.device.type, t.dtype, t.tolist()) for t in batched_codes] == [
        (t.device.type, t.dtype, t.tolist()) for t in scalar_codes
    ]
    assert [t.tolist() for t in batched_codes] == [[[3]], [], [], [[6]]]
    assert [t.item() for t in batched_output.multimodal_outputs["meta"]["finished"]] == [False, True, True, True]
    assert [t.item() for t in scalar_output.multimodal_outputs["meta"]["finished"]] == [False, True, True, True]
    assert torch.equal(results[False][4], results[True][4])
    assert talkers[True]._request_audio_states == talkers[False]._request_audio_states
    assert talkers[True]._penalty_histories is None  # consumed by sample()
    states = talkers[True]._request_audio_states
    assert states["req-live"]["recent_codes"] == [1, 2, 3]
    assert states["req-eos"] == {"finished": True, "step": 60, "max_tokens": 100, "recent_codes": [4, 4, 5]}
    assert states["req-cap"]["finished"] is True
    assert results[True][4].reshape(-1).tolist()[1:] == [_EOS, _EOS, _EOS]
    # The finished row embeds as zeros, the others as their codec id.
    assert torch.equal(results[True][0][2], torch.zeros(4))
    assert torch.equal(results[True][0][0], talkers[True].emb_code[0].weight[3])
    assert talkers[True]._decode_codec_ids == {}


def test_batched_decode_defers_eos_and_leaves_host_state_untouched() -> None:
    talker = _make_talker()
    talker._request_audio_states["req"] = {"finished": False, "step": 4}

    _, _, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([_EOS]),
        req_infos=[{"request_id": "req", "audio_state": {"finished": False}}],
    )

    # No host read in preprocess: the id is resolved in make_omni_output.
    assert talker._request_audio_states["req"]["finished"] is False
    assert updates[0]["codes"]["audio"].numel() == 0
    assert "req" in talker._decode_codec_ids
    output = talker.make_omni_output(
        torch.ones(1, 4),
        model_intermediate_buffer=[{"request_id": "req", "codes": updates[0]["codes"]}],
        request_token_spans=[(0, 1)],
    )
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert talker._request_audio_states["req"] == {"finished": True, "step": 4}
    assert talker._decode_codec_ids == {}


def test_batched_decode_builds_the_cpu_delta_in_make_omni_output() -> None:
    talker = _make_talker()
    talker._request_audio_states["req"] = {"finished": False, "step": 0, "recent_codes": [1]}
    _, _, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([5], dtype=torch.int32),
        req_infos=[{"request_id": "req", "audio_state": {"finished": False}}],
    )

    output = talker.make_omni_output(
        torch.ones(1, 4),
        model_intermediate_buffer=[{"request_id": "req", "codes": updates[0]["codes"]}],
        request_token_spans=[(0, 1)],
    )

    delta = output.multimodal_outputs["codes"]["audio"][0]
    assert delta.device.type == "cpu" and delta.dtype == torch.long and delta.tolist() == [[5]]
    assert talker._request_audio_states["req"] == {"finished": False, "step": 1, "recent_codes": [1, 5]}
    assert talker._penalty_histories[0].device.type == "cpu"
    assert talker._penalty_histories[0].tolist() == [1, 5]


def test_batched_decode_falls_back_to_scalar_preprocess_without_state(mocker) -> None:
    talker = _make_talker()
    prefill = mocker.patch.object(
        talker,
        "preprocess",
        return_value=(None, torch.full((1, 4), 5.0), {"codes": {"audio": torch.empty(0, dtype=torch.long)}}),
    )

    _, embeds, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([2, 3]),
        req_infos=[{"request_id": "new"}, {"request_id": "old", "audio_state": {"finished": False}}],
    )

    assert prefill.call_count == 1
    assert torch.equal(embeds[0], torch.full((4,), 5.0))
    assert torch.equal(embeds[1], talker.emb_code[0].weight[3])
    assert updates[0]["codes"]["audio"].numel() == 0
    assert updates[1]["codes"]["audio"].numel() == 0
    assert set(talker._decode_codec_ids) == {"old"}


def test_compute_logits_matches_boolean_mask_indexing() -> None:
    talker = _make_talker()
    hidden = torch.randn(4, 4, generator=torch.Generator().manual_seed(3))
    force = [True, False, False, True]
    mask = [False, True, False, False]
    expected = talker.head_code[0](hidden).float().clone()
    expected[torch.tensor(force)] = float("-inf")
    expected[torch.tensor(force), _EOS] = 0.0
    expected[torch.tensor(mask), _EOS] = float("-inf")

    talker._force_eos_rows = force
    talker._mask_eos_rows = mask
    logits = talker.compute_logits(hidden)

    assert torch.equal(logits, expected)
    assert talker._pending_force_eos_rows == force


def _fake_vllm_config(stage: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            multimodal_config=None,
            model_stage=stage,
        )
    )


def test_wrapper_exposes_the_batched_decode_hook_to_the_runner(mocker) -> None:
    from vllm_omni.worker import gpu_model_runner

    talker = _make_talker()
    talker.make_empty_intermediate_tensors = lambda *args, **kwargs: None
    mocker.patch.object(minicpmo_4_5_omni, "init_vllm_registered_model", return_value=talker)

    model = MiniCPMO45OmniForConditionalGeneration(vllm_config=_fake_vllm_config("tts"))

    assert model.preprocess_decode_batch == talker.preprocess_decode_batch
    assert model.use_async_omni_output is True
    # codes.audio is a CPU transport delta: nothing is GPU-resident.
    assert not hasattr(model, "gpu_resident_buffer_keys")
    warnings = mocker.patch.object(gpu_model_runner.logger, "warning")
    gpu_model_runner.OmniGPUModelRunner._warn_unexposed_stage_hooks(model)
    warnings.assert_not_called()


def test_wrapper_keeps_thinker_on_the_scalar_preprocess_path(mocker) -> None:
    thinker = nn.Module()
    thinker.make_empty_intermediate_tensors = lambda *args, **kwargs: None
    mocker.patch.object(minicpmo_4_5_omni, "init_vllm_registered_model", return_value=thinker)

    model = MiniCPMO45OmniForConditionalGeneration(vllm_config=_fake_vllm_config("llm"))

    assert getattr(model, "preprocess_decode_batch", None) is None
    assert not hasattr(model, "gpu_resident_buffer_keys")
