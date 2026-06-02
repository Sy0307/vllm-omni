# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    StepAudio2Token2WavCore,
    StepAudio2Token2WavForConditionalGeneration,
    _StreamState,
)


class _DummyModelConfig:
    def __init__(self) -> None:
        self.hf_config = SimpleNamespace(
            token2wav_path="/tmp/fake-token2wav",
            token2wav_float16=False,
        )
        self.model = "/tmp/fake-model"

    def get_hidden_size(self) -> int:
        return 8


def _build_model() -> StepAudio2Token2WavForConditionalGeneration:
    vllm_config = SimpleNamespace(
        model_config=_DummyModelConfig(),
        device_config=SimpleNamespace(device="cpu"),
    )
    return StepAudio2Token2WavForConditionalGeneration(vllm_config=vllm_config)


def test_step_audio2_prompt_warmup_restores_rng(monkeypatch):
    core = StepAudio2Token2WavCore(model_path="/tmp/fake-token2wav", device="cpu")
    prompt_wav = "/tmp/default.wav"
    calls = {"load": 0, "prepare": 0}

    def _ensure_models_loaded():
        calls["load"] += 1
        torch.rand(4)

    def _prepare_prompt(prompt):
        calls["prepare"] += 1
        torch.rand(4)
        return (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.ones(1, 192),
            torch.ones(1, 6, 80),
            torch.tensor([6], dtype=torch.int32),
        )

    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr(core, "_ensure_models_loaded", _ensure_models_loaded)
    monkeypatch.setattr(core, "_prepare_prompt", _prepare_prompt)

    torch.manual_seed(123)
    expected = torch.rand(8)
    torch.manual_seed(123)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        core.warmup_prompt(prompt_wav)
        actual_dtype = torch.get_default_dtype()
        actual = torch.rand(8, dtype=torch.float32)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert calls == {"load": 1, "prepare": 1}
    assert prompt_wav in core.cache
    assert actual_dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_step_audio2_token2wav_sync_path_not_misdetected_by_empty_runtime_info(monkeypatch):
    model = _build_model()

    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr(
        model.token2wav,
        "forward",
        lambda *args, **kwargs: torch.zeros(1, 16, dtype=torch.float32),
    )

    out = model.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        runtime_additional_information=[{}],
    )

    audio = out.multimodal_outputs["model_outputs"]
    assert isinstance(audio, torch.Tensor)
    assert model._stream_states_by_req == {}
    assert model._legacy_stream_state is None


def test_step_audio2_token2wav_async_chunk_batch_guard():
    model = _build_model()

    with pytest.raises(RuntimeError, match="only supports batch=1"):
        model._forward_async_chunk(
            input_ids=torch.tensor([1, 2, 3]),
            runtime_additional_information=[
                {"left_context_size": 0},
                {"left_context_size": 1},
            ],
        )


def test_step_audio2_token2wav_async_chunk_last_chunk_resets_state(monkeypatch):
    model = _build_model()

    monkeypatch.setattr("os.path.exists", lambda _: True)
    calls = {"setup": 0, "stream": 0, "reset": 0}

    def _setup_stream_for(prompt_wav, state):
        calls["setup"] += 1
        state.setup_done = True
        state.stream_cache = {}
        state.hift_cache_dict = {
            "mel": torch.zeros(1, 1, 0),
            "source": torch.zeros(1, 1, 0),
            "speech": torch.zeros(1, 0),
        }

    def _stream_chunk_for(audio_tokens, prompt_wav, last_chunk, state):
        calls["stream"] += 1
        return torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    def _reset_stream_for(state):
        calls["reset"] += 1
        state.setup_done = False
        state.finished = True
        state.stream_cache = None
        state.hift_cache_dict = {}

    monkeypatch.setattr(model.token2wav, "setup_stream_for", _setup_stream_for)
    monkeypatch.setattr(model.token2wav, "stream_chunk_for", _stream_chunk_for)
    monkeypatch.setattr(model.token2wav, "reset_stream_for", _reset_stream_for)

    out = model.forward(
        input_ids=torch.tensor([10, 11, 12]),
        positions=torch.tensor([0, 1, 2]),
        runtime_additional_information=[
            {"left_context_size": 1, "req_id": "rid-last", "stream_finished": True}
        ],
    )

    audio_list = out.multimodal_outputs["model_outputs"]
    assert isinstance(audio_list, list)
    assert len(audio_list) == 1
    assert torch.allclose(audio_list[0], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert calls == {"setup": 1, "stream": 1, "reset": 1}
    assert model._stream_states_by_req == {}


def test_step_audio2_token2wav_async_chunk_empty_eof_returns_zero_chunk():
    model = _build_model()

    out = model._forward_async_chunk(
        input_ids=torch.tensor([], dtype=torch.int64),
        runtime_additional_information=[{"left_context_size": 1, "req_id": "rid-empty"}],
    )

    audio_list = out.multimodal_outputs["model_outputs"]
    assert isinstance(audio_list, list)
    assert len(audio_list) == 1
    assert torch.equal(audio_list[0], torch.zeros(1, dtype=torch.float32))
    assert model._stream_states_by_req == {}


def test_step_audio2_token2wav_async_chunk_keeps_request_keyed_states(monkeypatch):
    model = _build_model()

    monkeypatch.setattr("os.path.exists", lambda _: True)
    calls = {"setup": [], "stream": [], "reset": []}

    def _setup_stream_for(prompt_wav, state):
        calls["setup"].append(id(state))
        state.setup_done = True
        state.stream_cache = {}
        state.hift_cache_dict = {
            "mel": torch.zeros(1, 1, 0),
            "source": torch.zeros(1, 1, 0),
            "speech": torch.zeros(1, 0),
        }

    def _stream_chunk_for(audio_tokens, prompt_wav, last_chunk, state):
        calls["stream"].append((tuple(audio_tokens), last_chunk, id(state)))
        return torch.tensor([0.1], dtype=torch.float32)

    def _reset_stream_for(state):
        calls["reset"].append(id(state))
        state.setup_done = False
        state.finished = True

    monkeypatch.setattr(model.token2wav, "setup_stream_for", _setup_stream_for)
    monkeypatch.setattr(model.token2wav, "stream_chunk_for", _stream_chunk_for)
    monkeypatch.setattr(model.token2wav, "reset_stream_for", _reset_stream_for)

    model.forward(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        runtime_additional_information=[{"left_context_size": 0, "req_id": "rid-a"}],
    )
    model.forward(
        input_ids=torch.tensor([3, 4]),
        positions=torch.tensor([0, 1]),
        runtime_additional_information=[{"left_context_size": 0, "req_id": "rid-b"}],
    )

    assert set(model._stream_states_by_req) == {"rid-a", "rid-b"}
    assert len(set(calls["setup"])) == 2

    state_b = model._stream_states_by_req["rid-b"]
    model.forward(
        input_ids=torch.tensor([5]),
        positions=torch.tensor([0]),
        runtime_additional_information=[
            {"left_context_size": 1, "req_id": "rid-a", "stream_finished": True}
        ],
    )

    assert set(model._stream_states_by_req) == {"rid-b"}
    assert model._stream_states_by_req["rid-b"] is state_b
    assert len(calls["reset"]) == 1
