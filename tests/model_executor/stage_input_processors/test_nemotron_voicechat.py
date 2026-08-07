# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
    _FULL_PAYLOAD_REPLACE_KEYS,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
    thinker2talker_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PAD = 12


def _thinker_output(prompt_len: int, generated: list[int]):
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=list(range(100, 100 + prompt_len)) + [_PAD],
        additional_information=None,
        outputs=[SimpleNamespace(token_ids=generated, cumulative_token_ids=None)],
    )


def test_thinker2talker_builds_pad_prompt_timeline() -> None:
    generated = [7, 8, 9, 10]
    inputs = thinker2talker_token_only([_thinker_output(prompt_len=5, generated=generated)])
    assert len(inputs) == 1
    prompt = inputs[0]
    # 1-token placeholder prompt; the timeline rides additional_information.
    assert prompt["prompt_token_ids"] == [0]
    info = prompt["additional_information"]
    assert info["nvc_logical_prompt_len"] == 5
    assert info["nvc_frame_count"] == 4
    assert info["nvc_text_timeline"] == [_PAD] * 5 + generated


def test_thinker2talker_skips_unfinished() -> None:
    out = _thinker_output(3, [1])
    out.finished = False
    assert thinker2talker_token_only([out]) == []


def test_full_payload_trims_prompt_region_rows() -> None:
    # Rows correspond to timeline steps t=1..T-1; with prompt_len P the NeMo
    # trim keeps rows for t >= P, i.e. row index P-1 onward.
    prompt_len = 4
    frames, q = 10, 31
    codes = torch.arange(frames * q, dtype=torch.long).reshape(frames, q) % 1024
    request = SimpleNamespace(
        additional_information={
            "codes": {"audio": codes},
            "nvc_logical_prompt_len": prompt_len,
        },
        additional_information_cpu=None,
    )
    payload = talker2code2wav_full_payload(request=request, is_finished=True)
    assert payload.codes.audio.shape == (frames - (prompt_len - 1), q)
    assert torch.equal(payload.codes.audio, codes[prompt_len - 1 :])
    assert bool(payload.meta.finished)


def test_full_payload_without_codes_is_empty_finished() -> None:
    # The sync producer must never emit a meta-only "pending" struct (it would
    # be shipped as the real payload); with no codes it ships an empty finished
    # payload regardless of the request's is_finished flag.
    payload = talker2code2wav_full_payload(request=SimpleNamespace(), is_finished=False)
    assert payload is not None
    assert bool(payload.meta.finished)
    assert payload.codes.audio.numel() == 0


def test_full_payload_replace_semantics_declared() -> None:
    # Full-sequence tensors must REPLACE in the accumulator, not concatenate.
    # The AR runner flattens nested payloads before accumulation, so the
    # flattened "codes.audio" key is the one that matters.
    assert "codes.audio" in _FULL_PAYLOAD_REPLACE_KEYS
    assert "codes" in _FULL_PAYLOAD_REPLACE_KEYS


def test_full_payload_missing_prompt_len_fails_explicitly() -> None:
    codes = torch.zeros((5, 31), dtype=torch.long)
    request = SimpleNamespace(
        additional_information={"codes": {"audio": codes}},
        additional_information_cpu=None,
    )
    with pytest.raises(ValueError, match="nvc_logical_prompt_len"):
        talker2code2wav_full_payload(request=request, is_finished=True)


def test_code2wav_distinguishes_empty_payload_from_missing_info() -> None:
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_code2wav import (
        NemotronVoiceChatCode2Wav,
    )

    extract = NemotronVoiceChatCode2Wav._codes_from_runtime_info
    # Explicit empty payload (e.g. zero frames after the prompt trim): the
    # empty tensor comes back as-is -> empty audio, never a placeholder-id
    # decode.
    empty = extract(None, {"codes": {"audio": torch.empty(0, dtype=torch.long)}})
    assert isinstance(empty, torch.Tensor) and empty.numel() == 0
    # Missing codes info entirely -> None (forward raises: placeholder
    # input_ids are never decoded).
    assert extract(None, {"meta": {}}) is None
    assert extract(None, None) is None
    # Normal payload passes through (nested and flattened key forms).
    codes = torch.zeros((3, 31), dtype=torch.long)
    assert extract(None, {"codes": {"audio": codes}}) is codes
    assert extract(None, {"codes.audio": codes}) is codes


def test_validate_code_stack_negatives() -> None:
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_code2wav import (
        validate_code_stack,
    )

    good = torch.zeros((4, 31), dtype=torch.long)
    assert validate_code_stack(good, 31, 1024).shape == (4, 31)
    with pytest.raises(ValueError, match="not divisible"):
        validate_code_stack(torch.zeros(100, dtype=torch.long), 31, 1024)
    with pytest.raises(ValueError, match="shaped"):
        validate_code_stack(torch.zeros((4, 30), dtype=torch.long), 31, 1024)
    bad = good.clone()
    bad[1, 5] = 1024
    with pytest.raises(ValueError, match="out-of-range"):
        validate_code_stack(bad, 31, 1024)


def test_tokenizer_resolution_failure_is_actionable(monkeypatch) -> None:
    from transformers import AutoConfig

    from vllm_omni.model_executor.models.nemotron_voicechat.configuration_nemotron_voicechat import (
        NemotronVoiceChatConfig,
    )

    def _boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(AutoConfig, "from_pretrained", _boom)
    with pytest.raises(RuntimeError, match="NEMOTRON_VOICECHAT_LLM_PATH"):
        NemotronVoiceChatConfig(
            data={"frame_length": 0.08, "source_sample_rate": 16000, "target_sample_rate": 22050},
            model={
                "inference_speaker_name": "Aria",
                "stt": {"model": {"perception": {"encoder": {}}, "pretrained_llm": "nvidia/nope"}},
                "speech_generation": {"model": {"tts_config": {}, "codec_config": {}}},
            },
        )


def test_sanitize_tts_model_cfg_disables_pretrained_loads(monkeypatch) -> None:
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
        sanitize_tts_model_cfg,
    )

    original = {
        "pretrained_lm_name": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "pretrained_codec_model": "/some/external/codec.ckpt",
        "tts_config": {
            "pretrained_text_name": "some/hub-llm",
            "backbone_type": "gemma3_text",
            "cas_config": {"pretrained_tokenizer_name": "some/hub-tokenizer", "keep": 1},
        },
    }
    monkeypatch.delenv("NEMOTRON_VOICECHAT_LLM_PATH", raising=False)
    cfg = sanitize_tts_model_cfg(original)
    # External weight-load triggers are gone; construction is config-only.
    assert "pretrained_text_name" not in cfg["tts_config"]
    assert "pretrained_codec_model" not in cfg
    assert "pretrained_tokenizer_name" not in cfg["tts_config"]["cas_config"]
    assert cfg["tts_config"]["cas_config"]["keep"] == 1
    # Tokenizer reference survives.
    assert cfg["pretrained_lm_name"] == "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    # The shared stage config is never mutated (deep copy).
    assert original["tts_config"]["pretrained_text_name"] == "some/hub-llm"

    # Air-gapped override reroutes the tokenizer reference.
    monkeypatch.setenv("NEMOTRON_VOICECHAT_LLM_PATH", "/local/nemotron")
    cfg = sanitize_tts_model_cfg(original)
    assert cfg["pretrained_lm_name"] == "/local/nemotron"


def test_code2wav_token_only_sizes_placeholder() -> None:
    talker_output = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(token_ids=[0] * 9, cumulative_token_ids=None)],
    )
    inputs = talker2code2wav_token_only([talker_output])
    assert len(inputs) == 1
    assert len(inputs[0]["prompt_token_ids"]) == 9


# =========================
# Async-chunk (streaming) producers
# =========================


def _fake_transfer_manager(codec_chunk_frames: int | None = None):
    extra = {}
    if codec_chunk_frames is not None:
        extra["codec_chunk_frames"] = codec_chunk_frames
    return SimpleNamespace(
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
        put_req_chunk={},
    )


def _streaming_request(prompt_len: int, generated: list[int], info: dict | None = None):
    return SimpleNamespace(
        external_req_id="req-0",
        prompt_token_ids=list(range(100, 100 + prompt_len)) + [_PAD],
        output_token_ids=generated,
        additional_information=info,
    )


def test_thinker2talker_async_chunk_cumulative_timeline() -> None:
    from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
        thinker2talker_async_chunk,
    )

    tm = _fake_transfer_manager()
    req = _streaming_request(prompt_len=3, generated=[7])
    payload = thinker2talker_async_chunk(tm, {}, req, is_finished=False)
    assert payload is not None
    assert payload.ids.all == [_PAD] * 3 + [7]
    assert payload.meta.num_processed_tokens == 3
    assert not bool(payload.meta.finished)

    # No new tokens since the last chunk -> skip.
    assert thinker2talker_async_chunk(tm, {}, req, is_finished=False) is None

    # More tokens -> longer cumulative timeline; final chunk sets finished.
    req.output_token_ids = [7, 8, 9]
    payload = thinker2talker_async_chunk(tm, {}, req, is_finished=True)
    assert payload.ids.all == [_PAD] * 3 + [7, 8, 9]
    assert bool(payload.meta.finished)


def test_talker2code2wav_async_chunk_cadence_and_left_context() -> None:
    from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
        talker2code2wav_async_chunk,
    )

    tm = _fake_transfer_manager(codec_chunk_frames=2)
    prompt_len = 3
    info = {"nvc_logical_prompt_len": prompt_len}
    req = _streaming_request(prompt_len=prompt_len, generated=[], info=info)

    def cumulative(rows: int) -> dict:
        # Talker rows are timeline steps t=1..; prompt region occupies the
        # first prompt_len-1 rows before acoustic frames start.
        return {"codes": {"audio": torch.arange((prompt_len - 1 + rows) * 31).reshape(-1, 31)}}

    # Only prompt-region rows so far -> nothing to ship.
    assert talker2code2wav_async_chunk(tm, cumulative(0), req, is_finished=False) is None
    # One new frame < chunk size 2 -> hold.
    assert talker2code2wav_async_chunk(tm, cumulative(1), req, is_finished=False) is None
    # Two frames -> first chunk, no left context.
    payload = talker2code2wav_async_chunk(tm, cumulative(2), req, is_finished=False)
    assert payload is not None
    assert payload.meta.left_context_size == 0
    assert payload.codes.audio.shape == (2, 31)
    # Third frame + finished -> cumulative history with left context 2.
    payload = talker2code2wav_async_chunk(tm, cumulative(3), req, is_finished=True)
    assert payload.meta.left_context_size == 2
    assert payload.codes.audio.shape == (3, 31)
    assert bool(payload.meta.finished)


def test_talker2code2wav_async_chunk_requires_prompt_len() -> None:
    from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
        talker2code2wav_async_chunk,
    )

    tm = _fake_transfer_manager()
    req = _streaming_request(prompt_len=3, generated=[], info=None)
    with pytest.raises(ValueError, match="logical prompt length"):
        talker2code2wav_async_chunk(tm, {"codes": {"audio": torch.zeros(5, 31, dtype=torch.long)}}, req)
