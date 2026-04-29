import sys
import types

import pytest

sys.modules.setdefault("torch", types.SimpleNamespace())

from examples.online_serving.qwen3_tts_nv_triton import benchmark_model


def test_parse_args_defaults_to_custom_voice(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_model.py"])

    args = benchmark_model.parse_args()

    assert args.task_type == "CustomVoice"
    assert args.model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def test_default_stage_config_uses_nv_custom_voice_talker():
    cfg = benchmark_model._build_talker_only_stage_config()
    engine_args = cfg["stage_args"][0]["engine_args"]

    assert engine_args["model_arch"] == "Qwen3TTSTalkerForConditionalGenerationNv"


def test_base_stage_config_uses_generic_talker():
    cfg = benchmark_model._build_talker_only_stage_config(task_type="Base")
    engine_args = cfg["stage_args"][0]["engine_args"]

    assert engine_args["model_arch"] == "Qwen3TTSTalkerForConditionalGeneration"


def test_base_input_includes_voice_clone_reference(monkeypatch):
    monkeypatch.setattr(benchmark_model, "_estimate_prompt_len", lambda *_args, **_kwargs: 7)

    prompt = benchmark_model.build_input(
        text="hello",
        speaker="vivian",
        language="English",
        model_name="/models/Qwen3-TTS-12Hz-1.7B-Base",
        task_type="Base",
        ref_audio="/tmp/ref.wav",
        ref_text="reference words",
    )

    assert prompt["prompt_token_ids"] == [0] * 7
    assert prompt["additional_information"] == {
        "task_type": ["Base"],
        "text": ["hello"],
        "language": ["English"],
        "ref_audio": ["/tmp/ref.wav"],
        "ref_text": ["reference words"],
        "x_vector_only_mode": [False],
    }


def test_base_input_requires_reference_audio_and_text(monkeypatch):
    monkeypatch.setattr(benchmark_model, "_estimate_prompt_len", lambda *_args, **_kwargs: 7)

    with pytest.raises(ValueError, match="--ref-audio and --ref-text"):
        benchmark_model.build_input(
            text="hello",
            speaker="vivian",
            language="English",
            model_name="/models/Qwen3-TTS-12Hz-1.7B-Base",
            task_type="Base",
            ref_audio=None,
            ref_text="reference words",
        )
