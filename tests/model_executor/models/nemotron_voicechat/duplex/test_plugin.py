# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the Nemotron VoiceChat duplex model plugin (RFC #7181)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, load_duplex_plugin, validate_duplex_plugin_sampling
from vllm_omni.model_executor.models.nemotron_voicechat.duplex import plugin as plugin_module
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
    NemotronVoiceChatClientRuntimeConfigError,
    NemotronVoiceChatDuplexPlugin,
    _render_tool_response,
)
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import NEMOTRON_VOICECHAT_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PLUGIN_PATH = "vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin.NemotronVoiceChatDuplexPlugin"

_FAKE_SPECIAL_IDS = {
    "nvc_text_bos_id": 0,
    "nvc_text_eos_id": 1,
    "nvc_text_pad_id": 12,
    "nvc_function_sotc_id": 20,
    "nvc_function_eotc_id": 21,
    "nvc_function_eotr_id": 22,
    "nvc_tokenizer_ref": "test-tokenizer",
}


def _plugin() -> NemotronVoiceChatDuplexPlugin:
    return NemotronVoiceChatDuplexPlugin(lambda *_: None)


def _frame() -> dict[str, object]:
    raw = np.zeros(1280, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(raw).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }


def _plan(plugin, *, input_seq: int, runtime_config: dict[str, object] | None = None):
    return plugin.plan_append(
        request_id="req",
        fence=DuplexFence("sid", epoch=3),
        session_config={},
        runtime_config=runtime_config
        or {
            "nvc_prompt_token_ids": [0, 42, 1],
            "nvc_text_pad_id": 12,
            "nvc_max_model_len": 8192,
        },
        seq=input_seq,
        turn_seq=input_seq,
        payload=_frame(),
        final=False,
        sampling_params=SamplingParams(),
    )


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(stt_cfg={}),
        max_model_len=8192,
    )


def _fake_tokenizer() -> SimpleNamespace:
    return SimpleNamespace(encode=lambda text, **_kwargs: [42])


@pytest.fixture
def patched_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMOTRON_VOICECHAT_LLM_PATH", raising=False)
    monkeypatch.setattr(
        plugin_module,
        "_load_tokenizer_runtime",
        lambda _model_config: (dict(_FAKE_SPECIAL_IDS), _fake_tokenizer()),
    )


def test_pipeline_declares_the_plugin() -> None:
    assert NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin == _PLUGIN_PATH
    assert NEMOTRON_VOICECHAT_PIPELINE.default_session_mode == "turn"


def test_duplex_deploy_selects_the_duplex_session_mode() -> None:
    from vllm_omni.config.stage_config import _DEPLOY_DIR, resolve_deploy_yaml

    duplex = resolve_deploy_yaml(_DEPLOY_DIR / "nemotron_labs_voicechat_duplex.yaml")
    assert duplex.get("pipeline") == "nemotron_voicechat"
    assert duplex.get("session_mode") == "duplex"

    streaming = resolve_deploy_yaml(_DEPLOY_DIR / "nemotron_labs_voicechat_streaming.yaml")
    assert streaming.get("pipeline") == "nemotron_voicechat"
    assert streaming.get("session_mode", NEMOTRON_VOICECHAT_PIPELINE.default_session_mode) == "turn"


def test_load_duplex_plugin_and_sampling_contract() -> None:
    plugin = load_duplex_plugin(_PLUGIN_PATH, lambda *_: None)
    assert plugin.plugin_id == "nemotron_voicechat"
    assert isinstance(plugin.data_plane, DuplexDataPlane)

    defaults = tuple(
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True) if stage_id == 0 else SamplingParams()
        for stage_id in range(3)
    )
    validate_duplex_plugin_sampling(plugin, sampling_defaults=defaults)

    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    assert len(configured) == 3
    stage0 = configured[0]
    assert stage0.temperature == 0.0 and stage0.max_tokens == 1
    assert stage0.top_p == 1.0 and stage0.top_k == 0
    assert stage0.ignore_eos is True
    assert all(params.output_kind == RequestOutputKind.DELTA for params in configured)


def test_capabilities_match_the_frame_locked_native_duplex_shape() -> None:
    capabilities = _plugin().capabilities(max_sessions=1).as_dict()
    assert capabilities["implementation_level"] == "model_native_duplex"
    assert capabilities["input_modes"] == ["append_audio_chunk"]
    assert capabilities["chunk_period_ms"] == 80
    assert capabilities["supports_core_resumable_request"] is True
    assert capabilities["supports_core_kv_lease"] is False
    assert capabilities["supports_model_native_turn_policy"] is True
    assert capabilities["supports_external_turn_signal"] is False
    assert capabilities["supports_client_commit"] is True
    assert capabilities["supports_barge_in"] is False
    assert capabilities["supports_multi_session"] is False
    assert capabilities["supports_session_resume"] is True
    assert capabilities["signal_sources"] == ["model_native", "client_event"]


def test_first_append_prefills_prompt_then_each_append_consumes_one_frame() -> None:
    plugin = _plugin()

    first = _plan(plugin, input_seq=1)
    later = _plan(plugin, input_seq=2)

    assert first.prompt["prompt_token_ids"] == [0, 42, 1, 12]
    assert later.prompt["prompt_token_ids"] == [12]
    first_duplex = first.prompt["model_intermediate_buffer"]["duplex"]
    later_duplex = later.prompt["model_intermediate_buffer"]["duplex"]
    assert first_duplex["source_input_seq"] == 1
    assert later_duplex["source_input_seq"] == 2
    # The old incarnation field is gone; the fence carries (session, epoch, turn).
    assert "incarnation" not in first_duplex
    assert first_duplex["epoch"] == 3
    assert first_duplex["mode"] == "append_audio_chunk"


def test_append_rejects_stage0_context_overflow() -> None:
    with pytest.raises(ValueError, match="max_model_len"):
        _plan(_plugin(), input_seq=8190)


def test_append_rejects_non_frame_payload() -> None:
    payload = _frame()
    payload["audio"] = base64.b64encode(np.zeros(640, dtype=np.float32).tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="exactly 1280 samples"):
        _plugin().plan_append(
            request_id="req",
            fence=DuplexFence("sid", epoch=0),
            session_config={},
            runtime_config={"nvc_prompt_token_ids": [0], "nvc_text_pad_id": 12, "nvc_max_model_len": 8192},
            seq=1,
            turn_seq=1,
            payload=payload,
            final=False,
            sampling_params=SamplingParams(),
        )


def test_decide_output_marks_stage0_frames_as_direct_text_responses() -> None:
    plugin = _plugin()
    decision = plugin.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(4, 5),
        segment_output_metadata={"k": 1},
        output=SimpleNamespace(),
    )
    assert decision is not None
    assert decision.metadata["nvc_text_token_ids"] == [4, 5]
    assert decision.metadata["duplex_direct_response"] is True
    assert decision.final_output_type == "text"

    assert (
        plugin.decide_output(
            stage_id=1,
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(4,),
            segment_output_metadata={},
            output=SimpleNamespace(),
        )
        is None
    )


def test_prepare_runtime_config_requires_native_full_duplex() -> None:
    plugin = _plugin()
    with pytest.raises(NemotronVoiceChatClientRuntimeConfigError, match="auto_response"):
        import asyncio

        asyncio.run(
            plugin.prepare_runtime_config(
                DuplexSessionConfig(instructions="hi", extra_body={}),
                model_config=_model_config(),
            )
        )


def test_prepare_runtime_config_rejects_private_keys() -> None:
    plugin = _plugin()
    with pytest.raises(NemotronVoiceChatClientRuntimeConfigError, match="server-owned"):
        plugin.validate_client_extra_body({"nvc_text_pad_id": 0})


async def _prepare(plugin: NemotronVoiceChatDuplexPlugin, **extra_body) -> dict[str, object]:
    return await plugin.prepare_runtime_config(
        DuplexSessionConfig(instructions="hi", extra_body={"auto_response": True, **extra_body}),
        model_config=_model_config(),
    )


def test_prepare_runtime_config_resolves_prompt_and_configures_data_plane(patched_tokenizer) -> None:
    plugin = _plugin()

    import asyncio

    runtime = asyncio.run(_prepare(plugin))

    assert runtime["nvc_prompt_token_ids"] == [0, 42, 1]
    assert runtime["nvc_max_model_len"] == 8192
    assert runtime["instructions"] == "hi"
    assert runtime["nvc_tools_signature"] == "[]"
    assert plugin.data_plane._special_ids is not None
    assert plugin._tokenizers["nvidia/NVIDIA-Nemotron-Nano-9B-v2"][1] is not None


def test_runtime_config_for_update_rejects_instructions_and_tools_changes(patched_tokenizer) -> None:
    plugin = _plugin()
    import asyncio

    current = asyncio.run(_prepare(plugin))

    updated = plugin.runtime_config_for_update(
        DuplexSessionConfig(instructions="hi", extra_body={"auto_response": True}),
        current,
    )
    assert updated["nvc_prompt_token_ids"] == current["nvc_prompt_token_ids"]

    with pytest.raises(NemotronVoiceChatClientRuntimeConfigError, match="instructions cannot change"):
        plugin.runtime_config_for_update(
            DuplexSessionConfig(instructions="other", extra_body={"auto_response": True}),
            current,
        )
    with pytest.raises(NemotronVoiceChatClientRuntimeConfigError, match="tools cannot change"):
        plugin.runtime_config_for_update(
            DuplexSessionConfig(instructions="hi", extra_body={"auto_response": True}),
            {**current, "nvc_tools_signature": '[{"name":"other"}]'},
        )


def test_function_output_becomes_versioned_nvidia_channel_tokens(patched_tokenizer) -> None:
    plugin = _plugin()
    encoded: list[str] = []

    def _encode(text: str, **_kwargs: object) -> list[int]:
        encoded.append(text)
        return [31, 32, 33]

    plugin._tokenizers["test-tokenizer"] = (dict(_FAKE_SPECIAL_IDS), SimpleNamespace(encode=_encode))
    config = DuplexSessionConfig(instructions="hi", extra_body={"auto_response": True})
    current: dict[str, object] = {**_FAKE_SPECIAL_IDS, "nvc_max_model_len": 8192}

    first = plugin.runtime_config_for_function_output(
        config, current, {"type": "function_call_output", "call_id": "call-1", "output": '{"result":20}'}
    )
    second = plugin.runtime_config_for_function_output(
        config, first, {"type": "function_call_output", "call_id": "call-2", "output": "plain text"}
    )

    assert _render_tool_response('{"result":20}') == '<TOOL_RESPONSE>[{"result":20}]</TOOL_RESPONSE>'
    assert encoded == [
        '<TOOL_RESPONSE>[{"result":20}]</TOOL_RESPONSE>',
        '<TOOL_RESPONSE>["plain text"]</TOOL_RESPONSE>',
    ]
    assert first["nvc_function_response_generation"] == 1
    assert second["nvc_function_response_generation"] == 2
    assert second["nvc_function_response_token_ids"] == [31, 32, 33]
    assert second["nvc_function_response_call_id"] == "call-2"
    assert second["nvc_function_response_batches"] == [
        {"generation": 1, "call_id": "call-1", "token_ids": [31, 32, 33]},
        {"generation": 2, "call_id": "call-2", "token_ids": [31, 32, 33]},
    ]
    third = plugin.runtime_config_for_function_output(
        config, second, {"type": "function_call_output", "call_id": "call-3", "output": '{"result": 1}'}
    )
    assert third["nvc_function_response_generation"] == 3
    assert len(third["nvc_function_response_batches"]) == 3

    session = {"function_response_generation": 0, "forced_function_tokens": []}
    NemotronVoiceChatThinkerForConditionalGeneration._sync_forced_function_response(session, second)

    assert session["function_response_generation"] == 2
    assert session["forced_function_token"] == 31
    assert session["forced_function_tokens"] == [31, 32, 33, 31, 32, 33]


def test_function_response_backlog_is_bounded(patched_tokenizer) -> None:
    """The per-append runtime snapshot embeds the backlog; it must not grow without bound."""
    plugin = _plugin()
    plugin._tokenizers["test-tokenizer"] = (
        dict(_FAKE_SPECIAL_IDS),
        SimpleNamespace(encode=lambda text, **kwargs: [31, 32, 33]),
    )
    config = DuplexSessionConfig(instructions="hi", extra_body={"auto_response": True})
    current: dict = {"nvc_tokenizer_ref": "test-tokenizer"}

    for index in range(8):
        current = plugin.runtime_config_for_function_output(
            config, current, {"type": "function_call_output", "call_id": f"call-{index}", "output": "1"}
        )
    assert current["nvc_function_response_generation"] == 8

    with pytest.raises(NemotronVoiceChatClientRuntimeConfigError, match="faster than the frame-locked"):
        plugin.runtime_config_for_function_output(
            config, current, {"type": "function_call_output", "call_id": "call-9", "output": "1"}
        )

    drained = plugin.runtime_config_after_model_output(
        current,
        {"nvc_function_response_consumed_generation": 8},
    )
    assert drained is not None
    assert drained["nvc_function_response_batches"] == []
    ninth = plugin.runtime_config_for_function_output(
        config,
        drained,
        {"type": "function_call_output", "call_id": "call-9", "output": "1"},
    )
    assert ninth["nvc_function_response_generation"] == 9
