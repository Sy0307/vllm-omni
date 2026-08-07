# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    DuplexInputAppendCommand,
    DuplexInputClearCommand,
    DuplexInputCloseCommand,
    DuplexInputCommitCommand,
    DuplexInputCompletionMode,
    DuplexInputEffect,
    DuplexInputFlushCommand,
    DuplexInputSnapshot,
    ordered_input_emissions,
    validate_serving_runtime_adapter,
)
from vllm_omni.experimental.fullduplex.openai.runtime_bridge import (
    NativeRuntimeBridgeMixin,
)
from vllm_omni.experimental.fullduplex.openai.serving import (
    OmniDuplexSessionHandler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_server_runtime_config_owns_cold_open_timeout() -> None:
    session = SimpleNamespace(
        runtime_config={"runtime_control_timeout_s": 240.0},
        config=SimpleNamespace(
            extra_body={"duplex_control_timeout_s": 1.0},
        ),
        capabilities=SimpleNamespace(
            implementation_level="model_native_duplex",
        ),
    )

    assert NativeRuntimeBridgeMixin._runtime_control_timeout_s(session) == 240.0


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_omni.experimental.fullduplex.openai.session_runner",
        "vllm_omni.experimental.fullduplex.openai.runtime_bridge",
        "vllm_omni.experimental.fullduplex.openai.serving",
    ],
)
def test_generic_openai_runtime_import_does_not_load_minicpmo45(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = f"""
import sys

import {module_name}

loaded = sorted(
    name
    for name in sys.modules
    if name == "vllm_omni.experimental.fullduplex.minicpmo45"
    or name.startswith("vllm_omni.experimental.fullduplex.minicpmo45.")
)
if loaded:
    raise SystemExit("generic OpenAI runtime loaded MiniCPM-o: " + ", ".join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_generic_session_update_does_not_own_minicpmo45_ref_audio_policy() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    source = (repo_root / "vllm_omni/experimental/fullduplex/openai/serving.py").read_text(encoding="utf-8")

    assert "ref_audio_required" not in source
    assert "MiniCPM-o native duplex audio output requires ref_audio" not in source


def test_generic_handler_requires_explicit_serving_runtime_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
from types import SimpleNamespace

from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler

chat_service = SimpleNamespace(engine_client=SimpleNamespace())
try:
    OmniDuplexSessionHandler(chat_service=chat_service)
except ValueError as exc:
    if "serving runtime adapter" not in str(exc).lower():
        raise
else:
    raise SystemExit("generic handler silently selected a model serving adapter")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _valid_runtime_adapter() -> SimpleNamespace:
    input_controller = SimpleNamespace(
        create_state=lambda: object(),
        snapshot=lambda state: DuplexInputSnapshot(),
        append=lambda state, command: DuplexInputEffect(),
        commit=lambda state, command: DuplexInputEffect(),
        clear=lambda state, command: DuplexInputEffect(),
        flush=lambda state, command: DuplexInputEffect(),
        close=lambda state, command: DuplexInputEffect(),
    )
    data_plane = SimpleNamespace(
        begin_request=lambda request_id: None,
        is_terminal=lambda request_id: False,
        mark_terminal=lambda request_id: None,
        close_stream=lambda request_id: None,
        close_session=lambda session_id, **kwargs: None,
        project=lambda result, **kwargs: (),
    )
    return SimpleNamespace(
        adapter_id="test-adapter",
        session_states={},
        input_controller=input_controller,
        data_plane=data_plane,
        clean_response_done_prefix="clean",
        interrupted_tts_prefix="interrupted",
        private_runtime_config_keys=frozenset({"private_key"}),
        create_session_state=lambda: object(),
        session_state=lambda session_id: object(),
        remove_session_state=lambda session_id: None,
        is_enabled=lambda config: True,
        capabilities=lambda **kwargs: object(),
        validate_client_extra_body=lambda extra_body: None,
        prepare_runtime_config=lambda config, **kwargs: {},
        runtime_config_for_update=lambda config, current: {},
        data_plane_context=lambda **kwargs: object(),
    )


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("session_states", None),
        ("clean_response_done_prefix", None),
        ("interrupted_tts_prefix", None),
        ("private_runtime_config_keys", {"private_key"}),
    ],
)
def test_runtime_adapter_validator_rejects_invalid_protocol_attributes(
    attribute: str,
    invalid_value: object,
) -> None:
    adapter = _valid_runtime_adapter()
    setattr(adapter, attribute, invalid_value)

    with pytest.raises(TypeError, match=attribute):
        validate_serving_runtime_adapter(adapter)


def test_runtime_adapter_validator_rejects_incomplete_data_plane() -> None:
    adapter = _valid_runtime_adapter()
    adapter.data_plane.close_session = None

    with pytest.raises(TypeError, match="data_plane.*close_session"):
        validate_serving_runtime_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_mode", "collect_outputs"),
    [
        (DuplexInputCompletionMode.APPEND_ACCEPTED, False),
        (DuplexInputCompletionMode.OUTPUT_PROJECTED, True),
    ],
)
async def test_runtime_input_completion_mode_controls_first_output_ack(
    completion_mode: DuplexInputCompletionMode,
    collect_outputs: bool,
) -> None:
    append_kwargs: list[dict[str, object]] = []

    async def append_duplex_input_async(session_id: str, **kwargs):
        assert session_id == "session"
        append_kwargs.append(kwargs)
        return {}

    adapter = _valid_runtime_adapter()
    adapter.input_completion_mode = completion_mode
    handler = OmniDuplexSessionHandler(
        chat_service=SimpleNamespace(
            engine_client=SimpleNamespace(
                append_duplex_input_async=append_duplex_input_async,
            )
        ),
        serving_runtime_adapter=adapter,
    )
    session = DuplexSession(
        "session",
        DuplexSessionConfig(),
        capabilities=DuplexCapabilities(supports_input_append=True),
    )

    async def send_json(payload: dict[str, object]) -> None:
        raise AssertionError(f"unexpected wire event: {payload}")

    assert await handler._append_runtime_input(
        session,
        {"frame": 1},
        final=False,
        send_json=send_json,
        mode="append_audio_chunk",
    ) == (True, False)
    assert append_kwargs[0]["collect_outputs"] is collect_outputs


def test_runtime_adapter_requires_a_model_owned_input_controller() -> None:
    adapter = _valid_runtime_adapter()
    adapter.input_controller = None

    with pytest.raises(TypeError, match="input_controller"):
        validate_serving_runtime_adapter(adapter)


def test_second_adapter_input_controller_can_keep_opaque_model_state() -> None:
    class OpaqueState:
        def __init__(self) -> None:
            self.commands: list[str] = []

    class SecondModelController:
        def create_state(self) -> object:
            return OpaqueState()

        def snapshot(self, state: object) -> DuplexInputSnapshot:
            assert isinstance(state, OpaqueState)
            return DuplexInputSnapshot(has_pending=bool(state.commands))

        def append(
            self,
            state: object,
            command: DuplexInputAppendCommand,
        ) -> DuplexInputEffect:
            assert isinstance(state, OpaqueState)
            state.commands.append("append")
            return DuplexInputEffect(append_payloads=(command.payload,))

        def commit(
            self,
            state: object,
            command: DuplexInputCommitCommand,
        ) -> DuplexInputEffect:
            assert isinstance(state, OpaqueState)
            state.commands.append("commit")
            return DuplexInputEffect()

        def clear(
            self,
            state: object,
            command: DuplexInputClearCommand,
        ) -> DuplexInputEffect:
            assert isinstance(state, OpaqueState)
            state.commands.append("clear")
            return DuplexInputEffect()

        def close(
            self,
            state: object,
            command: DuplexInputCloseCommand,
        ) -> DuplexInputEffect:
            assert isinstance(state, OpaqueState)
            state.commands.append("close")
            return DuplexInputEffect()

        def flush(
            self,
            state: object,
            command: DuplexInputFlushCommand,
        ) -> DuplexInputEffect:
            assert isinstance(state, OpaqueState)
            state.commands.append("flush")
            return DuplexInputEffect()

    controller = SecondModelController()
    state = controller.create_state()

    append = controller.append(
        state,
        DuplexInputAppendCommand(
            payload={"audio": "pcm"},
            operation_id="append-1",
            chunk_period_ms=80,
            allow_emit=True,
        ),
    )
    controller.commit(
        state,
        DuplexInputCommitCommand(operation_id="commit-1", chunk_period_ms=80),
    )
    controller.clear(state, DuplexInputClearCommand(reason="cancel"))
    controller.close(state, DuplexInputCloseCommand(abort=False))

    assert append.append_payloads == ({"audio": "pcm"},)
    assert controller.snapshot(state).has_pending is True
    assert not hasattr(state, "audio_buffer")
    assert not hasattr(state, "pending_input_continuation")


def test_input_effect_supports_multiple_ordered_model_emissions() -> None:
    first = SimpleNamespace(operation_id="frame-1")
    second = SimpleNamespace(operation_id="frame-2")
    effect = DuplexInputEffect(
        append_payloads=({"frame": 1}, {"frame": 2}),
        reservations=(first, second),
    )

    assert ordered_input_emissions(effect) == (
        ({"frame": 1}, first),
        ({"frame": 2}, second),
    )


def test_generic_runtime_does_not_expose_a_pcm_buffer_contract() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    generic_files = (
        repo_root / "vllm_omni/experimental/fullduplex/openai/runtime_adapter.py",
        repo_root / "vllm_omni/experimental/fullduplex/openai/session_runner.py",
        repo_root / "vllm_omni/experimental/fullduplex/openai/serving.py",
    )

    for path in generic_files:
        source = path.read_text(encoding="utf-8")
        assert "PcmAppendBuffer" not in source
        assert ".audio_buffer" not in source


def test_realtime_projects_incremental_model_transcript_without_response_id() -> None:
    protocol = NativeRealtimeSessionProtocol({})

    delta = protocol.encode_outbound_event(
        {
            "type": "input.transcript.delta",
            "session_id": "session",
            "epoch": 4,
            "source_input_seq": 8,
            "transcript_seq": 0,
            "delta": "hello ",
            "final": False,
        }
    )
    completed = protocol.encode_outbound_event(
        {
            "type": "input.transcript.delta",
            "session_id": "session",
            "epoch": 4,
            "source_input_seq": 8,
            "transcript_seq": 1,
            "delta": "world",
            "final": True,
        }
    )

    assert [event["type"] for event in delta] == [
        "conversation.item.added",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.delta",
    ]
    assert delta[-1]["delta"] == "hello "
    assert "response_id" not in delta[-1]
    assert [event["type"] for event in completed] == [
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.done",
    ]
    assert completed[-2]["transcript"] == "hello world"


def test_realtime_allocates_wire_ids_for_model_function_call_channel() -> None:
    protocol = NativeRealtimeSessionProtocol({})

    started = protocol.encode_outbound_event(
        {
            "type": "function_call.start",
            "session_id": "session",
            "epoch": 2,
            "source_input_seq": 5,
            "call_id": "weather-call",
            "name": "get_weather",
        }
    )
    response_id = started[0]["response"]["id"]
    item_id = started[-1]["item"]["id"]
    delta = protocol.encode_outbound_event(
        {
            "type": "function_call.delta",
            "session_id": "session",
            "epoch": 2,
            "call_id": "weather-call",
            "function_seq": 0,
            "delta": '{"city":"San',
        }
    )
    completed = protocol.encode_outbound_event(
        {
            "type": "function_call.done",
            "session_id": "session",
            "epoch": 2,
            "call_id": "weather-call",
            "function_seq": 1,
            "reason": "completed",
        }
    )

    assert [event["type"] for event in started] == [
        "response.created",
        "conversation.item.added",
        "conversation.item.created",
        "response.output_item.added",
    ]
    assert delta[0] == {
        "type": "response.function_call_arguments.delta",
        "event_id": delta[0]["event_id"],
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "call_id": "weather-call",
        "delta": '{"city":"San',
    }
    assert [event["type"] for event in completed] == [
        "response.function_call_arguments.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.done",
    ]
    assert completed[0]["arguments"] == '{"city":"San'
