# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Nemotron VoiceChat session-runner scenarios on the real duplex plugin.

Same harness contract as ``test_session_runner.py`` (a recording stage port +
the manager dispatch loop), but with the real ``NemotronVoiceChatDuplexPlugin``
so the frame-locked 80 ms cadence, prompt prefill and silence continuation are
exercised end to end without a GPU.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import pytest

from tests.engine.duplex.test_session_runner import (
    Harness,
    RecordingStagePort,
    _fake_encode_audio,
    append_audio,
    close_harness,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexOutputContext,
    DuplexRequestIdentity,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.input import (
    NEMOTRON_VOICECHAT_FRAME_SAMPLES,
)
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
    NemotronVoiceChatDuplexPlugin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "duplex-nemotron-test"
PAD_TOKEN_ID = 12

_FAKE_SPECIAL_IDS = {
    "nvc_text_bos_id": 0,
    "nvc_text_eos_id": 1,
    "nvc_text_pad_id": PAD_TOKEN_ID,
    "nvc_function_sotc_id": 20,
    "nvc_function_eotc_id": 21,
    "nvc_function_eotr_id": 22,
    "nvc_tokenizer_ref": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
}


def _fake_decode(token_ids: object, **_kwargs: object) -> str:
    ids = list(token_ids) if isinstance(token_ids, (list, tuple)) else []
    if ids == [7]:
        return '[{"name":"lookup","arguments":{"x":1}}]'
    return "hi" if ids else ""


_FAKE_TOKENIZER = SimpleNamespace(
    encode=lambda text, **_kwargs: [42],
    decode=_fake_decode,
)


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(stt_cfg={}),
        max_model_len=8192,
    )


def _plugin() -> NemotronVoiceChatDuplexPlugin:
    plugin = NemotronVoiceChatDuplexPlugin(_fake_encode_audio)
    # Pre-seed the per-checkpoint tokenizer cache so opening a session never
    # downloads a tokenizer in tests.
    plugin._tokenizers[_FAKE_SPECIAL_IDS["nvc_tokenizer_ref"]] = (dict(_FAKE_SPECIAL_IDS), _FAKE_TOKENIZER)
    return plugin


async def open_nemotron_harness(
    *,
    extra_body: dict[str, object] | None = None,
    runtime_config: DuplexSessionRuntimeConfig | None = None,
) -> Harness:
    plugin = _plugin()
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue = asyncio.Queue()
    results: asyncio.Queue = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=runtime_config or DuplexSessionRuntimeConfig(),
        model_config=_model_config(),
    )
    config = DuplexSessionConfig(
        model="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        modalities=["audio", "text"],
        instructions="You are NVIDIA Voice Chat.",
        extra_body={"auto_response": True, **(extra_body or {})},
    )
    await manager.handle(OpenDuplexSessionMessage(control_id="c-open", session_id=SESSION_ID, session_config=config))
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(manager=manager, port=port, output=output, results=results, runner=manager.runners[SESSION_ID])
    await harness.settle()
    return harness


def append_frame(*, value: float = 0.05, is_speech: bool | None = True) -> commands.AppendAudio:
    return append_audio(samples=NEMOTRON_VOICECHAT_FRAME_SAMPLES, value=value, is_speech=is_speech)


def submit(h: Harness, command: commands.DuplexCommand) -> None:
    h.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))


async def run(h: Harness, command: commands.DuplexCommand):
    submit(h, command)
    return await h.settle()


def stage0_request_id(h: Harness, epoch: int | None = None) -> str:
    fence = h.session.fence if epoch is None else DuplexFence(SESSION_ID, epoch=epoch)
    return duplex_resource_request_id(fence, "stage0")


def deliver(
    h: Harness,
    output: Any,
    *,
    stage_id: int = 0,
    segment_finished: bool = False,
    segment_token_ids: tuple[int, ...] = (),
    segment_output_metadata: dict[str, object] | None = None,
) -> bool:
    context = DuplexOutputContext(
        identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=h.session.fence),
        final_stage_id=h.port.stage_count - 1,
        segment_finished=segment_finished,
        segment_token_ids=tuple(segment_token_ids),
        segment_output_metadata=dict(segment_output_metadata or {}),
    )
    return h.runner.on_stage_output(stage_id, output, None, request_id=output.request_id, context=context)


async def deliver_and_settle(h: Harness, output: Any, **kwargs: Any):
    deliver(h, output, **kwargs)
    return await h.settle()


def thinker_output(request_id: str, token_id: int) -> SimpleNamespace:
    """One Stage-0 segment output carrying one sampled text-channel token."""
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text="", token_ids=[token_id], multimodal_output={})],
        multimodal_output={},
    )


@pytest.mark.asyncio
async def test_open_requires_native_full_duplex() -> None:
    plugin = _plugin()
    results: asyncio.Queue = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=RecordingStagePort(stage_count=3),
        output_sink=asyncio.Queue(),
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=_model_config(),
    )
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="c-open",
            session_id=SESSION_ID,
            session_config=DuplexSessionConfig(model="m", instructions="hi", extra_body={}),
        )
    )
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert not result.ok
    assert "auto_response" in (result.error_message or "")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_open_announces_nemotron_capabilities_and_reserves_stage0_request() -> None:
    h = await open_nemotron_harness()
    try:
        assert types(h.events)[0] == "session.created"
        created = h.events[0]
        capabilities = created.session["capabilities"]
        assert capabilities["implementation_level"] == "model_native_duplex"
        assert capabilities["chunk_period_ms"] == 80
        assert capabilities["supports_core_resumable_request"] is True
        assert capabilities["supports_core_kv_lease"] is False
        assert capabilities["supports_barge_in"] is False
        assert [context.request_id for context in h.port.ensured] == [stage0_request_id(h, epoch=0)]
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_each_frame_appends_one_position_to_the_resumable_stage0_request() -> None:
    h = await open_nemotron_harness()
    try:
        events = await run(h, append_frame())
        assert "error" not in types(events)
        assert len(h.port.submissions) == 1
        first = h.port.submissions[0]
        assert first.context.request_id == stage0_request_id(h, epoch=0)
        assert first.already_submitted is False
        # The first frame carries the fused text prompt plus one PAD anchor.
        assert first.prompt["prompt_token_ids"] == [0, 42, 1, PAD_TOKEN_ID]
        duplex = first.prompt["model_intermediate_buffer"]["duplex"]
        assert duplex["session_id"] == SESSION_ID
        assert duplex["source_input_seq"] == 1
        assert duplex["seq"] == 1
        assert "incarnation" not in duplex

        await run(h, append_frame())
        second = h.port.submissions[1]
        assert second.already_submitted is True
        assert second.prompt["prompt_token_ids"] == [PAD_TOKEN_ID]
        assert second.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 2
        assert h.session.input_seq == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_partial_packets_are_buffered_until_one_full_80ms_frame() -> None:
    h = await open_nemotron_harness()
    try:
        await run(h, append_audio(samples=1000))
        assert h.port.submissions == []
        assert h.runner.model_state.audio_buffer.has_pending()

        await run(h, append_audio(samples=280))
        assert len(h.port.submissions) == 1
        assert not h.runner.model_state.audio_buffer.has_pending()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_append_of_more_than_one_frame_is_rejected() -> None:
    h = await open_nemotron_harness()
    try:
        events = await run(h, append_audio(samples=NEMOTRON_VOICECHAT_FRAME_SAMPLES + 1))
        assert types(events) == ["input_audio_buffer.speech_started", "error"]
        assert events[-1].code == "bad_event"
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_during_an_open_response_schedules_an_80ms_silence_unit() -> None:
    h = await open_nemotron_harness()
    try:
        await run(h, append_frame())
        request_id = stage0_request_id(h)
        # The model opens a response by emitting a real text token.
        events = await deliver_and_settle(
            h,
            thinker_output(request_id, 5),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[5],
        )
        assert "response.speak" in types(events) or "response.created" in types(events)
        assert h.session.active_response_id is not None
        submissions_before = len(h.port.submissions)

        # ...then keeps listening: one PAD token must tick out another unit.
        events = await deliver_and_settle(
            h,
            thinker_output(request_id, PAD_TOKEN_ID),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[PAD_TOKEN_ID],
        )

        assert not [event for event in events if event.type == "error"], types(events)
        assert len(h.port.submissions) == submissions_before + 1, "no silence continuation was scheduled"
        silence = h.port.submissions[-1]
        payload = silence.prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert set(base64.b64decode(payload["audio"])) == {0}
        assert len(base64.b64decode(payload["audio"])) == NEMOTRON_VOICECHAT_FRAME_SAMPLES * 4
        assert payload["format"] == "pcm_f32le"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_function_output_rides_the_runtime_config_into_the_worker() -> None:
    """Queue a tool result and retire it when the worker reports consumption."""
    h = await open_nemotron_harness()
    try:
        await run(h, append_frame())
        request_id = stage0_request_id(h)
        # The model opens its function channel: SOTC, argument tokens, EOTC.
        for token_id in (20, 7, 21):
            events = await deliver_and_settle(
                h,
                thinker_output(request_id, token_id),
                stage_id=0,
                segment_finished=True,
                segment_token_ids=[token_id],
                segment_output_metadata={"nvc_function_token": [token_id]},
            )
        done = [event for event in h.events if event.type == "response.function_call_arguments.done"]
        assert done, types(h.events)
        call_id = done[-1].call_id
        assert isinstance(call_id, str) and call_id

        from vllm_omni.protocol.realtime.items import normalize_conversation_item

        events = await run(
            h,
            commands.CreateItem(
                item=normalize_conversation_item(
                    {"type": "function_call_output", "call_id": call_id, "output": '{"result": 20}'}
                )
            ),
        )
        assert not [event for event in events if event.type == "error"], types(events)
        runtime = h.session.runtime_config
        assert runtime["nvc_function_response_generation"] == 1
        assert runtime["nvc_function_response_call_id"] == call_id
        assert runtime["nvc_function_response_token_ids"] == [42]
        assert len(runtime["nvc_function_response_batches"]) == 1

        events = await deliver_and_settle(
            h,
            thinker_output(request_id, PAD_TOKEN_ID),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[PAD_TOKEN_ID],
            segment_output_metadata={"nvc_function_response_consumed_generation": 1},
        )
        assert not [event for event in events if event.type == "error"], types(events)
        assert h.session.runtime_config["nvc_function_response_batches"] == []
        assert h.session.runtime_config["nvc_function_response_generation"] == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_releases_the_resumable_stage0_request() -> None:
    h = await open_nemotron_harness()
    try:
        await run(h, append_frame())
        request_id = stage0_request_id(h)
        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close", session_id=SESSION_ID, reason="client_close")
        )
        result = await asyncio.wait_for(h.results.get(), timeout=2.0)
        events = await h.settle()

        assert result.ok and result.operation == "close"
        assert types(events) == ["session.closed"]
        assert h.port.cleanups == [([request_id], True)]
    finally:
        await close_harness(h)
