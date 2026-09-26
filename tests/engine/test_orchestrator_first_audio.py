# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.engine import FinishReason

from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState, _upstream_first_audio_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _orchestrator():
    obj = Orchestrator.__new__(Orchestrator)
    obj.stage_pools = [SimpleNamespace(final_output=False), SimpleNamespace(final_output=True)]
    obj.request_states = {"r": OrchestratorRequestState(request_id="r", final_stage_id=1)}
    obj.output_async_queue = asyncio.Queue()
    return obj


def _raw(*, first=False, required=False, terminal=False):
    return OmniEngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(
                request_id="r",
                new_token_ids=[],
                finish_reason=FinishReason.STOP if terminal else None,
                multimodal_output={
                    "model_outputs": torch.ones(4),
                    "sr": torch.tensor(24000),
                    "_omni_first_audio": torch.tensor(first),
                    "_omni_first_audio_required": torch.tensor(required),
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_codec_terminal_that_overtakes_first_audio_is_released_in_order(mocker):
    obj = _orchestrator()
    later = _raw(required=True)
    terminal = _raw(terminal=True)
    expected = [later.outputs[0], terminal.outputs[0]]
    await obj._route_upstream_first_audio(1, 0, later)
    await obj._route_upstream_first_audio(1, 0, terminal)
    assert not later.outputs and not terminal.outputs
    assert obj.output_async_queue.empty()
    released = []

    async def process(stage, replica, raw, terminals):
        assert obj.output_async_queue.qsize() == 1  # first PCM is already queued
        released.extend(raw.outputs)
        return []

    obj._process_llm_stage_outputs = process
    obj._handle_processed_outputs = mocker.AsyncMock()
    obj._finish_raw_terminal_requests = mocker.AsyncMock()
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    assert released == expected
    assert obj.request_states["r"].pending_first_audio_outputs == []
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    assert obj.output_async_queue.qsize() == 1


@pytest.mark.asyncio
async def test_unmarked_upstream_audio_and_regular_codec_outputs_are_untouched():
    obj = _orchestrator()
    for stage in (0, 1):
        raw = _raw()
        await obj._route_upstream_first_audio(stage, 0, raw)
        assert len(raw.outputs) == 1
        assert torch.equal(raw.outputs[0].multimodal_output["model_outputs"], torch.ones(4))
    assert obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_receive_late_first_audio():
    obj = _orchestrator()
    obj.request_states.clear()
    raw = _raw(first=True)
    await obj._route_upstream_first_audio(0, 0, raw)
    assert not raw.outputs and obj.output_async_queue.empty()


def test_first_audio_supports_both_speech_and_chat_consumers():
    audio = torch.arange(8, dtype=torch.float32)
    result = _upstream_first_audio_output("r", audio, torch.tensor(24000))
    assert not result.finished
    assert torch.equal(result.outputs[0].multimodal_output["audio"], audio)
    assert torch.equal(result.multimodal_output["audio"], audio)


@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [None, [], torch.empty(0)])
@pytest.mark.parametrize("codec_overtook", [False, True])
async def test_invalid_first_audio_reports_error_and_releases_request(mocker, audio, codec_overtook):
    obj = _orchestrator()
    obj._cfg_tracker = CfgCompanionTracker()
    obj._pd_kv_params = {}
    obj._running_counter = None
    obj._abort_request_ids = mocker.AsyncMock(return_value=[])
    obj._release_request_bindings = mocker.Mock()
    obj.request_states["healthy"] = OrchestratorRequestState(request_id="healthy", final_stage_id=1)
    if codec_overtook:
        await obj._route_upstream_first_audio(1, 0, _raw(required=True, terminal=True))
        assert obj.request_states["r"].pending_first_audio_outputs
    first = _raw(first=True)
    first.outputs[0].multimodal_output["model_outputs"] = audio
    await obj._route_upstream_first_audio(0, 0, first)

    error = obj.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage)
    assert error.request_id == "r" and "first-audio" in error.error
    assert not first.outputs and set(obj.request_states) == {"healthy"}
    obj._abort_request_ids.assert_awaited_once_with(["r"])
    obj._release_request_bindings.assert_called_once_with(["r"])
    # A late duplicate cannot restart the request or emit a second error.
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    assert obj.output_async_queue.empty()
