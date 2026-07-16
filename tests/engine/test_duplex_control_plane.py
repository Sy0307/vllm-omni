# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex_control_plane import (
    DuplexControlPlane,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
)
from vllm_omni.engine.duplex_types import DuplexFence
from vllm_omni.engine.messages import (
    AppendDuplexInputMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.resumable import ResumableSegmentPolicy


class _Extension:
    def configure_sampling_params(self, *, runtime_config, defaults):
        del runtime_config
        return tuple(f"configured-{stage_id}" for stage_id, _ in enumerate(defaults))

    def plan_append(
        self,
        *,
        request_id,
        fence,
        session_config,
        runtime_config,
        seq,
        turn_seq,
        mode,
        payload,
        final,
        sampling_params,
    ):
        del request_id, fence, session_config, runtime_config, seq, turn_seq, mode, payload, final
        assert sampling_params == "configured-0"
        return DuplexAppendPlan(
            prompt={"prompt_token_ids": [1, 2, 3]},
            segment_policy=ResumableSegmentPolicy(
                stop_token_ids=(99,),
                max_segment_tokens=4,
            ),
        )

    def segment_policy(self, sampling_params):
        del sampling_params
        return ResumableSegmentPolicy(
            stop_token_ids=(99,),
            max_segment_tokens=4,
        )

    def decide_output(self, **kwargs):
        del kwargs
        return None


class _TypedStagePort:
    stage_count = 2

    def __init__(self) -> None:
        self.ensure_calls: list[DuplexStageRequestContext] = []
        self.submit_calls: list[DuplexStageSubmission] = []
        self.cleanup_calls: list[tuple[list[str], bool]] = []

    def sampling_defaults(self) -> tuple[object, ...]:
        return ("default-0", "default-1")

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensure_calls.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        self.submit_calls.append(submission)
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=3,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanup_calls.append((request_ids, abort))


@pytest.mark.asyncio
async def test_control_plane_uses_frozen_typed_stage_context_without_request_state() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
    )
    fence = DuplexFence("typed-port")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={
                "input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value],
            },
            session_config={"voice": "test"},
            runtime_config={"runtime": "test"},
        )
    )
    assert (await result_sink.get()).ok is True

    context = stage_port.ensure_calls[-1]
    assert context.session_id == fence.session_id
    assert context.fence == fence
    assert context.stage_id == 0
    assert context.final_stage_id == 1
    assert context.sampling_params == ("configured-0", "configured-1")
    assert context.session_config == {"voice": "test"}
    assert context.runtime_config == {"runtime": "test"}
    with pytest.raises(FrozenInstanceError):
        context.stage_id = 1  # type: ignore[misc]

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
            final=True,
        )
    )
    append_result = await result_sink.get()
    assert append_result.ok is True
    assert append_result.stage_results[0]["replica_id"] == 3
    assert append_result.stage_results[0]["result"]["response_stage_id"] == 1

    submission = stage_port.submit_calls[-1]
    assert submission.context == stage_port.ensure_calls[-1]
    assert submission.prompt == {"prompt_token_ids": [1, 2, 3]}
    assert submission.segment_policy == ResumableSegmentPolicy(
        stop_token_ids=(99,),
        max_segment_tokens=4,
    )
    assert submission.already_submitted is False
    assert not hasattr(submission, "request_state")


def test_control_plane_accepts_only_typed_duplex_messages() -> None:
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_TypedStagePort(),
        result_sink=asyncio.Queue(),
    )

    assert plane.accepts(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=DuplexFence("typed-message"),
            session_id="typed-message",
            capabilities=DuplexRuntimeCapabilities().__dict__,
        )
    )
    assert plane.accepts(type("Lookalike", (), {"type": "open_duplex_session"})()) is False
