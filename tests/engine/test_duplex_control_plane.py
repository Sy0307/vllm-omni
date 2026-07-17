# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import msgspec
import pytest

from vllm_omni.engine.duplex_control_plane import (
    DuplexControlPlane,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex_lease import DuplexLeaseActivity, DuplexLeaseConfig
from vllm_omni.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
)
from vllm_omni.engine.duplex_types import DuplexFence
from vllm_omni.engine.messages import (
    AppendDuplexInputMessage,
    DuplexSessionLifecycleMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
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


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


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


@pytest.mark.parametrize(
    "message",
    [
        TouchDuplexSessionMessage(
            control_id="touch-1",
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        ),
        ResumeDuplexSessionMessage(
            control_id="resume-1",
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            expected_lease_generation=3,
        ),
        DuplexSessionLifecycleMessage(
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            event="expired",
            reason="idle_ttl_expired",
            lease_generation=4,
            submitted_request_ids=["req-submitted"],
            reserved_request_ids=["req-reserved"],
        ),
    ],
)
def test_duplex_lease_messages_round_trip(message) -> None:
    encoded = msgspec.json.encode(message)
    decoded = msgspec.json.decode(encoded, type=type(message))

    assert decoded == message


@pytest.mark.asyncio
async def test_control_plane_touches_append_signal_and_explicit_activity() -> None:
    clock = _Clock()
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=30.0, disconnect_grace_s=5.0),
        clock=clock,
    )
    fence = DuplexFence("sid-touch")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        )
    )
    await result_sink.get()
    session = plane.sessions.require(fence.session_id)

    clock.advance(1.0)
    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    assert (await result_sink.get()).ok is True
    assert session.lease.last_activity == 1.0

    clock.advance(1.0)
    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="signal",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
        )
    )
    assert (await result_sink.get()).ok is True
    assert session.lease.last_activity == 2.0

    submit_count = len(stage_port.submit_calls)
    clock.advance(1.0)
    await plane.handle(
        TouchDuplexSessionMessage(
            control_id="heartbeat",
            fence=fence,
            session_id=fence.session_id,
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        )
    )
    touch_result = await result_sink.get()
    assert touch_result.ok is True
    assert session.lease.last_activity == 3.0
    assert len(stage_port.submit_calls) == submit_count
    assert lifecycle_sink.empty()


@pytest.mark.asyncio
async def test_control_plane_resume_requires_expected_lease_generation() -> None:
    clock = _Clock()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_TypedStagePort(),
        result_sink=result_sink,
        lifecycle_sink=asyncio.Queue(),
        lease_config=DuplexLeaseConfig(),
        clock=clock,
    )
    fence = DuplexFence("sid-resume-control")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={},
        )
    )
    await result_sink.get()
    await plane.handle(
        TouchDuplexSessionMessage(
            control_id="detach",
            fence=fence,
            session_id=fence.session_id,
            activity=DuplexLeaseActivity.DETACH.value,
        )
    )
    await result_sink.get()

    await plane.handle(
        ResumeDuplexSessionMessage(
            control_id="resume",
            fence=fence,
            session_id=fence.session_id,
            expected_lease_generation=0,
        )
    )
    result = await result_sink.get()

    assert result.ok is True
    assert result.stage_results[0]["result"]["lease_generation"] == 1

    await plane.handle(
        ResumeDuplexSessionMessage(
            control_id="stale-resume",
            fence=fence,
            session_id=fence.session_id,
            expected_lease_generation=0,
        )
    )
    assert (await result_sink.get()).ok is False


@pytest.mark.asyncio
async def test_control_plane_reaps_one_session_through_cleanup_and_lifecycle_sink() -> None:
    clock = _Clock()
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=2.0, disconnect_grace_s=1.0),
        clock=clock,
    )
    for session_id in ("sid-a", "sid-b"):
        fence = DuplexFence(session_id)
        await plane.handle(
            OpenDuplexSessionMessage(
                control_id=f"open-{session_id}",
                fence=fence,
                session_id=session_id,
                capabilities={},
            )
        )
        await result_sink.get()
    clock.advance(1.0)
    plane.sessions.require("sid-b").touch(DuplexFence("sid-b"), DuplexLeaseActivity.HEARTBEAT)
    clock.advance(1.1)

    expired_count = await plane.reap_expired()

    assert expired_count == 1
    assert plane.sessions.get("sid-a") is None
    assert plane.sessions.get("sid-b") is not None
    assert stage_port.cleanup_calls == [([plane.stage_request_id(DuplexFence("sid-a"), stage_id=0)], False)]
    lifecycle = await lifecycle_sink.get()
    assert isinstance(lifecycle, DuplexSessionLifecycleMessage)
    assert lifecycle.session_id == "sid-a"
    assert lifecycle.reason == "idle_ttl_expired"
    assert result_sink.empty()


@pytest.mark.asyncio
async def test_control_plane_retries_expired_cleanup_before_publishing_lifecycle() -> None:
    class _FailOnceStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.failures_remaining = 1

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise RuntimeError("transient cleanup failure")
            await super().cleanup(request_ids, abort=abort)

    clock = _Clock()
    stage_port = _FailOnceStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0, disconnect_grace_s=1.0),
        clock=clock,
    )
    fence = DuplexFence("sid-retry-expiry")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-retry",
            fence=fence,
            session_id=fence.session_id,
            capabilities={},
        )
    )
    await result_sink.get()
    clock.advance(2.0)

    with pytest.raises(RuntimeError, match="transient cleanup failure"):
        await plane.reap_expired()
    assert lifecycle_sink.empty()

    assert await plane.reap_expired() == 1
    assert (await lifecycle_sink.get()).session_id == fence.session_id
