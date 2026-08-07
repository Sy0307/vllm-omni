# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexExecutionProfile,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import (
    DuplexControlPlane,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexFence,
    OpenDuplexSessionMessage,
    SignalDuplexTurnMessage,
)
from vllm_omni.experimental.fullduplex.engine.resource_lease import (
    DuplexResourceLeaseCoordinator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingProvider:
    def __init__(
        self,
        provider_id: str,
        calls: list[str],
        *,
        reserve_error: Exception | None = None,
        release_failures: int = 0,
    ) -> None:
        self.provider_id = provider_id
        self.calls = calls
        self.reserve_error = reserve_error
        self.release_failures = release_failures

    async def prewarm(self, batch_sizes: tuple[int, ...]) -> None:
        self.calls.append(f"prewarm:{self.provider_id}:{batch_sizes}")

    async def reserve(
        self,
        fence: DuplexFence,
        *,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
    ) -> object:
        self.calls.append(f"reserve:{self.provider_id}:{fence.session_id}")
        if self.reserve_error is not None:
            raise self.reserve_error
        return f"handle:{self.provider_id}:{fence.incarnation}"

    async def release(self, handle: object, *, abort: bool) -> None:
        self.calls.append(f"release:{self.provider_id}:{handle}:{abort}")
        if self.release_failures:
            self.release_failures -= 1
            raise RuntimeError(f"release failed: {self.provider_id}")

    async def advance_epoch(
        self,
        handle: object,
        *,
        cancelled_fence: DuplexFence,
        next_fence: DuplexFence,
    ) -> None:
        self.calls.append(f"advance:{self.provider_id}:{handle}:{cancelled_fence.epoch}->{next_fence.epoch}")


def test_resource_lease_reserves_in_order_and_releases_in_reverse() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        providers = tuple(_RecordingProvider(name, calls) for name in ("a", "b", "c"))
        coordinator = DuplexResourceLeaseCoordinator(providers)
        fence = DuplexFence("session", incarnation=4, epoch=2)

        await coordinator.reserve(fence, session_config={"voice": "a"}, runtime_config={})
        await coordinator.release(fence, abort=False)

        assert calls == [
            "reserve:a:session",
            "reserve:b:session",
            "reserve:c:session",
            "release:c:handle:c:4:False",
            "release:b:handle:b:4:False",
            "release:a:handle:a:4:False",
        ]
        assert coordinator.has_lease(fence) is False

    asyncio.run(scenario())


def test_resource_lease_rolls_back_reverse_order_when_reserve_fails() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        providers = (
            _RecordingProvider("a", calls),
            _RecordingProvider("b", calls),
            _RecordingProvider("c", calls, reserve_error=RuntimeError("capacity")),
        )
        coordinator = DuplexResourceLeaseCoordinator(providers)
        fence = DuplexFence("session", incarnation=1)

        with pytest.raises(RuntimeError, match="capacity"):
            await coordinator.reserve(fence, session_config={}, runtime_config={})

        assert calls == [
            "reserve:a:session",
            "reserve:b:session",
            "reserve:c:session",
            "release:b:handle:b:1:True",
            "release:a:handle:a:1:True",
        ]
        assert coordinator.has_lease(fence) is False

    asyncio.run(scenario())


def test_resource_release_retry_skips_already_released_handles() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        providers = (
            _RecordingProvider("a", calls),
            _RecordingProvider("b", calls, release_failures=1),
            _RecordingProvider("c", calls),
        )
        coordinator = DuplexResourceLeaseCoordinator(providers)
        fence = DuplexFence("session", incarnation=1)
        await coordinator.reserve(fence, session_config={}, runtime_config={})

        with pytest.raises(RuntimeError, match="release failed: b"):
            await coordinator.release(fence, abort=True)
        assert coordinator.has_lease(fence) is True

        await coordinator.release(fence, abort=True)
        assert calls.count("release:c:handle:c:1:True") == 1
        assert calls.count("release:b:handle:b:1:True") == 2
        assert calls.count("release:a:handle:a:1:True") == 1
        assert coordinator.has_lease(fence) is False

    asyncio.run(scenario())


def test_resource_prewarm_runs_once_per_provider() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        providers = tuple(_RecordingProvider(name, calls) for name in ("a", "b"))
        coordinator = DuplexResourceLeaseCoordinator(providers)

        await coordinator.prewarm((1, 2))
        await coordinator.prewarm((1, 2))

        assert calls == ["prewarm:a:(1, 2)", "prewarm:b:(1, 2)"]

    asyncio.run(scenario())


class _Extension:
    def __init__(
        self,
        providers: tuple[object, ...],
        *,
        profile: DuplexExecutionProfile = DuplexExecutionProfile(),
    ) -> None:
        self._providers = providers
        self._profile = profile

    def resource_lease_providers(self) -> tuple[object, ...]:
        return self._providers

    def execution_profile(self) -> DuplexExecutionProfile:
        return self._profile

    def configure_sampling_params(self, *, runtime_config, defaults):
        del runtime_config
        return defaults

    def plan_append(self, **kwargs):
        del kwargs
        return DuplexAppendPlan(prompt={"prompt_token_ids": [1]})

    def decide_output(self, **kwargs):
        del kwargs
        return None


class _StagePort:
    stage_count = 1

    def __init__(self, *, on_submit=None) -> None:
        self.on_submit = on_submit
        self.cleanup_calls: list[tuple[list[str], bool]] = []

    def sampling_defaults(self) -> tuple[object, ...]:
        return (object(),)

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        del context

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        if self.on_submit is not None:
            self.on_submit()
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=0,
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


def _open_message(fence: DuplexFence) -> OpenDuplexSessionMessage:
    return OpenDuplexSessionMessage(
        control_id=f"open-{fence.session_id}",
        fence=fence,
        session_id=fence.session_id,
        capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        session_config={"voice": "test"},
        runtime_config={},
    )


@pytest.mark.asyncio
async def test_control_plane_holds_admission_until_resource_release_retry_succeeds() -> None:
    calls: list[str] = []
    provider = _RecordingProvider("model", calls, release_failures=1)
    sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension((provider,)),
        stage_port=_StagePort(),
        result_sink=sink,
        max_sessions=1,
    )
    fence = DuplexFence("session", incarnation=1)

    await plane.handle(_open_message(fence))
    assert (await sink.get()).ok is True
    await plane.handle(
        CloseDuplexSessionMessage(
            control_id="close-1",
            fence=fence,
            session_id=fence.session_id,
            reason="client_close",
        )
    )
    assert (await sink.get()).ok is False
    assert plane.sessions.get(fence.session_id) is not None

    await plane.handle(
        CloseDuplexSessionMessage(
            control_id="close-2",
            fence=fence,
            session_id=fence.session_id,
            reason="client_close",
        )
    )
    assert (await sink.get()).ok is True
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_cancel_keeps_session_resource_lease_until_close() -> None:
    calls: list[str] = []
    provider = _RecordingProvider("model", calls)
    sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension((provider,)),
        stage_port=_StagePort(),
        result_sink=sink,
    )
    fence = DuplexFence("session", incarnation=1, epoch=0)
    next_fence = DuplexFence("session", incarnation=1, epoch=1)

    await plane.handle(_open_message(fence))
    assert (await sink.get()).ok is True
    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="cancel",
            fence=fence,
            next_fence=next_fence,
            session_id=fence.session_id,
            event="response.cancel",
        )
    )
    assert (await sink.get()).ok is True
    assert not any(call.startswith("release:model") for call in calls)
    assert [call for call in calls if call.startswith("advance:model")] == ["advance:model:handle:model:1:0->1"]
    assert plane._resource_leases.has_lease(next_fence)

    await plane.handle(
        CloseDuplexSessionMessage(
            control_id="close",
            fence=next_fence,
            session_id=next_fence.session_id,
            reason="client_close",
        )
    )
    assert (await sink.get()).ok is True
    assert [call for call in calls if call.startswith("release:model")] == ["release:model:handle:model:1:False"]


@pytest.mark.asyncio
async def test_control_plane_prewarms_once_and_records_deadline_miss_without_failing_step() -> None:
    calls: list[str] = []
    provider = _RecordingProvider("model", calls)
    clock = _Clock()
    sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(
            (provider,),
            profile=DuplexExecutionProfile(
                prewarm_batch_sizes=(1, 2),
                step_latency_budget_ms=80.0,
            ),
        ),
        stage_port=_StagePort(on_submit=lambda: clock.advance(0.1)),
        result_sink=sink,
        clock=clock,
    )
    fence = DuplexFence("session", incarnation=1)

    await plane.handle(_open_message(fence))
    assert (await sink.get()).ok is True
    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"frame": 1},
        )
    )
    assert (await sink.get()).ok is True

    snapshot = plane.execution_metrics.snapshot()
    assert snapshot.count == 1
    assert snapshot.p95_ms == pytest.approx(100.0)
    assert snapshot.max_ms == pytest.approx(100.0)
    assert snapshot.deadline_misses == 1
    assert calls.count("prewarm:model:(1, 2)") == 1
