# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from vllm.logger import init_logger

from vllm_omni.engine.duplex_lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
)
from vllm_omni.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
    DuplexRuntimeCapabilities,
    DuplexRuntimeExtension,
    SessionMode,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex_session import (
    DuplexAppendReservation,
    DuplexSessionExpiry,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
)
from vllm_omni.engine.duplex_types import DuplexFence
from vllm_omni.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionLifecycleMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.resumable import ResumableSegmentPolicy

logger = init_logger(__name__)


@dataclass(frozen=True)
class DuplexRequestIdentity:
    """Stable duplex identity owned by an orchestrator request record."""

    session_id: str
    fence: DuplexFence


@dataclass(frozen=True)
class DuplexStageRequestContext:
    """Immutable control-plane input for a stage request adapter."""

    request_id: str
    session_id: str
    fence: DuplexFence
    stage_id: int
    final_stage_id: int
    config_generation: int
    sampling_params: tuple[object, ...]
    session_config: Mapping[str, Any] = field(default_factory=dict)
    runtime_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_params", tuple(self.sampling_params))
        object.__setattr__(self, "session_config", MappingProxyType(dict(self.session_config)))
        object.__setattr__(self, "runtime_config", MappingProxyType(dict(self.runtime_config)))

    @property
    def stage_sampling_params(self) -> object:
        return self.sampling_params[self.stage_id]


@dataclass(frozen=True)
class DuplexStageSubmission:
    """Typed append transaction handed to the orchestrator stage adapter."""

    context: DuplexStageRequestContext
    prompt: Mapping[str, Any]
    segment_policy: ResumableSegmentPolicy | None
    already_submitted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", MappingProxyType(dict(self.prompt)))


@dataclass(frozen=True)
class DuplexStageSubmissionResult:
    request_id: str
    stage_id: int
    replica_id: int


@dataclass(frozen=True)
class DuplexOutputContext:
    """Model-neutral snapshot needed to interpret one routed stage output."""

    identity: DuplexRequestIdentity
    final_stage_id: int
    segment_finished: bool
    segment_token_ids: tuple[int, ...] = ()
    segment_output_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_token_ids", tuple(self.segment_token_ids))
        object.__setattr__(
            self,
            "segment_output_metadata",
            MappingProxyType(dict(self.segment_output_metadata)),
        )


class DuplexStagePort(Protocol):
    """Narrow stage/runtime surface required by the duplex control plane."""

    @property
    def stage_count(self) -> int: ...

    def sampling_defaults(self) -> tuple[object, ...]: ...

    def ensure_request(self, context: DuplexStageRequestContext) -> None: ...

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult: ...

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None: ...


class DuplexResultSink(Protocol):
    async def put(self, message: DuplexControlResultMessage) -> None: ...


class DuplexLifecycleSink(Protocol):
    async def put(self, message: DuplexSessionLifecycleMessage) -> None: ...


class DuplexControlPlane:
    _MESSAGE_TYPES = (
        OpenDuplexSessionMessage,
        AppendDuplexInputMessage,
        SignalDuplexTurnMessage,
        CloseDuplexSessionMessage,
        TouchDuplexSessionMessage,
        ResumeDuplexSessionMessage,
    )

    def __init__(
        self,
        *,
        extension: DuplexRuntimeExtension | None,
        stage_port: DuplexStagePort,
        result_sink: DuplexResultSink,
        lifecycle_sink: DuplexLifecycleSink | None = None,
        lease_config: DuplexLeaseConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._extension = extension
        self._stage_port = stage_port
        self._result_sink = result_sink
        self._lifecycle_sink = lifecycle_sink
        self._lease_config = lease_config or DuplexLeaseConfig()
        self._sessions = DuplexSessionRuntimeManager(clock=clock)
        self._pending_expirations: dict[tuple[str, int], DuplexSessionExpiry] = {}

    @property
    def sessions(self) -> DuplexSessionRuntimeManager:
        return self._sessions

    def accepts(self, message: object) -> bool:
        return isinstance(message, self._MESSAGE_TYPES)

    async def handle(self, message: object) -> None:
        if isinstance(message, OpenDuplexSessionMessage):
            await self.handle_open(message)
        elif isinstance(message, AppendDuplexInputMessage):
            await self.handle_append(message)
        elif isinstance(message, SignalDuplexTurnMessage):
            await self.handle_signal(message)
        elif isinstance(message, CloseDuplexSessionMessage):
            await self.handle_close(message)
        elif isinstance(message, TouchDuplexSessionMessage):
            await self.handle_touch(message)
        elif isinstance(message, ResumeDuplexSessionMessage):
            await self.handle_resume(message)
        else:
            raise TypeError(f"Unsupported duplex control message: {type(message).__name__}")

    @staticmethod
    def coerce_capabilities(raw: dict[str, object]) -> DuplexRuntimeCapabilities:
        input_modes: set[DuplexInputMode] = set()
        values = raw.get("input_modes")
        if isinstance(values, list):
            for value in values:
                try:
                    input_modes.add(DuplexInputMode(value))
                except ValueError:
                    logger.warning("Ignoring unknown duplex input mode: %s", value)
        return DuplexRuntimeCapabilities(
            input_modes=input_modes or {DuplexInputMode.TURN_COMMIT_ONLY},
            implementation_level=str(raw.get("implementation_level") or "serving_session_adapter"),
        )

    def sampling_params_for_config(self, runtime_config: dict[str, Any]) -> list[object]:
        defaults = self._stage_port.sampling_defaults()
        if self._extension is None:
            return list(defaults)
        configured = self._extension.configure_sampling_params(
            runtime_config=runtime_config,
            defaults=defaults,
        )
        if not isinstance(configured, tuple):
            raise TypeError("duplex runtime extension must return sampling parameters as a tuple")
        if len(configured) != len(defaults):
            raise ValueError("duplex runtime extension must return one sampling parameter per stage")
        for sampling_params in configured:
            policy = self._extension.segment_policy(sampling_params)
            if not isinstance(policy, ResumableSegmentPolicy):
                raise TypeError("duplex runtime extension segment_policy() must return ResumableSegmentPolicy")
        return list(configured)

    @staticmethod
    def stage_request_id(fence: DuplexFence, *, stage_id: int) -> str:
        return duplex_resource_request_id(fence, f"stage{stage_id}")

    def ensure_stage_request(
        self,
        session: DuplexSessionRuntimeState,
        *,
        stage_id: int,
        fence: DuplexFence | None = None,
    ) -> DuplexStageRequestContext | None:
        if stage_id >= self._stage_port.stage_count:
            return None
        effective_fence = fence or session.fence
        request_id = self.stage_request_id(effective_fence, stage_id=stage_id)
        session.reserve_stage_request(stage_id, request_id, fence=effective_fence)
        context = DuplexStageRequestContext(
            request_id=request_id,
            session_id=session.session_id,
            fence=effective_fence,
            stage_id=stage_id,
            final_stage_id=self._stage_port.stage_count - 1,
            config_generation=session.config_generation,
            sampling_params=tuple(self.sampling_params_for_config(session.runtime_config)),
            session_config=session.session_config,
            runtime_config=session.runtime_config,
        )
        self._stage_port.ensure_request(context)
        return context

    async def handle_open(self, message: OpenDuplexSessionMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        try:
            session_mode = SessionMode(message.session_mode)
            capabilities = self.coerce_capabilities(message.capabilities)
            native_append_modes = capabilities.input_modes - {DuplexInputMode.TURN_COMMIT_ONLY}
            if native_append_modes and self._extension is None:
                raise RuntimeError("duplex_runtime_extension_not_configured")
            session = self.sessions.open_session(
                message.fence,
                capabilities=capabilities,
                session_config=message.session_config,
                runtime_config=message.runtime_config,
                lease_config=self._lease_config,
            )
            request_context = self.ensure_stage_request(session, stage_id=0)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="open",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "implementation_level": capabilities.implementation_level,
                            "data_plane_session": True,
                            "session_mode": session_mode.value,
                            "scheduler_request_context": request_context is not None,
                            "request_id": request_context.request_id if request_context is not None else None,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("open_duplex_session failed: %s", exc)
            if session is not None and self.sessions.get(message.session_id) is session:
                reserved_request_ids = session.resource_request_ids()
                self.sessions.close_session(session.fence)
                await self._stage_port.cleanup(reserved_request_ids)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="open",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def handle_append(self, message: AppendDuplexInputMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        lease_operation_id = f"append:{message.control_id}"
        operation_started = False
        try:
            session = self.sessions.require(message.session_id)
            if message.expected_epoch is not None and message.expected_epoch != message.fence.epoch:
                raise ValueError("expected_epoch must match fence.epoch")
            mode = DuplexInputMode(message.mode)
            if message.operation_id is not None:
                completed = session.completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                )
                if completed is not None:
                    session.touch(message.fence, DuplexLeaseActivity.APPEND)
                    await self.put_result(
                        message.control_id,
                        fence=message.fence,
                        operation="append",
                        session_id=message.session_id,
                        stage_results=completed,
                    )
                    return
            session.begin_operation(message.fence, lease_operation_id)
            operation_started = True
            reservation = session.prepare_append(mode=mode, fence=message.fence)
            stage_results = await self.append_via_data_plane(
                message,
                session=session,
                reservation=reservation,
                mode=mode,
            )
            if message.operation_id is not None:
                session.record_completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                    stage_results=stage_results,
                )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=stage_results,
            )
        except Exception as exc:
            logger.exception("append_duplex_input failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )
        finally:
            if session is not None and operation_started and lease_operation_id in session.lease.active_operations:
                session.end_operation(session.fence, lease_operation_id)

    async def append_via_data_plane(
        self,
        message: AppendDuplexInputMessage,
        *,
        session: DuplexSessionRuntimeState,
        reservation: DuplexAppendReservation,
        mode: DuplexInputMode,
    ) -> list[dict[str, object]]:
        if self._stage_port.stage_count == 0:
            return [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": "duplex_data_plane_has_no_stage"},
                }
            ]
        if self._extension is None:
            raise RuntimeError("duplex_runtime_extension_not_configured")

        stage_id = 0
        request_id = self.stage_request_id(message.fence, stage_id=stage_id)
        existing_binding = session.stage_bindings.get(stage_id)
        already_submitted = existing_binding is not None and existing_binding.request_id == request_id
        request_context = self.ensure_stage_request(
            session,
            stage_id=stage_id,
            fence=message.fence,
        )
        if request_context is None:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        append_plan = self._extension.plan_append(
            request_id=request_id,
            fence=message.fence,
            session_config=dict(request_context.session_config),
            runtime_config=dict(request_context.runtime_config),
            seq=reservation.update.seq,
            turn_seq=reservation.update.turn_seq,
            mode=mode,
            payload=message.payload,
            final=message.final,
            sampling_params=request_context.stage_sampling_params,
        )
        if not isinstance(append_plan, DuplexAppendPlan):
            raise TypeError("duplex runtime extension plan_append() must return DuplexAppendPlan")
        submission = DuplexStageSubmission(
            context=request_context,
            prompt=append_plan.prompt,
            segment_policy=append_plan.segment_policy,
            already_submitted=already_submitted,
        )
        submission_result = await self._stage_port.submit(submission)
        if submission_result.request_id != request_id or submission_result.stage_id != stage_id:
            raise RuntimeError("duplex stage adapter returned a mismatched submission result")
        update = session.commit_append(reservation)
        session.bind_stage_request(stage_id, request_id, fence=message.fence)
        return [
            {
                "stage_id": stage_id,
                "replica_id": submission_result.replica_id,
                "result": {
                    "supported": True,
                    "implementation_level": session.capabilities.implementation_level,
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": request_context.final_stage_id,
                    "seq": update.seq,
                    "turn_id": update.turn_id,
                    "response_seq": message.fence.response_seq,
                    "turn_seq": update.turn_seq,
                    "mode": mode.value,
                    "resumable": True,
                },
            }
        ]

    async def handle_signal(self, message: SignalDuplexTurnMessage) -> None:
        try:
            cancel_events = {"barge_in", "input.cancel", "response.cancel"}
            if message.event not in {*cancel_events, "session.update"}:
                raise ValueError(f"unsupported duplex runtime signal: {message.event}")
            session = self.sessions.require(message.session_id)
            effective_next_fence = message.next_fence
            if message.event in cancel_events:
                if effective_next_fence is None:
                    raise ValueError(f"{message.event} requires next_fence")
                stale_request_ids = session.cancel_fence(message.fence, effective_next_fence)
                if stale_request_ids:
                    await self._stage_port.cleanup(stale_request_ids, abort=True)
            else:
                session.accept_fence(message.fence)
            if message.session_config is not None:
                session.replace_session_config(message.session_config)
            if message.runtime_config is not None:
                self.sampling_params_for_config(message.runtime_config)
                session.replace_runtime_config(message.runtime_config)
            session.touch(session.fence, DuplexLeaseActivity.SIGNAL)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="signal",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "data_plane_signal": True,
                            "event": message.event,
                            "fence": message.fence,
                            "next_fence": effective_next_fence,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("signal_duplex_turn failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="signal",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def handle_close(self, message: CloseDuplexSessionMessage) -> None:
        try:
            session = self.sessions.get(message.session_id)
            if session is None:
                await self.put_result(
                    message.control_id,
                    fence=message.fence,
                    operation="close",
                    session_id=message.session_id,
                    stage_results=[],
                )
                return
            stale_request_ids = session.resource_request_ids()
            submitted_request_ids = set(session.resource_request_ids(submitted=True))
            self.sessions.close_session(message.fence, reason=message.reason)
            submitted = [request_id for request_id in stale_request_ids if request_id in submitted_request_ids]
            reserved = [request_id for request_id in stale_request_ids if request_id not in submitted_request_ids]
            if submitted:
                await self._stage_port.cleanup(submitted, abort=True)
            if reserved:
                await self._stage_port.cleanup(reserved)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="close",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "data_plane_close": True,
                            "reason": message.reason,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("close_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="close",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def handle_touch(self, message: TouchDuplexSessionMessage) -> None:
        try:
            session = self.sessions.require(message.session_id)
            activity = DuplexLeaseActivity(message.activity)
            if activity is DuplexLeaseActivity.DETACH:
                session.detach(message.fence)
            else:
                session.touch(message.fence, activity)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="touch",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "activity": activity.value,
                            "lease_generation": session.lease.generation,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("touch_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="touch",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def handle_resume(self, message: ResumeDuplexSessionMessage) -> None:
        try:
            session = self.sessions.require(message.session_id)
            generation = session.resume(
                message.fence,
                expected_lease_generation=message.expected_lease_generation,
            )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="resume",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "lease_generation": generation,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("resume_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="resume",
                session_id=message.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def reap_expired(self, now: float | None = None) -> int:
        for item in self.sessions.collect_expired(now):
            self._pending_expirations.setdefault(
                (item.session_id, item.lease_generation),
                item,
            )
        completed = 0
        for key, item in list(self._pending_expirations.items()):
            if item.submitted_request_ids:
                await self._stage_port.cleanup(list(item.submitted_request_ids), abort=True)
            if item.reserved_request_ids:
                await self._stage_port.cleanup(list(item.reserved_request_ids))
            if self._lifecycle_sink is not None:
                await self._lifecycle_sink.put(
                    DuplexSessionLifecycleMessage(
                        fence=item.fence,
                        session_id=item.session_id,
                        event="expired",
                        reason=item.reason,
                        lease_generation=item.lease_generation,
                        submitted_request_ids=list(item.submitted_request_ids),
                        reserved_request_ids=list(item.reserved_request_ids),
                    )
                )
            self._pending_expirations.pop(key, None)
            completed += 1
        return completed

    @classmethod
    def _iter_result_dicts(cls, result: object):
        if isinstance(result, dict):
            yield result
        elif isinstance(result, list | tuple):
            for item in result:
                yield from cls._iter_result_dicts(item)

    @classmethod
    def _result_counts(cls, stage_results: list[dict[str, object]]) -> tuple[int, int]:
        unsupported_count = 0
        error_count = 0
        for item in stage_results:
            for result in cls._iter_result_dicts(item.get("result")):
                if result.get("supported") is False:
                    unsupported_count += 1
                if result.get("error"):
                    error_count += 1
        return unsupported_count, error_count

    async def put_result(
        self,
        control_id: str,
        *,
        fence: DuplexFence,
        operation: str,
        session_id: str,
        stage_results: list[dict[str, object]],
        error: str | None = None,
    ) -> None:
        if error is not None:
            stage_results = [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": error},
                }
            ]
        unsupported_count, error_count = self._result_counts(stage_results)
        await self._result_sink.put(
            DuplexControlResultMessage(
                control_id=control_id,
                fence=fence,
                operation=operation,
                session_id=session_id,
                ok=error_count == 0 and unsupported_count == 0,
                stage_results=stage_results,
                unsupported_count=unsupported_count,
                error_count=error_count,
            )
        )

    def segment_policy(
        self,
        context: DuplexOutputContext | None,
        params: object,
        *,
        resumable: bool,
    ) -> ResumableSegmentPolicy | None:
        if not resumable or context is None:
            return None
        if self._extension is None:
            raise RuntimeError("duplex_runtime_extension_not_configured")
        return self._extension.segment_policy(params)

    def decide_output(
        self,
        stage_id: int,
        output: object,
        context: DuplexOutputContext | None,
    ) -> DuplexOutputDecision | None:
        if context is None or self._extension is None:
            return None
        session = self._sessions.get(context.identity.session_id)
        if session is not None:
            try:
                session.touch(context.identity.fence, DuplexLeaseActivity.MODEL_OUTPUT)
            except RuntimeError:
                pass
        decision = self._extension.decide_output(
            stage_id=stage_id,
            final_stage_id=context.final_stage_id,
            segment_finished=context.segment_finished,
            segment_token_ids=context.segment_token_ids,
            segment_output_metadata=dict(context.segment_output_metadata),
            output=output,
        )
        if decision is not None and not isinstance(decision, DuplexOutputDecision):
            raise TypeError("duplex runtime extension decide_output() must return DuplexOutputDecision or None")
        return decision

    def session_for_identity(self, identity: DuplexRequestIdentity | None) -> DuplexSessionRuntimeState | None:
        if identity is None:
            return None
        return self._sessions.get(identity.session_id)

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]:
        return self._sessions.close_sessions_for_request_ids(request_ids)


__all__ = [
    "DuplexControlPlane",
    "DuplexOutputContext",
    "DuplexRequestIdentity",
    "DuplexResultSink",
    "DuplexLifecycleSink",
    "DuplexStagePort",
    "DuplexStageRequestContext",
    "DuplexStageSubmission",
    "DuplexStageSubmissionResult",
]
