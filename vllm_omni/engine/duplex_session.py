# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_omni.engine.duplex_lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
    DuplexSessionExpiry,
)
from vllm_omni.engine.duplex_types import DuplexFence

if TYPE_CHECKING:
    from vllm_omni.engine.duplex_runtime import (
        DuplexInputMode,
        DuplexRuntimeCapabilities,
    )


def _default_capabilities() -> DuplexRuntimeCapabilities:
    from vllm_omni.engine.duplex_runtime import DuplexRuntimeCapabilities

    return DuplexRuntimeCapabilities()


class DuplexFenceMismatchError(RuntimeError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence) -> None:
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


@dataclass
class DuplexStageBinding:
    request_id: str
    fence: DuplexFence


@dataclass
class DuplexRequestResource:
    stage_id: int
    request_id: str
    fence: DuplexFence
    submitted: bool = False


@dataclass
class DuplexInputAppend:
    seq: int
    turn_seq: int
    turn_id: int


@dataclass(frozen=True)
class DuplexAppendReservation:
    fence: DuplexFence
    mode: DuplexInputMode
    base_fence: DuplexFence
    base_input_seq: int
    base_input_turn_seq: int
    base_append_turn_key: tuple[int, int, int] | None
    update: DuplexInputAppend


@dataclass(frozen=True)
class DuplexCompletedAppend:
    fence: DuplexFence
    mode: DuplexInputMode
    final: bool
    stage_results: tuple[dict[str, object], ...]


@dataclass
class DuplexSessionRuntimeState:
    """Engine resource handles associated with a core-owned identity fence."""

    fence: DuplexFence
    lease: DuplexLeaseState
    _clock: Callable[[], float] = field(repr=False)
    capabilities: DuplexRuntimeCapabilities = field(default_factory=_default_capabilities)
    session_config: dict[str, Any] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    config_generation: int = 0
    stage_bindings: dict[int, DuplexStageBinding] = field(default_factory=dict)
    request_resources: dict[tuple[int, str], DuplexRequestResource] = field(default_factory=dict)
    input_seq: int = 0
    input_turn_seq: int = 0
    _append_turn_key: tuple[int, int, int] | None = None
    completed_appends: dict[str, DuplexCompletedAppend] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.fence.session_id

    @property
    def epoch(self) -> int:
        return self.fence.epoch

    @property
    def turn_id(self) -> int:
        return self.fence.turn_id

    def _validate_fence(self, fence: DuplexFence) -> None:
        if fence.session_id != self.session_id or fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, fence)
        current = self.fence
        if fence.epoch < current.epoch or (
            fence.epoch == current.epoch
            and (fence.turn_id < current.turn_id or fence.response_seq < current.response_seq)
        ):
            raise DuplexFenceMismatchError(current, fence)

    def accept_fence(self, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        if fence.epoch != self.fence.epoch:
            self.input_seq = 0
            self.input_turn_seq = 0
            self._append_turn_key = None
        self.fence = fence

    def touch(self, fence: DuplexFence, activity: DuplexLeaseActivity) -> None:
        self._validate_fence(fence)
        self.lease.touch(self._clock(), activity)

    def detach(self, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        self.lease.detach(self._clock())

    def resume(self, fence: DuplexFence, *, expected_lease_generation: int) -> int:
        self._validate_fence(fence)
        return self.lease.resume(
            self._clock(),
            expected_generation=expected_lease_generation,
        )

    def begin_operation(self, fence: DuplexFence, operation_id: str) -> None:
        self._validate_fence(fence)
        self.lease.begin_operation(self._clock(), operation_id)

    def end_operation(self, fence: DuplexFence, operation_id: str) -> None:
        self._validate_fence(fence)
        self.lease.end_operation(self._clock(), operation_id)

    def replace_session_config(self, session_config: dict[str, Any]) -> None:
        self.session_config = dict(session_config)
        self.config_generation += 1

    def replace_runtime_config(self, runtime_config: dict[str, Any]) -> None:
        self.runtime_config = dict(runtime_config)
        self.config_generation += 1

    def reserve_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        resource_key = (stage_id, request_id)
        existing = self.request_resources.get(resource_key)
        if existing is not None:
            if (
                existing.fence.session_id != fence.session_id
                or existing.fence.incarnation != fence.incarnation
                or existing.fence.epoch != fence.epoch
            ):
                raise ValueError(f"Duplex request resource already reserved with different identity: {request_id}")
            return
        self.request_resources[resource_key] = DuplexRequestResource(
            stage_id=stage_id,
            request_id=request_id,
            fence=fence,
        )

    def bind_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self.reserve_stage_request(stage_id, request_id, fence=fence)
        self.accept_fence(fence)
        resource = self.request_resources[(stage_id, request_id)]
        resource.fence = fence
        resource.submitted = True
        self.stage_bindings[stage_id] = DuplexStageBinding(request_id=request_id, fence=fence)

    def stage_request_ids(self, fence: DuplexFence | None = None) -> list[str]:
        return [
            binding.request_id for binding in self.stage_bindings.values() if fence is None or binding.fence == fence
        ]

    def resource_request_ids(
        self,
        fence: DuplexFence | None = None,
        *,
        submitted: bool | None = None,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                resource.request_id
                for resource in self.request_resources.values()
                if (fence is None or resource.fence == fence) and (submitted is None or resource.submitted is submitted)
            )
        )

    def prepare_append(self, *, mode: DuplexInputMode, fence: DuplexFence) -> DuplexAppendReservation:
        if mode not in self.capabilities.input_modes:
            raise ValueError(f"Duplex input mode {mode.value!r} is not supported by session {self.session_id}")
        self._validate_fence(fence)
        input_seq = 0 if fence.epoch != self.fence.epoch else self.input_seq
        input_turn_seq = 0 if fence.epoch != self.fence.epoch else self.input_turn_seq
        append_turn_key = None if fence.epoch != self.fence.epoch else self._append_turn_key
        turn_key = (fence.epoch, fence.turn_id, fence.response_seq)
        turn_seq = input_turn_seq + 1 if turn_key == append_turn_key else 1
        return DuplexAppendReservation(
            fence=fence,
            mode=mode,
            base_fence=self.fence,
            base_input_seq=self.input_seq,
            base_input_turn_seq=self.input_turn_seq,
            base_append_turn_key=self._append_turn_key,
            update=DuplexInputAppend(
                seq=input_seq + 1,
                turn_seq=turn_seq,
                turn_id=fence.turn_id,
            ),
        )

    def commit_append(self, reservation: DuplexAppendReservation) -> DuplexInputAppend:
        if (
            self.fence != reservation.base_fence
            or self.input_seq != reservation.base_input_seq
            or self.input_turn_seq != reservation.base_input_turn_seq
            or self._append_turn_key != reservation.base_append_turn_key
        ):
            raise RuntimeError("duplex append reservation is stale")
        self.accept_fence(reservation.fence)
        self.input_seq = reservation.update.seq
        self.input_turn_seq = reservation.update.turn_seq
        self._append_turn_key = (
            reservation.fence.epoch,
            reservation.fence.turn_id,
            reservation.fence.response_seq,
        )
        return reservation.update

    def completed_append(
        self,
        operation_id: str,
        *,
        fence: DuplexFence,
        mode: DuplexInputMode,
        final: bool,
    ) -> list[dict[str, object]] | None:
        self._validate_fence(fence)
        completed = self.completed_appends.get(operation_id)
        if completed is None:
            return None
        if completed.fence != fence or completed.mode != mode or completed.final != final:
            raise ValueError(f"duplex append operation {operation_id!r} was reused with different metadata")
        return [dict(result) for result in completed.stage_results]

    def record_completed_append(
        self,
        operation_id: str,
        *,
        fence: DuplexFence,
        mode: DuplexInputMode,
        final: bool,
        stage_results: list[dict[str, object]],
    ) -> None:
        self.completed_appends[operation_id] = DuplexCompletedAppend(
            fence=fence,
            mode=mode,
            final=final,
            stage_results=tuple(dict(result) for result in stage_results),
        )

    def append_input(self, *, mode: DuplexInputMode, fence: DuplexFence) -> DuplexInputAppend:
        return self.commit_append(self.prepare_append(mode=mode, fence=fence))

    def release_fence(self, fence: DuplexFence) -> list[str]:
        stale = self.resource_request_ids(fence)
        self.stage_bindings = {
            stage_id: binding for stage_id, binding in self.stage_bindings.items() if binding.fence != fence
        }
        self.request_resources = {
            resource_key: resource
            for resource_key, resource in self.request_resources.items()
            if resource.fence != fence
        }
        return stale

    def cancel_fence(self, cancelled_fence: DuplexFence, next_fence: DuplexFence) -> list[str]:
        if cancelled_fence.session_id != self.session_id or cancelled_fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if (
            next_fence.session_id != self.session_id
            or next_fence.incarnation != self.fence.incarnation
            or next_fence.epoch <= cancelled_fence.epoch
        ):
            raise DuplexFenceMismatchError(cancelled_fence, next_fence)
        current_key = (self.fence.epoch, self.fence.turn_id, self.fence.response_seq)
        cancelled_key = (cancelled_fence.epoch, cancelled_fence.turn_id, cancelled_fence.response_seq)
        next_key = (next_fence.epoch, next_fence.turn_id, next_fence.response_seq)
        if cancelled_key > current_key:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if next_key > current_key:
            self.accept_fence(next_fence)
        return self.release_fence(cancelled_fence)

    def close(self, fence: DuplexFence | None = None) -> list[str]:
        if fence is not None:
            self.accept_fence(fence)
        stale = self.resource_request_ids()
        self.stage_bindings.clear()
        self.request_resources.clear()
        return stale

    def terminate(self, fence: DuplexFence, *, reason: str) -> bool:
        self._validate_fence(fence)
        if not self.lease.mark_terminal(reason):
            return False
        self.close(fence)
        return True


class DuplexSessionRuntimeManager:
    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._sessions: dict[str, DuplexSessionRuntimeState] = {}
        self._clock = clock or time.monotonic

    def open_session(
        self,
        fence: DuplexFence,
        *,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
        lease_config: DuplexLeaseConfig | None = None,
    ) -> DuplexSessionRuntimeState:
        if not isinstance(fence, DuplexFence):
            raise TypeError("open_session requires DuplexFence")
        if fence.session_id in self._sessions:
            raise ValueError(f"Duplex session already exists: {fence.session_id}")
        session = DuplexSessionRuntimeState(
            fence=fence,
            lease=DuplexLeaseState(
                config=lease_config or DuplexLeaseConfig(),
                generation=0,
                last_activity=self._clock(),
            ),
            _clock=self._clock,
            capabilities=capabilities or _default_capabilities(),
            session_config=dict(session_config or {}),
            runtime_config=dict(runtime_config or {}),
        )
        self._sessions[fence.session_id] = session
        return session

    def get(self, session_id: str) -> DuplexSessionRuntimeState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> DuplexSessionRuntimeState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown duplex session: {session_id}")
        return session

    def close_session(
        self,
        fence: DuplexFence,
        *,
        reason: str = "explicit_close",
    ) -> DuplexSessionRuntimeState | None:
        if not isinstance(fence, DuplexFence):
            raise TypeError("close_session requires DuplexFence")
        session = self._sessions.get(fence.session_id)
        if session is not None:
            if not session.terminate(fence, reason=reason):
                return None
            if self._sessions.get(fence.session_id) is session:
                self._sessions.pop(fence.session_id)
        return session

    def collect_expired(self, now: float | None = None) -> list[DuplexSessionExpiry]:
        effective_now = self._clock() if now is None else now
        expired: list[DuplexSessionExpiry] = []
        for session_id, session in list(self._sessions.items()):
            if not session.lease.idle_expired(effective_now):
                continue
            submitted = tuple(session.resource_request_ids(submitted=True))
            reserved = tuple(session.resource_request_ids(submitted=False))
            if not session.terminate(session.fence, reason="idle_ttl_expired"):
                continue
            self._sessions.pop(session_id, None)
            expired.append(
                DuplexSessionExpiry(
                    session_id=session_id,
                    fence=session.fence,
                    lease_generation=session.lease.generation,
                    reason="idle_ttl_expired",
                    submitted_request_ids=submitted,
                    reserved_request_ids=reserved,
                )
            )
        return expired

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]:
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        for session_id, session in list(self._sessions.items()):
            stale = session.resource_request_ids()
            if request_id_set.isdisjoint(stale):
                continue
            if not session.terminate(session.fence, reason="request_cleanup"):
                continue
            self._sessions.pop(session_id, None)
            closed[session_id] = stale
        return closed


__all__ = [
    "DuplexAppendReservation",
    "DuplexCompletedAppend",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexLeaseActivity",
    "DuplexLeaseConfig",
    "DuplexLeaseState",
    "DuplexRequestResource",
    "DuplexSessionExpiry",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
]
