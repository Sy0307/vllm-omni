from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)


def _validate_identity_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a plain non-negative integer")


@dataclass(frozen=True, slots=True)
class PendingInputContinuation:
    incarnation: int
    epoch: int
    source_input_seq: int

    def __post_init__(self) -> None:
        _validate_identity_int("incarnation", self.incarnation)
        _validate_identity_int("epoch", self.epoch)
        _validate_identity_int("source_input_seq", self.source_input_seq)

    def is_stale(self, *, fence: DuplexFence, source_input_seq: int) -> bool:
        return (
            fence.incarnation != self.incarnation
            or fence.epoch != self.epoch
            or source_input_seq != self.source_input_seq
        )


@dataclass(frozen=True, slots=True)
class ActiveOutputContinuation:
    incarnation: int
    epoch: int
    output_id: str

    def __post_init__(self) -> None:
        _validate_identity_int("incarnation", self.incarnation)
        _validate_identity_int("epoch", self.epoch)
        if not isinstance(self.output_id, str) or not self.output_id.strip():
            raise ValueError("output_id must be a non-empty string")

    def is_stale(self, *, fence: DuplexFence, output_id: str) -> bool:
        return (
            fence.incarnation != self.incarnation
            or fence.epoch != self.epoch
            or output_id != self.output_id
        )


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    pending_input_continuation: PendingInputContinuation | None = None
    active_output_continuation: ActiveOutputContinuation | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.pending_input_continuation = None
        self.active_output_continuation = None
        self.continuation_units = 0
        self.pending_silence_task = None
