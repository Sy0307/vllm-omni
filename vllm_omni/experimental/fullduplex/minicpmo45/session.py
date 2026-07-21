from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    deferred_overlap_turn: bool = False
    deferred_overlap_turn_id: int | None = None
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    auto_response_waiting_for_speech: bool = False
    auto_response_new_turn_prefix_variant: str | None = None
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0

    def reserve_overlap_turn(
        self,
        *,
        session_turn_id: int,
        active_response_turn_id: int | None,
    ) -> int:
        """Latch one model turn for the complete overlapping input item."""
        if self.deferred_overlap_turn_id is None:
            reserved = int(session_turn_id)
            if active_response_turn_id is not None:
                reserved = max(reserved, int(active_response_turn_id) + 1)
            self.deferred_overlap_turn_id = reserved
        return self.deferred_overlap_turn_id

    def clear_overlap_turn(self) -> None:
        self.deferred_overlap_turn = False
        self.deferred_overlap_turn_id = None

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
        self.continuation_owner_id = None
        self.continuation_units = 0
