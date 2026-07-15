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
    committed_audio_payload: dict[str, object] | None = None
    deferred_response_create: bool = False
    auto_response_waiting_for_speech: bool = False
    auto_response_new_turn_prefix_variant: str | None = None
    data_plane_task: asyncio.Task[None] | None = None
    continuation_response_id: str | None = None
    continuation_units: int = 0

    def clear_continuation(self) -> None:
        self.continuation_response_id = None
        self.continuation_units = 0
