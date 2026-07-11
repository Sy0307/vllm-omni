# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .identity import DuplexFence

InputData: TypeAlias = bytes | str


@dataclass(frozen=True, slots=True)
class InputStarted:
    modality: str = "audio"


@dataclass(frozen=True, slots=True)
class InputChunk:
    data: InputData
    modality: str = "audio"


@dataclass(frozen=True, slots=True)
class InputCommitted:
    pass


@dataclass(frozen=True, slots=True)
class EngineAppendAccepted:
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class CommittedHistoryItem:
    role: str
    modality: str
    content: InputData


@dataclass(frozen=True, slots=True)
class HistoryCommitted:
    item: CommittedHistoryItem
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelListening:
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelSpeaking:
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    text: str
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelAudioDelta:
    data: bytes
    generated_cursor: int
    sent_cursor: int
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelSegmentEnded:
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class ModelTurnEnded:
    reason: str = "completed"
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class PlaybackAcknowledged:
    cursor: int
    committed_cursor: int | None = None
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class InterruptRequested:
    reason: str
    fence: DuplexFence | None = None


@dataclass(frozen=True, slots=True)
class SessionCloseRequested:
    reason: str = "closed"


@dataclass(frozen=True, slots=True)
class EngineFailed:
    message: str
    fence: DuplexFence | None = None


class ProtocolEventKind(str, Enum):
    MODEL_LISTENING = "model_listening"
    RESPONSE_STARTED = "response_started"
    TEXT_DELTA = "text_delta"
    AUDIO_DELTA = "audio_delta"
    SEGMENT_ENDED = "segment_ended"
    RESPONSE_COMPLETED = "response_completed"
    ENGINE_FAILED = "engine_failed"


@dataclass(frozen=True, slots=True)
class AppendToEngine:
    fence: DuplexFence
    chunk: InputChunk | None = None
    final: bool = False


@dataclass(frozen=True, slots=True)
class ReserveResponse:
    fence: DuplexFence


@dataclass(frozen=True, slots=True)
class EmitProtocolEvent:
    fence: DuplexFence
    kind: ProtocolEventKind
    payload: object | None = None


@dataclass(frozen=True, slots=True)
class CancelFence:
    fence: DuplexFence


@dataclass(frozen=True, slots=True)
class ResetStage1:
    fence: DuplexFence


@dataclass(frozen=True, slots=True)
class RebuildStage0Context:
    fence: DuplexFence
    committed_history: tuple[CommittedHistoryItem, ...]
    committed_playback_position: int


@dataclass(frozen=True, slots=True)
class CloseSessionResources:
    fence: DuplexFence


DomainEvent: TypeAlias = (
    InputStarted
    | InputChunk
    | InputCommitted
    | EngineAppendAccepted
    | HistoryCommitted
    | ModelListening
    | ModelSpeaking
    | ModelTextDelta
    | ModelAudioDelta
    | ModelSegmentEnded
    | ModelTurnEnded
    | PlaybackAcknowledged
    | InterruptRequested
    | SessionCloseRequested
    | EngineFailed
)

DuplexEffect: TypeAlias = (
    AppendToEngine
    | ReserveResponse
    | EmitProtocolEvent
    | CancelFence
    | ResetStage1
    | RebuildStage0Context
    | CloseSessionResources
)
