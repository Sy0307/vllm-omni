# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeAlias

from .events import (
    AppendToEngine,
    EmitProtocolEvent,
    EngineAppendAccepted,
    EngineFailed,
    HistoryCommitted,
    ModelAudioDelta,
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
)
from .identity import DuplexFence

EngineEvent: TypeAlias = (
    EngineAppendAccepted
    | EngineFailed
    | HistoryCommitted
    | ModelListening
    | ModelSpeaking
    | ModelTextDelta
    | ModelAudioDelta
    | ModelSegmentEnded
    | ModelTurnEnded
)


class DuplexEnginePort(Protocol):
    async def reserve(self, command: ReserveResponse) -> None: ...

    async def append(self, command: AppendToEngine) -> None: ...

    def events(self) -> AsyncIterator[EngineEvent]: ...

    async def cancel(self, fence: DuplexFence) -> None: ...

    async def reset(self, command: ResetStage1) -> None: ...

    async def rebuild(self, command: RebuildStage0Context) -> None: ...

    async def close(self, fence: DuplexFence) -> None: ...


class DuplexEventSink(Protocol):
    async def emit(self, event: EmitProtocolEvent) -> None: ...


__all__ = ["DuplexEnginePort", "DuplexEventSink", "EngineEvent"]
