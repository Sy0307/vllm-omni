# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed model events for the experimental Full-Duplex runtime."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias
from uuid import uuid4

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


def _validate_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_fence(fence: object) -> None:
    if not isinstance(fence, DuplexFence):
        raise TypeError("fence must be a DuplexFence")


@dataclass(frozen=True, slots=True)
class DuplexListen:
    fence: DuplexFence
    source_input_seq: int
    reason: str = "model_listen"

    def __post_init__(self) -> None:
        _validate_fence(self.fence)
        _validate_non_negative_int("source_input_seq", self.source_input_seq)
        _validate_non_empty_string("reason", self.reason)


@dataclass(frozen=True, slots=True)
class DuplexSpeakStart:
    fence: DuplexFence
    source_input_seq: int
    output_id: str

    def __post_init__(self) -> None:
        _validate_fence(self.fence)
        _validate_non_negative_int("source_input_seq", self.source_input_seq)
        _validate_non_empty_string("output_id", self.output_id)


@dataclass(frozen=True, slots=True)
class DuplexSpeakChunk:
    fence: DuplexFence
    output_id: str
    output_seq: int
    text_delta: str = ""
    audio_data: str = ""
    audio_format: str = "wav"
    audio_duration_ms: int | None = None
    audio_text_marks: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _validate_fence(self.fence)
        _validate_non_empty_string("output_id", self.output_id)
        _validate_non_negative_int("output_seq", self.output_seq)
        if not isinstance(self.text_delta, str):
            raise TypeError("text_delta must be a string")
        if not isinstance(self.audio_data, str):
            raise TypeError("audio_data must be a string")
        if not self.text_delta and not self.audio_data:
            raise ValueError("DuplexSpeakChunk must contain text or audio")
        _validate_non_empty_string("audio_format", self.audio_format)
        if self.audio_duration_ms is not None:
            _validate_non_negative_int("audio_duration_ms", self.audio_duration_ms)
        if not isinstance(self.audio_text_marks, tuple):
            raise TypeError("audio_text_marks must be a tuple")
        for mark in self.audio_text_marks:
            if not isinstance(mark, tuple) or len(mark) != 2:
                raise TypeError("audio_text_marks must contain integer pairs")
            _validate_non_negative_int("audio_text_marks", mark[0])
            _validate_non_negative_int("audio_text_marks", mark[1])


@dataclass(frozen=True, slots=True)
class DuplexSpeakEnd:
    fence: DuplexFence
    output_id: str
    reason: str = "completed"

    def __post_init__(self) -> None:
        _validate_fence(self.fence)
        _validate_non_empty_string("output_id", self.output_id)
        _validate_non_empty_string("reason", self.reason)


DuplexModelEvent: TypeAlias = DuplexListen | DuplexSpeakStart | DuplexSpeakChunk | DuplexSpeakEnd


class DuplexEventProtocolError(RuntimeError):
    """Raised when a model event violates Full-Duplex output ordering."""


@dataclass(frozen=True, slots=True)
class _CompletedOutput:
    next_output_seq: int


class DuplexOutputLedger:
    """Validate and allocate one ordered assistant output per duplex session."""

    def __init__(self, fence: DuplexFence, *, completed_output_limit: int = 256) -> None:
        _validate_fence(fence)
        if type(completed_output_limit) is not int or completed_output_limit <= 0:
            raise ValueError("completed_output_limit must be a positive integer")
        self._fence = fence
        self._completed_output_limit = completed_output_limit
        self._active_output_id: str | None = None
        self._active_output_source_input_seq: int | None = None
        self._next_output_seq = 0
        self._completed_source_input_seq: int | None = None
        self._completed: OrderedDict[str, _CompletedOutput] = OrderedDict()
        self._seen_listens: OrderedDict[tuple[int, int], None] = OrderedDict()

    @property
    def fence(self) -> DuplexFence:
        return self._fence

    @property
    def active_output_id(self) -> str | None:
        return self._active_output_id

    @property
    def next_output_seq(self) -> int:
        return self._next_output_seq

    def advance_epoch(self, fence: DuplexFence) -> None:
        _validate_fence(fence)
        if fence.session_id != self._fence.session_id or fence.incarnation != self._fence.incarnation:
            raise DuplexEventProtocolError(f"duplex event fence does not belong to this ledger: {fence!r}")
        if fence.epoch < self._fence.epoch:
            raise DuplexEventProtocolError(
                f"duplex event fence moved backwards: current={self._fence!r}, next={fence!r}"
            )
        if fence.epoch == self._fence.epoch:
            self._fence = fence
            return
        self._fence = fence
        self._active_output_id = None
        self._active_output_source_input_seq = None
        self._next_output_seq = 0
        self._completed_source_input_seq = None
        self._completed.clear()
        self._seen_listens.clear()

    def accept(self, event: DuplexModelEvent) -> bool:
        if not isinstance(event, (DuplexListen, DuplexSpeakStart, DuplexSpeakChunk, DuplexSpeakEnd)):
            raise TypeError(f"unsupported duplex model event: {type(event).__name__}")
        if event.fence.session_id != self._fence.session_id or event.fence.incarnation != self._fence.incarnation:
            raise DuplexEventProtocolError(f"duplex event fence does not belong to this ledger: {event.fence!r}")
        if event.fence.epoch < self._fence.epoch:
            return False
        if event.fence.epoch > self._fence.epoch:
            self.advance_epoch(event.fence)

        if isinstance(event, DuplexListen):
            key = (event.fence.epoch, event.source_input_seq)
            if key in self._seen_listens:
                return False
            self._seen_listens[key] = None
            self._seen_listens.move_to_end(key)
            while len(self._seen_listens) > self._completed_output_limit:
                self._seen_listens.popitem(last=False)
            return True

        if isinstance(event, DuplexSpeakStart):
            if (
                self._completed_source_input_seq is not None
                and event.source_input_seq <= self._completed_source_input_seq
            ):
                return False
            if event.output_id == self._active_output_id or event.output_id in self._completed:
                return False
            if self._active_output_id is not None:
                raise DuplexEventProtocolError(f"duplex output {self._active_output_id!r} is already active")
            self._active_output_id = event.output_id
            self._active_output_source_input_seq = event.source_input_seq
            self._next_output_seq = 0
            return True

        if isinstance(event, DuplexSpeakChunk):
            completed = self._completed.get(event.output_id)
            if completed is not None:
                if event.output_seq < completed.next_output_seq:
                    return False
                raise DuplexEventProtocolError(f"duplex output chunk arrived after end: output_id={event.output_id!r}")
            if self._active_output_id is None:
                raise DuplexEventProtocolError(f"duplex chunk references unknown output: output_id={event.output_id!r}")
            if event.output_id != self._active_output_id:
                raise DuplexEventProtocolError(f"duplex chunk references unknown output: output_id={event.output_id!r}")
            if event.output_seq < self._next_output_seq:
                return False
            if event.output_seq > self._next_output_seq:
                raise DuplexEventProtocolError(
                    f"duplex output chunk sequence gap: expected={self._next_output_seq}, actual={event.output_seq}"
                )
            self._next_output_seq += 1
            return True

        completed = self._completed.get(event.output_id)
        if completed is not None:
            return False
        if self._active_output_id is None or event.output_id != self._active_output_id:
            raise DuplexEventProtocolError(f"duplex end references unknown output: output_id={event.output_id!r}")
        self._completed[event.output_id] = _CompletedOutput(
            next_output_seq=self._next_output_seq,
        )
        self._completed.move_to_end(event.output_id)
        while len(self._completed) > self._completed_output_limit:
            self._completed.popitem(last=False)
        assert self._active_output_source_input_seq is not None
        if (
            self._completed_source_input_seq is None
            or self._active_output_source_input_seq > self._completed_source_input_seq
        ):
            self._completed_source_input_seq = self._active_output_source_input_seq
        self._active_output_id = None
        self._active_output_source_input_seq = None
        self._next_output_seq = 0
        return True

    def emit_listen(self, *, source_input_seq: int, reason: str = "model_listen") -> DuplexListen:
        event = DuplexListen(
            fence=self._fence,
            source_input_seq=source_input_seq,
            reason=reason,
        )
        self.accept(event)
        return event

    def emit_start(
        self,
        *,
        source_input_seq: int,
        output_id: str | None = None,
    ) -> DuplexSpeakStart:
        event = DuplexSpeakStart(
            fence=self._fence,
            source_input_seq=source_input_seq,
            output_id=output_id or uuid4().hex,
        )
        if not self.accept(event):
            raise DuplexEventProtocolError(f"duplex output start is a duplicate: {event.output_id!r}")
        return event

    def emit_chunk(
        self,
        *,
        source_input_seq: int,
        text_delta: str = "",
        audio_data: str = "",
        audio_format: str = "wav",
        audio_duration_ms: int | None = None,
        audio_text_marks: tuple[tuple[int, int], ...] = (),
    ) -> tuple[DuplexModelEvent, ...]:
        _validate_non_negative_int("source_input_seq", source_input_seq)
        events: list[DuplexModelEvent] = []
        if self._active_output_id is None:
            if self._completed_source_input_seq is not None and source_input_seq <= self._completed_source_input_seq:
                return ()
            events.append(self.emit_start(source_input_seq=source_input_seq))
        assert self._active_output_id is not None
        chunk = DuplexSpeakChunk(
            fence=self._fence,
            output_id=self._active_output_id,
            output_seq=self._next_output_seq,
            text_delta=text_delta,
            audio_data=audio_data,
            audio_format=audio_format,
            audio_duration_ms=audio_duration_ms,
            audio_text_marks=audio_text_marks,
        )
        self.accept(chunk)
        events.append(chunk)
        return tuple(events)

    def emit_end(self, *, reason: str = "completed") -> DuplexSpeakEnd:
        if self._active_output_id is None:
            raise DuplexEventProtocolError("cannot end duplex output when no output is active")
        event = DuplexSpeakEnd(
            fence=self._fence,
            output_id=self._active_output_id,
            reason=reason,
        )
        self.accept(event)
        return event


__all__ = [
    "DuplexEventProtocolError",
    "DuplexListen",
    "DuplexModelEvent",
    "DuplexOutputLedger",
    "DuplexSpeakChunk",
    "DuplexSpeakEnd",
    "DuplexSpeakStart",
]
