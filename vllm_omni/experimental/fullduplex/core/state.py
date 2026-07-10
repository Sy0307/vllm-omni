# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .events import (
    AppendToEngine,
    CancelFence,
    CloseSessionResources,
    CommittedHistoryItem,
    DomainEvent,
    DuplexEffect,
    EmitProtocolEvent,
    EngineAppendAccepted,
    EngineFailed,
    HistoryCommitted,
    InputChunk,
    InputCommitted,
    InputStarted,
    InterruptRequested,
    ModelAudioDelta,
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
    PlaybackAcknowledged,
    ProtocolEventKind,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
    SessionCloseRequested,
)
from .identity import DuplexFence
from .playback import PlaybackCursor, PlaybackCursorError


class DuplexSessionPhase(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class DuplexTurnPhase(str, Enum):
    IDLE = "idle"
    INPUT_STREAMING = "input_streaming"
    TURN_COMMITTED = "turn_committed"
    AWAITING_MODEL = "awaiting_model"
    RESPONDING = "responding"


@dataclass(frozen=True, slots=True)
class ActiveResponse:
    fence: DuplexFence
    started: bool = False
    segments_ended: int = 0


@dataclass(frozen=True, slots=True)
class DuplexSessionState:
    fence: DuplexFence
    session_phase: DuplexSessionPhase = DuplexSessionPhase.OPEN
    turn_phase: DuplexTurnPhase = DuplexTurnPhase.IDLE
    pending_input_chunks: int = 0
    active_response: ActiveResponse | None = None
    playback: PlaybackCursor = PlaybackCursor()
    last_committed_playback_position: int = 0
    committed_history: tuple[CommittedHistoryItem, ...] = ()
    terminal_reason: str | None = None
    capabilities: frozenset[str] = frozenset()
    last_terminal_fence: DuplexFence | None = None
    stale_event_count: int = 0
    duplicate_terminal_count: int = 0

    @classmethod
    def open(
        cls,
        session_id: str,
        *,
        capabilities: frozenset[str] = frozenset(),
    ) -> DuplexSessionState:
        return cls(
            fence=DuplexFence(session_id),
            capabilities=capabilities,
        )


class DuplexReducerError(RuntimeError):
    pass


class DuplexTransitionError(DuplexReducerError):
    def __init__(self, state: DuplexSessionState, event: DomainEvent, detail: str):
        super().__init__(
            f"illegal {type(event).__name__} transition from "
            f"session={state.session_phase.value}, turn={state.turn_phase.value}: {detail}"
        )
        self.state = state
        self.event = event


class DuplexFenceError(DuplexReducerError):
    pass


class DuplexMissingFenceError(DuplexFenceError):
    def __init__(self, event: DomainEvent):
        super().__init__(f"{type(event).__name__} is missing a DuplexFence")
        self.event = event


class DuplexFenceMismatchError(DuplexFenceError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence):
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


_FencedEvent = (
    EngineAppendAccepted
    | HistoryCommitted
    | InterruptRequested
    | ModelListening
    | ModelSpeaking
    | ModelTextDelta
    | ModelAudioDelta
    | ModelSegmentEnded
    | ModelTurnEnded
    | PlaybackAcknowledged
    | EngineFailed
)


def _is_stale(expected: DuplexFence, actual: DuplexFence) -> bool:
    if actual.epoch < expected.epoch:
        return True
    if actual.epoch > expected.epoch:
        return False
    no_newer_component = (
        actual.turn_id <= expected.turn_id
        and actual.response_seq <= expected.response_seq
    )
    return no_newer_component and actual != expected


def _validate_fence(
    state: DuplexSessionState,
    event: _FencedEvent,
) -> bool:
    fence = event.fence
    if fence is None:
        raise DuplexMissingFenceError(event)
    if fence.session_id != state.fence.session_id:
        raise DuplexFenceMismatchError(state.fence, fence)
    if fence == state.fence:
        return True
    if _is_stale(state.fence, fence):
        return False
    raise DuplexFenceMismatchError(state.fence, fence)


def _transition_error(
    state: DuplexSessionState,
    event: DomainEvent,
    detail: str,
) -> DuplexTransitionError:
    return DuplexTransitionError(state, event, detail)


def _require_active(
    state: DuplexSessionState,
    event: DomainEvent,
) -> ActiveResponse:
    active = state.active_response
    if active is None:
        raise _transition_error(state, event, "there is no active response")
    if active.fence != state.fence:
        raise DuplexFenceMismatchError(state.fence, active.fence)
    return active


def _close_state(
    state: DuplexSessionState,
    *,
    reason: str,
) -> DuplexSessionState:
    return replace(
        state,
        session_phase=DuplexSessionPhase.CLOSED,
        turn_phase=DuplexTurnPhase.IDLE,
        pending_input_chunks=0,
        active_response=None,
        terminal_reason=reason,
    )


def _close_effects(state: DuplexSessionState) -> tuple[DuplexEffect, ...]:
    effects: list[DuplexEffect] = []
    if state.active_response is not None:
        effects.extend((CancelFence(state.fence), ResetStage1(state.fence)))
    effects.append(CloseSessionResources(state.fence))
    return tuple(effects)


def reduce_event(
    state: DuplexSessionState,
    event: DomainEvent,
) -> tuple[DuplexSessionState, tuple[DuplexEffect, ...]]:
    if isinstance(event, SessionCloseRequested):
        if state.session_phase is DuplexSessionPhase.CLOSED:
            return replace(
                state,
                duplicate_terminal_count=state.duplicate_terminal_count + 1,
            ), ()
        return _close_state(state, reason=event.reason), _close_effects(state)

    if isinstance(event, EngineFailed):
        if not _validate_fence(state, event):
            return replace(
                state,
                stale_event_count=state.stale_event_count + 1,
            ), ()
        if state.session_phase is DuplexSessionPhase.CLOSED:
            return replace(
                state,
                duplicate_terminal_count=state.duplicate_terminal_count + 1,
            ), ()
        effects = (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.ENGINE_FAILED,
                payload=event.message,
            ),
            *_close_effects(state),
        )
        return _close_state(state, reason=event.message), effects

    if state.session_phase is not DuplexSessionPhase.OPEN:
        raise _transition_error(state, event, "session is not open")

    if isinstance(event, InterruptRequested):
        if not _validate_fence(state, event):
            return replace(
                state,
                stale_event_count=state.stale_event_count + 1,
            ), ()
        old_fence = state.fence
        new_fence = old_fence.next_epoch()
        committed_playback_position = max(
            state.last_committed_playback_position,
            state.playback.committed,
        )
        effects: tuple[DuplexEffect, ...] = (
            CancelFence(old_fence),
            ResetStage1(old_fence),
            RebuildStage0Context(
                new_fence,
                committed_history=state.committed_history,
                committed_playback_position=committed_playback_position,
            ),
        )
        return (
            replace(
                state,
                fence=new_fence,
                turn_phase=DuplexTurnPhase.IDLE,
                pending_input_chunks=0,
                active_response=None,
                playback=PlaybackCursor(),
                last_committed_playback_position=committed_playback_position,
                last_terminal_fence=None,
            ),
            effects,
        )

    if isinstance(
        event,
        (
            ModelListening,
            EngineAppendAccepted,
            HistoryCommitted,
            ModelSpeaking,
            ModelTextDelta,
            ModelAudioDelta,
            ModelSegmentEnded,
            ModelTurnEnded,
            PlaybackAcknowledged,
        ),
    ):
        if not _validate_fence(state, event):
            return replace(
                state,
                stale_event_count=state.stale_event_count + 1,
            ), ()

    if isinstance(event, InputStarted):
        if state.turn_phase is DuplexTurnPhase.IDLE:
            return replace(state, turn_phase=DuplexTurnPhase.INPUT_STREAMING), ()
        if state.turn_phase is DuplexTurnPhase.INPUT_STREAMING:
            return state, ()
        raise _transition_error(state, event, "input cannot start in this turn phase")

    if isinstance(event, InputChunk):
        if state.turn_phase not in (
            DuplexTurnPhase.IDLE,
            DuplexTurnPhase.INPUT_STREAMING,
        ):
            raise _transition_error(state, event, "input cannot be appended in this turn phase")
        next_state = replace(
            state,
            turn_phase=DuplexTurnPhase.INPUT_STREAMING,
            pending_input_chunks=state.pending_input_chunks + 1,
        )
        return next_state, (AppendToEngine(state.fence, chunk=event),)

    if isinstance(event, InputCommitted):
        if state.turn_phase not in (
            DuplexTurnPhase.IDLE,
            DuplexTurnPhase.INPUT_STREAMING,
        ):
            raise _transition_error(state, event, "input cannot be committed in this turn phase")
        fence = state.fence.next_turn()
        next_state = replace(
            state,
            fence=fence,
            turn_phase=DuplexTurnPhase.TURN_COMMITTED,
            pending_input_chunks=0,
            active_response=ActiveResponse(fence),
            playback=PlaybackCursor(),
            last_committed_playback_position=0,
        )
        return next_state, (
            ReserveResponse(fence),
            AppendToEngine(fence, final=True),
        )

    if isinstance(event, EngineAppendAccepted):
        _require_active(state, event)
        if state.turn_phase is not DuplexTurnPhase.TURN_COMMITTED:
            raise _transition_error(
                state,
                event,
                "engine append can only be accepted for a committed turn",
            )
        return replace(state, turn_phase=DuplexTurnPhase.AWAITING_MODEL), ()

    if isinstance(event, HistoryCommitted):
        _require_active(state, event)
        if state.turn_phase not in (
            DuplexTurnPhase.TURN_COMMITTED,
            DuplexTurnPhase.AWAITING_MODEL,
            DuplexTurnPhase.RESPONDING,
        ):
            raise _transition_error(state, event, "history cannot be committed in this phase")
        return replace(
            state,
            committed_history=(*state.committed_history, event.item),
        ), ()

    if isinstance(event, ModelListening):
        _require_active(state, event)
        if state.turn_phase not in (
            DuplexTurnPhase.AWAITING_MODEL,
            DuplexTurnPhase.RESPONDING,
        ):
            raise _transition_error(state, event, "model cannot listen in this turn phase")
        return state, (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.MODEL_LISTENING,
                payload=event,
            ),
        )

    if isinstance(event, ModelSpeaking):
        active = _require_active(state, event)
        if state.turn_phase not in (
            DuplexTurnPhase.AWAITING_MODEL,
            DuplexTurnPhase.RESPONDING,
        ):
            raise _transition_error(state, event, "model cannot speak in this turn phase")
        if active.started:
            return state, ()
        next_state = replace(
            state,
            turn_phase=DuplexTurnPhase.RESPONDING,
            active_response=replace(active, started=True),
        )
        return next_state, (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.RESPONSE_STARTED,
                payload=event,
            ),
        )

    if isinstance(event, ModelTextDelta):
        active = _require_active(state, event)
        if state.turn_phase is not DuplexTurnPhase.RESPONDING or not active.started:
            raise _transition_error(state, event, "text arrived before the response started")
        return state, (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.TEXT_DELTA,
                payload=event,
            ),
        )

    if isinstance(event, ModelAudioDelta):
        active = _require_active(state, event)
        if state.turn_phase is not DuplexTurnPhase.RESPONDING or not active.started:
            raise _transition_error(state, event, "audio arrived before the response started")
        try:
            playback = state.playback.mark_generated(event.generated_cursor)
            playback = playback.mark_sent(event.sent_cursor)
        except PlaybackCursorError as exc:
            raise _transition_error(state, event, str(exc)) from exc
        return replace(state, playback=playback), (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.AUDIO_DELTA,
                payload=event,
            ),
        )

    if isinstance(event, ModelSegmentEnded):
        active = _require_active(state, event)
        if state.turn_phase is not DuplexTurnPhase.RESPONDING or not active.started:
            raise _transition_error(state, event, "segment ended before the response started")
        next_state = replace(
            state,
            active_response=replace(
                active,
                segments_ended=active.segments_ended + 1,
            ),
        )
        return next_state, (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.SEGMENT_ENDED,
                payload=event,
            ),
        )

    if isinstance(event, ModelTurnEnded):
        if (
            state.active_response is None
            and state.last_terminal_fence == state.fence
            and state.turn_phase is DuplexTurnPhase.IDLE
        ):
            return replace(
                state,
                duplicate_terminal_count=state.duplicate_terminal_count + 1,
            ), ()
        _require_active(state, event)
        if state.turn_phase not in (
            DuplexTurnPhase.AWAITING_MODEL,
            DuplexTurnPhase.RESPONDING,
        ):
            raise _transition_error(state, event, "turn cannot end in this phase")
        next_state = replace(
            state,
            turn_phase=DuplexTurnPhase.IDLE,
            pending_input_chunks=0,
            active_response=None,
            last_terminal_fence=state.fence,
        )
        return next_state, (
            EmitProtocolEvent(
                state.fence,
                ProtocolEventKind.RESPONSE_COMPLETED,
                payload=event,
            ),
            ResetStage1(state.fence),
        )

    if isinstance(event, PlaybackAcknowledged):
        active_playback = state.turn_phase in (
            DuplexTurnPhase.AWAITING_MODEL,
            DuplexTurnPhase.RESPONDING,
        )
        terminal_playback = (
            state.turn_phase is DuplexTurnPhase.IDLE
            and state.active_response is None
            and state.last_terminal_fence == state.fence
        )
        if not active_playback and not terminal_playback:
            raise _transition_error(
                state,
                event,
                "playback cannot be acknowledged in this turn phase",
            )
        if (
            state.active_response is None
            and state.last_terminal_fence != state.fence
        ):
            raise _transition_error(state, event, "there is no response playback to acknowledge")
        try:
            playback = state.playback.acknowledge(
                played=event.cursor,
                committed=event.committed_cursor,
            )
        except PlaybackCursorError as exc:
            raise _transition_error(state, event, str(exc)) from exc
        return replace(
            state,
            playback=playback,
            last_committed_playback_position=max(
                state.last_committed_playback_position,
                playback.committed,
            ),
        ), ()

    raise TypeError(f"unsupported duplex event: {type(event).__name__}")
