# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    CancelFence,
    CloseSessionResources,
    CommittedHistoryItem,
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
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.core.playback import PlaybackCursor
from vllm_omni.experimental.fullduplex.core.state import (
    DuplexFenceMismatchError,
    DuplexMissingFenceError,
    DuplexSessionPhase,
    DuplexSessionState,
    DuplexTransitionError,
    DuplexTurnPhase,
    reduce_event,
)


def committed_state(session_id: str = "s") -> DuplexSessionState:
    state, _ = reduce_event(DuplexSessionState.open(session_id), InputCommitted())
    return state


def awaiting_state(session_id: str = "s") -> DuplexSessionState:
    state = committed_state(session_id)
    state, _ = reduce_event(state, EngineAppendAccepted(fence=state.fence))
    return state


def responding_state(session_id: str = "s") -> DuplexSessionState:
    state = awaiting_state(session_id)
    state, _ = reduce_event(state, ModelSpeaking(fence=state.fence))
    return state


def state_for_phase(phase: DuplexTurnPhase) -> DuplexSessionState:
    state = DuplexSessionState.open("matrix")
    if phase is DuplexTurnPhase.IDLE:
        return state
    if phase is DuplexTurnPhase.INPUT_STREAMING:
        return reduce_event(state, InputStarted())[0]
    state = reduce_event(state, InputCommitted())[0]
    if phase is DuplexTurnPhase.TURN_COMMITTED:
        return state
    state = reduce_event(state, EngineAppendAccepted(fence=state.fence))[0]
    if phase is DuplexTurnPhase.AWAITING_MODEL:
        return state
    return reduce_event(state, ModelSpeaking(fence=state.fence))[0]


def test_fence_advances_turn_without_advancing_epoch():
    state = DuplexSessionState.open("s")

    new_state, effects = reduce_event(state, InputCommitted())

    assert new_state.fence == DuplexFence("s", epoch=0, turn_id=1, response_seq=1)
    assert state.fence == DuplexFence("s")
    assert new_state.turn_phase is DuplexTurnPhase.TURN_COMMITTED
    assert [type(effect) for effect in effects] == [ReserveResponse, AppendToEngine]
    assert all(effect.fence == new_state.fence for effect in effects)
    assert effects[-1].final is True


def test_append_acknowledgement_gates_model_output():
    state = committed_state()

    with pytest.raises(DuplexTransitionError):
        reduce_event(state, ModelSpeaking(fence=state.fence))

    acknowledged, effects = reduce_event(state, EngineAppendAccepted(fence=state.fence))
    assert acknowledged.turn_phase is DuplexTurnPhase.AWAITING_MODEL
    assert effects == ()


def test_input_streaming_emits_fenced_incremental_append():
    state = DuplexSessionState.open("s")
    started, _ = reduce_event(state, InputStarted(modality="audio"))
    streamed, effects = reduce_event(started, InputChunk(data=b"pcm", modality="audio"))

    assert effects == (AppendToEngine(state.fence, chunk=InputChunk(data=b"pcm", modality="audio")),)
    assert streamed.turn_phase is DuplexTurnPhase.INPUT_STREAMING
    assert streamed.pending_input_chunks == 1


def test_empty_turn_is_legal_after_append_acknowledgement():
    state = awaiting_state()

    ended, effects = reduce_event(state, ModelTurnEnded(fence=state.fence))

    assert ended.active_response is None
    assert ended.turn_phase is DuplexTurnPhase.IDLE
    assert ended.playback == PlaybackCursor()
    assert [type(effect) for effect in effects] == [EmitProtocolEvent, ResetStage1]


def test_model_segment_end_does_not_finish_response_but_turn_end_does():
    state = responding_state()

    segmented, segment_effects = reduce_event(state, ModelSegmentEnded(fence=state.fence))
    ended, turn_effects = reduce_event(segmented, ModelTurnEnded(fence=state.fence))

    assert segmented.active_response is not None
    assert segmented.active_response.segments_ended == 1
    assert segmented.turn_phase is DuplexTurnPhase.RESPONDING
    assert segment_effects[0].kind is ProtocolEventKind.SEGMENT_ENDED
    assert ended.active_response is None
    assert ended.turn_phase is DuplexTurnPhase.IDLE
    assert turn_effects[-1] == ResetStage1(state.fence)


def test_clean_turn_preserves_epoch_and_advances_identity_once():
    first = responding_state()
    ended, _ = reduce_event(first, ModelTurnEnded(fence=first.fence))

    second, effects = reduce_event(ended, InputCommitted())

    assert second.fence == DuplexFence("s", epoch=0, turn_id=2, response_seq=2)
    assert second.turn_phase is DuplexTurnPhase.TURN_COMMITTED
    assert [type(effect) for effect in effects] == [ReserveResponse, AppendToEngine]


def test_duplicate_turn_terminal_is_idempotent_and_counted():
    state = responding_state()
    ended, _ = reduce_event(state, ModelTurnEnded(fence=state.fence))

    duplicate, effects = reduce_event(ended, ModelTurnEnded(fence=state.fence))

    assert effects == ()
    assert duplicate.duplicate_terminal_count == ended.duplicate_terminal_count + 1
    assert duplicate.fence == ended.fence


def test_interruption_rebuilds_from_committed_history_and_playback():
    state = responding_state()
    old = state.fence
    item = CommittedHistoryItem(role="assistant", modality="text", content="hello")
    state, _ = reduce_event(
        state,
        ModelAudioDelta(
            data=b"pcm",
            generated_cursor=140,
            sent_cursor=120,
            fence=old,
        ),
    )
    state, _ = reduce_event(
        state,
        PlaybackAcknowledged(cursor=80, committed_cursor=60, fence=old),
    )
    state, _ = reduce_event(state, HistoryCommitted(item=item, fence=old))

    new_state, effects = reduce_event(
        state,
        InterruptRequested(reason="test", fence=old),
    )

    assert new_state.fence == old.next_epoch()
    assert new_state.active_response is None
    assert new_state.turn_phase is DuplexTurnPhase.IDLE
    assert new_state.committed_history == (item,)
    assert new_state.last_committed_playback_position == 60
    assert effects == (
        CancelFence(old),
        ResetStage1(old),
        RebuildStage0Context(
            new_state.fence,
            committed_history=(item,),
            committed_playback_position=60,
        ),
    )

    repeated, repeated_effects = reduce_event(
        new_state,
        InterruptRequested(reason="repeat", fence=new_state.fence),
    )
    assert repeated.last_committed_playback_position == 60
    assert repeated_effects[-1] == RebuildStage0Context(
        repeated.fence,
        committed_history=(item,),
        committed_playback_position=60,
    )

    stale, stale_effects = reduce_event(
        repeated,
        InterruptRequested(reason="late", fence=old),
    )
    assert stale.fence == repeated.fence
    assert stale.stale_event_count == repeated.stale_event_count + 1
    assert stale_effects == ()


def test_new_turn_resets_playback_checkpoint_before_interruption():
    first = responding_state("two-turn")
    first, _ = reduce_event(
        first,
        ModelAudioDelta(
            data=b"first",
            generated_cursor=140,
            sent_cursor=120,
            fence=first.fence,
        ),
    )
    first, _ = reduce_event(
        first,
        PlaybackAcknowledged(
            cursor=80,
            committed_cursor=60,
            fence=first.fence,
        ),
    )
    first, _ = reduce_event(first, ModelTurnEnded(fence=first.fence))

    second, _ = reduce_event(first, InputCommitted())
    assert second.last_committed_playback_position == 0
    second, _ = reduce_event(second, EngineAppendAccepted(fence=second.fence))
    second, _ = reduce_event(second, ModelSpeaking(fence=second.fence))
    second, _ = reduce_event(
        second,
        ModelAudioDelta(
            data=b"second",
            generated_cursor=40,
            sent_cursor=30,
            fence=second.fence,
        ),
    )
    second, _ = reduce_event(
        second,
        PlaybackAcknowledged(
            cursor=20,
            committed_cursor=10,
            fence=second.fence,
        ),
    )

    interrupted, effects = reduce_event(
        second,
        InterruptRequested(reason="turn-two", fence=second.fence),
    )
    assert effects[-1].committed_playback_position == 10

    repeated, repeated_effects = reduce_event(
        interrupted,
        InterruptRequested(reason="repeat", fence=interrupted.fence),
    )
    assert repeated.last_committed_playback_position == 10
    assert repeated_effects[-1].committed_playback_position == 10


def test_stale_output_is_classified_and_dropped():
    state = responding_state()
    old = state.fence
    interrupted, _ = reduce_event(
        state,
        InterruptRequested(reason="test", fence=old),
    )

    classified, effects = reduce_event(interrupted, ModelTextDelta(text="late", fence=old))

    assert effects == ()
    assert classified.stale_event_count == interrupted.stale_event_count + 1
    assert classified.fence == interrupted.fence


def test_missing_and_mismatched_fences_fail_loudly():
    state = responding_state()

    with pytest.raises(DuplexMissingFenceError):
        reduce_event(state, ModelTextDelta(text="missing"))
    with pytest.raises(DuplexMissingFenceError):
        reduce_event(state, EngineFailed(message="missing"))
    with pytest.raises(DuplexMissingFenceError):
        reduce_event(state, InterruptRequested(reason="missing"))

    with pytest.raises(DuplexFenceMismatchError):
        reduce_event(state, ModelTextDelta(text="wrong", fence=DuplexFence("other", 0, 1, 1)))

    future = DuplexFence("s", epoch=1, turn_id=state.fence.turn_id, response_seq=state.fence.response_seq)
    with pytest.raises(DuplexFenceMismatchError):
        reduce_event(state, ModelTextDelta(text="future", fence=future))
    with pytest.raises(DuplexFenceMismatchError):
        reduce_event(state, InterruptRequested(reason="future", fence=future))


def test_model_output_updates_distinct_playback_positions():
    state = responding_state()
    text, text_effects = reduce_event(state, ModelTextDelta(text="hello", fence=state.fence))
    audio, audio_effects = reduce_event(
        text,
        ModelAudioDelta(
            data=b"pcm",
            generated_cursor=140,
            sent_cursor=120,
            fence=state.fence,
        ),
    )
    acknowledged, ack_effects = reduce_event(
        audio,
        PlaybackAcknowledged(cursor=80, committed_cursor=60, fence=state.fence),
    )

    assert text_effects[0].kind is ProtocolEventKind.TEXT_DELTA
    assert audio_effects[0].kind is ProtocolEventKind.AUDIO_DELTA
    assert all(effect.fence == state.fence for effect in (*text_effects, *audio_effects))
    assert acknowledged.playback == PlaybackCursor(
        generated=140,
        sent=120,
        played=80,
        committed=60,
    )
    assert ack_effects == ()


def terminal_idle_state() -> DuplexSessionState:
    state = responding_state("terminal")
    item = CommittedHistoryItem("assistant", "text", "completed response")
    state, _ = reduce_event(state, HistoryCommitted(item=item, fence=state.fence))
    state, _ = reduce_event(
        state,
        ModelAudioDelta(
            data=b"pcm",
            generated_cursor=140,
            sent_cursor=120,
            fence=state.fence,
        ),
    )
    return reduce_event(state, ModelTurnEnded(fence=state.fence))[0]


@pytest.mark.parametrize(
    ("state_factory", "is_legal"),
    [
        (terminal_idle_state, True),
        (lambda: DuplexSessionState.open("unrelated"), False),
    ],
    ids=("terminal_idle", "unrelated_idle"),
)
def test_idle_playback_acknowledgement_transition_matrix(state_factory, is_legal):
    state = state_factory()
    event = PlaybackAcknowledged(
        cursor=80,
        committed_cursor=60,
        fence=state.fence,
    )

    if not is_legal:
        with pytest.raises(DuplexTransitionError):
            reduce_event(state, event)
        return

    acknowledged, effects = reduce_event(state, event)
    assert effects == ()
    assert acknowledged.playback == PlaybackCursor(
        generated=140,
        sent=120,
        played=80,
        committed=60,
    )
    assert acknowledged.committed_history == state.committed_history


ALL_PHASES = frozenset(DuplexTurnPhase)
INPUT_PHASES = frozenset((DuplexTurnPhase.IDLE, DuplexTurnPhase.INPUT_STREAMING))
MODEL_PHASES = frozenset((DuplexTurnPhase.AWAITING_MODEL, DuplexTurnPhase.RESPONDING))
HISTORY_PHASES = frozenset(
    (
        DuplexTurnPhase.TURN_COMMITTED,
        DuplexTurnPhase.AWAITING_MODEL,
        DuplexTurnPhase.RESPONDING,
    )
)


TRANSITION_CASES = (
    ("input_started", lambda state: InputStarted(), INPUT_PHASES),
    ("input_chunk", lambda state: InputChunk(data=b"pcm"), INPUT_PHASES),
    ("input_committed", lambda state: InputCommitted(), INPUT_PHASES),
    (
        "append_accepted",
        lambda state: EngineAppendAccepted(fence=state.fence),
        frozenset((DuplexTurnPhase.TURN_COMMITTED,)),
    ),
    ("model_listening", lambda state: ModelListening(fence=state.fence), MODEL_PHASES),
    ("model_speaking", lambda state: ModelSpeaking(fence=state.fence), MODEL_PHASES),
    (
        "text_delta",
        lambda state: ModelTextDelta(text="delta", fence=state.fence),
        frozenset((DuplexTurnPhase.RESPONDING,)),
    ),
    (
        "audio_delta",
        lambda state: ModelAudioDelta(
            data=b"pcm",
            generated_cursor=1,
            sent_cursor=1,
            fence=state.fence,
        ),
        frozenset((DuplexTurnPhase.RESPONDING,)),
    ),
    (
        "segment_ended",
        lambda state: ModelSegmentEnded(fence=state.fence),
        frozenset((DuplexTurnPhase.RESPONDING,)),
    ),
    ("turn_ended", lambda state: ModelTurnEnded(fence=state.fence), MODEL_PHASES),
    (
        "playback_acknowledged",
        lambda state: PlaybackAcknowledged(cursor=0, fence=state.fence),
        MODEL_PHASES,
    ),
    (
        "history_committed",
        lambda state: HistoryCommitted(
            item=CommittedHistoryItem("user", "text", "hello"),
            fence=state.fence,
        ),
        HISTORY_PHASES,
    ),
    (
        "interrupt",
        lambda state: InterruptRequested(reason="matrix", fence=state.fence),
        ALL_PHASES,
    ),
    ("close", lambda state: SessionCloseRequested(reason="matrix"), ALL_PHASES),
    (
        "engine_failed",
        lambda state: EngineFailed(message="matrix", fence=state.fence),
        ALL_PHASES,
    ),
)


@pytest.mark.parametrize("phase", tuple(DuplexTurnPhase), ids=lambda phase: phase.value)
@pytest.mark.parametrize(
    ("event_name", "event_factory", "legal_phases"),
    TRANSITION_CASES,
    ids=[case[0] for case in TRANSITION_CASES],
)
def test_turn_phase_transition_matrix(phase, event_name, event_factory, legal_phases):
    del event_name
    state = state_for_phase(phase)
    event = event_factory(state)

    if phase in legal_phases:
        reduce_event(state, event)
    else:
        with pytest.raises(DuplexTransitionError):
            reduce_event(state, event)


def test_close_and_engine_failure_release_fenced_resources():
    state = responding_state()
    closed, close_effects = reduce_event(state, SessionCloseRequested(reason="client"))

    assert closed.session_phase is DuplexSessionPhase.CLOSED
    assert closed.terminal_reason == "client"
    assert close_effects == (
        CancelFence(state.fence),
        ResetStage1(state.fence),
        CloseSessionResources(state.fence),
    )

    duplicate, effects = reduce_event(closed, SessionCloseRequested(reason="client"))
    assert effects == ()
    assert duplicate.duplicate_terminal_count == closed.duplicate_terminal_count + 1

    failing = responding_state("failed")
    failed, failure_effects = reduce_event(
        failing,
        EngineFailed(message="boom", fence=failing.fence),
    )
    assert failed.session_phase is DuplexSessionPhase.CLOSED
    assert failed.terminal_reason == "boom"
    assert failure_effects[-1] == CloseSessionResources(failing.fence)


def test_state_events_and_history_are_frozen():
    state = DuplexSessionState.open("s")
    event = InputCommitted()
    history = CommittedHistoryItem("user", "text", "hello")

    with pytest.raises(FrozenInstanceError):
        state.pending_input_chunks = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        event.extra = True  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        history.content = "changed"  # type: ignore[misc]
