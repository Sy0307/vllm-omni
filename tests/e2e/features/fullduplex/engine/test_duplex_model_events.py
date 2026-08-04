# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.engine.model_events import (
    DuplexEventProtocolError,
    DuplexListen,
    DuplexOutputLedger,
    DuplexSpeakChunk,
    DuplexSpeakEnd,
    DuplexSpeakStart,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_events_are_immutable_and_accept_valid_fields() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=2)
    listen = DuplexListen(fence=fence, source_input_seq=3)
    start = DuplexSpeakStart(
        fence=fence,
        source_input_seq=3,
        output_id="output-1",
    )
    chunk = DuplexSpeakChunk(
        fence=fence,
        output_id="output-1",
        output_seq=0,
        text_delta="hello",
        audio_data="YQ==",
        audio_format="pcm16",
        audio_duration_ms=20,
        audio_text_marks=((0, 5),),
    )
    end = DuplexSpeakEnd(fence=fence, output_id="output-1")

    assert listen.source_input_seq == start.source_input_seq == 3
    assert chunk.output_seq == 0
    assert end.reason == "completed"
    with pytest.raises(FrozenInstanceError):
        chunk.output_seq = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda f: DuplexListen(fence=f, source_input_seq=-1), "source_input_seq"),
        (lambda f: DuplexListen(fence=f, source_input_seq=True), "source_input_seq"),
        (lambda f: DuplexListen(fence=f, source_input_seq=0, reason=""), "reason"),
        (
            lambda f: DuplexSpeakStart(fence=f, source_input_seq=0, output_id=" "),
            "output_id",
        ),
        (
            lambda f: DuplexSpeakChunk(fence=f, output_id="output", output_seq=False, text_delta="x"),
            "output_seq",
        ),
        (
            lambda f: DuplexSpeakChunk(fence=f, output_id="output", output_seq=0),
            "text or audio",
        ),
        (
            lambda f: DuplexSpeakChunk(
                fence=f,
                output_id="output",
                output_seq=0,
                audio_data="YQ==",
                audio_duration_ms=-1,
            ),
            "audio_duration_ms",
        ),
        (
            lambda f: DuplexSpeakChunk(
                fence=f,
                output_id="output",
                output_seq=0,
                text_delta="x",
                audio_text_marks=((0, -1),),
            ),
            "audio_text_marks",
        ),
        (lambda f: DuplexSpeakEnd(fence=f, output_id="", reason="done"), "output_id"),
        (lambda f: DuplexSpeakEnd(fence=f, output_id="output", reason=" "), "reason"),
    ],
)
def test_model_events_reject_illegal_fields(factory, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory(DuplexFence("session"))


def test_input_can_listen_while_existing_output_remains_active() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=2)
    ledger = DuplexOutputLedger(fence)

    start, first = ledger.emit_chunk(
        source_input_seq=4,
        text_delta="hello",
        audio_data="YQ==",
        audio_format="pcm16",
    )
    listen = ledger.emit_listen(source_input_seq=5)

    assert isinstance(start, DuplexSpeakStart)
    assert isinstance(first, DuplexSpeakChunk)
    assert listen.source_input_seq == 5
    assert ledger.active_output_id == start.output_id


def test_ledger_emits_one_start_and_monotonic_chunks_then_end() -> None:
    fence = DuplexFence("session")
    ledger = DuplexOutputLedger(fence)

    start, first = ledger.emit_chunk(source_input_seq=7, text_delta="a")
    (second,) = ledger.emit_chunk(source_input_seq=8, text_delta="b")
    end = ledger.emit_end()

    assert isinstance(start, DuplexSpeakStart)
    assert first.output_id == second.output_id == end.output_id == start.output_id
    assert (first.output_seq, second.output_seq) == (0, 1)
    assert ledger.active_output_id is None


def test_ledger_drops_completed_input_replays_before_starting_newer_output() -> None:
    fence = DuplexFence("session")
    ledger = DuplexOutputLedger(fence)

    first_start, first_chunk = ledger.emit_chunk(source_input_seq=7, text_delta="first")
    first_end = ledger.emit_end()

    assert first_chunk.output_id == first_end.output_id == first_start.output_id
    assert (
        ledger.accept(
            DuplexSpeakStart(
                fence=fence,
                source_input_seq=7,
                output_id="duplicate-start",
            )
        )
        is False
    )
    assert ledger.emit_chunk(source_input_seq=7, text_delta="duplicate") == ()
    assert ledger.emit_chunk(source_input_seq=6, text_delta="older") == ()

    next_start, next_chunk = ledger.emit_chunk(source_input_seq=8, text_delta="next")
    (continued_chunk,) = ledger.emit_chunk(source_input_seq=9, text_delta="continued")

    assert next_start.output_id != first_start.output_id
    assert next_chunk.output_id == continued_chunk.output_id == next_start.output_id
    assert (next_chunk.output_seq, continued_chunk.output_seq) == (0, 1)


def test_ledger_ignores_exact_duplicate_events() -> None:
    fence = DuplexFence("session")
    ledger = DuplexOutputLedger(fence)
    start = DuplexSpeakStart(fence=fence, source_input_seq=1, output_id="output")
    chunk = DuplexSpeakChunk(
        fence=fence,
        output_id="output",
        output_seq=0,
        text_delta="a",
    )
    end = DuplexSpeakEnd(fence=fence, output_id="output")

    assert ledger.accept(start) is True
    assert ledger.accept(start) is False
    assert ledger.accept(chunk) is True
    assert ledger.accept(chunk) is False
    assert ledger.accept(end) is True
    assert ledger.accept(end) is False


def test_ledger_rejects_gap_unknown_output_and_chunk_after_end() -> None:
    fence = DuplexFence("session")
    ledger = DuplexOutputLedger(fence)
    ledger.accept(DuplexSpeakStart(fence=fence, source_input_seq=1, output_id="output"))

    with pytest.raises(DuplexEventProtocolError, match="gap"):
        ledger.accept(
            DuplexSpeakChunk(
                fence=fence,
                output_id="output",
                output_seq=1,
                text_delta="future",
            )
        )
    with pytest.raises(DuplexEventProtocolError, match="unknown output"):
        DuplexOutputLedger(fence).accept(
            DuplexSpeakChunk(
                fence=fence,
                output_id="missing",
                output_seq=0,
                text_delta="x",
            )
        )

    ledger.accept(
        DuplexSpeakChunk(
            fence=fence,
            output_id="output",
            output_seq=0,
            text_delta="last",
        )
    )
    ledger.accept(DuplexSpeakEnd(fence=fence, output_id="output"))
    with pytest.raises(DuplexEventProtocolError, match="after end"):
        ledger.accept(
            DuplexSpeakChunk(
                fence=fence,
                output_id="output",
                output_seq=1,
                text_delta="late",
            )
        )


def test_ledger_rejects_second_live_output() -> None:
    fence = DuplexFence("session")
    ledger = DuplexOutputLedger(fence)
    ledger.accept(DuplexSpeakStart(fence=fence, source_input_seq=1, output_id="first"))

    with pytest.raises(DuplexEventProtocolError, match="already active"):
        ledger.accept(DuplexSpeakStart(fence=fence, source_input_seq=2, output_id="second"))


def test_new_epoch_discards_active_and_completed_output_state() -> None:
    old_fence = DuplexFence("session", incarnation=4, epoch=2)
    new_fence = DuplexFence("session", incarnation=4, epoch=3)
    ledger = DuplexOutputLedger(old_fence)
    start, _ = ledger.emit_chunk(source_input_seq=1, text_delta="old")
    ledger.emit_end()

    ledger.advance_epoch(new_fence)
    new_start, new_chunk = ledger.emit_chunk(source_input_seq=0, text_delta="new")

    assert new_start.fence == new_fence
    assert new_chunk.output_seq == 0
    assert new_start.output_id != start.output_id


def test_old_epoch_event_is_dropped_and_future_epoch_resets_ledger() -> None:
    current_fence = DuplexFence("session", incarnation=1, epoch=2)
    old_fence = DuplexFence("session", incarnation=1, epoch=1)
    future_fence = DuplexFence("session", incarnation=1, epoch=3)
    ledger = DuplexOutputLedger(current_fence)

    assert ledger.accept(DuplexListen(fence=old_fence, source_input_seq=1)) is False
    assert ledger.accept(DuplexListen(fence=future_fence, source_input_seq=0)) is True
    assert ledger.fence == future_fence


def test_event_from_another_session_or_incarnation_is_rejected() -> None:
    ledger = DuplexOutputLedger(DuplexFence("session", incarnation=1))

    with pytest.raises(DuplexEventProtocolError, match="fence"):
        ledger.accept(DuplexListen(fence=DuplexFence("other", incarnation=1), source_input_seq=0))
    with pytest.raises(DuplexEventProtocolError, match="fence"):
        ledger.accept(DuplexListen(fence=DuplexFence("session", incarnation=2), source_input_seq=0))
