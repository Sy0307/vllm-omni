# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.core.playback import PlaybackCursor, PlaybackCursorError


def test_fence_advances_without_mutating_prior_identity():
    fence = DuplexFence("session")

    next_turn = fence.next_turn()
    next_epoch = next_turn.next_epoch()

    assert fence == DuplexFence("session")
    assert next_turn == DuplexFence("session", epoch=0, turn_id=1, response_seq=1)
    assert next_epoch == DuplexFence("session", epoch=1, turn_id=1, response_seq=1)


def test_fence_is_frozen_and_slotted():
    fence = DuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")


def test_playback_cursor_tracks_all_positions_immutably():
    cursor = PlaybackCursor()

    generated = cursor.mark_generated(140)
    sent = generated.mark_sent(120)
    acknowledged = sent.acknowledge(played=80, committed=60)

    assert cursor == PlaybackCursor()
    assert generated == PlaybackCursor(generated=140)
    assert sent == PlaybackCursor(generated=140, sent=120)
    assert acknowledged == PlaybackCursor(
        generated=140,
        sent=120,
        played=80,
        committed=60,
    )


@pytest.mark.parametrize(
    ("played", "committed"),
    [
        (121, None),
        (80, 81),
        (-1, None),
    ],
)
def test_playback_cursor_rejects_impossible_acknowledgements(played, committed):
    cursor = PlaybackCursor(generated=140, sent=120)

    with pytest.raises(PlaybackCursorError):
        cursor.acknowledge(played=played, committed=committed)


@pytest.mark.parametrize(
    ("operation", "position"),
    [
        ("generated", 99),
        ("sent", 81),
        ("sent", 121),
    ],
)
def test_playback_cursor_rejects_non_monotonic_output_positions(operation, position):
    cursor = PlaybackCursor(generated=120, sent=100)

    with pytest.raises(PlaybackCursorError):
        if operation == "generated":
            cursor.mark_generated(position)
        else:
            cursor.mark_sent(position)


@pytest.mark.parametrize(
    "positions",
    [
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
        (10, 11, 0, 0),
        (10, 9, 10, 0),
        (10, 9, 8, 9),
    ],
)
def test_playback_cursor_rejects_invalid_direct_construction(positions):
    with pytest.raises(PlaybackCursorError):
        PlaybackCursor(*positions)
