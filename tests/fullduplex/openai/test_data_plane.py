# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.data_plane import DuplexDataPlaneCursors


def test_text_cursor_is_scoped_to_complete_fence():
    cursors = DuplexDataPlaneCursors()
    first = DuplexFence("s", epoch=0, turn_id=1, response_seq=1)
    second = first.next_turn()

    assert cursors.text_delta(first, "hello") == "hello"
    assert cursors.text_delta(first, "hello world") == " world"
    assert cursors.text_delta(second, "new") == "new"


def test_audio_cursor_is_scoped_to_epoch_as_well_as_turn():
    cursors = DuplexDataPlaneCursors()
    old = DuplexFence("s", epoch=0, turn_id=1, response_seq=1)
    interrupted = old.next_epoch()

    assert cursors.audio_delta(old, [1, 2, 3]) == [1, 2, 3]
    assert cursors.audio_delta(old, [1, 2, 3, 4]) == [4]
    assert cursors.audio_delta(interrupted, [9]) == [9]


def test_release_removes_all_cursor_state_for_fence():
    cursors = DuplexDataPlaneCursors()
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    cursors.text_delta(fence, "old")
    cursors.audio_delta(fence, [1, 2])

    cursors.release(fence)

    assert cursors.text_delta(fence, "fresh") == "fresh"
    assert cursors.audio_delta(fence, [7]) == [7]
