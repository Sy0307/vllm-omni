# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
    MiniCPMO45Stage1State,
    MiniCPMO45StageFenceMismatchError,
)


def test_stage1_transient_state_is_owned_by_complete_fence():
    fence = DuplexFence("s", epoch=1, turn_id=2, response_seq=3)
    state = MiniCPMO45Stage1State(fence=fence)
    state.consumed_tokens = 7
    state.token2wav_buffer.extend([1, 2, 3])

    state.require(fence)

    assert state.consumed_tokens == 7
    assert state.token2wav_buffer == [1, 2, 3]


def test_stage1_rejects_reusing_transient_state_for_next_turn():
    old = DuplexFence("s", epoch=1, turn_id=2, response_seq=3)
    state = MiniCPMO45Stage1State(fence=old)

    with pytest.raises(MiniCPMO45StageFenceMismatchError):
        state.require(old.next_turn())


def test_stage1_reset_clears_transient_state_without_touching_stage0_context():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    state = MiniCPMO45Stage1State(
        fence=fence,
        consumed_tokens=9,
        token2wav_buffer=[4, 5],
        audio_offset=120,
    )

    reset = state.reset(fence.next_turn())

    assert reset.fence == fence.next_turn()
    assert reset.consumed_tokens == 0
    assert reset.token2wav_buffer == []
    assert reset.audio_offset == 0
