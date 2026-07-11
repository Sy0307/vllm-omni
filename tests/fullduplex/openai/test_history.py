# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.history import (
    DuplicateResponseLifecycleError,
    LateResponseOutputError,
    ResponseLifecycleLedger,
)


def test_response_lifecycle_is_exactly_once_per_fence():
    ledger = ResponseLifecycleLedger()
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    response_id = ledger.start(fence)
    ledger.append_text(fence, "hello")
    transcript = ledger.finish(fence)

    assert response_id.endswith("-e0-t1-r1")
    assert transcript == "hello"
    with pytest.raises(DuplicateResponseLifecycleError):
        ledger.start(fence)
    with pytest.raises(DuplicateResponseLifecycleError):
        ledger.finish(fence)


def test_audio_after_terminal_is_rejected():
    ledger = ResponseLifecycleLedger()
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    ledger.start(fence)
    ledger.finish(fence)

    with pytest.raises(LateResponseOutputError):
        ledger.append_audio(fence, b"late")


def test_distinct_fences_never_share_transcript_state():
    ledger = ResponseLifecycleLedger()
    first = DuplexFence("s", turn_id=1, response_seq=1)
    second = first.next_turn()
    ledger.start(first)
    ledger.append_text(first, "first tail")
    ledger.finish(first)

    ledger.start(second)
    ledger.append_text(second, "second")

    assert ledger.transcript(second) == "second"
