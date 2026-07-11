# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.experimental.fullduplex.core.events import EmitProtocolEvent, ProtocolEventKind
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.realtime import RealtimeEventProjector


def test_realtime_projection_uses_effect_fence_without_mutating_identity():
    fence = DuplexFence("s", epoch=2, turn_id=3, response_seq=4)
    projector = RealtimeEventProjector()

    events = projector.project(
        EmitProtocolEvent(
            fence=fence,
            kind=ProtocolEventKind.RESPONSE_STARTED,
        )
    )

    assert events == (
        {
            "type": "response.created",
            "response": {
                "id": "resp-s-e2-t3-r4",
                "status": "in_progress",
            },
        },
    )
    assert fence == DuplexFence("s", epoch=2, turn_id=3, response_seq=4)


def test_realtime_projection_rejects_late_audio_via_shared_lifecycle():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    projector = RealtimeEventProjector()
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_COMPLETED))

    try:
        projector.project(EmitProtocolEvent(fence, ProtocolEventKind.AUDIO_DELTA, payload=b"late"))
    except Exception as exc:
        assert type(exc).__name__ == "LateResponseOutputError"
    else:
        raise AssertionError("late audio must fail")
