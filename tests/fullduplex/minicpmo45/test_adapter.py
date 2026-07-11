# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.experimental.fullduplex.core.events import (
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45ModelEventAdapter


def test_listen_output_maps_to_model_listening_with_same_fence():
    fence = DuplexFence("s", epoch=2, turn_id=3, response_seq=4)

    events = MiniCPMO45ModelEventAdapter().map_output({"is_listen": True}, fence)

    assert events == (ModelListening(fence=fence),)


def test_speaking_segment_emits_speaking_text_and_segment_end_only():
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    events = MiniCPMO45ModelEventAdapter().map_output(
        {
            "is_listen": False,
            "text": "first sentence",
            "tts_is_last_chunk": True,
            "turn_end": False,
        },
        fence,
    )

    assert events == (
        ModelSpeaking(fence=fence),
        ModelTextDelta(text="first sentence", fence=fence),
        ModelSegmentEnded(fence=fence),
    )
    assert not any(isinstance(event, ModelTurnEnded) for event in events)


def test_only_model_turn_end_metadata_finishes_response():
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    events = MiniCPMO45ModelEventAdapter().map_output(
        {
            "is_listen": False,
            "text": "",
            "tts_is_last_chunk": True,
            "turn_end": True,
        },
        fence,
    )

    assert events[-2:] == (
        ModelSegmentEnded(fence=fence),
        ModelTurnEnded(reason="model_turn_eos", fence=fence),
    )
