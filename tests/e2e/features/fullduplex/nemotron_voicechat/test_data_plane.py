from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.engine.duplex_runtime import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.nemotron_voicechat.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_RUNTIME = {
    "nvc_text_bos_id": 0,
    "nvc_text_eos_id": 1,
    "nvc_text_pad_id": 12,
    "nvc_function_sotc_id": 20,
    "nvc_function_eotc_id": 21,
    "nvc_tokenizer_ref": "test-tokenizer",
}


def _projector(encode_audio=lambda *_: None) -> NemotronVoiceChatDataPlaneSession:
    projector = NemotronVoiceChatDataPlaneSession(encode_audio)
    projector.configure_runtime(_RUNTIME, tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "text"))
    return projector


def _stage0_output(token_id: int) -> object:
    completion = SimpleNamespace(multimodal_output={"nvc_text_token_ids": [token_id]})
    return SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(request_id="req-0", outputs=[completion]),
    )


def _stage2_output(audio: np.ndarray, *, finished: bool = False) -> object:
    completion = SimpleNamespace(
        multimodal_output={
            "model_outputs": [audio],
            "sr": [22050],
        }
    )
    return SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(request_id="req-0", outputs=[completion], finished=finished),
    )


def test_function_channel_projects_completed_call_without_ending_speech() -> None:
    projector = _projector()
    projector._decode = lambda token_ids: (
        '[{"name":"weather","arguments":{"city":"Shanghai"}}]' if token_ids == [99] else ""
    )
    context = NemotronVoiceChatDataPlaneContext(epoch=0)

    events = []
    for function_token in (20, 99, 21):
        completion = SimpleNamespace(
            multimodal_output={"nvc_text_token_ids": [12], "nvc_function_token": [function_token]}
        )
        output = SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id="req-fc", outputs=[completion]),
        )
        events.extend(projector.project_output(output, context=context))

    listen = [event for event in events if event.get("is_listen") is True]
    function = [event for event in events if event.get("function_call") is True]
    assert len(listen) == 3
    assert len(function) == 1
    assert function[0]["name"] == "weather"
    assert function[0]["arguments"] == '{"city":"Shanghai"}'


def test_new_epoch_request_does_not_inherit_partial_function_or_frame_state() -> None:
    projector = _projector(lambda *_: "audio")
    old_request = duplex_resource_request_id(DuplexFence("sid", epoch=1), "stage0")
    new_request = duplex_resource_request_id(DuplexFence("sid", epoch=2), "stage0")

    def stage0(request_id: str, *, text_token: int, function_token: int | None = None) -> object:
        metadata: dict[str, object] = {"nvc_text_token_ids": [text_token]}
        if function_token is not None:
            metadata["nvc_function_token"] = [function_token]
        completion = SimpleNamespace(multimodal_output=metadata)
        return SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id=request_id, outputs=[completion]),
        )

    projector.begin_request(old_request)
    list(projector.project_output(stage0(old_request, text_token=42, function_token=20)))
    projector.begin_request(new_request)

    # EOTC in the new epoch must not close the old epoch's partial function.
    new_events = list(projector.project_output(stage0(new_request, text_token=1, function_token=21)))
    assert not [event for event in new_events if event.get("function_call") is True]
    assert not [event for event in new_events if event.get("end_of_turn") is True]

    audio = SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(
            request_id=new_request,
            outputs=[
                SimpleNamespace(multimodal_output={"model_outputs": [np.ones(1764, dtype=np.float32)], "sr": [22050]})
            ],
        ),
    )
    audio_events = list(projector.project_output(audio))
    assert sum(event.get("end_of_turn") is True for event in audio_events) == 1
