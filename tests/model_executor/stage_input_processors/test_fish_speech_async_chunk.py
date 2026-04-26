from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.fish_speech import (
    slow_ar_to_dac_decoder_async_chunk,
)


def _manager(*, chunk_frames=25, left_context=25, initial_frames=4):
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_frames,
                }
            }
        ),
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
    )


def _request(rid="req"):
    return SimpleNamespace(external_req_id=rid, additional_information=None, is_finished=lambda: False)


def _pooling_output(frame_id: int):
    return {"audio_codes": torch.tensor([[frame_id, frame_id + 100]], dtype=torch.long)}


def test_fish_async_chunk_uses_initial_chunk_only_once():
    manager = _manager(chunk_frames=25, left_context=25, initial_frames=4)
    request = _request()

    emitted = None
    for frame_id in range(1, 5):
        emitted = slow_ar_to_dac_decoder_async_chunk(manager, _pooling_output(frame_id), request)

    assert emitted is not None
    assert emitted["left_context_size"] == 0
    assert emitted["code_predictor_codes"] == [1, 2, 3, 4, 101, 102, 103, 104]
    assert manager.request_payload["req"]["_fish_speech_sent_frames"] == 4

    for frame_id in range(5, 29):
        assert slow_ar_to_dac_decoder_async_chunk(manager, _pooling_output(frame_id), request) is None

    emitted = slow_ar_to_dac_decoder_async_chunk(manager, _pooling_output(29), request)

    assert emitted is not None
    assert emitted["left_context_size"] == 4
    assert len(emitted["code_predictor_codes"]) == 58
    assert manager.request_payload["req"]["_fish_speech_sent_frames"] == 29


def test_fish_async_chunk_can_emit_compact_tensor(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_COMPACT_CODES", "1")
    manager = _manager(chunk_frames=4, left_context=0, initial_frames=4)
    request = _request()

    emitted = None
    for frame_id in range(1, 5):
        emitted = slow_ar_to_dac_decoder_async_chunk(manager, _pooling_output(frame_id), request)

    assert emitted is not None
    codes = emitted["code_predictor_codes"]
    assert isinstance(codes, torch.Tensor)
    assert codes.dtype == torch.int16
    assert codes.tolist() == [1, 2, 3, 4, 101, 102, 103, 104]
