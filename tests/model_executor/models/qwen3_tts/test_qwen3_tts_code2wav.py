# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav as code2wav_mod
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
    Qwen3TTSCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_QUANTIZERS = 2
_TOTAL_UPSAMPLE = 4
_OUTPUT_SAMPLE_RATE = 24000


class _FakeDecoder(nn.Module):
    def __init__(self, total_upsample: int = _TOTAL_UPSAMPLE):
        super().__init__()
        self.total_upsample = total_upsample
        self.decode_calls: list[dict[str, int]] = []
        self.cudagraph_calls: list[dict[str, int | torch.device]] = []
        self.fail_batch_decode = False

    def to(self, *args, **kwargs):
        return self

    def chunked_decode(
        self,
        codes: torch.Tensor,
        *,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        if self.fail_batch_decode and codes.shape[0] > 1:
            raise RuntimeError("synthetic batch decode failure")
        self.decode_calls.append(
            {
                "batch_size": int(codes.shape[0]),
                "frames": int(codes.shape[-1]),
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
            }
        )
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1).expand(codes.shape[0], 1, -1).clone()

    def enable_cudagraph(self, **kwargs):
        self.cudagraph_calls.append(kwargs)


def _fake_dec_config():
    return SimpleNamespace(
        num_quantizers=_NUM_QUANTIZERS,
        sliding_window=0,
    )


def _make_model(
    *,
    stage_connector_config=None,
    async_chunk: bool = False,
    device: torch.device | None = None,
) -> Qwen3TTSCode2Wav:
    dec_config = _fake_dec_config()
    tok_config = SimpleNamespace(
        decoder_config=dec_config,
        output_sample_rate=_OUTPUT_SAMPLE_RATE,
    )
    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Config.from_pretrained",
            return_value=tok_config,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Decoder._from_config",
            return_value=_FakeDecoder(),
        ),
    ):
        model = Qwen3TTSCode2Wav(
            vllm_config=SimpleNamespace(
                load_config=SimpleNamespace(),
                model_config=SimpleNamespace(
                    model="unused",
                    revision=None,
                    stage_connector_config=stage_connector_config,
                    async_chunk=async_chunk,
                ),
                device_config=SimpleNamespace(device=device or torch.device("cpu")),
            )
        )
    return model


def _load_weights_noop(model: Qwen3TTSCode2Wav) -> set[str]:
    class _FakeModelLoader:
        class Source:
            def __init__(self, **_: object):
                pass

        def __init__(self, _load_config: object):
            pass

        def _get_weights_iterator(self, _source: object):
            return iter(())

    class _FakeAutoWeightsLoader:
        def __init__(self, *_: object, **__: object):
            pass

        def load_weights(self, _weights: object) -> set[str]:
            return {"decoder.fake_weight"}

    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.DefaultModelLoader",
            _FakeModelLoader,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.AutoWeightsLoader",
            _FakeAutoWeightsLoader,
        ),
    ):
        return model.load_weights(iter(()))


def test_forward_trims_context_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 2}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_trims_trailing_padding_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_connector_codec_chunking_does_not_override_decode_chunking():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
            }
        },
    )

    loaded = _load_weights_noop(model)

    assert loaded == {"decoder.fake_weight"}
    assert model._decode_chunk_frames == 300
    assert model._decode_left_context_frames == 25

    model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    assert model.decoder.decode_calls[-1] == {
        "batch_size": 1,
        "frames": 6,
        "chunk_size": 300,
        "left_context_size": 25,
    }


def test_forward_batches_equal_length_requests():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(24, dtype=torch.long),
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
        seq_token_counts=[12, 12],
    )

    assert len(out.multimodal_outputs["model_outputs"]) == 2
    assert model.decoder.decode_calls == [
        {
            "batch_size": 2,
            "frames": 6,
            "chunk_size": 300,
            "left_context_size": 25,
        }
    ]


def test_forward_buckets_mixed_length_requests():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(20, dtype=torch.long),
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
        seq_token_counts=[12, 8],
    )

    assert len(out.multimodal_outputs["model_outputs"]) == 2
    assert model.decoder.decode_calls == [
        {
            "batch_size": 1,
            "frames": 6,
            "chunk_size": 300,
            "left_context_size": 25,
        },
        {
            "batch_size": 1,
            "frames": 4,
            "chunk_size": 300,
            "left_context_size": 25,
        },
    ]


def test_forward_falls_back_to_per_request_when_batch_decode_fails():
    model = _make_model()
    model.decoder.fail_batch_decode = True

    out = model.forward(
        input_ids=torch.arange(24, dtype=torch.long),
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
        seq_token_counts=[12, 12],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert all(audio.numel() > 0 for audio in audios)
    assert model.decoder.decode_calls == [
        {
            "batch_size": 1,
            "frames": 6,
            "chunk_size": 300,
            "left_context_size": 25,
        },
        {
            "batch_size": 1,
            "frames": 6,
            "chunk_size": 300,
            "left_context_size": 25,
        },
    ]


def test_forward_rejects_multi_request_metadata_without_request_splits():
    model = _make_model()

    with pytest.raises(ValueError, match="seq_token_counts"):
        model.forward(
            input_ids=torch.arange(24, dtype=torch.long),
            runtime_additional_information=[
                {"meta": {"left_context_size": 0}},
                {"meta": {"left_context_size": 0}},
            ],
        )


def test_forward_rejects_multi_request_metadata_with_only_forward_context_splits(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(code2wav_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        code2wav_mod,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_slices=[12, 12]),
    )

    with pytest.raises(ValueError, match="seq_token_counts"):
        model.forward(
            input_ids=torch.arange(24, dtype=torch.long),
            runtime_additional_information=[
                {"meta": {"left_context_size": 0}},
                {"meta": {"left_context_size": 0}},
            ],
        )


def test_forward_skips_terminal_malformed_chunk_without_warning(caplog):
    model = _make_model()

    with caplog.at_level(logging.WARNING):
        out = model.forward(
            input_ids=torch.tensor([2150], dtype=torch.long),
            runtime_additional_information=[{"meta": {"finished": torch.tensor(True)}}],
        )

    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model.decoder.decode_calls == []
    assert "not divisible by num_quantizers" not in caplog.text


def test_forward_skips_single_token_sentinel_without_warning(caplog):
    model = _make_model()

    with caplog.at_level(logging.WARNING):
        out = model.forward(input_ids=torch.tensor([2150], dtype=torch.long))

    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model.decoder.decode_calls == []
    assert "not divisible by num_quantizers" not in caplog.text


def test_forward_rejects_multi_value_finished_metadata():
    model = _make_model()

    with pytest.raises(ValueError, match="scalar bool metadata"):
        model.forward(
            input_ids=torch.arange(12, dtype=torch.long),
            runtime_additional_information=[{"meta": {"finished": [True, False]}}],
        )


def test_decode_chunking_can_be_overridden_separately():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_chunk_frames": 400,
                "decode_left_context_frames": 17,
            }
        },
    )

    _load_weights_noop(model)

    assert model._decode_chunk_frames == 400
    assert model._decode_left_context_frames == 17


def test_decode_chunking_override_is_passed_to_cudagraph():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_chunk_frames": 400,
                "decode_left_context_frames": 17,
            }
        },
    )

    _load_weights_noop(model)

    assert model.decoder.cudagraph_calls[-1] == {
        "device": torch.device("cuda"),
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 72,
        "decode_chunk_size": 400,
        "decode_left_context": 17,
        "capture_batch_sizes": None,
    }


def test_decode_capture_batch_sizes_are_passed_to_cudagraph():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_capture_batch_sizes": [1, 2, 4, 8],
            }
        },
    )

    _load_weights_noop(model)

    assert model.decoder.cudagraph_calls[-1]["capture_batch_sizes"] == [1, 2, 4, 8]


def test_invalid_decode_chunking_is_rejected():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "decode_chunk_frames": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="decode_chunk_frames=0"):
        _load_weights_noop(model)
