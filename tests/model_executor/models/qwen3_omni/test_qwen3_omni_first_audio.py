# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni Talker first-frame audio (CPU, tiny random Code2Wav).

- The Talker's first-frame decoder computes exactly Code2Wav's streaming
  chunk 0 for a one-frame first chunk (no left context).
- The decoder is opt-in and only exists for a streaming Talker.
- A chunk ramp adds its decode windows to Code2Wav's graph sizes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

from vllm_omni.model_executor.models.qwen3_omni.first_frame_decoder import Qwen3OmniFirstFrameDecoder
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
    plan_decode_groups,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_Q = 4
_CODEBOOK = 16


def _code2wav() -> Qwen3OmniMoeCode2Wav:
    torch.manual_seed(0)
    config = Qwen3OmniMoeCode2WavConfig(
        codebook_size=_CODEBOOK,
        num_quantizers=_Q,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_dim=32,
        sliding_window=8,
    )
    model = Qwen3OmniMoeCode2Wav(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    return model.eval()


def test_first_frame_pcm_equals_code2wav_streaming_chunk0():
    code2wav = _code2wav()
    decoder = Qwen3OmniFirstFrameDecoder(code2wav, sample_rate=24000)
    codes = torch.randint(0, _CODEBOOK, (3, _Q))

    with torch.inference_mode():
        for row in range(codes.shape[0]):
            pcm = decoder.decode(codes[row : row + 1])
            # What the Code2Wav stage returns for a lone one-frame chunk 0.
            chunk0 = code2wav.chunked_decode_streaming(
                codes[row].reshape(1, _Q, 1), left_context_size=[0], seq_token_counts=[_Q]
            )[0]
            assert pcm.dtype == torch.float32
            assert torch.equal(pcm[0], chunk0.reshape(-1).float())
            # One frame minus the causal right-edge trim.
            assert 0 < pcm.shape[-1] < int(code2wav.total_upsample)
        batched = decoder.decode(codes)
    assert batched.shape == (3, pcm.shape[-1])


def test_first_frame_is_a_prefix_of_a_longer_first_chunk():
    """Trimming the delivered samples from a longer chunk 0 keeps the stream contiguous."""
    code2wav = _code2wav()
    decoder = Qwen3OmniFirstFrameDecoder(code2wav, sample_rate=24000)
    codes = torch.randint(0, _CODEBOOK, (_Q, 4))

    with torch.inference_mode():
        first = decoder.decode(codes[:, :1].T)[0]
        chunk0 = code2wav.chunked_decode_streaming(
            codes.reshape(1, _Q, 4), left_context_size=[0], seq_token_counts=[4 * _Q]
        )[0]
    torch.testing.assert_close(chunk0.reshape(-1)[: first.numel()].float(), first)


@pytest.mark.parametrize(
    ("env", "async_chunk", "v2"),
    [
        ("0", True, True),  # switched off
        (None, False, True),  # full-payload Code2Wav input
        (None, True, False),  # V1 runner: nothing decodes or delivers the frame
    ],
)
def test_first_frame_decoder_needs_streaming_mrv2_talker(monkeypatch, env, async_chunk, v2):
    if env is None:
        monkeypatch.delenv("VLLM_OMNI_TALKER_FIRST_AUDIO", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_TALKER_FIRST_AUDIO", env)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk, use_v2_model_runner=v2))

    assert Qwen3OmniMoeForConditionalGeneration._build_first_frame_decoder(vllm_config, None, "") is None


def test_chunk_ramp_adds_exact_code2wav_graph_sizes(monkeypatch):
    import vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper as wrapper_module

    real = wrapper_module.CUDAGraphDecoderWrapper
    wrapper = MagicMock(compute_capture_sizes=real.compute_capture_sizes)
    monkeypatch.setattr(wrapper_module, "CUDAGraphDecoderWrapper", wrapper)
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 25, "codec_chunk_ramp": [1, 2, 4, 8, 16, 25]}
    code2wav = SimpleNamespace(config=SimpleNamespace(num_quantizers=_Q))
    code2wav.enable_cudagraph = lambda **kw: Qwen3OmniMoeCode2Wav.enable_cudagraph(
        code2wav, device=torch.device("cuda"), **kw
    )
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.code2wav = code2wav
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(async_chunk=True, enforce_eager=False, stage_connector_config={"extra": extra})
    )
    for batches in ([], [2, 4, 8]):
        extra["codec_graph_batch_sizes"] = batches
        Qwen3OmniMoeForConditionalGeneration._maybe_enable_code2wav_cudagraph(model)
        args = wrapper.call_args.kwargs
        sizes = args["capture_sizes"]
        assert {1, 3, 7, 15, 31, 50} <= set(sizes)
        assert set(real.compute_capture_sizes(codec_chunk_frames=25, codec_left_context_frames=25)) <= set(sizes)
        assert set(args["extra_capture_shapes"]) == {
            (b, size) for b in batches for size in sizes if size <= 50 and b * size <= 325
        }
    assert code2wav._streaming_batch_sizes[50] == [1, 2, 4]
    extra.pop("codec_chunk_ramp")
    extra.pop("codec_graph_batch_sizes")
    Qwen3OmniMoeForConditionalGeneration._maybe_enable_code2wav_cudagraph(model)
    assert wrapper.call_args.kwargs["capture_sizes"] is None


_SIZES = [1, 2, 3, 4, 7, 8, 15, 16, 25, 31, 32, 50, 64]


def _bucket(length):
    return next((size for size in _SIZES if size >= length), None)


def test_decode_groups_keep_a_uniform_batch_whole_and_split_by_the_largest_graph():
    assert plan_decode_groups([50] * 6, _bucket, lambda size: [1, 2, 4, 8]) == [(list(range(6)), 50)]
    calls = plan_decode_groups([25] * 20, _bucket, lambda size: [1, 2, 4, 8, 16])
    assert [len(rows) for rows, _length in calls] == [16, 4]
    # Row counts are captured per graph size: long windows split into smaller calls.
    calls = plan_decode_groups([1] * 8 + [50] * 8, _bucket, lambda size: [1, 2, 4, 8] if size < 50 else [1, 2, 4])
    assert [(len(rows), length) for rows, length in calls] == [(8, 1), (4, 50), (4, 50)]
    assert plan_decode_groups([], _bucket, lambda size: [1, 2]) == []


class _PaddingGraphs:
    """Replays like CUDAGraphDecoderWrapper: pads rows/frames to a captured shape, trims the output."""

    def __init__(self, model, batch_sizes):
        self.model = model
        self.batch_sizes = sorted(batch_sizes)
        self.calls = []

    def _get_padded_size(self, length):
        return _bucket(length)

    def decode(self, codes):
        rows, _q, frames = codes.shape
        size = _bucket(frames)
        batch = next(b for b in self.batch_sizes if b >= rows)
        self.calls.append((rows, frames))
        padded = torch.zeros(batch, codes.shape[1], size, dtype=codes.dtype)
        padded[:rows, :, :frames] = codes
        out = self.model(padded)
        return out[:rows, :, : out.shape[-1] - (size - frames) * int(self.model.total_upsample)].clone()


def test_grouped_streaming_decode_matches_one_padded_batch():
    code2wav = _code2wav()
    lengths = [1, 3, 50, 7, 31]
    lefts = [0, 1, 25, 3, 15]
    codes = torch.zeros(len(lengths), _Q, max(lengths), dtype=torch.long)
    for row, length in enumerate(lengths):
        codes[row, :, :length] = torch.randint(0, _CODEBOOK, (_Q, length))
    counts = [length * _Q for length in lengths]

    with torch.inference_mode():
        # One call padded to the longest row (the default path).
        reference = code2wav.chunked_decode_streaming(codes, left_context_size=lefts, seq_token_counts=counts)
        graphs = _PaddingGraphs(code2wav, [1, 2, 4])
        code2wav._cudagraph_wrapper, code2wav._cudagraph_enabled = graphs, True
        code2wav._streaming_batch_sizes = {size: [1, 2, 4] for size in _SIZES}
        grouped = code2wav.chunked_decode_streaming(codes, left_context_size=lefts, seq_token_counts=counts)

    assert len(graphs.calls) > 1
    for want, got in zip(reference, grouped, strict=True):
        assert want.shape == got.shape
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
