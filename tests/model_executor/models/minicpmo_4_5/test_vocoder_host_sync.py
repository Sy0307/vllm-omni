# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 vocoder stage's sync-free replacements are exact.

Each replacement (DiT timestep embedding, HiFT inverse STFT, codec id upload,
generation-runner output copy) removes a host sync from the per-step path; it
must produce bitwise the same values as the call it replaces.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import hifigan
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import _codec_tensor
from vllm_omni.worker.gpu_generation_model_runner import _HostCopyBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cosyvoice2_estimator(hidden_size: int = 32):
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    embedder = decoder_dit.TimestepEmbedder(hidden_size)
    return SimpleNamespace(t_embedder=embedder), decoder_dit


def _time_embedding(backend, estimator, time):
    return BatchedToken2Wav._time_embedding(backend, estimator, time)


def test_time_embedding_matches_cosyvoice2_embedder_bitwise():
    estimator, _ = _cosyvoice2_estimator()
    backend = SimpleNamespace(_timestep_freqs={})
    timeline = 1 - torch.cos(torch.linspace(0, 1, 11) * 0.5 * torch.pi)
    for step in range(10):
        time = timeline[step].expand(6)
        with torch.inference_mode():
            expected = estimator.t_embedder(time)
            actual = _time_embedding(backend, estimator, time)
        assert torch.equal(actual, expected)


def test_time_embedding_builds_the_frequency_table_once(monkeypatch):
    estimator, decoder_dit = _cosyvoice2_estimator()
    backend = SimpleNamespace(_timestep_freqs={})
    time = torch.full((4,), 0.25)
    with torch.inference_mode():
        expected = estimator.t_embedder(time)

    def rebuilt(*_args, **_kwargs):
        raise AssertionError("the per-call frequency table must not be rebuilt")

    monkeypatch.setattr(decoder_dit.TimestepEmbedder, "timestep_embedding", staticmethod(rebuilt))
    with torch.inference_mode():
        first = _time_embedding(backend, estimator, time)
        second = _time_embedding(backend, estimator, time)
    assert len(backend._timestep_freqs) == 1
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_time_embedding_falls_back_for_other_embedders():
    backend = SimpleNamespace(_timestep_freqs={})
    estimator = SimpleNamespace(t_embedder=lambda time: time[:, None] * 2)
    time = torch.tensor([0.5, 1.0])
    assert torch.equal(_time_embedding(backend, estimator, time), time[:, None] * 2)
    assert backend._timestep_freqs == {}


def _spectrum(batch: int, frames: int, n_fft: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    bins = n_fft // 2 + 1
    magnitude = torch.rand(batch, bins, frames, generator=generator) * 3
    phase = torch.randn(batch, bins, frames, generator=generator)
    return magnitude, phase


@pytest.mark.parametrize("batch,frames", [(1, 2), (1, 7), (3, 50), (8, 301)])
def test_istft_without_host_sync_matches_torch_istft_bitwise(batch, frames):
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft, periodic=True)
    magnitude, phase = _spectrum(batch, frames, n_fft)
    spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

    expected = torch.istft(spec, n_fft, hop, n_fft, window=window)
    actual = hifigan._istft_without_host_sync(spec, n_fft, hop, window, {})

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_hift_istft_reuses_the_envelope_per_frame_count(monkeypatch):
    n_fft, hop = 16, 4
    window = torch.from_numpy(hifigan.get_window("hann", n_fft, fftbins=True).astype("float32"))
    hift = SimpleNamespace(
        _use_cached_istft=True,
        istft_params={"n_fft": n_fft, "hop_len": hop},
        _get_stft_window=lambda tensor: window,
    )
    built = []
    original = hifigan._istft_envelope

    def counted(*args):
        built.append(args[-1])
        return original(*args)

    monkeypatch.setattr(hifigan, "_istft_envelope", counted)
    for seed, frames in enumerate((20, 20, 33, 20)):
        magnitude, phase = _spectrum(2, frames, n_fft, seed)
        clipped = torch.clip(magnitude, max=1e2)
        reference = torch.istft(
            torch.complex(clipped * torch.cos(phase), clipped * torch.sin(phase)),
            n_fft,
            hop,
            n_fft,
            window=window,
        )
        assert torch.equal(HiFTGenerator._istft(hift, magnitude, phase), reference)
    assert built == [20, 33]


def test_codec_tensor_uploads_host_ids_as_long_on_the_segment_device():
    segment = torch.zeros(3, dtype=torch.int32)
    ids = _codec_tensor(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), segment)
    assert ids.dtype == torch.long and ids.tolist() == [1, 2, 3, 4]
    assert _codec_tensor([[5, 6], [7, 8]], segment).tolist() == [5, 6, 7, 8]
    assert _codec_tensor(None, torch.tensor([9, 10], dtype=torch.int32)).tolist() == [9, 10]


def test_host_copy_batch_matches_blocking_copy_on_cpu():
    to_host = _HostCopyBatch(pin_memory=False)
    base = torch.arange(12.0).reshape(3, 4)
    strided = base.t()
    copied = to_host.copy(strided)
    to_host.wait()
    assert copied.is_contiguous()
    assert torch.equal(copied, strided)
    assert to_host.copy(base).data_ptr() == base.data_ptr()


def test_shared_hift_keeps_native_istft_without_opt_in(monkeypatch):
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft)
    hift = SimpleNamespace(
        istft_params={"n_fft": n_fft, "hop_len": hop},
        _get_stft_window=lambda tensor: window,
    )

    def unexpected_cache(*args):
        raise AssertionError("shared HiFT must not opt in implicitly")

    monkeypatch.setattr(hifigan, "_istft_without_host_sync", unexpected_cache)
    magnitude, phase = _spectrum(2, 20, n_fft, 0)
    assert HiFTGenerator._istft(hift, magnitude, phase).shape == (2, 76)
    assert not hasattr(hift, "_istft_envelopes")
