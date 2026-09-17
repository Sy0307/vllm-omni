# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torchaudio

from vllm_omni.model_executor.models.moss_tts.reference_encoder import _prep_wav_sync

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_prepared_waveform_exact(sr, channels):
    generator = torch.Generator().manual_seed(123)
    waveform = torch.rand(channels, sr, generator=generator).tolist()
    expected = torch.tensor(waveform, dtype=torch.float32)
    if sr != 24000:
        expected = torchaudio.functional.resample(expected, sr, 24000)
    assert torch.equal(_prep_wav_sync(waveform, sr, 24000), expected)
    assert torch.equal(_prep_wav_sync(waveform, sr, 24000), expected)


def test_prepared_waveform_owns_storage():
    original = torch.arange(16, dtype=torch.float32).numpy()
    prepared = _prep_wav_sync(original, 24000, 24000)
    original[:] = -1
    assert torch.equal(prepared, torch.arange(16, dtype=torch.float32).unsqueeze(0))
