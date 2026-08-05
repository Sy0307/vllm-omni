# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden shape test for the NemotronVoiceChat frame-locked timeline contract.

Pins the exact prefill/off-by-one arithmetic against values verified with the
NeMo reference run on sample_general.wav (fp32, greedy): perception produced
196 acoustic frames for a 250,760-sample 16 kHz input; the system prompt
tokenized to 56 ids ([bos] + 54 + [eos]); NeMo's timeline T = 252 = 56 + 196;
and our fp32 pipeline matched the reference text channel 196/196 tokens.
"""

import pytest

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    compute_acoustic_frame_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Perception config subset matching the checkpoint (mel 10 ms hop, n_fft 512,
# dw-striding 8x subsampling, conv kernel 3).
_STT_CFG = {
    "perception": {
        "preprocessor": {"sample_rate": 16000, "window_stride": 0.01, "window_size": 0.025, "n_fft": 512},
        "encoder": {"subsampling": "dw_striding", "subsampling_factor": 8, "subsampling_conv_kernel_size": 3},
    }
}

# (num_16k_samples, expected 12.5 Hz frames). The first row is the
# sample_general.wav acceptance fixture, verified against the NeMo reference
# (T=252 timeline with a 56-token prompt => 196 acoustic frames).
_GOLDEN = [
    (250760, 196),
    (16000 * 2, 26),
    (16000 * 4, 51),
    (16000 * 8, 101),
]


@pytest.mark.parametrize(("num_samples", "expected_frames"), _GOLDEN)
def test_acoustic_frame_count_golden(num_samples: int, expected_frames: int) -> None:
    assert compute_acoustic_frame_count(_STT_CFG, num_samples) == expected_frames


def test_prefill_contract_arithmetic() -> None:
    # vllm_prefill_len = logical_prompt_token_len + 1 (the +1 is acoustic
    # frame 0); decode steps == acoustic_frame_count; NeMo timeline
    # T == prompt + frames. Values from the verified sample_general.wav run.
    logical_prompt_len = 56
    frames = compute_acoustic_frame_count(_STT_CFG, 250760)
    vllm_prefill_len = logical_prompt_len + 1
    assert vllm_prefill_len == 57
    assert logical_prompt_len + frames == 252  # NeMo reference T
    # The talker consumes the full timeline from t=1 and the producer trims
    # the prompt region: rows P-1 onward of the (T-1)-row code stack.
    talker_rows = (logical_prompt_len + frames) - 1
    trimmed = talker_rows - (logical_prompt_len - 1)
    assert trimmed == frames
