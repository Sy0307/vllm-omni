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


def test_timeline_max_model_len_validation() -> None:
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        validate_timeline_fits_max_model_len,
    )

    # The acceptance fixture (56 + 1 + 196 = 253) fits the shipped 8192 limit.
    validate_timeline_fits_max_model_len(56, 196, 8192)
    # Exactly at the limit: the vLLM request occupies prompt + 1 (frame-0
    # placeholder) + frames tokens, so 57 + 8135 == 8192 is allowed...
    validate_timeline_fits_max_model_len(56, 8135, 8192)
    # ...and one more frame must be rejected, naming every offending length.
    with pytest.raises(ValueError) as excinfo:
        validate_timeline_fits_max_model_len(56, 8136, 8192)
    message = str(excinfo.value)
    for fragment in (
        "vllm_prefill_len=57",
        "acoustic_frame_count=8136",
        "vllm_total_len=8193",
        "max_model_len=8192",
        "logical_prompt_token_len=56",
        "logical_total_len=8192",
    ):
        assert fragment in message


def _bare_thinker():
    import torch
    from torch import nn

    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        NemotronVoiceChatThinkerForConditionalGeneration,
    )

    thinker = NemotronVoiceChatThinkerForConditionalGeneration.__new__(NemotronVoiceChatThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker._use_function_head = True
    thinker._dtype = torch.float32
    thinker.function_head = nn.Linear(8, 16, bias=False)
    return thinker


def test_postprocess_appends_function_token_to_timeline() -> None:
    import torch

    thinker = _bare_thinker()
    hidden = torch.randn(3, 8)
    expected = int(thinker.function_head(hidden[-1:]).argmax(dim=-1))

    # No existing timeline: starts one.
    update = thinker.postprocess(hidden)
    assert int(update["nvc_prev_function_token"]) == expected
    assert update["nvc_function_tokens"].tolist() == [expected]

    # Existing timeline: appended, not replaced.
    existing = torch.tensor([7, 8, 9], dtype=torch.long)
    update = thinker.postprocess(hidden, nvc_function_tokens=existing)
    assert update["nvc_function_tokens"].tolist() == [7, 8, 9, expected]


def test_postprocess_skips_intermediate_prefill_chunks() -> None:
    import torch

    thinker = _bare_thinker()
    hidden = torch.randn(4, 8)
    # Intermediate chunk: 0 computed + 4 rows < prompt_len 10 -> no token.
    assert thinker.postprocess(hidden, _omni_is_prefill=True, _omni_num_computed_tokens=0, _omni_prompt_len=10) == {}
    # Final chunk reaches the last prefill position (acoustic frame 0).
    update = thinker.postprocess(hidden, _omni_is_prefill=True, _omni_num_computed_tokens=6, _omni_prompt_len=10)
    assert "nvc_prev_function_token" in update
    assert update["nvc_function_tokens"].numel() == 1


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


def test_vendored_mel_filterbank_matches_librosa() -> None:
    librosa = pytest.importorskip("librosa")

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.librosa_mel import (
        mel as vendored_mel,
    )

    for kwargs in (
        {"sr": 16000, "n_fft": 512, "n_mels": 128, "fmin": 0.0, "fmax": 8000.0, "norm": "slaney"},
        {"sr": 22050, "n_fft": 1024, "n_mels": 80, "fmin": 0.0, "fmax": None, "norm": "slaney", "htk": True},
        {"sr": 16000, "n_fft": 512, "n_mels": 80, "fmin": 0.0, "fmax": 8000.0, "norm": None},
    ):
        ours = vendored_mel(**kwargs)
        reference = librosa.filters.mel(**kwargs)
        assert (ours == reference).all(), f"vendored mel diverged for {kwargs}"
