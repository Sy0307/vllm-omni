# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_make_omni_output_preserves_per_request_ref_code():
    model = object.__new__(Qwen3TTSTalkerForConditionalGeneration)
    hidden = torch.randn(4, 8)
    ref_code_1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    ref_code_2 = torch.tensor([[5, 6], [7, 8], [9, 0]], dtype=torch.long)

    info_dicts = [
        {
            "audio_codes": torch.tensor([[1, 1], [2, 2]], dtype=torch.long),
            "ref_code": ref_code_1,
            "ref_code_len": 2,
            "codec_streaming": True,
        },
        {
            "audio_codes": torch.tensor([[3, 3], [4, 4]], dtype=torch.long),
            "ref_code": ref_code_2,
            "ref_code_len": 3,
            "codec_streaming": True,
        },
    ]

    output = Qwen3TTSTalkerForConditionalGeneration.make_omni_output(
        model,
        hidden,
        model_intermediate_buffer=info_dicts,
    )

    ref_codes = output.multimodal_outputs["ref_code"]
    assert len(ref_codes) == 2
    assert torch.equal(ref_codes[0], ref_code_1)
    assert torch.equal(ref_codes[1], ref_code_2)
