# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
    talker2code2wav,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("rows", [0, 4, 8, 9, 30])
def test_native_full_payload_preserves_legacy_dedelay_and_tail(rows):
    audio = torch.arange(rows * 8).reshape(rows, 8) % 1026
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": audio}})])
    legacy = talker2code2wav([out])[0]["prompt_token_ids"]
    native = talker2code2wav_full_payload(None, {"codes.audio": audio}, None)
    assert native["codes"]["audio"].tolist() == legacy
    assert native["meta"]["finished"]
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == legacy


def test_native_control_slot_without_audio_payload():
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output=None)])
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == [0]
