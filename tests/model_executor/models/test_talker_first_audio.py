# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker first-frame audio is on by default for streaming MRv2 Talkers."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.common.talker_first_audio import talker_first_audio_enabled

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("env", "async_chunk", "v2", "expected"),
    [
        (None, True, True, True),  # default
        ("1", True, True, True),
        ("0", True, True, False),
        (None, False, True, False),
        (None, True, False, False),
    ],
)
def test_talker_first_audio_default_and_conditions(monkeypatch, env, async_chunk, v2, expected):
    if env is None:
        monkeypatch.delenv("VLLM_OMNI_TALKER_FIRST_AUDIO", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_TALKER_FIRST_AUDIO", env)
    config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk, use_v2_model_runner=v2))
    assert talker_first_audio_enabled(config) is expected
