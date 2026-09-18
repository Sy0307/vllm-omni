# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for init_omni_model_state factory dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.worker_v2.model_states import (
    init_omni_model_state,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_vllm_config(architectures):
    model_config = SimpleNamespace(architectures=architectures)
    return SimpleNamespace(model_config=model_config)


@patch("vllm_omni.worker_v2.model_states._upstream_init_model_state")
def test_unknown_arch_delegates_to_upstream(mock_upstream):
    mock_upstream.return_value = MagicMock()
    cfg = _make_vllm_config(["LlamaForCausalLM"])
    model = MagicMock()
    device = torch.device("cpu")

    state = init_omni_model_state(cfg, model, None, device)

    mock_upstream.assert_called_once_with(cfg, model, None, device)
    assert state is mock_upstream.return_value


@patch("vllm_omni.worker_v2.model_states._upstream_init_model_state")
def test_none_architectures_delegates_to_upstream(mock_upstream):
    mock_upstream.return_value = MagicMock()
    cfg = _make_vllm_config(None)
    model = MagicMock()
    device = torch.device("cpu")

    init_omni_model_state(cfg, model, None, device)

    mock_upstream.assert_called_once()


@pytest.mark.parametrize("flag", ["has_preprocess", "has_postprocess", "have_multimodal_outputs"])
def test_unlisted_model_with_omni_capability_uses_omni_state(monkeypatch, flag):
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    model = SimpleNamespace(**{flag: True})
    state = init_omni_model_state(_make_vllm_config(["UnlistedModel"]), model, None, torch.device("cpu"))
    assert isinstance(state, OmniModelState)


def test_disabled_capabilities_preserve_upstream_dispatch(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("vllm_omni.worker_v2.model_states._upstream_init_model_state", lambda *args: sentinel)
    model = SimpleNamespace(has_preprocess=False, has_postprocess=False, have_multimodal_outputs=False)
    assert init_omni_model_state(_make_vllm_config(["OtherModel"]), model, None, torch.device("cpu")) is sentinel


@pytest.mark.parametrize("architecture", ["Qwen3TTSTalkerForConditionalGeneration", "Qwen3TTSCode2Wav"])
def test_qwen3_tts_dispatches_to_omni_model_state(monkeypatch, architecture):
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    state = init_omni_model_state(_make_vllm_config([architecture]), SimpleNamespace(), None, torch.device("cpu"))
    assert isinstance(state, OmniModelState)
