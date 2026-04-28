# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
from pytest_mock import MockerFixture

_MODELS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
    )
)
_COMMON = os.path.join(_MODELS, "common")


def _load_common_module(mocker: MockerFixture):
    platforms_mock = mocker.MagicMock()
    platforms_mock.current_omni_platform.supports_torch_inductor.return_value = False

    logger_mock = mocker.MagicMock()
    logger_mock.init_logger = lambda name: mocker.MagicMock()

    weight_utils_mock = mocker.MagicMock()
    weight_utils_mock.default_weight_loader = lambda p, w: None

    vllm_parallel_mock = mocker.MagicMock()
    vllm_parallel_mock.VocabParallelEmbedding = torch.nn.Embedding

    common_pkg = types.ModuleType("vllm_omni.model_executor.models.common")
    common_pkg.__path__ = [_COMMON]
    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = [_MODELS]

    mocker.patch.dict(
        sys.modules,
        {
            "vllm_omni": mocker.MagicMock(),
            "vllm_omni.platforms": platforms_mock,
            "vllm.logger": logger_mock,
            "vllm.config": mocker.MagicMock(),
            "vllm.model_executor.model_loader.weight_utils": weight_utils_mock,
            "vllm.model_executor.layers.vocab_parallel_embedding": vllm_parallel_mock,
            "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
            "vllm_omni.model_executor.models": models_pkg,
            "vllm_omni.model_executor.models.common": common_pkg,
        },
    )

    path = os.path.join(_COMMON, "qwen3_code_predictor.py")
    spec = importlib.util.spec_from_file_location(
        "vllm_omni.model_executor.models.common.qwen3_code_predictor", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_config_defaults_keep_cache_disabled(mocker: MockerFixture) -> None:
    mod = _load_common_module(mocker)

    config = mod.CodePredictorWrapperConfig()

    assert config.use_cache is False
    assert config.cache_config is None


def test_cache_state_allocates_expected_shapes(mocker: MockerFixture) -> None:
    mod = _load_common_module(mocker)

    state = mod.CodePredictorCacheState.allocate(
        batch_size=2,
        max_seq_len=5,
        num_layers=3,
        num_key_value_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert state.keys.shape == (3, 2, 4, 5, 8)
    assert state.values.shape == (3, 2, 4, 5, 8)
    assert state.seq_lens.tolist() == [[0, 0], [0, 0], [0, 0]]
    assert state.enabled is True


def test_disabled_cache_state_has_no_tensors(mocker: MockerFixture) -> None:
    mod = _load_common_module(mocker)

    state = mod.CodePredictorCacheState.disabled()

    assert state.keys is None
    assert state.values is None
    assert state.seq_lens is None
    assert state.enabled is False


def test_cache_state_reset_clears_sequence_lengths(mocker: MockerFixture) -> None:
    mod = _load_common_module(mocker)
    state = mod.CodePredictorCacheState.allocate(
        batch_size=2,
        max_seq_len=5,
        num_layers=3,
        num_key_value_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state.seq_lens[:] = torch.tensor([[3, 5], [2, 4], [1, 3]])

    state.reset()

    assert state.seq_lens.tolist() == [[0, 0], [0, 0], [0, 0]]


def test_attention_cached_step_matches_refill_last_token(
    mocker: MockerFixture,
) -> None:
    mod = _load_common_module(mocker)
    config = types.SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=8,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    attention = mod.CodePredictorAttention(config)
    hidden_states = torch.randn(2, 4, 8)
    rotary_emb = mod._RotaryEmbedding(config)
    full_position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(2, -1)
    step_position_ids = full_position_ids[:, -1:]
    cache_state = mod.CodePredictorCacheState.allocate(
        batch_size=2,
        max_seq_len=4,
        num_layers=1,
        num_key_value_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for step in range(3):
        attention.forward_cached_step(
            hidden_states[:, step : step + 1],
            rotary_emb(hidden_states[:, step : step + 1], full_position_ids[:, step : step + 1]),
            cache_state,
            layer_idx=0,
        )
    refill_last = attention(hidden_states, rotary_emb(hidden_states, full_position_ids))[:, -1:]

    cached_last = attention.forward_cached_step(
        hidden_states[:, -1:],
        rotary_emb(hidden_states[:, -1:], step_position_ids),
        cache_state,
        layer_idx=0,
    )

    torch.testing.assert_close(cached_last, refill_last, atol=1e-6, rtol=1e-5)
    assert cache_state.seq_lens.tolist() == [[4, 4]]
