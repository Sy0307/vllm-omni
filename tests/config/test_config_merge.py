# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy

import pytest

from vllm_omni.config.stage_config import _merge_config_fields, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_selected_fields_merge_recursively_without_mutating_inputs():
    base = {"section": {"nested": {"a": 1, "b": 2}}, "replace": {"old": 1}}
    overlay = {"section": {"nested": {"b": 0}}, "replace": {"new": 2}}
    before = deepcopy((base, overlay))
    merged = _merge_config_fields(base, overlay, deep_merge_keys=frozenset({"section"}))
    assert merged == {"section": {"nested": {"a": 1, "b": 0}}, "replace": {"new": 2}}
    assert (base, overlay) == before


@pytest.mark.parametrize("value", [None, False, 0, [1, 2]])
def test_non_mapping_overlay_replaces_mapping(value):
    assert _merge_config_fields({"section": {"a": 1}}, {"section": value}, deep_merge_keys=frozenset({"section"})) == {
        "section": value
    }


def test_recursive_deploy_inheritance(tmp_path):
    # Test raw YAML merging independently of the current flat speech schema.
    (tmp_path / "base.yaml").write_text("speech_cache:\n  nested:\n    a: 1\n    b: 2\n")
    (tmp_path / "middle.yaml").write_text("base_config: base.yaml\nspeech_cache:\n  nested:\n    b: 0\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("base_config: middle.yaml\nasync_chunk: false\n")
    merged = resolve_deploy_yaml(leaf)
    assert merged["speech_cache"] == {"nested": {"a": 1, "b": 0}}
    assert merged["async_chunk"] is False


def test_single_stage_streaming_capability_uses_central_async_resolver():
    from vllm_omni.config.stage_config import DeployConfig, _resolve_pipeline_async_chunk_enabled
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE

    assert _resolve_pipeline_async_chunk_enabled(QWEN3_TTS_FUSED_PIPELINE, DeployConfig())
    with pytest.raises(ValueError, match="requires async_chunk=True"):
        _resolve_pipeline_async_chunk_enabled(QWEN3_TTS_FUSED_PIPELINE, DeployConfig(async_chunk=False))


def test_single_stage_tts_rejects_unsupported_decoder_configuration(mocker):
    from vllm_omni.model_executor.models.qwen3_tts.first_audio import talker_stream_decode_enabled

    config = mocker.Mock()
    config.additional_config = {"talker_stream_decode": True}
    config.model_config.stage_connector_config = {}
    config.model_config.use_v2_model_runner = True
    config.model_config.async_chunk = True
    config.parallel_config.tensor_parallel_size = config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.distributed_executor_backend = "uni"
    config.cache_config.enable_prefix_caching = True
    mocker.patch(
        "vllm_omni.model_executor.models.qwen3_tts.first_audio.current_omni_platform.is_cuda", return_value=True
    )
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        talker_stream_decode_enabled(config)
    config.cache_config.enable_prefix_caching = False
    assert talker_stream_decode_enabled(config)
