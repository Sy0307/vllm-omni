# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS first-frame decode eligibility."""

from typing import Any

import torch

from vllm_omni.platforms import current_omni_platform


def talker_stream_decode_enabled(vllm_config: Any) -> bool:
    """The Talker decodes every frame itself and emits PCM as its final output."""
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    return (
        Predictor._parse_bool_config(extra.get("talker_stream_decode"))
        and current_omni_platform.is_cuda()
        and bool(getattr(model, "use_v2_model_runner", False))
        and bool(getattr(model, "async_chunk", False))
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and not vllm_config.cache_config.enable_prefix_caching
    )


def stream_ref_context_frames(vllm_config: Any) -> int:
    """Reference-code context frames for a voice-clone stream (same keys as the Code2Wav input processor)."""
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    return int(extra.get("ref_code_context_frames") or extra.get("codec_left_context_frames", 25))


def talker_first_audio_enabled(vllm_config: Any) -> bool:
    """First-frame delivery requires CUDA MRv2 streaming with an in-process Talker.

    Unsupported runners retain the regular codec path. Prefix-cache replay
    and distributed Talkers likewise retain the regular path.
    """
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    return (
        Predictor._parse_bool_config(extra.get("talker_first_audio"))
        and current_omni_platform.is_cuda()
        and torch.device(vllm_config.device_config.device).type == "cuda"
        and bool(getattr(model, "use_v2_model_runner", False))
        and bool(getattr(model, "async_chunk", False))
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.distributed_executor_backend in (None, "uni")
        and not vllm_config.cache_config.enable_prefix_caching
    )
