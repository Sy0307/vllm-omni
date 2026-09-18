# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_intermediate_buffer_keeps_nested_gpu_resident_tensor_on_device() -> None:
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to verify GPU residency")
    buffer = OmniIntermediateBuffer(max_num_reqs=1)
    tensor = torch.ones((2, 3), device="cuda")
    buffer.update(0, {"embed": {"prefill": tensor}}, gpu_resident_keys={("embed", "prefill")})
    stored = buffer.buffers[0]["embed"]["prefill"]
    assert isinstance(stored, torch.Tensor)
    assert stored.device.type == "cuda"
    assert stored.shape == (2, 3)


def test_intermediate_buffer_keeps_codes_ref_on_device() -> None:
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to verify GPU residency")
    buffer = OmniIntermediateBuffer(max_num_reqs=1)
    ref_code = torch.ones((4, 16), device="cuda", dtype=torch.long)
    buffer.update(0, {"codes": {"ref": ref_code}}, gpu_resident_keys={("codes", "ref")})
    stored = buffer.buffers[0]["codes"]["ref"]
    assert isinstance(stored, torch.Tensor)
    assert stored.device.type == "cuda"
    assert stored.dtype == torch.long


def test_prompt_builder_long_tensor_cache_reuses_tensor() -> None:
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._long_tensor_cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = builder._long_tensor([1, 2, 3], device)
    second = builder._long_tensor([1, 2, 3], device)
    assert first is second
    assert first.device.type == device.type
    if first.device.type == "cuda":
        assert first.device.index == torch.accelerator.current_device_index()
    assert first.dtype == torch.long
    assert first.tolist() == [[1, 2, 3]]
