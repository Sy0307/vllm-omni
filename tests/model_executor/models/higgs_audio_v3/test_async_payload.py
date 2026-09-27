# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
    HiggsAudioV3TalkerForConditionalGeneration as C,
)
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_two_inflight_steps_keep_owned_codes_and_masks():
    t = C.__new__(C)
    torch.nn.Module.__init__(t)
    t.num_codebooks = 2
    t.use_async_omni_output = True
    first = torch.tensor([[10, 11, 1, 0], [20, 21, 0, 0], [30, 31, 1, 1], [40, 41, 1, 0]])
    t._finish_audio_staging(first, 4, torch.device("cpu"), 4)
    first.zero_()  # Next graph replay overwrites original GPU storage.
    a = t.post_sample_multimodal_outputs(
        req_ids=["b", "a", "c", "invalid"], invalid_req_indices=[3], multimodal_outputs=None
    )
    second = torch.tensor([[50, 51, 1, 0]])
    t._finish_audio_staging(second, 1, torch.device("cpu"), 1)
    b = t.post_sample_multimodal_outputs(req_ids=["new"], invalid_req_indices=[], multimodal_outputs=None)
    second.zero_()
    assert t.post_sample_multimodal_outputs(req_ids=[], invalid_req_indices=[], multimodal_outputs=None) is None
    # Finish callbacks out of order. They cannot read current model state.
    rb = t.finalize_multimodal_outputs_from_cpu_snapshot(b)["codes"]["audio"]
    ra = t.finalize_multimodal_outputs_from_cpu_snapshot(a)["codes"]["audio"]
    assert [x.tolist() for x in ra] == [[[10, 11]], [], [[30, 31]], []]
    assert rb[0].tolist() == [[50, 51]]
    a["_higgs_audio_snapshot"].zero_()
    assert ra[0].tolist() == [[10, 11]]
    assert t._fast_audio_direct_rows == 0
    assert t.finalize_multimodal_outputs_from_cpu_snapshot(None) is None


@pytest.mark.parametrize("enabled", [False, True])
def test_whole_output_explicit_opt_in(enabled):
    r = object.__new__(GPUARModelRunner)
    r.use_async_scheduling = True
    r.speculative_config = None
    r.model_config = SimpleNamespace(async_chunk=False, enable_return_routed_experts=False)
    r.model = SimpleNamespace(use_async_omni_output=True, supports_async_whole_payload=enabled, has_postprocess=False)
    assert r._should_use_async_omni_output() == enabled
    r.use_async_scheduling = False
    assert not r._should_use_async_omni_output()
