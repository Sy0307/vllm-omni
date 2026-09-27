# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail", [False, True])
def test_runner_closes_metadata_scope_on_return_and_error(monkeypatch, fail):
    events = []

    def begin(**kw):
        events.append("begin")

    def end():
        events.append("end")

    def forward(*args, **kw):
        assert events == ["begin"]
        events.extend(["warmup", "capture"])
        if fail:
            raise RuntimeError("capture failed")
        return torch.ones(2)

    r = object.__new__(OmniGPUModelRunner)
    r.model = SimpleNamespace(
        supports_omni_decode_step_metadata=True, update_decode_step_metadata=begin, finish_decode_step_forward=end
    )
    r.input_batch = SimpleNamespace(req_ids=["a", "b"])
    r._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", forward)
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            r._model_forward()
    else:
        torch.testing.assert_close(r._model_forward(), torch.ones(2))
    assert events == ["begin", "warmup", "capture", "end"]
