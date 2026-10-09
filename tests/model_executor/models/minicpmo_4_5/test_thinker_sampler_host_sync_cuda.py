# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Thinker's duplex sampler must not block the host on the GPU.

With async scheduling, the step after a launch samples while the GPU still
runs that launch; a synchronizing write here serializes every Thinker step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45DuplexSampler
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.utils.device_copy import index_to_device

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_TERMINATOR = 7


def _sampler(mocker, deferred):
    params = SamplingParams(temperature=0.7, top_k=100, top_p=0.8, seed=42, max_tokens=20)
    info = {
        "req_id": "r",
        "sampling_params": params,
        "duplex": {"data_plane": True, "session_id": "s", "seq": 0, "payload": {}},
    }
    model = mocker.Mock(
        spec=MiniCPMO45OmniForConditionalGeneration,
        _mrv2_duplex_infos=[info],
        prepare_duplex_sampling=mocker.Mock(),
        _minicpmo45_native_duplex_token_ids=lambda: {},
        _minicpmo45_chunk_terminator_token_ids=lambda _: {_TERMINATOR},
        _sample_minicpmo45_native_duplex_rows_deferred=mocker.Mock(side_effect=deferred),
        _record_minicpmo45_duplex_terminator=mocker.Mock(),
        _commit_minicpmo45_duplex_pending_samples=mocker.Mock(),
    )
    base = mocker.Mock(spec=Sampler, side_effect=lambda logits, _batch: logits)
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        num_computed_prefill_tokens_np=np.array([4]),
        num_scheduled_tokens=np.array([1]),
        prefill_len_np=np.array([4]),
    )
    return MiniCPMO45DuplexSampler(base, model), model, batch


def test_deferred_and_lookahead_rows_select_tokens_without_host_sync(mocker):
    device = torch.device("cuda")
    drawn = torch.tensor([_TERMINATOR], device=device)
    sampler, model, batch = _sampler(mocker, lambda *args, **kwargs: drawn)
    host = torch.tensor([[_TERMINATOR, _TERMINATOR, 0]], dtype=torch.long, pin_memory=True)
    model._minicpmo45_duplex_pending_samples = mocker.Mock(host=host, event=None)
    # Warm the pinned host allocator and CUDA context outside the check.
    index_to_device([1], device)
    logits = [torch.randn(1, 16, device=device) for _ in range(2)]
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            deferred_logits = sampler(logits[0], batch)
            # The terminator is committed at the next call; async scheduling
            # then samples one lookahead step whose outcome is that terminator.
            lookahead_logits = sampler(logits[1], batch)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    for selected in (deferred_logits, lookahead_logits):
        assert selected.argmax(dim=-1).tolist() == [_TERMINATOR]
        assert torch.isinf(selected).sum().item() == selected.shape[-1] - 1
    model._sample_minicpmo45_native_duplex_rows_deferred.assert_called_once()
    assert sampler._requests["r"][1] == [_TERMINATOR]
