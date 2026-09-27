# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("tokens,requests,replay_query", [(256, 64, 212), (192, 64, 170), (64, 64, 48)])
def test_full_capture_covers_unbalanced_replay(tokens, requests, replay_query):
    buffers = InputBuffers(requests, tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(requests, tokens, buffers)
    assert int(batch.num_scheduled_tokens.max()) < replay_query
    state = OmniModelState.__new__(OmniModelState)
    with patch.object(DefaultModelState, "prepare_attn", return_value={}) as backend:
        # Capture construction uses mode NONE plus for_capture=True.
        state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (),
            torch.empty(0),
            [],
            KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
            for_capture=True,
        )
    captured_batch = backend.call_args.args[0]
    assert captured_batch.max_query_len >= replay_query
    assert captured_batch.max_query_len == tokens
    assert captured_batch.query_start_loc is batch.query_start_loc
    assert batch.max_query_len is None


@pytest.mark.parametrize("for_capture,bound", [(False, None), (False, 8), (True, 4)])
def test_runtime_and_explicit_capture_bounds_are_preserved(for_capture, bound):
    buffers = InputBuffers(64, 256, torch.device("cpu"))
    batch = InputBatch.make_dummy(64, 256, buffers, max_query_len=bound)
    state = OmniModelState.__new__(OmniModelState)
    with patch.object(DefaultModelState, "prepare_attn", return_value={}) as backend:
        state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (),
            torch.empty(0),
            [],
            KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
            for_capture=for_capture,
        )
    assert backend.call_args.args[0] is batch
    assert backend.call_args.kwargs["for_capture"] is for_capture
