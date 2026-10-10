# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode-burst configuration and per-step batch layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.worker_v2.decode_burst import BurstOutputBatch, burst_step_batch, decode_burst_steps

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "v2,mode,processor,expected",
    [
        (False, "duplex", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 1),
        (True, "turn", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 1),
        (True, "duplex", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 4),
        (True, "duplex", "qwen3_omni.thinker2talker", 1),
        (True, "duplex", None, 1),
    ],
)
def test_decode_burst_steps(mocker, v2, mode, processor, expected):
    config = VllmConfig()
    config.model_config = mocker.Mock(
        spec=OmniModelConfig,
        use_v2_model_runner=v2,
        session_mode=mode,
        custom_process_next_stage_input_func=(
            f"vllm_omni.model_executor.stage_input_processors.{processor}" if processor else None
        ),
    )
    assert decode_burst_steps(config) == expected


def test_uniform_burst_layout_keeps_each_request_contiguous():
    batch = BurstOutputBatch.uniform(num_reqs=3, steps=4)
    np.testing.assert_array_equal(batch.query_start_loc_np, [0, 4, 8, 12])
    np.testing.assert_array_equal(batch.num_scheduled_tokens, [4, 4, 4])
    assert batch.num_tokens_after_padding == 12
    assert not batch.is_prefilling_np.any()


def test_burst_step_batch_advances_only_host_lengths():
    from dataclasses import dataclass

    @dataclass
    class _Batch:
        num_reqs: int
        num_computed_tokens_np: np.ndarray
        seq_lens_cpu_upper_bound: torch.Tensor

    batch = _Batch(2, np.array([10, 20]), torch.tensor([11, 21, 0]))
    step = burst_step_batch(batch, 2)
    np.testing.assert_array_equal(step.num_computed_tokens_np, [12, 22])
    assert step.seq_lens_cpu_upper_bound.tolist() == [13, 23, 0]
    # The scheduled batch is left untouched.
    assert batch.seq_lens_cpu_upper_bound.tolist() == [11, 21, 0]
