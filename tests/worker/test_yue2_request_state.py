# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2's worker-side sampler follows retained engine outputs on recompute."""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter
from vllm_omni.model_executor.models.yue2 import yue2 as y
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def model(mocker):
    """Run real state/sampling control flow with CPU staging and fixed draws."""
    model = object.__new__(y.Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._states = {}
    model._pending = None
    model._step_rows = []
    model._step_discard = None
    model._decode_t0 = {}
    model._last_mm = None
    model._max_model_len = 4096
    model._stage = torch.empty(1024, dtype=torch.long)
    model._pinned = torch.empty((2, 64), dtype=torch.long)
    model._synthesis = mocker.Mock(spec=y._SynthesisQueue)
    mocker.patch.object(model, "_finish_request_safely")
    mocker.patch.object(model, "_sampler_graph", return_value=None)
    # CPU copies complete synchronously; this replaces only CUDA completion.
    event = mocker.Mock(spec=torch.Event)
    mocker.patch.object(y.torch, "Event", return_value=event)
    mocker.patch.object(y, "sample_rows", return_value=(torch.tensor([y.CODEC_OFFSET + 7]), torch.tensor([False])))
    constants = y._RowConstants("r", "semantic", 1.0, 0.9, 100, 1.1, 2, 1, 3, 17, False, 8)
    model._states["r"] = y._RequestState("r", constants, prompt_len=3, prefix_ids=[1, 2, 3])
    return model


def test_pending_draw_is_discarded_after_chunked_recompute(model) -> None:
    """A preempted in-flight draw never advances history or the frame limit."""
    logits = torch.zeros((1, y.VOCAB_SIZE))
    model._step_rows = [("r", 0, 3)]
    model.sample(logits, None)
    state = model._states["r"]
    # Other requests run while r waits for KV space. Resolving the host copy
    # may resolve a tentative token, but the resumed row must roll it back.
    model._step_rows = []
    model.sample(logits[:0], None)
    assert state.job is None
    # A partial recompute must neither accept the pending draw nor draw again.
    model._step_rows = [("r", 0, 2)]
    model._step_discard = [True]
    model.sample(logits, None)
    assert state.job is None
    # Completing recompute has retained zero outputs: discard the old draw.
    model._step_rows = [("r", 2, 1)]
    model._step_discard = [False]
    model.sample(logits, None)
    assert state.history == []
    # Only this new draw is retained in the next decode input.
    model._step_rows = [("r", 3, 1)]
    model.sample(logits, None)
    assert state.history == [y.CODEC_OFFSET + 7]
    assert not state.finish_ready and not state.truncated


@pytest.mark.parametrize("token", [y.MUSIC_END, y.CODEC_OFFSET + 7])
def test_discarded_terminal_draw_does_not_start_synthesis(model, mocker, token: int) -> None:
    """Dropped EOS/frame-limit draws cannot prematurely finish a song."""
    state = model._states["r"]
    state.history = [y.CODEC_OFFSET + 2] * 2
    mocker.patch.object(y, "sample_rows", return_value=(torch.tensor([token]), torch.tensor([False])))
    logits = torch.zeros((1, y.VOCAB_SIZE))
    model._step_rows = [("r", 4, 1)]
    model.sample(logits, None)
    model._step_rows = []
    model.sample(logits[:0], None)
    assert state.job is None
    model._step_rows = [("r", 0, 5)]
    model.sample(logits, None)
    assert state.history == [y.CODEC_OFFSET + 2] * 2
    assert not state.finish_ready and not state.truncated and not state.end_drawn
    assert state.job is None and not state.finished
    model._finish_request_safely.assert_not_called()


def test_recompute_clears_terminal_state_and_cancels_stale_job(model) -> None:
    """Rolling back a drawn end also releases synthesis and revives the row."""
    state = model._states["r"]
    state.history = [y.CODEC_OFFSET + 2] * 2
    state.end_drawn = True
    state.finished = state.finish_ready = True
    state.hold_steps = 3

    def song(keep):
        yield
        return torch.zeros((2, 0))

    job = y._SynthesisJob("r", song)
    state.job = job
    model._reconcile_history(state, 2)
    model._synthesis.cancel.assert_called_once_with(job)
    assert state.job is None and state.hold_steps == 0
    assert not state.finished and not state.finish_ready and not state.truncated and not state.end_drawn


def test_accepted_draw_still_reaches_frame_limit(model) -> None:
    """Confirmed codec frames retain the existing delayed finish behavior."""
    logits = torch.zeros((1, y.VOCAB_SIZE))
    for comp, span in ((0, 3), (3, 1), (4, 1)):
        model._step_rows = [("r", comp, span)]
        model.sample(logits, None)
    state = model._states["r"]
    assert len(state.history) == 2
    model._resolve_pending()
    model._reconcile_history(state, 3)
    assert len(state.history) == 3
    assert state.finish_ready and state.truncated


def test_real_prepare_hook_admits_reordered_requests(model) -> None:
    """Runner opt-in metadata reaches the actual YuE2 admission hook."""
    runner = object.__new__(GPUARModelRunner)
    runner.model = model
    runner.requests = {}
    for rid, seed in (("a", 7), ("b", 9)):
        runner.requests[rid] = CachedRequestState(
            req_id=rid,
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=SamplingParams(extra_args={y.KEY_PREFIX_IDS: [1, 2, 3], y.KEY_SEED: seed}),
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
    runner.discard_request_mask = CpuGpuBuffer(2, dtype=torch.bool, device=torch.device("cpu"), pin_memory=False)
    runner.discard_request_mask.np[:] = [True, False]
    ids = torch.tensor([1, 2, 1, 2, 3], dtype=torch.int32)
    positions = torch.tensor([0, 1, 0, 1, 2])
    runner._call_prepare_runner_inputs(
        model.prepare_runner_inputs,
        req_ids=["b", "a"],
        input_ids=ids,
        positions=positions,
        inputs_embeds=None,
        num_computed_tokens=np.array([0, 0]),
        num_scheduled_tokens=np.array([2, 3]),
        input_ids_buffer=ids,
    )
    assert model._states["b"].constants.seed == 9
    assert model._states["a"].constants.seed == 7
    assert model._states["b"].prompt_len == 3
    assert model._states["b"].prefix_ids == [1, 2, 3]
    assert model._step_discard == [True, False]


@pytest.mark.parametrize("spare", [0, 1, 4095, 4096, 5000])
def test_hold_budget_matches_engine_headroom(spare: int) -> None:
    """Adapter metadata advertises only the HOLD steps left after clipping."""
    frames = 250
    prompt = {"prompt_token_ids": [1] * (y.CONTEXT - frames - 1 - spare)}
    adapter = object.__new__(Yue2Adapter)
    request = OpenAICreateSpeechRequest(model="m-a-p/YuE2-3B", input="lyrics", max_new_tokens=frames)
    params = adapter.apply_sampling_overrides([SamplingParams()], request, prompt)[0]
    assert params.extra_args[y.KEY_MAX_HOLD_STEPS] == min(spare, 4096)
    assert params.max_tokens == frames + 1 + min(spare, 4096)
