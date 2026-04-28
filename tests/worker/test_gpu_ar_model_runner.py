from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class _SamplingParams:
    logprobs = None
    stop_token_id = None
    stop_token_ids = None


class _SchedulerOutput:
    def __init__(self):
        self.omni_acoustic_inner_loop_extra_slots = {"rid": 2}
        self.num_scheduled_tokens = {"rid": 3}
        self.scheduled_spec_decode_tokens = {}
        self.scheduled_encoder_inputs = {}


class _InputBatch:
    num_reqs = 1
    req_ids = ["rid"]
    num_computed_tokens_cpu = [4]


def _make_runner():
    runner = object.__new__(GPUARModelRunner)
    runner.use_async_scheduling = False
    runner.omni_prefix_cache = None
    runner.input_batch = _InputBatch()
    runner.requests = {"rid": SimpleNamespace(sampling_params=_SamplingParams())}
    runner.speculative_config = None
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner.model = SimpleNamespace(greedy_group0_tokens=lambda hidden: hidden.argmax(dim=-1))
    return runner


def test_acoustic_inner_loop_fast_path_requires_greedy_model_helper():
    runner = _make_runner()
    scheduler_output = _SchedulerOutput()

    assert runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)

    runner.model = SimpleNamespace()

    assert not runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)


@pytest.mark.parametrize(
    "attr,value",
    [
        ("use_async_scheduling", True),
        ("omni_prefix_cache", object()),
        ("speculative_config", object()),
    ],
)
def test_acoustic_inner_loop_fast_path_rejects_unsupported_runner_modes(attr, value):
    runner = _make_runner()
    setattr(runner, attr, value)

    assert not runner._should_run_fast_acoustic_inner_loop(_SchedulerOutput(), None)


def test_acoustic_inner_loop_fast_path_rejects_spec_encoder_and_ubatching():
    runner = _make_runner()
    scheduler_output = _SchedulerOutput()
    scheduler_output.scheduled_spec_decode_tokens = {"rid": [1]}
    assert not runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)

    scheduler_output = _SchedulerOutput()
    scheduler_output.scheduled_encoder_inputs = {"rid": object()}
    assert not runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)

    runner = _make_runner()
    runner.parallel_config.use_ubatching = True
    assert not runner._should_run_fast_acoustic_inner_loop(_SchedulerOutput(), None)


def test_store_fast_acoustic_token_keeps_gpu_tensor_until_output_boundary():
    runner = object.__new__(GPUARModelRunner)
    runner._fast_acoustic_sampled_token_ids = torch.empty(3, dtype=torch.int64)

    runner._store_fast_acoustic_token(0, torch.tensor([11], dtype=torch.int64))
    runner._store_fast_acoustic_token(1, torch.tensor(12, dtype=torch.int64))

    assert runner._fast_acoustic_sampled_token_ids.tolist()[:2] == [11, 12]
