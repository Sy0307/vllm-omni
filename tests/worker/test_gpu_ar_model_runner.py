import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingType

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class _SamplingParams:
    def __init__(self, **overrides):
        self.logprobs = None
        self.prompt_logprobs = None
        self.stop = None
        self.stop_token_id = None
        self.stop_token_ids = None
        self.all_stop_token_ids = None
        self.ignore_eos = False
        self.min_tokens = 0
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = -1
        self.min_p = 0.0
        self.seed = None
        self.sampling_type = None
        self.do_sample = False
        self.use_beam_search = False
        self.n = 1
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.repetition_penalty = 1.0
        self.encoder_repetition_penalty = 1.0
        self.logits_processors = None
        self.allowed_token_ids = None
        self.structured_outputs = None
        self.guided_decoding = None
        self.extra_args = None
        for key, value in overrides.items():
            setattr(self, key, value)


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 50},
        {"min_p": 0.1},
        {"seed": 1234},
        {"sampling_type": SamplingType.RANDOM},
        {"sampling_type": SamplingType.RANDOM_SEED},
        {"do_sample": True},
        {"early_stopping": True},
        {"early_stopping": "never"},
        {"use_beam_search": True},
        {"n": 2},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"repetition_penalty": 1.1},
        {"encoder_repetition_penalty": 1.1},
        {"logits_processors": [object()]},
        {"allowed_token_ids": [1, 2]},
        {"prompt_logprobs": 1},
        {"logprobs": 1},
        {"stop": ["<eos>"]},
        {"stop_token_ids": [2]},
        {"stop_token_id": 2},
        {"all_stop_token_ids": {2}},
        {"ignore_eos": True},
        {"min_tokens": 1},
        {"structured_outputs": object()},
        {"guided_decoding": object()},
        {"extra_args": {"bad_words": ["x"]}},
        {"unknown_constraint": object()},
    ],
)
def test_acoustic_inner_loop_fast_path_rejects_non_greedy_sampling_params(overrides):
    runner = _make_runner()
    runner.requests["rid"].sampling_params = _SamplingParams(**overrides)

    assert not runner._should_run_fast_acoustic_inner_loop(_SchedulerOutput(), None)


def test_store_fast_acoustic_token_keeps_gpu_tensor_until_output_boundary():
    runner = object.__new__(GPUARModelRunner)
    runner._fast_acoustic_sampled_token_ids = torch.empty(3, dtype=torch.int64)

    runner._store_fast_acoustic_token(0, torch.tensor([11], dtype=torch.int64))
    runner._store_fast_acoustic_token(1, torch.tensor(12, dtype=torch.int64))

    assert runner._fast_acoustic_sampled_token_ids.tolist()[:2] == [11, 12]


def test_fast_acoustic_loop_has_single_hidden_append_and_no_per_step_cpu_item():
    source = inspect.getsource(GPUARModelRunner._run_acoustic_inner_loop)

    assert source.count("hidden_chunks.append(hidden_states[:1])") == 1
    fast_branch = source.split("if use_fast_path:", 1)[1].split("else:", 1)[0]
    assert ".item()" not in fast_branch
