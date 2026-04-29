import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingType

from vllm_omni.worker import gpu_ar_model_runner as ar_runner
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


class _Qwen3TTSNVModel:
    def greedy_group0_tokens(self, hidden):
        return hidden.argmax(dim=-1)


_Qwen3TTSNVModel.__module__ = "vllm_omni.model_executor.models.qwen3_tts_nv.qwen3_tts_talker_nv"


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
    runner.parallel_config = SimpleNamespace(use_ubatching=False, num_ubatches=1)
    runner.model = _Qwen3TTSNVModel()
    return runner


def test_acoustic_inner_loop_fast_path_requires_greedy_model_helper():
    runner = _make_runner()
    scheduler_output = _SchedulerOutput()

    assert runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)

    runner.model = SimpleNamespace()

    assert not runner._should_run_fast_acoustic_inner_loop(scheduler_output, None)


def test_acoustic_inner_loop_fast_path_requires_qwen3_tts_nv_model():
    runner = _make_runner()
    runner.model = SimpleNamespace(greedy_group0_tokens=lambda hidden: hidden.argmax(dim=-1))

    assert not runner._should_run_fast_acoustic_inner_loop(_SchedulerOutput(), None)


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
    source = inspect.getsource(GPUARModelRunner._run_fast_qwen3_tts_nv_acoustic_inner_loop)

    assert source.count("hidden_chunks.append(hidden_states[:1])") == 1
    assert ".item()" not in source


def test_fast_acoustic_loop_hoists_shape_stable_prep_and_keeps_dynamic_prep_per_step(monkeypatch):
    runner = _make_runner()
    scheduler_output = _SchedulerOutput()
    scheduler_output.total_num_scheduled_tokens = 3
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="multi"))
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2))
    runner.supports_mm_inputs = False
    runner.parallel_config.num_ubatches = 1
    runner.input_batch.num_computed_tokens_cpu = [4]
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([4])
    runner.input_batch.sampling_metadata = None
    runner.input_ids = SimpleNamespace(gpu=torch.empty(1, dtype=torch.int64))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.attn_groups = []

    counts = {
        "determine": 0,
        "ubatch": 0,
        "prepare_inputs": 0,
        "slot_mappings": 0,
        "attention_metadata": 0,
        "preprocess": 0,
        "prepare_runner_inputs": 0,
    }

    class _KVContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    runner.maybe_get_kv_connector_output = lambda scheduler_output: _KVContext()
    runner._make_single_token_scheduler_output = lambda scheduler_output, req_id: SimpleNamespace(
        num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={}
    )
    def prepare_inputs(sub_output, num_scheduled_tokens_np):
        counts["prepare_inputs"] += 1
        return torch.tensor([0]), None

    runner._prepare_inputs = prepare_inputs

    def determine(**kwargs):
        counts["determine"] += 1
        return None, SimpleNamespace(num_tokens=1, num_reqs=1), False, 1, None

    runner._determine_batch_execution_and_padding = determine

    def ubatch(*args):
        counts["ubatch"] += 1
        return None, None

    monkeypatch.setattr(ar_runner, "maybe_create_ubatch_slices", ubatch)
    def get_slot_mappings(**kwargs):
        counts["slot_mappings"] += 1
        return {}, None

    runner._get_slot_mappings = get_slot_mappings

    def build_attention_metadata(**kwargs):
        counts["attention_metadata"] += 1
        return None, None

    runner._build_attention_metadata = build_attention_metadata

    def preprocess(sub_output, num_tokens_padded, intermediate_tensors):
        counts["preprocess"] += 1
        return (
            torch.zeros(1, dtype=torch.int64),
            None,
            torch.zeros(1, dtype=torch.int64),
            intermediate_tensors,
            {},
            None,
        )

    runner._preprocess = preprocess
    runner._model_forward = lambda **kwargs: torch.ones(1, 2)
    runner.extract_multimodal_outputs = lambda model_output: (model_output, {})
    runner.model.greedy_group0_tokens = lambda hidden: torch.tensor([1])

    def prepare_runner_inputs(**kwargs):
        counts["prepare_runner_inputs"] += 1
        return kwargs["input_ids"], kwargs["positions"]

    runner.model.prepare_runner_inputs = prepare_runner_inputs
    runner._ensure_fast_acoustic_token_buffer = lambda max_steps: setattr(
        runner, "_fast_acoustic_sampled_token_ids", torch.empty(max_steps, dtype=torch.int64)
    )
    runner._process_additional_information_updates = lambda *args: None

    output = runner._run_acoustic_inner_loop(scheduler_output, None, None, None)

    assert output.sampled_token_ids == [[1, 1, 1]]
    assert counts["determine"] == 1
    assert counts["ubatch"] == 1
    assert counts["prepare_inputs"] == 3
    assert counts["slot_mappings"] == 3
    assert counts["attention_metadata"] == 3
    assert counts["preprocess"] == 3
    assert counts["prepare_runner_inputs"] == 3
