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
        self.bad_words = None
        self.logit_bias = None
        self.extra_args = None
        for key, value in overrides.items():
            setattr(self, key, value)


class _Qwen3TTSNVModel:
    def __init__(self):
        self._prev_hidden_buffer = torch.zeros(1, 2)

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
    runner.parallel_config = SimpleNamespace(use_ubatching=False, num_ubatches=1, data_parallel_size=1)
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
        {"bad_words": ["x"]},
        {"logit_bias": {1: 1.0}},
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


def test_fast_acoustic_graph_state_prefills_fixed_k_gpu_buffers():
    runner = object.__new__(GPUARModelRunner)
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.hidden_size = 2

    state = runner._prepare_fast_acoustic_graph_state(
        max_steps=4,
        start_num_computed=10,
        precomputed_slot_mappings_by_group={0: torch.tensor([101, 102, 103, 104], dtype=torch.int64)},
    )

    assert state.positions_buffer.tolist() == [10, 11, 12, 13]
    assert state.seq_lens_buffer.tolist() == [11, 12, 13, 14]
    assert state.slot_mapping_buffers[0].tolist() == [101, 102, 103, 104]
    assert state.sampled_tokens_buffer.dtype == torch.int64
    assert state.valid_mask.tolist() == [False, False, False, False]
    assert state.valid_count.shape == (1,)
    assert int(state.valid_count[0]) == 0
    assert state.hidden_buffer.shape == (4, 2)
    assert state.mm_buffers == {}
    assert state.input_ids_buffer.shape == (1,)
    assert state.positions_live_buffer.shape == (1,)


@pytest.mark.parametrize("generated", [0, 1, 2])
def test_fast_acoustic_graph_state_masks_post_stop_slots_to_scratch(generated):
    runner = object.__new__(GPUARModelRunner)
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.hidden_size = 2

    state = runner._prepare_fast_acoustic_graph_state(
        max_steps=4,
        start_num_computed=10,
        precomputed_slot_mappings_by_group={0: torch.tensor([101, 102, 103, 104], dtype=torch.int64)},
    )
    state.valid_mask[:generated] = True
    runner._mask_fast_acoustic_post_stop_slots(state)

    scratch_slot = int(state.scratch_slot[0])
    assert scratch_slot not in {101, 102, 103, 104}
    assert state.slot_mapping_buffers[0].tolist() == [
        101 + idx if idx < generated else scratch_slot for idx in range(4)
    ]


def test_fast_acoustic_gpu_step_state_uses_static_buffers_without_cpu_mirrors():
    runner = object.__new__(GPUARModelRunner)
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.hidden_size = 2
    runner.seq_lens = SimpleNamespace(
        np=torch.zeros(4, dtype=torch.int32).numpy(),
        gpu=torch.zeros(4, dtype=torch.int32),
    )
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=[999],
        num_computed_tokens_cpu_tensor=torch.tensor([999], dtype=torch.int32),
    )
    state = runner._prepare_fast_acoustic_graph_state(
        max_steps=3,
        start_num_computed=20,
        precomputed_slot_mappings_by_group={0: torch.tensor([201, 202, 203], dtype=torch.int64)},
    )
    state.valid_mask[2] = True
    positions = torch.zeros(1, dtype=torch.int64)
    live_slots = {0: torch.zeros(1, dtype=torch.int64)}

    runner._set_fast_acoustic_decode_step_state_gpu(
        positions=positions,
        slot_mappings_by_group=live_slots,
        graph_state=state,
        step=2,
    )

    assert int(positions[0]) == 22
    assert int(runner.seq_lens.gpu[0]) == 23
    assert int(live_slots[0][0]) == 203
    assert runner.input_batch.num_computed_tokens_cpu[0] == 999
    assert int(runner.input_batch.num_computed_tokens_cpu_tensor[0]) == 999
    assert runner.seq_lens.np[0] == 0


def test_fast_acoustic_loop_has_no_python_break_or_per_step_cpu_sync():
    source = inspect.getsource(GPUARModelRunner._run_fast_qwen3_tts_nv_acoustic_inner_loop)
    capture_start = "# Fast acoustic K loop begins."
    capture_end = "# Fast acoustic K loop ends."

    start_idx = source.index(capture_start)
    end_idx = source.index(capture_end)
    capture_scope = source[start_idx:end_idx]
    graph_ready_scope = capture_scope[capture_scope.index("if graph_ready:") : capture_scope.index("else:")]
    output_construction = source[end_idx:]

    assert "break" not in capture_scope
    assert "current_pos = start_num_computed + inner_idx" not in capture_scope
    assert "current_pos=" not in capture_scope
    assert "_set_fast_acoustic_decode_step_state(" not in capture_scope
    assert "_set_fast_acoustic_decode_step_state_gpu(" in capture_scope
    assert "hidden_chunks.append" not in graph_ready_scope
    assert "multimodal_chunks.setdefault" not in graph_ready_scope
    assert ".item()" not in capture_scope
    assert ".cpu()" not in capture_scope
    assert '.to("cpu")' not in capture_scope
    assert 'valid_count_tensor.detach().to("cpu")' in output_construction


def test_graph_ready_fast_acoustic_captures_then_replays_same_shape(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ACOUSTIC_GRAPH_READY", "1")
    runner = _make_runner()
    runner.requests["rid"].sampling_params = _SamplingParams(stop_token_ids=[7])
    scheduler_output = _SchedulerOutput()
    scheduler_output.num_scheduled_tokens = {"rid": 4}
    scheduler_output.total_num_scheduled_tokens = 4
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.hidden_size = 2
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type="multi"),
        parallel_config=runner.parallel_config,
        compilation_config=SimpleNamespace(
            static_forward_context=None,
            fast_moe_cold_start=False,
        ),
    )
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2))
    runner.supports_mm_inputs = False
    runner.parallel_config.num_ubatches = 1
    runner.input_batch.num_computed_tokens_cpu = [4]
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([4])
    runner.input_batch.sampling_metadata = None
    runner.input_ids = SimpleNamespace(gpu=torch.empty(1, dtype=torch.int64))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.attn_groups = []
    runner.query_start_loc = SimpleNamespace(
        np=torch.zeros(4, dtype=torch.int32).numpy(),
        copy_to_gpu=lambda: None,
    )
    runner.seq_lens = SimpleNamespace(
        np=torch.zeros(4, dtype=torch.int32).numpy(),
        gpu=torch.zeros(4, dtype=torch.int32),
        copy_to_gpu=lambda: None,
    )

    class _KVContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    runner.maybe_get_kv_connector_output = lambda scheduler_output: _KVContext()
    runner._make_single_token_scheduler_output = lambda scheduler_output, req_id: SimpleNamespace(
        num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={}
    )
    runner._prepare_inputs = lambda sub_output, num_scheduled_tokens_np: (torch.tensor([0]), None)
    runner._determine_batch_execution_and_padding = lambda **kwargs: (
        None,
        SimpleNamespace(num_tokens=1, num_reqs=1),
        False,
        1,
        None,
    )
    monkeypatch.setattr(ar_runner, "maybe_create_ubatch_slices", lambda *args: (None, None))
    runner._get_slot_mappings = lambda **kwargs: ({}, None)
    runner._build_attention_metadata = lambda **kwargs: (None, None)
    runner._preprocess = lambda sub_output, num_tokens_padded, intermediate_tensors: (
        torch.zeros(1, dtype=torch.int64),
        None,
        torch.zeros(1, dtype=torch.int64),
        intermediate_tensors,
        {},
        None,
    )
    forward_calls = []
    hidden_by_pos = {
        4: torch.tensor([[1.0, 4.0]]),
        5: torch.tensor([[2.0, 5.0]]),
        6: torch.tensor([[99.0, 99.0]]),
        7: torch.tensor([[100.0, 100.0]]),
    }

    def model_forward(**kwargs):
        pos = int(kwargs["positions"][0])
        forward_calls.append(pos)
        return hidden_by_pos[pos].clone()

    runner._model_forward = model_forward
    runner.extract_multimodal_outputs = lambda model_output: (model_output, {})
    tokens = iter([5, 7, 9, 10])
    runner.model.greedy_group0_tokens = lambda hidden: torch.tensor([next(tokens)])
    runner._ensure_fast_acoustic_token_buffer = lambda max_steps: setattr(
        runner, "_fast_acoustic_sampled_token_ids", torch.empty(max_steps, dtype=torch.int64)
    )
    additional_updates = []

    def process_additional_information_updates(hidden_states, multimodal_outputs, num_scheduled_tokens_np, sub_output):
        additional_updates.append((hidden_states.clone(), int(num_scheduled_tokens_np[0])))

    runner._process_additional_information_updates = process_additional_information_updates
    corrected = []

    def apply_correction(start_num_computed, generated, max_steps):
        corrected.append(("rid", generated, max_steps))
        return start_num_computed + generated

    runner._apply_fast_acoustic_corrected_num_computed_tokens = apply_correction
    capture_calls = []
    replay_calls = []

    def capture_acoustic_graph(body):
        capture_calls.append(True)
        body()
        state = runner._fast_acoustic_graph_state
        token_snapshot = state.sampled_tokens_buffer.clone()
        valid_mask_snapshot = state.valid_mask.clone()
        valid_count_snapshot = state.valid_count.clone()
        hidden_snapshot = state.hidden_buffer.clone()
        prev_hidden_snapshot = runner.model._prev_hidden_buffer.clone()

        def replay():
            state.sampled_tokens_buffer.copy_(token_snapshot)
            state.valid_mask.copy_(valid_mask_snapshot)
            state.valid_count.copy_(valid_count_snapshot)
            state.hidden_buffer.copy_(hidden_snapshot)
            runner.model._prev_hidden_buffer.copy_(prev_hidden_snapshot)

        state.graph = SimpleNamespace(replay=replay)
        state.graph_max_steps = state.max_steps
        return state

    def replay_acoustic_graph():
        replay_calls.append(True)
        state = runner._fast_acoustic_graph_state
        state.graph.replay()
        return state

    runner.capture_acoustic_graph = capture_acoustic_graph
    runner.replay_acoustic_graph = replay_acoustic_graph

    output = runner._run_acoustic_inner_loop(scheduler_output, None, None, None)

    assert capture_calls == [True]
    assert replay_calls == []
    assert forward_calls == [4, 5, 6, 7]
    assert runner._last_acoustic_inner_loop_path == "fast_graph_ready"
    assert runner._fast_acoustic_valid_token_mask.tolist() == [True, True, False, False]
    assert int(runner._fast_acoustic_valid_count.cpu()) == 2
    assert output.sampled_token_ids == [[5, 7]]
    assert torch.equal(output.pooler_output[0]["hidden"], torch.tensor([[1.0, 4.0], [2.0, 5.0]]))
    assert len(additional_updates) == 1
    assert torch.equal(additional_updates[0][0], torch.tensor([[1.0, 4.0], [2.0, 5.0]]))
    assert additional_updates[0][1] == 2
    assert torch.equal(runner.model._prev_hidden_buffer, torch.tensor([[2.0, 5.0]]))
    assert runner.input_batch.num_computed_tokens_cpu[0] == 6
    assert corrected == [("rid", 2, 4)]

    runner.input_batch.num_computed_tokens_cpu = [4]
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([4])
    output = runner._run_acoustic_inner_loop(scheduler_output, None, None, None)

    assert capture_calls == [True]
    assert replay_calls == [True]
    assert forward_calls == [4, 5, 6, 7]
    assert output.sampled_token_ids == [[5, 7]]


def test_acoustic_inner_loop_output_keeps_audio_codes_gpu_resident_without_hidden_cpu_payload(monkeypatch):
    runner = object.__new__(GPUARModelRunner)
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2))
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="audio"))
    runner.supports_mm_inputs = False

    def fail_build_mm_cpu(multimodal_outputs):
        raise AssertionError("audio_codes should stay GPU-resident until output processing")

    monkeypatch.setattr(ar_runner, "build_mm_cpu", fail_build_mm_cpu)

    output = runner._build_acoustic_inner_loop_output(
        req_id="rid",
        generated=2,
        sampled_token_ids=[3, 4],
        hidden_chunks=[torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])],
        multimodal_chunks={
            "audio_codes": [torch.tensor([[11, 12]]), torch.tensor([[13, 14]])],
        },
        kv_connector_output=None,
        ec_connector_output=None,
        num_nans_in_logits={},
        cudagraph_stats=None,
    )

    payload = output.pooler_output[0]
    assert "hidden" not in payload
    assert torch.equal(payload["audio_codes"], torch.tensor([[11, 12], [13, 14]]))


def test_acoustic_inner_loop_text_output_skips_payload_concat(monkeypatch):
    runner = object.__new__(GPUARModelRunner)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    runner.supports_mm_inputs = False

    def fail_cat(*args, **kwargs):
        raise AssertionError("text outputs should not build hidden/audio payloads")

    monkeypatch.setattr(torch, "cat", fail_cat)

    output = runner._build_acoustic_inner_loop_output(
        req_id="rid",
        generated=1,
        sampled_token_ids=[3],
        hidden_chunks=[torch.tensor([[1.0, 2.0]])],
        multimodal_chunks={"audio_codes": [torch.tensor([[11, 12]])]},
        kv_connector_output=None,
        ec_connector_output=None,
        num_nans_in_logits={},
        cudagraph_stats=None,
    )

    assert output.pooler_output is None


def test_acoustic_inner_loop_output_builder_has_no_eager_mm_cpu_payload_copy():
    source = inspect.getsource(GPUARModelRunner._build_acoustic_inner_loop_output)

    assert "build_mm_cpu" not in source
    assert '.to("cpu")' not in source


def test_fast_acoustic_loop_hoists_generic_prep_out_of_substep_loop(monkeypatch):
    runner = _make_runner()
    scheduler_output = _SchedulerOutput()
    scheduler_output.total_num_scheduled_tokens = 3
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.hidden_size = 2
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type="multi"),
        parallel_config=runner.parallel_config,
        compilation_config=SimpleNamespace(
            static_forward_context=None,
            fast_moe_cold_start=False,
        ),
    )
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2))
    runner.supports_mm_inputs = False
    runner.parallel_config.num_ubatches = 1
    runner.input_batch.num_computed_tokens_cpu = [4]
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([4])
    runner.input_batch.sampling_metadata = None
    runner.input_ids = SimpleNamespace(gpu=torch.empty(1, dtype=torch.int64))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.attn_groups = []
    runner.query_start_loc = SimpleNamespace(
        np=torch.zeros(4, dtype=torch.int32).numpy(),
        copy_to_gpu=lambda: None,
    )
    runner.seq_lens = SimpleNamespace(
        np=torch.zeros(4, dtype=torch.int32).numpy(),
        gpu=torch.zeros(4, dtype=torch.int32),
        copy_to_gpu=lambda: None,
    )

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
    assert counts["prepare_inputs"] == 1
    assert counts["slot_mappings"] == 1
    assert counts["attention_metadata"] == 1
    assert counts["preprocess"] == 1
    assert counts["prepare_runner_inputs"] == 1
