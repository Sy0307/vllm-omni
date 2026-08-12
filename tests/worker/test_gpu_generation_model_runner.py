import contextlib
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_omni.worker.gpu_generation_model_runner as gen_runner_module
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self, req_ids=("req-1",)):
        self.req_ids = list(req_ids)
        self.req_id_to_index = {req_id: index for index, req_id in enumerate(req_ids)}
        self.num_reqs = len(self.req_ids)
        self.vocab_size = 10


def _make_runner(multimodal_outputs, *, req_ids=("req-1",)):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch(req_ids)
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
    runner._generation_finished_req_ids = set()
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.multimodal_outputs is None
    assert output.inter_stage_outputs is None


def test_sample_tokens_dict_maps_only_completed_rows():
    completed_audio = torch.ones(4)
    sample_rate = torch.tensor(22050, dtype=torch.int32)
    runner = _make_runner(
        {
            "audio": [completed_audio, None],
            "sr": [sample_rate, None],
        },
        req_ids=("done", "pending"),
    )

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[1] is None
    torch.testing.assert_close(output.multimodal_outputs[0]["audio"], completed_audio)
    torch.testing.assert_close(output.multimodal_outputs[0]["sr"], sample_rate)


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert "audio" in output.multimodal_outputs[0]
    assert "unused" not in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0]["audio"].shape == (1, 4)


class _StubSchedulerOutput:
    def __init__(self, total_num_scheduled_tokens):
        self.total_num_scheduled_tokens = total_num_scheduled_tokens
        self.num_scheduled_tokens = {"req-1": total_num_scheduled_tokens}
        self.finished_req_ids = set()
        self.kv_connector_metadata = None
        self.scheduled_new_reqs = []


def _make_guard_runner():
    # Stubbed far enough that a span escaping the guard reaches the real
    # `_prepare_inputs`, i.e. fails the way the reported crash does.
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.cache_config = CacheConfig()
    runner.input_batch = _DummyInputBatch()
    runner.model = object()
    runner._update_states = lambda scheduler_output: None
    runner.synchronize_input_prep = contextlib.nullcontext
    runner.attach_omni_connector_output = lambda result: result
    return runner


@pytest.mark.parametrize("total_num_scheduled_tokens", [0, 1])
def test_execute_model_runs_stepwise_without_input_prep(
    monkeypatch,
    total_num_scheduled_tokens,
):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()
    runner.device = torch.device("cpu")
    runner.model_intermediate_buffer = {
        "req-a": {"value": "a"},
        "req-b": {"value": "b"},
    }
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        take_finished_request_ids=lambda: {"req-b"},
    )
    captured = {}

    def model_forward(**kwargs):
        captured.update(kwargs)
        return {"audio": [None, torch.ones(4)]}

    runner._model_forward = model_forward
    runner.extract_multimodal_outputs = lambda outputs: (None, outputs)
    sampled_output = object()
    sample_calls = []
    runner.sample_tokens = lambda: sample_calls.append(True) or sampled_output
    runner._prepare_inputs = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("zero-token stepwise work reached _prepare_inputs")
    )
    scheduler_output = _StubSchedulerOutput(total_num_scheduled_tokens)
    scheduler_output.stepwise_req_ids = ["req-a", "req-b"]

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    if total_num_scheduled_tokens == 0:
        assert output is sampled_output
        assert sample_calls == [True]
    else:
        assert output is None
        assert sample_calls == []
    assert captured["request_ids"] == ["req-a", "req-b"]
    assert captured["model_intermediate_buffer"] == [
        {"value": "a"},
        {"value": "b"},
    ]
    assert captured["input_ids"].numel() == 0
    assert runner._generation_finished_req_ids == {"req-b"}
    assert runner._stepwise_output_req_ids == ["req-a", "req-b"]


def test_execute_model_flushes_finished_state_without_stepwise_work(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()
    flush_calls = []
    runner.model = SimpleNamespace(
        flush_finished_requests=lambda: flush_calls.append(True),
    )
    runner.attach_omni_connector_output = lambda output: output
    scheduler_output = _StubSchedulerOutput(0)
    scheduler_output.stepwise_req_ids = []
    scheduler_output.finished_req_ids = {"done"}
    runner.model.on_requests_finished = lambda request_ids: None

    GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert flush_calls == [True]


def test_execute_model_drops_cancelled_request_from_stepwise_work(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()
    lifecycle_calls = []
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        on_requests_finished=lambda request_ids: lifecycle_calls.append(("finish", set(request_ids))),
        flush_finished_requests=lambda: lifecycle_calls.append(("flush", None)),
    )
    runner.attach_omni_connector_output = lambda output: output
    runner._execute_stepwise_generation = lambda *_args: (_ for _ in ()).throw(
        AssertionError("cancelled request reached stepwise forward")
    )
    scheduler_output = _StubSchedulerOutput(0)
    scheduler_output.num_scheduled_tokens = {}
    scheduler_output.stepwise_req_ids = ["cancelled"]
    scheduler_output.finished_req_ids = {"cancelled"}

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert lifecycle_calls == [
        ("finish", {"cancelled"}),
        ("flush", None),
    ]


def test_execute_model_reinitializes_resubmitted_stepwise_request(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()
    lifecycle_calls = []
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        on_requests_finished=lambda request_ids: lifecycle_calls.append(("finish", set(request_ids))),
        flush_finished_requests=lambda: lifecycle_calls.append(("flush", None)),
    )
    runner.model_intermediate_buffer = {"same-id": {"generation": "new"}}
    captured_request_ids = []
    runner._execute_stepwise_generation = lambda _output, request_ids, _corrections: captured_request_ids.extend(
        request_ids
    )
    scheduler_output = _StubSchedulerOutput(1)
    scheduler_output.num_scheduled_tokens = {"same-id": 1}
    scheduler_output.stepwise_req_ids = ["same-id"]
    scheduler_output.finished_req_ids = {"same-id"}
    scheduler_output.scheduled_new_reqs = [SimpleNamespace(req_id="same-id")]

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert output is None
    assert captured_request_ids == ["same-id"]
    assert lifecycle_calls == [
        ("finish", {"same-id"}),
        ("flush", None),
    ]


def test_sample_tokens_uses_explicit_stepwise_output_order():
    runner = _make_runner(
        {"audio": [torch.ones(2), None]},
        req_ids=("stale-input-batch",),
    )
    runner._stepwise_output_req_ids = ["done", "pending"]

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.req_ids == ["done", "pending"]
    assert output.req_id_to_index == {"done": 0, "pending": 1}
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[1] is None


@pytest.mark.parametrize("total", [-1, -512, 0])
def test_execute_model_skips_non_positive_scheduled_span(monkeypatch, total):
    """#5196: a negative span is truthy, so it used to reach `_prepare_inputs`,
    whose `assert total_num_scheduled_tokens > 0` killed the stage EngineCore."""
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(total))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
