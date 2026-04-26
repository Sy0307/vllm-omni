from __future__ import annotations

import queue
from contextlib import nullcontext
from collections import deque
from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeScheduler:
    def __init__(
        self,
        ready_sequence: list[bool],
        *,
        unfinished: bool = False,
    ):
        self.ready_sequence = deque(ready_sequence)
        self.unfinished = unfinished

    def fish_dac_has_ready_work(self) -> bool:
        if not self.ready_sequence:
            return False
        return self.ready_sequence.popleft()

    def has_unfinished_requests(self) -> bool:
        return self.unfinished


def _make_core(
    monkeypatch,
    *,
    ready_sequence: list[bool],
    unfinished: bool = False,
):
    monkeypatch.setenv("VLLM_FISH_DAC_ENGINE_SIDE_LOOP", "1")
    monkeypatch.setenv("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_MAX_STEPS", "8")
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="dac_decoder", async_chunk=True)
    )
    core.scheduler = _FakeScheduler(ready_sequence, unfinished=unfinished)
    core.output_queue = queue.Queue()
    core.input_queue = queue.Queue()
    core.post_step_calls = []

    def post_step(model_executed):
        core.post_step_calls.append(model_executed)

    core.post_step = post_step
    return core


def test_fish_dac_engine_side_loop_drains_ready_steps(monkeypatch):
    core = _make_core(monkeypatch, ready_sequence=[True, True, False])
    calls = []

    def step_fn():
        calls.append(len(calls))
        return {0: f"out-{len(calls)}"}, True

    core.step_fn = step_fn

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is True
    assert len(calls) == 3
    assert core.post_step_calls == [True, True, True]
    assert [core.output_queue.get_nowait() for _ in range(3)] == [
        (0, "out-1"),
        (0, "out-2"),
        (0, "out-3"),
    ]


def test_fish_dac_engine_side_loop_stops_for_input_queue(monkeypatch):
    core = _make_core(monkeypatch, ready_sequence=[True, True])
    core.input_queue.put_nowait(("add", None))
    calls = []

    def step_fn():
        calls.append(len(calls))
        return {0: "out"}, True

    core.step_fn = step_fn

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is True
    assert len(calls) == 1
    assert core.output_queue.get_nowait() == (0, "out")


def test_fish_dac_engine_side_loop_does_not_spin_after_empty_step(monkeypatch):
    core = _make_core(monkeypatch, ready_sequence=[True, True, True])
    calls = []

    def step_fn():
        calls.append(len(calls))
        return {}, False

    core.step_fn = step_fn

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is False
    assert len(calls) == 1
    assert core.output_queue.empty()


def test_fish_dac_engine_side_loop_waits_within_idle_budget(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_IDLE_US", "300")
    monkeypatch.setenv("VLLM_FISH_DAC_ENGINE_SIDE_LOOP_POLL_US", "100")
    core = _make_core(
        monkeypatch,
        ready_sequence=[False, False, True, False],
        unfinished=True,
    )
    calls = []
    now = [0.0]
    sleeps = []

    monkeypatch.setattr(
        "vllm_omni.engine.stage_engine_core_proc.time.monotonic",
        lambda: now[0],
    )

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("vllm_omni.engine.stage_engine_core_proc.time.sleep", fake_sleep)

    def step_fn():
        calls.append(len(calls))
        return {0: f"out-{len(calls)}"}, True

    core.step_fn = step_fn

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is True
    assert len(calls) == 2
    assert sleeps[:2] == [0.0001, 0.0001]
    assert [core.output_queue.get_nowait() for _ in range(2)] == [
        (0, "out-1"),
        (0, "out-2"),
    ]


def test_fish_dac_engine_has_work_when_ready_chunk_arrives(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_DAC_READY_WAKEUP", "1")
    core = _make_core(monkeypatch, ready_sequence=[True])

    assert StageEngineCoreProc.has_work(core) is True


def test_fish_dac_ready_wakeup_enqueues_engine_wakeup(monkeypatch):
    core = _make_core(monkeypatch, ready_sequence=[])
    wakeups = []
    core.input_queue = SimpleNamespace(
        put_nowait=lambda item: wakeups.append(item),
    )

    StageEngineCoreProc._fish_dac_ready_wakeup(core)

    assert wakeups
    assert wakeups[0][1] is None


def test_fish_dac_direct_worker_executes_and_outputs(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_DAC_DIRECT_WORKER", "1")
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="dac_decoder", async_chunk=True)
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    model_output = SimpleNamespace()
    core.output_queue = queue.Queue()
    core.post_step_calls = []
    core.aborts_processed = False
    core.log_error_detail = lambda _scheduler_output: nullcontext()
    core.log_iteration_details = lambda _scheduler_output: nullcontext()

    class _Future:
        def result(self):
            return model_output

    core.model_executor = SimpleNamespace(
        execute_model=lambda output, non_block=False: _Future(),
        sample_tokens=lambda grammar_output: None,
    )

    class _Scheduler:
        def __init__(self):
            self.updated = None

        def fish_dac_worker_schedule(self):
            return scheduler_output

        def get_grammar_bitmask(self, output):
            assert output is scheduler_output
            return None

        def fish_dac_worker_update(self, output, runner_output):
            self.updated = (output, runner_output)
            return {0: "direct-output"}

    scheduler = _Scheduler()
    core.scheduler = scheduler
    core.post_step = lambda model_executed: core.post_step_calls.append(model_executed)
    core._process_aborts_queue = lambda: setattr(core, "aborts_processed", True)

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is True
    assert scheduler.updated == (scheduler_output, model_output)
    assert core.aborts_processed is True
    assert core.post_step_calls == [True]
    assert core.output_queue.get_nowait() == (0, "direct-output")


def test_fish_dac_direct_worker_drains_multiple_ready_batches(monkeypatch):
    monkeypatch.setenv("VLLM_FISH_DAC_DIRECT_WORKER", "1")
    monkeypatch.setenv("VLLM_FISH_DAC_DIRECT_WORKER_MAX_STEPS", "8")
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="dac_decoder", async_chunk=True)
    )
    scheduler_outputs = deque(
        [
            SimpleNamespace(total_num_scheduled_tokens=1, name="batch-1"),
            SimpleNamespace(total_num_scheduled_tokens=1, name="batch-2"),
            None,
        ]
    )
    core.output_queue = queue.Queue()
    core.input_queue = queue.Queue()
    core.post_step_calls = []
    core.log_error_detail = lambda _scheduler_output: nullcontext()
    core.log_iteration_details = lambda _scheduler_output: nullcontext()

    class _Future:
        def __init__(self, output):
            self.output = output

        def result(self):
            return self.output

    core.model_executor = SimpleNamespace(
        execute_model=lambda output, non_block=False: _Future(f"model-{output.name}"),
        sample_tokens=lambda grammar_output: None,
    )

    class _Scheduler:
        def __init__(self):
            self.updated = []

        def fish_dac_worker_schedule(self):
            return scheduler_outputs.popleft()

        def get_grammar_bitmask(self, output):
            return None

        def fish_dac_worker_update(self, output, runner_output):
            self.updated.append((output.name, runner_output))
            return {output.name: f"direct-{output.name}"}

        def has_unfinished_requests(self):
            return False

        def fish_dac_has_ready_work(self):
            return bool(scheduler_outputs and scheduler_outputs[0] is not None)

    scheduler = _Scheduler()
    core.scheduler = scheduler
    core.post_step = lambda model_executed: core.post_step_calls.append(model_executed)
    core._process_aborts_queue = lambda: None

    model_executed = StageEngineCoreProc._process_engine_step(core)

    assert model_executed is True
    assert scheduler.updated == [
        ("batch-1", "model-batch-1"),
        ("batch-2", "model-batch-2"),
    ]
    assert core.post_step_calls == [True, True]
    assert [core.output_queue.get_nowait() for _ in range(2)] == [
        ("batch-1", "direct-batch-1"),
        ("batch-2", "direct-batch-2"),
    ]
