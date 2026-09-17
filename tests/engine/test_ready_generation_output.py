# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Completed generation output delivery preserves the batch queue contract."""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def ready(value=None, error=None):
    f: Future[Any] = Future()
    if error:
        f.set_exception(error)
    else:
        f.set_result(value)
    return f


def core(queue):
    events: list[Any] = []
    obj = SimpleNamespace(
        batch_queue=deque(queue),
        capture_iteration_details=lambda out: nullcontext("iteration"),
        log_error_detail=lambda out: nullcontext(),
        _process_aborts_queue=lambda: events.append("abort"),
        _attach_iteration_details=lambda out, details: events.append(("attach", details)),
    )

    def update(sched, out):
        events.append(("update", sched, out))
        return {"output": out}

    obj.scheduler = SimpleNamespace(update_from_output=update)
    return obj, events


def test_ready_oldest_is_delivered_before_new_batch():
    older = (ready("audio"), "old", ready())
    newer = (ready("next"), "new", ready())
    obj, events = core([newer, older])
    assert StageEngineCoreProc._drain_ready_generation_output(obj) == ({"output": "audio"}, False)
    assert list(obj.batch_queue) == [newer]
    assert events == ["abort", ("update", "old", "audio"), ("attach", "iteration")]


def test_pending_oldest_does_not_block_or_reorder():
    pending: tuple[Future[Any], str, Future[Any]] = (Future(), "old", ready())
    newer = (ready("next"), "new", ready())
    obj, events = core([newer, pending])
    assert StageEngineCoreProc._drain_ready_generation_output(obj) is None
    assert list(obj.batch_queue) == [newer, pending] and not events


def test_empty_queue():
    obj, events = core([])
    assert StageEngineCoreProc._drain_ready_generation_output(obj) is None
    assert not events


def test_output_error_propagates_without_scheduler_update():
    obj, events = core([(ready(error=ValueError("worker")), "batch", ready())])
    with pytest.raises(ValueError, match="worker"):
        StageEngineCoreProc._drain_ready_generation_output(obj)
    assert not events


def test_execution_failure_is_not_hidden_by_empty_sampling_result():
    obj, events = core([(ready(None), "batch", ready(error=ValueError("execution")))])
    with pytest.raises(ValueError, match="execution"):
        StageEngineCoreProc._drain_ready_generation_output(obj)
    assert not events


def test_enabled_delivery_does_not_execute_next_batch(monkeypatch):
    obj, events = core([(ready("audio"), "batch", ready())])
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.__dict__.update(obj.__dict__)
    engine._omni_drain_ready_generation = True

    def upstream(self):
        raise AssertionError("next batch must not run first")

    monkeypatch.setattr(EngineCoreProc, "step_with_batch_queue", upstream)
    assert engine.step_with_batch_queue() == ({"output": "audio"}, False)


@pytest.mark.parametrize("enabled", [False, True])
def test_pending_output_uses_upstream_path(monkeypatch, enabled):
    pending: Future[Any] = Future()
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.batch_queue = deque([(pending, "batch", ready())])
    engine._omni_drain_ready_generation = enabled
    calls = []

    def upstream(self):
        calls.append(self)
        return None, True

    monkeypatch.setattr(EngineCoreProc, "step_with_batch_queue", upstream)
    assert engine.step_with_batch_queue() == (None, True)
    assert calls == [engine]
    assert engine.batch_queue[-1][0] is pending


def test_disabled_completed_output_uses_upstream_path(monkeypatch):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.batch_queue = deque([(ready("audio"), "batch", ready())])
    engine._omni_drain_ready_generation = False
    calls = []

    def upstream(self):
        calls.append(self)
        return None, True

    monkeypatch.setattr(EngineCoreProc, "step_with_batch_queue", upstream)
    assert engine.step_with_batch_queue() == (None, True)
    assert calls == [engine]
    assert len(engine.batch_queue) == 1


def test_ready_delivery_notifies_completion_observer_before_scheduler():
    future = ready("audio")
    obj, events = core([(future, "batch", ready())])

    class Observer:
        def consumed(self, completed):
            assert completed is future
            events.append("consumed")

    obj._omni_completion_observer = Observer()
    StageEngineCoreProc._drain_ready_generation_output(obj)
    assert events[:2] == ["consumed", "abort"]
