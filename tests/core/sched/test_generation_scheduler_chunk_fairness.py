# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk admission must not rerun consumed payloads or overlap one stream."""

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from tests.core.sched.test_generation_scheduler_restore import _scheduler_with_parked_generation_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(name: str, *, in_flight: int = 0, computed: int = 0) -> Request:
    request = Request(name, [1, 2], SamplingParams(max_tokens=4), pooling_params=None)
    request.num_in_flight_tokens = in_flight
    request.num_computed_tokens = computed
    request.external_req_id = name
    return request


@pytest.fixture
def scheduler(monkeypatch):
    scheduler, _ = _scheduler_with_parked_generation_request(monkeypatch, use_v2_model_runner=True)
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.create_request_queue", create_request_queue)
    scheduler._native_data_plane = True
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = SimpleNamespace(finished_requests=set())
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.running = []
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.requests = {}
    scheduler._process_pending_omni_inputs = lambda model_mode: None
    scheduler._postprocess_omni_schedule_output = lambda output: None
    scheduler._restore_omni_wait_queues = lambda: None

    def advance(output):
        for rid, count in output.num_scheduled_tokens.items():
            req = scheduler.requests[rid]
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens += count
            req.num_in_flight_tokens += count

    scheduler._update_after_schedule = advance
    return scheduler


def test_ready_first_chunk_runs_while_another_stream_is_in_flight(scheduler):
    running = _request("running", in_flight=2)
    running.status = RequestStatus.RUNNING
    first = _request("first")
    scheduler.running = [running]
    scheduler.waiting.add_request(first)
    scheduler.requests = {r.request_id: r for r in (running, first)}

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"first": 2}
    assert running.num_in_flight_tokens == 2
    assert len(output.scheduled_new_reqs) == scheduler.max_num_running_reqs == 1


def test_completed_chunk_yields_to_waiting_stream_then_makes_progress(scheduler):
    continuation = _request("continuation")
    continuation.status = RequestStatus.RUNNING
    first = _request("first")
    scheduler.running = [continuation]
    scheduler.waiting.add_request(first)
    scheduler.requests = {r.request_id: r for r in (continuation, first)}

    first_output = scheduler.schedule()
    next_output = scheduler.schedule()

    assert first_output.num_scheduled_tokens == {"first": 2}
    assert next_output.num_scheduled_tokens == {"continuation": 2}
    assert {r.request_id for r in scheduler.running} == {"first", "continuation"}


def test_waiting_in_flight_entry_does_not_block_or_repeat(scheduler):
    pending = _request("pending", in_flight=2)
    ready = _request("ready")
    scheduler.requests = {r.request_id: r for r in (pending, ready)}
    scheduler.waiting.add_request(pending)
    scheduler.waiting.add_request(ready)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"ready": 2}
    assert list(scheduler.waiting) == [pending]


@pytest.mark.parametrize("terminal", [False, True])
def test_completed_payload_is_not_executed_again_without_new_chunk(scheduler, terminal):
    completed = _request("completed", computed=2)
    completed.status = RequestStatus.RUNNING
    scheduler.running = [completed]
    scheduler.requests = {"completed": completed}
    if terminal:
        scheduler.input_coordinator.finished_requests.add("completed")

    output = scheduler.schedule()

    assert not output.num_scheduled_tokens
    assert scheduler._pending_finish_reqs == ([completed] if terminal else [])


def test_chunk_wait_status_and_request_state_survive_requeue(scheduler):
    completed = _request("completed", computed=2)
    completed.status = RequestStatus.WAITING_FOR_CHUNK
    scheduler.running = [completed]
    scheduler._requeue_completed_native_chunks()

    assert completed.status == RequestStatus.WAITING_FOR_CHUNK
    assert completed.num_computed_tokens == 2
    assert list(scheduler.waiting) == [completed]
    assert not scheduler.running


def test_legacy_generation_keeps_lifetime_admission(scheduler):
    scheduler._native_data_plane = False
    request = _request("legacy")
    request.status = RequestStatus.RUNNING
    scheduler.running = [request]

    scheduler._requeue_completed_native_chunks()

    assert scheduler.running == [request]
    assert not scheduler.waiting
