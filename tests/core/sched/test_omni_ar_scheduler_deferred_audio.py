# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Held terminal outputs of single-stage audio stages."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler():
    held = {
        "a": {"client_index": 1, "finish_reason": FinishReason.STOP, "stop_reason": 7},
        "b": {"client_index": 0, "finish_reason": FinishReason.LENGTH, "stop_reason": None},
    }
    return SimpleNamespace(omni_deferred_finishes=held, omni_tail_outputs=deque())


def test_audio_of_a_step_in_flight_at_the_stop_is_forwarded():
    scheduler = _scheduler()
    outputs: dict[int, list] = defaultdict(list)
    payload = {"model_outputs": torch.ones(2, 3)}
    runner_output = SimpleNamespace(req_id_to_index={"a": 0, "z": 1})
    OmniARScheduler._forward_deferred_audio(scheduler, outputs, "a", [payload, None], runner_output)
    OmniARScheduler._forward_deferred_audio(scheduler, outputs, "z", [payload, payload], runner_output)
    (out,) = outputs[1]
    assert out.request_id == "a" and out.finish_reason is None and out.multimodal_output is payload
    assert list(outputs) == [1]


def test_tail_outputs_follow_the_queue_and_release_the_hold():
    scheduler = _scheduler()
    audio = {"model_outputs": torch.ones(2, 3)}
    scheduler.omni_tail_outputs.extend([("a", audio, False), ("b", None, True), ("a", None, False), ("a", audio, True)])
    outputs: dict[int, list] = defaultdict(list)
    OmniARScheduler._emit_tail_outputs(scheduler, outputs)
    a = outputs[1]
    assert [o.finish_reason for o in a] == [None, FinishReason.STOP] and a[1].stop_reason == 7
    (b,) = outputs[0]
    assert b.finish_reason == FinishReason.LENGTH and b.multimodal_output is None
    assert not scheduler.omni_deferred_finishes and not scheduler.omni_tail_outputs
    # Late items of a request whose terminal output went out are dropped.
    scheduler.omni_tail_outputs.append(("a", audio, False))
    OmniARScheduler._emit_tail_outputs(scheduler, outputs)
    assert len(outputs[1]) == 2


def test_fused_preemption_finishes_with_error_without_requeue(monkeypatch):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.omni_defer_audio_finish = True
    request = SimpleNamespace(request_id="preempted", client_index=2, status=RequestStatus.RUNNING)
    scheduler.requests = {request.request_id: request}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = defaultdict(set)

    def finish_requests(request_ids, status):
        request.status = status
        scheduler.requests.pop(request.request_id)
        scheduler.finished_req_ids.update(request_ids)
        scheduler.finished_req_ids_dict[request.client_index].update(request_ids)

    def unexpected_requeue(*args, **kwargs):
        pytest.fail("Fused audio must not re-enter upstream's preemption/recompute path")

    scheduler.finish_requests = finish_requests
    monkeypatch.setattr(Scheduler, "_preempt_request", unexpected_requeue)
    scheduler._preempt_request(request, 1.0)
    assert request.status == RequestStatus.FINISHED_ERROR and not scheduler.requests
    assert scheduler.finished_req_ids == {"preempted"}
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=True)
    [error] = outputs[2].outputs
    assert error.finish_reason == FinishReason.ERROR and "KV-cache preemption" in error.stop_reason
    assert outputs[2].finished_requests == {"preempted"}
    outputs = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=True)
    assert outputs == {}


def test_other_stages_keep_upstream_preemption(monkeypatch):
    calls = []
    monkeypatch.setattr(Scheduler, "_preempt_request", lambda *args: calls.append(args))
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.omni_defer_audio_finish = False
    request = SimpleNamespace(request_id="normal")
    scheduler._preempt_request(request, 1.0, drop_stale_output=True)
    assert calls == [(scheduler, request, 1.0, True)]
