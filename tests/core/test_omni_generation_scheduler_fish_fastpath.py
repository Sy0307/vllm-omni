from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Request:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.external_req_id = request_id
        self.status = RequestStatus.RUNNING
        self.prompt_token_ids = [0, 1, 2]
        self.num_computed_tokens = 3
        self.num_cached_tokens = -1
        self.client_index = 0
        self.stop_reason = None
        self.trace_headers = None
        self.num_external_computed_tokens = 0
        self.num_nans_in_logits = None
        self.events = ["scheduled"]

    def __hash__(self):
        return hash(self.request_id)

    def is_finished(self):
        return self.status == RequestStatus.FINISHED_STOPPED

    def get_finished_reason(self):
        return "stop"

    def take_events(self):
        events = self.events
        self.events = []
        return events


class _Queue(list):
    def remove_requests(self, requests):
        for request in list(requests):
            if request in self:
                self.remove(request)


def _make_scheduler(request: _Request, *, finished: bool):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._fish_dac_update_fastpath = True
    scheduler._fish_dac_rearm_in_update = False
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        finished_requests={request.request_id} if finished else set(),
        cleanup_calls=[],
        rearm_calls=[],
    )

    def cleanup(req_id, ext_id):
        scheduler.chunk_transfer_adapter.cleanup_calls.append((req_id, ext_id))

    scheduler.chunk_transfer_adapter.cleanup = cleanup

    def rearm_running_chunk_request(request):
        scheduler.chunk_transfer_adapter.rearm_calls.append(request.request_id)
        request.status = RequestStatus.WAITING_FOR_CHUNK
        return True

    scheduler.chunk_transfer_adapter.rearm_running_chunk_request = rearm_running_chunk_request
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = _Queue()
    scheduler.skipped_waiting = _Queue()
    scheduler.finished_req_ids_dict = {}
    scheduler._handle_stopped_request = lambda req: True
    scheduler._free_request = lambda req: {"freed": req.request_id}
    return scheduler


def test_fish_dac_update_fastpath_emits_audio_without_finishing():
    request = _Request("req-1")
    scheduler = _make_scheduler(request, finished=False)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 3})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-1": 0},
        pooler_output=[{"audio": "chunk"}],
    )

    outputs = OmniGenerationScheduler._fish_try_update_dac_fastpath(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert list(outputs) == [0]
    eco = outputs[0].outputs[0]
    assert eco.request_id == "req-1"
    assert eco.pooling_output == {"audio": "chunk"}
    assert eco.finish_reason is None
    assert scheduler.running == [request]
    assert scheduler.chunk_transfer_adapter.cleanup_calls == []
    assert scheduler.chunk_transfer_adapter.rearm_calls == []


def test_fish_dac_update_fastpath_can_rearm_live_chunk_request():
    request = _Request("req-1")
    scheduler = _make_scheduler(request, finished=False)
    scheduler._fish_dac_rearm_in_update = True
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 3})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-1": 0},
        pooler_output=[{"audio": "chunk"}],
    )

    outputs = OmniGenerationScheduler._fish_try_update_dac_fastpath(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    eco = outputs[0].outputs[0]
    assert eco.request_id == "req-1"
    assert eco.pooling_output == {"audio": "chunk"}
    assert scheduler.chunk_transfer_adapter.rearm_calls == ["req-1"]
    assert request.status == RequestStatus.WAITING_FOR_CHUNK
    assert scheduler.running == []


def test_fish_dac_direct_worker_schedules_bucketed_ready_chunk():
    req1 = _Request("req-1")
    req1.prompt_token_ids = [0] * 40
    req1.num_computed_tokens = 0
    req1.additional_information = {"next_stage_prompt_len": 40}
    req2 = _Request("req-2")
    req2.prompt_token_ids = [0] * 250
    req2.num_computed_tokens = 0
    req2.additional_information = {"next_stage_prompt_len": 250}

    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._fish_dac_direct_worker = True
    scheduler._fish_dac_worker_batch_size = 1
    scheduler._fish_dac_direct_worker_prefetch = 2
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.requests = {req1.request_id: req1, req2.request_id: req2}
    scheduler.running = []
    scheduler.waiting = _Queue()
    scheduler.log_stats = False
    scheduler._fish_dac_bucket_frames = [4, 25, 50]
    scheduler._fish_dac_sched_fastpath_profile = False
    scheduler._fish_dac_fastpath_steps = 0
    scheduler.kv_cache_manager = SimpleNamespace(new_step_starts=lambda: None)

    class _Adapter:
        finished_requests = set()

        def __init__(self):
            self.leftovers = []

        def drain_ready_chunks(self, max_chunks):
            assert max_chunks == 2
            return [req1, req2]

        def process_pending_chunks(self, waiting, running):
            raise AssertionError("ready chunks should bypass pending processing")

        def prepend_ready_chunks(self, requests):
            self.leftovers.extend(requests)

        def postprocess_scheduler_output(self, scheduler_output):
            pass

    adapter = _Adapter()
    scheduler.chunk_transfer_adapter = adapter

    def make_output(**kwargs):
        return SimpleNamespace(
            num_scheduled_tokens=kwargs["num_scheduled_tokens"],
            total_num_scheduled_tokens=sum(kwargs["num_scheduled_tokens"].values()),
        )

    scheduler._fish_make_dac_fastpath_output = make_output

    output = OmniGenerationScheduler.fish_dac_worker_schedule(scheduler)

    assert output.num_scheduled_tokens == {"req-1": 40}
    assert req1.status == RequestStatus.RUNNING
    assert req1.num_computed_tokens == 40
    assert scheduler.running == []
    assert adapter.leftovers == [req2]


def test_fish_dac_direct_worker_can_coalesce_mixed_buckets():
    req1 = _Request("req-1")
    req1.prompt_token_ids = [0] * 40
    req1.num_computed_tokens = 0
    req1.additional_information = {"next_stage_prompt_len": 40}
    req2 = _Request("req-2")
    req2.prompt_token_ids = [0] * 250
    req2.num_computed_tokens = 0
    req2.additional_information = {"next_stage_prompt_len": 250}

    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._fish_dac_direct_worker = True
    scheduler._fish_dac_direct_worker_mixed_bucket = True
    scheduler._fish_dac_worker_batch_size = 2
    scheduler._fish_dac_direct_worker_prefetch = 2
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.requests = {req1.request_id: req1, req2.request_id: req2}
    scheduler.running = []
    scheduler.waiting = _Queue()
    scheduler.log_stats = False
    scheduler._fish_dac_bucket_frames = [4, 25, 50]
    scheduler._fish_dac_sched_fastpath_profile = False
    scheduler._fish_dac_fastpath_steps = 0
    scheduler.kv_cache_manager = SimpleNamespace(new_step_starts=lambda: None)

    class _Adapter:
        finished_requests = set()

        def __init__(self):
            self.leftovers = []

        def drain_ready_chunks(self, max_chunks):
            assert max_chunks == 2
            return [req1, req2]

        def process_pending_chunks(self, waiting, running):
            raise AssertionError("ready chunks should bypass pending processing")

        def prepend_ready_chunks(self, requests):
            self.leftovers.extend(requests)

        def postprocess_scheduler_output(self, scheduler_output):
            pass

    adapter = _Adapter()
    scheduler.chunk_transfer_adapter = adapter

    def make_output(**kwargs):
        return SimpleNamespace(
            num_scheduled_tokens=kwargs["num_scheduled_tokens"],
            total_num_scheduled_tokens=sum(kwargs["num_scheduled_tokens"].values()),
        )

    scheduler._fish_make_dac_fastpath_output = make_output

    output = OmniGenerationScheduler.fish_dac_worker_schedule(scheduler)

    assert output.num_scheduled_tokens == {"req-1": 40, "req-2": 250}
    assert req1.status == RequestStatus.RUNNING
    assert req2.status == RequestStatus.RUNNING
    assert req1.num_computed_tokens == 40
    assert req2.num_computed_tokens == 250
    assert adapter.leftovers == []


def test_fish_dac_update_fastpath_finishes_and_cleans_up():
    request = _Request("req-1")
    scheduler = _make_scheduler(request, finished=True)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 3})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-1": 0},
        pooler_output=[{"audio": "final"}],
    )

    outputs = OmniGenerationScheduler._fish_try_update_dac_fastpath(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    eco = outputs[0].outputs[0]
    assert eco.finish_reason == "stop"
    assert eco.kv_transfer_params == {"freed": "req-1"}
    assert scheduler.running == []
    assert scheduler.chunk_transfer_adapter.cleanup_calls == [("req-1", "req-1")]
