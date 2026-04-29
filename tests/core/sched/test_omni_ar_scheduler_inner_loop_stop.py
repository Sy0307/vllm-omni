from __future__ import annotations

from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler_for_update(request: Request) -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = SimpleNamespace(remove_requests=lambda _requests: None)
    scheduler.skipped_waiting = SimpleNamespace(remove_requests=lambda _requests: None)
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.recompute_kv_load_failures = False
    scheduler.structured_output_manager = SimpleNamespace(should_advance=lambda _request: False)
    scheduler.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
    scheduler.kv_event_publisher = SimpleNamespace(publish=lambda _batch: None)
    scheduler.chunk_transfer_adapter = None
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.waiting_for_transfer_free = set()
    scheduler.transfer_triggered_requests = set()
    scheduler.active_kv_transfers = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler.kv_transfer_criteria = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler.max_model_len = 64
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler.acoustic_inner_loop_steps = 4

    def _free_request(self, stopped_request):
        self.finished_req_ids_dict[stopped_request.client_index].add(stopped_request.request_id)
        self.requests.pop(stopped_request.request_id, None)
        return None

    scheduler._free_request = MethodType(_free_request, scheduler)
    scheduler._get_routed_experts = MethodType(lambda _self, _request: None, scheduler)
    scheduler.make_stats = MethodType(lambda *_args, **_kwargs: None, scheduler)
    return scheduler


def test_inner_loop_stop_token_finishes_and_releases_unused_decode_slots():
    request = Request(
        request_id="req-inner-stop",
        prompt_token_ids=[11, 12],
        sampling_params=SamplingParams(max_tokens=16, stop_token_ids=[2150]),
        pooling_params=None,
        client_index=0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    # The base scheduler has already advanced computed tokens by the four
    # scheduled decode slots: one regular decode slot plus three inner-loop
    # slots. Only two tokens are actually generated before the stop token.
    request.num_computed_tokens = 6
    request.num_output_placeholders = 3

    scheduler = _make_scheduler_for_update(request)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 4},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens={},
        omni_acoustic_inner_loop_extra_slots={request.request_id: 3},
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[100, 2150]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={request.request_id: 0},
        kv_extracted_req_ids=None,
    )

    outputs = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)

    assert request.is_finished()
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 2150
    assert list(request.output_token_ids) == [100, 2150]
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == request.num_tokens == 4
    assert request not in scheduler.running
    assert request.request_id not in scheduler.requests
    assert scheduler._reserve_acoustic_inner_loop_slots() == {}

    output = outputs[0].outputs[0]
    assert output.request_id == request.request_id
    assert output.new_token_ids == [100, 2150]
    assert outputs[0].finished_requests == {request.request_id}
