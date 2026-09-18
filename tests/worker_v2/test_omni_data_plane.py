# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lifecycle contracts for OmniRunnerDataPlane and NativeOutputWorker: FIFO
publish, terminal/abort exactly once, failure propagation, bounded close."""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker_v2.delivery import DeliveryCancelledError, DeliveryState, OmniDeliveryManager
from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _complete(plane, payloads, token=21):
    return plane.complete_outputs(
        req_ids=["internal"], inter_stage_outputs=payloads, sampled_token_ids=[[token]] * len(payloads)
    )


def _new_request(req_id="internal", external_req_id="external"):
    return SimpleNamespace(
        req_id=req_id,
        external_req_id=external_req_id,
        prompt_token_ids=[10, 11],
        num_computed_tokens=2,
        resumable=True,
        additional_information=SimpleNamespace(entries={}),
        sampling_params=SimpleNamespace(stop_token_ids=[2150]),
    )


def _bare_plane(*, delivery_timeout_s=1.0, shutdown_timeout_s=1.0):
    plane = object.__new__(OmniRunnerDataPlane)
    plane.__dict__.update(
        _native_requests={},
        _native_output_lock=threading.RLock(),
        _native_send_lock=threading.Lock(),
        _native_output_error_lock=threading.Lock(),
        _native_output_error=None,
        _native_output_closed=False,
        _native_outputs_in_flight=defaultdict(int),
        _native_terminal_pending=set(),
        _put_req_chunk=defaultdict(int),
        _ramp_chunk_count=defaultdict(int),
        _delivery_manager=OmniDeliveryManager(
            delivery_timeout_s=delivery_timeout_s, shutdown_timeout_s=shutdown_timeout_s
        ),
    )
    return plane


def _make_plane(replace_send):
    p = _bare_plane()
    p.record = SimpleNamespace(batches=[], cleaned=[])
    if replace_send:
        p.send_chunks = lambda entries, **_kw: p.record.batches.append(entries) or len(entries)
    p.cleanup_finished_request = p.record.cleaned.append
    return p


def _yield_plane(replace_send):
    p = _make_plane(replace_send)
    yield p
    if getattr(p, "_native_output_worker", None) is not None:
        p._stop_output_worker()


@pytest.fixture
def raw_plane():  # keeps the real connector send path
    yield from _yield_plane(False)


@pytest.fixture
def plane():
    yield from _yield_plane(True)


def _gated_connector():
    """put() blocks its first call until released (or forever without release)."""
    conn = SimpleNamespace(
        put_keys=[], put_started=threading.Event(), release=threading.Event(), close_called=threading.Event()
    )

    def put(**kwargs):
        conn.put_keys.append(kwargs["put_key"])
        if len(conn.put_keys) == 1:
            conn.put_started.set()
            assert conn.release.wait(timeout=2)
        return True, 1, None

    conn.put, conn.close = put, conn.close_called.set
    return conn


def _start_save_thread(plane, connector):
    plane.__dict__.update(
        _omni_connector=connector,
        _stage_id=0,
        _next_stage_id=1,
        _lock=threading.Lock(),
        _pending_save_reqs={},
        _pending_save_counts=defaultdict(int),
        _deferred_send_cleanup=set(),
        _request_ids_mapping={},
        _send_side_request_payload={},
        _code_prompt_token_ids=defaultdict(list),
        _cached_ic={},
        _work_available=threading.Event(),
        _stop_event=threading.Event(),
        _custom_process_batch_func=lambda **kwargs: kwargs["pooling_outputs"],
        _can_send=True,
        _MAX_SEND_RETRIES=0,
        is_data_transfer_rank=lambda: True,
        _connector_send_error_sink=plane._record_output_error,
        _recv_thread=None,
    )
    plane._save_thread = threading.Thread(target=plane._save_loop, daemon=True)
    plane._save_thread.start()


def _stop_save_thread(plane):
    plane._stop_event.set()
    plane._work_available.set()
    plane._save_thread.join(timeout=2)
    assert not plane._save_thread.is_alive()


class _Output:
    """Fake AsyncModelRunnerOutput; get_output may block on a gate or raise."""

    copy_event = None

    def __init__(self, rid="internal", gate=None, started=None, error=None, payload=None, token=1):
        self.rid, self.gate, self.started, self.error = rid, gate, started, error
        self.payload, self.token = payload, token

    def get_output(self):
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=3)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(req_ids=[self.rid], inter_stage_outputs=self.payload, sampled_token_ids=[[self.token]])


def test_emit_cohort_one_batch_and_trims_non_resumable_history(plane):
    plane.send_chunk = lambda *_a, **_kw: pytest.fail("must not fall back to per-request send_chunk")
    plane.register_request(_new_request("r0", "ext-0"))
    plane.register_request(_new_request("r1", "ext-1"))
    emitted = plane.emit_chunks(
        req_ids=["r0", "r1"],
        inter_stage_outputs=[{"codes.audio": "c0"}, {"codes.audio": "c1"}],
        sampled_token_ids=[[21]],
        terminal_req_ids={"r1"},
    )
    assert emitted == 2 and len(plane.record.batches) == 1  # the whole step cohort goes out in one batch
    (req0, payload0), (req1, _) = plane.record.batches[0]
    assert (req0.external_req_id, payload0) == ("ext-0", {"codes": {"audio": "c0"}})
    assert req0.all_token_ids == [10, 11, 21] and req0.is_finished() is False
    assert req1.is_finished() is True and plane.record.cleaned == ["r1"]
    # A non-resumable request omits token history after its first chunk.
    plane.register_request(_new_request("r2", "ext-2"))
    state = plane._native_requests["r2"]
    state.resumable, plane._put_req_chunk["ext-2"] = False, 1
    state.output_token_ids.extend(range(1024))
    plane.emit_chunks(
        req_ids=["r2"], inter_stage_outputs=[{"h": "d"}], sampled_token_ids=[[1024]], terminal_req_ids={"r2"}
    )
    request, _ = plane.record.batches[1][0]
    assert request.output_token_count == 1025 and request.all_token_ids == []


def test_terminal_waits_for_reserved_outputs_and_emits_once(plane):
    batches, cleaned = plane.record.batches, plane.record.cleaned
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.reserve_outputs(["internal"])
    # The terminal defers while reservations are in flight, and is idempotent.
    assert plane.request_terminal({"internal"}) == 0
    assert plane.request_terminal({"internal"}) == 0 and batches == []
    assert _complete(plane, [{"codes.audio": "c0"}]) == 1 and batches[0][0][0].is_finished() is False
    assert _complete(plane, [{"codes.audio": "c1"}], token=22) == 2  # data chunk, then the released terminal
    chunk, terminal = batches[1][0], batches[2][0]
    assert chunk[1] == {"codes": {"audio": "c1"}} and chunk[0].output_token_ids == [21, 22]
    assert terminal[1] is None and terminal[0].is_finished() is True and cleaned == ["internal"]
    # Stale terminal/complete after cleanup are no-ops; cleanup ran exactly once.
    assert plane.request_terminal({"internal"}) == 0 and _complete(plane, [{"codes.audio": "stale"}]) == 0
    assert cleaned == ["internal"]
    # A receive-only stage cleans its terminal locally without sending.
    plane._can_send = False
    plane.register_request(_new_request("rx"))
    assert plane.request_terminal({"rx"}) == 0 and cleaned == ["internal", "rx"]


@pytest.mark.parametrize("fail_point", ["terminal", "deferred"])
def test_enqueue_failure_propagates_and_holds_lifecycle(plane, fail_point):
    plane.send_chunks = lambda _e, **_kw: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    plane.register_request(_new_request())
    if fail_point == "terminal":
        with pytest.raises(RuntimeError, match="enqueue failed"):
            plane.request_terminal({"internal"})
    else:
        plane.reserve_outputs(["internal"])
        plane.request_terminal({"internal"})
        with pytest.raises(RuntimeError, match="enqueue failed"):
            _complete(plane, [{"codes.audio": "x"}])
        assert plane._native_outputs_in_flight["internal"] == 1
    # Lifecycle state is held for retry; cleanup must not run.
    assert "internal" in plane._native_requests and "internal" in plane._native_terminal_pending
    assert plane.record.cleaned == []
    # After recovery the held terminal is emitted once, after any held data.
    plane.send_chunks = lambda entries, **_kw: plane.record.batches.append(entries) or len(entries)
    recovered = plane.request_terminal(set()) if fail_point == "terminal" else _complete(plane, [{"codes.audio": "x"}])
    assert recovered == (1 if fail_point == "terminal" else 2) and "internal" not in plane._native_requests


def test_abort_before_enqueue_emits_terminal_once_and_drops_stale(plane):
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.request_terminal({"internal"})
    assert plane.abort_requests({"internal"}) == 1
    request, payload = plane.record.batches[0][0]
    assert request.is_finished() is True and payload is None
    assert plane.record.cleaned == ["internal"] and "internal" not in plane._native_requests
    # A duplicate abort is a no-op; a stale deferred output completes silently.
    assert plane.abort_requests({"internal"}) == 0 and _complete(plane, [{"codes.audio": "stale"}]) == 0
    assert len(plane.record.batches) == 1


def test_abort_cannot_overtake_committed_output(plane):
    data_in_lock, release_data, terminal_sent = threading.Event(), threading.Event(), threading.Event()
    original = plane._send_chunk_entries

    def gated(entries, **kwargs):
        if entries[0][0].is_finished():
            terminal_sent.set()
            return original(entries, **kwargs)
        data_in_lock.set()
        assert release_data.wait(timeout=2)  # hold the data send inside _native_send_lock
        return original(entries, **kwargs)

    plane._send_chunk_entries = gated
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    threads = [
        threading.Thread(target=lambda: _complete(plane, [{"codes.audio": "c0"}])),
        threading.Thread(target=plane.abort_requests, args=({"internal"},)),
    ]
    threads[0].start()
    try:
        assert data_in_lock.wait(timeout=2)
        threads[1].start()
        overtook = terminal_sent.wait(timeout=0.1)  # the terminal must not pass the committed data send
        release_data.set()
    finally:
        release_data.set()
        for t in threads:
            t.join(timeout=2)
    assert not overtook


def test_output_worker_fifo_and_scheduler_not_blocked(plane):
    batches, send_started, release_send = [], threading.Event(), threading.Event()

    def send_chunks(entries, **_kw):
        if not entries[0][0].is_finished():
            send_started.set()
            assert release_send.wait(timeout=2)
        batches.append(entries)
        return len(entries)

    plane.send_chunks = send_chunks
    plane._start_output_worker(max_pending_batches=2)
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
    assert send_started.wait(timeout=2)
    # A stuck deferred send must not block the scheduler path.
    assert plane.request_terminal({"internal"}) == 0 and batches == []
    release_send.set()
    plane.drain_outputs()
    assert [e[0].is_finished() for batch in batches for e in batch] == [False, True]


def test_worker_failure_surfaces_and_close_drains_in_order(plane):
    events = []
    plane.send_chunks = lambda _e, **_kw: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    plane.shutdown_omni_connectors = lambda: events.append("shutdown")
    plane._start_output_worker(max_pending_batches=2)
    plane.register_request(_new_request())
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
    with pytest.raises(RuntimeError, match="enqueue failed"):
        plane.drain_outputs()
    with pytest.raises(RuntimeError, match="enqueue failed"):  # an errored worker rejects new submissions
        plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 1}], sampled_token_ids=[[22]])
    # close() drains pending batches before shutting connectors down.
    plane._native_output_error = None
    plane.send_chunks = lambda entries, **_kw: events.append("send") or len(entries)
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 1}], sampled_token_ids=[[22]])
    plane.close()
    assert events == ["send", "shutdown"]


def test_permanent_put_failure_quarantines_and_holds_state(raw_plane):
    puts = []
    connector = SimpleNamespace(put=lambda **_kw: puts.append(1) or (False, 0, None), close=lambda: None)
    _start_save_thread(raw_plane, connector)
    raw_plane.register_request(_new_request())
    raw_plane.reserve_outputs(["internal"])
    raw_plane.request_terminal({"internal"})
    try:
        with pytest.raises(RuntimeError, match="connector send failed"):
            raw_plane.complete_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
        assert len(puts) == 1 and raw_plane._native_output_error is not None  # fail fast, no retry
        assert raw_plane._native_outputs_in_flight["internal"] == 1
        assert "internal" in raw_plane._native_terminal_pending
        with pytest.raises(RuntimeError, match="quarantined"):  # the quarantine rejects new tickets
            raw_plane._delivery_manager.create_ticket(request_id="other", put_key="other_0_0")
    finally:
        _stop_save_thread(raw_plane)


def test_delivery_timeout_drops_queued_ticket(raw_plane):
    raw_plane._delivery_manager = OmniDeliveryManager(delivery_timeout_s=0.05, shutdown_timeout_s=0.5)
    connector = _gated_connector()
    _start_save_thread(raw_plane, connector)
    req = SimpleNamespace(request_id="external")
    _, first = raw_plane._enqueue_chunk_payload(req, {"c": 1}, wait_for_delivery=True)
    _, second = raw_plane._enqueue_chunk_payload(req, {"c": 2}, wait_for_delivery=True)
    try:
        assert connector.put_started.wait(timeout=1)
        with pytest.raises(TimeoutError, match="delivery timed out"):
            first.wait()
        assert raw_plane._delivery_manager.is_quarantined
        connector.release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and "external" in raw_plane._pending_save_counts:
            time.sleep(0.001)
        # After quarantine the queued ticket is dropped: never put, marked FAILED.
        assert connector.put_keys == [first.put_key] and second.state is DeliveryState.FAILED
        assert "external" not in raw_plane._pending_save_counts and "external" not in raw_plane._pending_save_reqs
    finally:
        connector.release.set()
        _stop_save_thread(raw_plane)


def test_close_cancels_waiting_delivery_once_and_shutdown_stays_bounded():
    # A queued delivery waiter is cancelled exactly once on shutdown.
    manager = OmniDeliveryManager(delivery_timeout_s=30.0, shutdown_timeout_s=0.5)
    ticket = manager.create_ticket(request_id="r", put_key="k")
    manager.shutdown(RuntimeError("close"))
    manager.shutdown(RuntimeError("again"))  # idempotent: a queued waiter is cancelled at most once
    with pytest.raises(DeliveryCancelledError, match="cancelled"):
        ticket.wait()
    assert ticket.state is DeliveryState.CANCELLED
    # A connector put() that cannot be cancelled must not hang shutdown.
    plane = _bare_plane(delivery_timeout_s=0.05, shutdown_timeout_s=0.2)
    connector = _gated_connector()
    _start_save_thread(plane, connector)
    assert plane._enqueue_chunk_payload(SimpleNamespace(request_id="external"), {"c": 0}, wait_for_delivery=True)[0]
    try:
        assert connector.put_started.wait(timeout=1)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="shutdown exceeded"):
            plane.shutdown_omni_connectors()
        assert time.monotonic() - start < 1.5 and connector.close_called.is_set()
    finally:
        connector.release.set()
        _stop_save_thread(plane)


@pytest.mark.parametrize("abort", [False, True])
def test_materialization_fences_terminal_and_abort(plane, abort):
    batches, cleaned, gate, started = plane.record.batches, plane.record.cleaned, threading.Event(), threading.Event()
    plane.get_omni_connector_output = lambda: None
    plane._start_output_worker(max_pending_batches=2)
    materializer = NativeOutputWorker(2)
    try:
        plane.register_request(_new_request())
        plane.reserve_outputs(["internal"])
        output = materializer.submit(
            _Output(gate=gate, started=started, payload={"codes.audio": "c0"}, token=21), plane
        )
        assert started.wait(timeout=3)
        assert plane.request_terminal({"internal"}) == 0  # fenced behind the in-flight materialization
        if abort:
            assert plane.abort_requests({"internal"}) == 1
        gate.set()
        output.get_output()
        materializer.close()
        plane.drain_outputs()
        assert cleaned == ["internal"] and len(batches) == (1 if abort else 2)
        if not abort:
            assert batches[0][0][1] == {"codes": {"audio": "c0"}} and not batches[0][0][0].is_finished()
        assert batches[-1][0][0].is_finished()
    finally:
        gate.set()
        materializer.close()


def test_native_worker_fifo_thread_affinity_and_signals():
    owner, gate, calls = threading.get_ident(), threading.Event(), []

    def enqueue_outputs(**kw):
        calls.append(("publish", kw["req_ids"], threading.get_ident()))

    plane = SimpleNamespace(
        enqueue_outputs=enqueue_outputs,
        get_omni_connector_output=lambda: (calls.append(("signals", threading.get_ident())), "ready")[1],
    )
    worker = NativeOutputWorker(2)
    try:
        a = worker.submit(_Output("a", gate=gate, payload={"v": 1}), plane)
        b = worker.submit(_Output("b", payload={"v": 1}), plane)
        gate.set()
        b.get_output()  # publication of b implies a already published (FIFO, single worker thread)
        publishes = [c for c in calls if c[0] == "publish"]
        assert [c[1] for c in publishes] == [["a"], ["b"]]
        assert all(c[2] != owner for c in publishes)  # published on the worker thread
        result = a.get_output()
        assert a.get_output() is result and result.inter_stage_outputs is None  # resolution is cached
        assert result.omni_connector_output == "ready"
        assert calls[-2:] == [("signals", owner)] * 2  # signals resolve lazily on the caller thread
        worker.close()
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit(_Output("c"), plane)
    finally:
        gate.set()
        worker.close()


def test_native_worker_capacity_error_cache_and_tp_gates():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    plane = SimpleNamespace(enqueue_outputs=lambda **kw: None, get_omni_connector_output=lambda: None)
    worker, gate = NativeOutputWorker(1), threading.Event()
    try:
        first = worker.submit(_Output("a", gate=gate), plane)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker.submit, _Output("b"), plane)
            time.sleep(0.05)
            assert not future.done()  # capacity 1: pending materialization is bounded
            gate.set()
            second = future.result(3)
        assert first.get_output().req_ids == ["a"] and second.get_output().req_ids == ["b"]
        bad = worker.submit(_Output("x", error=ValueError("crash")), plane)
    finally:
        gate.set()
        worker.close()
    # A failed materialization is cached: repeated reads raise the same error.
    for _ in range(2):
        with pytest.raises(ValueError, match="crash"):
            bad.get_output()
    parallel = SimpleNamespace(tensor_parallel_size=2)
    runner = SimpleNamespace(_omni_data_plane=object(), vllm_config=SimpleNamespace(parallel_config=parallel))
    assert OmniGPUModelRunner._uses_native_output_materializer(runner) is False  # TP>1 keeps TP consumers
    parallel.tensor_parallel_size = 1
    assert OmniGPUModelRunner._uses_native_output_materializer(runner) is True


def test_accumulate_and_pop_decode_delta():
    plane = object.__new__(OmniRunnerDataPlane)
    plane._lock, plane._request_ids_mapping = threading.Lock(), {"internal": "external"}
    decode = torch.tensor([[1.0], [2.0]])
    accumulated = {
        "embed": {"prefill": "prefill", "decode": decode, "decode_token_start": 1, "decode_token_end": 3},
        "ids": {"prompt": [1, 2], "output": [3, 4]},
    }
    plane._send_side_request_payload = {"external": accumulated}
    plane._local_stage_payload_cache = {"internal": accumulated}
    payload = plane.pop_local_stage_payload("internal")
    # pop hands the delta to the model and acks its rows in connector accumulation.
    assert payload["embed"]["decode"] is decode
    assert accumulated["embed"] == {"prefill": "prefill"} and accumulated["ids"] == {"prompt": [1, 2]}
    # Absolute decode spans do not duplicate when re-accumulated.
    plane._send_side_request_payload = {}
    span = {"embed": {"decode": decode, "decode_token_start": 1, "decode_token_end": 3}}
    plane._accumulate_payload("external", span)
    merged = plane._accumulate_payload("external", span)
    assert merged["embed"]["decode_token_start"] == 1 and torch.equal(merged["embed"]["decode"], decode)
