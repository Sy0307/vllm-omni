from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Full, Queue
from types import SimpleNamespace
from typing import Any

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)


@dataclass
class _NativeRequestState:
    request_id: str
    external_req_id: str
    prompt_token_ids: list[int]
    additional_information: Any = None
    sampling_params: Any = None
    num_computed_tokens: int = 0
    resumable: bool = False
    output_token_ids: list[int] = field(default_factory=list)
    finished: bool = False

    def snapshot(self, *, include_token_history: bool) -> SimpleNamespace:
        prompt = list(self.prompt_token_ids) if include_token_history else []
        output = list(self.output_token_ids) if include_token_history else []
        finished = self.finished
        request = SimpleNamespace(
            request_id=self.request_id,
            req_id=self.request_id,
            external_req_id=self.external_req_id,
            prompt_token_ids=prompt,
            output_token_ids=output,
            all_token_ids=prompt + output,
            output_token_count=len(self.output_token_ids),
            additional_information=self.additional_information,
            sampling_params=self.sampling_params,
            num_computed_tokens=self.num_computed_tokens,
            resumable=self.resumable,
        )
        request.is_finished = lambda: finished
        return request


@dataclass(frozen=True)
class _NativeOutputBatch:
    req_ids: list[str]
    inter_stage_outputs: list[Any | None] | None
    sampled_token_ids: list[list[int]] | None


_NATIVE_OUTPUT_STOP = object()
_NATIVE_OUTPUT_QUEUE_DEPTH = 8


class OmniRunnerDataPlane(OmniConnectorModelRunnerMixin):
    """MRv2-owned stage transport and request-side payload state."""

    def __init__(self, vllm_config: Any, model_config: Any) -> None:
        self._native_requests: dict[str, _NativeRequestState] = {}
        self._native_output_lock = threading.RLock()
        self._native_send_lock = threading.Lock()
        self._native_outputs_in_flight: dict[str, int] = defaultdict(int)
        self._native_terminal_pending: set[str] = set()
        self.init_omni_connectors(vllm_config=vllm_config, model_config=model_config)
        self._can_send = self._custom_process_func is not None
        self._start_output_worker(max_pending_batches=_NATIVE_OUTPUT_QUEUE_DEPTH)

    def _start_output_worker(self, *, max_pending_batches: int) -> None:
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self._native_output_queue: Queue[_NativeOutputBatch | object] = Queue(maxsize=max_pending_batches)
        self._native_output_error_lock = threading.Lock()
        self._native_output_error: BaseException | None = None
        self._native_output_closed = False
        stage_id = getattr(self, "_stage_id", "unknown")
        self._native_output_worker = threading.Thread(
            target=self._output_worker_loop,
            name=f"omni-native-output-{stage_id}",
            daemon=True,
        )
        self._native_output_worker.start()

    def _record_output_error(self, error: BaseException) -> None:
        with self._native_output_error_lock:
            if self._native_output_error is None:
                self._native_output_error = error

    def _raise_output_error(self) -> None:
        with self._native_output_error_lock:
            error = self._native_output_error
        if error is not None:
            raise RuntimeError(f"native output worker failed: {error}") from error

    def _output_worker_loop(self) -> None:
        while True:
            batch = self._native_output_queue.get()
            try:
                if batch is _NATIVE_OUTPUT_STOP:
                    return
                if self._native_output_error is not None:
                    continue
                assert isinstance(batch, _NativeOutputBatch)
                self.complete_outputs(
                    req_ids=batch.req_ids,
                    inter_stage_outputs=batch.inter_stage_outputs,
                    sampled_token_ids=batch.sampled_token_ids,
                )
            except BaseException as error:
                self._record_output_error(error)
            finally:
                self._native_output_queue.task_done()

    def enqueue_outputs(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
    ) -> None:
        """Transfer one completed model batch to the ordered output worker."""
        self._raise_output_error()
        if self._native_output_closed:
            raise RuntimeError("native output worker is closed")
        batch = _NativeOutputBatch(
            req_ids=req_ids,
            inter_stage_outputs=inter_stage_outputs,
            sampled_token_ids=sampled_token_ids,
        )
        while True:
            try:
                self._native_output_queue.put(batch, timeout=0.1)
                break
            except Full:
                self._raise_output_error()
        self._raise_output_error()

    def drain_outputs(self) -> None:
        self._native_output_queue.join()
        self._raise_output_error()

    def _stop_output_worker(self) -> None:
        worker = getattr(self, "_native_output_worker", None)
        if worker is None:
            return
        self._native_output_closed = True
        self._native_output_queue.put(_NATIVE_OUTPUT_STOP)
        worker.join()
        self._native_output_worker = None

    def register_request(self, request_data: Any) -> None:
        self._raise_output_error()
        req_id = str(request_data.req_id)
        external_req_id = str(getattr(request_data, "external_req_id", None) or req_id)
        with self._native_output_lock:
            self._native_outputs_in_flight.pop(req_id, None)
            self._native_terminal_pending.discard(req_id)
            self._native_requests[req_id] = _NativeRequestState(
                request_id=req_id,
                external_req_id=external_req_id,
                prompt_token_ids=list(getattr(request_data, "prompt_token_ids", None) or []),
                additional_information=getattr(request_data, "additional_information", None),
                sampling_params=getattr(request_data, "sampling_params", None),
                num_computed_tokens=int(getattr(request_data, "num_computed_tokens", 0) or 0),
                resumable=bool(getattr(request_data, "resumable", False)),
            )

    def reserve_outputs(self, req_ids: list[str]) -> None:
        """Keep request state alive until deferred runner outputs are consumed."""
        self._raise_output_error()
        with self._native_output_lock:
            for req_id in dict.fromkeys(req_ids):
                if req_id in self._native_requests:
                    self._native_outputs_in_flight[req_id] += 1

    def _ready_terminal_requests(self) -> set[str]:
        return {
            req_id for req_id in self._native_terminal_pending if self._native_outputs_in_flight.get(req_id, 0) == 0
        }

    def _emit_ready_terminals(self) -> int:
        ready = self._ready_terminal_requests()
        if not ready:
            return 0
        emitted = self.emit_chunks(
            req_ids=[],
            inter_stage_outputs=None,
            sampled_token_ids=None,
            terminal_req_ids=ready,
        )
        self._native_terminal_pending.difference_update(ready)
        for req_id in ready:
            self._native_outputs_in_flight.pop(req_id, None)
        return emitted

    def request_terminal(self, req_ids: set[str]) -> int:
        """Emit terminal markers only after all deferred outputs are enqueued."""
        self._raise_output_error()
        with self._native_output_lock:
            self._native_terminal_pending.update(req_id for req_id in req_ids if req_id in self._native_requests)
            return self._emit_ready_terminals()

    def abort_requests(self, req_ids: set[str]) -> int:
        """Cancel deferred outputs and terminate each live request once."""
        with self._native_output_lock:
            active_req_ids = {req_id for req_id in req_ids if req_id in self._native_requests}
            if not active_req_ids:
                return 0
            for req_id in active_req_ids:
                self._native_outputs_in_flight.pop(req_id, None)
                self._native_terminal_pending.discard(req_id)
            return self.emit_chunks(
                req_ids=[],
                inter_stage_outputs=None,
                sampled_token_ids=None,
                terminal_req_ids=active_req_ids,
            )

    def complete_outputs(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
    ) -> int:
        """Commit one deferred output batch, then release its lifecycle holds."""
        send_lock_held = False
        with self._native_output_lock:
            entries, finished_req_ids = self._build_chunk_entries(
                req_ids=req_ids,
                inter_stage_outputs=inter_stage_outputs,
                sampled_token_ids=sampled_token_ids,
                terminal_req_ids=set(),
            )
            # Claim connector ordering before publishing the updated request
            # state. request_terminal() can still register a pending terminal
            # while the enqueue runs, but abort cannot overtake this output.
            if entries:
                self._native_send_lock.acquire()
                send_lock_held = True
        try:
            emitted = self._send_chunk_entries(
                entries,
                send_lock_held=send_lock_held,
            )
        finally:
            if send_lock_held:
                self._native_send_lock.release()
        with self._native_output_lock:
            self._cleanup_finished_entries(finished_req_ids)
            for req_id in dict.fromkeys(req_ids):
                in_flight = self._native_outputs_in_flight.get(req_id, 0)
                if in_flight > 1:
                    self._native_outputs_in_flight[req_id] = in_flight - 1
                elif in_flight == 1:
                    self._native_outputs_in_flight.pop(req_id, None)
            return emitted + self._emit_ready_terminals()

    def register_receivers(self, handles: list[Any]) -> None:
        for handle in handles:
            self.register_chunk_recv(handle)

    @classmethod
    def _copy_payload_structure(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._copy_payload_structure(item) for key, item in value.items()}
        if isinstance(value, list):
            return list(value)
        return value

    def pop_local_stage_payload(self, req_id: str) -> Any:
        """Hand one accumulated delta to the model and acknowledge its rows.

        Connector accumulation and ``model_intermediate_buffer`` are separate
        ownership domains. Once decode rows are handed to the model, remove
        them from connector accumulation so the next chunk cannot replay the
        same rows into Talker's ``cached_decode``.
        """
        with self._lock:
            payload = self._local_stage_payload_cache.pop(req_id, None)
            if payload is None:
                return None
            delivered = self._copy_payload_structure(payload)
            external_req_id = self._request_ids_mapping.get(req_id, req_id)
            accumulated = self._send_side_request_payload.get(external_req_id)
            if isinstance(accumulated, dict):
                embed = accumulated.get("embed")
                if isinstance(embed, dict):
                    embed.pop("decode", None)
                    embed.pop("decode_token_start", None)
                    embed.pop("decode_token_end", None)
                hidden_states = accumulated.get("hidden_states")
                if isinstance(hidden_states, dict):
                    hidden_states.pop("output", None)
                ids = accumulated.get("ids")
                if isinstance(ids, dict):
                    ids.pop("output", None)
            return delivered

    def emit_chunks(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
        terminal_req_ids: set[str],
    ) -> int:
        with self._native_output_lock:
            entries, finished_req_ids = self._build_chunk_entries(
                req_ids=req_ids,
                inter_stage_outputs=inter_stage_outputs,
                sampled_token_ids=sampled_token_ids,
                terminal_req_ids=terminal_req_ids,
            )
            emitted = self._send_chunk_entries(entries)
            self._cleanup_finished_entries(finished_req_ids)
            return emitted

    def _build_chunk_entries(
        self,
        *,
        req_ids: list[str],
        inter_stage_outputs: list[Any | None] | None,
        sampled_token_ids: list[list[int]] | None,
        terminal_req_ids: set[str],
    ) -> tuple[list[tuple[Any, Any | None]], list[str]]:
        if not getattr(self, "_can_send", True):
            return [], []
        payload_by_req = {
            req_id: inter_stage_outputs[index]
            for index, req_id in enumerate(req_ids)
            if inter_stage_outputs is not None and index < len(inter_stage_outputs)
        }
        sampled_by_req = {
            req_id: sampled_token_ids[index]
            for index, req_id in enumerate(req_ids)
            if sampled_token_ids is not None and index < len(sampled_token_ids)
        }
        entries: list[tuple[Any, Any | None]] = []
        finished_req_ids: list[str] = []
        for req_id in dict.fromkeys([*req_ids, *sorted(terminal_req_ids)]):
            state = self._native_requests.get(req_id)
            if state is None:
                continue
            state.output_token_ids.extend(int(token_id) for token_id in sampled_by_req.get(req_id, []))
            state.finished = req_id in terminal_req_ids
            payload = payload_by_req.get(req_id)
            if isinstance(payload, dict):
                payload = unflatten_payload(payload)
            if payload is None and not state.finished:
                continue
            include_token_history = bool(state.resumable or self._put_req_chunk.get(state.external_req_id, 0) == 0)
            entries.append(
                (
                    state.snapshot(include_token_history=include_token_history),
                    payload,
                )
            )
            if state.finished:
                finished_req_ids.append(req_id)
        return entries, finished_req_ids

    def _send_chunk_entries(
        self,
        entries: list[tuple[Any, Any | None]],
        *,
        send_lock_held: bool = False,
    ) -> int:
        if not entries:
            return 0
        if send_lock_held:
            return self.send_chunks(entries, propagate_errors=True)
        with self._native_send_lock:
            return self.send_chunks(entries, propagate_errors=True)

    def _cleanup_finished_entries(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self._native_requests.pop(req_id, None)
            self.cleanup_finished_request(req_id)

    def close(self) -> None:
        error = None
        try:
            self.drain_outputs()
        except BaseException as exc:
            error = exc
        finally:
            self._stop_output_worker()
            self.shutdown_omni_connectors()
        if error is not None:
            raise error
