# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import threading
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.fish_dac_ready_queue = (
            (
                os.environ.get("VLLM_FISH_DAC_READY_QUEUE", "0") == "1"
                or os.environ.get("VLLM_FISH_DAC_DEDICATED_WORKER", "0") == "1"
            )
            and self.connector.stage_id != 0
            and self.model_mode != "ar"
        )
        self._ready_chunk_reqs: deque[Request] = deque()
        self._ready_chunk_req_ids: set[str] = set()
        self._ready_chunk_lock = threading.Lock()
        self._ready_callback = None
        self.ready_only_scheduling = (
            os.environ.get("VLLM_FISH_CHUNK_READY_ONLY_SCHED", "0") == "1"
            and self.connector.stage_id != 0
            and self.model_mode != "ar"
        )
        self.fish_dac_burst_recv = (
            os.environ.get("VLLM_FISH_DAC_BURST_RECV", "0") == "1"
            and self.connector.stage_id != 0
            and self.model_mode != "ar"
        )
        self.fish_dac_burst_max_chunks = max(
            1,
            int(os.environ.get("VLLM_FISH_DAC_BURST_MAX_CHUNKS", "4") or 4),
        )
        self.fish_dac_inline_poll = (
            os.environ.get("VLLM_FISH_DAC_INLINE_POLL", "0") == "1"
            and self.connector.stage_id != 0
            and self.model_mode != "ar"
        )
        self.fish_dac_inline_poll_profile = (
            os.environ.get("VLLM_FISH_DAC_INLINE_POLL_PROFILE", "0") == "1"
            and self.fish_dac_inline_poll
        )

    def set_ready_callback(self, callback) -> None:
        """Register a callback fired when a new chunk becomes schedulable."""
        self._ready_callback = callback

    def _notify_ready_callback(self) -> None:
        callback = self._ready_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.debug("Fish DAC ready callback failed", exc_info=True)

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        pooling_output: torch.Tensor | None = None,
        request: Request | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        task = {
            "pooling_output": pooling_output,
            "request": request,
            "is_finished": request.is_finished(),
        }
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1
            if self.fish_dac_burst_recv:
                payload_data = self._collect_fish_dac_burst_payloads(
                    first_payload=payload_data,
                    req_id=req_id,
                    external_req_id=external_req_id,
                    target_stage_id=target_stage_id,
                    stage_id=stage_id,
                )

            if self.model_mode == "ar":
                self._update_request_payload(external_req_id, payload_data)
                request.additional_information = payload_data
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)

                new_ids = payload_data.get("code_predictor_codes", [])
                # Preserve previously attached request metadata (e.g. prompt
                # conditioning tensors) and update only per-chunk fields.
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                if isinstance(new_ids, torch.Tensor):
                    new_ids_len = int(new_ids.numel())
                    prompt_len = int(payload_data.get("next_stage_prompt_len", new_ids_len))
                    request.prompt_token_ids = [0] * prompt_len
                    info["code_predictor_codes"] = new_ids
                else:
                    new_ids_len = len(new_ids) if hasattr(new_ids, "__len__") else int(new_ids is not None)
                    request.prompt_token_ids = new_ids
                for key, value in payload_data.items():
                    if key in {"code_predictor_codes", "finished"}:
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                if new_ids_len == 0 and not payload_data.get("finished"):
                    return True

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            if self.fish_dac_ready_queue:
                self._mark_ready_chunk(request)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _mark_ready_chunk(self, request: Request) -> None:
        request_id = request.request_id
        marked = False
        with self._ready_chunk_lock:
            if request_id in self._ready_chunk_req_ids:
                return
            self._ready_chunk_reqs.append(request)
            self._ready_chunk_req_ids.add(request_id)
            marked = True
        if marked:
            self._notify_ready_callback()

    @staticmethod
    def _remove_from_deque(items: deque[Any], request_id: str) -> bool:
        if not items:
            return False
        kept: deque[Any] = deque()
        removed = False
        while items:
            item = items.popleft()
            if getattr(item, "request_id", None) == request_id:
                removed = True
                continue
            kept.append(item)
        items.extend(kept)
        return removed

    def drain_ready_chunks(self, max_chunks: int) -> list[Request]:
        """Pop chunk-ready Fish DAC requests collected by the recv thread.

        The request objects already carry the decoded per-chunk metadata from
        ``_poll_single_request``.  Draining removes them from the adapter's
        WAITING_FOR_CHUNK side queues so the scheduler can submit them directly
        without waiting for a restore pass.
        """
        if not self.fish_dac_ready_queue or max_chunks <= 0:
            return []

        ready: list[Request] = []
        with self._ready_chunk_lock:
            while self._ready_chunk_reqs and len(ready) < max_chunks:
                request = self._ready_chunk_reqs.popleft()
                self._ready_chunk_req_ids.discard(request.request_id)
                ready.append(request)

        for request in ready:
            request_id = request.request_id
            self._finished_load_reqs.discard(request_id)
            self.requests_with_ready_chunks.discard(request_id)
            self._remove_from_deque(
                self.waiting_for_chunk_waiting_requests,
                request_id,
            )
            self._remove_from_deque(
                self.waiting_for_chunk_running_requests,
                request_id,
            )

        if len(ready) < max_chunks:
            ready.extend(
                self._drain_finished_side_queue_chunks(max_chunks - len(ready))
            )

        return ready

    def prepend_ready_chunks(self, requests: list[Request]) -> None:
        """Put drained ready chunks back at the front of the ready queue."""
        if not requests:
            return
        with self._ready_chunk_lock:
            for request in reversed(requests):
                request_id = request.request_id
                if request_id in self._ready_chunk_req_ids:
                    continue
                self._ready_chunk_reqs.appendleft(request)
                self._ready_chunk_req_ids.add(request_id)
                self.requests_with_ready_chunks.add(request_id)

    def rearm_running_chunk_request(self, request: Request) -> bool:
        """Park a live DAC request directly for its next upstream chunk.

        The normal Stage1 lifecycle leaves a chunk-finished request in the
        scheduler's running queue until the next scheduler pass moves it back
        into WAITING_FOR_CHUNK. For Fish DAC streaming that extra pass happens
        for every chunk. This method lets the update path re-arm the connector
        immediately after emitting audio, keeping the request in the adapter's
        side queue until a real chunk is ready again.
        """
        if self.connector.stage_id == 0:
            return False

        request_id = request.request_id
        if request_id in self.finished_requests:
            return False

        self.requests_with_ready_chunks.discard(request_id)
        self._finished_load_reqs.discard(request_id)
        self._remove_from_deque(
            self.waiting_for_chunk_waiting_requests,
            request_id,
        )
        self._remove_from_deque(
            self.waiting_for_chunk_running_requests,
            request_id,
        )
        with self._ready_chunk_lock:
            self._ready_chunk_req_ids.discard(request_id)
            if self._ready_chunk_reqs:
                self._ready_chunk_reqs = deque(
                    req for req in self._ready_chunk_reqs
                    if req.request_id != request_id
                )

        request.status = RequestStatus.WAITING_FOR_CHUNK
        self.request_ids_mapping[request_id] = request.external_req_id
        self.waiting_for_chunk_running_requests.append(request)
        self.load_async(request)
        return True

    def _drain_finished_side_queue_chunks(self, max_chunks: int) -> list[Request]:
        """Drain ready chunks that reached side queues after the ready pop.

        The recv thread can set ``_finished_load_reqs`` immediately before or
        during a scheduler tick.  If the request has not been restored to the
        main scheduler queues yet, waiting until the next tick leaves a batch
        opportunity on the table.  This helper lets the Stage1 fastpath consume
        those side-queue chunks directly.
        """
        if max_chunks <= 0 or not self._finished_load_reqs:
            return []

        ready: list[Request] = []

        def drain_from(side_queue: deque[Any]) -> None:
            if len(ready) >= max_chunks or not side_queue:
                return
            pending: deque[Any] = deque()
            while side_queue:
                request = side_queue.popleft()
                request_id = request.request_id
                if (
                    len(ready) < max_chunks
                    and request_id in self._finished_load_reqs
                ):
                    self._finished_load_reqs.discard(request_id)
                    self.requests_with_ready_chunks.discard(request_id)
                    with self._ready_chunk_lock:
                        self._ready_chunk_req_ids.discard(request_id)
                    ready.append(request)
                else:
                    pending.append(request)
            side_queue.extend(pending)

        drain_from(self.waiting_for_chunk_waiting_requests)
        drain_from(self.waiting_for_chunk_running_requests)
        return ready

    def has_ready_chunks(self) -> bool:
        """Return whether the Fish DAC recv thread has queued ready chunks."""
        if not self.fish_dac_ready_queue:
            return False
        with self._ready_chunk_lock:
            if self._ready_chunk_reqs:
                return True
        if not self._finished_load_reqs:
            return False
        for request in self.waiting_for_chunk_waiting_requests:
            if request.request_id in self._finished_load_reqs:
                return True
        for request in self.waiting_for_chunk_running_requests:
            if request.request_id in self._finished_load_reqs:
                return True
        return False

    @staticmethod
    def _payload_code_len(payload: dict[str, Any]) -> int:
        codes = payload.get("code_predictor_codes", [])
        if isinstance(codes, torch.Tensor):
            return int(codes.numel())
        if hasattr(codes, "__len__"):
            return len(codes)
        return int(codes is not None)

    def _collect_fish_dac_burst_payloads(
        self,
        *,
        first_payload: dict[str, Any],
        req_id: str,
        external_req_id: str,
        target_stage_id: int,
        stage_id: int,
    ) -> dict[str, Any]:
        """Drain consecutive ready Fish DAC chunks for one request.

        This reduces Stage1 request-lifecycle churn when Stage0 has already
        produced multiple chunks. Chunks are kept as separate decode windows so
        left-context trimming remains equivalent to the streaming path.
        """
        payloads = [first_payload]
        while (
            len(payloads) < self.fish_dac_burst_max_chunks
            and not bool(payloads[-1].get("finished"))
        ):
            chunk_id = self.get_req_chunk[req_id]
            connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"
            try:
                result = self.connector.get(
                    str(target_stage_id),
                    str(stage_id),
                    connector_get_key,
                )
            except Exception as exc:
                logger.error(
                    "SharedMemoryConnector burst get failed for req %s: %s",
                    connector_get_key,
                    exc,
                )
                break
            if result is None:
                break
            payload_data, _size = result
            if not payload_data:
                break
            self.get_req_chunk[req_id] += 1
            payloads.append(payload_data)

        if len(payloads) == 1:
            return first_payload

        merged = dict(payloads[-1])
        code_chunks: list[Any] = []
        left_context_sizes: list[int] = []
        prompt_lens: list[int] = []
        for payload in payloads:
            code_len = self._payload_code_len(payload)
            if code_len <= 0:
                continue
            codes = payload.get("code_predictor_codes")
            code_chunks.append(codes)
            left_context_sizes.append(int(payload.get("left_context_size", 0) or 0))
            prompt_lens.append(int(payload.get("next_stage_prompt_len", code_len) or code_len))

        if code_chunks:
            merged["code_predictor_codes"] = code_chunks[0]
            merged["code_predictor_chunks"] = code_chunks
            merged["left_context_sizes"] = left_context_sizes
            merged["next_stage_prompt_lens"] = prompt_lens
            merged["next_stage_prompt_len"] = prompt_lens[0]
            merged["fish_burst_chunk_count"] = len(code_chunks)
        merged["finished"] = any(bool(payload.get("finished")) for payload in payloads)
        if os.environ.get("VLLM_FISH_DAC_BURST_PROFILE", "0") == "1":
            logger.info(
                "[Stage-%s] Fish DAC burst recv: req=%s chunks=%d finished=%s",
                stage_id,
                external_req_id,
                len(code_chunks),
                merged["finished"],
            )
        return merged

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the payload data for a request in the connector.

        Args:
            connector: OmniConnectorBase instance
            req_id: Request ID to update
            payload_data: New payload data to store
        """
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin_payload = self.request_payload[req_id]
        override_keys = payload_data.pop("override_keys", [])
        for key, value in payload_data.items():
            if key == "finished":
                continue
            elif key in override_keys:
                payload_data[key] = value
            elif isinstance(value, torch.Tensor) and key in origin_payload:
                payload_data[key] = torch.cat([origin_payload[key], value], dim=0)
            elif isinstance(value, list) and key in origin_payload:
                payload_data[key] = origin_payload[key] + value

        self.request_payload[req_id] = payload_data
        return payload_data

    def _send_single_request(self, task: dict):
        pooling_output = task["pooling_output"]
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data = None
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                    is_finished=is_finished,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if not payload_data:
            return

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            self.put_req_chunk[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            finished_flag = payload_data.get("finished")
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id)

        if is_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        with self._ready_chunk_lock:
            self._ready_chunk_req_ids.discard(request_id)
            if self._ready_chunk_reqs:
                kept = deque(
                    req for req in self._ready_chunk_reqs
                    if req.request_id != request_id
                )
                self._ready_chunk_reqs = kept

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        if self.ready_only_scheduling:
            self._restore_ready_chunks(
                waiting_queue,
                self.waiting_for_chunk_waiting_requests,
                RequestStatus.WAITING,
            )
            self._restore_ready_chunks(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
            )
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            request.status = RequestStatus.PREEMPTED
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        if self.ready_only_scheduling:
            self._restore_ready_chunks(
                waiting_queue,
                self.waiting_for_chunk_waiting_requests,
                RequestStatus.WAITING,
            )
            self._restore_ready_chunks(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
            )
            return

        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    continue
                # Requests that waiting for chunk
                if self._try_inline_poll_ready_chunk(request, target_status):
                    finished_load_reqs.discard(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _try_inline_poll_ready_chunk(
        self,
        request: Request,
        target_status: RequestStatus,
    ) -> bool:
        """Synchronously poll once before moving a DAC request off-queue.

        The background recv loop is still the fallback.  This only handles
        requests that are about to enter WAITING_FOR_CHUNK, so it does not race
        with a request that is already being polled by the recv thread.
        """
        if not self.fish_dac_inline_poll:
            return False
        request_id = request.request_id
        self._cancelled_load_reqs.discard(request_id)
        self.request_ids_mapping[request_id] = request.external_req_id
        try:
            ready = self._poll_single_request(request)
        except Exception as exc:
            logger.warning("Inline Fish DAC chunk poll failed for %s: %s", request_id, exc)
            return False
        if not ready:
            return False
        request.status = target_status
        if self.fish_dac_inline_poll_profile:
            logger.info(
                "[Stage-%s] Fish DAC inline poll hit: req=%s",
                self.connector.stage_id,
                request.external_req_id,
            )
        return True

    def _restore_ready_chunks(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
    ) -> None:
        """Move only chunk-ready requests back to the scheduler queue.

        In streaming DAC workloads most downstream requests spend many scheduler
        ticks waiting for the next chunk.  Keeping those requests off the main
        scheduler queues avoids a 1ms remove/restore cycle while the background
        recv loop continues polling the connector.
        """
        if not waiting_for_chunk_list or not self._finished_load_reqs:
            return

        pending: deque[Any] = deque()
        while waiting_for_chunk_list:
            request = waiting_for_chunk_list.popleft()
            if request.request_id in self._finished_load_reqs:
                request.status = target_status
                self._finished_load_reqs.remove(request.request_id)
                self.requests_with_ready_chunks.add(request.request_id)
                if target_status == RequestStatus.RUNNING:
                    queue.append(request)
                else:
                    queue.add_request(request)
            else:
                pending.append(request)
        waiting_for_chunk_list.extend(pending)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)
