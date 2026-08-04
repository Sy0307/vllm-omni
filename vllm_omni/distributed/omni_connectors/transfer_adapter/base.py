# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from typing import Any

from vllm_omni.outputs import StageTransferFailure

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()
        self._cancelled_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()
        self._transfer_failures: dict[str, StageTransferFailure] = {}
        self._transfer_failure_lock = threading.Lock()

        self.stop_event = threading.Event()
        self._recv_cond = threading.Condition()
        self._save_cond = threading.Condition()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data.

        Process each pending request exactly once per pass.  When no request
        made progress, back off 1 ms instead of tight-spinning on failed
        shm_open syscalls (which can burn a full CPU core).
        """
        while not self.stop_event.is_set():
            n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                if not self._pending_load_reqs:
                    break
                request = self._pending_load_reqs.popleft()
                request_id = request.request_id
                if request_id in self._cancelled_load_reqs:
                    self._cancelled_load_reqs.discard(request_id)
                    continue
                self.request_ids_mapping[request_id] = request.external_req_id
                try:
                    is_success = self._poll_single_request(request)
                    if is_success:
                        any_success = True
                    else:
                        self._pending_load_reqs.append(request)
                except Exception as e:
                    self._pending_load_reqs.append(request)
                    logger.warning(f"Error receiving data for {request_id}: {e}")

            # Timeout is the fallback for lock-free append/notify races.
            with self._recv_cond:
                if not self._pending_load_reqs and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.1)
                elif not any_success and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            while self._pending_save_reqs:
                task = self._pending_save_reqs.popleft()
                try:
                    success = self._send_single_request(task)
                except Exception:
                    logger.warning(
                        "Error saving data for %s",
                        task.get("internal_request_id") or task.get("request_id"),
                        exc_info=True,
                    )
                    success = False
                if success is False:
                    self._requeue_or_fail_send(task)

            with self._save_cond:
                if not self._pending_save_reqs and not self.stop_event.is_set():
                    self._save_cond.wait(timeout=0.1)

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    _MAX_SEND_RETRIES = 3

    def _requeue_or_fail_send(self, task: dict[str, Any]) -> None:
        retry_count = int(task.get("_retry_count", 0)) + 1
        if retry_count <= self._MAX_SEND_RETRIES:
            task["_retry_count"] = retry_count
            self._pending_save_reqs.appendleft(task)
            logger.warning(
                "Re-enqueuing failed stage send for %s (retry %d/%d)",
                task.get("internal_request_id") or task.get("request_id"),
                retry_count,
                self._MAX_SEND_RETRIES,
            )
            return

        self._record_transfer_failure_from_task(
            task,
            code="stage_payload_transport_failed",
            message="connector send failed after bounded retries",
        )
        self._settle_failed_send(task)

    def _record_transfer_failure_from_task(
        self,
        task: dict[str, Any],
        *,
        code: str,
        message: str,
    ) -> None:
        internal_request_id = task.get("internal_request_id") or task.get("request_id")
        if not internal_request_id:
            logger.error("Cannot deliver %s without an internal request ID", code)
            return
        source_stage = int(task.get("stage_id", getattr(self.connector, "stage_id", 0)))
        destination_stage = int(task.get("next_stage_id", source_stage + 1))
        failure = StageTransferFailure(
            internal_request_id=internal_request_id,
            external_request_id=task.get("external_request_id"),
            source_stage=source_stage,
            destination_stage=destination_stage,
            code=code,
            message=message,
        )
        with self._transfer_failure_lock:
            self._transfer_failures.setdefault(internal_request_id, failure)

    def _settle_failed_send(self, task: dict[str, Any]) -> None:
        """Subclass hook for pending-save and cleanup bookkeeping."""

    def drain_transfer_failures(self) -> dict[str, StageTransferFailure]:
        with self._transfer_failure_lock:
            failures = self._transfer_failures
            self._transfer_failures = {}
        return failures

    def shutdown(self):
        """Stop background loops and close the connector."""
        self.stop_event.set()
        with self._recv_cond:
            self._recv_cond.notify_all()
        with self._save_cond:
            self._save_cond.notify_all()
        if self.connector is not None:
            try:
                self.connector.close()
            except Exception:
                pass
