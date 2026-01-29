# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """
    Connector that uses SharedMemory for large objects and inline data for small objects.
    Acts as a unified replacement for the legacy IPC fallback logic.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.put_requests: dict[str, int] = defaultdict(int)
        self.get_requests: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.request_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}
        # Default threshold matches legacy behavior (64KB)
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            metadata = {}
            # if size > self.threshold:
            if True:  # TODO: correct put & get logic
                # Use Shared Memory
                meta = shm_write_bytes(payload, name=put_key)
                # meta contains {'name': ..., 'size': ...}
                metadata[put_key] = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata[put_key] = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def get(self, from_stage: str, to_stage: str, get_key: str, metadata=None) -> tuple[Any, int] | None:
        from multiprocessing import shared_memory as shm_pkg

        # Wait for shared memory to be available (with retry logic)
        max_retries = 30
        retry_delay = 0.1  # 100ms between retries
        shm = None

        for attempt in range(max_retries):
            try:
                shm = shm_pkg.SharedMemory(name=get_key)
                break  # Successfully opened, exit retry loop
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Max retries reached, return None
                    logger.warning(f"Shared memory '{get_key}' not found after {max_retries} retries")
                    return None, 0

        if shm is None:
            return None, 0

        try:
            data_bytes = shm_read_bytes({"name": get_key, "size": shm.size})
            obj = self.deserialize_obj(data_bytes)
            return obj, shm.size
        finally:
            shm.close()

        # TODO: update another read method

    def cleanup(self, request_id: str) -> None:
        """Best-effort, idempotent cleanup of per-request in-memory state.

        Note: SHM segments created by `put` are unlinked by the consumer in
        `shm_read_bytes`. We intentionally do NOT attempt to unlink SHM by
        request_id here, because (a) the connector currently doesn't track
        created SHM keys per request, and (b) unlinking from the sender side
        before the receiver reads would cause hangs.
        """
        try:
            # Counters / markers
            self.put_requests.pop(request_id, None)
            self.get_requests.pop(request_id, None)
            self.finished_requests.discard(request_id)

            # Streaming caches
            self.request_payload.pop(request_id, None)
            self.request_prompt_token_ids.pop(request_id, None)
            self.code_prompt_token_ids.pop(request_id, None)

            # Mapping can contain request_id as key or value.
            # Example: {internal_req_id -> external_req_id}
            self.request_ids_mapping.pop(request_id, None)
            keys_to_delete = [k for k, v in self.request_ids_mapping.items() if v == request_id]
            for k in keys_to_delete:
                self.request_ids_mapping.pop(k, None)
        except Exception as e:
            logger.warning("SharedMemoryConnector cleanup failed for request_id=%s: %s", request_id, e)

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
