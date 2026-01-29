# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors import adapter as adapter_mod
from vllm_omni.distributed.omni_connectors.adapter import get_chunk_for_generation, put_chunk
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector


def test_shm_connector_cleanup_is_idempotent_and_clears_state():
    connector = SharedMemoryConnector(config={"stage_id": 1})
    request_id = "external_req"

    connector.put_requests[request_id] = 3
    connector.get_requests[request_id] = 2
    connector.finished_requests.add(request_id)
    connector.request_payload[request_id] = {"tmp": True}
    connector.request_prompt_token_ids[request_id] = [1, 2, 3]
    connector.code_prompt_token_ids[request_id] = [[1], [2]]

    # Mapping can include request_id as key or value (internal -> external).
    connector.request_ids_mapping[request_id] = "some_value"
    connector.request_ids_mapping["internal_req"] = request_id

    connector.cleanup(request_id)

    assert request_id not in connector.put_requests
    assert request_id not in connector.get_requests
    assert request_id not in connector.finished_requests
    assert request_id not in connector.request_payload
    assert request_id not in connector.request_prompt_token_ids
    assert request_id not in connector.code_prompt_token_ids
    assert request_id not in connector.request_ids_mapping
    assert request_id not in connector.request_ids_mapping.values()

    # Idempotent
    connector.cleanup(request_id)


def test_get_chunk_for_generation_cleans_up_on_finished(monkeypatch):
    connector = SharedMemoryConnector(config={"stage_id": 2})
    request_id = "rid_1"

    payload = {"code_predictor_codes": [10, 20], "finished": True}

    monkeypatch.setattr(adapter_mod, "get_through_connector", lambda *args, **kwargs: payload)

    request = MagicMock()
    request.external_req_id = request_id
    request.status = None
    request.prompt_token_ids = []

    get_chunk_for_generation(connector, request)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request_id not in connector.get_requests
    assert request_id not in connector.finished_requests


def test_put_chunk_cleans_up_sender_state_on_finished(monkeypatch):
    connector = SharedMemoryConnector(config={"stage_id": 0})
    request_id = "rid_sender"

    # Make connector.put succeed without actually allocating SHM.
    monkeypatch.setattr(connector, "put", MagicMock(return_value=(True, 1, None)))

    request = MagicMock()
    request.external_req_id = request_id
    request.prompt_token_ids = [1, 2, 3]

    def process(pooling_output, request):
        return {"code_predictor_codes": [1], "finished": True}

    put_chunk(connector, pooling_output={}, request=request, custom_process_input_func=process)

    assert request_id not in connector.put_requests
    assert request_id not in connector.request_prompt_token_ids

