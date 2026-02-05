# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Temporary compatibility shim for vllm_omni.entrypoints.omni_stage.py / omni_llm.py.

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    from .connectors.base import OmniConnectorBase

from vllm_omni.entrypoints.stage_utils import OmniStageTaskType

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[dict[str, Any]], None],
    metrics: Any,
) -> bool:
    """Send payload via OmniConnector and enqueue notification/metrics; return True on success."""
    try:
        t0 = time.time()

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": original_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time(),
            },
        }

        # Send data via connector
        success, serialized_size, metadata = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)

        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "type": OmniStageTaskType.GENERATE,
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            # Merge connector metadata (e.g. shm handle or inline data) into queue payload
            if metadata:
                notify_payload["connector_metadata"] = metadata

            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0

            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        logger.warning(
            "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
            req_id,
            e,
        )
        return False


def try_recv_via_connector(
    task: dict[str, Any],
    connectors: dict[Any, Any],
    stage_id: int,
) -> tuple[Any, dict[str, Any] | None]:
    """Resolve engine_inputs from connector/IPC payload; returns (engine_inputs, rx_metrics) or (None, None)."""
    rid = task["request_id"]

    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        to_stage = str(stage_id)

        if not from_stage:
            logger.error(
                "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
            )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                connector_metadata = task.get("connector_metadata")
                payload = connector.get(from_stage, to_stage, str(rid), metadata=connector_metadata)
                _t_end = time.time()

                if payload:
                    if isinstance(payload, tuple):
                        payload_data, serialized_size = payload
                    else:
                        payload_data = payload
                        serialized_size = len(connector.serialize_obj(payload_data))
                else:
                    payload_data = None
                    serialized_size = 0

                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    decode_ms = (_t_end - _t_start) * 1000.0

                    rx_metrics = {"rx_decode_time_ms": decode_ms, "rx_transfer_bytes": serialized_size}
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                logger.error("[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e)
                return None, None
        else:
            logger.error(
                "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            )
            return None, None
    else:
        # Queue path (e.g. Stage-0 seed): task should carry direct inputs, but still decode SHM/IPC if present.

        # Try to use the new stage_utils which uses OmniSerializer
        from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc_with_metrics

        try:
            ein, metrics = maybe_load_from_ipc_with_metrics(task, "engine_inputs", "engine_inputs_shm")
            # If metrics are empty or zero, we might want to populate dummy metrics
            return ein, metrics
        except Exception:
            # If engine_inputs is missing, it might be a different kind of payload,
            # but for Stage-0 seed it should be there.
            # We'll return None to let caller handle error if strictly required.
            return None, None


def update_request_payload(connector: "OmniConnectorBase", req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
    """Update the payload data for a request in the connector.

    Args:
        connector: OmniConnectorBase instance
        req_id: Request ID to update
        payload_data: New payload data to store
    """
    origin_payload = connector.request_payload[req_id]
    for key, value in payload_data.items():
        if key == "finished":
            continue
        elif isinstance(value, torch.Tensor) and key in origin_payload:
            payload_data[key] = torch.cat([origin_payload[key], value], dim=0)
        elif isinstance(value, list) and key in origin_payload:
            payload_data[key] = origin_payload[key] + value

    connector.request_payload[req_id] = payload_data
    return payload_data


def get_chunk(
    connector: "OmniConnectorBase",
    scheduler_output: SchedulerOutput,
) -> None:
    """Fetch connector chunks and populate scheduler_output.additional_information (in-place)."""
    stage_id = connector.stage_id
    if stage_id == 0:
        return

    target_stage_id = stage_id - 1
    # Handle new requests
    for new_req_data in scheduler_output.scheduled_new_reqs:
        connector.request_ids_mapping[new_req_data.req_id] = new_req_data.external_req_id
        req_id = new_req_data.external_req_id
        chunk_id = connector.get_requests[req_id]
        connector_get_key = f"{req_id}_{target_stage_id}_{chunk_id}"
        payload_data = get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key)
        if payload_data:
            new_req_data.additional_information = payload_data
            connector.request_payload[req_id] = payload_data
            if payload_data.get("finished"):
                connector.finished_requests.add(req_id)

    # Handle cached/running requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    if not hasattr(cached_reqs, "additional_information"):
        cached_reqs.additional_information = {}

    for i, cached_req_id in enumerate(cached_reqs.req_ids):
        req_id = connector.request_ids_mapping.get(cached_req_id, cached_req_id)
        if req_id in connector.finished_requests:
            continue
        chunk_id = connector.get_requests[req_id]
        connector_get_key = f"{req_id}_{target_stage_id}_{chunk_id}"
        payload_data = get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key)
        if payload_data:
            payload_data = update_request_payload(connector, req_id, payload_data)
            cached_reqs.additional_information[cached_req_id] = payload_data
            if payload_data.get("finished"):
                connector.finished_requests.add(req_id)


def get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key):
    # TODO: add correct check mechanism for the payload_data
    try:
        chunk_id = int(str(connector_get_key).rsplit("_", 1)[1])
    except Exception:
        chunk_id = 0
    max_wait = 3000 if chunk_id == 0 else 300
    for _ in range(max_wait):
        result = connector.get(
            from_stage=str(target_stage_id),
            to_stage=str(stage_id),
            get_key=connector_get_key,
        )
        payload_data = None
        if result:
            payload_data, size = result
            if payload_data:
                connector.request_prompt_token_ids[req_id] = payload_data.get("thinker_input_ids", [])
                connector.get_requests[req_id] += 1
                logger.debug("[Stage-%d] Received one chunk for request %s", stage_id, connector_get_key)
                break
        time.sleep(0.01)
    return payload_data


def get_chunk_for_generation(
    connector: "OmniConnectorBase",
    request: Request,
) -> None:
    """Fetch one connector chunk and update request metadata + prompt_token_ids (in-place)."""
    stage_id = connector.stage_id
    target_stage_id = stage_id - 1
    request_id = request.external_req_id

    if request_id in connector.finished_requests:
        return

    chunk_id = connector.get_requests[request_id]
    connector_get_key = f"{request_id}_{target_stage_id}_{chunk_id}"
    payload_data = get_through_connector(connector, target_stage_id, stage_id, request_id, connector_get_key)
    if not payload_data:
        return

    # Persist codec streaming metadata on request.additional_information (survives scheduler->worker serialization).
    if isinstance(payload_data, dict):
        ai = request.additional_information
        if not isinstance(ai, dict):
            ai = {}
            request.additional_information = ai
        for k in (
            "codec_context_frames",
            "codec_context_codes",
            "codec_total_frames",
            "codec_chunk_frames",
            "codec_num_code_groups",
            "codec_layout",
            "codec_streaming",
        ):
            if k in payload_data:
                ai[k] = payload_data[k]

    try:
        if isinstance(payload_data, dict):
            codes = payload_data.get("code_predictor_codes", None)
            clen = len(codes) if isinstance(codes, list) else None
            logger.info(
                "[Stage-%d] recv chunk=%s finished=%s chunk_len=%s ctx_len=%s codec_streaming=%s ctx_frames=%s",
                stage_id,
                connector_get_key,
                bool(payload_data.get("finished")),
                clen,
                (
                    len(payload_data.get("codec_context_codes", []))
                    if isinstance(payload_data.get("codec_context_codes"), list)
                    else None
                ),
                payload_data.get("codec_streaming"),
                payload_data.get("codec_context_frames"),
            )
    except Exception:
        pass

    # Upstream finished producing chunks; don't force request.status (stage still needs to consume tokens).
    if payload_data.get("finished"):
        connector.finished_requests.add(request_id)
        request.status = RequestStatus.FINISHED_STOPPED
    request.prompt_token_ids = payload_data.get("code_predictor_codes", [])
    request.num_computed_tokens = 0


def put_chunk(
    connector: "OmniConnectorBase",
    pooling_output: dict[str, Any],
    request: Request,
    custom_process_input_func: Callable[[dict[str, Any], Request], dict[str, Any] | None] | None = None,
) -> None:
    """Send one pooling_output chunk to next stage via connector (optionally processed)."""
    stage_id = connector.stage_id
    next_stage_id = stage_id + 1
    request_id = request.external_req_id
    chunk_id = connector.put_requests[request_id]
    connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"
    payload_data = None

    # TODO: add default process_input_func to handle the payload_data ?
    if custom_process_input_func:
        try:
            payload_data = custom_process_input_func(
                connector=connector,
                pooling_output=pooling_output,
                request=request,
            )
        except Exception as e:
            logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if not payload_data:
            return

        # Qwen3-Omni thinker->talker: merge split payload parts on the first chunk only.
        if (
            stage_id == 0
            and chunk_id == 0
            and (("thinker_embeddings" in payload_data) or ("thinker_hidden_states" in payload_data))
        ):
            if connector.request_payload.get(request_id) is None:
                if not payload_data.get("finished"):
                    connector.request_payload[request_id] = payload_data
                    return
            else:
                save_payload = connector.request_payload.pop(request_id)
                if (
                    isinstance(save_payload, dict)
                    and isinstance(save_payload.get("thinker_embeddings"), torch.Tensor)
                    and isinstance(payload_data.get("thinker_embeddings"), torch.Tensor)
                ):
                    payload_data["thinker_embeddings"] = torch.cat(
                        (save_payload.get("thinker_embeddings"), payload_data.get("thinker_embeddings")), dim=0
                    )
                if (
                    isinstance(save_payload, dict)
                    and isinstance(save_payload.get("thinker_hidden_states"), torch.Tensor)
                    and isinstance(payload_data.get("thinker_hidden_states"), torch.Tensor)
                ):
                    payload_data["thinker_hidden_states"] = torch.cat(
                        (save_payload.get("thinker_hidden_states"), payload_data.get("thinker_hidden_states")), dim=0
                    )
                logger.debug("[Stage-%d] Merged embeddings and hidden states for request %s", stage_id, request_id)

        # Frame-aligned codec streaming: repack per-frame codes into (left_context + chunk) windows.
        if isinstance(payload_data, dict) and "code_predictor_codes" in payload_data and stage_id in (0, 1):
            raw_cfg = getattr(connector, "config", {}) or {}
            # Connector config commonly nests user options under {"extra": {...}}.
            cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
            if stage_id == 1 or bool(cfg.get("codec_streaming", False)):
                # async_chunk requires streaming windows; disallow explicit codec_streaming=False.
                req_streaming = payload_data.get("codec_streaming")
                if isinstance(req_streaming, torch.Tensor):
                    try:
                        req_streaming = bool(req_streaming.item())
                    except Exception:
                        req_streaming = None
                if req_streaming is False:
                    raise ValueError(
                        "codec_streaming=False is not supported for async_chunk code2wav pipelines. "
                        "Enable codec_streaming or switch to a non-streaming stage config."
                    )

                chunk_size = int(cfg.get("codec_chunk_frames", 25))
                left_context_size = int(cfg.get("codec_left_context_frames", 25))
                if chunk_size <= 0 or left_context_size < 0:
                    raise ValueError(
                        f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
                        f"codec_left_context_frames={left_context_size}"
                    )

                frame_codes = payload_data.get("code_predictor_codes", [])
                appended_frame = False
                if isinstance(frame_codes, list) and len(frame_codes) > 0:
                    connector.code_prompt_token_ids[request_id].append(frame_codes)
                    appended_frame = True
                elif not payload_data.get("finished"):
                    # For non-finished steps we require one frame per payload.
                    return

                length = len(connector.code_prompt_token_ids[request_id])
                chunk_length = length % chunk_size
                if chunk_length != 0 and not payload_data.get("finished"):
                    return

                # On finished: flush remainder (if any).
                context_length = chunk_length if chunk_length != 0 else chunk_size
                if payload_data.get("finished") and (not appended_frame) and chunk_length == 0:
                    # No remainder to flush; avoid resending the last full chunk.
                    payload_data["code_predictor_codes"] = []
                    payload_data["codec_context_codes"] = []
                    payload_data["codec_context_frames"] = 0
                    payload_data["codec_total_frames"] = 0
                    payload_data["codec_chunk_frames"] = 0
                    payload_data["codec_num_code_groups"] = 0
                    payload_data["codec_layout"] = "codebook_major"
                elif length <= 0:
                    # No codes to decode; still forward finished marker.
                    payload_data["code_predictor_codes"] = []
                    payload_data["codec_context_codes"] = []
                    payload_data["codec_context_frames"] = 0
                    payload_data["codec_total_frames"] = 0
                    payload_data["codec_chunk_frames"] = 0
                    payload_data["codec_num_code_groups"] = 0
                    payload_data["codec_layout"] = "codebook_major"
                else:
                    end_index = min(length, left_context_size + context_length)
                    ctx_frames = max(0, int(end_index - context_length))
                    window_frames = connector.code_prompt_token_ids[request_id][-end_index:]
                    # Send chunk tokens via prompt_token_ids; send left-context separately via codec_context_codes.
                    if ctx_frames > 0:
                        ctx_part = window_frames[:ctx_frames]
                        payload_data["codec_context_codes"] = (
                            torch.tensor(ctx_part).transpose(0, 1).reshape(-1).tolist()
                        )
                    else:
                        payload_data["codec_context_codes"] = []
                    chunk_part = window_frames[ctx_frames:]
                    payload_data["code_predictor_codes"] = torch.tensor(chunk_part).transpose(0, 1).reshape(-1).tolist()
                    payload_data["codec_context_frames"] = int(ctx_frames)
                    payload_data["codec_total_frames"] = int(end_index)
                    payload_data["codec_chunk_frames"] = int(context_length)
                    payload_data["codec_num_code_groups"] = int(
                        len(connector.code_prompt_token_ids[request_id][-1])
                        if connector.code_prompt_token_ids[request_id]
                        else 0
                    )
                    payload_data["codec_layout"] = "codebook_major"
        success, size, metadata = connector.put(
            from_stage=str(stage_id), to_stage=str(next_stage_id), put_key=connector_put_key, data=payload_data
        )

        if success:
            connector.put_requests[request_id] += 1
            logger.debug("[Stage-%d] Sent %s", stage_id, connector_put_key)


def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
    """Compute talker prompt length for chat-style prompt ids (system/user/assistant)."""
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091
    im_start_indexes = [i for i in range(len(prompt_ids)) if prompt_ids[i] == im_start_token_id]
    im_start_indexes.append(len(prompt_ids))
    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = im_start_indexes[i]
        e = im_start_indexes[i + 1]
        role = prompt_ids[s + 1]
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len
