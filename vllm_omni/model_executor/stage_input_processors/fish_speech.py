"""Stage input processor for Fish Speech S2 Pro: Slow AR → DAC Decoder."""

import os
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    encode_cuda_ipc_pool_tensor,
)

logger = init_logger(__name__)


def _fish_cuda_ipc_relay_enabled() -> bool:
    return os.environ.get("VLLM_FISH_CUDA_IPC_RELAY", "0") == "1"


def _fish_cuda_ipc_stage0_enabled() -> bool:
    return (
        _fish_cuda_ipc_relay_enabled()
        and os.environ.get("VLLM_FISH_CUDA_IPC_STAGE0", "0") == "1"
    )


def _fish_pooled_gpu_relay_enabled() -> bool:
    return os.environ.get("VLLM_FISH_POOLED_GPU_RELAY", "0") == "1"


def _compact_fish_codes_if_enabled(flat_codes: torch.Tensor) -> torch.Tensor:
    compact = os.environ.get("VLLM_FISH_COMPACT_CODES", "0") == "1"
    dtype_name = os.environ.get("VLLM_FISH_COMPACT_CODE_DTYPE", "int16").strip().lower()
    if not compact:
        return flat_codes
    if dtype_name in {"int32", "torch.int32"}:
        return flat_codes.to(dtype=torch.int32)
    if dtype_name not in {"int16", "torch.int16"}:
        logger.warning_once(
            "Ignoring unsupported VLLM_FISH_COMPACT_CODE_DTYPE=%r; using int16.",
            dtype_name,
        )
    return flat_codes.to(dtype=torch.int16)


def _make_pooled_gpu_relay_codes(
    *,
    request_payload: dict[str, Any],
    request_id: str,
    flat_codes: torch.Tensor,
    num_codebooks: int,
) -> dict[str, Any]:
    if not flat_codes.is_cuda:
        relay_device = os.environ.get("VLLM_FISH_GPU_RELAY_DEVICE", "cuda")
        flat_codes = flat_codes.to(device=relay_device, non_blocking=True)

    frame_count = int(flat_codes.numel() // num_codebooks)
    bucket_spec = os.environ.get("VLLM_FISH_GPU_RELAY_BUCKET_FRAMES", "4,50")
    buckets = sorted(int(part.strip()) for part in bucket_spec.split(",") if part.strip())
    frame_capacity = frame_count
    for bucket in buckets:
        if frame_count <= bucket:
            frame_capacity = bucket
            break
    frame_capacity = max(frame_capacity, int(os.environ.get("VLLM_FISH_GPU_RELAY_POOL_FRAMES", "0") or 0))
    frame_capacity = max(frame_capacity, frame_count)
    num_slots = int(os.environ.get("VLLM_FISH_GPU_RELAY_POOL_SLOTS", "8"))
    num_slots = max(1, num_slots)

    pool = request_payload.get("_fish_gpu_relay_pool")
    if (
        not isinstance(pool, torch.Tensor)
        or pool.shape[0] < num_slots
        or pool.shape[1] != num_codebooks
        or pool.shape[2] < frame_capacity
        or pool.device != flat_codes.device
        or pool.dtype != torch.long
    ):
        generation = int(request_payload.get("_fish_gpu_relay_pool_generation", 0)) + 1
        pool = torch.empty(
            (num_slots, num_codebooks, frame_capacity),
            device=flat_codes.device,
            dtype=torch.long,
        )
        request_payload["_fish_gpu_relay_pool"] = pool
        request_payload["_fish_gpu_relay_pool_generation"] = generation
        request_payload["_fish_gpu_relay_sent_handles"] = set()
        request_payload["_fish_gpu_relay_events"] = None

    use_events = os.environ.get("VLLM_FISH_GPU_RELAY_EVENT", "0") == "1"
    if use_events and request_payload.get("_fish_gpu_relay_events") is None:
        try:
            request_payload["_fish_gpu_relay_events"] = [
                torch.cuda.Event(enable_timing=False, interprocess=True)
                for _ in range(num_slots)
            ]
        except Exception as exc:
            logger.warning_once(
                "Fish pooled GPU relay disabled CUDA event handoff and will use sync: %s",
                exc,
            )
            use_events = False

    chunk_idx = int(request_payload.get("_fish_gpu_relay_chunk_idx", 0))
    request_payload["_fish_gpu_relay_chunk_idx"] = chunk_idx + 1
    slot = chunk_idx % num_slots
    slot_view = pool[slot]
    slot_view[:, :frame_count].copy_(flat_codes.view(num_codebooks, frame_count), non_blocking=True)

    event_handle = None
    if use_events:
        events = request_payload.get("_fish_gpu_relay_events")
        if isinstance(events, list):
            event = events[slot]
            event.record(torch.cuda.current_stream(slot_view.device))
    elif os.environ.get("VLLM_FISH_GPU_RELAY_SYNC", "1") == "1":
        torch.cuda.current_stream(slot_view.device).synchronize()

    generation = int(request_payload.get("_fish_gpu_relay_pool_generation", 0))
    relay_id = f"fish:{request_id}:{generation}:{slot}"
    sent_handles = request_payload.setdefault("_fish_gpu_relay_sent_handles", set())
    include_handle = relay_id not in sent_handles
    sent_handles.add(relay_id)
    if include_handle and use_events:
        events = request_payload.get("_fish_gpu_relay_events")
        if isinstance(events, list):
            event_handle = events[slot].ipc_handle()
    return encode_cuda_ipc_pool_tensor(
        slot_view,
        relay_id,
        include_handle=include_handle,
        event_handle=event_handle,
    )


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract the last frame of audio codes from the pooling output."""
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if (
            _fish_cuda_ipc_stage0_enabled()
            and os.environ.get("VLLM_FISH_CUDA_IPC_SKIP_ZERO_FILTER", "0") == "1"
            and frame.is_cuda
        ):
            # The post-sample Fish path only emits real codebook frames. Avoid
            # a per-frame GPU->CPU sync just to filter zero padding.
            return frame.to(torch.long).reshape(-1)
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Fish Speech async_chunk: {tuple(audio_codes.shape)}")


def slow_ar_to_dac_decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async processor: wait for Slow AR to finish, then pass all codes to DAC decoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    slow_ar_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    dac_inputs: list[OmniTokensPrompt] = []

    for output in slow_ar_outputs:
        out = output.outputs[0]
        # audio_codes shape: [num_frames, num_codebooks]
        audio_codes = out.multimodal_output["audio_codes"].to(torch.long)
        # Filter zero-padded frames.
        valid_mask = audio_codes.any(dim=1)
        audio_codes = audio_codes[valid_mask]
        # Codebook-major flat: [num_codebooks * num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        dac_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return dac_inputs


def slow_ar_to_dac_decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async streaming processor: emit code chunks as they are produced.

    Accumulates per-step codes and emits fixed-size chunks with left context
    overlap for smooth audio transitions, analogous to
    ``talker2code2wav_async_chunk`` in Qwen3 TTS.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            if _fish_cuda_ipc_stage0_enabled():
                transfer_manager.code_prompt_token_ids[request_id].append(frame.detach().to(dtype=torch.long))
            else:
                transfer_manager.code_prompt_token_ids[request_id].append(
                    frame.detach().to(device="cpu", dtype=torch.long)
                )
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames", 0))

    # Per-request override.
    additional_information = getattr(request, "additional_information", None)
    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )

    request_payload = transfer_manager.request_payload.setdefault(request_id, {})
    if initial_chunk_size > chunk_size:
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": True,
            }
        return None

    sent_frames = int(request_payload.get("_fish_speech_sent_frames", 0))
    pending = length - sent_frames

    if pending <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": True,
            }
        return None

    if sent_frames == 0 and initial_chunk_size > 0:
        if pending < initial_chunk_size and not finished:
            return None
        context_length = min(pending, initial_chunk_size)
        chunk_end = context_length
        left_context_size = 0
        window_frames = transfer_manager.code_prompt_token_ids[request_id][:chunk_end]
    else:
        if pending < chunk_size and not finished:
            return None
        context_length = pending if finished else chunk_size
        chunk_end = sent_frames + context_length
        left_start = max(0, sent_frames - left_context_size_config)
        left_context_size = sent_frames - left_start
        window_frames = transfer_manager.code_prompt_token_ids[request_id][left_start:chunk_end]

    # Pack into codebook-major flat codes.
    stacked_frames = torch.stack(window_frames, dim=0)
    flat_codes = stacked_frames.transpose(0, 1).reshape(-1).contiguous()
    if _fish_cuda_ipc_relay_enabled() and not flat_codes.is_cuda:
        try:
            relay_device = os.environ.get("VLLM_FISH_CUDA_IPC_RELAY_DEVICE", "cuda")
            flat_codes = flat_codes.to(device=relay_device, non_blocking=True)
        except Exception as exc:
            logger.warning_once("Fish CUDA IPC chunk relay fell back to CPU: %s", exc)
    request_payload["_fish_speech_sent_frames"] = chunk_end
    if _fish_pooled_gpu_relay_enabled():
        code_predictor_codes = _make_pooled_gpu_relay_codes(
            request_payload=request_payload,
            request_id=request_id,
            flat_codes=flat_codes,
            num_codebooks=stacked_frames.shape[1],
        )
    elif (
        os.environ.get("VLLM_FISH_TENSOR_RELAY", "0") == "1"
        or os.environ.get("VLLM_FISH_COMPACT_CODES", "0") == "1"
        or _fish_cuda_ipc_relay_enabled()
    ):
        code_predictor_codes: torch.Tensor | list[int] = _compact_fish_codes_if_enabled(flat_codes)
    else:
        code_predictor_codes = flat_codes.tolist()

    return {
        "code_predictor_codes": code_predictor_codes,
        "next_stage_prompt_len": int(flat_codes.numel()),
        "left_context_size": left_context_size,
        "finished": finished,
    }
