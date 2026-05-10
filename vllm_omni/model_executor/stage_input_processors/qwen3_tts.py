"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = init_logger(__name__)

_CODEBOOK_SIZE = 2048
_QWEN3_TTS_CODE_BUFFER_STATE_KEY = "qwen3_tts_code_buffer"


def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []
    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            # Non-async decode should only run once, after talker has
            # accumulated the final code sequence.
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})

        # audio_codes shape: [num_frames, Q] where Q=num_quantizers (16)
        audio_codes = mm_codes["audio"].to(torch.long)
        token_ids = output.cumulative_token_ids

        # token_ids provides an upper bound on the newly generated codec span.
        # audio_codes may still contain zero-padded / invalid rows, so trim only
        # after filtering valid frames instead of trying to align EOS indices.
        seq_len = max(len(token_ids) - 1, 0)
        # Filter invalid frames: zero-padded (EOS) and frames containing
        # out-of-range values (e.g. stop_token_id=2150 exceeds
        # codebook_size=2048). Keep this aligned with the async frame filter.
        valid_mask = (
            audio_codes.any(dim=1)
            & (audio_codes.min(dim=1).values >= 0)
            & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        )
        audio_codes = audio_codes[valid_mask]
        if seq_len > 0 and audio_codes.ndim == 2 and int(audio_codes.shape[0]) > seq_len:
            audio_codes = audio_codes[-seq_len:]
        ref_code = mm_codes.get("ref")
        ref_code_len = mm.get("meta", {}).get("ref_code_len")
        if isinstance(ref_code_len, torch.Tensor):
            ref_code_len = int(ref_code_len.reshape(-1)[-1].item()) if ref_code_len.numel() > 0 else 0
        elif ref_code_len is None:
            ref_code_len = 0
        else:
            ref_code_len = int(ref_code_len)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code = ref_code.to(torch.long).cpu().contiguous()
            if ref_code.ndim == 1:
                num_quantizers = int(audio_codes.shape[1]) if audio_codes.ndim == 2 and audio_codes.shape[1] > 0 else 16
                if ref_code.numel() % num_quantizers != 0:
                    logger.warning(
                        "Ignoring malformed ref_code with %d elements not divisible by num_quantizers=%d",
                        ref_code.numel(),
                        num_quantizers,
                    )
                    ref_code = None
                else:
                    ref_code = ref_code.reshape(-1, num_quantizers)
            elif ref_code.ndim != 2:
                logger.warning("Ignoring malformed ref_code shape %s", tuple(ref_code.shape))
                ref_code = None
            if isinstance(ref_code, torch.Tensor) and ref_code_len > 0 and int(ref_code.shape[0]) > ref_code_len:
                logger.warning(
                    "Trimming ref_code from %d frames to ref_code_len=%d before Code2Wav.",
                    int(ref_code.shape[0]),
                    ref_code_len,
                )
                ref_code = ref_code[:ref_code_len]
            if not isinstance(ref_code, torch.Tensor):
                ref_code_len = 0
            else:
                ref_code_len = int(ref_code.shape[0])
                audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
        else:
            ref_code_len = 0
        # Code2Wav expects codebook-major flat: [Q*num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        additional_information = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
                speaker=extract_speaker_from_prompt(prompt, index=i),
                language=extract_language_from_prompt(prompt, index=i),
            )
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )
    return code2wav_inputs


class _CodecFrameBuffer:
    """Append-only CPU buffer for per-request codec frames."""

    __slots__ = ("_data", "_length")

    def __init__(self, first_frame: torch.Tensor | None = None, *, initial_capacity: int = 8):
        self._data: torch.Tensor | None = None
        self._length = 0
        if first_frame is not None:
            self.append(first_frame, initial_capacity=initial_capacity)

    @classmethod
    def from_tensor(cls, data: torch.Tensor) -> "_CodecFrameBuffer":
        buffer = cls()
        data = data.to(torch.long).detach().cpu().reshape(data.shape[0], -1).contiguous()
        capacity = max(8, int(data.shape[0]))
        buffer._data = torch.empty((capacity, int(data.shape[1])), dtype=torch.long)
        buffer._data[: data.shape[0]].copy_(data)
        buffer._length = int(data.shape[0])
        return buffer

    def append(self, frame: torch.Tensor, *, initial_capacity: int = 8) -> bool:
        frame_data = frame.to(torch.long).detach().cpu().reshape(-1).contiguous()
        if not _is_valid_codec_frame(frame_data):
            return False
        if self._data is None:
            capacity = max(initial_capacity, 1)
            self._data = torch.empty((capacity, int(frame_data.numel())), dtype=torch.long)
        else:
            if frame_data.numel() != self._data.shape[1]:
                raise ValueError(
                    f"Qwen3-TTS codec frame quantizer count changed from {self._data.shape[1]} to {frame_data.numel()}"
                )
            if self._length >= self._data.shape[0]:
                new_capacity = max(self._data.shape[0] * 2, self._length + 1)
                new_data = torch.empty((new_capacity, self._data.shape[1]), dtype=torch.long)
                new_data[: self._length].copy_(self._data[: self._length])
                self._data = new_data
        self._data[self._length].copy_(frame_data)
        self._length += 1
        return True

    def __len__(self) -> int:
        return self._length

    @property
    def shape(self) -> tuple[int, int]:
        if self._data is None:
            return (0, 0)
        return (self._length, int(self._data.shape[1]))

    def tail(self, count: int) -> torch.Tensor:
        if self._data is None or self._length <= 0:
            return torch.empty((0, 0), dtype=torch.long)
        count = min(max(int(count), 0), self._length)
        return self._data[self._length - count : self._length].contiguous()


def _is_valid_codec_frame(frame: torch.Tensor) -> bool:
    if frame.numel() == 0:
        return False
    # Qwen3-TTS prefill emits an all-zero placeholder frame; non-async stage
    # processing applies the same rule before handing codes to Code2Wav.
    if not bool(frame.any().item()):
        return False
    if bool((frame < 0).any().item()) or bool((frame >= _CODEBOOK_SIZE).any().item()):
        return False
    return True


def _to_cpu_code_frame(frame: Any) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        return frame.to(torch.long).detach().cpu().reshape(-1).contiguous()
    return torch.as_tensor(frame, dtype=torch.long).reshape(-1).contiguous()


def _extract_last_frame(pooling_output: OmniPayload) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0:
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")


def _request_processing_state(transfer_manager: Any) -> dict[str, dict[str, Any]]:
    state = getattr(transfer_manager, "request_processing_state", None)
    if state is None:
        state = {}
        transfer_manager.request_processing_state = state
    return state


def _request_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    return _request_processing_state(transfer_manager).setdefault(request_id, {})


def _get_code_buffer(transfer_manager: Any, request_id: str) -> Any:
    return _request_processing_state(transfer_manager).get(request_id, {}).get(_QWEN3_TTS_CODE_BUFFER_STATE_KEY)


def _set_code_buffer(transfer_manager: Any, request_id: str, buffer: Any) -> None:
    _request_state(transfer_manager, request_id)[_QWEN3_TTS_CODE_BUFFER_STATE_KEY] = buffer


def _iter_code_buffers(transfer_manager: Any):
    for req_id, state in _request_processing_state(transfer_manager).items():
        if isinstance(state, dict) and _QWEN3_TTS_CODE_BUFFER_STATE_KEY in state:
            yield req_id, state[_QWEN3_TTS_CODE_BUFFER_STATE_KEY]


def _append_code_frame(transfer_manager: Any, request_id: str, frame: torch.Tensor) -> None:
    buffer = _get_code_buffer(transfer_manager, request_id)
    if isinstance(buffer, torch.Tensor):
        buffer = _CodecFrameBuffer.from_tensor(buffer)
        _set_code_buffer(transfer_manager, request_id, buffer)
    elif buffer is None:
        buffer = _CodecFrameBuffer()
    if isinstance(buffer, _CodecFrameBuffer):
        if buffer.append(frame):
            _set_code_buffer(transfer_manager, request_id, buffer)
            frames_by_request = getattr(transfer_manager, "code_prompt_token_ids", None)
            if hasattr(frames_by_request, "pop"):
                frames_by_request.pop(request_id, None)
        return
    raise TypeError(f"Invalid Qwen3-TTS codec buffer type: {type(buffer)!r}")


def _buffer_frame_count(buf: Any) -> int:
    if buf is None:
        return 0
    if isinstance(buf, _CodecFrameBuffer):
        return len(buf)
    if isinstance(buf, torch.Tensor):
        return int(buf.shape[0])
    raise TypeError(f"Invalid Qwen3-TTS codec buffer type: {type(buf)!r}")


def _buffer_tail(buf: Any, count: int) -> torch.Tensor:
    if buf is None:
        return torch.empty((0, 0), dtype=torch.long)
    if isinstance(buf, _CodecFrameBuffer):
        return buf.tail(count)
    if isinstance(buf, torch.Tensor):
        count = min(max(int(count), 0), int(buf.shape[0]))
        if count == 0:
            num_quantizers = int(buf.shape[1]) if buf.ndim >= 2 else 0
            return torch.empty((0, num_quantizers), dtype=torch.long)
        return buf[-count:].to(torch.long).cpu().contiguous()
    raise TypeError(f"Invalid Qwen3-TTS codec buffer type: {type(buf)!r}")


def _legacy_code_prompt_tensor(transfer_manager: Any, request_id: str) -> torch.Tensor:
    frames_by_request = getattr(transfer_manager, "code_prompt_token_ids", None)
    if not hasattr(frames_by_request, "get"):
        return torch.empty((0, 0), dtype=torch.long)
    frames = frames_by_request.get(request_id, [])
    if not frames:
        return torch.empty((0, 0), dtype=torch.long)

    rows: list[torch.Tensor] = []
    num_quantizers = None
    for frame in frames:
        row = _to_cpu_code_frame(frame)
        if not _is_valid_codec_frame(row):
            continue
        if num_quantizers is None:
            num_quantizers = int(row.numel())
        elif row.numel() != num_quantizers:
            raise ValueError(
                f"Qwen3-TTS legacy codec frame quantizer count changed from {num_quantizers} to {row.numel()}"
            )
        rows.append(row)
    if not rows:
        return torch.empty((0, 0), dtype=torch.long)
    return torch.stack(rows, dim=0).contiguous()


def _cached_legacy_code_prompt_tensor(
    transfer_manager: Any,
    request_id: str,
    legacy_cache: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if legacy_cache is not None:
        cached = legacy_cache.get(request_id)
        if cached is not None:
            return cached
    legacy_frames = _legacy_code_prompt_tensor(transfer_manager, request_id)
    if legacy_cache is not None:
        legacy_cache[request_id] = legacy_frames
    return legacy_frames


def _frame_count(
    transfer_manager: Any,
    request_id: str,
    legacy_cache: dict[str, torch.Tensor] | None = None,
) -> int:
    buffer = _get_code_buffer(transfer_manager, request_id)
    if buffer is not None:
        return _buffer_frame_count(buffer)
    return int(_cached_legacy_code_prompt_tensor(transfer_manager, request_id, legacy_cache).shape[0])


def _active_request_count(transfer_manager: Any) -> int:
    active = 0
    seen: set[str] = set()
    for req_id, buf in _iter_code_buffers(transfer_manager):
        if isinstance(buf, _CodecFrameBuffer) and len(buf) > 0:
            active += 1
            seen.add(req_id)
            continue
        if isinstance(buf, torch.Tensor) and buf.shape[0] > 0:
            active += 1
            seen.add(req_id)
    for req_id, frames in transfer_manager.code_prompt_token_ids.items():
        # This is a legacy load heuristic, not the codec-frame source of truth.
        # Older tests/paths may use zero placeholders here to represent active
        # requests before a valid Qwen3-TTS codec frame exists.
        if req_id not in seen and len(frames) > 0:
            active += 1
    return active


def _window_tensor(
    transfer_manager: Any,
    request_id: str,
    end_index: int,
    legacy_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    buffer = _get_code_buffer(transfer_manager, request_id)
    if buffer is not None:
        return _buffer_tail(buffer, end_index)
    legacy_frames = _cached_legacy_code_prompt_tensor(transfer_manager, request_id, legacy_cache)
    if legacy_frames.shape[0] == 0:
        return legacy_frames
    return legacy_frames[-end_index:].contiguous()


def _normalize_ref_code(ref_code: torch.Tensor, num_quantizers: int) -> torch.Tensor | None:
    ref_code = ref_code.to(torch.long).cpu().contiguous()
    if ref_code.ndim == 1:
        if ref_code.numel() % num_quantizers != 0:
            logger.warning(
                "Ignoring malformed ref_code with %d elements not divisible by num_quantizers=%d",
                ref_code.numel(),
                num_quantizers,
            )
            return None
        ref_code = ref_code.reshape(-1, num_quantizers)
    elif ref_code.ndim != 2:
        logger.warning("Ignoring malformed ref_code shape %s", tuple(ref_code.shape))
        return None
    return ref_code


def _ref_code_context_policy(cfg: dict[str, Any]) -> tuple[str, int | None]:
    policy = str(cfg.get("ref_code_context_policy", "full")).lower()
    if policy not in {"full", "tail", "first_only"}:
        raise ValueError(
            f"Invalid ref_code_context_policy for Qwen3-TTS: {policy!r}. Expected one of: full, tail, first_only."
        )

    tail_frames_value = cfg.get("ref_code_context_tail_frames")
    tail_frames = None if tail_frames_value is None else int(tail_frames_value)
    if policy == "tail" and (tail_frames is None or tail_frames <= 0):
        raise ValueError("ref_code_context_tail_frames must be > 0 when ref_code_context_policy='tail'.")
    if tail_frames is not None and tail_frames < 0:
        raise ValueError("ref_code_context_tail_frames must be >= 0.")
    return policy, tail_frames


def _select_ref_code_context(
    ref_frames: torch.Tensor,
    *,
    policy: str,
    tail_frames: int | None,
    is_first_chunk: bool,
) -> torch.Tensor | None:
    if policy == "first_only" and not is_first_chunk:
        return None
    if policy == "tail":
        assert tail_frames is not None
        return ref_frames[-tail_frames:].contiguous()
    return ref_frames


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            _append_code_frame(transfer_manager, request_id, frame)
        ref_code = pooling_output.get("codes", {}).get("ref")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0 and request_payload.get(request_id) is None:
            request_payload[request_id] = ref_code.to(torch.long).cpu().contiguous()
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)
    ref_context_policy, ref_context_tail_frames = _ref_code_context_policy(cfg)

    # Per-request override takes priority over dynamic IC.
    fixed_initial_chunk_size = configured_initial_chunk_size > 0
    initial_chunk_size = configured_initial_chunk_size
    additional_information = getattr(request, "additional_information", None)

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            fixed_initial_chunk_size = True

    # Dynamic IC: cache per request so boundaries stay stable for its lifetime.
    if not fixed_initial_chunk_size:
        _ic_cache = getattr(transfer_manager, "_cached_ic", None)
        if _ic_cache is None:
            _ic_cache = {}
            transfer_manager._cached_ic = _ic_cache
        if request_id not in _ic_cache:
            max_ic = max_ic_for_chunk_size(chunk_size)
            active = _active_request_count(transfer_manager)
            capacity = getattr(transfer_manager, "scheduler_max_num_seqs", 1)
            _ic_cache[request_id] = compute_dynamic_initial_chunk_size(active, capacity, max_ic)
        initial_chunk_size = _ic_cache[request_id]

    if chunk_size <= 0 or left_context_size_config < 0 or configured_initial_chunk_size < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )

    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    legacy_frame_cache: dict[str, torch.Tensor] = {}
    length = _frame_count(transfer_manager, request_id, legacy_frame_cache)

    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    use_first_chunk = initial_chunk_size > 0 and initial_chunk_size < chunk_size

    if use_first_chunk and length <= initial_chunk_size:
        if not finished and length < initial_chunk_size:
            return None
        context_length = length if finished and length < initial_chunk_size else initial_chunk_size
    else:
        # The initial chunk is only for TTFA. After that, return to the normal
        # codec chunk size so Code2Wav is not flooded by repeated tiny windows.
        initial_coverage = initial_chunk_size if use_first_chunk else 0
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = _window_tensor(transfer_manager, request_id, end_index, legacy_frame_cache)
    if window_frames.shape[0] == 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None
    left_context_size = min(left_context_size, int(window_frames.shape[0]))
    ref_context_frames = 0

    # Prepend ref_code as decoder context. The default "full" policy keeps the
    # previous behavior for quality; opt-in policies can trade speaker context
    # for lower Stage1 work in high-throughput voice-clone runs. Use `.get()`
    # (not `.pop()`) because the selected context may still be needed later.
    ref_code = request_payload.get(request_id)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_frames = _normalize_ref_code(ref_code, int(window_frames.shape[1]))
        if ref_frames is not None and ref_frames.numel() > 0:
            put_req_chunk = getattr(transfer_manager, "put_req_chunk", {})
            emitted_chunks = put_req_chunk.get(request_id, 0) if hasattr(put_req_chunk, "get") else 0
            ref_frames = _select_ref_code_context(
                ref_frames,
                policy=ref_context_policy,
                tail_frames=ref_context_tail_frames,
                is_first_chunk=int(emitted_chunks) <= 0,
            )
        if ref_frames is not None and ref_frames.numel() > 0:
            window_frames = torch.cat((ref_frames, window_frames), dim=0)
            ref_context_frames = int(ref_frames.shape[0])
            left_context_size += ref_context_frames

    code_predictor_codes = window_frames.transpose(0, 1).contiguous().reshape(-1)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
            codec_window_frames=int(window_frames.shape[0]),
            codec_ref_context_frames=ref_context_frames,
        ),
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
    )
