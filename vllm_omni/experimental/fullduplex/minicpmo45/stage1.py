from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.engine.intermediate import populate_tts_handoff_from_omni_payload
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy


class MiniCPMO45StageFenceMismatchError(ValueError):
    pass


@dataclass(slots=True)
class MiniCPMO45Stage1State:
    """Transient Stage1 data that must never cross a response fence."""

    fence: DuplexFence
    consumed_tokens: int = 0
    token2wav_buffer: list[int] = field(default_factory=list)
    audio_offset: int = 0

    def require(self, fence: DuplexFence) -> None:
        if fence != self.fence:
            raise MiniCPMO45StageFenceMismatchError(
                f"MiniCPM-o Stage1 fence mismatch: expected {self.fence!r}, got {fence!r}"
            )

    def reset(self, fence: DuplexFence) -> MiniCPMO45Stage1State:
        if fence.session_id != self.fence.session_id:
            raise MiniCPMO45StageFenceMismatchError(
                f"MiniCPM-o Stage1 session changed: expected {self.fence.session_id!r}, got {fence.session_id!r}"
            )
        return MiniCPMO45Stage1State(fence=fence)


class MiniCPMO45Stage1DuplexRuntime:
    """MiniCPM-o 4.5 native duplex runtime for the loaded TTS/token2wav stage."""

    runtime_impl = "vllm_omni_minicpmo45_stage1_runtime"
    owned_runtime = False
    stage_role = "tts"
    supports_multiple_native_duplex_sessions = True

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str = "cuda") -> None:
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.session_config: dict[str, Any] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self._duplex_stage_payload_store: dict[str, dict[str, Any]] = {}
        self._duplex_stage_payload_cache: Any | None = None

    @classmethod
    def can_wrap(cls, stage_model: Any) -> bool:
        stage = getattr(stage_model, "model_stage", None)
        return stage in {"tts", "talker"} or hasattr(stage_model, "talker")

    def open_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            raise ValueError("MiniCPM-o stage1 duplex session_id is required")
        session_config = kwargs.get("session_config")
        self.session_config = dict(session_config) if isinstance(session_config, dict) else {}
        self.sessions[session_id] = {"session_config": dict(self.session_config)}
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
            "stage_runtime_ready": True,
        }

    def append_duplex_input(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        mode = kwargs.get("mode")
        payload = kwargs.get("payload")
        if mode not in {MiniCPMO45DuplexPolicy.STAGE_HANDOFF_MODE, MiniCPMO45DuplexPolicy.TTS_HANDOFF_MODE}:
            raise ValueError(f"MiniCPM-o stage1 duplex expects append_stage_handoff, got {mode!r}")
        if not isinstance(payload, dict):
            raise TypeError("append_stage_handoff payload must be a dict")
        if session_id not in self.sessions:
            raise KeyError(f"unknown MiniCPM-o stage1 duplex session: {session_id}")
        native = self._run_tts_handoff(payload, session_id=session_id)
        native.setdefault("stage_role", self.stage_role)
        native.setdefault("runtime_impl", self.runtime_impl)
        native.setdefault("owned_runtime", self.owned_runtime)
        native.setdefault("stage_runtime_ready", True)
        native.setdefault("uses_model_runner_scheduler", False)
        native.setdefault("runner_kv_backed", False)
        native.setdefault("experimental_worker_control_rpc", True)
        native.setdefault("per_step_tensor_handoff", False)
        native.setdefault("runner_local_payload_ref", True)
        return native

    def signal_duplex_turn(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        event = kwargs.get("event")
        if event in {"barge_in", "input.cancel", "response.cancel"}:
            self._close_streams(session_id=session_id)
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "event": event,
        }

    def close_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        self._close_streams(session_id=session_id)
        if session_id:
            self.sessions.pop(session_id, None)
        else:
            self.sessions.clear()
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "reason": kwargs.get("reason"),
        }

    def stop(self) -> None:
        for session_id in list(self.sessions):
            self._close_streams(session_id=session_id)
        self.sessions.clear()

    def cleanup(self) -> None:
        self.stop()

    def _run_tts_handoff(self, payload: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        call = getattr(self.stage_model, "forward", None)
        if not callable(call):
            talker = (
                getattr(self.stage_model, "talker", None)
                or getattr(self.stage_model, "model", None)
                or self.stage_model
            )
            if callable(talker):
                call = talker
            else:
                call = getattr(talker, "forward", None)
            if not callable(call):
                return {
                    "is_listen": False,
                    "text": self._handoff_text(payload),
                    "audio_waveform": None,
                    "end_of_turn": bool(payload.get("end_of_turn", False)),
                    "stage_runtime_ready": False,
                    "reason": "stage1_loaded_model_does_not_expose_tts_forward",
                }

        info = self._normalize_handoff_payload(payload, session_id=session_id)
        result = call(additional_information=info)
        waveform = None
        if isinstance(result, tuple) and len(result) == 2:
            mel_or_waveform, waveform_chunk = result
            waveform = waveform_chunk if waveform_chunk is not None else mel_or_waveform
        else:
            waveform = result
        waveform_numel = None
        if hasattr(waveform, "numel"):
            waveform_numel = int(waveform.numel())
        elif waveform is not None:
            with suppress(Exception):
                waveform_numel = int(np.asarray(waveform).size)
        sample_rate_hz = int(self._stage_param("sample_rate", 24000))
        audio_duration_ms = None
        if waveform_numel is not None and sample_rate_hz > 0:
            audio_duration_ms = int(round(waveform_numel * 1000.0 / sample_rate_hz))
        text = self._handoff_text(payload)
        audio_text_marks = []
        if audio_duration_ms is not None and text:
            audio_text_marks.append(
                {
                    "text_chars": len(text),
                    "audio_end_ms": audio_duration_ms,
                }
            )
        return {
            "is_listen": False,
            "text": text,
            "audio_waveform": waveform,
            "end_of_turn": bool(payload.get("end_of_turn", False)),
            "audio_duration_ms": audio_duration_ms,
            "audio_duration_is_cumulative": False,
            "audio_text_marks": audio_text_marks,
            "sample_rate_hz": sample_rate_hz,
            "tts_token_shape": list(info["tts_token_ids"].shape)
            if hasattr(info.get("tts_token_ids"), "shape")
            else None,
            "tts_hidden_shape": list(info["tts_hidden_states"].shape)
            if hasattr(info.get("tts_hidden_states"), "shape")
            else None,
            "waveform_numel": waveform_numel,
        }

    def _normalize_handoff_payload(self, payload: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        info = self._resolve_handoff_payload_ref(payload)
        info["session_id"] = session_id
        info.setdefault("global_request_id", session_id)
        info.setdefault("request_id", session_id)
        info.setdefault("_omni_req_id", session_id)
        if info.get("omni_payload") is None:
            raise ValueError(
                "MiniCPM-o stage1 duplex handoff requires omni_payload; "
                "direct tts_token_ids/tts_hidden_states payloads are not accepted"
            )
        self._populate_tts_fields_from_omni_payload(info)
        if info.get("tts_token_ids") is None or info.get("tts_hidden_states") is None:
            raise ValueError("MiniCPM-o stage1 duplex omni_payload must contain ids.output and hidden_states.output")
        tts_token_ids = self._to_tensor(info.get("tts_token_ids"), dtype_name="long")
        tts_hidden_states = self._to_tensor(info.get("tts_hidden_states"), dtype_name="float")
        info["tts_token_ids"], info["tts_hidden_states"] = self._normalize_tts_condition_tensors(
            tts_token_ids,
            tts_hidden_states,
        )
        return info

    def put_duplex_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        payload_ref = str(kwargs.get("payload_ref") or "")
        payload = kwargs.get("payload")
        if not payload_ref:
            raise ValueError("MiniCPM-o stage1 duplex payload_ref is required")
        if not isinstance(payload, dict):
            raise TypeError("MiniCPM-o stage1 duplex stage payload must be a dict")
        payload_cache = getattr(self, "_duplex_stage_payload_cache", None)
        put_local_stage_payload = getattr(payload_cache, "put_local_stage_payload", None)
        if callable(put_local_stage_payload):
            put_local_stage_payload(payload_ref, dict(payload))
            uses_runner_local_payload_cache = True
        else:
            self._duplex_stage_payload_store[payload_ref] = dict(payload)
            uses_runner_local_payload_cache = False
        return {
            "payload_ref": payload_ref,
            "payload_cached": True,
            "uses_runner_local_payload_cache": uses_runner_local_payload_cache,
            "stage_role": self.stage_role,
            "runtime_impl": self.runtime_impl,
        }

    def set_duplex_stage_payload_cache(self, cache: Any) -> None:
        self._duplex_stage_payload_cache = cache

    def _resolve_handoff_payload_ref(self, payload: dict[str, Any]) -> dict[str, Any]:
        info = dict(payload)
        payload_ref = info.get("omni_payload_ref")
        if payload_ref is None:
            return info
        if not isinstance(payload_ref, str) or not payload_ref:
            raise ValueError("MiniCPM-o stage1 duplex omni_payload_ref must be a non-empty string")
        stored = self._pop_runner_local_stage_payload(payload_ref)
        if stored is None:
            stored = self._duplex_stage_payload_store.pop(payload_ref, None)
        if not isinstance(stored, dict):
            raise KeyError(f"MiniCPM-o stage1 duplex payload ref not found: {payload_ref}")
        merged = dict(stored)
        merged.setdefault("omni_payload_ref", payload_ref)
        for key, value in info.items():
            if key not in {"type", "omni_payload_ref"}:
                merged[key] = value
        return merged

    def _pop_runner_local_stage_payload(self, payload_ref: str) -> Any | None:
        cache = getattr(self, "_duplex_stage_payload_cache", None)
        pop_local_stage_payload = getattr(cache, "pop_local_stage_payload", None)
        if not callable(pop_local_stage_payload):
            return None
        return pop_local_stage_payload(payload_ref)

    @staticmethod
    def _populate_tts_fields_from_omni_payload(info: dict[str, Any]) -> None:
        if info.get("tts_token_ids") is not None and info.get("tts_hidden_states") is not None:
            return
        omni_payload = info.get("omni_payload")
        if omni_payload is None:
            return
        if isinstance(omni_payload, dict):
            payload = omni_payload
        else:
            from vllm_omni.data_entry_keys import deserialize_payload

            payload = deserialize_payload(omni_payload)
        if isinstance(payload, dict):
            populate_tts_handoff_from_omni_payload(info, payload)

    def _stage_param(self, name: str, default: Any) -> Any:
        for target in (
            self.stage_model,
            getattr(self.stage_model, "talker", None),
            getattr(self.stage_model, "model", None),
        ):
            if target is None:
                continue
            value = getattr(target, name, None)
            if value is not None:
                return value
            value = getattr(target, name.upper(), None)
            if value is not None:
                return value
        return default

    @staticmethod
    def _to_tensor(value: Any, *, dtype_name: str) -> Any:
        if value is None or hasattr(value, "detach"):
            return value
        import torch

        dtype = torch.long if dtype_name == "long" else torch.float32
        if getattr(value, "tensor_data", None) is not None:
            return MiniCPMO45Stage1DuplexRuntime._decode_tensor_payload(value, dtype=dtype)
        if (
            isinstance(value, (list, tuple))
            and value
            and all(getattr(item, "tensor_data", None) is not None for item in value)
        ):
            tensors = [MiniCPMO45Stage1DuplexRuntime._decode_tensor_payload(item, dtype=dtype) for item in value]
            return torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)
        return torch.as_tensor(value, dtype=dtype)

    @staticmethod
    def _decode_tensor_payload(value: Any, *, dtype: Any) -> Any:
        from vllm_omni.data_entry_keys import deserialize_tensor_entry

        return deserialize_tensor_entry(value).to(dtype=dtype)

    @staticmethod
    def _normalize_tts_condition_tensors(tts_token_ids: Any, tts_hidden_states: Any) -> tuple[Any, Any]:
        if tts_token_ids is None or tts_hidden_states is None:
            return tts_token_ids, tts_hidden_states

        token_ids = tts_token_ids.reshape(-1)
        hidden = tts_hidden_states
        while getattr(hidden, "ndim", 0) > 2 and 1 in hidden.shape[:-1]:
            for dim, size in enumerate(hidden.shape[:-1]):
                if size == 1:
                    hidden = hidden.squeeze(dim)
                    break
        if getattr(hidden, "ndim", 0) == 1:
            hidden = hidden.unsqueeze(0)
        elif getattr(hidden, "ndim", 0) > 2:
            hidden = hidden.reshape(-1, hidden.shape[-1])

        token_len = int(token_ids.shape[0])
        hidden_len = int(hidden.shape[0]) if getattr(hidden, "ndim", 0) >= 1 else 0
        if hidden_len != token_len and hidden_len > 0:
            common = min(token_len, hidden_len)
            token_ids = token_ids[-common:]
            hidden = hidden[-common:, :]
        return token_ids, hidden

    @staticmethod
    def _handoff_text(payload: dict[str, Any]) -> str:
        text = payload.get("llm_output_text", "")
        if isinstance(text, list):
            return str(text[0]) if text else ""
        return str(text)

    def _close_streams(self, *, session_id: str | None = None) -> None:
        target = (
            getattr(self.stage_model, "talker", None) or getattr(self.stage_model, "model", None) or self.stage_model
        )
        if session_id:
            finished_fn = getattr(target, "on_requests_finished", None)
            if callable(finished_fn):
                with suppress(Exception):
                    finished_fn({session_id})
        close_fn = getattr(target, "close_streams", None) or getattr(target, "cleanup", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()
