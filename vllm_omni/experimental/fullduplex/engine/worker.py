from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from typing import Any

NativeDuplexProvider = Callable[[Any, dict[str, Any] | None], Any | None]

NATIVE_DUPLEX_REQUIRED_METHODS = (
    "open_duplex_session",
    "append_duplex_input",
    "signal_duplex_turn",
    "close_duplex_session",
)

_NATIVE_DUPLEX_PROVIDERS: list[NativeDuplexProvider] = []
_DEFAULT_PROVIDERS_BOOTSTRAPPED = False


def register_native_duplex_provider(provider: NativeDuplexProvider) -> None:
    if provider not in _NATIVE_DUPLEX_PROVIDERS:
        _NATIVE_DUPLEX_PROVIDERS.append(provider)


def validate_native_duplex_target(target: Any) -> None:
    missing = [name for name in NATIVE_DUPLEX_REQUIRED_METHODS if not callable(getattr(target, name, None))]
    if missing:
        raise RuntimeError(f"native duplex target missing required methods: {', '.join(missing)}")


def get_native_duplex_target(worker: Any, capabilities: dict[str, Any] | None = None) -> Any | None:
    if capabilities is not None and capabilities.get("implementation_level") != "model_native_duplex":
        return None
    _bootstrap_default_providers()
    for provider in tuple(_NATIVE_DUPLEX_PROVIDERS):
        target = provider(worker, capabilities)
        if target is not None:
            return target
    return None


def is_passive_native_duplex_stage(target: Any) -> bool:
    return bool(getattr(target, "is_passive_native_duplex_stage", False))


def open_native_duplex_session(
    target: Any,
    *,
    session_id: str,
    epoch: int,
    capabilities: dict[str, Any],
    session_config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    validate_native_ref_audio_config(session_config)
    validate_native_duplex_target(target)
    native = target.open_duplex_session(
        session_id=session_id,
        epoch=epoch,
        capabilities=capabilities,
        session_config=session_config,
    )
    return (
        {
            "supported": True,
            "session_id": session_id,
            "epoch": epoch,
            "implementation_level": "model_native_duplex",
            "native_result": as_native_result_dict(native, target=target),
        },
        target,
    )


def append_native_duplex_input(
    target: Any,
    *,
    session_id: str,
    epoch: int,
    seq: int,
    mode: str,
    payload: Any,
    final: bool,
) -> dict[str, Any]:
    native = target.append_duplex_input(
        session_id=session_id,
        epoch=epoch,
        seq=seq,
        mode=mode,
        payload=payload,
        final=final,
    )
    return {
        "supported": True,
        "session_id": session_id,
        "epoch": epoch,
        "seq": seq,
        "mode": mode,
        "final": final,
        "implementation_level": "model_native_duplex",
        "native_result": as_native_result_dict(native, target=target),
    }


def decode_native_ref_audio_from_config(session_config: dict[str, Any]) -> Any | None:
    validate_native_ref_audio_config(session_config)
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    audio_data = extra_body.get("ref_audio_data")
    if audio_data is None:
        return None
    if not isinstance(audio_data, str):
        raise TypeError("native duplex ref_audio_data must be base64 pcm_f32le")
    fmt = extra_body.get("ref_audio_format") or "pcm_f32le"
    if fmt != "pcm_f32le":
        raise ValueError(f"unsupported native duplex ref_audio_format: {fmt!r}")
    import numpy as np

    try:
        raw = base64.b64decode(audio_data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid native duplex ref_audio_data") from exc
    if len(raw) % 4:
        raise ValueError("invalid native duplex ref_audio_data length")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def validate_native_ref_audio_config(session_config: dict[str, Any]) -> None:
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    if any(key in session_config for key in ("ref_audio_path", "tts_ref_audio_path")) or any(
        key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")
    ):
        raise ValueError("native duplex ref_audio_path is not accepted; resolve ref_audio in serving first")


def as_native_result_dict(native: Any, *, target: Any | None = None) -> dict[str, Any]:
    if native is None:
        result: dict[str, Any] = {}
    elif isinstance(native, dict):
        result = dict(native)
    else:
        model_dump = getattr(native, "model_dump", None)
        if callable(model_dump):
            result = dict(model_dump())
        elif hasattr(native, "__dict__"):
            result = dict(native.__dict__)
        else:
            result = {"value": native}

    normalize_native_audio_result(result)
    normalize_native_costs(result)
    if target is not None:
        runtime_impl = getattr(target, "runtime_impl", None)
        if isinstance(runtime_impl, str) and runtime_impl:
            result.setdefault("runtime_impl", runtime_impl)
        owned_runtime = getattr(target, "owned_runtime", None)
        if isinstance(owned_runtime, bool):
            result.setdefault("owned_runtime", owned_runtime)
    return result


def normalize_native_audio_result(result: dict[str, Any]) -> None:
    if result.get("audio_data") is not None:
        result.pop("audio_waveform", None)
        return
    waveform = result.pop("audio_waveform", None)
    if waveform is None:
        return
    try:
        import numpy as np

        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        audio_bytes = np.asarray(waveform, dtype=np.float32).reshape(-1).tobytes()
        result["audio_data"] = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as exc:
        result["audio_data_error"] = str(exc)


def normalize_native_costs(result: dict[str, Any]) -> None:
    cost_fields = {
        "cost_llm": "cost_llm_ms",
        "cost_tts_prep": "cost_tts_prep_ms",
        "cost_tts": "cost_tts_ms",
        "cost_token2wav": "cost_token2wav_ms",
        "cost_all": "cost_all_ms",
    }
    for src, dst in cost_fields.items():
        if dst in result or src not in result:
            continue
        value = result.get(src)
        if isinstance(value, int | float):
            result[dst] = float(value) * 1000.0


def _bootstrap_default_providers() -> None:
    global _DEFAULT_PROVIDERS_BOOTSTRAPPED
    if _DEFAULT_PROVIDERS_BOOTSTRAPPED:
        return
    _DEFAULT_PROVIDERS_BOOTSTRAPPED = True
    from vllm_omni.experimental.fullduplex.minicpmo45.worker import (
        maybe_load_minicpmo_native_duplex_target,
    )

    register_native_duplex_provider(maybe_load_minicpmo_native_duplex_target)
