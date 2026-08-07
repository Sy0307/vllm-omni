from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Protocol

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.engine.model_events import DuplexModelEvent


class ServingRuntimeConfigError(ValueError):
    """A model serving plugin rejected client-visible runtime configuration."""

    def __init__(self, message: str, *, code: str = "invalid_duplex_runtime_config") -> None:
        super().__init__(message)
        self.code = code


class DuplexInputCompletionMode(str, Enum):
    """Point at which one model-owned input append is acknowledged."""

    APPEND_ACCEPTED = "append_accepted"
    OUTPUT_PROJECTED = "output_projected"


@dataclass(frozen=True, slots=True)
class DuplexInputAppendCommand:
    payload: object
    operation_id: str
    chunk_period_ms: int
    allow_emit: bool
    fence: DuplexFence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a non-empty string")
        if type(self.chunk_period_ms) is not int or self.chunk_period_ms <= 0:
            raise ValueError("chunk_period_ms must be a positive integer")
        if not isinstance(self.allow_emit, bool):
            raise TypeError("allow_emit must be a boolean")
        if self.fence is not None and not isinstance(self.fence, DuplexFence):
            raise TypeError("fence must be a DuplexFence or null")


@dataclass(frozen=True, slots=True)
class DuplexInputCommitCommand:
    operation_id: str
    chunk_period_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a non-empty string")
        if type(self.chunk_period_ms) is not int or self.chunk_period_ms <= 0:
            raise ValueError("chunk_period_ms must be a positive integer")


@dataclass(frozen=True, slots=True)
class DuplexInputClearCommand:
    reason: str
    clear_force_listen: bool = True
    clear_buffer: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.clear_force_listen, bool):
            raise TypeError("clear_force_listen must be a boolean")
        if not isinstance(self.clear_buffer, bool):
            raise TypeError("clear_buffer must be a boolean")


@dataclass(frozen=True, slots=True)
class DuplexInputCloseCommand:
    abort: bool

    def __post_init__(self) -> None:
        if not isinstance(self.abort, bool):
            raise TypeError("abort must be a boolean")


@dataclass(frozen=True, slots=True)
class DuplexInputFlushCommand:
    chunk_period_ms: int

    def __post_init__(self) -> None:
        if type(self.chunk_period_ms) is not int or self.chunk_period_ms <= 0:
            raise ValueError("chunk_period_ms must be a positive integer")


@dataclass(frozen=True, slots=True)
class DuplexInputSnapshot:
    pending_byte_count: int = 0
    has_pending: bool = False
    has_reserved: bool = False
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_payload: object | None = None
    committed_operation_id: str | None = None
    committed_reserved_bytes: int = 0


@dataclass(frozen=True, slots=True)
class DuplexInputEffect:
    append_payloads: tuple[object, ...] = ()
    reservations: tuple[DuplexInputReservation, ...] = ()
    released_bytes: int = 0


def ordered_input_emissions(
    effect: DuplexInputEffect,
) -> tuple[tuple[object, DuplexInputReservation], ...]:
    """Pair model work with its rollback token without imposing packet size."""
    if len(effect.append_payloads) != len(effect.reservations):
        raise RuntimeError("Duplex input controller must return one reservation per ordered append payload")
    return tuple(zip(effect.append_payloads, effect.reservations, strict=True))


class DuplexInputController(Protocol):
    """Model-owned packetization and input-state lifecycle."""

    def create_state(self) -> object: ...

    def snapshot(self, state: object) -> DuplexInputSnapshot: ...

    def append(
        self,
        state: object,
        command: DuplexInputAppendCommand,
    ) -> DuplexInputEffect: ...

    def commit(
        self,
        state: object,
        command: DuplexInputCommitCommand,
    ) -> DuplexInputEffect: ...

    def clear(
        self,
        state: object,
        command: DuplexInputClearCommand,
    ) -> DuplexInputEffect: ...

    def flush(
        self,
        state: object,
        command: DuplexInputFlushCommand,
    ) -> DuplexInputEffect: ...

    def close(
        self,
        state: object,
        command: DuplexInputCloseCommand,
    ) -> DuplexInputEffect: ...


class DuplexInputReservation(Protocol):
    operation_id: str

    @property
    def active(self) -> bool: ...

    @property
    def byte_count(self) -> int: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _validate_identity_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a plain non-negative integer")


@dataclass(frozen=True, slots=True)
class DuplexPendingInputContinuation:
    incarnation: int
    epoch: int
    source_input_seq: int

    def __post_init__(self) -> None:
        _validate_identity_int("incarnation", self.incarnation)
        _validate_identity_int("epoch", self.epoch)
        _validate_identity_int("source_input_seq", self.source_input_seq)

    def is_stale(self, *, fence: DuplexFence, source_input_seq: int) -> bool:
        return (
            fence.incarnation != self.incarnation
            or fence.epoch != self.epoch
            or source_input_seq != self.source_input_seq
        )


@dataclass(frozen=True, slots=True)
class DuplexActiveOutputContinuation:
    incarnation: int
    epoch: int
    output_id: str

    def __post_init__(self) -> None:
        _validate_identity_int("incarnation", self.incarnation)
        _validate_identity_int("epoch", self.epoch)
        if not isinstance(self.output_id, str) or not self.output_id.strip():
            raise ValueError("output_id must be a non-empty string")

    def is_stale(self, *, fence: DuplexFence, output_id: str) -> bool:
        return fence.incarnation != self.incarnation or fence.epoch != self.epoch or output_id != self.output_id


@dataclass(frozen=True, slots=True)
class DuplexRuntimeProjectionBatch:
    events: tuple[DuplexModelEvent, ...] = ()
    controls: tuple[object, ...] = ()


@dataclass(slots=True)
class ServingRuntimeSessionState:
    """Framework lifecycle state plus an opaque model-owned input state."""

    input_state: object
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    pending_input_continuation: object | None = None
    active_output_continuation: object | None = None
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.pending_input_continuation = None
        self.active_output_continuation = None
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


class RuntimeDataPlane(Protocol):
    def begin_request(self, request_id: str) -> None: ...

    def is_terminal(self, request_id: str | None) -> bool: ...

    def mark_terminal(self, request_id: str) -> None: ...

    def close_stream(self, request_id: str) -> None: ...

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None: ...

    def project(self, result: object, *, context: object | None = None) -> Iterable[DuplexModelEvent]: ...


class ServingRuntimeAdapter(Protocol):
    adapter_id: str
    input_completion_mode: DuplexInputCompletionMode
    session_states: MutableMapping[str, object]
    input_controller: DuplexInputController
    data_plane: RuntimeDataPlane
    clean_response_done_prefix: str
    interrupted_tts_prefix: str
    private_runtime_config_keys: frozenset[str]

    def create_session_state(self) -> ServingRuntimeSessionState: ...

    def session_state(self, session_id: str) -> ServingRuntimeSessionState: ...

    def remove_session_state(self, session_id: str) -> None: ...

    def is_enabled(self, config: object) -> bool: ...

    def capabilities(self, *, max_sessions: int) -> object: ...

    def validate_client_extra_body(self, extra_body: object) -> None: ...

    async def prepare_runtime_config(self, config: object, *, model_config: Any) -> dict[str, object]: ...

    def runtime_config_for_update(
        self,
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]: ...

    def data_plane_context(
        self,
        *,
        fence: DuplexFence,
        source_input_seq: int,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> object: ...


def load_serving_runtime_adapter(
    path: str,
    encode_audio,
) -> ServingRuntimeAdapter:
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid duplex serving runtime adapter path: {path!r}")
    adapter_type = getattr(import_module(module_name), attribute_name)
    return validate_serving_runtime_adapter(adapter_type(encode_audio))


def validate_serving_runtime_adapter(adapter: object) -> ServingRuntimeAdapter:
    required_methods = (
        "create_session_state",
        "session_state",
        "remove_session_state",
        "is_enabled",
        "capabilities",
        "validate_client_extra_body",
        "prepare_runtime_config",
        "runtime_config_for_update",
        "data_plane_context",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(f"Duplex serving runtime adapter is missing callable method(s): {', '.join(missing)}")
    if not isinstance(getattr(adapter, "adapter_id", None), str) or not adapter.adapter_id:
        raise TypeError("Duplex serving runtime adapter must declare adapter_id")
    completion_mode = getattr(
        adapter,
        "input_completion_mode",
        DuplexInputCompletionMode.APPEND_ACCEPTED,
    )
    if not isinstance(completion_mode, DuplexInputCompletionMode):
        raise TypeError("Duplex serving runtime adapter input_completion_mode must be a DuplexInputCompletionMode")
    if not isinstance(getattr(adapter, "session_states", None), MutableMapping):
        raise TypeError("Duplex serving runtime adapter must declare mutable session_states")
    input_controller = getattr(adapter, "input_controller", None)
    input_controller_methods = (
        "create_state",
        "snapshot",
        "append",
        "commit",
        "clear",
        "flush",
        "close",
    )
    missing_input_controller_methods = [
        name for name in input_controller_methods if not callable(getattr(input_controller, name, None))
    ]
    if missing_input_controller_methods:
        raise TypeError(
            "Duplex serving runtime adapter input_controller is missing "
            "callable method(s): " + ", ".join(missing_input_controller_methods)
        )
    for prefix_name in ("clean_response_done_prefix", "interrupted_tts_prefix"):
        if not isinstance(getattr(adapter, prefix_name, None), str):
            raise TypeError(f"Duplex serving runtime adapter must declare {prefix_name}")
    private_keys = getattr(adapter, "private_runtime_config_keys", None)
    if not isinstance(private_keys, frozenset) or any(not isinstance(key, str) for key in private_keys):
        raise TypeError("Duplex serving runtime adapter must declare private_runtime_config_keys as frozenset[str]")
    data_plane = getattr(adapter, "data_plane", None)
    if data_plane is None:
        raise TypeError("Duplex serving runtime adapter must declare data_plane")
    data_plane_methods = (
        "begin_request",
        "is_terminal",
        "mark_terminal",
        "close_stream",
        "close_session",
        "project",
    )
    missing_data_plane_methods = [name for name in data_plane_methods if not callable(getattr(data_plane, name, None))]
    if missing_data_plane_methods:
        raise TypeError(
            "Duplex serving runtime adapter data_plane is missing callable method(s): "
            + ", ".join(missing_data_plane_methods)
        )
    return adapter  # type: ignore[return-value]


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id", payload.get("model_turn_id")))


def coerce_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
