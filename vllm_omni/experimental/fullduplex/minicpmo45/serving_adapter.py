from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.adapter import (
    MiniCPMO45NativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    DuplexInputAppendCommand,
    DuplexInputClearCommand,
    DuplexInputCloseCommand,
    DuplexInputCommitCommand,
    DuplexInputCompletionMode,
    DuplexInputEffect,
    DuplexInputFlushCommand,
    DuplexInputSnapshot,
    DuplexRuntimeProjectionBatch,
    ServingRuntimeSessionState,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


class MiniCPMO45InputController:
    """Own MiniCPM packetization without leaking its buffer into serving."""

    @staticmethod
    def create_state() -> MiniCPMO45ServingSessionState:
        return MiniCPMO45ServingSessionState()

    @staticmethod
    def snapshot(state: object) -> DuplexInputSnapshot:
        if not isinstance(state, MiniCPMO45ServingSessionState):
            raise TypeError("invalid MiniCPM serving input state")
        return DuplexInputSnapshot(
            pending_byte_count=state.audio_buffer.pending_byte_count,
            has_pending=state.audio_buffer.has_pending(),
            has_reserved=state.audio_buffer.has_reserved(),
            input_since_commit=state.input_since_commit,
            speech_since_commit=state.speech_since_commit,
            committed_payload=state.committed_audio_payload,
            committed_operation_id=state.committed_audio_operation_id,
            committed_reserved_bytes=state.committed_audio_reserved_bytes,
        )

    @staticmethod
    def append(
        state: object,
        command: DuplexInputAppendCommand,
    ) -> DuplexInputEffect:
        if not isinstance(state, MiniCPMO45ServingSessionState):
            raise TypeError("invalid MiniCPM serving input state")
        if not isinstance(command.payload, dict):
            raise TypeError("MiniCPM append payload must be a dictionary")
        reservation = state.audio_buffer.prepare_append(
            command.payload,
            operation_id=command.operation_id,
            chunk_period_ms=command.chunk_period_ms,
            allow_emit=command.allow_emit,
        )
        if reservation is None:
            return DuplexInputEffect()
        payloads = () if reservation.payload is None else (reservation.payload,)
        return DuplexInputEffect(
            append_payloads=payloads,
            reservations=(reservation,),
        )

    @staticmethod
    def commit(
        state: object,
        command: DuplexInputCommitCommand,
    ) -> DuplexInputEffect:
        if not isinstance(state, MiniCPMO45ServingSessionState):
            raise TypeError("invalid MiniCPM serving input state")
        reservation = state.audio_buffer.prepare_commit(
            operation_id=command.operation_id,
            chunk_period_ms=command.chunk_period_ms,
        )
        payloads = () if reservation.payload is None else (reservation.payload,)
        return DuplexInputEffect(
            append_payloads=payloads,
            reservations=(reservation,),
        )

    @staticmethod
    def clear(
        state: object,
        command: DuplexInputClearCommand,
    ) -> DuplexInputEffect:
        if not isinstance(state, MiniCPMO45ServingSessionState):
            raise TypeError("invalid MiniCPM serving input state")
        if command.clear_buffer:
            state.audio_buffer.clear()
        if command.clear_force_listen:
            state.audio_buffer.clear_force_listen()
        state.input_since_commit = False
        state.speech_since_commit = False
        return DuplexInputEffect(released_bytes=state.clear_committed_audio())

    @staticmethod
    def flush(
        state: object,
        command: DuplexInputFlushCommand,
    ) -> DuplexInputEffect:
        if not isinstance(state, MiniCPMO45ServingSessionState):
            raise TypeError("invalid MiniCPM serving input state")
        payload = state.audio_buffer.flush(
            chunk_period_ms=command.chunk_period_ms,
        )
        return DuplexInputEffect(
            append_payloads=() if payload is None else (payload,),
        )

    def close(
        self,
        state: object,
        command: DuplexInputCloseCommand,
    ) -> DuplexInputEffect:
        del command
        return self.clear(
            state,
            DuplexInputClearCommand(reason="session_close"),
        )


class MiniCPMO45SideControlServingAdapter:
    """Nominal MiniCPM serving extension for typed events plus side controls."""

    def project_runtime_batches(
        self,
        result: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[DuplexRuntimeProjectionBatch]:
        raise NotImplementedError


class MiniCPMO45ServingRuntimeAdapter(MiniCPMO45SideControlServingAdapter):
    """MiniCPM-owned serving state, input packing, and output projection."""

    adapter_id = "minicpmo45"
    input_completion_mode = DuplexInputCompletionMode.APPEND_ACCEPTED
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = MiniCPMO45NativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.input_controller = MiniCPMO45InputController()
        self.session_states: dict[str, ServingRuntimeSessionState] = {}
        self.data_plane = MiniCPMO45DataPlaneSession(encode_audio)

    def create_session_state(self) -> ServingRuntimeSessionState:
        return ServingRuntimeSessionState(
            input_state=self.input_controller.create_state(),
        )

    def project_runtime_batches(
        self,
        result: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[DuplexRuntimeProjectionBatch]:
        for batch in self.data_plane.project_runtime_batches(
            result,
            context=context,
        ):
            yield DuplexRuntimeProjectionBatch(
                events=batch.events,
                controls=batch.tts_segment_controls,
            )

    def session_state(self, session_id: str) -> ServingRuntimeSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        return MiniCPMO45NativeDuplexServingAdapter.is_enabled(config)  # type: ignore[arg-type]

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities.minicpmo45_native(max_sessions=max_sessions)

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        MiniCPMO45NativeDuplexServingAdapter.validate_client_extra_body(extra_body)

    @staticmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]:
        return await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
            config,  # type: ignore[arg-type]
            model_config=model_config,
        )

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        return MiniCPMO45NativeDuplexServingAdapter.runtime_config_for_update(
            config,  # type: ignore[arg-type]
            dict(current),
        )

    @staticmethod
    def data_plane_context(
        *,
        fence: DuplexFence,
        source_input_seq: int,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> MiniCPMO45DataPlaneContext:
        return MiniCPMO45DataPlaneContext(
            fence=fence,
            source_input_seq=source_input_seq,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )
