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
    MiniCPMO45ProjectionBatch,
)
from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities

EncodeAudio = Callable[[object, int, str, float | None], str | None]


class MiniCPMO45SideControlServingAdapter:
    """Nominal MiniCPM serving extension for typed events plus side controls."""

    def project_runtime_batches(
        self,
        result: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[MiniCPMO45ProjectionBatch]:
        raise NotImplementedError


class MiniCPMO45ServingRuntimeAdapter(MiniCPMO45SideControlServingAdapter):
    """MiniCPM-owned serving state, input packing, and output projection."""

    adapter_id = "minicpmo45"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = MiniCPMO45NativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, MiniCPMO45ServingSessionState] = {}
        self.data_plane = MiniCPMO45DataPlaneSession(encode_audio)

    def create_session_state(self) -> MiniCPMO45ServingSessionState:
        return MiniCPMO45ServingSessionState()

    def project_runtime_batches(
        self,
        result: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[MiniCPMO45ProjectionBatch]:
        return self.data_plane.project_runtime_batches(result, context=context)

    def session_state(self, session_id: str) -> MiniCPMO45ServingSessionState:
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
