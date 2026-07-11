# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64

from vllm_omni.experimental.fullduplex.core.events import EmitProtocolEvent, ProtocolEventKind
from vllm_omni.experimental.fullduplex.openai.history import ResponseLifecycleLedger


class RealtimeEventProjector:
    """Purely project fenced core effects into OpenAI Realtime events."""

    def __init__(self, lifecycle: ResponseLifecycleLedger | None = None) -> None:
        self.lifecycle = lifecycle or ResponseLifecycleLedger()

    def project(self, effect: EmitProtocolEvent) -> tuple[dict[str, object], ...]:
        fence = effect.fence
        if effect.kind is ProtocolEventKind.RESPONSE_STARTED:
            response_id = self.lifecycle.start(fence)
            return (
                {
                    "type": "response.created",
                    "response": {"id": response_id, "status": "in_progress"},
                },
            )
        response_id = self.lifecycle.response_id(fence)
        if effect.kind is ProtocolEventKind.TEXT_DELTA:
            text = str(effect.payload or "")
            self.lifecycle.append_text(fence, text)
            return ({"type": "response.output_text.delta", "response_id": response_id, "delta": text},)
        if effect.kind is ProtocolEventKind.AUDIO_DELTA:
            data = effect.payload if isinstance(effect.payload, bytes) else b""
            self.lifecycle.append_audio(fence, data)
            return (
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(data).decode("ascii"),
                },
            )
        if effect.kind is ProtocolEventKind.SEGMENT_ENDED:
            return ({"type": "response.output_audio.done", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.RESPONSE_COMPLETED:
            transcript = self.lifecycle.finish(fence)
            return (
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "transcript": transcript,
                    },
                },
            )
        if effect.kind is ProtocolEventKind.MODEL_LISTENING:
            return ({"type": "response.listen", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.ENGINE_FAILED:
            return ({"type": "error", "error": str(effect.payload or "engine failure")},)
        return ()
