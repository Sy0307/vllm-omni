from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSession

logger = init_logger(__name__)


class ChatFallbackProjectorMixin:
    """Project generic chat completion output into duplex response events."""

    async def _run_response(self, session: DuplexSession, send_json) -> None:
        response_id = session.begin_response()
        self._remember_response_conversation_mode(session, response_id)
        epoch = session.epoch
        request_id = f"duplex-{session.session_id}-{epoch}-{session.input_commit_seq}"
        session.active_request_id = f"chatcmpl-{request_id}"
        await send_json(
            self._response_created_payload(
                session,
                response_id,
                epoch=epoch,
                request_id=session.active_request_id,
            )
        )

        try:
            request = self._build_chat_request(session, request_id)
            result = await self._chat_service.create_chat_completion(request, raw_request=None)
            if isinstance(result, ErrorResponse):
                await send_json({"type": "error", "error": result.message, "code": result.type or "chat_error"})
                session.end_response(commit_text=False)
                return
            if hasattr(result, "__aiter__"):
                await self._drain_streaming_response(session, result, epoch, response_id, send_json)
            else:
                await self._emit_full_response(session, result, epoch, response_id, send_json)
            if session.epoch == epoch:
                should_commit = self._should_commit_response_to_history(response_id)
                committed_message = session.end_response(commit_text=should_commit)
                if should_commit:
                    session.register_history_item(f"item_{response_id}", committed_message)
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": epoch,
                        "committed": committed_message is not None,
                        "playback": session.playback.as_dict(),
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Duplex response failed: %s", exc)
            session.end_response(commit_text=False)
            await send_json(
                {
                    "type": "error",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "error": str(exc),
                    "code": "response_error",
                }
            )

    def _build_chat_request(self, session: DuplexSession, request_id: str) -> ChatCompletionRequest:
        messages: list[dict[str, object]] = []
        if session.config.instructions:
            messages.append({"role": "system", "content": session.config.instructions})
        messages.extend(session.history)

        kwargs: dict[str, Any] = {
            "model": session.config.model or self._chat_service.model_config.model,
            "messages": messages,
            "stream": True,
        }
        if session.config.temperature is not None:
            kwargs["temperature"] = session.config.temperature
        if session.config.max_tokens is not None:
            kwargs["max_tokens"] = session.config.max_tokens
        kwargs.update(session.config.extra_body)

        request = ChatCompletionRequest(**kwargs)
        object.__setattr__(request, "modalities", session.config.modalities)
        object.__setattr__(request, "request_id", request_id)
        object.__setattr__(
            request,
            "chat_template_kwargs",
            {"use_tts_template": session.config.use_tts_template},
        )
        return request

    async def _drain_streaming_response(
        self,
        session: DuplexSession,
        result: AsyncGenerator[str, None],
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        async for raw_chunk in result:
            if session.epoch != epoch:
                return
            for payload in self._parse_sse_payloads(raw_chunk):
                if payload == "[DONE]":
                    continue
                if isinstance(payload, dict):
                    await self._emit_chat_payload(session, payload, epoch, response_id, send_json)

    async def _emit_full_response(
        self,
        session: DuplexSession,
        result: Any,
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        if hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json", exclude_unset=True)
        else:
            payload = {"response": str(result)}
        await self._emit_chat_payload(session, payload, epoch, response_id, send_json)

    def _parse_sse_payloads(self, raw_chunk: str) -> list[dict[str, object] | str]:
        payloads: list[dict[str, object] | str] = []
        for line in raw_chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                payloads.append(data)
                continue
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON duplex stream payload: %s", data)
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
        return payloads

    async def _emit_chat_payload(
        self,
        session: DuplexSession,
        payload: dict[str, object],
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        modality = payload.get("modality")
        if modality not in {None, "text", "audio"}:
            await send_json(
                {
                    "type": "error",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": epoch,
                    "code": "unsupported_response_modality",
                    "error": f"Unsupported chat response modality: {modality}",
                }
            )
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            await send_json(
                {
                    "type": "response.message",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": epoch,
                    "payload": payload,
                }
            )
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            message = choice.get("message")
            content = None
            if isinstance(delta, dict):
                content = delta.get("content")
            elif isinstance(message, dict):
                content = message.get("content")

            if isinstance(content, str) and content:
                if modality == "audio":
                    session.mark_audio_sent()
                    await send_json(
                        {
                            "type": "response.output_audio.delta",
                            "session_id": session.session_id,
                            "response_id": response_id,
                            "epoch": epoch,
                            "audio": content,
                            "format": session.config.response_format,
                        }
                    )
                else:
                    session.append_assistant_text(content)
                    await send_json(
                        {
                            "type": "response.text.delta",
                            "session_id": session.session_id,
                            "response_id": response_id,
                            "epoch": epoch,
                            "delta": content,
                        }
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and modality != "audio":
                await send_json(
                    {
                        "type": "response.output_item.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": epoch,
                        "finish_reason": finish_reason,
                        "modality": modality,
                    }
                )
