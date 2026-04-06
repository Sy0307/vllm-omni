"""WebSocket handler for streaming video input understanding.

Accepts video frames incrementally via WebSocket, buffers them, and
generates text + optional audio responses using the existing Qwen3-Omni
multi-stage pipeline (thinker -> talker -> code2wav).

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import io
import json
import struct
import time as _time
import uuid
import wave
from typing import Any

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger

from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64
_CODEC_FRAME_SAMPLES = 1920  # CausalConv upsample artifact


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default=["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Max frames to keep in the buffer.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt.",
    )
    use_audio_in_video: bool = Field(
        default=False,
        description="Extract and interleave audio from video frames.",
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )


class OmniStreamingVideoHandler:
    """Handles WebSocket sessions for streaming video input.

    Supports:
    - Concurrent frame reception during query processing (reader/processor split)
    - PCM audio input (``audio.chunk``)
    - Async-chunk incremental audio output via ``engine_client.generate()``
    - Multi-turn conversation history
    - Soft interrupt (new query cancels current generation)
    """

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._engine_client = engine_client

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            # Session state
            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            audio_buffer = bytearray()  # raw PCM16 16kHz mono
            message_history: list[dict[str, Any]] = []
            active_request_id: str | None = None
            prev_request_id: str | None = None  # for abort on new query
            interrupt_event = asyncio.Event()

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _reader() -> None:
                """Receive WebSocket messages and enqueue them."""
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout")
                            await msg_queue.put(None)
                            return

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue

                        if not isinstance(msg, dict):
                            await self._send_error(websocket, "Messages must be JSON objects")
                            continue

                        await msg_queue.put(msg)
                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _cancel_active_query() -> None:
                """Signal soft interrupt for the active query."""
                nonlocal active_request_id
                if active_request_id is not None:
                    interrupt_event.set()
                    logger.info("Interrupt signaled for %s", active_request_id)

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, prev_request_id

                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        return

                    msg_type = msg.get("type")

                    if msg_type == "video.frame":
                        frame_data = msg.get("data", "")
                        if not frame_data:
                            continue
                        if len(frame_data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(frame_data)
                            Image.open(io.BytesIO(raw_bytes)).verify()
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
                            continue
                        max_buf = config.max_frames
                        if len(frame_buffer) >= max_buf:
                            frame_buffer.pop(0)
                        frame_buffer.append(frame_data)

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            continue
                        audio_buffer.extend(pcm_bytes)

                    elif msg_type == "video.query":
                        # Interrupt any active query
                        await _cancel_active_query()

                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                audio_buffer.extend(base64.b64decode(audio_data_b64))
                            except Exception:
                                pass

                        if not frame_buffer:
                            await self._send_error(websocket, "No frames buffered")
                            continue

                        # Abort previous request so the thinker scheduler
                        # releases KV blocks before the new prefill.
                        if prev_request_id and self._engine_client:
                            try:
                                await self._engine_client.abort(prev_request_id)
                            except Exception:
                                pass
                            # Give scheduler time to process the abort
                            await asyncio.sleep(0.1)

                        request_id = f"video-{uuid.uuid4().hex[:12]}"
                        active_request_id = request_id
                        interrupt_event.clear()

                        await self._process_query(
                            websocket,
                            config,
                            frame_buffer,
                            audio_buffer,
                            message_history,
                            query_text,
                            request_id,
                            interrupt_event,
                        )

                        prev_request_id = active_request_id
                        active_request_id = None
                        audio_buffer.clear()

                    elif msg_type == "video.done":
                        await _cancel_active_query()
                        if active_request_id and self._engine_client:
                            try:
                                await self._engine_client.abort(active_request_id)
                            except Exception:
                                pass
                        await websocket.send_json({"type": "session.done"})
                        return

                    elif msg_type == "ping":
                        # Heartbeat — respond with pong
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass

                    else:
                        await self._send_error(websocket, f"Unknown type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass

        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(self, websocket: WebSocket) -> StreamingVideoSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        try:
            config = StreamingVideoSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is not None:
            await self._process_query_engine(
                websocket, config, frame_buffer, audio_buffer,
                message_history, query_text, request_id, interrupt_event,
            )
        else:
            await self._process_query_chat(
                websocket, config, frame_buffer, audio_buffer,
                message_history, query_text,
            )

    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

    async def _process_query_engine(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
    ) -> None:
        """Direct engine_client.generate() path for async_chunk audio."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        messages, user_message = self._build_messages(
            config, frame_buffer, audio_buffer, message_history, query_text,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
        }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list

        try:
            chat_request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        # Preprocess to engine prompt
        try:
            engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        except Exception as e:
            await self._send_error(websocket, f"Preprocess failed: {e}")
            return

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        audio_chunk_count = 0
        audio_samples_sent = 0
        previous_text = ""
        interrupted = False
        t_start = _time.monotonic()
        t_first_text = None

        try:
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
            )

            async for output in result_gen:
                # Soft interrupt: drain without sending
                if interrupt_event.is_set():
                    if not interrupted:
                        logger.info("Generation interrupted — draining")
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                out_type = getattr(output, "final_output_type", "text")

                if out_type == "audio":
                    if not text_done_sent:
                        full_text = "".join(text_parts)
                        await websocket.send_json(
                            {"type": "response.text.done", "text": full_text}
                        )
                        text_done_sent = True

                    audio_chunk_count += 1
                    b64, audio_samples_sent = self._extract_audio_delta_b64(
                        output, audio_samples_sent,
                    )
                    if b64:
                        await websocket.send_json({
                            "type": "response.audio.delta",
                            "data": b64,
                            "format": "wav",
                        })
                else:
                    delta_text, previous_text = self._extract_text_delta(
                        output, previous_text,
                    )
                    if delta_text:
                        if t_first_text is None:
                            t_first_text = _time.monotonic()
                        text_parts.append(delta_text)
                        await websocket.send_json(
                            {"type": "response.text.delta", "delta": delta_text}
                        )

            if audio_chunk_count > 0:
                await websocket.send_json({"type": "response.audio.done"})

            # Commit turn to history
            response_text = "".join(text_parts)
            message_history.append(user_message)
            message_history.append({"role": "assistant", "content": response_text})

            t_end = _time.monotonic()
            logger.info(
                "[TIMING] total=%.2fs first_text=%.2fs audio_chunks=%d",
                t_end - t_start,
                (t_first_text - t_start) if t_first_text else -1,
                audio_chunk_count,
            )

        except Exception:
            logger.exception("Engine query failed")
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = "".join(text_parts)
            await websocket.send_json({"type": "response.text.done", "text": full_text})

    # ------------------------------------------------------------------
    # Chat-service path (SSE fallback, no streaming audio)
    # ------------------------------------------------------------------

    async def _process_query_chat(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
    ) -> None:
        """Fallback path via chat_service.create_chat_completion()."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        messages, user_message = self._build_messages(
            config, frame_buffer, audio_buffer, message_history, query_text,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
        }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list

        try:
            request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        await websocket.send_json({"type": "response.start"})
        full_text = ""

        try:
            generator = await self._chat_service.create_chat_completion(
                request, raw_request=None,
            )
            if hasattr(generator, "__aiter__"):
                async for chunk_str in generator:
                    if not isinstance(chunk_str, str):
                        continue
                    for line in chunk_str.strip().split("\n"):
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content and isinstance(content, str):
                            full_text += content
                            await websocket.send_json(
                                {"type": "response.text.delta", "delta": content}
                            )
        except Exception as e:
            logger.error("Chat query failed: %s", e)
            await self._send_error(websocket, f"Generation failed: {e}")
            return

        await websocket.send_json({"type": "response.text.done", "text": full_text})

        # Commit turn
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": full_text})

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build OpenAI-style messages list and the current user message.

        Returns (messages, user_message).
        """
        # Sample frames
        frames = frame_buffer
        if len(frames) > config.num_frames:
            indices = np.linspace(0, len(frames) - 1, config.num_frames, dtype=int)
            frames = [frame_buffer[i] for i in indices]

        # Build user content parts
        user_content: list[dict] = []
        for frame_b64 in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        # Audio input
        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": wav_b64,
                    "format": "wav",
                },
            })

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        # Build full messages
        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        # Add text-only history (strip images/audio from old turns).
        # Keep only the last turn (2 messages) to keep prompt short
        # enough for single-step mm_encoder scheduling.  When prompt
        # exceeds ~50 tokens, the V1 scheduler splits mm_encoder and
        # text prefill, causing incomplete thinker embeddings and
        # garbled audio.
        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))

        messages.append(user_message)

        return messages, user_message

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:
        """Wrap raw PCM16 mono in a WAV container and return base64."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _extract_audio_delta_b64(
        cls, result: OmniRequestOutput, samples_sent: int,
    ) -> tuple[str | None, int]:
        """Extract only NEW audio samples since last call.

        Returns (b64_wav_or_none, updated_samples_sent).
        """
        try:
            audio_np = cls._extract_audio_raw(result)
            if audio_np is None:
                return None, samples_sent

            total_samples = len(audio_np)
            trim = _CODEC_FRAME_SAMPLES if total_samples > _CODEC_FRAME_SAMPLES * 2 else 0
            effective_start = max(trim, samples_sent)
            if effective_start >= total_samples:
                return None, samples_sent

            delta_np = audio_np[effective_start:]
            if len(delta_np) == 0:
                return None, samples_sent

            b64 = cls._encode_audio_wav_b64(delta_np)
            return b64, total_samples
        except Exception:
            logger.exception("Failed to extract audio delta")
            return None, samples_sent

    @staticmethod
    def _extract_audio_raw(result: OmniRequestOutput):
        """Extract raw audio numpy array from OmniRequestOutput."""
        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return None
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return None
        mm_output = getattr(outputs[0], "multimodal_output", None)
        if not isinstance(mm_output, dict):
            return None
        audio_data = mm_output.get("audio")
        if audio_data is None:
            return None
        if isinstance(audio_data, list):
            audio_tensor = torch.cat(audio_data, dim=-1)
        else:
            audio_tensor = audio_data
        audio_np = audio_tensor.float().detach().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        return audio_np

    @staticmethod
    def _encode_audio_wav_b64(audio_np) -> str:
        """Encode numpy float32 audio to base64 WAV (24kHz)."""
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio
        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin

        audio_obj = CreateAudio(
            audio_tensor=audio_np,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            stream_format="audio",
            base64_encode=True,
        )
        mixin = AudioMixin()
        resp = mixin.create_audio(audio_obj)
        return resp.audio_data

    @staticmethod
    def _extract_text_delta(
        result: OmniRequestOutput, previous_text: str,
    ) -> tuple[str, str]:
        """Extract incremental text delta from OmniRequestOutput."""
        if result.final_output_type != "text":
            return "", previous_text

        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return "", previous_text

        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return "", previous_text

        text = getattr(outputs[0], "text", None)
        if not isinstance(text, str) or not text:
            return "", previous_text

        if text.startswith(previous_text):
            return text[len(previous_text):], text
        return text, text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _preprocess_to_engine_prompt(self, request) -> Any:
        """Use the chat handler's preprocessing to build an engine prompt."""
        handler = self._chat_service
        renderer = handler.renderer

        _conversation, engine_prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None)
            or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            renderer=renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return engine_prompts[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _text_only_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Strip multimodal content from a message for history."""
        content = msg.get("content")
        if isinstance(content, str):
            return msg
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            return {"role": msg.get("role", "user"), "content": " ".join(text_parts)}
        return {"role": msg.get("role", "user"), "content": str(content) if content else ""}

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
