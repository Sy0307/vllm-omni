# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocketDisconnect

INPUT_EVENTS = frozenset(
    {
        "input.text.append",
        "input_text.append",
        "push_text",
        "input.audio.append",
        "input_audio_buffer.append",
        "push_chunk",
        "input.commit",
        "input_audio_buffer.commit",
        "response.create",
    }
)
MODEL_OUTPUT_EVENTS = frozenset(
    {
        "response.created",
        "response.listen",
        "response.speak",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio.delta",
        "response.audio.delta",
        "response.output_audio.done",
        "response.audio.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.text.delta",
        "response.text.done",
        "response.message",
        "response.output_item.done",
        "response.content_part.done",
        "response.done",
        "runtime.control",
    }
)
DOMAIN_TERMINAL_EVENTS = frozenset(
    {
        "response.done",
        "response.listen",
        "audio.cancelled",
        "input.cancelled",
        "session.closed",
    }
)


def is_input_event(event_type: str) -> bool:
    return event_type in INPUT_EVENTS


@dataclass(frozen=True, slots=True)
class DuplexAppendTaskMeta:
    epoch: int
    mode: str
    final: bool
    response_bound: bool


@dataclass
class DuplexWebSocketActor:
    """Own ordered WebSocket I/O queues, but no domain identity."""

    websocket: Any
    current_epoch: Callable[[], int | None] | None = None
    session_closed: Callable[[], bool] | None = None
    output_queue: asyncio.Queue[dict[str, object] | None] = field(default_factory=asyncio.Queue)
    mailbox: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    outbound_protocol: Any | None = None
    native_append_tasks: dict[asyncio.Task[bool], DuplexAppendTaskMeta] = field(default_factory=dict)
    active_response_task: asyncio.Task[None] | None = None
    closing: bool = False
    close_reason: str | None = None
    stale_output_dropped: int = 0

    async def enqueue_event(self, event: dict[str, object]) -> None:
        await self.mailbox.put(event)

    async def next_event(self) -> dict[str, object]:
        event = await self.mailbox.get()
        self.mailbox.task_done()
        return event

    async def send_json(self, payload: dict[str, object]) -> None:
        await self.output_queue.put(payload)

    async def close_writer(self) -> None:
        await self.output_queue.put(None)

    async def writer_loop(self) -> None:
        while True:
            payload = await self.output_queue.get()
            try:
                if payload is None:
                    return
                raw_realtime = payload.pop("_realtime_raw", False) is True
                if not raw_realtime and self._is_stale_model_output(payload):
                    self.stale_output_dropped += 1
                    continue
                try:
                    if raw_realtime:
                        await self.websocket.send_json(payload)
                    elif self.outbound_protocol is not None:
                        for projected in self.outbound_protocol.encode_outbound_event(payload):
                            await self.websocket.send_json(projected)
                    else:
                        await self.websocket.send_json(payload)
                except (WebSocketDisconnect, RuntimeError):
                    return
            finally:
                self.output_queue.task_done()

    def _is_stale_model_output(self, payload: dict[str, object]) -> bool:
        event_type = payload.get("type")
        if event_type in DOMAIN_TERMINAL_EVENTS:
            return False
        if event_type not in MODEL_OUTPUT_EVENTS:
            return False
        if self.closing and event_type not in {"response.listen", "runtime.control"}:
            return True
        expected_epoch = self.current_epoch() if self.current_epoch is not None else None
        if (
            self.session_closed is not None
            and self.session_closed()
            and event_type
            not in {
                "response.listen",
                "runtime.control",
            }
        ):
            return True
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and isinstance(expected_epoch, int) and epoch != expected_epoch

    def track_append_task(
        self,
        task: asyncio.Task[bool],
        *,
        epoch: int,
        mode: str,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.native_append_tasks[task] = DuplexAppendTaskMeta(epoch, mode, final, response_bound)
        task.add_done_callback(self.native_append_tasks.pop)

    def has_response_bound_append_tasks(self) -> bool:
        return any(meta.response_bound for meta in self.native_append_tasks.values())

    async def cancel_append_tasks(self, timeout_s: float = 0.25, *, response_bound_only: bool = False) -> bool:
        tasks = [
            task for task, meta in self.native_append_tasks.items() if not response_bound_only or meta.response_bound
        ]
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except TimeoutError:
            pass
        return True


__all__ = [
    "DOMAIN_TERMINAL_EVENTS",
    "DuplexAppendTaskMeta",
    "DuplexWebSocketActor",
    "is_input_event",
]
