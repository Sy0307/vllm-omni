# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocketDisconnect


CONTROL_EVENTS = frozenset(
    {
        "session.close",
        "close_session",
        "input.cancel",
        "response.cancel",
        "barge_in",
        "input_audio_buffer.clear",
        "output_audio_buffer.clear",
        "turn.signal",
        "signal_turn",
        "playback.ack",
        "audio.playback_ack",
    }
)
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
        "response.output_item.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio.delta",
        "response.audio.delta",
        "response.output_audio.done",
        "response.audio.done",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
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


def is_control_event(event_type: str) -> bool:
    return event_type in CONTROL_EVENTS


def is_input_event(event_type: str) -> bool:
    return event_type in INPUT_EVENTS


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


@dataclass(frozen=True, slots=True)
class DuplexAppendTaskMeta:
    epoch: int
    mode: str
    final: bool
    response_bound: bool


@dataclass
class DuplexWebSocketActor:
    """Own WebSocket I/O and prioritized queues, but no domain identity."""

    websocket: Any
    current_epoch: Callable[[], int | None] | None = None
    output_queue: asyncio.Queue[dict[str, object] | None] = field(default_factory=asyncio.Queue)
    input_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    control_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    event_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    session: Any | None = None
    outbound_protocol: Any | None = None
    native_append_tasks: dict[asyncio.Task[None], DuplexAppendTaskMeta] = field(default_factory=dict)
    active_response_task: asyncio.Task[None] | None = None
    runtime_opened: bool = False
    runtime_closed: bool = False
    closing: bool = False
    lifecycle_state: str = "opening"
    close_reason: str | None = None
    stale_output_dropped: int = 0
    control_events_seen: int = 0
    input_events_seen: int = 0
    cancel_count: int = 0
    overlap_speech_ms: int = 0
    last_response_id: str | None = None
    _next_event_seq: int = 0
    _deferred_events: list[dict[str, object]] = field(default_factory=list)

    def transition(self, state: str, *, reason: str | None = None) -> None:
        self.lifecycle_state = state
        if reason is not None:
            self.close_reason = reason
        if self.session is None:
            return
        if state in {"closing", "closed"}:
            self.session.mark_closing()
        elif state == "listening":
            self._set_turn_state("user_speaking")
        elif state == "generating":
            self._set_turn_state("assistant_generating")

    def _set_turn_state(self, value: str) -> None:
        current = getattr(self.session, "turn_state", None)
        try:
            self.session.turn_state = type(current)(value)
        except (TypeError, ValueError):
            self.session.turn_state = value

    async def enqueue_event(self, event: dict[str, object]) -> None:
        event["_duplex_actor_seq"] = self._next_event_seq
        self._next_event_seq += 1
        event_type = event.get("type")
        if event_type in {"__timeout__", "__disconnect__"} or (
            isinstance(event_type, str) and is_control_event(event_type)
        ):
            self.control_events_seen += 1
            await self.control_queue.put(event)
        elif isinstance(event_type, str) and is_input_event(event_type):
            self.input_events_seen += 1
            if self.output_generation_in_flight():
                event["_duplex_overlap_candidate"] = True
            await self.input_queue.put(event)
        else:
            await self.event_queue.put(event)

    def output_generation_in_flight(self) -> bool:
        if self.session is None:
            return self.lifecycle_state == "generating" or self.has_response_bound_append_tasks()
        if self.assistant_playback_active() or self.lifecycle_state == "generating":
            return True
        if self.session.active_response_id is not None or self.session.active_request_id is not None:
            return True
        if self.active_response_task is not None and not self.active_response_task.done():
            return True
        return self.has_response_bound_append_tasks()

    def assistant_playback_active(self) -> bool:
        if self.session is None:
            return False
        policy = getattr(getattr(self.session, "config", None), "playback_commit_policy", None)
        if _enum_value(policy) != "ack_only":
            return False
        playback = self.session.playback
        return playback.sent_ms > playback.committed_ms

    async def next_event(self) -> dict[str, object]:
        while True:
            ready = [
                (self._event_priority(event), self._event_seq(event), event)
                for event in self._deferred_events
            ]
            self._deferred_events.clear()
            for queue in (self.control_queue, self.event_queue, self.input_queue):
                with suppress(asyncio.QueueEmpty):
                    event = queue.get_nowait()
                    ready.append((self._event_priority(event), self._event_seq(event), event))
            if not ready:
                ready = await self._wait_for_events()
            if not ready:
                continue
            ready.sort(key=lambda item: (item[0], item[1]))
            selected = ready[0][2]
            self._deferred_events.extend(event for _, _, event in ready[1:])
            selected.pop("_duplex_actor_seq", None)
            return selected

    async def _wait_for_events(self) -> list[tuple[int, int, dict[str, object]]]:
        tasks = [
            asyncio.create_task(self.control_queue.get()),
            asyncio.create_task(self.event_queue.get()),
            asyncio.create_task(self.input_queue.get()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*pending)
        ready = [
            (self._event_priority(task.result()), self._event_seq(task.result()), task.result())
            for task in tasks
            if task in done
        ]
        for queue in (self.control_queue, self.event_queue, self.input_queue):
            while True:
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                ready.append((self._event_priority(event), self._event_seq(event), event))
        return ready

    @staticmethod
    def _event_seq(event: dict[str, object]) -> int:
        value = event.get("_duplex_actor_seq")
        return value if isinstance(value, int) else 0

    @staticmethod
    def _event_priority(event: dict[str, object]) -> int:
        event_type = event.get("type")
        if event_type in {
            "__timeout__",
            "__disconnect__",
            "response.cancel",
            "barge_in",
            "input_audio_buffer.clear",
            "output_audio_buffer.clear",
        }:
            return 0
        if event_type == "input.cancel" or (isinstance(event_type, str) and is_input_event(event_type)):
            return 2
        if event_type in {"session.close", "close_session"}:
            return 3
        return 1

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
        if event_type not in MODEL_OUTPUT_EVENTS:
            return False
        if self.closing and event_type not in {"response.listen", "runtime.control"}:
            return True
        expected_epoch = self.current_epoch() if self.current_epoch is not None else None
        if expected_epoch is None and self.session is not None:
            if _enum_value(getattr(self.session, "state", None)) == "closed" and event_type not in {
                "response.listen",
                "runtime.control",
            }:
                return True
            expected_epoch = getattr(self.session, "epoch", None)
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and isinstance(expected_epoch, int) and epoch != expected_epoch

    def track_append_task(
        self,
        task: asyncio.Task[None],
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
            task
            for task, meta in self.native_append_tasks.items()
            if not response_bound_only or meta.response_bound
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

    def drain_input_queue(self) -> int:
        drained = 0
        kept = []
        for event in self._deferred_events:
            event_type = event.get("type")
            if isinstance(event_type, str) and is_input_event(event_type):
                drained += 1
            else:
                kept.append(event)
        self._deferred_events = kept
        while True:
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
        return drained


__all__ = [
    "DuplexAppendTaskMeta",
    "DuplexWebSocketActor",
    "is_control_event",
    "is_input_event",
]
