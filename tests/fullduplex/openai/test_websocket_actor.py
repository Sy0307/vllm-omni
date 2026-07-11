# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.openai.websocket import DuplexWebSocketActor


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(dict(payload))


@pytest.mark.asyncio
async def test_control_event_preempts_earlier_audio_input():
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "pcm"})
    await actor.enqueue_event({"type": "response.cancel"})

    assert await actor.next_event() == {"type": "response.cancel"}
    assert await actor.next_event() == {"type": "input_audio_buffer.append", "audio": "pcm"}


@pytest.mark.asyncio
async def test_writer_is_single_owner_of_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "one"})
    await actor.send_json({"type": "two"})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "one"}, {"type": "two"}]


@pytest.mark.asyncio
async def test_stale_fence_payload_is_dropped_before_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket, current_epoch=lambda: 2)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "response.audio.delta", "epoch": 1})
    await actor.send_json({"type": "response.audio.delta", "epoch": 2})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "response.audio.delta", "epoch": 2}]
    assert actor.stale_output_dropped == 1
