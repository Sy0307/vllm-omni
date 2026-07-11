import pytest

from tests.entrypoints.openai_api.test_duplex_handler import FakeChatService, TimedWebSocket
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FenceRecordingEngine:
    output_modalities = ["text", "audio"]

    def __init__(self) -> None:
        self.calls: list[tuple[str, DuplexFence | None]] = []

    async def open_duplex_session_async(self, session_id, *, fence=None, **kwargs):
        del session_id, kwargs
        self.calls.append(("open", fence))
        return {"ok": True}

    async def append_duplex_input_async(self, session_id, *, fence=None, **kwargs):
        del session_id, kwargs
        self.calls.append(("append", fence))
        return {"ok": True}

    async def signal_duplex_turn_async(self, session_id, *, fence=None, **kwargs):
        del session_id, kwargs
        self.calls.append(("signal", fence))
        return {"ok": True}

    async def close_duplex_session_async(self, session_id, *, fence=None, **kwargs):
        del session_id, kwargs
        self.calls.append(("close", fence))
        return {"ok": True}


@pytest.mark.asyncio
async def test_async_omni_forwards_fence_for_all_runtime_operations():
    engine = FenceRecordingEngine()
    app = object.__new__(AsyncOmni)
    app.engine = engine
    fence = DuplexFence("sid", epoch=1, turn_id=2)

    await app.open_duplex_session_async("sid", fence=fence)
    await app.append_duplex_input_async(
        "sid", mode="append_audio_chunk", payload={}, fence=fence, collect_outputs=False
    )
    await app.signal_duplex_turn_async("sid", event="turn.end", fence=fence)
    await app.close_duplex_session_async("sid", reason="done", fence=fence)

    assert engine.calls == [(operation, fence) for operation in ("open", "append", "signal", "close")]


@pytest.mark.asyncio
async def test_openai_handler_passes_current_session_fence_for_runtime_controls():
    engine = FenceRecordingEngine()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid",
        config=DuplexSessionConfig(),
        epoch=1,
        turn_id=2,
    )
    ws = TimedWebSocket()
    fence = DuplexFence("sid", epoch=1, turn_id=2)

    await handler._open_runtime_session(session, ws.send_json)
    await handler._signal_runtime_session(session, "turn.end", send_json=ws.send_json)
    await handler._close_runtime_session(session, reason="done", send_json=ws.send_json)

    assert engine.calls == [(operation, fence) for operation in ("open", "signal", "close")]


@pytest.mark.asyncio
async def test_runtime_signal_uses_fence_captured_before_session_turn_advances():
    engine = FenceRecordingEngine()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(session_id="sid", config=DuplexSessionConfig())
    captured_fence = DuplexFence("sid", epoch=0, turn_id=0)
    session.turn_id = 1

    await handler._signal_runtime_session(
        session,
        "input.commit",
        fence=captured_fence,
    )

    assert engine.calls == [("signal", captured_fence)]
