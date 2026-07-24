import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEngineClient:
    def __init__(self, *, async_chunk: bool):
        self.async_chunk = async_chunk


class FakeAppState:
    def __init__(
        self,
        serving: object | None,
        *,
        async_chunk: bool,
        duplex_handler: object | None,
    ):
        self.engine_client = FakeEngineClient(async_chunk=async_chunk)
        self.openai_serving_duplex = duplex_handler
        self.openai_serving_realtime = serving


class FakeApp:
    def __init__(
        self,
        serving: object | None,
        *,
        async_chunk: bool,
        duplex_handler: object | None,
    ):
        self.state = FakeAppState(
            serving,
            async_chunk=async_chunk,
            duplex_handler=duplex_handler,
        )


class FakeWebSocket:
    def __init__(
        self,
        serving: object | None,
        *,
        async_chunk: bool = False,
        duplex_handler: object | None = None,
        query_params: dict[str, str] | None = None,
    ):
        self.app = FakeApp(
            serving,
            async_chunk=async_chunk,
            duplex_handler=duplex_handler,
        )
        self.query_params = query_params or {}
        self.accept = AsyncMock()
        self.send_json = AsyncMock()
        self.close = AsyncMock()


class FakeRealtimeConnection:
    def __init__(self, handle_connection: AsyncMock):
        self.handle_connection = handle_connection


def test_async_chunk_uses_standard_realtime_connection(monkeypatch):
    serving = object()
    websocket = FakeWebSocket(serving, async_chunk=True)
    handle_connection = AsyncMock()

    def build_connection(actual_websocket, actual_serving):
        assert actual_websocket is websocket
        assert actual_serving is serving
        return FakeRealtimeConnection(handle_connection)

    monkeypatch.setattr(api_server, "RealtimeConnection", build_connection)

    asyncio.run(api_server.realtime_websocket(websocket))

    handle_connection.assert_awaited_once_with()
    websocket.accept.assert_not_awaited()
    websocket.send_json.assert_not_awaited()
    websocket.close.assert_not_awaited()


def test_duplex_opt_in_uses_duplex_handler(monkeypatch):
    serving = object()
    duplex_handler = AsyncMock()
    websocket = FakeWebSocket(
        serving,
        async_chunk=True,
        duplex_handler=duplex_handler,
        query_params={"duplex": "true"},
    )

    def unexpected_connection(*_args):
        raise AssertionError("standard realtime connection must not be created")

    monkeypatch.setattr(api_server, "RealtimeConnection", unexpected_connection)

    asyncio.run(api_server.realtime_websocket(websocket))

    duplex_handler.handle_realtime_session.assert_awaited_once_with(websocket)


@pytest.mark.parametrize(
    ("duplex_handler", "query_params"),
    [
        (None, {"duplex": "1"}),
        (AsyncMock(), {}),
        (AsyncMock(), {"duplex": "false"}),
    ],
)
def test_without_duplex_opt_in_uses_standard_realtime_connection(
    monkeypatch,
    duplex_handler,
    query_params,
):
    serving = object()
    websocket = FakeWebSocket(
        serving,
        async_chunk=True,
        duplex_handler=duplex_handler,
        query_params=query_params,
    )
    handle_connection = AsyncMock()

    def build_connection(actual_websocket, actual_serving):
        assert actual_websocket is websocket
        assert actual_serving is serving
        return FakeRealtimeConnection(handle_connection)

    monkeypatch.setattr(api_server, "RealtimeConnection", build_connection)

    asyncio.run(api_server.realtime_websocket(websocket))

    handle_connection.assert_awaited_once_with()
    if duplex_handler is not None:
        duplex_handler.handle_realtime_session.assert_not_awaited()


def test_missing_realtime_serving_returns_unsupported(monkeypatch):
    websocket = FakeWebSocket(None, async_chunk=True)

    def unexpected_connection(*_args):
        raise AssertionError("standard realtime connection must not be created")

    monkeypatch.setattr(api_server, "RealtimeConnection", unexpected_connection)

    asyncio.run(api_server.realtime_websocket(websocket))

    websocket.accept.assert_awaited_once_with()
    websocket.send_json.assert_awaited_once_with(
        {
            "type": "error",
            "error": "Realtime API is not available",
            "code": "unsupported",
        }
    )
    websocket.close.assert_awaited_once_with()
