"""Static host for the MiniCPM-o 4.5 continuous duplex voice-chat web UI.

A minimal FastAPI + uvicorn app that serves index.html and the two AudioWorklet
JS files. It also proxies the Realtime WebSocket by default, so proxied notebook
or studio environments only need to expose this one UI port.

The page derives the WS host from its own origin unless ``--ws-base`` injects an
explicit base. Same-origin mode forwards ``/v1/realtime`` to ``--ws-backend``.
Gradio is intentionally avoided because it interferes with AudioWorklet
registration.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("realtime_web")

STATIC_DIR = Path(__file__).parent / "static"


def _join_ws_url(base: str, path: str, query: str) -> str:
    return base.rstrip("/") + path + (("?" + query) if query else "")


async def _pump_client_to_backend(client: WebSocket, backend) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                with contextlib.suppress(Exception):
                    await backend.close()
                return
            if "text" in message and message["text"] is not None:
                await backend.send(message["text"])
            elif "bytes" in message and message["bytes"] is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(client: WebSocket, backend) -> None:
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


def _is_expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def build_app(ws_base: str = "", ws_backend: str = "ws://127.0.0.1:8099") -> FastAPI:
    """Build the FastAPI app serving the static UI.

    Args:
        ws_base: Optional explicit duplex WS base (e.g. ``ws://10.137.72.57:8099``).
            Empty string -> the client uses this same origin.
        ws_backend: Backend WS base used by the same-origin proxy.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="MiniCPM-o Realtime Web")
    index_path = STATIC_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("__DUPLEX_WS_BASE__", ws_base)
        return HTMLResponse(html)

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(content="ok", media_type="text/plain")

    @app.websocket("/v1/realtime")
    async def realtime_proxy(websocket: WebSocket) -> None:
        await websocket.accept()
        backend_url = _join_ws_url(ws_backend, "/v1/realtime", urlencode(websocket.query_params.multi_items()))
        logger.info("Proxying realtime websocket to %s", backend_url)
        try:
            async with websockets.connect(backend_url, max_size=64 * 1024 * 1024) as backend:
                client_to_backend = asyncio.create_task(_pump_client_to_backend(websocket, backend))
                backend_to_client = asyncio.create_task(_pump_backend_to_client(websocket, backend))
                done, pending = await asyncio.wait(
                    {client_to_backend, backend_to_client},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, BaseException) and not _is_expected_proxy_close(result):
                        raise result
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            return
        except Exception:
            logger.exception("Realtime websocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    # serve worklet JS + app.js under /static
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o realtime web UI host")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--ws-base",
        default="",
        help="Explicit duplex WS base, e.g. ws://10.137.72.57:8099. "
        "Empty -> client uses this same origin and the built-in WS proxy.",
    )
    parser.add_argument(
        "--ws-backend",
        default="ws://127.0.0.1:8099",
        help="Backend duplex WS base used by the built-in same-origin proxy.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Serving realtime web UI on %s:%d (ws_base=%r, ws_backend=%r)",
        args.host,
        args.port,
        args.ws_base,
        args.ws_backend,
    )
    app = build_app(ws_base=args.ws_base, ws_backend=args.ws_backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
