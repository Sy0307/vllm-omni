from fastapi.testclient import TestClient

from vllm_omni.experimental.fullduplex.web.server import _join_ws_url, build_app


def test_join_ws_url_preserves_realtime_query():
    assert (
        _join_ws_url(
            "ws://127.0.0.1:8099/",
            "/v1/realtime",
            "duplex=1&model=openbmb%2FMiniCPM-o-4_5",
        )
        == "ws://127.0.0.1:8099/v1/realtime?duplex=1&model=openbmb%2FMiniCPM-o-4_5"
    )


def test_build_app_serves_health_and_injected_client_config():
    app = build_app(
        ws_backend="ws://127.0.0.1:9001",
        model="local/MiniCPM-o-4_5",
    )
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.text == "ok"

    index = client.get("/")
    assert index.status_code == 200
    assert '"model": "local/MiniCPM-o-4_5"' in index.text
    assert '"realtimePath": "v1/realtime"' in index.text
    assert "__FULL_DUPLEX_CONFIG__" not in index.text


def test_build_app_exposes_realtime_websocket_and_static_assets():
    app = build_app()
    paths = {route.path for route in app.routes}

    assert "/v1/realtime" in paths
    assert "/static" in paths

    client = TestClient(app)
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/pcm_worklet.js").status_code == 200
    assert client.get("/static/playback_worklet.js").status_code == 200
