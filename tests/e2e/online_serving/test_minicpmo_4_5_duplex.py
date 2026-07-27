"""CI coverage for the MiniCPM-o 4.5 native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import websockets
from huggingface_hub import snapshot_download

from tests.e2e.online_serving.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
    run_demo,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_multi_session,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.experimental.fullduplex.client import build_realtime_url

pytestmark = pytest.mark.omni

_MODEL = "openbmb/MiniCPM-o-4_5"
_DEPLOY_CONFIG = modify_stage_config(
    get_deploy_config_path("minicpmo_4_5_duplex.yaml"),
    updates={
        "base_config": get_deploy_config_path("minicpmo_4_5.yaml"),
        "stages": {
            0: {"kv_cache_memory_bytes": 6 * 1024 * 1024 * 1024},
            1: {"kv_cache_memory_bytes": 512 * 1024 * 1024},
            2: {"kv_cache_memory_bytes": 256 * 1024 * 1024},
        },
    },
)
_CORE_DEPLOY_CONFIG = modify_stage_config(
    _DEPLOY_CONFIG,
    updates={
        "stages": {
            0: {"enforce_eager": True},
            1: {"enforce_eager": True},
        }
    },
)
_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "minicpmo_4_5"
_RESPONSE_REQUIRED_WAV = _ASSET_DIR / "response_required_16k.wav"
_RESPONSE_REQUIRED_SHA256 = "2e5fd4eb3ee434ce107ee3a0591fa624a33f7683c7462f45fe651c443c9af941"
_SOFT_INTERRUPT_WAV = _ASSET_DIR / "soft_interrupt_16k.wav"
_SOFT_INTERRUPT_SHA256 = "cadae6d0ddc510310f16f8775d6379f30a5195369ccb54ff10a8e3f9a1f4a2ea"
_REF_AUDIO_RELATIVE_PATH = Path("assets") / "HT_ref_audio.wav"

_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_DEPLOY_CONFIG,
            use_stage_cli=True,
            server_args=["--trust-remote-code"],
        ),
        id="three-stage-single-gpu",
    )
]
_CORE_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CORE_DEPLOY_CONFIG,
            use_stage_cli=True,
            server_args=["--trust-remote-code"],
        ),
        id="three-stage-single-gpu",
    )
]


def _validated_wav(path: Path, expected_sha256: str) -> Path:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"MiniCPM-o duplex fixture SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with wave.open(str(path), "rb") as wav_file:
        actual_format = (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getcomptype(),
        )
    expected_format = (16_000, 1, 2, "NONE")
    if actual_format != expected_format:
        raise AssertionError(f"MiniCPM-o duplex fixture must be 16 kHz mono PCM16, got {actual_format}")
    return path


def _validated_input_wav() -> Path:
    return _validated_wav(_RESPONSE_REQUIRED_WAV, _RESPONSE_REQUIRED_SHA256)


def _validated_soft_interrupt_wav() -> Path:
    return _validated_wav(_SOFT_INTERRUPT_WAV, _SOFT_INTERRUPT_SHA256)


def _resolve_ref_audio(model_prefix: str) -> Path:
    if model_prefix:
        model_root = Path(model_prefix) / _MODEL
    else:
        model_root = Path(snapshot_download(_MODEL, local_files_only=True))
    ref_audio = model_root / _REF_AUDIO_RELATIVE_PATH
    if not ref_audio.is_file():
        raise FileNotFoundError(f"MiniCPM-o checkpoint ref audio is missing: {ref_audio}")
    return ref_audio


def _realtime_url(omni_server) -> str:
    return f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1"


async def _receive_protocol_events(ws, required_types: set[str], *, timeout_s: float) -> list[dict[str, object]]:
    async def receive() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        seen: set[str] = set()
        while not required_types.issubset(seen):
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            event_type = event.get("type")
            if event_type == "error":
                raise AssertionError(f"WebSocket protocol smoke received an error: {event}")
            if isinstance(event_type, str):
                seen.add(event_type)
        return events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _run_protocol_smoke(*, url: str, model: str, ref_audio: Path) -> list[dict[str, object]]:
    session_id = f"duplex-ci-protocol-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {"minicpmo45_native_duplex": True},
                    },
                }
            )
        )
        events = await _receive_protocol_events(
            ws,
            {"session.created", "session.updated"},
            timeout_s=60,
        )
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(await _receive_protocol_events(ws, {"session.closed"}, timeout_s=60))
    return events


def _demo_args(*, omni_server, input_wav: Path, ref_audio: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        url=_realtime_url(omni_server),
        model=omni_server.model,
        session_id=f"duplex-ci-single-{uuid.uuid4().hex}",
        input_wav=str(input_wav),
        ref_audio=str(ref_audio),
        turn_input_wav=[],
        output_dir=str(output_dir),
        output_audio_format="pcm16",
        chunk_ms=200,
        realtime_input=True,
        first_turn_ms=1400,
        turn_duration_ms=[],
        first_turn_transcript="duplex CI speech",
        omit_transcript_hints=False,
        validation_mode="response-required",
        temperature=0.0,
        scenario="sequential",
        require_audio=True,
        require_distinct_inputs=False,
        expect_empty_turn=[],
        short_ack_ms=350,
        turns=1,
        timeout_s=120.0,
        model_policy_settle_ms=2000,
    )


def _multi_session_args(
    *,
    omni_server,
    input_wav: Path,
    ref_audio: Path,
    output_dir: Path,
    response_required: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        url=_realtime_url(omni_server),
        model=omni_server.model,
        sessions=2,
        input_wav=str(input_wav),
        ref_audio=str(ref_audio),
        session_input_wav=[],
        session_expected_token=[],
        turn_input_wav=[],
        output_dir=str(output_dir),
        realtime_input=True,
        chunk_ms=200,
        turns=1,
        first_turn_ms=1400,
        turn_duration_ms=[],
        response_required=response_required,
        temperature=0.0 if response_required else None,
        disconnect_session_index=0,
        takeover_session_index=1,
        resume_after_ms=100,
        expire_session_index=None,
        expire_after_s=40.0,
        verify_admission_limit=None,
        model_policy_settle_ms=2000,
        timeout_s=180.0,
    )


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _CORE_SERVER_PARAMS, indirect=True)
def test_duplex_websocket_protocol_smoke(omni_server, model_prefix: str) -> None:
    ref_audio = _resolve_ref_audio(model_prefix)
    events = asyncio.run(
        _run_protocol_smoke(
            url=_realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
        )
    )
    event_types = [event.get("type") for event in events]
    assert "session.created" in event_types
    assert "session.updated" in event_types
    assert event_types[-1] == "session.closed"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_duplex_single_session_response_required(omni_server, model_prefix: str, tmp_path: Path) -> None:
    result = asyncio.run(
        run_demo(
            _demo_args(
                omni_server=omni_server,
                input_wav=_validated_input_wav(),
                ref_audio=_resolve_ref_audio(model_prefix),
                output_dir=tmp_path / "single_session",
            )
        )
    )
    assert result["ok"] is True
    assert result["audio_delta_count"] > 0
    assert result["done_count"] == 1
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_duplex_two_sessions_resume_and_takeover(omni_server, model_prefix: str, tmp_path: Path) -> None:
    result = asyncio.run(
        run_multi_session(
            _multi_session_args(
                omni_server=omni_server,
                input_wav=_validated_input_wav(),
                ref_audio=_resolve_ref_audio(model_prefix),
                output_dir=tmp_path / "multi_session",
                response_required=True,
            )
        )
    )
    assert result["ok"] is True
    assert result["session_count"] == 2
    assert result["resume"]["ok"] is True
    assert result["takeover"]["ok"] is True
    assert not result["failures"]
    assert all(session["audio_delta_count"] > 0 for session in result["sessions"])
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
