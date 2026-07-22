import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples" / "online_serving" / "minicpmo" / "realtime_web"
APP_ROOT = ROOT / "app"
STATIC_ROOT = APP_ROOT / "static"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_page_exposes_focused_call_conversation_and_log_surfaces():
    html = (APP_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="callButton"' in html
    assert 'id="muteButton"' in html
    assert 'id="connectionState"' in html
    assert 'id="modelState"' in html
    assert 'id="conversation"' in html
    assert 'id="eventLog"' in html
    assert "<details" in html
    assert "Automatic barge-in" not in html
    assert "Server VAD" not in html


def test_client_uses_proxy_relative_realtime_url_and_model_policy_session():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "new URL(config.realtimePath, window.location.href)" in source
    assert "url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'" in source
    assert "url.searchParams.set('autostart', '0')" in source
    assert "url.searchParams.set('minicpmo45_native_duplex', '1')" in source
    assert "auto_response: true" in source
    assert "input_audio_buffer.append" in source
    assert "input_audio_buffer.commit" not in source
    assert "playback.ack" in source
    assert "event.event || event" in source
    assert "response.audio.delta" in source
    assert "response.audio_transcript.delta" in source
    assert "conversation.item.input_audio_transcription" in source
    assert "force_barge_in" not in source
    assert "server_vad" not in source
    assert "type: 'response.create'" not in source
    assert 'type: "response.create"' not in source


def test_web_server_requires_ref_audio_for_audio_output_session():
    source = (ROOT / "server.py").read_text(encoding="utf-8")

    ref_audio_arg = re.search(
        r"parser\.add_argument\(\s*\"--ref-audio\",(?P<body>.*?)\n\s*\)",
        source,
        re.DOTALL,
    )
    assert ref_audio_arg is not None
    assert "required=True" in ref_audio_arg.group("body")
    assert "Optional reference voice" not in ref_audio_arg.group("body")


def test_client_sends_ref_audio_in_realtime_session_contract():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "if (config.refAudio) session.ref_audio = config.refAudio;" in source
    assert "extraBody.ref_audio" not in source


def test_client_has_transactional_cleanup_and_visible_event_logging():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "async function stopSession" in source
    assert "type: 'session.close'" in source
    assert "waitForSessionClosed" in source
    assert "SESSION_CLOSE_TIMEOUT_MS" in source
    assert "case 'session.closed':" in source
    assert "track.stop()" in source
    assert "clearInterval(sendTimer)" in source
    assert "appendEventLog(event)" in source
    assert "stopSession({ terminal: false })" in source


def test_client_keeps_microphone_upload_active_during_assistant_playback():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    upload_gate = re.search(r"function microphoneUploadEnabled\(\) \{(?P<body>.*?)\n  \}", source, re.DOTALL)
    begin_assistant = re.search(r"function beginAssistant\(responseId\) \{(?P<body>.*?)\n  \}", source, re.DOTALL)

    assert upload_gate is not None
    assert "return running && !muted;" in upload_gate.group("body")
    assert "assistantActive" not in upload_gate.group("body")
    assert begin_assistant is not None
    assert "pendingCapture = []" not in begin_assistant.group("body")


def test_audio_worklets_define_capture_and_playback_processors():
    capture = (STATIC_ROOT / "pcm_worklet.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "registerProcessor('fullduplex-pcm-capture'" in capture
    assert "Int16Array" in capture
    assert "registerProcessor('fullduplex-pcm-playback'" in playback
    assert "playback-drained" in playback
    assert "clear" in playback


def test_playback_worklet_buffers_first_200ms_and_reports_underruns():
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "INITIAL_PLAYBACK_BUFFER_MS = 200" in app
    assert "initialBufferMs" in app
    assert "responseId" in app
    assert "playback-underrun" in app
    assert "underrunMs" in app
    assert "initialBufferFrames" in playback
    assert "playback-underrun" in playback
    assert "underrunFrames" in playback
    assert "underrunMs" in playback
