from types import SimpleNamespace

from tests.helpers import runtime
from tests.helpers.runtime import OpenAIClientHandler


class _BinaryResponse:
    content = b"fake-wav-bytes"
    response = SimpleNamespace(headers={"content-type": "audio/wav"})


class _StreamResponse:
    response = SimpleNamespace(headers={"content-type": "audio/wav"})

    def iter_bytes(self):
        yield b"fake-"
        yield b"wav-bytes"


def test_audio_speech_core_model_skips_wav_transcription(monkeypatch):
    def _fail_transcribe(_raw_bytes):
        raise AssertionError("core_model should not run Whisper transcription")

    monkeypatch.setattr(runtime, "convert_audio_bytes_to_text", _fail_transcribe)

    client = OpenAIClientHandler(run_level="core_model", log_stats=False)
    response = client._process_non_stream_audio_speech_response(
        _BinaryResponse(),
        response_format="wav",
        wall_start=0.0,
    )

    assert response.success
    assert response.audio_bytes == b"fake-wav-bytes"
    assert response.audio_content is None


def test_audio_speech_core_model_skips_stream_wav_transcription(monkeypatch):
    def _fail_transcribe(_raw_bytes):
        raise AssertionError("core_model should not run Whisper transcription")

    monkeypatch.setattr(runtime, "convert_audio_bytes_to_text", _fail_transcribe)

    client = OpenAIClientHandler(run_level="core_model", log_stats=False)
    response = client._process_stream_audio_speech_response(
        _StreamResponse(),
        response_format="wav",
        wall_start=0.0,
    )

    assert response.success
    assert response.audio_bytes == b"fake-wav-bytes"
    assert response.audio_content is None


def test_audio_speech_full_model_transcribes_wav(monkeypatch):
    calls = []

    def _transcribe(raw_bytes):
        calls.append(raw_bytes)
        return "hello"

    monkeypatch.setattr(runtime, "convert_audio_bytes_to_text", _transcribe)

    client = OpenAIClientHandler(run_level="full_model", log_stats=False)
    response = client._process_non_stream_audio_speech_response(
        _BinaryResponse(),
        response_format="wav",
        wall_start=0.0,
    )

    assert response.success
    assert response.audio_content == "hello"
    assert calls == [b"fake-wav-bytes"]
