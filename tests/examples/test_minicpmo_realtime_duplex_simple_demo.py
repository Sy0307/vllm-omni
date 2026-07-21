import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEMO_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_duplex_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "minicpmo_realtime_duplex_simple_demo_test",
        DEMO_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_post_commit_decision_ignores_streaming_listens_before_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "session.created"},
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert commit_index == 3
    assert demo._post_commit_model_decision(events, commit_index) is None


def test_post_commit_decision_accepts_listen_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "listen"


def test_post_commit_decision_accepts_drained_speak_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-a"}},
        {"type": "response.audio.delta", "response_id": "resp-a", "delta": "AAAA"},
        {"type": "response.done", "response_id": "resp-a"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "speak"


def test_post_commit_decision_ignores_cancelled_response():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"status": "cancelled"}},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) is None


def test_commit_ack_search_starts_after_stream_cursor():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._input_committed_index(events, after_index=1) == 2


def test_exact_model_unit_boundary_does_not_require_an_extra_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert not demo._has_residual_model_unit(b"\0" * (unit_bytes * 6), chunk_period_ms=1000)


def test_partial_model_unit_requires_a_post_commit_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert demo._has_residual_model_unit(b"\0" * (unit_bytes * 6 + 512), chunk_period_ms=1000)


def test_latest_streaming_decision_is_used_at_exact_boundary():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._latest_model_decision(events, after_index=0) == "listen"


def test_open_streaming_response_requires_post_commit_drain():
    demo = _load_demo_module()

    assert demo._response_in_progress([{"type": "response.created"}])
    assert not demo._response_in_progress(
        [
            {"type": "response.created"},
            {"type": "response.done"},
        ]
    )
