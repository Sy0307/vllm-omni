from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.openai.vad import (
    SileroStreamingVAD,
    SileroVADConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_silero_streaming_vad_debounces_and_latches_one_utterance():
    probabilities = iter([0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    detector = SileroStreamingVAD(
        SileroVADConfig(
            threshold=0.5,
            prefix_padding_ms=0,
            silence_duration_ms=64,
            min_speech_duration_ms=64,
        ),
        frame_scorer=lambda _: next(probabilities),
    )
    frame = np.zeros(512, dtype=np.float32)

    candidate = detector.process(frame)
    started = detector.process(frame)
    active = detector.process(frame)
    trailing_silence = detector.process(frame)
    stopped = detector.process(frame)
    next_candidate = detector.process(frame)
    next_started = detector.process(frame)

    assert candidate.is_speech is False
    assert candidate.speech_started is False
    assert started.speech_started is True
    assert started.speech_active is True
    assert active.speech_started is False
    assert active.speech_active is True
    assert trailing_silence.speech_active is True
    assert trailing_silence.speech_stopped is False
    assert stopped.speech_stopped is True
    assert stopped.speech_active is False
    assert next_candidate.speech_started is False
    assert next_started.speech_started is True


def test_silero_streaming_vad_rejects_noise_and_short_transient():
    probabilities = iter([0.1, 0.95, 0.1, 0.2, 0.1])
    detector = SileroStreamingVAD(
        SileroVADConfig(min_speech_duration_ms=64),
        frame_scorer=lambda _: next(probabilities),
    )

    results = [detector.process(np.zeros(512, dtype=np.float32)) for _ in range(5)]

    assert all(result.speech_started is False for result in results)
    assert all(result.speech_active is False for result in results)


def test_silero_streaming_vad_buffers_partial_model_windows():
    calls = 0

    def score(_: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.9

    detector = SileroStreamingVAD(
        SileroVADConfig(min_speech_duration_ms=32),
        frame_scorer=score,
    )

    partial = detector.process(np.zeros(511, dtype=np.float32))
    complete = detector.process(np.zeros(1, dtype=np.float32))

    assert calls == 1
    assert partial.speech_started is False
    assert complete.speech_started is True
