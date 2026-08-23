from __future__ import annotations

import binascii
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pybase64 as base64

SILERO_VAD_MIN_THRESHOLD = 0.15


class ServerVADUnavailableError(RuntimeError):
    """Raised when the optional server VAD runtime is not installed."""


@dataclass(frozen=True, slots=True)
class SileroVADConfig:
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 96

    @classmethod
    def from_turn_detection(cls, turn_detection: dict[str, object]) -> SileroVADConfig:
        defaults = cls()

        def number(name: str, default: int | float) -> int | float:
            value = turn_detection.get(name, default)
            return value if isinstance(value, int | float) and not isinstance(value, bool) else default

        return cls(
            threshold=float(number("threshold", defaults.threshold)),
            prefix_padding_ms=max(0, int(number("prefix_padding_ms", defaults.prefix_padding_ms))),
            silence_duration_ms=max(0, int(number("silence_duration_ms", defaults.silence_duration_ms))),
            min_speech_duration_ms=max(
                32,
                int(number("min_speech_duration_ms", defaults.min_speech_duration_ms)),
            ),
        )


@dataclass(frozen=True, slots=True)
class StreamingVADResult:
    is_speech: bool
    speech_active: bool
    speech_started: bool = False
    speech_stopped: bool = False
    speech_probability: float = 0.0
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None

    def as_hint(self) -> dict[str, object]:
        hint: dict[str, object] = {
            "backend": "silero",
            "is_speech": self.is_speech,
            "speech_active": self.speech_active,
            "speech_started": self.speech_started,
            "speech_stopped": self.speech_stopped,
            "speech_probability": self.speech_probability,
        }
        if self.speech_start_ms is not None:
            hint["speech_start_ms"] = self.speech_start_ms
        if self.speech_end_ms is not None:
            hint["speech_end_ms"] = self.speech_end_ms
        return hint


class SileroStreamingVAD:
    """Per-session streaming Silero VAD with onset debounce and utterance latch.

    Silero's model is stateful, so an instance must never be shared across
    Realtime sessions. Audio is normalized to 16 kHz and scored in the model's
    512-sample (32 ms) windows. Loading is lazy so deployments which use the
    default model-owned mode do not need the optional ``silero-vad`` package.
    """

    _SAMPLE_RATE_HZ = 16_000
    _WINDOW_SAMPLES = 512
    _NEGATIVE_THRESHOLD_GAP = SILERO_VAD_MIN_THRESHOLD

    def __init__(
        self,
        config: SileroVADConfig,
        *,
        frame_scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config
        self._frame_scorer = frame_scorer
        self._model: object | None = None
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample: int | None = None
        self._silence_samples = 0
        self._processed_samples = 0

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample = None
        self._silence_samples = 0
        self._processed_samples = 0
        reset_states = getattr(self._model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def process_base64(
        self,
        audio: object,
        *,
        fmt: object,
        sample_rate_hz: object,
    ) -> StreamingVADResult:
        if fmt != "pcm_f32le" or not isinstance(audio, str):
            raise ValueError("Silero server VAD requires decoded pcm_f32le audio")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Silero server VAD received invalid base64 audio") from exc
        if len(raw) % 4:
            raise ValueError("Silero server VAD received an incomplete pcm_f32le frame")
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        rate = int(sample_rate_hz) if isinstance(sample_rate_hz, int | float) else self._SAMPLE_RATE_HZ
        if rate <= 0:
            raise ValueError("Silero server VAD requires a positive sample rate")
        if rate != self._SAMPLE_RATE_HZ and samples.size > 1:
            target_size = max(1, int(round(samples.size * self._SAMPLE_RATE_HZ / rate)))
            source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
            target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
            samples = np.interp(target_x, source_x, samples).astype(np.float32)
        return self.process(samples)

    def process(self, samples: np.ndarray) -> StreamingVADResult:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))

        started = False
        stopped = False
        start_ms: int | None = None
        end_ms: int | None = None
        max_probability = 0.0
        contained_speech = self._speech_active
        min_speech_samples = int(self.config.min_speech_duration_ms * self._SAMPLE_RATE_HZ / 1000)
        min_silence_samples = int(self.config.silence_duration_ms * self._SAMPLE_RATE_HZ / 1000)
        negative_threshold = max(0.0, self.config.threshold - self._NEGATIVE_THRESHOLD_GAP)

        while self._pending.size >= self._WINDOW_SAMPLES:
            frame = self._pending[: self._WINDOW_SAMPLES]
            self._pending = self._pending[self._WINDOW_SAMPLES :]
            frame_start = self._processed_samples
            self._processed_samples += self._WINDOW_SAMPLES
            probability = min(1.0, max(0.0, float(self._score_frame(frame))))
            max_probability = max(max_probability, probability)

            if self._speech_active:
                contained_speech = True
                if probability < negative_threshold:
                    self._silence_samples += self._WINDOW_SAMPLES
                    if self._silence_samples >= min_silence_samples:
                        speech_end_sample = self._processed_samples - self._silence_samples
                        self._speech_active = False
                        self._silence_samples = 0
                        stopped = True
                        end_ms = max(0, int(round(speech_end_sample * 1000 / self._SAMPLE_RATE_HZ)))
                else:
                    self._silence_samples = 0
                continue

            if probability >= self.config.threshold:
                if self._candidate_samples == 0:
                    self._candidate_start_sample = frame_start
                self._candidate_samples += self._WINDOW_SAMPLES
                if self._candidate_samples >= min_speech_samples:
                    candidate_start = self._candidate_start_sample or 0
                    prefix_samples = int(self.config.prefix_padding_ms * self._SAMPLE_RATE_HZ / 1000)
                    self._speech_active = True
                    self._candidate_samples = 0
                    self._candidate_start_sample = None
                    self._silence_samples = 0
                    contained_speech = True
                    started = True
                    start_ms = max(
                        0,
                        int(round((candidate_start - prefix_samples) * 1000 / self._SAMPLE_RATE_HZ)),
                    )
            else:
                self._candidate_samples = 0
                self._candidate_start_sample = None

        return StreamingVADResult(
            is_speech=contained_speech,
            speech_active=self._speech_active,
            speech_started=started,
            speech_stopped=stopped,
            speech_probability=max_probability,
            speech_start_ms=start_ms,
            speech_end_ms=end_ms,
        )

    def _score_frame(self, frame: np.ndarray) -> float:
        if self._frame_scorer is not None:
            return float(self._frame_scorer(frame))
        if self._model is None:
            try:
                import torch
                from silero_vad import load_silero_vad
            except ImportError as exc:
                raise ServerVADUnavailableError(
                    "server_vad requires the optional 'silero-vad' package; install vllm-omni[server-vad]"
                ) from exc
            self._model = load_silero_vad(onnx=False)
            self._frame_scorer = lambda value: float(
                self._model(torch.from_numpy(np.ascontiguousarray(value)), self._SAMPLE_RATE_HZ).item()
            )
        return float(self._frame_scorer(frame))
