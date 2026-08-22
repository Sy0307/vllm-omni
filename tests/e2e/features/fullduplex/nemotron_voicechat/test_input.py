from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    NemotronVoiceChatPcmAppendBuffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: np.ndarray, *, sample_rate_hz: int = 16000, fmt: str = "pcm_f32le") -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(np.asarray(samples, dtype=np.float32).tobytes()).decode("ascii"),
        "format": fmt,
        "sample_rate_hz": sample_rate_hz,
    }


def _append(buffer, samples, operation_id):
    return buffer.prepare_append(
        _payload(np.asarray(samples, dtype=np.float32)),
        operation_id=operation_id,
        chunk_period_ms=80,
        allow_emit=True,
    )


def test_irregular_browser_packets_emit_exact_ordered_80ms_frame() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    assert _append(buffer, np.arange(1000), "packet-1") is None

    reservation = _append(buffer, np.arange(1000, 1280), "packet-2")

    assert reservation is not None
    assert reservation.operation_id == "packet-2"
    decoded = np.frombuffer(base64.b64decode(reservation.payload["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.arange(1280, dtype=np.float32))
    assert buffer.pending_byte_count == 0
