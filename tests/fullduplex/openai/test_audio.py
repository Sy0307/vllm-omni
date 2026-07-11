# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np

from vllm_omni.experimental.fullduplex.openai.audio import convert_input_audio_with_rate


def test_pcm16_input_conversion_is_pure_and_reports_target_rate():
    pcm16 = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    encoded = base64.b64encode(pcm16).decode("ascii")

    converted, fmt, sample_rate = convert_input_audio_with_rate(
        encoded,
        "pcm16",
        sample_rate_hz=16_000,
    )

    assert encoded == base64.b64encode(pcm16).decode("ascii")
    assert fmt == "pcm_f32le"
    assert sample_rate == 16_000
    samples = np.frombuffer(base64.b64decode(converted), dtype="<f4")
    assert np.allclose(samples, [-1.0, 0.0, 32767 / 32768])
