# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.worker.gpu_generation_worker import _make_compilation_times


def test_make_compilation_times_matches_vllm_020_shape():
    result = _make_compilation_times(0.0)

    assert result.language_model == 0.0
    assert result.encoder == 0.0


if __name__ == "__main__":
    test_make_compilation_times_matches_vllm_020_shape()


def test_make_compilation_times_filters_to_current_fields():
    result = _make_compilation_times(1.5, speculative_model=2.0)

    assert result.language_model == 1.5
    assert result.encoder == 0.0
