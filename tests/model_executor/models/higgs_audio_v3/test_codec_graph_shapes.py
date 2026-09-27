# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
    HiggsAudioV3Code2WavForConditionalGeneration as Codec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_matched_streaming_graphs_cover_every_exact_batch():
    model = SimpleNamespace(
        config=SimpleNamespace(
            codec_graph_single_max_frames=33,
            codec_graph_batch_sizes=list(range(1, 65)),
            codec_graph_frame_sizes=[8, 33],
        )
    )
    shapes = Codec._decode_graph_shapes(model)
    assert len(shapes) == 159
    assert all((b, f) in shapes for b in range(1, 65) for f in (8, 33))
    assert all((1, f) in shapes for f in range(1, 34))
    assert (2, 32) not in shapes


def test_default_graph_shapes_are_preserved():
    shapes = Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace()))
    assert len(shapes) == 162
    assert (1, 150) in shapes and (16, 54) in shapes


def test_invalid_graph_shape_rejected():
    with pytest.raises(ValueError):
        Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace(codec_graph_batch_sizes=[0])))
