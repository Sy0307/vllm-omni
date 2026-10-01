# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_batch_preprocess_projects_new_non_streaming_texts_once():
    torch.manual_seed(0)
    text_embedding = torch.nn.Embedding(64, 4)
    projection = torch.nn.Linear(4, 4)
    projected_shapes: list[tuple[int, ...]] = []

    def text_projection(x: torch.Tensor) -> torch.Tensor:
        projected_shapes.append(tuple(x.shape))
        return projection(x)

    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._text_embedding = text_embedding
    builder._text_projection = text_projection
    builder._batched_text_embeds = {}

    ids_a = list(range(1, 13))
    ids_b = list(range(20, 31))
    # Serving stores the assistant-template ids wrapped in a one-element list.
    serving_ids = [ids_a]
    buf: dict[str, dict[str, Any]] = {
        "a": {"req_id": "a", "task_type": ["CustomVoice"], PRECOMPUTED_TEXT_IDS_KEY: [ids_a]},
        "b": {"req_id": "b", "task_type": ["VoiceDesign"], PRECOMPUTED_TEXT_IDS_KEY: [torch.tensor(ids_b)]},
        "streaming": {
            "req_id": "streaming",
            "task_type": ["CustomVoice"],
            "non_streaming_mode": [False],
            PRECOMPUTED_TEXT_IDS_KEY: serving_ids,
        },
        "base": {"req_id": "base", "task_type": ["Base"], PRECOMPUTED_TEXT_IDS_KEY: serving_ids},
        "built": {
            "req_id": "built",
            "task_type": ["CustomVoice"],
            "embed": {"prefill": torch.zeros(1)},
            PRECOMPUTED_TEXT_IDS_KEY: serving_ids,
        },
    }

    builder.preprocess_infos_batch(req_infos=list(buf.values()), device=torch.device("cpu"))

    # One embedding + projection over both requests' text tokens (template stripped).
    assert projected_shapes == [(1, (len(ids_a) - 8) + (len(ids_b) - 8), 4)]
    assert set(builder._batched_text_embeds) == {"a", "b"}
    with torch.no_grad():
        for req_id, ids in (("a", ids_a), ("b", ids_b)):
            expected = projection(text_embedding(torch.tensor([ids[3:-5]])))
            torch.testing.assert_close(builder._batched_text_embeds[req_id], expected)
            assert buf[req_id][PRECOMPUTED_TEXT_IDS_KEY].tolist() == [ids]
    for skipped in ("streaming", "base", "built"):
        assert buf[skipped][PRECOMPUTED_TEXT_IDS_KEY] is serving_ids
