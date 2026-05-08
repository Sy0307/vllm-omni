# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_thinker_to_talker_latent_adds_token_embeddings():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    embeds = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    latent = model._build_thinker_to_talker_latent(hidden, embeds, input_ids=None)

    assert torch.equal(latent, hidden + embeds)


def test_build_talker_decode_reply_cache_appends_text_eos_pad():
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    model.embed_text_eos_token = torch.tensor([[100.0, 101.0]])
    model.embed_text_pad_token = torch.tensor([[200.0, 201.0]])
    thinker_result = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    reply = model._build_talker_decode_reply_cache(
        thinker_result,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        reply,
        torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 3.0],
                [100.0, 101.0],
                [200.0, 201.0],
            ]
        ),
    )
