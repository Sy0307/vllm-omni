# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text participates in every low-precision codebook addition, as in HF."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_prefill_does_not_reassociate_codebooks_before_text():
    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(2, 4, _weight=torch.full((2, 4), 256.0, dtype=torch.bfloat16))
    model.n_vq, model.audio_vocab_size, model.audio_pad_token_id = 12, 2, 2
    model._stacked_audio_emb_w = None
    model.audio_embeddings = torch.nn.ModuleList(
        [torch.nn.Embedding(2, 4, _weight=torch.ones(2, 4, dtype=torch.bfloat16)) for _ in range(12)]
    )
    codes = torch.zeros((1, 12), dtype=torch.long)

    _, result, _ = model.preprocess(torch.tensor([1]), None, codes={"ref": codes})

    # At 256, adding 1 rounds back to 256 in BF16 at each of the 12 steps.
    # Summing codebooks first would instead return 268.
    torch.testing.assert_close(result, torch.full((1, 4), 256.0, dtype=torch.bfloat16), rtol=0, atol=0)
