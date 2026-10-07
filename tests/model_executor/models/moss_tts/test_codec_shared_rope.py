# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import apply_rope, rope_angles

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shared_rope_table_matches_inline_rotation(dtype):
    torch.manual_seed(7)
    batch, heads, frames, dim = 3, 4, 15, 64
    q = torch.randn(batch, heads, frames, dim, dtype=dtype)
    k = torch.randn(batch, heads, frames, dim, dtype=dtype)
    offset = torch.tensor([0, 37, 5017])

    expected = apply_rope(q, k, offset, 10_000.0)
    table = rope_angles(offset, frames, dim, 10_000.0)
    actual = apply_rope(q, k, offset, 10_000.0, table=table)

    assert table[0].shape == (batch, 1, frames, dim // 2)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
