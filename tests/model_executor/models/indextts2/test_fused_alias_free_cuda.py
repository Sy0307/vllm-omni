# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm_omni.model_executor.models.common.alias_free_activation import (
    AliasFreeActivation1d,
    OfficialFusedAliasFreeActivation1d,
)
from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta

pytestmark = pytest.mark.core_model


def test_official_fused_alias_free_falls_back_on_cpu():
    torch.manual_seed(37)
    activation = SnakeBeta(3, alpha_logscale=True)
    eager = AliasFreeActivation1d(copy.deepcopy(activation)).eval()
    fused = OfficialFusedAliasFreeActivation1d(copy.deepcopy(activation)).eval()
    hidden = torch.randn(1, 3, 257)

    with torch.inference_mode():
        expected = eager(hidden)
        actual = fused(hidden)

    torch.testing.assert_close(actual, expected)
    assert fused.fused_activation_active is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_official_fused_alias_free_extension_matches_eager():
    torch.manual_seed(131)
    device = torch.device("cuda")
    activation = SnakeBeta(3, alpha_logscale=True)
    eager = AliasFreeActivation1d(copy.deepcopy(activation)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    fused = OfficialFusedAliasFreeActivation1d(copy.deepcopy(activation)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, 3, 4103, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = eager(hidden)
        actual = fused(hidden)

    error = (actual.float() - expected.float()).abs()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert fused.fused_activation_loaded is True
    assert fused.fused_activation_active is True
    assert error.max().item() < 0.08
    assert error.mean().item() < 0.01
    assert relative_l2.item() < 0.03
    assert cosine.item() > 0.999
