# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check MOSImage kernels against the eager rounding contract."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.mosimage.ops import _MATCHED_REDUCTION, try_rms_norm, try_rotate_qk, try_situ

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA required"),
]


@pytest.mark.parametrize("shape,sliced", [((2, 257, 3072), False), ((2, 257, 12, 128), True)])
@torch.inference_mode()
def test_rms_norm_rounding(shape, sliced):
    torch.manual_seed(23)
    if sliced:
        raw = torch.randn(*shape[:-2], 3 * shape[-2] * shape[-1], device="cuda", dtype=torch.bfloat16)
        x = raw.chunk(3, -1)[1].unflatten(-1, shape[-2:])
    else:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = y.to(x.dtype) * weight
    actual = try_rms_norm(x, weight)
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 16, 24, 128), (2, 257, 12, 128)])
@torch.inference_mode()
def test_rotary_rounding(shape):
    torch.manual_seed(17)
    q, k = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    rope = torch.randn(1, shape[1] + 3, 128, device="cuda", dtype=torch.bfloat16)[:, 3:]
    sin, cos = rope.chunk(2, -1)
    sin, cos = [torch.cat((t, t), -1).unsqueeze(-2) for t in (sin, cos)]
    expected = []
    for x in (q, k):
        a, b = x.chunk(2, -1)
        expected.append(x * cos + torch.cat((-b, a), -1) * sin)
    actual = try_rotate_qk(q, k, rope)
    assert actual is not None
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("batch,sequence,heads", [(1, 6620, 24), (2, 257, 12), (1, 17, 3)])
@pytest.mark.skipif(not _MATCHED_REDUCTION, reason="Fused Q/K reduction requires PyTorch 2.13")
@torch.inference_mode()
def test_qk_norm_rope_rounding(batch, sequence, heads):
    from vllm_omni.diffusion.models.mosimage.ops import try_qk_norm_rope

    torch.manual_seed(57)
    packed = torch.randn(batch, sequence, 3 * heads * 128, device="cuda", dtype=torch.bfloat16) * 3
    q, k, _ = [t.unflatten(-1, (heads, 128)) for t in packed.chunk(3, -1)]
    qw, kw = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    expected = try_rotate_qk(try_rms_norm(q, qw), try_rms_norm(k, kw), rope)
    actual = try_qk_norm_rope(q, k, qw, kw, rope)
    assert actual is not None
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    for x, w, a in zip((q, k), (qw, kw), actual):
        xf = x.float()
        norm = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w
        first, second = norm.chunk(2, -1)
        eager = norm * cos + torch.cat((-second, first), -1) * sin
        torch.testing.assert_close(a, eager, atol=0, rtol=0)


@pytest.mark.parametrize("batch,sequence", [(1, 6620), (2, 257)])
@torch.inference_mode()
def test_gated_residual_rounding(batch, sequence):
    from vllm_omni.diffusion.models.mosimage.ops import gated_residual

    torch.manual_seed(63)
    x, y = torch.randn(2, batch, sequence, 3072, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch, 6, 3072, device="cuda", dtype=torch.bfloat16)[:, 2]
    torch.testing.assert_close(gated_residual(x, y, gate), x + y * gate[:, None], atol=0, rtol=0)


def test_training_fallback():
    with torch.enable_grad():
        x = torch.ones(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        rope = torch.ones(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        assert try_rms_norm(x, weight) is None
        assert try_rotate_qk(x, x, rope) is None


@pytest.mark.parametrize("batch,sequence", [(1, 6620), (2, 257), (1, 17)])
@torch.inference_mode()
def test_rms_modulation_rounding(batch, sequence):
    torch.manual_seed(37)
    x = torch.randn(batch, sequence, 3072, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(batch, 6, 3072, device="cuda", dtype=torch.bfloat16)
    shift, scale = modulation[:, 0], modulation[:, 1]
    # Reference is the preceding round's norm followed by eager AdaLN.
    norm = try_rms_norm(x, weight)
    assert norm is not None
    expected = norm * (1 + scale[:, None]) + shift[:, None]
    actual = try_rms_norm(x, weight, scale, shift)
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 74, 92])
@torch.inference_mode()
def test_rms_packed_rows_eager(seed):
    torch.manual_seed(seed)
    x = torch.randn(2, 257, 3072, device="cuda", dtype=torch.bfloat16) * 3
    weight = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 2, 3072, device="cuda", dtype=torch.bfloat16).unbind(1)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
    expected = expected * (1 + scale[:, None]) + shift[:, None]
    actual = try_rms_norm(x, weight, scale, shift)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_situ_all_finite_bf16_values():
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)].reshape(1, -1, 1)
    for hidden, gate in [(values, values.flip(1)), (values.flip(1), values)]:
        expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
        actual = try_situ(hidden, gate)
        assert actual is not None
        # Libdevice/Triton and eager differ at near-zero underflow boundaries.
        # Bound that absolute error explicitly; require exact ordinary outputs.
        torch.testing.assert_close(actual, expected, atol=2e-36, rtol=0)
        normal = expected.abs() > 1e-30
        torch.testing.assert_close(actual[normal], expected[normal], atol=0, rtol=0)


@torch.inference_mode()
def test_situ_strided_ffn():
    torch.manual_seed(41)
    up = torch.randn(2, 257, 16384, device="cuda", dtype=torch.bfloat16) * 5
    hidden, gate = up.chunk(2, -1)
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
    actual = try_situ(hidden, gate)
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
