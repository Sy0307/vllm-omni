# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-grouped paged decode attention vs vLLM's TRITON_ATTN unified_attention."""

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

BLOCK = 16
HEAD = 128


def _case(query_lens, seq_lens, num_heads, num_kv_heads, pad_reqs, pad_tokens, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    query_lens = list(query_lens) + [0] * (pad_reqs - len(query_lens))
    seq_lens = list(seq_lens) + [0] * (pad_reqs - len(seq_lens))
    cu = [0]
    for n in query_lens:
        cu.append(cu[-1] + n)
    assert cu[-1] <= pad_tokens
    blocks = [(s + BLOCK - 1) // BLOCK for s in seq_lens]
    num_blocks = sum(blocks) + 8
    perm = torch.randperm(num_blocks, device="cuda", generator=g).to(torch.int32)
    block_table = torch.zeros((pad_reqs, 64), dtype=torch.int32, device="cuda")
    start = 0
    for i, n in enumerate(blocks):
        block_table[i, :n] = perm[start : start + n]
        start += n
    # LBNHC page: [block, token, kv head, K | V] like vLLM's TRITON_ATTN cache.
    kv = torch.randn((num_blocks, BLOCK, num_kv_heads, 2 * HEAD), device="cuda", dtype=dtype, generator=g)
    q = torch.randn((pad_tokens, num_heads, HEAD), device="cuda", dtype=dtype, generator=g)
    return dict(
        q=q,
        k=kv[..., :HEAD],
        v=kv[..., HEAD:],
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        block_table=block_table,
        num_tokens=cu[-1],
    )


def _reference(c, max_seqlen_q):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    out = torch.zeros_like(c["q"])
    ones = torch.ones((len(c["seqused_k"]), c["k"].shape[2]), device="cuda")
    unified_attention(
        q=c["q"],
        k=c["k"],
        v=c["v"],
        out=out,
        cu_seqlens_q=c["cu_seqlens_q"],
        max_seqlen_q=max_seqlen_q,
        seqused_k=c["seqused_k"],
        max_seqlen_k=1024,
        softmax_scale=HEAD**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=c["block_table"],
        softcap=0,
        q_descale=None,
        k_descale=ones,
        v_descale=ones,
    )
    return out


def _ours(c, **kwargs):
    from vllm_omni.attention.triton_decode_attention import paged_attention

    out = torch.zeros_like(c["q"])
    paged_attention(
        c["q"], c["k"], c["v"], out, c["cu_seqlens_q"], c["seqused_k"], c["block_table"], HEAD**-0.5, **kwargs
    )
    return out


CASES = {
    # (query lens, seq lens, padded reqs, padded tokens)
    "decode128": ([1] * 124, [60 + 7 * i % 241 for i in range(124)], 128, 128),
    "decode_small": ([1] * 5, [1, 17, 32, 33, 300], 8, 8),
    "mixed": ([1] * 60 + [176] + [1] * 3, [60 + 5 * i % 200 for i in range(60)] + [230, 9, 64, 65], 128, 304),
    "prefill_only": ([37, 1, 130], [37, 90, 400], 4, 176),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("heads", [(32, 8), (16, 8), (8, 8), (28, 4)])
@pytest.mark.parametrize("decode_heads", [None, 0, 1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_bitwise_equal_to_unified_attention(case, heads, decode_heads, dtype):
    num_heads, num_kv_heads = heads
    if decode_heads and num_kv_heads % decode_heads:
        pytest.skip("grouping must divide the KV heads")
    query_lens, seq_lens, pad_reqs, pad_tokens = CASES[case]
    c = _case(query_lens, seq_lens, num_heads, num_kv_heads, pad_reqs, pad_tokens, dtype)
    n = c["num_tokens"]
    ref = _reference(c, max_seqlen_q=pad_tokens)
    out = _ours(c, decode_heads=decode_heads)
    assert torch.equal(out[:n], ref[:n])
    assert not out[n:].any(), "padding rows must not be written"


@torch.inference_mode()
def test_cuda_graph_replay_with_new_lengths():
    c = _case([1] * 100, [50 + i for i in range(100)], 32, 8, 128, 128, torch.bfloat16)
    out = torch.zeros_like(c["q"])
    from vllm_omni.attention.triton_decode_attention import paged_attention

    def run():
        paged_attention(c["q"], c["k"], c["v"], out, c["cu_seqlens_q"], c["seqused_k"], c["block_table"], HEAD**-0.5)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Replay with a prefill chunk and different context lengths.
    lens = [1] * 90 + [30] + [0] * 37
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    c["cu_seqlens_q"].copy_(cu)
    seq = torch.zeros(128, dtype=torch.int32, device="cuda")
    seq[:91] = torch.arange(91, device="cuda", dtype=torch.int32) * 3 + 30
    c["seqused_k"].copy_(seq)
    out.zero_()
    graph.replay()
    ref = _reference(c, max_seqlen_q=128)
    n = int(cu[-1])
    assert torch.equal(out[:n], ref[:n])


@torch.inference_mode()
def test_patch_routes_supported_calls_and_falls_back(monkeypatch):
    from vllm.v1.attention.backends import triton_attn

    from vllm_omni.attention import triton_decode_attention as tda

    monkeypatch.setattr(triton_attn, "unified_attention", triton_attn.unified_attention)
    monkeypatch.setattr(triton_attn, "triton_reshape_and_cache_flash", triton_attn.triton_reshape_and_cache_flash)
    calls = []
    real = tda.paged_attention

    def recording_attention(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(tda, "paged_attention", recording_attention)
    assert tda.install_triton_attention_patch()
    assert not tda.install_triton_attention_patch()  # idempotent
    patched = triton_attn.unified_attention

    c = _case([1] * 20 + [40], [100] * 20 + [40], 32, 8, 32, 64, torch.bfloat16)
    ones = torch.ones((32, 8), device="cuda")
    segm = dict(
        seq_threshold_3D=16,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty((16, 32, 16, HEAD), device="cuda"),
        softmax_segm_max=torch.empty((16, 32, 16), device="cuda"),
        softmax_segm_expsum=torch.empty((16, 32, 16), device="cuda"),
    )

    def call(fn, **overrides):
        out = torch.zeros_like(c["q"])
        args = dict(
            q=c["q"],
            k=c["k"],
            v=c["v"],
            out=out,
            cu_seqlens_q=c["cu_seqlens_q"],
            max_seqlen_q=64,
            seqused_k=c["seqused_k"],
            max_seqlen_k=128,
            softmax_scale=HEAD**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=c["block_table"],
            softcap=0,
            q_descale=None,
            k_descale=ones,
            v_descale=ones,
            **segm,
        )
        args.update(overrides)
        fn(**args)
        return out

    n = c["num_tokens"]
    assert torch.equal(call(patched)[:n], call(patched.__wrapped__)[:n])
    assert len(calls) == 1
    # Sliding window is not handled: upstream kernel, same result.
    assert torch.equal(call(patched, window_size=(31, 0))[:n], call(patched.__wrapped__, window_size=(31, 0))[:n])
    assert len(calls) == 1


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cache_write_matches_upstream_with_padding_and_graph_replay(monkeypatch, use_pdl, dtype):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

    from vllm_omni.attention import triton_decode_attention as tda

    if use_pdl and not tda._USE_PDL:
        pytest.skip("PDL requires Hopper or newer")
    monkeypatch.setattr(tda, "_USE_PDL", use_pdl)
    key = torch.randn(37, 8, HEAD, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    cache = torch.zeros(8, BLOCK, 8, 2 * HEAD, device="cuda", dtype=dtype)
    expected = torch.zeros_like(cache)
    slots = torch.randperm(8 * BLOCK, device="cuda")[:37]
    slots[-3:] = -1
    scale = torch.ones(1, device="cuda")

    def write():
        tda._cache_write_pdl(
            key,
            value,
            cache[..., :HEAD],
            cache[..., HEAD:],
            slots,
            "auto",
            scale,
            scale,
            upstream=triton_reshape_and_cache_flash,
        )

    write()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    for _ in range(2):
        key.normal_()
        value.normal_()
        graph.replay()
        triton_reshape_and_cache_flash(
            key, value, expected[..., :HEAD], expected[..., HEAD:], slots, "auto", scale, scale
        )
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
