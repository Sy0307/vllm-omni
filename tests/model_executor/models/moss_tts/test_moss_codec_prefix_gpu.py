# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU contracts for prefix prefill and snapshot restore (no model weights required)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.codec_prefix import flat_restore_many, flat_snapshot, prefix_attention

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]


@pytest.mark.parametrize(
    "context,capacity,lengths",
    [
        (17, 32, [0, 1, 16, 17, 31, 32, 33, 257]),
        (125, 140, [0, 14, 15, 16, 124, 125, 126, 414]),
        (400, 880, [0, 32, 479, 480, 881, 13248]),
    ],
)
@pytest.mark.parametrize("graph", [False, True])
def test_prefix_attention_and_ring_tail_match_chronological_oracle(context, capacity, lengths, graph):
    torch.manual_seed(119)
    h, d, s = 2, 64, len(lengths) + 3
    lens = torch.tensor(lengths, device="cuda", dtype=torch.long)
    cu = F.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))
    starts = cu[:-1].to(torch.long)
    slots = torch.arange(len(lengths) - 1, -1, -1, device="cuda") + 1
    qkv = torch.randn(sum(lengths), 3, h, d, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(1)
    cache = torch.randn(2, s, h, capacity, d, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()

    def run():
        return prefix_attention(q, k, v, cache, slots, starts, lens, cu, context)

    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            result = run()
        cache.copy_(before)
        capture.replay()
    else:
        result = run()
    expected_cache = before.clone()
    for i, length in enumerate(lengths):
        if length == 0:
            continue
        start = sum(lengths[:i])
        slot = int(slots[i])
        # Block the independent FP32 oracle to avoid materializing a 13k-square mask.
        for lo in range(0, length, 128):
            hi = min(lo + 128, length)
            klo = max(0, lo - context + 1)
            posq = torch.arange(lo, hi, device="cuda")
            posk = torch.arange(klo, hi, device="cuda")
            distance = posq[:, None] - posk[None, :]
            mask = (distance >= 0) & (distance < context)
            expected = F.scaled_dot_product_attention(
                q[start + lo : start + hi].transpose(0, 1).float(),
                k[start + klo : start + hi].transpose(0, 1).float(),
                v[start + klo : start + hi].transpose(0, 1).float(),
                mask,
            )
            torch.testing.assert_close(
                result[start + lo : start + hi].transpose(0, 1), expected.to(q.dtype), atol=0.016, rtol=0.025
            )
        kept = torch.arange(max(0, length - capacity), length, device="cuda")
        expected_cache[0, slot, :, kept % capacity] = k[start + kept].transpose(0, 1)
        expected_cache[1, slot, :, kept % capacity] = v[start + kept].transpose(0, 1)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    # A new query after prefill must see the correct chronological tail.
    from vllm_omni.model_executor.models.moss_tts.slot_attention import slot_ring_flash_attention_rows

    valid = [i for i, n in enumerate(lengths) if n]
    follow = torch.randn(len(valid), 3, 3, h, d, device="cuda", dtype=torch.bfloat16)
    fq, fk, fv = follow.permute(2, 0, 3, 1, 4).unbind(0)
    next_audio = slot_ring_flash_attention_rows(
        fq,
        fk,
        fv,
        cache,
        lens[valid],
        slots[valid],
        torch.full((len(valid),), 3, device="cuda", dtype=torch.long),
        context,
    )
    for row, i in enumerate(valid):
        length = lengths[i]
        start = sum(lengths[:i])
        first = max(0, length - context + 1)
        all_k = torch.cat((k[start + first : start + length].transpose(0, 1), fk[row]), dim=1)
        all_v = torch.cat((v[start + first : start + length].transpose(0, 1), fv[row]), dim=1)
        distance = (
            torch.arange(length, length + 3, device="cuda")[:, None]
            - torch.arange(first, length + 3, device="cuda")[None, :]
        )
        expected = F.scaled_dot_product_attention(
            fq[row].float(), all_k.float(), all_v.float(), (distance >= 0) & (distance < context)
        )
        torch.testing.assert_close(next_audio[row], expected.to(q.dtype), atol=0.016, rtol=0.025)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dim", [5, 64])
def test_flat_snapshot_restore_independent_slots(monkeypatch, dim):
    from vllm_omni.model_executor.models.moss_tts import codec_prefix as module

    caches = [torch.randn(2, 8, 2, n, dim, device="cuda", dtype=torch.bfloat16) for n in (17, 140, 880)]
    codec = SimpleNamespace(_decoder_slot_offsets=torch.arange(40, device="cuda").view(5, 8))
    monkeypatch.setattr(module, "state_caches", lambda _: caches)
    first = flat_snapshot(codec, 1)
    second = flat_snapshot(codec, 3)
    original = [x.clone() for x in caches]
    offsets = codec._decoder_slot_offsets.clone()
    for cache in caches:
        cache[:, 1].fill_(9)
        cache[:, 3].fill_(7)
    flat_restore_many(codec, [(0, first), (5, first), (7, second)])
    for actual, expected in zip(caches, original):
        torch.testing.assert_close(actual[:, 0], expected[:, 1], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 5], expected[:, 1], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 7], expected[:, 3], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 2], expected[:, 2], atol=0, rtol=0)
        actual[:, 0].zero_()
        torch.testing.assert_close(actual[:, 5], expected[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(codec._decoder_slot_offsets[:, 5], offsets[:, 1], atol=0, rtol=0)


@pytest.mark.parametrize("dim", [5, 64])
def test_coalesced_slot_copies_preserve_owner_and_other_requests(monkeypatch, dim):
    from vllm_omni.model_executor.models.moss_tts import codec_prefix as module

    caches = [torch.randn(2, 8, 2, n, dim, device="cuda", dtype=torch.bfloat16) for n in (17, 140, 880)]
    codec = SimpleNamespace(_decoder_slot_offsets=torch.arange(40, device="cuda").view(5, 8))
    monkeypatch.setattr(module, "state_caches", lambda _: caches)
    originals = [x.clone() for x in caches]
    offsets = codec._decoder_slot_offsets.clone()
    module.copy_prefix_slots(codec, [(1, 0), (1, 5), (3, 7)])
    for actual, expected in zip(caches, originals):
        for source, target in [(1, 0), (1, 5), (3, 7)]:
            torch.testing.assert_close(actual[:, target], expected[:, source], atol=0, rtol=0)
        for untouched in (1, 2, 3, 4, 6):
            torch.testing.assert_close(actual[:, untouched], expected[:, untouched], atol=0, rtol=0)
        actual[:, 0].zero_()
        torch.testing.assert_close(actual[:, 5], expected[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(codec._decoder_slot_offsets[:, 5], offsets[:, 1], atol=0, rtol=0)
