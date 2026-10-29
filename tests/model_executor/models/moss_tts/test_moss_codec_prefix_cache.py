# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""State ownership, bounded admission and invalid fresh-slot contracts."""

from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig, SchedulerConfig, VllmConfig

from vllm_omni.model_executor.models.moss_tts import codec_prefix as cp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prefix_compile_domain_does_not_inherit_talker_range_or_cache(monkeypatch):
    from vllm_omni.model_executor.models.moss_tts import packed_codec as pc

    compilation = CompilationConfig(
        compile_ranges_endpoints=[512],
        compile_sizes=[1, 64],
        cache_dir="ordinary",
    )
    config = VllmConfig(
        compilation_config=compilation,
        scheduler_config=SchedulerConfig(max_model_len=32768, max_num_batched_tokens=512, is_encoder_decoder=False),
    )
    compilation.local_cache_dir = "ordinary/rank0"
    compilation.static_forward_context = {"talker": object()}
    compilation.traced_files = {"talker.py"}
    captured = []
    monkeypatch.setattr(pc, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(pc, "_PrefixDecodeCompileAdapter", lambda codec, **kwargs: captured.append(kwargs))
    codec = SimpleNamespace(number_channels=2, enable_channel_interleave=True, downsample_rate=3840)
    pc.CUDAGraphPackedDecoderWrapper(
        codec,
        frame_sizes=[60, 1920],
        max_segments=65,
        scratch_base=64,
        num_quantizers=12,
        vllm_config=config,
        prefix_prefill=True,
    )
    own = captured[0]["vllm_config"]
    assert own.compilation_config.compile_ranges_endpoints == [1920]
    assert own.scheduler_config.max_num_batched_tokens == 1920
    assert own.compilation_config.compile_sizes == []
    assert own.compilation_config.cache_dir == own.compilation_config.local_cache_dir == ""
    assert compilation.compile_ranges_endpoints == [512]
    assert compilation.cache_dir == "ordinary"
    assert config.scheduler_config.max_num_batched_tokens == 512


@pytest.fixture
def state(monkeypatch):
    banks = [torch.arange(2 * 8 * n).view(2, 8, 1, n, 1).float() for n in (5, 17)]
    codec = SimpleNamespace(_decoder_slot_offsets=torch.arange(24).view(3, 8), downsample_rate=4)
    monkeypatch.setattr(cp, "state_caches", lambda _: banks)
    return codec, banks


def test_lru_bytes_and_owned_snapshot(state):
    codec, banks = state
    cache = cp.PrefixStateCache(codec, 0)
    assert not cache.put(("disabled", 15), 0)
    cache.max_bytes = cache.entry_bytes * 2
    assert cache.put(("a", 15), 0)
    saved = cache.get(("a", 15))
    original = saved.bank.clone()
    for bank in banks:
        bank[:, 0].zero_()
    torch.testing.assert_close(saved.bank, original, atol=0, rtol=0)
    assert cache.put(("b", 15), 1)
    assert cache.get(("a", 15)) is saved
    assert cache.put(("c", 15), 2)
    assert list(cache.entries) == [("a", 15), ("c", 15)]
    assert cache.get(("b", 15)) is None
    assert cache.get(("a", 30)) is None
    assert cache.get(("changed-reference", 15)) is None
    assert cache.bytes == 2 * cache.entry_bytes <= cache.max_bytes
    cache.clear()
    assert cache.bytes == 0 and not cache.entries


def make_accelerator(codec):
    free = list(range(8))
    session = SimpleNamespace(_free_stream_slots=free, acquire=lambda: free.pop(0))
    decoder = SimpleNamespace(_stream_max_step_frames=15, _stream_req_slots={})
    accelerator = cp.CodecPrefixAccelerator.__new__(cp.CodecPrefixAccelerator)
    accelerator.codec = codec
    accelerator.session = session
    accelerator.cache = cp.PrefixStateCache(codec, 0)
    accelerator.prefill = None
    accelerator.stats = Counter()
    return accelerator, decoder


def test_admission_cap_and_reference_residual(state):
    codec, _ = state
    acc, decoder = make_accelerator(codec)
    acc.cache.max_bytes = acc.cache.entry_bytes * 2
    decoded = []
    decoder._decode_streaming_batch = lambda items: decoded.extend(items)
    codes = torch.arange(2 * 36).view(2, 36)
    items = [(i, str(i), codes, False) for i in range(4)]
    prepared, trims = acc.prepare(decoder, items, dict.fromkeys(range(4), 33 * 4), {i: str(i) for i in range(4)})
    assert len(decoded) == 8 and all(x[2].shape[1] == 15 for x in decoded)
    assert acc.cache.stats["insertions"] == 2
    assert acc.cache.stats["evictions"] == 0
    assert len(acc.cache.entries) == 2
    assert all(value == 3 * 4 for value in trims.values())
    for item in prepared:
        torch.testing.assert_close(item[2], codes[:, 30:], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["active", "short", "duplicate", "capacity"])
def test_invalid_batch_does_not_acquire_slots(state, failure):
    codec, _ = state
    acc, decoder = make_accelerator(codec)
    codes = torch.ones(2, 33)
    items = [(0, "valid", codes, False), (1, "invalid", codes, False)]
    if failure == "active":
        decoder._stream_req_slots["invalid"] = 7
    elif failure == "short":
        items[1] = (1, "invalid", codes[:, :2], False)
    elif failure == "duplicate":
        items[1] = (1, "valid", codes, False)
    else:
        acc.session._free_stream_slots[:] = [0]
    before = dict(decoder._stream_req_slots)
    available = list(acc.session._free_stream_slots)
    with pytest.raises((ValueError, RuntimeError)):
        acc.prepare(decoder, items, {0: 120, 1: 120}, {})
    assert before == decoder._stream_req_slots
    assert available == acc.session._free_stream_slots


def test_negative_capacity_rejected(state):
    with pytest.raises(ValueError, match="nonnegative"):
        cp.PrefixStateCache(state[0], -1)


def test_same_batch_reference_is_prefilled_once(state, monkeypatch):
    codec, _ = state
    acc, decoder = make_accelerator(codec)
    decoded, copies = [], []
    decoder._decode_streaming_batch = lambda items: decoded.extend(items)
    monkeypatch.setattr(cp, "copy_prefix_slots", lambda codec, entries: copies.extend(entries))
    codes = torch.ones(2, 33)
    items = [(i, str(i), codes, False) for i in range(4)]
    _, trims = acc.prepare(decoder, items, dict.fromkeys(range(4), 30 * 4), dict.fromkeys(range(4), "same"))
    assert len(decoded) == 2
    assert copies == [(0, 1), (0, 2), (0, 3)]
    assert acc.stats["coalesced_requests"] == 3
    assert set(trims.values()) == {0}
    assert len(set(decoder._stream_req_slots.values())) == 4


@pytest.mark.parametrize("length", [1, 14, 15, 16, 33])
def test_aligned_prefix_retains_reference_remainder_and_target_frames(state, length):
    codec, _ = state
    acc, decoder = make_accelerator(codec)
    calls = []
    decoder._decode_streaming_batch = lambda items: calls.extend(items)
    codes = torch.arange(2 * (length + 3)).view(2, length + 3)
    items, trims = acc.prepare(decoder, [(0, "a", codes, False)], {0: length * 4}, {0: "key"})
    boundary = length // 15 * 15
    if boundary:
        torch.testing.assert_close(torch.cat([item[2] for item in calls], dim=1), codes[:, :boundary])
    else:
        assert not calls
    torch.testing.assert_close(items[0][2], codes[:, boundary:])
    assert trims == {0: (length - boundary) * 4}
    assert all(item[2].shape[1] <= 15 for item in calls)
