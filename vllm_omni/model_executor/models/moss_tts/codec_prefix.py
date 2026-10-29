# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in MOSS codec prefix prefill and bounded, model-local state reuse."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from . import audio_tokenizer_v2 as av
from .packed_codec import CUDAGraphPackedDecoderWrapper

if TYPE_CHECKING:
    from .modeling_moss_tts_codec import MossTTSCodecDecoder, _MossCodecStreamSession

_DecodeItem = tuple[int, str, torch.Tensor, bool]
_PrefixKey = tuple[str, int]


@triton.jit
def _store_prefix_tail(
    k,
    v,
    cache,
    slots,
    starts,
    lengths,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    states: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    dim: tl.constexpr,
    block: tl.constexpr,
):
    segment = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2) * block + tl.arange(0, block)
    length = tl.load(lengths + segment)
    slot = tl.load(slots + segment)
    start = tl.load(starts + segment)
    # Map each physical ring column to its last nonnegative prefix position.
    pos = length - 1 - ((length - 1 - row) % capacity)
    mask = (row < capacity) & (pos >= 0) & (pos < length)
    columns = tl.arange(0, dim)
    kval = tl.load(k + (start + pos[:, None]) * ks0 + head * ks1 + columns[None, :], mask[:, None], 0)
    vval = tl.load(v + (start + pos[:, None]) * vs0 + head * vs1 + columns[None, :], mask[:, None], 0)
    dst = ((slot * heads + head) * capacity + row[:, None]) * dim + columns[None, :]
    tl.store(cache + dst, kval, mask[:, None])
    tl.store(cache + states * heads * capacity * dim + dst, vval, mask[:, None])


@torch.library.custom_op("vllm_omni::moss_codec_prefix_attention", mutates_args=("cache",))
def prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    tok_starts: torch.Tensor,
    lengths: torch.Tensor,
    cu: torch.Tensor,
    context: int,
) -> torch.Tensor:
    """All queries start at position zero; no history is read from the ring."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    n, h, dim = q.shape
    x = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=n,
        cu_seqlens_q=cu,
        max_seqlen_k=n,
        cu_seqlens_k=cu,
        causal=True,
        window_size=[context - 1, 0],
        fa_version=3,
    )
    _store_prefix_tail[(lengths.shape[0], h, triton.cdiv(cache.shape[3], 32))](
        k,
        v,
        cache,
        slots,
        tok_starts,
        lengths,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cache.shape[1],
        h,
        cache.shape[3],
        dim,
        32,
        num_warps=4,
    )
    return x


def state_caches(codec: av.MossAudioTokenizerModel) -> list[torch.Tensor]:
    return [
        layer.self_attn._streaming_state.kv_cache.cache
        for stage in codec.decoder
        if isinstance(stage, av.MossAudioTokenizerProjectedTransformer)
        for layer in stage.transformer.layers
    ]


@dataclass
class FlatSnapshot:
    bank: torch.Tensor
    offsets: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.bank.numel() * self.bank.element_size() + self.offsets.numel() * self.offsets.element_size()


@triton.jit
def _copy_state_bank(
    layout,
    banks,
    sources,
    destinations,
    dtype_hint,
    mode: tl.constexpr,
    block: tl.constexpr,
    aligned: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cache = tl.load(layout + tile * 4).to(tl.pointer_type(dtype_hint.dtype.element_ty))
    size = tl.load(layout + tile * 4 + 1)
    offset = tl.load(layout + tile * 4 + 2)
    start = tl.load(layout + tile * 4 + 3)
    if aligned:
        cache = tl.multiple_of(cache, 16)
        size = tl.multiple_of(size, 16)
        offset = tl.multiple_of(offset, 16)
    idx = start + tl.arange(0, block)
    if mode == 0 or mode == 2:
        source = tl.load(sources + row)
        value = tl.load(cache + source * size + idx, idx < size, 0)
    if mode == 0 or mode == 1:
        bank = tl.load(banks + row).to(tl.pointer_type(dtype_hint.dtype.element_ty))
        if aligned:
            bank = tl.multiple_of(bank, 16)
    if mode == 1:
        value = tl.load(bank + offset + idx, idx < size, 0)
    if mode == 0:
        tl.store(bank + offset + idx, value, idx < size)
    else:
        destination = tl.load(destinations + row)
        tl.store(cache + destination * size + idx, value, idx < size)


class _StateBankLayout:
    """One immutable descriptor table for this decoder state pool's lifetime."""

    def __init__(self, codec: av.MossAudioTokenizerModel) -> None:
        self.caches = state_caches(codec)
        self.sizes = [c[:, 0].numel() for c in self.caches]
        self.total = sum(self.sizes)
        self.offsets = codec._decoder_slot_offsets
        self.metadata = None
        self.block = 2048
        self.aligned = all(size % 32 == 0 for size in self.sizes)
        if self.offsets.device.type == "cuda":
            rows, offset = [], 0
            dtype = self.caches[0].dtype
            for cache, size in zip(self.caches, self.sizes, strict=True):
                if not cache.is_contiguous() or cache.dtype != dtype:
                    raise ValueError("Prefix state banks require contiguous caches with one dtype")
                # Tile K and V separately. Every launched CTA does useful work,
                # and contiguous address math needs no per-element division.
                half = size // 2
                for kv in range(2):
                    pointer = cache.data_ptr() + kv * cache.shape[1] * half * cache.element_size()
                    for start in range(0, half, self.block):
                        rows.append([pointer, half, offset + kv * half, start])
                offset += size
            self.metadata = torch.tensor(rows, device=self.offsets.device, dtype=torch.int64)

    def copy(
        self,
        *,
        banks: Sequence[torch.Tensor] = (),
        sources: Sequence[int] = (),
        destinations: Sequence[int] = (),
        mode: Literal[0, 1, 2],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.offsets.device
        count = len(sources) if mode != 1 else len(destinations)
        # One owned pinned allocation: no pageable H2D synchronization between
        # large state copies. The host allocator tracks the async copy before
        # recycling storage; never mutate or manually reuse this slab in flight.
        host = torch.empty((3, count), dtype=torch.long, pin_memory=True)
        view = host.numpy()
        view.fill(0)
        if banks:
            view[0] = [bank.data_ptr() for bank in banks]
        if sources:
            view[1] = sources
        if destinations:
            view[2] = destinations
        pointers, src, dst = host.to(device, non_blocking=True).unbind(0)
        _copy_state_bank[(count, self.metadata.shape[0])](
            self.metadata, pointers, src, dst, self.caches[0], mode, self.block, self.aligned, num_warps=4
        )
        return src, dst


def _state_layout(codec: av.MossAudioTokenizerModel) -> _StateBankLayout:
    layout = getattr(codec, "_prefix_state_layout", None)
    if layout is None or layout.offsets is not codec._decoder_slot_offsets:
        layout = _StateBankLayout(codec)
        codec._prefix_state_layout = layout
    return layout


def flat_snapshot(codec: av.MossAudioTokenizerModel, slot: int) -> FlatSnapshot:
    layout = _state_layout(codec)
    bank = torch.empty(layout.total, device=layout.offsets.device, dtype=layout.caches[0].dtype)
    if layout.metadata is not None:
        layout.copy(banks=[bank], sources=[slot], mode=0)
    else:
        offset = 0
        for cache, size in zip(layout.caches, layout.sizes, strict=True):
            bank[offset : offset + size].view(cache[:, slot].shape).copy_(cache[:, slot])
            offset += size
    return FlatSnapshot(bank, layout.offsets[:, slot].clone())


def flat_restore_many(codec: av.MossAudioTokenizerModel, entries: Sequence[tuple[int, FlatSnapshot]]) -> None:
    """Restore every layer and request in one launch; each slot owns its state."""
    if not entries:
        return
    layout = _state_layout(codec)
    for _, saved in entries:
        saved.bank.record_stream(torch.accelerator.current_stream())
        saved.offsets.record_stream(torch.accelerator.current_stream())
    _, slots = layout.copy(
        banks=[saved.bank for _, saved in entries], destinations=[slot for slot, _ in entries], mode=1
    )
    layout.offsets.index_copy_(1, slots, torch.stack([saved.offsets for _, saved in entries], dim=1))


def copy_prefix_slots(codec: av.MossAudioTokenizerModel, entries: Sequence[tuple[int, int]]) -> None:
    """Copy prefix owners to disjoint, independent follower slots in one launch."""
    if not entries:
        return
    layout = _state_layout(codec)
    sources, destinations = layout.copy(
        sources=[src for src, _ in entries], destinations=[dst for _, dst in entries], mode=2
    )
    layout.offsets.index_copy_(1, destinations, layout.offsets.index_select(1, sources))


@prefix_attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    tok_starts: torch.Tensor,
    lengths: torch.Tensor,
    cu: torch.Tensor,
    context: int,
) -> torch.Tensor:
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


class PrefixStateCache:
    """Bounded GPU snapshots, owned by one loaded codec and its fixed ring layout."""

    def __init__(self, codec: av.MossAudioTokenizerModel, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("codec_prefix_cache_max_bytes must be nonnegative")
        self.codec = codec
        self.max_bytes = max_bytes
        self.entries: OrderedDict[_PrefixKey, FlatSnapshot] = OrderedDict()
        self.bytes = 0
        self.stats = Counter()
        self.entry_bytes = sum(c[:, 0].numel() * c.element_size() for c in state_caches(codec))
        self.entry_bytes += codec._decoder_slot_offsets[:, 0].numel() * codec._decoder_slot_offsets.element_size()

    def get(self, key: _PrefixKey) -> FlatSnapshot | None:
        entry = self.entries.get(key)
        self.stats["hits" if entry is not None else "misses"] += 1
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def put(self, key: _PrefixKey, slot: int) -> bool:
        if self.entry_bytes > self.max_bytes:
            return False
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        while self.bytes + self.entry_bytes > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.nbytes
            self.stats["evictions"] += 1
        entry = flat_snapshot(self.codec, slot)
        self.entries[key] = entry
        self.bytes += entry.nbytes
        self.stats["insertions"] += 1
        return True

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0


class CodecPrefixAccelerator:
    """Prepare only fresh request slots; target audio remains on the streaming path."""

    def __init__(
        self,
        codec: av.MossAudioTokenizerModel,
        session: _MossCodecStreamSession,
        config: VllmConfig,
        *,
        prefill: bool,
        cache_bytes: int,
    ) -> None:
        self.codec = codec
        self.session = session
        self.cache = PrefixStateCache(codec, cache_bytes)
        self.prefill: CUDAGraphPackedDecoderWrapper | None = None
        self.stats = Counter()
        stream = session._packed_wrapper
        if stream is None or not session._ring_headroom:
            raise ValueError("Codec prefix acceleration requires packed decode with ring headroom")
        if prefill:
            capability = current_platform.get_device_capability()
            if not current_platform.is_cuda() or capability is None or capability.major != 9:
                raise ValueError("Codec prefix prefill currently requires a Hopper CUDA GPU (FA3)")
            sizes = [60, 120, 240, 480, 960, 1920]
            self.prefill = CUDAGraphPackedDecoderWrapper(
                codec,
                frame_sizes=sizes,
                max_segments=stream.max_segments,
                scratch_base=stream.scratch_base,
                num_quantizers=stream.num_quantizers,
                vllm_config=config,
                prefix_prefill=True,
            )
            self.prefill.warmup(session._device)
            session.reset_slots(list(range(session._total_state_capacity)))
        # Compile state-copy kernels before requests, using only scratch slots.
        # Otherwise the first cache hit pays seconds of Triton JIT compilation.
        scratch = stream.scratch_base
        if cache_bytes >= self.cache.entry_bytes:
            warm_snapshot = flat_snapshot(codec, scratch)
            flat_restore_many(codec, [(scratch + 1, warm_snapshot)])
        copy_prefix_slots(codec, [(scratch, scratch + 1)])
        torch.accelerator.synchronize(session._device)
        session.reset_slots(list(range(scratch, session._total_state_capacity)))

    def _prefill_fresh(self, items: list[tuple[int, torch.Tensor]]) -> None:
        assert self.prefill is not None
        # First-fit decreasing avoids repeatedly padding half-empty prefix graphs.
        bins: list[tuple[list[tuple[int, torch.Tensor]], int]] = []
        for slot, codes in sorted(items, key=lambda item: item[1].shape[1], reverse=True):
            frames = int(codes.shape[1])
            for index, (batch, total) in enumerate(bins):
                if total + frames <= self.prefill.max_frames and len(batch) + 1 < self.prefill.max_segments:
                    batch.append((slot, codes))
                    bins[index] = (batch, total + frames)
                    break
            else:
                bins.append(([(slot, codes)], frames))
        for batch, _ in bins:
            self.prefill.decode([slot for slot, _ in batch], [codes for _, codes in batch])
            self.stats["prefill_graphs"] += 1

    def prepare(
        self,
        decoder: MossTTSCodecDecoder,
        items: Sequence[_DecodeItem],
        prefix_trims: Mapping[int, int],
        prefix_keys: Mapping[int, str],
    ) -> tuple[list[_DecodeItem], dict[int, int]]:
        """Return remaining codes/trims after real prefix work, never placeholder PCM."""
        prepared = []
        trims = dict(prefix_trims)
        restores = []
        fast = []
        ordinary = []
        admissions = []
        owners = {}
        followers = []
        # Bound cold-batch snapshot work as well as retained cache memory.
        admission_limit = self.cache.max_bytes // self.cache.entry_bytes
        spf = int(self.codec.downsample_rate)
        step = decoder._stream_max_step_frames
        # Validate the whole call before acquiring any slots or enqueueing writes.
        fresh = set()
        for index, request_id, codes, _ in items:
            prefix = int(prefix_trims.get(index, 0)) // spf
            if prefix // step:
                if prefix > codes.shape[1] or request_id in decoder._stream_req_slots or request_id in fresh:
                    raise ValueError("Codec prefix may only initialize a fresh request with complete reference codes")
                fresh.add(request_id)
        if len(fresh) > len(self.session._free_stream_slots):
            raise RuntimeError("MOSS codec state capacity exhausted during prefix preparation")
        for index, request_id, codes, finished in items:
            prefix = int(prefix_trims.get(index, 0)) // spf
            # Keep the reference/target boundary on the ordinary chunk grid.
            # The reference remainder is decoded with the first target chunk.
            boundary = prefix // step * step
            if not boundary:
                prepared.append((index, request_id, codes, finished))
                continue
            key = (prefix_keys[index], boundary) if index in prefix_keys else None
            owner = owners.get(key) if key is not None else None
            saved = self.cache.get(key) if key is not None and owner is None and self.cache.max_bytes else None
            slot = self.session.acquire()
            if slot is None:
                raise RuntimeError("MOSS codec state capacity exhausted during prefix preparation")
            decoder._stream_req_slots[request_id] = slot
            if owner is not None:
                followers.append((owner, slot))
                self.stats["coalesced_requests"] += 1
            elif saved is not None:
                restores.append((slot, saved))
            else:
                prefix_codes = codes[:, :boundary]
                if self.prefill is not None and boundary <= self.prefill.max_frames:
                    fast.append((slot, prefix_codes))
                else:
                    ordinary.append((index, request_id, prefix_codes, False))
                if key is not None and len(admissions) < admission_limit:
                    admissions.append((key, slot))
            if key is not None and owner is None:
                owners[key] = slot
            prepared.append((index, request_id, codes[:, boundary:], finished))
            trims[index] = (prefix - boundary) * spf
        if restores:
            flat_restore_many(self.codec, restores)
        if fast:
            self._prefill_fresh(fast)
        if ordinary:
            for start in range(0, max(item[2].shape[1] for item in ordinary), step):
                wave = [
                    (i, req, codes[:, start : start + step], False)
                    for i, req, codes, _ in ordinary
                    if start < codes.shape[1]
                ]
                decoder._decode_streaming_batch(wave)
            self.stats["ordinary_prefix_requests"] += len(ordinary)
        if followers:
            copy_prefix_slots(self.codec, followers)
        for key, slot in admissions:
            self.cache.put(key, slot)
        return prepared, trims
