from __future__ import annotations

import os
from typing import Any

import torch

_LOAD_ERROR: Exception | None = None
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})

try:
    from vllm_omni import _C  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on optional extension
    _LOAD_ERROR = exc


def is_available() -> bool:
    return _LOAD_ERROR is None and hasattr(torch.ops.vllm_omni_fish_kvcache_attn, "decode")


def load_error() -> Exception | None:
    return _LOAD_ERROR


def is_fish_kvcache_attn_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_FISH_KVCACHE_ATTN", "").lower() in _ENABLED_VALUES


if is_available():

    @torch.library.register_fake("vllm_omni_fish_kvcache_attn::decode")
    def _decode_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        scale: float,
        max_seq_len: int,
    ) -> torch.Tensor:
        del q, k_cache, v_cache, block_table, seq_lens, scale, max_seq_len
        return out


def max_supported_seq_len() -> int:
    return int(os.environ.get("VLLM_OMNI_FISH_KVCACHE_ATTN_MAX_SEQ_LEN", "1024"))


def _is_sliding_window_disabled(sliding_window: Any) -> bool:
    if sliding_window is None:
        return True
    if isinstance(sliding_window, (list, tuple)) and len(sliding_window) == 2:
        return int(sliding_window[0]) == -1 and int(sliding_window[1]) == -1
    return False


def can_use_fish_kvcache_attn(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    dcp_world_size: int,
    use_cascade: bool,
    alibi_slopes: Any,
    sliding_window: Any,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> bool:
    if not is_fish_kvcache_attn_enabled():
        return False
    if not is_available():
        return False
    if max_query_len != 1 or use_cascade or dcp_world_size != 1:
        return False
    if block_table is None or alibi_slopes is not None or not _is_sliding_window_disabled(sliding_window):
        return False
    if output_scale is not None or output_block_scale is not None:
        return False
    if query.dim() != 3 or key_cache.dim() != 4 or value_cache.dim() != 4:
        return False
    if query.shape[-1] != 128 or key_cache.shape[-1] != 128:
        return False
    if key_cache.shape[1] != 16:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        return False
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return False
    if max_seq_len <= 0 or max_seq_len > max_supported_seq_len():
        return False
    if not (
        query.is_contiguous()
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and block_table.is_contiguous()
        and seq_lens.is_contiguous()
    ):
        return False
    return True


def fish_decode_kvcache_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    max_seq_len: int,
) -> torch.Tensor:
    return torch.ops.vllm_omni_fish_kvcache_attn.decode(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        float(scale),
        int(max_seq_len),
    )
