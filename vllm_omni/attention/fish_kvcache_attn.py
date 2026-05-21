from __future__ import annotations

import os
from typing import Any

import torch

_LOAD_ERROR: Exception | None = None
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_SMALL_PATH_MAX_SEQ_LEN = 1024
_WORKSPACE_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

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


def is_fish_kvcache_attn_required() -> bool:
    return os.environ.get("VLLM_OMNI_FISH_KVCACHE_ATTN_REQUIRED", "").lower() in _ENABLED_VALUES


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
        partial_m: torch.Tensor,
        partial_l: torch.Tensor,
        partial_acc: torch.Tensor,
    ) -> torch.Tensor:
        del q, k_cache, v_cache, block_table, seq_lens, scale, max_seq_len
        del partial_m, partial_l, partial_acc
        return out


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
    if block_table.dim() != 2 or seq_lens.dim() != 1:
        return False
    if block_table.shape[0] != query.shape[0] or seq_lens.shape[0] != query.shape[0]:
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
    if max_seq_len <= 0:
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


def _workspace_cache_key(
    query: torch.Tensor,
    num_splits: int,
) -> tuple[Any, ...]:
    return (
        query.device.type,
        query.device.index,
        num_splits,
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[2]),
    )


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _raise_workspace_capture_miss() -> None:
    raise RuntimeError("Fish kvcache attention workspace was not prewarmed before CUDA graph capture")


def _get_decode_workspace(
    query: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_seq_len <= _SMALL_PATH_MAX_SEQ_LEN:
        key = (query.device.type, query.device.index, "empty")
        workspace = _WORKSPACE_CACHE.get(key)
        if workspace is None:
            if _is_cuda_graph_capturing():
                _raise_workspace_capture_miss()
            workspace = (
                torch.empty((0,), device=query.device, dtype=torch.float32),
                torch.empty((0,), device=query.device, dtype=torch.float32),
                torch.empty((0,), device=query.device, dtype=torch.float32),
            )
            _WORKSPACE_CACHE[key] = workspace
        return workspace

    num_splits = (int(max_seq_len) + _SMALL_PATH_MAX_SEQ_LEN - 1) // _SMALL_PATH_MAX_SEQ_LEN
    key = _workspace_cache_key(query, num_splits)
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        if _is_cuda_graph_capturing():
            _raise_workspace_capture_miss()
        total_rows = int(query.shape[0]) * int(query.shape[1])
        head_dim = int(query.shape[2])
        workspace = (
            torch.empty((num_splits, total_rows), device=query.device, dtype=torch.float32),
            torch.empty((num_splits, total_rows), device=query.device, dtype=torch.float32),
            torch.empty((num_splits, total_rows, head_dim), device=query.device, dtype=torch.float32),
        )
        _WORKSPACE_CACHE[key] = workspace
    return workspace


def prewarm_fish_kvcache_attn_workspace(
    query: torch.Tensor,
    max_seq_len: int,
) -> None:
    _get_decode_workspace(query, int(max_seq_len))


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
    partial_m, partial_l, partial_acc = _get_decode_workspace(query, int(max_seq_len))
    return torch.ops.vllm_omni_fish_kvcache_attn.decode(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        float(scale),
        int(max_seq_len),
        partial_m,
        partial_l,
        partial_acc,
    )
