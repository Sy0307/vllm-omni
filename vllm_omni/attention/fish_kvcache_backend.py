from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.attention.fish_kvcache_attn import (
    can_use_fish_kvcache_attn,
    fish_decode_kvcache_attn,
    is_available,
    is_fish_kvcache_attn_enabled,
    load_error,
    max_supported_seq_len,
)

logger = init_logger(__name__)

_FIRST_HIT_LOGGED = False
_FIRST_MISS_LOGGED = False


def _fish_kvcache_enabled() -> bool:
    return is_fish_kvcache_attn_enabled()


def _effective_max_seq_len(attn_metadata: Any) -> int:
    max_seq_len = int(attn_metadata.max_seq_len)
    if int(attn_metadata.max_query_len) == 1:
        return min(max_seq_len, max_supported_seq_len())
    return max_seq_len


def _forward_with_fish_kvcache(
    impl: Any,
    original_forward: Callable[..., Any],
    layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    output: Any,
    output_scale: Any = None,
    output_block_scale: Any = None,
) -> Any:
    global _FIRST_HIT_LOGGED, _FIRST_MISS_LOGGED

    if attn_metadata is not None and not attn_metadata.use_cascade:
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        effective_max_seq_len = _effective_max_seq_len(attn_metadata)
        can_use = can_use_fish_kvcache_attn(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=effective_max_seq_len,
            dcp_world_size=impl.dcp_world_size,
            use_cascade=attn_metadata.use_cascade,
            alibi_slopes=impl.alibi_slopes,
            sliding_window=impl.sliding_window,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        if can_use:
            if not _FIRST_HIT_LOGGED:
                _FIRST_HIT_LOGGED = True
                logger.info(
                    "Fish decode-only kvcache attention fast path hit: "
                    "query_shape=%s key_cache_shape=%s block_table_shape=%s "
                    "seq_lens_shape=%s max_seq_len=%s",
                    tuple(q.shape),
                    tuple(key_cache.shape),
                    tuple(attn_metadata.block_table.shape),
                    tuple(attn_metadata.seq_lens.shape),
                    effective_max_seq_len,
                )
            fish_decode_kvcache_attn(
                q,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                out,
                scale=float(impl.scale),
                max_seq_len=effective_max_seq_len,
            )
            return output

        if not _FIRST_MISS_LOGGED:
            _FIRST_MISS_LOGGED = True
            logger.info(
                "Fish decode-only kvcache attention fast path miss: "
                "query_shape=%s query_dtype=%s key_cache_shape=%s key_dtype=%s "
                "block_table_shape=%s block_dtype=%s seq_lens_shape=%s seq_dtype=%s "
                "max_query_len=%s max_seq_len=%s dcp_world_size=%s use_cascade=%s",
                tuple(q.shape),
                q.dtype,
                tuple(key_cache.shape),
                key_cache.dtype,
                tuple(attn_metadata.block_table.shape) if attn_metadata.block_table is not None else None,
                getattr(attn_metadata.block_table, "dtype", None),
                tuple(attn_metadata.seq_lens.shape),
                attn_metadata.seq_lens.dtype,
                attn_metadata.max_query_len,
                effective_max_seq_len,
                impl.dcp_world_size,
                attn_metadata.use_cascade,
            )

    return original_forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def install_fish_kvcache_attn_backend(model: Any) -> int:
    """Install the Fish kvcache fast path on this Fish SlowAR model only."""
    if not _fish_kvcache_enabled():
        return 0
    if not is_available():
        logger.warning(
            "VLLM_OMNI_FISH_KVCACHE_ATTN=1 but native extension is unavailable: %r",
            load_error(),
        )
        return 0

    installed = 0
    for layer in getattr(model, "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        attention_layer = getattr(self_attn, "attn", None)
        impl = getattr(attention_layer, "impl", None)
        if impl is None or getattr(impl, "_fish_kvcache_attn_installed", False):
            continue

        original_forward = impl.forward

        def fish_forward(
            this_impl: Any,
            layer: Any,
            query: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            attn_metadata: Any,
            output: Any,
            output_scale: Any = None,
            output_block_scale: Any = None,
            *,
            _original_forward: Callable[..., Any] = original_forward,
        ) -> Any:
            return _forward_with_fish_kvcache(
                this_impl,
                _original_forward,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        impl.forward = types.MethodType(fish_forward, impl)
        impl._fish_kvcache_attn_installed = True
        installed += 1

    if installed:
        logger.info("Installed Fish decode-only kvcache attention backend on %d attention layers", installed)
    return installed
