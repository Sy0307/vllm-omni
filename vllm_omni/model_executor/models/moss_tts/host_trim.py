# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Host-path trims for the single-stage MOSS pipeline (each behind its own env flag).

MOSS_TRIM_HASH=1: prefix-cache hashes cover prompt blocks only. Generated audio tokens are sampled and never
appear in a later prompt, so their blocks cannot produce a prefix hit; hashing them (one sha256 per full block,
plus a hasher call per request per step) is pure host cost. Request.update_block_hashes stops at the first block
that extends past the prompt once outputs exist, and the coordinators cache at most the hashed blocks.
"""

from __future__ import annotations

import os

_HASH_BLOCK = 0


def flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def install_hash_trim() -> bool:
    if not flag("MOSS_TRIM_HASH"):
        return False
    from vllm.v1.core import kv_cache_coordinator
    from vllm.v1.request import Request

    if getattr(Request.update_block_hashes, "_moss_trim", False):
        return True
    update = Request.update_block_hashes

    def update_block_hashes(self) -> None:
        if self._block_hasher is None:
            return
        block = _HASH_BLOCK
        if block and self._output_token_ids and (len(self.block_hashes) + 1) * block > self.num_prompt_tokens:
            return
        update(self)

    update_block_hashes._moss_trim = True
    Request.update_block_hashes = update_block_hashes

    def wrap(cls) -> None:
        cache = cls.__dict__.get("cache_blocks")
        if cache is None or getattr(cache, "_moss_trim", False):
            return

        def cache_blocks(self, request, num_computed_tokens):
            global _HASH_BLOCK
            block = getattr(self, "hash_block_size", None) or self.scheduler_block_size
            if not _HASH_BLOCK:
                _HASH_BLOCK = block
            if request._output_token_ids:
                num_computed_tokens = min(num_computed_tokens, len(request.block_hashes) * block)
            return cache(self, request, num_computed_tokens)

        cache_blocks._moss_trim = True
        cls.cache_blocks = cache_blocks

    for name in ("KVCacheCoordinator", "HybridKVCacheCoordinator"):
        cls = getattr(kv_cache_coordinator, name, None)
        if cls is not None:
            wrap(cls)
    from vllm.logger import init_logger

    init_logger(__name__).info("MOSS host trim: prefix-cache hashes cover prompt blocks only")
    return True
