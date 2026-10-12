# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Share the KV blocks of a constant prompt head across requests (MOSS_SHARED_HEAD=1).

The serving layer salts every MOSS request's prefix-cache hash with its full conditioning, so even the
constant chat-template head (identical tokens, pad audio codes, same positions -> identical KV) is stored
once per request. A salt of the form ``h<N>|<salt>`` moves the salt from block 0 to the block containing
token N: blocks wholly inside the head hash by tokens only (shared by all requests), every later block
chains from the salted one.
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("MOSS_SHARED_HEAD", "0") == "1"


def head_salt(salt: str | None, prompt_ids: list[int], codes, head: list[int], pad_id: int) -> str | None:
    """Prefix ``salt`` with the head length when the prompt provably starts with the constant head."""
    n = len(head)
    if not salt or n == 0 or len(prompt_ids) <= n or prompt_ids[:n] != head:
        return salt
    if codes is None or codes.shape[0] < n or not bool((codes[:n] == pad_id).all()):
        return salt
    return f"h{n}|{salt}"


def install() -> bool:
    if not enabled():
        return False
    from vllm.v1.core import block_pool, kv_cache_utils

    orig = kv_cache_utils.generate_block_hash_extra_keys
    if getattr(orig, "_moss_shared_head", False):
        return True

    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        salt = getattr(request, "cache_salt", None)
        if not salt or not salt.startswith("h") or "|" not in salt[:12]:
            return orig(request, start_token_idx, end_token_idx, start_mm_idx)
        head = int(salt[1 : salt.index("|")])
        request.cache_salt = None
        try:
            keys, nxt = orig(request, start_token_idx, end_token_idx, start_mm_idx)
        finally:
            request.cache_salt = salt
        if start_token_idx <= head < end_token_idx:
            keys = (keys or ()) + (("cache_salt", salt),)
        return keys, nxt

    generate_block_hash_extra_keys._moss_shared_head = True
    from vllm.logger import init_logger
    from vllm.v1.core import kv_cache_manager

    log = init_logger(__name__)
    get_computed = kv_cache_manager.KVCacheManager.get_computed_blocks
    stats = {"n": 0, "head": 0, "tokens": 0}

    def get_computed_blocks(self, request):
        res = get_computed(self, request)
        num = res[1]
        salt = getattr(request, "cache_salt", None) or ""
        if salt.startswith("h"):
            stats["n"] += 1
            stats["tokens"] += num
            stats["head"] += num >= int(salt[1 : salt.index("|")]) // 16 * 16
            if stats["n"] % 1000 == 0:
                log.info(
                    "MOSS shared head: %d requests, %.1f%% hit the shared head, mean cached %.1f tokens",
                    stats["n"],
                    100.0 * stats["head"] / stats["n"],
                    stats["tokens"] / stats["n"],
                )
        return res

    kv_cache_manager.KVCacheManager.get_computed_blocks = get_computed_blocks
    log.info("MOSS shared head: prefix-cache salt starts after the constant prompt head")
    kv_cache_utils.generate_block_hash_extra_keys = generate_block_hash_extra_keys
    block_pool.generate_block_hash_extra_keys = generate_block_hash_extra_keys
    return True
