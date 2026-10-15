# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Share the KV blocks of a constant prompt head across requests (MOSS_SHARED_HEAD=1).

The serving layer salts every MOSS request's prefix-cache hash with its full conditioning, so even the
constant chat-template head (identical tokens, pad audio codes, same positions -> identical KV) is stored
once per request. Validated MOSS sampling metadata moves the salt from block 0 to the block containing
the end of the head: blocks wholly inside the head hash by tokens only, every later block chains from
the salted one. Public cache salts remain opaque strings.
"""

from __future__ import annotations

import os

HEAD_TOKENS_KEY = "_moss_shared_head_tokens"


def enabled() -> bool:
    return os.environ.get("MOSS_SHARED_HEAD", "0") == "1"


def validated_head_length(prompt_ids: list[int], codes, head: list[int], pad_id: int) -> int | None:
    """Return the head length only for identical tokens with pad-only audio conditioning."""
    n = len(head)
    if n == 0 or len(prompt_ids) <= n or prompt_ids[:n] != head:
        return None
    if codes is None or codes.shape[0] < n or not bool((codes[:n] == pad_id).all()):
        return None
    return n


def install() -> bool:
    if not enabled():
        return False
    from vllm.v1.core import block_pool, kv_cache_utils

    orig = kv_cache_utils.generate_block_hash_extra_keys
    if getattr(orig, "_moss_shared_head", False):
        return True

    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        salt = getattr(request, "cache_salt", None)
        params = getattr(request, "sampling_params", None)
        head = (getattr(params, "extra_args", None) or {}).get(HEAD_TOKENS_KEY)
        prompt_ids = getattr(request, "prompt_token_ids", None)
        if not salt or type(head) is not int or prompt_ids is None or not 0 < head < len(prompt_ids):
            return orig(request, start_token_idx, end_token_idx, start_mm_idx)
        request.cache_salt = None
        try:
            keys, nxt = orig(request, start_token_idx, end_token_idx, start_mm_idx)
        finally:
            request.cache_salt = salt
        if start_token_idx <= head < end_token_idx:
            keys = (keys or ()) + (("cache_salt", salt),)
        return keys, nxt

    generate_block_hash_extra_keys._moss_shared_head = True
    kv_cache_utils.generate_block_hash_extra_keys = generate_block_hash_extra_keys
    block_pool.generate_block_hash_extra_keys = generate_block_hash_extra_keys
    return True
