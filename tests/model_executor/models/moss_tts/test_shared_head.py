# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared heads use validated MOSS metadata; public salts stay opaque."""

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core import block_pool, kv_cache_utils
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request

from vllm_omni.model_executor.models.moss_tts import shared_head

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def installed(monkeypatch):
    # Let monkeypatch restore both module aliases after install().
    monkeypatch.setattr(kv_cache_utils, "generate_block_hash_extra_keys", kv_cache_utils.generate_block_hash_extra_keys)
    monkeypatch.setattr(block_pool, "generate_block_hash_extra_keys", block_pool.generate_block_hash_extra_keys)
    manager_method = KVCacheManager.get_computed_blocks
    shared_head.install()
    assert KVCacheManager.get_computed_blocks is manager_method
    return kv_cache_utils.generate_block_hash_extra_keys


def _request(salt, head=None):
    params = SamplingParams(max_tokens=1, extra_args={shared_head.HEAD_TOKENS_KEY: head} if head is not None else None)
    return Request("r", list(range(100)), params, None, cache_salt=salt)


@pytest.mark.parametrize("salt", ["hello", "hello|tenant", "h32|tenant", "h999|tenant", "", None])
def test_public_salts_remain_opaque(installed, salt):
    request = _request(salt)
    first, _ = installed(request, 0, 32, 0)
    later, _ = installed(request, 32, 64, 0)
    assert first == ((("cache_salt", salt),) if salt else None)
    assert later is None
    assert request.cache_salt == salt


@pytest.mark.parametrize("block_size", [16, 32])
def test_only_validated_head_is_shared(installed, block_size):
    head = 2 * block_size
    for salt in ("conditioning-a", "conditioning-b"):
        request = _request(salt, head)
        assert installed(request, 0, block_size, 0)[0] is None
        assert installed(request, block_size, head, 0)[0] is None
        assert installed(request, head, head + block_size, 0)[0] == (("cache_salt", salt),)
        assert installed(request, head + block_size, head + 2 * block_size, 0)[0] is None
        assert request.cache_salt == salt


@pytest.mark.parametrize("head", [True, -1, 0, 100, "32"])
def test_invalid_metadata_uses_normal_hashing(installed, head):
    assert installed(_request("hello", head), 0, 16, 0)[0] == (("cache_salt", "hello"),)


def test_head_requires_identical_tokens_and_pad_conditioning():
    codes = torch.full((5, 3), 99)
    assert shared_head.validated_head_length([1, 2, 3, 4, 5], codes, [1, 2, 3], 99) == 3
    assert shared_head.validated_head_length([1, 2, 4, 4, 5], codes, [1, 2, 3], 99) is None
    codes[1, 0] = 1
    assert shared_head.validated_head_length([1, 2, 3, 4, 5], codes, [1, 2, 3], 99) is None


def test_install_is_idempotent(installed):
    shared_head.install()
    assert kv_cache_utils.generate_block_hash_extra_keys is installed
    assert block_pool.generate_block_hash_extra_keys is installed
