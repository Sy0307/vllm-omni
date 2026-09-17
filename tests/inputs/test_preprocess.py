# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni metadata must survive vLLM 0.29 sync and async rendering."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.renderers import BaseRenderer

from vllm_omni.inputs.preprocess import omni_renderer_cls

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TokenRenderer(BaseRenderer):
    def render_messages(self, messages, params):
        raise NotImplementedError


@pytest.fixture
def renderer():
    instance = object.__new__(omni_renderer_cls(TokenRenderer))
    instance.model_config = SimpleNamespace(enable_prompt_embeds=True)
    instance._process_multimodal = Mock(return_value={"type": "multimodal", "prompt_token_ids": [1, 2]})
    instance._process_multimodal_async = AsyncMock(return_value={"type": "multimodal", "prompt_token_ids": [1, 2]})
    return instance


def render(renderer, prompt, asynchronous):
    if asynchronous:
        return asyncio.run(renderer._process_singleton_async(prompt, skip_mm_cache=True))
    return renderer._process_singleton(prompt, skip_mm_cache=True)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"height": 512}])
def test_explicit_processor_kwargs_without_media(renderer, asynchronous, kwargs):
    prompt = {"prompt_token_ids": [1, 2], "mm_processor_kwargs": kwargs, "multi_modal_uuids": {"image": ["id"]}}
    result = render(renderer, prompt, asynchronous)
    method = renderer._process_multimodal_async if asynchronous else renderer._process_multimodal
    method.assert_called_once_with(
        [1, 2], {}, mm_processor_kwargs=kwargs, mm_uuids={"image": ["id"]}, skip_mm_cache=True
    )
    assert result["type"] == "multimodal"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["token", "multimodal", "embeds"])
def test_render_preserves_pipeline_metadata(renderer, asynchronous, kind):
    metadata = {
        "additional_information": {"speaker": "test"},
        "model_intermediate_buffer": object(),
        "cache_salt": "salt",
    }
    prompt = {"prompt_token_ids": [1, 2], "prompt": "hello", **metadata}
    if kind == "multimodal":
        prompt["multi_modal_data"] = {"audio": object()}
    elif kind == "embeds":
        prompt["prompt_embeds"] = torch.zeros(2, 4)
    result = render(renderer, prompt, asynchronous)
    assert result["type"] == kind
    for key, value in metadata.items():
        assert result[key] is value
    assert result["prompt"] == "hello"
    if kind == "embeds":
        assert result["prompt_embeds"] is prompt["prompt_embeds"]
    if kind != "multimodal":
        renderer._process_multimodal.assert_not_called()
        renderer._process_multimodal_async.assert_not_called()


def test_embedding_validation_is_not_bypassed(renderer):
    renderer.model_config.enable_prompt_embeds = False
    with pytest.raises(ValueError, match="enable-prompt-embeds"):
        renderer._process_singleton({"prompt_embeds": torch.zeros(2, 4)})
