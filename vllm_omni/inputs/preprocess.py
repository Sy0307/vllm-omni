# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve Omni prompt metadata through vLLM's rendering pipeline."""

from functools import cache
from typing import Any

from vllm.config import VllmConfig
from vllm.renderers import BaseRenderer
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config, tokenizer_args_from_config


class OmniRendererMixin(BaseRenderer):
    """Extend the configured renderer without replacing its tokenizer or caches.

    vLLM 0.29 renders raw prompts before constructing engine requests. Keep
    pipeline metadata at that boundary, including for async rendering and
    embedding prompts. Text tokenization remains owned by the base renderer.
    """

    @staticmethod
    def _with_omni_metadata(inputs: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
        for key in ("prompt", "cache_salt", "additional_information", "model_intermediate_buffer"):
            if key in prompt:
                inputs[key] = prompt[key]
        return inputs

    @staticmethod
    def _requires_processor(prompt: dict[str, Any]) -> bool:
        # An explicitly empty kwargs dict still requests multimodal processing
        # for AR image models whose processor supplies an output scaffold.
        return "prompt_embeds" not in prompt and "mm_processor_kwargs" in prompt and not prompt.get("multi_modal_data")

    def _process_singleton(self, prompt, *, skip_mm_cache: bool = False):
        if self._requires_processor(prompt):
            inputs = self._process_multimodal(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = super()._process_singleton(prompt, skip_mm_cache=skip_mm_cache)
        return self._with_omni_metadata(inputs, prompt)

    async def _process_singleton_async(self, prompt, *, skip_mm_cache: bool = False):
        if self._requires_processor(prompt):
            inputs = await self._process_multimodal_async(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = await super()._process_singleton_async(prompt, skip_mm_cache=skip_mm_cache)
        return self._with_omni_metadata(inputs, prompt)


@cache
def omni_renderer_cls(renderer_cls: type[BaseRenderer]) -> type[BaseRenderer]:
    return type(f"Omni{renderer_cls.__name__}", (OmniRendererMixin, renderer_cls), {})


def build_omni_renderer(vllm_config: VllmConfig) -> BaseRenderer:
    model_config = vllm_config.model_config
    tokenizer = cached_tokenizer_from_config(model_config)
    renderer_mode, *_ = tokenizer_args_from_config(model_config)
    renderer_cls = RENDERER_REGISTRY.load_renderer_cls(renderer_mode)
    return omni_renderer_cls(renderer_cls)(vllm_config, tokenizer)
