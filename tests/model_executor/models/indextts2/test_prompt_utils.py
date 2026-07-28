# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_executor.models.indextts2 import prompt_utils


def test_indextts25_prompt_len_uses_three_conditioning_tokens(monkeypatch):
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return [11, 12, 13, 14], 0

    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        fake_prepare,
    )

    prompt_len = prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
        text_normalization=False,
    )

    # Official prepare_gpt_inputs wraps the language-prefixed tokenizer IDs
    # with start_text=0 and stop_text=1 before appending start_mel.
    assert prompt_len == 3 + (4 + 2) + 1
    assert captured["text_normalization"] is False


def test_indextts25_prompt_len_filters_existing_text_wrapper_ids(monkeypatch):
    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        lambda *args, **kwargs: ([0, 58838, 1, 42, 0], 0),
    )

    prompt_len = prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
    )

    assert prompt_len == 3 + (2 + 2) + 1


def test_indextts25_prompt_ids_preserve_placeholder_value(monkeypatch):
    monkeypatch.setattr(
        prompt_utils,
        "estimate_indextts2_prefill_prompt_len",
        lambda *args, **kwargs: 10,
    )

    prompt_ids = prompt_utils.build_indextts2_prefill_prompt_ids(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
        placeholder_token_id=17,
    )

    assert prompt_ids == [17] * 10
