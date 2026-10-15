# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Local-v1.5 continuation prompt fast path equals the processor layout."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import _local_continuation_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Processor:
    """Segment API of the model's processor with a deterministic tokenizer."""

    def __init__(self):
        self.model_config = SimpleNamespace(
            n_vq=3,
            im_start_token_id=1,
            im_end_token_id=2,
            audio_start_token_id=3,
            audio_pad_token_id=1024,
            audio_assistant_slot_token_id=5,
        )
        self.encoded = []

    def _encode_text(self, text):
        self.encoded.append(text)
        return [100 + ord(c) % 50 for c in text]

    def _user_prompt_prefix_ids(self):
        return [1] + self._encode_text("user\n") + self._encode_text("<ref>")

    def _user_prompt_after_reference_ids(self, language, fields):
        return self._encode_text(f"lang={fields.get('language', language)}")

    def _assistant_prompt_prefix_ids(self):
        return self._encode_text("</user>") + [2] + self._encode_text("\n") + [1] + self._encode_text("assistant")

    def reference(self, text, language, codes):
        # Mirrors the processor's _build_continuation_codes + _pad (one item).
        fields = {} if language is None else {"language": language}
        ids = (
            self._user_prompt_prefix_ids()
            + self._encode_text("None")
            + self._user_prompt_after_reference_ids(language, fields)
            + self._encode_text(text)
            + self._assistant_prompt_prefix_ids()
            + [3]
        )
        text_rows = torch.full((len(ids), 4), 1024, dtype=torch.long)
        text_rows[:, 0] = torch.tensor(ids)
        audio_rows = torch.full((codes.shape[0], 4), 1024, dtype=torch.long)
        audio_rows[:, 0] = 5
        audio_rows[:, 1:] = codes
        unified = torch.cat([text_rows, audio_rows])
        return unified[:, 0].tolist(), unified[:, 1:].contiguous()


@pytest.mark.parametrize("language", [None, "English"])
def test_matches_processor_and_caches_constant_segments(language):
    proc = _Processor()
    codes = torch.randint(0, 1024, (7, 3), dtype=torch.int32)
    kwargs = {"text": "ref words target words"}
    if language is not None:
        kwargs["language"] = language
    ids, audio = _local_continuation_prompt(proc, kwargs, [codes])
    expected_ids, expected_audio = proc.reference(kwargs["text"], language, codes.long())
    assert ids == expected_ids
    assert audio.dtype == torch.int64 and torch.equal(audio, expected_audio)

    proc.encoded.clear()
    _local_continuation_prompt(proc, {**kwargs, "text": "other"}, [codes])
    assert proc.encoded == ["other"]


def test_unsupported_layouts_fall_back():
    proc = _Processor()
    codes = torch.zeros((2, 3), dtype=torch.long)
    assert _local_continuation_prompt(proc, {"text": "t", "instruction": "x"}, [codes]) is None
    assert _local_continuation_prompt(proc, {"text": "t"}, [codes, codes]) is None
    assert _local_continuation_prompt(proc, {"text": "t"}, ["path.wav"]) is None
    assert _local_continuation_prompt(proc, {"text": "t"}, [torch.zeros((2, 4), dtype=torch.long)]) is None


@pytest.mark.parametrize("enabled,valid", [(True, True), (True, False), (False, True)])
def test_shared_head_metadata_is_validated_without_mutating_defaults(monkeypatch, enabled, valid):
    from vllm import SamplingParams

    from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import MossTTSAdapter
    from vllm_omni.model_executor.models.moss_tts.shared_head import HEAD_TOKENS_KEY

    monkeypatch.setenv("MOSS_SHARED_HEAD", "1" if enabled else "0")
    proc = SimpleNamespace(
        model_config=SimpleNamespace(audio_pad_token_id=99),
        _vllm_omni_continuation_segments={None: ([1, 2],)},
    )
    adapter = SimpleNamespace(_moss_variant="local", _get_moss_processor=lambda: proc)
    # A caller-supplied internal key is overwritten or removed after validation.
    default = SamplingParams(max_tokens=20, extra_args={HEAD_TOKENS_KEY: 99, "other": 1})
    defaults = [default]
    prompt = {
        "cache_salt": "hello|tenant",
        "prompt_token_ids": [1, 2, 3],
        "additional_information": {"codes": {"ref": torch.full((3, 2), 99 if valid else 0)}},
    }
    result = MossTTSAdapter.apply_sampling_overrides(
        adapter, defaults, SimpleNamespace(max_new_tokens=None, language=None), prompt
    )
    assert result[0].extra_args.get(HEAD_TOKENS_KEY) == (2 if enabled and valid else None)
    assert result[0].extra_args["other"] == 1
    assert defaults == [default] and default.extra_args[HEAD_TOKENS_KEY] == 99
    assert prompt["cache_salt"] == "hello|tenant"
