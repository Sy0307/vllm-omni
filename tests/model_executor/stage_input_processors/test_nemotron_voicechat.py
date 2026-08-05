# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
    _FULL_PAYLOAD_REPLACE_KEYS,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
    thinker2talker_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PAD = 12


def _thinker_output(prompt_len: int, generated: list[int]):
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=list(range(100, 100 + prompt_len)) + [_PAD],
        additional_information=None,
        outputs=[SimpleNamespace(token_ids=generated, cumulative_token_ids=None)],
    )


def test_thinker2talker_builds_pad_prompt_timeline() -> None:
    generated = [7, 8, 9, 10]
    inputs = thinker2talker_token_only([_thinker_output(prompt_len=5, generated=generated)])
    assert len(inputs) == 1
    prompt = inputs[0]
    # 1-token placeholder prompt; the timeline rides additional_information.
    assert prompt["prompt_token_ids"] == [0]
    info = prompt["additional_information"]
    assert info["nvc_logical_prompt_len"] == 5
    assert info["nvc_frame_count"] == 4
    assert info["nvc_text_timeline"] == [_PAD] * 5 + generated


def test_thinker2talker_skips_unfinished() -> None:
    out = _thinker_output(3, [1])
    out.finished = False
    assert thinker2talker_token_only([out]) == []


def test_full_payload_trims_prompt_region_rows() -> None:
    # Rows correspond to timeline steps t=1..T-1; with prompt_len P the NeMo
    # trim keeps rows for t >= P, i.e. row index P-1 onward.
    prompt_len = 4
    frames, q = 10, 31
    codes = torch.arange(frames * q, dtype=torch.long).reshape(frames, q) % 1024
    request = SimpleNamespace(
        additional_information={
            "codes": {"audio": codes},
            "nvc_logical_prompt_len": prompt_len,
        },
        additional_information_cpu=None,
    )
    payload = talker2code2wav_full_payload(request=request, is_finished=True)
    assert payload.codes.audio.shape == (frames - (prompt_len - 1), q)
    assert torch.equal(payload.codes.audio, codes[prompt_len - 1 :])
    assert bool(payload.meta.finished)


def test_full_payload_not_finished_is_pending() -> None:
    payload = talker2code2wav_full_payload(request=SimpleNamespace(), is_finished=False)
    assert payload is not None
    assert not bool(payload.meta.finished)
    assert payload.codes is None


def test_full_payload_replace_semantics_declared() -> None:
    # Full-sequence tensors must REPLACE in the accumulator, not concatenate.
    assert "codes" in _FULL_PAYLOAD_REPLACE_KEYS


def test_code2wav_token_only_sizes_placeholder() -> None:
    talker_output = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(token_ids=[0] * 9, cumulative_token_ids=None)],
    )
    inputs = talker2code2wav_token_only([talker_output])
    assert len(inputs) == 1
    assert len(inputs[0]["prompt_token_ids"]) == 9
