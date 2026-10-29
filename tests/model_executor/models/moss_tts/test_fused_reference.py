# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Continuation history belongs to the codec stream through detach and reuse."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.moss_tts.test_fused_audio import (
    N_VQ,
    _early_fused,
    _first_frames,
    _fused,
    _run,
    _stop,
    _stream_audio,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


def _reference(values):
    codes = torch.tensor(values).unsqueeze(1).expand(-1, N_VQ).clone()
    return {"codes": {"ref": codes}, "meta": {"ref_code_len": len(values)}}


def test_reference_primes_once_without_becoming_output():
    codec, fused, sender = _fused(initial=1, chunk=3, packed=True)
    info = _reference([1, 2, 3, 4])
    fused.add_request(0, "a", info)
    info["codes"]["ref"].fill_(80)
    delivered = []
    for code in [10, 11, 12, 13, 14, None]:
        _, audio = _run(fused, [0], {0: code})["a"]
        if audio:
            delivered.append(audio)
    _stop(fused, sender, "a")
    audio, finals = _stream_audio(delivered, sender, "a")
    assert [x for chunk in audio for x in chunk] == [10, 11, 12, 13, 14]
    assert [code for call in codec.calls[:2] for _, codes, _ in call for code in codes] == [1, 2, 3, 4, 10]
    assert all(len(codes) <= 3 for call in codec.calls for _, codes, _ in call)
    assert sum(1 in codes for call in codec.calls for _, codes, _ in call) == 1
    assert finals.count(True) == 1


def test_reference_survives_detach_and_talker_slot_reuse():
    codec, fused, sender = _fused(initial=1, chunk=3, fused_decode_first_wait_steps=100)
    fused.add_request(0, "old", _reference([1, 2]))
    for code in [10, None]:
        _run(fused, [0], {0: code})
    sender.deferred.add("old")
    fused.finish(["old"])
    fused.add_request(0, "new", _reference([7, 8]))
    for _ in range(6):
        fused.drain_detached_audio()
        torch.accelerator.synchronize()
    old_calls = [codes for call in codec.calls for req, codes, _ in call if req == "old"]
    assert old_calls == [[1, 2, 10]]
    assert [x for req, codes, _ in sender.sent if req == "old" for x in codes] == [10]
    assert fused.streams[0].reference[:, 0].tolist() == [7, 8]


def test_cancelled_reference_does_not_prime_reused_slot():
    codec, fused, _ = _fused(initial=1, chunk=3)
    fused.add_request(0, "old", _reference([1, 2]))
    fused.finish(["old"])
    fused.add_request(0, "new")
    for code in [10, 11, None]:
        _run(fused, [0], {0: code})
    assert all(req != "old" for call in codec.calls for req, _, _ in call)
    assert codec.calls[0] == [("new", [10], False)]


def test_invalid_reference_preserves_current_slot():
    _, fused, _ = _fused(initial=1, chunk=3)
    fused.add_request(0, "existing")
    info = _reference([1])
    info["meta"]["ref_code_len"] = 2
    with pytest.raises(ValueError, match="complete codec reference"):
        fused.add_request(0, "invalid", info)
    assert fused.slot_of == {"existing": 0}
    assert fused.streams[0].req_id == "existing"


def test_early_first_sender_receives_only_target_after_reference():
    codec, fused, sender, first = _early_fused()
    fused.add_request(0, "a", _reference([1, 2, 3, 4]))
    assert _first_frames(fused, {"a": 10}) == ["a"]
    delivered = []
    for code in [10, 11, 12, None]:
        _, audio = _run(fused, [0], {0: code})["a"]
        if audio:
            delivered.append(audio)
    _stop(fused, sender, "a")
    audio, finals = _stream_audio(delivered, sender, "a")
    assert first.sent == [(["a"], [10], [True])]
    assert [x for chunk in audio for x in chunk] == [11, 12]
    assert [code for call in codec.calls[:2] for _, codes, _ in call for code in codes] == [1, 2, 3, 4, 10]
    assert all(len(codes) <= 3 for call in codec.calls for _, codes, _ in call)
    assert finals.count(True) == 1
