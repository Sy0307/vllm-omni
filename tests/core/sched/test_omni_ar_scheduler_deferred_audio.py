# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Held terminal outputs of single-stage audio stages."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.engine import FinishReason

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler():
    held = {
        "a": {"client_index": 1, "finish_reason": FinishReason.STOP, "stop_reason": 7},
        "b": {"client_index": 0, "finish_reason": FinishReason.LENGTH, "stop_reason": None},
    }
    return SimpleNamespace(omni_deferred_finishes=held, omni_tail_outputs=deque())


def test_audio_of_a_step_in_flight_at_the_stop_is_forwarded():
    scheduler = _scheduler()
    outputs: dict[int, list] = defaultdict(list)
    payload = {"model_outputs": torch.ones(2, 3)}
    runner_output = SimpleNamespace(req_id_to_index={"a": 0, "z": 1})
    OmniARScheduler._forward_deferred_audio(scheduler, outputs, "a", [payload, None], runner_output)
    OmniARScheduler._forward_deferred_audio(scheduler, outputs, "z", [payload, payload], runner_output)
    (out,) = outputs[1]
    assert out.request_id == "a" and out.finish_reason is None and out.multimodal_output is payload
    assert list(outputs) == [1]


def test_tail_outputs_follow_the_queue_and_release_the_hold():
    scheduler = _scheduler()
    audio = {"model_outputs": torch.ones(2, 3)}
    scheduler.omni_tail_outputs.extend([("a", audio, False), ("b", None, True), ("a", None, False), ("a", audio, True)])
    outputs: dict[int, list] = defaultdict(list)
    OmniARScheduler._emit_tail_outputs(scheduler, outputs)
    a = outputs[1]
    assert [o.finish_reason for o in a] == [None, FinishReason.STOP] and a[1].stop_reason == 7
    (b,) = outputs[0]
    assert b.finish_reason == FinishReason.LENGTH and b.multimodal_output is None
    assert not scheduler.omni_deferred_finishes and not scheduler.omni_tail_outputs
    # Late items of a request whose terminal output went out are dropped.
    scheduler.omni_tail_outputs.append(("a", audio, False))
    OmniARScheduler._emit_tail_outputs(scheduler, outputs)
    assert len(outputs[1]) == 2
