# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequestState multimodal DELTA drain and consolidation guard."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.engine.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# NOTE: detokenizer and logprobs aren't really used here, but we mock them since
# some of the utils called in vLLM superclass assert require them to be None.
_DETOK = MagicMock(
    output_token_ids=[0],
    get_next_output_text=MagicMock(return_value=""),
    num_output_tokens=MagicMock(return_value=1),
)
_LOGPROBS = MagicMock(logprobs=None, cumulative_logprob=None, prompt_logprobs=None)

_DEFAULT_STATE_KWARGS = dict(
    request_id="r",
    external_req_id="r",
    parent_req=None,
    request_index=0,
    lora_request=None,
    prompt=None,
    prompt_token_ids=[0],
    prompt_embeds=None,
    logprobs_processor=_LOGPROBS,
    detokenizer=_DETOK,
    max_tokens_param=None,
    arrival_time=0.0,
    queue=None,
    log_stats=False,
    stream_interval=1,
)


def _make_state(output_kind: RequestOutputKind):
    return OmniRequestState(**_DEFAULT_STATE_KWARGS, output_kind=output_kind)


def test_init_empty_dict():
    """Ensure mm_accumulated is initially empty."""
    assert _make_state(RequestOutputKind.CUMULATIVE).mm_accumulated == {}
    assert _make_state(RequestOutputKind.DELTA).mm_accumulated == {}


def test_delta_drains_per_step():
    """Ensure that requests with delta outputs clear mm_accumulated data after emitting."""
    s = _make_state(RequestOutputKind.DELTA)
    t1, t2 = torch.ones(5), torch.ones(3)

    # Each call to _new_completion_output will clear mm_accumulated
    s.add_multimodal_tensor(t1, "audio")
    out1 = s._new_completion_output([1], None, None)
    assert torch.equal(out1.multimodal_output["audio"], t1)
    assert s.mm_accumulated == {}

    s.add_multimodal_tensor(t2, "audio")
    out2 = s._new_completion_output([2], None, None)
    assert torch.equal(out2.multimodal_output["audio"], t2)
    assert s.mm_accumulated == {}

    # If we have no mm accumulated data, get an empty dict back
    out3 = s._new_completion_output([3], None, None)
    assert out3.multimodal_output == {}
    assert out3.multimodal_output == {}


def test_cumulative_does_not_drain():
    """Ensure that calling _new_completion_output doesn't clear accumulated mm data."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    t1 = torch.ones(5)
    s.add_multimodal_tensor(t1, "audio")
    s._new_completion_output([1], None, None)
    assert "audio" in s.mm_accumulated
    assert torch.equal(s.mm_accumulated["audio"], t1)


def test_finish_consolidates_non_delta():
    """Ensure tensor consolidation is called for non delta images."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    len_1 = 5
    len_2 = 4
    s.add_multimodal_tensor(torch.ones(len_1), "audio")
    s.add_multimodal_tensor(torch.ones(len_2), "audio")

    result = s.make_request_output([1], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)

    audio = result.outputs[0].multimodal_output["audio"]
    assert isinstance(audio, torch.Tensor) and audio.shape[0] == len_1 + len_2


def test_finish_skips_consolidation_for_delta():
    s = _make_state(RequestOutputKind.DELTA)
    s.add_multimodal_tensor(torch.ones(5), "audio")
    s.add_multimodal_tensor(torch.ones(3), "audio")
    assert isinstance(s.mm_accumulated["audio"], list)

    result = s.make_request_output([1], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)

    # Audio should still be a list (not consolidated into a single tensor)
    audio = result.outputs[0].multimodal_output["audio"]
    assert isinstance(audio, list) and len(audio) == 2
    assert s.mm_accumulated == {}
