# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression for the ``output_kind`` coercion in ``AsyncOmni.generate``.

PR #2911 unconditionally rewrote every SamplingParams.output_kind to DELTA,
which silently overrode callers that had explicitly asked for FINAL_ONLY
(e.g. the non-streaming PD prefill path). This coerces only the vLLM
default CUMULATIVE; explicit DELTA / FINAL_ONLY is preserved and the
caller's instance is cloned before mutation.

This is a replay of the in-function coercion loop — a contract guard, not
a full integration test.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _coerce(sp_list):
    req_sp_list = list(sp_list)
    for idx in range(len(req_sp_list)):
        sp = req_sp_list[idx]
        if not isinstance(sp, SamplingParams):
            continue
        if sp.output_kind != RequestOutputKind.CUMULATIVE:
            continue
        if not sp.skip_clone:
            sp = sp.clone()
            sp.skip_clone = True
            req_sp_list[idx] = sp
        sp.output_kind = RequestOutputKind.DELTA
    return req_sp_list


def test_cumulative_default_becomes_delta():
    sp = SamplingParams()
    result = _coerce([sp])[0]
    assert result.output_kind == RequestOutputKind.DELTA


def test_explicit_final_only_is_preserved():
    sp = SamplingParams(output_kind=RequestOutputKind.FINAL_ONLY)
    result = _coerce([sp])[0]
    assert result is sp
    assert result.output_kind == RequestOutputKind.FINAL_ONLY


def test_explicit_delta_is_preserved():
    sp = SamplingParams(output_kind=RequestOutputKind.DELTA)
    result = _coerce([sp])[0]
    assert result is sp
    assert result.output_kind == RequestOutputKind.DELTA


def test_cumulative_caller_instance_is_cloned_not_mutated():
    sp = SamplingParams()
    result = _coerce([sp])[0]
    assert result is not sp
    assert sp.output_kind == RequestOutputKind.CUMULATIVE
    assert result.output_kind == RequestOutputKind.DELTA
