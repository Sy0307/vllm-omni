# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm_omni.outputs import OmniRequestOutput

DUPLEX_OUTPUT_DECISION_KEY = "_vllm_omni.experimental.fullduplex.duplex_output_decision"
DUPLEX_OUTPUT_CONTEXT_KEY = "_vllm_omni.experimental.fullduplex.duplex_output_context"


def _custom_output(output: object) -> dict[str, object] | None:
    custom_output = getattr(output, "_custom_output", None)
    if not isinstance(custom_output, dict):
        return None
    return custom_output


def attach_duplex_output_context(
    output: object,
    context: object,
) -> object:
    """Attach immutable Full-Duplex execution context to a final output."""
    custom_output = dict(_custom_output(output) or {})
    custom_output[DUPLEX_OUTPUT_CONTEXT_KEY] = context
    setattr(output, "_custom_output", custom_output)
    return output


def get_duplex_output_context(output: object) -> Any | None:
    custom_output = _custom_output(output)
    if custom_output is None:
        return None
    return custom_output.get(DUPLEX_OUTPUT_CONTEXT_KEY)


def attach_duplex_output_decision(
    output: OmniRequestOutput,
    decision: object,
) -> OmniRequestOutput:
    custom_output = dict(_custom_output(output) or {})
    custom_output[DUPLEX_OUTPUT_DECISION_KEY] = decision
    output._custom_output = custom_output
    return output


def get_duplex_output_decision(output: object) -> Any | None:
    custom_output = _custom_output(output)
    if custom_output is None:
        return None
    return custom_output.get(DUPLEX_OUTPUT_DECISION_KEY)


__all__ = [
    "DUPLEX_OUTPUT_CONTEXT_KEY",
    "DUPLEX_OUTPUT_DECISION_KEY",
    "attach_duplex_output_context",
    "attach_duplex_output_decision",
    "get_duplex_output_context",
    "get_duplex_output_decision",
]
