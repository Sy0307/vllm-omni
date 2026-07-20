# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplexSamplingRow:
    """Request-local context for an optional model sampling hook.

    The generic runner constructs rows only when the loaded model exposes
    ``prepare_duplex_sampling``. Models without that hook do not allocate or
    scan this metadata.
    """

    row_idx: int
    request_id: str
    session_id: str | None
    incarnation: int
    seq: int | None
    payload: dict[str, object] | None
    max_tokens: int | None


__all__ = ["DuplexSamplingRow"]
