# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DuplexSamplingRow:
    """Generic request context for an optional model-owned duplex sampler."""

    row_idx: int
    request_id: str
    session_id: str | None
    incarnation: int
    seq: int | None
    payload: dict[str, Any] | None
    max_tokens: int | None


__all__ = ["DuplexSamplingRow"]
