# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplexFence:
    """Stable identity fence shared by duplex engine control messages."""

    session_id: str
    epoch: int = 0
    turn_id: int = 0
    response_seq: int = 0
    incarnation: int = 0


__all__ = ["DuplexFence"]
