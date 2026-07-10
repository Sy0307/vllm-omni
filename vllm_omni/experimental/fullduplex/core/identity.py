# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class DuplexFence:
    session_id: str
    epoch: int = 0
    turn_id: int = 0
    response_seq: int = 0

    def next_turn(self) -> DuplexFence:
        return replace(
            self,
            turn_id=self.turn_id + 1,
            response_seq=self.response_seq + 1,
        )

    def next_epoch(self) -> DuplexFence:
        return replace(self, epoch=self.epoch + 1)
