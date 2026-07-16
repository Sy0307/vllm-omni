# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec


class ResumableSegmentPolicy(msgspec.Struct, frozen=True):
    """Scheduler-owned boundaries for one resumable generation segment."""

    stop_token_ids: tuple[int, ...] = ()
    max_segment_tokens: int | None = None

    def reached_boundary(self, new_token_ids: list[int], output_token_count: int) -> bool:
        if self.stop_token_ids and any(int(token_id) in self.stop_token_ids for token_id in new_token_ids):
            return True
        return self.max_segment_tokens is not None and output_token_count >= self.max_segment_tokens


__all__ = ["ResumableSegmentPolicy"]
