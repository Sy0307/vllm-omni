# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence

T = TypeVar("T")


class DuplexDataPlaneCursors:
    """Cumulative output cursors keyed only by the canonical fence."""

    def __init__(self) -> None:
        self._text: dict[DuplexFence, str] = {}
        self._audio_offsets: dict[DuplexFence, int] = {}

    def text_delta(self, fence: DuplexFence, cumulative: str) -> str:
        previous = self._text.get(fence, "")
        delta = cumulative[len(previous) :] if cumulative.startswith(previous) else cumulative
        self._text[fence] = cumulative
        return delta

    def audio_delta(self, fence: DuplexFence, cumulative: Sequence[T]) -> list[T]:
        previous = self._audio_offsets.get(fence, 0)
        if len(cumulative) < previous:
            previous = 0
        self._audio_offsets[fence] = len(cumulative)
        return list(cumulative[previous:])

    def release(self, fence: DuplexFence) -> None:
        self._text.pop(fence, None)
        self._audio_offsets.pop(fence, None)

    def release_session(self, session_id: str) -> None:
        for fence in set(self._text) | set(self._audio_offsets):
            if fence.session_id == session_id:
                self.release(fence)
