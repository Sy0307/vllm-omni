# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence


class DuplicateResponseLifecycleError(RuntimeError):
    pass


class LateResponseOutputError(RuntimeError):
    pass


@dataclass(slots=True)
class _ResponseRecord:
    response_id: str
    text_parts: list[str] = field(default_factory=list)
    audio_bytes: int = 0
    finished: bool = False


class ResponseLifecycleLedger:
    """Per-fence protocol projection state, separate from domain identity."""

    def __init__(self) -> None:
        self._records: dict[DuplexFence, _ResponseRecord] = {}

    @staticmethod
    def response_id(fence: DuplexFence) -> str:
        return f"resp-{fence.session_id}-e{fence.epoch}-t{fence.turn_id}-r{fence.response_seq}"

    def start(self, fence: DuplexFence) -> str:
        if fence in self._records:
            raise DuplicateResponseLifecycleError(f"response already started for {fence!r}")
        response_id = self.response_id(fence)
        self._records[fence] = _ResponseRecord(response_id=response_id)
        return response_id

    def _active(self, fence: DuplexFence) -> _ResponseRecord:
        record = self._records.get(fence)
        if record is None:
            raise LateResponseOutputError(f"response was not started for {fence!r}")
        if record.finished:
            raise LateResponseOutputError(f"response already finished for {fence!r}")
        return record

    def append_text(self, fence: DuplexFence, text: str) -> None:
        record = self._active(fence)
        if text:
            record.text_parts.append(text)

    def append_audio(self, fence: DuplexFence, data: bytes) -> None:
        self._active(fence).audio_bytes += len(data)

    def transcript(self, fence: DuplexFence) -> str:
        record = self._records.get(fence)
        return "" if record is None else "".join(record.text_parts)

    def finish(self, fence: DuplexFence) -> str:
        record = self._records.get(fence)
        if record is None:
            raise LateResponseOutputError(f"response was not started for {fence!r}")
        if record.finished:
            raise DuplicateResponseLifecycleError(f"response already finished for {fence!r}")
        record.finished = True
        return "".join(record.text_parts)

    def release(self, fence: DuplexFence) -> None:
        self._records.pop(fence, None)
