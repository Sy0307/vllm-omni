# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned PCM accumulation for a stage that decodes audio in place."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AudioRequest:
    samples_per_frame: int
    chunk_frames: int
    active: bool = True
    emitted: bool = False
    pending: list[torch.Tensor] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, pcm: torch.Tensor, ended: bool) -> torch.Tensor | None:
        with self.lock:
            if not self.active:
                return None
            if pcm.numel():
                self.pending.append(pcm)
            frames = sum(part.numel() for part in self.pending) // self.samples_per_frame
            out = None
            if frames and (ended or not self.emitted or frames >= self.chunk_frames):
                out = torch.cat(self.pending) if len(self.pending) > 1 else self.pending[0]
                self.pending.clear()
                self.emitted = True
            if ended:
                self.active = False
                self.pending.clear()
            return out

    def finish(self) -> None:
        with self.lock:
            self.active = False
            self.pending.clear()


class StreamingAudioBuffer:
    def __init__(self, samples_per_frame: int, chunk_frames: int) -> None:
        self.samples_per_frame = samples_per_frame
        self.chunk_frames = chunk_frames
        self.requests: dict[str, AudioRequest] = {}

    def add(self, request_id: str) -> AudioRequest:
        if request_id not in self.requests:
            self.requests[request_id] = AudioRequest(self.samples_per_frame, self.chunk_frames)
        return self.requests[request_id]

    def finish(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            state = self.requests.pop(request_id, None)
            if state is not None:
                # Already queued outputs retain this object, never a fresh lookup
                # by ID. A late copy cannot resurrect a cancelled request.
                state.finish()


@dataclass
class StreamingAudioOutput:
    wav: torch.Tensor
    valid: torch.Tensor
    sample_rate: torch.Tensor
    event: torch.cuda.Event | None
    query_start_loc: np.ndarray
    requests: list[AudioRequest | None]
    length_end: np.ndarray

    def to_cpu(
        self,
        copy_stream: torch.cuda.Stream,
        copy_tensor: Callable[[torch.Tensor], torch.Tensor],
    ) -> StreamingAudioOutput:
        if self.event is not None:
            copy_stream.wait_event(self.event)
        return StreamingAudioOutput(
            copy_tensor(self.wav),
            copy_tensor(self.valid),
            self.sample_rate,
            None,
            self.query_start_loc,
            self.requests,
            self.length_end,
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        valid = self.valid.numpy().astype(bool)
        output: list[dict[str, torch.Tensor] | None] = []
        for i, state in enumerate(self.requests):
            payload = None
            start, end = self.query_start_loc[i : i + 2]
            if state is not None and end > start:
                pcm = self.wav[start:end][torch.from_numpy(valid[start:end])].reshape(-1)
                ended = bool(self.length_end[i]) or not valid[end - 1]
                chunk = state.push(pcm, ended)
                if chunk is not None:
                    payload = {"model_outputs": chunk, "sr": self.sample_rate}
            output.append(payload)
        return output
