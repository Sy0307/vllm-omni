# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    EngineAppendAccepted,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.core.ports import EngineEvent
from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import DelegationBridge, StubDelegationBridge
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.policy import JoyVLPolicy, sample_frames

GenerateFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


class JoyVLDuplexAdapter:
    """A small typed engine port for the JoyVL demonstration."""

    def __init__(
        self,
        generate: GenerateFn,
        *,
        persona: str = "default",
        num_frames: int = 4,
        chunk_frames: int = 200,
        frame_seconds: float = 1.0,
        delegation: DelegationBridge | None = None,
    ) -> None:
        self._generate = generate
        self._policy = JoyVLPolicy(
            persona=persona,
            num_frames=num_frames,
            chunk_frames=chunk_frames,
            frame_seconds=frame_seconds,
            delegation=delegation or StubDelegationBridge(),
        )
        self._frames: list[str] = []
        self._pending_query: str | None = None
        self._commands: asyncio.Queue[AppendToEngine | None] = asyncio.Queue()
        self._cancelled: set[DuplexFence] = set()

    async def reserve(self, command: ReserveResponse) -> None:
        return None

    async def append(self, command: AppendToEngine) -> None:
        await self._commands.put(command)

    async def cancel(self, fence: DuplexFence) -> None:
        self._cancelled.add(fence)

    async def reset(self, command: ResetStage1) -> None:
        return None

    async def rebuild(self, command: RebuildStage0Context) -> None:
        return None

    async def close(self, fence: DuplexFence) -> None:
        await self._commands.put(None)

    async def events(self) -> AsyncIterator[EngineEvent]:
        while (command := await self._commands.get()) is not None:
            if command.chunk is not None:
                self._accept_input(
                    command.chunk.modality,
                    command.chunk.data,
                )
            if not command.final:
                continue

            fence = command.fence
            yield EngineAppendAccepted(fence=fence)
            action = await self._generate_action()
            if fence in self._cancelled:
                continue
            if action.action is not Action.SILENCE and action.text:
                yield ModelSpeaking(fence=fence)
                yield ModelTextDelta(text=action.text, fence=fence)
            yield ModelTurnEnded(fence=fence)

    def _accept_input(self, modality: str, data: bytes | str) -> None:
        if modality == "video":
            if not isinstance(data, str):
                raise TypeError("JoyVL video input must be a data URL string")
            self._policy.tick()
            self._frames.append(data)
        elif modality == "text":
            if not isinstance(data, str):
                raise TypeError("JoyVL text input must be a string")
            self._pending_query = data

    async def _generate_action(self):
        policy = self._policy
        await policy.fold_delegations()
        policy.set_query(self._pending_query)
        self._pending_query = None

        parts = [
            {"type": "image_url", "image_url": {"url": frame}}
            for frame in sample_frames(self._frames, policy.num_frames)
        ]
        messages, _ = policy.build_messages(parts)
        action = policy.commit(await self._generate(messages))

        frames = [(str(index), frame) for index, frame in enumerate(self._frames)]
        await policy.submit_if_delegate(action, frames)
        if policy.needs_flush():
            await policy.flush(frames)
            self._frames.clear()
        return action


__all__ = ["JoyVLDuplexAdapter"]
