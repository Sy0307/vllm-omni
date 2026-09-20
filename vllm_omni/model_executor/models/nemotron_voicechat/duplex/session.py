# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.input import (
    NemotronVoiceChatPcmAppendBuffer,
)


@dataclass(slots=True)
class NemotronVoiceChatSessionState(DefaultDuplexModelSessionState):
    """Per-session model state of one Nemotron VoiceChat duplex session (owned by the session runner).

    Every flag lives on the framework's ``DefaultDuplexModelSessionState``;
    the only model-owned part is the 80 ms PCM packetizer.
    """

    audio_buffer: NemotronVoiceChatPcmAppendBuffer = field(default_factory=NemotronVoiceChatPcmAppendBuffer)


__all__ = ["NemotronVoiceChatSessionState"]
