# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm_omni.engine.duplex_types import DuplexFence


class DuplexFenceMismatchError(RuntimeError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence) -> None:
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


class SessionMode(str, Enum):
    TURN = "turn"
    DUPLEX = "duplex"


class DuplexInputMode(str, Enum):
    APPEND_TOKENS = "append_tokens"
    APPEND_AUDIO_CHUNK = "append_audio_chunk"
    REPLACE_LATEST_CHUNK = "replace_latest_chunk"
    REENCODE_CONTEXT = "reencode_context"
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    TURN_COMMIT_ONLY = "turn_commit_only"


@dataclass
class DuplexRuntimeCapabilities:
    input_modes: set[DuplexInputMode] = field(default_factory=lambda: {DuplexInputMode.TURN_COMMIT_ONLY})
    implementation_level: str = "serving_session_adapter"


@dataclass
class DuplexStageBinding:
    request_id: str
    fence: DuplexFence


@dataclass
class DuplexInputAppend:
    seq: int
    turn_seq: int
    turn_id: int


@dataclass
class DuplexSessionRuntimeState:
    """Engine resource handles associated with a core-owned identity fence."""

    fence: DuplexFence
    capabilities: DuplexRuntimeCapabilities = field(default_factory=DuplexRuntimeCapabilities)
    session_config: dict[str, Any] = field(default_factory=dict)
    stage_bindings: dict[int, DuplexStageBinding] = field(default_factory=dict)
    input_seq: int = 0
    input_turn_seq: int = 0
    _append_turn_key: tuple[int, int, int] | None = None

    @property
    def session_id(self) -> str:
        return self.fence.session_id

    @property
    def epoch(self) -> int:
        return self.fence.epoch

    @property
    def turn_id(self) -> int:
        return self.fence.turn_id

    def accept_fence(self, fence: DuplexFence) -> None:
        if fence.session_id != self.session_id or fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, fence)
        current = self.fence
        if fence.epoch < current.epoch or (
            fence.epoch == current.epoch
            and (fence.turn_id < current.turn_id or fence.response_seq < current.response_seq)
        ):
            raise DuplexFenceMismatchError(current, fence)
        if fence.epoch != self.fence.epoch:
            self.input_seq = 0
            self.input_turn_seq = 0
            self._append_turn_key = None
        self.fence = fence

    def bind_stage_request(
        self,
        stage_id: int,
        request_id: str,
        *,
        fence: DuplexFence,
    ) -> None:
        self.accept_fence(fence)
        self.stage_bindings[stage_id] = DuplexStageBinding(
            request_id=request_id,
            fence=fence,
        )

    def stage_request_ids(self, fence: DuplexFence | None = None) -> list[str]:
        return [
            binding.request_id for binding in self.stage_bindings.values() if fence is None or binding.fence == fence
        ]

    def append_input(
        self,
        *,
        mode: DuplexInputMode,
        fence: DuplexFence,
    ) -> DuplexInputAppend:
        if mode not in self.capabilities.input_modes:
            raise ValueError(f"Duplex input mode {mode.value!r} is not supported by session {self.session_id}")
        self.accept_fence(fence)
        self.input_seq += 1
        turn_key = (fence.epoch, fence.turn_id, fence.response_seq)
        if turn_key != self._append_turn_key:
            self._append_turn_key = turn_key
            self.input_turn_seq = 0
        self.input_turn_seq += 1
        update = DuplexInputAppend(
            seq=self.input_seq,
            turn_seq=self.input_turn_seq,
            turn_id=fence.turn_id,
        )
        return update

    def release_fence(self, fence: DuplexFence) -> list[str]:
        stale = self.stage_request_ids(fence)
        self.stage_bindings = {
            stage_id: binding for stage_id, binding in self.stage_bindings.items() if binding.fence != fence
        }
        return stale

    def cancel_fence(
        self,
        cancelled_fence: DuplexFence,
        next_fence: DuplexFence,
    ) -> list[str]:
        if cancelled_fence.session_id != self.session_id or cancelled_fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if (
            next_fence.session_id != self.session_id
            or next_fence.incarnation != self.fence.incarnation
            or next_fence.epoch <= cancelled_fence.epoch
        ):
            raise DuplexFenceMismatchError(cancelled_fence, next_fence)

        current_key = (self.fence.epoch, self.fence.turn_id, self.fence.response_seq)
        cancelled_key = (
            cancelled_fence.epoch,
            cancelled_fence.turn_id,
            cancelled_fence.response_seq,
        )
        next_key = (next_fence.epoch, next_fence.turn_id, next_fence.response_seq)
        if cancelled_key > current_key:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if next_key > current_key:
            self.accept_fence(next_fence)
        return self.release_fence(cancelled_fence)

    def close(self, fence: DuplexFence | None = None) -> list[str]:
        if fence is not None:
            self.accept_fence(fence)
        stale = self.stage_request_ids()
        self.stage_bindings.clear()
        return stale


class DuplexSessionRuntimeManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DuplexSessionRuntimeState] = {}

    def open_session(
        self,
        fence: DuplexFence,
        *,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, Any] | None = None,
    ) -> DuplexSessionRuntimeState:
        if not isinstance(fence, DuplexFence):
            raise TypeError("open_session requires DuplexFence")
        if fence.session_id in self._sessions:
            raise ValueError(f"Duplex session already exists: {fence.session_id}")
        session = DuplexSessionRuntimeState(
            fence=fence,
            capabilities=capabilities or DuplexRuntimeCapabilities(),
            session_config=dict(session_config or {}),
        )
        self._sessions[fence.session_id] = session
        return session

    def get(self, session_id: str) -> DuplexSessionRuntimeState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> DuplexSessionRuntimeState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown duplex session: {session_id}")
        return session

    def close_session(self, fence: DuplexFence) -> DuplexSessionRuntimeState | None:
        if not isinstance(fence, DuplexFence):
            raise TypeError("close_session requires DuplexFence")
        session = self._sessions.get(fence.session_id)
        if session is not None:
            session.close(fence)
            if self._sessions.get(fence.session_id) is session:
                self._sessions.pop(fence.session_id)
        return session

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]:
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        for session_id, session in list(self._sessions.items()):
            stale = session.stage_request_ids()
            if request_id_set.isdisjoint(stale):
                continue
            self._sessions.pop(session_id, None)
            session.close()
            closed[session_id] = stale
        return closed


def duplex_data_plane_request_info(result: dict[str, object]) -> tuple[str | None, int | None]:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None, None
    for item in stage_results:
        if not isinstance(item, dict):
            continue
        inner = item.get("result")
        if not isinstance(inner, dict) or inner.get("data_plane_append") is not True:
            continue
        request_id = inner.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        response_stage_id = inner.get("response_stage_id")
        return request_id, response_stage_id if isinstance(response_stage_id, int) else None
    return None, None


def duplex_resource_request_id(fence: DuplexFence, role: str) -> str:
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        raise ValueError(f"invalid duplex resource role: {role!r}")
    incarnation = f"-i{fence.incarnation}" if fence.incarnation else ""
    return f"duplex-{fence.session_id}{incarnation}-e{fence.epoch}-{role}"


_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return bool(sample_count) and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN)
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8))


def duplex_first_append_context_reserve(session_config: object) -> int:
    if not isinstance(session_config, dict):
        return 48
    sources: list[dict[str, Any]] = [session_config]
    if isinstance(session_config.get("extra_body"), dict):
        sources.append(session_config["extra_body"])
    for source in sources:
        exact = source.get("duplex_first_append_context_tokens")
        if isinstance(exact, int) and exact >= 0:
            return exact
    reserve = 48
    for source in sources:
        ref = source.get("ref_audio_data")
        if not isinstance(ref, str) or not ref:
            continue
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            continue
        reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
        break
    return reserve


def duplex_new_user_turn_prefix_reserve(session_config: object, *, variant: object = None) -> int:
    if not isinstance(session_config, dict):
        return 0
    sources: list[dict[str, Any]] = [session_config]
    if isinstance(session_config.get("extra_body"), dict):
        sources.append(session_config["extra_body"])
    if isinstance(variant, str) and variant:
        for source in sources:
            by_variant = source.get("duplex_new_user_turn_prefix_tokens_by_variant")
            if isinstance(by_variant, dict) and isinstance(by_variant.get(variant), int):
                return max(0, by_variant[variant])
    for source in sources:
        exact = source.get("duplex_new_user_turn_prefix_tokens")
        if isinstance(exact, int) and exact >= 0:
            return exact
    return 0


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    """Plan the token-shaped scheduler admission for the current Omni port."""
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(session_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1
    if (
        seq > 1
        and duplex_payload_is_exact_chunks(payload)
        and not (isinstance(payload, dict) and payload.get("new_user_turn") is True)
    ):
        token_budget += 1
    if isinstance(payload, dict) and payload.get("new_user_turn") is True:
        token_budget += duplex_new_user_turn_prefix_reserve(
            session_config,
            variant=payload.get("new_user_turn_prefix_variant"),
        )
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
    extra_body = session_config.get("extra_body")
    raw_token_id = session_config.get("duplex_scheduler_token_id")
    if raw_token_id is None and isinstance(extra_body, dict):
        raw_token_id = extra_body.get("duplex_scheduler_token_id")
    try:
        token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


__all__ = [
    "DuplexInputAppend",
    "DuplexInputMode",
    "DuplexRuntimeCapabilities",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
    "SessionMode",
    "build_duplex_data_plane_prompt",
    "duplex_data_plane_request_info",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_new_user_turn_prefix_reserve",
    "duplex_payload_is_exact_chunks",
    "duplex_resource_request_id",
    "duplex_scheduler_token_budget",
]
