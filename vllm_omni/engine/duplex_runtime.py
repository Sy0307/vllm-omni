# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from types import MappingProxyType
from typing import Any, Protocol

from vllm_omni.engine.duplex_types import DuplexFence
from vllm_omni.engine.resumable import ResumableSegmentPolicy


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


class DuplexOutputAction(str, Enum):
    DIRECT_RESPONSE = "direct_response"


@dataclass
class DuplexRuntimeCapabilities:
    input_modes: set[DuplexInputMode] = field(default_factory=lambda: {DuplexInputMode.TURN_COMMIT_ONLY})
    implementation_level: str = "serving_session_adapter"


@dataclass(frozen=True)
class DuplexAppendPlan:
    prompt: dict[str, Any]
    segment_policy: ResumableSegmentPolicy | None = None


@dataclass(frozen=True)
class DuplexOutputDecision:
    action: DuplexOutputAction
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_output_type: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DuplexRuntimeExtension(Protocol):
    """Model adapter used by the stable orchestrator duplex control plane."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]: ...

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan: ...

    def segment_policy(self, sampling_params: object) -> ResumableSegmentPolicy: ...

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None: ...


def load_duplex_runtime_extension(path: str | None) -> DuplexRuntimeExtension | None:
    if not path:
        return None
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid duplex runtime extension path: {path!r}")
    extension_type = getattr(import_module(module_name), attribute_name)
    return validate_duplex_runtime_extension(extension_type())


def validate_duplex_runtime_extension(
    extension: object,
    *,
    sampling_defaults: tuple[object, ...] | None = None,
) -> DuplexRuntimeExtension:
    required_methods = (
        "configure_sampling_params",
        "plan_append",
        "segment_policy",
        "decide_output",
    )
    missing = [name for name in required_methods if not callable(getattr(extension, name, None))]
    if missing:
        raise TypeError(f"Duplex runtime extension is missing callable method(s): {', '.join(missing)}")
    typed_extension = extension  # type: ignore[assignment]
    if sampling_defaults is not None:
        configured = typed_extension.configure_sampling_params(
            runtime_config={},
            defaults=sampling_defaults,
        )
        if not isinstance(configured, tuple):
            raise TypeError("Duplex runtime extension must return sampling parameters as a tuple")
        if len(configured) != len(sampling_defaults):
            raise ValueError("Duplex runtime extension must return one sampling parameter per stage")
        for stage_id, (value, default) in enumerate(zip(configured, sampling_defaults, strict=True)):
            if default is not None and not isinstance(value, type(default)):
                raise TypeError(
                    "Duplex runtime extension sampling parameter type mismatch "
                    f"for stage {stage_id}: expected {type(default).__name__}, got {type(value).__name__}"
                )
            policy = typed_extension.segment_policy(value)
            if not isinstance(policy, ResumableSegmentPolicy):
                raise TypeError("Duplex runtime extension segment_policy() must return ResumableSegmentPolicy")
    return typed_extension


from vllm_omni.engine.duplex_session import (  # noqa: E402
    DuplexAppendReservation,
    DuplexCompletedAppend,
    DuplexFenceMismatchError,
    DuplexInputAppend,
    DuplexRequestResource,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    DuplexStageBinding,
)


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
    encoded_session_id = base64.urlsafe_b64encode(fence.session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return f"duplex-s.{encoded_session_id}.i.{fence.incarnation}.e.{fence.epoch}.r.{role}"


__all__ = [
    "DuplexAppendPlan",
    "DuplexCompletedAppend",
    "DuplexAppendReservation",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexInputMode",
    "DuplexOutputAction",
    "DuplexOutputDecision",
    "DuplexRuntimeCapabilities",
    "DuplexRuntimeExtension",
    "DuplexRequestResource",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
    "SessionMode",
    "duplex_data_plane_request_info",
    "duplex_resource_request_id",
    "load_duplex_runtime_extension",
    "validate_duplex_runtime_extension",
]
