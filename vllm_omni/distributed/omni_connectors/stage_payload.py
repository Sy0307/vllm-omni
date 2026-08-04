# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public contracts for payloads emitted between pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import msgspec

from vllm_omni.data_entry_keys import OmniPayloadStruct

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.payload_schema import (
        StagePayloadSchema,
    )


class StageBoundary(str, Enum):
    """A lifecycle boundary carried independently from payload data."""

    NONE = "none"
    SEGMENT_END = "segment_end"
    STREAM_END = "stream_end"


class StageRoute(msgspec.Struct, frozen=True, kw_only=True):
    """The directed pipeline edge that owns an envelope."""

    source_stage: int
    destination_stage: int

    def __post_init__(self) -> None:
        if type(self.source_stage) is not int or type(self.destination_stage) is not int:
            raise ValueError("stage IDs must be integers")
        if self.source_stage < 0 or self.destination_stage < 0:
            raise ValueError("stage IDs must be non-negative")
        if self.source_stage == self.destination_stage:
            raise ValueError("source stage and destination stage must differ")


class StageSessionFence(msgspec.Struct, frozen=True, kw_only=True):
    """Optional full-duplex session incarnation identity."""

    session_id: str
    incarnation: int
    epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string")
        if type(self.incarnation) is not int or type(self.epoch) is not int:
            raise TypeError("session fence counters must be integers")
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.incarnation < 0:
            raise ValueError("incarnation must be non-negative")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")


class StagePayloadIdentity(msgspec.Struct, frozen=True, kw_only=True):
    """Cross-stage identity with source-local diagnostic information."""

    external_request_id: str
    source_request_id: str
    session_fence: StageSessionFence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.external_request_id, str) or not isinstance(
            self.source_request_id,
            str,
        ):
            raise TypeError("request identities must be strings")
        if self.session_fence is not None and not isinstance(
            self.session_fence,
            StageSessionFence,
        ):
            raise TypeError("session_fence must be a StageSessionFence or None")
        if not self.external_request_id.strip():
            raise ValueError("external_request_id must not be empty")
        if not self.source_request_id.strip():
            raise ValueError("source_request_id must not be empty")


class StagePayloadEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable, versioned data-plane message between two stages.

    The struct containers are frozen. Tensors contained by ``payload`` are
    read-only by contract after the envelope is enqueued.
    """

    schema_version: int
    route: StageRoute
    identity: StagePayloadIdentity
    chunk_seq: int
    payload: OmniPayloadStruct | None = None
    boundary: StageBoundary = StageBoundary.NONE

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.route, StageRoute):
            raise TypeError("route must be a StageRoute")
        if not isinstance(self.identity, StagePayloadIdentity):
            raise TypeError("identity must be a StagePayloadIdentity")
        if type(self.chunk_seq) is not int or self.chunk_seq < 0:
            raise ValueError("chunk_seq must be non-negative")
        if self.payload is not None and not isinstance(self.payload, OmniPayloadStruct):
            raise TypeError("payload must be an OmniPayloadStruct or None")
        if not isinstance(self.boundary, StageBoundary):
            raise TypeError("boundary must be a StageBoundary")
        _require_payload_or_boundary(self.payload, self.boundary)


@dataclass(frozen=True, slots=True)
class StageRequestView:
    """A read-only snapshot of request state exposed to a processor."""

    internal_request_id: str
    external_request_id: str
    resumable: bool
    terminal: bool
    additional_information: Mapping[str, object] | None
    model_intermediate_buffer: Mapping[str, object] | None

    def __post_init__(self) -> None:
        if not self.internal_request_id.strip():
            raise ValueError("internal_request_id must not be empty")
        if not self.external_request_id.strip():
            raise ValueError("external_request_id must not be empty")
        if self.additional_information is not None:
            object.__setattr__(
                self,
                "additional_information",
                MappingProxyType(dict(self.additional_information)),
            )
        if self.model_intermediate_buffer is not None:
            object.__setattr__(
                self,
                "model_intermediate_buffer",
                MappingProxyType(dict(self.model_intermediate_buffer)),
            )


@dataclass(frozen=True, slots=True)
class NormalizedStageOutput:
    """Named output sources without legacy keyword precedence rules."""

    output_payload: OmniPayloadStruct | None
    request_payload: OmniPayloadStruct | None
    raw_output: object | None = None


@dataclass(frozen=True, slots=True)
class StagePayloadBuildContext:
    """The sole input accepted by a migrated stage payload processor."""

    source_output: NormalizedStageOutput
    request: StageRequestView
    route: StageRoute
    identity: StagePayloadIdentity
    chunk_seq: int
    boundary: StageBoundary

    def __post_init__(self) -> None:
        if self.chunk_seq < 0:
            raise ValueError("chunk_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class NoPayloadYet:
    """A processor needs more source output before emitting an envelope."""


@dataclass(frozen=True, slots=True)
class PayloadEmission:
    """A payload, a lifecycle boundary, or both, ready for transport."""

    payload: OmniPayloadStruct | None
    boundary: StageBoundary = StageBoundary.NONE

    def __post_init__(self) -> None:
        if self.payload is not None and not isinstance(self.payload, OmniPayloadStruct):
            raise TypeError("payload must be an OmniPayloadStruct or None")
        if not isinstance(self.boundary, StageBoundary):
            raise TypeError("boundary must be a StageBoundary")
        _require_payload_or_boundary(self.payload, self.boundary)


ProcessorResult = NoPayloadYet | PayloadEmission


class StagePayloadProcessor(Protocol):
    """Producer-side processor contract for one pipeline stage."""

    def process(self, context: StagePayloadBuildContext) -> ProcessorResult: ...

    def drop_request(self, identity: StagePayloadIdentity) -> None: ...


def try_normalize_stage_payload(value: Any) -> OmniPayloadStruct | None:
    """Best-effort normalization for a processor's named payload view.

    Legacy processors may consume model-native mapping outputs that are not
    ``OmniPayloadStruct`` values. Those mappings remain available through
    ``NormalizedStageOutput.raw_output``; failure to normalize them must not
    prevent the processor from seeing them.
    """
    from vllm_omni.data_entry_keys import to_dict, to_struct

    if value is None:
        return None
    if isinstance(value, OmniPayloadStruct):
        return to_struct(to_dict(value))
    if not isinstance(value, Mapping):
        return None
    try:
        return to_struct(dict(value))
    except (msgspec.ValidationError, TypeError, ValueError):
        return None


_ENVELOPE_WIRE_FIELDS = frozenset(
    {
        "schema_version",
        "route",
        "identity",
        "chunk_seq",
        "payload",
        "boundary",
    }
)


def decode_stage_payload_wire(
    value: Any,
    *,
    schema: StagePayloadSchema,
) -> StagePayloadEnvelope | dict[str, Any]:
    """Decode an envelope, or one release-cycle legacy raw payload.

    The generic connector decoder intentionally returns type-erased mappings.
    Only a mapping with every required envelope header is treated as a new
    envelope. All other mappings take the bounded legacy payload path and are
    returned without applying the new typed payload contract. This preserves
    the pre-envelope compatibility surface for one release cycle; callers must
    keep those values on their existing legacy projection path.
    """
    from vllm_omni.data_entry_keys import to_dict, to_struct
    from vllm_omni.distributed.omni_connectors.payload_schema import (
        StagePayloadContractError,
    )

    if isinstance(value, StagePayloadEnvelope):
        envelope = value
    elif isinstance(value, Mapping) and _ENVELOPE_WIRE_FIELDS.issubset(value):
        unknown_fields = set(value).difference(_ENVELOPE_WIRE_FIELDS)
        if unknown_fields:
            names = ", ".join(sorted(str(field) for field in unknown_fields))
            raise StagePayloadContractError(f"unknown stage payload envelope fields: {names}")
        try:
            route = msgspec.convert(value["route"], StageRoute)
            identity = msgspec.convert(value["identity"], StagePayloadIdentity)
            boundary = msgspec.convert(value["boundary"], StageBoundary)
            raw_payload = value["payload"]
            if raw_payload is None or isinstance(raw_payload, OmniPayloadStruct):
                payload = raw_payload
            elif isinstance(raw_payload, Mapping):
                payload = to_struct(dict(raw_payload))
            else:
                raise TypeError("envelope payload must be a mapping, OmniPayloadStruct, or None")
            envelope = StagePayloadEnvelope(
                schema_version=value["schema_version"],
                route=route,
                identity=identity,
                chunk_seq=value["chunk_seq"],
                payload=payload,
                boundary=boundary,
            )
        except (KeyError, TypeError, ValueError, msgspec.ValidationError) as exc:
            raise StagePayloadContractError(f"invalid stage payload envelope header: {exc}") from exc
    elif isinstance(value, OmniPayloadStruct):
        return to_dict(value)
    elif isinstance(value, Mapping):
        return dict(value)
    else:
        raise StagePayloadContractError(f"stage payload wire value must be a mapping, received {type(value).__name__}")

    if envelope.schema_version != schema.schema_version:
        raise StagePayloadContractError("envelope schema_version does not match the selected edge schema")
    if envelope.route != schema.route:
        raise StagePayloadContractError("envelope route does not match the selected edge schema")
    return envelope


def _require_payload_or_boundary(
    payload: OmniPayloadStruct | None,
    boundary: StageBoundary,
) -> None:
    if payload is None and boundary is StageBoundary.NONE:
        raise ValueError("an emission must contain a payload or boundary")


__all__ = [
    "NoPayloadYet",
    "NormalizedStageOutput",
    "PayloadEmission",
    "ProcessorResult",
    "StageBoundary",
    "StagePayloadBuildContext",
    "StagePayloadEnvelope",
    "StagePayloadIdentity",
    "StagePayloadProcessor",
    "StageRequestView",
    "StageRoute",
    "StageSessionFence",
    "decode_stage_payload_wire",
    "try_normalize_stage_payload",
]
