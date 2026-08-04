# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-owned validation and accumulation for stage payload envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import msgspec
import torch

from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct
from vllm_omni.distributed.omni_connectors.stage_payload import (
    StageBoundary,
    StagePayloadEnvelope,
    StagePayloadIdentity,
    StageRoute,
    StageSessionFence,
)
from vllm_omni.distributed.omni_connectors.tensor_span import (
    get_tensor_span,
    merge_tensor_spans,
)

_EMBED_SPAN_GROUPS: tuple[tuple[str, str, str], ...] = (("decode", "decode_token_start", "decode_token_end"),)


class StagePayloadContractError(ValueError):
    """An envelope does not match its configured pipeline edge contract."""


class StagePayloadValidationError(StagePayloadContractError):
    """A payload field is absent from, or invalid for, the edge schema."""


class StagePayloadSequenceError(StagePayloadContractError):
    """An envelope skipped a request-global sequence number."""


class PayloadMergeMode(str, Enum):
    DELTA = "delta"
    SNAPSHOT = "snapshot"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class TensorFieldConstraint:
    rank: int
    dtypes: frozenset[torch.dtype]
    concat_dim: int = 0

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("tensor rank must be non-negative")
        if not self.dtypes:
            raise ValueError("tensor dtypes must not be empty")
        if self.rank == 0 and self.concat_dim != 0:
            raise ValueError("a scalar tensor cannot declare a concat dimension")
        if self.rank > 0 and not -self.rank <= self.concat_dim < self.rank:
            raise ValueError("concat_dim must address the declared tensor rank")


@dataclass(frozen=True, slots=True)
class StagePayloadFieldRule:
    mode: PayloadMergeMode
    tensor: TensorFieldConstraint | None = None


@dataclass(frozen=True, slots=True)
class StagePayloadSchema:
    schema_version: int
    route: StageRoute
    fields: Mapping[tuple[str, ...], StagePayloadFieldRule]
    legacy: bool = False

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")

        copied: dict[tuple[str, ...], StagePayloadFieldRule] = {}
        for raw_path, rule in self.fields.items():
            path = tuple(raw_path)
            if not path or any(not isinstance(part, str) or not part for part in path):
                raise ValueError("field path components must be non-empty strings")
            if path in copied:
                raise ValueError(f"duplicate field path: {'.'.join(path)}")
            copied[path] = rule
        object.__setattr__(self, "fields", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class StagePayloadApplyResult:
    payload: OmniPayloadStruct | None
    boundary: StageBoundary
    duplicate: bool


@dataclass(frozen=True, slots=True)
class _LegacyStagePayloadSchemaTemplate:
    schema_version: int = 1

    def bind(self, route: StageRoute) -> StagePayloadSchema:
        return StagePayloadSchema(
            schema_version=self.schema_version,
            route=route,
            fields={},
            legacy=True,
        )


LEGACY_STAGE_PAYLOAD_SCHEMA = _LegacyStagePayloadSchemaTemplate()


class StagePayloadAccumulator:
    """Validate and atomically merge envelopes for one destination request."""

    def __init__(
        self,
        schema: StagePayloadSchema,
        *,
        expected_external_request_id: str,
        expected_session_fence: StageSessionFence | None,
    ) -> None:
        if not expected_external_request_id.strip():
            raise ValueError("expected_external_request_id must not be empty")
        self._schema = schema
        self._expected_external_request_id = expected_external_request_id
        self._expected_session_fence = expected_session_fence
        self._source_identity: StagePayloadIdentity | None = None
        self._next_chunk_seq = 0
        self._payload: OmniPayloadStruct | None = None

    @property
    def next_chunk_seq(self) -> int:
        return self._next_chunk_seq

    def apply(self, envelope: StagePayloadEnvelope) -> StagePayloadApplyResult:
        self._validate_envelope_contract(envelope)

        if envelope.chunk_seq < self._next_chunk_seq:
            return StagePayloadApplyResult(
                payload=self._payload,
                boundary=StageBoundary.NONE,
                duplicate=True,
            )
        if envelope.chunk_seq > self._next_chunk_seq:
            raise StagePayloadSequenceError(
                f"chunk_seq gap: expected {self._next_chunk_seq}, received {envelope.chunk_seq}"
            )

        merged_payload = self._merge_payload(envelope.payload)

        # Commit identity, payload, and sequence only after every field has
        # validated and the complete candidate can be reconstructed.
        if self._source_identity is None:
            self._source_identity = envelope.identity
        self._payload = merged_payload
        self._next_chunk_seq += 1
        return StagePayloadApplyResult(
            payload=merged_payload,
            boundary=envelope.boundary,
            duplicate=False,
        )

    def _validate_envelope_contract(self, envelope: StagePayloadEnvelope) -> None:
        if envelope.schema_version != self._schema.schema_version:
            raise StagePayloadContractError(
                f"schema_version mismatch: expected {self._schema.schema_version}, received {envelope.schema_version}"
            )
        if envelope.route != self._schema.route:
            raise StagePayloadContractError(f"route mismatch: expected {self._schema.route}, received {envelope.route}")
        if envelope.identity.external_request_id != self._expected_external_request_id:
            raise StagePayloadContractError(
                "external_request_id mismatch: "
                f"expected {self._expected_external_request_id!r}, "
                f"received {envelope.identity.external_request_id!r}"
            )
        if envelope.identity.session_fence != self._expected_session_fence:
            raise StagePayloadContractError(
                "session_fence mismatch: "
                f"expected {self._expected_session_fence!r}, "
                f"received {envelope.identity.session_fence!r}"
            )
        if self._source_identity is not None:
            if envelope.identity.source_request_id != self._source_identity.source_request_id:
                raise StagePayloadContractError("source_request_id changed after the first accepted envelope")
            if envelope.identity != self._source_identity:
                raise StagePayloadContractError("source identity changed after the first accepted envelope")

    def _merge_payload(
        self,
        incoming: OmniPayloadStruct | None,
    ) -> OmniPayloadStruct | None:
        if incoming is None:
            return self._payload

        if self._schema.legacy:
            return _merge_legacy_payload(self._payload, incoming)

        incoming_dict = to_dict(incoming)
        fields = list(_iter_declared_payload_fields(incoming_dict, self._schema))
        candidate = _copy_payload_dict(self._payload)

        for path, value in fields:
            rule = self._schema.fields.get(path)
            if rule is None:
                raise StagePayloadValidationError(f"field {'.'.join(path)} is not declared by the edge schema")
            _validate_field(path, value, rule)
            previous = _get_path(candidate, path)
            merged = _merge_field(path, previous, value, rule)
            _set_path(candidate, path, merged)

        try:
            return to_struct(candidate)
        except (msgspec.ValidationError, TypeError, ValueError) as exc:
            raise StagePayloadValidationError(f"merged payload does not match OmniPayloadStruct: {exc}") from exc


def _iter_declared_payload_fields(
    payload: Mapping[str, Any],
    schema: StagePayloadSchema,
    prefix: tuple[str, ...] = (),
):
    for key, value in payload.items():
        path = (*prefix, key)
        if path in schema.fields:
            yield path, value
            continue
        if isinstance(value, Mapping):
            yielded = False
            for nested in _iter_declared_payload_fields(value, schema, path):
                yielded = True
                yield nested
            if yielded:
                continue
        yield path, value


def validate_payload_for_schema(
    payload: OmniPayloadStruct | None,
    schema: StagePayloadSchema,
) -> None:
    """Validate one emitted payload before it reaches the connector."""
    if payload is None:
        return
    if schema.legacy:
        try:
            to_struct(to_dict(payload))
        except (msgspec.ValidationError, TypeError, ValueError) as exc:
            raise StagePayloadValidationError(f"legacy payload does not match OmniPayloadStruct: {exc}") from exc
        return
    payload_dict = to_dict(payload)
    for path, value in _iter_declared_payload_fields(payload_dict, schema):
        rule = schema.fields.get(path)
        if rule is None:
            raise StagePayloadValidationError(f"field {'.'.join(path)} is not declared by the edge schema")
        _validate_field(path, value, rule)


def _copy_payload_dict(payload: OmniPayloadStruct | None) -> dict[str, Any]:
    if payload is None:
        return {}
    copied: dict[str, Any] = {}
    for key, value in to_dict(payload).items():
        copied[key] = dict(value) if isinstance(value, Mapping) else value
    return copied


def _merge_legacy_payload(
    existing: OmniPayloadStruct | None,
    incoming: OmniPayloadStruct,
) -> OmniPayloadStruct:
    """Reproduce the pre-envelope generic append/override behavior."""
    incoming_dict = to_dict(incoming)
    if existing is None:
        return to_struct(_copy_plain_payload_dict(incoming_dict))

    origin = to_dict(existing)
    merged = _copy_plain_payload_dict(origin)
    raw_override_keys = incoming_dict.get("meta", {}).get("override_keys", [])
    override_keys = {tuple(key) if isinstance(key, list) else key for key in raw_override_keys}

    for key, value in incoming_dict.items():
        if isinstance(value, Mapping):
            origin_sub = origin.get(key)
            merged_sub = dict(origin_sub) if isinstance(origin_sub, Mapping) else {}
            span_handled: set[str] = set()
            if key == "embed" and isinstance(origin_sub, Mapping):
                for tensor_key, start_key, end_key in _EMBED_SPAN_GROUPS:
                    if tensor_key not in value or (key, tensor_key) in override_keys:
                        continue
                    span = merge_tensor_spans(
                        get_tensor_span(
                            origin_sub,
                            tensor_key=tensor_key,
                            start_key=start_key,
                            end_key=end_key,
                        ),
                        get_tensor_span(
                            value,
                            tensor_key=tensor_key,
                            start_key=start_key,
                            end_key=end_key,
                        ),
                    )
                    if span is None:
                        continue
                    tensor, start, end = span
                    merged_sub[tensor_key] = tensor
                    merged_sub[start_key] = start
                    merged_sub[end_key] = end
                    span_handled.update((tensor_key, start_key, end_key))
            for qualifier, current in value.items():
                if qualifier in span_handled:
                    continue
                if key == "meta" and qualifier == "finished":
                    merged_sub[qualifier] = current
                    continue
                if (key, qualifier) in override_keys:
                    merged_sub[qualifier] = current
                    continue
                previous = merged_sub.get(qualifier)
                merged_sub[qualifier] = _legacy_merge_value(previous, current)
            merged[key] = merged_sub
            continue

        if key in override_keys:
            merged[key] = value
        else:
            merged[key] = _legacy_merge_value(origin.get(key), value)

    try:
        return to_struct(merged)
    except (msgspec.ValidationError, TypeError, ValueError) as exc:
        raise StagePayloadValidationError(f"legacy payload does not match OmniPayloadStruct: {exc}") from exc


def _copy_plain_payload_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: dict(value) if isinstance(value, Mapping) else value for key, value in payload.items()}


def _legacy_merge_value(previous: Any, current: Any) -> Any:
    if isinstance(previous, torch.Tensor) and isinstance(current, torch.Tensor):
        try:
            return torch.cat((previous, current), dim=0)
        except RuntimeError as exc:
            raise StagePayloadValidationError(f"legacy tensor delta cannot be concatenated: {exc}") from exc
    if isinstance(previous, list) and isinstance(current, list):
        return [*previous, *current]
    return current


def _validate_field(
    path: tuple[str, ...],
    value: Any,
    rule: StagePayloadFieldRule,
) -> None:
    constraint = rule.tensor
    if constraint is None:
        return
    if not isinstance(value, torch.Tensor):
        raise StagePayloadValidationError(f"field {'.'.join(path)} must be a torch.Tensor")
    if value.ndim != constraint.rank:
        raise StagePayloadValidationError(
            f"field {'.'.join(path)} must have rank {constraint.rank}; received rank {value.ndim}"
        )
    if value.dtype not in constraint.dtypes:
        expected = ", ".join(sorted(str(dtype) for dtype in constraint.dtypes))
        raise StagePayloadValidationError(
            f"field {'.'.join(path)} must have dtype in {{{expected}}}; received {value.dtype}"
        )


def _merge_field(
    path: tuple[str, ...],
    previous: Any,
    current: Any,
    rule: StagePayloadFieldRule,
) -> Any:
    if rule.mode in (PayloadMergeMode.SNAPSHOT, PayloadMergeMode.REPLACE):
        return current
    if previous is None:
        return list(current) if isinstance(current, list) else current
    if isinstance(previous, torch.Tensor) and isinstance(current, torch.Tensor):
        concat_dim = rule.tensor.concat_dim if rule.tensor is not None else 0
        try:
            return torch.cat((previous, current), dim=concat_dim)
        except RuntimeError as exc:
            raise StagePayloadValidationError(f"field {'.'.join(path)} cannot be concatenated: {exc}") from exc
    if isinstance(previous, list) and isinstance(current, list):
        return [*previous, *current]
    raise StagePayloadValidationError(f"DELTA field {'.'.join(path)} must contain tensors or lists")


def _get_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current = payload
    for part in path[:-1]:
        nested = current.get(part)
        if nested is None:
            nested = {}
            current[part] = nested
        if not isinstance(nested, dict):
            raise StagePayloadValidationError(f"field {'.'.join(path)} conflicts with scalar parent {part}")
        current = nested
    current[path[-1]] = value


__all__ = [
    "LEGACY_STAGE_PAYLOAD_SCHEMA",
    "PayloadMergeMode",
    "StagePayloadAccumulator",
    "StagePayloadApplyResult",
    "StagePayloadContractError",
    "StagePayloadFieldRule",
    "StagePayloadSchema",
    "StagePayloadSequenceError",
    "StagePayloadValidationError",
    "TensorFieldConstraint",
    "validate_payload_for_schema",
]
