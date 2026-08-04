# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor loading and the bounded compatibility adapter."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

import torch

from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct
from vllm_omni.distributed.omni_connectors.payload_schema import (
    StagePayloadContractError,
    StagePayloadSchema,
)
from vllm_omni.distributed.omni_connectors.stage_payload import (
    NoPayloadYet,
    PayloadEmission,
    ProcessorResult,
    StageBoundary,
    StagePayloadBuildContext,
    StagePayloadEnvelope,
    StagePayloadIdentity,
    StagePayloadProcessor,
    StageRequestView,
    StageRoute,
)

logger = logging.getLogger(__name__)


class LegacyProcessorMode(str, Enum):
    ASYNC_CHUNK = "async_chunk"
    FULL_PAYLOAD = "full_payload"
    PASS_THROUGH = "pass_through"


class LegacyStagePayloadProcessorAdapter:
    """Call one legacy ABI shape without runtime signature probing."""

    def __init__(
        self,
        processor: Callable[..., OmniPayloadStruct | dict[str, Any] | None] | None,
        *,
        mode: LegacyProcessorMode,
        transfer_manager: object,
        request_provider: Callable[[StageRequestView], object],
        normalize_async_transport_flags: bool = False,
    ) -> None:
        if mode is not LegacyProcessorMode.PASS_THROUGH and processor is None:
            raise ValueError(f"legacy {mode.value} mode requires a processor")
        self._processor = processor
        self._mode = mode
        self._transfer_manager = transfer_manager
        self._request_provider = request_provider
        self._normalize_async_transport_flags = normalize_async_transport_flags
        if normalize_async_transport_flags and mode is not LegacyProcessorMode.ASYNC_CHUNK:
            raise ValueError("transport flag normalization is only valid for legacy async chunks")

    def process(self, context: StagePayloadBuildContext) -> ProcessorResult:
        if self._mode is LegacyProcessorMode.PASS_THROUGH:
            value = (
                context.source_output.output_payload
                if context.source_output.output_payload is not None
                else context.source_output.request_payload
            )
            return convert_legacy_result(value, context)

        request = self._request_provider(context.request)
        raw_output = context.source_output.raw_output
        if raw_output is None:
            raw_output = context.source_output.output_payload

        assert self._processor is not None
        if self._mode is LegacyProcessorMode.ASYNC_CHUNK:
            value = self._processor(
                transfer_manager=self._transfer_manager,
                multimodal_output=raw_output,
                request=request,
                is_finished=(context.request.terminal or context.boundary is not StageBoundary.NONE),
            )
        else:
            value = self._processor(
                transfer_manager=self._transfer_manager,
                pooling_output=raw_output,
                request=request,
            )
        if self._normalize_async_transport_flags and value is not None:
            value = _normalize_async_chunk_transport_flags(value, context)
        return convert_legacy_result(
            value,
            context,
            processor_type=_processor_type_name(self._processor),
        )

    def drop_request(self, identity: StagePayloadIdentity) -> None:
        del identity


def convert_legacy_result(
    value: OmniPayloadStruct | dict[str, Any] | None,
    context: StagePayloadBuildContext,
    *,
    processor_type: str | None = None,
) -> ProcessorResult:
    """Convert exactly one legacy return value into the new result union."""
    if value is None:
        if context.boundary is StageBoundary.NONE:
            return NoPayloadYet()
        return PayloadEmission(payload=None, boundary=context.boundary)

    payload = to_struct(value) if isinstance(value, dict) else value
    return PayloadEmission(
        payload=payload,
        boundary=_legacy_boundary(
            payload,
            fallback=context.boundary,
            processor_type=processor_type,
        ),
    )


def derive_legacy_transport_boundary(
    payload: OmniPayloadStruct | Mapping[str, Any],
) -> StageBoundary:
    """Read only the two flags owned by the old generic transport."""
    return _legacy_boundary(
        payload,
        fallback=StageBoundary.NONE,
        processor_type=None,
    )


def reconcile_envelope_legacy_boundary(
    envelope: StagePayloadEnvelope,
) -> None:
    """Reject true deprecated flags that disagree with envelope authority."""
    if envelope.payload is None:
        return
    derived = derive_legacy_transport_boundary(envelope.payload)
    if derived is StageBoundary.NONE:
        return
    if derived is not envelope.boundary:
        raise StagePayloadContractError("deprecated payload boundary flags disagree with the envelope boundary")


def load_stage_payload_processor(path: str) -> StagePayloadProcessor:
    """Resolve one zero-argument processor factory or processor object."""
    resolved = _load_dotted_object(path)
    candidate = resolved
    if not callable(getattr(candidate, "process", None)):
        if not callable(resolved):
            raise TypeError(f"stage payload processor {path!r} is neither a processor nor a factory")
        candidate = resolved()
    if not callable(getattr(candidate, "process", None)):
        raise TypeError(f"stage payload processor {path!r} has no callable process()")
    if not callable(getattr(candidate, "drop_request", None)):
        raise TypeError(f"stage payload processor {path!r} has no callable drop_request()")
    return candidate


def load_stage_payload_schema(
    path: str,
    *,
    expected_route: StageRoute | None = None,
) -> StagePayloadSchema:
    """Resolve and route-check one explicit edge schema."""
    resolved = _load_dotted_object(path)
    candidate = resolved() if callable(resolved) else resolved
    if not isinstance(candidate, StagePayloadSchema):
        raise TypeError(f"stage payload schema {path!r} is not a StagePayloadSchema")
    if expected_route is not None and candidate.route != expected_route:
        raise ValueError(
            f"stage payload schema route {candidate.route} does not match configured route {expected_route}"
        )
    return candidate


def _load_dotted_object(path: str) -> Any:
    if not path or "." not in path:
        raise ValueError("a stage payload object path must be a dotted import path")
    module_path, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"module {module_path!r} has no stage payload object {object_name!r}") from exc


def _normalize_async_chunk_transport_flags(
    value: OmniPayloadStruct | dict[str, Any],
    context: StagePayloadBuildContext,
) -> OmniPayloadStruct:
    """Reproduce the legacy chunk adapter's transport-owned flag writes.

    Some model processors use ``meta.finished`` for a downstream codec turn,
    while the old chunk adapter overwrote that field with the actual request
    terminal state before transport. Preserve that behavior inside the
    compatibility bridge so model-level boundaries cannot close a duplex
    stream accidentally.
    """
    payload = to_struct(value) if isinstance(value, dict) else value
    payload_dict = to_dict(payload)
    meta = dict(payload_dict.get("meta") or {})
    meta["finished"] = torch.tensor(
        context.boundary is StageBoundary.STREAM_END,
        dtype=torch.bool,
    )
    if meta.get("is_segment_finished") is None:
        meta["is_segment_finished"] = torch.tensor(
            context.boundary is StageBoundary.SEGMENT_END,
            dtype=torch.bool,
        )
    payload_dict["meta"] = meta
    return to_struct(payload_dict)


def _legacy_boundary(
    payload: OmniPayloadStruct | Mapping[str, Any],
    *,
    fallback: StageBoundary,
    processor_type: str | None,
) -> StageBoundary:
    if isinstance(payload, Mapping):
        raw_meta = payload.get("meta")
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        finished = meta.get("finished")
        segment_finished = meta.get("is_segment_finished")
    else:
        meta = payload.meta
        finished = meta.finished if meta is not None else None
        segment_finished = meta.is_segment_finished if meta is not None else None

    finished_true = _is_truthy_scalar(finished)
    segment_true = _is_truthy_scalar(segment_finished)
    if finished_true:
        if segment_true:
            _warn_conflicting_legacy_boundaries(processor_type)
        return StageBoundary.STREAM_END
    if segment_true:
        return StageBoundary.SEGMENT_END
    if segment_finished is not None and fallback is StageBoundary.SEGMENT_END:
        return StageBoundary.NONE
    return fallback


def _is_truthy_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and bool(value.item())
    return bool(value) if value is not None else False


_WARNED_CONFLICTING_BOUNDARIES: set[str] = set()


def _warn_conflicting_legacy_boundaries(processor_type: str | None) -> None:
    key = processor_type or "<unknown>"
    if key in _WARNED_CONFLICTING_BOUNDARIES:
        return
    _WARNED_CONFLICTING_BOUNDARIES.add(key)
    logger.warning(
        "Legacy stage payload processor %s emitted both finished and "
        "is_segment_finished; treating the result as STREAM_END. Migrate the "
        "processor to an explicit StageBoundary.",
        key,
    )


def _processor_type_name(processor: Callable[..., Any]) -> str:
    return f"{processor.__module__}.{getattr(processor, '__qualname__', type(processor).__qualname__)}"


__all__ = [
    "LegacyProcessorMode",
    "LegacyStagePayloadProcessorAdapter",
    "convert_legacy_result",
    "derive_legacy_transport_boundary",
    "load_stage_payload_processor",
    "load_stage_payload_schema",
    "reconcile_envelope_legacy_boundary",
]
