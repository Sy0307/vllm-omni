# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental duplex runtime loading and compatibility exports."""

from __future__ import annotations

from importlib import import_module

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexExecutionProfile,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
    DuplexRuntimeCapabilities,
    DuplexRuntimeExtension,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_resource_request_belongs_to_session,
    duplex_resource_request_id,
)


def duplex_execution_profile(extension: object | None) -> DuplexExecutionProfile:
    if extension is None:
        return DuplexExecutionProfile()
    factory = getattr(extension, "execution_profile", None)
    if not callable(factory):
        return DuplexExecutionProfile()
    profile = factory()
    if not isinstance(profile, DuplexExecutionProfile):
        raise TypeError("Duplex runtime extension execution_profile() must return DuplexExecutionProfile")
    return profile


def duplex_resource_lease_providers(extension: object | None) -> tuple[object, ...]:
    if extension is None:
        return ()
    factory = getattr(extension, "resource_lease_providers", None)
    if not callable(factory):
        return ()
    providers = factory()
    if not isinstance(providers, tuple):
        raise TypeError("Duplex runtime extension resource_lease_providers() must return a tuple")
    return providers


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
    return typed_extension


from vllm_omni.experimental.fullduplex.engine.duplex_session import (  # noqa: E402, F401
    DuplexAppendReservation,
    DuplexCompletedAppend,
    DuplexFenceMismatchError,
    DuplexInputAppend,
    DuplexRequestResource,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    DuplexStageBinding,
)
from vllm_omni.experimental.fullduplex.engine.lease import (  # noqa: E402, F401
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
    DuplexSessionExpiry,
)
from vllm_omni.experimental.fullduplex.engine.model_events import (  # noqa: E402, F401
    DuplexEventProtocolError,
    DuplexListen,
    DuplexModelEvent,
    DuplexOutputLedger,
    DuplexSpeakChunk,
    DuplexSpeakEnd,
    DuplexSpeakStart,
)

__all__ = [
    "DuplexAppendPlan",
    "DuplexAppendReservation",
    "DuplexCompletedAppend",
    "DuplexEventProtocolError",
    "DuplexExecutionProfile",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexInputMode",
    "DuplexLeaseActivity",
    "DuplexLeaseConfig",
    "DuplexLeaseState",
    "DuplexListen",
    "DuplexModelEvent",
    "DuplexOutputAction",
    "DuplexOutputDecision",
    "DuplexOutputLedger",
    "DuplexRequestResource",
    "DuplexRuntimeCapabilities",
    "DuplexRuntimeExtension",
    "DuplexSessionExpiry",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
    "DuplexSpeakChunk",
    "DuplexSpeakEnd",
    "DuplexSpeakStart",
    "SessionMode",
    "duplex_data_plane_request_info",
    "duplex_execution_profile",
    "duplex_resource_lease_providers",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_id",
    "load_duplex_runtime_extension",
    "validate_duplex_runtime_extension",
]
