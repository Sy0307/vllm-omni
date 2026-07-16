# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility exports for the stable engine contract and MiniCPM planner."""

from vllm_omni.engine.duplex_runtime import (
    DuplexFenceMismatchError,
    DuplexInputAppend,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    DuplexStageBinding,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
    build_duplex_data_plane_prompt,
    duplex_first_append_context_reserve,
    duplex_first_append_unit_count,
    duplex_new_user_turn_prefix_reserve,
    duplex_payload_is_exact_chunks,
    duplex_scheduler_token_budget,
)

__all__ = [
    "DuplexFenceMismatchError",
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
