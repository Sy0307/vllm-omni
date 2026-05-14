# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(slots=True)
class OmniPreparedStep:
    request_ids: list[str]
    token_slices: list[slice] = field(default_factory=list)
    fallback_request_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class OmniStepRunner(Protocol):
    @classmethod
    def from_runner(cls, runner: Any) -> OmniStepRunner:
        ...

    def supports_step(
        self,
        *,
        runner: Any,
        request_ids: list[str],
        num_scheduled_tokens: Sequence[int],
        is_prefill_by_req: Mapping[str, bool],
    ) -> bool:
        ...

    def prepare_step(
        self,
        *,
        request_ids: list[str],
        runner: Any,
        input_ids: torch.Tensor,
        req_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> OmniPreparedStep:
        ...

    def run_step(
        self,
        *,
        prepared: OmniPreparedStep,
        runner: Any,
    ) -> None:
        ...

    def commit_step(
        self,
        *,
        prepared: OmniPreparedStep,
        runner: Any,
        inputs_embeds: torch.Tensor,
    ) -> None:
        ...

    def free_request(self, request_id: str) -> None:
        ...
