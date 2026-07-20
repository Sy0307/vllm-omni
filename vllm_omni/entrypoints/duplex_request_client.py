# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lazy compatibility exports for the experimental duplex request client."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.experimental.fullduplex.request_client import (
        DuplexEnginePort,
        DuplexRequestClient,
        DuplexRequestOutputPort,
    )

__all__ = ["DuplexEnginePort", "DuplexRequestClient", "DuplexRequestOutputPort"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from vllm_omni.experimental.fullduplex import request_client

    return getattr(request_client, name)
