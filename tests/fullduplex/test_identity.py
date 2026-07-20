# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence as ExperimentalDuplexFence
from vllm_omni.experimental.fullduplex.engine.duplex_types import DuplexFence as EngineDuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fence_is_frozen_and_slotted():
    fence = ExperimentalDuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")


def test_core_identity_reexports_engine_duplex_fence():
    assert ExperimentalDuplexFence is EngineDuplexFence
