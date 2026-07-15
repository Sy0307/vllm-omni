# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex_types import DuplexFence as StableDuplexFence
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence as ExperimentalDuplexFence


def test_fence_is_frozen_and_slotted():
    fence = ExperimentalDuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")


def test_experimental_fence_reexports_stable_engine_type():
    assert ExperimentalDuplexFence is StableDuplexFence
