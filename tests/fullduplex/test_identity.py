# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence


def test_fence_is_frozen_and_slotted():
    fence = DuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")
