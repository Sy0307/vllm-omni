# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

import vllm_omni.utils.device_copy as device_copy

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def test_index_ring_wrap_waits_for_queued_copies(monkeypatch):
    monkeypatch.setattr(device_copy._PinnedRing, "SIZE", 8)
    monkeypatch.setattr(device_copy, "_RING", None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the stream busy so the first copy is still queued at the wrap.
        torch.cuda._sleep(200_000_000)
        first = device_copy.index_to_device([17] * 6, "cuda")
        second = device_copy.index_to_device([99] * 6, "cuda")
    torch.accelerator.synchronize()
    assert first.tolist() == [17] * 6
    assert second.tolist() == [99] * 6
