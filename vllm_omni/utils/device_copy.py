# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Host-to-device copies that do not block the host on queued GPU work.

A copy from pageable host memory (``tensor.to("cuda")``, ``torch.tensor(...,
device="cuda")``, ``torch.as_tensor(..., device="cuda")``) makes the CPU wait
until every kernel already queued on the stream has finished. In a per-step
path under async scheduling that removes the CPU/GPU overlap the batch queue
exists for. Staging the data in
pinned memory and copying with ``non_blocking`` keeps the host running ahead;
the caching host allocator keeps the pinned source alive until the copy's
stream event completes.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import torch


def tensor_to_device(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """``tensor.to(device)`` for a host tensor, without a host sync."""
    if tensor.device.type != "cpu" or torch.device(device).type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class _PinnedRing:
    """Pinned int64 staging reused round-robin for small index lists.

    Building a pinned tensor per call costs tens of microseconds, which adds
    up over the handful of index lists every engine step uploads. Rows are
    handed out in order and only reused after the ring wraps. Nothing bounds
    how long a queued copy stays pending, so a wrap first waits for the device
    to drain; at a few hundred indices per step that is one sync every few
    thousand steps.
    """

    SIZE = 1 << 20

    def __init__(self) -> None:
        self.buf = torch.empty(self.SIZE, dtype=torch.long, pin_memory=True)
        self.np = self.buf.numpy()
        self.pos = 0
        self.lock = threading.Lock()

    def stage(self, values: Sequence[int]) -> torch.Tensor | None:
        n = len(values)
        with self.lock:
            if self.pos + n > self.SIZE:
                if torch.cuda.is_current_stream_capturing():
                    return None
                from vllm_omni.platforms import current_omni_platform

                current_omni_platform.synchronize()
                self.pos = 0
            start = self.pos
            self.pos += n
        self.np[start : start + n] = values
        return self.buf[start : start + n]


_RING: _PinnedRing | None = None


def index_to_device(values: Sequence[int], device: torch.device | str, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """A host index list as a device tensor, without a host sync."""
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    global _RING
    if _RING is None:
        _RING = _PinnedRing()
    staged = _RING.stage(values) if 0 < len(values) <= 4096 else None
    if staged is None:
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    out = staged.to(device, non_blocking=True)
    return out if dtype == torch.long else out.to(dtype)
