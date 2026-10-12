# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned UVA staging for the MRV2 runner's per-step host-to-device uploads.

Upstream stages each small per-step upload (staged-write contents, the batch
index mapping, query start locations) in a freshly allocated pinned tensor
and issues an H2D copy. Each call costs several torch ops and a host caching
allocator round trip on the engine thread. These helpers write the values
into preallocated pinned, device-mapped ring buffers instead.

A ring slot is rewritten only ``depth`` uploads later, after the GPU work of
earlier steps that read it has completed (the same lifetime contract as the
upstream UVA index pools these uploads accompany).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBufferPool


class GrowOnlyUvaBufferPool(UvaBufferPool):
    """A UVA ring whose buffers replaced by growth stay alive.

    An earlier step's kernel may still read a replaced buffer, and pinned
    memory returned to the host caching allocator can be handed out again
    before it does. Growth doubles capacity, so few buffers are retained.
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int) -> None:
        super().__init__(size, dtype, max_concurrency)
        self._retired: list[object] = []

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        slot = (self._curr + 1) % self.max_concurrency
        if len(x) > self._uva_bufs[slot].cpu.shape[0]:
            self._retired.append(self._uva_bufs[slot])
        return super().copy_to_uva(x)


def stage_write_contents_in_uva(tensors: Sequence[StagedWriteTensor]) -> None:
    """Route staged-write contents through UVA, as upstream does for UVA tensors."""
    if not is_uva_available():
        return
    for tensor in tensors:
        if tensor.write_contents is None:
            tensor.write_contents = GrowOnlyUvaBufferPool(tensor.num_rows, tensor.dtype, tensor.max_concurrency)


class UvaUploads:
    """Drop-in ``async_tensor_h2d`` for 1-D NumPy uploads through UVA rings.

    Other inputs keep the upstream implementation. Results are owned device
    tensors (or ``out``), exactly as upstream returns them.
    """

    def __init__(self, depth: int) -> None:
        self._depth = depth
        self._pools: dict[torch.dtype, GrowOnlyUvaBufferPool] = {}

    def __call__(
        self,
        data: list | np.ndarray | torch.Tensor,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dtype is None and out is not None:
            dtype = out.dtype
        if not isinstance(data, np.ndarray) or data.ndim != 1 or dtype is None:
            return async_tensor_h2d(data, device=device, dtype=dtype, out=out)
        pool = self._pools.get(dtype)
        if pool is None:
            pool = self._pools[dtype] = GrowOnlyUvaBufferPool(max(len(data), 256), dtype, self._depth)
        staged = pool.copy_to_uva(data)
        if out is not None:
            return out.copy_(staged, non_blocking=True)
        return staged.clone()
