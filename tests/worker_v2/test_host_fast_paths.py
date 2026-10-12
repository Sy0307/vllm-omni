# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UVA staging of the runner's per-step uploads matches the upstream uploads."""

import numpy as np
import pytest
import torch
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

from vllm_omni.worker_v2.host_fast_paths import GrowOnlyUvaBufferPool, UvaUploads, stage_write_contents_in_uva

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available() or not is_uva_available(), reason="needs CUDA UVA"),
]


def test_uploads_match_upstream_for_device_and_out_targets():
    uploads = UvaUploads(depth=4)
    rng = np.random.default_rng(0)
    results = []
    # Sizes cross the initial capacity, so the ring grows while earlier
    # uploads may still be in flight.
    for n in (3, 200, 257, 1000, 5, 1000, 2):
        data = rng.integers(0, 1 << 20, n).astype(np.intp)
        got = uploads(data, device="cuda", dtype=torch.int32)
        out = torch.full((n + 3,), -1, dtype=torch.int32, device="cuda")
        returned = uploads(data, out=out[:n])
        want = async_tensor_h2d(data, device="cuda", dtype=torch.int32)
        results.append((got, out, returned, want, n))
    torch.accelerator.synchronize()
    for got, out, returned, want, n in results:
        assert got.dtype == torch.int32 and got.is_cuda
        assert got.data_ptr() != out.data_ptr()
        assert torch.equal(got, want)
        assert returned.data_ptr() == out.data_ptr()
        assert torch.equal(out[:n], want) and out[n:].eq(-1).all()


def test_other_inputs_keep_the_upstream_upload():
    uploads = UvaUploads(depth=4)
    for data in ([1, 2, 3], torch.tensor([4, 5]), np.zeros((2, 2), dtype=np.int64)):
        got = uploads(data, device="cuda", dtype=torch.int64)
        assert torch.equal(got.cpu(), torch.as_tensor(np.asarray(data)).to(torch.int64))
    assert not uploads._pools


def test_growth_keeps_replaced_buffers_alive():
    pool = GrowOnlyUvaBufferPool(4, torch.int32, max_concurrency=2)
    first = pool.copy_to_uva(np.arange(3))
    replaced = pool._uva_bufs[(pool._curr + 1) % 2]
    pool.copy_to_uva(np.arange(9))  # grows the other slot
    grown = pool.copy_to_uva(np.arange(20))  # grows the first slot
    assert any(buf is replaced for buf in pool._retired)
    assert len(pool._retired) == 2
    torch.accelerator.synchronize()
    assert first.tolist() == [0, 1, 2] and grown.tolist() == list(range(20))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_staged_writes_through_uva_match_the_upload_path(dtype):
    device = torch.device("cuda")
    reference = StagedWriteTensor((6, 8), dtype, device)
    candidate = StagedWriteTensor((6, 8), dtype, device)
    stage_write_contents_in_uva([candidate])
    assert reference.write_contents is None and candidate.write_contents is not None
    rng = np.random.default_rng(1)
    for _ in range(5):
        for tensor in (reference, candidate):
            tensor.stage_write_elem(1, 7)
        rows = rng.choice(6, 3, replace=False)
        for row in rows:
            start = int(rng.integers(0, 4))
            values = rng.integers(-50, 50, int(rng.integers(1, 9 - start))).tolist()
            for tensor in (reference, candidate):
                tensor.stage_write(int(row), start, values)
        for tensor in (reference, candidate):
            tensor.apply_write()
    torch.accelerator.synchronize()
    assert torch.equal(reference.gpu, candidate.gpu)
