# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.compilation.cuda_graph import CUDAGraphMode

from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner


def test_profile_dispatch_uses_eager_without_cudagraph_manager():
    runner = object.__new__(OmniGPUModelRunner)
    runner.cudagraph_manager = None
    runner.dp_size = 1
    runner.dp_rank = 0

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1,
        num_toks=8,
        uniform_tok_count=8,
        is_profile=True,
    )

    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    assert batch_desc.num_tokens == 8
    assert batch_desc.num_reqs == 1
    assert num_tokens_across_dp is None


def test_non_profile_dispatch_uses_cudagraph_manager():
    runner = object.__new__(OmniGPUModelRunner)
    expected = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=1)
    calls = []
    runner.cudagraph_manager = SimpleNamespace(dispatch=lambda *args: calls.append(args) or expected)
    runner.dp_size = 1
    runner.dp_rank = 0

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1,
        num_toks=8,
        uniform_tok_count=8,
        is_profile=False,
    )

    assert batch_desc is expected
    assert calls == [(1, 8, 8)]
    assert num_tokens_across_dp is None


if __name__ == "__main__":
    test_profile_dispatch_uses_eager_without_cudagraph_manager()
    test_non_profile_dispatch_uses_cudagraph_manager()


def test_fullgraph_requires_cudagraph_manager_contract():
    source = __import__("inspect").getsource(OmniGPUModelRunner.execute_model)

    assert "assert self.cudagraph_manager is not None" in source
