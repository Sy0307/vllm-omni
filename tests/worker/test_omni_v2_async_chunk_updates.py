# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.core.sched.output import OmniCachedRequestData
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner


class _Array:
    def __init__(self):
        self.np = [0]
        self.writes = []

    def stage_write_elem(self, idx, value):
        self.writes.append((idx, value))


class _TokenIds:
    def __init__(self):
        self.writes = []

    def stage_write(self, idx, start, values):
        self.writes.append((idx, start, list(values)))


class _ReqStates:
    def __init__(self):
        self.req_id_to_index = {"r1": 0}
        self.prompt_len = SimpleNamespace(np=[0])
        self.prefill_len = SimpleNamespace(np=[0])
        self.total_len = _Array()
        self.all_token_ids = _TokenIds()
        self.num_computed_tokens = _Array()
        self.num_computed_prefill_tokens = [7]
        self.applied = False

    def apply_staged_writes(self):
        self.applied = True


class _IntermediateBuffer:
    def __init__(self):
        self.buffers = [{"stale": "value", "req_id": "r1"}]

    def remove_request(self, idx):
        self.buffers[idx] = {}


def test_async_chunk_update_clears_stale_intermediate_buffer():
    runner = object.__new__(OmniGenerationModelRunner)
    runner.req_states = _ReqStates()
    runner.model_state = SimpleNamespace(intermediate_buffer=_IntermediateBuffer())
    cached = OmniCachedRequestData(
        req_ids=["r1"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
        prompt_token_ids={"r1": [1, 2, 3]},
        additional_information={"r1": {"fresh": "value"}},
    )

    runner._handle_async_chunk_updates(SimpleNamespace(scheduled_cached_reqs=cached))

    assert runner.model_state.intermediate_buffer.buffers[0] == {}
    assert runner.req_states.prompt_len.np[0] == 3
    assert runner.req_states.prefill_len.np[0] == 3
    assert runner.req_states.all_token_ids.writes == [(0, 0, [1, 2, 3])]
    assert runner.req_states.num_computed_prefill_tokens[0] == 0
    assert runner.req_states.applied is True


if __name__ == "__main__":
    test_async_chunk_update_clears_stale_intermediate_buffer()
