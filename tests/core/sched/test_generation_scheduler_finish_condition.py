# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("native,terminal,expected", [(True, False, False), (True, True, True), (False, False, True)])
def test_terminal_arrival_does_not_finish_previous_native_output(native, terminal, expected):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._native_data_plane = native
    scheduler.chunk_transfer_adapter = None if native else SimpleNamespace(is_done_receiving_chunks=lambda _: True)
    scheduler.input_coordinator = SimpleNamespace(finished_requests={"r1"})
    output = SimpleNamespace(input_terminal_req_ids={"r1"} if terminal else set())
    assert scheduler._input_execution_is_terminal("r1", output) is expected
