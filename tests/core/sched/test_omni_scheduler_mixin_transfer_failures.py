# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput, StageTransferFailure

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FailureAdapter:
    def __init__(self, failures):
        self.failures = failures
        self.drain_calls = 0

    def drain_transfer_failures(self):
        self.drain_calls += 1
        failures = self.failures
        self.failures = {}
        return failures


class _FailureScheduler(OmniSchedulerMixin):
    def __init__(self, connector_output, adapter_failures=None):
        self._latest_omni_connector_output = connector_output
        self.input_coordinator = None
        self.chunk_transfer_adapter = _FailureAdapter(adapter_failures or {})
        self.requests = {
            "internal-req-1": SimpleNamespace(
                request_id="internal-req-1",
                resumable=True,
                stop_reason=None,
            )
        }
        self.finish_calls = []

    def finish_requests(self, req_ids, status):
        self.finish_calls.append((set(req_ids), status))


def _failure(internal_request_id="internal-req-1"):
    return StageTransferFailure(
        internal_request_id=internal_request_id,
        external_request_id="external-req-1",
        source_stage=0,
        destination_stage=1,
        code="stage_payload_processor_failed",
        message="legacy processor raised ValueError",
    )


def test_transfer_failure_is_consumed_before_missing_coordinator_early_return():
    failure = _failure()
    scheduler = _FailureScheduler(
        OmniConnectorOutput(
            transfer_failures={"internal-req-1": failure},
        )
    )

    scheduler._consume_pending_connector_output("ar")

    assert len(scheduler.finish_calls) == 1
    req_ids, status = scheduler.finish_calls[0]
    assert req_ids == {"internal-req-1"}
    assert getattr(status, "name", str(status)).endswith("FINISHED_ERROR")
    request = scheduler.requests["internal-req-1"]
    assert request.resumable is False
    assert request.stop_reason == ("stage_payload_processor_failed: legacy processor raised ValueError")


def test_transfer_failure_preserves_error_terminal_for_engine_output():
    failure = _failure()
    scheduler = _FailureScheduler(
        OmniConnectorOutput(
            transfer_failures={"internal-req-1": failure},
        )
    )

    scheduler._consume_pending_connector_output("ar")

    terminal = scheduler._pop_stage_transfer_error_output("internal-req-1")
    assert terminal is not None
    assert terminal.request_id == "internal-req-1"
    assert terminal.finish_reason is FinishReason.ERROR
    assert terminal.stop_reason == ("stage_payload_processor_failed: legacy processor raised ValueError")
    assert scheduler._pop_stage_transfer_error_output("internal-req-1") is None


def test_connector_and_chunk_adapter_failures_merge_by_internal_request_id():
    failure = _failure()
    scheduler = _FailureScheduler(
        OmniConnectorOutput(
            transfer_failures={"internal-req-1": failure},
        ),
        adapter_failures={"internal-req-1": failure},
    )

    scheduler._consume_pending_connector_output("generation")

    assert scheduler.chunk_transfer_adapter.drain_calls == 1
    assert len(scheduler.finish_calls) == 1
    assert scheduler.finish_calls[0][0] == {"internal-req-1"}


def test_failure_for_already_freed_request_is_ignored():
    unknown = _failure("already-freed")
    scheduler = _FailureScheduler(OmniConnectorOutput(transfer_failures={"already-freed": unknown}))

    scheduler._consume_pending_connector_output("ar")

    assert scheduler.finish_calls == []


def test_failure_message_is_sanitized_and_bounded():
    failure = StageTransferFailure(
        internal_request_id="internal-req-1",
        external_request_id=None,
        source_stage=0,
        destination_stage=1,
        code="stage_payload_contract_failed",
        message="line one\nline two\x00" + "x" * 600,
    )

    assert "\n" not in failure.message
    assert "\x00" not in failure.message
    assert len(failure.message) <= 512
