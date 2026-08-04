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

    def drain_transfer_failures(self):
        failures = self.failures
        self.failures = {}
        return failures


class _FailureScheduler(OmniSchedulerMixin):
    def __init__(self, connector_output, adapter_failures=None):
        self._latest_omni_connector_output = connector_output
        self.input_coordinator = None
        self.chunk_transfer_adapter = _FailureAdapter(adapter_failures or {})
        self.requests = {
            "internal-processor-error": SimpleNamespace(
                request_id="internal-processor-error",
                resumable=True,
                stop_reason=None,
            )
        }
        self.finish_calls = []

    def finish_requests(self, req_ids, status):
        self.finish_calls.append((set(req_ids), status))


def _failure():
    return StageTransferFailure(
        internal_request_id="internal-processor-error",
        external_request_id="external-processor-error",
        source_stage=0,
        destination_stage=1,
        code="stage_payload_processor_failed",
        message="legacy processor raised ValueError",
    )


def test_transfer_failure_becomes_error_terminal_without_input_coordinator():
    failure = _failure()
    scheduler = _FailureScheduler(
        OmniConnectorOutput(
            transfer_failures={"internal-processor-error": failure},
        )
    )

    scheduler._consume_pending_connector_output("ar")

    assert len(scheduler.finish_calls) == 1
    request_ids, status = scheduler.finish_calls[0]
    assert request_ids == {"internal-processor-error"}
    assert getattr(status, "name", str(status)).endswith("FINISHED_ERROR")
    request = scheduler.requests["internal-processor-error"]
    assert request.resumable is False
    terminal = scheduler._pop_stage_transfer_error_output("internal-processor-error")
    assert terminal is not None
    assert terminal.finish_reason is FinishReason.ERROR
    assert terminal.stop_reason == ("stage_payload_processor_failed: legacy processor raised ValueError")


def test_transfer_failure_message_is_sanitized_and_bounded():
    failure = StageTransferFailure(
        internal_request_id="internal-processor-error",
        external_request_id=None,
        source_stage=0,
        destination_stage=1,
        code="stage_payload_processor_failed",
        message="line one\nline two\x00" + "x" * 600,
    )

    assert "\n" not in failure.message
    assert "\x00" not in failure.message
    assert len(failure.message) <= 512
