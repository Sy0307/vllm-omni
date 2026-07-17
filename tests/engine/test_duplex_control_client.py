# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.engine.duplex_control_client import DuplexControlClient
from vllm_omni.engine.duplex_lease import DuplexLeaseActivity
from vllm_omni.engine.duplex_types import DuplexFence
from vllm_omni.engine.messages import (
    DuplexControlResultMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)


class _Transport:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, key, message, **kwargs):
        self.calls.append((key, message, kwargs))
        return DuplexControlResultMessage(
            control_id=message.control_id,
            fence=message.fence,
            operation="touch" if isinstance(message, TouchDuplexSessionMessage) else "resume",
            session_id=message.session_id,
            ok=True,
            stage_results=[],
        )


def test_control_client_routes_touch_and_resume_by_control_id() -> None:
    transport = _Transport()
    control_ids = iter(("touch-id", "resume-id"))
    client = DuplexControlClient(transport, control_id_factory=lambda: next(control_ids))
    fence = DuplexFence("sid-client")

    assert (
        client.touch(
            fence.session_id,
            fence=fence,
            activity=DuplexLeaseActivity.PLAYBACK_ACK,
            timeout=2.0,
        )["ok"]
        is True
    )
    assert (
        client.resume(
            fence.session_id,
            fence=fence,
            expected_lease_generation=7,
            timeout=3.0,
        )["ok"]
        is True
    )

    touch_key, touch_message, touch_kwargs = transport.calls[0]
    assert touch_key == ("duplex", "touch-id")
    assert isinstance(touch_message, TouchDuplexSessionMessage)
    assert touch_message.activity == DuplexLeaseActivity.PLAYBACK_ACK.value
    assert touch_kwargs["timeout"] == 2.0

    resume_key, resume_message, resume_kwargs = transport.calls[1]
    assert resume_key == ("duplex", "resume-id")
    assert isinstance(resume_message, ResumeDuplexSessionMessage)
    assert resume_message.expected_lease_generation == 7
    assert resume_kwargs["timeout"] == 3.0
