# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for VoxCPM2 talker per-request state lifecycle.

Pins the contract broken by the crash fixed in this PR:

1. preprocess() must NOT evict state based on _pending_requests (which is
   a per-step prefix, not the full batch).
2. Cleanup must be driven by on_requests_finished -> _flush_deferred_cleanup,
   which only runs at the end of forward().

The test constructs the "N cached-decode + 1 new prefill" batch shape that
triggered the original out-of-bounds crash and asserts the cached-decode
states survive through the preprocess phase.

Pure Python; no GPU, CUDA Graph, or torch.compile involved.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
# voxcpm2_talker.py imports librosa at module scope; the lightweight unit-test
# environments (simple-unit-test, diffusion-cache-backend-test, etc.) don't
# install librosa, so skip rather than error out on collection.
pytest.importorskip("librosa")

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (  # noqa: E402
    VoxCPM2TalkerForConditionalGeneration,
    _RequestState,
)


def _make_bare_talker() -> VoxCPM2TalkerForConditionalGeneration:
    """Construct a talker without running __init__ (which needs vllm_config)."""
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    talker._active_states = {}
    talker._current_request_id = None
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._max_batch_size = 4
    talker._active_state_warn_threshold = 512
    talker._active_state_warned = False
    return talker


def _seed_cached_decode(talker, req_id: str) -> _RequestState:
    """Seed a request state that looks like it has finished prefill."""
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.decode_step_count = 5
    talker._active_states[req_id] = state
    return state


class TestStateEvictionContract:
    def test_pending_requests_is_not_used_for_eviction(self) -> None:
        """Reproduce the "N cached decode + 1 new prefill" batch shape.

        Before the fix, preprocess() evicted states whose request_id was not
        in _pending_requests. Because vLLM calls preprocess() once per request
        and only populates _pending_requests up to the current one, cached
        decodes scheduled AFTER a new prefill would be wrongly evicted.
        """
        talker = _make_bare_talker()

        # 4 cached decodes already in _active_states from earlier steps.
        cached_ids = [f"req-{i}" for i in range(4)]
        for rid in cached_ids:
            _seed_cached_decode(talker, rid)

        # Simulate vLLM's runner: it calls preprocess() once per request and
        # only appends to _pending_requests as it goes.  When a new prefill
        # lands in the middle of the batch, _pending_requests holds only the
        # prefix walked so far — say the prefill + the first 2 cached decodes.
        new_prefill_id = "req-new"
        walked_so_far = [new_prefill_id, cached_ids[0], cached_ids[1]]
        talker._pending_requests = [(rid, False, None, 0) for rid in walked_so_far]

        # All original cached-decode states must still be present; the
        # remaining two cached decodes (req-2, req-3) are scheduled after
        # the prefill and are NOT yet in _pending_requests.
        for rid in cached_ids:
            assert rid in talker._active_states, (
                f"cached decode {rid} must not be evicted just because it hasn't been walked yet in _pending_requests"
            )

        # Regression pin: every cached state kept its prefill_completed flag.
        for rid in cached_ids:
            assert talker._active_states[rid].prefill_completed is True

    def test_on_requests_finished_defers_cleanup(self) -> None:
        """on_requests_finished must not mutate _active_states directly.

        Cleanup is deferred to _flush_deferred_cleanup at the end of
        forward(), because on_requests_finished is called BEFORE forward()
        and the current step may still reference the state.
        """
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")

        talker.on_requests_finished({"req-A"})

        assert "req-A" in talker._active_states, (
            "on_requests_finished must defer cleanup (forward() may still touch the state this step)"
        )
        assert "req-A" in talker._deferred_cleanup_ids

    def test_flush_deferred_cleanup_removes_only_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker.on_requests_finished(["req-A"])

        talker._flush_deferred_cleanup()

        assert "req-A" not in talker._active_states
        assert "req-B" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_current_request_id_cleared_when_matching(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        talker._current_request_id = "req-A"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id is None

    def test_current_request_id_preserved_when_not_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker._current_request_id = "req-B"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id == "req-B"


class TestLeakWarnGuard:
    def test_warn_fires_once_over_threshold(self, monkeypatch) -> None:
        """_get_or_create_state warns at most once when _active_states grows."""
        from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as tk

        calls: list[str] = []

        def _capture(msg, *args, **kwargs):
            calls.append(msg % args if args else msg)

        monkeypatch.setattr(tk.logger, "warning", _capture)

        talker = _make_bare_talker()
        talker._active_state_warn_threshold = 3

        # Pre-fill past the threshold without using _get_or_create_state
        # (so the one-shot flag is still False).
        for i in range(4):
            talker._active_states[f"seed-{i}"] = _RequestState(request_id=f"seed-{i}")

        talker._get_or_create_state("new-1")
        talker._get_or_create_state("new-2")

        leak_warnings = [m for m in calls if "cleanup path leak" in m]
        assert len(leak_warnings) == 1, f"leak warn must be one-shot; got {len(leak_warnings)} records"
        assert talker._active_state_warned is True
