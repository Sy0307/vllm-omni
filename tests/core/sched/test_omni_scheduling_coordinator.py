# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniSchedulingCoordinator.

These tests use mock request objects and mock queues.  They do not require
GPU, vLLM runtime, or any connector.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
    uses_native_mrv2_data_plane,
)
from vllm_omni.core.sched.output import OmniChunkRecvHandle

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class _RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    WAITING_FOR_CHUNK = "waiting_for_chunk"
    WAITING_FOR_INPUT = "waiting_for_input"
    FINISHED_STOPPED = "finished_stopped"


# Patch RequestStatus for tests that don't import vllm
try:
    from vllm.v1.request import RequestStatus
except ImportError:
    RequestStatus = _RequestStatus  # type: ignore[misc,assignment]

if not hasattr(RequestStatus, "WAITING_FOR_INPUT"):
    coord_mod.RequestStatus = _RequestStatus  # type: ignore[assignment]
    RequestStatus = _RequestStatus  # type: ignore[misc,assignment]


def _make_request(req_id: str, status: str = "waiting") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=req_id,
        status=status,
        additional_information=None,
        prompt_token_ids=[],
        num_prompt_tokens=0,
        num_computed_tokens=0,
        num_output_placeholders=0,
        _all_token_ids=[],
        _output_token_ids=[],
    )


class MockQueue:
    """Simplified queue that mimics the Scheduler waiting queue interface."""

    def __init__(self, items: list | None = None):
        self._items: list = list(items or [])

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def add_request(self, request):
        self._items.append(request)

    def prepend_requests(self, requests):
        self._items = list(requests) + self._items

    def remove(self, request):
        self._items.remove(request)

    def remove_requests(self, requests):
        remove_set = set(id(r) for r in requests)
        self._items = [r for r in self._items if id(r) not in remove_set]


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestFullPayloadCoordinatorSelection(unittest.TestCase):
    """Tests for the (model_arch, model_stage) whitelist gate.

    The init_omni_connectors arch allowlist is keyed by ``model_arch`` and
    is a superset of the stages registered here -- consumer-wait stages
    must be registered explicitly in ``_FULL_PAYLOAD_INPUT_STAGES``, while
    the init allowlist covers both producer- and consumer-side runners.
    These tests pin which ``(arch, stage)`` pairs the gate fires for today.
    """

    # Expected whitelist (model_arch, model_stage).  Hardcoded to avoid the
    # tautology of importing _FULL_PAYLOAD_INPUT_STAGES and asserting it
    # against itself; any drift between this matrix and the whitelist will
    # fail loudly here.
    EXPECTED_FULL_PAYLOAD_INPUT_STAGES: frozenset[tuple[str, str]] = frozenset(
        {
            ("Qwen3OmniMoeForConditionalGeneration", "talker"),
            ("Qwen3OmniMoeForConditionalGeneration", "code2wav"),
            ("Qwen2_5OmniForConditionalGeneration", "talker"),
            ("Qwen2_5OmniForConditionalGeneration", "code2wav"),
            ("CovoAudioForConditionalGeneration", "code2wav"),
            ("MiMoAudioModel", "code2wav"),
            ("Qwen3TTSCode2Wav", "code2wav"),
            ("CosyVoice3Model", "cosyvoice3_code2wav"),
            ("IndexTTS2S2MelDecoder", "indextts2_s2mel_decoder"),
            ("DyninOmniForConditionalGeneration", "token2image"),
            ("DyninOmniForConditionalGeneration", "token2audio"),
        }
    )

    def test_whitelist_matches_expected_matrix(self):
        """_FULL_PAYLOAD_INPUT_STAGES must equal the hardcoded expected matrix.

        Catches both accidental additions (which would silently enable the
        consumer-wait gate for a new arch) and accidental removals (which
        would silently disable an enabled arch).
        """
        from vllm_omni.core.sched.omni_scheduling_coordinator import _FULL_PAYLOAD_INPUT_STAGES

        self.assertEqual(
            frozenset(_FULL_PAYLOAD_INPUT_STAGES),
            self.EXPECTED_FULL_PAYLOAD_INPUT_STAGES,
            msg="_FULL_PAYLOAD_INPUT_STAGES drifted from the expected matrix; "
            "update EXPECTED_FULL_PAYLOAD_INPUT_STAGES if intentional.",
        )

    def test_all_whitelisted_arch_stage_pairs_fire_gate(self):
        """Every (arch, stage) pair in the expected matrix must fire
        the gate when stage_id > 0 and async_chunk=False.
        """
        for arch, stage in self.EXPECTED_FULL_PAYLOAD_INPUT_STAGES:
            model_config = SimpleNamespace(
                stage_id=1,
                async_chunk=False,
                model_arch=arch,
                model_stage=stage,
            )
            self.assertTrue(
                uses_full_payload_input_coordinator(model_config),
                msg=f"expected gate to fire for {arch}/{stage}",
            )

    def test_other_arch_or_stage_or_mode_does_not_fire(self):
        cases = [
            SimpleNamespace(
                stage_id=1, async_chunk=True, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage="talker"
            ),
            SimpleNamespace(
                stage_id=0, async_chunk=False, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage="thinker"
            ),
            SimpleNamespace(
                stage_id=1,
                async_chunk=False,
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_stage="some_future_stage",
            ),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="Qwen3TTSForConditionalGeneration", model_stage="code2wav"
            ),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="MingFlashOmniForConditionalGeneration", model_stage="talker"
            ),
            SimpleNamespace(stage_id=1, async_chunk=False, model_arch=None, model_stage="talker"),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage=None
            ),
        ]
        for model_config in cases:
            self.assertFalse(
                uses_full_payload_input_coordinator(model_config),
                msg=f"expected gate OFF for {model_config}",
            )


class TestNativeMRV2DataPlaneSelection(unittest.TestCase):
    def test_qwen3_omni_async_chunk_uses_native_data_plane_only_on_mrv2(self):
        model_config = SimpleNamespace(
            async_chunk=True,
            supports_native_mrv2_data_plane=True,
        )

        self.assertTrue(uses_native_mrv2_data_plane(model_config, use_v2_model_runner=True))
        self.assertFalse(uses_native_mrv2_data_plane(model_config, use_v2_model_runner=False))

    def test_sync_or_other_arch_keeps_existing_path(self):
        sync_config = SimpleNamespace(
            async_chunk=False,
            supports_native_mrv2_data_plane=True,
        )
        unsupported = SimpleNamespace(
            async_chunk=True,
            supports_native_mrv2_data_plane=False,
        )

        self.assertFalse(uses_native_mrv2_data_plane(sync_config, use_v2_model_runner=True))
        self.assertFalse(uses_native_mrv2_data_plane(unsupported, use_v2_model_runner=True))


class TestChunkCoordinatorStateTransition(unittest.TestCase):
    """Test 5: process_pending_chunks transitions WAITING_FOR_CHUNK → target."""

    def test_ready_request_transitions_to_waiting(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIn("r1", coord.requests_with_ready_chunks)

    def test_non_ready_stays_waiting_for_chunk(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_new_waiter_emits_minimal_runner_registration_handle(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("internal", status=RequestStatus.WAITING)
        req.external_req_id = "external"
        waiting = MockQueue([req])

        coord.process_pending_chunks(
            waiting,
            [],
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        assert len(coord.pending_chunk_registrations) == 1
        handle = coord.pending_chunk_registrations[0]
        assert isinstance(handle, OmniChunkRecvHandle)
        assert handle.request_id == "internal"
        assert handle.external_req_id == "external"

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)
        req = _make_request("r1")
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )
        self.assertNotEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_talker_decode_horizon_grants_exact_row_credits(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.RUNNING)
        req._output_token_ids = [100]
        waiting = MockQueue()
        running = [req]
        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"decode_token_end": 4}},
            model_mode="ar",
        )

        for generated_count in (1, 2, 3):
            req._output_token_ids = [100] * generated_count
            coord.process_pending_chunks(waiting, running, set(), set())
            self.assertEqual(req.status, RequestStatus.RUNNING)
            self.assertIn(req, running)

        req._output_token_ids = [100] * 4
        coord.process_pending_chunks(waiting, running, set(), set())

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        self.assertNotIn(req, running)

    def test_talker_exhausted_credit_resumes_only_after_upstream_finishes(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.RUNNING)
        req._output_token_ids = [100, 101]
        waiting = MockQueue()
        running = [req]
        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"decode_token_end": 2}},
            model_mode="ar",
        )

        coord.process_pending_chunks(waiting, running, set(), set())
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

        coord.restore_queues(waiting, running)
        coord.process_pending_chunks(waiting, running, {"r1"}, {"r1"})

        self.assertEqual(req.status, RequestStatus.RUNNING)
        self.assertIn("r1", coord.finished_requests)

    def test_talker_initial_chunk_readiness_is_consumed_once(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(waiting, running, {"r1"}, set())
        self.assertEqual(req.status, RequestStatus.WAITING)

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="r1")],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        )
        coord.postprocess_scheduler_output(scheduler_output)
        coord.process_pending_chunks(waiting, running, set(), set())

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_talker_decode_horizon_ignores_stale_async_metadata(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.RUNNING)
        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"decode_token_end": 4}},
            model_mode="ar",
        )

        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"decode_token_end": 3}},
            model_mode="ar",
        )

        self.assertEqual(coord.decode_token_horizons["r1"], 4)

    def test_talker_credit_reserves_async_output_placeholders(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.RUNNING)
        req._output_token_ids = [100]
        req.num_output_placeholders = 1
        waiting = MockQueue()
        running = [req]
        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"decode_token_end": 2}},
            model_mode="ar",
        )

        coord.process_pending_chunks(waiting, running, set(), set())

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        self.assertNotIn(req, running)


class TestChunkCoordinatorRestoreQueues(unittest.TestCase):
    """Test 6: restore_queues returns waiting-for-chunk requests."""

    def test_restore(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_for_chunk_waiting.append(r1)
        coord._waiting_for_chunk_running.append(r2)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertIn(r2, running)
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 0)
        self.assertEqual(len(coord._waiting_for_chunk_running), 0)


class TestChunkCoordinatorFinishedSignal(unittest.TestCase):
    """Test 8: chunk_finished_req_ids → finished_requests."""

    def test_finished_signal(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )

        self.assertIn("r1", coord.finished_requests)


class TestChunkCoordinatorUpdateRequestMetadata(unittest.TestCase):
    """Test update_request_metadata applies scheduling metadata to requests."""

    def test_ar_mode_no_longer_sets_additional_information(self):
        """AR mode only processes scheduling metadata, not full payloads."""
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        requests = {"r1": req}

        # Only scheduling metadata is passed now (full payload stays in model runner)
        request_metadata = {"r1": {"next_stage_prompt_len": 50}}

        coord.update_request_metadata(requests, request_metadata, model_mode="ar")

        # next_stage_prompt_len should update prompt_token_ids
        self.assertEqual(len(req.prompt_token_ids), 50)
        self.assertEqual(req.num_prompt_tokens, 50)
        # additional_information should NOT be set
        self.assertIsNone(getattr(req, "additional_information", None))

    def test_ar_mode_uses_exact_prompt_ids_instead_of_zero_placeholders(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        req.num_prompt_tokens = 3
        req._all_token_ids = [0, 0, 0]

        coord.update_request_metadata(
            {"r1": req},
            {
                "r1": {
                    "next_stage_prompt_len": 3,
                    "next_stage_prompt_ids": [3071, 872, 3071],
                }
            },
            model_mode="ar",
        )

        self.assertEqual(req.prompt_token_ids, [3071, 872, 3071])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req._all_token_ids, [3071, 872, 3071])
        self.assertEqual(req._output_token_ids, [])
        self.assertEqual(req.num_computed_tokens, 0)

    def test_ar_mode_does_not_replace_prompt_ids_after_decode_started(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        req = _make_request("r1")
        req.prompt_token_ids = [11, 12]
        req.num_prompt_tokens = 2
        req._all_token_ids = [11, 12, 99]
        req._output_token_ids = [99]
        req.num_computed_tokens = 3

        coord.update_request_metadata(
            {"r1": req},
            {"r1": {"next_stage_prompt_ids": [3071, 872, 3071]}},
            model_mode="ar",
        )

        self.assertEqual(req.prompt_token_ids, [11, 12])
        self.assertEqual(req._all_token_ids, [11, 12, 99])
        self.assertEqual(req._output_token_ids, [99])
        self.assertEqual(req.num_computed_tokens, 3)

    def test_generation_mode(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        req.num_prompt_tokens = 3
        req.num_computed_tokens = 3
        req._all_token_ids = [0, 0, 0, 99]
        req._output_token_ids = [99]
        requests = {"r1": req}

        request_metadata = {
            "r1": {
                "code_predictor_codes": [10, 20, 30],
                "left_context_size": 25,
            }
        }

        coord.update_request_metadata(requests, request_metadata, model_mode="generation")

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req._all_token_ids, [10, 20, 30])
        self.assertEqual(req._output_token_ids, [])
        self.assertIsNone(req.additional_information)
        self.assertEqual(req._omni_initial_model_buffer, {"meta": {"left_context_size": 25}})

    def test_generation_mode_flattens_tensor_code_predictor_codes(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": {"code_predictor_codes": torch.tensor([[1, 2, 3]], dtype=torch.long)}},
            model_mode="generation",
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req._all_token_ids, [1, 2, 3])
        self.assertEqual(req._output_token_ids, [])

    def test_generation_mode_flattens_nested_code_predictor_codes(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": {"code_predictor_codes": [[1, 2], [3, 4]]}},
            model_mode="generation",
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(req.num_prompt_tokens, 4)
        self.assertEqual(req._all_token_ids, [1, 2, 3, 4])
        self.assertEqual(req._output_token_ids, [])


class TestChunkCoordinatorPostprocess(unittest.TestCase):
    """Test postprocess_scheduler_output clears ready chunks."""

    def test_clear_ready(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        coord.requests_with_ready_chunks = {"r1", "r2"}

        new_req = SimpleNamespace(req_id="r1")
        cached_reqs = SimpleNamespace(req_ids=["r2"])
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=cached_reqs,
        )

        coord.postprocess_scheduler_output(scheduler_output)

        self.assertEqual(coord.requests_with_ready_chunks, set())


class TestWaitingForInputTransition(unittest.TestCase):
    """Test B8: process_pending_full_payload_inputs transitions WAITING_FOR_INPUT."""

    def test_transition_on_recv(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_stays_waiting_for_input_if_not_received(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)

    def test_restore_queues_includes_waiting_for_input(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        coord._waiting_for_input.append(r1)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_full_payload_mode_auto_transitions_waiting_to_waiting_for_input(self):
        """In full_payload_mode (async_chunk=False), fresh WAITING requests on
        non-Stage-0 should be transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)
        self.assertEqual(len(coord.pending_input_registrations), 1)

    def test_async_chunk_mode_does_not_auto_transition(self):
        """In async_chunk mode, fresh WAITING requests should NOT be
        transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_pending_input_registrations(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(len(coord.pending_input_registrations), 1)
        self.assertEqual(coord.pending_input_registrations[0].request_id, "r1")

    def test_idle_cycles_retain_received_marker_before_request_appears(self):
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        coord._full_payload_input_received.add("late")
        coord.finished_requests.add("late")

        waiting = MockQueue()
        running: list = []

        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())

        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)

        late_req = _make_request("late", status=RequestStatus.WAITING)
        waiting.add_request(late_req)

        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())

        self.assertEqual(late_req.status, RequestStatus.WAITING)
        self.assertEqual(coord.pending_input_registrations, [])
        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)


class TestTimeoutDetection(unittest.TestCase):
    """Regression tests for orphaned pending-recv timeout detection.

    Covers the full lifecycle:
      1. Request enters WAITING_FOR_CHUNK from either waiting or running queue
      2. restore_queues() moves it back to the scheduler queue
      3. Timeout fires via collect_timed_out_request_ids()
      4. Scheduler removes from both queues and calls _free_request()
    """

    def test_waiting_since_recorded_on_chunk_wait(self):
        """_waiting_since is set when a request enters WAITING_FOR_CHUNK."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_chunks(
            waiting,
            [],
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertIn("r1", coord._waiting_since)
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_waiting_since_cleared_on_chunk_arrival(self):
        """_waiting_since is cleared when a chunk arrives."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])

        coord.process_pending_chunks(
            waiting,
            [],
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )

        self.assertNotIn("r1", coord._waiting_since)

    def test_waiting_since_recorded_on_input_wait(self):
        """_waiting_since is set when a request enters WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            [],
            stage_recv_req_ids=set(),
        )

        self.assertIn("r1", coord._waiting_since)

    def test_waiting_since_cleared_on_input_arrival(self):
        """_waiting_since is cleared when input data arrives."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        coord._waiting_for_input.append(req)
        coord._waiting_since["r1"] = 0.0

        waiting = MockQueue()
        coord.process_pending_full_payload_inputs(
            waiting,
            [],
            stage_recv_req_ids={"r1"},
        )

        self.assertNotIn("r1", coord._waiting_since)
        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_collect_timed_out_request_ids_no_timeout(self):
        """No IDs returned when nothing has timed out."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        import time

        coord._waiting_since["r1"] = time.monotonic()

        result = coord.collect_timed_out_request_ids(timeout_s=300.0)
        self.assertEqual(result, set())

    def test_collect_timed_out_request_ids_expired(self):
        """Timed-out IDs are returned and _waiting_since is cleared."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        coord._waiting_since["r1"] = 0.0  # epoch → definitely expired
        coord._waiting_since["r2"] = 0.0

        import time

        coord._waiting_since["r3"] = time.monotonic() + 9999  # far future

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1", "r2"})
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r2", coord._waiting_since)
        self.assertIn("r3", coord._waiting_since)

    def test_collect_removes_from_coordinator_queues(self):
        """Timed-out requests are defensively removed from internal queues."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_for_chunk_waiting.append(r1)
        coord._waiting_for_input.append(r2)
        coord._waiting_since["r1"] = 0.0
        coord._waiting_since["r2"] = 0.0

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1", "r2"})
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 0)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_free_finished_request_clears_all_lifecycle_state(self):
        """free_finished_request makes stale connector events harmless."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_since["r1"] = 0.0
        coord._full_payload_input_received.add("r1")
        coord.finished_requests.add("r1")
        coord.requests_with_ready_chunks.add("r1")
        coord.decode_token_horizons["r1"] = 7
        coord._waiting_for_chunk_waiting.extend([r1, r2])
        coord._waiting_for_chunk_running.extend([r1, r2])
        coord._waiting_for_input.extend([r1, r2])
        coord.pending_chunk_registrations = [
            OmniChunkRecvHandle(request_id="r1"),
            OmniChunkRecvHandle(request_id="r2"),
        ]
        coord.pending_input_registrations = [
            OmniChunkRecvHandle(request_id="r1"),
            OmniChunkRecvHandle(request_id="r2"),
        ]

        coord.free_finished_request("r1")

        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r1", coord._full_payload_input_received)
        self.assertNotIn("r1", coord.finished_requests)
        self.assertNotIn("r1", coord.requests_with_ready_chunks)
        self.assertNotIn("r1", coord.decode_token_horizons)
        for queue in (
            coord._waiting_for_chunk_waiting,
            coord._waiting_for_chunk_running,
            coord._waiting_for_input,
        ):
            self.assertEqual([request.request_id for request in queue], ["r2"])
        self.assertEqual(
            [handle.request_id for handle in coord.pending_chunk_registrations],
            ["r2"],
        )
        self.assertEqual(
            [handle.request_id for handle in coord.pending_input_registrations],
            ["r2"],
        )

    def test_generation_metadata_tracks_terminal_input_until_it_is_scheduled(self):
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )
        requests = {
            "r1": _make_request("r1"),
            "r2": _make_request("r2"),
        }

        coord.update_request_metadata(
            requests,
            {"r1": {"input_terminal": True}},
            model_mode="generation",
        )

        previous_output = SimpleNamespace(
            num_scheduled_tokens={"r2": 1},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["r2"]),
        )
        self.assertEqual(
            coord.get_scheduled_input_terminal_req_ids(previous_output),
            set(),
        )
        coord.postprocess_scheduler_output(previous_output)
        self.assertEqual(coord.input_terminal_req_ids, {"r1"})

        terminal_output = SimpleNamespace(
            num_scheduled_tokens={"r1": 1},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["r1"]),
        )
        self.assertEqual(
            coord.get_scheduled_input_terminal_req_ids(terminal_output),
            {"r1"},
        )
        coord.postprocess_scheduler_output(terminal_output)
        self.assertEqual(coord.input_terminal_req_ids, set())

    def test_timeout_from_running_queue_full_lifecycle(self):
        """End-to-end: request from running → WAITING_FOR_CHUNK → restore →
        timeout → removed from running list.

        This is the critical regression case: WAITING_FOR_CHUNK requests
        that originated from self.running are placed back into self.running
        by restore_queues(), but their status remains WAITING_FOR_CHUNK.
        The scheduler must remove from BOTH queues unconditionally.
        """
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        # 1) Request starts in running queue with WAITING status
        req = _make_request("r1", status=RequestStatus.WAITING)
        running = [req]
        waiting = MockQueue()

        # 2) process_pending_chunks: moves to WAITING_FOR_CHUNK
        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        self.assertIn("r1", coord._waiting_since)
        self.assertEqual(len(coord._waiting_for_chunk_running), 1)

        # 3) restore_queues: back to running (status stays WAITING_FOR_CHUNK)
        coord.restore_queues(waiting, running)
        self.assertIn(req, running)
        self.assertEqual(len(coord._waiting_for_chunk_running), 0)
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

        # 4) Force timeout by setting _waiting_since to epoch
        coord._waiting_since["r1"] = 0.0

        timed_out_ids = coord.collect_timed_out_request_ids(timeout_s=1.0)
        self.assertEqual(timed_out_ids, {"r1"})

        # 5) Scheduler removes from both queues (simulating the scheduler path)
        timed_out_id_set = {id(req)}
        running = [r for r in running if id(r) not in timed_out_id_set]
        waiting.remove_requests([req])

        self.assertNotIn(req, running)
        self.assertEqual(len(waiting), 0)

    def test_timeout_from_waiting_queue_full_lifecycle(self):
        """End-to-end: request from waiting → WAITING_FOR_CHUNK → restore →
        timeout → removed from waiting queue."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 1)

        coord.restore_queues(waiting, running)
        self.assertIn(req, waiting)

        coord._waiting_since["r1"] = 0.0
        timed_out_ids = coord.collect_timed_out_request_ids(timeout_s=1.0)
        self.assertEqual(timed_out_ids, {"r1"})

        waiting.remove_requests([req])
        self.assertEqual(len(waiting), 0)


class TestOverflowPreemption(unittest.TestCase):
    """Tests for P1-1: overflow requests must get WAITING status.

    Overflow happens when multiple WAITING_FOR_CHUNK requests in
    ``_waiting_for_chunk_running`` receive their chunk in the same cycle.
    ``_process_chunk_queue`` restores them to RUNNING (``continue``
    path) while RUNNING requests without chunks are moved out.  If the
    net result exceeds ``scheduler_max_num_seqs``, the tail is pushed
    to ``waiting_queue`` and must have status == WAITING.
    """

    def test_overflow_sets_waiting_status(self):
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=1,
            stage_id=1,
            async_chunk=True,
        )

        # r1 is currently RUNNING in the queue.
        # r2, r3 were previously moved to _waiting_for_chunk_running.
        r1 = _make_request("r1", status=RequestStatus.RUNNING)
        r2 = _make_request("r2", status=RequestStatus.WAITING_FOR_CHUNK)
        r3 = _make_request("r3", status=RequestStatus.WAITING_FOR_CHUNK)

        running = [r1]
        waiting = MockQueue([])
        coord._waiting_for_chunk_running.extend([r2, r3])

        # restore_queues puts r2, r3 back into running
        coord.restore_queues(waiting, running)
        self.assertEqual(len(running), 3)

        # Now process_pending_chunks with r2, r3 chunks ready:
        # _process_chunk_queue will:
        #   r1 (RUNNING) → no chunk → move to _waiting_for_chunk_running
        #   r2 (WAITING_FOR_CHUNK, chunk ready) → set RUNNING, stay in running
        #   r3 (WAITING_FOR_CHUNK, chunk ready) → set RUNNING, stay in running
        # running = [r2, r3], len=2 > max=1 → overflow
        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r2", "r3"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(len(running), 1)
        self.assertEqual(len(waiting), 1)
        overflow_req = list(waiting)[0]
        self.assertEqual(
            overflow_req.status,
            RequestStatus.WAITING,
            f"Overflowed request should have WAITING status, got {overflow_req.status}",
        )

    def test_overflow_does_not_strand_request(self):
        """Without the fix, the overflowed request would keep its
        RUNNING status in the waiting queue and never be re-scheduled."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=1,
            stage_id=1,
            async_chunk=True,
        )

        r1 = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        r2 = _make_request("r2", status=RequestStatus.WAITING_FOR_CHUNK)
        coord._waiting_for_chunk_running.extend([r1, r2])

        running: list = []
        waiting = MockQueue([])

        coord.restore_queues(waiting, running)
        self.assertEqual(len(running), 2)

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1", "r2"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(len(running), 1)
        self.assertEqual(len(waiting), 1)
        for req in waiting:
            self.assertNotEqual(req.status, RequestStatus.RUNNING, "Overflowed request must not keep RUNNING status")


if __name__ == "__main__":
    unittest.main()
