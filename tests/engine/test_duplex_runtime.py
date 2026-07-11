import pytest

from vllm_omni.engine.orchestrator import _duplex_force_listen_count
from vllm_omni.experimental.fullduplex.engine.omni import (
    DuplexAdapterPattern,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_new_user_turn_prefix_reserve,
    duplex_scheduler_token_budget,
)
from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
    MiniCPMO45DuplexPolicy,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence


def test_duplex_runtime_tracks_stage_bindings_and_barge_in_epoch():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-1"),
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            adapter_patterns={DuplexAdapterPattern.CHUNK_GROUP_APPEND},
            input_modes={DuplexInputMode.APPEND_TOKENS},
            supports_kv_lease=True,
            supports_stage_resumption=True,
        ),
    )
    session.bind_stage_request(stage_id=0, request_id="req-stage0", replica_id=1, fence=session.fence)
    session.bind_stage_request(stage_id=1, request_id="req-stage1", replica_id=0, fence=session.fence)

    update = session.append_input({"tokens": [1, 2, 3]}, mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)
    next_fence = DuplexFence("sid-1", epoch=1)
    stale_request_ids = session.release_fence(session.fence)
    session.accept_fence(next_fence)

    assert update.seq == 1
    assert session.fence == next_fence
    assert stale_request_ids == ["req-stage0", "req-stage1"]
    assert session.stage_bindings == {}
    assert session.input_seq == 0
    assert session.pending_inputs == []


def test_duplex_runtime_core_kv_lease_is_not_model_internal_state():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-model-internal"),
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            supports_kv_lease=True,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
        ),
    )

    session.bind_stage_request(stage_id=0, request_id="req-stage0", fence=session.fence)

    assert session.stage_bindings[0].lease_active is False


def test_duplex_runtime_core_kv_lease_marks_stage_binding_active():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-core-lease"),
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            supports_kv_lease=True,
            supports_core_kv_lease=True,
        ),
    )

    session.bind_stage_request(stage_id=0, request_id="req-stage0", fence=session.fence)

    assert session.stage_bindings[0].lease_active is True


def test_duplex_runtime_pending_inputs_store_metadata_not_raw_payload():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-audio"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    audio_payload = {
        "type": "audio",
        "audio": "A" * 4096,
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }

    update = session.append_input(audio_payload, mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)

    assert not hasattr(update, "payload")
    assert update.payload_meta == {
        "type": "dict",
        "keys": ["audio", "format", "sample_rate_hz", "type"],
        "audio_bytes": 3072,
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }
    assert session.pending_inputs == [update]


def test_duplex_runtime_tracks_turn_local_append_sequence():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-turn-seq"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    first = session.append_input({"is_speech": True}, mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    second = session.append_input({"is_speech": True}, mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    next_turn = DuplexFence("sid-turn-seq", turn_id=1, response_seq=1)
    third = session.append_input(
        {"is_speech": True, "new_user_turn": True},
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_turn,
    )
    fourth = session.append_input(
        {"is_speech": True},
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_turn,
    )

    assert [first.seq, second.seq, third.seq, fourth.seq] == [1, 2, 3, 4]
    assert [first.turn_seq, second.turn_seq, third.turn_seq, fourth.turn_seq] == [1, 2, 1, 2]
    assert [first.turn_id, second.turn_id, third.turn_id, fourth.turn_id] == [0, 0, 1, 1]


def test_duplex_runtime_rejects_unsupported_append_mode():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-2"),
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.TURN_COMMIT_ONLY}),
    )

    with pytest.raises(ValueError, match="not supported"):
        session.append_input({"tokens": [1]}, mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)


def test_duplex_runtime_serializes_capability_patterns():
    caps = DuplexRuntimeCapabilities(
        adapter_patterns={
            DuplexAdapterPattern.CHUNK_GROUP_APPEND,
            DuplexAdapterPattern.EXPERIMENTAL_WORKER_CONTROL_RPC,
            DuplexAdapterPattern.PER_STEP_TENSOR_HANDOFF,
        },
        input_modes={
            DuplexInputMode.APPEND_TOKENS,
            DuplexInputMode.ROLLBACK_TO_CHECKPOINT,
        },
        supports_audio_truncate=True,
        chunk_period_ms=1000,
        target_barge_in_latency_ms=300,
    )

    data = caps.as_dict()

    assert data["adapter_patterns"] == [
        "chunk_group_append",
        "experimental_worker_control_rpc",
        "per_step_tensor_handoff",
    ]
    assert data["input_modes"] == ["append_tokens", "rollback_to_checkpoint"]
    assert data["supports_audio_truncate"] is True
    assert data["chunk_period_ms"] == 1000
    assert data["target_barge_in_latency_ms"] == 300


def test_duplex_data_plane_request_info_extracts_structured_stage_result():
    request_id, response_stage_id = duplex_data_plane_request_info(
        {
            "stage_results": [
                {"result": {"supported": True}},
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0-s1",
                        "response_stage_id": 1,
                    }
                },
            ]
        }
    )

    assert request_id == "duplex-sid-e0-stage0-s1"
    assert response_stage_id == 1


def test_duplex_data_plane_request_info_rejects_missing_request_id():
    assert duplex_data_plane_request_info(
        {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "",
                        "response_stage_id": 1,
                    }
                }
            ]
        }
    ) == (None, None)


def test_duplex_scheduler_token_budget_estimates_pcm_slots():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
        == 16
    )


def test_duplex_scheduler_token_budget_ignores_client_budget_fields():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "duplex_num_input_tokens": 999,
                "num_input_tokens": 999,
            }
        )
        == 16
    )


def test_duplex_new_user_turn_prefix_reserve_uses_precomputed_count():
    assert duplex_new_user_turn_prefix_reserve({"extra_body": {"duplex_new_user_turn_prefix_tokens": 7}}) == 7


def test_duplex_new_user_turn_prefix_reserve_uses_variant_count():
    assert (
        duplex_new_user_turn_prefix_reserve(
            {
                "extra_body": {
                    "duplex_new_user_turn_prefix_tokens": 99,
                    "duplex_new_user_turn_prefix_tokens_by_variant": {
                        MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE: 5,
                    },
                }
            },
            variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE,
        )
        == 5
    )


def test_minicpmo_new_user_turn_prefix_variants_match_hf_duplex_streaming_prefill():
    # MiniCPMODuplex.streaming_prefill feeds only <unit> plus media per
    # append. Chat-role prefixes are used by the simplex streaming path and
    # must not be injected into native full-duplex KV mid-session.
    assert (
        MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(
            MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE
        )
        == ""
    )
    assert (
        MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_INTERRUPTED_TTS)
        == ""
    )
    assert (
        MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_LISTEN_ONLY) == ""
    )


def test_duplex_force_listen_default_matches_hf_zero():
    assert _duplex_force_listen_count(None) == 0
    assert _duplex_force_listen_count({}) == 0
    assert _duplex_force_listen_count({"force_listen_count": 3}) == 3
    assert _duplex_force_listen_count({"force_listen_count": "bad"}) == 0
