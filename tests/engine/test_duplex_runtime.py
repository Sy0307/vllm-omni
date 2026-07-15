import pytest

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.engine.omni import (
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    build_duplex_data_plane_prompt,
    duplex_data_plane_request_info,
    duplex_new_user_turn_prefix_reserve,
    duplex_resource_request_id,
    duplex_scheduler_token_budget,
)
from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
    MiniCPMO45DuplexPolicy,
)


def test_duplex_runtime_tracks_stage_bindings_and_barge_in_epoch():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-1"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_TOKENS},
        ),
    )
    session.bind_stage_request(stage_id=0, request_id="req-stage0", fence=session.fence)
    session.bind_stage_request(stage_id=1, request_id="req-stage1", fence=session.fence)

    update = session.append_input(mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)
    next_fence = DuplexFence("sid-1", epoch=1)
    stale_request_ids = session.release_fence(session.fence)
    session.accept_fence(next_fence)

    assert update.seq == 1
    assert session.fence == next_fence
    assert stale_request_ids == ["req-stage0", "req-stage1"]
    assert session.stage_bindings == {}
    assert session.input_seq == 0


def test_duplex_runtime_cancel_fence_rejects_late_append_and_accepts_next_epoch():
    manager = DuplexSessionRuntimeManager()
    cancelled_fence = DuplexFence("sid-cancel-race")
    next_fence = DuplexFence("sid-cancel-race", epoch=1)
    session = manager.open_session(
        cancelled_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(
        stage_id=0,
        request_id="req-cancelled",
        fence=cancelled_fence,
    )

    stale_request_ids = session.cancel_fence(cancelled_fence, next_fence)

    assert stale_request_ids == ["req-cancelled"]
    assert session.fence == next_fence
    assert session.stage_bindings == {}
    with pytest.raises(RuntimeError, match="fence mismatch"):
        session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=cancelled_fence,
        )
    update = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_fence,
    )
    assert update.seq == 1


def test_duplex_runtime_stale_close_preserves_live_session_and_bindings():
    manager = DuplexSessionRuntimeManager()
    current_fence = DuplexFence("sid-stale-close", epoch=1)
    session = manager.open_session(
        current_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(0, "req-live", fence=current_fence)

    with pytest.raises(RuntimeError, match="fence mismatch"):
        manager.close_session(DuplexFence("sid-stale-close"))

    assert manager.get("sid-stale-close") is session
    assert session.stage_request_ids() == ["req-live"]


def test_duplex_runtime_reopen_rejects_late_append_from_old_incarnation():
    manager = DuplexSessionRuntimeManager()
    old_fence = DuplexFence("sid-reopen", incarnation=0)
    old_session = manager.open_session(
        old_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    manager.close_session(old_fence)

    new_fence = DuplexFence("sid-reopen", incarnation=1)
    new_session = manager.open_session(
        new_fence,
        capabilities=old_session.capabilities,
    )

    with pytest.raises(RuntimeError, match="fence mismatch"):
        new_session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=old_fence,
        )
    assert (
        new_session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=new_fence,
        ).seq
        == 1
    )


def test_duplex_prompt_expands_incarnation_metadata():
    fence = DuplexFence("sid-incarnation", incarnation=3)

    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={"is_speech": True},
        final=False,
    )

    assert prompt["model_intermediate_buffer"]["duplex"]["incarnation"] == 3


def test_duplex_runtime_tracks_turn_local_append_sequence():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-turn-seq"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    first = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    second = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    next_turn = DuplexFence("sid-turn-seq", turn_id=1, response_seq=1)
    third = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_turn,
    )
    fourth = session.append_input(
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
        session.append_input(mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)


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


def test_resource_state_rejects_fence_regression_and_requires_explicit_fence():
    current = DuplexFence("sid", epoch=2, turn_id=3, response_seq=4)
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        current,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )

    for stale in (
        DuplexFence("sid", epoch=1, turn_id=99, response_seq=99),
        DuplexFence("sid", epoch=2, turn_id=2, response_seq=4),
        DuplexFence("sid", epoch=2, turn_id=3, response_seq=3),
    ):
        with pytest.raises(RuntimeError, match="fence mismatch"):
            session.accept_fence(stale)
        assert session.fence == current

    with pytest.raises(TypeError, match="fence"):
        session.bind_stage_request(0, "request")
    with pytest.raises(TypeError, match="fence"):
        session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK)
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.open_session("legacy-session")
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.close_session("legacy-session")


def test_resource_request_id_is_derived_from_fence_and_role():
    fence = DuplexFence("sid-with-dashes", epoch=7, turn_id=11, response_seq=13)

    assert duplex_resource_request_id(fence, "stage0") == "duplex-sid-with-dashes-e7-stage0"
    assert duplex_resource_request_id(fence, "stage1") == "duplex-sid-with-dashes-e7-stage1"


def test_placeholder_budget_is_planned_inside_omni_engine_boundary():
    fence = DuplexFence("sid", turn_id=1, response_seq=1)
    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        seq=2,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={
            "audio": "AAAAAA==",
            "format": "pcm_f32le",
            "duplex_num_input_tokens": 999,
            "num_input_tokens": 999,
        },
        final=False,
    )

    assert len(prompt["prompt_token_ids"]) == 16
    assert prompt["model_intermediate_buffer"]["duplex"]["fence"] == fence
    assert prompt["model_intermediate_buffer"]["duplex"]["scheduler_token_budget"] == 16
