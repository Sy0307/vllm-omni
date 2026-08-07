from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine import duplex_runtime
from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexExecutionProfile,
    duplex_data_plane_request_info,
    duplex_resource_request_belongs_to_session,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
    MiniCPMO45DuplexRuntimeExtension,
    build_duplex_data_plane_prompt,
    duplex_scheduler_token_budget,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_fence_is_immutable():
    fence = DuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")


def test_duplex_fence_contains_only_session_incarnation_and_epoch() -> None:
    assert [field.name for field in fields(DuplexFence)] == [
        "session_id",
        "incarnation",
        "epoch",
    ]


def test_output_progress_does_not_partition_input_sequence() -> None:
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-independent-input-output", incarnation=3, epoch=1),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    first = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )
    session.output_ledger.emit_chunk(
        source_input_seq=first.input_seq,
        audio_data="YQ==",
        audio_format="pcm16",
    )
    second = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )

    assert first.input_seq == 1
    assert second.input_seq == 2


def test_rejected_append_reservation_does_not_consume_input_sequence() -> None:
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-stale-reservation"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    first = session.prepare_append(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )
    stale = session.prepare_append(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )

    assert session.commit_append(first).input_seq == 1
    with pytest.raises(RuntimeError, match="reservation is stale"):
        session.commit_append(stale)
    assert session.input_seq == 1


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

    assert update.input_seq == 1
    assert session.fence == next_fence
    assert stale_request_ids == ["req-stage0", "req-stage1"]
    assert session.stage_bindings == {}


def test_duplex_runtime_tracks_same_request_id_for_each_pipeline_stage():
    manager = DuplexSessionRuntimeManager()
    fence = DuplexFence("sid-shared-pipeline-request")
    session = manager.open_session(fence)

    session.reserve_stage_request(0, "req-shared", fence=fence)
    session.bind_stage_request(0, "req-shared", fence=fence)
    session.bind_stage_request(1, "req-shared", fence=fence)

    assert session.stage_bindings[0].request_id == "req-shared"
    assert session.stage_bindings[1].request_id == "req-shared"
    assert session.resource_request_ids() == ["req-shared"]
    assert session.input_seq == 0


def test_duplex_runtime_extension_validation_rejects_missing_methods():
    class IncompleteExtension:
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults

    with pytest.raises(TypeError, match="plan_append"):
        duplex_runtime.validate_duplex_runtime_extension(IncompleteExtension())


def test_duplex_runtime_extension_validation_rejects_stage_count_mismatch():
    class WrongStageCountExtension(MiniCPMO45DuplexRuntimeExtension):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults[:1]

    with pytest.raises(ValueError, match="one sampling parameter per stage"):
        duplex_runtime.validate_duplex_runtime_extension(
            WrongStageCountExtension(),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


def test_duplex_runtime_extension_validation_rejects_sampling_type_mismatch():
    class WrongSamplingTypeExtension(MiniCPMO45DuplexRuntimeExtension):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return tuple(object() for _ in defaults)

    with pytest.raises(TypeError, match="sampling parameter type"):
        duplex_runtime.validate_duplex_runtime_extension(
            WrongSamplingTypeExtension(),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prewarm_batch_sizes": (1, 1)}, "duplicates"),
        ({"prewarm_batch_sizes": (0,)}, "positive integers"),
        ({"prewarm_batch_sizes": [1]}, "tuple"),
        ({"step_latency_budget_ms": 0.0}, "positive finite"),
        ({"step_latency_budget_ms": float("inf")}, "positive finite"),
    ],
)
def test_duplex_execution_profile_rejects_invalid_values(kwargs, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DuplexExecutionProfile(**kwargs)


def test_duplex_runtime_extension_defaults_to_no_execution_policy_or_resources() -> None:
    extension = MiniCPMO45DuplexRuntimeExtension()

    assert duplex_runtime.duplex_execution_profile(extension) == DuplexExecutionProfile()
    assert duplex_runtime.duplex_resource_lease_providers(extension) == ()


def test_duplex_runtime_extension_exposes_optional_execution_policy_and_resources() -> None:
    provider = object()

    class Extension(MiniCPMO45DuplexRuntimeExtension):
        def execution_profile(self):
            return DuplexExecutionProfile(
                prewarm_batch_sizes=(1, 2),
                step_latency_budget_ms=80.0,
            )

        def resource_lease_providers(self):
            return (provider,)

    extension = Extension()

    assert duplex_runtime.duplex_execution_profile(extension).prewarm_batch_sizes == (1, 2)
    assert duplex_runtime.duplex_resource_lease_providers(extension) == (provider,)


def test_duplex_output_decision_metadata_is_immutable():
    decision = DuplexOutputDecision(
        action=DuplexOutputAction.DIRECT_RESPONSE,
        metadata={"model_listen": True},
    )

    with pytest.raises(TypeError):
        decision.metadata["model_listen"] = False


def _decide_minicpmo_output(
    output: object,
    *,
    segment_token_ids: tuple[int, ...] = (),
    segment_output_metadata: dict | None = None,
):
    return MiniCPMO45DuplexRuntimeExtension().decide_output(
        stage_id=0,
        final_stage_id=1,
        segment_finished=True,
        segment_token_ids=segment_token_ids,
        segment_output_metadata=segment_output_metadata or {},
        output=output,
    )


def test_minicpmo_extension_owns_stage_sampling_overrides():
    defaults = (
        SamplingParams(max_tokens=4),
        SamplingParams(max_tokens=8),
    )

    configured = MiniCPMO45DuplexRuntimeExtension().configure_sampling_params(
        runtime_config={
            "duplex_stage_max_tokens": {"0": 20},
            "duplex_stage_sampling_params": {"1": {"stop_token_ids": [151645]}},
        },
        defaults=defaults,
    )

    assert configured[0].max_tokens == 20
    assert configured[1].stop_token_ids == [151645]
    assert defaults[0].max_tokens == 4
    assert 151645 not in (defaults[1].stop_token_ids or [])


def test_minicpmo_output_decision_uses_raw_streaming_token_snapshot():
    decision = _decide_minicpmo_output(
        SimpleNamespace(outputs=[SimpleNamespace()]),
        segment_token_ids=(151705,),
        segment_output_metadata={"special_token_ids": {"listen_token_id": 151705}},
    )

    assert decision is not None
    assert decision.action is DuplexOutputAction.DIRECT_RESPONSE
    assert decision.metadata["duplex_native_decision"] == "listen"
    assert decision.metadata["model_listen"] is True


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_minicpmo_output_decision_ignores_output_level_token_history(attr):
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace()],
        **{attr: [42, 151705]},
    )

    assert _decide_minicpmo_output(output) is None


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_minicpmo_output_decision_uses_completion_token_ids(attr):
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(**{attr: [42, 151705]})],
    )

    assert _decide_minicpmo_output(output) is not None


def test_minicpmo_output_decision_uses_completion_stop_reason():
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(stop_reason=151705)],
    )

    assert _decide_minicpmo_output(output) is not None


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
    assert update.input_seq == 1


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
        ).input_seq
        == 1
    )


def test_duplex_prompt_expands_incarnation_metadata():
    fence = DuplexFence("sid-incarnation", incarnation=3)

    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        runtime_config={},
        input_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={"is_speech": True},
        final=False,
    )

    assert prompt["model_intermediate_buffer"]["duplex"]["incarnation"] == 3


def test_duplex_runtime_tracks_monotonic_input_sequence_until_epoch_change():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-turn-seq"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    first = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    second = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    third = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )
    fourth = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )
    next_epoch = DuplexFence("sid-turn-seq", epoch=1)
    after_cancel = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_epoch,
    )

    assert [first.input_seq, second.input_seq, third.input_seq, fourth.input_seq] == [1, 2, 3, 4]
    assert after_cancel.input_seq == 1


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


def test_resource_state_rejects_fence_regression_and_requires_explicit_fence():
    current = DuplexFence("sid", epoch=2)
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        current,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )

    for stale in (DuplexFence("sid", epoch=1),):
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
    fence = DuplexFence("sid-with-dashes", epoch=7)

    assert duplex_resource_request_id(fence, "stage0") == "duplex-s.c2lkLXdpdGgtZGFzaGVz.i.0.e.7.r.stage0"
    assert duplex_resource_request_id(fence, "stage1") == "duplex-s.c2lkLXdpdGgtZGFzaGVz.i.0.e.7.r.stage1"


def test_resource_request_id_codec_separates_session_id_from_incarnation():
    embedded_incarnation = duplex_resource_request_id(
        DuplexFence("foo-i1", incarnation=0),
        "stage0",
    )
    actual_incarnation = duplex_resource_request_id(
        DuplexFence("foo", incarnation=1),
        "stage0",
    )

    assert embedded_incarnation != actual_incarnation


def test_resource_request_id_session_membership_uses_encoded_identity():
    request_id = duplex_resource_request_id(
        DuplexFence("sid-with-dashes", incarnation=2, epoch=7),
        "stage0",
    )

    assert duplex_resource_request_belongs_to_session(request_id, "sid-with-dashes") is True
    assert duplex_resource_request_belongs_to_session(request_id, "sid") is False
    assert duplex_resource_request_belongs_to_session("duplex-s.invalid.i.x.e.7.r.stage0", "sid") is False


def test_placeholder_budget_is_planned_inside_omni_engine_boundary():
    fence = DuplexFence("sid", epoch=1)
    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        runtime_config={},
        input_seq=2,
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
