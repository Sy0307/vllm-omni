# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import msgspec
import pytest
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.payload_schema import (
    PayloadMergeMode,
    StagePayloadContractError,
    StagePayloadFieldRule,
    StagePayloadSchema,
    TensorFieldConstraint,
)
from vllm_omni.distributed.omni_connectors.stage_payload import (
    NoPayloadYet,
    NormalizedStageOutput,
    PayloadEmission,
    StageBoundary,
    StagePayloadBuildContext,
    StagePayloadEnvelope,
    StagePayloadIdentity,
    StageRequestView,
    StageRoute,
    StageSessionFence,
    decode_stage_payload_wire,
    try_normalize_stage_payload,
)
from vllm_omni.distributed.omni_connectors.stage_payload_processor import (
    reconcile_envelope_legacy_boundary,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _identity() -> StagePayloadIdentity:
    return StagePayloadIdentity(
        external_request_id="request-1",
        source_request_id="internal-1",
    )


def _payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(codes=CodesStruct(audio=torch.tensor([1, 2], dtype=torch.long)))


def test_payload_and_segment_boundary_can_coexist():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        identity=_identity(),
        chunk_seq=3,
        payload=_payload(),
        boundary=StageBoundary.SEGMENT_END,
    )

    assert envelope.payload is not None
    assert envelope.boundary is StageBoundary.SEGMENT_END


def test_finish_only_stream_boundary_is_valid():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=1, destination_stage=2),
        identity=StagePayloadIdentity(
            external_request_id="request-2",
            source_request_id="internal-2",
        ),
        chunk_seq=4,
        payload=None,
        boundary=StageBoundary.STREAM_END,
    )

    assert envelope.payload is None
    assert envelope.boundary is StageBoundary.STREAM_END


def test_session_fence_is_optional_and_can_identify_a_duplex_incarnation():
    assert _identity().session_fence is None

    fence = StageSessionFence(session_id="session-1", incarnation=2, epoch=7)
    identity = StagePayloadIdentity(
        external_request_id="request-1",
        source_request_id="internal-1",
        session_fence=fence,
    )

    assert identity.session_fence == fence


@pytest.mark.parametrize(
    ("payload", "boundary"),
    [
        (None, StageBoundary.NONE),
    ],
)
def test_envelope_rejects_empty_non_boundary_emission(payload, boundary):
    with pytest.raises(ValueError, match="payload or boundary"):
        StagePayloadEnvelope(
            schema_version=1,
            route=StageRoute(source_stage=0, destination_stage=1),
            identity=_identity(),
            chunk_seq=0,
            payload=payload,
            boundary=boundary,
        )


def test_payload_emission_rejects_empty_non_boundary_result():
    with pytest.raises(ValueError, match="payload or boundary"):
        PayloadEmission(payload=None, boundary=StageBoundary.NONE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"route": {"source_stage": 0, "destination_stage": 1}},
        {
            "identity": {
                "external_request_id": "request-1",
                "source_request_id": "internal-1",
            }
        },
        {"chunk_seq": True},
        {"boundary": "none"},
    ],
)
def test_envelope_direct_constructor_rejects_untyped_wire_fields(overrides):
    values = {
        "schema_version": 1,
        "route": StageRoute(source_stage=0, destination_stage=1),
        "identity": _identity(),
        "chunk_seq": 0,
        "payload": _payload(),
        "boundary": StageBoundary.NONE,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        StagePayloadEnvelope(**values)


def test_payload_emission_rejects_untyped_boundary():
    with pytest.raises(TypeError, match="boundary"):
        PayloadEmission(payload=_payload(), boundary="none")


def test_no_payload_yet_is_the_only_empty_non_boundary_result():
    assert isinstance(NoPayloadYet(), NoPayloadYet)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"chunk_seq": -1}, "chunk_seq"),
    ],
)
def test_envelope_rejects_invalid_version_or_sequence(overrides, message):
    values = {
        "schema_version": 1,
        "route": StageRoute(source_stage=0, destination_stage=1),
        "identity": _identity(),
        "chunk_seq": 0,
        "payload": _payload(),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        StagePayloadEnvelope(**values)


@pytest.mark.parametrize(
    ("source_stage", "destination_stage"),
    [(-1, 1), (0, -1), (1, 1), (False, 1)],
)
def test_route_rejects_invalid_stage_relationships(source_stage, destination_stage):
    with pytest.raises(ValueError, match="stage"):
        StageRoute(
            source_stage=source_stage,
            destination_stage=destination_stage,
        )


@pytest.mark.parametrize(
    ("external_request_id", "source_request_id"),
    [("", "internal-1"), ("request-1", "")],
)
def test_identity_rejects_empty_request_ids(
    external_request_id,
    source_request_id,
):
    with pytest.raises(ValueError, match="request_id"):
        StagePayloadIdentity(
            external_request_id=external_request_id,
            source_request_id=source_request_id,
        )


def test_request_view_snapshots_mutable_mappings():
    additional_information = {"turn": 1}
    model_intermediate_buffer = {"cursor": 3}
    request = StageRequestView(
        internal_request_id="internal-1",
        external_request_id="request-1",
        resumable=True,
        terminal=False,
        additional_information=additional_information,
        model_intermediate_buffer=model_intermediate_buffer,
    )

    additional_information["turn"] = 2
    model_intermediate_buffer["cursor"] = 4

    assert request.additional_information == {"turn": 1}
    assert request.model_intermediate_buffer == {"cursor": 3}
    with pytest.raises(TypeError):
        request.additional_information["turn"] = 5


def test_context_and_wire_containers_are_frozen():
    route = StageRoute(source_stage=0, destination_stage=1)
    identity = _identity()
    request = StageRequestView(
        internal_request_id="internal-1",
        external_request_id="request-1",
        resumable=True,
        terminal=False,
        additional_information=None,
        model_intermediate_buffer=None,
    )
    context = StagePayloadBuildContext(
        source_output=NormalizedStageOutput(
            output_payload=_payload(),
            request_payload=None,
        ),
        request=request,
        route=route,
        identity=identity,
        chunk_seq=0,
        boundary=StageBoundary.NONE,
    )
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=route,
        identity=identity,
        chunk_seq=0,
        payload=_payload(),
    )

    with pytest.raises(FrozenInstanceError):
        context.chunk_seq = 1
    with pytest.raises(AttributeError):
        envelope.chunk_seq = 1


def test_raw_model_mapping_that_is_not_an_omni_payload_stays_unnormalized():
    raw_output = {"pooler_features": torch.tensor([[1.0, 2.0]])}

    assert try_normalize_stage_payload(raw_output) is None


def test_valid_payload_mapping_is_normalized_for_migrated_processors():
    normalized = try_normalize_stage_payload({"codes": {"audio": torch.tensor([1, 2], dtype=torch.long)}})

    assert isinstance(normalized, OmniPayloadStruct)
    assert normalized.codes.audio.tolist() == [1, 2]


def test_normalizing_a_struct_copies_mutable_wire_containers():
    original = _payload()

    normalized = try_normalize_stage_payload(original)
    original.codes.audio = torch.tensor([9], dtype=torch.long)

    assert normalized is not original
    assert normalized.codes is not original.codes
    assert normalized.codes.audio.tolist() == [1, 2]


def _wire_schema() -> StagePayloadSchema:
    return StagePayloadSchema(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        fields={
            ("codes", "audio"): StagePayloadFieldRule(
                mode=PayloadMergeMode.DELTA,
                tensor=TensorFieldConstraint(
                    rank=1,
                    dtypes=frozenset({torch.int64}),
                ),
            ),
            ("meta", "finished"): StagePayloadFieldRule(mode=PayloadMergeMode.REPLACE),
            ("meta", "is_segment_finished"): StagePayloadFieldRule(mode=PayloadMergeMode.REPLACE),
        },
    )


def _encode_tensor(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError
    tensor = value.detach().cpu().contiguous()
    return {
        "__tensor__": True,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": tensor.numpy().tobytes(),
    }


def test_type_erased_msgpack_round_trip_reconstructs_envelope():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        identity=_identity(),
        chunk_seq=7,
        payload=_payload(),
        boundary=StageBoundary.SEGMENT_END,
    )
    encoded = msgspec.msgpack.encode(envelope, enc_hook=_encode_tensor)
    type_erased = msgspec.msgpack.decode(encoded)

    decoded = decode_stage_payload_wire(type_erased, schema=_wire_schema())

    assert isinstance(decoded, StagePayloadEnvelope)
    assert decoded.schema_version == 1
    assert decoded.route == envelope.route
    assert decoded.identity == envelope.identity
    assert decoded.chunk_seq == 7
    assert decoded.boundary is StageBoundary.SEGMENT_END
    assert decoded.payload.codes.audio.tolist() == [1, 2]


def test_legacy_raw_payload_remains_readable():
    raw_payload = {
        "codes": {"audio": [[1]]},
        "meta": {"model_specific_phase": "decode"},
    }
    decoded = decode_stage_payload_wire(
        raw_payload,
        schema=_wire_schema(),
    )

    assert decoded == raw_payload


def test_envelope_wire_rejects_unknown_header_fields():
    wire = {
        "schema_version": 1,
        "route": {"source_stage": 0, "destination_stage": 1},
        "identity": {
            "external_request_id": "request-1",
            "source_request_id": "internal-1",
            "session_fence": None,
        },
        "chunk_seq": 0,
        "payload": {"codes": {"audio": torch.tensor([1], dtype=torch.long)}},
        "boundary": "none",
        "abort": True,
    }

    with pytest.raises(StagePayloadContractError, match="unknown.*abort"):
        decode_stage_payload_wire(wire, schema=_wire_schema())


def test_matching_deprecated_flag_does_not_override_envelope_boundary():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        identity=_identity(),
        chunk_seq=0,
        payload=OmniPayloadStruct(meta=MetaStruct(finished=torch.tensor(True))),
        boundary=StageBoundary.STREAM_END,
    )

    reconcile_envelope_legacy_boundary(envelope)


def test_true_deprecated_flag_disagreeing_with_envelope_is_rejected():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        identity=_identity(),
        chunk_seq=0,
        payload=OmniPayloadStruct(meta=MetaStruct(finished=torch.tensor(True))),
        boundary=StageBoundary.SEGMENT_END,
    )

    with pytest.raises(StagePayloadContractError, match="disagree"):
        reconcile_envelope_legacy_boundary(envelope)


def test_false_deprecated_flag_does_not_suppress_envelope_boundary():
    envelope = StagePayloadEnvelope(
        schema_version=1,
        route=StageRoute(source_stage=0, destination_stage=1),
        identity=_identity(),
        chunk_seq=0,
        payload=OmniPayloadStruct(meta=MetaStruct(is_segment_finished=torch.tensor(False))),
        boundary=StageBoundary.SEGMENT_END,
    )

    reconcile_envelope_legacy_boundary(envelope)
