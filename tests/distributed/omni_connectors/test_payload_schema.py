# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.distributed.omni_connectors.payload_schema import (
    PayloadMergeMode,
    StagePayloadAccumulator,
    StagePayloadContractError,
    StagePayloadFieldRule,
    StagePayloadSchema,
    StagePayloadSequenceError,
    StagePayloadValidationError,
    TensorFieldConstraint,
    validate_payload_for_schema,
)
from vllm_omni.distributed.omni_connectors.stage_payload import (
    StageBoundary,
    StagePayloadEnvelope,
    StagePayloadIdentity,
    StageRoute,
    StageSessionFence,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


ROUTE = StageRoute(source_stage=1, destination_stage=2)


def _schema(*, schema_version: int = 1) -> StagePayloadSchema:
    return StagePayloadSchema(
        schema_version=schema_version,
        route=ROUTE,
        fields={
            ("codes", "audio"): StagePayloadFieldRule(
                mode=PayloadMergeMode.DELTA,
                tensor=TensorFieldConstraint(
                    rank=1,
                    dtypes=frozenset({torch.int64}),
                    concat_dim=0,
                ),
            ),
            ("embed", "decode"): StagePayloadFieldRule(
                mode=PayloadMergeMode.SNAPSHOT,
                tensor=TensorFieldConstraint(
                    rank=2,
                    dtypes=frozenset({torch.float32}),
                ),
            ),
            ("ids", "output"): StagePayloadFieldRule(
                mode=PayloadMergeMode.DELTA,
            ),
            ("meta", "cache_epoch"): StagePayloadFieldRule(
                mode=PayloadMergeMode.REPLACE,
            ),
        },
    )


def _identity(
    *,
    external_request_id: str = "request-1",
    source_request_id: str = "source-internal-1",
    session_fence: StageSessionFence | None = None,
) -> StagePayloadIdentity:
    return StagePayloadIdentity(
        external_request_id=external_request_id,
        source_request_id=source_request_id,
        session_fence=session_fence,
    )


def _envelope(
    *,
    seq: int,
    audio: torch.Tensor | None = None,
    decode: torch.Tensor | None = None,
    output_ids: list[int] | None = None,
    cache_epoch: int | None = None,
    boundary: StageBoundary = StageBoundary.NONE,
    identity: StagePayloadIdentity | None = None,
    route: StageRoute = ROUTE,
    payload: OmniPayloadStruct | None = None,
) -> StagePayloadEnvelope:
    if payload is None and any(value is not None for value in (audio, decode, output_ids, cache_epoch)):
        payload = OmniPayloadStruct(
            codes=CodesStruct(audio=audio) if audio is not None else None,
            embed=EmbeddingsStruct(decode=decode) if decode is not None else None,
            ids=IdsStruct(output=output_ids) if output_ids is not None else None,
            meta=(MetaStruct(cache_epoch=cache_epoch) if cache_epoch is not None else None),
        )
    return StagePayloadEnvelope(
        schema_version=1,
        route=route,
        identity=identity or _identity(),
        chunk_seq=seq,
        payload=payload,
        boundary=boundary,
    )


def _accumulator(
    *,
    schema: StagePayloadSchema | None = None,
    expected_fence: StageSessionFence | None = None,
) -> StagePayloadAccumulator:
    return StagePayloadAccumulator(
        schema or _schema(),
        expected_external_request_id="request-1",
        expected_session_fence=expected_fence,
    )


def test_delta_appends_and_replace_overwrites():
    accumulator = _accumulator()

    first = accumulator.apply(
        _envelope(
            seq=0,
            audio=torch.tensor([1, 2], dtype=torch.long),
            cache_epoch=0,
        )
    )
    second = accumulator.apply(
        _envelope(
            seq=1,
            audio=torch.tensor([3], dtype=torch.long),
            cache_epoch=1,
        )
    )

    assert first.payload.codes.audio.tolist() == [1, 2]
    assert second.payload.codes.audio.tolist() == [1, 2, 3]
    assert second.payload.meta.cache_epoch == 1


def test_list_delta_extends_instead_of_nesting():
    accumulator = _accumulator()

    accumulator.apply(_envelope(seq=0, output_ids=[1, 2]))
    result = accumulator.apply(_envelope(seq=1, output_ids=[3]))

    assert result.payload.ids.output == [1, 2, 3]


def test_snapshot_replaces_the_previous_cumulative_value():
    accumulator = _accumulator()
    first_decode = torch.tensor([[1.0], [2.0]])
    second_decode = torch.tensor([[4.0], [5.0], [6.0]])

    accumulator.apply(_envelope(seq=0, decode=first_decode))
    result = accumulator.apply(_envelope(seq=1, decode=second_decode))

    assert torch.equal(result.payload.embed.decode, second_decode)


@pytest.mark.parametrize(
    "audio",
    [
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    ],
)
def test_codes_audio_rejects_wrong_rank_or_dtype(audio):
    accumulator = _accumulator()

    with pytest.raises(StagePayloadValidationError):
        accumulator.apply(_envelope(seq=0, audio=audio))

    assert accumulator.next_chunk_seq == 0


@pytest.mark.parametrize(
    "audio",
    [
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    ],
)
def test_producer_validation_rejects_codes_audio_before_transport(audio):
    payload = OmniPayloadStruct(codes=CodesStruct(audio=audio))

    with pytest.raises(StagePayloadValidationError):
        validate_payload_for_schema(payload, _schema())


def test_unknown_payload_field_is_rejected_atomically():
    accumulator = _accumulator()
    invalid = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([1], dtype=torch.long)),
        request_id="not-declared-by-this-edge",
    )

    with pytest.raises(StagePayloadValidationError, match="request_id"):
        accumulator.apply(_envelope(seq=0, payload=invalid))

    result = accumulator.apply(_envelope(seq=0, audio=torch.tensor([2], dtype=torch.long)))
    assert result.payload.codes.audio.tolist() == [2]


def test_duplicate_delta_and_boundary_are_idempotent():
    accumulator = _accumulator()
    envelope = _envelope(
        seq=0,
        audio=torch.tensor([1, 2], dtype=torch.long),
        boundary=StageBoundary.SEGMENT_END,
    )

    first = accumulator.apply(envelope)
    duplicate = accumulator.apply(envelope)

    assert first.boundary is StageBoundary.SEGMENT_END
    assert duplicate.duplicate is True
    assert duplicate.boundary is StageBoundary.NONE
    assert duplicate.payload.codes.audio.tolist() == [1, 2]


def test_future_sequence_gap_is_a_contract_failure():
    accumulator = _accumulator()

    with pytest.raises(StagePayloadSequenceError, match="expected 0"):
        accumulator.apply(_envelope(seq=1, audio=torch.tensor([1], dtype=torch.long)))

    assert accumulator.next_chunk_seq == 0


def test_finish_only_boundary_advances_without_changing_payload():
    accumulator = _accumulator()
    first = accumulator.apply(_envelope(seq=0, audio=torch.tensor([1], dtype=torch.long)))

    finished = accumulator.apply(_envelope(seq=1, boundary=StageBoundary.STREAM_END))

    assert finished.payload is first.payload
    assert finished.boundary is StageBoundary.STREAM_END
    assert accumulator.next_chunk_seq == 2


def test_segment_boundary_does_not_reset_request_global_sequence():
    accumulator = _accumulator()

    accumulator.apply(
        _envelope(
            seq=0,
            audio=torch.tensor([1], dtype=torch.long),
            boundary=StageBoundary.SEGMENT_END,
        )
    )
    result = accumulator.apply(_envelope(seq=1, audio=torch.tensor([2], dtype=torch.long)))

    assert result.payload.codes.audio.tolist() == [1, 2]
    assert accumulator.next_chunk_seq == 2


def test_route_mismatch_is_rejected():
    accumulator = _accumulator()

    with pytest.raises(StagePayloadContractError, match="route"):
        accumulator.apply(
            _envelope(
                seq=0,
                audio=torch.tensor([1], dtype=torch.long),
                route=StageRoute(source_stage=0, destination_stage=2),
            )
        )


def test_external_identity_and_fence_are_destination_validated():
    fence = StageSessionFence(session_id="session-1", incarnation=1, epoch=2)
    accumulator = _accumulator(expected_fence=fence)

    with pytest.raises(StagePayloadContractError, match="external_request_id"):
        accumulator.apply(
            _envelope(
                seq=0,
                audio=torch.tensor([1], dtype=torch.long),
                identity=_identity(
                    external_request_id="wrong-request",
                    session_fence=fence,
                ),
            )
        )

    with pytest.raises(StagePayloadContractError, match="session_fence"):
        accumulator.apply(
            _envelope(
                seq=0,
                audio=torch.tensor([1], dtype=torch.long),
                identity=_identity(),
            )
        )


def test_source_identity_is_pinned_by_first_accepted_envelope():
    accumulator = _accumulator()
    accumulator.apply(_envelope(seq=0, audio=torch.tensor([1], dtype=torch.long)))

    with pytest.raises(StagePayloadContractError, match="source_request_id"):
        accumulator.apply(
            _envelope(
                seq=1,
                audio=torch.tensor([2], dtype=torch.long),
                identity=_identity(source_request_id="restarted-source"),
            )
        )


def test_schema_version_mismatch_is_rejected():
    accumulator = _accumulator(schema=_schema(schema_version=2))

    with pytest.raises(StagePayloadContractError, match="schema_version"):
        accumulator.apply(_envelope(seq=0, audio=torch.tensor([1], dtype=torch.long)))


def test_schema_copies_and_freezes_field_rules():
    fields = {("meta", "cache_epoch"): StagePayloadFieldRule(mode=PayloadMergeMode.REPLACE)}
    schema = StagePayloadSchema(schema_version=1, route=ROUTE, fields=fields)
    fields.clear()

    assert ("meta", "cache_epoch") in schema.fields
    with pytest.raises(TypeError):
        schema.fields[("codes", "audio")] = StagePayloadFieldRule(mode=PayloadMergeMode.DELTA)


def test_schema_rejects_empty_field_path():
    with pytest.raises(ValueError, match="field path"):
        StagePayloadSchema(
            schema_version=1,
            route=ROUTE,
            fields={(): StagePayloadFieldRule(mode=PayloadMergeMode.REPLACE)},
        )
