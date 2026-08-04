# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.stage_payload import (
    NoPayloadYet,
    NormalizedStageOutput,
    PayloadEmission,
    StageBoundary,
    StagePayloadBuildContext,
    StagePayloadIdentity,
    StageRequestView,
    StageRoute,
)
from vllm_omni.distributed.omni_connectors.stage_payload_processor import (
    LegacyProcessorMode,
    LegacyStagePayloadProcessorAdapter,
    convert_legacy_result,
    load_stage_payload_processor,
    load_stage_payload_schema,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(*, finished=None, segment_finished=None) -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([1], dtype=torch.long)),
        meta=MetaStruct(
            finished=finished,
            is_segment_finished=segment_finished,
        ),
    )


def _context(
    *,
    boundary: StageBoundary = StageBoundary.NONE,
    raw_output=None,
    terminal: bool = False,
) -> StagePayloadBuildContext:
    route = StageRoute(source_stage=0, destination_stage=1)
    identity = StagePayloadIdentity(
        external_request_id="request-1",
        source_request_id="internal-1",
    )
    return StagePayloadBuildContext(
        source_output=NormalizedStageOutput(
            output_payload=_payload(),
            request_payload=None,
            raw_output=raw_output,
        ),
        request=StageRequestView(
            internal_request_id="internal-1",
            external_request_id="request-1",
            resumable=not terminal,
            terminal=terminal,
            additional_information=None,
            model_intermediate_buffer=None,
        ),
        route=route,
        identity=identity,
        chunk_seq=0,
        boundary=boundary,
    )


def test_new_processor_receives_exactly_one_context():
    processor = load_stage_payload_processor("tests.distributed.omni_connectors.fixtures.build_processor")
    context = _context()

    result = processor.process(context)

    assert isinstance(result, NoPayloadYet)
    assert processor.contexts == [context]


def test_loader_instantiates_processor_classes():
    processor = load_stage_payload_processor("tests.distributed.omni_connectors.fixtures.RecordingProcessor")
    context = _context()

    result = processor.process(context)

    assert isinstance(result, NoPayloadYet)
    assert processor.contexts == [context]


def test_failure_injection_processor_exception_escapes():
    processor = load_stage_payload_processor("tests.distributed.omni_connectors.fixtures.build_failing_processor")

    with pytest.raises(RuntimeError, match="injected stage payload failure"):
        processor.process(_context())


def test_schema_loader_validates_the_expected_route():
    route = StageRoute(source_stage=0, destination_stage=1)

    schema = load_stage_payload_schema(
        "tests.distributed.omni_connectors.fixtures.STAGE_0_TO_1_SCHEMA",
        expected_route=route,
    )

    assert schema.route == route


def test_schema_loader_can_supply_the_runtime_route():
    schema = load_stage_payload_schema(
        "tests.distributed.omni_connectors.fixtures.WRONG_ROUTE_SCHEMA",
    )

    assert schema.route == StageRoute(source_stage=1, destination_stage=2)


def test_schema_loader_rejects_a_different_route():
    with pytest.raises(ValueError, match="route"):
        load_stage_payload_schema(
            "tests.distributed.omni_connectors.fixtures.WRONG_ROUTE_SCHEMA",
            expected_route=StageRoute(source_stage=0, destination_stage=1),
        )


def test_legacy_async_adapter_uses_only_the_multimodal_call_shape():
    seen = []
    request = object()
    transfer_manager = object()
    raw_output = object()

    def legacy(**kwargs):
        seen.append(kwargs)
        return _payload()

    adapter = LegacyStagePayloadProcessorAdapter(
        legacy,
        mode=LegacyProcessorMode.ASYNC_CHUNK,
        transfer_manager=transfer_manager,
        request_provider=lambda _view: request,
    )
    adapter.process(_context(raw_output=raw_output, terminal=True))

    assert seen == [
        {
            "transfer_manager": transfer_manager,
            "multimodal_output": raw_output,
            "request": request,
            "is_finished": True,
        }
    ]


def test_legacy_full_adapter_uses_only_the_pooling_call_shape():
    seen = []
    request = object()
    transfer_manager = object()
    raw_output = object()

    def legacy(**kwargs):
        seen.append(kwargs)
        return _payload()

    adapter = LegacyStagePayloadProcessorAdapter(
        legacy,
        mode=LegacyProcessorMode.FULL_PAYLOAD,
        transfer_manager=transfer_manager,
        request_provider=lambda _view: request,
    )
    adapter.process(_context(raw_output=raw_output, terminal=True))

    assert seen == [
        {
            "transfer_manager": transfer_manager,
            "pooling_output": raw_output,
            "request": request,
        }
    ]


def test_legacy_none_without_boundary_becomes_no_payload_yet():
    result = convert_legacy_result(None, _context())

    assert isinstance(result, NoPayloadYet)


def test_legacy_none_at_boundary_becomes_finish_only_emission():
    result = convert_legacy_result(
        None,
        _context(boundary=StageBoundary.STREAM_END),
    )

    assert result == PayloadEmission(
        payload=None,
        boundary=StageBoundary.STREAM_END,
    )


def test_legacy_false_segment_flag_suppresses_inferred_segment_boundary():
    result = convert_legacy_result(
        _payload(segment_finished=torch.tensor(False)),
        _context(boundary=StageBoundary.SEGMENT_END),
    )

    assert result.boundary is StageBoundary.NONE


def test_legacy_stream_finished_wins_over_segment_finished():
    result = convert_legacy_result(
        _payload(
            finished=torch.tensor(True),
            segment_finished=torch.tensor(True),
        ),
        _context(boundary=StageBoundary.SEGMENT_END),
    )

    assert result.boundary is StageBoundary.STREAM_END


def test_legacy_processor_exception_escapes_adapter():
    def broken(**_kwargs):
        raise ValueError("bad codec rank")

    adapter = LegacyStagePayloadProcessorAdapter(
        broken,
        mode=LegacyProcessorMode.ASYNC_CHUNK,
        transfer_manager=object(),
        request_provider=lambda _view: object(),
    )

    with pytest.raises(ValueError, match="bad codec rank"):
        adapter.process(_context())


def test_chunk_legacy_adapter_does_not_promote_model_finished_to_stream_end():
    def legacy(**_kwargs):
        return _payload(
            finished=torch.tensor(True),
            segment_finished=torch.tensor(False),
        )

    adapter = LegacyStagePayloadProcessorAdapter(
        legacy,
        mode=LegacyProcessorMode.ASYNC_CHUNK,
        transfer_manager=object(),
        request_provider=lambda _view: object(),
        normalize_async_transport_flags=True,
    )

    result = adapter.process(
        _context(boundary=StageBoundary.SEGMENT_END),
    )

    assert isinstance(result, PayloadEmission)
    assert result.boundary is StageBoundary.NONE
    assert result.payload.meta.finished.item() is False
    assert result.payload.meta.is_segment_finished.item() is False


def test_chunk_legacy_adapter_marks_real_request_terminal_as_stream_end():
    def legacy(**_kwargs):
        return _payload(
            finished=torch.tensor(False),
            segment_finished=torch.tensor(False),
        )

    adapter = LegacyStagePayloadProcessorAdapter(
        legacy,
        mode=LegacyProcessorMode.ASYNC_CHUNK,
        transfer_manager=object(),
        request_provider=lambda _view: object(),
        normalize_async_transport_flags=True,
    )

    result = adapter.process(
        _context(boundary=StageBoundary.STREAM_END, terminal=True),
    )

    assert isinstance(result, PayloadEmission)
    assert result.boundary is StageBoundary.STREAM_END
    assert result.payload.meta.finished.item() is True
