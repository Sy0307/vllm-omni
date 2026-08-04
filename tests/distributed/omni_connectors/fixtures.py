# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Importable fixtures for stage-payload configuration tests."""

import torch

from vllm_omni.distributed.omni_connectors.payload_schema import (
    PayloadMergeMode,
    StagePayloadFieldRule,
    StagePayloadSchema,
    TensorFieldConstraint,
)
from vllm_omni.distributed.omni_connectors.stage_payload import (
    NoPayloadYet,
    StageRoute,
)

STAGE_0_TO_1_SCHEMA = StagePayloadSchema(
    schema_version=1,
    route=StageRoute(source_stage=0, destination_stage=1),
    fields={
        ("codes", "audio"): StagePayloadFieldRule(
            mode=PayloadMergeMode.DELTA,
            tensor=TensorFieldConstraint(
                rank=1,
                dtypes=frozenset({torch.int64}),
            ),
        )
    },
)

STAGE_0_TO_2_SCHEMA = StagePayloadSchema(
    schema_version=1,
    route=StageRoute(source_stage=0, destination_stage=2),
    fields={},
)

WRONG_ROUTE_SCHEMA = StagePayloadSchema(
    schema_version=1,
    route=StageRoute(source_stage=1, destination_stage=2),
    fields={},
)


class RecordingProcessor:
    def __init__(self) -> None:
        self.contexts = []
        self.dropped = []

    def process(self, context):
        self.contexts.append(context)
        return NoPayloadYet()

    def drop_request(self, identity):
        self.dropped.append(identity)


def build_processor() -> RecordingProcessor:
    return RecordingProcessor()


class FailingProcessor:
    def process(self, context):
        del context
        raise RuntimeError("injected stage payload failure")

    def drop_request(self, identity):
        del identity


def build_failing_processor() -> FailingProcessor:
    return FailingProcessor()


def legacy_async_processor(**kwargs):
    return kwargs.get("multimodal_output")


def legacy_full_processor(**kwargs):
    return kwargs.get("pooling_output")
