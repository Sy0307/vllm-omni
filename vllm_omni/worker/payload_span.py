# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for explicit thinker decode span metadata."""

import torch

from vllm_omni.distributed.omni_connectors.tensor_span import (
    TensorSpan,
)
from vllm_omni.distributed.omni_connectors.tensor_span import (
    get_tensor_span as get_tensor_span,
)
from vllm_omni.distributed.omni_connectors.tensor_span import (
    merge_tensor_spans as merge_tensor_spans,
)

THINKER_DECODE_EMBEDDINGS_KEY = "thinker_decode_embeddings"
THINKER_OUTPUT_TOKEN_IDS_KEY = "thinker_output_token_ids"
THINKER_DECODE_TOKEN_START_KEY = "thinker_decode_embeddings_token_start"
THINKER_DECODE_TOKEN_END_KEY = "thinker_decode_embeddings_token_end"

CACHED_THINKER_DECODE_EMBEDDINGS_KEY = "cached_thinker_decode_embeddings"
CACHED_THINKER_DECODE_TOKEN_START_KEY = "cached_thinker_decode_embeddings_token_start"
CACHED_THINKER_DECODE_TOKEN_END_KEY = "cached_thinker_decode_embeddings_token_end"


def get_tensor_span_row(span: TensorSpan | None, index: int) -> torch.Tensor | None:
    if span is None:
        return None
    tensor, start, end = span
    if index < start or index >= end:
        return None
    return tensor[index - start]
