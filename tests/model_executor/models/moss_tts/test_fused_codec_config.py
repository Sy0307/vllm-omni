# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.moss_tts.fused_codec import _codec_vllm_config, fused_codec_options

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("backend,tp,pp", [("mp", 1, 1), ("uni", 2, 1), ("uni", 1, 2)])
def test_fused_codec_rejects_executor_without_in_process_delivery(backend, tp, pp):
    config = Mock(spec=VllmConfig)
    config.model_config.async_chunk = config.model_config.use_v2_model_runner = True
    config.parallel_config.distributed_executor_backend = backend
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.pipeline_parallel_size = pp
    with pytest.raises(ValueError, match="uniprocess executor"):
        _codec_vllm_config(config, {})


@pytest.mark.parametrize("options", [[], None, False])
def test_fused_codec_rejects_non_mapping_options(options):
    config = Mock(spec=VllmConfig)
    config.additional_config = {"moss_fused_codec": options}
    with pytest.raises(ValueError, match="must be a mapping"):
        fused_codec_options(config)
