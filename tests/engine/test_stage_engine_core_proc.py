from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


def test_omni_request_wire_roundtrip_preserves_model_intermediate_buffer():
    combined_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    request = OmniEngineCoreRequest(
        request_id="internal",
        prompt_token_ids=[0, 0, 0],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        model_intermediate_buffer={"combined_embeds": combined_embeds},
    )

    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(encoded)

    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    scheduler_request = SimpleNamespace()
    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 0),
    ):
        result, _ = engine.preprocess_add_request(decoded)

    assert isinstance(result.model_intermediate_buffer, dict)
    assert torch.equal(
        result.model_intermediate_buffer["combined_embeds"],
        combined_embeds,
    )


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
        model_intermediate_buffer={"combined_embeds": "frame"},
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}
    assert result.model_intermediate_buffer == {"combined_embeds": "frame"}
