# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import _normalize_reference
from vllm_omni.model_executor.models.minicpmo_4_5.reference_audio import (
    decode_reference_audio,
    encode_reference_audio,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    tts2code2wav_async_chunk,
    tts2code2wav_full_payload,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reference_survives_engine_ipc_buffer_and_codec_bridge() -> None:
    waveform = torch.linspace(-1, 1, 96256)
    payload = encode_reference_audio(waveform)
    request = OmniEngineCoreRequest(
        request_id="ref-request",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        model_intermediate_buffer={"codes": {"ref": payload}, "meta": {"ref_audio_sr": 16000}},
    )
    wire = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    assert isinstance(wire.model_intermediate_buffer["codes"]["ref"]["data"], bytes)
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(
        0,
        SimpleNamespace(
            req_id=request.request_id, mm_features=[], model_intermediate_buffer=wire.model_intermediate_buffer
        ),
    )
    stored = buffer.buffers[0]
    torch.testing.assert_close(decode_reference_audio(stored["codes"]["ref"]), waveform, rtol=0, atol=0)
    wire.model_intermediate_buffer = stored
    for processor, kwargs in ((tts2code2wav_full_payload, {}), (tts2code2wav_async_chunk, {"is_finished": True})):
        codec = processor(SimpleNamespace(), torch.tensor([10, 20]), wire, **kwargs)
        torch.testing.assert_close(codec.codes.ref, waveform, rtol=0, atol=0)
    compact, compact_sr = _normalize_reference(payload, 16000)
    legacy, legacy_sr = _normalize_reference(waveform.tolist(), 16000)
    assert compact_sr == legacy_sr
    torch.testing.assert_close(compact, legacy, rtol=0, atol=0)
    buffer.remove_request(0)
    buffer.add_request(0, SimpleNamespace(req_id="replacement", mm_features=[]))
    assert "codes" not in buffer.buffers[0]
    assert request.request_id not in buffer.req_id_to_index
