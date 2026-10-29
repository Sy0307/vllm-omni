# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Continuation keeps assistant reference codes; generation starts a new segment."""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.moss_tts.test_stage0_codec_prime import decoder
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    talker2codec,
    talker2codec_raw_async_chunk,
    talker2codec_raw_async_chunk_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def prompt(continuation):
    return {
        "prompt_token_ids": [1, 2, 3, 4],
        "additional_information": {
            "codes": {"ref": torch.tensor([[1024, 1024], [1024, 1024], [1, 2], [3, 4]])},
            "meta": {"ref_code_len": 2} if continuation else {},
        },
    }


@pytest.mark.parametrize("continuation", [False, True])
def test_sync_bridge_uses_stage_client_call_contract(continuation):
    output = SimpleNamespace(multimodal_outputs={"codes": {"audio": torch.tensor([[5, 6], [7, 8]])}})
    # StageEngineCoreClient passes selected outputs, original prompt, and the
    # multimodal-data flag positionally (there is no stage index argument).
    result = talker2codec([output], prompt(continuation), False)[0]
    expected = [[1, 3, 5, 7], [2, 4, 6, 8]] if continuation else [[5, 7], [6, 8]]
    assert result["prompt_token_ids"] == torch.tensor(expected).flatten().tolist()
    assert result["additional_information"]["meta"]["left_context_size"] == (2 if continuation else 0)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_stream_prefix_once_and_generation_unchanged(batched, serialized):
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"initial_codec_chunk_frames": 1, "codec_chunk_frames": 2})
    )
    requests = []
    for mode in (True, False):
        info = prompt(mode)["additional_information"]
        requests.append(
            SimpleNamespace(
                request_id=f"internal-{mode}",
                external_req_id=str(mode),
                additional_information=serialize_additional_information(info) if serialized else info,
            )
        )
    packets = []
    for index in range(3):
        outputs = [{"codes": {"audio": torch.tensor([[5 + index, 6 + index]])}} for _ in requests]
        if batched:
            step = talker2codec_raw_async_chunk_batch(manager, outputs, requests, [index == 2] * 2)
        else:
            step = [talker2codec_raw_async_chunk(manager, out, req, index == 2) for out, req in zip(outputs, requests)]
        packets.append(step)
        for req, packet in zip(requests, step):
            if packet is not None:
                manager.put_req_chunk[req.external_req_id] += 1
    assert packets[0][0].codes.audio.reshape(2, -1).tolist() == [[1, 3, 5], [2, 4, 6]]
    assert packets[0][0].meta.left_context_size == 2
    assert packets[0][1].codes.audio.tolist() == [5, 6]
    assert packets[0][1].meta.left_context_size == 0
    assert packets[1] == [None, None]
    for packet in packets[2]:
        assert packet.codes.audio.reshape(2, -1).tolist() == [[6, 7], [7, 8]]
        assert packet.meta.left_context_size == 0
        assert bool(packet.meta.finished)
    assert not manager.request_payload and not manager.code_prompt_token_ids


def test_stream_decoder_trims_reference_pcm_but_retains_history():
    d = decoder()

    # The fake codec is causal cumulative sum: reference sums 3 + 7, then
    # target sums 11 + 15. An unprimed decoder would produce 11, 26.
    def decode(codes, left, finished):
        return d.forward(
            input_ids=torch.tensor(codes),
            seq_token_counts=[len(codes)],
            runtime_additional_information=[
                {
                    "request_id": "a",
                    "meta": {
                        "req_id": ["a"],
                        "left_context_size": left,
                        "finished": finished,
                    },
                }
            ],
        ).multimodal_outputs["model_outputs"][0]

    first = decode([1, 3, 5, 2, 4, 6], 2, False)
    last = decode([7, 8], 0, True)
    torch.testing.assert_close(torch.cat([first, last], -1), torch.tensor([[21.0, 36.0], [21.0, 36.0]]))
    assert not d._stream_req_slots
    torch.testing.assert_close(decode([5, 6], 0, True), torch.tensor([[11.0], [11.0]]))
