# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
from vllm_omni.model_executor.models.qwen3_tts.streaming_code2wav import StreamingCode2Wav
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingCodecDecoder

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_streaming_graphs_preserve_reordered_ring_state_and_owned_pcm():
    decoder = _make_decoder().to(device="cuda", dtype=torch.bfloat16)
    # Real tokenizer configs declare head_dim; the small CPU fixture omits it.
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    backend = StreamingCode2Wav(
        decoder, num_slots=2, max_batch_size=2, batch_sizes=[2], frame_sizes=[1, 25], capture=True
    )
    assert all(graph.sizes == [2] for graph in backend.graphs.values())
    oracle = StreamingCodecDecoder(decoder, 2)
    caches = [{"skip_first_audio": True}, {}]
    positions = [0, 0]
    kept = None
    for ordinal, length in enumerate([1, 25, 25, 25, 25, 25, 25, 25, 7]):
        # Replay one live row into a B2 bucket, then resume both streams.
        order = [1] if ordinal == 7 else ([0, 1] if ordinal % 2 == 0 else [1, 0])
        codes = torch.randint(0, 32, (len(order), 2, length), device="cuda")
        terminal = ordinal == 8
        actual = backend.decode(
            codes,
            [length] * len(order),
            request_ids=[str(index) for index in order],
            caches=[caches[index] for index in order],
            terminal=[terminal] * len(order),
            legacy_decoder=decoder,
            chunk_size=300,
            left_context_size=25,
        )
        # Match the graph's B2 compute shape, including its scratch row.
        value = codes.transpose(1, 2).to(torch.int32).contiguous()
        value = torch.nn.functional.pad(value, (0, 0, 0, 25 - length if terminal else 0, 0, 2 - len(order)))
        expected = oracle(
            value,
            torch.tensor(order + [oracle.scratch_slot] * (2 - len(order)), device="cuda", dtype=torch.int32),
            torch.tensor(
                [positions[index] for index in order] + [0] * (2 - len(order)), device="cuda", dtype=torch.int32
            ),
        )
        for row, index in enumerate(order):
            skip = int(index == 0 and positions[index] == 0)
            torch.testing.assert_close(
                actual[row], expected[row : row + 1, skip * oracle.spf : length * oracle.spf], rtol=1e-5, atol=1e-6
            )
            positions[index] += length
        if ordinal == 1:
            kept = (actual[0], actual[0].clone())
    assert kept is not None
    torch.testing.assert_close(*kept, rtol=0, atol=0)
    backend.release(["0", "1"])
    value = torch.randint(0, 32, (1, 2, 1), device="cuda")
    actual = backend.decode(
        value,
        [1],
        request_ids=["fresh"],
        caches=[{}],
        terminal=[True],
        legacy_decoder=decoder,
        chunk_size=300,
        left_context_size=25,
    )[0]
    expected = oracle(
        torch.nn.functional.pad(value.transpose(1, 2).to(torch.int32), (0, 0, 0, 0, 0, 1)).contiguous(),
        torch.tensor([0, oracle.scratch_slot], device="cuda", dtype=torch.int32),
        torch.tensor([0, 0], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected[:1], rtol=1e-5, atol=1e-6)
