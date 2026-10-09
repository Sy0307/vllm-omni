# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU validation of the fused MRv2 Talker decode-output kernel (Triton interpreter).

Run with TRITON_INTERPRET=1 (set below before any Triton import). Compares
``_mrv2_decode_output_gpu`` against ``make_omni_output_mrv2``'s eager Torch
path on randomized decode batches, native and non-native, with padding.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import random  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts  # noqa: E402


def make_talker(slots, controls):
    talker = object.__new__(tts.MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._codec_eos_id = 6561
    talker._tts_config = SimpleNamespace(max_position_embeddings=4096)
    talker._mrv2_decode_rows_logged = True
    talker._mrv2_codec_controls = controls
    meta = {key: [torch.tensor(0)] for key in tts._DUPLEX_OUTPUT_META_KEYS}
    talker._mrv2_metadata_by_slot = {slot: meta for slot in range(slots)}
    return talker


def case(rng: random.Random, native: bool):
    slots = 16
    num_reqs = rng.randint(1, slots)
    num_tokens = num_reqs + rng.choice([0, 0, 1, 3])
    slot_ids = rng.sample(range(slots), num_reqs)
    prompt = torch.tensor([rng.choice([2, 10, 400, 4090, 4100]) for _ in range(slots)], dtype=torch.int32)
    steps = [rng.choice([-1, 0, 1, 5, 24, 25, 27, 29, 30, 49, 50, 51, 2047, 2048, 3000]) for _ in range(num_reqs)]
    seq_lens = torch.tensor([int(prompt[s]) + st for s, st in zip(slot_ids, steps)], dtype=torch.int32)
    ids = torch.tensor([rng.choice([0, 17, 6560, 6561]) for _ in range(num_tokens)], dtype=torch.int32)
    empty = torch.tensor([rng.random() < 0.2 for _ in range(slots)], dtype=torch.bool)
    controls = torch.tensor(
        [[rng.choice([-1, 0, 10, 25, 50]), rng.choice([-1, 0, 3, 30]), rng.choice([-1, 0, 1])] for _ in range(slots)],
        dtype=torch.long,
    )
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        has_prefill=False,
        is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        idx_mapping_np=np.array(slot_ids),
        idx_mapping=torch.tensor(slot_ids, dtype=torch.int32),
        input_ids=ids,
        seq_lens=seq_lens,
        query_start_loc_np=np.arange(num_reqs + 1),
        logits_indices=torch.arange(num_reqs),
    )
    req_states = SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt))
    talker = make_talker(slots, controls)
    talker._mrv2_empty_speech = empty
    hidden = torch.zeros(num_tokens, 4)
    out = talker.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=req_states,
        model_intermediate_buffer=[{"native_duplex": native} for _ in range(num_reqs)],
    )
    ref_codes = out.multimodal_outputs["codes"]["audio"]
    ref_valid = out.multimodal_outputs["meta"]["codec_frame_valid"]
    ref_forced, ref_mask = talker._mrv2_forced_eos, talker._mrv2_mask_eos
    codes, valid, forced, mask = tts._mrv2_decode_output_gpu(
        ids,
        batch.idx_mapping,
        seq_lens,
        prompt,
        empty,
        controls if native else None,
        num_reqs=num_reqs,
        eos_id=talker._codec_eos_id,
        context_limit=4095,
    )
    assert torch.equal(codes, ref_codes) and codes.dtype == ref_codes.dtype and codes.shape == ref_codes.shape
    assert torch.equal(valid, ref_valid) and valid.dtype == torch.bool, (valid, ref_valid)
    assert torch.equal(forced, ref_forced), (forced, ref_forced, steps)
    if native:
        assert torch.equal(mask, ref_mask), (mask, ref_mask, steps)
    else:
        assert mask is None and ref_mask is None


def main(cases: int = 300):
    rng = random.Random(0)
    n = 0
    for _ in range(cases):
        for native in (False, True):
            case(rng, native)
            n += 1
    print(f"OK: {n} randomized decode batches match the eager Torch path bit for bit")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
