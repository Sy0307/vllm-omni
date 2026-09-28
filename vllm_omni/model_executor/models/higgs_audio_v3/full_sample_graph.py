# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental dense decode sampler graph; preserves request torch RNG draws.

The first use of each exact shape compares eager and captured state transitions
with identical noise before replaying the actual transition. CPU payload staging
remains outside capture. This module does not handle mixed prefill batches.
"""

import copy

import torch
from vllm.v1.outputs import SamplerOutput

_STATE = (
    "_decode_last_codes",
    "_decode_has_codes",
    "_decode_delay_count",
    "_decode_eoc_countdown",
    "_decode_generation_done",
)


def run_dense_sample(model, hidden, logits, metadata, force_audio_inputs=False, active_batch=None):
    batch = hidden.shape[0]
    if active_batch is None and getattr(model.config, "audio_mrv2_sample_graph_buckets", False):
        bucket = next((n for n in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128) if n >= batch), batch)
        if bucket != batch:
            # Dummy rows own no request state. The batch-state gather replaces
            # these rows before any later request can use them.
            model._ensure_decode_state_capacity(bucket, hidden.device)
            padded_hidden = hidden.new_zeros((bucket, hidden.shape[1]))
            padded_hidden[:batch].copy_(hidden)
            padded_metadata = copy.copy(metadata)
            for name in ("temperature", "top_k", "top_p"):
                value = getattr(metadata, name)
                if value is not None:
                    padded = value.new_ones(bucket)
                    padded[:batch].copy_(value)
                    setattr(padded_metadata, name, padded)
            original_ids = model._last_step_input_ids
            padded_ids = original_ids.new_full((bucket,), model._audio_continuation_id)
            if not force_audio_inputs:
                padded_ids[:batch].copy_(original_ids)
            model._last_step_input_ids = padded_ids
            try:
                output = run_dense_sample(
                    model,
                    padded_hidden,
                    logits.new_empty((bucket, logits.shape[1])),
                    padded_metadata,
                    force_audio_inputs=force_audio_inputs,
                    active_batch=batch,
                )
            finally:
                model._last_step_input_ids = original_ids
            return SamplerOutput(sampled_token_ids=output.sampled_token_ids[:batch], logprobs_tensors=None)
    actual = batch if active_batch is None else active_batch
    effective_ids = (
        torch.full((batch,), model._audio_continuation_id, dtype=model._last_step_input_ids.dtype, device=hidden.device)
        if force_audio_inputs
        else model._last_step_input_ids
    )
    books = model.num_codebooks
    vocab = model.modality_head.vocab_size
    noise = torch.empty((batch * books, vocab), dtype=torch.float32, device=hidden.device)
    if actual < batch:
        noise[actual * books :].fill_(1)
    if metadata.generators:
        for i in range(actual):
            noise[i * books : (i + 1) * books].exponential_(generator=metadata.generators.get(i))
    else:
        noise[: actual * books].exponential_()
    fields = ("temperature", "top_k", "top_p")
    # force_audio_inputs controls capture-time Python branches (including
    # the mixed-tail assertion). Never reuse that graph for real EOS inputs.
    key = (
        batch,
        hidden.dtype,
        hidden.device,
        bool(force_audio_inputs),
        bool(metadata.all_greedy),
        tuple(getattr(metadata, f) is None for f in fields),
        tuple(getattr(model, name).data_ptr() for name in _STATE),
    )
    cache = getattr(model, "_dense_sample_graphs", None)
    if cache is None:
        cache = model._dense_sample_graphs = {}
    entry = cache.get(key)
    if entry is None:
        static_hidden = hidden.clone()
        static_input_ids = effective_ids.clone()
        static_noise = noise.clone()
        static_metadata = copy.copy(metadata)
        for name in fields:
            value = getattr(metadata, name)
            setattr(static_metadata, name, None if value is None else value.clone())
        # Ensure the reusable staging exists before capture. Each graph keeps
        # its own reference if a later larger batch grows the model buffer.
        staging = model._get_audio_gpu_staging_buffer(batch, hidden.device)
        state = {name: getattr(model, name).clone() for name in _STATE}
        original_ids = model._last_step_input_ids
        original_qsl = model._last_step_query_start_loc
        original_mode_rows = getattr(model, "_step_audio_mode_rows", 0)
        original_tail_rows = getattr(model, "_step_audio_tail_rows", 0)
        original_direct_rows = getattr(model, "_fast_audio_direct_rows", 0)
        # Dense decode has one token per row. Own this input instead of
        # capturing a view of the runner's resizable staging buffer.
        static_qsl = torch.arange(batch + 1, dtype=torch.int32, device=hidden.device)

        def restore():
            for name, value in state.items():
                getattr(model, name).copy_(value)

        try:
            model._in_audio_sample_graph = True
            # Outer eligibility and real-input validation already ran. The
            # owned graph inputs represent exactly one audio-state row each.
            # Normalize capture-time host dispatch; never capture an outer
            # mixed-batch assertion or a stock-sampler branch into this graph.
            model._step_audio_mode_rows = 0
            model._step_audio_tail_rows = 0
            model._fast_audio_direct_rows = batch
            model._audio_graph_noise = static_noise
            model._last_step_input_ids = static_input_ids
            model._last_step_query_start_loc = static_qsl
            model._last_logits_hidden = static_hidden
            reference = model.sample(logits, static_metadata)
            expected_ids = reference.sampled_token_ids.clone()
            expected_staging = staging.clone()
            expected_state = {name: getattr(model, name).clone() for name in _STATE}
            restore()
            model._last_logits_hidden = static_hidden
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model.sample(logits, static_metadata)
            # CUDA capture records work; it does not execute the transition.
            # Validate only after replay from the preserved initial state.
            restore()
            graph.replay()
            torch.testing.assert_close(output.sampled_token_ids, expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(staging, expected_staging, atol=0, rtol=0)
            for name, expected in expected_state.items():
                torch.testing.assert_close(getattr(model, name), expected, atol=0, rtol=0)
            restore()
            print(f"HIGGS_DENSE_SAMPLE_GRAPH_PARITY batch={batch} exact=True", flush=True)
        finally:
            model._in_audio_sample_graph = False
            model._step_audio_mode_rows = original_mode_rows
            model._step_audio_tail_rows = original_tail_rows
            model._fast_audio_direct_rows = original_direct_rows
            model._audio_graph_noise = None
            model._last_step_input_ids = original_ids
            model._last_step_query_start_loc = original_qsl
            model._last_logits_hidden = None
        # Graphs reference storage, not Python owners. Model caches and state
        # can grow after capture; keep their original allocations alive.
        keepalive = [getattr(model, name) for name in _STATE]
        for name in ("_row_index_cache", "_codebook_index_cache", "_boc_frame_cache"):
            keepalive.extend(getattr(model, name).values())
        keepalive.append(static_qsl)
        entry = (graph, static_hidden, static_input_ids, static_noise, static_metadata, output, staging, keepalive)
        cache[key] = entry
    graph, static_hidden, static_input_ids, static_noise, static_metadata, output, staging, keepalive = entry
    static_hidden.copy_(hidden)
    static_input_ids.copy_(effective_ids)
    static_noise.copy_(noise)
    for name in fields:
        value = getattr(metadata, name)
        if value is not None:
            getattr(static_metadata, name).copy_(value)
    graph.replay()
    model._finish_audio_staging(staging[:actual], actual, hidden.device, actual)
    return SamplerOutput(sampled_token_ids=output.sampled_token_ids.clone(), logprobs_tensors=None)
