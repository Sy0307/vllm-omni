# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Higgs delay-pattern sampling and owned audio output on Model Runner V2."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


class HiggsModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        model._use_external_decode_cudagraph = True
        logger.info("Higgs MRV2 custom sampler and owned audio snapshots enabled")
        if getattr(model.config, "audio_mrv2_static_inputs", False):
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, model.model.embed_tokens.weight.shape[1]),
                dtype=self.dtype,
                device=device,
            )
        self.direct_payload = bool(getattr(model.config, "audio_mrv2_direct_payload", False))
        self.rng_seeds = None
        if getattr(model.config, "audio_mrv2_batched_rng", False):
            if vllm_config.model_config.max_model_len * model.num_codebooks * model.modality_head.vocab_size >= 2**32:
                raise ValueError("Higgs MRV2 RNG counter domain exceeds uint32")
            self.rng_seeds = torch.zeros(self.scheduler_config.max_num_seqs, dtype=torch.int64, device=device)
        self.sampling_slots = None
        self.request_slots = {}
        if getattr(model.config, "audio_mrv2_slot_parameters", False):
            capacity = self.scheduler_config.max_num_seqs
            self.sampling_slots = {
                "temperature": torch.zeros(capacity, device=device),
                "top_p": torch.ones(capacity, device=device),
                "top_k": torch.full((capacity,), -1, dtype=torch.long, device=device),
                "zeros": torch.zeros(capacity, device=device),
                "ones": torch.ones(capacity, device=device),
            }
        self.requests = {}
        self.generators = {}
        self.metadata_key = None
        self.metadata = None
        logger.info(
            "Higgs MRV2 options: static=%s slot_parameters=%s direct_payload=%s batched_rng=%s",
            self._static_inputs_embeds is not None,
            self.sampling_slots is not None,
            self.direct_payload,
            self.rng_seeds is not None,
        )
        if not model.use_async_omni_output:
            raise ValueError("Higgs MRV2 requires audio_async_payload")

    def add_request(self, req_index, new_req_data):
        params = new_req_data.sampling_params
        is_warmup = str(new_req_data.req_id).startswith("_warmup_")
        if not is_warmup and (
            params.frequency_penalty != 0
            or params.presence_penalty != 0
            or params.repetition_penalty != 1
            or params.logprobs is not None
            or params.prompt_logprobs is not None
            or params.min_tokens != 0
            or params.allowed_token_ids
            or params.bad_words
            or params.logit_bias
            or params.min_p != 0
        ):
            raise ValueError("Higgs MRV2 currently supports temperature/top-k/top-p sampling without text penalties")
        super().add_request(req_index, new_req_data)
        self.requests[new_req_data.req_id] = new_req_data
        self.request_slots[new_req_data.req_id] = req_index
        if self.sampling_slots is not None:
            for name in ("temperature", "top_p", "top_k"):
                self.sampling_slots[name][req_index : req_index + 1].fill_(getattr(params, name))
        self.metadata_key = None
        if self.rng_seeds is not None:
            import secrets

            seed = params.seed if params.seed is not None else secrets.randbits(63)
            seed = seed % 2**64
            self.rng_seeds[req_index] = seed if seed < 2**63 else seed - 2**64
        if params.seed is not None:
            self.generators[new_req_data.req_id] = torch.Generator(device=self.device).manual_seed(params.seed)

    def remove_request(self, req_index_or_id):
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            req_id = self.intermediate_buffer.buffers[index].get("req_id")
            if req_id is not None:
                self.model.on_requests_finished({req_id})
                self.requests.pop(req_id, None)
                self.request_slots.pop(req_id, None)
                self.generators.pop(req_id, None)
                self.metadata_key = None
        super().remove_request(req_index_or_id)

    def run_preprocess(self, input_batch, model_inputs, req_states=None, mtp_batch_descriptor_dispatcher=None):
        model = self.model
        model._resolve_token_ids()
        audio_mode = True
        for row, req_id in enumerate(input_batch.req_ids):
            request = self.requests[req_id]
            prompt = request.prompt_token_ids
            slot = int(input_batch.idx_mapping_np[row])
            end = int(req_states.num_computed_prefill_tokens[slot]) + int(input_batch.num_scheduled_tokens[row])
            audio_mode &= bool(prompt and prompt[-1] == model._audio_continuation_id and end >= len(prompt))
        model.update_decode_step_metadata(
            input_ids=model_inputs["input_ids"][: input_batch.num_tokens],
            positions=input_batch.positions,
            omni_query_start_loc=input_batch.query_start_loc[: input_batch.num_reqs + 1],
            req_ids=input_batch.req_ids,
            audio_prompt_mode_rows=input_batch.num_reqs if audio_mode else 0,
        )

        if self._static_inputs_embeds is not None:
            self._prepare_static_embeddings(input_batch, model_inputs)

    def _prepare_static_embeddings(self, input_batch, model_inputs):
        # Feedback and reference substitution are request-dependent and belong
        # outside the captured backbone. Use only actual tokens: graph padding
        # must not change the one-token-per-request decode classification.
        n = input_batch.num_tokens
        ids = model_inputs["input_ids"][:n]
        safe_ids = torch.where(ids < 0, torch.zeros_like(ids), ids)
        embeds = self.model.model.embed_tokens(safe_ids)
        info = model_inputs.get("model_intermediate_buffer")
        if input_batch.has_prefill and info:
            embeds = self.model._apply_ref_audio_substitution(
                embeds,
                ids,
                input_batch.positions[:n],
                info,
            )
        embeds = self._apply_audio_feedback(embeds, ids, input_batch)
        self._static_inputs_embeds[:n].copy_(embeds)
        padded = input_batch.num_tokens_after_padding
        self._static_inputs_embeds[n:padded].zero_()

    def _apply_audio_feedback(self, embeds, ids, input_batch):
        if input_batch.num_tokens == input_batch.num_reqs:
            return self.model._apply_audio_feedback(embeds, ids)
        # Every scheduled request has a nonempty span. Gather its first token
        # for all rows, then mask on-device. Boolean indexing would first ask
        # the GPU how many rows matched, synchronizing every mixed prefill.
        n = input_batch.num_reqs
        qsl = input_batch.query_start_loc[: n + 1]
        starts = qsl[:-1].long()
        model = self.model
        model._ensure_decode_state_capacity(n, embeds.device)
        audio = model.multimodal_embedding(model._decode_last_codes[:n]).to(embeds.dtype)
        use_audio = model._decode_has_codes[:n] & ((qsl[1:] - qsl[:-1]) == 1)
        current = embeds.index_select(0, starts)
        result = embeds.clone()
        result.index_copy_(0, starts, torch.where(use_audio[:, None], audio, current))
        return result

    def run_postprocess(self, hidden_states, input_batch):
        self.model.finish_decode_step_forward()

    def postprocess_model_output(self, model_output, input_batch, req_states):
        return model_output, None

    def sampling_metadata(self, req_ids, slots=None):
        key = tuple(req_ids)
        if self.metadata_key != key:
            params = [self.requests[r].sampling_params for r in req_ids]
            n = len(params)
            if self.sampling_slots is not None:
                if slots is None:
                    slots = torch.tensor([self.request_slots[r] for r in req_ids], device=self.device)
                temperature, top_p, top_k = (
                    self.sampling_slots[name].index_select(0, slots) for name in ("temperature", "top_p", "top_k")
                )
                zeros = self.sampling_slots["zeros"][:n]
                ones = self.sampling_slots["ones"][:n]
            else:
                temperature = torch.tensor([p.temperature for p in params], device=self.device)
                top_p = torch.tensor([p.top_p for p in params], device=self.device)
                top_k = torch.tensor([p.top_k for p in params], device=self.device)
                zeros = torch.zeros(n, device=self.device)
                ones = torch.ones(n, device=self.device)
            self.metadata = SamplingMetadata(
                temperature=temperature,
                all_greedy=all(p.temperature == 0 for p in params),
                all_random=all(p.temperature != 0 for p in params),
                top_p=top_p,
                top_k=top_k,
                generators={i: self.generators[r] for i, r in enumerate(req_ids) if r in self.generators},
                max_num_logprobs=None,
                no_penalties=True,
                prompt_token_ids=None,
                frequency_penalties=zeros,
                presence_penalties=zeros,
                repetition_penalties=ones,
                output_token_ids=[[] for _ in params],
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                logitsprocs=LogitsProcessors(),
            )
            self.metadata_key = key
        return self.metadata

    def sample_omni_output(self, hidden_states, input_batch, req_states, grammar_output):
        # Upstream warmup exercises its text sampler, including grammar and
        # penalties, with synthetic non-audio prompts. Keep that path intact.
        if all(str(r).startswith("_warmup_") for r in input_batch.req_ids):
            return None
        if grammar_output is not None or input_batch.num_draft_tokens:
            raise ValueError("Higgs MRV2 does not support grammar or speculative sampling")
        metadata = self.sampling_metadata(input_batch.req_ids, input_batch.idx_mapping[: input_batch.num_reqs])
        hidden = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(hidden, metadata)
        noise = None
        if self.rng_seeds is not None and self.model._step_audio_mode_rows == input_batch.num_reqs:
            from .mrv2_rng import exponential_noise

            noise = exponential_noise(
                self.rng_seeds,
                input_batch.idx_mapping[: input_batch.num_reqs],
                input_batch.seq_lens,
                req_states.prompt_len.gpu,
                self.model.num_codebooks,
                self.model.modality_head.vocab_size,
            )
        self.model._mrv2_audio_noise = noise
        try:
            sampled = self.model.sample(logits, metadata)
        finally:
            self.model._mrv2_audio_noise = None
        count, rejected = get_num_sampled_and_rejected(
            torch.ones(input_batch.num_reqs, dtype=torch.int32, device=hidden.device),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            req_states.prefill_len.gpu,
        )
        output = SamplerOutput(sampled.sampled_token_ids, sampled.logprobs_tensors, None, count, rejected)
        payload = self.model.post_sample_multimodal_outputs(
            req_ids=input_batch.req_ids,
            invalid_req_indices=[],
            multimodal_outputs=None,
        )
        return output, count, rejected, payload

    def finalize_audio_snapshot(self, payload: dict[str, Any], num_sampled: list[int]):
        if payload and "_higgs_audio_snapshot" in payload and getattr(self, "direct_payload", False):
            from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

            snapshot = payload["_higgs_audio_snapshot"]
            if snapshot.device.type != "cpu":
                raise RuntimeError("Higgs snapshot finalized before D2H")
            codes = snapshot[:, : self.model.num_codebooks].clone()
            valid = snapshot[:, self.model.num_codebooks].tolist()
            invalid = set(payload.get("_higgs_invalid_rows", ()))
            return RequestOutputSnapshot(
                [
                    {"codes.audio": codes[i : i + 1]} if valid[i] and n and i not in invalid else None
                    for i, n in enumerate(num_sampled)
                ]
            )
        if payload and "_higgs_audio_snapshot" in payload:
            payload = dict(payload)
            payload["_higgs_invalid_rows"] = tuple(i for i, n in enumerate(num_sampled) if n == 0)
        return self.model.finalize_multimodal_outputs_from_cpu_snapshot(payload)
