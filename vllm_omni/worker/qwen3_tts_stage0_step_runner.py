# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

from vllm_omni.worker.omni_step_runner import OmniPreparedStep

logger = init_logger(__name__)


@dataclass(slots=True)
class Qwen3TTSSlot:
    req_id: str | None
    prompt_len: int
    num_computed_tokens: int
    text_offset: int
    codec_len: int
    emitted_chunks: int
    finished: bool


class Qwen3TTSSlotTable:
    def __init__(
        self,
        *,
        max_slots: int,
        hidden_size: int,
        num_quantizers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_slots = max_slots
        self.hidden_size = hidden_size
        self.num_quantizers = num_quantizers
        self.slots = [Qwen3TTSSlot(None, 0, 0, 0, 0, 0, False) for _ in range(max_slots)]
        self.req_to_slot: dict[str, int] = {}
        self.free_slots = list(range(max_slots - 1, -1, -1))

        self.input_ids = torch.empty((max_slots,), dtype=torch.long, device=device)
        self.inputs_embeds = torch.empty((max_slots, hidden_size), dtype=dtype, device=device)
        self.last_talker_hidden = torch.empty((max_slots, hidden_size), dtype=dtype, device=device)
        self.text_step = torch.empty((max_slots, hidden_size), dtype=dtype, device=device)
        self.next_embeds = torch.empty((max_slots, hidden_size), dtype=dtype, device=device)
        self.sampled_codes = torch.empty((max_slots, num_quantizers), dtype=torch.long, device=device)
        self.codec_frames: dict[str, list[torch.Tensor]] = defaultdict(list)

    def allocate(self, req_id: str) -> int:
        existing = self.req_to_slot.get(req_id)
        if existing is not None:
            return existing
        if not self.free_slots:
            raise RuntimeError("Qwen3-TTS Stage0 slot table exhausted")
        slot = self.free_slots.pop()
        self.req_to_slot[req_id] = slot
        self.slots[slot] = Qwen3TTSSlot(req_id, 0, 0, 0, 0, 0, False)
        return slot

    def free(self, req_id: str) -> None:
        slot = self.req_to_slot.pop(req_id, None)
        if slot is None:
            return
        self.slots[slot] = Qwen3TTSSlot(None, 0, 0, 0, 0, 0, False)
        self.codec_frames.pop(req_id, None)
        self.free_slots.append(slot)


@dataclass(slots=True)
class Qwen3TTSPreparedStep(OmniPreparedStep):
    slot_indices: list[int] = field(default_factory=list)
    query_offsets: list[int] = field(default_factory=list)
    next_embeds: torch.Tensor | None = None
    sampled_codes: torch.Tensor | None = None


@dataclass(slots=True)
class Qwen3TTSStage0StepStats:
    fast_path_steps: int = 0
    fast_path_requests: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)


def flatten_codec_frames_for_code2wav(frames_fq: torch.Tensor) -> torch.Tensor:
    if frames_fq.ndim != 2:
        raise ValueError(f"expected [frames, quantizers], got {tuple(frames_fq.shape)}")
    return frames_fq.transpose(0, 1).contiguous().reshape(-1)


class Qwen3TTSStage0StepRunner:
    def __init__(
        self,
        *,
        max_slots: int,
        hidden_size: int,
        num_quantizers: int = 16,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        log_every_n_steps: int = 1000,
    ) -> None:
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_quantizers <= 0:
            raise ValueError(f"num_quantizers must be positive, got {num_quantizers}")
        device = device or torch.device("cpu")
        self.table = Qwen3TTSSlotTable(
            max_slots=max_slots,
            hidden_size=hidden_size,
            num_quantizers=num_quantizers,
            device=device,
            dtype=dtype,
        )
        self.max_slots = max_slots
        self.hidden_size = hidden_size
        self.num_quantizers = num_quantizers
        self.stats = Qwen3TTSStage0StepStats()
        self.log_every_n_steps = log_every_n_steps
        self._last_fallback_reason: str | None = None

    @classmethod
    def from_runner(cls, runner: Any) -> Qwen3TTSStage0StepRunner:
        mtp_buffer = runner.talker_mtp_inputs_embeds.gpu
        model = getattr(runner, "model", None)
        talker_config = getattr(model, "talker_config", None)
        num_quantizers = int(getattr(talker_config, "num_code_groups", 0) or 16)
        step_runner = cls(
            max_slots=int(mtp_buffer.shape[0]),
            hidden_size=int(mtp_buffer.shape[-1]),
            num_quantizers=num_quantizers,
            device=mtp_buffer.device,
            dtype=mtp_buffer.dtype,
        )
        step_runner.table.input_ids = runner.talker_mtp_input_ids.gpu
        step_runner.table.inputs_embeds = runner.talker_mtp_inputs_embeds.gpu
        step_runner.table.last_talker_hidden = runner.last_talker_hidden.gpu
        step_runner.table.text_step = runner.text_step.gpu
        return step_runner

    def _reject(self, reason: str) -> bool:
        self._last_fallback_reason = reason
        return False

    def supports_step(
        self,
        *,
        runner: Any,
        request_ids: list[str],
        num_scheduled_tokens: Sequence[int],
        is_prefill_by_req: Mapping[str, bool],
    ) -> bool:
        if not request_ids:
            return self._reject("empty")
        model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        if not bool(getattr(model_config, "async_chunk", False)):
            return self._reject("async_chunk_disabled")
        if getattr(model_config, "model_stage", None) != "qwen3_tts":
            return self._reject("wrong_stage")
        if not bool(getattr(runner, "has_talker_mtp", False)):
            return self._reject("no_talker_mtp")
        if len(num_scheduled_tokens) != len(request_ids):
            return self._reject("shape_mismatch")
        if any(int(n) != 1 for n in num_scheduled_tokens):
            return self._reject("non_decode_step")
        if any(bool(is_prefill_by_req.get(req_id, True)) for req_id in request_ids):
            return self._reject("prefill")
        self._last_fallback_reason = None
        return True

    def prepare_step(
        self,
        *,
        request_ids: list[str],
        runner: Any,
        input_ids: torch.Tensor,
        req_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> Qwen3TTSPreparedStep:
        batch_size = len(request_ids)
        if batch_size > self.max_slots:
            raise RuntimeError(f"Qwen3-TTS Stage0 slot batch too large: {batch_size} > {self.max_slots}")
        if req_embeds.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden_size={self.hidden_size}, got {req_embeds.shape[-1]}")

        req_index_by_id = {req_id: idx for idx, req_id in enumerate(runner.input_batch.req_ids)}
        slot_indices: list[int] = []
        query_offsets: list[int] = []
        if self.table.input_ids[:batch_size].data_ptr() != input_ids[:batch_size].data_ptr():
            self.table.input_ids[:batch_size].copy_(input_ids[:batch_size].to(device=self.table.input_ids.device))
        if self.table.inputs_embeds[:batch_size].data_ptr() != req_embeds[:batch_size].data_ptr():
            self.table.inputs_embeds[:batch_size].copy_(
                req_embeds[:batch_size].to(device=self.table.inputs_embeds.device)
            )
        if self.table.last_talker_hidden[:batch_size].data_ptr() != last_talker_hidden[:batch_size].data_ptr():
            self.table.last_talker_hidden[:batch_size].copy_(
                last_talker_hidden[:batch_size].to(device=self.table.last_talker_hidden.device)
            )
        if self.table.text_step[:batch_size].data_ptr() != text_step[:batch_size].data_ptr():
            self.table.text_step[:batch_size].copy_(text_step[:batch_size].to(device=self.table.text_step.device))
        for batch_idx, req_id in enumerate(request_ids):
            slot_indices.append(self.table.allocate(req_id))
            req_index = req_index_by_id[req_id]
            query_offsets.append(int(runner.query_start_loc.cpu[req_index]))

        return Qwen3TTSPreparedStep(
            request_ids=list(request_ids),
            slot_indices=slot_indices,
            query_offsets=query_offsets,
            metadata={"batch_size": batch_size},
        )

    def _talker_kwargs(self, runner: Any, request_ids: list[str], device: torch.device) -> dict[str, Any]:
        subtalker_params = getattr(runner.vllm_config.model_config, "subtalker_sampling_params", None)
        if not isinstance(subtalker_params, dict):
            subtalker_params = {}
        talker_kwargs: dict[str, Any] = {
            "do_sample": subtalker_params.get("do_sample"),
            "temperature": subtalker_params.get("temperature"),
            "top_k": subtalker_params.get("top_k"),
            "top_p": subtalker_params.get("top_p"),
        }
        if not request_ids:
            return talker_kwargs
        first_req_id = request_ids[0]
        first_sp = getattr(runner.requests[first_req_id], "sampling_params", None)
        extra_args = getattr(first_sp, "extra_args", None) if first_sp is not None else None
        seed = extra_args.get("qwen3_tts_request_seed") if isinstance(extra_args, dict) else None
        if seed is None:
            return talker_kwargs
        generators = getattr(runner, "_talker_mtp_generators", None)
        if generators is None:
            generators = {}
            runner._talker_mtp_generators = generators
        generator = generators.get(first_req_id)
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators[first_req_id] = generator
        talker_kwargs["generator"] = generator
        return talker_kwargs

    def run_step(
        self,
        *,
        prepared: Qwen3TTSPreparedStep,
        runner: Any,
    ) -> None:
        batch_size = int(prepared.metadata["batch_size"])
        if batch_size == 0:
            return
        if hasattr(runner, "_determine_batch_execution_and_padding"):
            cudagraph_mode, batch_desc, _, _, _ = runner._determine_batch_execution_and_padding(
                num_tokens=batch_size,
                num_reqs=batch_size,
                num_scheduled_tokens_np=np.ones(batch_size, dtype=np.int32),
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
            )
            if not isinstance(runner.talker_mtp, CUDAGraphWrapper):
                cudagraph_mode = CUDAGraphMode.NONE
                num_tokens_padded = batch_size
            else:
                num_tokens_padded = batch_desc.num_tokens
            talker_kwargs = self._talker_kwargs(runner, prepared.request_ids, self.table.input_ids.device)
            with set_forward_context(
                None,
                runner.vllm_config,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
            ):
                next_embeds, sampled_codes = runner.talker_mtp(
                    self.table.input_ids[:num_tokens_padded],
                    self.table.inputs_embeds[:num_tokens_padded],
                    self.table.last_talker_hidden[:num_tokens_padded],
                    self.table.text_step[:num_tokens_padded],
                    **talker_kwargs,
                )
        else:
            next_embeds, sampled_codes = runner.talker_mtp(
                self.table.input_ids[:batch_size],
                self.table.inputs_embeds[:batch_size],
                self.table.last_talker_hidden[:batch_size],
                self.table.text_step[:batch_size],
            )
        prepared.next_embeds = next_embeds[:batch_size]
        prepared.sampled_codes = sampled_codes[:batch_size]

    def commit_step(
        self,
        *,
        prepared: Qwen3TTSPreparedStep,
        runner: Any,
        inputs_embeds: torch.Tensor,
    ) -> None:
        if prepared.next_embeds is None or prepared.sampled_codes is None:
            raise RuntimeError("run_step must be called before commit_step")
        out_key = getattr(runner.model, "talker_mtp_output_key", ("codes", "audio"))
        for idx, req_id in enumerate(prepared.request_ids):
            start_offset = prepared.query_offsets[idx]
            inputs_embeds[start_offset : start_offset + 1] = prepared.next_embeds[idx : idx + 1]
            codes = prepared.sampled_codes[idx : idx + 1]
            update = getattr(runner, "_update_talker_mtp_output", None)
            if update is None:
                self._update_runner_buffer(runner, req_id, out_key, codes)
            else:
                update(req_id, out_key, codes)
        self.record_fast_path(batch_size=len(prepared.request_ids))

    def _update_runner_buffer(self, runner: Any, req_id: str, out_key: tuple[str, str], value: torch.Tensor) -> None:
        type_key, qual = out_key
        existing = runner.model_intermediate_buffer.setdefault(req_id, {})
        existing_sub = existing.setdefault(type_key, {})
        existing_sub[qual] = value.detach().clone()
        req_state = runner.requests.get(req_id)
        if req_state is not None:
            setattr(req_state, "additional_information_cpu", existing)

    def record_fast_path(self, *, batch_size: int) -> None:
        self.stats.fast_path_steps += 1
        self.stats.fast_path_requests += int(batch_size)
        if self.log_every_n_steps > 0 and self.stats.fast_path_steps % self.log_every_n_steps == 0:
            logger.info(
                "Qwen3-TTS Stage0 fast path stats: steps=%d, requests=%d, fallback=%s",
                self.stats.fast_path_steps,
                self.stats.fast_path_requests,
                self.stats.fallback_reasons,
            )

    def record_fallback(self, reason: str) -> None:
        self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1

    def free_request(self, request_id: str) -> None:
        self.table.free(request_id)
