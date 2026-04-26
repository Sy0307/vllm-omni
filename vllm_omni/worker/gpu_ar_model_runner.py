"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, make_empty_encoder_model_runner_output
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
)
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    cudagraph_stats: Any
    # OMNI: multimodal_outputs field for omni-specific multimodal handling
    multimodal_outputs: Any
    # slot_mappings for attention/drafter (aligned with upstream v1 API)
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None


class GPUARModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin):
    """Autoregressive GPU model runner that returns hidden states per request.

    Follows the v0.12 two-phase execute/sample flow from GPUModelRunner, and
    reuses Omni hooks for additional_information / multimodal outputs. This
    class only overrides sample_tokens to expose hidden states + multimodal
    outputs per request while keeping Async output semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    def _build_model_sampler_output_token_ids(self) -> list[list[int]]:
        """Build decoded-token history for custom model samplers.

        vLLM only populates sampling_metadata.output_token_ids when penalties or
        logits processors require it. CosyVoice3's custom RAS sampler also
        depends on this history, so we reconstruct it directly from the input
        batch for prefer_model_sampler models.
        """
        req_output_token_ids = getattr(self.input_batch, "req_output_token_ids", [])
        req_ids = list(getattr(self.input_batch, "req_ids", []))
        output_token_ids = [list(req_output_token_ids[idx] or []) for idx in range(len(req_ids))]

        sampled_token_ids_cpu = getattr(self.input_batch, "sampled_token_ids_cpu", None)
        async_copy_ready_event = getattr(self.input_batch, "async_copy_ready_event", None)
        prev_req_id_to_index = getattr(self.input_batch, "prev_req_id_to_index", None)
        if sampled_token_ids_cpu is None or not output_token_ids or prev_req_id_to_index is None:
            return output_token_ids

        sampled_token_ids: list[list[int]] | None = None
        for index, req_id in enumerate(req_ids):
            prev_index = prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_history = output_token_ids[index]
            if not req_history or req_history[-1] != -1:
                continue
            if sampled_token_ids is None:
                assert async_copy_ready_event is not None
                async_copy_ready_event.synchronize()
                sampled_token_ids = sampled_token_ids_cpu.tolist()
            new_ids = list(sampled_token_ids[prev_index])
            if not new_ids:
                continue
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            first_placeholder = req_history.index(-1)
            num_placeholders = len(req_history) - first_placeholder
            num_to_replace = min(num_sampled_ids, num_placeholders)
            req_history[first_placeholder : first_placeholder + num_to_replace] = new_ids[:num_to_replace]

        return output_token_ids

    def _sampling_metadata_for_model_sampler(self, sampling_metadata):
        output_token_ids = self._build_model_sampler_output_token_ids()
        if output_token_ids == sampling_metadata.output_token_ids:
            return sampling_metadata
        return replace(sampling_metadata, output_token_ids=output_token_ids)

    def capture_model(self) -> int:
        # Setup merged/unified decode BEFORE graph capture so forward()
        # includes codebook injection + inline sampling + Fast AR.
        # VLLM_FISH_MERGED_DECODE=1: full merge (eliminates talker_mtp)
        # VLLM_FISH_UNIFIED_DECODE=1: partial merge (inline sampling only)
        import os
        merged = os.environ.get("VLLM_FISH_MERGED_DECODE", "0") == "1"
        unified_only = os.environ.get("VLLM_FISH_UNIFIED_DECODE", "0") == "1"
        allow_unified_experiment = os.environ.get("VLLM_FISH_ALLOW_UNSAFE_UNIFIED_DECODE", "0") == "1"
        if (merged or unified_only) and not allow_unified_experiment:
            logger.warning(
                "VLLM_FISH_MERGED_DECODE/VLLM_FISH_UNIFIED_DECODE are disabled by default: "
                "the current inline sampler/code relay path is experimental and can truncate Fish Speech audio. "
                "Set VLLM_FISH_ALLOW_UNSAFE_UNIFIED_DECODE=1 only for controlled profiling."
            )
            merged = False
            unified_only = False

        if (merged or unified_only) and not getattr(self.model, "_unified_decode", False):
            setup_name = "setup_merged_decode" if merged else "setup_unified_decode"
            setup_fn = getattr(self.model, setup_name, None)
            if callable(setup_fn):
                max_bs = max(
                    self.scheduler_config.max_num_seqs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
                setup_fn(max_bs, self.device)
                logger.info(
                    "Fish unified decode setup: max_batch_size=%d, merged=%s",
                    max_bs,
                    merged,
                )

        result = super().capture_model()

        # Set _codes_valid so talker_mtp graph captures the buffer-read
        # path instead of the Fast AR re-computation path.
        if getattr(self.model, "_unified_decode", False):
            self.model._codes_valid = True

        self._capture_talker_mtp_graphs()
        return result

    def _capture_talker_mtp_graphs(self) -> None:
        from vllm_omni.worker.gpu_model_runner import CUDAGraphWrapper

        if not self.has_talker_mtp or not isinstance(self.talker_mtp, CUDAGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.distributed.parallel_state import graph_capture

        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        capture_sizes = sorted(capture_sizes, reverse=True)
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_forward_context(
                            None,
                            self.vllm_config,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_forward_context(
                        None,
                        self.vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.cuda.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )

        if self.routed_experts_initialized:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    output = make_empty_encoder_model_runner_output(scheduler_output)
                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids
                    return output

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    self._dummy_run(1)

                # Capture KV extraction results before early return;
                # sample_tokens() is skipped on this path so the IDs
                # would otherwise be silently overwritten next step.
                kv_ids = self.kv_extracted_req_ids
                self.kv_extracted_req_ids = None

                if not has_kv_transfer_group():
                    output = EMPTY_MODEL_RUNNER_OUTPUT
                else:
                    output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                if kv_ids:
                    output = copy(output)
                    output.kv_extracted_req_ids = kv_ids

                return output

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

            has_separate_kv_update = not all(
                all(g.backend.forward_includes_kv_cache_update for g in self.attn_groups[id])
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded if pad_attn or has_separate_kv_update else num_tokens_unpadded,
                num_reqs_padded=(num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

        # Let the model adjust inputs before forward (e.g. restore input_ids
        # for multimodal position detection, fix decode position offsets).
        if hasattr(self.model, "prepare_runner_inputs"):
            input_ids, positions = self.model.prepare_runner_inputs(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                req_ids=req_ids[:num_reqs],
                num_computed_tokens=[int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(num_reqs)],
                num_scheduled_tokens=[int(num_scheduled_tokens_np[i]) for i in range(num_reqs)],
                input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        with (
            nullcontext(),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,  # OMNI: required for KV cache operations
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )

            # [Omni] Map pending ropes metadata to req_ids.
            if hasattr(self.model, "flush_pending_metadata"):
                self.model.flush_pending_metadata(list(req_ids))

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(model_output)

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens_padded)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: pass slot_mappings for drafter
        )
        self.kv_connector_output = kv_connector_output

        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                sampler_output = model_sample(
                    logits,
                    self._sampling_metadata_for_model_sampler(sampling_metadata),
                )
                if sampler_output is not None:
                    return sampler_output
            self.input_batch.update_async_output_token_ids()
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None

        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: unpack slot_mappings for drafter
        ) = self.execute_model_state
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        # Correct padding values of prompt_token_ids to match the logits vocabulary size
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,  # OMNI: pass slot_mappings to drafter (upstream v1 API)
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle() or spec_config.uses_draft_model() or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                        self.optimistic_seq_lens_cpu,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(
                        len(self.input_batch.req_ids), self.num_spec_tokens
                    )
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if self.speculative_config is not None:
            self.finalize_kv_connector()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        post_sample_codes_cpu: dict[str, torch.Tensor] = {}
        fish_cuda_ipc_relay = os.environ.get("VLLM_FISH_CUDA_IPC_RELAY", "0") == "1"
        fish_cuda_ipc_stage0 = (
            fish_cuda_ipc_relay
            and os.environ.get("VLLM_FISH_CUDA_IPC_STAGE0", "0") == "1"
        )
        if (
            getattr(self.model, "_post_sample_codebook", False)
            and isinstance(sample_hidden_states, torch.Tensor)
            and valid_sampled_token_ids
        ):
            with record_function_or_nullcontext("gpu_model_runner: fish_post_sample_codebook"):
                sem_begin = getattr(self.model, "_semantic_begin_id", 0)
                sem_end = getattr(self.model, "_semantic_end_id", 0)
                hidden_row_indices: list[int] = []
                token_ids: list[int] = []
                code_req_ids: list[str] = []
                for rid in req_ids_output_copy:
                    idx = req_id_to_index_output_copy.get(rid)
                    if idx is None or idx >= len(valid_sampled_token_ids):
                        continue
                    ids = valid_sampled_token_ids[idx]
                    if not ids:
                        continue
                    tok = int(ids[0])
                    if tok < sem_begin or tok > sem_end:
                        continue
                    if idx >= int(sample_hidden_states.shape[0]):
                        continue
                    hidden_row_indices.append(idx)
                    token_ids.append(tok)
                    code_req_ids.append(rid)

                if hidden_row_indices:
                    from vllm_omni.worker.gpu_model_runner import CUDAGraphWrapper

                    row_indices = torch.tensor(hidden_row_indices, device=self.device, dtype=torch.long)
                    h = sample_hidden_states.index_select(0, row_indices).to(device=self.device, dtype=torch.bfloat16)
                    toks = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                    decode_batch_size = len(hidden_row_indices)
                    _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=decode_batch_size,
                        num_reqs=decode_batch_size,
                        num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    if not isinstance(self.talker_mtp, CUDAGraphWrapper):
                        _cudagraph_mode = CUDAGraphMode.NONE
                        num_tokens_padded = decode_batch_size
                    else:
                        num_tokens_padded = batch_desc.num_tokens

                    req_input_ids = self.talker_mtp_input_ids.gpu[:num_tokens_padded]
                    req_embeds = self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded]
                    last_talker_hidden = self.last_talker_hidden.gpu[:num_tokens_padded]
                    text_step = self.text_step.gpu[:num_tokens_padded]
                    req_input_ids[:decode_batch_size].copy_(toks)
                    embeds = self.model.embed_input_ids(toks.reshape(-1, 1).to(torch.long)).reshape(decode_batch_size, -1)
                    req_embeds[:decode_batch_size].copy_(embeds.to(device=self.device, dtype=self.dtype))
                    last_talker_hidden[:decode_batch_size].copy_(h.reshape(decode_batch_size, -1))
                    text_step[:decode_batch_size].zero_()

                    with set_forward_context(
                        None,
                        self.vllm_config,
                        cudagraph_runtime_mode=_cudagraph_mode,
                        batch_descriptor=batch_desc,
                    ):
                        _, codes = self.talker_mtp(req_input_ids, req_embeds, last_talker_hidden, text_step)
                    codes = codes[:decode_batch_size]
                    out_key = getattr(self.model, "talker_mtp_output_key", "audio_codes")
                    codes_cpu = None if fish_cuda_ipc_stage0 else codes.detach().to("cpu").contiguous()
                    for row, rid in enumerate(code_req_ids):
                        one = codes[row : row + 1]
                        self._update_intermediate_buffer(
                            rid,
                            {
                                "_post_sample_prev_codes": one,
                            },
                        )
                        self.model_intermediate_buffer.get(rid, {}).pop(out_key, None)
                        if codes_cpu is None:
                            post_sample_codes_cpu[rid] = one.detach().contiguous()
                        else:
                            post_sample_codes_cpu[rid] = codes_cpu[row : row + 1]

        # kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )

        self._process_additional_information_updates(
            hidden_states, multimodal_outputs, num_scheduled_tokens_np, scheduler_output
        )

        inline_dac_flush_cpu: dict[str, torch.Tensor] = {}
        if valid_sampled_token_ids and getattr(self.model, "_inline_dac", False):
            with record_function_or_nullcontext("gpu_model_runner: fish_inline_dac_finish_flush"):
                inline_dac_flush_cpu = self._flush_inline_dac_on_finished_samples(
                    req_ids_output_copy,
                    req_id_to_index_output_copy,
                    valid_sampled_token_ids,
                )

        # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
        # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
        mm_cpu: dict[str, object] = {}
        if isinstance(multimodal_outputs, dict) and multimodal_outputs:
            fish_code_output_key = getattr(self.model, "talker_mtp_output_key", "audio_codes")
            for k, v in multimodal_outputs.items():
                try:
                    if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                        if fish_cuda_ipc_stage0 and k == fish_code_output_key:
                            mm_cpu[k] = v.detach().contiguous()
                        else:
                            mm_cpu[k] = v.detach().to("cpu").contiguous()
                    elif isinstance(v, dict):
                        sub_dict: dict[str, torch.Tensor] = {}
                        for sk, sv in v.items():
                            if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                if fish_cuda_ipc_stage0 and str(sk) == fish_code_output_key:
                                    sub_dict[str(sk)] = sv.detach().contiguous()
                                else:
                                    sub_dict[str(sk)] = sv.detach().to("cpu").contiguous()
                        if sub_dict:
                            mm_cpu[k] = sub_dict
                    elif isinstance(v, list):
                        if len(v) == 0:
                            continue
                        cpu_list = []
                        for elem in v:
                            if isinstance(elem, torch.Tensor):
                                cpu_list.append(elem.detach().to("cpu").contiguous())
                            else:
                                cpu_list.append(elem)
                        mm_cpu[k] = cpu_list
                except Exception as e:
                    logger.error(f"Error in merge multimodal outputs: {e}")

        pooler_output: list[dict[str, object]] = []
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy[rid]
            start = int(self.query_start_loc.cpu[idx])
            sched = int(num_scheduled_tokens_np[idx])
            end = start + sched
            hidden_slice = hidden_states_cpu[start:end]
            payload: dict[str, object] = {"hidden": hidden_slice}
            if mm_cpu:
                mm_payload: dict[str, object] = {}
                for k, v in mm_cpu.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                        mm_payload[k] = v[start:end].contiguous()
                    elif isinstance(v, dict):
                        mm_payload[k] = {sk: sv[start:end].contiguous() for sk, sv in v.items()}
                    elif isinstance(v, list):
                        element = v[idx] if idx < len(v) else v[0]
                        # Clone tensors to avoid cross-request aliasing
                        if isinstance(element, torch.Tensor):
                            element = element.clone()
                        mm_payload[k] = element
                    elif isinstance(v, torch.Tensor):
                        # List-derived tensor payloads are request-invariant; clone to
                        # avoid accidental cross-request aliasing on downstream mutation.
                        mm_payload[k] = v.clone()
                    else:
                        mm_payload[k] = v
                payload.update(mm_payload)
            if rid in post_sample_codes_cpu:
                payload[getattr(self.model, "talker_mtp_output_key", "audio_codes")] = post_sample_codes_cpu[rid]
            if rid in inline_dac_flush_cpu:
                flush_wav = inline_dac_flush_cpu[rid]
                prev_wav = payload.get("model_outputs")
                if isinstance(prev_wav, torch.Tensor) and prev_wav.numel() > 0:
                    payload["model_outputs"] = torch.cat(
                        [prev_wav.reshape(-1), flush_wav.reshape(-1)]
                    ).contiguous()
                else:
                    payload["model_outputs"] = flush_wav.reshape(-1).contiguous()
                if "sr" not in payload:
                    payload["sr"] = torch.tensor(
                        int(getattr(self.model, "_dac_sample_rate", 44100)),
                        dtype=torch.int32,
                    )
            pooler_output.append(payload)
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.routed_experts_initialized:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")
            output = OmniModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )
            output.kv_extracted_req_ids = kv_extracted_req_ids

        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _flush_inline_dac_on_finished_samples(
        self,
        req_ids_output_copy: list[str],
        req_id_to_index_output_copy: dict[str, int],
        valid_sampled_token_ids: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        flush = getattr(self.model, "flush_inline_dac", None)
        if not callable(flush):
            return {}

        flushed: dict[str, torch.Tensor] = {}
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy.get(rid)
            if idx is None or idx >= len(valid_sampled_token_ids):
                continue
            sampled_ids = valid_sampled_token_ids[idx]
            if not self._fish_sample_contains_stop(rid, sampled_ids):
                continue
            req_info = self.model_intermediate_buffer.get(rid)
            if not isinstance(req_info, dict):
                continue
            try:
                delta = flush(req_info)
            except Exception as exc:
                logger.error("Inline DAC finish flush failed for request %s: %s", rid, exc)
                continue
            if isinstance(delta, torch.Tensor) and delta.numel() > 0:
                flushed[rid] = delta.detach().to("cpu").contiguous()
        return flushed

    def _fish_sample_contains_stop(self, req_id: str, sampled_ids: list[int]) -> bool:
        if not sampled_ids:
            return False

        stop_ids: set[int] = set()
        req_state = self.requests.get(req_id)
        sampling_params = getattr(req_state, "sampling_params", None)
        if sampling_params is not None:
            for attr in ("stop_token_ids", "all_stop_token_ids"):
                ids = getattr(sampling_params, attr, None)
                if ids is None:
                    continue
                try:
                    stop_ids.update(int(tok) for tok in ids)
                except TypeError:
                    stop_ids.add(int(ids))

        if any(int(tok) in stop_ids for tok in sampled_ids):
            return True

        sem_begin = getattr(self.model, "_semantic_begin_id", None)
        sem_end = getattr(self.model, "_semantic_end_id", None)
        if sem_begin is None or sem_end is None:
            return False
        return any(int(tok) < int(sem_begin) or int(tok) > int(sem_end) for tok in sampled_ids)

    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
