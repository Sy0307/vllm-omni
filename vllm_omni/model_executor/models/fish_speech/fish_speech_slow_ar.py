"""Fish Speech S2 Pro -- Slow AR model (Stage 0).

Uses vLLM's ``Qwen3Model`` as the transformer backbone.  Adds:
  - Multi-codebook input embedding (text + summed codebook embeddings at
    semantic-token positions).
  - Semantic logit masking.
  - Nested Fast AR for residual codebook prediction (``talker_mtp``).
  - ``preprocess`` / ``postprocess`` hooks for vLLM-omni's AR scheduler.

Analogous to ``Qwen3TTSTalkerForConditionalGeneration`` in qwen3_tts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.voice_cache import VoiceEmbeddingCache

from .configuration_fish_speech import FishSpeechConfig, FishSpeechFastARConfig, FishSpeechSlowARConfig
from .dac_encoder import _load_dac_codec, encode_reference_audio_codes
from .fish_speech_fast_ar import FishSpeechFastAR
from .prompt_utils import build_fish_voice_clone_prompt_ids

logger = init_logger(__name__)


def _remap_fish_speech_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    n_head: int,
    n_local_heads: int,
    head_dim: int,
    fast_n_head: int,
    fast_n_local_heads: int,
    fast_head_dim: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Transform Fish Speech HF weight names/values to Qwen3-compatible format.

    Key transformations:
      - ``wqkv`` → split into ``q_proj``, ``k_proj``, ``v_proj``
      - ``wo`` → ``o_proj``
      - ``w1`` → ``gate_proj``, ``w3`` → ``up_proj``, ``w2`` → ``down_proj``
      - ``attention_norm`` → ``input_layernorm``
      - ``ffn_norm`` → ``post_attention_layernorm``
      - ``text_model.model.embeddings`` → ``model.embed_tokens``
      - ``audio_decoder.*`` → ``fast_ar.*`` with similar transforms
    """
    q_size_text = n_head * head_dim
    kv_size_text = n_local_heads * head_dim
    q_size_fast = fast_n_head * fast_head_dim
    kv_size_fast = fast_n_local_heads * fast_head_dim

    for name, tensor in weights:
        # --- Text model (Slow AR) ---
        if name.startswith("text_model.model."):
            suffix = name[len("text_model.model.") :]

            # Embeddings
            if suffix == "embeddings.weight":
                yield "model.embed_tokens.weight", tensor
                continue

            # Norm
            if suffix == "norm.weight":
                yield "model.norm.weight", tensor
                continue

            # Layer weights
            if suffix.startswith("layers."):
                # layers.{N}.attention.wqkv.weight → split into q/k/v
                if ".attention.wqkv.weight" in suffix:
                    layer_prefix = suffix.split(".attention.wqkv.weight")[0]
                    q = tensor[:q_size_text, :]
                    k = tensor[q_size_text : q_size_text + kv_size_text, :]
                    v = tensor[q_size_text + kv_size_text :, :]
                    yield f"model.{layer_prefix}.self_attn.q_proj.weight", q
                    yield f"model.{layer_prefix}.self_attn.k_proj.weight", k
                    yield f"model.{layer_prefix}.self_attn.v_proj.weight", v
                    continue

                new_suffix = suffix
                new_suffix = new_suffix.replace(".attention.wo.", ".self_attn.o_proj.")
                new_suffix = new_suffix.replace(".attention.q_norm.", ".self_attn.q_norm.")
                new_suffix = new_suffix.replace(".attention.k_norm.", ".self_attn.k_norm.")
                new_suffix = new_suffix.replace(".attention_norm.", ".input_layernorm.")
                new_suffix = new_suffix.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w3.", ".mlp.up_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_suffix = new_suffix.replace(".ffn_norm.", ".post_attention_layernorm.")
                yield f"model.{new_suffix}", tensor
                continue

            # Fallback for any other text_model.model.* weights
            yield f"model.{suffix}", tensor
            continue

        # --- Audio decoder (Fast AR) ---
        if name.startswith("audio_decoder."):
            suffix = name[len("audio_decoder.") :]

            # Codebook embeddings (belongs to the main model, not Fast AR).
            if suffix == "codebook_embeddings.weight":
                yield "codebook_embeddings.weight", tensor
                continue

            # Fast AR embeddings, output, norm.
            if suffix == "embeddings.weight":
                yield "fast_ar.fast_embeddings.weight", tensor
                continue
            if suffix == "output.weight":
                yield "fast_ar.fast_output.weight", tensor
                continue
            if suffix == "norm.weight":
                yield "fast_ar.fast_norm.weight", tensor
                continue

            # Fast AR projection in.
            if suffix.startswith("fast_project_in."):
                yield f"fast_ar.fast_project_in.{suffix[len('fast_project_in.') :]}", tensor
                continue

            # Fast AR layer weights.
            if suffix.startswith("layers."):
                if ".attention.wqkv.weight" in suffix:
                    layer_prefix = suffix.split(".attention.wqkv.weight")[0]
                    q = tensor[:q_size_fast, :]
                    k = tensor[q_size_fast : q_size_fast + kv_size_fast, :]
                    v = tensor[q_size_fast + kv_size_fast :, :]
                    yield f"fast_ar.model.{layer_prefix}.self_attn.q_proj.weight", q
                    yield f"fast_ar.model.{layer_prefix}.self_attn.k_proj.weight", k
                    yield f"fast_ar.model.{layer_prefix}.self_attn.v_proj.weight", v
                    continue

                new_suffix = suffix
                new_suffix = new_suffix.replace(".attention.wo.", ".self_attn.o_proj.")
                new_suffix = new_suffix.replace(".attention.q_norm.", ".self_attn.q_norm.")
                new_suffix = new_suffix.replace(".attention.k_norm.", ".self_attn.k_norm.")
                new_suffix = new_suffix.replace(".attention_norm.", ".input_layernorm.")
                new_suffix = new_suffix.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w3.", ".mlp.up_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_suffix = new_suffix.replace(".ffn_norm.", ".post_attention_layernorm.")
                yield f"fast_ar.model.{new_suffix}", tensor
                continue

            yield f"fast_ar.{suffix}", tensor
            continue

        # Pass through any other weights.
        yield name, tensor


class FishSpeechSlowARForConditionalGeneration(nn.Module):
    """vLLM-AR Slow AR model for Fish Speech S2 Pro.

    Stage 0: text → semantic tokens (+ residual codebook codes via Fast AR).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config: FishSpeechConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.text_config: FishSpeechSlowARConfig = config.text_config
        self.fast_ar_config: FishSpeechFastARConfig = config.audio_decoder_config

        self._semantic_begin_id = int(config.semantic_start_token_id)
        self._semantic_end_id = int(config.semantic_end_token_id)
        self._audio_pad_token_id = int(config.audio_pad_token_id)
        self._codebook_size = int(self.text_config.codebook_size)
        self._num_codebooks = int(self.text_config.num_codebooks)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(self.text_config.hidden_size)
        self.talker_mtp_output_key = "audio_codes"
        self.talker_mtp_graph_safe = True
        self.gpu_resident_buffer_keys: set[str] = {
            "last_slow_ar_hidden",
            "_merged_prev_codes",
            "_post_sample_prev_codes",
        }

        # Merged decode: codebook inject + transformer + logits + sampling
        # + Fast AR all inside forward(), captured in a single CUDA graph.
        # Eliminates talker_mtp graph launch and Python dispatch overhead.
        self._merged_decode = False
        self._unified_decode = False  # subset: inline logits+sampling+FastAR
        self._unified_decode_batch_size = 0  # set by runner (used outside graph)
        self._output_codes: torch.Tensor | None = None
        self._output_sampled_token: torch.Tensor | None = None
        self._unified_logits: torch.Tensor | None = None
        self._decode_rows: torch.Tensor | None = None
        self._codes_valid = False
        self.prefer_model_sampler = False
        # Persistent VQ buffers for merged decode (allocated by setup_merged_decode).
        self._vq_codes: torch.Tensor | None = None
        self._vq_mask: torch.Tensor | None = None
        self._codebook_offsets: torch.Tensor | None = None
        self._vq_scale: float = 1.0

        # Single-stage DAC inline: enabled when engine_output_type is "audio".
        self._inline_dac = False
        self._dac_codec: torch.nn.Module | None = None
        self._dac_sample_rate: int = 44100
        self._dac_num_codebooks: int = 10
        self._vocode_stride: int = 10
        self._initial_vocode_stride: int = 4
        self._vocode_left_context: int = 25
        self._use_async_vocoder = os.environ.get(
            "VLLM_OMNI_FISH_ASYNC_VOCODER",
            os.environ.get("VLLM_FISH_ASYNC_INLINE_DAC", "0"),
        ) == "1"
        self._vocoder_device: torch.device | None = None
        self._vocoder_stream: torch.cuda.Stream | None = None
        self._post_sample_codebook = os.environ.get("VLLM_FISH_POST_SAMPLE_CODEBOOK", "0") == "1"
        if self._post_sample_codebook:
            logger.info("Fish Speech post-sample FastAR code relay enabled")

        # Qwen3 transformer backbone.
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Fish Speech uses interleaved (GPT-J) RoPE, not NeoX style.
        # vLLM's Qwen3Attention defaults to NeoX (is_neox_style=True).
        # Replace with interleaved RoPE to match training.
        self._fix_rope_style()

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Multi-codebook embedding table: codebook_size * num_codebooks entries.
        self.codebook_embeddings = nn.Embedding(
            self._codebook_size * self._num_codebooks,
            self.text_config.hidden_size,
        )

        # Fast AR (residual codebook predictor).
        predictor_compilation = dataclasses.replace(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        self._fast_ar_vllm_config = dataclasses.replace(vllm_config, compilation_config=predictor_compilation)
        from vllm.config.vllm import set_current_vllm_config as _set_cfg

        with _set_cfg(self._fast_ar_vllm_config):
            self.fast_ar = FishSpeechFastAR(
                vllm_config=self._fast_ar_vllm_config,
                config=self.fast_ar_config,
                slow_ar_config=self.text_config,
                prefix="fast_ar",
            )
        if self.talker_mtp_graph_safe:
            self.fast_ar._disable_compile_for_graph = True

        # Constant logit mask: allow only semantic tokens + im_end.
        vocab = int(self.text_config.vocab_size)
        semantic_mask = torch.zeros((vocab,), dtype=torch.bool)
        lo = self._semantic_begin_id
        hi = min(self._semantic_end_id + 1, vocab)
        if hi > lo:
            semantic_mask[lo:hi] = True
        # Also allow <|im_end|> (token 151645 in Qwen3 tokeniser).
        im_end_id = 151645
        if im_end_id < vocab:
            semantic_mask[im_end_id] = True
        self.register_buffer("_semantic_allowed_mask", semantic_mask, persistent=False)

        # In-memory LRU cache for DAC-encoded reference audio codes.
        self._voice_cache = VoiceEmbeddingCache()

        # Tokeniser (lazy).
        self._tokenizer = None

    def _fix_rope_style(self) -> None:
        """Replace NeoX-style RoPE with interleaved (GPT-J) style.

        Fish Speech was trained with interleaved RoPE (complex-number pairs),
        but vLLM's Qwen3Attention defaults to NeoX style.  We rebuild the
        rotary embedding with ``is_neox_style=False`` for each attention layer.
        """
        from vllm.model_executor.layers.rotary_embedding import get_rope

        for layer in self.model.layers:
            attn = layer.self_attn
            # Extract parameters from the existing RoPE to rebuild it.
            head_dim = attn.head_dim
            max_position = self.text_config.max_position_embeddings
            rope_params = getattr(self.text_config, "rope_scaling", None) or {}
            rope_params.setdefault("rope_theta", getattr(self.text_config, "rope_theta", 1000000.0))
            attn.rotary_emb = get_rope(
                head_size=head_dim,
                max_position=max_position,
                is_neox_style=False,
                rope_parameters=rope_params,
            )
        logger.info("Fixed RoPE style to interleaved (GPT-J) for %d layers", len(self.model.layers))

    # -------------------- Merged decode setup --------------------

    def setup_unified_decode(self, max_batch_size: int, device: torch.device) -> None:
        """Allocate persistent GPU buffers for inline logits/sample/Fast AR."""
        self._unified_decode = True
        self.prefer_model_sampler = True

        # Unified decode buffers (sampling + Fast AR output).
        self._output_sampled_token = torch.zeros(
            max_batch_size, dtype=torch.long, device=device,
        )
        self._output_codes = torch.zeros(
            max_batch_size, self._num_codebooks, dtype=torch.long, device=device,
        )
        self._unified_logits = torch.zeros(
            max_batch_size, self.text_config.vocab_size, device=device, dtype=torch.float32,
        )
        self._decode_rows = torch.full(
            (max_batch_size,), -1, dtype=torch.long, device=device,
        )
        self._sampling_temperature = torch.full(
            (max_batch_size,), 0.8, device=device,
        )
        self._sampling_top_k = torch.full(
            (max_batch_size,), 30, dtype=torch.long, device=device,
        )
        self._sampling_top_p = torch.full(
            (max_batch_size,), 0.9, device=device,
        )

        logger.info(
            "Unified decode enabled: max_batch_size=%d, device=%s",
            max_batch_size, device,
        )

    def setup_merged_decode(self, max_batch_size: int, device: torch.device) -> None:
        """Allocate persistent GPU buffers for merged decode mode.

        Merged decode combines codebook injection + transformer + logits +
        sampling + Fast AR inside the decode step. Per-request codes are kept
        in runner state and only copied into these row buffers for the current
        scheduled batch.
        """
        if not self._unified_decode:
            self.setup_unified_decode(max_batch_size, device)
        self._merged_decode = True

        # VQ persistent buffers: codebook codes from previous step.
        # forward() reads these to inject codebook embeddings, then
        # writes new codes after Fast AR.
        self._vq_codes = torch.zeros(
            max_batch_size, self._num_codebooks, dtype=torch.long, device=device,
        )
        self._vq_mask = torch.zeros(
            max_batch_size, dtype=torch.bool, device=device,
        )
        self._codebook_offsets = (
            torch.arange(self._num_codebooks, device=device, dtype=torch.long)
            * self._codebook_size
        )
        self._vq_scale = 1.0 / math.sqrt(self._num_codebooks + 1)

        logger.info(
            "Merged decode enabled: max_batch_size=%d, device=%s",
            max_batch_size, device,
        )

    # -------------------- vLLM required hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # Merged decode: inject codebook embeddings from persistent VQ buffer.
        # This replaces the separate talker_mtp step.
        if (
            self._merged_decode
            and inputs_embeds is not None
            and self._vq_codes is not None
            and self._decode_rows is not None
        ):
            max_bs = self._vq_codes.shape[0]
            bs = min(self._unified_decode_batch_size, max_bs)
            if bs > 0:
                rows = self._decode_rows[:bs].clamp(min=0, max=max(inputs_embeds.shape[0] - 1, 0))
                valid = self._vq_mask[:bs] & (self._decode_rows[:bs] >= 0)
                selected_inputs = inputs_embeds.index_select(0, rows)
                vq_codes = self._vq_codes[:bs]
                offset_parts = vq_codes + self._codebook_offsets[None, :]
                cb_embeds = self.codebook_embeddings(offset_parts)
                cb_sum = cb_embeds.sum(dim=1).to(inputs_embeds.dtype)
                combined = (selected_inputs + cb_sum) * self._vq_scale
                updated = torch.where(
                    valid.unsqueeze(-1), combined, selected_inputs,
                )
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds.index_copy_(0, rows, updated)

        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

        # Merged/unified decode: inline logits + sampling + Fast AR.
        if (
            self._unified_decode
            and isinstance(hidden_states, torch.Tensor)
            and self._unified_decode_batch_size > 0
        ):
            self._inline_decode_codebooks(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        # In unified decode mode, logits were already computed inside forward().
        # Return the pre-computed masked logits for decode-only batches.
        # Mixed prefill+decode batches fall through to the standard path.
        if self._unified_decode and self._unified_logits is not None:
            bs = self._unified_decode_batch_size
            if bs > 0 and hidden_states.shape[0] == bs:
                return self._unified_logits[:bs]

        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None

        # Mask to semantic tokens + im_end only.
        logits = logits.masked_fill(~self._semantic_allowed_mask, float("-inf"))
        return logits

    # -------------------- Unified decode: inline sampling + Fast AR ----------

    @torch.no_grad()
    def _inline_decode_codebooks(self, hidden_states: torch.Tensor) -> None:
        """Constrained semantic sampling + Fast AR codebook loop, inline in forward().

        This replaces the separate talker_mtp graph launch with in-graph computation.
        Modeled after sglang's _decode_codebooks().

        Always processes max_batch_size entries (the persistent buffer size)
        to keep CUDA graph shapes fixed.  Extra entries for padded positions
        produce garbage that is never consumed.

        Writes to persistent buffers:
          _output_sampled_token: [max_bs] semantic token IDs (vocab space)
          _output_codes:         [max_bs, num_codebooks] codebook codes
          _unified_logits:       [max_bs, vocab] masked logits for compute_logits
        """
        # Slice to the scheduled decode rows. This avoids doing Fast AR on
        # prefill rows in mixed batches and keeps request state row-stable.
        max_bs = self._output_codes.shape[0]
        bs = min(self._unified_decode_batch_size, max_bs)
        if bs <= 0:
            return
        if self._decode_rows is not None:
            rows = self._decode_rows[:bs].clamp(min=0, max=max(hidden_states.shape[0] - 1, 0))
            h = hidden_states.index_select(0, rows)
        else:
            h = hidden_states[:bs]

        # 1. Compute logits + semantic mask
        logits = self.logits_processor(self.lm_head, h)
        logits = logits.masked_fill(~self._semantic_allowed_mask, float("-inf"))

        # Store for compute_logits to return.
        self._unified_logits[:bs] = logits

        # 2. Constrained sampling: top-k → temperature scaling → top-p → multinomial
        # All operations are CUDA graph safe.
        biased_logits = logits.to(torch.float32)

        # Top-k filtering.
        _TOP_K = 30  # Fixed for graph safety (matches default config).
        top_k_logits, top_k_indices = torch.topk(biased_logits, _TOP_K, dim=-1)

        # Temperature scaling.
        temperature = self._sampling_temperature[:bs].unsqueeze(1).clamp(min=1e-5)
        scaled_logits = top_k_logits / temperature

        # Top-p filtering.
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        top_p = self._sampling_top_p[:bs].unsqueeze(1)
        top_p_mask = cum_probs > top_p
        top_p_mask[..., 0] = False  # Always keep at least one token.
        sorted_logits = sorted_logits.masked_fill(top_p_mask, float("-inf"))
        # Unsort back to top-k order.
        scaled_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(scaled_logits, dim=-1)
        # multinomial is CUDA graph safe in PyTorch 2.x+.
        local_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        semantic_token = top_k_indices.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)

        self._output_sampled_token[:bs] = semantic_token

        # 3. Fast AR codebook loop (re-prefill, all within the same graph).
        codes = self.fast_ar(
            slow_ar_hidden=h.reshape(bs, -1),
            semantic_token_id=semantic_token,
            do_sample=False,  # argmax for CUDA graph safety (matches sglang).
        )
        self._output_codes[:bs] = codes
        self._codes_valid = True

        # NOTE: Do NOT update _vq_codes here.  This method runs inside CUDA
        # graph for both prefill and decode batches.  During prefill, the
        # hidden states fed here are the first max_bs tokens of the prompt
        # (NOT the last token), so the codes would be wrong.
        # Instead, the runner copies _output_codes → _vq_codes in _preprocess
        # before the next decode step, where it knows the correct decode index.

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> Any:
        """Return pre-computed sampled token from unified decode.

        Called by vLLM's _sample() when prefer_model_sampler=True.
        Returns a SamplerOutput with the semantic tokens already sampled
        inside forward()'s _inline_decode_codebooks().

        Only handles decode-only batches (num_reqs == decode_batch_size).
        Mixed prefill+decode batches return None → standard sampler.
        """
        if not self._unified_decode or self._output_sampled_token is None:
            return None
        bs = self._unified_decode_batch_size
        if bs <= 0 or logits is None or logits.numel() == 0:
            return None

        num_reqs = logits.shape[0]
        # Only return pre-computed tokens for decode-only batches.
        if num_reqs != bs:
            return None

        from vllm.v1.outputs import SamplerOutput

        sampled = self._output_sampled_token[:bs].unsqueeze(-1).to(torch.int32)
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    # -------------------- Omni multimodal output plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            parent_output = model_outputs
        else:
            hidden = model_outputs
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information") or []

            audio_codes_list: list[torch.Tensor] = []
            for info in info_dicts:
                if not isinstance(info, dict):
                    continue
                ac = info.get("audio_codes")
                if isinstance(ac, torch.Tensor):
                    audio_codes_list.append(ac)

            if not audio_codes_list:
                logger.debug("make_omni_output: no audio_codes found in info_dicts (len=%d)", len(info_dicts))
                if self._inline_dac:
                    return OmniOutput(
                        text_hidden_states=hidden,
                        multimodal_outputs={
                            "model_outputs": [torch.zeros((0,), dtype=torch.float32)],
                            "sr": [torch.tensor(self._dac_sample_rate, dtype=torch.int32)],
                        },
                    )
                return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

            audio_codes = torch.cat(audio_codes_list, dim=0)
            span_len = int(audio_codes.shape[0])
            hidden = hidden[:span_len]
            mm: dict[str, torch.Tensor] = {"audio_codes": audio_codes}
            parent_output = OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

        if not self._inline_dac:
            return parent_output

        # ---- Single-stage inline DAC decode ----
        return self._inline_dac_output(parent_output, **kwargs)

    # -------------------- Single-stage inline DAC --------------------

    def _ensure_dac_loaded(self) -> None:
        """Lazily load DAC codec for single-stage inline decode."""
        if self._dac_codec is not None:
            return

        from .dac_utils import DAC_NUM_CODEBOOKS, DAC_SAMPLE_RATE, build_dac_codec
        from ..fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder

        codec_path = os.path.join(self.model_path, "codec.pth")
        if not os.path.exists(codec_path):
            try:
                from transformers.utils.hub import cached_file

                cached = cached_file(self.model_path, "codec.pth")
                if cached is not None:
                    codec_path = cached
            except Exception:
                pass
        if not os.path.exists(codec_path):
            raise FileNotFoundError(f"codec.pth not found at {codec_path}.")

        codec = build_dac_codec()
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)

        # Bake weight norm for inference.
        from torch.nn.utils.parametrize import remove_parametrizations

        for module in codec.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if not parametrizations:
                continue
            for name in list(parametrizations.keys()):
                remove_parametrizations(module, name, leave_parametrized=True)

        # Prune encode-only components.
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

        # Fix numpy scalars for torch.compile.
        if hasattr(codec, "hop_length"):
            codec.hop_length = int(codec.hop_length)
        if hasattr(codec, "frame_length"):
            codec.frame_length = int(codec.frame_length)

        ar_device = self.vllm_config.device_config.device
        vocoder_device = ar_device
        vocoder_device_str = os.environ.get("VLLM_OMNI_FISH_VOCODER_DEVICE")
        if vocoder_device_str:
            try:
                vocoder_device = torch.device(vocoder_device_str)
            except Exception:
                logger.warning(
                    "Invalid VLLM_OMNI_FISH_VOCODER_DEVICE=%s; using AR device",
                    vocoder_device_str,
                )
                vocoder_device = ar_device

        codec = codec.to(device=vocoder_device, dtype=torch.float32)
        codec.eval()

        if vocoder_device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        FishSpeechDACDecoder._cache_attention_masks(codec, vocoder_device)

        if os.environ.get("VLLM_FISH_DAC_COMPILE", "0") == "1":
            try:
                import torch._dynamo as _dynamo

                _dynamo.config.recompile_limit = max(256, _dynamo.config.recompile_limit)
                codec.decode = torch.compile(
                    codec.decode, mode="default", dynamic=True, fullgraph=False
                )
                logger.info("Single-stage: torch.compile enabled for DAC codec.decode")
            except Exception as exc:
                logger.warning("Single-stage: torch.compile on DAC failed: %s", exc)
        else:
            logger.info("Single-stage: DAC torch.compile disabled (set VLLM_FISH_DAC_COMPILE=1 to enable)")

        self._dac_codec = codec
        self._vocoder_device = vocoder_device
        if self._use_async_vocoder and vocoder_device.type == "cuda":
            self._vocoder_stream = torch.cuda.Stream(device=vocoder_device)
        self._dac_sample_rate = DAC_SAMPLE_RATE
        self._dac_num_codebooks = DAC_NUM_CODEBOOKS
        logger.info(
            "Single-stage DAC codec loaded (vocoder_device=%s, ar_device=%s, async=%s)",
            vocoder_device,
            ar_device,
            self._vocoder_stream is not None,
        )

    @torch.no_grad()
    def _decode_all(self, codes_fq: torch.Tensor) -> torch.Tensor:
        """Decode [N, Q] codes → waveform tensor."""
        self._ensure_dac_loaded()
        assert self._dac_codec is not None
        codec_device = next(self._dac_codec.parameters()).device
        codes_qf = codes_fq.to(device=codec_device).transpose(0, 1).long()
        total_frames = codes_qf.shape[1]
        feature_lengths = torch.tensor([total_frames], device=codec_device, dtype=torch.long)
        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(codes_qf.unsqueeze(0), feature_lengths)
        audio_len = int(audio_lengths[0].item()) if audio_lengths.numel() > 0 else int(wav_batch.shape[-1])
        return wav_batch[0, 0, :audio_len].to(dtype=torch.float32).reshape(-1)

    @torch.no_grad()
    def _submit_decode_async(
        self,
        codes_fq: torch.Tensor,
    ) -> tuple[torch.cuda.Event | None, tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor]:
        """Submit a full accumulated-code DAC decode on the vocoder stream."""
        self._ensure_dac_loaded()
        assert self._dac_codec is not None
        if self._vocoder_stream is None:
            return None, self._decode_all(codes_fq)

        vocoder_device = self._vocoder_device
        assert vocoder_device is not None
        with torch.cuda.stream(self._vocoder_stream):
            codes_qf = codes_fq.to(device=vocoder_device, non_blocking=True).transpose(0, 1).long()
            total_frames = codes_qf.shape[1]
            feature_lengths = torch.tensor([total_frames], device=vocoder_device, dtype=torch.long)
            with torch.amp.autocast("cuda", enabled=False):
                wav_batch, audio_lengths = self._dac_codec.decode(codes_qf.unsqueeze(0), feature_lengths)
            audio_len_t = audio_lengths[0] if audio_lengths.numel() > 0 else None
            done_event = self._vocoder_stream.record_event()
        return done_event, (wav_batch[0, 0], audio_len_t)

    def _collect_pending_inline_dac(
        self,
        req_info: dict[str, Any],
        *,
        block: bool = False,
    ) -> torch.Tensor | None:
        pending = req_info.get("_pending_decode")
        if pending is None:
            return None

        done_event, wav_info, frames_at_submit = pending
        t_sync = time.perf_counter()
        if done_event is not None:
            if not block:
                query = getattr(done_event, "query", None)
                if callable(query) and not query():
                    return None
            done_event.synchronize()
        sync_ms = (time.perf_counter() - t_sync) * 1000
        if sync_ms > 2.0:
            logger.info("Inline DAC async collect waited %.2fms (frames=%d)", sync_ms, frames_at_submit)

        wav_gpu, audio_len_t = wav_info
        audio_len = int(audio_len_t.item()) if audio_len_t is not None else int(wav_gpu.shape[-1])
        full_wav = wav_gpu[:audio_len].to(dtype=torch.float32, device="cpu")
        emitted_samples = int(req_info.get("_emitted_samples", 0) or 0)
        delta = full_wav[emitted_samples:].contiguous()
        req_info["_emitted_samples"] = int(full_wav.numel())
        req_info["_last_vocoded_at"] = int(frames_at_submit)
        req_info["_pending_decode"] = None
        return delta

    def flush_inline_dac(self, req_info: dict[str, Any]) -> torch.Tensor:
        """Decode any pending inline-DAC frames for a request at finish."""
        pending_delta = self._collect_pending_inline_dac(req_info, block=True)
        if pending_delta is not None and pending_delta.numel() > 0:
            return pending_delta
        return self._decode_inline_dac_delta(req_info, force=True, request_label="finish")

    def _decode_inline_dac_delta(
        self,
        req_info: dict[str, Any],
        *,
        force: bool,
        request_label: object,
    ) -> torch.Tensor:
        empty_wav = torch.zeros((0,), dtype=torch.float32)
        codes_list = req_info.get("_all_codes")
        if not codes_list:
            return empty_wav

        total_frames = sum(c.shape[0] for c in codes_list)
        last_vocoded_at = int(req_info.get("_last_vocoded_at", 0) or 0)
        new_since_vocode = total_frames - last_vocoded_at
        stride = self._initial_vocode_stride if last_vocoded_at == 0 else self._vocode_stride

        if new_since_vocode <= 0 or (not force and new_since_vocode < stride):
            return empty_wav

        # Decode only the unseen tail plus a small left context. The context
        # keeps DAC boundaries stable without replaying the full utterance.
        all_codes = torch.cat(codes_list, dim=0)
        codes_list.clear()
        codes_list.append(all_codes)
        decode_start = 0
        if last_vocoded_at > 0:
            decode_start = max(0, last_vocoded_at - self._vocode_left_context)
        left_context_frames = last_vocoded_at - decode_start
        decode_codes = all_codes[decode_start:]

        try:
            full_wav = self._decode_all(decode_codes)
        except Exception as exc:
            logger.error("Inline DAC decode failed for req %s: %s", request_label, exc)
            return empty_wav

        full_wav = full_wav.cpu()
        if left_context_frames > 0 and decode_codes.shape[0] > 0:
            trim_samples = int(round(full_wav.numel() * left_context_frames / int(decode_codes.shape[0])))
            full_wav = full_wav[min(trim_samples, full_wav.numel()):]

        req_info["_last_vocoded_at"] = total_frames
        delta = full_wav.contiguous()
        return delta if delta.numel() > 0 else empty_wav

    def _inline_dac_output(self, parent_output: OmniOutput, **kwargs: Any) -> OmniOutput:
        """Accumulate codes and decode DAC inline at stride boundaries."""
        sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
        empty_wav = torch.zeros((0,), dtype=torch.float32)

        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        req_infos: list[dict[str, Any]] = [info for info in info_dicts if isinstance(info, dict)]
        batch_size = max(len(req_infos), 1)

        mm = parent_output.multimodal_outputs or {}
        all_codes_combined = mm.get("audio_codes")

        deltas: list[torch.Tensor] = []
        for i, req_info in enumerate(req_infos):
            delta_from_pending = None
            pending_active = False
            if self._use_async_vocoder:
                delta_from_pending = self._collect_pending_inline_dac(req_info)
                pending_active = req_info.get("_pending_decode") is not None

            # 1) Accumulate this step's codes.
            if isinstance(all_codes_combined, torch.Tensor) and i < all_codes_combined.shape[0]:
                latest_codes = all_codes_combined[i : i + 1]
                valid = latest_codes.any(dim=1)
                if valid.any():
                    codes_list = req_info.get("_all_codes")
                    if codes_list is None:
                        codes_list = []
                        req_info["_all_codes"] = codes_list
                    codes_list.append(latest_codes[valid].detach())

            codes_list = req_info.get("_all_codes")
            if not codes_list:
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            if pending_active:
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            if not self._use_async_vocoder:
                deltas.append(self._decode_inline_dac_delta(req_info, force=False, request_label=i))
                continue

            total_frames = sum(c.shape[0] for c in codes_list)
            last_vocoded_at = int(req_info.get("_last_vocoded_at", 0) or 0)
            new_since_vocode = total_frames - last_vocoded_at
            stride = self._initial_vocode_stride if last_vocoded_at == 0 else self._vocode_stride
            if new_since_vocode < stride:
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            all_codes = torch.cat(codes_list, dim=0)
            codes_list.clear()
            codes_list.append(all_codes)
            try:
                done_event, payload = self._submit_decode_async(all_codes)
            except Exception as exc:
                logger.error("Inline DAC async submit failed for req %s: %s", i, exc)
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            if done_event is None:
                full_wav = payload.cpu() if isinstance(payload, torch.Tensor) else None
                if full_wav is None:
                    deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                    continue
                emitted_samples = int(req_info.get("_emitted_samples", 0) or 0)
                delta = full_wav[emitted_samples:].contiguous()
                req_info["_emitted_samples"] = int(full_wav.numel())
                req_info["_last_vocoded_at"] = total_frames
                if delta.numel() > 0:
                    deltas.append(delta)
                elif delta_from_pending is not None:
                    deltas.append(delta_from_pending)
                else:
                    deltas.append(empty_wav)
                continue

            req_info["_pending_decode"] = (done_event, payload, total_frames)
            deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)

        while len(deltas) < batch_size:
            deltas.append(empty_wav)

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={
                "model_outputs": deltas,
                "sr": [sr_tensor] * batch_size,
            },
        )

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        if span_len > 1:
            # --- Prefill ---
            prompt_embeds_buf = info_dict.get("slow_ar_prompt_embeds")
            is_first_prefill = not isinstance(prompt_embeds_buf, torch.Tensor) or prompt_embeds_buf.ndim != 2
            dev = input_ids.device

            if is_first_prefill:
                if bool(info_dict.get("fish_structured_voice_clone", False)):
                    prompt_embeds = self._build_structured_voice_clone_prefill_embeds(info_dict)
                else:
                    prompt_embeds = self._build_prefill_embeds(input_ids, info_dict)
                prompt_embeds_buf = prompt_embeds.detach().to("cpu").contiguous()
                if not prompt_embeds_buf.is_pinned():
                    prompt_embeds_buf = prompt_embeds_buf.pin_memory()
                total_prompt_len = int(prompt_embeds_buf.shape[0])
                next_offset = min(span_len, total_prompt_len)

                info_update: dict[str, Any] = {
                    "slow_ar_prompt_embeds": prompt_embeds_buf if next_offset < total_prompt_len else None,
                    "prefill_offset": next_offset,
                }

                take = prompt_embeds_buf[:span_len]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_embed = self.embed_input_ids(
                        torch.tensor([self._audio_pad_token_id], device=dev, dtype=torch.long)
                    ).reshape(1, -1)
                    take = torch.cat([take, pad_embed.expand(pad_n, -1)], dim=0)
                prompt_embeds = take.to(device=dev, dtype=torch.bfloat16, non_blocking=True)

                zeros = torch.zeros(
                    (prompt_embeds.shape[0], self._num_codebooks),
                    device=dev,
                    dtype=torch.long,
                )
                info_update["audio_codes"] = zeros

                input_ids_out = input_ids.clone()
                input_ids_out[:] = self._audio_pad_token_id
                return input_ids_out, prompt_embeds, info_update

            else:
                # Subsequent prefill chunk.
                offset = int(info_dict.get("prefill_offset", 0) or 0)
                total_prompt_len = int(prompt_embeds_buf.shape[0])
                s = max(0, min(offset, total_prompt_len))
                e = max(0, min(offset + span_len, total_prompt_len))
                take = prompt_embeds_buf[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_embed = self.embed_input_ids(
                        torch.tensor([self._audio_pad_token_id], device=dev, dtype=torch.long)
                    ).reshape(1, -1)
                    take = torch.cat([take, pad_embed.expand(pad_n, -1)], dim=0)
                prompt_embeds = take.to(device=dev, dtype=torch.bfloat16, non_blocking=True)
                next_offset = offset + span_len

                zeros = torch.zeros((prompt_embeds.shape[0], self._num_codebooks), device=dev, dtype=torch.long)
                return (
                    input_ids.clone().fill_(self._audio_pad_token_id),
                    prompt_embeds,
                    {
                        "slow_ar_prompt_embeds": prompt_embeds_buf if next_offset < total_prompt_len else None,
                        "prefill_offset": next_offset,
                        "audio_codes": zeros,
                    },
                )

        # --- Decode: span_len == 1 ---
        dev = input_ids.device

        last_hidden = info_dict.get("last_slow_ar_hidden")
        if not isinstance(last_hidden, torch.Tensor):
            logger.warning(
                "preprocess decode: last_slow_ar_hidden not found (keys=%s), "
                "returning plain embed (mtp_inputs will NOT be set)",
                list(info_dict.keys()),
            )
            embeds = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).reshape(1, -1)
            return input_ids, embeds.to(dtype=torch.bfloat16), {}

        token_embed = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(
            device=dev, dtype=torch.bfloat16
        )  # [1, 1, H]

        inputs_embeds_out = token_embed.reshape(1, -1)

        if self._merged_decode:
            # Merged decode: codebook injection happens inside forward(),
            # not in talker_mtp.  Set VQ mask based on token type.
            is_semantic = (input_ids[0] >= self._semantic_begin_id) & (
                input_ids[0] <= self._semantic_end_id
            )
            # Find batch index for this request (runner calls preprocess per-req).
            # _vq_mask is indexed by decode request order (set by runner).
            # We just return the plain embed; forward() handles injection.
            return input_ids, inputs_embeds_out, {}

        # Non-merged: prepare mtp_inputs for talker_mtp.
        info_update = {
            "mtp_inputs": (
                last_hidden.to(device=dev, dtype=torch.bfloat16).reshape(1, -1),
                torch.zeros(1, self.text_config.hidden_size, device=dev, dtype=torch.bfloat16),
            ),
        }
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **info_dict: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            logger.debug("postprocess: empty hidden_states")
            return {}
        last = hidden_states[-1, :].detach().contiguous()
        logger.debug("postprocess: saved last_slow_ar_hidden shape=%s", tuple(last.shape))
        update: dict[str, Any] = {"last_slow_ar_hidden": last}
        if self._merged_decode and self._output_codes is not None:
            decode_idx = info_dict.get("_merged_decode_idx")
            if decode_idx is not None:
                try:
                    idx = int(decode_idx)
                    if 0 <= idx < int(self._output_codes.shape[0]):
                        update["_merged_prev_codes"] = self._output_codes[idx : idx + 1]
                except (TypeError, ValueError):
                    pass
        return update

    # -------------------- prompt construction --------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return self._tokenizer

    def _build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
    ) -> torch.Tensor:
        """Build prefill embeddings, adding codebook embeddings at semantic positions.

        For text-only prefill (no reference audio), this is just embed_input_ids.
        For voice cloning, reference codes are embedded with codebook offsets.
        """
        dev = input_ids.device
        # Basic text embeddings.
        base_embeds = self.embed_input_ids(input_ids.reshape(1, -1).to(torch.long))  # [1, T, H]

        # Check for reference codebook codes (for voice cloning).
        ref_codes = info_dict.get("ref_codes")
        if not isinstance(ref_codes, torch.Tensor) or ref_codes.numel() == 0:
            return base_embeds.squeeze(0).to(dtype=torch.bfloat16)

        # ref_codes: [T_ref, num_codebooks] -- codebook codes for reference audio positions.
        ref_codes = ref_codes.to(device=dev, dtype=torch.long)
        ref_positions = info_dict.get("ref_positions")
        if not isinstance(ref_positions, torch.Tensor):
            return base_embeds.squeeze(0).to(dtype=torch.bfloat16)

        ref_positions = ref_positions.to(device=dev, dtype=torch.long).reshape(-1)
        seq_len = int(input_ids.shape[0])
        codebook_sum = torch.zeros_like(base_embeds)  # [1, T, H]

        for pos_idx in range(int(ref_positions.shape[0])):
            pos = int(ref_positions[pos_idx].item())
            if pos < 0 or pos >= seq_len:
                continue
            for cb_idx in range(min(int(ref_codes.shape[1]), self._num_codebooks)):
                code = ref_codes[pos_idx, cb_idx].clamp(min=0)
                code_with_offset = code + cb_idx * self._codebook_size
                emb = self.codebook_embeddings(code_with_offset.unsqueeze(0))
                codebook_sum[0, pos, :] += emb.squeeze(0).to(dtype=base_embeds.dtype)

        result = base_embeds + codebook_sum
        return result.squeeze(0).to(dtype=torch.bfloat16)

    def _build_structured_voice_clone_prefill_embeds(self, info_dict: dict[str, Any]) -> torch.Tensor:
        tokenizer = self._get_tokenizer()
        ref_text = info_dict.get("ref_text")
        text = info_dict.get("text")
        ref_audio_sr = info_dict.get("ref_audio_sr")
        if not isinstance(ref_text, str) or not isinstance(text, str):
            raise ValueError("Fish Speech structured voice clone requires string text and ref_text")

        # --- Voice cache: reuse DAC codes for uploaded (named) voices ---
        _voice_cache_key: str | None = None
        voice_name = info_dict.get("voice_name")
        voice_created_at = info_dict.get("voice_created_at")
        if isinstance(voice_name, str) and voice_name:
            _created_at = float(voice_created_at) if voice_created_at is not None else 0.0
            if _created_at <= 0:
                logger.warning(
                    "Voice '%s' has no created_at timestamp; DAC code caching disabled for this request",
                    voice_name,
                )
            else:
                _voice_cache_key = self._voice_cache.make_cache_key(
                    voice_name,
                    xvec_only=False,
                    created_at=_created_at,
                )
                _cached = self._voice_cache.get(_voice_cache_key)
                if _cached is not None:
                    ref_codes_fq = _cached["ref_codes_fq"].to(
                        device=self.codebook_embeddings.weight.device,
                        dtype=torch.long,
                    )
                    _voice_cache_key = None  # hit → don't store again
                    logger.debug("Voice cache HIT for Fish Speech voice '%s'", voice_name)
                    return self._apply_codebook_embeddings(
                        tokenizer,
                        text,
                        ref_text,
                        ref_codes_fq,
                    )

        if not isinstance(ref_audio_sr, int):
            raise ValueError("Fish Speech structured voice clone requires integer ref_audio_sr")

        ref_audio_wav_raw = info_dict.get("ref_audio_wav")
        if ref_audio_wav_raw is None:
            raise ValueError("Fish Speech structured voice clone requires ref_audio_wav")
        if isinstance(ref_audio_wav_raw, torch.Tensor):
            ref_audio_wav = ref_audio_wav_raw.cpu().numpy()
        else:
            ref_audio_wav = np.asarray(ref_audio_wav_raw, dtype=np.float32)

        if (
            _voice_cache_key is None
            and os.environ.get("VLLM_FISH_REF_AUDIO_CODE_CACHE", "1") == "1"
        ):
            try:
                wav_for_hash = np.ascontiguousarray(ref_audio_wav, dtype=np.float32)
                digest = hashlib.blake2b(
                    wav_for_hash.view(np.uint8),
                    digest_size=16,
                ).hexdigest()
                _voice_cache_key = (
                    f"inline-ref-audio:{int(ref_audio_sr)}:"
                    f"{tuple(wav_for_hash.shape)}:{digest}:icl"
                )
                _cached = self._voice_cache.get(_voice_cache_key)
                if _cached is not None:
                    ref_codes_fq = _cached["ref_codes_fq"].to(
                        device=self.codebook_embeddings.weight.device,
                        dtype=torch.long,
                    )
                    logger.debug("Voice cache HIT for inline Fish Speech reference audio")
                    return self._apply_codebook_embeddings(
                        tokenizer,
                        text,
                        ref_text,
                        ref_codes_fq,
                    )
            except Exception:
                logger.exception("Failed to build inline Fish Speech ref_audio cache key")
                _voice_cache_key = None

        ref_codes_fq = encode_reference_audio_codes(
            self.model_path,
            ref_audio_wav,
            ref_audio_sr,
            device=self.codebook_embeddings.weight.device,
        )

        # Cache miss: store DAC codes for future reuse.
        if _voice_cache_key is not None:
            self._voice_cache.put(
                _voice_cache_key,
                {"ref_codes_fq": ref_codes_fq.detach().cpu()},
            )
            logger.debug("Voice cache STORE for Fish Speech voice '%s'", voice_name)

        return self._apply_codebook_embeddings(tokenizer, text, ref_text, ref_codes_fq)

    def _apply_codebook_embeddings(
        self,
        tokenizer: Any,
        text: str,
        ref_text: str,
        ref_codes_fq: torch.Tensor,
    ) -> torch.Tensor:
        """Build prefill embeddings from DAC codes and inject codebook conditioning."""
        semantic_token_ids = (ref_codes_fq[:, 0] + self._semantic_begin_id).tolist()
        prompt_ids, _, _ = build_fish_voice_clone_prompt_ids(
            tokenizer,
            text,
            ref_text,
            semantic_token_ids,
        )
        prompt_ids = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=self.codebook_embeddings.weight.device,
        )
        embeds = self.embed_input_ids(prompt_ids.unsqueeze(0)).squeeze(0).to(dtype=torch.bfloat16)

        audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")
        start_pos = (prompt_ids == int(audio_start_id)).nonzero(as_tuple=False)
        end_pos = (prompt_ids == int(audio_end_id)).nonzero(as_tuple=False)
        if start_pos.numel() == 0 or end_pos.numel() == 0:
            return embeds
        s = int(start_pos[0].item()) + 1
        e = int(end_pos[0].item())
        if e <= s:
            return embeds

        frames_in_prompt = e - s
        if ref_codes_fq.device != embeds.device:
            ref_codes_fq = ref_codes_fq.to(device=embeds.device, dtype=torch.long)
        frames = min(int(ref_codes_fq.shape[0]), int(frames_in_prompt))
        if frames <= 0:
            return embeds

        q = min(int(ref_codes_fq.shape[1]), self._num_codebooks)
        offsets = (torch.arange(q, device=embeds.device, dtype=torch.long) * self._codebook_size).unsqueeze(0)
        ref_codes_slice = ref_codes_fq[:frames, :q]
        if bool((ref_codes_slice < 0).any().item()):
            logger.warning("Fish Speech structured clone saw negative DAC codes; clamping them to zero")
        code_with_offset = ref_codes_slice.clamp(min=0) + offsets
        codebook_sum = self.codebook_embeddings(code_with_offset).sum(dim=1).to(dtype=embeds.dtype)

        result = embeds.clone()
        result[s : s + frames] = (result[s : s + frames] + codebook_sum) / math.sqrt(self._num_codebooks + 1)
        return result.to(dtype=torch.bfloat16)

    # -------------------- GPU-side MTP fast-path --------------------

    @torch.inference_mode()
    def inject_codebook_embeddings(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        audio_codes: torch.Tensor,
    ) -> torch.Tensor:
        bsz = int(input_ids.shape[0])
        dev = input_embeds.device
        input_ids = input_ids.reshape(bsz, 1).to(dtype=torch.long, device=dev)
        audio_codes = audio_codes.reshape(bsz, self._num_codebooks).to(dtype=torch.long, device=dev)

        inputs_embeds_out = input_embeds.reshape(bsz, -1).clone()
        semantic_mask = (input_ids[:, 0] >= self._semantic_begin_id) & (input_ids[:, 0] <= self._semantic_end_id)
        semantic_codes = audio_codes.clamp(min=0, max=self._codebook_size - 1)
        offsets = (
            torch.arange(self._num_codebooks, device=dev, dtype=semantic_codes.dtype) * self._codebook_size
        ).unsqueeze(0)
        codebook_sum = self.codebook_embeddings(semantic_codes + offsets).sum(dim=1).to(dtype=torch.bfloat16)
        norm_embeds = (inputs_embeds_out + codebook_sum) / math.sqrt(self._num_codebooks + 1)
        return torch.where(semantic_mask.unsqueeze(-1), norm_embeds, inputs_embeds_out)

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU fast-path: run Fast AR to predict residual codebook codes.

        Returns (inputs_embeds, audio_codes).

        When unified decode is active, codes were pre-computed in forward()'s
        _inline_decode_codebooks() and stored in persistent buffers.  This
        method becomes a lightweight reader: fetch codes, compute codebook
        embeddings, inject into input_embeds.

        When unified decode is NOT active (fallback), runs the full Fast AR
        codebook loop as before.
        """
        bsz = int(input_ids.shape[0])
        dev = input_embeds.device

        input_ids = input_ids.reshape(bsz, 1).to(dtype=torch.long, device=dev)

        if self._unified_decode and self._codes_valid and self._output_codes is not None:
            # Read pre-computed codes from the persistent buffer.
            # _codes_valid is set after the first _inline_decode_codebooks() call.
            audio_codes = self._output_codes[:bsz]
        else:
            # Fallback: run Fast AR as before.
            past_hidden = last_talker_hidden.reshape(bsz, -1).to(dtype=torch.bfloat16, device=dev)
            audio_codes = self.fast_ar(
                slow_ar_hidden=past_hidden,
                semantic_token_id=input_ids.reshape(bsz),
                do_sample=True,
                temperature=0.8,
                top_k=30,
                top_p=0.9,
            )  # [B, num_codebooks]

        inputs_embeds_out = self.inject_codebook_embeddings(input_ids, input_embeds, audio_codes)
        return inputs_embeds_out, audio_codes.to(dtype=torch.long)

    # -------------------- Prompt length estimation --------------------

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        **kwargs: Any,
    ) -> int:
        """Estimate prompt length for placeholder allocation."""
        info = additional_information or {}
        text = info.get("text", [""])[0] if isinstance(info.get("text"), list) else info.get("text", "")
        # Conservative estimate: tokenize text length + overhead.
        return max(2, len(str(text)) // 2 + 64)

    # -------------------- Weight loading --------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Fish Speech → Qwen3 format transformation.

        Transforms weight names (wqkv → q/k/v split, w1/w2/w3 → gate/up/down,
        etc.) and routes to the correct sub-modules.
        """
        n_head = self.text_config.num_attention_heads
        n_local_heads = self.text_config.num_key_value_heads
        head_dim = self.text_config.head_dim
        fast_n_head = self.fast_ar_config.num_attention_heads
        fast_n_local_heads = self.fast_ar_config.num_key_value_heads
        fast_head_dim = self.fast_ar_config.head_dim

        remapped = _remap_fish_speech_weights(
            weights,
            n_head,
            n_local_heads,
            head_dim,
            fast_n_head,
            fast_n_local_heads,
            fast_head_dim,
        )

        # Qwen3Model uses stacked_params_mapping for q/k/v → qkv_proj
        # and gate/up → gate_up_proj.  Feed the remapped weights through
        # the standard Qwen3 loading path.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in remapped:
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle tied embeddings → lm_head.
            if name == "model.embed_tokens.weight" and self.text_config.tie_word_embeddings:
                # Also load into lm_head if present.
                lm_key = "lm_head.weight"
                if lm_key in params_dict:
                    p = params_dict[lm_key]
                    wl = getattr(p, "weight_loader", default_weight_loader)
                    wl(p, loaded_weight)
                    loaded_params.add(lm_key)

            # Try stacked params mapping (q/k/v → qkv_proj, gate/up → gate_up_proj).
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                handled = True
                break

            if handled:
                continue

            # Direct parameter mapping.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info("Loaded %d weights for FishSpeechSlowARForConditionalGeneration", len(loaded_params))

        # Truncate RoPE cos/sin caches to bf16 precision to match training.
        # Without this, f32 RoPE values cause logit divergence and premature EOS.
        truncated = 0
        for module in self.modules():
            if hasattr(module, "cos_sin_cache") and isinstance(module.cos_sin_cache, torch.Tensor):
                cache = module.cos_sin_cache
                module.cos_sin_cache = cache.to(torch.bfloat16).to(cache.dtype)
                truncated += 1
        if truncated:
            logger.info("Truncated %d RoPE cos_sin_cache buffers to bf16 precision", truncated)

        if not getattr(self, "talker_mtp_graph_safe", False):
            try:
                self.fast_ar.warmup_compile(
                    device=self.codebook_embeddings.weight.device,
                    dtype=torch.bfloat16,
                    batch_sizes=(1,),
                )
            except Exception as exc:
                logger.warning("Fish Speech Fast AR compile warmup failed: %s", exc)

        codec_device = self.codebook_embeddings.weight.device
        _load_dac_codec(
            self.model_path,
            device=codec_device,
            dtype=torch.float32,
        )

        # Enable single-stage inline DAC when engine_output_type is "audio".
        engine_output_type = getattr(
            getattr(self.vllm_config, "model_config", None), "engine_output_type", None
        )
        if engine_output_type is None:
            engine_output_type = os.environ.get("VLLM_FISH_SINGLE_STAGE_OUTPUT", "")
        if engine_output_type == "audio":
            self._inline_dac = True
            logger.info("Single-stage mode: inline DAC decode enabled")

        return loaded_params
