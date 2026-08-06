# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 talker for NemotronVoiceChat: frame-locked text -> 31-quantizer codes.

``LLM_AR`` stage driven per frame by the vLLM engine, with the vendored NeMo
EAR-TTS model (28-layer Gemma3-style backbone + MoG head) executed inside the
per-request ``preprocess`` hook. Classifier-free guidance (internal batch
doubling) and MoG sampling happen inside the vendored forward exactly as in
NeMo — vLLM's sampler never touches the audio codes.

Execution contract (mirrors the NeMo ``NemotronVoiceChat.offline_inference``
TTS half):

* The request's ``additional_information`` carries the full frame-aligned text
  timeline (built by ``thinker2talker_token_only``), INCLUDING the PAD prompt
  region — the talker's KV state at the first acoustic frame depends on having
  processed it (NeMo trims prompt-region codes only after the loop).
* vLLM prompt = one placeholder token (timeline position 0; NeMo's loop starts
  at t=1). Each decode step t runs ``infer_codes_one_step`` with
  ``current_subword_id=timeline[t]``, ``prev_subword_id=timeline[t-1]`` (the
  first step uses the warmup's last init subword, as in NeMo), and the previous
  step's code stack; the resulting ``[1, 31]`` codes accumulate under
  ``("codes","audio")`` in the request state for the full-payload producer.
* Warmup: ``set_init_inputs(speaker_name=...)`` (pre-baked "Aria" latent) +
  one ``tts_model(**init_inputs)`` forward seeds ``past_key_values`` and the
  initial code, exactly as in the reference.
* Completion: after consuming timeline position ``T-1`` the stage emits a stop
  placeholder token (the deploy yaml sets ``stop_token_ids: [1]``); codes for
  positions ``1..T-1`` are shipped and the producer trims the prompt region.

Heavyweight per-request state (HF ``past_key_values`` etc.) lives in a
model-side session dict keyed by request id (PersonaPlex pattern), cleaned up
via ``on_requests_finished``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.nemotron_voicechat.runtime_info import (
    merge_runtime_info,
    require_request_id,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_CONTINUE_TOKEN = 0
_STOP_TOKEN = 1


def sanitize_tts_model_cfg(tts_model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Make the vendored DuplexEARTTS construction config-only.

    Every parameter comes from the unified checkpoint streamed in
    ``load_weights``; the vendored constructors must never download or load
    external pretrained weights. Returns a deep copy so the shared stage
    config is never mutated:

    * ``tts_config.pretrained_text_name`` would make ``RVQEARTTSModel``
      download a full LLM backbone — dropped (the backbone builds from
      ``backbone_type``/``backbone_config``).
    * ``pretrained_codec_model`` would make ``setup_audio_codec`` load an
      external codec checkpoint at construction — dropped (the codec weights
      arrive with the ``tts_model.audio_codec.*`` subtree, and the RVQ
      embedding binding is overwritten by the same ``load_state_dict``).
    * ``tts_config.cas_config.pretrained_tokenizer_name`` is dropped, matching
      NeMo's own inference override (the CAS vocab is baked into the weights).
    * ``pretrained_lm_name`` survives ONLY as the tokenizer reference (the
      context-LM load path is disabled by ``context_hidden_size: null``); the
      ``NEMOTRON_VOICECHAT_LLM_PATH`` override keeps it air-gap friendly.
    """
    import copy
    import os

    cfg = copy.deepcopy(tts_model_cfg)
    cfg.pop("pretrained_codec_model", None)
    tts_config = cfg.get("tts_config")
    if isinstance(tts_config, dict):
        tts_config.pop("pretrained_text_name", None)
        cas_cfg = tts_config.get("cas_config")
        if isinstance(cas_cfg, dict):
            cas_cfg.pop("pretrained_tokenizer_name", None)
    llm_override = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH")
    if llm_override:
        cfg["pretrained_lm_name"] = llm_override
    return cfg


class NemotronVoiceChatTalkerForConditionalGeneration(nn.Module):
    """vLLM AR stage wrapping the vendored NeMo EAR-TTS per-frame step."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        talker_cfg = getattr(self.config, "talker_config", None)
        self._hidden = int(getattr(talker_cfg, "hidden_size", 1152))
        self._vocab = max(int(getattr(talker_cfg, "vocab_size", 1024)), 2)
        self._dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self._speaker_name = getattr(self.config, "inference_speaker_name", "Aria") or "Aria"

        # Omni AR runner contract flags.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_full_prefix_cached_hidden_states = False

        # Stateful per-request execution is only validated at batch size 1
        # (the shipped offline scope); fail fast on silent multi-request use.
        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        if max_num_seqs != 1:
            raise NotImplementedError(
                f"NemotronVoiceChat talker supports max_num_seqs=1 only (got {max_num_seqs}); "
                "the vendored EAR-TTS session state is not batch-safe."
            )

        # Vendored DuplexEARTTS; constructed in load_weights.
        self.tts: nn.Module | None = None
        # request_id -> mutable session state (past_key_values, prev code, ...).
        self._sessions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # vLLM model protocol (placeholder token loop).
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self._hidden), device=input_ids.device, dtype=self._dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        # The per-frame compute happens in preprocess; the "hidden state" only
        # transports the stop flag (column 0) to compute_logits.
        if inputs_embeds is not None:
            return inputs_embeds
        assert input_ids is not None
        return self.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        hs = hidden_states
        flag = hs[..., 0].to(torch.float32)  # 1.0 => finished
        logits = torch.full((hs.shape[0], self._vocab), -1.0e4, device=hs.device, dtype=torch.float32)
        logits[:, _CONTINUE_TOKEN] = (1.0 - flag) * 10.0
        logits[:, _STOP_TOKEN] = flag * 10.0
        return logits

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {})
            audio = codes.get("audio") if isinstance(codes, dict) else None
            if isinstance(audio, torch.Tensor):
                codes_list.append(audio)
        if not codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": torch.cat(codes_list, dim=0)}},
        )

    # ------------------------------------------------------------------
    # Per-frame vendored TTS step.
    # ------------------------------------------------------------------
    def _timeline(self, info: dict[str, Any]) -> torch.Tensor:
        timeline = info.get("nvc_text_timeline")
        if timeline is None or info.get("nvc_logical_prompt_len") is None:
            # Explicit failure — no prompt_token_ids fallback: an implicit
            # timeline would silently skip the prompt-region trim downstream.
            raise ValueError(
                "NemotronVoiceChat talker request is missing its timeline metadata "
                "('nvc_text_timeline'/'nvc_logical_prompt_len' in additional_information); "
                "the stage cannot synthesize speech without the thinker's frame-locked tokens."
            )
        if isinstance(timeline, torch.Tensor):
            return timeline.reshape(-1).to(torch.long)
        return torch.as_tensor(list(timeline), dtype=torch.long)

    def _init_session(self, request_id: str, info: dict[str, Any], device: torch.device) -> dict[str, Any]:
        assert self.tts is not None
        timeline = self._timeline(info).to(device)
        if timeline.numel() < 2:
            raise ValueError(
                f"NemotronVoiceChat talker timeline for request {request_id} has "
                f"{timeline.numel()} positions; need at least 2 (prompt + one frame)."
            )
        # Warmup: pre-baked speaker latent + init prompt forward (NeMo
        # offline_inference lines around get_init_inputs/first forward).
        self.tts.set_init_inputs(speaker_name=self._speaker_name)
        init_inputs = self.tts.get_init_inputs(B=1)
        generation_config = self.tts._get_generation_config(self._guidance_enabled())
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": self._guidance_enabled()})
        with torch.inference_mode():
            outputs = self.tts.tts_model(**init_inputs)
        session = {
            "timeline": timeline,
            "step": 1,  # NeMo's loop starts at t=1
            "past_key_values": outputs.past_key_values,
            "code": init_inputs["code"][:, -1:],
            "first_context_subword_id": init_inputs["subword_ids"][:, -1].unsqueeze(-1),
            "tts_initialized": False,
            "generation_config": generation_config,
            "codes_rows": [],
        }
        self._sessions[request_id] = session
        return session

    def _guidance_enabled(self) -> bool:
        return bool(getattr(self.config, "tts_cfg", {}).get("inference_guidance_enabled", True))

    def _step_session(self, session: dict[str, Any]) -> tuple[torch.Tensor, bool]:
        """One NeMo TTS step; returns (codes_row [1, Q], finished)."""
        assert self.tts is not None
        timeline: torch.Tensor = session["timeline"]
        t = int(session["step"])
        total = int(timeline.numel())
        current_subword_id = timeline[t].reshape(1, 1)
        if not session["tts_initialized"]:
            prev_subword_id = session["first_context_subword_id"]
            session["tts_initialized"] = True
        else:
            prev_subword_id = timeline[t - 1].reshape(1, 1)
        current_subword_mask = torch.ones(1, 1, device=timeline.device, dtype=torch.bool)
        with torch.inference_mode():
            code, past_key_values = self.tts.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=session["code"],
                past_key_values=session["past_key_values"],
                guidance_enabled=self._guidance_enabled(),
                generation_config=session["generation_config"],
                ignore_eos_flag_stop=True,
            )
        session["code"] = code
        session["past_key_values"] = past_key_values
        session["step"] = t + 1
        finished = (t + 1) >= total
        return code.reshape(1, -1).to(torch.long), finished

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        info = merge_runtime_info(info_dict)
        device = input_ids.device
        span = int(input_ids.shape[0])
        is_prefill_raw = info.get("_omni_is_prefill")
        is_prefill = bool(is_prefill_raw) if isinstance(is_prefill_raw, bool) else span > 1
        request_id = require_request_id(info, "talker")

        embeds = torch.zeros((span, self._hidden), device=device, dtype=self._dtype)
        if is_prefill or request_id not in self._sessions:
            # The 1-token vLLM prefill is the session boundary: run the vendored
            # warmup here so the first decode step is a pure NeMo t=1 step.
            self._init_session(request_id, info, device)
            return input_ids, embeds, {}

        session = self._sessions[request_id]
        codes_row, finished = self._step_session(session)
        session["codes_rows"].append(codes_row.cpu())
        cumulative = torch.cat(session["codes_rows"], dim=0)
        if finished:
            embeds[:, 0] = 1.0  # stop flag -> compute_logits emits the stop token
        info_update: dict[str, Any] = {
            "codes": {"audio": cumulative},
            "meta": {"nvc_tts_step": int(session["step"])},
        }
        return input_ids, embeds, info_update

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._sessions.pop(str(request_id), None)

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Build the vendored DuplexEARTTS and load the ``tts_model.*`` subtree.

        The backbone/MoG run in the engine dtype (bf16 by default); the codec
        submodule inside DuplexEARTTS stays fp32 (it is not used by this stage —
        decode happens in code2wav — but keeping its dtype canonical avoids
        surprises if warmup paths touch it).
        """
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.duplex_ear_tts import (
            DuplexEARTTS,
        )

        tts_model_cfg = dict(getattr(self.config, "tts_cfg", {}) or {})
        if not tts_model_cfg:
            raise ValueError(
                "NemotronVoiceChat checkpoint config lacks 'model.speech_generation.model'; cannot build the talker."
            )
        tts_model_cfg = sanitize_tts_model_cfg(tts_model_cfg)
        tts_data = dict(getattr(self.config, "tts_data", {}) or {})
        tts_data.setdefault("source_sample_rate", 22050)
        tts_data.setdefault("target_sample_rate", int(getattr(self.config, "target_sample_rate", 22050)))
        tts_data.setdefault("frame_length", float(getattr(self.config, "frame_length", 0.08)))
        # DuplexEARTTS consumes the whole speech_generation section layout
        # ({"data": ..., "model": ...}), matching NeMo's constructor call.
        tts = DuplexEARTTS({"data": tts_data, "model": tts_model_cfg})

        prefix = "tts_model."
        state: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if not name.startswith(prefix):
                continue
            target = name.removeprefix(prefix)
            if target.startswith("audio_codec."):
                state[target] = tensor.to(dtype=torch.float32)
            else:
                state[target] = tensor.to(dtype=self._dtype)
        if not state:
            raise ValueError(
                "No 'tts_model.*' tensors found in the checkpoint; the NemotronVoiceChat "
                "unified model.safetensors is required for the talker stage."
            )
        missing, unexpected = tts.load_state_dict(state, strict=False)
        real_missing = [m for m in missing if not m.startswith("audio_codec.")]
        if real_missing:
            raise ValueError(f"NemotronVoiceChat talker is missing TTS weights: {sorted(real_missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat talker ignoring %d unexpected TTS tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        tts = tts.to(device=device)
        tts.tts_model.to(dtype=self._dtype)
        if hasattr(tts, "audio_codec") and tts.audio_codec is not None:
            tts.audio_codec.to(dtype=torch.float32)
        self.tts = tts.eval()
        logger.info(
            "NemotronVoiceChat talker loaded EAR-TTS: %d tensors, speaker=%s, guidance=%s",
            len(state),
            self._speaker_name,
            self._guidance_enabled(),
        )
        return {name for name, _ in self.named_parameters()}
