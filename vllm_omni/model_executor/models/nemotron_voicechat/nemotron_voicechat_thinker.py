# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 thinker for NemotronVoiceChat: 16 kHz speech -> frame-locked text.

``LLM_AR`` stage combining:

* the vendored NeMo FastConformer perception (mel + 24-layer encoder + linear
  projection to the LLM width, 12.5 Hz frames);
* the additive channel fusion from the NeMo duplex STT model
  (``fused[t] = E(text_prev) * w_text + timeline[t] * w_audio +
  E(func_prev) * w_func`` with the checkpoint's AddFusion weights);
* the NemotronH (hybrid Mamba) backbone on vLLM paged execution via
  ``init_vllm_registered_model`` — this is where vLLM earns its keep;
* the separate ``lm_head`` (text channel, sampled by the vLLM engine) and
  ``function_head`` (greedy, maintained internally per request).

Timeline contract (verified against NeMo ``_init_inference``/``_step_zero``/
``_step_inference``; the golden shape test guards it):

* The vLLM prompt covers ``logical_prompt_token_len + 1`` positions: the system
  prompt tokens (``[bos] + tokens + [eos]``) fused with a PAD text channel
  (NeMo's ``_get_bos_embedding`` returns the PAD embedding, so position 0 is
  uniform with the rest of the prompt region), plus acoustic frame 0 as the
  final prefill position. The first engine-sampled token is therefore NeMo's
  ``gen_text[prompt_len]`` — the first acoustic-frame prediction.
* Each decode step t consumes conformer frame t (held in per-request state)
  fused with the previously sampled text token and the previous greedy
  function token. Text EOS never terminates generation (``ignore_eos``); the
  example sets ``min_tokens = max_tokens = acoustic_frame_count``.
* The function channel exists for numerical correctness (fusion weight 2.0)
  even though tool calling is out of scope; the greedy token timeline is
  exposed for debugging via the request info.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_LM_ARCHITECTURE = "NemotronHForCausalLM"


def compute_acoustic_frame_count(nemo_stt_cfg: dict[str, Any], num_samples: int) -> int:
    """Exact 12.5 Hz frame count for ``num_samples`` of 16 kHz audio.

    Replicates the vendored mel featurizer length arithmetic followed by the
    dw-striding conv subsampling length recurrence (NOT ``duration / 0.08``).
    The thinker validates this against the real perception output at prefill,
    and the golden shape test pins it against the NeMo reference.
    """
    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.subsampling import (
        calc_length,
    )

    perception = nemo_stt_cfg.get("perception") or {}
    pre = perception.get("preprocessor") or {}
    encoder = perception.get("encoder") or {}
    sample_rate = int(pre.get("sample_rate", 16000))
    window_stride = float(pre.get("window_stride", 0.01))
    window_size = float(pre.get("window_size", 0.025))
    n_fft = int(pre.get("n_fft") or 2 ** int(np.ceil(np.log2(window_size * sample_rate))))
    hop = int(window_stride * sample_rate)
    pad_amount = n_fft // 2 * 2  # torch.stft center padding on both sides
    mel_len = (num_samples + pad_amount - n_fft) // hop + 1
    subsampling_factor = int(encoder.get("subsampling", "dw_striding") and encoder.get("subsampling_factor", 8))
    conv_kernel = int(encoder.get("subsampling_conv_kernel_size", 3) or 3)
    repeat = int(np.log2(subsampling_factor))
    out = calc_length(
        torch.tensor([mel_len]),
        all_paddings=2 * (conv_kernel // 2),
        kernel_size=conv_kernel,
        stride=2,
        ceil_mode=False,
        repeat_num=repeat,
    )
    return int(out[0])


class NemotronVoiceChatThinkerForConditionalGeneration(nn.Module, HasInnerState, IsHybrid):
    """vLLM-native thinker (perception + fusion + NemotronH + dual heads).

    Declares the hybrid-Mamba interfaces (``IsHybrid``/``HasInnerState``) so
    vLLM configures the mamba state cache for the wrapped NemotronH backbone;
    the state shape/dtype classmethods mirror ``NemotronHForCausalLM`` but read
    the NemotronH sub-config (``get_text_config()``) instead of the top-level
    checkpoint config.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig") -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        intermediate_size = hf_config.mamba_num_heads * hf_config.mamba_head_dim
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        stt_cfg: dict[str, Any] = dict(getattr(config, "stt_cfg", {}) or {})
        if not stt_cfg:
            raise ValueError("NemotronVoiceChat checkpoint config lacks 'model.stt.model'; cannot build the thinker.")
        self.stt_cfg = stt_cfg
        text_cfg = config.get_text_config()
        self._hidden = int(getattr(text_cfg, "hidden_size", 4480))
        self._vocab = int(getattr(text_cfg, "vocab_size", 131072))
        self._dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)

        # NemotronH backbone (+ lm_head) on vLLM native layers.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_cfg,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=[_LM_ARCHITECTURE],
        )

        # Shared channel embedding + separate function head (both distinct
        # checkpoints subtrees from the backbone).
        self.embed_tokens = nn.Embedding(self._vocab, self._hidden)
        self.function_head = nn.Linear(self._hidden, self._vocab, bias=False)

        # Vendored perception (mel + FastConformer + projection).
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import (
            AudioPerceptionModule,
        )

        perception_cfg = stt_cfg.get("perception")
        if not perception_cfg:
            raise ValueError("'model.stt.model.perception' missing from the checkpoint config.")
        from omegaconf import DictConfig

        self.perception = AudioPerceptionModule(DictConfig(perception_cfg))

        # AddFusion weights (the checkpoint uses fuse_method="add" defaults;
        # the same duplex_*_weight keys double as the fusion weights).
        use_function_head = bool(stt_cfg.get("use_function_head", True))
        self._w_text = float(stt_cfg.get("duplex_text_channel_weight", 1.0))
        self._w_audio = float(stt_cfg.get("duplex_user_channel_weight", 1.0))
        self._w_func = float(stt_cfg.get("duplex_function_channel_weight", 1.0)) if use_function_head else 0.0
        self._use_function_head = use_function_head
        fuse_method = stt_cfg.get("fuse_method", "add") or "add"
        if fuse_method != "add":
            raise NotImplementedError(
                f"NemotronVoiceChat thinker only implements fuse_method='add' "
                f"(this checkpoint uses it); got {fuse_method!r}."
            )
        if bool(stt_cfg.get("predict_user_text", False)):
            raise NotImplementedError(
                "NemotronVoiceChat thinker does not implement the user-ASR channel "
                "(predict_user_text=True); this checkpoint has it disabled."
            )

        self._special_ids: dict[str, int] | None = None

        # Omni AR runner contract.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True  # capture per-step hidden for the function channel
        self.requires_full_prefix_cached_hidden_states = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}

        # request_id -> per-request state (audio frames, prefill embeds, counters).
        self._sessions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Special token ids (from the checkpoint's token strings).
    # ------------------------------------------------------------------
    def _resolve_special_ids(self) -> dict[str, int]:
        if self._special_ids is not None:
            return self._special_ids
        import os

        from transformers import AutoTokenizer

        ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or self.stt_cfg.get(
            "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        )
        tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
        ids = {
            "bos": tok.convert_tokens_to_ids(self.stt_cfg.get("bos_token", "<s>")),
            "eos": tok.convert_tokens_to_ids(self.stt_cfg.get("eos_token", "</s>")),
            "pad": tok.convert_tokens_to_ids(self.stt_cfg.get("pad_token", "<SPECIAL_12>")),
        }
        for name, value in ids.items():
            if value is None or value < 0:
                raise ValueError(f"NemotronVoiceChat could not resolve the {name} token id from {ref!r}.")
        self._special_ids = ids
        return ids

    # ------------------------------------------------------------------
    # vLLM model protocol.
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Placeholder; preprocess replaces every position with fused embeds.
        return torch.zeros((input_ids.shape[0], self._hidden), device=input_ids.device, dtype=self._dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        # vLLM >= 0.25 dropped the sampling_metadata parameter.
        return self.language_model.compute_logits(hidden_states)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    # ------------------------------------------------------------------
    # Fusion helpers.
    # ------------------------------------------------------------------
    def _fuse(
        self,
        text_ids: torch.Tensor,
        timeline_embeds: torch.Tensor,
        func_ids: torch.Tensor,
    ) -> torch.Tensor:
        """AddFusion over the three channels; shapes [T] ids, [T, H] timeline."""
        text_e = self.embed_tokens(text_ids)
        fused = text_e * self._w_text + timeline_embeds.to(text_e.dtype) * self._w_audio
        if self._use_function_head:
            fused = fused + self.embed_tokens(func_ids) * self._w_func
        return fused

    # ------------------------------------------------------------------
    # Per-request session.
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_info(info_dict: dict[str, Any]) -> dict[str, Any]:
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            return merged
        return info_dict

    def _request_id(self, info: dict[str, Any]) -> str:
        request_id = info.get("request_id")
        if request_id is None:
            meta = info.get("meta")
            if isinstance(meta, dict):
                request_id = meta.get("request_id")
        if request_id is None:
            raise RuntimeError("NemotronVoiceChat thinker requires a request_id in the runtime info.")
        return str(request_id)

    def _load_audio(self, info: dict[str, Any], device: torch.device) -> torch.Tensor:
        audio = info.get("nvc_audio")
        if audio is None:
            raise ValueError(
                "NemotronVoiceChat thinker request is missing its audio: pass "
                "additional_information={'nvc_audio': <16 kHz mono float waveform>, "
                "'nvc_sr': 16000, 'nvc_prompt_token_ids': [...]}."
            )
        wav = torch.as_tensor(np.asarray(audio), dtype=torch.float32, device=device).reshape(-1)
        if wav.numel() == 0:
            raise ValueError("NemotronVoiceChat thinker received empty audio input.")
        sr = int(info.get("nvc_sr", 16000))
        expected_sr = int(getattr(self.config, "source_sample_rate", 16000))
        if sr != expected_sr:
            raise ValueError(
                f"NemotronVoiceChat thinker expects {expected_sr} Hz audio, got {sr} Hz — "
                "resample client-side (the offline example does this)."
            )
        return wav

    def _init_session(self, request_id: str, info: dict[str, Any], device: torch.device) -> dict[str, Any]:
        pad_id = self._resolve_special_ids()["pad"]
        wav = self._load_audio(info, device)

        prompt_ids = info.get("nvc_prompt_token_ids")
        if prompt_ids is None:
            raise ValueError(
                "NemotronVoiceChat thinker request is missing 'nvc_prompt_token_ids' "
                "(the system prompt token ids, [bos] + tokens + [eos])."
            )
        prompt_ids = torch.as_tensor(list(prompt_ids), dtype=torch.long, device=device)
        if prompt_ids.numel() == 0:
            raise ValueError("NemotronVoiceChat thinker requires a non-empty system prompt.")

        # Perception: mel + FastConformer + projection -> [N, hidden] frames.
        with torch.inference_mode():
            encoded, encoded_len = self.perception(
                input_signal=wav.unsqueeze(0).to(next(self.perception.parameters()).dtype),
                input_signal_length=torch.tensor([wav.numel()], device=device),
            )
        n_frames = int(encoded_len.reshape(-1)[0])
        audio_embeds = encoded[0, :n_frames].to(self._dtype)  # [N, H]
        expected = info.get("nvc_expected_frames")
        if expected is not None and int(expected) != n_frames:
            raise ValueError(
                f"NemotronVoiceChat frame-count mismatch: the request was sized for "
                f"{int(expected)} acoustic frames but perception produced {n_frames}. "
                "The example must size min_tokens/max_tokens with compute_acoustic_frame_count()."
            )

        # Fused prefill embeds for positions 0..P (P prompt slots + frame 0).
        # All prompt positions use the PAD text channel (NeMo's
        # _get_bos_embedding returns the PAD embedding) and PAD function channel.
        p = int(prompt_ids.numel())
        timeline = torch.cat([self.embed_tokens(prompt_ids).to(self._dtype), audio_embeds[0:1]], dim=0)  # [P+1, H]
        pad_ids = torch.full((p + 1,), pad_id, dtype=torch.long, device=device)
        prefill = self._fuse(pad_ids, timeline, pad_ids)  # [P+1, H]

        session = {
            "audio_embeds": audio_embeds,
            "prefill_embeds": prefill,
            "prompt_len": p,
            "n_frames": n_frames,
            "func_token": int(pad_id),
            "func_tokens": [],
            "decode_step": 0,
        }
        self._sessions[request_id] = session
        return session

    # ------------------------------------------------------------------
    # Omni AR hooks.
    # ------------------------------------------------------------------
    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        info = self._merge_info(info_dict)
        device = input_ids.device
        span = int(input_ids.shape[0])
        is_prefill_raw = info.get("_omni_is_prefill")
        is_prefill = bool(is_prefill_raw) if isinstance(is_prefill_raw, bool) else span > 1
        request_id = self._request_id(info)

        if is_prefill or request_id not in self._sessions:
            session = self._sessions.get(request_id) or self._init_session(request_id, info, device)
            offset = max(0, int(info.get("_omni_num_computed_tokens", 0) or 0))
            prefill: torch.Tensor = session["prefill_embeds"]
            if offset + span > prefill.shape[0]:
                raise ValueError(
                    f"NemotronVoiceChat thinker prefill span [{offset}, {offset + span}) exceeds "
                    f"the fused prompt length {prefill.shape[0]}; the request prompt must be "
                    "the system prompt tokens plus exactly one placeholder position."
                )
            return input_ids, prefill[offset : offset + span].to(self._dtype), {}

        session = self._sessions[request_id]
        pad_id = self._resolve_special_ids()["pad"]

        # Function channel: greedy argmax over the PREVIOUS position's hidden
        # (captured by postprocess). At the first decode step that hidden is the
        # prefill's last position (acoustic frame 0), matching NeMo's
        # gen_function[prompt_len].
        if self._use_function_head:
            hs = info.get("hidden_states", {})
            last_hidden = hs.get("last") if isinstance(hs, dict) else None
            if isinstance(last_hidden, torch.Tensor) and last_hidden.numel() > 0:
                with torch.inference_mode():
                    func_logits = self.function_head(last_hidden.reshape(1, -1).to(self._dtype))
                session["func_token"] = int(func_logits.argmax(dim=-1))
                session["func_tokens"].append(session["func_token"])

        session["decode_step"] = int(session["decode_step"]) + 1
        frame_idx = session["decode_step"]  # decode step k consumes frame k
        n_frames = int(session["n_frames"])
        if frame_idx >= n_frames:
            raise ValueError(
                f"NemotronVoiceChat thinker was asked for decode step {frame_idx} but only "
                f"{n_frames} acoustic frames exist; max_tokens must equal the frame count."
            )
        text_prev = input_ids.reshape(-1)[:1].to(torch.long)
        func_prev = torch.full_like(text_prev, session["func_token"] if self._use_function_head else pad_id)
        frame = session["audio_embeds"][frame_idx : frame_idx + 1]
        fused = self._fuse(text_prev, frame, func_prev)  # [1, H]

        info_update: dict[str, Any] = {
            "meta": {"nvc_frame": frame_idx},
            "nvc_function_tokens": torch.tensor(session["func_tokens"], dtype=torch.long),
            "nvc_text_pad_id": pad_id,
        }
        return input_ids, fused, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """Capture the step's last hidden for the next function-channel argmax."""
        if hidden_states is None or hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1, :].detach()}}

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._sessions.pop(str(request_id), None)

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route the unified checkpoint into the thinker's components.

        * ``stt_model.llm.*`` -> vLLM NemotronH (renamed to ``backbone.*``; the
          separate ``stt_model.embed_tokens`` doubles as ``backbone.embeddings``
          since the checkpoint keeps a single shared text embedding).
        * ``stt_model.lm_head.weight`` -> the backbone's ``lm_head``.
        * ``stt_model.embed_tokens/function_head`` -> the fusion embedding /
          function head.
        * ``stt_model.perception.*`` -> the vendored Conformer.
        * everything else (``rnnt_*``, ``tts_model.*``) is skipped lazily.
        """
        loaded: set[str] = set()
        lm_weights: list[tuple[str, torch.Tensor]] = []
        perception_state: dict[str, torch.Tensor] = {}
        embed_weight: torch.Tensor | None = None

        for name, tensor in weights:
            if name.startswith("stt_model.llm."):
                lm_weights.append(("backbone." + name.removeprefix("stt_model.llm."), tensor))
            elif name == "stt_model.lm_head.weight":
                lm_weights.append(("lm_head.weight", tensor))
            elif name == "stt_model.embed_tokens.weight":
                embed_weight = tensor
            elif name == "stt_model.function_head.weight":
                with torch.no_grad():
                    self.function_head.weight.copy_(tensor)
                loaded.add("function_head.weight")
            elif name.startswith("stt_model.perception."):
                perception_state[name.removeprefix("stt_model.perception.")] = tensor

        if embed_weight is None:
            raise ValueError("Checkpoint is missing 'stt_model.embed_tokens.weight'.")
        if not lm_weights:
            raise ValueError("Checkpoint is missing the 'stt_model.llm.*' backbone subtree.")
        if not perception_state:
            raise ValueError("Checkpoint is missing the 'stt_model.perception.*' subtree.")
        if "function_head.weight" not in loaded and self._use_function_head:
            raise ValueError("Checkpoint is missing 'stt_model.function_head.weight'.")

        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight)
        loaded.add("embed_tokens.weight")

        # The vLLM NemotronH model expects an input embedding; the checkpoint's
        # shared embed_tokens fills that slot (inputs_embeds bypass it at runtime).
        lm_weights.append(("backbone.embeddings.weight", embed_weight))
        lm_loaded = self.language_model.load_weights(iter(lm_weights))
        loaded |= {f"language_model.{n}" for n in lm_loaded}

        missing, unexpected = self.perception.load_state_dict(perception_state, strict=False)
        if missing:
            raise ValueError(f"NemotronVoiceChat perception is missing weights: {sorted(missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat perception ignoring %d unexpected tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        self.perception.to(device=device, dtype=self._dtype).eval()
        self.embed_tokens.to(device=device, dtype=self._dtype)
        self.function_head.to(device=device, dtype=self._dtype)
        loaded |= {f"perception.{n}" for n in perception_state}
        logger.info(
            "NemotronVoiceChat thinker loaded: %d backbone tensors, %d perception tensors, "
            "fusion weights (text=%.1f, audio=%.1f, function=%.1f)",
            len(lm_weights),
            len(perception_state),
            self._w_text,
            self._w_audio,
            self._w_func,
        )
        return loaded
