# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-2 Code2Wav for NemotronVoiceChat: RVQ-VAE codec decode to 22.05 kHz PCM.

``LLM_GENERATION`` stage consuming the talker's per-frame 31-quantizer code
stacks and decoding them with the vendored NeMo RVQ-VAE codec. The decode runs
in fp32 (the NeMo reference wraps codec decode in an fp32 context; bf16 collapses
the reconstruction).

Input contract (mirrors ``PersonaPlexCode2Wav``): the connector payload carries
``{"codes": {"audio": Tensor[frames, num_quantizers]}}`` per request; the
scheduler-side ``input_ids`` are placeholders. Codes must lie in
``[0, codebook_size)`` — out-of-range codes are a hard error (an early bf16-drift
signal upstream), never clamped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def validate_code_stack(codes: torch.Tensor, num_quantizers: int, codebook_size: int) -> torch.Tensor:
    """Validate and reshape a code payload to ``[frames, num_quantizers]``.

    Out-of-range codes are a hard error (an early numeric-drift signal
    upstream), never clamped.
    """
    q = num_quantizers
    if codes.dim() == 1:
        if codes.numel() % q != 0:
            raise ValueError(
                f"NemotronVoiceChat code2wav received a flat code sequence of length "
                f"{codes.numel()} which is not divisible by num_quantizers={q}."
            )
        codes = codes.reshape(-1, q)
    if codes.dim() != 2 or codes.shape[-1] != q:
        raise ValueError(f"NemotronVoiceChat code2wav expects codes shaped [frames, {q}], got {tuple(codes.shape)}.")
    codes = codes.to(dtype=torch.long)
    bad = (codes < 0) | (codes >= codebook_size)
    if bool(bad.any()):
        bad_vals = codes[bad]
        raise ValueError(
            "NemotronVoiceChat code2wav received out-of-range codec codes "
            f"(min={int(bad_vals.min())}, max={int(bad_vals.max())}, count={int(bad.sum())}); "
            f"valid range is [0, {codebook_size}). Refusing to clamp — this usually "
            "signals numeric drift or a corrupted stage payload upstream."
        )
    return codes


class NemotronVoiceChatCode2Wav(nn.Module):
    """RVQ-VAE codec decoder stage (GenerationModelRunner)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        codec_cfg = getattr(self.config, "code2wav_config", None)
        self._num_quantizers = int(getattr(codec_cfg, "num_quantizers", 31))
        self._codebook_size = int(getattr(codec_cfg, "codebook_size", 1024))
        self._sample_rate = int(getattr(codec_cfg, "sample_rate", 22050))

        # Runner-facing capability flags (same contract as PersonaPlexCode2Wav).
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Constructed in load_weights (owns the tts_model.audio_codec.* subtree).
        self.audio_codec: nn.Module | None = None

    # ------------------------------------------------------------------
    # Runner-facing placeholder hooks.
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        return [{"meta": {"nemotron_voicechat_dummy_profile": True}} for _ in range(num_reqs)]

    # ------------------------------------------------------------------
    # Decode.
    # ------------------------------------------------------------------
    def _codes_from_runtime_info(self, runtime_info: Mapping[str, Any] | None) -> torch.Tensor | None:
        if not isinstance(runtime_info, Mapping):
            return None
        codes = runtime_info.get("codes")
        if isinstance(codes, Mapping):
            codes = codes.get("audio")
        if isinstance(codes, list | tuple) and codes:
            codes = torch.as_tensor(codes, dtype=torch.long)
        if isinstance(codes, torch.Tensor) and codes.numel() > 0:
            return codes
        return None

    def _validate_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return validate_code_stack(codes, self._num_quantizers, self._codebook_size)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        infos = runtime_additional_information or []
        num_req = max(len(infos), 1)
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        if self.audio_codec is None:
            raise RuntimeError("NemotronVoiceChatCode2Wav.forward called before load_weights().")
        device = next(self.audio_codec.parameters()).device

        for i in range(num_req):
            runtime_info = infos[i] if i < len(infos) else None
            meta = runtime_info.get("meta") if isinstance(runtime_info, Mapping) else None
            if isinstance(meta, Mapping) and meta.get("nemotron_voicechat_dummy_profile") is True:
                continue
            codes = self._codes_from_runtime_info(runtime_info)
            if codes is None:
                if input_ids is not None and input_ids.numel() > 0 and num_req == 1:
                    codes = input_ids.reshape(-1)
                else:
                    continue
            codes = self._validate_codes(codes).to(device=device)
            frames = codes.shape[0]
            if frames == 0:
                continue
            lens = torch.tensor([frames], device=device, dtype=torch.long)
            # NeMo decodes the codec strictly in fp32.
            with torch.autocast(device_type=device.type, enabled=False):
                wav, wav_len = self.audio_codec.decode(codes.unsqueeze(0), lens)
            wav = wav.reshape(-1)[: int(wav_len.reshape(-1)[0])]
            audios[i] = wav.detach().to(dtype=torch.float32).cpu()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        raise TypeError(f"NemotronVoiceChatCode2Wav expected OmniOutput, got {type(model_outputs)}")

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Build the vendored RVQ-VAE codec and load ``tts_model.audio_codec.*``.

        The vLLM safetensors iterator streams the full unified checkpoint; only
        the codec subtree is materialized here (everything else is skipped
        without copying). The codec stays fp32.
        """
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_vae_codec import (
            RVQVAEModel,
        )

        codec_cfg = dict(getattr(self.config, "tts_cfg", {}).get("codec_config") or {})
        if not codec_cfg:
            raise ValueError(
                "NemotronVoiceChat checkpoint config lacks "
                "'model.speech_generation.model.codec_config'; cannot build the RVQ-VAE codec."
            )
        from omegaconf import DictConfig

        codec = RVQVAEModel(DictConfig(codec_cfg))

        prefix = "tts_model.audio_codec."
        state: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = tensor.to(dtype=torch.float32)
        if not state:
            raise ValueError(
                f"No '{prefix}*' tensors found in the checkpoint; the NemotronVoiceChat "
                "unified model.safetensors is required for the code2wav stage."
            )
        missing, unexpected = codec.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"NemotronVoiceChat code2wav is missing codec weights: {sorted(missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat code2wav ignoring %d unexpected codec tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32).eval()
        self.audio_codec = codec
        logger.info(
            "NemotronVoiceChat code2wav loaded RVQ-VAE codec: %d tensors, num_quantizers=%d, sr=%d",
            len(state),
            self._num_quantizers,
            self._sample_rate,
        )
        return {name for name, _ in self.named_parameters()}
