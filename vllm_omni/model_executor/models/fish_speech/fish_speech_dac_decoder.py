"""Fish Speech S2 Pro -- DAC Decoder (Stage 1).

Loads the DAC codec from ``codec.pth`` and decodes codebook indices
[num_codebooks, T] → audio waveform at 44.1 kHz.

Analogous to ``Qwen3TTSCode2Wav`` in qwen3_tts.

Requires the ``fish-speech`` package for the DAC model architecture.
Install with: ``pip install fish-speech``
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.fish_speech.dac_utils import (
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_SAMPLE_RATE,
    build_dac_codec,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class FishSpeechDACDecoder(nn.Module):
    """Stage-1 DAC decoder for Fish Speech S2 Pro (GenerationModelRunner).

    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the DAC codec decoder.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._codec: nn.Module | None = None
        self._num_codebooks: int = DAC_NUM_CODEBOOKS
        self._output_sample_rate: int = DAC_SAMPLE_RATE
        self._hop_length: int = DAC_HOP_LENGTH
        self._logged_codec_stats = False
        self._dac_warmup_done = False
        self._profile_decode_calls = 0
        self._profile_decode_ms = 0.0

    def _bake_weight_norm(self, codec: nn.Module) -> None:
        baked = 0
        for module in codec.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if not parametrizations:
                continue
            for name in list(parametrizations.keys()):
                remove_parametrizations(module, name, leave_parametrized=True)
                baked += 1
        if baked > 0:
            logger.info("Baked %d DAC parametrized weights for inference", baked)

    @staticmethod
    def _cache_attention_masks(codec: nn.Module, device: torch.device | None = None) -> None:
        for module in codec.modules():
            if not hasattr(module, "make_mask") or not hasattr(module, "make_window_limited_mask"):
                continue

            base_make_mask = module.make_mask
            base_make_window_mask = module.make_window_limited_mask
            mask_cache: dict[int, torch.Tensor] = {}
            window_mask_cache: dict[int, torch.Tensor] = {}

            def make_mask_cached(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_mask,
                _device=device,
            ):
                if x_lens is not None:
                    result = _orig(max_length, x_lens)
                    return result.to(device=_device, non_blocking=True) if _device else result
                key = int(max_length)
                cached = mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    if _device is not None:
                        cached = cached.to(device=_device)
                    mask_cache[key] = cached
                return cached

            def make_window_mask_cached(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_window_mask,
                _device=device,
            ):
                if x_lens is not None:
                    result = _orig(max_length, x_lens)
                    return result.to(device=_device, non_blocking=True) if _device else result
                key = int(max_length)
                cached = window_mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    if _device is not None:
                        cached = cached.to(device=_device)
                    window_mask_cache[key] = cached
                return cached

            module.make_mask = make_mask_cached
            module.make_window_limited_mask = make_window_mask_cached

    def _ensure_codec_loaded(self) -> None:
        if self._codec is not None:
            return

        codec_path = os.path.join(self.model_path, "codec.pth")
        if not os.path.exists(codec_path):
            # Try HuggingFace cache.
            try:
                from transformers.utils.hub import cached_file

                cached = cached_file(self.model_path, "codec.pth")
                if cached is not None:
                    codec_path = cached
            except Exception:
                pass

        if not os.path.exists(codec_path):
            raise FileNotFoundError(
                f"codec.pth not found at {codec_path}. Make sure the Fish Speech S2 Pro model includes codec.pth."
            )

        codec = build_dac_codec()

        # Load weights.
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        # Some checkpoints wrap under "generator" key.
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)
        self._bake_weight_norm(codec)

        # Decode path only uses quantizer.decode() + decoder; prune
        # encode-only components before moving to device to avoid
        # unnecessary GPU allocation.
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

        # Fix numpy scalars that break torch.compile/Dynamo guard checks.
        if hasattr(codec, "hop_length"):
            codec.hop_length = int(codec.hop_length)
        if hasattr(codec, "frame_length"):
            codec.frame_length = int(codec.frame_length)

        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32)
        codec.eval()

        # Enable TF32 for float32 matmul/conv: ~2x throughput on Ampere+.
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Cache attention masks AFTER .to(device) so cached masks land on
        # the correct device (Triton kernels reject CPU tensors).
        self._cache_attention_masks(codec, device)

        dac_bucket_spec = os.environ.get(
            "VLLM_FISH_DAC_BUCKET_FRAMES",
            "4,50" if os.environ.get("VLLM_FISH_DAC_COMPILE", "0") == "1" else "",
        ).strip()

        # Compile decode to fuse conv kernels and reduce dispatch overhead.
        # Disabled by default: dynamic shapes cause costly recompilation
        # (~30s per new frame count) which dominates latency in production.
        # Set VLLM_FISH_DAC_COMPILE=1 to enable (useful for fixed-length only).
        if os.environ.get("VLLM_FISH_DAC_COMPILE", "0") == "1":
            try:
                import torch._dynamo as _dynamo

                _dynamo.config.recompile_limit = max(256, _dynamo.config.recompile_limit)
                codec.decode = torch.compile(
                    codec.decode,
                    mode="default",
                    dynamic=not bool(dac_bucket_spec),
                    fullgraph=False,
                )
                logger.info(
                    "Enabled torch.compile on DAC codec.decode (dynamic=%s, buckets=%r, recompile_limit=%d)",
                    not bool(dac_bucket_spec),
                    dac_bucket_spec or None,
                    _dynamo.config.recompile_limit,
                )
            except Exception as exc:
                logger.warning("torch.compile on DAC codec.decode failed: %s", exc)
        else:
            logger.info("DAC codec.decode torch.compile disabled (set VLLM_FISH_DAC_COMPILE=1 to enable)")

        self._codec = codec

        if os.environ.get("VLLM_FISH_DAC_WARMUP_BUCKETS", "0") == "1":
            self._warmup_decode_buckets()

        logger.info(
            "Fish Speech DAC codec loaded from %s (device=%s, sample_rate=%d)",
            codec_path,
            device,
            self._output_sample_rate,
        )

    def _parse_bucket_frames(self) -> list[int]:
        bucket_spec = os.environ.get(
            "VLLM_FISH_DAC_BUCKET_FRAMES",
            "4,50" if os.environ.get("VLLM_FISH_DAC_COMPILE", "0") == "1" else "",
        ).strip()
        if not bucket_spec:
            return []
        try:
            return sorted(
                int(part.strip())
                for part in bucket_spec.split(",")
                if part.strip()
            )
        except ValueError:
            logger.warning_once(
                "Ignoring invalid VLLM_FISH_DAC_BUCKET_FRAMES=%r",
                bucket_spec,
            )
            return []

    def _bucket_for_frames(self, frames: int, buckets: list[int]) -> int:
        for bucket in buckets:
            if frames <= bucket:
                return bucket
        return frames

    @torch.no_grad()
    def _warmup_decode_buckets(self) -> None:
        if self._dac_warmup_done or self._codec is None:
            return
        self._dac_warmup_done = True
        buckets = self._parse_bucket_frames() or [4, 25, 50]
        batch_spec = os.environ.get("VLLM_FISH_DAC_WARMUP_BATCH_SIZES", "1,2,4")
        try:
            batch_sizes = [
                int(part.strip())
                for part in batch_spec.split(",")
                if part.strip()
            ]
        except ValueError:
            logger.warning_once(
                "Ignoring invalid VLLM_FISH_DAC_WARMUP_BATCH_SIZES=%r; using 1,2,4",
                batch_spec,
            )
            batch_sizes = [1, 2, 4]
        device = self.vllm_config.device_config.device
        logger.info(
            "Warming DAC decode buckets: frames=%s batch_sizes=%s",
            buckets,
            batch_sizes,
        )
        for frames in buckets:
            for batch_size in batch_sizes:
                codes = torch.ones(
                    (batch_size, self._num_codebooks, frames),
                    device=device,
                    dtype=torch.long,
                )
                lengths = torch.full(
                    (batch_size,),
                    frames,
                    device=device,
                    dtype=torch.long,
                )
                with torch.amp.autocast("cuda", enabled=False):
                    self._codec.decode(codes, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info("Finished DAC decode bucket warmup")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

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
        """Decode codec codes into audio waveform.

        input_ids layout per request: flat codes [num_codebooks * num_frames].
        Codes are codebook-major: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ...].
        """
        self._ensure_codec_loaded()
        assert self._codec is not None

        q = self._num_codebooks
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        burst_output = self._try_decode_runtime_code_chunks(
            runtime_additional_information,
            input_ids.device if input_ids is not None else self.vllm_config.device_config.device,
            q,
            sr_tensor,
            empty,
        )
        if burst_output is not None:
            return burst_output

        runtime_code_ids: list[torch.Tensor | None] = []
        has_runtime_codes = False
        if runtime_additional_information is not None:
            device = input_ids.device if input_ids is not None else self.vllm_config.device_config.device
            for info in runtime_additional_information:
                codes = info.get("code_predictor_codes") if isinstance(info, dict) else None
                if codes is None:
                    runtime_code_ids.append(None)
                    continue
                if isinstance(codes, torch.Tensor):
                    flat_codes = codes.reshape(-1).to(device=device, dtype=torch.long, non_blocking=True)
                else:
                    flat_codes = torch.as_tensor(codes, device=device, dtype=torch.long).reshape(-1)
                next_len = info.get("next_stage_prompt_len") if isinstance(info, dict) else None
                if isinstance(next_len, int) and 0 < next_len <= flat_codes.numel():
                    flat_codes = flat_codes[:next_len]
                has_runtime_codes = has_runtime_codes or flat_codes.numel() > 0
                runtime_code_ids.append(flat_codes)

        if not has_runtime_codes and (input_ids is None or input_ids.numel() == 0):
            # Output length must match the number of scheduled requests so the
            # generation runner can unpack one multimodal output per request.
            num_req = max(len(runtime_code_ids), 1)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if has_runtime_codes:
            request_ids_list = [
                codes if codes is not None else empty.to(dtype=torch.long)
                for codes in runtime_code_ids
            ]
        else:
            assert input_ids is not None
            ids = input_ids.reshape(-1).to(dtype=torch.long)
            request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        num_req = len(request_ids_list)
        parsed_ctx_frames = [0] * num_req
        parsed_total_frames = [0] * num_req
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if "left_context_size" in info:
                    left_context_size[i] = info["left_context_size"]

        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "DAC decoder input_ids length %d not divisible by num_codebooks %d; returning empty audio.",
                        n,
                        q,
                    )
                continue
            frames = n // q
            codes_qf = flat.reshape(q, frames)
            parsed_ctx_frames[i] = ctx_frames
            parsed_total_frames[i] = frames
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "DAC decoder: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        if os.environ.get("VLLM_FISH_DAC_FAKE_DECODE", "0") == "1":
            device = valid_codes_qf[0].device
            for local_idx, codes_qf in enumerate(valid_codes_qf):
                idx = valid_indices[local_idx]
                ctx_frames = parsed_ctx_frames[idx]
                total_frames = parsed_total_frames[idx]
                audio_frames = max(int(total_frames) - int(ctx_frames), 0)
                audio_len = audio_frames * self._hop_length
                if audio_len > 0:
                    audios[idx] = torch.zeros(
                        (audio_len,),
                        device=device,
                        dtype=torch.float32,
                    )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        buckets = self._parse_bucket_frames()
        grouped: dict[int, list[int]] = {}
        for local_idx, codes_qf in enumerate(valid_codes_qf):
            frame_capacity = self._bucket_for_frames(int(codes_qf.shape[1]), buckets)
            grouped.setdefault(frame_capacity, []).append(local_idx)

        if len(grouped) > 1:
            logger.debug(
                "DAC decoder split mixed frame batch into buckets: %s",
                {k: len(v) for k, v in grouped.items()},
            )

        for frame_capacity, local_indices in grouped.items():
            feature_lengths = torch.tensor(
                [valid_codes_qf[i].shape[1] for i in local_indices],
                device=valid_codes_qf[0].device,
                dtype=torch.long,
            )
            codes_bqf = torch.zeros(
                (len(local_indices), q, frame_capacity),
                device=valid_codes_qf[0].device,
                dtype=torch.long,
            )
            for row, local_idx in enumerate(local_indices):
                codes_qf = valid_codes_qf[local_idx]
                frame_count = int(codes_qf.shape[1])
                codes_bqf[row, :, :frame_count] = codes_qf

            profile_decode = os.environ.get("VLLM_FISH_DAC_PROFILE_DECODE", "0") == "1"
            if profile_decode and codes_bqf.device.type == "cuda":
                torch.cuda.synchronize(codes_bqf.device)
            profile_start = time.perf_counter() if profile_decode else 0.0
            with torch.amp.autocast("cuda", enabled=False):
                wav_batch, audio_lengths = self._codec.decode(codes_bqf, feature_lengths)
            if profile_decode:
                if codes_bqf.device.type == "cuda":
                    torch.cuda.synchronize(codes_bqf.device)
                elapsed_ms = (time.perf_counter() - profile_start) * 1000.0
                self._profile_decode_calls += 1
                self._profile_decode_ms += elapsed_ms
                if self._profile_decode_calls <= 10 or self._profile_decode_calls % 100 == 0:
                    avg_ms = self._profile_decode_ms / self._profile_decode_calls
                    logger.info(
                        "DAC decode profile: calls=%d last_ms=%.3f avg_ms=%.3f batch=%d bucket_frames=%d lengths=%s",
                        self._profile_decode_calls,
                        elapsed_ms,
                        avg_ms,
                        codes_bqf.shape[0],
                        frame_capacity,
                        [int(x) for x in feature_lengths.detach().cpu().tolist()],
                    )

            for row, local_idx in enumerate(local_indices):
                idx = valid_indices[local_idx]
                ctx_frames = parsed_ctx_frames[idx]
                total_frames = parsed_total_frames[idx]
                audio_len = int(audio_lengths[row].item()) if audio_lengths.numel() > row else int(wav_batch.shape[-1])
                wav = wav_batch[row, 0, :audio_len]
                # Trim context frames (left overlap for streaming).
                if ctx_frames > 0:
                    # Decode length may deviate from (frames * hop_length) due to model
                    # internals (padding/rounding). Use proportional trimming to keep
                    # overlap removal aligned with the actual decoded length.
                    denom = max(int(total_frames), 1)
                    cut = int(ctx_frames / denom * wav.shape[0])
                    cut = max(0, min(cut, int(wav.shape[0])))
                    if cut < wav.shape[0]:
                        wav = wav[cut:]
                    else:
                        logger.warning(
                            "Context trim %d >= decoded length %d; returning empty audio.",
                            cut,
                            wav.shape[0],
                        )
                        continue
                if wav.shape[0] > 0:
                    audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    @torch.no_grad()
    def _try_decode_runtime_code_chunks(
        self,
        runtime_additional_information: list[dict[str, Any]] | None,
        device: torch.device,
        q: int,
        sr_tensor: torch.Tensor,
        empty: torch.Tensor,
    ) -> OmniOutput | None:
        """Decode multiple ready streaming chunks per request in one forward."""
        if runtime_additional_information is None:
            return None

        per_request_chunks: list[list[tuple[torch.Tensor, int]]] = []
        has_burst = False
        for info in runtime_additional_information:
            if not isinstance(info, dict):
                per_request_chunks.append([])
                continue
            chunks = info.get("code_predictor_chunks")
            if not chunks:
                per_request_chunks.append([])
                continue
            has_burst = True
            left_contexts = info.get("left_context_sizes", [])
            prompt_lens = info.get("next_stage_prompt_lens", [])
            parsed_chunks: list[tuple[torch.Tensor, int]] = []
            for idx, codes in enumerate(chunks):
                if isinstance(codes, torch.Tensor):
                    flat_codes = codes.reshape(-1).to(device=device, dtype=torch.long, non_blocking=True)
                else:
                    flat_codes = torch.as_tensor(codes, device=device, dtype=torch.long).reshape(-1)
                next_len = (
                    prompt_lens[idx]
                    if isinstance(prompt_lens, list) and idx < len(prompt_lens)
                    else flat_codes.numel()
                )
                if isinstance(next_len, int) and 0 < next_len <= flat_codes.numel():
                    flat_codes = flat_codes[:next_len]
                if flat_codes.numel() > 0:
                    left_context = (
                        left_contexts[idx]
                        if isinstance(left_contexts, list) and idx < len(left_contexts)
                        else 0
                    )
                    parsed_chunks.append((flat_codes, int(left_context or 0)))
            per_request_chunks.append(parsed_chunks)

        if not has_burst:
            return None

        num_req = len(per_request_chunks)
        audios_by_req: list[list[torch.Tensor]] = [[] for _ in range(num_req)]
        srs = [sr_tensor] * num_req
        jobs: list[tuple[int, torch.Tensor, int, int]] = []
        for req_idx, chunks in enumerate(per_request_chunks):
            for flat, ctx_frames in chunks:
                n = flat.numel()
                if n == 0 or n % q != 0:
                    if n > 0:
                        logger.warning(
                            "DAC decoder burst input_ids length %d not divisible by num_codebooks %d; skipping.",
                            n,
                            q,
                        )
                    continue
                frames = n // q
                jobs.append((req_idx, flat.reshape(q, frames), ctx_frames, frames))

        if not jobs:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": srs,
                },
            )

        if os.environ.get("VLLM_FISH_DAC_FAKE_DECODE", "0") == "1":
            for req_idx, _codes_qf, ctx_frames, total_frames in jobs:
                audio_frames = max(int(total_frames) - int(ctx_frames), 0)
                audio_len = audio_frames * self._hop_length
                if audio_len > 0:
                    audios_by_req[req_idx].append(
                        torch.zeros((audio_len,), device=device, dtype=torch.float32)
                    )
            audios = [
                torch.cat(chunks).contiguous() if chunks else empty
                for chunks in audios_by_req
            ]
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        buckets = self._parse_bucket_frames()
        grouped: dict[int, list[int]] = {}
        for job_idx, (_req_idx, codes_qf, _ctx_frames, _total_frames) in enumerate(jobs):
            frame_capacity = self._bucket_for_frames(int(codes_qf.shape[1]), buckets)
            grouped.setdefault(frame_capacity, []).append(job_idx)

        for frame_capacity, job_indices in grouped.items():
            feature_lengths = torch.tensor(
                [jobs[i][1].shape[1] for i in job_indices],
                device=device,
                dtype=torch.long,
            )
            codes_bqf = torch.zeros(
                (len(job_indices), q, frame_capacity),
                device=device,
                dtype=torch.long,
            )
            for row, job_idx in enumerate(job_indices):
                codes_qf = jobs[job_idx][1]
                frame_count = int(codes_qf.shape[1])
                codes_bqf[row, :, :frame_count] = codes_qf

            with torch.amp.autocast("cuda", enabled=False):
                wav_batch, audio_lengths = self._codec.decode(codes_bqf, feature_lengths)

            for row, job_idx in enumerate(job_indices):
                req_idx, _codes_qf, ctx_frames, total_frames = jobs[job_idx]
                audio_len = int(audio_lengths[row].item()) if audio_lengths.numel() > row else int(wav_batch.shape[-1])
                wav = wav_batch[row, 0, :audio_len]
                if ctx_frames > 0:
                    denom = max(int(total_frames), 1)
                    cut = int(ctx_frames / denom * wav.shape[0])
                    cut = max(0, min(cut, int(wav.shape[0])))
                    if cut < wav.shape[0]:
                        wav = wav[cut:]
                    else:
                        continue
                if wav.shape[0] > 0:
                    audios_by_req[req_idx].append(wav.to(dtype=torch.float32).reshape(-1))

        audios = [
            torch.cat(chunks).contiguous() if chunks else empty
            for chunks in audios_by_req
        ]
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"FishSpeechDACDecoder expected (audio_tensor, sr), got {type(model_outputs)}")
        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # DAC codec weights are loaded lazily from codec.pth, not from the main checkpoint.
        return set()
