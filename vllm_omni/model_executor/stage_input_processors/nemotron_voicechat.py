# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage transfer processors for the NemotronVoiceChat 3-stage pipeline.

thinker (0) -> talker (1): token path only. The thinker's frame-locked text
timeline becomes the talker's ``prompt_token_ids``:
``[text_pad_id] * logical_prompt_token_len + generated frame tokens``. The
talker must process the PAD prompt region too (its KV state at the first
acoustic frame depends on it — NeMo trims prompt-region codes only after the
loop), so the prompt slots are reconstructed here, and the timeline metadata
rides ``additional_information``.

talker (1) -> code2wav (2): PersonaPlex-style full payload. The talker
accumulates one 31-quantizer code stack per frame under ``("codes","audio")``;
the producer ships the prompt-trimmed ``[frames, 31]`` stack once the request
finishes, and the sync placeholder builder sizes stage-2's dummy prompt.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)

# Full-sequence tensors are re-shipped whole; the full-payload accumulator must
# replace (not concatenate) them if the producer fires more than once.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset({"codes"})

# <SPECIAL_12> in the Nemotron-Nano-9B-v2 vocab; the checkpoint config's
# stt pad_token. The thinker also reports it in its latent metadata, which
# takes precedence when present.
_DEFAULT_TEXT_PAD_ID = 12


def _info_get(info: Any, key: str) -> Any:
    if isinstance(info, dict):
        if key in info:
            return info[key]
        meta = info.get("meta")
        if isinstance(meta, dict):
            return meta.get(key)
    return None


def thinker2talker_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Build the talker's engine inputs from the thinker's finished outputs.

    ``prompt_token_ids`` = the full frame-aligned text timeline including the
    PAD prompt region. The thinker's vLLM prompt covers
    ``logical_prompt_token_len + 1`` positions (the +1 is acoustic frame 0), so
    ``logical_prompt_token_len = len(thinker prompt) - 1`` and the generated
    tokens are exactly the acoustic-frame text channel.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for thinker_output in source_outputs:
        if not getattr(thinker_output, "finished", False):
            continue
        output = thinker_output.outputs[0]
        generated = list(getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or [])
        prompt_ids = list(getattr(thinker_output, "prompt_token_ids", None) or [])
        logical_prompt_len = max(len(prompt_ids) - 1, 0)
        pad_id = _DEFAULT_TEXT_PAD_ID
        info = getattr(thinker_output, "additional_information", None)
        reported_pad = _info_get(info, "nvc_text_pad_id")
        if reported_pad is not None:
            pad_id = int(reported_pad)
        timeline = [pad_id] * logical_prompt_len + [int(t) for t in generated]
        inputs.append(
            OmniTokensPrompt(
                # The vLLM-side prompt is one placeholder (timeline position 0;
                # the NeMo TTS loop starts at t=1). Each decode step consumes
                # one timeline position from additional_information, so the
                # talker needs len(timeline) - 1 decode steps.
                prompt_token_ids=[0],
                additional_information={
                    "nvc_text_timeline": timeline,
                    "nvc_logical_prompt_len": logical_prompt_len,
                    "nvc_frame_count": len(generated),
                    "nvc_text_pad_id": pad_id,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def _empty_finished_payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    pooling_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct | None:
    """Producer: ship the talker's prompt-trimmed ``[frames, 31]`` code stack.

    Codes accumulate per frame under ``("codes","audio")`` in the request's
    additional_information; the prompt-region rows were never emitted by the
    talker (it only stores codes for generated positions), so no extra trim is
    needed here. Ships once, at request finish, with replace semantics.
    """
    # NOTE: no is_finished gating — in sync full-payload mode the producer only
    # runs at flush time (request already finished), and the flush-time request
    # object may still report is_finished()=False. Returning a meta-only
    # "pending" struct here would be SENT as the real payload and starve the
    # code2wav stage of its codes.
    del transfer_manager, is_finished

    def _codes_from(src: Any) -> torch.Tensor | None:
        if not isinstance(src, dict):
            return None
        nested = src.get("codes")
        audio = nested.get("audio") if isinstance(nested, dict) else None
        return audio if audio is not None else src.get("codes.audio")

    audio = None
    for source in (
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
        pooling_output,
        kwargs.get("multimodal_output"),
    ):
        audio = _codes_from(source)
        if audio is not None:
            break
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        import logging

        logging.getLogger(__name__).error(
            "nemotron_voicechat talker full-payload producer found no codes: "
            "pooling_output keys=%s, request.additional_information keys=%s",
            sorted(pooling_output.keys()) if isinstance(pooling_output, dict) else type(pooling_output).__name__,
            sorted(getattr(request, "additional_information", None) or {})
            if isinstance(getattr(request, "additional_information", None), dict)
            else type(getattr(request, "additional_information", None)).__name__,
        )
        return _empty_finished_payload()
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    # Rows correspond to NeMo timeline steps t = 1..T-1. Trim the prompt
    # region: NeMo keeps gen_codes rows t = P..T-1 (P = logical prompt len),
    # i.e. row indices P-1 onward here. P >= 1 always (a system prompt is
    # required), so this never drops acoustic frames.
    prompt_len = _info_get(getattr(request, "additional_information", None), "nvc_logical_prompt_len")
    if prompt_len is None:
        prompt_len = _info_get(getattr(request, "additional_information_cpu", None), "nvc_logical_prompt_len")
    trim = max(int(prompt_len) - 1, 0) if prompt_len is not None else 0
    audio = audio[trim:]
    if audio.shape[0] == 0:
        return _empty_finished_payload()
    # 2-D [frames, Q] payload: the chunk/full-payload transfer adapters route
    # >=2-D tensors through the payload channel (1-D would be coerced into
    # prompt token ids and dropped from the info merge).
    return OmniPayloadStruct(
        codes=CodesStruct(audio=audio.detach().to(dtype=torch.long, device="cpu")),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync placeholder builder for stage 2 (codes arrive via the connector)."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = len(token_ids)
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * max(n_frames, 1),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


__all__ = [
    "thinker2talker_token_only",
    "talker2code2wav_full_payload",
    "talker2code2wav_token_only",
]
