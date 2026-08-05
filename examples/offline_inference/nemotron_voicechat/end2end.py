# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline speech-to-speech with NVIDIA-NemotronLabs-VoiceChat-11B.

Feeds a 16 kHz user utterance (plus a spoken-style system prompt) through the
3-stage nemotron_voicechat pipeline and writes the agent's reply as text and a
22.05 kHz WAV.

Usage:
    python end2end.py \
        --checkpoint /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
        --wav /path/to/user_question_16k.wav \
        --output-dir results/nemotron_voicechat

The system prompt defaults to the NeMo reference default. The text tokenizer
resolves from the ``nvidia/NVIDIA-Nemotron-Nano-9B-v2`` HF id (or the
``NEMOTRON_VOICECHAT_LLM_PATH`` env var for air-gapped runs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# The NeMo reference offline script's default system prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


def load_wav_16k_mono(path: str) -> np.ndarray:
    import librosa

    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="NVIDIA-NemotronLabs-VoiceChat-11B directory")
    parser.add_argument("--wav", required=True, help="user speech (any sr; resampled to 16 kHz mono)")
    parser.add_argument("--output-dir", default="results/nemotron_voicechat")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    if not Path(args.wav).is_file():
        raise FileNotFoundError(f"Input WAV not found: {args.wav}")
    ckpt = Path(args.checkpoint)
    raw_cfg = json.loads((ckpt / "config.json").read_text())
    stt_cfg = raw_cfg["model"]["stt"]["model"]

    from transformers import AutoTokenizer
    from vllm.sampling_params import SamplingParams

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        compute_acoustic_frame_count,
    )

    tok_ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or stt_cfg.get(
        "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=True)
    bos_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("bos_token", "<s>"))
    eos_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("eos_token", "</s>"))
    pad_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("pad_token", "<SPECIAL_12>"))
    # NeMo encode_system_prompt: [bos] + tokens + [eos].
    prompt_ids = [bos_id] + tokenizer.encode(args.system_prompt, add_special_tokens=False) + [eos_id]

    wav = load_wav_16k_mono(args.wav)
    n_frames = compute_acoustic_frame_count(stt_cfg, int(wav.shape[0]))
    print(f"input: {wav.shape[0]} samples @16kHz -> {n_frames} acoustic frames; prompt {len(prompt_ids)} tokens")

    inputs = {
        # vLLM prompt = system prompt + one placeholder position (acoustic
        # frame 0); the thinker replaces every position with fused embeddings.
        "prompt_token_ids": prompt_ids + [pad_id],
        "additional_information": {
            # Plain float list: raw ndarrays do not survive the engine-core
            # message serialization.
            "nvc_audio": wav.astype(np.float32).tolist(),
            "nvc_sr": 16000,
            "nvc_prompt_token_ids": prompt_ids,
            "nvc_expected_frames": n_frames,
        },
    }
    sampling_params_list = [
        # thinker: greedy, frame-locked generation length.
        SamplingParams(temperature=0.0, min_tokens=n_frames, max_tokens=n_frames, ignore_eos=True, seed=0),
        # talker: placeholder loop, stops on the stage's stop token.
        SamplingParams(temperature=0.0, max_tokens=16384, stop_token_ids=[1], detokenize=False, seed=0),
        # code2wav: single decode step.
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False, seed=0),
    ]

    omni = Omni(model=str(ckpt), trust_remote_code=True)
    outputs = omni.generate(inputs, sampling_params_list)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.wav).stem

    agent_text = None
    audio = None
    sr = 22050
    for request_output in outputs:
        for stage_output in request_output.outputs if hasattr(request_output, "outputs") else []:
            text = getattr(stage_output, "text", None)
            if text:
                agent_text = text
        mm = getattr(request_output, "multimodal_output", None) or {}
        model_outputs = mm.get("model_outputs") if isinstance(mm, dict) else None
        if model_outputs is not None:
            candidate = model_outputs[0] if isinstance(model_outputs, list) else model_outputs
            arr = np.asarray(candidate.float().cpu() if hasattr(candidate, "cpu") else candidate, dtype=np.float32)
            if arr.size > 0:
                audio = arr.reshape(-1)
            sr_val = mm.get("sr")
            if sr_val is not None:
                sr_item = sr_val[0] if isinstance(sr_val, list) else sr_val
                sr = int(sr_item.item() if hasattr(sr_item, "item") else sr_item)

    if agent_text is not None:
        # The frame-locked text channel emits PAD tokens on silent frames;
        # strip them (NeMo's post-inference detokenizer does the same).
        pad_token = stt_cfg.get("pad_token", "<SPECIAL_12>")
        agent_text = " ".join(agent_text.replace(pad_token, " ").split())
        text_path = out_dir / f"{stem}_output.txt"
        text_path.write_text(agent_text + "\n")
        print(f"agent text -> {text_path}\n{agent_text}")
    else:
        print("WARNING: no agent text in outputs")

    if audio is not None and audio.size > 0:
        wav_path = out_dir / f"{stem}_output.wav"
        sf.write(wav_path, audio, sr)
        print(f"agent audio -> {wav_path} ({audio.size / sr:.2f}s @ {sr} Hz)")
    else:
        raise RuntimeError("No audio produced by the code2wav stage.")


if __name__ == "__main__":
    main()
