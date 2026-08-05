# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump NeMo reference intermediates for NemotronVoiceChat parity debugging.

Runs the official NeMo offline inference on one WAV and saves the agent text,
the frame-locked text-token timeline, the function-channel timeline, the
31-quantizer code stacks, and the decoded waveform. These dumps anchor the
vLLM-Omni golden shape test and stage-by-stage debugging.

REQUIRES the NeMo-Speech environment (branch nemotron-labs-voicechat), NOT the
vLLM-Omni venv:

    /path/to/voicechat_venv/bin/python dump_nemo_reference.py \
        --checkpoint /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
        --wav sample_general.wav --output-dir results/nemo_reference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--output-dir", default="results/nemo_reference")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an AI voice assistant developed by NVIDIA. "
            "Your name is NVIDIA Voice Chat. "
            "Answer in a spoken, conversational style rather than a written one. "
            "Do not repeat the same sentence over and over again. "
            "Start the conversation by greeting the user."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
        build_model,
        encode_system_prompt,
        load_wav_16k_mono,
    )

    model = build_model(args.checkpoint, device=args.device)

    _, input_signal, input_signal_lens = load_wav_16k_mono(args.wav, device=args.device)
    prompt_tokens, prompt_token_lens = encode_system_prompt(model, args.system_prompt, device=args.device)

    with torch.no_grad():
        result = model.offline_inference(
            input_signal=input_signal,
            input_signal_lens=input_signal_lens,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
            decode_audio=True,
            temperature=0.0,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.wav).stem
    dump = {
        "text": result.get("text"),
        "tokens_text": result.get("tokens_text"),
        "tokens_len": result.get("tokens_len"),
        "tokens_audio": result.get("tokens_audio"),
        "tokens_function": result.get("tokens_function", result.get("tokens_function_pred")),
        "prompt_tokens": prompt_tokens,
        "prompt_token_lens": prompt_token_lens,
        "audio_len": result.get("audio_len"),
    }
    dump = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in dump.items()}
    torch.save(dump, out_dir / f"{stem}_reference.pt")

    (out_dir / f"{stem}_reference.txt").write_text((result.get("text") or [""])[0] + "\n")
    audio = result.get("audio")
    if isinstance(audio, torch.Tensor):
        import soundfile as sf

        wav = audio[0].float().cpu().numpy()
        n = int(result["audio_len"][0]) if "audio_len" in result else wav.shape[-1]
        sf.write(out_dir / f"{stem}_reference.wav", wav[:n], 22050)
    print(f"reference dumps -> {out_dir}/{stem}_reference.{{pt,txt,wav}}")
    print("text:", (result.get("text") or [""])[0])
    for key in ("tokens_text", "tokens_audio", "tokens_function"):
        val = dump.get(key)
        if isinstance(val, torch.Tensor):
            print(f"{key}: shape {tuple(val.shape)}")


if __name__ == "__main__":
    main()
