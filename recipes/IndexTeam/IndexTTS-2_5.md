# IndexTTS-2.5 for multilingual voice-cloned TTS on 1x GPU

## Summary

- Vendor: IndexTeam
- Model: `IndexTeam/IndexTTS-2.5`
- Task: Multilingual text-to-speech with voice cloning, emotion control, and
  native speed control
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API, or
  offline inference through `Omni`
- Maintainer: Community

## When to use this recipe

Use this recipe to serve IndexTTS-2.5 as a two-stage TTS system. Stage 0 is an
autoregressive talker; Stage 1 uses EnhancedCodec, S2Mel CFM/DiT, and BigVGAN
to produce 22.05 kHz mono speech. Each request supplies synthesis text and
reference audio, or an uploaded audio voice, for zero-shot voice cloning.

IndexTTS-2.5 adds multilingual text processing and model-native speed control
to the IndexTTS-2 serving contract. It is not an async-chunk streaming model:
Stage 1 starts after the complete semantic-code sequence is available.

## References

- Upstream model:
  [IndexTeam/IndexTTS-2.5 on Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-2.5)
- Online serving example:
  [`examples/online_serving/text_to_speech/README.md#indextts-2-and-indextts-25`](../../examples/online_serving/text_to_speech/README.md#indextts-2-and-indextts-25)
- Offline inference example:
  [`examples/offline_inference/text_to_speech/README.md#indextts-2-and-indextts-25`](../../examples/offline_inference/text_to_speech/README.md#indextts-2-and-indextts-25)
- OpenAI-compatible client:
  [`examples/online_serving/text_to_speech/indextts2/speech_client.py`](../../examples/online_serving/text_to_speech/indextts2/speech_client.py)
- Standard deploy config:
  [`vllm_omni/deploy/indextts2_5.yaml`](../../vllm_omni/deploy/indextts2_5.yaml)
- Experimental GPT-latent deploy config:
  [`vllm_omni/deploy/indextts2_5_latent.yaml`](../../vllm_omni/deploy/indextts2_5_latent.yaml)

## Hardware Support

### GPU

### 1x NVIDIA H20 96GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

Install vLLM-Omni with the IndexTTS text-processing dependencies:

```bash
pip install 'vllm-omni[indextts2]'
```

Obtain the native IndexTTS-2.5 bundle and point `MODEL` at its root. The model
loader accepts the upstream nested `checkpoints/` layout.

#### Command

Start the standard code-only server from the repository root:

```bash
MODEL_VERSION=2.5 \
MODEL=/path/to/indextts-2.5 \
bash examples/online_serving/text_to_speech/indextts2/run_server.sh
```

This selects `vllm_omni/deploy/indextts2_5.yaml`, which uses
`use_gpt_latent=false`. To launch directly:

```bash
vllm serve /path/to/indextts-2.5 \
  --omni \
  --trust-remote-code \
  --port 8092 \
  --deploy-config vllm_omni/deploy/indextts2_5.yaml
```

#### Verification

Send a multilingual voice-cloning request with the bundled client. Local audio
paths are converted to base64 data URLs before transmission:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --model-version 2.5 \
  --model /path/to/indextts-2.5 \
  --lang zh \
  --text "你好，这是 IndexTTS-2.5 语音合成测试。" \
  --ref-audio /path/to/reference.wav \
  --output indextts2_5.wav
```

The public `speed` field is handled natively. Its accepted range is
`[0.5, 2.0]`; values above `1.0` produce shorter, faster speech:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --model-version 2.5 \
  --model /path/to/indextts-2.5 \
  --speed 1.25 \
  --text "这是使用模型原生语速控制的测试。" \
  --ref-audio /path/to/reference.wav \
  --output indextts2_5_speed.wav
```

For raw HTTP requests, put language and text-normalization controls under
`extra_params`:

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/indextts-2.5",
    "input": "This is an IndexTTS-2.5 test.",
    "response_format": "wav",
    "speed": 1.0,
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_AUDIO>",
    "extra_params": {
      "lang": "en",
      "text_normalization": true
    }
  }' \
  --output indextts2_5_en.wav
```

Emotion controls are also passed in `extra_params`. Use one emotion source at a
time; for example:

```json
{
  "emo_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "emo_alpha": 0.8
}
```

For offline inference:

```bash
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
  --model /path/to/indextts-2.5 \
  --model-version 2.5 \
  --lang zh \
  --speed 1.0 \
  --text "你好，这是离线语音合成测试。" \
  --ref-audio /path/to/reference.wav
```

#### Notes

- Hardware scope: the standard recipe has been exercised on one NVIDIA H20.
  It does not claim an unverified throughput or quality result.
- Audio output: 22.05 kHz mono WAV.
- Voice cloning: reference audio is required on the documented raw request
  path. Alternatively, `voice` may name an uploaded audio voice; there is no
  built-in text-only preset voice.
- Native speed: serving maps `speed` to the model's duration factor and skips
  generic waveform speed adjustment, so speed is applied exactly once.
- Languages: common codes include `zh`, `en`, `zhen` (mixed Chinese/English),
  `ja`, and `yue`; `Mandarin` is accepted as an alias for `zh`.
- Japanese normalization: `ja` tokenization does not automatically expand
  numbers, dates, or percentages. Write these inputs as readable Japanese text
  before inference.
- Emotion controls: `use_emo_text`, `emo_vector`, and `emo_audio` are
  alternative conditioning modes. Their precedence is `use_emo_text` >
  `emo_vector` > `emo_audio` > the speaker-reference emotion.
- Sampling difference: Stage 0 uses plain vLLM sampling, not the upstream
  default `num_beams=3` beam search. Use upstream `num_beams=1` for parity
  comparisons; output may differ from the official beam-search result.
- Streaming boundary: `stream=true` is accepted as an HTTP response mode, but
  `async_chunk=false`; audio is emitted after the full semantic sequence has
  reached S2Mel.
- Experimental variant: `indextts2_5_latent.yaml` enables a vLLM-Omni-specific
  GPT-latent path with nearest-neighbor alignment. It has no official runnable
  reference output and should not be used as the standard parity baseline.
