# Ming-3.0-TTS Tiny

Online speech synthesis through the OpenAI-compatible `/v1/audio/speech` API.

## Model and compatibility

| HF model | Backbone | Omni pipeline | Reference conditioning |
| --- | --- | --- | --- |
| [Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) | Ling V3 MoE | `ming3_tts` | Speaker embedding, prompt latents, multi-speaker Podcast |

The public Tiny checkpoint retains `model_type=lingv3_tts`, architecture
`LingV3TTSForConditionalGeneration`, and task contract
`lingv3_sft_multitask_v1`. These serialized names remain supported aliases;
Omni modules, deployment configuration, and request controls use `ming3_tts`.

The model card describes the vendor's native CLI and Python APIs. The commands
below are its vLLM-Omni serving equivalents; the native environment pins do
not replace vLLM's matching runtime dependencies. See the
[HF validation record](Ming-3.0-TTS-HF-validation.md) for exact revisions,
commands, results, and limitations.

## Deployment

Validated with `.venv`, Python 3.12, vLLM 0.32.0, Torch 2.13.0+cu130, and H200.
Tiny uses a CUDA/BF16 AR/Flow stage followed by a streaming AudioVAE.
Its HF ID selects the pipeline automatically; remote code is not required.
Tiny's default context limit is 4096 tokens.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve inclusionAI/Ming-3.0-tts-tiny \
  --omni --port 8000
```

Use `--revision <HF-commit>` to pin a release and `--download-dir <cache>` to
select the HF weight cache. For `file://` references, add
`--allowed-local-media-path /absolute/path/to/references` at launch.

## Voice cloning from the model card

The public reference clip has transcript `在此奉劝大家别乱打美白针。`.
The `seed` task accepts a reference waveform and its transcript.
Explicit Flow temperature zero follows the model card's generation example;
omitting it uses the checkpoint default, currently 1.5.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "傍晚的风从山口吹过来，远处的云慢慢染成了金色。",
    "ref_audio": "https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs/10002287-00000094.wav",
    "ref_text": "在此奉劝大家别乱打美白针。",
    "language": "Chinese",
    "response_format": "wav",
    "max_new_tokens": 200,
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1234
  }' --output ming3_tiny_clone.wav
```

Omit `ref_text` for speaker-only conditioning. Use the checkpoint tokenizer;
reference transcripts must describe the supplied waveform exactly. For
single-speaker synthesis, `language` adds the target-language prompt line.

## Instructions and voice design

Short structured controls use Chinese keys: `情感`, `语速`,
`基频`, `音量`, `方言`, `风格`, and `IP`. Freeform instructions are strings.
Full single-speaker `audio_sequence` JSON captions retain their supplied field
order. Tiny preserves the release's task-specific structured-caption order.

- Emotion: reference audio plus `instructions: "{\"情感\":\"高兴\"}"`.
- Freeform: reference audio plus a natural-language `instructions` string.
- Voice design: omit references and provide
  `instructions: "{\"风格\":\"年轻女声，声音清晰自然。\"}"`.
- IP control: use `instructions: "{\"IP\":\"<vendor voice name>\"}"`.
  Tiny does not expose built-in IP names through the uploaded `voice` field.

Do not add reference transcripts to speaker-only control tasks. The Tiny model
card's dialect example maps to:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "今日天气几好，我哋食完饭一齐去海边行下啦。",
    "ref_audio": "https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs/10002287-00000094.wav",
    "instructions": "{\"方言\":\"广粤话\"}",
    "response_format": "wav",
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1234
  }' --output ming3_tiny_dialect.wav
```

## Podcast: Tiny

Provide an ordered reference clip and accurate transcript for each speaker.
Each input line starts with `speaker_N:` and uses one of those speakers.
Tiny infers the target language from the dialogue when `language` is omitted.
Podcast does not accept `instructions`; the checkpoint must declare
`tts_config.task_decode_alignment.podcast=old_shift`.

For a server allowing `/absolute/path/to/references`:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "speaker_1:你好，今天我们聊聊语音合成。\nspeaker_2:好呀，我们开始吧。",
    "ref_audio": [
      "file:///absolute/path/to/references/speaker_1.wav",
      "file:///absolute/path/to/references/speaker_2.wav"
    ],
    "ref_text": "speaker_1:在此奉劝大家别乱打美白针。\nspeaker_2:对此，乔宝云认为有夸大的成分。",
    "response_format": "wav",
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1235
  }' --output ming3_tiny_podcast.wav
```

Those example transcripts require the corresponding Seed-TTS reference clips;
replace both audio paths and transcripts together when using other references.
The adapter derives Podcast decode bounds from the turn text. An explicit
`max_new_tokens` can tighten the maximum but cannot be below the derived minimum.
The HF validation found a turn-2 speaker-similarity warning for the settings
above; the earlier temperature 1.5 / seed 1234 control retained correct speaker
matching for these references. See the [validation record](Ming-3.0-TTS-HF-validation.md#podcast-sampling-control)
for both results.

## Streaming, voices, and sampling

- Non-streaming output is mono 44.1 kHz WAV. Set `response_format: "pcm"` and
  `stream_format: "audio"` for raw PCM streaming through `/v1/audio/speech`.
- Upload reference clips through `/v1/audio/voices`; reuse their names in
  `voice`. Uploaded references remain distinct from built-in IP controls.
- Reference clips use the endpoint's 1–30 second duration limits.
- Tiny Flow overrides use `extra_params.temperature` and `extra_params.ming3_cfg`.
  The former `lingv3_cfg` remains an alias; conflicting aliases are rejected.
- Tiny uses `cond + cfg * (cond - uncond)`: `ming3_cfg=1` corresponds to the
  native sampler's `uncond + 2 * (cond - uncond)`. Numeric CFG values therefore
  require formula alignment before comparing backends.
- Decode limits count Flow steps: Tiny emits one 80 ms latent frame per step.
