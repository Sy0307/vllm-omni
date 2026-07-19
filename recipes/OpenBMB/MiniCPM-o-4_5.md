# MiniCPM-o 4.5

> Online serving and offline inference for omni multimodal chat
> (text / image / audio / video → text + 24 kHz speech)

## Summary

- Vendor: OpenBMB
- Model: [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Task: Omni multimodal chat — accepts text / image / audio / video input;
  emits text and 24 kHz mono speech in the same response
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions`
  API (plus Gradio demo), and offline inference via `Omni.generate`
- Maintainer: [`@tc-mb`](https://github.com/tc-mb) (MiniCPM-V / MiniCPM-o team)

## When to use this recipe

Use this recipe as a known-good starting point for serving
`openbmb/MiniCPM-o-4_5` on vLLM-Omni. MiniCPM-o 4.5 is the omni member
of the MiniCPM-o family — it runs a multimodal thinker, a streaming
MiniCPMTTS codec talker, and a separate batched Code2Wav stage so a single
`/v1/chat/completions` call can return text and 24 kHz speech in one
shot. The recipe covers the shipped GPU layouts (single / 2 / 3 / 8 GPUs):
  the default co-locates both stages on one GPU, and the larger scale-out
  layouts (2 / 3 / 8 GPUs) are selected via `--deploy-config`.

## References

- Default deploy configs (auto-loaded by HF `model_type=minicpmo` +
  `hf_config.version="4.5"`):
  - Single-GPU layout (default):
    [`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
  - 2-GPU layout (thinker on GPU 0, talker on GPU 1):
    [`vllm_omni/deploy/minicpmo_4_5_2gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2gpu.yaml)
  - 3-GPU layout (thinker TP=2):
    [`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
  - 8x RTX 4090 layout:
    [`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
- Online example + Gradio demo:
  [`examples/online_serving/minicpmo/`](../../examples/online_serving/minicpmo/)
- Offline end-to-end example:
  [`examples/offline_inference/minicpmo/`](../../examples/offline_inference/minicpmo/)
- Pipeline / talker source:
  [`vllm_omni/model_executor/models/minicpmo_4_5/`](../../vllm_omni/model_executor/models/minicpmo_4_5/)
- Stage-input processors (thinker → talker and talker → Code2Wav):
  [`vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`](../../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py)
- Upstream model card:
  [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Integration PR:
  [vllm-project/vllm-omni#3642](https://github.com/vllm-project/vllm-omni/pull/3642)

## Hardware Support

Three GPU layouts ship with default deploy configs. Every layout uses the
same strict three-stage topology. The Talker emits codec chunks only;
Code2Wav consumes them through a shared-memory async connector.

| Layout | Thinker | Talker | Code2Wav | Typical hardware |
| --- | --- | --- | --- | --- |
| 2-GPU (default) | GPU 0 | GPU 1 | GPU 1 | 2x A100/H100/H200 80GB |
| 3-GPU (thinker TP=2) | GPU 0,1 (TP=2) | GPU 2 | GPU 2 | 3x large-memory GPUs |
| 8x RTX 4090 24GB | GPU 0–3 (TP=4) | GPU 4 | GPU 5 | 8x RTX 4090 consumer |

## GPU

### 1 x GPU (default — single command)

The default
[`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
puts the thinker on GPU 0 (`~70 %` memory, `enforce_eager: true`) and
co-locates the codec-only Talker and Code2Wav stages on GPU 1. This is
the recommended starting layout — works on
any pair of 80GB-class GPUs (A100, H100, H200) and on most 40GB+
pairs as long as the thinker model weights fit.

#### Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: >= 0.21.0 (or current `main`)
- Optional Talker dep: `stepaudio2-minicpmo` (see Notes for why this is
  required and how to install it)

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

The deploy config is auto-loaded by the model registry — no
`--deploy-config` flag needed for this default single-GPU layout.

#### Verification

**Quick smoke test (text-only output)**:

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Briefly introduce yourself."}],
        "modalities": ["text"]
    }'
```

**Text + speech in one response** (the headline 4.5 feature). The TTS
path is gated by a Jinja flag on the chat template. Pass
`use_tts_template=true` via the **top-level** `chat_template_kwargs`
field (curl does not flatten nested `extra_body`):

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello, then introduce vLLM in one sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": true}
    }'
```

When using the OpenAI Python SDK, the same flag can also be sent as
`extra_body={"chat_template_kwargs": {"use_tts_template": True}}`
because the client merges `extra_body` into the request root.

Response carries text in one choice's `message.content` and base64 WAV
in another choice's `message.audio.data` (24 kHz mono, see Notes). With
`modalities: ["text", "audio"]` you typically get two `choices` entries
(one text, one audio).

**Gradio demo (text + image + audio + video UI)**:

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh
# or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` and try a text prompt with the **"Generate
speech output (TTS)"** checkbox on / off.

#### Notes

- Memory budget: thinker uses `gpu_memory_utilization: 0.7`; Talker and
  Code2Wav use separate 0.45 and 0.30 stage budgets on GPU 1.
- `--trust-remote-code` is required — the HF repo ships a custom
  `MiniCPMO` config / model class.
- Pin: `enforce_eager: true` on all stages. CUDA graph capture remains
  outside the currently validated configuration.
- The default and batching configs support `max_num_seqs: 4`. Talker AR
  state and Code2Wav caches are request-owned; Code2Wav batches only
  exact-shape-compatible chunks and does not fall back to serial decode.

### 2 x GPU (talker on its own GPU)

Use
[`vllm_omni/deploy/minicpmo_4_5_2gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2gpu.yaml)
when you have two GPUs and want to give the talker + Token2Wav vocoder a
dedicated card instead of sharing GPU 0 with the thinker. The thinker
runs on GPU 0 (`~90 %` mem, TP=1) and the talker on GPU 1 (`~75 %` mem,
`max_num_seqs: 1`). This relieves the memory pressure of the default
single-GPU co-located layout and is the recommended step up when a
second 80GB-class card is available but full 3-way TP scale-out is not
needed.

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_2gpu.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Verification and Notes mirror the single-GPU section; the only
difference is that the talker no longer competes with the thinker for
GPU 0 memory.

### 3 x GPU (thinker TP=2)

Use
[`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
when you have a third GPU available and want the thinker on 2-way tensor
parallel. Talker and Code2Wav share GPU 2 in this conservative layout.

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_3gpu.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Verification and Notes mirror the single-GPU section; thinker latency
roughly halves under load thanks to TP=2.

### 8 x RTX 4090 24GB (consumer-GPU layout)

Use
[`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
on an 8x RTX 4090 host. Thinker uses 4-way TP across GPUs 0–3
(`~85 %` mem each ≈ 20.4 GiB/card), Talker uses GPU 4, and Code2Wav
uses GPU 5. GPUs 6–7 are left free.

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

#### Notes

- `max_model_len` is capped at 4096 in this layout — 8192 still OOMs on
  4090s. Raise it if your cards have more headroom (e.g. 4090 D /
  custom 32 GB SKUs), but verify with a long-prompt run before
  promoting.
- All other knobs match the single-GPU section; the only difference is
  the per-card memory pressure on the thinker shards.

## Notes (applies to all layouts)

- **Code2Wav dependency**: Stage 2 loads `Token2wav` from the
  MiniCPM-o-flavored
  vocoder (PyPI package `stepaudio2-minicpmo` — NOT the upstream
  `stepfun-ai/Step-Audio2`, whose `Token2wav.__init__` signature
  rejects `n_timesteps`). Install via the published extra:

  ```bash
  pip install 'vllm-omni[minicpmo]'
  ```

  Equivalent direct install: `pip install stepaudio2-minicpmo`. A
  missing dep raises `ImportError` at first request with the same
  install hint instead of silently emitting empty audio.

- **TTS trigger**: speech output requires
  `chat_template_kwargs.use_tts_template=true` so the chat template
  appends `<|tts_bos|>` before generation. Without it, Stage-1 talker
  receives no TTS token span and returns silent audio (not text-only).
  For **curl**, put `chat_template_kwargs` at the request root; nested
  `extra_body.chat_template_kwargs` is ignored. The OpenAI Python SDK
  may use `extra_body` because it flattens those fields into the root.

- **Output audio**: 24 kHz mono WAV inside the OpenAI-style
  `message.audio.data` (base64). The Gradio demo's WAV player decodes
  this automatically.

- **Routing**: MiniCPM-o 4.5 and 2.6 both ship `architectures=
  ["MiniCPMO"]` in HF config; routing is disambiguated by
  `hf_config.version == "4.5"` via the
  `hf_config_predicate` on the 4.5 pipeline. A 2.6 checkpoint loaded
  with this recipe's `--deploy-config` will be rejected at startup
  rather than silently misrouted.

- **Async chunking**: enabled in all deploy configs. Talker sends
  25-code chunks with three-code left context to Code2Wav through
  `SharedMemoryConnector`; terminal chunks flush held lookahead state.
- **Response choices**: text and audio are separate choices. SDK clients
  should select the choice whose `message.audio.data` is populated rather
  than assuming `choices[0]` contains audio.
