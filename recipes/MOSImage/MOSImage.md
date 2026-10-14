# MOSImage text-to-image

## Summary

- Model: MOSImage, loaded from a local Diffusers-format checkpoint.
- Task: Text-to-image through the offline engine and `/v1/images/generations`.
- Hardware validated: NVIDIA H200; one GPU for complete image generation.

MOSImage combines a Qwen3.5-9B text encoder, a dense joint image/text DiT and
a FLUX.2 VAE. The validated DiT has 42 blocks, hidden size 3072, 24 attention
heads, FFN size 8192 and two text-refinement blocks. Its 128-channel packed
latents have spatial stride 16. Weights and DiT activations use BF16; noise,
CFG, nearest-grid Euler updates and VAE decoding use FP32.

## Features

| Feature | Configuration |
| --- | --- |
| Tensor parallelism | `--tensor-parallel-size 2`, 4 or 8 for this checkpoint |
| Ulysses sequence parallelism (USP) | `--ulysses-degree 2` |
| CFG parallelism | `--cfg-parallel-size 2` |
| Request batching | `--max-num-seqs 4` |
| Continuous batching | `--step-execution --max-num-seqs 4` |
| Distributed layerwise offloading (DLO) | Stream DiT, text-refinement and text-encoder blocks using the config below |
| Skip Layer Guidance (SLG) | Two-pass unconditional skipping or three-pass positive SLG via `extra_params` |

DLO also works on one GPU: the AllGather selector streams complete blocks
without a collective at group size one. With multiple DP ranks it gathers
weight shards across DP; otherwise it uses the SP group. The text encoder
remains replicated independently of DiT TP/SP. Use `weight_transfer=rank-local`
for independent transfers; optional leading resident DiT layers require that
mode. Full-checkpoint multi-GPU generation is not qualified by this recipe;
the two-GPU DLO check used a small native transformer with real NCCL AllGather.
Ring attention and approximate denoising caches are not supported.

SLG layer IDs are zero-based main-DiT indices, so `[22]` skips the 23rd block.
The active window is `slg_start <= step_index / num_steps < slg_stop`.
Optional `slg_rescaling_scale` applies per-image rescaling in denoised space.
SLG is disabled by default; layer and scale choices below are example settings.

## Kernel fusion

The eager CUDA path fuses RMSNorm/AdaLN, Q/K RMSNorm with RoPE, SiTU FFN
activation, and gated residual additions while preserving BF16 rounding
boundaries. Full-forward requests also reuse immutable text conditioning and
RoPE. Matched reductions target PyTorch 2.13; unsupported cases use Torch
reductions, unfused kernels or eager operators.

## Environment and checkpoint

Validated on Linux with Python 3.12, PyTorch 2.13.0+cu130, vLLM 0.31.0
(precompiled build), Transformers 5.14.1 and Diffusers 0.40.0.
Install vLLM-Omni following the [installation guide](../../docs/getting_started/installation/README.md).

Set `MODEL_DIR` to a checkpoint containing `model_index.json` with
`_class_name: MOSImagePipeline`, plus `transformer/`, `text_encoder/`,
`tokenizer/`, `scheduler/` and `vae/`. This recipe does not assume a public
Hugging Face model identifier.

## Deploy

One GPU, weights resident:

```bash
MODEL_DIR=/path/to/MOSImage
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --omni --served-model-name MOSImage --model-class-name MOSImagePipeline \
  --dtype bfloat16 --enforce-eager --cache-backend none --port 8091
```

For DLO, add this option to the same command:

```bash
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"allgather"}}}'
```

For example, select two visible GPUs and add `--ulysses-degree 2` for USP.
Parallel degrees multiply; choose enough visible GPUs for the resulting
world size.

## Three demo requests

Run these commands from the repository root. The included
[pagoda prompt](pagoda_prompt.txt) is the original 9502-byte nutrition-page
prompt used for the measurements. Change the prompt file for another scene.
`jq` and `base64` are required to build requests and save the returned images.

```bash
jq -n --rawfile prompt recipes/MOSImage/pagoda_prompt.txt \
  '{model:"MOSImage",prompt:($prompt|rtrimstr("\n")),negative_prompt:"",size:"1024x1024",n:1,
    num_inference_steps:50,guidance_scale:3.0,flow_shift:5.0,seed:0,
    generator_device:"cuda",response_format:"b64_json"}' > baseline.json
```

Baseline, CFG 3:

```bash
curl --fail-with-body -sS http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' --data-binary @baseline.json \
  | jq -r '.data[0].b64_json' | base64 -d > baseline.png
```

Two-pass, CFG 3 with skipped unconditional layer 22:

```bash
jq '.extra_params={uncond_skip_layers:[22],slg_start:0.01,slg_stop:1.0}' \
  baseline.json > uncond.json
curl --fail-with-body -sS http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' --data-binary @uncond.json \
  | jq -r '.data[0].b64_json' | base64 -d > uncond.png
```

Three-pass, CFG 5 plus skipped positive layer 22, SLG scale 3:

```bash
jq '.guidance_scale=5.0 |
    .extra_params={slg_layers:[22],slg_scale:3.0,slg_start:0.01,slg_stop:1.0}' \
  baseline.json > slg.json
curl --fail-with-body -sS http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' --data-binary @slg.json \
  | jq -r '.data[0].b64_json' | base64 -d > slg.png
```

## Demo performance

One H200, eager, batch 1, 1024×1024, 50 steps, shift 5, CUDA seed 0 and the
same prompt above. Each entry is one full offline generation after normal
startup warmup, including text encoding, denoising, VAE decoding and PIL
conversion; model initialization and image saving are excluded.

| Demo | Resident, seconds | DLO, seconds |
| --- | ---: | ---: |
| Baseline, CFG 3 | 11.735 | 18.711 |
| Two-pass, CFG 3, uncond skip 22 | 11.657 | 18.287 |
| Three-pass, CFG 5, SLG scale 3, cond skip 22 | 18.516 | 26.805 |

These are single-run functional measurements, not throughput benchmarks.
Separate worker-local memory measurements gave peak PyTorch allocations of
30.31 GiB resident and 8.96 GiB DLO, and peak allocator reservations of
34.44 GiB and 13.12 GiB. The peak occurs during FP32 VAE decoding. These
figures exclude CUDA-context and other allocations outside PyTorch's allocator.
All three DLO images matched their corresponding resident images pixel-for-pixel;
they are generated outputs, not ground-truth images.
The deployment command and all three curl requests above were also executed
with DLO, returned HTTP 200 and reproduced the same generated pixels.

## Offline inference

The shared text-to-image entrypoint also accepts SLG through `--extra-body`.
Set `--timesteps-shift` explicitly so the runner's legacy shift aliases agree
with the requested flow shift. This command also reproduced the corresponding
resident image pixel-for-pixel:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$MODEL_DIR" --prompt "$(cat recipes/MOSImage/pagoda_prompt.txt)" \
  --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 3.0 \
  --seed 0 --timesteps-shift 5.0 --enforce-eager \
  --extra-body '{"flow_shift":5.0,"uncond_skip_layers":[22],"slg_start":0.01,"slg_stop":1.0}' \
  --output uncond_offline.png
```
