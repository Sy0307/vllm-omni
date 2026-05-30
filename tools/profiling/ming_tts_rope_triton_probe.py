#!/usr/bin/env python3
"""Probe whether a fused Triton RoPE apply is worth integrating."""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    _apply_rotary_pos_emb_from_trig,
)


@triton.jit
def _rope_kernel(
    x,
    cos,
    sin,
    out,
    total: tl.constexpr,
    dim: tl.constexpr,
    seq_len: tl.constexpr,
    heads: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    stride_d: tl.constexpr,
    cos_stride_s: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < total
    d = offsets % dim
    tmp = offsets // dim
    s = tmp % seq_len
    tmp = tmp // seq_len
    h = tmp % heads
    b = tmp // heads

    half = dim // 2
    pair_d = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, -1.0, 1.0)

    x0 = tl.load(x + b * stride_b + h * stride_h + s * stride_s + d * stride_d, mask=mask, other=0.0)
    x1 = tl.load(x + b * stride_b + h * stride_h + s * stride_s + pair_d * stride_d, mask=mask, other=0.0)
    c = tl.load(cos + s * cos_stride_s + d, mask=mask, other=0.0)
    sn = tl.load(sin + s * cos_stride_s + d, mask=mask, other=0.0)
    y = x0 * c + sign * x1 * sn
    tl.store(out + offsets, y, mask=mask)


def rope_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    out = torch.empty((batch, heads, seq_len, dim), device=x.device, dtype=x.dtype)
    total = batch * heads * seq_len * dim
    grid = (triton.cdiv(total, 256),)
    _rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        total,
        dim,
        seq_len,
        heads,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        block=256,
    )
    return out


def _time_us(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=37)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    dtype = torch.bfloat16
    base = torch.randn(
        (args.batch_size, args.seq_len, args.heads, args.head_dim),
        device=args.device,
        dtype=dtype,
    )
    x = base.transpose(1, 2)
    cos = torch.randn((1, args.seq_len, args.head_dim), device=args.device, dtype=dtype)
    sin = torch.randn((1, args.seq_len, args.head_dim), device=args.device, dtype=dtype)

    for _ in range(10):
        ref = _apply_rotary_pos_emb_from_trig(x, cos, sin, 1.0)
        candidate = rope_triton(x, cos[0], sin[0])
    torch.cuda.synchronize()

    diff = (ref - candidate).float().abs()
    pytorch_us = _time_us(lambda: _apply_rotary_pos_emb_from_trig(x, cos, sin, 1.0), args.repeats)
    triton_us = _time_us(lambda: rope_triton(x, cos[0], sin[0]), args.repeats)

    summary = {
        "shape": list(x.shape),
        "x_stride": list(x.stride()),
        "equivalence": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        },
        "timers_us": {
            "pytorch_mean": float(stats.mean(pytorch_us)),
            "pytorch_stdev": float(stats.stdev(pytorch_us)) if len(pytorch_us) > 1 else 0.0,
            "triton_mean": float(stats.mean(triton_us)),
            "triton_stdev": float(stats.stdev(triton_us)) if len(triton_us) > 1 else 0.0,
            "speedup": float(stats.mean(pytorch_us) / stats.mean(triton_us)),
        },
        "samples_us": {
            "pytorch": pytorch_us[:10],
            "triton": triton_us[:10],
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
