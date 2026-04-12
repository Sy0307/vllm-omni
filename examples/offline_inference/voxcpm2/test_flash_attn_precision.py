"""Test flash_attn precision vs native SDPA for VoxCPM2 base_lm.

Loads the native model, runs a prefill + N decode steps with both
native SDPA and flash_attn_func, comparing outputs at each step.

Usage:
    python test_flash_attn_precision.py --model /path/to/VoxCPM2 --steps 20
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def run_decode_step_sdpa(attn, hidden, pos_emb, pos_id, kv_cache):
    """Native SDPA decode step (reference)."""
    bsz, _ = hidden.size()
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    v = attn.v_proj(hidden)

    q = q.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    if pos_emb is not None:
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

    key_cache, value_cache = kv_cache
    key_cache[:, :, pos_id, :] = k.squeeze(2)
    value_cache[:, :, pos_id, :] = v.squeeze(2)

    attn_mask = torch.arange(key_cache.size(2), device=key_cache.device) <= pos_id
    q = q.contiguous()
    key_cache_c = key_cache.contiguous()
    value_cache_c = value_cache.contiguous()

    out = F.scaled_dot_product_attention(q, key_cache_c, value_cache_c, attn_mask=attn_mask, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(bsz, attn.num_heads * attn.head_dim)
    return attn.o_proj(out)


def run_decode_step_flash(attn, hidden, pos_emb, pos_id, kv_cache):
    """Flash attention decode step."""
    from flash_attn import flash_attn_func

    bsz, _ = hidden.size()
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    v = attn.v_proj(hidden)

    q = q.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    if pos_emb is not None:
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

    key_cache, value_cache = kv_cache
    key_cache[:, :, pos_id, :] = k.squeeze(2)
    value_cache[:, :, pos_id, :] = v.squeeze(2)

    valid_len = pos_id + 1
    q_flash = q.transpose(1, 2)
    k_flash = key_cache[:, :, :valid_len, :].transpose(1, 2)
    v_flash = value_cache[:, :, :valid_len, :].transpose(1, 2)

    out = flash_attn_func(q_flash.contiguous(), k_flash.contiguous(), v_flash.contiguous(), causal=True)
    out = out.reshape(bsz, attn.num_heads * attn.head_dim)
    return attn.o_proj(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    from voxcpm.core import VoxCPM

    print("Loading model...")
    native = VoxCPM.from_pretrained(args.model, load_denoiser=False, optimize=False)
    tts = native.tts_model.to("cuda")
    dtype = tts.base_lm.embed_tokens.weight.dtype
    device = torch.device("cuda")

    print(f"Model loaded. dtype={dtype}, base_lm layers={len(tts.base_lm.layers)}")

    # Test single layer attention
    layer = tts.base_lm.layers[0]
    attn = layer.self_attn
    rope = tts.base_lm.rope_emb

    hidden_size = attn.hidden_size
    num_kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim
    max_len = 256

    # Setup KV caches (two separate copies for independent testing)
    kv_sdpa = (
        torch.zeros(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype),
        torch.zeros(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype),
    )
    kv_flash = (
        torch.zeros(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype),
        torch.zeros(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype),
    )

    print(f"\nRunning {args.steps} decode steps, comparing SDPA vs flash_attn...")
    print(f"{'Step':>4} | {'Max Abs Diff':>14} | {'Cosine Sim':>12} | {'Rel Norm Diff':>14}")
    print("-" * 60)

    max_diffs = []
    for step in range(args.steps):
        hidden = torch.randn(1, hidden_size, device=device, dtype=dtype) * 0.1
        pos_ids = torch.tensor([step], device=device)
        pos_emb = rope(pos_ids)

        with torch.no_grad():
            out_sdpa = run_decode_step_sdpa(attn, hidden, pos_emb, step, kv_sdpa)
            out_flash = run_decode_step_flash(attn, hidden, pos_emb, step, kv_flash)

        abs_diff = (out_sdpa - out_flash).abs().max().item()
        cos_sim = F.cosine_similarity(out_sdpa.flatten(), out_flash.flatten(), dim=0).item()
        norm_sdpa = out_sdpa.norm().item()
        rel_diff = abs_diff / norm_sdpa if norm_sdpa > 0 else 0

        max_diffs.append(abs_diff)
        print(f"{step:4d} | {abs_diff:14.6e} | {cos_sim:12.8f} | {rel_diff:14.6e}")

    print("-" * 60)
    print(f"Max abs diff across all steps: {max(max_diffs):.6e}")
    print(f"Avg abs diff: {sum(max_diffs)/len(max_diffs):.6e}")

    if max(max_diffs) < 1e-3:
        print("\nVERDICT: PASS — flash_attn matches native SDPA within tolerance")
    else:
        print("\nVERDICT: FAIL — significant precision difference detected")


if __name__ == "__main__":
    main()
