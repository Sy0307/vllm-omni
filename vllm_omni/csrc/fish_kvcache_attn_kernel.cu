#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <limits>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 128;
constexpr int kSmallMaxSeqLen = 1024;
constexpr int kLongTokensPerSplit = 1024;

template <typename scalar_t>
__device__ inline float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void fish_decode_kvcache_attn_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int max_seq_len,
    float scale) {
  const int row = blockIdx.x;
  const int b = row / num_q_heads;
  const int qh = row - b * num_q_heads;
  const int kvh = qh / (num_q_heads / num_kv_heads);
  const int tid = threadIdx.x;

  extern __shared__ float shared[];
  float* scores = shared;
  float* scratch = shared + kSmallMaxSeqLen;
  __shared__ float alpha_shared;
  __shared__ float beta_shared;
  __shared__ float m_shared;
  __shared__ float l_shared;

  const int seq_len = seq_lens[b];
  if (seq_len <= 0) {
    if (tid < kHeadDim) {
      out[(row * kHeadDim) + tid] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const scalar_t* q_row = q + row * kHeadDim;
  if (seq_len > kSmallMaxSeqLen) {
    float acc = 0.0f;
    if (tid == 0) {
      m_shared = -FLT_MAX;
      l_shared = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; ++t) {
      const int logical_block = t / block_size;
      if (logical_block >= max_blocks_per_seq) {
        break;
      }
      const int block_offset = t - logical_block * block_size;
      const int physical_block = block_table[b * max_blocks_per_seq + logical_block];
      const scalar_t* k_row =
          k_cache + ((physical_block * block_size + block_offset) * num_kv_heads + kvh) *
                        kHeadDim;

      float partial = 0.0f;
      if (tid < kHeadDim) {
        partial = to_float(q_row[tid]) * to_float(k_row[tid]);
      }
      scratch[tid] = partial;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        const float score = scratch[0] * scale;
        const float m_old = m_shared;
        const float m_new = fmaxf(m_old, score);
        alpha_shared = __expf(m_old - m_new);
        beta_shared = __expf(score - m_new);
        l_shared = l_shared * alpha_shared + beta_shared;
        m_shared = m_new;
      }
      __syncthreads();

      if (tid < kHeadDim) {
        const scalar_t* v_row =
            v_cache + ((physical_block * block_size + block_offset) * num_kv_heads + kvh) *
                          kHeadDim;
        acc = acc * alpha_shared + beta_shared * to_float(v_row[tid]);
      }
      __syncthreads();
    }

    if (tid < kHeadDim) {
      const float denom = fmaxf(l_shared, 1.0e-20f);
      out[row * kHeadDim + tid] = from_float<scalar_t>(acc / denom);
    }
    return;
  }

  float local_max = -FLT_MAX;

  constexpr int kWarpSize = 32;
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid >> 5;
  const int num_warps = blockDim.x >> 5;

  const int num_logical_blocks = (seq_len + block_size - 1) / block_size;
  for (int logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
    const int physical_block = block_table[b * max_blocks_per_seq + logical_block];
    const int block_token_start = logical_block * block_size;
    const int tokens_in_block = min(block_size, seq_len - block_token_start);
    const scalar_t* k_block =
        k_cache + ((physical_block * block_size * num_kv_heads + kvh) * kHeadDim);

    for (int block_offset = warp_id; block_offset < tokens_in_block; block_offset += num_warps) {
      const int t = block_token_start + block_offset;
      const scalar_t* k_row = k_block + block_offset * num_kv_heads * kHeadDim;

      float partial = 0.0f;
#pragma unroll
      for (int d = lane; d < kHeadDim; d += kWarpSize) {
        partial += to_float(q_row[d]) * to_float(k_row[d]);
      }
      float dot = warp_sum(partial) * scale;
      if (lane == 0) {
        scores[t] = dot;
        local_max = fmaxf(local_max, dot);
      }
    }
  }

  scratch[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];

  float local_sum = 0.0f;
  for (int t = tid; t < seq_len; t += blockDim.x) {
    const float weight = __expf(scores[t] - max_score);
    scores[t] = weight;
    local_sum += weight;
  }

  scratch[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / fmaxf(scratch[0], 1.0e-20f);

  if (tid < kHeadDim) {
    float acc = 0.0f;
    for (int logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
      const int physical_block = block_table[b * max_blocks_per_seq + logical_block];
      const int block_token_start = logical_block * block_size;
      const int tokens_in_block = min(block_size, seq_len - block_token_start);
      const scalar_t* v_block =
          v_cache + ((physical_block * block_size * num_kv_heads + kvh) * kHeadDim);
#pragma unroll
      for (int block_offset = 0; block_offset < 16; ++block_offset) {
        if (block_offset < tokens_in_block) {
          const int t = block_token_start + block_offset;
          const scalar_t* v_row = v_block + block_offset * num_kv_heads * kHeadDim;
          acc += scores[t] * inv_sum * to_float(v_row[tid]);
        }
      }
    }
    out[row * kHeadDim + tid] = from_float<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void fish_decode_kvcache_attn_partial_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int num_splits,
    float scale) {
  const int row = blockIdx.x;
  const int split_id = blockIdx.y;
  const int b = row / num_q_heads;
  const int qh = row - b * num_q_heads;
  const int kvh = qh / (num_q_heads / num_kv_heads);
  const int tid = threadIdx.x;
  const int total_rows = gridDim.x;

  __shared__ float scratch[kThreads];
  __shared__ float alpha_shared;
  __shared__ float beta_shared;
  __shared__ float m_shared;
  __shared__ float l_shared;

  float acc = 0.0f;
  if (tid == 0) {
    m_shared = -FLT_MAX;
    l_shared = 0.0f;
  }
  __syncthreads();

  const int seq_len = seq_lens[b];
  const int split_begin = split_id * kLongTokensPerSplit;
  const int split_end = min(seq_len, split_begin + kLongTokensPerSplit);

  if (seq_len <= 0 || split_begin >= split_end) {
    if (tid == 0) {
      partial_m[split_id * total_rows + row] = -FLT_MAX;
      partial_l[split_id * total_rows + row] = 0.0f;
    }
    if (tid < kHeadDim) {
      partial_acc[(split_id * total_rows + row) * kHeadDim + tid] = 0.0f;
    }
    return;
  }

  const scalar_t* q_row = q + row * kHeadDim;
  for (int t = split_begin; t < split_end; ++t) {
    const int logical_block = t / block_size;
    const int block_offset = t - logical_block * block_size;
    const int physical_block = block_table[b * max_blocks_per_seq + logical_block];
    const scalar_t* k_row =
        k_cache + ((physical_block * block_size + block_offset) * num_kv_heads + kvh) *
                      kHeadDim;

    float partial = 0.0f;
    if (tid < kHeadDim) {
      partial = to_float(q_row[tid]) * to_float(k_row[tid]);
    }
    scratch[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      const float score = scratch[0] * scale;
      const float m_old = m_shared;
      const float m_new = fmaxf(m_old, score);
      alpha_shared = __expf(m_old - m_new);
      beta_shared = __expf(score - m_new);
      l_shared = l_shared * alpha_shared + beta_shared;
      m_shared = m_new;
    }
    __syncthreads();

    if (tid < kHeadDim) {
      const int logical_block = t / block_size;
      const int block_offset = t - logical_block * block_size;
      const int physical_block = block_table[b * max_blocks_per_seq + logical_block];
      const scalar_t* v_row =
          v_cache + ((physical_block * block_size + block_offset) * num_kv_heads + kvh) *
                        kHeadDim;
      acc = acc * alpha_shared + beta_shared * to_float(v_row[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_m[split_id * total_rows + row] = m_shared;
    partial_l[split_id * total_rows + row] = l_shared;
  }
  if (tid < kHeadDim) {
    partial_acc[(split_id * total_rows + row) * kHeadDim + tid] = acc;
  }
}

template <typename scalar_t>
__global__ void fish_decode_kvcache_attn_combine_kernel(
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    int num_splits) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int total_rows = gridDim.x;

  __shared__ float global_m_shared;
  __shared__ float global_l_shared;

  if (tid == 0) {
    float global_m = -FLT_MAX;
    for (int split_id = 0; split_id < num_splits; ++split_id) {
      global_m = fmaxf(global_m, partial_m[split_id * total_rows + row]);
    }
    float global_l = 0.0f;
    for (int split_id = 0; split_id < num_splits; ++split_id) {
      const float m = partial_m[split_id * total_rows + row];
      const float l = partial_l[split_id * total_rows + row];
      if (l > 0.0f) {
        global_l += l * __expf(m - global_m);
      }
    }
    global_m_shared = global_m;
    global_l_shared = global_l;
  }
  __syncthreads();

  if (tid < kHeadDim) {
    if (global_l_shared <= 1.0e-20f) {
      out[row * kHeadDim + tid] = from_float<scalar_t>(0.0f);
      return;
    }
    float acc = 0.0f;
    for (int split_id = 0; split_id < num_splits; ++split_id) {
      const float l = partial_l[split_id * total_rows + row];
      if (l > 0.0f) {
        const float weight =
            __expf(partial_m[split_id * total_rows + row] - global_m_shared);
        acc += partial_acc[(split_id * total_rows + row) * kHeadDim + tid] *
               weight;
      }
    }
    out[row * kHeadDim + tid] = from_float<scalar_t>(acc / global_l_shared);
  }
}

}  // namespace

torch::Tensor fish_decode_kvcache_attn_cuda(torch::Tensor q,
                                            torch::Tensor k_cache,
                                            torch::Tensor v_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor seq_lens,
                                            torch::Tensor out,
                                            double scale,
                                            int64_t max_seq_len,
                                            torch::Tensor partial_m,
                                            torch::Tensor partial_l,
                                            torch::Tensor partial_acc) {
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, q_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [blocks, block_size, kv_heads, head_dim]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "v_cache must have the same shape as k_cache");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [batch, max_blocks_per_seq]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(partial_m.is_contiguous(), "partial_m must be contiguous");
  TORCH_CHECK(partial_l.is_contiguous(), "partial_l must be contiguous");
  TORCH_CHECK(partial_acc.is_contiguous(), "partial_acc must be contiguous");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() &&
                  q.scalar_type() == v_cache.scalar_type() &&
                  q.scalar_type() == out.scalar_type(),
              "q, k_cache, v_cache, and out must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "only fp16 and bf16 are supported");
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::ScalarType::Int,
              "seq_lens must be int32");
  TORCH_CHECK(partial_m.scalar_type() == at::ScalarType::Float &&
                  partial_l.scalar_type() == at::ScalarType::Float &&
                  partial_acc.scalar_type() == at::ScalarType::Float,
              "partial workspace tensors must be fp32");

  const int batch_size = q.size(0);
  const int num_q_heads = q.size(1);
  const int head_dim = q.size(2);
  const int block_size = k_cache.size(1);
  const int num_kv_heads = k_cache.size(2);
  const int kv_head_dim = k_cache.size(3);
  const int max_blocks_per_seq = block_table.size(1);

  TORCH_CHECK(head_dim == kHeadDim && kv_head_dim == kHeadDim,
              "only head_dim=128 is supported");
  TORCH_CHECK(block_size == 16, "only KV cache block_size=16 is supported");
  TORCH_CHECK(batch_size == block_table.size(0) && batch_size == seq_lens.size(0),
              "batch dimensions do not match");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive");
  TORCH_CHECK(max_seq_len <= max_blocks_per_seq * block_size,
              "max_seq_len exceeds block_table capacity");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const dim3 grid(batch_size * num_q_heads);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (max_seq_len <= kSmallMaxSeqLen) {
    const size_t shared_bytes =
        static_cast<size_t>(kSmallMaxSeqLen + kThreads) * sizeof(float);
    if (q.scalar_type() == at::ScalarType::Half) {
      fish_decode_kvcache_attn_kernel<at::Half><<<grid, block, shared_bytes,
                                                  stream>>>(
          q.data_ptr<at::Half>(), k_cache.data_ptr<at::Half>(),
          v_cache.data_ptr<at::Half>(), block_table.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(), out.data_ptr<at::Half>(), batch_size,
          num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
          static_cast<int>(max_seq_len), static_cast<float>(scale));
    } else {
      fish_decode_kvcache_attn_kernel<at::BFloat16><<<grid, block, shared_bytes,
                                                     stream>>>(
          q.data_ptr<at::BFloat16>(), k_cache.data_ptr<at::BFloat16>(),
          v_cache.data_ptr<at::BFloat16>(), block_table.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(), out.data_ptr<at::BFloat16>(), batch_size,
          num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
          static_cast<int>(max_seq_len), static_cast<float>(scale));
    }
  } else {
    const int num_splits =
        (static_cast<int>(max_seq_len) + kLongTokensPerSplit - 1) /
        kLongTokensPerSplit;
    TORCH_CHECK(partial_m.dim() == 2 && partial_m.size(0) == num_splits &&
                    partial_m.size(1) == batch_size * num_q_heads,
                "partial_m has incompatible shape");
    TORCH_CHECK(partial_l.dim() == 2 && partial_l.size(0) == num_splits &&
                    partial_l.size(1) == batch_size * num_q_heads,
                "partial_l has incompatible shape");
    TORCH_CHECK(partial_acc.dim() == 3 && partial_acc.size(0) == num_splits &&
                    partial_acc.size(1) == batch_size * num_q_heads &&
                    partial_acc.size(2) == kHeadDim,
                "partial_acc has incompatible shape");
    const dim3 split_grid(batch_size * num_q_heads, num_splits);
    if (q.scalar_type() == at::ScalarType::Half) {
      fish_decode_kvcache_attn_partial_kernel<at::Half><<<split_grid, block, 0,
                                                         stream>>>(
          q.data_ptr<at::Half>(), k_cache.data_ptr<at::Half>(),
          v_cache.data_ptr<at::Half>(), block_table.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(), partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(), partial_acc.data_ptr<float>(),
          num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
          num_splits, static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      fish_decode_kvcache_attn_combine_kernel<at::Half><<<grid, block, 0,
                                                         stream>>>(
          seq_lens.data_ptr<int32_t>(), out.data_ptr<at::Half>(),
          partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(), num_splits);
    } else {
      fish_decode_kvcache_attn_partial_kernel<at::BFloat16><<<split_grid, block,
                                                            0, stream>>>(
          q.data_ptr<at::BFloat16>(), k_cache.data_ptr<at::BFloat16>(),
          v_cache.data_ptr<at::BFloat16>(), block_table.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(), partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(), partial_acc.data_ptr<float>(),
          num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
          num_splits, static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      fish_decode_kvcache_attn_combine_kernel<at::BFloat16><<<grid, block, 0,
                                                            stream>>>(
          seq_lens.data_ptr<int32_t>(), out.data_ptr<at::BFloat16>(),
          partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(), num_splits);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
