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
  float* scratch = shared + max_seq_len;

  const int seq_len = seq_lens[b];
  if (seq_len <= 0) {
    if (tid < kHeadDim) {
      out[(row * kHeadDim) + tid] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const scalar_t* q_row = q + row * kHeadDim;
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

}  // namespace

torch::Tensor fish_decode_kvcache_attn_cuda(torch::Tensor q,
                                            torch::Tensor k_cache,
                                            torch::Tensor v_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor seq_lens,
                                            torch::Tensor out,
                                            double scale,
                                            int64_t max_seq_len) {
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
  const size_t shared_bytes =
      static_cast<size_t>(max_seq_len + kThreads) * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
