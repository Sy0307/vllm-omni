#include <torch/extension.h>

torch::Tensor fish_decode_kvcache_attn_cuda(torch::Tensor q,
                                            torch::Tensor k_cache,
                                            torch::Tensor v_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor seq_lens,
                                            torch::Tensor out,
                                            double scale,
                                            int64_t max_seq_len);

torch::Tensor fish_decode_kvcache_attn(torch::Tensor q,
                                       torch::Tensor k_cache,
                                       torch::Tensor v_cache,
                                       torch::Tensor block_table,
                                       torch::Tensor seq_lens,
                                       torch::Tensor out,
                                       double scale,
                                       int64_t max_seq_len) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  return fish_decode_kvcache_attn_cuda(q, k_cache, v_cache, block_table,
                                       seq_lens, out, scale, max_seq_len);
}

TORCH_LIBRARY(vllm_omni_fish_kvcache_attn, m) {
  m.def(
      "decode(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
      "Tensor seq_lens, Tensor(a!) out, float scale, int max_seq_len) -> "
      "Tensor(a!)");
}

TORCH_LIBRARY_IMPL(vllm_omni_fish_kvcache_attn, CUDA, m) {
  m.impl("decode", &fish_decode_kvcache_attn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("has_fish_kvcache_attn", []() { return true; });
}

