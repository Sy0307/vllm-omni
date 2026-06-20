from __future__ import annotations

from pathlib import Path

import torch


def test_qwen3_tts_talker_declares_prefill_and_ref_gpu_resident() -> None:
    source = Path("vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py").read_text()
    assert '("embed", "prefill")' in source
    assert '("codes", "ref")' in source


def test_intermediate_buffer_keeps_nested_gpu_resident_tensor_on_device() -> None:
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    if not torch.cuda.is_available():
        return
    buffer = OmniIntermediateBuffer(max_num_reqs=1)
    tensor = torch.ones((2, 3), device="cuda")
    buffer.update(0, {"embed": {"prefill": tensor}}, gpu_resident_keys={("embed", "prefill")})
    stored = buffer.buffers[0]["embed"]["prefill"]
    assert isinstance(stored, torch.Tensor)
    assert stored.device.type == "cuda"
    assert stored.shape == (2, 3)


def test_intermediate_buffer_keeps_codes_ref_on_device() -> None:
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    if not torch.cuda.is_available():
        return
    buffer = OmniIntermediateBuffer(max_num_reqs=1)
    ref_code = torch.ones((4, 16), device="cuda", dtype=torch.long)
    buffer.update(0, {"codes": {"ref": ref_code}}, gpu_resident_keys={("codes", "ref")})
    stored = buffer.buffers[0]["codes"]["ref"]
    assert isinstance(stored, torch.Tensor)
    assert stored.device.type == "cuda"
    assert stored.dtype == torch.long


def test_qwen3_tts_code2wav_codec_stats_log_does_not_extract_gpu_values() -> None:
    source = Path("vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py").read_text()
    start = source.index('"Code2Wav codec:')
    block = source[start : start + 500]
    assert ".item()" not in block
    assert "torch.unique" not in block


def test_omni_model_state_uses_cpu_seq_len_upper_bound_when_available() -> None:
    source = Path("vllm_omni/worker_v2/model_states/omni_model_state.py").read_text()
    assert 'seq_lens_cpu_upper_bound = getattr(input_batch, "seq_lens_cpu_upper_bound", None)' in source
    assert "max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())" in source


def test_prompt_builder_long_tensor_cache_reuses_tensor() -> None:
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._long_tensor_cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = builder._long_tensor([1, 2, 3], device)
    second = builder._long_tensor([1, 2, 3], device)
    assert first is second
    assert first.device.type == device.type
    if first.device.type == "cuda":
        assert first.device.index == torch.accelerator.current_device_index()
    assert first.dtype == torch.long
    assert first.tolist() == [[1, 2, 3]]


def test_prompt_builder_mel_spectrogram_does_not_sync_for_range_logging() -> None:
    source = Path("vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py").read_text()
    start = source.index("def mel_spectrogram")
    block = source[start : start + 1200]
    assert "torch.min(y)" not in block
    assert "torch.max(y)" not in block
