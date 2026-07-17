from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)


class _RotaryStub(nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, position_ids


def _make_decoder_transformer_stub():
    model = object.__new__(Qwen3TTSTokenizerV2DecoderTransformerModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=0)
    model.input_proj = nn.Identity()
    model.output_proj = nn.Identity()
    model.norm = nn.Identity()
    model.rotary_emb = _RotaryStub()
    model.layers = nn.ModuleList()
    return model


def test_auto_position_ids_skip_packed_sequence_detection():
    model = _make_decoder_transformer_stub()
    captured = {}

    def fake_get_mask(self, **kwargs):
        captured.update(kwargs)
        return {"full_attention": None}

    model._get_cached_causal_mask_mapping = MethodType(fake_get_mask, model)
    model(inputs_embeds=torch.randn(2, 7, 4))

    assert captured["position_ids"] is None


def test_explicit_position_ids_keep_packed_sequence_detection():
    model = _make_decoder_transformer_stub()
    captured = {}

    def fake_get_mask(self, **kwargs):
        captured.update(kwargs)
        return {"full_attention": None}

    model._get_cached_causal_mask_mapping = MethodType(fake_get_mask, model)
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 3, 4]])
    model(inputs_embeds=torch.randn(1, 7, 4), position_ids=position_ids)

    assert captured["position_ids"] is position_ids
