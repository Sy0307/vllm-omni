from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)


def _make_minimal_talker():
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = SimpleNamespace(codec_pad_id=7, num_code_groups=16)
    return model


def test_single_token_prefill_uses_prefill_path():
    model = _make_minimal_talker()
    full_prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    trailing_text = torch.ones((2, 4), dtype=torch.float32)
    tts_pad = torch.full((1, 4), 0.5, dtype=torch.float32)
    ref_code = torch.arange(32, dtype=torch.long).reshape(2, 16)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, tts_pad, 2, ref_code

    model._build_prompt_embeds = fake_build_prompt_embeds

    input_ids = torch.tensor([123], dtype=torch.long)
    out_ids, out_embeds, update = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=3,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1
    assert update["meta"]["talker_text_offset"] == 0
    assert update["meta"]["ref_code_len"] == 2
    assert torch.equal(update["embed"]["prefill"], full_prompt_embeds)
    assert torch.equal(update["embed"]["tts_pad"], tts_pad)
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text)
    assert torch.equal(update["codes"]["ref"], ref_code)
    assert update["codes"]["audio"].shape == (1, 16)


def test_single_token_prefill_can_be_inferred_from_token_progress():
    model = _make_minimal_talker()
    full_prompt_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    trailing_text = torch.ones((1, 4), dtype=torch.float32)
    tts_pad = torch.zeros((1, 4), dtype=torch.float32)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, tts_pad, None, None

    model._build_prompt_embeds = fake_build_prompt_embeds

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_num_computed_tokens=0,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1


def test_decode_advances_trailing_text_by_offset_without_rewriting_tail():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        embed={"tts_pad": tts_pad},
        meta={"talker_text_offset": 1},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [123]
    assert torch.equal(out_embeds.cpu(), torch.full((1, 4), 123.0, dtype=torch.bfloat16))
    assert "hidden_states" not in update
    assert update["meta"]["talker_text_offset"] == 2
    past_hidden, text_step = update["mtp_inputs"]
    assert torch.equal(past_hidden.cpu(), last_hidden.reshape(1, -1).to(torch.bfloat16))
    assert torch.equal(text_step.cpu(), trailing_text[1:2].to(torch.bfloat16))


def test_decode_advances_trailing_text_offset_across_multiple_steps():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state_tail = trailing_text
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)
    meta = {"talker_text_offset": 0}
    seen_steps = []

    for _ in range(3):
        _, _, update = model.preprocess(
            input_ids=torch.tensor([123], dtype=torch.long),
            input_embeds=None,
            text=["hello"],
            task_type=["CustomVoice"],
            hidden_states={"trailing_text": state_tail, "last": last_hidden},
            embed={"tts_pad": tts_pad},
            meta=meta,
            _omni_is_prefill=False,
            _omni_num_computed_tokens=2,
            _omni_prompt_len=2,
        )
        seen_steps.append(update["mtp_inputs"][1].cpu())
        if "hidden_states" in update and "trailing_text" in update["hidden_states"]:
            state_tail = update["hidden_states"]["trailing_text"]
        meta = update["meta"]

    assert torch.equal(seen_steps[0], trailing_text[0:1].to(torch.bfloat16))
    assert torch.equal(seen_steps[1], trailing_text[1:2].to(torch.bfloat16))
    assert torch.equal(seen_steps[2], tts_pad.to(torch.bfloat16))
    assert meta["talker_text_offset"] == 0
    assert state_tail.numel() == 0


def test_decode_compacts_long_trailing_text_after_large_offset():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(130 * 4, dtype=torch.float32).reshape(130, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)

    _, _, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        embed={"tts_pad": tts_pad},
        meta={"talker_text_offset": 64},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert torch.equal(update["mtp_inputs"][1].cpu(), trailing_text[64:65].to(torch.bfloat16))
    assert update["meta"]["talker_text_offset"] == 0
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text[65:])
