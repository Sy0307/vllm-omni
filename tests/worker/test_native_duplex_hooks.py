from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minicpmo_duplex_policy_case(
    state: SimpleNamespace,
    payload: dict[str, object],
):
    from vllm_omni.model_executor.duplex import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    session_key = ("sid-policy", 1)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": 7,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-policy",
        session_id=session_key[0],
        incarnation=session_key[1],
        seq=3,
        payload=payload,
        max_tokens=20,
    )
    return model, row


def test_minicpmo_model_hook_owns_duplex_sampling_rows_and_force_listen():
    from vllm_omni.model_executor.duplex import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    listen_id = 7
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": listen_id,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-hook", 2): state})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-hook",
        session_id="sid-hook",
        incarnation=2,
        seq=3,
        payload={"force_listen": True, "is_speech": True},
        max_tokens=20,
    )

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert model._minicpmo45_active_duplex_rows == [0]
    assert model._minicpmo45_duplex_row_sessions == {0: ("sid-hook", 2)}
    assert model._minicpmo45_duplex_row_payloads == {0: row.payload}
    assert model._minicpmo45_duplex_row_max_tokens == {0: 20}
    assert logits[0, listen_id].item() == 0.0
    assert torch.isneginf(logits[0, :listen_id]).all()
    assert torch.isneginf(logits[0, listen_id + 1 :]).all()


def test_minicpmo_model_hook_turn_end_latch_forces_silence_to_listen():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert logits[0, 7].item() == 0.0
    assert torch.isneginf(logits[0, :7]).all()
    assert torch.isneginf(logits[0, 8:]).all()


def test_minicpmo_model_hook_new_speech_clears_turn_end_latch():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    model._minicpmo45_turn_ended_sessions = {("sid-policy", 1)}
    logits = torch.zeros((1, 16), dtype=torch.float32)
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert model._minicpmo45_turn_ended_sessions == set()
    assert state.last_terminator_token is None
    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_mid_turn_speech_redirects_listen_to_tts_bos():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 8] = -2.0
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.isneginf(logits[0, 7])
    assert logits[0, 8].item() == 30.0
    assert logits[0, 10].item() == 20.0


def test_minicpmo_model_hook_new_user_turn_opens_turn_without_forcing_speak():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=8,
        pending_terminator_token=8,
    )
    model, row = _minicpmo_duplex_policy_case(
        state,
        {"is_speech": True, "new_user_turn": True, "force_speak": True},
    )
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 5.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert state.current_turn_ended is True
    assert state.last_terminator_token is None
    assert torch.equal(logits, original_logits)


def test_generic_ar_runner_builds_typed_duplex_sampling_rows():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-duplex", "req-plain"])
    runner.model_intermediate_buffer = {
        "req-duplex": {
            "duplex": {
                "data_plane": True,
                "session_id": "sid-runner-hook",
                "incarnation": 4,
                "seq": 4,
                "payload": {"is_speech": True},
            }
        }
    }
    runner.requests = {
        "req-duplex": SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=32),
        )
    }

    rows = runner._duplex_sampling_rows()

    assert len(rows) == 1
    assert rows[0].row_idx == 0
    assert rows[0].request_id == "req-duplex"
    assert rows[0].session_id == "sid-runner-hook"
    assert rows[0].incarnation == 4
    assert rows[0].seq == 4
    assert rows[0].payload == {"is_speech": True}
    assert rows[0].max_tokens == 32


def test_generic_ar_runner_has_no_minicpmo_sampler_state_or_typeerror_probe():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    source = inspect.getsource(GPUARModelRunner)

    assert "_minicpmo45_duplex_row" not in source
    assert "_minicpmo45_native_duplex_token_ids" not in source
    assert 'if "duplex_rows" not in str(exc)' not in source


def test_minicpmo_model_cleans_incarnation_state_when_request_finishes():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    request_id = "duplex-sid-cleanup-i3-e0-stage0"
    session_key = ("sid-cleanup", 3)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model = SimpleNamespace()
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: object()})
    model._minicpmo45_duplex_request_sessions = {request_id: session_key}
    model._minicpmo45_force_listen_applied_segments = {
        (request_id, 1),
        ("duplex-sid-other-e0-stage0", 2),
    }

    model.on_requests_finished({request_id})

    assert model._minicpmo45_duplex_data_plane_helper.sessions == {}
    assert model._minicpmo45_duplex_request_sessions == {}
    assert model._minicpmo45_force_listen_applied_segments == {
        ("duplex-sid-other-e0-stage0", 2),
    }


def test_minicpmo_stage0_rejects_invalid_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    with pytest.raises(ValueError, match="invalid native duplex ref_audio_data"):
        MiniCPMO45Stage0DuplexRuntime._decode_ref_audio_from_session_config(
            {
                "extra_body": {
                    "ref_audio_data": "a",
                    "ref_audio_format": "pcm_f32le",
                }
            }
        )


def test_minicpmo_tts_native_duplex_exports_segment_text_not_accumulated_condition_text():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        _ar_last_chunk_flags = [False]
        _ar_last_emitted_text = "你好，你有什莫想聊的吗？"

        def __call__(self, **kwargs):
            del kwargs
            return None, torch.zeros(8, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        runtime_additional_information=[
            {
                "native_duplex": True,
                "llm_output_text": ["你好，你有什莫想聊的吗？你好，你有什莫想聊的吗？"],
                "meta": {
                    "native_duplex_segment_text": "你好，你有什莫想聊的吗？",
                },
            }
        ],
    )

    text_bytes = output.multimodal_outputs["meta.llm_output_text_utf8"].detach().cpu().tolist()
    assert bytes(text_bytes).decode("utf-8") == "你好，你有什莫想聊的吗？"
    assert int(output.multimodal_outputs["meta.audio_text_total_chars"].item()) == len("你好，你有什莫想聊的吗？")


def test_minicpmo_tts_native_duplex_exports_model_turn_end_metadata():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        _ar_last_chunk_flags = [True]
        _ar_turn_end_flags = [True]

        def __call__(self, **kwargs):
            del kwargs
            return None, torch.zeros(0, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        runtime_additional_information=[
            {
                "native_duplex": True,
                "meta": {"native_duplex_segment_text": ""},
            }
        ],
    )

    assert int(output.multimodal_outputs["meta.turn_end"].item()) == 1


def test_minicpmo_stage0_special_token_ids_are_tokenizer_derived():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_bos|>": 105,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [self.ids[text]] if text in self.ids else [201, self.ids["<|tts_bos|>"]]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    runtime._require_special_token_ids()
    assert runtime.tts_bos_token_id == 105
    assert runtime.stage_padding_token_id() == 102
    assert runtime._special_token_ids()["chunk_tts_eos_token_id"] == 109


def test_minicpmo_stage0_rejects_unknown_special_token_fallbacks():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [self.unk_token_id]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    with pytest.raises(ValueError, match=r"<\|tts_bos\|>"):
        runtime._require_special_token_ids()


def test_minicpmo_stage0_data_plane_prefill_matches_official_unit_format():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [201, 5],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-data-plane-prefill")

    # Official duplex format: each unit is <unit> + audio embeddings with no
    # per-chunk assistant header or <|tts_bos|> boundary. Decoding starts right
    # after the audio so the first sampled token is the listen/speak decision.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=1)

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert result["prompt_suffix_len"] == 0

    # Subsequent units must close the previous unit with </unit> first,
    # mirroring the official finalize_unit() feed.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=2)

    assert result["success"] is True
    assert result["input_token_ids"] == [2, 1, 11]
    assert result["prompt_suffix_len"] == 0


def test_minicpmo_stage0_data_plane_next_append_reinjects_previous_listen():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-speech-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=3,
        last_terminator_token=3,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [3, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 3
    assert state.current_turn_ended is True


def test_minicpmo_stage0_data_plane_new_user_turn_inserts_official_prefix_after_unit_close():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-user-turn-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=10,
        last_terminator_token=10,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
        new_user_turn=True,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [10, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 10
    assert state.current_turn_ended is True


def test_minicpmo_stage0_data_plane_new_user_turn_uses_clean_done_prefix_variant():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-user-turn-clean-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=10,
        last_terminator_token=10,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
        new_user_turn=True,
        new_user_turn_prefix_variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [10, 2, 1, 11]


def test_minicpmo_stage0_data_plane_new_user_turn_preserves_audio_cache():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    stale_cache = object()

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)
            self.audio_past_key_values = stale_cache

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-user-turn-audio-cache",
        audio_chunk_idx=1,
        audio_past_key_values=stale_cache,
        pending_terminator_token=8,
        last_terminator_token=8,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
        new_user_turn=True,
        new_user_turn_prefix_variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE,
    )

    assert result["success"] is True
    assert state.audio_past_key_values is stale_cache
    assert runtime.thinker.audio_past_key_values is stale_cache


def test_minicpmo_stage0_data_plane_final_first_chunk_does_not_add_silence_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        first_chunk_ms = 10
        sample_rate = 1000

        def __init__(self):
            super().__init__()
            self.seen_audio = None
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            self.seen_audio = np.asarray(data["audio_features"], dtype=np.float32)
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    class _MelProcessor:
        sample_rate = 1000

        def get_config(self):
            return {"effective_first_chunk_ms": 10}

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(
        _streaming_mel_processor=_MelProcessor(),
        get_streaming_chunk_size=lambda: 10,
    )
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-first-chunk-padding")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.arange(8, dtype=np.float32),
        seq=1,
        final=True,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert runtime.stage_model.seen_audio is not None
    np.testing.assert_allclose(
        runtime.stage_model.seen_audio.reshape(-1),
        np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
    )


def test_minicpmo_stage0_runtime_uses_loaded_vllm_embed_tokens_when_get_input_embeddings_is_broken():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, 2))
            self.calls = []

        def forward(self, input_ids):
            ids = input_ids.reshape(-1).tolist()
            self.calls.append(ids)
            return torch.tensor([[float(i), 0.0] for i in ids], dtype=torch.float32)

    class _Thinker:
        def __init__(self):
            self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=_Embed()))

        def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
            raise AttributeError("'Qwen3ForCausalLM' object has no attribute 'get_input_embeddings'")

    thinker = _Thinker()
    stage_model = SimpleNamespace(model_stage="llm", thinker=thinker, processor=None)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = stage_model
    runtime.thinker = thinker
    runtime.device = "cpu"

    embeds = runtime._embed_token(11)

    assert embeds.shape == (1, 2)
    assert thinker.llm.model.embed_tokens.calls == [[11]]


def test_minicpmo_remote_config_patch_handles_nested_and_dict_configs():
    from vllm_omni.experimental.fullduplex.minicpmo45.compat import (
        patch_minicpmo_remote_config,
    )

    nested = SimpleNamespace(base_model_tp_plan=None)
    config = SimpleNamespace(
        base_model_tp_plan=None,
        text_config=nested,
        tts_config={},
    )

    patch_minicpmo_remote_config(config)

    assert config.base_model_tp_plan == {}
    assert nested.base_model_tp_plan == {}
    assert config.tts_config["top_p"] == 0.8
    assert config.tts_config["top_k"] == 100
    assert config.tts_config["temperature"] == 0.8
    assert config.tts_config["repetition_penalty"] == 1.05


def test_minicpmo_stage0_short_audio_buffers_without_context_mutation():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Processor:
        def get_streaming_chunk_size(self):
            return 16000

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.processor = _Processor()
    runtime._require_special_token_ids = lambda: None
    state = _MiniCPMO45Stage0SessionState(session_id="sid")

    result = runtime._stage_prefill_embeddings_only(state, np.zeros(1600, dtype=np.float32))

    assert result["success"] is False
    assert result["reason"]
    assert len(state.audio_buffer) >= 1600
    assert state.context_embeds == []


def test_minicpmo_stage0_native_sampler_penalizes_repeated_text_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    repeated = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, repeated] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[repeated] * 8],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_override_model_at_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            200: "我",
            201: "喜",
            202: "欢",
            203: "。",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202, 203]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_cut_before_natural_boundary_minimum():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "。"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_rewrite_model_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            108386: "你",
            104256: "好",
            3837: "，",
            100644: "今",
            99172: "天",
            100281: "想",
            27442: "聊",
            99217: "什",
            1773: "。",
            99218: "么",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"。": [1773]}.get(text, [])

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    period = 1773
    continuation = 99218
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, period] = 30.0
    logits[0, continuation] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 108386, 104256, 3837, 100644, 99172, 100281, 27442, 99217]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[period]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_preserves_early_model_turn_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151717] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151717]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_keeps_hard_chunk_cap():
    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[200] * (MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK - 1)],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_cuts_before_request_length_cap():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.max_new_speak_tokens_per_chunk = 64
    model._minicpmo45_duplex_row_max_tokens = {0: 20}
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706] + [200] * 18],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_ignores_pending_placeholders():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    newline = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, newline] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[-1, -1, -1]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[newline]]


def test_minicpmo_stage0_native_sampler_converts_mid_turn_listen_to_tts_bos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151703]]
    assert state.current_turn_ended is False


def test_minicpmo_stage0_native_sampler_forced_listen_yields_floor():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_row_payloads = {0: {"force_listen": True}}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151705]]
    assert state.current_turn_ended is True


def test_minicpmo_stage0_native_sampler_uses_runner_duplex_rows():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    metadata = SimpleNamespace(
        prompt_token_ids=torch.tensor([[1, 2, 3]]),
    )

    rows = model._minicpmo45_native_duplex_prompt_rows(
        metadata,
        unit_id=151683,
        batch_size=1,
        duplex_rows=[0],
    )

    assert rows == [0]


def test_minicpmo_stage0_session_context_includes_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.unit_token_id = 151683
    runtime.processor = SimpleNamespace()
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.device = "cpu"
    token_map = {
        "<|im_start|>system\nUse speech.\n<|audio_start|>": [1, 2, 3],
        "<|audio_end|><|im_end|>": [4, 5],
    }
    runtime._stage_runtime_ready = lambda: True
    runtime._require_special_token_ids = lambda: None
    runtime._decode_ref_audio_from_session_config = lambda _config: np.array([0.1, -0.1], dtype=np.float32)
    runtime._encode_text = lambda text: token_map[text]
    runtime._embed_token = lambda token_id: torch.full((1, 2), float(token_id))
    runtime._stage_ref_audio_embeddings = lambda ref_audio, state=None: torch.tensor([[10.0, 11.0], [12.0, 13.0]])

    state = _MiniCPMO45Stage0SessionState(session_id="sid-ref")
    runtime._prepare_session_context(state, {"instructions": "Use speech.", "extra_body": {"ref_audio_data": "x"}})

    assert state.context_token_ids == [1, 2, 3, 151683, 151683, 4, 5]
    assert len(state.context_embeds) == 6
