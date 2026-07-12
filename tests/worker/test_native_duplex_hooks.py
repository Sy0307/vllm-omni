from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.worker.mixins import OmniWorkerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _register_test_native_duplex_provider(monkeypatch):
    from vllm_omni.experimental.fullduplex.engine import worker as native_duplex

    def provider(worker, capabilities):
        return getattr(worker, "_test_native_duplex_target", None)

    monkeypatch.setattr(native_duplex, "_DEFAULT_PROVIDERS_BOOTSTRAPPED", False)
    monkeypatch.setattr(native_duplex, "_NATIVE_DUPLEX_PROVIDERS", [provider])


class _PlainModelMethods:
    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare(self, **_kwargs):
        self.prepare_calls += 1

    def prefill(self, **_kwargs):
        return {"success": True}

    def generate(self, **_kwargs):
        return {"is_listen": True}


class _ExplicitNativeDuplexTarget:
    runtime_impl = "test_explicit_runtime"
    owned_runtime = False

    def __init__(self, *, fail_close: bool = False) -> None:
        self.fail_close = fail_close
        self.open_calls = []
        self.append_calls = []
        self.signal_calls = []
        self.close_calls = []

    def open_duplex_session(self, **kwargs):
        self.open_calls.append(kwargs)
        return {"opened": kwargs["session_id"]}

    def append_duplex_input(self, **kwargs):
        self.append_calls.append(kwargs)
        return {
            "text": "explicit",
            "audio_waveform": np.array([0.25, -0.25], dtype=np.float32),
            "cost_all": 0.002,
        }

    def signal_duplex_turn(self, **kwargs):
        self.signal_calls.append(kwargs)
        return {"event": kwargs["event"]}

    def close_duplex_session(self, **kwargs):
        self.close_calls.append(kwargs)
        if self.fail_close:
            raise RuntimeError("close failed")
        return {"closed": kwargs["session_id"]}


_USE_DEFAULT_TEST_TARGET = object()


class _Worker(OmniWorkerMixin):
    def __init__(self, model, native_duplex_target=_USE_DEFAULT_TEST_TARGET) -> None:
        self.model_runner = SimpleNamespace(model=model)
        if native_duplex_target is _USE_DEFAULT_TEST_TARGET:
            native_duplex_target = None if getattr(model, "model_stage", None) is not None else model
        self._test_native_duplex_target = native_duplex_target


class _SplitMiniCPMOStageModel:
    def __init__(self, model_stage: str) -> None:
        self.model_stage = model_stage
        self.name_or_path = "/tmp/minicpmo45"
        self.config = SimpleNamespace(_name_or_path="/tmp/minicpmo45")
        if model_stage == "llm":
            self.thinker = SimpleNamespace(llm=SimpleNamespace(), get_audio_hidden_states=lambda data: [])
            self.talker = None
            self.model = self.thinker
        elif model_stage == "tts":
            self.thinker = None
            self.talker = SimpleNamespace()
            self.model = self.talker
        else:
            self.thinker = None
            self.talker = None
            self.model = SimpleNamespace()


def _mark_runner_context_contract(fn):
    fn.uses_scheduler_metadata = True
    fn.uses_runner_kv_cache = True
    fn.vllm_omni_runner_context_contract = True
    return fn


def _minicpmo_tts_handoff_payload(
    torch,
    *,
    token_ids: list[int],
    hidden,
    text: str = "hello",
    end_of_turn: bool = False,
) -> dict:
    from vllm_omni.data_entry_keys import serialize_payload

    return {
        "omni_payload": serialize_payload(
            {
                "ids": {"output": token_ids},
                "hidden_states": {"output": hidden},
            }
        ),
        "llm_output_text": [text],
        "end_of_turn": end_of_turn,
    }


def test_worker_minicpmo_stage0_reuses_loaded_llm_stage_without_full_model_load():
    import vllm_omni.experimental.fullduplex.minicpmo45.stage0 as stage0_runtime

    assert not hasattr(stage0_runtime, "MiniCPMO45FullModelDuplexRuntime")

    worker = _Worker(_SplitMiniCPMOStageModel("llm"))

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert getattr(target, "stage_role") == "llm"
    assert getattr(target, "owned_runtime", True) is False


def test_worker_minicpmo_stage0_uses_runner_context_forward_boundary():
    model = _SplitMiniCPMOStageModel("llm")
    calls = []

    def runner_forward(**kwargs):
        calls.append(kwargs)
        return {"logits": "logits", "hidden_states": "hidden"}

    worker = _Worker(model)
    worker.model_runner.supports_native_duplex_runner_context = True
    _mark_runner_context_contract(runner_forward)
    worker.model_runner.duplex_forward_with_runner_context = runner_forward

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert callable(getattr(model, "duplex_forward_with_runner_context", None))
    assert model.duplex_forward_with_runner_context(session_id="sid", inputs_embeds="embeds") == {
        "logits": "logits",
        "hidden_states": "hidden",
    }
    assert calls == [{"session_id": "sid", "inputs_embeds": "embeds"}]


def test_worker_minicpmo_stage0_runner_context_hook_is_contract_marked():
    model = _SplitMiniCPMOStageModel("llm")

    def runner_forward(**kwargs):
        return kwargs

    worker = _Worker(model)
    worker.model_runner.supports_native_duplex_runner_context = True
    _mark_runner_context_contract(runner_forward)
    worker.model_runner.duplex_forward_with_runner_context = runner_forward

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    attached = getattr(model, "duplex_forward_with_runner_context")
    assert target is not None
    assert getattr(attached, "uses_scheduler_metadata") is True
    assert getattr(attached, "uses_runner_kv_cache") is True
    assert getattr(attached, "vllm_omni_runner_context_contract") is True


def test_worker_minicpmo_stage0_replaces_model_local_runner_context_with_runner_hook():
    model = _SplitMiniCPMOStageModel("llm")
    calls = []

    def model_local_forward(**_kwargs):
        raise AssertionError("model-local hook must be replaced by runner hook")

    def runner_forward(**kwargs):
        calls.append(kwargs)
        return {"from_runner": True}

    model.duplex_forward_with_runner_context = model_local_forward
    worker = _Worker(model)
    worker.model_runner.supports_native_duplex_runner_context = True
    _mark_runner_context_contract(runner_forward)
    worker.model_runner.duplex_forward_with_runner_context = runner_forward

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert model.duplex_forward_with_runner_context(session_id="sid") == {"from_runner": True}
    assert calls == [{"session_id": "sid"}]
    assert getattr(model.duplex_forward_with_runner_context, "vllm_omni_runner_context_contract") is True


def test_worker_minicpmo_stage0_does_not_attach_uncontracted_runner_forward():
    model = _SplitMiniCPMOStageModel("llm")

    def runner_forward(**kwargs):
        del kwargs
        raise AssertionError("uncontracted runner forward must not be attached")

    worker = _Worker(model)
    worker.model_runner.duplex_forward_with_runner_context = runner_forward

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert not callable(getattr(model, "duplex_forward_with_runner_context", None))


def test_worker_minicpmo_stage0_does_not_attach_runner_forward_without_contract_marker():
    model = _SplitMiniCPMOStageModel("llm")

    def runner_forward(**kwargs):
        del kwargs
        raise AssertionError("runner forward without contract marker must not be attached")

    worker = _Worker(model)
    worker.model_runner.supports_native_duplex_runner_context = True
    runner_forward.uses_scheduler_metadata = True
    runner_forward.uses_runner_kv_cache = True
    worker.model_runner.duplex_forward_with_runner_context = runner_forward

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert not callable(getattr(model, "duplex_forward_with_runner_context", None))


def test_minicpmo_stage0_open_rejects_model_local_unmarked_runner_context_method():
    import pytest

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    def local_forward(**_kwargs):
        raise AssertionError("model-local duplex forward must not be trusted as runner/KV-backed")

    stage_model = SimpleNamespace(
        model_stage="llm",
        processor=SimpleNamespace(tokenizer=None),
        thinker=SimpleNamespace(),
        config=SimpleNamespace(_name_or_path="/tmp/minicpmo45"),
        duplex_forward_with_runner_context=local_forward,
    )
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, model_path="/tmp/minicpmo45", device="cpu")

    with pytest.raises(RuntimeError, match="runner-context contract"):
        runtime.open_duplex_session(session_id="sid-local-hook", session_config={})


def test_worker_minicpmo_stage_target_selection_does_not_pick_inner_plain_llm():
    model = _SplitMiniCPMOStageModel("llm")
    model.model.generate = lambda *args, **kwargs: None
    worker = _Worker(model)

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not model.model
    assert getattr(target, "stage_role") == "llm"
    assert getattr(target, "owned_runtime", True) is False


def test_worker_minicpmo_stage1_reuses_loaded_tts_stage_without_passive_fallback():
    worker = _Worker(_SplitMiniCPMOStageModel("tts"))

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert target is not None
    assert getattr(target, "stage_role") == "tts"
    assert getattr(target, "owned_runtime", True) is False


def test_worker_native_duplex_uses_provider_registry(monkeypatch):
    from vllm_omni.experimental.fullduplex.engine import worker as native_duplex

    class _RegisteredTarget:
        def open_duplex_session(self, **kwargs):
            return {"opened": kwargs["session_id"]}

    calls = []

    def provider(worker, capabilities):
        calls.append((worker, capabilities))
        return _RegisteredTarget()

    monkeypatch.setattr(native_duplex, "_DEFAULT_PROVIDERS_BOOTSTRAPPED", True)
    monkeypatch.setattr(native_duplex, "_NATIVE_DUPLEX_PROVIDERS", [provider])

    worker = _Worker(SimpleNamespace())
    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})

    assert isinstance(target, _RegisteredTarget)
    assert calls == [(worker, {"implementation_level": "model_native_duplex"})]
    assert not any(name.startswith("_maybe_load_minicpmo") for name in dir(OmniWorkerMixin))


def test_worker_mixin_does_not_guess_native_target_from_plain_model_methods():
    model = _PlainModelMethods()
    worker = _Worker(model, native_duplex_target=None)

    target = worker._get_native_duplex_target({"implementation_level": "model_native_duplex"})
    result = worker.open_duplex_session_async(
        "sid-no-provider",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={"instructions": "Be brief."},
    )

    assert target is None
    assert result["supported"] is False
    assert result["reason"] == "worker_duplex_session_not_implemented"
    assert model.prepare_calls == 0


def test_worker_module_does_not_export_legacy_native_method_adapter():
    from vllm_omni.experimental.fullduplex.engine import worker as native_duplex

    assert not hasattr(native_duplex, "NativeDuplexMethodAdapter")
    assert not hasattr(OmniWorkerMixin, "_NativeDuplexMethodAdapter")
    for helper_name in (
        "_decode_native_audio_payload",
        "_decode_native_ref_audio_from_config",
        "_as_native_result_dict",
        "_native_duplex_generate_kwargs",
        "_native_kv_cache_length",
    ):
        assert not hasattr(OmniWorkerMixin, helper_name)


def test_worker_rejects_incomplete_native_duplex_provider_contract():
    class _OpenOnlyTarget:
        def open_duplex_session(self, **kwargs):
            return {"opened": kwargs["session_id"]}

    worker = _Worker(SimpleNamespace(), native_duplex_target=_OpenOnlyTarget())

    result = worker.open_duplex_session_async(
        "sid-incomplete-provider",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )

    assert result["supported"] is False
    assert "missing required methods" in result["error"]
    assert "append_duplex_input" in result["error"]
    assert "signal_duplex_turn" in result["error"]
    assert "close_duplex_session" in result["error"]


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


def test_minicpmo_stage0_runtime_generates_tts_handoff_from_loaded_stage(monkeypatch):
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        eos_token_id = 2
        unk_token_id = -1
        all_special_ids = []
        all_special_tokens = []

        ids = {
            "<unit>": 3,
            "</unit>": 4,
            "<|listen|>": 5,
            "<|speak|>": 6,
            "<|chunk_eos|>": 7,
            "<|chunk_tts_eos|>": 8,
            "<|turn_eos|>": 9,
            "<|tts_pad|>": 0,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, 99)

        def encode(self, text, add_special_tokens=False):
            return [11]

        def decode(self, ids, skip_special_tokens=True):
            return "hello" if ids else ""

    class _Processor:
        tokenizer = _Tokenizer()

        def get_streaming_chunk_size(self):
            return 4

        def process_audio_streaming(self, audio, chunk_idx=0):
            return {
                "audio_features": torch.ones(1, 80, 4),
                "audio_feature_lens": [[torch.tensor(4)]],
            }

    class _Thinker:
        def __init__(self):
            self.embed_calls = []
            self.forward_inputs = []
            self.logit_ids = [6, 42, 7]

        def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
            ids = input_ids.reshape(-1).tolist()
            self.embed_calls.append(ids)
            return torch.tensor([[float(i), 0.0] for i in ids], dtype=torch.float32)

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            self.forward_inputs.append(inputs_embeds.detach().clone())
            return inputs_embeds, inputs_embeds.unsqueeze(0)

        def compute_logits(self, hidden_states):
            token_id = self.logit_ids.pop(0)
            logits = torch.full((1, 100), -1.0e9)
            logits[0, token_id] = 1.0
            return logits

    def runner_forward(*, session_id, inputs_embeds, context_len, previous_context_len, reset_kv):
        del session_id, previous_context_len, reset_kv
        stage_model.thinker.forward_inputs.append(inputs_embeds.detach().clone())
        logits = stage_model.thinker.compute_logits(inputs_embeds)
        return {
            "logits": logits,
            "hidden_states": inputs_embeds,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "kv_cache_length": int(context_len),
            "sampled_token_id": int(torch.argmax(logits, dim=-1).item()),
        }

    _mark_runner_context_contract(runner_forward)

    def model_local_duplex_method_must_not_run(*args, **kwargs):
        raise AssertionError("stage0 must use runner-context forward, not model-local duplex methods")

    stage_model = SimpleNamespace(
        model_stage="llm",
        processor=_Processor(),
        thinker=_Thinker(),
        config=SimpleNamespace(_name_or_path="/tmp/minicpmo45"),
        duplex_forward_with_runner_context=runner_forward,
        prepare=model_local_duplex_method_must_not_run,
        prefill=model_local_duplex_method_must_not_run,
        generate=model_local_duplex_method_must_not_run,
    )
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, model_path="/tmp/minicpmo45", device="cpu")
    runtime.open_duplex_session(session_id="sid-stage0", session_config={"instructions": "brief"})

    result = runtime.append_duplex_input(
        session_id="sid-stage0",
        mode="append_audio_chunk",
        payload={
            "audio": base64.b64encode(np.zeros(4, dtype=np.float32).tobytes()).decode("ascii"),
            "format": "pcm_f32le",
        },
    )

    assert result["stage_runtime_ready"] is True
    assert result["is_listen"] is False
    assert result["runtime_impl"] == "vllm_omni_minicpmo45_stage0_experimental_worker_runtime"
    assert result["uses_model_runner_scheduler"] is True
    assert result["runner_kv_backed"] is True
    assert result["experimental_worker_control_rpc"] is True
    assert result["experimental_eager_decoder"] is False
    assert result["per_step_tensor_handoff"] is False
    assert result["runner_local_payload_ref"] is True
    assert result["text"] == "hello"
    assert result["requires_stage_handoff"] is True
    assert result["stage_handoff"]["target_stage_role"] == "tts"
    assert result["stage_handoff"]["mode"] == "append_stage_handoff"
    handoff_payload = result["stage_handoff"]["payload"]
    from vllm_omni.data_entry_keys import deserialize_payload

    omni_payload = deserialize_payload(handoff_payload["omni_payload"])
    assert omni_payload["ids"]["output"] == [42]
    assert omni_payload["hidden_states"]["output"].shape == (1, 2)
    assert handoff_payload["llm_output_text"] == ["hello"]
    assert handoff_payload["meta"]["native_duplex_segment_text"] == "hello"
    assert "tts_token_ids" not in result
    assert "tts_hidden_states" not in result
    assert "omni_payload" not in result
    assert stage_model.thinker.forward_inputs


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
                "minicpmo45_native_duplex": True,
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
                "minicpmo45_native_duplex": True,
                "meta": {"native_duplex_segment_text": ""},
            }
        ],
    )

    assert int(output.multimodal_outputs["meta.turn_end"].item()) == 1


def test_minicpmo_stage0_decode_uses_runner_sampled_token():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    state = _MiniCPMO45Stage0SessionState(session_id="sid-decode")
    state.last_forward_metadata = {
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
        "sampled_token_id": 11,
    }

    assert runtime._decode_next_token(None, state) == 11


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
    runtime.session_config = {}
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
    runtime.session_config = {}
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
    runtime.session_config = {}
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
    runtime.session_config = {}
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
    runtime.session_config = {}
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
    runtime.session_config = {}
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


def test_minicpmo_stage0_context_window_preserves_system_prefix_and_recent_context():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.session_config = {}
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-window",
        session_config={
            "extra_body": {
                "stage0_context_max_tokens": 6,
                "stage0_context_previous_max_tokens": 4,
            }
        },
    )
    state.system_context_len = 2
    state.context_embeds = [torch.tensor([[float(i)]]) for i in range(10)]

    runtime._enforce_context_window(state)

    assert [int(embed.item()) for embed in state.context_embeds] == [0, 1, 6, 7, 8, 9]


def test_minicpmo_stage0_prefill_rolls_back_context_when_runner_forward_fails():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

        def duplex_forward_with_runner_context(self, **_kwargs):
            raise RuntimeError("runner forward failed")

    _mark_runner_context_contract(_StageModel.duplex_forward_with_runner_context)

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
        }.get(token, 0),
    )
    runtime.processor = SimpleNamespace(
        get_streaming_chunk_size=lambda: 4,
    )
    runtime.device = "cpu"
    runtime.session_config = {}
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-prefill-rollback")
    state.context_embeds = [runtime._embed_token(1)]
    state.audio_buffer = np.array([9.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="runner forward failed"):
        runtime._stage_prefill(state, np.zeros(4, dtype=np.float32))

    assert len(state.context_embeds) == 1
    assert state.audio_chunk_idx == 0
    assert state.pending_logits is None
    assert state.audio_buffer.tolist() == pytest.approx([9.0, 0.0, 0.0, 0.0, 0.0])


def test_minicpmo_stage0_open_requires_runner_context_by_default(monkeypatch):
    import pytest

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    stage_model = SimpleNamespace(
        model_stage="llm",
        processor=SimpleNamespace(tokenizer=None),
        thinker=SimpleNamespace(),
        config=SimpleNamespace(_name_or_path="/tmp/minicpmo45"),
    )
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, model_path="/tmp/minicpmo45", device="cpu")

    with pytest.raises(RuntimeError, match="duplex_forward_with_runner_context"):
        runtime.open_duplex_session(session_id="sid-no-runner", session_config={})


def test_minicpmo_stage0_forward_prefers_runner_context_hook():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.embed = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embed

        def duplex_forward_with_runner_context(self, **kwargs):
            self.calls.append(kwargs)
            hidden_states = kwargs["inputs_embeds"] + 1
            logits = torch.zeros(hidden_states.shape[0], 16)
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "kv_cache_length": kwargs["context_len"],
                "sampled_token_id": 5,
            }

    _mark_runner_context_contract(_StageModel.duplex_forward_with_runner_context)

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = None
    runtime.processor = None
    runtime.device = "cpu"
    runtime.session_config = {}
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-runner")
    state.context_embeds = [runtime._embed_token(1), runtime._embed_token(2)]

    logits, hidden_states = runtime._forward_context(state)

    assert logits.shape == (2, 16)
    assert hidden_states.shape == (2, 4)
    assert runtime.stage_model.calls[0]["session_id"] == "sid-runner"
    assert runtime.stage_model.calls[0]["context_len"] == 2
    assert state.last_forward_metadata == {
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
        "kv_cache_length": 2,
        "sampled_token_id": 5,
    }


def test_minicpmo_stage0_forward_appends_only_new_embeds_to_runner_kv():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.embed = torch.nn.Embedding(16, 4)

        def get_input_embeddings(self):
            return self.embed

        def duplex_forward_with_runner_context(self, **kwargs):
            self.calls.append(kwargs)
            hidden_states = kwargs["inputs_embeds"] + 1
            logits = torch.zeros(hidden_states.shape[0], 16)
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "kv_cache_length": kwargs["context_len"],
                "sampled_token_id": 5,
            }

    _mark_runner_context_contract(_StageModel.duplex_forward_with_runner_context)

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = None
    runtime.processor = None
    runtime.device = "cpu"
    runtime.session_config = {}
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-runner-delta")
    state.context_embeds = [runtime._embed_token(1), runtime._embed_token(2)]

    runtime._forward_context(state)
    state.context_embeds.append(runtime._embed_token(3))
    runtime._forward_context(state)

    first_call, second_call = runtime.stage_model.calls
    assert first_call["inputs_embeds"].shape[0] == 2
    assert first_call["previous_context_len"] == 0
    assert first_call["reset_kv"] is True
    assert second_call["inputs_embeds"].shape[0] == 1
    assert second_call["previous_context_len"] == 2
    assert second_call["reset_kv"] is False


def test_minicpmo_stage0_forward_rejects_runner_without_scheduler_kv_metadata():
    import pytest
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embed

        def duplex_forward_with_runner_context(self, **kwargs):
            hidden_states = kwargs["inputs_embeds"] + 1
            return {
                "logits": torch.zeros(hidden_states.shape[0], 16),
                "hidden_states": hidden_states,
                "uses_model_runner_scheduler": False,
                "runner_kv_backed": False,
            }

    _mark_runner_context_contract(_StageModel.duplex_forward_with_runner_context)

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = None
    runtime.processor = None
    runtime.device = "cpu"
    runtime.session_config = {}
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-runner-unbacked")
    state.context_embeds = [runtime._embed_token(1)]

    with pytest.raises(RuntimeError, match="scheduler/KV-backed"):
        runtime._forward_context(state)


def test_minicpmo_stage0_decode_requires_runner_sampled_token():
    import pytest
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.session_config = {}
    runtime.listen_token_id = 5
    state = _MiniCPMO45Stage0SessionState(session_id="sid-runner-sample")
    state.last_forward_metadata = {
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
    }

    with pytest.raises(RuntimeError, match="sampled_token_id"):
        runtime._decode_next_token(torch.zeros(1, 16), state)


def test_minicpmo_stage0_forward_rejects_unscheduled_vllm_forward():
    import pytest
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, *args, **kwargs):
            raise AssertionError("unscheduled vLLM forward must not be used")

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = None
    runtime.processor = None
    runtime.device = "cpu"
    runtime.session_config = {}
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-no-runner")
    state.context_embeds = [runtime._embed_token(1)]

    with pytest.raises(RuntimeError, match="without scheduler attention metadata"):
        runtime._forward_context(state)


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


def test_minicpmo_stage1_runtime_keys_and_resets_tts_stream_by_session():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _Talker:
        def __init__(self):
            self.forward_infos = []
            self.finished = []

        def forward(self, additional_information=None, **kwargs):
            self.forward_infos.append(dict(additional_information or {}))
            return None, torch.tensor([0.1, -0.1], dtype=torch.float32)

        def on_requests_finished(self, finished_req_ids):
            self.finished.append(list(finished_req_ids))

    talker = _Talker()
    runtime = MiniCPMO45Stage1DuplexRuntime(
        SimpleNamespace(model_stage="tts", talker=talker),
        model_path="/tmp/minicpmo45",
        device="cpu",
    )
    runtime.open_duplex_session(session_id="sid-stage1", session_config={})

    result = runtime.append_duplex_input(
        session_id="sid-stage1",
        mode="append_stage_handoff",
        payload=_minicpmo_tts_handoff_payload(
            torch,
            token_ids=[42],
            hidden=torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        ),
    )
    runtime.signal_duplex_turn(session_id="sid-stage1", event="barge_in")
    runtime.close_duplex_session(session_id="sid-stage1", reason="session_close")

    assert result["stage_runtime_ready"] is True
    assert result["audio_waveform"].tolist() == pytest.approx([0.1, -0.1])
    assert talker.forward_infos[0]["global_request_id"] == "sid-stage1"
    assert talker.forward_infos[0]["request_id"] == "sid-stage1"
    assert talker.finished == [["sid-stage1"], ["sid-stage1"]]
    assert "sid-stage1" not in runtime.sessions


def test_minicpmo_stage1_runtime_prefers_loaded_stage_forward_over_inner_talker():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _Talker:
        def forward(self, additional_information=None, **kwargs):
            raise AssertionError("stage1 native duplex must use the loaded stage forward first")

    class _StageModel:
        model_stage = "tts"

        def __init__(self):
            self.talker = _Talker()
            self.forward_infos = []

        def forward(self, additional_information=None, **kwargs):
            self.forward_infos.append(dict(additional_information or {}))
            return None, torch.tensor([0.2, -0.2], dtype=torch.float32)

    stage_model = _StageModel()
    runtime = MiniCPMO45Stage1DuplexRuntime(stage_model, model_path="/tmp/minicpmo45", device="cpu")
    runtime.open_duplex_session(session_id="sid-stage1-forward", session_config={})

    result = runtime.append_duplex_input(
        session_id="sid-stage1-forward",
        mode="append_stage_handoff",
        payload=_minicpmo_tts_handoff_payload(
            torch,
            token_ids=[42],
            hidden=torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        ),
    )

    assert result["audio_waveform"].tolist() == pytest.approx([0.2, -0.2])
    assert stage_model.forward_infos[0]["request_id"] == "sid-stage1-forward"


def test_minicpmo_stage1_runtime_squeezes_handoff_hidden_states():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _Talker:
        def __init__(self):
            self.forward_infos = []

        def forward(self, additional_information=None, **kwargs):
            self.forward_infos.append(dict(additional_information or {}))
            return None, torch.tensor([0.25], dtype=torch.float32)

    talker = _Talker()
    runtime = MiniCPMO45Stage1DuplexRuntime(
        SimpleNamespace(model_stage="tts", talker=talker),
        model_path="/tmp/minicpmo45",
        device="cpu",
    )
    runtime.open_duplex_session(session_id="sid-stage1-shape", session_config={})

    result = runtime.append_duplex_input(
        session_id="sid-stage1-shape",
        mode="append_stage_handoff",
        payload=_minicpmo_tts_handoff_payload(
            torch,
            token_ids=[101, 102],
            hidden=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
            text="ok",
        ),
    )

    info = talker.forward_infos[0]
    assert info["tts_token_ids"].shape == (2,)
    assert info["tts_hidden_states"].shape == (2, 3)
    assert result["tts_token_shape"] == [2]
    assert result["tts_hidden_shape"] == [2, 3]
    assert result["waveform_numel"] == 1


def test_minicpmo_stage1_runtime_rejects_legacy_direct_tts_tensor_payload():
    import pytest
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _Talker:
        def __init__(self):
            self.forward_infos = []

        def forward(self, additional_information=None, **kwargs):
            self.forward_infos.append(dict(additional_information or {}))
            return None, torch.tensor([0.25], dtype=torch.float32)

    talker = _Talker()
    runtime = MiniCPMO45Stage1DuplexRuntime(
        SimpleNamespace(model_stage="tts", talker=talker),
        model_path="/tmp/minicpmo45",
        device="cpu",
    )
    runtime.open_duplex_session(session_id="sid-stage1-compact", session_config={})

    with pytest.raises(ValueError, match="omni_payload"):
        runtime.append_duplex_input(
            session_id="sid-stage1-compact",
            mode="append_stage_handoff",
            payload={
                "tts_token_ids": [42],
                "tts_hidden_states": [[0.1, 0.2, 0.3]],
                "llm_output_text": ["hello"],
                "end_of_turn": False,
            },
        )
    assert talker.forward_infos == []


def test_minicpmo_stage1_runtime_accepts_omni_payload_handoff():
    import torch

    from vllm_omni.data_entry_keys import serialize_payload
    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _Talker:
        def __init__(self):
            self.forward_infos = []

        def forward(self, additional_information=None, **kwargs):
            self.forward_infos.append(dict(additional_information or {}))
            return None, torch.tensor([0.5], dtype=torch.float32)

    hidden = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    talker = _Talker()
    runtime = MiniCPMO45Stage1DuplexRuntime(
        SimpleNamespace(model_stage="tts", talker=talker),
        model_path="/tmp/minicpmo45",
        device="cpu",
    )
    runtime.open_duplex_session(session_id="sid-stage1-omni-payload", session_config={})

    result = runtime.append_duplex_input(
        session_id="sid-stage1-omni-payload",
        mode="append_stage_handoff",
        payload={
            "omni_payload": serialize_payload(
                {
                    "ids": {"output": [42]},
                    "hidden_states": {"output": hidden},
                }
            ),
            "llm_output_text": ["hello"],
            "end_of_turn": False,
        },
    )

    info = talker.forward_infos[0]
    assert info["tts_token_ids"].shape == (1,)
    assert info["tts_hidden_states"].shape == (1, 3)
    assert torch.equal(info["tts_token_ids"], torch.tensor([42]))
    assert torch.allclose(info["tts_hidden_states"], hidden)
    assert result["waveform_numel"] == 1


def test_minicpmo_stage1_runtime_resolves_omni_payload_ref_from_payload_store():
    import torch

    from vllm_omni.data_entry_keys import serialize_payload
    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    hidden = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    runtime = MiniCPMO45Stage1DuplexRuntime.__new__(MiniCPMO45Stage1DuplexRuntime)
    runtime._duplex_stage_payload_store = {
        "sid:1": {
            "omni_payload": serialize_payload(
                {
                    "ids": {"output": [42]},
                    "hidden_states": {"output": hidden},
                }
            ),
            "llm_output_text": ["hello"],
        }
    }

    info = runtime._normalize_handoff_payload(
        {
            "type": "minicpmo45_tts_handoff_ref",
            "omni_payload_ref": "sid:1",
        },
        session_id="sid",
    )

    assert "sid:1" not in runtime._duplex_stage_payload_store
    assert info["tts_token_ids"].tolist() == [42]
    assert torch.allclose(info["tts_hidden_states"], hidden)
    assert info["llm_output_text"] == ["hello"]


def test_minicpmo_stage1_runtime_prefers_runner_local_payload_cache_for_payload_ref():
    import torch

    from vllm_omni.data_entry_keys import serialize_payload
    from vllm_omni.experimental.fullduplex.minicpmo45.stage1 import (
        MiniCPMO45Stage1DuplexRuntime,
    )

    class _PayloadCache:
        def __init__(self, payload):
            self.payloads = {"sid:runner": payload}

        def pop_local_stage_payload(self, payload_ref):
            return self.payloads.pop(payload_ref, None)

    hidden = torch.tensor([[0.5, 0.6]], dtype=torch.float32)
    payload = {
        "omni_payload": serialize_payload(
            {
                "ids": {"output": [7]},
                "hidden_states": {"output": hidden},
            }
        ),
        "llm_output_text": ["runner-cache"],
    }
    runtime = MiniCPMO45Stage1DuplexRuntime.__new__(MiniCPMO45Stage1DuplexRuntime)
    runtime._duplex_stage_payload_store = {
        "sid:runner": {"omni_payload": "stale-local-store"},
    }
    runtime._duplex_stage_payload_cache = _PayloadCache(payload)

    info = runtime._normalize_handoff_payload(
        {
            "type": "minicpmo45_tts_handoff_ref",
            "omni_payload_ref": "sid:runner",
        },
        session_id="sid",
    )

    assert info["tts_token_ids"].tolist() == [7]
    assert torch.allclose(info["tts_hidden_states"], hidden)
    assert info["llm_output_text"] == ["runner-cache"]
    assert runtime._duplex_stage_payload_store["sid:runner"]["omni_payload"] == "stale-local-store"


def test_worker_put_duplex_stage_payload_stages_runtime_payload_and_runner_cache():
    from vllm_omni.worker.mixins import OmniWorkerMixin

    class _Runtime:
        def __init__(self):
            self.calls = []

        def put_duplex_stage_payload(self, **kwargs):
            self.calls.append(kwargs)
            return {"payload_cached": True, "payload_ref": kwargs["payload_ref"]}

    class _Runner:
        def __init__(self):
            self.payloads = {}

        def put_local_stage_payload(self, req_id, payload):
            self.payloads[req_id] = payload

    class _Worker(OmniWorkerMixin):
        def __init__(self):
            self.model_runner = _Runner()
            self._omni_native_duplex_sessions = {"sid": _Runtime()}

    worker = _Worker()
    payload = {"omni_payload": "serialized"}

    result = worker.put_duplex_stage_payload_async(
        "sid",
        epoch=0,
        seq=1,
        payload_ref="sid:0:1:stage1",
        payload=payload,
    )

    assert result["supported"] is True
    assert result["native_result"]["payload_cached"] is True
    assert worker._omni_native_duplex_sessions["sid"].calls[0]["payload"] == payload
    assert worker.model_runner.payloads["sid:0:1:stage1"] == payload


def test_worker_native_duplex_uses_explicit_provider_lifecycle():
    target = _ExplicitNativeDuplexTarget()
    worker = _Worker(SimpleNamespace(), native_duplex_target=target)

    opened = worker.open_duplex_session_async(
        "sid-explicit",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={"instructions": "Be brief."},
    )
    appended = worker.append_duplex_input_async(
        "sid-explicit",
        epoch=0,
        seq=1,
        mode="append_audio_chunk",
        payload={"audio": "AAAA", "format": "pcm_f32le"},
        final=False,
    )
    signalled = worker.signal_duplex_turn_async(
        "sid-explicit",
        epoch=0,
        event="input.commit",
        payload={"reason": "test"},
    )
    closed = worker.close_duplex_session_async(
        "sid-explicit",
        epoch=0,
        reason="session_close",
    )

    assert opened["supported"] is True
    assert target.open_calls[0]["session_config"] == {"instructions": "Be brief."}
    assert appended["native_result"]["text"] == "explicit"
    assert appended["native_result"]["audio_data"] == base64.b64encode(
        np.array([0.25, -0.25], dtype=np.float32).tobytes()
    ).decode("ascii")
    assert appended["native_result"]["cost_all_ms"] == 2.0
    assert signalled["native_result"]["event"] == "input.commit"
    assert closed["supported"] is True
    assert "sid-explicit" not in worker._native_duplex_sessions()


def test_worker_native_duplex_rejects_second_session_while_runtime_busy():
    target = _ExplicitNativeDuplexTarget()
    worker = _Worker(SimpleNamespace(), native_duplex_target=target)

    first = worker.open_duplex_session_async(
        "sid-a",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )
    second = worker.open_duplex_session_async(
        "sid-b",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )

    assert first["supported"] is True
    assert second["supported"] is False
    assert second["reason"] == "native_duplex_session_busy"
    assert second["active_session_ids"] == ["sid-a"]
    assert len(target.open_calls) == 1


def test_worker_native_duplex_close_failure_keeps_target_for_retry():
    target = _ExplicitNativeDuplexTarget(fail_close=True)
    worker = _Worker(SimpleNamespace(), native_duplex_target=target)
    worker.open_duplex_session_async(
        "sid-close-fails",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )

    result = worker.close_duplex_session_async(
        "sid-close-fails",
        epoch=0,
        reason="session_close",
    )

    assert result["supported"] is False
    assert "close failed" in result["error"]
    assert "sid-close-fails" in worker._native_duplex_sessions()


def test_worker_native_duplex_rejects_ref_audio_path_before_open():
    target = _ExplicitNativeDuplexTarget()
    worker = _Worker(SimpleNamespace(), native_duplex_target=target)

    result = worker.open_duplex_session_async(
        "sid-untrusted-ref",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={"extra_body": {"ref_audio_path": "/tmp/ref.wav"}},
    )

    assert result["supported"] is False
    assert "ref_audio_path is not accepted" in result["error"]
    assert target.open_calls == []


def test_worker_native_duplex_does_not_fall_back_to_loaded_model_signal():
    class _SignalModel:
        def __init__(self):
            self.signals = []

        def signal_duplex_turn(self, **kwargs):
            self.signals.append(kwargs)
            return {"event": kwargs["event"]}

    model = _SignalModel()
    worker = _Worker(model, native_duplex_target=None)

    result = worker.signal_duplex_turn_async(
        "sid-no-session",
        epoch=2,
        event="input.cancel",
        payload={"reason": "test"},
    )

    assert result["supported"] is False
    assert result["reason"] == "worker_duplex_signal_not_implemented"
    assert model.signals == []


def test_minicpmo_transformers_cache_compat_supports_legacy_indexing():
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    from vllm_omni.experimental.fullduplex.minicpmo45.worker import (
        patch_minicpmo_transformers_compat,
    )

    patch_minicpmo_transformers_compat()

    dynamic_cache = DynamicCache()
    dynamic_cache.key_cache.append(np.zeros((1, 1, 2, 3), dtype=np.float32))
    dynamic_cache.value_cache.append(np.ones((1, 1, 2, 3), dtype=np.float32))
    assert dynamic_cache[0][0].shape == (1, 1, 2, 3)
    assert dynamic_cache[0][1].sum() == 6

    encoder_decoder_cache = EncoderDecoderCache(dynamic_cache, DynamicCache())
    assert encoder_decoder_cache[0][0].shape == (1, 1, 2, 3)
    assert encoder_decoder_cache.key_cache[0].shape == (1, 1, 2, 3)


def test_minicpmo_remote_config_patch_handles_nested_and_dict_configs():
    from vllm_omni.experimental.fullduplex.minicpmo45.worker import (
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
    state = _MiniCPMO45Stage0SessionState(session_id="sid")

    result = runtime._stage_prefill(state, np.zeros(1600, dtype=np.float32))

    assert result["success"] is False
    assert result["reason"]
    assert len(state.audio_buffer) >= 1600
    assert state.context_embeds == []
    assert state.pending_logits is None


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


def test_minicpmo_stage0_runtime_does_not_cut_at_punctuation():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Tokenizer:
        text = {
            200: "我",
            201: "喜",
            202: "欢",
            203: "，",
            204: "坏",
        }

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime.speak_token_id = 151706
    runtime.listen_token_id = 151705
    runtime.chunk_eos_token_id = 151718
    runtime.chunk_tts_eos_token_id = 151721
    runtime.turn_eos_token_id = 151717
    runtime.unit_token_id = 151683
    runtime.unit_end_token_id = 151684
    runtime.tts_bos_token_id = 151703
    runtime.tts_eos_token_id = 151704
    runtime.tts_pad_token_id = 151722
    runtime.chunk_terminator_token_ids = [
        runtime.listen_token_id,
        runtime.chunk_eos_token_id,
        runtime.chunk_tts_eos_token_id,
    ]
    runtime.turn_terminator_token_ids = [runtime.turn_eos_token_id]
    runtime.chunk_speak_token_ids = [runtime.speak_token_id]
    runtime._embed_token = lambda token_id: torch.tensor([[float(token_id), 0.0]])
    runtime._append_token_and_forward = lambda state, token_id: (torch.tensor([[float(token_id), 1.0]]), object())

    remaining = [runtime.speak_token_id, 200, 201, 202, 203, 204, runtime.chunk_eos_token_id]

    def _decode_next_token(_logits, _state):
        token_id = remaining.pop(0)
        return token_id

    runtime._decode_next_token = _decode_next_token
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-runtime-boundary",
        session_config={
            "extra_body": {
                "min_new_speak_tokens_before_chunk_boundary": 4,
                "max_new_speak_tokens_per_chunk": 64,
            }
        },
        pending_logits=object(),
    )

    result = runtime._stage_generate(state, force_listen=False)

    assert result["is_listen"] is False
    assert result["text"] == "我喜欢，坏"
    assert result["n_tokens"] == 5
    assert remaining == []


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
    model._minicpmo45_duplex_row_sessions = {0: "sid-native"}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={"sid-native": state})
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
    model._minicpmo45_duplex_row_sessions = {0: "sid-native"}
    model._minicpmo45_duplex_row_payloads = {0: {"force_listen": True}}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={"sid-native": state})
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

    rows = model._native_duplex_prompt_rows(
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
    assert state.system_context_len == 6
