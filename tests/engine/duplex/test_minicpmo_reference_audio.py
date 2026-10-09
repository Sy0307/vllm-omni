# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def plugin(monkeypatch):
    tokenizer = SimpleNamespace(
        all_special_ids=[90],
        unk_token_id=-1,
        eos_token_id=99,
        encode=lambda text, add_special_tokens=False: [ord(char) for char in text],
        convert_tokens_to_ids=lambda token: {"<|listen|>": 91}.get(token, -1),
    )
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)
    return module.MiniCPMO45DuplexPlugin(lambda *args: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [("audio",), ("text", "audio"), ("text",)])
async def test_session_without_reference_skips_voice_resolution(plugin, mocker, modalities):
    resolve = mocker.patch.object(module, "resolve_ref_audio", new_callable=mocker.AsyncMock)
    config = DuplexSessionConfig(modalities=modalities, instructions="Streaming Omni Conversation.")

    runtime = await plugin.prepare_runtime_config(config, model_config=None)

    resolve.assert_not_awaited()
    assert config.ref_audio is None
    assert not any(key.startswith("ref_audio") for key in runtime)
    assert runtime["duplex_first_append_context_tokens"] == len(
        "<|im_start|>system\nStreaming Omni Conversation.<|im_end|>"
    )


@pytest.mark.asyncio
async def test_explicit_reference_keeps_audio_conditioning_and_exact_reserve(plugin, mocker):
    resolve = mocker.patch.object(
        module,
        "resolve_ref_audio",
        new_callable=mocker.AsyncMock,
        return_value=(np.zeros(4801, dtype=np.float32), 16000),
    )
    config = DuplexSessionConfig(modalities=("audio",), ref_audio="data:audio/wav;base64,AAA=")

    runtime = await plugin.prepare_runtime_config(config, model_config=None)

    resolve.assert_awaited_once_with("data:audio/wav;base64,AAA=", model_config=None)
    assert runtime["ref_audio_format"] == "pcm_f32le"
    assert runtime["ref_audio_sample_rate_hz"] == 16000
    assert len(module.b64decode(runtime["ref_audio_data"])) == 4800 * 4
    assert config.ref_audio is None
    assert (
        runtime["duplex_first_append_context_tokens"]
        == len("<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|><|audio_end|><|im_end|>") + 3
    )
