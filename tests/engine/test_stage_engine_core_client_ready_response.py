from types import SimpleNamespace

import msgspec

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient


def test_ready_response_fallback_decodes_schema_drift(monkeypatch):
    def fail_upstream(self, payload):
        raise msgspec.ValidationError("Object missing required field `block_size`")

    monkeypatch.setattr(
        "vllm_omni.engine.stage_engine_core_client.MPClient._apply_ready_response",
        fail_upstream,
    )

    client = object.__new__(StageEngineCoreClient)
    client.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=65536),
        cache_config=SimpleNamespace(num_gpu_blocks=7),
    )
    client.stats_update_address = None

    payload = msgspec.msgpack.encode(
        {
            "max_model_len": 32768,
            "num_gpu_blocks": 11,
            "dp_stats_address": "tcp://127.0.0.1:1234",
            "kv_cache_config": {"num_blocks": 1},
        }
    )

    client._apply_ready_response(payload)

    assert client.vllm_config.model_config.max_model_len == 32768
    assert client.vllm_config.cache_config.num_gpu_blocks == 18
    assert client.stats_update_address == "tcp://127.0.0.1:1234"
