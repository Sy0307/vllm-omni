# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.metrics.stats import OrchestratorAggregator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("output_kind", [RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA])
def test_stage_pool_resumable_metrics_do_not_recount_prior_segments(monkeypatch, output_kind) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    params = SamplingParams(output_kind=output_kind)
    totals = []
    for segment in range(1, 4):
        now = 100.0 + segment
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        frames = 24000 * (segment if output_kind == RequestOutputKind.CUMULATIVE else 1)
        output = SimpleNamespace(
            request_id="req-stream",
            multimodal_output={"audio": SimpleNamespace(shape=(frames,)), "sample_rate": 24000},
            outputs=[],
        )
        pool.record_output_timestamps([output], output_ts=now)
        metrics = pool.build_stage_metrics(
            [output], submit_ts=100.0, request_timestamp=100.0, replica_id=0, sampling_params=params
        )
        totals.append(metrics)

    assert [m.audio_generated_frames for m in totals] == [24000, 24000, 24000]
    assert [m.output_unit_count for m in totals] == [24000, 24000, 24000]
    assert sum(m.audio_duration_s for m in totals) == pytest.approx(3.0)
    assert sum(m.stage_gen_time_ms for m in totals) == pytest.approx(3000.0)


def test_stage_pool_cumulative_audio_metrics_release_request_watermark(monkeypatch) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    params = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
    output = SimpleNamespace(
        request_id="req-stream",
        multimodal_output={"audio": SimpleNamespace(shape=(24000,)), "sample_rate": 24000},
        outputs=[],
    )
    durations = []
    for now in [101.0, 102.0]:
        monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: now)
        pool.record_output_timestamps([output], output_ts=now)
        metrics = pool.build_stage_metrics(
            [output], submit_ts=100.0, request_timestamp=100.0, replica_id=0, sampling_params=params
        )
        durations.append(metrics.audio_duration_s)
    assert durations == [1.0, 0.0]

    pool.release_binding("req-stream")
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: 201.0)
    pool.record_output_timestamps([output], output_ts=201.0)
    restarted = pool.build_stage_metrics(
        [output], submit_ts=200.0, request_timestamp=200.0, replica_id=0, sampling_params=params
    )
    assert restarted.audio_duration_s == pytest.approx(1.0)
    assert restarted.stage_gen_time_ms == pytest.approx(1000.0)


def test_duplex_response_audio_frames_match_serialized_stage_duration(monkeypatch) -> None:
    pool = StagePool(
        2,
        [SimpleNamespace(stage_type="llm", final_output=True, final_output_type="audio")],
        output_processor=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    session = DuplexEngineSession(session_id="test", config=DuplexSessionConfig(model="test"), num_stages=3)
    session.begin_response()
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time.time", lambda: 101.0)
    for index in range(2):
        output = SimpleNamespace(
            request_id=f"req-{index}",
            multimodal_output={"audio": SimpleNamespace(shape=(24000,)), "sample_rate": 24000},
            outputs=[],
        )
        pool.record_output_timestamps([output], output_ts=101.0)
        metrics = pool.build_stage_metrics(
            [output],
            submit_ts=100.0,
            request_timestamp=100.0,
            replica_id=0,
            sampling_params=SamplingParams(output_kind=RequestOutputKind.CUMULATIVE),
        )
        snapshot = OrchestratorAggregator._merge_stage_metric_event(None, metrics)
        total = session.accumulate_response_stage_metrics({"2": snapshot})["2"]
    assert total["audio_frames"] == 48000
    assert total["output_unit_count"] == 48000
    assert total["audio_duration_s"] == pytest.approx(total["audio_frames"] / total["audio_sample_rate"])
