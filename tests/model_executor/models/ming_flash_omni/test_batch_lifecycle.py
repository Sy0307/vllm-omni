# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auxiliary capture runs after eval and profiling does not start a producer."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration as Talker,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_batch import MingTalkerBatchEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_capture_requires_eval_and_release_closes_idle_runtime(mocker):
    engine = mocker.Mock()
    model = SimpleNamespace(
        training=True,
        device=torch.device("cuda"),
        _batch_rows=2,
        _batch_engine=None,
        _batch_vae=None,
        _stream_jobs={},
        _build_batch_engine=mocker.Mock(return_value=engine),
    )
    with pytest.raises(RuntimeError, match="eval"):
        Talker.capture_auxiliary_graphs(model)
    model._build_batch_engine.assert_not_called()
    model.training = False
    Talker.capture_auxiliary_graphs(model)
    Talker.capture_auxiliary_graphs(model)
    model._build_batch_engine.assert_called_once()
    engine.start.assert_not_called()
    model._stream_jobs["active"] = object()
    with pytest.raises(RuntimeError, match="active requests"):
        Talker.release_auxiliary_graphs(model)
    model._stream_jobs.clear()
    Talker.release_auxiliary_graphs(model)
    engine.close.assert_called_once()
    assert model._batch_engine is None and model._batch_vae is None


def test_idle_engine_start_is_idempotent_and_close_joins_thread(mocker):
    from threading import Condition

    engine = object.__new__(MingTalkerBatchEngine)
    engine.closed = False
    engine.thread = None
    engine.cond = Condition()
    engine.stream = mocker.Mock()

    def idle():
        with engine.cond:
            engine.cond.wait_for(lambda: engine.closed)

    engine._run = idle
    engine.start()
    thread = engine.thread
    assert thread is not None
    engine.start()
    assert engine.thread is thread
    engine.close()
    assert not thread.is_alive()
    engine.stream.synchronize.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        engine.start()
