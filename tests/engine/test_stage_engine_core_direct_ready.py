from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.engine import stage_engine_core_proc as core_proc_module


def test_uniproc_tp1_binds_native_data_plane_to_scheduler_inbox() -> None:
    plane = SimpleNamespace(set_omni_connector_output_sink=MagicMock())
    runner = SimpleNamespace(_omni_data_plane=plane)
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    )
    scheduler = SimpleNamespace(enqueue_omni_connector_output=MagicMock())

    assert core_proc_module._bind_native_data_plane_ready_sink(executor, scheduler)

    plane.set_omni_connector_output_sink.assert_called_once_with(scheduler.enqueue_omni_connector_output)


def test_uniproc_tp2_keeps_output_carried_ready_fallback() -> None:
    plane = SimpleNamespace(set_omni_connector_output_sink=MagicMock())
    runner = SimpleNamespace(_omni_data_plane=plane)
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1)
    )
    scheduler = SimpleNamespace(enqueue_omni_connector_output=MagicMock())

    assert not core_proc_module._bind_native_data_plane_ready_sink(executor, scheduler)
    plane.set_omni_connector_output_sink.assert_not_called()
