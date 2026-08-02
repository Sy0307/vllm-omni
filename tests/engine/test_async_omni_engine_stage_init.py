import contextlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import types

import pytest

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import (
    LogicalStageInitPlan,
    ReplicaInitPlan,
    build_stage0_input_processor,
    compute_replica_layout,
)
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_llm_metadata(
    stage_id: int,
    *,
    replica_id: int = 0,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime_cfg={},
        prompt_expand_func=None,
        final_output=final_output,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"sp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[] if stage_id == 0 else [stage_id - 1],
        engine_output_type="token_ids",
        replica_id=replica_id,
        is_comprehension=is_comprehension,
    )


def _make_diffusion_metadata(stage_id: int, *, replica_id: int = 0, final_output_type: str = "image"):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="diffusion",
        runtime_cfg={"devices": str(replica_id)},
        prompt_expand_func=None,
        final_output=True,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"dsp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[],
        cfg_kv_collect_func=None,
        replica_id=replica_id,
    )


def _make_llm_plan(
    stage_idx: int,
    *,
    stage_id: int,
    vllm_config: object,
    num_replicas: int = 1,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=stage_id,
            stage_type="llm",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_llm_metadata(
                    stage_id,
                    replica_id=replica_id,
                    final_output=final_output,
                    final_output_type=final_output_type,
                    is_comprehension=is_comprehension and replica_id == 0,
                ),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
                stage_vllm_config=vllm_config,
                executor_class=object,
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=replicas,
    )


def _make_diffusion_plan(
    stage_idx: int,
    *,
    stage_id: int,
    num_replicas: int = 1,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=stage_id,
            stage_type="diffusion",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_diffusion_metadata(stage_id, replica_id=replica_id),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=replicas,
    )


def test_stage_engine_core_client_module_reload_keeps_forward_refs_deferred():
    """Regression test for forward references in make_async_mp_client."""
    import vllm_omni.engine.stage_engine_core_client as client_mod

    importlib.reload(client_mod)

    assert client_mod.StageEngineCoreClientBase.make_async_mp_client.__annotations__["return"] == (
        "StageEngineCoreClient | DPLBStageEngineCoreClient"
    )


def test_compute_replica_layout_splits_diffusion_devices_by_world_size():
    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={"parallel_config": {"tensor_parallel_size": 2}},
        runtime={"devices": "0,1,2,3", "num_replicas": 2},
    )

    replicas_per_stage, replica_devices_map = compute_replica_layout([stage_cfg])

    assert replicas_per_stage == [2]
    assert replica_devices_map == {0: ["0,1", "2,3"]}


def test_collect_initialized_clients_for_cleanup_deduplicates_clients():
    shared = types.SimpleNamespace(name="shared")
    extra = types.SimpleNamespace(name="extra")

    cleanup_clients = StageRuntime._collect_initialized_clients_for_cleanup(
        stage_pools=[types.SimpleNamespace(clients=[shared, None])],
        initialized_clients_by_stage={0: [shared], 1: [extra]},
    )

    assert cleanup_clients == [shared, extra]


def test_initialize_local_diffusion_replica_restores_device_visibility_after_local_init(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.engine.stage_engine_startup import StageReplicaResources
    from vllm_omni.platforms import current_omni_platform

    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    plan = _make_diffusion_plan(0, stage_id=0).replicas[0]

    env_var = current_omni_platform.device_control_env_var
    old_env = os.environ.get(env_var)
    os.environ[env_var] = "0,1"
    runtime._init_visible_devices_baseline = "0,1"

    monkeypatch.setattr(runtime_mod, "inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(
        runtime_mod,
        "launch_diffusion_stage_replica",
        lambda **_: (types.SimpleNamespace(), StageReplicaResources()),
    )

    try:
        runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=1)
        assert os.environ.get(env_var) == "0,1"
    finally:
        if old_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_env


def test_initialize_local_diffusion_replica_applies_and_restores_stage_env(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.engine.stage_engine_startup import StageReplicaResources

    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )
    plan = _make_diffusion_plan(0, stage_id=0).replicas[0]
    plan.metadata.runtime_cfg["env"] = {"OMNI_TEST_STAGE_ENV": "diffusion"}
    observed = []

    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)
    monkeypatch.setattr(runtime_mod, "inject_kv_stage_info", lambda *_: None)

    def _capture_launch(**_kwargs):
        observed.append(os.environ.get("OMNI_TEST_STAGE_ENV"))
        return types.SimpleNamespace(), StageReplicaResources()

    monkeypatch.setattr(
        runtime_mod,
        "launch_diffusion_stage_replica",
        _capture_launch,
    )

    runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=1)

    assert observed == ["diffusion"]
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_scoped_spawn_env_applies_device_and_stage_env_under_one_lock(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env
    from vllm_omni.platforms import current_omni_platform

    device_env = current_omni_platform.device_control_env_var
    monkeypatch.setenv(device_env, "baseline")
    monkeypatch.setenv("OMNI_TEST_STAGE_ENV", "baseline")
    observations = []
    spawn_lock = threading.Lock()

    def _observe(stage_id, device, value):
        with scoped_spawn_device_env(
            device,
            spawn_lock,
            stage_id=stage_id,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": value}},
        ):
            observations.append(
                (
                    stage_id,
                    os.environ.get(device_env),
                    os.environ.get("OMNI_TEST_STAGE_ENV"),
                )
            )
            time.sleep(0.01)

    threads = [
        threading.Thread(target=_observe, args=(0, "0", "stage0")),
        threading.Thread(target=_observe, args=(1, "1", "stage1")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(observations) == [
        (0, "0", "stage0"),
        (1, "1", "stage1"),
    ]
    assert os.environ[device_env] == "baseline"
    assert os.environ["OMNI_TEST_STAGE_ENV"] == "baseline"


def test_scoped_spawn_env_serializes_distinct_stage_runtime_locks(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env

    runtimes = [
        StageRuntime(
            stage_configs=[],
            model="dummy-model",
            config_path="dummy-config",
            stage_init_timeout=1,
            diffusion_batch_size=1,
            async_chunk=False,
        )
        for _ in range(2)
    ]
    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def _hold_first_scope():
        with scoped_spawn_device_env(
            "0",
            runtimes[0]._spawn_device_lock,
            stage_id=0,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "stage0"}},
        ):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def _enter_second_scope():
        assert first_entered.wait(timeout=2)
        with scoped_spawn_device_env(
            "1",
            runtimes[1]._spawn_device_lock,
            stage_id=1,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "stage1"}},
        ):
            second_entered.set()

    first_thread = threading.Thread(target=_hold_first_scope)
    second_thread = threading.Thread(target=_enter_second_scope)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()
    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_headless_spawn_env_shares_process_global_lock(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import (
        launch_headless_replica_group,
        scoped_spawn_device_env,
    )

    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)
    headless_entered = threading.Event()
    release_headless = threading.Event()
    colocated_entered = threading.Event()

    class _Manager:
        def shutdown(self):
            pass

    def _launch_headless(_replica_id):
        headless_entered.set()
        assert release_headless.wait(timeout=2)
        return _Manager()

    def _run_headless():
        launch_headless_replica_group(
            stage_id=0,
            omni_dp_size_local=1,
            per_replica_devices=["0"],
            launch_one=_launch_headless,
            wait_for_replicas=lambda _managers: None,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "headless"}},
        )

    def _run_colocated():
        assert headless_entered.wait(timeout=2)
        with scoped_spawn_device_env(
            "1",
            threading.Lock(),
            stage_id=1,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "colocated"}},
        ):
            colocated_entered.set()

    headless_thread = threading.Thread(target=_run_headless)
    colocated_thread = threading.Thread(target=_run_colocated)
    headless_thread.start()
    assert headless_entered.wait(timeout=2)
    colocated_thread.start()
    try:
        assert not colocated_entered.wait(timeout=0.1)
    finally:
        release_headless.set()
        headless_thread.join(timeout=2)
        colocated_thread.join(timeout=2)

    assert not headless_thread.is_alive()
    assert not colocated_thread.is_alive()
    assert colocated_entered.is_set()
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


@pytest.mark.parametrize(
    ("stage_runtime", "expected_env"),
    [
        ({"env": {"OMNI_TEST_STAGE_ENV": "headless-stage"}}, "headless-stage"),
        (None, None),
    ],
)
def test_headless_dp_coordinator_spawn_shares_process_global_lock(
    monkeypatch,
    stage_runtime,
    expected_env,
):
    import vllm_omni.engine.stage_engine_startup as startup

    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)
    other_scope_entered = threading.Event()
    release_other_scope = threading.Event()
    coordinator_constructed = threading.Event()
    observed = []
    errors = []

    class _Coordinator:
        proc = types.SimpleNamespace(pid=1234)

        def shutdown(self):
            pass

    def _construct_coordinator(*_args, **_kwargs):
        observed.append(os.environ.get("OMNI_TEST_STAGE_ENV"))
        coordinator_constructed.set()
        return _Coordinator()

    def _hold_other_stage_scope():
        with startup.scoped_spawn_device_env(
            "1",
            threading.Lock(),
            stage_id=1,
            stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "other-stage"}},
        ):
            other_scope_entered.set()
            assert release_other_scope.wait(timeout=2)

    def _launch_headless_stage():
        try:
            startup.launch_headless_llm_replicas(
                vllm_config=types.SimpleNamespace(
                    parallel_config=types.SimpleNamespace(
                        data_parallel_size_local=1,
                        data_parallel_rank=0,
                    ),
                    needs_dp_coordinator=True,
                    model_config=types.SimpleNamespace(is_moe=False),
                ),
                executor_class=object,
                log_stats=False,
                omni_master_address="127.0.0.1",
                omni_master_port=12345,
                stage_id=0,
                stage_config=types.SimpleNamespace(runtime=stage_runtime),
                omni_dp_size_local=1,
                per_replica_devices=["0"],
            )
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(startup, "DPCoordinator", _construct_coordinator)
    monkeypatch.setattr(startup, "launch_headless_replica_group", lambda **_kwargs: None)

    other_thread = threading.Thread(target=_hold_other_stage_scope)
    headless_thread = threading.Thread(target=_launch_headless_stage)
    other_thread.start()
    assert other_scope_entered.wait(timeout=2)
    headless_thread.start()
    try:
        assert not coordinator_constructed.wait(timeout=0.1)
    finally:
        release_other_scope.set()
        other_thread.join(timeout=2)
        headless_thread.join(timeout=2)

    assert not other_thread.is_alive()
    assert not headless_thread.is_alive()
    assert not errors
    assert coordinator_constructed.is_set()
    assert observed == [expected_env]
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_scoped_spawn_env_reaches_spawned_child_and_grandchild(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env
    from vllm_omni.platforms import current_omni_platform

    device_env = current_omni_platform.device_control_env_var
    stage_env = "OMNI_TEST_STAGE_ENV"
    monkeypatch.setenv(device_env, "baseline")
    monkeypatch.delenv(stage_env, raising=False)

    grandchild_code = (
        "import json, os, sys; print(json.dumps([os.environ.get(sys.argv[1]), os.environ.get(sys.argv[2])]))"
    )
    child_code = (
        "import json, os, subprocess, sys; "
        "child = [os.environ.get(sys.argv[1]), os.environ.get(sys.argv[2])]; "
        "grandchild = json.loads(subprocess.check_output("
        "[sys.executable, '-c', sys.argv[3], sys.argv[1], sys.argv[2]], "
        "text=True)); "
        "print(json.dumps([child, grandchild]))"
    )
    with scoped_spawn_device_env(
        "3",
        threading.Lock(),
        stage_id=3,
        stage_runtime_cfg={"env": {stage_env: "spawned"}},
    ):
        result = subprocess.check_output(
            [
                sys.executable,
                "-c",
                child_code,
                device_env,
                stage_env,
                grandchild_code,
            ],
            text=True,
        )

    assert json.loads(result) == [["3", "spawned"], ["3", "spawned"]]
    assert os.environ[device_env] == "baseline"
    assert stage_env not in os.environ


def test_scoped_spawn_env_applies_env_without_device_override(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env

    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)

    with scoped_spawn_device_env(
        None,
        threading.Lock(),
        stage_id=0,
        stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "env-only"}},
    ):
        assert os.environ["OMNI_TEST_STAGE_ENV"] == "env-only"

    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_launch_stage_replica_constructs_manager_inside_stage_env_scope(monkeypatch):
    import vllm.utils.network_utils as network_utils

    import vllm_omni.engine.stage_engine_core_proc_manager as proc_manager_mod
    import vllm_omni.engine.stage_engine_startup as startup
    from vllm_omni.platforms import current_omni_platform

    device_env = current_omni_platform.device_control_env_var
    monkeypatch.setenv(device_env, "baseline")
    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)
    observed = []
    addresses = types.SimpleNamespace()

    class _FakeManager:
        def __init__(self, **_kwargs):
            observed.append(
                (
                    os.environ.get(device_env),
                    os.environ.get("OMNI_TEST_STAGE_ENV"),
                )
            )

    @contextlib.contextmanager
    def _fake_socket_ctx(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(startup, "get_engine_zmq_addresses", lambda _cfg: addresses)
    monkeypatch.setattr(network_utils, "get_open_zmq_ipc_path", lambda: "ipc://fake")
    monkeypatch.setattr(startup, "zmq_socket_ctx", _fake_socket_ctx)
    monkeypatch.setattr(startup, "wait_for_engine_startup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        proc_manager_mod,
        "StageEngineCoreProcManager",
        _FakeManager,
    )

    with startup.launch_stage_replica(
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(),
            cache_config=None,
        ),
        executor_class=object,
        log_stats=False,
        stage_id=0,
        stage_visible_devices="0",
        spawn_device_lock=threading.Lock(),
        stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "stage0"}},
    ):
        pass

    assert observed == [("0", "stage0")]
    assert os.environ[device_env] == "baseline"
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_stage_runtime_env_rolls_back_if_an_assignment_fails(monkeypatch):
    from vllm_omni.engine.stage_init_utils import stage_runtime_env

    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)

    with pytest.raises(ValueError):
        with stage_runtime_env(
            0,
            {
                "env": {
                    "OMNI_TEST_STAGE_ENV": "temporary",
                    "INVALID=ENV=KEY": "value",
                }
            },
        ):
            pytest.fail("invalid environment must fail before entering scope")

    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_headless_replica_group_applies_stage_env_at_launch_boundary(monkeypatch):
    from vllm_omni.engine.stage_engine_startup import (
        launch_headless_replica_group,
    )
    from vllm_omni.platforms import current_omni_platform

    device_env = current_omni_platform.device_control_env_var
    monkeypatch.setenv(device_env, "baseline")
    monkeypatch.delenv("OMNI_TEST_STAGE_ENV", raising=False)
    observations = []

    class _Manager:
        def shutdown(self):
            pass

    def _launch_one(replica_id):
        observations.append(
            (
                replica_id,
                os.environ.get(device_env),
                os.environ.get("OMNI_TEST_STAGE_ENV"),
            )
        )
        return _Manager()

    launch_headless_replica_group(
        stage_id=1,
        omni_dp_size_local=2,
        per_replica_devices=["2", "3"],
        launch_one=_launch_one,
        wait_for_replicas=lambda _managers: None,
        stage_runtime_cfg={"env": {"OMNI_TEST_STAGE_ENV": "headless"}},
    )

    assert observations == [
        (0, "2", "headless"),
        (1, "3", "headless"),
    ]
    assert os.environ[device_env] == "baseline"
    assert "OMNI_TEST_STAGE_ENV" not in os.environ


def test_initialize_local_diffusion_replica_passes_stage_init_timeout_and_inline_flag(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.engine.stage_engine_startup import StageReplicaResources

    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=4,
        async_chunk=False,
    )

    plan = _make_diffusion_plan(0, stage_id=0).replicas[0]

    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime_mod, "inject_kv_stage_info", lambda *_: None)

    def _capture_launch_diffusion_stage_replica(**kwargs):
        captured["stage_id"] = kwargs["metadata"].stage_id
        captured["stage_init_timeout"] = kwargs["stage_init_timeout"]
        captured["batch_size"] = kwargs["batch_size"]
        captured["use_inline"] = kwargs["use_inline"]
        captured["omni_master_server"] = kwargs["omni_master_server"]
        return types.SimpleNamespace(), StageReplicaResources()

    monkeypatch.setattr(runtime_mod, "launch_diffusion_stage_replica", _capture_launch_diffusion_stage_replica)

    runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=302)

    assert captured == {
        "stage_id": 0,
        "stage_init_timeout": 302,
        "batch_size": 4,
        "use_inline": True,
        "omni_master_server": None,
    }


def test_stage_runtime_initializes_stage_pools(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod

    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace(), types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2, is_comprehension=True),
        _make_llm_plan(1, stage_id=1, vllm_config=cfg1, final_output=True),
    ]

    stage0_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=True,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0"),
    )
    stage0_client_r1 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0r1"),
    )
    stage1_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=True,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp1"),
    )
    initialized_clients = {
        0: [stage0_client_r0, stage0_client_r1],
        1: [stage1_client_r0],
    }

    stage0_output_processor = object()
    stage1_output_processor = object()
    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)
    monkeypatch.setattr(runtime, "_initialize_stage_replicas", lambda *_: initialized_clients)
    monkeypatch.setattr(
        runtime_mod,
        "build_llm_stage_output_processor",
        lambda plan, _cfg, **_kw: stage0_output_processor if plan.stage_idx == 0 else stage1_output_processor,
    )

    runtime.initialize()

    assert len(runtime.stage_pools) == 2
    assert runtime.stage_pools[0].stage_client is stage0_client_r0
    assert runtime.stage_pools[1].stage_client is stage1_client_r0
    assert runtime.stage_pools[0].stage_vllm_config is cfg0
    assert runtime.stage_pools[1].stage_vllm_config is cfg1
    assert runtime.stage_pools[0].output_processor is stage0_output_processor
    assert runtime.stage_pools[1].output_processor is stage1_output_processor


def test_build_logical_stage_init_plans_applies_replica_device_splits(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod

    runtime = StageRuntime(
        stage_configs=[
            types.SimpleNamespace(
                stage_id=0, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="0")
            ),
            types.SimpleNamespace(
                stage_id=1, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="1,2,3")
            ),
        ],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    metadata_by_stage = {
        0: _make_llm_metadata(0),
        1: _make_llm_metadata(1),
    }

    monkeypatch.setattr(
        runtime_mod,
        "extract_legacy_stage_metadata",
        lambda cfg: types.SimpleNamespace(**metadata_by_stage[cfg.stage_id].__dict__),
    )
    monkeypatch.setattr(runtime_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(runtime_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
    monkeypatch.setattr(runtime_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(
        runtime_mod,
        "build_vllm_config",
        lambda stage_cfg, *_args, **_kwargs: (types.SimpleNamespace(tag=f"cfg-{stage_cfg.stage_id}"), object),
    )

    stage_plans = runtime._build_logical_stage_init_plans(
        omni_transfer_config=None,
        replicas_per_stage=[1, 3],
        replica_devices_map={1: ["1", "2", "3"]},
    )

    assert [plan.stage_id for plan in stage_plans] == [0, 1]
    assert [replica.stage_cfg.runtime.devices for replica in stage_plans[1].replicas] == ["1", "2", "3"]
    assert [replica.replica_id for replica in stage_plans[1].replicas] == [0, 1, 2]
    assert all(replica.num_replicas == 3 for replica in stage_plans[1].replicas)


def test_initialize_stage_replicas_collects_results_by_stage_and_replica_id(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=123,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2),
        _make_llm_plan(1, stage_id=1, vllm_config=cfg1, num_replicas=2),
    ]

    clients = {
        (0, 0): types.SimpleNamespace(name="stage0-replica0"),
        (0, 1): types.SimpleNamespace(name="stage0-replica1"),
        (1, 0): types.SimpleNamespace(name="stage1-replica0"),
        (1, 1): types.SimpleNamespace(name="stage1-replica1"),
    }

    def _initialize_replica(plan, _stage_init_timeout):
        time.sleep(0.02 * (3 - plan.metadata.stage_id - plan.replica_id))
        return clients[(plan.metadata.stage_id, plan.replica_id)]

    monkeypatch.setattr(runtime, "_initialize_replica", _initialize_replica)

    initialized_clients = runtime._initialize_stage_replicas(stage_plans, stage_init_timeout=123)

    assert initialized_clients == {
        0: [clients[(0, 0)], clients[(0, 1)]],
        1: [clients[(1, 0)], clients[(1, 1)]],
    }


def test_remote_replicas_use_distinct_init_group_keys():
    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=123,
        diffusion_batch_size=1,
        async_chunk=False,
    )
    plan = _make_llm_plan(
        1,
        stage_id=1,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64)),
        num_replicas=2,
    )

    for replica in plan.replicas:
        replica.launch_mode = "remote"
        replica.metadata.runtime_cfg = None

    assert [runtime._replica_init_group_key(replica) for replica in plan.replicas] == [
        "remote:1:0",
        "remote:1:1",
    ]


def test_initialize_stages_cleans_up_successful_replicas_after_partial_multi_replica_failure(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)

    def _initialize_replica(plan, _stage_init_timeout):
        if plan.replica_id == 0:
            return initialized_client
        time.sleep(0.05)
        raise RuntimeError("replica launch failed")

    monkeypatch.setattr(runtime, "_initialize_replica", _initialize_replica)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(runtime, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        runtime.initialize()

    assert captured_cleanup == [[initialized_client]]


def test_initialize_stages_cleans_up_late_successful_replicas_after_early_multi_replica_failure(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)

    def _initialize_stage_replicas(_stage_plans, _stage_init_timeout):
        exc = RuntimeError("replica launch failed")
        exc._initialized_clients_by_stage = {0: [None, initialized_client]}
        raise exc

    monkeypatch.setattr(runtime, "_initialize_stage_replicas", _initialize_stage_replicas)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(runtime, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        runtime.initialize()

    assert captured_cleanup == [[initialized_client]]


def test_initialize_local_llm_replica_passes_stage_init_timeout_to_complete_stage_handshake(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.platforms import current_omni_platform

    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=302,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace(inputs=["in"], outputs=["out"], frontend_stats_publish_address=None)
    captured_timeout: int | None = None
    captured_runtime_cfg = None

    plan = ReplicaInitPlan(
        replica_id=0,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=types.SimpleNamespace(engine_args={}, runtime=types.SimpleNamespace(devices="0")),
        metadata=types.SimpleNamespace(stage_id=0, runtime_cfg={"devices": "0"}),
        stage_connector_spec={},
        omni_kv_connector=(None, None, None),
        stage_vllm_config=fake_vllm_config,
        executor_class=object,
        engine_args_dict={},
    )

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    def _capture_acquire_device_locks(*_args):
        nonlocal captured_timeout
        captured_timeout = _args[2]
        return []

    monkeypatch.setattr(runtime_mod, "acquire_device_locks", _capture_acquire_device_locks)

    from vllm_omni.engine.stage_engine_startup import StageReplicaResources

    @contextlib.contextmanager
    def _fake_launch_stage_replica(**_kwargs):
        nonlocal captured_runtime_cfg
        captured_runtime_cfg = _kwargs["stage_runtime_cfg"]
        yield StageReplicaResources(
            manager=types.SimpleNamespace(shutdown=lambda: None),
            addresses=fake_addresses,
        )

    monkeypatch.setattr(runtime_mod, "launch_stage_replica", _fake_launch_stage_replica)
    monkeypatch.setattr(
        runtime_mod.StageEngineCoreClientBase,
        "make_async_mp_client",
        staticmethod(lambda **_: types.SimpleNamespace(shutdown=lambda: None)),
    )

    try:
        runtime._initialize_local_llm_replica(plan, 302)
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert captured_timeout == 302
    assert captured_runtime_cfg == {"devices": "0"}


def test_build_engine_args_cli_tokenizer_overrides_inferred_base_tokenizer(tmp_path):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"model_subdir": "llm"},
        default_sampling_params={},
    )

    engine_args = build_engine_args_dict(
        stage_cfg,
        str(tmp_path),
        cli_tokenizer="/external/tokenizer",
    )

    assert engine_args["model"] == os.path.join(str(tmp_path), "llm")
    assert engine_args["tokenizer"] == "/external/tokenizer"


def test_build_engine_args_stage_model_overrides_parent_model():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"model": "/stage/model"},
        default_sampling_params={},
    )

    engine_args = build_engine_args_dict(
        stage_cfg,
        "/parent/model",
    )

    assert engine_args["model"] == "/stage/model"


def test_build_engine_args_keeps_full_payload_connector_spec():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=1,
        stage_type="llm",
        engine_args={"async_chunk": False},
        default_sampling_params={},
    )
    connector_spec = {
        "name": "SharedMemoryConnector",
        "extra": {"role": "sender"},
    }

    engine_args = build_engine_args_dict(
        stage_cfg,
        "/parent/model",
        stage_connector_spec=connector_spec,
    )

    assert engine_args["stage_connector_spec"] == connector_spec


def test_build_engine_args_keeps_stage_owned_tokenizer_subdir(tmp_path):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"model_subdir": "llm", "tokenizer_subdir": "tokenizer"},
        default_sampling_params={},
    )

    engine_args = build_engine_args_dict(
        stage_cfg,
        str(tmp_path),
        cli_tokenizer="/external/tokenizer",
    )

    assert engine_args["model"] == os.path.join(str(tmp_path), "llm")
    assert engine_args["tokenizer"] == os.path.join(str(tmp_path), "tokenizer")


def test_build_stage0_input_processor_uses_omni_input_preprocessor(monkeypatch):
    import vllm_omni.engine.stage_init_utils as init_mod

    class DummyInputProcessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer or object()
            self.input_preprocessor = None

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer

    monkeypatch.setattr(init_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(init_mod, "OmniInputPreprocessor", DummyOmniInputPreprocessor)

    input_processor = build_stage0_input_processor(
        types.SimpleNamespace(model_config=types.SimpleNamespace(try_get_generation_config=lambda: {}))
    )

    assert isinstance(input_processor.input_preprocessor, DummyOmniInputPreprocessor)
    assert input_processor.input_preprocessor.renderer is input_processor.renderer


def test_build_stage0_input_processor_does_not_resolve_tokenizer_when_skipped(
    monkeypatch,
):
    import vllm_omni.engine.stage_init_utils as init_mod

    token_only_renderer = object()
    seen = {}

    class DummyInputProcessor:
        def __init__(self, vllm_config, renderer=None):
            if renderer is None:
                raise AssertionError("skip_tokenizer_init must supply a renderer")
            seen["renderer"] = renderer
            self.renderer = renderer
            self.input_preprocessor = None

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            seen["preprocessor_renderer"] = renderer

    monkeypatch.setattr(init_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(
        init_mod,
        "_build_token_only_renderer",
        lambda _config: token_only_renderer,
    )
    monkeypatch.setattr(
        init_mod,
        "OmniInputPreprocessor",
        DummyOmniInputPreprocessor,
    )

    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            skip_tokenizer_init=True,
            try_get_generation_config=lambda: {},
        )
    )
    build_stage0_input_processor(config)

    assert seen["renderer"] is token_only_renderer
    assert seen["preprocessor_renderer"] is token_only_renderer


def test_inject_kv_stage_info_infers_sender_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {
                "need_send_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {"need_recv_cache": True},
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage0, 0, [stage0, stage1])

    assert stage0.engine_args["omni_kv_config"]["stage_id"] == 0
    assert stage0.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}


def test_inject_kv_stage_info_infers_receiver_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {"need_send_cache": True},
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {
                "need_recv_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage1, 1, [stage0, stage1])

    assert stage1.engine_args["omni_kv_config"]["stage_id"] == 1
    assert stage1.engine_args["omni_kv_config"]["engine_input_source"] == [0]
    assert stage1.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}


def test_resolve_stage_configs_injects_global_diffusion_attention_when_missing(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    stage_cfg = types.SimpleNamespace(
        stage_type="diffusion",
        engine_args=types.SimpleNamespace(
            diffusion_attention_config=None,
            lora_path=None,
            lora_scale=None,
            enable_sleep_mode=None,
            quantization_config=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg], None),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "FLASH_ATTN"},
        trust_remote_code=False,
    )

    diffusion_attention_config = stage_configs[0].engine_args.diffusion_attention_config
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"


def test_resolve_stage_configs_preserves_stage_diffusion_attention(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    existing_attention = AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
    stage_cfg = types.SimpleNamespace(
        stage_type="diffusion",
        engine_args=types.SimpleNamespace(
            diffusion_attention_config=existing_attention,
            lora_path=None,
            lora_scale=None,
            enable_sleep_mode=None,
            quantization_config=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg], None),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "FLASH_ATTN"},
        trust_remote_code=False,
    )

    assert stage_configs[0].engine_args.diffusion_attention_config is existing_attention


def test_resolve_stage_configs_does_not_inject_diffusion_attention_into_llm_stage(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    stage_cfg = types.SimpleNamespace(
        stage_type="llm",
        engine_args=types.SimpleNamespace(
            attention_config={"backend": "FLASH_ATTN"},
            enable_sleep_mode=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg], None),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "TORCH_SDPA"},
        trust_remote_code=False,
    )

    assert stage_configs[0].engine_args.attention_config == {"backend": "FLASH_ATTN"}
    assert not hasattr(stage_configs[0].engine_args, "diffusion_attention_config")


def test_extract_legacy_stage_metadata_rocm_does_not_inject_diffusion_attention(monkeypatch):
    """ROCm default attention logic only applies to LLM stages, not diffusion."""
    from vllm_omni.engine.stage_init_utils import extract_legacy_stage_metadata

    monkeypatch.setattr("vllm_omni.engine.stage_init_utils.current_omni_platform.is_rocm", lambda: True)

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={},
        runtime={},
        engine_input_source=[],
        final_output=False,
        final_output_type=None,
    )

    metadata = extract_legacy_stage_metadata(stage_cfg)

    assert metadata.stage_type == "diffusion"
    assert "diffusion_attention_config" not in stage_cfg.engine_args


def test_build_engine_args_dict_normalizes_diffusion_attention_config():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={
            "diffusion_attention_config": {
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"cross": {"backend": "TORCH_SDPA"}},
            }
        },
        runtime={},
    )

    engine_args_dict = build_engine_args_dict(stage_cfg, model="dummy-model")

    diffusion_attention_config = engine_args_dict["diffusion_attention_config"]
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"
    assert diffusion_attention_config.per_role["cross"].backend == "TORCH_SDPA"


def test_build_engine_args_dict_uses_diffusion_attention_config_key():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={
            "diffusion_attention_config": {
                "default": {"backend": "FLASH_ATTN"},
            }
        },
        runtime={},
    )

    engine_args_dict = build_engine_args_dict(stage_cfg, model="dummy-model")

    assert "attention_config" not in engine_args_dict
    assert engine_args_dict["diffusion_attention_config"].default.backend == "FLASH_ATTN"


def test_omni_master_server_allocates_globally_unique_route_ports(monkeypatch):
    """Regression: two stages must never draw the same ZMQ port.

    ``get_open_ports_list`` only dedups within a single call, so per-route
    allocation used to let a later stage reuse an earlier stage's port. The
    second engine to ``bind()`` then died with ``zmq.error.ZMQError: Address
    already in use`` (flaky multi-stage startup, e.g. Qwen3-Omni thinker/talker/
    code2wav). ``OmniMasterServer`` now dedups every port it hands out.
    """
    import itertools

    from vllm_omni.engine import stage_engine_startup as ses

    # A colliding prefix (repeats + the master port 9000) followed by an endless
    # fresh stream, so the only way to succeed is to redraw the collisions.
    supply = itertools.chain([9000, 9000, 9001, 9001, 9000], itertools.count(9002))

    def fake_get_open_ports_list(count):
        return [next(supply) for _ in range(count)]

    monkeypatch.setattr(ses, "get_open_ports_list", fake_get_open_ports_list)

    server = ses.OmniMasterServer(
        master_address="127.0.0.1",
        master_port=9000,  # seed: a route must not reuse the registration port
        stage_ids=[0, 1, 2],
    )

    ports = []
    for sid in (0, 1, 2):
        alloc = server.get_allocation(sid)
        for addr in (
            alloc.handshake_bind_address,
            alloc.input_bind_address,
            alloc.output_bind_address,
        ):
            ports.append(ses._port_from_zmq_address(addr))

    assert len(ports) == len(set(ports)), f"duplicate route ports allocated: {ports}"
    assert 9000 not in ports, "route reused the master registration port"


def test_port_from_zmq_address_parsing():
    from vllm_omni.engine.stage_engine_startup import _port_from_zmq_address

    assert _port_from_zmq_address("tcp://127.0.0.1:34277") == 34277
    assert _port_from_zmq_address(None) is None
    assert _port_from_zmq_address("ipc:///tmp/sock") is None
    assert _port_from_zmq_address("tcp://host:not-a-port") is None
