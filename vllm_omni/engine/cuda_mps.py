# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Private NVIDIA MPS lifetime for explicitly opted-in local stage processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def physical_gpu_uuid(device: str) -> str:
    """Resolve a physical ordinal/UUID through NVML without initializing CUDA."""
    from vllm.utils.import_utils import import_pynvml

    nvml = import_pynvml()
    nvml.nvmlInit()
    try:
        if device.isdigit():
            handle = nvml.nvmlDeviceGetHandleByIndex(int(device))
        elif device.startswith("GPU-"):
            handle = nvml.nvmlDeviceGetHandleByUUID(device)
        else:
            raise ValueError("cuda_mps requires one physical NVIDIA GPU ordinal or GPU UUID")
        uuid = nvml.nvmlDeviceGetUUID(handle)
        return uuid.decode() if isinstance(uuid, bytes) else str(uuid)
    finally:
        nvml.nvmlShutdown()


class CudaMPSServer:
    """Own a private control socket; never stop an operator's existing daemon."""

    def __init__(self, gpu_uuid: str) -> None:
        control = shutil.which("nvidia-cuda-mps-control")
        if control is None:
            raise RuntimeError("cuda_mps requires nvidia-cuda-mps-control on PATH")
        self._control = control
        self._directory: Path | None = None
        self._closed = False
        inherited_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        if inherited_pipe:
            self.env = {"CUDA_MPS_PIPE_DIRECTORY": inherited_pipe, "CUDA_VISIBLE_DEVICES": gpu_uuid}
            self._run(input="get_server_list\n")
            logger.info("Using operator-managed MPS at %s for %s", inherited_pipe, gpu_uuid)
            return

        self._directory = root = Path(tempfile.mkdtemp(prefix="vllm-omni-mps-"))
        (root / "pipe").mkdir()
        (root / "log").mkdir()
        self.env = {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "CUDA_MPS_PIPE_DIRECTORY": str(root / "pipe"),
            "CUDA_MPS_LOG_DIRECTORY": str(root / "log"),
        }
        started = False
        try:
            self._run("-d")
            started = True
            self._run(input="get_server_list\n")
        except BaseException:
            if started:
                try:
                    self.close()
                except Exception:
                    logger.exception("MPS startup cleanup failed; control files remain at %s", root)
            else:
                shutil.rmtree(root)
                self._directory = None
            raise
        logger.info("Started private MPS for %s at %s", gpu_uuid, self.env["CUDA_MPS_PIPE_DIRECTORY"])

    def _run(self, *args: str, input: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._control, *args],
            input=input,
            env=os.environ | self.env,
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._directory is not None:
            # Keep the control socket if quit fails, so cleanup can be retried.
            self._run(input="quit\n")
            shutil.rmtree(self._directory)
            self._directory = None
        self._closed = True
