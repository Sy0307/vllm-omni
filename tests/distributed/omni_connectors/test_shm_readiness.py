# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import struct
import subprocess
import sys

import pytest

from vllm_omni.distributed.omni_connectors.utils.shm_readiness import ShmReadiness

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.skipif(sys.platform != "linux", reason="inotify")]


@pytest.fixture
def notifier(tmp_path):
    notifier = ShmReadiness(str(tmp_path))
    yield notifier
    notifier.close()


def test_publication_requires_writer_close(notifier, tmp_path):
    path = tmp_path / "shm_request_0_0_lockfile.lock"
    with path.open("wb") as writer:
        writer.write(b"published")
        writer.flush()
        assert notifier.wait(0)[0] == set()
    assert notifier.wait(1)[0] == {"request_0_0"}


def test_cross_process_publication_batches_keys(notifier, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "[(Path(sys.argv[1])/('shm_'+k+'_lockfile.lock')).write_bytes(b'x') for k in ['a_0_0','b_0_1']]",
            str(tmp_path),
        ],
        check=True,
    )
    assert notifier.wait(1)[0] == {"a_0_0", "b_0_1"}


def test_local_wake_before_wait_is_retained(notifier):
    notifier.wake()
    assert notifier.wait(0) == (set(), False)


def test_queue_overflow_requires_key_reconciliation():
    assert ShmReadiness._decode(struct.pack("iIII", -1, 0x4000, 0, 0)) == (set(), True)


def test_watch_invalidation_falls_back_instead_of_silently_losing_notifications():
    with pytest.raises(OSError, match="invalidated"):
        ShmReadiness._decode(struct.pack("iIII", 1, 0x8000, 0, 0))
