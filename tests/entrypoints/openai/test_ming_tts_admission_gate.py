import asyncio

import pytest

from vllm_omni.entrypoints.openai.serving_speech import MingTTSAdmissionGate


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_releases_full_batch_together():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=1000)

    async def wait(name):
        cohort = await gate.wait("same-key")
        return name, cohort

    first = asyncio.create_task(wait("first"))
    await asyncio.sleep(0)
    assert not first.done()

    second = asyncio.create_task(wait("second"))
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert results[0][0] == "first"
    assert results[1][0] == "second"
    assert results[0][1] == results[1][1]


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_separates_keys():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=20)

    first = asyncio.create_task(gate.wait("a"))
    second = asyncio.create_task(gate.wait("b"))

    cohorts = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert cohorts[0] != cohorts[1]
