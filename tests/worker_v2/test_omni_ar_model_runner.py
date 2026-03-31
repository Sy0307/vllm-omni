"""Unit tests for OmniARModelRunner v2."""

import numpy as np
import pytest
import torch

from vllm_omni.worker_v2.omni_ar_model_runner import (
    OmniARModelRunner,
    _async_copy_mm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------
# _build_pooler_output_from_cpu (was _build_pooler_output)
# ---------------------------------------------------------------


def test_build_pooler_output_basic():
    """Verify _build_pooler_output_from_cpu slices per-request hidden + mm."""
    hidden = torch.randn(6, 8)
    mm = {"audio": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )

    assert len(pooler) == 2
    assert pooler[0]["hidden"].shape == (3, 8)
    assert pooler[1]["hidden"].shape == (3, 8)
    assert pooler[0]["audio"].shape == (3, 2)


def test_build_pooler_output_empty_mm():
    hidden = torch.randn(4, 8)

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        {},
        query_start_loc_np=np.array([0]),
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_reqs=1,
    )
    assert len(pooler) == 1
    assert "hidden" in pooler[0]
    assert len(pooler[0]) == 1


# ---------------------------------------------------------------
# _async_copy_mm (was _copy_mm_to_cpu)
# ---------------------------------------------------------------


def test_copy_mm_to_cpu_tensor():
    total = 10
    t = torch.randn(10, 4)
    result = _async_copy_mm({"feat": t}, total)
    assert "feat" in result
    assert result["feat"].shape == (10, 4)
    assert result["feat"].device == torch.device("cpu")


def test_copy_mm_to_cpu_dict():
    total = 10
    d = {"inner": torch.randn(10, 2)}
    result = _async_copy_mm({"nested": d}, total)
    assert "nested" in result
    assert "inner" in result["nested"]


def test_copy_mm_to_cpu_list():
    result = _async_copy_mm({"items": [torch.randn(3), "text"]}, 10)
    assert "items" in result
    assert isinstance(result["items"][0], torch.Tensor)
    assert result["items"][1] == "text"


def test_copy_mm_to_cpu_empty():
    assert _async_copy_mm({}, 10) == {}


# ---------------------------------------------------------------
# Slicing via _build_pooler_output_from_cpu (was _slice_mm_payload)
# ---------------------------------------------------------------


def test_slice_mm_payload_tensor():
    hidden = torch.randn(6, 4)
    mm_cpu = {"feat": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[0]["feat"].shape == (3, 2)
    assert pooler[1]["feat"].shape == (3, 2)


def test_slice_mm_payload_list():
    hidden = torch.randn(6, 4)
    mm_cpu = {"items": [torch.randn(2), torch.randn(3)]}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert isinstance(pooler[0]["items"], torch.Tensor)
    assert isinstance(pooler[1]["items"], torch.Tensor)


def test_slice_mm_payload_dict():
    hidden = torch.randn(6, 4)
    mm_cpu = {"nested": {"a": torch.randn(6, 2)}}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[1]["nested"]["a"].shape == (3, 2)
