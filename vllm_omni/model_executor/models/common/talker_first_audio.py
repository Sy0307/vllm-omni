# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""When a Talker stage decodes each stream's first frame itself."""

import os
from typing import Any


def talker_first_audio_enabled(vllm_config: Any) -> bool:
    """Whether the Talker decodes and delivers each stream's first frame (default on).

    Needs Model Runner V2, whose model state decodes the frame after sampling
    and hands it to the client, and streaming (async-chunk) Code2Wav input,
    whose first chunk is then a context-free decode of that same frame.
    ``VLLM_OMNI_TALKER_FIRST_AUDIO=0`` turns it off.
    """
    model_config = vllm_config.model_config
    return (
        os.environ.get("VLLM_OMNI_TALKER_FIRST_AUDIO", "1") == "1"
        and bool(getattr(model_config, "use_v2_model_runner", False))
        and bool(getattr(model_config, "async_chunk", False))
    )
