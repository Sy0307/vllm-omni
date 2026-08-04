from types import SimpleNamespace

import numpy as np

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexOutputContext,
    DuplexRequestIdentity,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.engine.model_events import (
    DuplexListen,
    DuplexSpeakChunk,
    DuplexSpeakEnd,
    DuplexSpeakStart,
)
from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.experimental.fullduplex.output import (
    attach_duplex_output_context,
)


def _encode_audio(value, _sample_rate, _response_format, _speed):
    samples = np.asarray(value, dtype=np.float32).reshape(-1)
    return f"wav-{samples.size}" if samples.size else None


def _context(fence: DuplexFence, *, source_input_seq: int) -> MiniCPMO45DataPlaneContext:
    return MiniCPMO45DataPlaneContext(
        fence=fence,
        source_input_seq=source_input_seq,
        auto_responds=True,
        response_format="wav",
        modalities=("audio",),
    )


def _output(
    fence: DuplexFence,
    *,
    source_input_seq: int,
    request_id: str = "duplex-model-events-stage0",
    text: str = "",
    samples: int = 0,
    finished: bool = False,
    listen: bool = False,
    tts_segment_end: bool = False,
    speech_end: bool = False,
):
    multimodal_output: dict[str, object] = {
        "sr": 24000,
        "meta.tts_is_last_chunk": np.asarray([tts_segment_end], dtype=np.bool_),
        "meta.duplex_speech_end": np.asarray([speech_end], dtype=np.bool_),
    }
    if samples:
        multimodal_output["audio"] = np.zeros(samples, dtype=np.float32)
    if listen:
        multimodal_output.update(
            {
                "duplex_native_decision": "listen",
                "model_listen": True,
            }
        )
    output = SimpleNamespace(
        request_id=request_id,
        finished=finished,
        outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
        multimodal_output=multimodal_output,
        metrics={},
        _custom_output={},
    )
    return attach_duplex_output_context(
        output,
        DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=fence.session_id,
                fence=fence,
            ),
            final_stage_id=2,
            segment_finished=finished,
            source_input_seq=source_input_seq,
        ),
    )


def test_model_listen_projects_one_typed_listen_event() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=2)
    data_plane = MiniCPMO45DataPlaneSession(_encode_audio)

    events = tuple(
        data_plane.project_output(
            _output(fence, source_input_seq=7, finished=True, listen=True),
            context=_context(fence, source_input_seq=8),
        )
    )

    assert events == (
        DuplexListen(
            fence=fence,
            source_input_seq=7,
            reason="model_listen",
        ),
    )


def test_first_visible_chunk_starts_and_speech_end_finishes_one_output() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=0)
    data_plane = MiniCPMO45DataPlaneSession(_encode_audio)

    events = tuple(
        data_plane.project_output(
            _output(
                fence,
                source_input_seq=3,
                text="hello",
                samples=240,
                finished=True,
                tts_segment_end=True,
                speech_end=True,
            ),
            context=_context(fence, source_input_seq=4),
        )
    )

    assert [type(event) for event in events] == [
        DuplexSpeakStart,
        DuplexSpeakChunk,
        DuplexSpeakEnd,
    ]
    start, chunk, end = events
    assert start.source_input_seq == 3
    assert chunk.output_id == start.output_id
    assert chunk.output_seq == 0
    assert chunk.text_delta == "hello"
    assert chunk.audio_data == "wav-240"
    assert end.output_id == start.output_id


def test_tts_segment_end_does_not_end_active_speech_output() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=0)
    data_plane = MiniCPMO45DataPlaneSession(_encode_audio)

    first = tuple(
        data_plane.project_output(
            _output(
                fence,
                source_input_seq=1,
                samples=100,
                finished=True,
                tts_segment_end=True,
            ),
            context=_context(fence, source_input_seq=1),
        )
    )
    second = tuple(
        data_plane.project_output(
            _output(
                fence,
                source_input_seq=2,
                samples=180,
            ),
            context=_context(fence, source_input_seq=2),
        )
    )

    assert isinstance(first[0], DuplexSpeakStart)
    first_chunk = next(event for event in first if isinstance(event, DuplexSpeakChunk))
    assert not any(isinstance(event, DuplexSpeakEnd) for event in first)
    second_chunk = next(event for event in second if isinstance(event, DuplexSpeakChunk))
    assert second_chunk.output_id == first_chunk.output_id
    assert second_chunk.output_seq == 1


def test_listen_during_speech_does_not_end_or_replace_active_output() -> None:
    fence = DuplexFence("session", incarnation=1, epoch=0)
    data_plane = MiniCPMO45DataPlaneSession(_encode_audio)

    first = tuple(
        data_plane.project_output(
            _output(fence, source_input_seq=1, samples=100),
            context=_context(fence, source_input_seq=1),
        )
    )
    listen = tuple(
        data_plane.project_output(
            _output(fence, source_input_seq=2, finished=True, listen=True),
            context=_context(fence, source_input_seq=2),
        )
    )
    later = tuple(
        data_plane.project_output(
            _output(fence, source_input_seq=3, samples=180),
            context=_context(fence, source_input_seq=3),
        )
    )

    first_chunk = next(event for event in first if isinstance(event, DuplexSpeakChunk))
    assert listen == (DuplexListen(fence=fence, source_input_seq=2),)
    later_chunk = next(event for event in later if isinstance(event, DuplexSpeakChunk))
    assert later_chunk.output_id == first_chunk.output_id
    assert later_chunk.output_seq == 1


def test_old_epoch_output_projects_no_model_event() -> None:
    old_fence = DuplexFence("session", incarnation=1, epoch=0)
    current_fence = DuplexFence("session", incarnation=1, epoch=1)
    data_plane = MiniCPMO45DataPlaneSession(_encode_audio)

    events = tuple(
        data_plane.project_output(
            _output(old_fence, source_input_seq=5, samples=100),
            context=_context(current_fence, source_input_seq=1),
        )
    )

    assert events == ()
