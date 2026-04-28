from vllm_omni.core.sched.acoustic_inner_loop import (
    acoustic_inner_loop_extra_slots,
    corrected_num_computed_tokens,
)


def test_extra_slots_for_single_decode_request():
    assert (
        acoustic_inner_loop_extra_slots(
            max_inner_steps=4,
            num_running_requests=1,
            num_computed_tokens=32,
            num_prompt_tokens=32,
            num_output_tokens=10,
            max_tokens=64,
        )
        == 3
    )


def test_extra_slots_limited_by_remaining_tokens():
    assert (
        acoustic_inner_loop_extra_slots(
            max_inner_steps=4,
            num_running_requests=1,
            num_computed_tokens=32,
            num_prompt_tokens=32,
            num_output_tokens=62,
            max_tokens=64,
        )
        == 1
    )


def test_extra_slots_disabled_for_prefill_multi_request_or_spec():
    base = dict(
        max_inner_steps=4,
        num_running_requests=1,
        num_computed_tokens=32,
        num_prompt_tokens=32,
        num_output_tokens=10,
        max_tokens=64,
    )
    assert acoustic_inner_loop_extra_slots(**(base | {"num_computed_tokens": 31})) == 0
    assert acoustic_inner_loop_extra_slots(**(base | {"num_running_requests": 2})) == 0
    assert acoustic_inner_loop_extra_slots(**base, has_spec_tokens=True) == 0
    assert acoustic_inner_loop_extra_slots(**base, uses_structured_output=True) == 0


def test_corrects_unused_scheduled_slots_after_early_stop():
    assert (
        corrected_num_computed_tokens(
            num_computed_tokens_after_schedule=40,
            num_scheduled_tokens=4,
            num_generated_tokens=2,
        )
        == 38
    )
