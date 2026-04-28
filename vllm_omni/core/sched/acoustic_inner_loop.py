from __future__ import annotations


def acoustic_inner_loop_extra_slots(
    *,
    max_inner_steps: int,
    num_running_requests: int,
    num_computed_tokens: int,
    num_prompt_tokens: int,
    num_output_tokens: int,
    max_tokens: int,
    has_spec_tokens: bool = False,
    uses_structured_output: bool = False,
) -> int:
    if max_inner_steps <= 1:
        return 0
    if num_running_requests != 1:
        return 0
    if has_spec_tokens or uses_structured_output:
        return 0
    if num_computed_tokens < num_prompt_tokens:
        return 0

    remaining = max_tokens - num_output_tokens
    if remaining <= 1:
        return 0
    return min(max_inner_steps, remaining) - 1


def corrected_num_computed_tokens(
    *,
    num_computed_tokens_after_schedule: int,
    num_scheduled_tokens: int,
    num_generated_tokens: int,
) -> int:
    unused_slots = max(0, num_scheduled_tokens - num_generated_tokens)
    return num_computed_tokens_after_schedule - unused_slots
