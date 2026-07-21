from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommitAction(str, Enum):
    """How one native audio commit participates in response scheduling."""

    START_AUTO_RESPONSE = "start_auto_response"
    DEFER_ACTIVE_RESPONSE = "defer_active_response"
    COMMIT_ONLY = "commit_only"


@dataclass(frozen=True, slots=True)
class CommitSnapshot:
    auto_responds: bool
    speech_since_commit: bool
    auto_response_waiting_for_speech: bool
    active_response_id: str | None
    overlap_speech_ms: int
    native_response_in_progress: bool
    playback_active: bool


def decide_commit_action(snapshot: CommitSnapshot) -> CommitAction:
    """Choose response scheduling from an immutable session snapshot."""
    if snapshot.playback_active and snapshot.overlap_speech_ms > 0:
        return CommitAction.DEFER_ACTIVE_RESPONSE
    can_start_response = snapshot.active_response_id is None and (
        snapshot.auto_response_waiting_for_speech or not snapshot.native_response_in_progress
    )
    if snapshot.auto_responds and snapshot.speech_since_commit and can_start_response:
        return CommitAction.START_AUTO_RESPONSE
    if snapshot.native_response_in_progress and snapshot.overlap_speech_ms > 0:
        return CommitAction.DEFER_ACTIVE_RESPONSE
    return CommitAction.COMMIT_ONLY


__all__ = ["CommitAction", "CommitSnapshot", "decide_commit_action"]
