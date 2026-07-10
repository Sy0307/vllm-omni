# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace


class PlaybackCursorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlaybackCursor:
    generated: int = 0
    sent: int = 0
    played: int = 0
    committed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.committed <= self.played <= self.sent <= self.generated:
            raise PlaybackCursorError(
                "playback positions must satisfy "
                "0 <= committed <= played <= sent <= generated"
            )

    def mark_generated(self, cursor: int) -> PlaybackCursor:
        if cursor < 0:
            raise PlaybackCursorError("generated cursor cannot be negative")
        if cursor < self.generated:
            raise PlaybackCursorError(
                "generated cursor cannot move backwards: "
                f"{cursor} < {self.generated}"
            )
        return replace(self, generated=cursor)

    def mark_sent(self, cursor: int) -> PlaybackCursor:
        if cursor < 0:
            raise PlaybackCursorError("sent cursor cannot be negative")
        if cursor < self.sent:
            raise PlaybackCursorError(
                f"sent cursor cannot move backwards: {cursor} < {self.sent}"
            )
        if cursor > self.generated:
            raise PlaybackCursorError(
                "sent cursor cannot exceed generated cursor: "
                f"{cursor} > {self.generated}"
            )
        return replace(self, sent=cursor)

    def acknowledge(
        self,
        *,
        played: int,
        committed: int | None = None,
    ) -> PlaybackCursor:
        if played < 0:
            raise PlaybackCursorError("played cursor cannot be negative")
        if committed is None:
            committed = played
        if committed < 0:
            raise PlaybackCursorError("committed cursor cannot be negative")

        next_played = max(self.played, played)
        next_committed = max(self.committed, committed)
        if next_played > self.sent:
            raise PlaybackCursorError(
                f"played cursor exceeds sent cursor: {next_played} > {self.sent}"
            )
        if next_committed > next_played:
            raise PlaybackCursorError(
                "committed cursor cannot exceed played cursor: "
                f"{next_committed} > {next_played}"
            )
        return replace(self, played=next_played, committed=next_committed)
