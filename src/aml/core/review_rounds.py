"""An owner's review rounds: round 1, round 2 and the settle rounds after it (C.2, C.11.2).

Round 1 reviews the owner's members and the re-entered ones; round 2 reviews what
round 1's resolver or fence sent on. The resolver asks for no round after round 2,
but the fence can still find a round-2 component changed: its members were
nominated at the round's pin, and an earlier component of the same round wrote rows
they might now be issued (C.11.1 #3). Such a component goes to the next round, a
settle round, which rebuilds, nominates, reviews and commits its members at a fresh
pin after every earlier commit, exactly as round 2 does. Each settle round reviews a
position once more under a judge child of its own round (C.13).

The first component of a round meets no delta of its own round, so a round commits
at least one component unless another writer changed the scope inside its window;
``MAX_ROUND`` bounds the rounds, and only a component the fence still finds changed
in that last round ends ``target_changed`` (C.11.2 #2).
"""

from __future__ import annotations

from typing import Final, Literal, cast

RoundNo = Literal[1, 2, 3, 4, 5, 6, 7, 8]
"""An owner's review round: 1, 2, or a settle round up to ``MAX_ROUND``."""

MAX_ROUND: Final = 8
"""The owner's last review round; its changed components end ``target_changed``."""


def next_round(round_no: RoundNo) -> RoundNo | None:
    """The round after ``round_no``, or None after the last one."""
    return None if round_no >= MAX_ROUND else cast(RoundNo, round_no + 1)


__all__ = ["MAX_ROUND", "RoundNo", "next_round"]
