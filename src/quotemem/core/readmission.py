"""The request ids an acknowledged Add is stored under once a forget overtook it.

A forget of a user that commits after an Add's input was saved and before it was
ordered deletes that input, and the Writer refuses the Add's Run for good
(``erase_fenced``). The agent service then completes the same messages under a
derived request id (``service/add_followup.py``). The id is a function of the
original id and the admission number alone, so a restart derives the same ids, and
every read of the original request (its result, its add-items, Why, its in-chat
forget report) finds the result stored under one of them (`admitted_request_ids`).
"""

from __future__ import annotations

import hashlib
from typing import Final

MAX_READMISSIONS: Final = 3
"""Re-admissions of one Add whose saved input forgets overtook, before its refusal is told."""
READMITTED_PREFIX: Final = "readmitted-"


def readmitted_request_id(request_id: str, admission: int) -> str:
    """The request id of an Add's ``admission``-th re-admission (1 through the maximum)."""
    original = hashlib.sha256(request_id.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{READMITTED_PREFIX}{admission}-{original}"


def admitted_request_ids(request_id: str) -> tuple[str, ...]:
    """Every id one Add's result can be stored under: its own first, then each re-admission's.

    At most one of them holds a result: an admission is re-admitted only after the
    Writer refused it for good.
    """
    return (request_id, *(
        readmitted_request_id(request_id, admission)
        for admission in range(1, MAX_READMISSIONS + 1)
    ))
