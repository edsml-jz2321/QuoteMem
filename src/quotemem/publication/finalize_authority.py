"""Writer epoch ordering for committed finalize children.

This join validates stored input meaning. It grants no held lease, current read
authorization, Candidate membership or permission to commit business effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from quotemem.core.errors import QuoteMemError

if TYPE_CHECKING:
    from quotemem.store.writer import Writer


def validate_finalize_history_epoch(
    writer: Writer, child: Mapping[str, Any], owner: Mapping[str, Any],
) -> None:
    """Retain terminal child epochs across actual same-owner cold adoption."""
    lease = writer.get_control("lease:writer")
    if lease is None or not (
        child["writer_epoch"] <= owner["writer_epoch"] <= lease["value"]["writer_epoch"]
    ):
        raise QuoteMemError("lease_lost", "Finalize history changed its writer epoch ordering")
