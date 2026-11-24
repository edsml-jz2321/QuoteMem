"""Seal order bookkeeping registered by the Assistant log workspace."""

from __future__ import annotations

__all__ = ["AgentLogSeal"]

from heavenbase.entity import Entity, field
from heavenbase.types import Integer


class AgentLogSeal(Entity):
    """The last allocated seal sequence of one assistant conversation."""

    identifier = "agent-log-seal"

    seq = field(Integer).default(0).desc("Last allocated sequence for a turn summary")
