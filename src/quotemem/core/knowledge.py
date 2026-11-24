"""Request-owned recorded-time selections with current privacy authority."""

from dataclasses import dataclass

from quotemem.core.recall_snapshot import RecallSnapshot


@dataclass(frozen=True, slots=True)
class KnowledgeReadView:
    """Selected revisions projected as active heads for one guarded Search.

    The root contains current Evidence and fences, only the selected memories,
    and no Evidence routes. It must not outlive the enclosing read guard.
    ``storage_generation`` binds the reconstruction to the sole Writer.
    """

    root: RecallSnapshot
    scope_id: str
    recorded_as_of: int
    storage_generation: int
