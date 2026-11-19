"""Process-local finalize execution identity from plan.md §9.

The Writer issues and weakly registers these objects. Matching field values alone do
not authorize a transition; the Writer also checks object identity and durable
lease, Run, scope and epoch state before every effect.
"""

from dataclasses import dataclass
from typing import Literal

from aml.core.identity import ULID, ScopeId


@dataclass(eq=False, slots=True, weakref_slot=True)
class FinalizeLeaseTicket:
    """Identify one existing cycle without copying its durable execution state.

    Applications retain the issued object across heartbeat and held transition.
    They cannot acquire authority by constructing or editing a ticket. The
    state is refreshed by the Writer only after the matching lease CAS succeeds.
    """

    scope_id: ScopeId
    lease_generation: int
    owner_run_id: ULID
    writer_epoch: int
    state: Literal["preparing", "held"]
