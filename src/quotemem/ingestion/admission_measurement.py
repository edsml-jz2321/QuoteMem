"""Content-free measurement of extraction admission and whole-reply refusals.

Measurement only: nothing here gates, retries or persists extraction work, and
no model or source text reaches a count. Each fresh extraction reply is counted
once through the observer: an admitted reply by its admission record's notes
(``fact:user_unattested``, ``group:quote_not_found`` …), a refused one by its
output fault. A replayed stored reply is not a fresh reply and is not counted.
"""

from __future__ import annotations

from quotemem.core.extraction_records import WholeOutputFault
from quotemem.ingestion.extraction_admission import Admission
from quotemem.service.request_measurement import ExtractionObservation, observe_extraction


def observe_admission(admission: Admission) -> None:
    """Count one fresh reply that program admission admitted.

    Args:
        admission (Admission): Its admission; its record's notes are counted.
    """
    observe_extraction(ExtractionObservation.from_admission(admission.record))


def observe_refusal(fault: WholeOutputFault) -> None:
    """Count one fresh reply refused whole.

    Args:
        fault (WholeOutputFault): The fault, the key the refusal counts read.
    """
    observe_extraction(ExtractionObservation(refusal=fault))
