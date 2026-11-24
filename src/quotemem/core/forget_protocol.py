"""Pure verification of the current forget receipt reservation.

Admission, owner cancellation and maintenance state transitions remain the
single Writer's responsibility. A valid receipt pair alone proves no deletion.
"""

from quotemem.core.errors import QuoteMemError
from quotemem.core.forget import ForgetPrepareReceiptV3, ForgetReceiptV3
from quotemem.core.identity import ArtifactId


def verify_forget_receipt_pair(
    prepare: ForgetPrepareReceiptV3,
    final: ForgetReceiptV3,
    *,
    prepare_ref: ArtifactId,
) -> None:
    """Refuse any changed reservation or substituted prepare reference.

    Every prepare field other than its phase literals must survive byte for
    byte. Models are revalidated so constructed or mutated instances cannot
    bypass the strict receipt schema at this proof boundary.
    """
    prepare = ForgetPrepareReceiptV3.model_validate(prepare.model_dump(mode="python"))
    final = ForgetReceiptV3.model_validate(final.model_dump(mode="python"))
    original = prepare.model_dump(mode="json")
    completed = final.model_dump(mode="json")
    if final.prepare_ref != prepare_ref or any(
        completed.get(key) != value for key, value in original.items()
        if key not in {"receipt_kind", "status"}
    ):
        raise QuoteMemError("binding_mismatch", "Forget receipts describe different reservations")
