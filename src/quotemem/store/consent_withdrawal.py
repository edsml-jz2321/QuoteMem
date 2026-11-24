"""A user's consent withdrawal: the durable premise of erasing all their sources.

Consent withdrawal is a privacy-erase trigger: advancing the scope's
``consent_epoch:{s}`` fences every operation admitted under the old epoch, and
the erase of the withdrawn sources (``ConsentWithdrawalInputV1``) is admitted at
the advanced epoch. Until now no route advanced the epoch; the
agent profile's ``delete-user`` is its producer.

In one Writer section the withdrawal advances ``commit_seq`` and
``consent_epoch:{s}`` together and publishes the scope's new fence through the
read model's privacy path. An Add owner holding the scope keeps its admitted
epochs, so its next Writer action is refused ``consent_fenced`` and
``store.finalize_privacy`` rebases or cancels its cycle. Nothing else changes:
the erase that follows is the ordinary ``PrivacyCleanup`` owner.

The withdrawal is refused while the scope is quarantined or a forget, erase or
maintenance fence holds it; the caller waits for those to end first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quotemem.core.errors import QuoteMemError
from quotemem.core.query import ScopeV1

if TYPE_CHECKING:
    from quotemem.store.writer import Writer


def withdraw_consent(writer: Writer, scope: ScopeV1, *, now: int) -> int:
    """Advance the scope's consent epoch and publish it (the module's transaction).

    Args:
        writer: The sole Writer; its mutex must not be held by the caller.
        scope: The registered scope whose consent is withdrawn.
        now: Trusted time in epoch milliseconds for the counter rows.

    Returns:
        The advanced consent epoch.

    Raises:
        QuoteMemError: ``scope_denied`` for an unregistered scope; ``operation_conflict``
            while it is quarantined or fenced; storage and publication failures,
            after which the read model is failed closed for recovery.
    """
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.writer import RowWrite

    key = f"consent_epoch:{scope.scope_id}"
    with writer.write_scope():
        registration = writer.get_control(f"scope:{scope.scope_id}")
        if registration is None or registration["value"]["scope_kind"] != scope.scope_kind:
            raise QuoteMemError("scope_denied", "Consent withdrawal requires a registered scope")
        if writer.is_scope_quarantined(scope.scope_id):
            raise QuoteMemError("operation_conflict", "A quarantined scope must be repaired first")
        if writer.indexed_reader().scope_fences(scope.scope_id):
            raise QuoteMemError("operation_conflict",
                                "A forget or maintenance fence holds the scope")
        rows: list[RowWrite] = []
        for name in ("commit_seq", key):
            previous = writer.get_control(name)
            if previous is None:
                raise QuoteMemError("service_unhealthy", "Required Control counter is absent")
            rows.append(RowWrite("control", writer.materialize_control(
                name, {"seq": previous["value"]["seq"] + 1}, previous["version"] + 1, now,
                previous["updated_by_run_id"],
            )))
        _, plan = writer.plan_rows(rows, None)
        writer.write_plan(plan)
        publisher = StoreReadModel.get(writer)
        if publisher is not None:
            try:
                publisher.publish_privacy(
                    writer, scope, None, writer.get_counter("commit_seq"), maintenance=False,
                )
            except BaseException:
                publisher.fail_closed(discard_finalize_baselines=True)
                raise
        return writer.get_counter(key)
