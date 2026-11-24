"""Backend-owned storage strategy adapter contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["StrategyAdapter"]

from heavenbase.utils import Any, Sequence

from ..utils.ids import check_id

if TYPE_CHECKING:
    from ..execution.contracts import QueryFragment, RowOp
    from ..execution.frame.result import ResultFrame
    from ..schema.model import EntitySchema
    from .base import Backend


def _strategy_identifier(value: Any, *, resolver: Any) -> str:
    from ..strategies import StorageStrategy

    return StorageStrategy.identifier_of(value, resolver=resolver)


class StrategyAdapter:
    """Implement one storage strategy inside one backend instance.

    Subclasses declare `identifier` and `strategy`, then implement `ensure` and
    `upsert`. Override `can_execute` with `execute` only when the adapter also owns
    query fragments. Reset and Tag metadata have safe defaults.

    Args:
        backend (Backend): Backend instance that owns physical connectivity,
            transactions, and this adapter's lifecycle.

    Raises:
        ValueError: If the subclass identifier or strategy is invalid.
    """

    identifier = ""
    strategy = ""

    def __init__(self, backend: "Backend") -> None:
        """Initialize this strategy adapter from its owning backend.

        Returns:
            None: This initializer stores the adapter identity and strategy binding.
        """
        self.backend = backend
        self.identifier = check_id(type(self).identifier, "strategy adapter identifier")
        try:
            self.strategy = _strategy_identifier(type(self).strategy, resolver=backend.module_resolver)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Unknown strategy for adapter {self.identifier!r}: {type(self).strategy!r}") from exc

    def register(self, *, replace: bool = False) -> "StrategyAdapter":
        """Register this adapter with its owning backend.

        Args:
            replace (bool): Replace a different adapter for the same strategy.

        Returns:
            StrategyAdapter: Registered adapter.

        Raises:
            TypeError: If the subclass does not implement `ensure` and `upsert`.
            ValueError: If the strategy already has a different adapter and `replace`
                is false.
        """
        return self.backend.register_strategy_adapter(self, replace=replace)

    def supports(self, schema: EntitySchema, binding: Any) -> bool:
        """Return whether this adapter owns one storage binding.

        The default accepts the adapter's strategy on its owning backend. Override
        when support also depends on logical type, field, or backend configuration.

        Args:
            schema (EntitySchema): Entity schema being registered or queried.
            binding (Any): Storage binding for one field.

        Returns:
            bool: True when the binding targets this backend and strategy.
        """
        del schema
        raw_strategy = getattr(binding, "strategy", None)
        if raw_strategy is None or getattr(binding, "backend", None) != self.backend.name:
            return False
        try:
            return _strategy_identifier(raw_strategy, resolver=self.backend.module_resolver) == self.strategy
        except (KeyError, TypeError, ValueError):
            return False

    def ensure(self, schema: EntitySchema, binding: Any) -> None:
        """Prepare strategy-specific physical structures for one binding.

        Args:
            schema (EntitySchema): Entity schema being registered.
            binding (Any): Supported storage binding for one field.

        Returns:
            None: This function does not return a value.

        Raises:
            NotImplementedError: If the subclass does not implement physical setup.
        """
        raise NotImplementedError(f"{type(self).__name__}.ensure() is required")

    def prune(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Release strategy state for fields of the entity that are no longer bound here.

        ``Backend.prune`` calls this with every binding of the entity that
        this adapter supports under the live registration, so an adapter that indexed a
        field under an earlier registration can stop maintaining it and drop
        the rows it owns. That includes a field ``schema`` no longer declares,
        whose state an adapter that persists it must find from its own records.
        The default keeps no state and does nothing.

        Args:
            schema (EntitySchema): Entity schema of the live registration.
            bindings (Sequence[Any]): Supported storage bindings the entity keeps.

        Returns:
            None: This function does not return a value.
        """
        del schema, bindings

    def abandon(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Undo what a registration that failed before it was published set up here.

        ``Backend.abandon`` calls this with every binding of the entity that
        this adapter supports under the registration that stays live (none
        when the entity had no registration), once a registration has failed.
        An adapter releases the fields that only the failed registration bound,
        and leaves alone the stored state that existed before it began, which
        another workspace on the same database may still use. The default calls
        ``prune``, which suits an adapter whose state is private to this
        backend.

        Args:
            schema (EntitySchema): Entity schema of the registration that stays live, or of the failed one when none does.
            bindings (Sequence[Any]): Supported storage bindings of the registration that stays live.

        Returns:
            None: This function does not return a value.
        """
        self.prune(schema, bindings)

    def validate_upsert(self, rows: Sequence["RowOp"]) -> None:
        """Check strategy-owned preconditions for a batch of backend row mutations.

        The owning backend filters out mutations whose ``RowOp.strategies`` flag
        is false. Validation must leave persistent state unchanged so a backend
        that may autocommit can run it before writing any rows.

        Args:
            rows: Backend-ready row mutations.
        """
        del rows

    def upsert(self, rows: Sequence["RowOp"]) -> None:
        """Maintain strategy-owned structures for backend row mutations.

        Only mutations whose ``RowOp.strategies`` flag is true reach this
        method; the owning backend filters the rest.

        Args:
            rows (Sequence[RowOp]): Backend-ready row mutations.

        Returns:
            None: This function does not return a value.

        Raises:
            NotImplementedError: If the subclass does not implement mutation handling.
        """
        raise NotImplementedError(f"{type(self).__name__}.upsert() is required")

    def can_execute(self, fragment: "QueryFragment") -> bool:
        """Return whether this adapter owns one compiled query fragment.

        Args:
            fragment (QueryFragment): Backend-ready query fragment.

        Returns:
            bool: False by default; query-capable adapters override this method.
        """
        del fragment
        return False

    def execute(self, fragment: "QueryFragment") -> ResultFrame:
        """Execute one adapter-owned query fragment.

        Args:
            fragment (QueryFragment): Backend-ready query fragment.

        Returns:
            ResultFrame: Rows matched by the adapter.

        Raises:
            TypeError: Always, unless a query-capable subclass overrides this method.
        """
        raise TypeError(f"{self.identifier} does not execute fragment op={fragment.op!r}")

    def reset(self) -> None:
        """Clear adapter state after backend reconnect or drop.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """

    def to_dict(self) -> dict[str, Any]:
        """Return the stable ``strategy-adapters`` Tag envelope.

        Args:
            None.

        Returns:
            dict[str, Any]: Adapter identity, strategy reference, and metadata.
        """
        return {"identifier": self.identifier, "strategy": self.strategy, "metadata": {}}
