"""Neutral row, query, and handler execution contracts."""

from __future__ import annotations

__all__ = ["LOGICAL_OPS", "HandlerContext", "QueryFragment", "RowOp"]

from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, Iterable, Mapping, Sequence, copy_data, dataclass, field, raise_mismatch
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver, module_resolver
from ..utils.ids import check_entity_id
from .frame.merge import difference, intersect, union
from .frame.result import ResultFrame

if TYPE_CHECKING:
    from .handlers.registry import HandlerRegistry

LOGICAL_OPS = ("and", "or", "not")


@dataclass(init=False)
class HandlerContext:
    """Carry the explicit authorities and route facts used during compilation.

    The context is deliberately owned by the neutral execution layer. Provider
    handlers receive a live backend only as an optional protocol object; shared
    execution code never imports backend implementations or their registries.
    """

    backend: str
    backend_identifier: str
    strategy: str
    backend_instance: Any | None
    entity_id: str | None
    strategy_options: dict[str, Any]
    scalar_indexed: bool
    unsafe: bool
    resolver: ModuleResolver
    handlers: "HandlerRegistry | None"
    top_k: int
    metric: str
    exact: bool
    _config: Any

    def __init__(
        self,
        backend: str,
        backend_identifier: str,
        strategy: str,
        strategy_options: Mapping[str, Any] | None = None,
        *,
        top_k: int | None = None,
        metric: str | None = None,
        exact: bool = False,
        backend_instance: Any | None = None,
        entity_id: str | None = None,
        scalar_indexed: bool = False,
        unsafe: bool = False,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        handlers: "HandlerRegistry | None" = None,
    ) -> None:
        """Create a resolver- and config-bound handler compilation context.

        Args:
            backend (str): Routed workspace backend instance name.
            backend_identifier (str): Canonical backend type identifier.
            strategy (str): Storage strategy identifier for the routed field.
            strategy_options (Mapping[str, Any] | None): Strategy-specific options.
            top_k (int | None): Default vector near result limit.
            metric (str | None): Default vector near metric name.
            exact (bool): Require an exhaustive near fragment. A native handler
                may set `notes["exact"] = True` only when its executor enforces
                that guarantee; the router otherwise uses the portable scorer.
            backend_instance (Any | None): Optional live backend protocol object.
            entity_id (str | None): Optional entity identifier bound to compilation.
            scalar_indexed (bool): Whether scalar value indexes may be used natively.
            unsafe (bool): Whether unsafe query optimizations are allowed.
            resolver (ModuleResolver | None): Context-owned module resolver.
            config (Any | None): Config authority paired with ``resolver``.
                When both are omitted, the default Context's modules and
                config are used.
            handlers (HandlerRegistry | None): Optional pre-seeded handler registry.

        Returns:
            None: This function does not return a value.

        Raises:
            TypeError: If authority inputs or ``strategy_options`` are invalid.
            ValueError: If resolver/config pairing disagrees with the backend instance.
        """
        self.backend = str(backend)
        self.backend_identifier = str(backend_identifier)
        self.strategy = str(strategy)
        self.backend_instance = backend_instance
        bound_resolver = getattr(backend_instance, "_module_resolver", None)
        bound_config = getattr(backend_instance, "_config", None)
        if resolver is not None and bound_resolver is not None and resolver is not bound_resolver:
            raise ValueError("HandlerContext resolver differs from the backend construction authority")
        if config is not None and bound_config is not None and config is not bound_config:
            raise ValueError("HandlerContext config differs from the backend construction authority")
        selected_resolver = bound_resolver if bound_resolver is not None else resolver
        selected_config = bound_config if bound_config is not None else config
        if selected_resolver is None and selected_config is not None:
            from heavenbase.context.runtime import _config_context

            owner = _config_context(selected_config)
            if owner is None:
                raise TypeError("HandlerContext config-only authority is unbound; supply its paired resolver")
            selected_resolver = owner.modules()
        elif selected_resolver is not None and selected_config is None:
            selected_config = getattr(selected_resolver, "authority_config", None)
            if selected_config is None:
                from heavenbase.context.runtime import _default_context_config_of

                selected_config = _default_context_config_of(selected_resolver)
                if selected_config is None:
                    raise TypeError("HandlerContext resolver-only authority has no bound config; supply its paired config")
        self.resolver = module_resolver(selected_resolver)
        self.entity_id = str(entity_id) if entity_id is not None else None
        self.scalar_indexed = bool(scalar_indexed)
        if not isinstance(unsafe, bool):
            raise TypeError("HandlerContext unsafe must be a boolean")
        self.unsafe = unsafe
        if not isinstance(exact, bool):
            raise TypeError("HandlerContext exact must be a boolean")
        self.exact = exact
        if strategy_options is None:
            self.strategy_options = {}
        else:
            try:
                self.strategy_options = dict(strategy_options)
            except (TypeError, ValueError) as exc:
                raise TypeError("HandlerContext strategy_options must be a mapping; pass near parameters as top_k=... and metric=...") from exc
        if selected_config is None:
            # Neither was given: self.resolver is the default Context's
            # modules, so read that Context's config, which is not CM_HVNB
            # after set_default_context installs a Context with its own config.
            from heavenbase.context.authority import omitted_config

            selected_config = omitted_config(self.resolver)
        self._config = selected_config
        if not callable(getattr(self._config, "get", None)):
            raise TypeError("HandlerContext config must provide get(key, default=...)")
        owner = getattr(self._config, "_context", None)
        if owner is not None and owner.modules() is not self.resolver:
            raise ValueError("HandlerContext resolver and config belong to different Contexts")
        bound_config = getattr(self.resolver, "authority_config", None)
        if isinstance(self._config, ConfigManager) and bound_config is not None and bound_config is not self._config:
            raise ValueError("HandlerContext resolver and config belong to different Contexts")
        self.handlers = handlers
        if handlers is not None and handlers.module_resolver is not self.resolver:
            raise ValueError("HandlerContext handlers and resolver must share one module authority")
        self.top_k = int(top_k) if top_k is not None else int(self._config.get("heavenbase.query.near.default_top_k", default=50))
        self.metric = str(metric) if metric is not None else str(self._config.get("heavenbase.query.near.default_metric", default="cosine"))

    @property
    def config(self) -> Any:
        """Return the explicit config authority used by handler policy reads."""
        return self._config

    def indexable(self, dtype: Any) -> bool:
        """Return whether the resolved backend policy can index ``dtype``.

        Args:
            dtype (Any): Logical type class or instance being inspected.

        Returns:
            bool: ``True`` when the backend type reports the dtype as indexable.

        Raises:
            TypeError: If the backend type does not expose ``indexable(dtype, resolver=...)``.
        """
        owner = type(self.backend_instance) if self.backend_instance is not None else self.resolver.resolve("backend_type", self.backend_identifier)
        policy = getattr(owner, "indexable", None)
        if not callable(policy):
            raise TypeError(f"Backend type {self.backend_identifier!r} does not provide indexable(dtype, resolver=...)")
        return bool(policy(dtype, resolver=self.resolver))

    def fallback_reason(self, op: str, dtype: Any, strategy: Any | None = None) -> str | None:
        """Return the resolver-local fallback reason for one capability.

        Args:
            op (str): Logical operation identifier.
            dtype (Any): Logical type class, instance, identifier, or alias.
            strategy (Any | None): Storage strategy override. Defaults to the context
                strategy when omitted.

        Returns:
            str | None: Best matching fallback explanation, or ``None``.
        """
        return self._handler_registry().fallback_reason(
            self.backend_identifier,
            op,
            dtype,
            self.strategy if strategy is None else strategy,
        )

    def unsupported_reason(self, op: str, dtype: Any, strategy: Any | None = None) -> str | None:
        """Return the resolver-local unsupported-operation reason.

        Args:
            op (str): Logical operation identifier.
            dtype (Any): Logical type class, instance, identifier, or alias.
            strategy (Any | None): Storage strategy override. Defaults to the context
                strategy when omitted.

        Returns:
            str | None: Best matching unsupported explanation, or ``None``.
        """
        return self._handler_registry().unsupported_reason(
            self.backend_identifier,
            op,
            dtype,
            self.strategy if strategy is None else strategy,
        )

    def _handler_registry(self) -> "HandlerRegistry":
        """Return the supplied registry or lazily seed a standalone one."""
        if self.handlers is None:
            from .handlers.seed import seed_handlers

            self.handlers = seed_handlers(resolver=self.resolver)
        return self.handlers


@dataclass
class RowOp:
    """Carry one backend-ready replacement upsert or idempotent delete.

    `RowOp` is the only write value a backend receives. The workspace has already
    applied defaults, derives, logical validation, field routing, and encoding.

    Args:
        entity_id (str): Logical entity identifier for the physical destination.
        object_id (Any): Authoritative row identity used as the physical key.
        values (dict[str, Any]): Encoded replacement fragment for this backend.
            Delete operations normally use an empty mapping.
        op (str): Mutation kind.
            Supported values:
            - `upsert`: Replace the stored row fragment.
            - `delete`: Remove the stored row fragment idempotently.
        strategies (bool): Whether backend strategy adapters maintain their
            structures (such as sparse-GRAM keyword rows) for this mutation.
            `False` asserts that the row's adapter-bound fields are empty and
            that no adapter-owned rows exist for it, so the backend writes only
            its primary storage. A backend that cannot honor the hint runs
            full maintenance. Defaults to `True`.
    """

    entity_id: str
    object_id: Any
    values: dict[str, Any] = field(default_factory=dict)
    op: str = "upsert"
    strategies: bool = True

    def __post_init__(self) -> None:
        """Validate identity and isolate backend-ready values.

        Args:
            None.

        Returns:
            None: This initializer normalizes the mutation in place.

        Raises:
            ValueError: If the identity or operation is invalid.
            TypeError: If `values` is not a field mapping or `strategies` is not a boolean.
        """
        self.entity_id = check_entity_id(self.entity_id)
        if self.object_id in (None, ""):
            raise ValueError("RowOp object_id is required")
        self.op = raise_mismatch(("upsert", "delete"), str(self.op), name="row mutation op")
        try:
            self.values = copy_data(dict(self.values))
        except (TypeError, ValueError) as exc:
            raise TypeError("RowOp values must be a mapping of backend-ready field values") from exc
        if not isinstance(self.strategies, bool):
            raise TypeError("RowOp strategies must be a boolean")

    def is_upsert(self) -> bool:
        """Return whether this mutation replaces a row fragment.

        Args:
            None.

        Returns:
            bool: `True` for an `upsert` operation.
        """
        return self.op == "upsert"

    def is_delete(self) -> bool:
        """Return whether this mutation deletes a row fragment.

        Args:
            None.

        Returns:
            bool: `True` for a `delete` operation.
        """
        return self.op == "delete"

    def to_row(self, *, fields: Iterable[str] | None = None) -> dict[str, Any]:
        """Return a replacement fragment with authoritative identity attached.

        Args:
            fields (Iterable[str] | None): Optional physical field names retained
                from `values`. `object_id` is attached after filtering.

        Returns:
            dict[str, Any]: Isolated backend-ready replacement row.

        Raises:
            ValueError: If this is a delete operation with no replacement row.
        """
        if self.is_delete():
            raise ValueError("Delete RowOp has no replacement row")
        selected = set(fields) if fields is not None else None
        row = copy_data(self.values if selected is None else {key: value for key, value in self.values.items() if key in selected})
        row["object_id"] = self.object_id
        return row


@dataclass
class QueryFragment:
    """Represent one backend-ready read operation produced by a handler.

    Handlers compile logical query operations into fragments. Workspaces bind the
    fragments to an entity and backend route, then backends execute the opaque
    provider payload. Shared routing and explain code should use this protocol
    instead of reading common keys from ``notes`` directly.

    Args:
        backend (str): Workspace backend instance name targeted by this fragment.
        op (str): Logical operation identifier such as ``"eq"``, ``"near"``,
            ``"and"``, or ``"not"``.
        field (str): Logical field name, or ``"*"`` for whole-row and logical
            fragments. An empty logical field is normalized to ``"*"``.
        payload (Any): Backend-executable provider payload or Python row callable.
        returns (list[str]): Field names expected in backend result rows.
        notes (dict[str, Any]): Open provider metadata. Common routing facts are
            exposed through the methods on this class.

    Raises:
        ValueError: If required identity is empty or a ``"not"`` fragment does not
            contain exactly one child.
        TypeError: If collections are malformed or a logical payload contains values
            other than ``QueryFragment`` instances.
    """

    backend: str
    op: str
    field: str
    payload: Any
    returns: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and isolate the fragment envelope.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If required identity is empty or a ``"not"`` fragment does
                not contain exactly one child.
            TypeError: If collections are malformed or a logical payload contains
                values other than ``QueryFragment`` instances.
        """
        self.backend = "" if self.backend is None else str(self.backend).strip()
        self.op = "" if self.op is None else str(self.op).strip()
        self.field = "" if self.field is None else str(self.field).strip()
        if not self.backend:
            raise ValueError("QueryFragment backend must be non-empty")
        if not self.op:
            raise ValueError("QueryFragment op must be non-empty")
        if not self.field:
            if self.is_logical():
                self.field = "*"
            else:
                raise ValueError("QueryFragment field must be non-empty")
        if isinstance(self.returns, (str, bytes)):
            raise TypeError("QueryFragment returns must be an iterable of field names")
        try:
            self.returns = [str(name) for name in self.returns]
        except TypeError as exc:
            raise TypeError("QueryFragment returns must be iterable") from exc
        if not isinstance(self.notes, Mapping):
            raise TypeError("QueryFragment notes must be a mapping")
        self.notes = dict(self.notes)
        if self.is_logical():
            self.payload = self.children()

    def is_logical(self) -> bool:
        """Return whether this fragment composes child fragments.

        Args:
            None.

        Returns:
            bool: ``True`` for ``"and"``, ``"or"``, and ``"not"`` fragments.
        """
        return self.op in LOGICAL_OPS

    def children(self) -> list["QueryFragment"]:
        """Return validated children for a logical fragment.

        Args:
            None.

        Returns:
            list[QueryFragment]: Logical child fragments, or an empty list for a
            non-logical fragment.

        Raises:
            TypeError: If a logical payload is not iterable or contains another type.
            ValueError: If a ``"not"`` fragment does not contain exactly one child.
        """
        if not self.is_logical():
            return []
        try:
            children = list(self.payload)
        except TypeError as exc:
            raise TypeError(f"Logical QueryFragment {self.op!r} payload must be iterable") from exc
        if not all(isinstance(child, QueryFragment) for child in children):
            raise TypeError(f"Logical QueryFragment {self.op!r} payload must contain QueryFragment children")
        if self.op == "not" and len(children) != 1:
            raise ValueError("Logical 'not' fragment requires exactly one child")
        return children

    def kind(self, default: str | None = None) -> str | None:
        """Return the provider execution kind.

        Args:
            default (str | None): Value returned when the handler did not declare a
                provider-native kind.

        Returns:
            str | None: Provider kind such as ``"sql_where"``, or ``default``.
        """
        value = self.notes.get("kind")
        return str(value) if value is not None else default

    def mode(self) -> str:
        """Return the effective execution mode.

        Args:
            None.

        Returns:
            str: Explicit handler mode, ``"native"`` when a provider kind exists,
            or ``"py_scan"`` for a portable row-payload fragment.
        """
        value = self.notes.get("mode")
        if value is not None:
            return str(value)
        return "native" if self.kind() else "py_scan"

    def uses_scan(self) -> bool:
        """Return whether this fragment tree requires Python row scanning.

        Args:
            None.

        Returns:
            bool: ``True`` when this fragment or any logical child uses scan mode.
        """
        if self.mode() == "py_scan":
            return True
        return self.is_logical() and any(child.uses_scan() for child in self.children())

    def fallback_reason(self) -> str | None:
        """Return the first Python fallback reason in this fragment tree.

        Args:
            None.

        Returns:
            str | None: First declared fallback reason, or ``None``.
        """
        reason = self.notes.get("fallback_reason")
        if reason:
            return str(reason)
        for child in self.children():
            reason = child.fallback_reason()
            if reason:
                return reason
        return None

    def unsupported_reason(self) -> str | None:
        """Return the first unsupported-operation reason in this fragment tree.

        Args:
            None.

        Returns:
            str | None: First declared unsupported reason, or ``None``.
        """
        reason = self.notes.get("unsupported_reason")
        if reason:
            return str(reason)
        if self.notes.get("unsupported"):
            return "Unsupported operation"
        for child in self.children():
            reason = child.unsupported_reason()
            if reason:
                return reason
        return None

    def bind(self, entity_id: str, *, backend: str | None = None) -> "QueryFragment":
        """Bind this fragment tree to one entity and optional backend route.

        Args:
            entity_id (str): Entity type identifier read by the fragment tree.
            backend (str | None): Optional routed workspace backend instance name.

        Returns:
            QueryFragment: This fragment for fluent route assembly.

        Raises:
            ValueError: If ``entity_id`` or an explicit ``backend`` is empty.
        """
        entity_id = "" if entity_id is None else str(entity_id).strip()
        if not entity_id:
            raise ValueError("QueryFragment entity_id must be non-empty")
        if backend is not None:
            backend = str(backend).strip()
            if not backend:
                raise ValueError("QueryFragment backend must be non-empty")
            self.backend = backend
        self.notes["entity_id"] = entity_id
        for child in self.children():
            child.bind(entity_id, backend=backend)
        return self

    def entity_id(self) -> str | None:
        """Return the entity type identifier bound by the workspace router.

        Args:
            None.

        Returns:
            str | None: Bound entity identifier, or ``None`` before routing.
        """
        value = self.notes.get("entity_id")
        return str(value) if value is not None else None

    def filter_fragment(self) -> "QueryFragment | None":
        """Return the filter attached to a near fragment.

        The near router records the filter in both provider payloads and shared
        notes. Provider adapters should use this method instead of depending on
        either representation.

        Args:
            None.

        Returns:
            QueryFragment | None: Attached filter fragment, or ``None``.

        Raises:
            TypeError: If the attached filter is not a ``QueryFragment``.
        """
        value = self.payload.get("filter") if isinstance(self.payload, Mapping) and self.payload.get("filter") is not None else self.notes.get("filter")
        if value is None:
            return None
        if not isinstance(value, QueryFragment):
            raise TypeError(f"QueryFragment filter must be a QueryFragment, got {type(value).__name__}")
        return value

    def mark_scan(self, reason: str | None = None) -> "QueryFragment":
        """Mark this fragment as a Python scan fallback.

        Args:
            reason (str | None): Optional human-readable fallback reason surfaced by
                ``query.explain()`` and fallback warnings.

        Returns:
            QueryFragment: This fragment for fluent handler compilation.
        """
        self.notes["mode"] = "py_scan"
        if reason:
            self.notes["fallback_reason"] = str(reason)
        return self

    def execute(
        self,
        execute: Callable[["QueryFragment"], ResultFrame],
        universe: Callable[[str | None], ResultFrame],
    ) -> ResultFrame:
        """Execute this logical fragment by composing child result frames.

        Args:
            execute (Callable[[QueryFragment], ResultFrame]): Callable that executes
                one child fragment.
            universe (Callable[[str | None], ResultFrame]): Callable returning the
                entity universe for ``not``.

        Returns:
            ResultFrame: Composed result frame.

        Raises:
            TypeError: If this fragment is not a logical operation.
            ValueError: If a ``not`` fragment does not contain exactly one child.
        """
        if not self.is_logical():
            raise TypeError(f"Cannot compose non-logical fragment op {self.op!r}")
        entity_id = self.entity_id()
        children = self.children()
        if entity_id:
            for child in children:
                child.bind(entity_id)
        if self.op == "and":
            return intersect([execute(child) for child in children])
        if self.op == "or":
            return union([execute(child) for child in children])
        return difference(universe(entity_id), execute(children[0]))
