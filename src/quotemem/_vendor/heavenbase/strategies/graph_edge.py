"""Graph edge storage strategy."""

from __future__ import annotations

__all__ = [
    "GraphEdge",
    "resolve_graph_endpoint",
]

from heavenbase.utils import Any, Literal, Mapping, raise_mismatch

from .base import StorageStrategy


class GraphEdge(StorageStrategy):
    """Store a HyperG field as graph relationships (one edge per mini-row).

    ``target`` names the endpoint column — the HyperG column whose ``Identifier`` value is the edge's
    destination node; the remaining columns are edge properties. Use it whenever the endpoint is a plain
    ``Identifier``, or to disambiguate when more than one column is an ``Identifier``. When ``target`` is
    omitted the endpoint is auto-detected as the sole ``Identifier[Entity]`` column.

    The endpoint's *target entity* is taken from its ``Identifier[Entity]`` scope, else it defaults to the
    *owning* entity (a self-referential edge). So a CROSS-ENTITY edge MUST type the endpoint column as
    ``Identifier[TargetEntity]`` — a plain ``Identifier`` resolves to the owning entity, so a cross-entity
    target id would not be found and the strict write / traversal would fail loudly. There is no
    ``entity=`` argument by design: declare the target entity on the column's type.

    Edge semantics — a record stores only its OUTGOING edges; a node's role (start / middle / end) is not
    declared, it emerges from the data:
      - The field value is the record's full set of out-edges: one mini-row per edge, an empty list ``[]``
        meaning no out-edges — a terminal/end node for this relationship. To say "no downstream" use ``[]``;
        never write a row with a missing endpoint (the ``target`` column is required when a row is present).
      - Inbound edges are NOT stored on the record (a "middle" node looks the same as a "start" node in its
        own row); they live on the source records, so query who points here with
        ``ws.traverse(field, id, direction="in")``.
      - On upsert the field value REPLACES that field's out-edges. Use ``set`` when patch semantics are
        required.
      - Target endpoints are strong references. A workspace write rejects a missing endpoint; deleting a
        referenced target uses this field's ``on_delete`` policy.

    Examples:
        cites       = field(HyperG["cited": Identifier]).store(strategy=GraphEdge(target="cited"))                                 # self-ref (plain Identifier -> owning entity)
        authored_by = field(HyperG["author": Identifier[Author], "position": Integer]).store(strategy=GraphEdge(target="author"))  # cross-entity (typed -> Author)
    """

    identifier = "graph"
    option_fields = ("on_delete",)

    def __init__(self, target: str | None = None, *, on_delete: Literal["restrict", "detach"] = "restrict") -> None:
        """Create graph-edge placement metadata.

        Args:
            target (str | None): Optional HyperG child column that stores the
                destination object id. When omitted, HeavenBase requires exactly
                one scoped `Identifier[Entity]` child.
            on_delete (str): Target deletion policy.
                Supported values:
                - `restrict`: Reject deletion while a surviving source points to
                  the target. This is the default.
                - `detach`: Remove matching edges before deleting the target.

        Returns:
            None: This initializer stores placement metadata.

        Raises:
            ValueError: If `on_delete` is unsupported.
        """
        self.on_delete = raise_mismatch(("restrict", "detach"), str(on_delete).lower(), name="GraphEdge on_delete")
        super().__init__(on_delete=self.on_delete if self.on_delete != "restrict" else None)
        self.target = target

    @classmethod
    def normalize_options(cls, options: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize serialized graph-edge options.

        Args:
            options (Mapping[str, Any]): Strategy option mapping.

        Returns:
            dict[str, Any]: Normalized non-default `on_delete` option.

        Raises:
            ValueError: If an option name or deletion policy is unsupported.
        """
        data = super().normalize_options(options)
        if "on_delete" not in data:
            return data
        policy = raise_mismatch(("restrict", "detach"), str(data["on_delete"]).lower(), name="GraphEdge on_delete")
        return {"on_delete": policy} if policy != "restrict" else {}

    @property
    def params(self) -> dict[str, str]:
        """Return the endpoint column as per-field placement metadata.

        Returns:
            dict[str, str]: A `target` mapping when explicitly configured,
            otherwise an empty mapping.
        """
        return {"target": self.target} if self.target is not None else {}


def resolve_graph_endpoint(field_schema: Any, target: str | None, owning_entity: str) -> tuple[str, str, list[str]]:
    """Resolve endpoint metadata for one graph-edge field.

    Args:
        field_schema (Any): Field schema with a HyperG logical type.
        target (str | None): Explicit endpoint child column, or `None` to
            auto-detect the sole scoped `Identifier[Entity]` child.
        owning_entity (str): Entity identifier used for self-referential plain
            `Identifier` endpoints.

    Returns:
        tuple[str, str, list[str]]: Endpoint column, target entity identifier,
        and edge-property child columns.

    Raises:
        ValueError: If the explicit endpoint is missing or not an `Identifier`,
            or automatic detection does not find exactly one scoped endpoint.
    """
    columns = list(getattr(field_schema.dtype, "fields", []))
    if target is not None:
        endpoint = next((column for column in columns if column.name == target), None)
        if endpoint is None:
            raise ValueError(f"GraphEdge field {field_schema.name!r}: target column {target!r} is not a column of the HyperG field")
        if type(endpoint.dtype).identifier != "identifier":
            raise ValueError(f"GraphEdge field {field_schema.name!r}: target column {target!r} must be an Identifier (it is the edge endpoint)")
        target_entity = getattr(endpoint.dtype, "target_entity", None) or owning_entity
    else:
        endpoints = [column for column in columns if type(column.dtype).identifier == "identifier" and getattr(column.dtype, "target_entity", None)]
        if len(endpoints) != 1:
            raise ValueError(
                f"GraphEdge field {field_schema.name!r}: pass GraphEdge(target='<column>') to name the endpoint, "
                f"or declare exactly one Identifier[Entity] column to auto-detect it (found {len(endpoints)})"
            )
        endpoint = endpoints[0]
        target_entity = endpoint.dtype.target_entity
    prop_cols = [column.name for column in columns if column.name != endpoint.name]
    return endpoint.name, target_entity, prop_cols
