"""Private workspace service assembly."""

from __future__ import annotations

from heavenbase.utils import Any, Callable

from ..catalog.audit import _CatalogAuditService
from ..catalog.service import _CatalogService
from ..graph.mutation import _GraphMutationService
from ..graph.service import _GraphService
from ..mcp.service import _MCPService
from ..metaschema.service import _MetaSchemaService
from ..query.explain import _QueryExplainService
from ..query.router import _WorkspaceRouter
from ..query.search import _SearchService
from ..query.service import _QueryService
from ..rows.access import _RowAccessService
from ..rows.codec import _RowCodec
from ..rows.crud import _CrudService
from ..rows.inputs import _RowInputService
from ..rows.lookup import _RowLookupService
from ..rows.materialize import _RowMaterializer
from ..rows.store import _RowStore
from ..rows.writer import _EntityWriter
from ..schema.metadata import _SchemaMetadataService
from ..schema.placement import _SchemaPlacementService, _StorageProfileService
from ..schema.service import _SchemaService
from .extensions import _ExtensionService
from .maintenance import _MaintenanceService

__all__ = ("attach_workspace_services", "workspace_service_roles")

ServiceFactory = Callable[[Any], Any]
ServiceSpec = tuple[str, str, ServiceFactory]


def _query_explain(workspace: Any) -> _QueryExplainService:
    """Build query explanation after the router dependency is attached."""
    return _QueryExplainService(workspace._router)


_SERVICE_SPECS: tuple[ServiceSpec, ...] = (
    ("_row_store", "backend row-fragment lookup, scans, counts, merges, and query hydration", _RowStore),
    ("_row_lookup", "object-id validation, entity disambiguation, and batch lookups", _RowLookupService),
    ("_row_codec", "field routing plus logical-type encoding and decoding at backend boundaries", _RowCodec),
    ("_catalog", "Catalog row identity, signatures, derives, and write-row construction", _CatalogService),
    ("_catalog_audit", "Catalog consistency audit and repair planning", _CatalogAuditService),
    ("_schema_metadata", "stable schema export, hook references, and query capability metadata", _SchemaMetadataService),
    ("_metaschema", "MetaSchema row ids, writes, publishing, and registry metadata refresh", _MetaSchemaService),
    ("_storage_profiles", "workspace storage-profile overrides over process defaults", _StorageProfileService),
    ("_schema_placement", "storage-plan resolution and storage-stability validation", _SchemaPlacementService),
    ("_schema", "entity normalization, registration, alteration, and descriptions", _SchemaService),
    ("_materializer", "defaults, derives, object-id materialization, and schema-version checks", _RowMaterializer),
    ("_search", "Catalog and MetaSchema card search", _SearchService),
    ("_row_inputs", "public row input, patch, and delete-target normalization", _RowInputService),
    ("_graph_mutation", "strong GraphEdge endpoint validation and target-deletion policy", _GraphMutationService),
    ("_crud", "row command semantics for upsert, set, get, delete, exists, and count", _CrudService),
    ("_row_access", "row list/item views and Python collection protocol adapters", _RowAccessService),
    ("_writer", "routed backend write fan-out and Catalog cleanup planning", _EntityWriter),
    ("_router", "query route orchestration, frame merging, hydration, ordering, and projection", _WorkspaceRouter),
    ("_query_explain", "honest route diagnostics over router compilation metadata", _query_explain),
    ("_query", "fluent, JSON, execute, explain, and card-search query entry points", _QueryService),
    ("_graph_service", "public GraphEdge traversal, shortest-path, audit, and cleanup commands", _GraphService),
    ("_mcp", "workspace and backend MCP Toolkit exposure", _MCPService),
    ("_extension_service", "extension lifecycle, APIs, metadata, and extension-aware entity lists", _ExtensionService),
    ("_maintenance", "drop, refresh, audit, repair, and health operations", _MaintenanceService),
)


def workspace_service_roles() -> tuple[tuple[str, str], ...]:
    """Return private workspace service slots and their one-line responsibilities.

    Args:
        None.

    Returns:
        tuple[tuple[str, str], ...]: Ordered ``(slot, objective)`` pairs for every attached service.
    """
    return tuple((slot, objective) for slot, objective, _ in _SERVICE_SPECS)


def attach_workspace_services(workspace: Any) -> None:
    """Attach private role services to a prepared workspace facade.

    Args:
        workspace (Any): Workspace facade whose identity, backends, handlers, capabilities, storage resolver,
            entity map, and storage-plan map already exist.

    Returns:
        None: The services are attached to ``workspace`` in dependency order.
    """
    for slot, _, factory in _SERVICE_SPECS:
        setattr(workspace, slot, factory(workspace))
