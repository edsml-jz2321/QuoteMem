"""Shared dashboard HTTP models and the canonical operation catalog.

The operation catalog is the single source for Starlette routes, runtime
request/response validation, the published OpenAPI document, and generated
frontend protocol types. Transport adapters may map exceptions to status codes,
but they must not maintain a second path or parameter catalog.
"""

from __future__ import annotations

__all__ = [
    "DASHBOARD_OPERATIONS",
    "ApiOperation",
    "AssistEvent",
    "ErrorResponse",
    "openapi_schema",
]

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, field_validator
from pydantic.json_schema import models_json_schema

JsonObject = dict[str, JsonValue]


class ProtocolModel(BaseModel):
    """Strict shared-model base for dashboard HTTP objects.

    ``advisory_fields`` names the optional request fields a service method may
    decline: the HTTP adapter omits them for a method that cannot accept them,
    instead of refusing the request. Only a field whose absence never changes
    what the method does belongs there; every other field is always passed.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    advisory_fields: ClassVar[frozenset[str]] = frozenset()


class ExtensibleResponse(BaseModel):
    """Response base for engine-owned payloads with dynamic JSON metadata."""

    model_config = ConfigDict(extra="allow")


class EmptyParameters(ProtocolModel):
    """Operation with no path, query, or request-body parameters."""


class ConfigTreeQuery(ProtocolModel):
    """Configuration tree selection."""

    scope: Optional[str] = None
    version: Optional[int] = None


class ConfigValueQuery(ProtocolModel):
    """Resolved configuration-value inspection parameters."""

    key: str = Field(min_length=1)
    reveal: bool = False


class ConfigHistoryQuery(ProtocolModel):
    """Configuration history selection."""

    scope: Optional[str] = None
    limit: int = Field(default=20, ge=1)


class ConfigDiffQuery(ProtocolModel):
    """Configuration diff endpoints."""

    scope_a: Optional[str] = None
    version_a: Optional[int] = None
    scope_b: Optional[str] = None
    version_b: Optional[int] = None


class EntitiesQuery(ProtocolModel):
    """Entity-card display switches."""

    include_system: bool = Field(default=False, alias="system")
    include_counts: bool = Field(default=True, alias="counts")


class SearchQuery(ProtocolModel):
    """Workspace search parameters."""

    text: str = Field(default="", alias="q")
    limit: Optional[int] = Field(default=None, ge=1)


class AssistantKeyQuery(ProtocolModel):
    """Assistant provider-field inspection parameters."""

    provider: str = ""
    reveal: bool = False
    field: Literal["api_key", "base_url"] = "api_key"


class AssistProgressQuery(ProtocolModel):
    """Live assistant progress token."""

    progress_id: str = Field(default="", alias="id")


class EntityPath(ProtocolModel):
    """Entity route path."""

    entity_id: str = Field(min_length=1)


class EntityFieldPath(EntityPath):
    """Entity and field route path."""

    field: str = Field(min_length=1)


class EntityRowPath(EntityPath):
    """Entity and object route path."""

    object_id: str = Field(min_length=1)


class ConfigSetRequest(ProtocolModel):
    """Write one scoped configuration value."""

    key: str = Field(min_length=1)
    value: JsonValue
    scope: Optional[str] = None
    parse: Literal["auto", "json", "raw"] = "auto"


class ConfigKeyRequest(ProtocolModel):
    """Mutate one scoped configuration key."""

    key: str = Field(min_length=1)
    scope: Optional[str] = None


class WorkspaceOpenRequest(ProtocolModel):
    """Open one registered workspace."""

    ws_id: str = Field(min_length=1)


class WorkspacePath(ProtocolModel):
    """Registered workspace route path."""

    ws_id: str = Field(min_length=1)


class DashboardPreferencesRequest(ProtocolModel):
    """Update one or more shared dashboard appearance preferences."""

    theme: Optional[Literal["theme-light", "theme-dark", "theme-anthropic"]] = None
    language: Optional[Literal["en", "zh"]] = None
    font: Optional[Literal["small", "default", "large"]] = None
    font_scale: Optional[Literal["small", "default", "large", "xlarge"]] = None


class EntityRowsRequest(ProtocolModel):
    """Read one page of entity rows."""

    filter: Optional[JsonObject] = None
    order: Optional[list[str]] = None
    select: Optional[list[str]] = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, gt=0)


class AssistTurn(ProtocolModel):
    """One prior assistant conversation turn."""

    role: Literal["user", "assistant"]
    content: str


class AssistRequest(ProtocolModel):
    """One blocking or streaming assistant request."""

    question: str = Field(min_length=1)
    context: Optional[JsonObject] = None
    history: Optional[list[AssistTurn]] = None
    progress_id: Optional[str] = None
    preset: Optional[str] = None
    read_only: Optional[bool] = None
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    turn_id: Optional[str] = Field(default=None, max_length=128)
    # Session-log identifiers only: an override written before 0.1.2.3 records no log and is served without them.
    advisory_fields: ClassVar[frozenset[str]] = frozenset({"conversation_id", "turn_id"})

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        """Reject whitespace-only questions at the protocol boundary."""
        if not value.strip():
            raise ValueError("assist requires a non-empty question")
        return value


class AssistStreamRequest(ProtocolModel):
    """One streaming assistant request."""

    question: str = Field(min_length=1)
    context: Optional[JsonObject] = None
    history: Optional[list[AssistTurn]] = None
    preset: Optional[str] = None
    read_only: Optional[bool] = None
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    turn_id: Optional[str] = Field(default=None, max_length=128)
    # Session-log identifiers only: an override written before 0.1.2.3 records no log and is served without them.
    advisory_fields: ClassVar[frozenset[str]] = frozenset({"conversation_id", "turn_id"})

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        """Reject whitespace-only questions at the protocol boundary."""
        if not value.strip():
            raise ValueError("assist requires a non-empty question")
        return value


class MetaResponse(ProtocolModel):
    """Dashboard package identity."""

    name: str
    version: str


class ConfigMetaResponse(ProtocolModel):
    """Configuration store metadata."""

    scopes: list[str]
    scope: str
    scope_chain: list[str]
    versions: dict[str, Optional[int]]
    root: str


class ConfigTreeResponse(ProtocolModel):
    """Merged configuration tree and leaf origins."""

    tree: JsonObject
    origins: dict[str, Literal["builtin", "override", "custom"]]
    scope: str
    version: Optional[int]


class ConfigMutationResponse(ProtocolModel):
    """Successful configuration mutation."""

    ok: Literal[True]
    key: str


class ConfigResetResponse(ConfigMutationResponse):
    """Successful configuration reset."""

    origin: Literal["builtin", "custom"]


class ConfigHistoryRow(ExtensibleResponse):
    """One retained configuration version."""

    scope: str
    ver: int
    gen: int
    pkg_ver: Optional[str] = None
    created_at: JsonValue = None


class ConfigHistoryResponse(RootModel[list[ConfigHistoryRow]]):
    """Retained configuration versions."""


class ConfigDiffResponse(ProtocolModel):
    """Unified configuration diff."""

    diff: str


class WorkspaceCard(ExtensibleResponse):
    """Registered workspace summary."""

    id: str
    active: bool
    open: bool


class WorkspacesResponse(RootModel[list[WorkspaceCard]]):
    """Registered workspace summaries."""


class WorkspaceRemoveResponse(ProtocolModel):
    """Workspace unregister result."""

    ws_id: str
    removed: bool
    was_open: bool


class WorkspaceBackendHealth(ExtensibleResponse):
    """One backend in an open workspace health payload."""

    name: str
    type: Optional[str] = None
    provider: Optional[str] = None
    capabilities: JsonObject = Field(default_factory=dict)


class WorkspaceHealthResponse(ProtocolModel):
    """Open workspace health payload."""

    ws_id: str
    workspace: str
    entities: list[str]
    backends: list[WorkspaceBackendHealth]


class DashboardPreferencesResponse(ProtocolModel):
    """Shared dashboard appearance preferences."""

    theme: Literal["theme-light", "theme-dark", "theme-anthropic"]
    language: Literal["en", "zh"]
    font: Literal["small", "default", "large"]
    font_scale: Literal["small", "default", "large", "xlarge"]
    configured: bool


class EntityCard(ProtocolModel):
    """One entity browser card."""

    entity_id: str
    name: str
    desc: Optional[str] = None
    tags: list[str]
    system: bool
    extension: Optional[str] = None
    count: Optional[int] = None


class EntitiesResponse(RootModel[list[EntityCard]]):
    """Entity browser cards."""


class EntitySchemaResponse(ProtocolModel):
    """Entity schema, placement, and capability metadata."""

    schema_: JsonObject = Field(alias="schema")
    storage: dict[str, list[JsonObject]]
    capabilities: JsonObject
    backends: list[JsonObject]


class EntityRowsResponse(ProtocolModel):
    """One entity row page."""

    rows: list[JsonObject]
    columns: list[str]
    total: int
    has_more: bool
    offset: int
    limit: int


class EntityRowResponse(RootModel[JsonObject]):
    """One entity row."""


class ColumnStatsResponse(ExtensibleResponse):
    """Column statistics with kind-specific extension fields."""

    field: str
    dtype: str
    kind: str
    total: int
    scanned: int
    sampled: bool
    nulls: int
    null_pct: float


class SearchResponse(ProtocolModel):
    """Ranked workspace search rows with the bound the cap applied.

    Ranking scores every matching card before the cap applies, so ``total`` is
    the exact match count rather than a floor, and ``truncated`` says whether
    the cap withheld cards.
    """

    rows: list[JsonObject]
    n: int
    total: int
    truncated: bool


class AssistantKeyResponse(ProtocolModel):
    """Resolved assistant provider-field status."""

    provider: str
    field: Literal["api_key", "base_url"]
    set: bool
    tail: str
    source: Literal["environment", "fallback", "literal"]
    env_var: Optional[str] = None
    env_set: Optional[bool] = None
    value: Optional[str] = None


class ConfigValueResponse(ProtocolModel):
    """Resolved configuration value and its effective source."""

    key: str
    set: bool
    tail: str
    source: Literal["environment", "manual"]
    env_var: Optional[str] = None
    env_set: Optional[bool] = None
    value: Optional[str] = None


class AssistFrame(ProtocolModel):
    """Lossless table frame returned by an assistant query tool."""

    entity: str
    spec: JsonObject
    columns: list[str]
    rows: list[JsonObject]
    n: int
    # The engine cap cuts result groups before the table clip does, and it knows
    # the exact match count, so the frame carries the bound it was rendered under.
    total: Optional[int] = None
    truncated: bool = False


class AssistSQLFrame(ProtocolModel):
    """Lossless bounded result from one assistant SQL execution tool."""

    database_id: str
    sql: str
    params: JsonObject
    columns: list[str]
    rows: list[JsonObject]
    n: int
    total: Optional[int] = None
    limit: Optional[int] = None
    truncated: bool = False
    readonly: bool = True


class AssistTimelineEvent(ExtensibleResponse):
    """One assistant reasoning/tool timeline item."""

    type: Literal["think", "tool"]
    text: Optional[str] = None
    name: Optional[str] = None
    args: Optional[str] = None
    result: Optional[str] = None
    frame: Optional[AssistFrame] = None
    sql_frame: Optional[AssistSQLFrame] = None


class AssistResponse(ProtocolModel):
    """Completed assistant response."""

    text: str
    think: str
    served_model: str
    frames: list[AssistFrame]
    sql_frames: list[AssistSQLFrame]
    events: list[AssistTimelineEvent]
    elapsed: float
    truncated: bool = False
    capture_warning: Optional[str] = None


class AssistProgressLine(ProtocolModel):
    """One live assistant activity line."""

    type: Literal["tool", "think"]
    text: str


class AssistProgressResponse(ProtocolModel):
    """Live assistant activity feed."""

    lines: list[AssistProgressLine]


class AssistEvent(ExtensibleResponse):
    """One server-sent assistant stream event."""

    kind: Literal["think", "text", "tool", "tool_result", "done", "error"]
    text: Optional[str] = None
    think: Optional[str] = None
    served_model: Optional[str] = None
    frames: Optional[list[AssistFrame]] = None
    sql_frames: Optional[list[AssistSQLFrame]] = None
    events: Optional[list[AssistTimelineEvent]] = None
    elapsed: Optional[float] = None
    truncated: Optional[bool] = None
    name: Optional[str] = None
    args: Optional[str] = None
    frame: Optional[AssistFrame] = None
    sql_frame: Optional[AssistSQLFrame] = None
    capture_warning: Optional[str] = None


class ErrorResponse(ProtocolModel):
    """Dashboard protocol error."""

    error: str


@dataclass(frozen=True)
class ApiOperation:
    """One canonical dashboard HTTP operation.

    Attributes:
        operation_id: Stable generated-client operation identifier.
        method: HTTP method.
        path: URL path template.
        summary: Short OpenAPI summary.
        service_method: ``DashboardAPI`` method bound to the route.
        response_model: Successful response model.
        path_model: Path-parameter model.
        query_model: Query-parameter model.
        body_model: Optional JSON request model.
        media_type: Successful response media type.
        stream: Whether the response is a server-sent event stream.
    """

    operation_id: str
    method: Literal["DELETE", "GET", "POST"]
    path: str
    summary: str
    service_method: str
    response_model: type[BaseModel]
    path_model: type[BaseModel] = EmptyParameters
    query_model: type[BaseModel] = EmptyParameters
    body_model: Optional[type[BaseModel]] = None
    media_type: str = "application/json"
    stream: bool = False


DASHBOARD_OPERATIONS = (
    ApiOperation("dashboardMeta", "GET", "/api/meta", "Read dashboard metadata", "meta", MetaResponse),
    ApiOperation("configMeta", "GET", "/api/config/meta", "Read configuration metadata", "config_meta", ConfigMetaResponse),
    ApiOperation(
        "configTree",
        "GET",
        "/api/config/tree",
        "Read the merged configuration tree",
        "config_tree",
        ConfigTreeResponse,
        query_model=ConfigTreeQuery,
    ),
    ApiOperation(
        "configValueStatus",
        "GET",
        "/api/config/value-status",
        "Inspect one resolved configuration value",
        "config_value_status",
        ConfigValueResponse,
        query_model=ConfigValueQuery,
    ),
    ApiOperation(
        "configSet",
        "POST",
        "/api/config/set",
        "Set one configuration value",
        "config_set",
        ConfigMutationResponse,
        body_model=ConfigSetRequest,
    ),
    ApiOperation(
        "configUnset",
        "POST",
        "/api/config/unset",
        "Remove one custom configuration value",
        "config_unset",
        ConfigMutationResponse,
        body_model=ConfigKeyRequest,
    ),
    ApiOperation(
        "configReset",
        "POST",
        "/api/config/reset",
        "Reset one configuration value",
        "config_reset",
        ConfigResetResponse,
        body_model=ConfigKeyRequest,
    ),
    ApiOperation(
        "configHistory",
        "GET",
        "/api/config/history",
        "Read configuration history",
        "config_history",
        ConfigHistoryResponse,
        query_model=ConfigHistoryQuery,
    ),
    ApiOperation(
        "configDiff",
        "GET",
        "/api/config/diff",
        "Compare configuration versions",
        "config_diff",
        ConfigDiffResponse,
        query_model=ConfigDiffQuery,
    ),
    ApiOperation(
        "dashboardPreferences",
        "GET",
        "/api/preferences",
        "Read shared dashboard preferences",
        "preferences",
        DashboardPreferencesResponse,
    ),
    ApiOperation(
        "dashboardPreferencesUpdate",
        "POST",
        "/api/preferences",
        "Update shared dashboard preferences",
        "update_preferences",
        DashboardPreferencesResponse,
        body_model=DashboardPreferencesRequest,
    ),
    ApiOperation("workspaces", "GET", "/api/workspaces", "List registered workspaces", "workspaces", WorkspacesResponse),
    ApiOperation(
        "workspaceRemove",
        "DELETE",
        "/api/workspaces/{ws_id}",
        "Unregister a workspace",
        "remove_workspace",
        WorkspaceRemoveResponse,
        path_model=WorkspacePath,
    ),
    ApiOperation(
        "workspaceOpen",
        "POST",
        "/api/workspace/open",
        "Open a registered workspace",
        "open",
        WorkspaceHealthResponse,
        body_model=WorkspaceOpenRequest,
    ),
    ApiOperation("workspaceHealth", "GET", "/api/workspace/health", "Read open workspace health", "health", WorkspaceHealthResponse),
    ApiOperation(
        "workspaceEntities",
        "GET",
        "/api/workspace/entities",
        "List workspace entities",
        "entities",
        EntitiesResponse,
        query_model=EntitiesQuery,
    ),
    ApiOperation(
        "workspaceEntitySchema",
        "GET",
        "/api/workspace/entity/{entity_id}/schema",
        "Read one entity schema",
        "entity_schema",
        EntitySchemaResponse,
        path_model=EntityPath,
    ),
    ApiOperation(
        "workspaceEntityRows",
        "POST",
        "/api/workspace/entity/{entity_id}/rows",
        "Read one page of entity rows",
        "entity_rows",
        EntityRowsResponse,
        path_model=EntityPath,
        body_model=EntityRowsRequest,
    ),
    ApiOperation(
        "workspaceColumnStats",
        "GET",
        "/api/workspace/entity/{entity_id}/stats/{field}",
        "Read statistics for one entity field",
        "column_stats",
        ColumnStatsResponse,
        path_model=EntityFieldPath,
    ),
    ApiOperation(
        "workspaceEntityRow",
        "GET",
        "/api/workspace/entity/{entity_id}/row/{object_id}",
        "Read one entity row",
        "entity_row",
        EntityRowResponse,
        path_model=EntityRowPath,
    ),
    ApiOperation(
        "workspaceSearch",
        "GET",
        "/api/workspace/search",
        "Search the open workspace",
        "search",
        SearchResponse,
        query_model=SearchQuery,
    ),
    ApiOperation(
        "assistantKey",
        "GET",
        "/api/assistant/key",
        "Inspect one assistant provider field",
        "assistant_key_status",
        AssistantKeyResponse,
        query_model=AssistantKeyQuery,
    ),
    ApiOperation(
        "assist",
        "POST",
        "/api/assist",
        "Run one assistant turn",
        "assist",
        AssistResponse,
        body_model=AssistRequest,
    ),
    ApiOperation(
        "assistProgress",
        "GET",
        "/api/assist/progress",
        "Read live assistant progress",
        "assist_progress",
        AssistProgressResponse,
        query_model=AssistProgressQuery,
    ),
    ApiOperation(
        "assistStream",
        "POST",
        "/api/assist/stream",
        "Stream one assistant turn",
        "assist_stream",
        AssistEvent,
        body_model=AssistStreamRequest,
        media_type="text/event-stream",
        stream=True,
    ),
)


def _component_models() -> list[type[BaseModel]]:
    models: list[type[BaseModel]] = [ErrorResponse, AssistEvent]
    for operation in DASHBOARD_OPERATIONS:
        models.extend((operation.path_model, operation.query_model, operation.response_model))
        if operation.body_model is not None:
            models.append(operation.body_model)
    return list(dict.fromkeys(models))


def _model_ref(model: type[BaseModel]) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{model.__name__}"}


def _parameters(model: type[BaseModel], location: Literal["path", "query"]) -> list[dict[str, Any]]:
    if model is EmptyParameters:
        return []
    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
    required = set(schema.get("required") or [])
    parameters = []
    for field_name, field in model.model_fields.items():
        wire_name = field.alias or field_name
        parameters.append(
            {
                "name": wire_name,
                "in": location,
                "required": location == "path" or wire_name in required,
                "schema": schema["properties"][wire_name],
            }
        )
    return parameters


def openapi_schema() -> dict[str, Any]:
    """Build the OpenAPI 3.1 document from the canonical operation catalog.

    Args:
        None.

    Returns:
        dict[str, Any]: JSON-serializable OpenAPI document.
    """
    models = _component_models()
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="HeavenBase Dashboard API",
        ref_template="#/components/schemas/{model}",
    )
    paths: dict[str, Any] = {}
    for operation in DASHBOARD_OPERATIONS:
        parameters = _parameters(operation.path_model, "path") + _parameters(operation.query_model, "query")
        operation_schema: dict[str, Any] = {
            "operationId": operation.operation_id,
            "summary": operation.summary,
            "parameters": parameters,
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {operation.media_type: {"schema": _model_ref(operation.response_model)}},
                },
                "400": {
                    "description": "Invalid request",
                    "content": {"application/json": {"schema": _model_ref(ErrorResponse)}},
                },
                "403": {
                    "description": "Local host or origin rejected",
                    "content": {"application/json": {"schema": _model_ref(ErrorResponse)}},
                },
                "404": {
                    "description": "Resource not found",
                    "content": {"application/json": {"schema": _model_ref(ErrorResponse)}},
                },
                "500": {
                    "description": "Application or provider failure",
                    "content": {"application/json": {"schema": _model_ref(ErrorResponse)}},
                },
            },
        }
        if operation.body_model is not None:
            operation_schema["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": _model_ref(operation.body_model)}},
            }
        paths.setdefault(operation.path, {})[operation.method.lower()] = operation_schema
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "HeavenBase Dashboard API",
            "version": "1.0.0",
            "description": "Loopback-local protocol used by the HeavenBase dashboard.",
        },
        "paths": paths,
        "components": {"schemas": definitions.get("$defs", {})},
    }
