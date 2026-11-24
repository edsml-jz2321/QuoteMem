# HeavenBase

<p align="center">
  <strong>One data model. Many backends. A structured data surface for applications and agents.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/heavenbase/"><img src="https://img.shields.io/pypi/v/heavenbase?label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/heavenbase/"><img src="https://img.shields.io/pypi/pyversions/heavenbase" alt="Supported Python versions"></a>
  <a href="https://github.com/Magolor/HeavenBase/actions/workflows/code-quality.yml"><img src="https://github.com/Magolor/HeavenBase/actions/workflows/code-quality.yml/badge.svg" alt="Code quality"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/heavenbase" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://ahvn.top/">Documentation</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#backends">Backends</a> ·
  <a href="#contributing">Contributing</a>
</p>

HeavenBase is a Python data engine for defining structured entities once, placing their fields across the storage systems you already use, and accessing them through one typed workspace.

It is not a database replacement. HeavenBase coordinates schema, storage placement, query routing, metadata, and agent-facing tools above SQL, vector, search, graph, and file backends.

> [!NOTE]
> HeavenBase is alpha software. The core data model and workspace workflows are usable today, but APIs may still change between releases. This README follows the checked-out source; the PyPI badge shows the latest published package.

## Why HeavenBase?

AI applications often need data from several systems, but each system has its own schema, client, query language, and operational limits. HeavenBase provides one explicit layer for that coordination:

- **Model once.** Define typed entities with stable object identity and JSON-compatible schemas.
- **Place fields deliberately.** Keep scalar data in SQL, embeddings in a vector store, searchable text in a search backend, and relationships in a graph backend.
- **Query consistently.** Use Python expressions or JSON queries for filtering, projection, ordering, aggregation, vector search, and traversal.
- **Stay inspectable.** Query `Catalog` and `MetaSchema` like ordinary entities, and use `explain()` to see where and how a query will run.
- **Expose safe tools to agents.** Turn a workspace into a profile-scoped MCP server without creating a second data API.

## Built on HeavenBase

- **[GlossWise](https://github.com/Magolor/GlossWise)** is a public downstream application for terminology-safe translation context. It builds its Python SDK, CLI, HeavenBase extension, agent Skill, search and OCR workflows, and MCP profiles on one HeavenBase workspace model.

## Installation

HeavenBase supports Python 3.10 through 3.13.

```bash
pip install heavenbase
hb setup
```

`hb setup` initializes the global configuration and stable default workspace. Creating another workspace does not activate it or replace that default. The `debug` workspace preset requires no external services and is the recommended place to start.

Install optional integrations only when you need them:

| Extra | Includes | Install |
| --- | --- | --- |
| `interop` | Pydantic, NumPy, pandas, PyArrow, and SQLAlchemy interop | `pip install "heavenbase[interop]"` |
| `daft` | Daft and PyArrow interop | `pip install "heavenbase[daft]"` |
| `sql` | PostgreSQL- and MySQL-protocol families, SQL Server, Oracle, Trino, and ClickHouse drivers | `pip install "heavenbase[sql]"` |
| `full` | Maintained database, vector, search, graph, and LLM provider clients available for the current platform, plus SQLGlot for dialect-aware SQL parsing | `pip install "heavenbase[full]"` |

For source development, see [Contributing](#contributing).

## Quickstart

Define an entity, open a workspace, write a row, and query it:

```python
import heavenbase as hb


class Product(hb.Entity):
    name = hb.field(hb.ShortText).desc("Display name")
    description = hb.field(hb.LongText).desc("Searchable description")
    price = hb.field(hb.Float).desc("List price")
    tags = hb.field(hb.Array[hb.ShortText]).default([])


ws = hb.HeavenBase("shop", preset="debug")
ws.register(Product)

product_id = ws.upsert(
    Product,
    {
        "name": "Oak desk",
        "description": "Writing desk with a cable tray",
        "price": 129.0,
        "tags": ["office", "furniture"],
    },
)

query = (
    ws.query(Product)
    .where(Product.price < 150)
    .where(Product.tags.array_contains("office"))
    .select("name", "price")
)

print(product_id)
print(query.execute().rows())
```

`register()` establishes an Entity class and its storage plan; it does not
write an Entity instance. Row operations such as `upsert`, `set`, and `delete`
only operate after that class is already present in the workspace and never
create schema state as a side effect. The expert-level exception is an active
Entity-kind `MetaSchema` write carrying a full Entity definition in `meta`;
that command delegates to schema registration when the class is absent, while
the caller-authored metadata row remains non-authoritative.

Queries are immutable specification values: each transformation returns a new
query, and every `execute()` recomputes. Its `ResultFrame` return value is a
transport container, so application code should explicitly export it through
`rows()`, `scalar()`, `to_pandas()`, `to_pyarrow()`, or another known format
rather than print or implicitly coerce the frame itself.

`hb.HeavenBase("shop", ...)` is the idempotent durable entry point: it creates
and registers `shop` when absent, or opens the compatible registered workspace
when present. It never changes the active workspace. An explicit `preset=` or
`backends=` value is a construction-spec assertion and raises
`FileExistsError` when it conflicts with the registered definition; evolving
Entity schemas are not part of that comparison. When both settings are omitted
for an existing id, HeavenBase uses the persisted construction spec rather than
reinterpreting current defaults. Use
`hb.HeavenBase.load("shop")` when absence must instead raise `KeyError`. Use
the returned object directly, as above, or call `ws.activate()` when later
name-free loads should select it.

For ready Backend instances, compatibility compares only the complete mapping
from workspace backend name to canonical Backend identifier; connection details
and other runtime state are deliberately ignored. `load()` can reuse the live
facade. After that facade is released or the process restarts, reconstruction
requires matching same-Context live instances or explicit Backend input with
the same name-to-identifier topology.

Workspace manifests are complete reconstructive shell definitions. Version 2
stores one top-level `construction` envelope instead of a parallel constructor
`config`: replayable Backend settings are isolated under
`{kind: "backends", config: ...}`, while live objects record only
`{kind: "runtime", identifiers: {name: identifier}}`. Replayable construction
may be replaced in full with `preset=` or `backends=` during import. Runtime
input may change connection details but must match every recorded Backend name
and identifier; an identifier alone never fabricates connection configuration
or a live client.

Pass `detached=True` for a caller-owned workspace that must not be registered,
selected, or found by `load()`:

```python
scratch = hb.HeavenBase(
    "scratch",
    backends={"main": {"type": "inmem"}},
    detached=True,
)
```

Detached is a lifecycle policy, not a storage policy: it does not imply
temporary or isolated Backend data, cleanup is not automatic, and the workspace
cannot be activated. Retain the facade and call `scratch.drop()` explicitly
when destructive Backend cleanup is intended.

The stable workspace id also derives its configuration layer: `shop` inherits the
base `heavenbase` config and overlays `heavenbase.shop`. No separate workspace
scope is stored, and activation does not change which layer a workspace uses.
Explicit constructor or manifest construction settings remain authoritative.
Later operations may observe scoped edits when they resolve configuration, but
already-built resources are not guaranteed to update; reopening the workspace is
the safe synchronization boundary.

Projected entity rows retain `object_id`, so a compact query result can still be used with `get`, `set`, and `delete`. If a row omits `object_id` but provides `name`, HeavenBase derives a deterministic identifier.

Use `ws.query(Document).near("embedding", query_vector, top_k=8, exact=True)`
when exhaustive metric Top-K is required. Filters apply before Top-K. Proven
native implementations execute on the provider; other routes use an exhaustive
scan or fail if enumeration is unavailable. Inspect `explain()` for the selected
route. For caller-owned immutable knowledge generations, add
`cache_version="knowledge-v1"` to the exact near clause to permit bounded
workspace-owned candidate reuse. Omit it for live reads; external writes require
a new version or `ws.refresh(entity=Document)`. The token does not select a
historical database snapshot. Query results are never cached; only candidate rows
are reused, and only within one `cache_version`. `cache_version` is experimental
and may change without a deprecation period; see the
[exact-near contract](docs/resources/architecture/query-values-and-results.md#exact-nearest-neighbors).

The same query can be expressed as JSON for APIs and agent-authored calls:

```python
rows = ws.query_json(
    Product,
    {
        "filter": {
            "$and": [
                {"price": {"$lt": 150}},
                {"tags": {"$array_contains": "office"}},
            ]
        },
        "select": ["name", "price"],
        "limit": 10,
    },
).execute().rows()
```

Inspect the planned execution before relying on backend-native behavior:

```python
for step in query.explain()["steps"]:
    if step.get("node") == "filter":
        print(step["field"], step["backend"], step["handler_mode"])
```

`handler_mode` distinguishes native execution from an exact Python scan fallback. Unsupported operations and fallback reasons are reported in the same plan.

## Core Capabilities

| Area | What HeavenBase provides |
| --- | --- |
| Entity model | Typed fields, validation, defaults, computed values, descriptions, indexes, uniqueness, JSON schema, and stable `object_id` identity |
| Workspace | Schema registration, CRUD, backend ownership, storage routing, system metadata, manifests, lifecycle, and repair tools |
| Query | Python expressions, JSON filters, dotted JSON paths, projection, ordering, pagination, aggregation over declared or nested JSON values, JSON map explosion, `HAVING`, vector-near search, and graph traversal |
| Storage | Field-level placement through `InlineColumn`, `JsonField`, `SideTable`, `SparseGramIndex`, `VectorIndex`, `InvertedIndex`, `GraphEdge`, and `ExternalRef` |
| Discovery | Queryable `Catalog` and `MetaSchema` rows for workspace state, direct Registry tag inspection for module declarations, and exact handler-support inspection |
| Diagnostics | `Query.explain()` reports the selected backend, strategy, handler, native or fallback mode, and unsupported reasons |
| Agent interfaces | Workspace MCP profiles, toolkits, prompts, memory and agent extensions, configured LLM calls, and a local dashboard |
| Extensibility | Registered logical types, storage strategies, backends, query handlers, extensions, MCP profiles, and serializers |

The working model is deliberately small:

```text
Application or agent
        │
        ├── Python API
        ├── JSON query
        └── MCP tools
                │
        HeavenBase workspace
        ├── Entity schemas
        ├── Storage and query routing
        ├── Catalog and MetaSchema
        └── Execution diagnostics
                │
      SQL · Vector · Search · Graph · File
```

## Backends

HeavenBase includes backend adapters across several storage families. Optional providers require their own Python packages, credentials, and reachable services.

| Family | Built-in adapters |
| --- | --- |
| Local and file | In-memory, JSON, pickle, SQLite, DuckDB, TinyDB |
| SQL and analytics | PostgreSQL, Supabase, MySQL, seekdb, OceanBase, Dolt, Microsoft SQL Server, Oracle, StarRocks, Trino, ClickHouse |
| Document, key-value, and text search | MongoDB, Redis, Elasticsearch, OpenSearch |
| Vector and hybrid search | pgvector, LanceDB, Chroma, Milvus, Milvus Lite, Pinecone, Qdrant, Weaviate, RediSearch |
| Graph and multi-model | Neo4j, SurrealDB |

The [provider reference](docs/resources/reference/backends.md) lists canonical
configuration selectors, aliases, protocol-compatible routes, and local-file
behavior.

Start with a preset and move to explicit backend configuration when deployment placement matters:

| Preset | Default layout | Intended use |
| --- | --- | --- |
| `debug` | SQLite rows, in-memory vectors and search | Quickstarts, tests, and local experiments without Docker |
| `local-lts` | PostgreSQL, LanceDB, Elasticsearch | Durable self-hosted development |
| `web-lts` | Supabase rows, pgvector, hosted search | Managed or shared environments |

Executable Backend support is operation-specific. Consult the [generated
capability matrix](docs/resources/reports/capabilities.md) or inspect the exact
registered handler keys at runtime:

```python
hb.capabilities.backends(hb.Vector, op="near")
ws.capabilities.ops(hb.ShortText, hb.InlineColumn, backend="main")
```

Module tags are descriptive declarations and candidate hints, not proof that a
route can execute. Inspect a Backend type without constructing it through the
owning Context, or inspect construction/live overlays on a concrete instance:

```python
modules = ws.context.modules()
assert modules.tag("backend_type", "inmem", "vector").value is True

main = ws.backends.get("main")
print(main.tag("backend-availability").value)
```

Static claims come only from the installed module record's `meta.tags`.
Definitions are optional, so an unknown claim remains visible as unresolved
metadata. Exact handlers, compiler output, and concrete resource checks still
decide execution; use `explain()` for that decision.

## Agent and LLM Interfaces

Every workspace can expose a compact MCP surface:

```python
ws.serve(
    name="shop-mcp",
    profile="agent",
    transport="http",
    host="127.0.0.1",
    port=7001,
)
```

The `agent` profile provides schema discovery and ordinary data operations such as `define_entity`, `list_entities`, `describe_entity`, `upsert`, `get`, `set`, `count`, `query`, and `explain`. Use `profile="full"` only for trusted administrative workflows.

The same selected profile can be installed as a credential-free agent skill
and served from the parser-neutral CLI. Omit the workspace id to select the
active/default workspace:

```python
skill = ws.to_skill(profile="agent", port=7001)
skill.install()
```

```bash
hb ws skill shop --profile agent --port 7001
hb mcp workspace shop --profile agent --transport http --port 7001
```

The generated skill declares the MCP dependency and teaches discovery and safe
editing, but contains no workspace construction or backend credentials.

HeavenBase also includes preset-driven LLM utilities for chat, embeddings, image generation, sessions, caching, and MCP tool loops:

```python
import heavenbase as hb

llm = hb.LLM(preset="mock")
print(llm.chat("Reply with hb-ok"))
```

The mock preset is suitable for local verification. Online presets read provider credentials from the environment.

## Command Line and Dashboard

The `hb` CLI is a shallow interface over the same configuration, workspace, LLM, prompt, and MCP APIs:

```bash
hb --help
hb setup
hb ws list
hb ws create shop --preset debug
hb ws activate shop
hb ws deactivate
hb config set heavenbase.query.near.default_top_k 25 --scope shop
hb ws presets show debug
hb dashboard
```

Workspace creation and import never activate implicitly. The equivalent Python
operations are `ws.activate()` and `ws.deactivate()`; advanced name-only callers use
`context.activate_workspace("shop")` and `context.deactivate_workspace()`.
`HeavenBase.load()` without an id resolves the active workspace and then falls
back to the configured default. Loading requires an existing workspace
registration and raises `KeyError` on a miss. Run `hb setup` to idempotently
ensure the configured default; creation never happens as a side effect of
loading.

`hb dashboard` opens a loopback-local workbench for configuration, read-only
workspace inspection, and a tool-using assistant. While an assistant turn runs,
the send control becomes a stop control; interrupting keeps the partial turn
without retrying it. The composer defaults to the `chat` preset, shows the
selected model and reasoning mode, and lets each conversation switch between
read-only and full-access tools. On macOS the dashboard prefers a standalone system WebView
and falls back to the normal browser when unavailable; use `--browser` for a
browser tab, `--app` for a Chromium app window, or `--no-open` to serve only the
printed local URL. On macOS,
`hb dashboard --install-app` creates `~/Applications/HeavenBase.app` with a Dock
icon and a Finder-owned loopback-server lifecycle. Re-running `--install-app`
prompts before replacing an existing bundle; pass `--yes` or `-y` for a
non-interactive replacement. Dashboard appearance and the last successfully
opened workspace are persisted in `~/.heavenbase`, so browser, Chromium
`--app`, and installed-app hosts resume the same selection. The workspace
picker opens registered workspaces and can unregister them after confirmation;
unregistering does not delete backing database data.

## Database Analyst Extension

Enable the optional `database` extension to import an external SQL schema as
seven editable `db-*` entity types. Imports may select tables and columns,
collect column statistics and low-cardinality values, and preserve descriptions,
synonyms, and corrected analytical types across refreshes:

```python
ws.enable_extension("database")
result = ws.database.ingest(
    "warehouse-sql",
    name="warehouse",
    tables=["sales.orders", "sales.customers"],
    columns={"sales.orders": ["order_id", "customer_id", "total"]},
)
database_id = result["database"][0]
frame = ws.database.run_sql(database_id, "SELECT COUNT(*) AS n FROM sales.orders")
```

The scoped `database` MCP profile lets an assistant inspect and edit those
entities, run read-only SQL against the live source, save successful SQL as a
verified `DbSnippet`, and retrieve relevant `DbRule` guidance. Persisted source
metadata and errors are redacted; secret-bearing direct connections must be
re-ingested after process restart or configured as named workspace SQL backends.
When this workspace is open in the dashboard, chat discovers that MCP profile
automatically. Read-only mode can search rules/snippets and execute guarded SQL;
full-access mode can additionally ingest a configured backend, edit `db-*`
context, save a verified snippet, and curate a rule. The setup tool accepts only
a named backend reference, never connection credentials. SQL answers render the
executed statement and bounded actual result table beside the final analysis;
the same evidence is available as structured `sql_frames` in blocking and
streamed dashboard responses. There is not yet a
dedicated extension page or automatic-learning loop. See the [database
reference](docs/resources/reference/database.md) for selection, profiling,
configuration, chat, and safety details.

## Project Status and Limits

HeavenBase favors explicit, inspectable behavior over optimistic capability claims:

- A registered query operation does not guarantee native execution on every backend. Use `explain()` to verify the concrete route.
- One workspace coordinates multi-backend writes, but independent backends do not form a distributed transaction.
- Workspace manifests replay construction settings, requested optional Extension roots, and schemas. Required and transitive Extensions are recomputed; row data is not exported.
- JSON and pickle file backends are local-development tools. Treat pickle data as trusted-local only.
- Some portable operations intentionally use exact scan or fold execution when a provider cannot prove equivalent native semantics.

See the [changelog](CHANGELOG.md) for release-specific changes and [current goals](docs/goals/current.md) for the checked-out source status.

## Documentation

- [Public documentation](https://ahvn.top/) — installation, concepts, guides, integrations, and workshops.
- [Public API tiers](docs/resources/reference/public-api.md) — the recommended user surface and advanced extension APIs.
- [External extension tutorial](docs/resources/reference/external-extensions.md) — package, install, enable, and test an out-of-tree Extension.
- [Workspace presets](docs/resources/reference/workspace-presets.md) — built-in layouts and explicit configuration.
- [MCP reference](docs/resources/reference/mcp.md) — profiles, transports, serializers, and workspace tools.
- [LLM reference](docs/resources/reference/llm.md) — presets, providers, gateways, sessions, and caching.
- [Built-in tags](docs/resources/reference/tags.md) — generated definitions, defaults, subjects, and built-in claims.
- [Capability matrix](docs/resources/reports/capabilities.md) — generated backend and operation support.
- [Canonical tags](docs/resources/architecture/canonical-tags.md) — static and effective module metadata, optional definitions, and execution-proof boundaries.
- [Engineering guide](docs/README.md) — architecture, repository contracts, and maintainer documentation.

## Contributing

Clone the repository and use the checked-in `uv` environment:

```bash
git clone https://github.com/Magolor/HeavenBase.git
cd HeavenBase
uv sync --extra dev
bash scripts/test.bash
bash scripts/flake.bash --ci
```

Tests must run through `scripts/test.bash`; it owns the project marker policy and environment setup. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

## License

HeavenBase is available under the [MIT License](LICENSE).
