# Elasticsearch module

This folder is the complete built-in Elasticsearch lego piece. It uses the
same folder protocol as an independently installed HeavenBase module:

```text
elasticsearch/
  meta.yaml       # declarative Registry contributions
  backend/        # physical storage and execution role
  handler/        # provider-native query compilation role
  __init__.py     # lazy provider facade
```

`meta.yaml` contributes both `backend_type:elasticsearch` and
`handler:elasticsearch`. Discovery reads that metadata without importing the
Backend or Handler code; setup publishes both records atomically, and runtime
resolution later imports the selected role through its recorded target.

The module depends inward on shared Backend, Handler, schema, and query
contracts. It must not own planner branches, workspace policy, or a second
provider registry. Provider behavior is covered by the Elasticsearch sections
of `tests/backends/test_backends.py`, `tests/backends/test_native_backends.py`,
`tests/core/test_logical_types.py`, and `tests/query/test_scalar_index.py`.

To extract this module into another distribution, keep the folder shape and
record identities, update only the Python package coordinates in `meta.yaml`,
and install the folder through `Context.modules().install(...)` before opening
a workspace. The consumer-facing backend and handler contracts stay the same.
