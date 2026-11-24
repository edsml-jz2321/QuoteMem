"""Python query surface an agent needs to write code against a workspace."""

from __future__ import annotations

__all__ = ["render_python_guide"]

from heavenbase.utils import Any, Dict

# Every fenced block below is executed by
# ``tests/extensions/test_python_extension.py`` against a live workspace, so
# this text cannot drift away from the API it documents.
_GUIDE = """# Writing Python for `run_python`

`hb`, `ws`, and entity classes are preloaded when the tools are built. When
two entities share a class name, the caller's own entity holds it. Look up
entities registered later through `ws.entities`. Statements run in order and a
trailing expression is the result, like a notebook cell. Names persist between
calls.

## Read rows

```python
rows = ws.rows(Paper)          # every row, as dicts — unbounded
ws.count(Paper)                # row count
```

## Query with the fluent surface

```python
frame = (
    ws.query(Paper)
    .where(Paper.year >= 2022)
    .group_by(Paper.venue)
    .aggregate(n=Paper.object_id.count(), best=Paper.score.max())
    .order_by("n", desc=True)
    .limit(10)
    .execute()
)
frame.rows()
```

Reducers on a field reference: `count(distinct=True)`, `sum`, `avg`, `min`,
`max`, `median`, `percentile(q=0.95)`.

Filters: `==`, `!=`, `<`, `<=`, `>`, `>=`, `.in_([...])`, `.contains("x")`.
Combine with `&` and `|`; wrap each side in parentheses.

## Json columns

Filter nested values with a path reference:

```python
ws.query(Doc).where(Doc.meta["review"]["score"] >= 8).execute().rows()
```

To aggregate over keys the query does not name, fold the whole column:

```python
ws.query(Trace).aggregate(totals=Trace.tool_calls.key_sums()).execute().scalar()
```

`key_sums` totals each key's values and `key_counts` counts the rows carrying
each key, both over every row and exactly. The result is one mapping per group,
so `group_by` on declared columns works normally, but `order_by` and `having`
cannot reach inside it — sort in Python when you need a ranking.

When neither shape fits, just compute it here — you are already in Python:

```python
from collections import Counter

tally = Counter()
for row in ws.rows(Trace):
    for tool, calls in (row.get("tool_calls") or {}).items():
        tally[tool] += calls
dict(tally.most_common(5))
```

Registering a permanent reducer with `hb.ext.register_aggregate_reducer` is a
maintainer action taken in real module code: the registry stores an importable
target, and a function defined here has no importable path.

## The JSON spec also works

```python
ws.query_json(Paper, {"group_by": ["venue"], "aggregate": {"n": {"count": {"field": "object_id"}}}}).execute().rows()
```

## Reading a result

```python
frame = ws.query(Paper).limit(3).execute()
frame.rows()          # list of dicts
frame.col_names()     # column names
len(frame)            # rows returned
frame.total           # rows that matched, when pagination cut the result
frame.truncated       # whether it cut
```

Return the frame itself as the last expression and the payload keeps `total`
and `truncated`. When `truncated` is true, report `total`, not `len(frame)`.

## Rules

- Prefer the `query` tool when a spec can say it: the engine validates a spec
  and names what is legal when it cannot. A wrong program returns a plausible
  number with no such signal.
- Do not compute a total by paging rows and adding them up. Aggregate in the
  query, or use `ws.rows()` which is unbounded.
- Say when a number came from your own arithmetic rather than from the engine.
"""


# Globals every program starts with besides the entity bindings.
_PRELUDE = {"hb", "ws", "__name__"}


def render_python_guide(workspace: Any = None) -> str:
    """Return the Python guide, with the workspace's own entities appended.

    The appended list names exactly the entity bindings a program starts
    with (see ``python_namespace``), so a class name two entities share is
    listed once, and the entity it does not bind is named with its
    ``ws.entities`` lookup instead.

    Args:
        workspace (Any): Open workspace whose entity names ground the examples.
            Defaults to `None`, which renders the generic guide.

    Returns:
        str: Guide text ready to hand to a model.
    """
    if workspace is None:
        return _GUIDE
    from .namespace import python_namespace

    return _render_workspace_guide(workspace, python_namespace(workspace))


def _render_workspace_guide(workspace: Any, namespace: Dict[str, Any]) -> str:
    """Describe the initial bindings and the current workspace lookups."""
    bound = {id(value) for value in namespace.values()}
    names = sorted(name for name in namespace if name not in _PRELUDE)
    text = f"{_GUIDE}\n## Bound in this workspace\n\n{', '.join(names)}\n"
    lookups = []
    for entity_id in sorted(workspace.entities):
        entity = workspace.entities[entity_id]
        if id(entity) not in bound:
            lookups.append(f'`ws.entities["{entity_id}"]`')
    if lookups:
        text += f"\nOther entities available by lookup\n\n{', '.join(lookups)}\n"
    return text
