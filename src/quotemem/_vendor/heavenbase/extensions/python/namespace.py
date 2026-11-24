"""Prepared globals for caller-authored analysis code."""

from __future__ import annotations

__all__ = ["python_namespace"]

import sys

from heavenbase.utils import Any, Dict


def python_namespace(workspace: Any) -> Dict[str, Any]:
    """Return the globals one Python program starts from.

    The workspace and its registered entity classes are preloaded, so a program
    reads and aggregates without first resolving names the caller would have to
    guess. Entities bind under their class name, which is what a schema
    description reports. When a caller's entity and a packaged or
    extension-provided entity share a class name (for example ``Session``), the
    caller's entity wins.

    Args:
        workspace (Any): Open workspace the program queries.

    Returns:
        Dict[str, Any]: Globals containing ``hb``, ``ws``, and one binding per
        registered entity class.
    """
    import heavenbase as hb

    namespace: Dict[str, Any] = {"hb": hb, "ws": workspace, "__name__": "__python__"}
    extension_owned = {str(entity_id) for extension in workspace.extensions().values() for entity_id in extension.get("entities", ())}
    entities = [(entity_id, workspace.entities[entity_id]) for entity_id in workspace.entities]
    builtin = [entity_id in extension_owned or _packaged_class(entity) for entity_id, entity in entities]
    ordered = [entity for (_, entity), packaged in zip(entities, builtin) if not packaged]
    ordered += [entity for (_, entity), packaged in zip(entities, builtin) if packaged]
    for entity in ordered:
        name = getattr(entity, "__name__", None)
        if name and name not in namespace:
            namespace[name] = entity
    return namespace


def _packaged_class(entity: Any) -> bool:
    """Return whether an entity class was defined by HeavenBase code rather than by the caller.

    HeavenBase's own entity classes live in an imported ``heavenbase.*`` module
    or in a loaded module artifact. A class compiled from a caller's schema
    mapping (``ws.register({...})``, an interop mapping) is also labelled
    ``heavenbase.*``, but no such module exists, so it counts as the caller's.
    """
    module = str(getattr(entity, "__module__", ""))
    if module.startswith("_heavenbase_artifact_"):
        return True
    return module.startswith("heavenbase.") and module in sys.modules
