"""Workspace-owned Toolkit reference resolution."""

from __future__ import annotations

__all__ = ["describe_toolkit_ref", "load_toolkit_ref", "parse_toolkit_ref"]

from heavenbase.utils import Any, Dict

from .toolkit import Toolkit


def parse_toolkit_ref(ref: str, namespace: str | None = None, version: str | None = None) -> Dict[str, str | None]:
    """Parse `namespace.name:version` or `namespace/name:version` refs.

    Args:
        ref (str): Toolkit id or qualified name.
        namespace (str | None): Namespace override for an unqualified name.
        version (str | None): Version override for an unqualified name.

    Returns:
        Dict[str, str | None]: Parsed id, name, namespace, and version parts.
    """
    text = str(ref or "").strip()
    ns = namespace
    ver = version
    if ":" in text:
        text, ver = text.rsplit(":", 1)
    if "/" in text:
        ns, text = text.split("/", 1)
    elif "." in text:
        ns, text = text.split(".", 1)
    return {"id": ref, "name": text, "namespace": ns, "version": ver}


def _is_negative_version(version: str | None) -> bool:
    try:
        return int(str(version).strip()) < 0 if version is not None else False
    except (TypeError, ValueError):
        return False


def _toolkit_ref_rows(parts: Dict[str, str | None], *, ws: Any = None) -> list[Dict[str, Any]]:
    rows = [row for row in Toolkit.list(ws=ws) if row.get("name") == parts["name"]]
    if parts["namespace"]:
        rows = [row for row in rows if row.get("namespace") == parts["namespace"]]
    if parts["version"] is not None and not _is_negative_version(parts["version"]):
        rows = [row for row in rows if str(row.get("version")) == str(parts["version"])]
    return rows


def _ambiguous_toolkit_message(name: str, rows: list[Dict[str, Any]]) -> str:
    refs = sorted({f"{row.get('namespace')}.{row.get('name')}:{row.get('version')}" for row in rows})
    return f"Ambiguous toolkit name {name!r}; use one of: {', '.join(refs)}"


def load_toolkit_ref(
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    *,
    ws: Any = None,
) -> Toolkit:
    """Load a Toolkit reference from one workspace.

    Args:
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified names.
        version (str | None): Optional version or negative version index.
        ws (Any): Workspace that owns the Toolkit row. When omitted,
            `HeavenBase.load()` resolves the selected workspace.

    Returns:
        Toolkit: Restored Toolkit runtime.
    """
    try:
        return Toolkit.load(ref, ws=ws)
    except KeyError as original:
        parts = parse_toolkit_ref(ref, namespace=namespace, version=version)
        try:
            return Toolkit.load(
                ws=ws,
                name=parts["name"],
                namespace=parts["namespace"],
                version=parts["version"],
            )
        except KeyError:
            if parts["namespace"]:
                raise original
            rows = _toolkit_ref_rows(parts, ws=ws)
            namespaces = sorted({str(row.get("namespace") or "default") for row in rows})
            if len(namespaces) == 1:
                return Toolkit.load(
                    ws=ws,
                    name=parts["name"],
                    namespace=namespaces[0],
                    version=parts["version"],
                )
            if len(namespaces) > 1:
                raise KeyError(_ambiguous_toolkit_message(str(parts["name"]), rows)) from None
            raise original


def describe_toolkit_ref(
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    *,
    ws: Any = None,
) -> Dict[str, Any]:
    """Describe a Toolkit reference from one workspace.

    Args:
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified names.
        version (str | None): Optional version or negative version index.
        ws (Any): Workspace that owns the Toolkit row. When omitted,
            `HeavenBase.load()` resolves the selected workspace.

    Returns:
        Dict[str, Any]: Matching Toolkit row.
    """
    try:
        return Toolkit.describe(ref, ws=ws)
    except KeyError as original:
        parts = parse_toolkit_ref(ref, namespace=namespace, version=version)
        try:
            return Toolkit.describe(
                ws=ws,
                name=parts["name"],
                namespace=parts["namespace"],
                version=parts["version"],
            )
        except KeyError:
            if parts["namespace"]:
                raise original
            rows = _toolkit_ref_rows(parts, ws=ws)
            namespaces = sorted({str(row.get("namespace") or "default") for row in rows})
            if len(namespaces) == 1:
                return Toolkit.describe(
                    ws=ws,
                    name=parts["name"],
                    namespace=namespaces[0],
                    version=parts["version"],
                )
            if len(namespaces) > 1:
                raise KeyError(_ambiguous_toolkit_message(str(parts["name"]), rows)) from None
            raise original
