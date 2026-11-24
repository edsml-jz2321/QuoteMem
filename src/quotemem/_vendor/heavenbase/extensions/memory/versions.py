"""Project-bound memory version chains."""

from __future__ import annotations

__all__ = ["MemoryVersions"]

import re
from time import time

from heavenbase.utils import Any, Mapping, hash_id

_SELECT = [
    "text",
    "title",
    "tags",
    "metadata",
    "source",
    "project_id",
    "keypath",
    "version",
    "previous_id",
    "deleted",
    "created_at",
    "updated_at",
]


def _segment(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return text.replace("-", ".")


def _title(keypath: str) -> str:
    leaf = str(keypath or "").split(".")[-1]
    return leaf.replace("-", " ").replace("_", " ").title() or "Memory"


class MemoryVersions:
    """Manage append-only memory versions in one project namespace.

    Args:
        workspace (Any): Workspace storing the memory entity.
        entity (str): Memory entity identifier.
        project_id (str): Project namespace. Empty values resolve to `default`.
    """

    def __init__(self, workspace: Any, entity: str, project_id: str) -> None:
        """Bind a version store to one workspace project.

        Args:
            workspace (Any): Workspace storing the memory entity.
            entity (str): Memory entity identifier.
            project_id (str): Project namespace. Empty values resolve to `default`.

        Returns:
            None: This initializer stores the workspace, entity, and namespace.
        """
        self.workspace = workspace
        self.entity = str(entity)
        self.project_id = _segment(project_id or "default")

    @property
    def prefix(self) -> str:
        """str: Full keypath prefix for this project namespace."""
        return f"project.{self.project_id}"

    def keypath(self, value: str = "") -> str:
        """Resolve a project-relative keypath.

        Args:
            value (str): Relative or already-prefixed keypath.

        Returns:
            str: Normalized keypath below this project's prefix.
        """
        path = ".".join(_segment(part) for part in str(value or "").split(".") if _segment(part))
        if not path:
            return self.prefix
        return path if path == self.prefix or path.startswith(f"{self.prefix}.") else f"{self.prefix}.{path}"

    def derive_keypath(self, text: str, title: str = "") -> str:
        """Derive a relative keypath from memory content.

        Args:
            text (str): Markdown memory body.
            title (str): Optional title used before the first Markdown heading.

        Returns:
            str: Normalized relative keypath suitable for `keypath`.
        """
        if title:
            return _segment(title)
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return _segment(stripped.lstrip("#").strip())
        return f"note.{hash_id(text, length=10)}"

    def title(self, keypath: str) -> str:
        """Build a short title from a keypath.

        Args:
            keypath (str): Relative or full memory keypath.

        Returns:
            str: Human-readable title derived from the final segment.
        """
        return _title(keypath)

    def rows(self) -> list[dict[str, Any]]:
        """Return every stored version in this project.

        The project scope is a query filter and no limit is applied, so the
        result is the whole project rather than the head of an entity-wide scan
        that could drop versions this project owns.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Every project row in workspace query order.
        """
        return self.workspace.query_json(self.entity, {"select": _SELECT, "filter": {"project_id": self.project_id}}).execute().rows()

    def current(self, keypath: str = "", *, include_deleted: bool = False) -> list[dict[str, Any]]:
        """Return the latest version for each keypath below a prefix.

        Args:
            keypath (str): Relative or full keypath prefix. Empty selects the project.
            include_deleted (bool): Whether to retain latest soft-delete tombstones.

        Returns:
            list[dict[str, Any]]: Latest rows sorted by keypath.
        """
        prefix = self.keypath(keypath)
        latest: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            row_keypath = str(row.get("keypath") or "")
            if row_keypath != prefix and not row_keypath.startswith(f"{prefix}."):
                continue
            previous = latest.get(row_keypath)
            if previous is None or int(row.get("version") or 0) >= int(previous.get("version") or 0):
                latest[row_keypath] = row
        rows = sorted(latest.values(), key=lambda row: str(row.get("keypath") or ""))
        return rows if include_deleted else [row for row in rows if not row.get("deleted")]

    def history(self, keypath: str) -> list[dict[str, Any]]:
        """Return the complete version chain for one keypath.

        The chain is read by its own unlimited query on `project_id` and
        `keypath` rather than folded out of a wider scan, so `append` derives
        the next version from the real tail of the chain.

        Args:
            keypath (str): Relative or full memory keypath.

        Returns:
            list[dict[str, Any]]: Every version row ordered from oldest to newest.
        """
        target = self.keypath(keypath)
        spec = {"select": _SELECT, "filter": {"project_id": self.project_id, "keypath": target}}
        rows = self.workspace.query_json(self.entity, spec).execute().rows()
        return sorted(rows, key=lambda row: int(row.get("version") or 0))

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search current rows using all whitespace-separated terms.

        Args:
            query (str): Case-insensitive terms matched across keypath, title,
                text, and tags.

        Returns:
            list[dict[str, Any]]: Current non-deleted rows matching every term.
        """
        terms = str(query or "").strip().lower().split()
        if not terms:
            return self.current()
        return [row for row in self.current() if all(term in self._search_text(row) for term in terms)]

    def append(
        self,
        keypath: str,
        text: str,
        *,
        title: str = "",
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        source: str | None = None,
        deleted: bool = False,
    ) -> dict[str, Any]:
        """Append one version to a memory keypath.

        Args:
            keypath (str): Relative or full memory keypath.
            text (str): Stored memory body.
            title (str): Optional short title.
            tags (list[str] | None): Optional memory tags.
            metadata (Mapping[str, Any] | None): Optional structured metadata.
            source (str | None): Optional producer or agent name.
            deleted (bool): Whether this version is a soft-delete tombstone.

        Returns:
            dict[str, Any]: Version metadata, stored row, and conflicts with the
            previous non-deleted content.

        Raises:
            ValueError: If the derived version already exists, which means the
                chain read here did not see the whole history.
        """
        target = self.keypath(keypath)
        history = self.history(target)
        previous = history[-1] if history else None
        conflicts = []
        if previous and not previous.get("deleted") and str(previous.get("text") or "") != str(text or ""):
            conflicts.append({"object_id": previous["object_id"], "version": previous.get("version"), "text": previous.get("text")})
        version = int(previous.get("version") or 0) + 1 if previous else 1
        now = int(time())
        object_id = f"mem-{hash_id(self.project_id, target, str(version), length=32)}"
        row = {
            "object_id": object_id,
            "text": text,
            "title": title,
            "tags": list(tags or []),
            "metadata": dict(metadata or {}),
            "source": source or str(self.workspace.context.config.get("heavenbase.mcp.memory.default_source", default="agent") or "agent"),
            "project_id": self.project_id,
            "keypath": target,
            "version": version,
            "previous_id": previous["object_id"] if previous else "",
            "deleted": bool(deleted),
            "created_at": now if previous is None else int(previous.get("created_at") or now),
            "updated_at": now,
        }
        self.workspace.upsert_many(self.entity, [row], overwrite=False)  # a colliding version means the chain read above was short
        return {"object_id": object_id, "keypath": target, "version": version, "row": self.workspace.get(object_id, entity=self.entity), "conflicts": conflicts}

    @staticmethod
    def _search_text(row: Mapping[str, Any]) -> str:
        return " ".join(
            [
                str(row.get("keypath") or ""),
                str(row.get("title") or ""),
                str(row.get("text") or ""),
                " ".join(str(tag) for tag in row.get("tags") or []),
            ]
        ).lower()
