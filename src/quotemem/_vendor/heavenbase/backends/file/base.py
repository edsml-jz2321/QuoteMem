"""Folder-backed backend base."""

from __future__ import annotations

__all__ = [
    "FileBackend",
]

import hashlib
import os
import threading
from pathlib import Path
from urllib.parse import quote, unquote

from heavenbase.utils import Any, Callable, Dict, Iterable, List, Optional, Sequence, copy_data
from heavenbase.utils.files import delete_dir, delete_file, file_uri_to_path, list_files, replace_file, temp_file, touch_dir
from heavenbase.utils.files import pj
from heavenbase.utils.serialize import dump_bin, dump_json, dumps_pickle, load_bin, load_json, loads_pickle

from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...utils.ids import check_entity_id
from ..base import backend_locked
from heavenbase.execution.contracts import RowOp
from ..inmem import InMemBackend

_FOLDER_LOCKS: dict[str, Any] = {}
_FOLDER_LOCKS_GUARD = threading.Lock()


def _folder_lock(path: str) -> Any:
    """Return the re-entrant lock this process shares among File Backends over one folder."""
    key = os.path.normcase(os.path.realpath(path))
    with _FOLDER_LOCKS_GUARD:
        lock = _FOLDER_LOCKS.get(key)
        if lock is None:
            lock = _FOLDER_LOCKS[key] = threading.RLock()
        return lock


def _reset_folder_locks_after_fork() -> None:
    """Free the folder locks in a forked child, where the threads of the parent that held them do not exist."""
    global _FOLDER_LOCKS_GUARD
    _FOLDER_LOCKS_GUARD = threading.Lock()
    _FOLDER_LOCKS.clear()


if hasattr(os, "register_at_fork"):  # POSIX only; Windows has no fork
    os.register_at_fork(after_in_child=_reset_folder_locks_after_fork)


_NAME_MAX = 255
"""Longest file name, in bytes, on the file systems HeavenBase supports (ext4, XFS, APFS, NTFS)."""

_TEMP_NAME_OVERHEAD = len(".") + len(".") + 8 + len(".tmp")
"""Characters ``temp_file`` adds around a record name: ``.{name}.XXXXXXXX.tmp``."""

_WINDOWS_PATH_MAX = 259
"""Longest full path Windows opens without long-path support: ``MAX_PATH`` less the terminating NUL."""

_PATH_MAX: int | None = _WINDOWS_PATH_MAX if os.name == "nt" else None
"""Longest full path Windows opens without long-path support (``MAX_PATH`` less the terminating NUL); no limit elsewhere."""

_PLAIN_PATH_MAX = _WINDOWS_PATH_MAX - _TEMP_NAME_OVERHEAD
"""Longest path Windows file calls take as it is: the temporary file beside it fits ``MAX_PATH``, and a folder this
long is within the 247 characters Windows creates."""

_EXTENDED_PREFIX = "\\\\?\\"
"""Prefix of a Windows extended-length path, which file calls take whole, past ``MAX_PATH``."""

_HASHED_SEP = "="
"""Separator of hashed record names; ``quote(..., safe="")`` never emits it, so no encoded name contains it."""

_DIGEST_CHARS = 32
_READABLE_CHARS = 32

_WINDOWS_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{digit}" for digit in range(10)), *(f"lpt{digit}" for digit in range(10))})


def _record_name_max(path: str) -> int:
    """Return the longest record-file name the file system at ``path`` holds, capped at ``_NAME_MAX``."""
    try:
        value = os.pathconf(path, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        return _NAME_MAX
    return min(value, _NAME_MAX) if value > 0 else _NAME_MAX


def _long_path(path: str, *, always: bool = False) -> str:
    """Return ``path`` in the form a Windows file call opens past ``MAX_PATH``; elsewhere ``path`` itself.

    Windows without long-path support (the default) refuses a full path longer than ``MAX_PATH`` unless it is an
    extended-length ``\\\\?\\`` path. Record files keep names that fit ``MAX_PATH`` where they can, but a record
    under a long encoded name (written on another file system) or a hashed name under a deep folder does not
    fit, so its path is extended. Paths short enough for any folder or file, and paths elsewhere, are returned
    unchanged; ``always`` extends every Windows path, for a call such as a tree removal that reaches paths
    below ``path``.
    """
    if os.name != "nt" or path.startswith(_EXTENDED_PREFIX) or not (always or len(path) > _PLAIN_PATH_MAX):
        return path
    path = os.path.abspath(path)
    if path.startswith("\\\\"):
        return f"{_EXTENDED_PREFIX}UNC\\{path[2:]}"
    return f"{_EXTENDED_PREFIX}{path}"


def _exists(path: str) -> bool:
    """Return whether the absolute ``path`` exists, following links; ``exists_path`` would resolve the path first."""
    return Path(path).exists()


def _touch_folder(path: str) -> str:
    """Create the folder ``path``, through its extended-length form when it is too long for Windows, and return it.

    The folder is returned resolved and in its ordinary form, as ``touch_dir`` returns it, so ``FileBackend.path``
    shows no ``\\\\?\\`` prefix.
    """
    folder = pj(path, abs=True)
    touch_dir(_long_path(folder))
    return folder


def _is_device_name(name: str) -> bool:
    """Return whether Windows reserves ``name`` for a device (``con.json``, ``nul.x.pkl``, ...)."""
    return name.split(".", 1)[0].lower() in _WINDOWS_DEVICE_NAMES


def _name_may_alias(row_id: str) -> bool:
    """Return whether the encoded name of ``row_id`` may open the record file of another id.

    ``quote`` keeps ASCII letters as they are and percent-encodes every other non-ASCII character with
    upper-case hex digits, so two ids share an encoded name up to case only when they differ in the case
    of an ASCII letter. A name with ``~`` may also be a Windows 8.3 short name (``123456~1.pkl``) that
    NTFS resolves to another record file.
    """
    return any((char.isascii() and char.isalpha()) or char == "~" for char in row_id)


class _PickleRecord(dict[str, Any]):
    """Cache a decoded row and the original bytes used to reconstruct it."""

    def __init__(self, row: Dict[str, Any], payload: bytes) -> None:
        super().__init__(row)
        self.payload = payload


class FileBackend(InMemBackend):
    """Folder-backed row store that persists each row as one record file.

    Notes:
        A record file is named by the percent-encoded object ID, ``quote(object_id, safe="")``, plus
        ``.json`` or ``.pkl``, as in 0.1.2.2. A record uses a hashed name instead when that name cannot
        serve it: it would not fit the file system's name limit together with the temporary name written
        beside it (on Windows, the full temporary path must also fit 259 characters), it is a Windows
        device name such as ``con.json``, or it already holds an ID that differs only by letter case (on a
        case-insensitive file system, the default on macOS and Windows). The hashed name is the first 32
        hex digits of the SHA-256 of the UTF-8 object ID, ``=``, then at most the first 32 characters of
        the encoded ID. A lookup reads the hashed name first, then the encoded name, and accepts an
        encoded-name file only when the ``object_id`` stored in it is the requested one. A record file it
        cannot read counts as present for ``exists``, as in 0.1.2.2. HeavenBase 0.1.2.2 finds a row stored
        under a hashed name only by scanning. On Windows, a record file whose path is too long for ``MAX_PATH``
        (a long encoded name another file system wrote, or a hashed name under a deep folder) is read, written
        and deleted through its extended-length ``\\\\?\\`` path, so it works without long-path support.
    """

    identifier = "file"
    aliases = ()
    default_path_key = "heavenbase.backend_defaults.file.path"
    default_path = "file://%/file"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "FileBackend":
        """Build the generic file backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``path``, ``fmt``, and optional ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and runtime policy.

        Returns:
            FileBackend: Configured folder-backed backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("path", None), cfg.pop("fmt", "json"), ws_id=ws_id, resolver=resolver, config=config)

    def __init__(
        self,
        name: str = "file",
        path: str | None = None,
        fmt: str = "json",
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> None:
        """Create a folder-backed backend.

        Args:
            name: Workspace backend instance name.
            path: Base folder for persisted row files. When omitted, uses the configured
                file backend default path.
            fmt: Record format. Supported values are ``"json"``, ``"pkl"``, and ``"pickle"``.
            ws_id: Optional workspace namespace. When set, files live under a workspace subfolder.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and runtime policy.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        configured_path = self._construction_config_get(type(self).default_path_key, default=type(self).default_path)
        self._base_path = file_uri_to_path(path or configured_path, home=self.config.root)
        self.path = _touch_folder(pj(self._base_path, self.ws_id) if self.ws_id else self._base_path)
        self._name_max = _record_name_max(self.path)
        self.fmt = fmt
        self._loaded_entities: Dict[str, set] = {}
        # Per workspace and entity: record file name -> (stat stamp, row read from that file or None).
        self._record_stamps: Dict[Any, Dict[str, Dict[str, tuple[tuple[int, int, int], Dict[str, Any] | None]]]] = {}

    def _construction_tag_claims(self) -> dict[str, Any]:
        """Return format-derived tags for the configurable generic File Backend."""
        claims = super()._construction_tag_claims()
        if self.type == "file":
            claims["binary-storage"] = "base64" if self.fmt == "json" else "bytes"
        return claims

    def _refresh_workspace(self) -> None:
        self._refresh_path()

    def _refresh_path(self) -> None:
        """Recompute self.path when ws_id is assigned after construction."""
        with self._lock:
            self.path = _touch_folder(pj(self._base_path, self.ws_id) if self.ws_id else self._base_path)
            self._name_max = _record_name_max(self.path)
            self.data.clear()
            self._page_keys.clear()
            self._loaded_entities.clear()
            self._record_stamps.clear()

    @backend_locked
    def drop(self) -> None:
        """Delete the current workspace directory for this file backend.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        delete_dir(_long_path(self.path, always=True))
        self.data.pop(self.ws_id, None)
        self._page_keys.clear()
        self._loaded_entities.pop(self.ws_id, None)
        self._record_stamps.pop(self.ws_id, None)

    @backend_locked
    def refresh(self, entity_id: str | None = None) -> None:
        """Clear cached file rows for this backend.

        Args:
            entity_id: Optional logical entity identifier to clear. When omitted, all cached rows for the current workspace are cleared.

        Returns:
            None: This function does not return a value.

        Warnings:
            This only clears in-memory snapshots; it does not delete persisted files. Normal row lookups read the current
            record file, and a reload of the entity folder reads again each record file whose size, modification time or
            inode changed. ``refresh`` makes the next reload read every record file, which is useful after a manual edit
            that keeps all three, or before inspecting backend ``data`` directly.

        Examples:
            ``backend.refresh("note")`` clears cached rows for one entity. ``backend.refresh()`` clears the whole workspace cache.
        """
        ns = self.ws_id
        if entity_id is None:
            self.data.pop(ns, None)
            self._page_keys.clear()
            self._loaded_entities.pop(ns, None)
            self._record_stamps.pop(ns, None)
            return
        entity_id = check_entity_id(entity_id)
        self.data.get(ns, {}).pop(entity_id, None)
        self._ids_changed(entity_id)
        self._loaded_entities.get(ns, set()).discard(entity_id)
        self._record_stamps.get(ns, {}).pop(entity_id, None)

    @property
    def suffix(self) -> str:
        """Return the file suffix used by this backend format.

        Returns:
            str: ``".json"`` for JSON records or ``".pkl"`` for pickle records.

        Raises:
            ValueError: If ``fmt`` is not a supported record format.
        """
        if self.fmt == "json":
            return ".json"
        if self.fmt in ("pkl", "pickle"):
            return ".pkl"
        raise ValueError(f"Unsupported file backend format {self.fmt!r}")

    def _entity_dir(self, entity_id: str) -> str:
        """Return the entity's folder, its links resolved and the variables in their targets expanded.

        Resolving is slow, so it is done once per save, delete, ``get_many``, ``exists_many`` and ``count``. Record
        names are joined to the folder as they are; the file helpers resolve and expand the path they are given, and the
        folder is expanded here so that a record is looked up where they read and write it.
        """
        return os.path.expandvars(pj(self.path, self._physical_name(entity_id)))

    def _encoded_name(self, row_id: str) -> str:
        """Return the 0.1.2.2 record name of ``row_id``: the percent-encoded id plus the format suffix."""
        return f"{quote(row_id, safe='')}{self.suffix}"

    def _writable(self, entity_dir: str, name: str) -> bool:
        """Return whether a record can be written under ``name`` in ``entity_dir``.

        The name must not be a Windows device name and must fit the file system's name limit together with
        the temporary name ``temp_file`` writes beside it; on Windows that temporary path must also fit
        ``MAX_PATH``.
        """
        if _is_device_name(name) or len(name) + _TEMP_NAME_OVERHEAD > self._name_max:
            return False
        return _PATH_MAX is None or len(os.path.abspath(os.path.join(entity_dir, name))) + _TEMP_NAME_OVERHEAD <= _PATH_MAX

    def _hashed_name(self, row_id: str) -> str:
        """Return the bounded, case-insensitively unique record name of ``row_id``.

        The name is the first 32 hex digits of the SHA-256 of the UTF-8 id, ``=``, then at most the first
        32 characters of the percent-encoded id for readability. It does not depend on the file system.
        """
        digest = hashlib.sha256(row_id.encode("utf-8", "surrogatepass")).hexdigest()[:_DIGEST_CHARS]
        readable = quote(row_id, safe="")[:_READABLE_CHARS]
        cut = readable.rfind("%", max(len(readable) - 2, 0))
        if cut >= 0:  # never end on a split escape
            readable = readable[:cut]
        return f"{digest}{_HASHED_SEP}{readable}{self.suffix}"

    def _find_record(
        self, entity_dir: str, row_id: str, *, need_row: bool, loader: Callable[[str], Dict[str, Any] | None] | None = None
    ) -> tuple[str | None, Dict[str, Any] | None]:
        """Return the path of the record file in ``entity_dir`` that holds ``row_id``, and its row when ``need_row`` or verified.

        The hashed name is checked first. The encoded name counts only when the ``object_id`` stored in it
        is ``row_id``: on a case-insensitive file system it may open the record of an id that differs only
        by letter case, and on NTFS a name with ``~`` may be the 8.3 short name of another record. The
        encoded name is looked up even when it is too long to write here, since a record may have been
        written under it on another file system. Without ``need_row``, an encoded-name record that cannot
        be read counts as present, as in 0.1.2.2.
        """
        load = self._load_record if loader is None else loader
        hashed = _long_path(os.path.join(entity_dir, self._hashed_name(row_id)))
        if _exists(hashed):
            return hashed, load(hashed) if need_row else None
        name = self._encoded_name(row_id)
        if len(name) > self._name_max or (os.name == "nt" and _is_device_name(name)):
            return None, None
        path = _long_path(os.path.join(entity_dir, name))
        if not _exists(path):
            return None, None
        if not need_row and not _name_may_alias(row_id):
            return path, None
        if need_row:
            row = load(path)
        else:
            try:
                row = load(path)
            except Exception:  # 0.1.2.2 never read it here: an unreadable record (bad JSON, missing pickle class) is present
                return path, None
        if row is None or str(row.get("object_id", row_id)) != row_id:
            return None, None
        return path, row

    def _load_entity(self, entity_id: str) -> Dict[Any, Dict[str, Any]]:
        """Return the rows of one entity's record files, reading only files changed since the last call.

        A record file is read again when its size, modification time or inode differ
        from the stamp taken when it was last read; otherwise the row read then is
        reused. Writes replace a record file through a new temporary file, so each one
        changes the stamp. The stamp is taken before the file is read, so a change
        made during the read is seen by the next call.
        """
        known = self._record_stamps.setdefault(self.ws_id, {}).get(entity_id, {})
        stamps: Dict[str, tuple[tuple[int, int, int], Dict[str, Any] | None]] = {}
        rows: Dict[Any, Dict[str, Any]] = {}
        suffix = self.suffix
        entity_dir = self._entity_dir(entity_id)
        try:
            with os.scandir(_long_path(entity_dir)) as listing:
                # The files ``list_files(entity_dir, ext=...)`` yields, in its order: a bare ``.json`` has no suffix.
                entries = sorted(
                    (entry for entry in listing if entry.name.endswith(suffix) and entry.name != suffix and not entry.is_dir()),
                    key=lambda entry: entry.name.lower(),
                )
        except (FileNotFoundError, NotADirectoryError):
            entries = []
        for entry in entries:
            try:
                stat = entry.stat()
                stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                seen = known.get(entry.name)
                row = seen[1] if seen is not None and seen[0] == stamp else self._load_record(_long_path(entry.path))
            except FileNotFoundError:
                continue  # deleted since the listing
            stamps[entry.name] = (stamp, row)
            row_id = None if row is None else row.get("object_id")
            if row_id is not None and (row_id not in rows or _HASHED_SEP in entry.name):
                rows[row_id] = row
        self._record_stamps[self.ws_id][entity_id] = stamps
        return rows

    def _ensure_loaded(self, entity_id: str) -> None:
        entity_id = check_entity_id(entity_id)
        loaded = self._loaded_entities.setdefault(self.ws_id, set())
        if entity_id in loaded:
            return
        self._load_all(entity_id)

    def _load_all(self, entity_id: str) -> None:
        """Replace the cached rows of one entity with its current record files."""
        ns = self.data.setdefault(self.ws_id, {})
        previous = ns.get(entity_id)
        loaded = ns[entity_id] = self._load_entity(entity_id)
        if previous is None or previous.keys() != loaded.keys():
            self._ids_changed(entity_id)
        self._loaded_entities.setdefault(self.ws_id, set()).add(entity_id)

    def _load_record(self, path: str) -> Dict[str, Any] | None:
        if self.fmt == "json":
            return load_json(path) or None
        return self._load_pickle_record(path, keep_payload=True)

    def _load_pickle_record(self, path: str, *, keep_payload: bool = False) -> Dict[str, Any] | None:
        try:
            payload = load_bin(path, strict=True)
        except FileNotFoundError:
            return None
        row = loads_pickle(payload)
        if row and keep_payload:
            return _PickleRecord(row, payload)
        return row or None

    def _copy_row(self, row: Dict[str, Any], selection: tuple[str, ...] | None = None) -> Dict[str, Any]:
        """Detach the selected record values using the format in which they were stored."""
        if isinstance(row, _PickleRecord):
            result = loads_pickle(row.payload)
            result.setdefault("object_id", row["object_id"])
            return self._project_row(result, selection)
        return copy_data(self._project_row(row, selection))

    def _save_record(self, entity_id: str, row_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        entity_dir = self._entity_dir(entity_id)
        found, _ = self._find_record(entity_dir, row_id, need_row=False)
        hashed = _long_path(os.path.join(entity_dir, self._hashed_name(row_id)))
        name = self._encoded_name(row_id)
        encoded = os.path.join(entity_dir, name) if self._writable(entity_dir, name) else None
        stale = None
        if found is not None and found in (hashed, encoded):
            path = found
        elif found is not None:
            path, stale = hashed, found
        elif encoded is not None and not _exists(encoded):
            path = encoded
        else:
            path = hashed
        tmp = temp_file(path)  # creates the folder when it is missing
        try:
            if self.fmt == "json":
                dump_json(values, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            else:
                payload = dumps_pickle(values)
                dump_bin(payload, tmp)
                values = _PickleRecord(values, payload)
            replace_file(tmp, path)
        except BaseException:
            delete_file(tmp)
            raise
        if stale is not None:
            delete_file(stale)
        return values

    def _delete_record(self, entity_id: str, row_id: str) -> None:
        entity_dir = self._entity_dir(entity_id)
        for _ in range(2):
            path, _row = self._find_record(entity_dir, row_id, need_row=False)
            if path is None:
                return
            delete_file(path)

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Validate uniqueness constraints against the entity's record files.

        Args:
            schema: Entity schema owning the resolved declarations.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Index state is rebuilt from the record files.

        Raises:
            ValueError: If stored record files violate a uniqueness constraint.
        """
        if storage_plan.backend_constraints(self.name):
            self._load_all(check_entity_id(schema.entity_id))
        super().ensure_indexes(schema, storage_plan)

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations to memory and persisted record files.

        Args:
            rows: Backend row mutations. Upserts rewrite one record file; deletes remove
                the record file if it exists. Each row reaches memory only after its file
                change succeeds, so a failing row leaves memory as it is on disk; rows
                before it in the call stay applied.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If the call would leave two rows sharing a value tuple of a
                unique constraint. No row or record file changes then.

        Notes:
            Unique constraints are checked against the entity's record files at the
            call, with every mutation of the call applied, before any record file
            changes. Each call lists the entity folder and reads again only the
            record files that changed since this backend last read them. The check
            and the writes hold a lock that every File Backend over the same folder
            shares within this process; a writer in another process is not
            serialized against them. A failure while writing record files can leave
            earlier rows of the call written: the File Backends do not claim
            ``atomic-upsert``.
        """
        with _folder_lock(self.path):
            ns = self.data.setdefault(self.ws_id, {})
            staged = [(row, None if row.is_delete() else row.to_row()) for row in rows]
            for entity_id in dict.fromkeys(row.entity_id for row in rows):
                if self._unique_fields.get(entity_id):
                    self._load_all(check_entity_id(entity_id))
                    prospective = dict(ns.get(entity_id, {}))
                    for row, item in staged:
                        if row.entity_id != entity_id:
                            continue
                        if item is None:
                            prospective.pop(row.object_id, None)
                        else:
                            prospective[row.object_id] = item
                    self._check_unique(entity_id, prospective)
                else:
                    self._ensure_loaded(entity_id)
            for row, item in staged:
                entity_rows = ns.setdefault(row.entity_id, {})
                if item is None:
                    self._delete_record(row.entity_id, str(row.object_id))
                    if entity_rows.pop(row.object_id, None) is not None:
                        self._ids_changed(row.entity_id)
                    continue
                saved = self._save_record(row.entity_id, str(row.object_id), item)
                if row.object_id not in entity_rows:
                    self._ids_changed(row.entity_id)
                entity_rows[row.object_id] = saved

    @backend_locked
    def compare_upsert(
        self,
        row: RowOp,
        *,
        revision_field: str,
        expected_revision: int | None,
    ) -> bool:
        """Conditionally replace one row and its record file.

        The current revision is read from the row's record file, not a cached copy.
        The comparison and the write hold the lock that every File Backend over the
        same folder shares within this process, so the operation is atomic within
        the process (``compare-upsert: process``), not across processes.

        Args:
            row (RowOp): Replacement upsert containing the next revision.
            revision_field (str): Row field holding the current integer revision.
            expected_revision (int | None): Required revision, or `None` when
                the row must be absent.

        Returns:
            bool: Whether the conditional replacement succeeded.

        Raises:
            ValueError: If `row` is a delete mutation, or if the replacement
                violates a unique constraint.
        """
        if not row.is_upsert():
            raise ValueError("compare_upsert requires an upsert RowOp")
        entity_id = check_entity_id(row.entity_id)
        with _folder_lock(self.path):
            _, current = self._find_record(self._entity_dir(entity_id), str(row.object_id), need_row=True)
            if expected_revision is None:
                if current is not None:
                    return False
            elif current is None or current.get(revision_field) != expected_revision:
                return False
            self.upsert([row])
            return True

    @backend_locked
    def upsert_owned(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply owner-checked Catalog mutations in this workspace's private directory."""
        if not self.ws_id or self.ws_id != owner:
            raise NotImplementedError("File ownership-guarded mutations require a matching workspace directory")
        return super().upsert_owned(rows, owner_field=owner_field, owner=owner)

    def _guard_registration_rows(self, entity_id: str, object_ids: Sequence[str]) -> Any:
        """Require a guard that also excludes writers in other processes."""
        raise NotImplementedError(f"Backend {self.type!r} cannot conditionally restore MetaSchema rows")

    @backend_locked
    def scan(self, entity_id: str) -> List[Dict[str, Any]]:
        """Load and return all rows for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            List[Dict[str, Any]]: Detached row fragments loaded from the entity folder.
        """
        entity_id = check_entity_id(entity_id)
        self._load_all(entity_id)
        return super().scan(entity_id)

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Iterate row fragments of one entity after reloading its record files.

        The entity folder is read once when iteration starts, exactly as
        ``scan`` reads it, so rows written by another process are seen; the
        in-memory rows then serve the iteration.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive scan batch size.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Yields:
            dict[str, Any]: Next detached row fragment.

        Raises:
            ValueError: If the page size is not positive.

        Notes:
            The generator holds the backend lock across ``yield``, like the
            in-memory backend; page with ``scan_page`` to release it between pages.
        """
        self._check_batch_size(batch_size)
        entity_id = check_entity_id(entity_id)
        with self._lock:
            self._load_all(entity_id)
            yield from super().iter_scan(entity_id, batch_size=batch_size, columns=columns)

    @backend_locked
    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> List[Dict[str, Any]]:
        """Return one ``object_id``-ordered page of rows, hydrating the entity from its record files first.

        The first page (``after=None``) reloads the entity folder exactly as
        ``scan`` does, so a fresh enumeration sees rows written by another
        process; later pages read the rows already loaded, which this
        backend's own writes keep current.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Dict[str, Any]]: Detached row fragments in ascending ``object_id`` order.

        Raises:
            ValueError: If the page size is not positive.
        """
        self._check_batch_size(batch_size)
        entity_id = check_entity_id(entity_id)
        if after is None:
            self._load_all(entity_id)
        else:
            self._ensure_loaded(entity_id)
        return super().scan_page(entity_id, after=after, batch_size=batch_size, columns=columns)

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> List[Optional[Dict[str, Any]]]:
        """Return requested rows by reading only the needed record files.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to read.
            columns: Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Optional[Dict[str, Any]]]: Detached row fragments aligned to ``object_ids``.
                Missing files produce ``None``.
        """
        entity_id = check_entity_id(entity_id)
        selection = self._row_columns(columns)
        out: List[Optional[Dict[str, Any]]] = []
        bucket = self.data.setdefault(self.ws_id, {}).setdefault(entity_id, {})
        pickle_loader = self._load_pickle_record if self.fmt == "pickle" else None
        entity_dir = self._entity_dir(entity_id)
        for object_id in object_ids:
            _, row = self._find_record(entity_dir, str(object_id), need_row=True, loader=pickle_loader)
            if row is not None:
                row.setdefault("object_id", object_id)
                if pickle_loader is not None:
                    # This fresh graph can go directly to the caller. Scans
                    # and writes populate their own detached row cache.
                    out.append(self._project_row(row, selection))
                    continue
                if row["object_id"] not in bucket:
                    self._ids_changed(entity_id)
                bucket[row["object_id"]] = row
                out.append(self._copy_row(row, selection))
            else:
                if bucket.pop(object_id, None) is not None:
                    self._ids_changed(entity_id)
                out.append(None)
        return out

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> List[bool]:
        """Return existence flags without loading the whole entity folder.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to test.

        Returns:
            List[bool]: Boolean flags aligned to ``object_ids``.
        """
        entity_id = check_entity_id(entity_id)
        bucket = self.data.setdefault(self.ws_id, {}).setdefault(entity_id, {})
        out = []
        entity_dir = self._entity_dir(entity_id)
        for object_id in object_ids:
            exists = self._find_record(entity_dir, str(object_id), need_row=False)[0] is not None
            if not exists and bucket.pop(object_id, None) is not None:
                self._ids_changed(entity_id)
            out.append(exists)
        return out

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return row count without loading every record body.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            int: Number of record files for the entity. An encoded-name file that stores the same id as
                an existing hashed-name copy is left out: it is the original of a move that was interrupted
                before removing it, and ``scan`` reads only the copy.
        """
        entity_id = check_entity_id(entity_id)
        entity_dir = self._entity_dir(entity_id)
        listed = _long_path(entity_dir)
        if not _exists(listed):
            return 0
        names = {os.path.basename(path) for path in list_files(listed, ext=self.suffix[1:], abs=True)}
        return sum(1 for name in names if not self._is_moved_original(entity_dir, name, names))

    def _is_moved_original(self, entity_dir: str, name: str, names: set[str]) -> bool:
        """Return whether encoded-name record ``name`` is the original of a move whose hashed copy is in ``names``.

        Only an encoded name whose id has its hashed name present is read, to confirm the id it stores.
        """
        if _HASHED_SEP in name:
            return False
        row_id = unquote(name[: -len(self.suffix)])
        if self._hashed_name(row_id) not in names:
            return False
        try:
            row = self._load_record(_long_path(os.path.join(entity_dir, name)))
        except Exception:  # an unreadable record is counted, as before
            return False
        return row is not None and str(row.get("object_id", row_id)) == row_id

    def to_dict(self) -> dict[str, Any]:
        """Return file backend metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend name, backend type, active path, and record format.
        """
        return {"name": self.name, "type": self.type, "path": self.path, "fmt": self.fmt}
