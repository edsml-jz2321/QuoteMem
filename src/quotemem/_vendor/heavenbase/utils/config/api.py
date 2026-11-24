"""Scope-aware configuration operations for presentation adapters."""

from __future__ import annotations

__all__ = ["ConfigOps"]

import difflib
import os
import shlex
import tempfile

from ..commands import cmd
from .data import dflat, dget
from .manager import ConfigManager
from ..serialize import dump_yaml, dumps_yaml, load_yaml, loads_json
from ..typing import Any
from ..values import raise_mismatch

_PARSE_MODES = ("auto", "json", "raw")
_COPY_SOURCES = ("base", "default")


class ConfigOps:
    """Normalize presentation input and delegate config lifecycle operations.

    Args:
        manager (ConfigManager): Configuration manager that owns persistence,
            history, scopes, and merged reads.
    """

    def __init__(self, manager: ConfigManager) -> None:
        """Bind operations to one configuration manager.

        Args:
            manager (ConfigManager): Configuration manager receiving operations.

        Returns:
            None: This initializer stores the manager.
        """
        if not isinstance(manager, ConfigManager):
            raise TypeError("ConfigOps expects a ConfigManager")
        self.manager = manager

    def get(
        self,
        key: str | None = None,
        *,
        scope: str | None = None,
        version: int | None = None,
        merged: bool = True,
    ) -> Any:
        """Read one config selection.

        Args:
            key (str | None): Optional flattened key below the selected data.
            scope (str | None): Optional relative or full config scope.
            version (int | None): Optional retained version. Negative values
                index retained versions from newest to oldest.
            merged (bool): Whether an unscoped read merges all active layers.

        Returns:
            Any: Selected value, layer mapping, or merged config mapping.
        """
        resolved = self._scope(scope)
        if version is not None:
            target = resolved or self.manager.scope
            resolved_version = self._version(target, version)
            data = {} if resolved_version is None else self.manager.layer(scope=target, version=resolved_version)
        elif resolved:
            data = self.manager.layer(scope=resolved)
        else:
            data = self.manager.load(refresh=True) if merged else self.manager.layer()
        return dget(data, key, default=None) if key else data

    def list(self, prefix: str | None = None, *, scope: str | None = None, merged: bool = True) -> list[dict[str, Any]]:
        """List flattened config key/value rows.

        Args:
            prefix (str | None): Optional flattened key prefix filter.
            scope (str | None): Optional relative or full config scope.
            merged (bool): Whether an unscoped read merges active layers.

        Returns:
            list[dict[str, Any]]: Ordered `key` and `value` row mappings.
        """
        rows = [{"key": key, "value": value} for key, value in dflat(self.get(scope=scope, merged=merged) or {})]
        return [row for row in rows if not prefix or row["key"].startswith(prefix)]

    def set(self, key: str, value: Any, *, scope: str | None = None, parse: str = "auto") -> bool:
        """Set one config key.

        Args:
            key (str): Flattened config key.
            value (Any): Raw value or presentation text.
            scope (str | None): Optional relative or full target scope.
            parse (str): Value parsing mode. Supported values:
                - `auto`: Parse booleans, nulls, integers, and floats.
                - `json`: Parse strict JSON.
                - `raw`: Store strings unchanged.

        Returns:
            bool: Result reported by `ConfigManager.set`.
        """
        return self.manager.set(key, self._parse(value, parse), scope=self._scope(scope))

    def add(self, key: str, value: Any, *, scope: str | None = None, parse: str = "auto") -> bool:
        """Append one value to a config list key.

        Args:
            key (str): Flattened list key, with or without a trailing `[]`.
            value (Any): Raw value or presentation text.
            scope (str | None): Optional relative or full target scope.
            parse (str): Value parsing mode documented by `set`.

        Returns:
            bool: Result reported by `ConfigManager.set`.
        """
        target = key if key.endswith("]") else f"{key}[]"
        return self.set(target, value, scope=scope, parse=parse)

    def unset(self, key: str, *, scope: str | None = None) -> bool:
        """Unset one config key.

        Args:
            key (str): Flattened config key to remove.
            scope (str | None): Optional relative or full target scope.

        Returns:
            bool: Result reported by `ConfigManager.unset`.
        """
        return self.manager.unset(key, scope=self._scope(scope))

    def edit(self, *, scope: str | None = None, editor: str | None = None) -> bool:
        """Edit one raw config layer in an external editor.

        Args:
            scope (str | None): Optional relative or full target scope.
            editor (str | None): Optional editor command. Defaults to `VISUAL`,
                `EDITOR`, or the platform editor.

        Returns:
            bool: Result reported by `ConfigManager.init` after YAML validation.

        Raises:
            ValueError: If the edited YAML root is not a mapping.
        """
        target = self._scope(scope) or self.manager.scope
        fd, path = tempfile.mkstemp(prefix=f"{self.manager.package}-config-", suffix=".yaml")
        os.close(fd)
        try:
            dump_yaml(self.manager.layer(scope=target), path)
            selected = editor or os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
            cmd([*shlex.split(selected, posix=os.name != "nt"), path], check=True)
            data = load_yaml(path, strict=True) or {}
            if not isinstance(data, dict):
                raise ValueError("Edited config layer must be a YAML mapping")
            return self.manager.init(scope=target, reset=True, data=data)
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def copy(
        self,
        key: str | None = None,
        *,
        source: str = "base",
        from_scope: str | None = None,
        from_version: int | None = None,
        target_scope: str | None = None,
    ) -> bool:
        """Copy a config subtree into one target scope.

        Args:
            key (str | None): Optional flattened subtree key. Empty copies the
                complete source layer.
            source (str): Source selection. Supported values:
                - `base`: Read the base or explicit `from_scope` layer.
                - `default`: Read packaged defaults.
            from_scope (str | None): Optional source scope used with `base`.
            from_version (int | None): Optional retained source version.
            target_scope (str | None): Optional destination scope.

        Returns:
            bool: Result reported by `ConfigManager.init` or `set`.

        Raises:
            ValueError: If `source` is unsupported.
        """
        if source not in _COPY_SOURCES:
            raise_mismatch(_COPY_SOURCES, source, name="config copy source")
        if source == "default":
            data = self.manager.load_default()
        else:
            resolved_source = self._scope(from_scope) if from_scope else self.manager.base_scope
            resolved_version = self._version(resolved_source, from_version)
            data = (
                self.manager.layer(scope=resolved_source, version=resolved_version)
                if resolved_version is not None
                else self.manager.layer(scope=resolved_source)
            )
        value = dget(data, key, default=None) if key else data
        if key is None:
            return self.manager.init(scope=self._scope(target_scope), reset=True, data=value)
        return self.manager.set(key, value, scope=self._scope(target_scope))

    def diff(
        self,
        *,
        scope_a: str | None = None,
        version_a: int | None = None,
        scope_b: str | None = None,
        version_b: int | None = None,
    ) -> str:
        """Build a unified diff between two config snapshots.

        Args:
            scope_a (str | None): Optional source scope.
            version_a (int | None): Optional source retained version.
            scope_b (str | None): Optional target scope.
            version_b (int | None): Optional target retained version.

        Returns:
            str: Unified YAML diff, or an empty string when snapshots match.
        """
        resolved_a = self._scope(scope_a) or self.manager.scope
        resolved_b = self._scope(scope_b) or self.manager.scope
        resolved_version_b = self._version(resolved_b, -1 if version_b is None else version_b)
        resolved_version_a = self._version(resolved_a, (-2 if resolved_a == resolved_b else -1) if version_a is None else version_a)
        data_a = self.manager.layer(scope=resolved_a, version=resolved_version_a) if resolved_version_a is not None else {}
        data_b = self.manager.layer(scope=resolved_b, version=resolved_version_b) if resolved_version_b is not None else {}
        label_a = f"{resolved_a} v{resolved_version_a}"
        label_b = f"{resolved_b} v{resolved_version_b}"
        return "".join(difflib.unified_diff(dumps_yaml(data_a).splitlines(True), dumps_yaml(data_b).splitlines(True), fromfile=label_a, tofile=label_b))

    def _scope(self, scope: str | None) -> str | None:
        if not scope:
            return None
        value = scope.strip().lower()
        if value == ".":
            return self.manager.base_scope
        return value if value == self.manager.base_scope or value.startswith(f"{self.manager.base_scope}.") else f"{self.manager.base_scope}.{value}"

    def _version(self, scope: str, version: int | None) -> int | None:
        if version is None or version >= 0:
            return version
        versions = sorted(int(row["ver"]) for row in self.manager.history(scope=scope, limit=1000))
        if not versions:
            return None
        try:
            return versions[version]
        except IndexError:
            return versions[0]

    @staticmethod
    def _parse(value: Any, mode: str) -> Any:
        if mode not in _PARSE_MODES:
            raise_mismatch(_PARSE_MODES, mode, name="config value parse mode")
        if mode == "raw" or not isinstance(value, str):
            return value
        if mode == "json":
            return loads_json(value)
        token = value.strip()
        lower = token.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if lower in {"null", "none"}:
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return value
