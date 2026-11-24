"""JSON folder backend."""

from __future__ import annotations

__all__ = [
    "JsonFileBackend",
]

from heavenbase.utils import Any, Dict

from ...modules import ModuleResolver
from .base import FileBackend


class JsonFileBackend(FileBackend):
    """Folder backend that persists rows as JSON files."""

    identifier = "json"
    default_path_key = "heavenbase.backend_defaults.json.path"
    default_path = "file://%/json"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "JsonFileBackend":
        """Build the JSON file backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``path`` and optional ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and runtime policy.

        Returns:
            JsonFileBackend: Configured JSON folder backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("path", None), ws_id=ws_id, resolver=resolver, config=config)

    def __init__(
        self,
        name: str = "json",
        path: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> None:
        """Create a JSON folder backend.

        Args:
            name: Workspace backend instance name.
            path: Base folder for JSON record files. When omitted, uses the configured
                JSON backend default path.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and runtime policy.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, path, "json", ws_id=ws_id, resolver=resolver, config=config)
