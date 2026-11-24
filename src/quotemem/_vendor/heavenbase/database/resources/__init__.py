"""Load packaged SQL resources owned by the database domain."""

from __future__ import annotations

__all__ = ["load_builtin_sql"]

from importlib import resources


def load_builtin_sql(name: str, package: str = "heavenbase.database") -> str:
    """Load a packaged SQL resource by relative path.

    Args:
        name (str): SQL resource filename below `resources/sql`.
        package (str): Python package containing the owner-local
            `resources/sql` folder.

    Returns:
        str: SQL resource text decoded as UTF-8.
    """
    return resources.files(package).joinpath("resources", "sql", name).read_text(encoding="utf-8")
