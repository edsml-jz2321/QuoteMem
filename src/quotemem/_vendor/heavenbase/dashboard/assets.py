"""Resolve dashboard GUI assets across source and wheel installations."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import sysconfig


def gui_file(area: str, *parts: str) -> Path:
    """Resolve one GUI asset from a source checkout or installed wheel."""
    relative = Path("share") / "heavenbase" / "gui" / area / Path(*parts)
    source = Path(__file__).resolve().parents[3] / "apps" / "gui" / area / Path(*parts)
    if source.is_file():
        return source

    installed = Path(sysconfig.get_path("data")) / relative
    if installed.is_file():
        return installed

    try:
        package = distribution("heavenbase")
    except PackageNotFoundError:
        return installed
    suffix = relative.as_posix()
    for entry in package.files or ():
        if entry.as_posix().endswith(suffix):
            candidate = Path(package.locate_file(entry))
            if candidate.is_file():
                return candidate
    return installed
