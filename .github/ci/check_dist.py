"""Check the built sdist and wheel against the repository checkout.

Run it from the root of a git checkout after `uv build --out-dir dist`; it stops at once
anywhere else. It fails when:

- the sdist does not hold exactly one top-level directory `quotemem-<version>/`;
- a file in the sdist, other than `PKG-INFO`, is not a tracked file of the checkout, or
  differs from the checkout's copy byte for byte (this catches an ignore file or any
  other file picked up from outside the repository);
- a required file is missing from the sdist;
- a path in the sdist or the wheel looks like local state (environments, caches,
  databases, keys, build output);
- the wheel holds anything except the `quotemem` package and its `.dist-info`, or lacks
  the licence files.

Usage: python .github/ci/check_dist.py dist
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

HEAVENBASE_LICENCE = "quotemem/_vendor/heavenbase-0.1.2.3.dist-info/licenses/LICENSE"
REQUIRED_SDIST = ("PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "NOTICE",
                  "src/quotemem/__init__.py", f"src/{HEAVENBASE_LICENCE}")
REQUIRED_WHEEL_LICENCES = ("LICENSE", "NOTICE")
LOCAL_STATE_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                     ".mypy_cache", ".pyright", "node_modules", ".git", "dist", "build",
                     "qm-workspace"}
LOCAL_STATE_SUFFIXES = (".pyc", ".pyo", ".db", ".db-wal", ".db-shm", ".key", ".tgz")
LOCAL_STATE_NAMES = {".env", ".DS_Store", "uv.lock"}


def _local_state(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    return (bool(LOCAL_STATE_PARTS.intersection(parts)) or name in LOCAL_STATE_NAMES
            or name.endswith(LOCAL_STATE_SUFFIXES))


def _require_checkout_root() -> None:
    """Stop unless the working directory is the root of a git checkout."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                         text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path.cwd().resolve():
        raise SystemExit("check_dist.py runs at the root of the git checkout it checks")


def _tracked_files() -> set[str]:
    listing = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True)
    return {name for name in listing.stdout.decode().split("\0") if name}


def _one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {directory}, found {len(found)}")
    return found[0]


def check_sdist(sdist: Path, tracked: set[str], errors: list[str]) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        tops = {PurePosixPath(member.name).parts[0] for member in archive.getmembers()}
        expected_top = sdist.name.removesuffix(".tar.gz")
        if tops != {expected_top}:
            errors.append(f"sdist top-level entries {sorted(tops)} != [{expected_top!r}]")
        names: set[str] = set()
        for member in members:
            relative = PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
            names.add(relative)
            if _local_state(relative):
                errors.append(f"sdist holds local state: {relative}")
            if relative == "PKG-INFO":
                continue
            if relative not in tracked:
                errors.append(f"sdist holds a file the checkout does not track: {relative}")
                continue
            handle = archive.extractfile(member)
            content = handle.read() if handle is not None else b""
            if content != Path(relative).read_bytes():
                errors.append(f"sdist copy differs from the checkout: {relative}")
        for required in REQUIRED_SDIST:
            if required not in names:
                errors.append(f"sdist lacks {required}")
        print(f"sdist {sdist.name}: {len(members)} files")


def check_wheel(wheel: Path, errors: list[str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    dist_info = next((name.split("/")[0] for name in names
                      if name.split("/")[0].endswith(".dist-info")), "")
    for name in names:
        top = name.split("/")[0]
        if top not in {"quotemem", dist_info}:
            errors.append(f"wheel holds a path outside quotemem/ and {dist_info}/: {name}")
        if _local_state(name):
            errors.append(f"wheel holds local state: {name}")
    for licence in REQUIRED_WHEEL_LICENCES:
        if f"{dist_info}/licenses/{licence}" not in names:
            errors.append(f"wheel lacks {dist_info}/licenses/{licence}")
    for required in ("quotemem/__init__.py", HEAVENBASE_LICENCE):
        if required not in names:
            errors.append(f"wheel lacks {required}")
    print(f"wheel {wheel.name}: {len(names)} files")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    _require_checkout_root()
    directory = Path(argv[1])
    errors: list[str] = []
    check_sdist(_one(directory, "*.tar.gz"), _tracked_files(), errors)
    check_wheel(_one(directory, "*.whl"), errors)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    print("DIST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
