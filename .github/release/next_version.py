"""Decide whether a push to main starts a release (.github/workflows/tag-release.yml).

    python3 -I .github/release/next_version.py TAGS_FILE [ROOT]

TAGS_FILE lists the repository's tag names, one per line. ROOT is the checkout
(default: the current folder).

When pyproject.toml and package.json name a version whose tag vX.Y.Z exists,
there is nothing to release. When they name a version without a tag, that
version is released, after these checks: both files name the same version,
it is MAJOR.MINOR.PATCH and higher than every vX.Y.Z tag, server.json names it,
and CHANGELOG.md has a dated section `## X.Y.Z - YYYY-MM-DD` for it.

It writes `release=true|false` and `version=X.Y.Z` to $GITHUB_OUTPUT when set,
and exits 1 with one ::error:: line per problem.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import tomllib
from pathlib import Path

VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def _key(match: re.Match[str]) -> tuple[int, int, int]:
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def decide(root: Path, tags: list[str]) -> tuple[bool, str | None, list[str]]:
    """Return (release, version, problems) for the checkout at root."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    py_version = project.get("project", {}).get("version")
    npm_version = package.get("version")

    if py_version == npm_version and f"v{py_version}" in tags:
        return False, py_version, []
    if py_version != npm_version:
        return (
            False,
            None,
            [
                f"pyproject.toml names version {py_version!r} and package.json names "
                f"{npm_version!r}; set both to the new version"
            ],
        )

    version = str(py_version)
    problems: list[str] = []
    match = VERSION.fullmatch(version)
    if match is None:
        return False, version, [f"version {version!r} is not MAJOR.MINOR.PATCH"]

    released = [m for m in (TAG.fullmatch(tag.strip()) for tag in tags) if m is not None]
    if released:
        highest = max(released, key=_key)
        if _key(match) <= _key(highest):
            problems.append(
                f"version {version} is not higher than the latest release tag {highest.group(0)}"
            )

    server_path = root / "server.json"
    server = json.loads(server_path.read_text(encoding="utf-8")) if server_path.exists() else {}
    listed = [server.get("version")] + [p.get("version") for p in server.get("packages", [])]
    if any(found != version for found in listed):
        problems.append(f"server.json names version(s) {listed}; set them to {version}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(
        rf"^## {re.escape(version)} - (\d{{4}}-\d{{2}}-\d{{2}})[ \t]*$", changelog, re.MULTILINE
    )
    if heading is None:
        problems.append(
            f"CHANGELOG.md has no dated section for {version}; add a heading "
            f"'## {version} - YYYY-MM-DD'"
        )
    else:
        try:
            datetime.date.fromisoformat(heading.group(1))
        except ValueError:
            problems.append(
                f"CHANGELOG.md section for {version} has an invalid date {heading.group(1)}"
            )

    return not problems, version, problems


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    tags = Path(argv[1]).read_text(encoding="utf-8").split()
    root = Path(argv[2]) if len(argv) == 3 else Path()
    release, version, problems = decide(root, tags)
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        print("Nothing was tagged.")
        return 1
    if release:
        print(f"version {version} has no tag yet: tag v{version} after CI passes")
    else:
        print(f"version {version} is already tagged v{version}: nothing to release")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"release={'true' if release else 'false'}\n")
            handle.write(f"version={version}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
