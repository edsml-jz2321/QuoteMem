"""Tests of next_version.py: python -m unittest discover -s .github/release"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from next_version import decide, main

TAGS = ["v0.1.0", "v0.3.0", "not-a-release"]


def write_tree(
    root: Path,
    py: str = "0.3.1",
    npm: str = "0.3.1",
    server: str | None = "0.3.1",
    changelog: str = "## 0.3.1 - 2026-12-01\n\n- A change.\n\n## 0.3.0 - 2026-11-24\n",
) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "quotemem"\nversion = "{py}"\n')
    (root / "package.json").write_text(json.dumps({"name": "quotemem", "version": npm}))
    if server is not None:
        entry = {"version": server, "packages": [{"identifier": "quotemem", "version": server}]}
        (root / "server.json").write_text(json.dumps(entry))
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n{changelog}")


class Decide(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_tagged_version_does_nothing(self) -> None:
        write_tree(self.root, py="0.3.0", npm="0.3.0", server="0.3.0", changelog="")
        self.assertEqual(decide(self.root, TAGS), (False, "0.3.0", []))

    def test_new_matching_version_is_released(self) -> None:
        write_tree(self.root)
        self.assertEqual(decide(self.root, TAGS), (True, "0.3.1", []))

    def test_first_release_without_tags(self) -> None:
        write_tree(self.root)
        self.assertEqual(decide(self.root, []), (True, "0.3.1", []))

    def test_mismatched_versions_fail(self) -> None:
        write_tree(self.root, py="0.3.1", npm="0.3.0")
        release, _, problems = decide(self.root, TAGS)
        self.assertFalse(release)
        self.assertEqual(len(problems), 1)
        self.assertIn(
            "pyproject.toml names version '0.3.1' and package.json names '0.3.0'", problems[0]
        )

    def test_mismatch_fails_even_if_one_version_is_tagged(self) -> None:
        write_tree(self.root, py="0.3.0", npm="0.3.1")
        self.assertFalse(decide(self.root, TAGS)[0])
        self.assertEqual(len(decide(self.root, TAGS)[2]), 1)

    def test_missing_changelog_section_fails(self) -> None:
        write_tree(self.root, changelog="## 0.3.0 - 2026-11-24\n")
        release, _, problems = decide(self.root, TAGS)
        self.assertFalse(release)
        self.assertEqual(
            problems,
            ["CHANGELOG.md has no dated section for 0.3.1; add a heading '## 0.3.1 - YYYY-MM-DD'"],
        )

    def test_undated_changelog_section_fails(self) -> None:
        for heading in ("## 0.3.1\n", "## 0.3.1 - Unreleased\n", "## 0.3.10 - 2026-12-01\n"):
            with self.subTest(heading=heading):
                write_tree(self.root, changelog=heading)
                self.assertFalse(decide(self.root, TAGS)[0])

    def test_invalid_changelog_date_fails(self) -> None:
        write_tree(self.root, changelog="## 0.3.1 - 2026-13-40\n")
        problems = decide(self.root, TAGS)[2]
        self.assertEqual(
            problems, ["CHANGELOG.md section for 0.3.1 has an invalid date 2026-13-40"]
        )

    def test_version_not_above_latest_tag_fails(self) -> None:
        write_tree(
            self.root, py="0.2.0", npm="0.2.0", server="0.2.0", changelog="## 0.2.0 - 2026-12-01\n"
        )
        problems = decide(self.root, TAGS)[2]
        self.assertEqual(
            problems, ["version 0.2.0 is not higher than the latest release tag v0.3.0"]
        )

    def test_tags_compare_as_numbers(self) -> None:
        write_tree(
            self.root,
            py="0.10.0",
            npm="0.10.0",
            server="0.10.0",
            changelog="## 0.10.0 - 2026-12-01\n",
        )
        self.assertEqual(decide(self.root, ["v0.9.0", "v0.3.0"]), (True, "0.10.0", []))

    def test_server_json_version_fails(self) -> None:
        write_tree(self.root, server="0.3.0")
        problems = decide(self.root, TAGS)[2]
        self.assertEqual(len(problems), 1)
        self.assertIn("server.json", problems[0])

    def test_not_semver_fails(self) -> None:
        write_tree(self.root, py="0.3.1rc1", npm="0.3.1rc1")
        self.assertEqual(
            decide(self.root, TAGS)[2], ["version '0.3.1rc1' is not MAJOR.MINOR.PATCH"]
        )


class Main(unittest.TestCase):
    def run_main(self, root: Path, tags: list[str]) -> tuple[int, str, str]:
        tags_file = root / "tags.txt"
        tags_file.write_text("\n".join(tags) + "\n")
        output = root / "github_output"
        output.write_text("")
        old = os.environ.get("GITHUB_OUTPUT")
        os.environ["GITHUB_OUTPUT"] = str(output)
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = main(["next_version.py", str(tags_file), str(root)])
        finally:
            if old is None:
                del os.environ["GITHUB_OUTPUT"]
            else:
                os.environ["GITHUB_OUTPUT"] = old
        return code, buffer.getvalue(), output.read_text()

    def test_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_tree(root)
            self.assertEqual(self.run_main(root, TAGS)[::2], (0, "release=true\nversion=0.3.1\n"))
            self.assertEqual(
                self.run_main(root, [*TAGS, "v0.3.1"])[::2], (0, "release=false\nversion=0.3.1\n")
            )
            write_tree(root, npm="0.3.0")
            code, stdout, output = self.run_main(root, TAGS)
            self.assertEqual((code, output), (1, ""))
            self.assertTrue(stdout.startswith("::error::"))
            self.assertIn("Nothing was tagged.", stdout)


if __name__ == "__main__":
    unittest.main()
