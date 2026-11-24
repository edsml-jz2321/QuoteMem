"""Installable agent-skill artifact for one workspace MCP profile."""

from __future__ import annotations

__all__ = ["WorkspaceSkill"]

import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlsplit

from heavenbase.utils import Any, Dict, List, Mapping, dumps_json, kebab

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_WILDCARD_HOSTS = {"0.0.0.0", "::", "[::]"}


def _skill_name(value: str) -> str:
    """Return one validated kebab-case skill identifier."""
    name = kebab(str(value or "").strip()).strip("-")
    if not name or len(name) > 64 or _SKILL_NAME.fullmatch(name) is None:
        raise ValueError("Workspace skill name must be 1-64 lowercase letters, digits, or single hyphen-separated words")
    return name


def _client_url(host: str, port: int) -> str:
    """Return the local client URL for one MCP server bind."""
    selected = "127.0.0.1" if host in _WILDCARD_HOSTS else host
    if ":" in selected and not selected.startswith("["):
        selected = f"[{selected}]"
    return f"http://{selected}:{port}/mcp"


def _validated_url(value: str) -> str:
    """Return a credential-free streamable-HTTP MCP URL."""
    url = str(value or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Workspace skill MCP URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("Workspace skill MCP URL must not contain credentials, a query string, or a fragment")
    return url


def _profile_instructions(profile: Any) -> List[str]:
    """Return extension-owned instructions from one resolved profile."""
    raw = dict(getattr(profile, "meta", None) or {}).get("workspace_skill") or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MCP profile {profile.name!r} workspace_skill metadata must be a mapping")
    instructions = raw.get("instructions") or []
    if not isinstance(instructions, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in instructions):
        raise TypeError(f"MCP profile {profile.name!r} workspace_skill instructions must be a string list")
    return [item.strip() for item in instructions]


class WorkspaceSkill:
    """Detached installable skill describing one live workspace MCP surface.

    The artifact contains only workspace identity, the selected exposure
    profile, public tool/schema guidance, and a credential-free MCP URL. It
    never serializes workspace construction or source-database credentials.
    Construction is side-effect free; call :meth:`save` or :meth:`install` to
    write a skill folder.
    """

    def __init__(
        self,
        workspace: Any,
        *,
        profile: str,
        name: str | None = None,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Describe one workspace/profile as an installable agent skill.

        Args:
            workspace: Live registered or detached HeavenBase workspace.
            profile: Explicit MCP profile exposed to the skill.
            name: Optional kebab-case skill name. Defaults to the workspace id.
            url: Optional already-served streamable-HTTP MCP URL. When omitted,
                the workspace MCP host/port settings produce a local URL.
            host: Optional local server host override used when ``url`` is omitted.
            port: Optional local server port override used when ``url`` is omitted.

        Raises:
            ValueError: If the name, profile, URL, or port is invalid.
            RuntimeError: If the selected profile cannot be built for the workspace.
        """
        from .support import mcp_host, mcp_port, workspace_mcp_profile

        selected_profile = str(profile or "").strip()
        if not selected_profile:
            raise ValueError("Workspace skill requires an explicit MCP profile")
        toolkit = workspace.to_mcp(profile=selected_profile)
        resolver = workspace.module_resolver
        config = workspace.context.config
        resolved = workspace_mcp_profile(workspace, selected_profile, resolver=resolver)
        server_host = mcp_host(host, config=config, resolver=resolver)
        server_port = mcp_port(port, config=config, resolver=resolver)
        if server_port < 1 or server_port > 65535:
            raise ValueError("Workspace skill MCP port must be between 1 and 65535")

        self.workspace_id = str(workspace.id)
        self.profile = resolved.name
        self.name = _skill_name(name or self.workspace_id)
        self.url = _validated_url(url or _client_url(server_host, server_port))
        self.host = server_host
        self.port = server_port
        self.local_server = url is None
        self.description = str(resolved.desc or toolkit.description or f"HeavenBase workspace {self.workspace_id}").strip()
        self.tools = tuple(toolkit.list_tools())
        self.entities = tuple(sorted(workspace.entities if resolved.entities == "*" else resolved.entities))
        self.instructions = tuple(_profile_instructions(resolved))

    @staticmethod
    def default_install_root() -> str:
        """Return the user skill directory used by Codex.

        Returns:
            str: Absolute ``$CODEX_HOME/skills`` or ``~/.codex/skills`` path.
        """
        base = Path(os.environ.get("CODEX_HOME") or "~/.codex").expanduser()
        return str((base / "skills").resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Return credential-free inspectable artifact metadata."""
        return {
            "name": self.name,
            "workspace": self.workspace_id,
            "profile": self.profile,
            "url": self.url,
            "tools": list(self.tools),
            "entities": list(self.entities),
        }

    def _skill_markdown(self) -> str:
        """Render the skill instruction document."""
        description = f"Use HeavenBase workspace {self.workspace_id!r} through its {self.profile!r} MCP profile. {self.description}"
        lines = [
            "---",
            f"name: {self.name}",
            f"description: {dumps_json(description)}",
            "---",
            "",
            f"# {self.workspace_id} workspace",
            "",
            f"Use the MCP server `{self.name}` at `{self.url}`. Its designated HeavenBase profile is `{self.profile}`.",
            "",
        ]
        if self.local_server:
            lines.extend(
                [
                    "## Start the server",
                    "",
                    "If the MCP tools are unavailable, start the workspace server and then reconnect the MCP client:",
                    "",
                    "```bash",
                    f"hb mcp workspace {self.workspace_id} --profile {self.profile} --transport http --host {self.host} --port {self.port}",
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "## Working method",
                "",
                "1. Discover before acting: call `list_entities` with `include_system=true`, then `describe_entity` for relevant profile-scoped entities before reading or editing rows.",
                "2. Treat workspace rows as durable context and source-execution tools as evidence. Never infer real-world values from schema metadata alone.",
                "3. Use the narrowest read that answers the question. Only call write tools when the user explicitly asks to change or retain workspace knowledge.",
                "4. Preserve user meaning when editing descriptions or rules, but summarize it into concise standalone context.",
                "5. State limitations and truncation. Never invent rows, fields, tool success, or source results.",
                "",
            ]
        )
        if self.instructions:
            lines.extend(["## Profile workflow", "", *(f"- {item}" for item in self.instructions), ""])
        lines.extend(
            [
                "## Exposed tools",
                "",
                *(f"- `{tool}`" for tool in self.tools),
                "",
                "## Entity scope",
                "",
                *(f"- `{entity}`" for entity in self.entities),
                "",
                "## Fallback MCP client",
                "",
                "When the host does not attach declared MCP dependencies dynamically, use the bundled client; it still calls the same MCP server:",
                "Resolve the script relative to this `SKILL.md` directory, regardless of the current working directory.",
                "",
                "```bash",
                "python scripts/client.py list",
                "python scripts/client.py call list_entities '{}'",
                "```",
            ]
        )
        return "\n".join(lines).rstrip()

    def _openai_yaml(self) -> str:
        """Render Codex skill metadata with one streamable-HTTP dependency."""
        display = self.workspace_id.replace("-", " ").replace("_", " ").strip().title() or self.name
        short = f"Use {self.workspace_id} via HeavenBase MCP"
        prompt = f"Use ${self.name} to work with HeavenBase workspace {self.workspace_id} through the {self.profile} MCP profile."
        dependency = f"HeavenBase workspace {self.workspace_id} ({self.profile} profile)"
        return "\n".join(
            [
                "interface:",
                f"  display_name: {dumps_json(display)}",
                f"  short_description: {dumps_json(short[:64])}",
                f"  default_prompt: {dumps_json(prompt)}",
                "dependencies:",
                "  tools:",
                '    - type: "mcp"',
                f"      value: {dumps_json(self.name)}",
                f"      description: {dumps_json(dependency)}",
                '      transport: "streamable_http"',
                f"      url: {dumps_json(self.url)}",
            ]
        )

    def _client_script(self) -> str:
        """Render a small deterministic MCP fallback client."""
        return f'''#!/usr/bin/env python3
"""Fallback client for the {self.name} MCP dependency."""

import argparse
import json

from heavenbase.toolkit import toolkit_from_fastmcp

URL = {self.url!r}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("list", "call"))
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{{}}")
    args = parser.parse_args()
    toolkit = toolkit_from_fastmcp(URL, name={self.name!r})
    if args.action == "list":
        value = [{{"name": name, "description": toolkit[name].description, "input_schema": toolkit[name].input_schema}} for name in toolkit]
    else:
        if not args.tool:
            parser.error("call requires TOOL and optional JSON arguments")
        payload = json.loads(args.arguments)
        if not isinstance(payload, dict):
            parser.error("arguments must decode to a JSON object")
        value = toolkit.run(args.tool, **payload)
    print(json.dumps(value, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
'''.rstrip()

    def _write(self, path: Path) -> None:
        """Write the complete skill into an empty staging directory."""
        (path / "agents").mkdir(parents=True, exist_ok=True)
        (path / "scripts").mkdir(parents=True, exist_ok=True)
        (path / "SKILL.md").write_text(f"{self._skill_markdown()}\n", encoding="utf-8")
        (path / "agents" / "openai.yaml").write_text(f"{self._openai_yaml()}\n", encoding="utf-8")
        client = path / "scripts" / "client.py"
        client.write_text(f"{self._client_script()}\n", encoding="utf-8")
        client.chmod(0o755)

    def save(self, path: str | os.PathLike[str], *, replace: bool = False) -> str:
        """Write this skill folder atomically.

        Args:
            path: Exact destination skill directory.
            replace: Replace an existing exact destination.

        Returns:
            str: Absolute installed skill directory.

        Raises:
            FileExistsError: If the target exists and ``replace`` is false.
        """
        target = Path(path).expanduser().resolve()
        if target.exists() and not replace:
            raise FileExistsError(f"{target} already exists; pass replace=True to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_root = Path(tempfile.mkdtemp(prefix=f".{self.name}-", dir=target.parent))
        staged = stage_root / self.name
        previous = stage_root / "previous"
        try:
            staged.mkdir()
            self._write(staged)
            if target.exists():
                os.replace(target, previous)
            try:
                os.replace(staged, target)
            except BaseException:
                if previous.exists() and not target.exists():
                    os.replace(previous, target)
                raise
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        return str(target)

    def install(self, root: str | os.PathLike[str] | None = None, *, replace: bool = False) -> str:
        """Install the skill under one Codex skill root.

        Args:
            root: Directory containing skill folders. Defaults to
                :meth:`default_install_root`.
            replace: Replace an existing skill with the same name.

        Returns:
            str: Absolute installed skill directory.
        """
        skill_root = Path(root or self.default_install_root()).expanduser().resolve()
        return self.save(skill_root / self.name, replace=replace)
