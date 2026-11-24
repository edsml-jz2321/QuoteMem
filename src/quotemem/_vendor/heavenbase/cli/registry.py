"""Registry for backend-neutral CLI command groups and aliases."""

from __future__ import annotations

__all__ = ["CLIRegistry"]

from heavenbase.utils.typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .spec import CommandGroupSpec, CommandSpec

Scope = Tuple[str, ...]


def _scope(value: Optional[str | Sequence[str]]) -> Scope:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace("/", ".").split(".") if part)
    return tuple(str(part) for part in value if str(part))


class CLIRegistry:
    """Mutable registry of root commands, command groups, and scoped aliases.

    Args:
        aliases: Optional initial alias mappings keyed by scope. Each scope maps
            canonical command or group names to alias sequences.
    """

    def __init__(self, aliases: Optional[Mapping[Sequence[str] | str | None, Mapping[str, Sequence[str]]]] = None) -> None:
        """Initialize an empty CLI registry.

        Constructor parameters are documented on ``CLIRegistry``.
        """
        self.root_commands: List[CommandSpec] = []
        self.groups: List[CommandGroupSpec] = []
        self.aliases: Dict[Scope, Dict[str, List[str]]] = {}
        for scope, mapping in (aliases or {}).items():
            self.register_aliases(scope, mapping)

    def register_aliases(self, scope: Optional[str | Sequence[str]], mapping: Mapping[str, Sequence[str]]) -> None:
        """Register aliases for a scope.

        Args:
            scope (str | Sequence[str] | None): Scope prefix for nested groups.
                An empty scope applies to root commands and groups.
            mapping (Mapping[str, Sequence[str]]): Canonical names mapped to
                alias sequences.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If an alias conflicts with a reserved or existing name.
        """
        scope_key = _scope(scope)
        scoped = self.aliases.setdefault(scope_key, {})
        reverse = self.alias_map(scope_key)
        reserved = set(self._canonical_names(scope_key))
        for canonical, aliases in mapping.items():
            name = str(canonical).strip()
            if not name:
                raise ValueError("Alias canonical name must be non-empty")
            scoped.setdefault(name, [])
            for raw_alias in aliases:
                alias = str(raw_alias).strip()
                if not alias or alias == name:
                    continue
                if alias in reserved:
                    raise ValueError(f"Alias {alias!r} conflicts with command {alias!r} in scope {scope_key!r}")
                if alias in scoped and alias != name:
                    raise ValueError(f"Alias {alias!r} conflicts with command {alias!r} in scope {scope_key!r}")
                target = reverse.get(alias)
                if target and target != name:
                    raise ValueError(f"Alias {alias!r} already points to {target!r} in scope {scope_key!r}")
                if alias not in scoped[name]:
                    scoped[name].append(alias)
                    reverse[alias] = name

    def alias_map(self, scope: Optional[str | Sequence[str]]) -> Dict[str, str]:
        """Return alias-to-canonical mapping for a scope.

        Args:
            scope (str | Sequence[str] | None): Scope prefix for nested groups.

        Returns:
            Dict[str, str]: Alias names mapped to canonical command or group names.
        """
        scoped = self.aliases.get(_scope(scope), {})
        return {alias: canonical for canonical, aliases in scoped.items() for alias in aliases}

    def aliases_for(self, scope: Optional[str | Sequence[str]], canonical: str) -> List[str]:
        """Return aliases for one canonical command or group.

        Args:
            scope (str | Sequence[str] | None): Scope prefix for nested groups.
            canonical (str): Canonical command or group name.

        Returns:
            List[str]: Registered aliases for ``canonical`` within ``scope``.
        """
        return list(self.aliases.get(_scope(scope), {}).get(canonical, []))

    def display_name(self, scope: Optional[str | Sequence[str]], canonical: str) -> str:
        """Return a help-table display name such as ``list|ls``.

        Args:
            scope (str | Sequence[str] | None): Scope prefix for nested groups.
            canonical (str): Canonical command or group name.

        Returns:
            str: Pipe-joined canonical name and aliases in stable order.
        """
        names = [canonical, *self.aliases_for(scope, canonical)]
        return "|".join(dict.fromkeys(names))

    def add_root(self, command: CommandSpec) -> CommandSpec:
        """Add a root-level command.

        Args:
            command (CommandSpec): Backend-neutral root command specification.

        Returns:
            CommandSpec: The registered ``command`` instance.

        Raises:
            ValueError: If a root command or group with the same name exists.
        """
        self._ensure_root_unique(command.name)
        self.root_commands.append(command)
        if command.aliases:
            self.register_aliases((), {command.name: command.aliases})
        return command

    def add_group(self, group: CommandGroupSpec) -> CommandGroupSpec:
        """Add a command group.

        Args:
            group (CommandGroupSpec): Backend-neutral command group specification.

        Returns:
            CommandGroupSpec: The registered ``group`` instance.

        Raises:
            ValueError: If a root command or duplicate group command name exists.
        """
        self._ensure_root_unique(group.name)
        seen: set[str] = set()
        for command in group.commands:
            if command.name in seen:
                raise ValueError(f"Command {command.name!r} appears twice in group {group.name!r}")
            seen.add(command.name)
        self.groups.append(group)
        if group.aliases:
            self.register_aliases((), {group.name: group.aliases})
        for command in group.commands:
            if command.aliases:
                self.register_aliases((group.name,), {command.name: command.aliases})
        return group

    def extend(self, groups: Iterable[CommandGroupSpec]) -> "CLIRegistry":
        """Add many groups and return self.

        Args:
            groups (Iterable[CommandGroupSpec]): Command groups to register.

        Returns:
            CLIRegistry: This registry after registering every group.
        """
        for group in groups:
            self.add_group(group)
        return self

    def _ensure_command_unique(self, commands: Sequence[CommandSpec], name: str) -> None:
        if any(command.name == name for command in commands):
            raise ValueError(f"Command {name!r} already exists")

    def _ensure_root_unique(self, name: str) -> None:
        self._ensure_command_unique(self.root_commands, name)
        if any(group.name == name for group in self.groups):
            raise ValueError(f"Command group {name!r} already exists")

    def _canonical_names(self, scope: Optional[str | Sequence[str]]) -> List[str]:
        scope_key = _scope(scope)
        if not scope_key:
            return [command.name for command in self.root_commands] + [group.name for group in self.groups]
        if len(scope_key) == 1:
            group = next((item for item in self.groups if item.name == scope_key[0]), None)
            return group.command_names() if group else []
        return []
