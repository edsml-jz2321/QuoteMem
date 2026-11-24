"""Pair runtime module and configuration authorities without hybrid defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..modules import ModuleResolver, module_resolver
from ..utils.config import ConfigManager

if TYPE_CHECKING:
    from .runtime import Context

__all__: list[str] = []


@dataclass(frozen=True, eq=False, slots=True)
class RuntimeAuthority:
    """Exact module and configuration authorities for one runtime operation."""

    resolver: ModuleResolver
    config: ConfigManager


def omitted_config(resolver: ModuleResolver | None = None) -> ConfigManager:
    """Return the config an omitted ``config=`` reads, paired with ``resolver``.

    A resolver bound to a Context pairs with that Context's ConfigManager, the
    default Context's modules with the default Context's config, and an
    omitted or unbound resolver with the installed default Context. Before
    the default exists, returns ``CM_HVNB`` without selecting a Context.

    Args:
        resolver (ModuleResolver | None): The module authority the caller
            selected, if any. Defaults to ``None``.

    Returns:
        ConfigManager: The config paired with ``resolver``.
    """
    from ..utils.config import CM_HVNB
    from .runtime import _default_context_config_of, _default_context_instance

    if resolver is not None:
        config = getattr(resolver, "authority_config", None)
        if config is None:
            config = _default_context_config_of(resolver)
        if config is not None:
            return config
    default = _default_context_instance()
    return CM_HVNB if default is None else default.config


def resolve_runtime_authority(
    *,
    context: "Context | None" = None,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
    label: str = "Runtime",
) -> RuntimeAuthority:
    """Resolve one coherent authority pair or reject an unowned partial pair.

    Args:
        context (Context | None): Machine composition root that owns both module
            and configuration authority. When supplied, ``resolver`` and
            ``config`` must be omitted. Defaults to ``None``.
        resolver (ModuleResolver | None): Module authority paired with
            ``config``. Defaults to ``None``.
        config (ConfigManager | None): Configuration authority paired with
            ``resolver``. An unowned ``CM_HVNB`` supplied alone selects
            both authorities from the process default. Defaults to ``None``.
        label (str): Error-message prefix for authority mismatch failures.
            Defaults to ``"Runtime"``.

    Returns:
        RuntimeAuthority: Coherent module and configuration authorities for
        one runtime operation.

    Raises:
        TypeError: If ``context`` is combined with ``resolver`` or ``config``,
            or if a partial pair cannot be bound to a Context.
        ValueError: If ``resolver`` and ``config`` belong to different
            Contexts.
        RuntimeError: If explicit Context mode is on and the call needs the
            process default Context (an omitted pair, or ``CM_HVNB`` before
            the default exists).
    """
    from .runtime import _config_context, _default_context_config_of, context_or_default, default_context

    if context is not None:
        if resolver is not None or config is not None:
            raise TypeError(f"{label} context cannot be combined with resolver or config")
        owner = context_or_default(context)
        return RuntimeAuthority(owner.modules(), owner.config)

    if resolver is None and config is None:
        default = default_context()
        return RuntimeAuthority(default.modules(), default.config)

    if config is not None and not isinstance(config, ConfigManager):
        raise TypeError(f"{label} config authority must be a ConfigManager")

    if resolver is None:
        owner = _config_context(config)
        if owner is None:
            from ..utils.config import CM_HVNB

            if config is CM_HVNB:
                default = context_or_default()
                return RuntimeAuthority(default.modules(), default.config)
            raise TypeError(f"{label} config-only authority is unbound; supply its Context or paired resolver")
        owner = context_or_default(owner)
        resolver = owner.modules()
    else:
        resolver = module_resolver(resolver)

    if config is None:
        config = getattr(resolver, "authority_config", None)
        if config is None:
            config = _default_context_config_of(resolver)
        if config is None:
            raise TypeError(f"{label} resolver-only authority has no bound config; supply its Context or paired config")
        if not isinstance(config, ConfigManager):
            raise TypeError(f"{label} resolver config authority must be a ConfigManager")

    owner = getattr(config, "_context", None)
    if owner is not None:
        owner = context_or_default(owner)
        if owner.modules() is not resolver:
            raise ValueError(f"{label} resolver and config belong to different Contexts")
    bound_config = getattr(resolver, "authority_config", None)
    if bound_config is not None and bound_config is not config:
        raise ValueError(f"{label} resolver and config belong to different Contexts")
    return RuntimeAuthority(resolver, config)
