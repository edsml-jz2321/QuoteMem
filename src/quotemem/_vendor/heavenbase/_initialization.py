"""Opt-in explicit Context authority for embedded applications."""

from ._lazy import bind_late

explicit_context_required = False

# ``utils`` may not import this module, so hash_id reads the mode through a named hook. It is bound on import,
# and explicit mode can only be selected after this import, so an unbound hook means the mode is off.
bind_late("heavenbase.context.explicit_mode", __name__, "explicit_context_required")


def configure_explicit_context() -> None:
    """Reject process-default authority before it is selected or installed.

    Importing Context first is safe because its upstream default is lazy.
    Repeating this selection is safe. A default already in use cannot be
    replaced or silently retained by an explicit-only application.

    Args:
        None.

    Returns:
        None: Selects explicit Context authority for the process.

    Raises:
        RuntimeError: If a process default Context already exists.
    """
    from .context.runtime import _DEFAULT_CONTEXT_LOCK, _default_context_instance
    from .utils.config.manager import CM_HVNB

    global explicit_context_required
    with CM_HVNB._lock, _DEFAULT_CONTEXT_LOCK:
        if _default_context_instance() is not None:
            raise RuntimeError("Explicit Context mode must be selected before using a default Context")
        explicit_context_required = True


def require_default_context_allowed(*, installing: bool = False, installer: str = "set_default_context()") -> None:
    """Fail before default construction or installation in explicit-only mode.

    Args:
        installing (bool): Whether the caller installs an explicit default
            (``set_default_context``) rather than building one.
        installer (str): The call that installs, named in the error when
            ``installing`` is true.

    Returns:
        None: Allows default Context selection when explicit mode is disabled.

    Raises:
        RuntimeError: If explicit Context authority has been selected.
    """
    if not explicit_context_required:
        return
    if installing:
        raise RuntimeError(f"Explicit Context mode is on: {installer} cannot install a process default Context")
    raise RuntimeError(
        "Explicit Context mode is on, but this call needed the process default Context: an omitted context=, "
        "hb.DEFAULT_CONTEXT or default_context(), or an omitted config= that fell back to CM_HVNB. "
        "Pass context=, or a config= bound to your Context, explicitly"
    )
