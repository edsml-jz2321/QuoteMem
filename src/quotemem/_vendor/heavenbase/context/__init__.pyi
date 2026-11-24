from .._initialization import configure_explicit_context as configure_explicit_context
from .bootstrap import BootstrapSpec as BootstrapSpec
from .bootstrap import load_bootstrap as load_bootstrap
from .runtime import Context as Context
from .runtime import context_or_default as context_or_default
from .runtime import default_context as default_context
from .runtime import set_default_context as set_default_context

# Attribute only, not in __all__: a star import must not build the process default Context.
DEFAULT_CONTEXT: Context

__all__: list[str]
