"""The known warnings HeavenBase raises on every start, silenced at the command line.

None of them comes from a value QuoteMem passes, and none asks the user to do anything:

- omegaconf's deprecation of ``register_new_resolver``, which HeavenBase's config
  loader calls;
- HeavenBase's own packaged defaults (``heavenbase.*`` keys such as backend ports)
  giving its ``env`` resolver integer defaults;
- HeavenBase's notice that two QuoteMem columns, the control ``key`` and the recall
  ``trigger``, are SQL reserved words; HeavenBase quotes every column, and the names
  are part of every existing store.

Each filter names the module that raises the warning and its exact message, so any
other warning, QuoteMem's own included, still shows.
"""

from __future__ import annotations

import warnings
from typing import Final

KNOWN_DEPENDENCY_WARNINGS: Final[tuple[tuple[str, str], ...]] = (
    (r"heavenbase\.utils\.config\.interpolation\Z",
     r"register_new_resolver\(\) is deprecated"),
    (r"heavenbase\.utils\.config\.interpolation\Z",
     r"Resolver 'env' (parameter 'default'|return value) expected str, got int "
     r"at full key 'heavenbase\."),
    (r"heavenbase\.workspace\.runtime\Z",
     r"Backend '[^']+' \(sqlite\) maps SQL column for "
     r"'agent-memory-(control' 'key'|recall' 'trigger') to reserved provider name"),
)
"""(module, message) regular expressions of the known HeavenBase ``UserWarning`` s."""


def silence_known_dependency_warnings() -> None:
    """Ignore exactly the `KNOWN_DEPENDENCY_WARNINGS` in this process."""
    for module, message in KNOWN_DEPENDENCY_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=UserWarning, module=module)
