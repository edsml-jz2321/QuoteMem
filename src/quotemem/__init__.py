"""QuoteMem: auditable agent memory with quoted evidence, versions and forgetting."""

import os
import sys

__version__ = "0.3.2"

# HeavenBase, the store, is included in this package; `heavenbase` is imported from here.
_INCLUDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if _INCLUDED not in sys.path:
    sys.path.insert(0, _INCLUDED)
_loaded = getattr(sys.modules.get("heavenbase"), "__file__", None)
if _loaded and not os.path.normcase(os.path.abspath(_loaded)).startswith(
        os.path.normcase(_INCLUDED + os.sep)):
    raise ImportError(f"quotemem uses the HeavenBase included in it ({_INCLUDED}), but another "
                      f"one is already imported ({_loaded}); import quotemem first")
