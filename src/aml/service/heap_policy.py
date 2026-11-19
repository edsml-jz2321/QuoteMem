"""The service process's garbage-collection policy (C.14; F.7 W02).

``serve_bench`` applies it in two steps, once per process:

- Before the runtime starts, ``tune_collector`` raises the young-generation
  threshold from 700 to ``YOUNG_THRESHOLD`` new tracked objects, so short-lived
  objects mostly die before a collection, and lowers the full-collection
  threshold to ``OLD_THRESHOLD`` middle collections, so a full collection is
  considered at every middle collection (CPython still runs it only once the
  objects promoted since the previous one reach a quarter of its survivors).
- Once the runtime has started (the workspace is open and recovered) and before
  the listener admits a request, ``freeze_started_heap`` runs a full collection
  and moves everything still alive to the permanent generation: the loaded
  application and the open workspace's runtime, which live as long as the
  service. Later full collections no longer traverse them.

Nothing is frozen after that. Freezing the survivors of every later full
collection hid every reference cycle that was alive during one, and under
continuous Add load that is each in-flight owner's working state: once it died
it stayed resident until a thaw, and a thaw waited for a quiet service or for
two hundred full collections. A rehearsal process grew by about 35 to 50 MB per
Add and a 32 GB host ran out of memory (LEAK, 2026-10-28). Cycles that die are
now reclaimed by the next full collection.

``limit_malloc_arenas`` caps glibc's malloc arenas before any worker thread
allocates (``MALLOC_ARENAS``).

A second service lifetime in the same process changes nothing. No callback is
installed and no lock is taken while collecting: collection timing never
decides a storage effect.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import threading

YOUNG_THRESHOLD = 10_000
"""New tracked objects between young collections (CPython's default is 700)."""

OLD_THRESHOLD = 1
"""Middle collections between full-collection checks (CPython's default is 10)."""

MALLOC_ARENAS = 2
"""glibc malloc arenas a service process may use (``M_ARENA_MAX``).

glibc gives each contending thread its own arena, up to eight per core, and an
arena keeps the memory its thread freed. A service's worker threads take turns
under the interpreter lock, so they gain nothing from separate arenas, while
every arena holds its own share of the decoded rows, vectors and bodies that
the bounded caches recycle. A single-user LoCoMo chain grew to 3.8 GiB RSS by
its 100th Add with the default arenas; with two it levelled off at about
1.8 GiB once the caches were full (Add 70), at the same CPU per Add. One
process per shard makes this the difference between fitting 16 shards on a
host or not.
"""

_M_ARENA_MAX = -8
"""glibc's ``mallopt`` parameter number for ``M_ARENA_MAX``."""

_done: set[str] = set()
_once = threading.Lock()


def _first(step: str) -> bool:
    with _once:
        if step in _done:
            return False
        _done.add(step)
        return True


def limit_malloc_arenas() -> bool:
    """Cap glibc's malloc arenas at ``MALLOC_ARENAS``, once per process, before workers start.

    An explicit ``MALLOC_ARENA_MAX`` in the environment wins, and a platform
    without glibc's ``mallopt`` is left unchanged. Arenas already created stay.

    Returns:
        bool: True when this call set the cap.
    """
    if not _first("arenas") or not sys.platform.startswith("linux") or (
        "MALLOC_ARENA_MAX" in os.environ
    ):
        return False
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except (OSError, AttributeError):
        return False
    return bool(mallopt(ctypes.c_int(_M_ARENA_MAX), ctypes.c_int(MALLOC_ARENAS)))


def tune_collector() -> bool:
    """Raise the young and lower the full-collection threshold, once per process.

    Returns:
        bool: True when this call tuned them; False when an earlier call had.
    """
    if not _first("tune"):
        return False
    _young, middle, _old = gc.get_threshold()
    gc.set_threshold(YOUNG_THRESHOLD, middle, OLD_THRESHOLD)
    return True


def freeze_started_heap() -> bool:
    """Collect and freeze the started heap, once per process.

    Only what is alive before the first request is frozen; everything the
    service builds afterwards stays collectable.

    Returns:
        bool: True when this call froze it; False when an earlier call had.
    """
    if not _first("freeze"):
        return False
    gc.collect()
    gc.freeze()
    return True
