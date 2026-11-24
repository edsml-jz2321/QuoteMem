"""HeavenBase package version.

Schema: ``MAJOR.MINOR.PATCH.N[devK]``.

- ``MAJOR.MINOR.PATCH``: release train (currently ``0.1.2``).
- ``N``: very small frequent updates (fourth integer segment).
- Optional ``devK`` suffix (``dev0``, ``dev1``, …) for in-development snapshots; omit for stabilized releases.

The ``heaven-style`` skill follows the same schema; on HeavenBase-aligned releases
its ``metadata.version`` matches ``__version__`` here. Between releases the skill may
lead with skill-only ``N`` (and optional ``devK``) bumps on the same train.
"""

__version__ = "0.1.2.3"
