"""Public configuration surface for HeavenBase."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "CM_HVNB",
    "ConfigEngine",
    "ConfigManager",
    "ConfigSnapshot",
    "ConfigSpec",
    "InterpolationPolicy",
    "copy_data",
    "dflat",
    "dget",
    "dmerge",
    "dset",
    "dsetdef",
    "dunflat",
    "dunset",
    "resolve_interpolations",
    "to_mutable_data",
]

_LAZY_EXPORTS = {
    "copy_data": (".data", "copy_data"),
    "dflat": (".data", "dflat"),
    "dget": (".data", "dget"),
    "dmerge": (".data", "dmerge"),
    "dset": (".data", "dset"),
    "dsetdef": (".data", "dsetdef"),
    "dunflat": (".data", "dunflat"),
    "dunset": (".data", "dunset"),
    "to_mutable_data": (".data", "to_mutable_data"),
    "ConfigEngine": (".engine", "ConfigEngine"),
    "InterpolationPolicy": (".interpolation", "InterpolationPolicy"),
    "CM_HVNB": (".manager", "CM_HVNB"),
    "ConfigManager": (".manager", "ConfigManager"),
    "ConfigSnapshot": (".snapshot", "ConfigSnapshot"),
    "ConfigSpec": (".spec", "ConfigSpec"),
    "resolve_interpolations": (".spec", "resolve_interpolations"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
