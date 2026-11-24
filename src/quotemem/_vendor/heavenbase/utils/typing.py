"""Small typing and dataclass re-export surface used during utility migration."""

from __future__ import annotations

__all__ = [
    "ABC",
    "Any",
    "AsyncGenerator",
    "Callable",
    "ClassVar",
    "Dict",
    "FrozenSet",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "Mapping",
    "MutableMapping",
    "Optional",
    "Protocol",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "Union",
    "abstractmethod",
    "dataclass",
    "datetime",
    "deepcopy",
    "defaultdict",
    "field",
    "overload",
]

import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)
