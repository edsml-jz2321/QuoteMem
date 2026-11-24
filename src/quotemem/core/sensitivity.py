"""The sensitive topic classes of memories: a content topic, never an identity."""

from __future__ import annotations

from typing import Literal

SensitivityClass = Literal[
    "health", "religion", "political_affiliation", "sexual_orientation",
    "gender_identity", "financial", "credentials",
]
"""The extract-v12 finding categories and stored sensitivity classes.

The first five are the protected classes (``core/sensitivity_rules.PROTECTED_CLASSES``).
"""
