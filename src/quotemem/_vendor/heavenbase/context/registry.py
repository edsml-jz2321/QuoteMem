"""Adapt a Context-owned HeavenBase Backend to standalone Registry snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import nullcontext
from typing import Any, Callable, ContextManager

from heavenbase.utils.registry import RegistryIntegrityError

from heavenbase.execution.contracts import RowOp
from heavenbase.schema.model import EntitySchema, FieldSchema
from heavenbase.placement import StorageBinding
from heavenbase.strategies import InlineColumn
from heavenbase.types import Artifact, Identifier, Integer, ShortText

_ENTITY_ID = "sys-registry-state"


class _RegistryIdentifier(Identifier):
    """Identifier carrier whose DDL is independent of ConfigManager."""

    default_max_chars = 255


class _RegistryShortText(ShortText):
    """Short-text carrier whose DDL is independent of ConfigManager."""

    default_max_chars = 255


def _schema() -> EntitySchema:
    return EntitySchema(
        _ENTITY_ID,
        "RegistryState",
        tags=["system", "control-plane"],
        fields={
            "object_id": FieldSchema("object_id", _RegistryIdentifier(), pk=True, required=True),
            "revision": FieldSchema("revision", Integer(), required=True),
            "payload": FieldSchema("payload", Artifact(), required=True),
            "digest": FieldSchema("digest", _RegistryShortText(), required=False),
            "updated_at": FieldSchema("updated_at", _RegistryShortText(), required=True),
        },
    )


def _bindings(backend_name: str) -> list[StorageBinding]:
    return [StorageBinding(_ENTITY_ID, field_name, backend_name, InlineColumn, "registry") for field_name in _schema().fields]


class HeavenBaseRegistryStore:
    """Adapt one caller-owned HeavenBase Backend to `RegistryStore`.

    The adapter preserves the existing private `sys-registry-state` rows and
    never constructs or disposes a Backend. Context owns that lifecycle.

    Args:
        backend (Any): Existing HeavenBase Backend with atomic `compare_upsert`.
        control (Callable[[], ContextManager[Any]] | None): Optional Context
            seam used while the Backend persists Registry state.
    """

    def __init__(self, backend: Any, *, control: Callable[[], ContextManager[Any]] | None = None) -> None:
        """Bind one caller-owned Backend to Registry persistence.

        Returns:
            None: This initializer does not return a value.
        """
        self.backend = backend
        self._control = control or nullcontext

    def prepare(self) -> None:
        """Ensure the private Registry state entity exists.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        with self._control():
            self.backend.ensure(_schema(), _bindings(self.backend.name))

    def read(self, namespace: str) -> tuple[int, bytes, str] | None:
        """Read one existing private Registry state row.

        Args:
            namespace (str): Stable Registry namespace identifier.

        Returns:
            tuple[int, bytes, str] | None: Revision, opaque payload, and digest,
            or `None` when no snapshot has been published.

        Raises:
            RegistryIntegrityError: If the Backend returns a different row id.
        """
        with self._control():
            row = self.backend.get_many(_ENTITY_ID, [namespace])[0]
        if row is None:
            return None
        if row.get("object_id") != namespace:
            raise RegistryIntegrityError(f"Registry row identity {row.get('object_id')!r} does not match {namespace!r}")
        return row.get("revision"), row.get("payload"), row.get("digest")  # type: ignore[return-value]

    def compare_and_set(
        self,
        namespace: str,
        *,
        expected_revision: int,
        revision: int,
        payload: bytes,
        digest: str,
    ) -> bool:
        """Publish one private Registry state row with Backend CAS.

        Args:
            namespace (str): Stable Registry namespace identifier.
            expected_revision (int): Current revision, or zero when absent.
            revision (int): New revision, exactly one above the expected value.
            payload (bytes): Complete canonical Registry snapshot.
            digest (str): Integrity digest for the snapshot.

        Returns:
            bool: Whether the expected revision matched and was replaced.

        Raises:
            ValueError: If the revision transition is invalid.
        """
        if revision != expected_revision + 1 or expected_revision < 0:
            raise ValueError(f"Invalid Registry revision transition {expected_revision} -> {revision}")
        row = RowOp(
            _ENTITY_ID,
            namespace,
            {
                "revision": revision,
                "payload": payload,
                "digest": digest,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        with self._control():
            return self.backend.compare_upsert(
                row,
                revision_field="revision",
                expected_revision=None if expected_revision == 0 else expected_revision,
            )


__all__ = ["HeavenBaseRegistryStore"]
