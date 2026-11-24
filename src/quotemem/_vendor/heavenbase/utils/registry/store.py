"""Persist opaque Registry snapshots."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import threading
from typing import Protocol, cast, runtime_checkable

__all__ = ["RegistryStore", "SQLiteRegistryStore"]


@runtime_checkable
class RegistryStore(Protocol):
    """Persist complete Registry namespace snapshots with compare-and-set."""

    def prepare(self) -> None:
        """Idempotently prepare the store for Registry reads and writes.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        ...

    def read(self, namespace: str) -> tuple[int, bytes, str] | None:
        """Read one opaque Registry namespace snapshot.

        Args:
            namespace (str): Stable Registry namespace identifier.

        Returns:
            tuple[int, bytes, str] | None: Revision, canonical payload, and
            digest, or `None` when the namespace has not been written.
        """
        ...

    def compare_and_set(
        self,
        namespace: str,
        *,
        expected_revision: int,
        revision: int,
        payload: bytes,
        digest: str,
    ) -> bool:
        """Atomically publish a snapshot when its expected revision matches.

        Args:
            namespace (str): Stable Registry namespace identifier.
            expected_revision (int): Current revision expected by the writer.
                Zero represents an absent namespace.
            revision (int): New revision, which must be exactly one greater
                than `expected_revision`.
            payload (bytes): Complete canonical Registry snapshot.
            digest (str): Integrity digest for `payload` and `revision`.

        Returns:
            bool: Whether the snapshot was published.
        """
        ...


class SQLiteRegistryStore:
    """Persist Registry snapshots in one standard-library SQLite database.

    The store owns one SQLite connection. Multiple store instances may point
    at the same file and use SQLite transactions for cross-process CAS.

    Args:
        path (str | Path): SQLite database path or URI.
        timeout (float): Maximum seconds to wait for a locked database.
        uri (bool): Whether `path` is interpreted as a SQLite URI.

    Raises:
        ValueError: If `timeout` is not positive.
    """

    def __init__(self, path: str | Path, *, timeout: float = 5.0, uri: bool = False) -> None:
        """Open one SQLite-backed Registry snapshot store.

        Returns:
            None: This function does not return a value.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        self.path = os.fspath(path)
        self._connection = sqlite3.connect(self.path, timeout=timeout, isolation_level=None, check_same_thread=False, uri=uri)
        self._connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        self._lock = threading.RLock()
        self._closed = False

    def _need_open(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("SQLiteRegistryStore is closed")
        return self._connection

    def prepare(self) -> None:
        """Create the private snapshot table when it is absent.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        with self._lock:
            self._need_open().execute("""CREATE TABLE IF NOT EXISTS registry_state (
                    namespace TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    digest TEXT NOT NULL
                )""")

    def read(self, namespace: str) -> tuple[int, bytes, str] | None:
        """Read one Registry snapshot without starting a write transaction.

        Args:
            namespace (str): Stable Registry namespace identifier.

        Returns:
            tuple[int, bytes, str] | None: Stored revision, payload, and digest,
            or `None` when the namespace is absent.
        """
        with self._lock:
            row = (
                self._need_open()
                .execute(
                    "SELECT revision, payload, digest FROM registry_state WHERE namespace = ?",
                    (namespace,),
                )
                .fetchone()
            )
        if row is None:
            return None
        return cast(tuple[int, bytes, str], row)

    def compare_and_set(
        self,
        namespace: str,
        *,
        expected_revision: int,
        revision: int,
        payload: bytes,
        digest: str,
    ) -> bool:
        """Atomically publish one Registry snapshot revision.

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
        with self._lock:
            connection = self._need_open()
            connection.execute("BEGIN IMMEDIATE")
            try:
                if expected_revision == 0:
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO registry_state(namespace, revision, payload, digest) VALUES (?, ?, ?, ?)",
                        (namespace, revision, payload, digest),
                    )
                else:
                    cursor = connection.execute(
                        "UPDATE registry_state SET revision = ?, payload = ?, digest = ? WHERE namespace = ? AND revision = ?",
                        (revision, payload, digest, namespace, expected_revision),
                    )
                won = cursor.rowcount == 1
                if won:
                    connection.commit()
                else:
                    connection.rollback()
                return won
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def close(self) -> None:
        """Close the owned SQLite connection idempotently.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> SQLiteRegistryStore:
        self.prepare()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
