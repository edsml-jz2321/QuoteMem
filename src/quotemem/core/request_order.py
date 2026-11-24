"""Content and commit coordinates for deterministic Add request tie ordering."""

from __future__ import annotations

from quotemem.core.add_read_view import (
    EntityNameRecord,
    EntityRecord,
    Sigma,
    TombstoneRecord,
    VersionRecord,
)

VersionOrder = tuple[int, Sigma, str, int]
"""Commit sequence, source chronology, statement and version; no minted identity."""


def version_order(version: VersionRecord) -> VersionOrder:
    """Return a revision's stable tie order, independently of its minted ids."""
    return version.commit_seq, version.sigma, version.snapshot.statement, version.version


def name_order(row: EntityNameRecord) -> tuple[int, int, str, str, int]:
    """Order readable names by kind, newest commit, normalized spelling and annotation."""
    return ("name", "alias", "former").index(row.name_kind), -row.commit_seq, (
        row.name_norm), row.name, row.annotation_index


def entity_order(record: EntityRecord) -> tuple[int, tuple[str, ...], str]:
    """Order entity ties by commit, readable names and type, never minted ids."""
    return (record.commit_seq,
            tuple(row.name_norm for row in sorted(record.names, key=name_order)),
            record.entity_type)


def tombstone_order(record: TombstoneRecord) -> tuple[int, str, str]:
    """Order forgotten targets by their durable sequence and stated content."""
    return record.tombstone_seq, record.statement, record.kind
