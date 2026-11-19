"""Cross-row proofs the single Writer applies to NL rows before any write (A.3, A.4).

S1b's row models check each row alone; the Writer's reference validation checks
that every reference exists and stays in its scope. These proofs check a written
row against the rows it depends on: the batch's post-images first, then the
stored rows. They decide no semantics. Which rows a transaction writes belongs
to its producer: S5b's commit and rejection operations, S6a's forget, S6b's
privacy erase and S8b's Entity SDK. The proofs only refuse rows whose shape
disagrees with their authorities:

- an Item agrees with its head revision (DM-02) and changes lifecycle only as
  A.3.1 allows;
- a new revision is its Item's new head in the same batch, one version after
  the previous head; unless it is erased, it anchors only the user or active
  entities of its scope (DM-03) and has a non-zero vector in the workspace's
  space and dimension (DM-09);
- a link is written with its source revision and never on an erased version,
  and a revision's ``about_entity`` links are exactly its snapshot's
  ``about_entity_ids`` (DM-04);
- a candidate changes only while pending, and a published one names the
  revision it wrote;
- an entity keeps its identity, and an erased one is terminal (A.3.6);
- an ``entity_name`` row is one annotation of an active Item's head (DM-05);
- Evidence keeps its bytes and bindings, and its withheld spans only grow
  while it is active (A.3.5, DM-07);
- a deletion leaves every surviving head with its about links and, while the
  Item is active, its names (DM-04, DM-05).

Every function reads plain JSON rows through the given lookups and returns
nothing; a failed proof raises `AMLError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from aml.core.entities import USER
from aml.core.errors import AMLError

RowLookup = Callable[[str, str], Mapping[str, Any] | None]
"""Return the row of a kind and identity, or None when it does not exist."""

MEMORY_ROW_KINDS: Final = frozenset({
    "item", "revision", "link", "candidate", "entity", "entity_name",
})
"""The NL kinds whose rows `check_row` proves against other rows."""

_ENDED: Final = frozenset({"superseded", "corrected", "retracted"})
_ITEM_IDENTITY: Final = ("scope_kind", "scope_id", "created_at")
_CANDIDATE_IDENTITY: Final = (
    "scope_kind", "scope_id", "owner_run_id", "unit_no", "position", "extraction_ref",
    "proposal_ref", "statement_digest", "created_commit_seq", "created_at",
)
_ENTITY_IDENTITY: Final = ("scope_kind", "scope_id", "mint_key", "created_by_run_id", "created_at")
_EVIDENCE_BYTES: Final = ("text", "content_hash", "embedding")


@dataclass(frozen=True, slots=True)
class VectorSpace:
    """The workspace's active embedding space and dimension (DM-09).

    Attributes:
        dimension (int): The workspace dimension, 1024 or 2048.
        embedding_space (str): The active ``embedding_space`` every vector names.
    """

    dimension: int
    embedding_space: str


def activation_mode(snapshot: Mapping[str, Any]) -> str:
    """Return the activation mode a non-erased head gives its Item (A.3.1).

    Args:
        snapshot (Mapping[str, Any]): The head's ``MemorySnapshotV2`` JSON.

    Returns:
        str: ``on_demand`` when the head has a sensitivity class, is
            ``restricted`` or is inferred or untrusted; else ``contextual``.
    """
    on_demand = (
        bool(snapshot["sensitivity_classes"])
        or snapshot["security_level"] == "restricted"
        or snapshot["trust_tier"] in {"inferred", "untrusted"}
    )
    return "on_demand" if on_demand else "contextual"


def check_row(
    kind: str,
    row: Mapping[str, Any],
    *,
    latest: RowLookup,
    stored: RowLookup,
    batch: Mapping[tuple[str, str], Mapping[str, Any]],
    space: Callable[[], VectorSpace],
) -> None:
    """Prove one NL row of a write batch against the rows it depends on.

    Args:
        kind (str): One of `MEMORY_ROW_KINDS`.
        row (Mapping[str, Any]): The row's strict JSON post-image.
        latest (RowLookup): The batch's post-image of a row, else its stored row.
        stored (RowLookup): The stored row, before this batch.
        batch (Mapping[tuple[str, str], Mapping[str, Any]]): Every post-image of
            the batch by kind and identity.
        space (Callable[[], VectorSpace]): Return the active embedding space
            and dimension; called only for a version that is not erased, so a
            batch without a vector never needs one.

    Raises:
        AMLError: ``binding_mismatch`` when a row loses the authority it names,
            ``parity_mismatch`` when a derived value or a transition disagrees,
            ``immutable_field`` when a fixed field changes, ``schema_invalid``
            when a version that is not erased has a zero vector, or an
            embedding code when a vector leaves the active space.
    """
    if kind == "item":
        _check_item(row, latest=latest, stored=stored)
    elif kind == "revision":
        _check_revision(row, latest=latest, batch=batch, space=space)
    elif kind == "link":
        _check_link(row, batch=batch)
    elif kind == "candidate":
        _check_candidate(row, latest=latest, stored=stored)
    elif kind == "entity":
        _check_entity(row, stored=stored)
    elif kind == "entity_name":
        _check_entity_name(row, latest=latest)
    else:
        raise AMLError("unsupported_variant", "No cross-row proof exists for this kind")


def check_batch(rows: Iterable[tuple[str, Mapping[str, Any]]]) -> None:
    """Prove what only the whole batch shows: each revision's about links (DM-04).

    Args:
        rows (Iterable[tuple[str, Mapping[str, Any]]]): The batch's post-images
            with their kinds.

    Raises:
        AMLError: ``parity_mismatch`` when a revision written in the batch does
            not carry exactly one ``about_entity`` link per ``about_entity_ids``.
    """
    revisions: dict[str, Mapping[str, Any]] = {}
    about: dict[str, set[str]] = {}
    for kind, row in rows:
        if kind == "revision":
            revisions[row["object_id"]] = row
        elif kind == "link" and row["link_type"] == "about_entity":
            about.setdefault(row["source_revision_id"], set()).add(row["target_key"])
    for identity, revision in revisions.items():
        if about.get(identity, set()) != set(revision["snapshot"]["about_entity_ids"]):
            raise AMLError("parity_mismatch", "A revision's about links differ from its snapshot")


def check_deletion(kind: str, row: Mapping[str, Any], *, latest: RowLookup) -> None:
    """Prove that deleting a stored row leaves its surviving head whole (DM-04, DM-05).

    An ``about_entity`` link of a head that stays an Item's head, and a name
    of a head that stays an active Item's head, are deleted only with that
    head's change or the Item's deletion. Every other deletion is its
    producer's (forget's retention unit, a re-derivation).

    Args:
        kind (str): The deleted row's kind.
        row (Mapping[str, Any]): The deleted row's stored image.
        latest (RowLookup): The transaction's post-image of a row, else its
            stored row; None for a row the transaction deletes.

    Raises:
        AMLError: ``parity_mismatch`` when a surviving head would lose one of
            its about links or one of its names.
    """
    if kind == "link" and row["link_type"] == "about_entity":
        source = latest("revision", row["source_revision_id"])
        item = None if source is None else latest("item", source["memory_id"])
        if item is not None and item["current_revision_id"] == row["source_revision_id"]:
            raise AMLError("parity_mismatch", "A surviving head keeps its about links (DM-04)")
    elif kind == "entity_name":
        item = latest("item", row["naming_memory_id"])
        if item is not None and item["lifecycle_state"] == "active" and (
            item["current_revision_id"] == row["naming_revision_id"]
        ):
            raise AMLError("parity_mismatch", "A surviving active head keeps its names (DM-05)")


def check_evidence_change(old: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    """Prove a changed Evidence row against its stored row (A.3.5, DM-07).

    Erasure keeps today's placeholders and clears bindings and spans; every
    other change keeps the bindings set at reception, and while the part is
    active it keeps its bytes and only adds withheld ranges, each stored range
    staying inside a new one.

    Args:
        old (Mapping[str, Any]): The stored Evidence row.
        row (Mapping[str, Any]): Its post-image in the batch.

    Raises:
        AMLError: ``immutable_field`` when bindings or bytes change, or
            ``parity_mismatch`` when a withheld range is dropped.
    """
    if row["state"] == "erased":
        return
    if row["entity_bindings"] != old["entity_bindings"]:
        raise AMLError("immutable_field", "Evidence bindings are fixed at reception")
    if old["state"] != "active":
        return
    if any(row[name] != old[name] for name in _EVIDENCE_BYTES):
        raise AMLError("immutable_field", "Withholding never changes Evidence bytes")
    current = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
    for span in old["withheld_spans"]:
        if not any(
            start <= span["byte_start"] and span["byte_end"] <= end for start, end in current
        ):
            raise AMLError("parity_mismatch", "Withheld spans only grow while Evidence is active")


def _check_item(row: Mapping[str, Any], *, latest: RowLookup, stored: RowLookup) -> None:
    head = latest("revision", row["current_revision_id"])
    if head is None or head["memory_id"] != row["object_id"]:
        raise AMLError("binding_mismatch", "An Item's head is a revision of that Item")
    snapshot = head["snapshot"]
    if (row["entity_id"], row["statement_digest"]) != (
        snapshot["entity_id"], snapshot["statement_digest"],
    ):
        raise AMLError("parity_mismatch", "Item keys differ from its head snapshot")
    if snapshot["statement"] and row["activation_mode"] != activation_mode(snapshot):
        raise AMLError("parity_mismatch", "Item activation differs from its head")
    old = stored("item", row["object_id"])
    if old is None:
        historical = head["reason_code"] == "historical_correction"
        if head["version"] != 1 or row["lifecycle_state"] != (
            "historical" if historical else "active"
        ):
            raise AMLError(
                "parity_mismatch",
                "A new Item is version 1, historical exactly for a correction of history",
            )
        return
    if any(old[name] != row[name] for name in _ITEM_IDENTITY):
        raise AMLError("immutable_field", "An Item keeps its scope and creation time")
    before, after = old["lifecycle_state"], row["lifecycle_state"]
    if before != after and (before != "active" or after not in _ENDED | {"erased"}):
        raise AMLError("parity_mismatch", "Only an active Item changes lifecycle (A.3.1)")
    if old["current_revision_id"] == row["current_revision_id"]:
        if after == "erased" and before != "erased":
            raise AMLError("parity_mismatch", "An Item is erased by its erase revision")
        return
    if before == "erased" or head["previous_revision_id"] != old["current_revision_id"]:
        raise AMLError("binding_mismatch", "A new head follows the Item's stored head")
    if head["op"] in {"merge", "revise"} and (before, after) != ("active", "active"):
        raise AMLError("parity_mismatch", "Merge and revise write only on an active head")
    if head["op"] == "erase" and after != ("erased" if before == "active" else before):
        raise AMLError("parity_mismatch", "An erase leaves an active Item erased, else unchanged")
    if head["op"] == "requote" and after != before:
        raise AMLError("parity_mismatch", "A requote never changes lifecycle")


def _check_revision(
    row: Mapping[str, Any], *, latest: RowLookup,
    batch: Mapping[tuple[str, str], Mapping[str, Any]], space: Callable[[], VectorSpace],
) -> None:
    item = batch.get(("item", row["memory_id"]))
    if item is None or item["current_revision_id"] != row["object_id"]:
        raise AMLError(
            "binding_mismatch", "A new revision is its Item's new head in the same batch",
        )
    previous = row["previous_revision_id"]
    if previous is not None:
        prior = latest("revision", previous)
        if prior is None or prior["memory_id"] != row["memory_id"] or (
            prior["version"] + 1 != row["version"]
        ):
            raise AMLError("binding_mismatch", "A revision follows its Item's previous version")
    snapshot = row["snapshot"]
    if not snapshot["statement"]:
        return
    for anchor in (snapshot["entity_id"], *snapshot["about_entity_ids"]):
        entity = None if anchor == USER else latest("entity", anchor)
        if anchor != USER and (
            entity is None or entity["state"] != "active"
            or (entity["scope_kind"], entity["scope_id"]) != (item["scope_kind"], item["scope_id"])
        ):
            raise AMLError(
                "binding_mismatch", "A version anchors the user or active same-scope entities",
            )
    active = space()
    if len(snapshot["embedding"]) != active.dimension:
        raise AMLError("embedding_dimension_mismatch", "Vector dimension differs from workspace")
    if not any(snapshot["embedding"]):
        raise AMLError("schema_invalid", "Only an erased version has a zero vector")
    if snapshot["embedding_space"] != active.embedding_space:
        raise AMLError("embedding_space_mismatch", "Vector belongs to a different embedding space")


def _check_link(
    row: Mapping[str, Any], *, batch: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    source = batch.get(("revision", row["source_revision_id"]))
    if source is None:
        raise AMLError("binding_mismatch", "A link is written with its source revision")
    snapshot = source["snapshot"]
    if not snapshot["statement"]:
        raise AMLError("parity_mismatch", "An erased version carries no link (A.3.3)")
    item = batch.get(("item", source["memory_id"]))
    link_type = row["link_type"]
    if link_type == "about_entity" and row["target_key"] not in snapshot["about_entity_ids"]:
        raise AMLError("parity_mismatch", "An about link projects one of its snapshot's anchors")
    if link_type == "corrects_history" and (
        item is None or item["lifecycle_state"] != "historical"
    ):
        raise AMLError("parity_mismatch", "Only a historical Item corrects history (C.9.5a)")
    if link_type in {"supersedes", "contradicts"} and row["target_key"] == source["memory_id"]:
        raise AMLError("parity_mismatch", "A link never targets its own Item")


def _check_candidate(row: Mapping[str, Any], *, latest: RowLookup, stored: RowLookup) -> None:
    old = stored("candidate", row["object_id"])
    if old is not None:
        if old["status"] != "pending":
            raise AMLError("immutable_field", "A settled candidate never changes (A.3.4)")
        if any(old[name] != row[name] for name in _CANDIDATE_IDENTITY):
            raise AMLError("immutable_field", "A candidate keeps its position and proposal")
        if row["reentry_count"] < old["reentry_count"]:
            raise AMLError("parity_mismatch", "reentry_count never decreases (C.3)")
    if row["status"] == "published":
        revision = latest("revision", row["revision_id"])
        if revision is None or revision["source_candidate_id"] != row["object_id"] or (
            revision["memory_id"] != row["memory_id"]
        ):
            raise AMLError("binding_mismatch", "A published candidate names the revision it wrote")


def _check_entity(row: Mapping[str, Any], *, stored: RowLookup) -> None:
    old = stored("entity", row["object_id"])
    if old is None:
        if row["state"] != "active":
            raise AMLError("parity_mismatch", "A new entity is active (A.3.6)")
        return
    if old["state"] == "erased":
        raise AMLError("immutable_field", "An erased entity is terminal (A.3.6)")
    if any(old[name] != row[name] for name in _ENTITY_IDENTITY) or (
        row["state"] == "active" and row["participant_key"] != old["participant_key"]
    ):
        raise AMLError("immutable_field", "An entity keeps its scope, keys and creator")


def _check_entity_name(row: Mapping[str, Any], *, latest: RowLookup) -> None:
    item = latest("item", row["naming_memory_id"])
    revision = latest("revision", row["naming_revision_id"])
    if item is None or revision is None or revision["memory_id"] != row["naming_memory_id"] or (
        item["current_revision_id"] != row["naming_revision_id"]
        or item["lifecycle_state"] != "active"
    ):
        raise AMLError("binding_mismatch", "A name derives from an active Item's head (A.3.7)")
    naming = revision["snapshot"]["naming"]
    index = row["annotation_index"]
    if index >= len(naming) or (
        naming[index]["entity_id"], naming[index]["name"], naming[index]["name_kind"],
    ) != (row["entity_id"], row["name"], row["name_kind"]):
        raise AMLError("parity_mismatch", "A name row is one annotation of its head (DM-05)")


def row_anchors(kind: str, row: Mapping[str, Any]) -> frozenset[str]:
    """Return the entities one NL row is anchored to (C.14; INV:P12, I10).

    These key the Writer's entity generations: a write that touches the row
    invalidates every nomination cache of these entities. An Item is anchored
    by its primary entity, a version by its snapshot's anchors, an about link
    by its target, a name or a tombstone by its entity, a candidate by its
    primary entity and an Evidence part by its SDK bindings.

    Args:
        kind (str): The row's kind.
        row (Mapping[str, Any]): The row's strict JSON image.

    Returns:
        frozenset[str]: Its anchors, ``user`` included; empty for other kinds.
    """
    if kind == "item":
        anchors: set[str] = {row["entity_id"]}
    elif kind == "revision":
        snapshot = row["snapshot"]
        anchors = {snapshot["entity_id"], *snapshot["about_entity_ids"]}
    elif kind == "link":
        anchors = {row["target_key"]} if row["link_type"] == "about_entity" else set()
    elif kind == "entity":
        anchors = {row["object_id"]}
    elif kind in {"entity_name", "candidate"}:
        anchors = {row["entity_id"]}
    elif kind == "tombstone":
        anchors = {row["entity_id"], *row["about_entity_ids"]}
    elif kind == "evidence":
        anchors = {binding["entity_id"] for binding in row["entity_bindings"] or ()}
    else:
        anchors = set()
    return frozenset(anchor for anchor in anchors if anchor)
