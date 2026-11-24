"""The Entity SDK: create, read, find, list and retype a scope's entities.

Trusted callers create an entity before they name it: a client names an
entity by Adding a message whose name span carries an `EntityBindingV1`
anchor, so ``create`` writes no names. Names are derived from naming
memories and returned only while they pass ``p_name``. ``find`` returns
the active entities with a readable name of that ``name_norm``: a candidate
list, never proof of identity. There is no delete, merge or rename,
and a participant's key is never exposed.

Every write is one ``entity_update`` run admitted with its ``entity_update:sdk``
input: idempotent per (principal, scope, request_id), refused as
``operation_conflict`` when the same request id carries another body, and
committed through the single Writer's NL (natural-language) transaction. ``set_type`` is a
compare-and-set on the observed type, allowed only for entities an SDK run
created; types are stored, so a type naming a sensitive topic is
stored as ``entity``.

Reads go through the guarded memory-read entry: an absent, erased or
other-scope id is refused with ``scope_denied``, and an unfinished forget
closes the scope.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Final, Literal

from pydantic import Field, TypeAdapter, ValidationError

from quotemem.core.admission import admission_identity, canonicalize_admission
from quotemem.core.entities import USER, EntityRef, NameKind
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ContentBindingDigest, new_ulid
from quotemem.core.query import ScopeV1
from quotemem.core.run_inputs import (
    EntityCreateV1,
    EntitySetTypeV1,
    EntityUpdateKeyV1,
    EntityUpdateSdkRunInputV1,
)
from quotemem.core.semantic import SystemTimestamp
from quotemem.core.sensitivity_rules import stored_entity_type as stored_type
from quotemem.core.statement import is_name, name_norm
from quotemem.core.technical_types import ExternalId
from quotemem.ingestion.authorization import check_add_grant
from quotemem.providers.auth import AuthRequest, PrincipalGrant
from quotemem.publication.history import MemoryReads, MemoryRows, cite
from quotemem.store.domain import DOMAIN_ADAPTERS, EntityNameRowV1, EntityRowV1
from quotemem.store.indexed import EntityNameRows, EntityRows
from quotemem.store.metadata import StrictModel
from quotemem.store.writer import RowWrite, WriteTicket

MAX_ENTITY_NAMES: Final = 32
"""The most names one entity view returns."""

MAX_FIND: Final = 64
MAX_LIST: Final = 128
_NAME_ORDER: Final[dict[NameKind, int]] = {"name": 0, "alias": 1, "former": 2}
_ULID: Final[TypeAdapter[str]] = TypeAdapter(ULID)
_ENTITY_REF: Final[TypeAdapter[str]] = TypeAdapter(EntityRef)
_EXTERNAL: Final[TypeAdapter[str]] = TypeAdapter(ExternalId)
_ENTITY: Final[TypeAdapter[EntityRowV1]] = TypeAdapter(EntityRowV1)
_NAME: Final[TypeAdapter[EntityNameRowV1]] = TypeAdapter(EntityNameRowV1)


class EntityNameV1(StrictModel):
    """One readable name of an entity and the memory that names it."""

    name: str
    kind: NameKind
    naming_memory_id: ULID


class EntityRecordV1(StrictModel):
    """One active entity of an exact scope; ``user`` is a virtual record of type ``user``."""

    entity_id: EntityRef
    scope: ScopeV1
    entity_type: str
    state: Literal["active"]
    created_at: SystemTimestamp | None
    updated_at: SystemTimestamp | None


class EntityViewV1(StrictModel):
    """An entity with its ``p_name`` names: name < alias < former, then ``name_norm``; ≤ 32."""

    entity: EntityRecordV1
    names: Annotated[tuple[EntityNameV1, ...], Field(max_length=MAX_ENTITY_NAMES)]


def stored_entity_type(text: str) -> str:
    """Return the stored form of an SDK entity type.

    One rule serves extraction and the SDK (``core.sensitivity_rules``); the SDK
    only maps its refusal to the caller's error.

    Args:
        text (str): The caller's type.

    Returns:
        str: The type after NFC and trimming, or ``entity`` when it names a
            sensitive topic.

    Raises:
        QuoteMemError: ``invalid_request`` for a type outside 1..64 characters.
    """
    try:
        return stored_type(text)
    except (TypeError, ValueError):
        raise QuoteMemError("invalid_request", "An entity type has 1 to 64 characters") from None


def _user(scope: ScopeV1) -> EntityRecordV1:
    return EntityRecordV1(
        entity_id=USER, scope=scope, entity_type="user", state="active", created_at=None,
        updated_at=None,
    )


def _record(scope: ScopeV1, row: EntityRowV1) -> EntityRecordV1:
    return EntityRecordV1(
        entity_id=row.object_id, scope=scope, entity_type=row.entity_type, state="active",
        created_at=row.created_at, updated_at=row.updated_at,
    )


def _cite_names(rows: MemoryRows, names: Sequence[EntityNameRowV1]) -> None:
    """Record every naming memory, head and Evidence that ``p_name`` reads for ``names``."""
    for row in names:
        item = rows.memory(row.naming_memory_id)
        if item is not None:
            cite(rows, rows.head(item).snapshot)


def _readable(rows: MemoryRows, names: Sequence[EntityNameRowV1]) -> list[EntityNameRowV1]:
    """Keep the name rows that pass ``p_name`` at the read clock; their reads are recorded."""
    read = rows.readability(rows.clock())
    kept: list[EntityNameRowV1] = []
    for row in names:
        item = rows.memory(row.naming_memory_id)
        if read.p_name(row, item, None if item is None else rows.head(item).snapshot):
            kept.append(row)
    return kept


def _view(record: EntityRecordV1, names: Sequence[EntityNameRowV1]) -> EntityViewV1:
    ordered = sorted(
        names, key=lambda row: (_NAME_ORDER[row.name_kind], row.name_norm, row.object_id),
    )
    return EntityViewV1(entity=record, names=tuple(
        EntityNameV1(name=row.name, kind=row.name_kind, naming_memory_id=row.naming_memory_id)
        for row in ordered[:MAX_ENTITY_NAMES]
    ))


def _names_of(rows: MemoryRows, entity_id: str) -> list[EntityNameRowV1]:
    query = EntityNameRows(
        scope_kind=rows.scope.scope_kind, scope_id=rows.scope.scope_id, entity_id=entity_id,
    )
    return [_NAME.validate_python(raw) for raw in rows.page(query)]


def _active(rows: MemoryRows, entity_id: str) -> EntityRowV1 | None:
    raw = rows.row("entity", entity_id)
    if raw is None:
        return None
    entity = _ENTITY.validate_python(raw)
    return entity if entity.state == "active" else None


class EntityService:
    """The Entity SDK of one QuoteMem service.

    Args:
        reads (MemoryReads): The guarded memory-read entry over the sole
            Writer, its registered read model, the authorization provider and
            the trusted clock; writes use the same owners.
    """

    def __init__(self, reads: MemoryReads) -> None:
        self._reads = reads

    def _read[T](self, scope: ScopeV1, read: Callable[[MemoryRows], T]) -> T:
        found: list[T] = []
        self._reads.consume(scope, read, found.append)
        return found[0]

    def get(self, *, scope: ScopeV1, entity_id: str) -> EntityViewV1:
        """Return one active entity of the exact scope with its readable names.

        Args:
            scope (ScopeV1): The caller's exact scope.
            entity_id (str): An entity ULID, or ``user``.

        Returns:
            EntityViewV1: The entity and its ``p_name`` names.

        Raises:
            QuoteMemError: ``invalid_request`` for a malformed id; ``scope_denied``
                for an absent, erased or other-scope id, with identical bytes.
        """
        try:
            selected = _ENTITY_REF.validate_python(entity_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request", "An entity id is a ULID or user") from None

        def read(rows: MemoryRows) -> EntityViewV1:
            rows.authorize()
            if selected == USER:
                record = _user(rows.scope)
            else:
                entity = _active(rows, selected)
                if entity is None:
                    raise QuoteMemError("scope_denied", "Entity is unavailable")
                record = _record(rows.scope, entity)
            names = _names_of(rows, selected)
            _cite_names(rows, names)
            rows.finish()
            return _view(record, _readable(rows, names))

        return self._read(scope, read)

    def find(self, *, scope: ScopeV1, name: str, limit: int = 16) -> tuple[EntityViewV1, ...]:
        """Return the active entities with a readable name of this ``name_norm``, oldest first.

        A match is a candidate list, never proof that two mentions are one entity.

        Args:
            scope (ScopeV1): The caller's exact scope.
            name (str): A valid name.
            limit (int): The most entities, 1 through 64.

        Returns:
            tuple[EntityViewV1, ...]: The entities by ``created_at``, ``user`` first.

        Raises:
            QuoteMemError: ``invalid_request`` for an invalid name or limit.
        """
        if not is_name(name):
            raise QuoteMemError("invalid_request", "find needs one valid name")
        if type(limit) is not int or not 1 <= limit <= MAX_FIND:
            raise QuoteMemError("invalid_request", "find limit is outside 1 to 64")
        norm = name_norm(name)

        def read(rows: MemoryRows) -> tuple[EntityViewV1, ...]:
            rows.authorize()
            query = EntityNameRows(
                scope_kind=rows.scope.scope_kind, scope_id=rows.scope.scope_id,
                name_norms=(norm,),
            )
            matches = [_NAME.validate_python(raw) for raw in rows.page(query)]
            entities: dict[str, EntityRowV1 | None] = {}
            for match in matches:
                if match.entity_id not in entities:
                    entities[match.entity_id] = (
                        None if match.entity_id == USER else _active(rows, match.entity_id)
                    )
            candidates = {
                entity_id: _names_of(rows, entity_id) for entity_id, row in entities.items()
                if entity_id == USER or row is not None
            }
            for names in (matches, *candidates.values()):
                _cite_names(rows, names)
            rows.finish()
            found = {row.entity_id for row in _readable(rows, matches)}
            views: list[tuple[int, str, EntityViewV1]] = []
            for entity_id, names in candidates.items():
                if entity_id not in found:
                    continue
                entity = entities[entity_id]
                record = _user(rows.scope) if entity is None else _record(rows.scope, entity)
                views.append((
                    -1 if entity is None else entity.created_at, entity_id,
                    _view(record, _readable(rows, names)),
                ))
            return tuple(view for _, _, view in sorted(views, key=lambda found: found[:2]))[
                :limit
            ]

        return self._read(scope, read)

    def list(
        self, *, scope: ScopeV1, after: str | None = None, limit: int = MAX_LIST,
    ) -> tuple[EntityViewV1, ...]:
        """Return a page of the scope's active entities by id, ``user`` last.

        Args:
            scope (ScopeV1): The caller's exact scope.
            after (str | None): List only entities with a greater id.
            limit (int): The most entities, 1 through 128.

        Returns:
            tuple[EntityViewV1, ...]: The page.

        Raises:
            QuoteMemError: ``invalid_request`` for an invalid cursor or limit.
        """
        if after is not None:
            try:
                _ENTITY_REF.validate_python(after, strict=True)
            except ValidationError:
                raise QuoteMemError("invalid_request", "A list cursor is an entity id") from None
        if type(limit) is not int or not 1 <= limit <= MAX_LIST:
            raise QuoteMemError("invalid_request", "list limit is outside 1 to 128")

        def read(rows: MemoryRows) -> tuple[EntityViewV1, ...]:
            rows.authorize()
            page: builtins.list[EntityRowV1] = []
            if after != USER:
                query = EntityRows(
                    scope_kind=rows.scope.scope_kind, scope_id=rows.scope.scope_id,
                    state="active",
                )
                page = [
                    _ENTITY.validate_python(raw)
                    for raw in rows.page(query, after=after, limit=limit)
                ]
            with_user = len(page) < limit and after != USER
            names = {row.object_id: _names_of(rows, row.object_id) for row in page}
            if with_user:
                names[USER] = _names_of(rows, USER)
            for found in names.values():
                _cite_names(rows, found)
            rows.finish()
            views = [
                _view(_record(rows.scope, row), _readable(rows, names[row.object_id]))
                for row in page
            ]
            if with_user:
                views.append(_view(_user(rows.scope), _readable(rows, names[USER])))
            return tuple(views)

        return self._read(scope, read)

    def create(self, *, scope: ScopeV1, request_id: str, entity_type: str) -> EntityRecordV1:
        """Create one entity of the exact scope; it has no names.

        Args:
            scope (ScopeV1): The caller's exact, registered scope.
            request_id (str): Idempotency key per (principal, scope, request_id).
            entity_type (str): Free text, stored as ``stored_entity_type`` returns it.

        Returns:
            EntityRecordV1: The entity the request created, also on a repeat.

        Raises:
            QuoteMemError: ``invalid_request`` for an invalid id or type;
                ``operation_conflict`` when the request id carries another
                body; ``scope_denied`` for an unregistered or unauthorized
                scope; ``erasure_in_progress`` during an erasure.
        """
        request = EntityCreateV1(
            op="create", request_id=_request_id(request_id),
            entity_type=stored_entity_type(entity_type),
        )
        return self._update(scope, request)

    def set_type(
        self, *, scope: ScopeV1, request_id: str, entity_id: str, expected_type: str,
        entity_type: str,
    ) -> EntityRecordV1:
        """Retype one SDK entity while it still has the observed type.

        Args:
            scope (ScopeV1): The caller's exact scope.
            request_id (str): Idempotency key per (principal, scope, request_id).
            entity_id (str): An entity an ``entity_update`` run created.
            expected_type (str): The type the caller observed.
            entity_type (str): The new type, stored as ``stored_entity_type`` returns it.

        Returns:
            EntityRecordV1: The retyped entity, also on a repeat.

        Raises:
            QuoteMemError: ``invalid_request`` for ``user``, an invalid id or type;
                ``scope_denied`` for an absent, erased or other-scope entity;
                ``operation_conflict`` for a changed body under one request id,
                a type that is no longer ``expected_type`` or an entity no SDK
                run created.
        """
        if entity_id == USER:
            raise QuoteMemError("invalid_request", "The user's type cannot change")
        try:
            selected = _ULID.validate_python(entity_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request", "An entity id is a ULID") from None
        request = EntitySetTypeV1(
            op="set_type", request_id=_request_id(request_id), entity_id=selected,
            expected_type=stored_entity_type(expected_type),
            entity_type=stored_entity_type(entity_type),
        )
        return self._update(scope, request)

    def _update(self, scope: ScopeV1, request: EntityCreateV1 | EntitySetTypeV1) -> EntityRecordV1:
        """Admit, run and finish one ``entity_update`` run under the Writer."""
        reads = self._reads
        writer = reads.writer
        exact = ScopeV1.model_validate(scope.model_dump(mode="json"))
        grant = reads.auth.resolve(AuthRequest(requested_scope=exact, now_ms=reads.clock()))
        check_add_grant(grant, exact, reads.clock())
        admission = canonicalize_admission({
            "v": 1, "run_kind": "entity_update", "commit_mode": "single_compensatable",
            "principal_id": grant.principal_id, **exact.model_dump(mode="json"),
            "target_kind": "scope", "target_locator": exact.scope_id,
            "raw_business_key": EntityUpdateKeyV1(request_id=request.request_id).model_dump(),
            "request_payload": request.model_dump(mode="json"), "options": {},
        })
        family, digest, payload = admission_identity(admission, writer.bind_content)
        with writer.mutex:
            check_add_grant(grant, exact, reads.clock())
            run = writer.indexed_reader().latest_run("entity_update", family)
            if run is not None and run["admission_digest"] != digest:
                raise QuoteMemError("operation_conflict", "The request id carries another body")
            if run is not None and run["status"] == "succeeded":
                body = writer.read_logical(run["input_ref"], "run_input")
                if not isinstance(body, EntityUpdateSdkRunInputV1) or body.request != request:
                    raise QuoteMemError("binding_mismatch", "An entity update changed its input")
                return self._written(exact, body.entity_id)
            if run is None:
                run = self._admit(grant, exact, family, digest)
            ticket = writer.issue_ticket(run["object_id"])
            selected = self._input(ticket, run, request, digest, payload)
            entity = self._effect(ticket, exact, run, selected)
            now = reads.clock()
            current = writer.get_row("run", run["object_id"])
            if current is None:
                raise QuoteMemError("reference_missing", "An entity update lost its run")
            finished = {**current, "status": "succeeded", "updated_at": now, "finished_at": now}
            writer.write_bookkeeping(ticket, [RowWrite("run", _run(finished))])
            return _record(exact, entity)

    def _admit(
        self, grant: PrincipalGrant, scope: ScopeV1, family: str, digest: ContentBindingDigest,
    ) -> dict[str, Any]:
        """Admit the request's run, empty and queued, at the scope's current epochs."""
        writer = self._reads.writer
        lease = writer.get_control("lease:writer")
        if lease is None:
            raise QuoteMemError("lease_lost", "Entity updates need the active Writer")
        ticket = writer.admit_run(_run({
            "object_id": new_ulid(), "run_kind": "entity_update",
            "commit_mode": "single_compensatable", "status": "queued", "stage": "await_input",
            "attempt": 0, "idempotency_key": family, "run_generation": 0,
            "admission_digest": digest, "principal_id": grant.principal_id,
            "scope_kind": scope.scope_kind, "scope_id": scope.scope_id,
            "target_kind": "scope", "target_id": scope.scope_id,
            "writer_epoch": lease["value"]["writer_epoch"],
            "admitted_erase_epoch": writer.get_counter(f"erase_epoch:{scope.scope_id}"),
            "admitted_consent_epoch": writer.get_counter(f"consent_epoch:{scope.scope_id}"),
            "created_at": self._reads.clock(),
        }))
        run = writer.get_row("run", ticket.run_id or "")
        if run is None:
            raise QuoteMemError("reference_missing", "An admitted entity update was lost")
        return run

    def _input(
        self, ticket: WriteTicket, run: dict[str, Any],
        request: EntityCreateV1 | EntitySetTypeV1, digest: ContentBindingDigest,
        payload: ContentBindingDigest,
    ) -> EntityUpdateSdkRunInputV1:
        """Write the run's input once and attach it; on a repeat, read the attached one."""
        writer = self._reads.writer
        found = writer.find_logical_candidates(run["object_id"], "run_input")
        if len(found) > 1:
            raise QuoteMemError("binding_mismatch", "An entity update has two inputs")
        if found:
            reference, body = found[0]
            if not isinstance(body, EntityUpdateSdkRunInputV1) or body.request != request:
                raise QuoteMemError("binding_mismatch", "An entity update changed its input")
        else:
            body = EntityUpdateSdkRunInputV1(
                admission_digest=digest, payload_digest=payload, run_kind="entity_update",
                source="sdk", business_key=EntityUpdateKeyV1(request_id=request.request_id),
                request=request,
                entity_id=request.entity_id if isinstance(request, EntitySetTypeV1) else new_ulid(),
            )
            reference = writer.put_logical_artifact(
                ticket, "run_input", body, created_at=self._reads.clock(),
            )
        if run["input_ref"] is None:
            now = self._reads.clock()
            attached = {**run, "input_ref": reference, "status": "running", "stage": "commit",
                        "started_at": now, "updated_at": now}
            writer.write_bookkeeping(ticket, [RowWrite("run", _run(attached))])
        elif run["input_ref"] != reference:
            raise QuoteMemError("binding_mismatch", "An entity update attached another input")
        return body

    def _effect(
        self, ticket: WriteTicket, scope: ScopeV1, run: dict[str, Any],
        selected: EntityUpdateSdkRunInputV1,
    ) -> EntityRowV1:
        """Write the run's one entity row, unless an earlier attempt already did."""
        writer = self._reads.writer
        stored = writer.get_row("entity", selected.entity_id)
        request = selected.request
        if stored is not None and isinstance(request, EntityCreateV1):
            entity = _ENTITY.validate_python(stored)
            if entity.created_by_run_id != run["object_id"]:
                raise QuoteMemError("binding_mismatch", "An SDK entity id belongs to another run")
            return entity
        now = self._reads.clock()
        sequence = writer.get_counter("commit_seq") + 1
        if isinstance(request, EntityCreateV1):
            entity = EntityRowV1(
                object_id=selected.entity_id, scope_kind=scope.scope_kind,
                scope_id=scope.scope_id, entity_type=request.entity_type, state="active",
                created_by_run_id=run["object_id"], commit_seq=sequence, created_at=now,
                updated_at=now,
            )
        else:
            entity = self._retyped(scope, request, stored, sequence, now)
            if entity is None:
                return _ENTITY.validate_python(stored)
        writer.commit_rows(
            ticket, [RowWrite("entity", entity)], updated_at=now,
            publish=lambda committed: self._reads.publisher.publish_commit(writer, committed),
        )
        return entity

    def _retyped(
        self, scope: ScopeV1, request: EntitySetTypeV1, stored: dict[str, Any] | None,
        sequence: int, now: int,
    ) -> EntityRowV1 | None:
        """The retyped row, or None when this run's retype is already written."""
        writer = self._reads.writer
        if stored is None or (stored["scope_kind"], stored["scope_id"]) != (
            scope.scope_kind, scope.scope_id,
        ) or stored["state"] != "active":
            raise QuoteMemError("scope_denied", "Entity is unavailable")
        entity = _ENTITY.validate_python(stored)
        creator = writer.get_row("run", entity.created_by_run_id)
        if creator is None or creator["run_kind"] != "entity_update":
            raise QuoteMemError("operation_conflict", "Only an SDK entity changes its type")
        if entity.entity_type == request.entity_type and entity.entity_type != (
            request.expected_type
        ):
            return None
        if entity.entity_type != request.expected_type:
            raise QuoteMemError("operation_conflict",
                                "The entity's type is no longer the observed one")
        return entity.model_copy(update={
            "entity_type": request.entity_type, "commit_seq": sequence, "updated_at": now,
        })

    def _written(self, scope: ScopeV1, entity_id: str) -> EntityRecordV1:
        """The current record of the entity a finished run wrote."""
        stored = self._reads.writer.get_row("entity", entity_id)
        if stored is None or stored["state"] != "active" or (
            stored["scope_kind"], stored["scope_id"],
        ) != (scope.scope_kind, scope.scope_id):
            raise QuoteMemError("scope_denied", "Entity is unavailable")
        return _record(scope, _ENTITY.validate_python(stored))


def _request_id(value: str) -> str:
    try:
        return _EXTERNAL.validate_python(value, strict=True)
    except ValidationError:
        raise QuoteMemError("invalid_request", "A request id has 1 to 1024 characters") from None


def _run(values: dict[str, Any]) -> Any:
    """Parse one ``entity_update`` run row with the store's strict run model."""
    return DOMAIN_ADAPTERS["run"].validate_python(values)
