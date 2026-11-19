"""Bounded residual inspection with explicit proof for unordered Artifact fragments.

An Artifact part stores no logical kind or position (A0.7). A
per-part credential scan therefore cannot prove that the recoverable payload is
clean. Complete manifests supply that ordering; otherwise only exact fragments
of a validated, content-free ingestion erasure block are accepted. This module
does not delete data or certify arbitrary text as free of unrecognized secrets.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Never

from aml.core.artifacts import ArtifactEnvelopeV1, ArtifactRowV1, ManifestV1, PartV1
from aml.core.errors import AMLError
from aml.core.identity import ArtifactId, artifact_id, canonical_json
from aml.core.privacy_persistence import IngestionErasurePlanV1
from aml.store.metadata import StrictModel
from aml.store.registry import extract_text_leaves

_PART_BYTES = 180000
_BODY_CHARS = 262144


@dataclass(frozen=True)
class ResidualScanSummary:
    """Return counts of inspected physical rows, never source bytes or locators."""

    artifacts: int
    manifests: int
    parts: int
    certified_orphan_parts: int


class PrivacyResidualScanner:
    """Inspect one optimistic-generation scan without retaining decoded source content.

    Args:
        recognizes: The same supported credential detector used by Add preflight.
        max_tracked_artifacts: Hard in-memory identity budget. Exceeding it fails
            closed; the caller cannot report completion from a truncated scan.
        report: Where a recognized credential goes instead of failing the scan:
            called once per inspected value that holds one, with that value's
            content-free kind (D.7.5 credential findings). Without it a credential
            fails the scan. Structural faults fail the scan either way.

    Notes:
        The caller owns table iteration and the final same-epoch/generation
        check. Its logical loader must be the authoritative ArtifactStore reader,
        including complete part/envelope/reference validation. Only identities
        are retained between calls, not decoded bodies or incoming boundary bytes.
    """

    def __init__(
        self,
        *,
        recognizes: Callable[[bytes], bool],
        max_tracked_artifacts: int = 1_000_000,
        report: Callable[[str], None] | None = None,
    ) -> None:
        if type(max_tracked_artifacts) is not int or max_tracked_artifacts < 1:
            raise AMLError("invalid_request", "Residual scan requires a positive identity budget")
        self._recognizes = recognizes
        self._report = report
        self._limit = max_tracked_artifacts
        self._tracked: set[ArtifactId] = set()
        self._seen: set[ArtifactId] = set()
        self._parts: set[ArtifactId] = set()
        self._covered: set[ArtifactId] = set()
        self._certified: set[ArtifactId] = set()
        self._manifests = 0
        self._failed = False

    def _reject(self, message: str) -> Never:
        self._failed = True
        raise AMLError("closure_incomplete", message)

    def _track(self, identity: ArtifactId) -> None:
        if identity not in self._tracked:
            if len(self._tracked) >= self._limit:
                self._reject("Residual scan exhausted its identity budget")
            self._tracked.add(identity)

    def inspect_value(self, value: object, *, kind: str | None = None) -> None:
        """Inspect decoded strings, including JSON object keys, with bounded traversal.

        A strict model's leaves declared as AML references, local identities or
        keyed digests hold product-generated identifiers, which can resemble a
        credential signature by chance; they are not inspected. Free text,
        external identifiers, undeclared strings and every untyped mapping or
        sequence, keys included, are.

        Args:
            value: A strict entity row or logical body, or a decoded untyped value;
                never an encoded fragment.
            kind: The value's content-free kind for a reported finding; its type
                name when absent.

        Raises:
            AMLError: If a supported credential signature is found and the scan has
                no ``report``. The error names the content-free leaf path, never the
                matched value.
        """
        for leaf in extract_text_leaves(value):
            if self._recognizes(leaf.value.encode("utf-8")):
                if self._report is not None:
                    self._report(kind or type(value).__name__)
                    return
                self._reject(
                    "Credential residual scan found a credential signature in "
                    f"{type(value).__name__} at {leaf.path or 'its root'}",
                )

    def _row(self, value: ArtifactRowV1 | Mapping[str, object]) -> ArtifactRowV1:
        raw = value.model_dump(mode="json") if isinstance(value, ArtifactRowV1) else dict(value)
        try:
            row = ArtifactRowV1.model_validate(raw)
            envelope = row.model_dump(mode="json", include=set(ArtifactEnvelopeV1.model_fields))
            if artifact_id(envelope) != row.object_id or row.size_bytes != len(
                row.body.encode("utf-8")
            ):
                raise ValueError("Invalid physical identity")
            return row
        except ValueError:
            self._failed = True
            raise AMLError(
                "closure_incomplete", "Residual scan found an invalid Artifact"
            ) from None

    def inspect_artifact(
        self,
        value: ArtifactRowV1 | Mapping[str, object],
        *,
        load_logical: Callable[[str, str], StrictModel],
    ) -> None:
        """Scan a physical row and decode a complete manifest at most once.

        Args:
            value: One complete physical Artifact row from the owned table scan.
            load_logical: ArtifactStore.read_logical, which must verify the full
                manifest, every part, its envelope, and reconstructed strict body.

        Raises:
            AMLError: On a credential, malformed identity, unavailable logical
                payload, or exhausted tracking budget. Missing manifests are
                assessed only by finish, independently of table iteration order.
        """
        row = self._row(value)
        self._track(row.object_id)
        if row.object_id in self._seen:
            return
        self.inspect_value(row.model_dump(mode="json", exclude={"body"}), kind="artifact")
        if row.artifact_kind == "part":
            part = PartV1.model_validate_json(row.body)
            if row.artifact_refs:
                self._reject("Physical part owns unexpected references")
            # Parts encode slices of canonical JSON, not decoded scalar values.
            # Escaping can resemble a credential even when the decoded text is
            # a permitted redaction. Manifest or exact block proof is mandatory
            # at finish, so raw fragment pattern matches cannot certify either
            # acceptance or rejection of the complete logical body.
            base64.b64decode(part.data, validate=True)
            self._parts.add(row.object_id)
        else:
            manifest = (
                ManifestV1.model_validate_json(row.body)
                if row.artifact_kind == "manifest"
                else None
            )
            if manifest is not None:
                for reference in manifest.parts:
                    self._track(reference)
            logical_kind = manifest.logical_kind if manifest is not None else row.artifact_kind
            try:
                logical = load_logical(row.object_id, logical_kind)
            except Exception:
                self._failed = True
                raise AMLError(
                    "closure_incomplete", "Residual scan could not verify a logical Artifact"
                ) from None
            if manifest is not None:
                encoded = canonical_json(logical).encode("utf-8")
                if (
                    len(encoded) != manifest.total_bytes
                    or hashlib.sha256(encoded).hexdigest() != manifest.body_digest
                ):
                    self._reject("Manifest logical scan lost its binding")
            elif canonical_json(logical) != row.body:
                self._reject("Direct logical scan lost its body binding")
            self.inspect_value(logical, kind=logical_kind)
            if isinstance(logical, IngestionErasurePlanV1):
                self.certify_cleanup_block(row, logical)
            if manifest is not None:
                self._covered.update(manifest.parts)
                self._manifests += 1
        self._seen.add(row.object_id)

    def certify_cleanup_block(
        self,
        envelope_source: ArtifactRowV1 | Mapping[str, object],
        logical: IngestionErasurePlanV1,
    ) -> None:
        """Prove exact fixed-slice identities from a strict content-free logical candidate.

        Args:
            envelope_source: An existing same-creator Artifact supplying exact
                tenant, scope, security, creator and body-reference metadata.
                The caller must already have verified its cleanup ownership.
            logical: An existing or deterministically reconstructed complete
                ingestion erasure block, validated by the owning plan loader.

        Raises:
            AMLError: For invalid proof shape, unprotected scope, a recognized
                credential, or exhausted identity budget. Creator kind alone
                never proves an orphan safe. Inline blocks certify no parts.
        """
        source = self._row(envelope_source)
        try:
            block = IngestionErasurePlanV1.model_validate(logical.model_dump(mode="json"))
        except (AttributeError, ValueError):
            self._failed = True
            raise AMLError("closure_incomplete", "Cleanup fragment proof is not strict") from None
        if source.scope_id is None or source.security_level != "restricted":
            self._reject("Cleanup fragment proof lacks its protected scope")
        self.inspect_value(block)
        text = canonical_json(block)
        if len(text) <= _BODY_CHARS:
            return
        payload = text.encode("utf-8")
        envelope: dict[str, Any] = source.model_dump(
            mode="json", include=set(ArtifactEnvelopeV1.model_fields)
        )
        envelope["artifact_kind"] = "part"
        for start in range(0, len(payload), _PART_BYTES):
            fragment = PartV1(
                encoding="base64",
                data=base64.b64encode(payload[start : start + _PART_BYTES]).decode("ascii"),
            )
            envelope["body"] = canonical_json(fragment)
            reference = artifact_id(envelope)
            self._track(reference)
            self._certified.add(reference)

    def finish(self) -> ResidualScanSummary:
        """Reject every unordered orphan lacking an exact complete logical proof.

        Returns:
            Counts only after all observed parts are covered; this does not itself
            prove table enumeration, reader quiescence or the final generation.

        Raises:
            AMLError: If any fragment has no manifest or validated cleanup proof.
        """
        if self._failed:
            raise AMLError("closure_incomplete", "A prior residual scan check failed")
        orphans = self._parts - self._covered
        if not orphans <= self._certified:
            self._reject("Residual scan found ambiguous orphan fragments")
        return ResidualScanSummary(
            len(self._seen), self._manifests, len(self._parts), len(orphans)
        )
