"""Exact source-owned participant provenance, scoped by the authenticated caller."""

from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from aml.store.metadata import OpaqueIdSpec, StrictModel


def _metadata_version(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("Source metadata version must be the integer 1")
    return value


def _opaque_identifier(value: object) -> object:
    if isinstance(value, str):
        value.encode("utf-8")
    return value


SourceMetadataVersion = Annotated[Literal[1], BeforeValidator(_metadata_version)]
SourceParticipantId = Annotated[
    str, Field(min_length=1, max_length=256), BeforeValidator(_opaque_identifier),
    OpaqueIdSpec("external"),
]


class SourceParticipantV1(StrictModel):
    """A source adapter's exact participant identity within one conversation.

    These opaque strings are never inferred from display names or session IDs.
    Their authority is limited to the authenticated scope and the original
    speaker; all three values must match before a prior binding can be reused.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: SourceMetadataVersion = 1
    namespace: SourceParticipantId
    conversation_id: SourceParticipantId
    participant_id: SourceParticipantId
