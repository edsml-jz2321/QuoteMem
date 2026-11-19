"""Distinct bindings for benchmark bundles and their individual raw source files."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryFile
from typing import IO, Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, JsonValue, field_validator, model_validator

from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.store.metadata import StrictModel

_COPY_CHUNK_BYTES = 1024 * 1024


class DatasetPart(StrictModel):
    """One publisher file; its SHA256 always identifies raw downloaded bytes."""

    path: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Keep source parts within a task-owned directory without ambiguous paths."""
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or "\\" in self.path
            or any(part in ("", ".", "..") for part in self.path.split("/"))
        ):
            raise ValueError("Dataset part path is not a relative canonical path")
        return self


def bundle_manifest_digest(payload: dict[str, JsonValue]) -> str:
    """Hash the complete declared manifest, excluding only its own digest field."""
    content = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


class DatasetBundleSource(StrictModel):
    """A frozen BEAM selection; manifest identity is separate from every file hash."""

    kind: Literal["dataset_bundle_v1"]
    dataset: Literal["beam"]
    variant: Literal["128k"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    selection: Literal["publisher_128k_all20_prefer_truncated_all_probes_v1"]
    parts: Annotated[tuple[DatasetPart, ...], Field(min_length=1)]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    license: Annotated[str, Field(min_length=1)]
    license_url: Annotated[str, Field(min_length=1)]
    expected_cases: Annotated[int, Field(gt=0)]

    @field_validator("parts", mode="before")
    @classmethod
    def parts_from_array(
        cls, value: JsonValue | tuple[DatasetPart, ...]
    ) -> tuple[JsonValue, ...] | tuple[DatasetPart, ...]:
        """Freeze the declared JSON array without accepting arbitrary iterable coercions."""
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, tuple):
            return value
        raise ValueError("Dataset parts must be an array")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require unique immutable URLs and the exact separately declared manifest hash."""
        if len({part.path for part in self.parts}) != len(self.parts):
            raise ValueError("Dataset bundle repeats a part path")
        for part in self.parts:
            url = urlsplit(part.url)
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or self.revision not in url.path
                or url.query
                or url.fragment
            ):
                raise ValueError("Dataset part URL must identify its immutable publisher revision")
        if bundle_manifest_digest(self.model_dump(mode="json")) != self.manifest_sha256:
            raise ValueError("Dataset bundle manifest digest does not match its exact parts")
        return self


def bundle_part_path(directory: Path, part: DatasetPart) -> Path:
    """Resolve a part without allowing existing symlinks to leave the source directory."""
    root = directory.resolve()
    path = root.joinpath(*PurePosixPath(part.path).parts).resolve()
    if not path.is_relative_to(root):
        raise AMLError("dataset_source_invalid", "Dataset part leaves its source directory")
    return path


@contextmanager
def verified_source(path: Path, *, size_bytes: int, sha256: str) -> Iterator[IO[bytes]]:
    """Yield a private seekable snapshot of the exact hash-verified source bytes.

    Args:
        path (Path): Local source to open once; later pathname replacement is irrelevant.
        size_bytes (int): Exact manifest length, also the temporary disk bound.
        sha256 (str): Expected SHA256 of the raw bytes, before any decoding.

    Yields:
        IO[bytes]: Borrowed snapshot positioned at zero. The caller must not retain it
            past this context. Copy reads are at most 1 MiB with constant copy memory;
            decoder memory is separate, and this is not an instantaneous RSS bound.
            Temporary disk use is at most the declared size. No persistent cache is created.

    Raises:
        AMLError: If the observed size or hash differs from the manifest.
        OSError: If the source or temporary file cannot be read or written. Both handles
            close on verification errors, caller decoding errors and normal exit.
    """
    if type(size_bytes) is not int or size_bytes <= 0:
        raise AMLError("dataset_hash_mismatch", "Dataset size must be positive")
    with TemporaryFile(mode="w+b") as snapshot:
        with path.open("rb") as source:
            digest = hashlib.sha256()
            copied = 0
            while chunk := source.read(min(_COPY_CHUNK_BYTES, size_bytes - copied + 1)):
                copied += len(chunk)
                if copied > size_bytes:
                    raise AMLError(
                        "dataset_hash_mismatch", "Dataset size differs from its source manifest"
                    )
                digest.update(chunk)
                snapshot.write(chunk)
            if copied != size_bytes or digest.hexdigest() != sha256:
                raise AMLError(
                    "dataset_hash_mismatch", "Dataset bytes differ from their source manifest"
                )
        snapshot.seek(0)
        yield snapshot


def verify_bundle(directory: Path, source: DatasetBundleSource) -> dict[str, Path]:
    """Check current raw files; returned paths are locators, not immutable attestations.

    A consumer must decode a ``verified_source`` snapshot rather than reopen a
    returned path under this earlier check's identity.

    Args:
        directory (Path): Explicit local directory containing the declared relative parts.
        source (DatasetBundleSource): Complete manifest with individual raw byte hashes.

    Returns:
        dict[str, Path]: Verified paths keyed by their canonical source-relative names.

    Raises:
        AMLError: For changed manifest metadata, missing bytes, escaped paths or hashes.
    """
    try:
        source = DatasetBundleSource.model_validate(source.model_dump())
        paths: dict[str, Path] = {}
        for part in source.parts:
            path = bundle_part_path(directory, part)
            if path.stat().st_size != part.size_bytes:
                raise ValueError("Dataset part size changed")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != part.sha256:
                    raise ValueError("Dataset part bytes changed")
            paths[part.path] = path
        return paths
    except AMLError:
        raise
    except (OSError, ValueError):
        raise AMLError(
            "dataset_hash_mismatch", "Dataset bundle failed raw source verification"
        ) from None
