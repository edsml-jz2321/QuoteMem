"""Explicit, resumable acquisition of revision-and-hash-pinned public benchmark files."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from aml.bench.bundle import DatasetBundleSource, DatasetPart, bundle_part_path
from aml.bench.datasets import BenchmarkSource, DatasetSource
from aml.core.errors import AMLError
from aml.core.platform_files import lock_exclusive, open_regular_file


def _regular_size(path: Path) -> int | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or (
        getattr(metadata, "st_file_attributes", 0) & 0x400
    ):
        raise AMLError("dataset_hash_mismatch", "Dataset paths must be singly linked regular files")
    return metadata.st_size


def _verified(path: Path, source: DatasetSource | DatasetPart) -> bool:
    if _regular_size(path) != source.size_bytes:
        return False
    descriptor = open_regular_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != source.size_bytes:
            return False
        return hashlib.file_digest(stream, "sha256").hexdigest() == source.sha256


def download_dataset(source: BenchmarkSource, path: Path, client: httpx.Client) -> None:
    """Acquire one exact public file, preserving partial bytes and rejecting changed artifacts.

    Args:
        source (DatasetSource): Explicit HTTPS publisher URL with its immutable revision,
            expected size, and SHA256 from the checked source manifest.
        path (Path): Task-owned regular output file. Partial state uses the adjacent
            `.part` file; an existing verified regular final file is reused without
            network access. Symlinks and special files are rejected.
        client (httpx.Client): Caller-owned downloader with bounded connection/read timeouts.
            No provider or registry authentication is required by these public sources.

    Returns:
        None: Verified bytes are atomically published without replacing a destination.

    Raises:
        AMLError: If the destination is occupied by different bytes, another downloader
            owns it, disk space is insufficient, range semantics fail, or transfer/checksum
            verification fails. A destination created during transfer and the verified
            partial are both preserved on conflict, including when their bytes match.
            Partial bytes remain available for a later explicit resume.
    """
    if isinstance(source, DatasetBundleSource):
        source = DatasetBundleSource.model_validate(source.model_dump())
        for part in source.parts:
            _download_file(part, bundle_part_path(path, part), client, revision=source.revision)
    else:
        _download_file(source, path, client, revision=source.revision)


def _download_file(
    source: DatasetSource | DatasetPart, path: Path, client: httpx.Client, *, revision: str
) -> None:
    parts = urlsplit(source.url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or revision not in parts.path
    ):
        raise AMLError("dataset_source_invalid", "Dataset acquisition requires a pinned HTTPS URL")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    lock_path = path.with_name(path.name + ".download.lock")
    try:
        descriptor = open_regular_file(lock_path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(descriptor, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise AMLError("dataset_download_failed", "Dataset lock must be a regular file")
            try:
                lock_exclusive(lock.fileno())
            except BlockingIOError:
                raise AMLError(
                    "dataset_download_busy", "Another downloader owns this destination"
                ) from None
            if _regular_size(path) is not None:
                if _verified(path, source):
                    return
                raise AMLError(
                    "dataset_hash_mismatch", "Existing dataset differs from its manifest"
                )
            offset = _regular_size(partial) or 0
            if offset > source.size_bytes:
                raise AMLError("dataset_hash_mismatch", "Partial dataset exceeds its expected size")
            if shutil.disk_usage(path.parent).free < source.size_bytes - offset + 64 * 1024 * 1024:
                raise AMLError(
                    "dataset_disk_insufficient", "Dataset download lacks a safety reserve"
                )
            if offset != source.size_bytes:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with client.stream("GET", source.url, headers=headers) as response:
                    expected_status = 206 if offset else 200
                    if response.status_code != expected_status:
                        raise AMLError(
                            "dataset_download_failed", "Dataset response status is invalid"
                        )
                    if offset:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", "")
                        )
                        if match is None or tuple(map(int, match.groups())) != (
                            offset,
                            source.size_bytes - 1,
                            source.size_bytes,
                        ):
                            raise AMLError(
                                "dataset_download_failed", "Dataset resume range is invalid"
                            )
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise AMLError("dataset_download_failed", "Dataset byte encoding changed")
                    descriptor = open_regular_file(
                        partial,
                        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                    )
                    with os.fdopen(descriptor, "ab") as output:
                        if not stat.S_ISREG(os.fstat(output.fileno()).st_mode):
                            raise AMLError(
                                "dataset_hash_mismatch", "Partial dataset must be a regular file"
                            )
                        for chunk in response.iter_raw():
                            if output.tell() + len(chunk) > source.size_bytes:
                                raise AMLError(
                                    "dataset_hash_mismatch", "Dataset exceeds its manifest"
                                )
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
            if not _verified(partial, source):
                raise AMLError(
                    "dataset_hash_mismatch", "Downloaded dataset differs from its manifest"
                )
            try:
                os.link(partial, path, follow_symlinks=False)
            except FileExistsError:
                raise AMLError(
                    "dataset_hash_mismatch", "Dataset destination appeared during acquisition"
                ) from None
            partial.unlink()
    except AMLError:
        raise
    except (OSError, httpx.HTTPError):
        raise AMLError(
            "dataset_download_failed", "Dataset transfer failed; partial bytes retained"
        ) from None


def download_source_file(
    source: DatasetPart, path: Path, client: httpx.Client, *, revision: str
) -> None:
    """Acquire raw source bytes before decoding establishes a dataset's actual case count.

    Args:
        source (DatasetPart): Exact publisher file URL, byte size and raw SHA256.
        path (Path): Task-owned destination, with adjacent resumable partial state.
        client (httpx.Client): Explicit bounded downloader, without provider credentials.
        revision (str): Immutable publisher revision required in the source URL.

    Returns:
        None: Verified bytes are available at the destination.

    Raises:
        AMLError: For invalid source identity, transfer failure or mismatched bytes.
    """
    _download_file(source, path, client, revision=revision)
