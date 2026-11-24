"""Load the pinned local o200k_base tokenizer (tiktoken 0.12.0) after verifying its bytes.

Tokenizer assets and the release wheel are caller-supplied verified downloads. This
module performs no acquisition, cache lookup, provider or service request. The bench counts
AMLB's Answer cut with this encoding.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiktoken import Encoding

_VERSION = "0.12.0"
_ASSET_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
# Configuration from the exact release's tiktoken_ext/openai_public.py (source revision
# 97e49cbadd500b5cc9dbb51a486f0b42e6701bee). Construct Encoding directly so neither the
# plugin registry nor the download/cache path runs.
_PATTERN = "|".join((
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
))
_SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


def checked_encoding(
    package_archive: Path, package_sha256: str, encoding_asset: Path,
) -> tuple[Encoding, dict[str, str]]:
    """Verify the installed tokenizer against its selected wheel, then build the encoding.

    Args:
        package_archive (Path): Exact platform wheel acquired from the public release.
        package_sha256 (str): Independently verified public digest of that wheel.
        encoding_asset (Path): Local o200k_base BPE file matching the pinned public digest.

    Returns:
        tuple[Encoding, dict[str, str]]: The o200k_base encoding, constructed directly so
            neither tiktoken's plugin registry nor its download cache runs, and the SHA-256
            of every installed wheel file that was compared.

    Raises:
        ValueError: The wheel, asset, installed files or imported implementation differ
            from the pinned release.
        ImportError: The pinned optional bench tokenizer dependency is unavailable.
        OSError: A local artifact cannot be read.
    """
    with package_archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != package_sha256:
            raise ValueError("Tokenizer release wheel differs from its selected public digest")
    asset = encoding_asset.read_bytes()
    if hashlib.sha256(asset).hexdigest() != _ASSET_SHA256:
        raise ValueError("Tokenizer asset differs from the pinned o200k_base release")
    distribution = importlib.metadata.distribution("tiktoken")
    if distribution.version != _VERSION:
        raise ValueError("The local tokenizer must be the exact tiktoken 0.12.0 release")
    files: dict[str, str] = {}
    with zipfile.ZipFile(package_archive) as archive:
        for entry in archive.infolist():
            name = entry.filename
            if entry.is_dir() or name == f"tiktoken-{_VERSION}.dist-info/RECORD":
                continue
            if (
                ".." in Path(name).parts or Path(name).is_absolute()
                or not name.startswith((
                    "tiktoken/", "tiktoken_ext/", f"tiktoken-{_VERSION}.dist-info/"
                ))
            ):
                raise ValueError("Tokenizer wheel contains an unexpected package path")
            expected = hashlib.sha256(archive.read(entry)).hexdigest()
            installed = Path(str(distribution.locate_file(name)))
            with installed.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Installed tokenizer differs from its selected wheel bytes")
            files[name] = actual
    if not {"tiktoken/core.py", "tiktoken_ext/openai_public.py"} <= files.keys():
        raise ValueError("Tokenizer wheel does not contain the selected encoding implementation")

    from tiktoken import Encoding

    native_file = importlib.import_module("tiktoken._tiktoken").__file__
    if (
        Path(importlib.import_module(Encoding.__module__).__file__ or "").resolve()
        != Path(str(distribution.locate_file("tiktoken/core.py"))).resolve()
        or native_file is None
        or not any(
            Path(str(distribution.locate_file(name))).resolve() == Path(native_file).resolve()
            for name in files if name.startswith("tiktoken/_tiktoken.")
        )
    ):
        raise ValueError("Tokenizer imports differ from the verified installed implementation")
    ranks: dict[bytes, int] = {}
    for line in asset.splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token, validate=True)] = int(rank)
    return Encoding(
        "o200k_base", pat_str=_PATTERN, mergeable_ranks=ranks, special_tokens=_SPECIAL_TOKENS,
    ), files

