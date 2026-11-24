"""Agent-profile settings: environment, data directory, token and store binding.

An agent store lives in one data directory:

- ``workspace/``: the QuoteMem workspace (mode 0700), served by the bench composition;
- ``agent.json``: the nonsecret binding of the store to its embedding model;
- ``token``: the generated HTTP bearer token (mode 0600) unless one is configured.

Provider endpoints are resolved from the environment on every start and never
written into the workspace (``quotemem.agent.routing``), so a gateway URL or key can
change without making the store unopenable. The embedding model and dimension
stay bound: vectors of another model would share no space with the stored ones.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast, get_args
from urllib.parse import urlsplit

from pydantic import ValidationError

from quotemem.core.bootstrap import WorkspaceConfiguration
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import AddMode, ReviewMode
from quotemem.core.platform_files import (
    ensure_private_directory,
    open_private_file,
    validate_private_path,
)
from quotemem.providers.config import DEFAULT_ADD_MODE, ProviderConfig
from quotemem.store.metadata import StrictModel

AGENT_JOB_ID: Final = "agent-local"
"""The fixed job identity of an agent store; the bench runtime binds one job per store."""
CANONICAL_CHAT_BASE_URL: Final = "https://chat.agent-route.invalid/v1"
"""The chat endpoint written into an agent workspace; ``quotemem.agent.routing`` maps it."""
CANONICAL_EMBEDDING_BASE_URL: Final = "https://embedding.agent-route.invalid/v1"
"""The embedding endpoint written into an agent workspace; routing maps it too."""
ROUTED_ENVIRONMENT: Final[Mapping[str, str]] = {
    "DASHSCOPE_API_KEY": "agent-routed", "OPENAI_API_KEY": "agent-routed",
}
"""Placeholder credentials for the core: the routing clients replace them on the wire."""
LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})

type StructuredOutput = Literal["json_schema", "json_object"]
type ReasoningParameters = Literal["off", "high", "none", "effort", "passthrough"]

_STRUCTURED_OUTPUTS = frozenset(get_args(StructuredOutput.__value__))
_REASONING = frozenset(get_args(ReasoningParameters.__value__))
DEFAULT_REASONING_TOKENS: Final = 24_576
"""The reasoning tokens ``high`` adds to every chat call's output bound
(``quotemem.agent.routing``).

DeepSeek v4.1 flash (1017) on DashScope at high effort, over one test conversation, a
correction batch and a statement forget (88 calls): reasoning tokens per call were
p50 1,884, p90 8,903 and at most 22,270 for extraction, p50 5,263, p90 22,633 and at most
22,712 for the judge, and at most 3,929 for a forget vote, with answers of at most 1,344
tokens. A one-member judge reply bounded at 4,096 answer tokens then has 28,672 in all, the
benchmark modes' four-member bound. At the 181 tokens a second measured, 24,576 reasoning
tokens take about 136 s, so ``DEFAULT_REASONING_SECONDS`` usually ends a long reasoning
reply first; the bound stops a faster runaway."""
DEFAULT_REASONING_SECONDS: Final = 120.0
"""How long ``high`` waits for one reasoning reply before asking again without thinking.

In the same run 86 of 88 calls answered within it (extraction at most 108 s); two judge
replies of 22,633 and 22,712 reasoning tokens (126 s and 135 s) were asked again without
thinking and answered in about 5 s. A call thus waits at most this long plus its answer
without thinking. A call with a shorter time limit waits at most half of it, and at most
that limit less 30 s, so the resend without thinking keeps time to answer
(``quotemem.agent.routing``)."""
_REASONING_TOKENS = (1_024, 131_072)
_REASONING_SECONDS = (10.0, 900.0)
_ADD_MODES = frozenset(get_args(AddMode))
AGENT_ADD_MODE: Final[AddMode] = "agent"
"""The Add mode a new agent store is created with: long-lived memory from live chat.

A store created before this default stored no mode and keeps ``DEFAULT_ADD_MODE``
(``lean-local``); a store's frozen mode is never rewritten."""
REVIEW_VARIABLE: Final = "QUOTEMEM_REVIEW"
"""The variable selecting a new store's review (``ReviewMode``); ``--review`` overrides it."""
_REVIEWS = frozenset(get_args(ReviewMode))


@dataclass(frozen=True)
class ChatEndpoint:
    """One OpenAI-compatible chat-completions endpoint.

    Attributes:
        base_url: The service root, ``https`` or loopback ``http``.
        model: The model name sent on the wire.
        api_key: The bearer credential; never represented or persisted.
        structured_output: ``json_schema`` sends strict schemas; ``json_object``
            sends JSON mode and states the schema in a system message.
        reasoning: ``off`` (the default) asks for no extended thinking on every call
            (``thinking: {"type": "disabled"}``, no ``reasoning_effort``) and sends nothing
            once the provider refuses that parameter; ``high`` asks every call for
            thinking (``thinking: {"type": "enabled"}``) and requests
            ``reasoning_effort: "high"`` (accepted by DashScope; no effect of the effort was
            measurable there), adds ``reasoning_room`` to the call's output
            bound and asks again without thinking when the reasoning reply is cut off or
            slower than ``reasoning_wait`` (``quotemem.agent.routing``); ``none`` sends no
            reasoning parameters, so the provider's own default applies; ``effort`` sends
            only ``reasoning_effort``; ``passthrough`` sends the core's parameters as built.
        reasoning_room: Under ``high``, the reasoning tokens added to each call's output
            bound (``QUOTEMEM_CHAT_REASONING_TOKENS``).
        reasoning_wait: Under ``high``, the longest wait in seconds for one reasoning reply
            (``QUOTEMEM_CHAT_REASONING_SECONDS``).
        model_pattern: A full-match pattern for reported model ids; None admits the
            configured name and that name with a date suffix.
    """

    base_url: str
    model: str
    api_key: str = field(repr=False)
    structured_output: StructuredOutput = "json_schema"
    reasoning: ReasoningParameters = "off"
    model_pattern: re.Pattern[str] | None = None
    reasoning_room: int = DEFAULT_REASONING_TOKENS
    reasoning_wait: float = DEFAULT_REASONING_SECONDS


@dataclass(frozen=True)
class EmbeddingEndpoint:
    """One OpenAI-compatible embeddings endpoint.

    Attributes:
        base_url: The service root, ``https`` or loopback ``http``.
        model: The model name sent on the wire.
        api_key: The bearer credential; never represented or persisted.
        send_dimensions: Send the store's dimension; models without that parameter
            return their native size, which is zero-padded when smaller.
        model_pattern: As for chat.
    """

    base_url: str
    model: str
    api_key: str = field(repr=False)
    send_dimensions: bool = True
    model_pattern: re.Pattern[str] | None = None


def _value(environment: Mapping[str, str], *names: str) -> tuple[str, str] | None:
    """The first nonblank variable among ``names``, with its name."""
    for name in names:
        value = environment.get(name, "").strip()
        if value:
            return name, value
    return None


def _required(environment: Mapping[str, str], *names: str) -> tuple[str, str]:
    found = _value(environment, *names)
    if found is None:
        raise QuoteMemError("provider_unavailable", f"Set {' or '.join(names)}")
    return found


def endpoint_url(value: str, variable: str) -> str:
    """Validate one provider service root.

    Args:
        value: The configured URL.
        variable: The variable that supplied it, named in a refusal.

    Returns:
        The URL without its trailing slash.

    Raises:
        QuoteMemError: ``schema_invalid`` unless it is ``https``, or ``http`` to a loopback
            host, with no credentials, query or fragment.
    """
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "http"} or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or (parsed.scheme == "http" and parsed.hostname not in LOOPBACK_HOSTS)
    ):
        raise QuoteMemError(
            "schema_invalid",
            f"{variable} must be an https URL (or http on loopback) without credentials",
        )
    return value.rstrip("/")


def _choice[T: str](
    environment: Mapping[str, str], name: str, allowed: frozenset[str], default: T,
) -> T:
    value = environment.get(name, "").strip()
    if not value:
        return default
    if value not in allowed:
        raise QuoteMemError("schema_invalid", f"{name} is one of {', '.join(sorted(allowed))}")
    return cast(T, value)


def _bounded(environment: Mapping[str, str], name: str, bounds: tuple[float, float],
             default: float) -> float:
    """A number variable within ``bounds``, else ``default`` when unset."""
    value = environment.get(name, "").strip()
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not bounds[0] <= number <= bounds[1]:
        raise QuoteMemError("schema_invalid",
                            f"{name} is a number from {bounds[0]:g} to {bounds[1]:g}")
    return number


def _pattern(environment: Mapping[str, str], name: str) -> re.Pattern[str] | None:
    value = environment.get(name, "").strip()
    if not value:
        return None
    try:
        return re.compile(value)
    except re.error:
        raise QuoteMemError("schema_invalid", f"{name} is not a valid regular expression") from None


def resolve_endpoints(environment: Mapping[str, str]) -> tuple[ChatEndpoint, EmbeddingEndpoint]:
    """Resolve both provider endpoints from the environment, without network access.

    Chat reads ``QUOTEMEM_CHAT_{BASE_URL,MODEL,API_KEY}`` and falls back to
    ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and ``OPENAI_API_KEY``. Embeddings read
    ``QUOTEMEM_EMBED_{BASE_URL,MODEL,API_KEY}``; the URL falls back to the chat
    URL and the key to ``DASHSCOPE_API_KEY``, then to the chat key.

    Raises:
        QuoteMemError: ``provider_unavailable`` naming a missing variable, or
            ``schema_invalid`` naming a malformed one. Values are never named.
    """
    chat_url_name, chat_url = _required(environment, "QUOTEMEM_CHAT_BASE_URL", "OPENAI_BASE_URL")
    _, chat_model = _required(environment, "QUOTEMEM_CHAT_MODEL", "OPENAI_MODEL")
    _, chat_key = _required(environment, "QUOTEMEM_CHAT_API_KEY", "OPENAI_API_KEY")
    chat = ChatEndpoint(
        base_url=endpoint_url(chat_url, chat_url_name), model=chat_model, api_key=chat_key,
        structured_output=_choice(
            environment, "QUOTEMEM_CHAT_STRUCTURED_OUTPUT", _STRUCTURED_OUTPUTS,
            cast(StructuredOutput, "json_schema"),
        ),
        reasoning=_choice(
            environment, "QUOTEMEM_CHAT_REASONING", _REASONING,
            cast(ReasoningParameters, "off"),
        ),
        model_pattern=_pattern(environment, "QUOTEMEM_CHAT_MODEL_PATTERN"),
        reasoning_room=int(_bounded(environment, "QUOTEMEM_CHAT_REASONING_TOKENS",
                                      _REASONING_TOKENS, DEFAULT_REASONING_TOKENS)),
        reasoning_wait=_bounded(environment, "QUOTEMEM_CHAT_REASONING_SECONDS",
                                   _REASONING_SECONDS, DEFAULT_REASONING_SECONDS),
    )
    embed_url = _value(environment, "QUOTEMEM_EMBED_BASE_URL")
    _, embed_model = _required(environment, "QUOTEMEM_EMBED_MODEL")
    embed_key = _value(environment, "QUOTEMEM_EMBED_API_KEY", "DASHSCOPE_API_KEY")
    send = environment.get("QUOTEMEM_EMBED_SEND_DIMENSIONS", "").strip().lower()
    if send not in {"", "true", "false", "1", "0"}:
        raise QuoteMemError("schema_invalid", "QUOTEMEM_EMBED_SEND_DIMENSIONS is true or false")
    embedding = EmbeddingEndpoint(
        base_url=chat.base_url if embed_url is None else endpoint_url(embed_url[1], embed_url[0]),
        model=embed_model, api_key=chat_key if embed_key is None else embed_key[1],
        send_dimensions=send not in {"false", "0"},
        model_pattern=_pattern(environment, "QUOTEMEM_EMBED_MODEL_PATTERN"),
    )
    return chat, embedding


def resolve_review(
    requested: ReviewMode | None, environment: Mapping[str, str],
) -> ReviewMode | None:
    """The review a start asks for: ``--review``, else ``QUOTEMEM_REVIEW``, else None.

    None keeps an existing store's review and gives a new store ``standard``.

    Raises:
        QuoteMemError: ``schema_invalid`` naming the values when either is not a review.
    """
    if requested is not None:
        if requested not in _REVIEWS:
            raise QuoteMemError("schema_invalid", "--review is one of fast, standard or full")
        return requested
    value = environment.get(REVIEW_VARIABLE, "").strip().lower()
    if not value:
        return None
    if value not in _REVIEWS:
        raise QuoteMemError("schema_invalid", f"{REVIEW_VARIABLE} is one of fast, standard or full")
    return cast(ReviewMode, value)


def default_data_dir(environment: Mapping[str, str]) -> Path:
    """``QUOTEMEM_DATA_DIR``, else ``$XDG_DATA_HOME/quotemem``, else ~/.local/share/quotemem."""
    configured = environment.get("QUOTEMEM_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = environment.get("XDG_DATA_HOME", "").strip()
    return (Path(base) if base else Path.home() / ".local" / "share") / "quotemem"


def prepare_data_dir(data_dir: Path) -> Path:
    """Create the private data directory and its workspace directory.

    Args:
        data_dir: The agent data directory; its parents are created as needed.

    Returns:
        The private workspace directory inside it.

    Raises:
        QuoteMemError: ``workspace_unsafe`` when an existing directory is not private to
            this user; it is never repaired.
    """
    data_dir = data_dir.expanduser()
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    ensure_private_directory(data_dir)
    workspace = data_dir / "workspace"
    ensure_private_directory(workspace)
    return workspace


def _visible_ascii(value: str) -> bool:
    return bool(value) and all(0x21 <= ord(character) <= 0x7E for character in value)


def load_token(data_dir: Path, environment: Mapping[str, str]) -> tuple[str, str]:
    """The HTTP bearer token and its source, creating one when none is configured.

    ``QUOTEMEM_TOKEN``, then ``QUOTEMEM_HTTP_AUTH_SECRET``, then ``<data-dir>/token``,
    which is generated (mode 0600) on first use.

    Returns:
        (token, source), the source being the variable name or ``file``.

    Raises:
        QuoteMemError: ``schema_invalid`` when a token is not visible ASCII.
    """
    configured = _value(environment, "QUOTEMEM_TOKEN", "QUOTEMEM_HTTP_AUTH_SECRET")
    if configured is not None:
        if not _visible_ascii(configured[1]):
            raise QuoteMemError("schema_invalid", f"{configured[0]} must be visible ASCII")
        return configured[1], configured[0]
    path = data_dir / "token"
    if not (path.exists() or path.is_symlink()):
        descriptor = open_private_file(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding="ascii") as stream:
        token = stream.read().strip()
    if not _visible_ascii(token):
        raise QuoteMemError("schema_invalid", "The token file must hold one visible ASCII token")
    return token, "file"


def token_source(data_dir: Path, environment: Mapping[str, str]) -> str:
    """Where ``load_token`` would take the token from, without creating one.

    Returns:
        The variable name, ``file`` for an existing token file, or ``generate``
        when the first ``quotemem serve`` would create it.

    Raises:
        QuoteMemError: ``schema_invalid`` for a token that is not visible ASCII.
    """
    configured = _value(environment, "QUOTEMEM_TOKEN", "QUOTEMEM_HTTP_AUTH_SECRET")
    if configured is not None:
        if not _visible_ascii(configured[1]):
            raise QuoteMemError("schema_invalid", f"{configured[0]} must be visible ASCII")
        return configured[0]
    path = data_dir / "token"
    if not (path.exists() or path.is_symlink()):
        return "generate"
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding="ascii") as stream:
        if not _visible_ascii(stream.read().strip()):
            raise QuoteMemError("schema_invalid",
                                "The token file must hold one visible ASCII token")
    return "file"


def read_token(data_dir: Path, environment: Mapping[str, str]) -> str:
    """The service token for a client command, without creating one.

    Raises:
        QuoteMemError: ``authentication_failed`` when none is configured or stored.
    """
    source = token_source(data_dir, environment)
    if source == "generate":
        raise QuoteMemError(
            "authentication_failed",
            "No token: pass the service's --data-dir (or set QUOTEMEM_DATA_DIR), or set "
            "QUOTEMEM_TOKEN_FILE to a file holding the service's token",
        )
    return load_token(data_dir, environment)[0]


def inspect_store(
    data_dir: Path, *, embedding_model: str, dimensions: Literal[1024, 2048] | None,
    add_mode: AddMode | None, review: ReviewMode | None = None,
) -> tuple[dict[str, object], Literal[1024, 2048]]:
    """Check an agent data directory as ``quotemem serve`` would open it, read-only.

    Nothing is created, bound or written: an absent directory, binding or
    workspace is reported as what the first serve would create.

    Returns:
        (the report, the vector dimension the store has or would be created with).

    Raises:
        QuoteMemError: ``workspace_unsafe`` for a directory that is not private,
            ``workspace_binding_mismatch`` for a store bound to another embedding
            model, dimension, Add mode or review, ``schema_invalid`` for a malformed file
            or a review the Add mode cannot take.
    """
    data_dir = data_dir.expanduser()
    report: dict[str, object] = {"data_dir": str(data_dir)}
    if not (data_dir.exists() or data_dir.is_symlink()):
        report["state"] = "new"
        new = agent_provider_config(dimensions=dimensions or 1024, add_mode=add_mode,
                                    review=review)
        report.update(dimensions=new.embedding_dimensions, add_mode=new.add_mode,
                      review=new.review_mode)
        return report, new.embedding_dimensions
    validate_private_path(data_dir, directory=True)
    workspace = data_dir / "workspace"
    if workspace.exists() or workspace.is_symlink():
        validate_private_path(workspace, directory=True)
    config = store_provider_config(workspace, dimensions=dimensions, add_mode=add_mode,
                                   review=review)
    binding = read_binding(data_dir)
    if binding is not None:
        check_binding(binding, embedding_model=embedding_model,
                      dimensions=config.embedding_dimensions)
        report["binding"] = describe_binding(binding)
    report["state"] = "existing" if stored_provider_config(workspace) is not None else "unopened"
    report.update(dimensions=config.embedding_dimensions,
                  add_mode=config.add_mode or DEFAULT_ADD_MODE, review=config.review_mode)
    return report, config.embedding_dimensions


class AgentBindingV1(StrictModel):
    """``agent.json``: what an agent store is bound to; nothing secret.

    Attributes:
        format_version: This file's format.
        job_id: The store's fixed job identity.
        embedding_model: The wire model whose vectors the store holds.
        embedding_dimensions: The stored vector dimension.
        chat_model: The chat model of the latest start, recorded and not enforced.
        created_at_ms: When the store was bound.
        updated_at_ms: The latest start that changed this file.
    """

    format_version: Literal[1] = 1
    job_id: str = AGENT_JOB_ID
    embedding_model: str
    embedding_dimensions: Literal[1024, 2048]
    chat_model: str
    created_at_ms: int
    updated_at_ms: int


def _write_binding(path: Path, binding: AgentBindingV1) -> None:
    staged = path.with_name(path.name + ".tmp")
    if staged.exists():
        staged.unlink()
    descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(binding.model_dump_json(indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def read_binding(data_dir: Path) -> AgentBindingV1 | None:
    """``agent.json``, or None when the store is not bound yet; never writes.

    Raises:
        QuoteMemError: ``schema_invalid`` for a malformed file.
    """
    path = data_dir / "agent.json"
    if not (path.exists() or path.is_symlink()):
        return None
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
    try:
        return AgentBindingV1.model_validate_json(raw)
    except ValidationError:
        raise QuoteMemError("schema_invalid", "agent.json is not a valid agent binding") from None


def check_binding(
    binding: AgentBindingV1, *, embedding_model: str, dimensions: Literal[1024, 2048],
) -> None:
    """Refuse an embedding model or dimension other than the store's.

    Raises:
        QuoteMemError: ``workspace_binding_mismatch`` naming the fix.
    """
    if binding.embedding_model != embedding_model:
        raise QuoteMemError(
            "workspace_binding_mismatch",
            "This store holds vectors of another embedding model; set QUOTEMEM_EMBED_MODEL "
            "back to the model named in agent.json, or use a new --data-dir",
        )
    if binding.embedding_dimensions != dimensions:
        raise QuoteMemError("workspace_binding_mismatch", "The store's embedding dimension differs")


def bind_store(
    data_dir: Path, *, embedding_model: str, dimensions: Literal[1024, 2048],
    chat_model: str, clock_ms: int | None = None,
) -> AgentBindingV1:
    """Create or check ``agent.json``.

    Returns:
        The binding as now stored.

    Raises:
        QuoteMemError: ``workspace_binding_mismatch`` when the store holds vectors of
            another embedding model or dimension; ``schema_invalid`` for a
            malformed file.
    """
    now = time.time_ns() // 1_000_000 if clock_ms is None else clock_ms
    path = data_dir / "agent.json"
    binding = read_binding(data_dir)
    if binding is None:
        binding = AgentBindingV1(
            embedding_model=embedding_model, embedding_dimensions=dimensions,
            chat_model=chat_model, created_at_ms=now, updated_at_ms=now,
        )
        _write_binding(path, binding)
        return binding
    check_binding(binding, embedding_model=embedding_model, dimensions=dimensions)
    if binding.chat_model != chat_model:
        binding = binding.model_copy(update={"chat_model": chat_model, "updated_at_ms": now})
        _write_binding(path, binding)
    return binding


def stored_provider_config(workspace: Path) -> ProviderConfig | None:
    """The provider configuration of an existing agent workspace, else None when new.

    Raises:
        QuoteMemError: ``workspace_binding_mismatch`` when the workspace was not made by
            the agent profile.
    """
    path = workspace / "workspace.json"
    if not (path.exists() or path.is_symlink()):
        return None
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        stored = WorkspaceConfiguration.model_validate_json(stream.read()).providers
    if (stored.openai_base_url, stored.model_studio_base_url) != (
        CANONICAL_CHAT_BASE_URL, CANONICAL_EMBEDDING_BASE_URL,
    ):
        raise QuoteMemError(
            "workspace_binding_mismatch", "This workspace was not created by the agent profile",
        )
    return stored


def agent_provider_config(
    *, dimensions: Literal[1024, 2048] = 1024, add_mode: AddMode | None = None,
    review: ReviewMode | None = None,
) -> ProviderConfig:
    """The provider configuration a new agent workspace is created with.

    The bench profile's storage pipeline, with endpoints that only the agent
    routing clients resolve. None selects ``AGENT_ADD_MODE``, stored explicitly, and
    the ``standard`` review, which is stored as None.

    Raises:
        QuoteMemError: ``schema_invalid`` for an unknown Add mode or review, or for a review
            other than the one the ``full`` or ``lean-local`` Add mode carries out.
    """
    if add_mode is not None and add_mode not in _ADD_MODES:
        raise QuoteMemError("schema_invalid", "The Add mode is invalid")
    if review is not None and review not in _REVIEWS:
        raise QuoteMemError("schema_invalid", f"{REVIEW_VARIABLE} is one of fast, standard or full")
    mode = add_mode or AGENT_ADD_MODE
    config = ProviderConfig(
        profile="bench", embedding_dimensions=dimensions,
        openai_base_url=CANONICAL_CHAT_BASE_URL,
        model_studio_base_url=CANONICAL_EMBEDDING_BASE_URL, add_mode=mode,
        review=review if mode == "agent" else None,
    )
    if review is not None and review != config.review_mode:
        raise QuoteMemError(
            "schema_invalid",
            f"The {mode} Add mode always reviews {config.review_mode}; --review and "
            f"{REVIEW_VARIABLE} choose the review of the agent Add mode",
        )
    return config


def store_provider_config(
    workspace: Path, *, dimensions: Literal[1024, 2048] | None, add_mode: AddMode | None,
    review: ReviewMode | None = None,
) -> ProviderConfig:
    """The configuration to open ``workspace`` with: its own when it exists.

    The review is fixed when the store is created: its extraction policy is released
    once, and a reopen under another one would be refused by the bootstrap. A store
    moves to another review by ``quotemem rebuild --review`` into a new data directory.

    Args:
        workspace: The agent workspace directory.
        dimensions: A requested dimension; None takes the store's (1024 when new).
        add_mode: A requested Add mode; None takes the store's (``AGENT_ADD_MODE`` when
            new; ``DEFAULT_ADD_MODE`` for an existing store that stored none).
        review: A requested review (``resolve_review``); None takes the store's
            (``standard`` when new).

    Raises:
        QuoteMemError: ``workspace_binding_mismatch`` when a request differs from an
            existing store; ``schema_invalid`` for a review a new store's Add mode
            cannot take.
    """
    stored = stored_provider_config(workspace)
    if stored is None:
        return agent_provider_config(dimensions=dimensions or 1024, add_mode=add_mode,
                                     review=review)
    if dimensions is not None and dimensions != stored.embedding_dimensions:
        raise QuoteMemError("workspace_binding_mismatch", "The store's embedding dimension differs")
    if add_mode is not None and add_mode != (stored.add_mode or DEFAULT_ADD_MODE):
        raise QuoteMemError("workspace_binding_mismatch",
                            "The store was created with another Add mode")
    if review is not None and review != stored.review_mode:
        raise QuoteMemError(
            "workspace_binding_mismatch",
            f"This store was created with review {stored.review_mode}, which is fixed for "
            f"the store; unset {REVIEW_VARIABLE} and --review (or set {stored.review_mode}) "
            f"to keep it, or switch with `quotemem rebuild --from <this data dir> --to "
            f"<new data dir> --review {review}`",
        )
    return stored


def describe_binding(binding: AgentBindingV1) -> dict[str, object]:
    """A JSON-ready view of the binding."""
    return cast(dict[str, object], json.loads(binding.model_dump_json()))
