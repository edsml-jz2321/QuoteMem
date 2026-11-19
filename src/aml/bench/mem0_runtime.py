"""Task-owned optional Mem0 runtime and bounded, observed compatible HTTP clients.

No Mem0/OpenAI import or credential read occurs until explicit runtime creation.
The AML dependency lock and provider defaults remain unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import time
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx

from aml.bench.datasets import decode_source_json
from aml.bench.mem0_binding import MEM0_VERSION, Mem0Endpoint, Mem0Registration, mem0_digest
from aml.bench.resources import ProviderCallObservation
from aml.core.errors import AMLError


class Mem0Memory(Protocol):
    """The pinned library's synchronous Add/Search surface, without optional imports."""

    def add(
        self, messages: list[dict[str, str]], *, user_id: str,
        metadata: dict[str, object], infer: bool,
    ) -> object: ...

    def search(
        self, query: str, *, top_k: int, filters: dict[str, str],
        threshold: float, rerank: bool, explain: bool,
    ) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Mem0ProviderCall:
    """Payload-free HTTP observation, with the retained scalar witness used by accounting."""

    sequence: int
    purpose: Literal["extraction", "embedding"]
    operation_id: str
    request_sha256: str
    response_sha256: str | None
    requested_model: str
    resolved_model: str | None
    provider_request_id: str | None
    status_code: int | None
    elapsed_ms: float
    failure: str | None
    observation: ProviderCallObservation


class Mem0Ledger:
    """One run's actual HTTP attempts and conservative stop state.

    Call/byte limits apply before further dispatch. Token ceilings stop after
    observed responses, so an in-flight response may exceed a token ceiling.
    Billing amounts remain unknown. Local library work needs the outer job limit.
    """

    def __init__(self, registration: Mem0Registration) -> None:
        self.registration = registration
        self.calls: list[Mem0ProviderCall] = []
        self.phase: Literal["add", "search"] = "add"
        self.operation_id = "setup"
        self.failure: str | None = None
        self._lock = Lock()
        self._counts: dict[str, int] = {"extraction": 0, "embedding": 0}
        self._inputs: dict[str, int] = {"extraction": 0, "embedding": 0}
        self._outputs = 0
        self._deadline = time.monotonic() + registration.method.configuration.max_run_seconds

    def begin(self, phase: Literal["add", "search"], identity: str) -> int:
        """Bind later attempts to one sequential local Add or Search operation."""
        self.phase, self.operation_id = phase, identity
        return len(self.calls)

    def check(self) -> None:
        """Refuse more work after an accounting/transport failure or the run time limit."""
        if time.monotonic() >= self._deadline:
            self.failure = "run_budget_exhausted"
        if self.failure is not None:
            raise AMLError("baseline_incomplete", "Mem0 stopped at its registered boundary")

    def dispatch(self, purpose: Literal["extraction", "embedding"], request: httpx.Request) -> int:
        endpoint: Mem0Endpoint = getattr(self.registration.method.configuration, purpose)
        with self._lock:
            self.check()
            if self._counts[purpose] >= endpoint.max_calls:
                self.failure = "call_budget_exhausted"
                self.check()
            if (
                self._inputs[purpose] >= endpoint.max_observed_input_tokens
                or self._outputs
                >= self.registration.method.configuration.max_observed_output_tokens
            ):
                self.failure = "observed_token_budget_exhausted"
                self.check()
            path = "/chat/completions" if purpose == "extraction" else "/embeddings"
            body = decode_source_json(request.content)
            config = self.registration.method.configuration
            if (
                request.method != "POST" or str(request.url) != endpoint.base_url + path
                or len(request.content) > endpoint.max_request_utf8_bytes
                or not isinstance(body, dict) or body.get("model") != endpoint.requested_model
                or purpose == "extraction" and (
                    body.get("max_tokens") != config.max_output_tokens_per_call
                    or body.get("temperature") != config.temperature
                    or body.get("top_p") != config.top_p or body.get("stream", False) is not False
                )
                or purpose == "embedding" and body.get("dimensions") != config.embedding_dimensions
            ):
                self.failure = "request_binding_mismatch"
                self.check()
            self._counts[purpose] += 1
            index = sum(self._counts.values())
            seconds = min(endpoint.timeout_ms / 1000, max(0.001, self._deadline - time.monotonic()))
            request.extensions["timeout"] = dict.fromkeys(
                ("connect", "read", "write", "pool"), seconds,
            )
            return index

    def observe(
        self, purpose: Literal["extraction", "embedding"], index: int, request: httpx.Request,
        body: bytes | None, status: int | None, elapsed_ms: float, failure: str | None,
    ) -> None:
        """Retain only observed model/usage scalars; never estimate an absent charge."""
        endpoint: Mem0Endpoint = getattr(self.registration.method.configuration, purpose)
        model: str | None = None
        request_id: str | None = None
        entered: int | None = None
        emitted: int | None = None
        if body is not None:
            try:
                raw = decode_source_json(body)
                if isinstance(raw, dict):
                    model_value, request_id_value = raw.get("model"), raw.get("id")
                    model = model_value if isinstance(model_value, str) else None
                    request_id = request_id_value if isinstance(request_id_value, str) else None
                    usage = raw.get("usage")
                    if isinstance(usage, dict):
                        input_value = usage.get("prompt_tokens", usage.get("total_tokens"))
                        output_value = usage.get("completion_tokens")
                        entered = (
                            input_value if type(input_value) is int and input_value >= 0 else None
                        )
                        emitted = (
                            output_value if type(output_value) is int and output_value >= 0
                            else None
                        )
            except (ValueError, AMLError):
                failure = failure or "invalid_response"
        with self._lock:
            self._inputs[purpose] += entered or 0
            self._outputs += emitted or 0
            if status is not None and not 200 <= status < 300:
                failure = failure or "http_error"
            if failure is None:
                if model != endpoint.expected_resolved_model:
                    failure = "model_resolution_mismatch"
                elif entered is None or purpose == "extraction" and emitted is None:
                    failure = "usage_unknown"
                elif (
                    self._inputs[purpose] > endpoint.max_observed_input_tokens
                    or self._outputs
                    > self.registration.method.configuration.max_observed_output_tokens
                ):
                    failure = "observed_token_budget_exhausted"
            request_digest = hashlib.sha256(request.content).hexdigest()
            response_digest = hashlib.sha256(body).hexdigest() if body is not None else None
            witness = {
                "purpose": purpose, "operation_id": self.operation_id, "sequence": index,
                "request_sha256": request_digest, "response_sha256": response_digest,
                "requested_model": endpoint.requested_model, "resolved_model": model,
                "provider_request_id": request_id, "status_code": status,
                "elapsed_ms": elapsed_ms, "failure": failure,
                "input_tokens": entered, "output_tokens": emitted,
            }
            witness_digest = mem0_digest(witness)
            observation = ProviderCallObservation(
                call_key=f"mem0-http-{index:06d}", witness_sha256=witness_digest,
                provider_snapshot=f"{endpoint.provider}:{endpoint.requested_model}",
                phase=self.phase, cycle_key=None, outcome="failed" if failure else "succeeded",
                input_tokens=entered, output_tokens=emitted,
                usage_witness_sha256=(
                    witness_digest if entered is not None or emitted is not None else None
                ),
                cost_usd=None, cost_witness_sha256=None,
            )
            self.calls.append(Mem0ProviderCall(
                index, purpose, self.operation_id, request_digest, response_digest,
                endpoint.requested_model, model, request_id, status, elapsed_ms, failure,
                observation,
            ))
            self.failure = self.failure or failure


class Mem0Transport(httpx.BaseTransport):
    """Observe and bound the two SDK clients without changing Mem0's memory algorithm."""

    def __init__(
        self, ledger: Mem0Ledger, purpose: Literal["extraction", "embedding"],
        transport: httpx.BaseTransport,
    ) -> None:
        self.ledger = ledger
        self.purpose: Literal["extraction", "embedding"] = purpose
        self.transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        index = self.ledger.dispatch(self.purpose, request)
        start = time.monotonic()
        endpoint: Mem0Endpoint = getattr(
            self.ledger.registration.method.configuration, self.purpose,
        )
        body: bytes | None = None
        status: int | None = None
        failure: str | None = None
        try:
            response = self.transport.handle_request(request)
            status = response.status_code
            try:
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    self.ledger.check()
                    size += len(chunk)
                    if size > endpoint.max_response_utf8_bytes:
                        failure = "response_byte_budget_exhausted"
                        raise AMLError("baseline_incomplete", "Mem0 response exceeded its budget")
                    chunks.append(chunk)
                body = b"".join(chunks)
            finally:
                response.close()
            headers = httpx.Headers(response.headers)
            headers.pop("content-encoding", None)
            headers.pop("content-length", None)
            return httpx.Response(status, headers=headers, content=body, request=request)
        except Exception:
            failure = failure or "transport_error"
            raise
        finally:
            self.ledger.observe(
                self.purpose, index, request, body, status,
                (time.monotonic() - start) * 1000, failure,
            )

    def close(self) -> None:
        self.transport.close()


class _SDKClient(Protocol):
    def with_options(self, **kwargs: Any) -> _SDKClient: ...
    def close(self) -> None: ...


class _MemoryFactory(Protocol):
    def from_config(self, config: dict[str, object]) -> Mem0Memory: ...


def mem0_distribution_digest() -> str:
    """Bind every installed distribution name/version in the dedicated runtime.

    Args:
        None.

    Returns:
        Canonical digest of the complete sorted installed version inventory.
        Wheel installation and dependency-lock resolution remain separate evidence.
    """
    return mem0_digest(sorted(
        (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
        for distribution in importlib.metadata.distributions()
    ))


def _verify_install(registration: Mem0Registration, wheel: Path, dependency_lock: Path) -> None:
    method = registration.method
    try:
        installed_version = importlib.metadata.version("mem0ai")
    except importlib.metadata.PackageNotFoundError:
        raise AMLError(
            "provider_unavailable", "The pinned optional Mem0 runtime is not installed",
        ) from None
    if (
        hashlib.sha256(wheel.read_bytes()).hexdigest() != method.wheel_sha256
        or hashlib.sha256(dependency_lock.read_bytes()).hexdigest() != method.dependency_lock_sha256
        or installed_version != MEM0_VERSION
        or mem0_distribution_digest() != method.installed_distributions_sha256
    ):
        raise AMLError("baseline_invalid", "Mem0 wheel, lock or installed version differs")
    distribution = importlib.metadata.distribution("mem0ai")
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist()
                   if name.startswith("mem0/") and not name.endswith("/")]
        if not members or any(".." in Path(name).parts for name in members):
            raise AMLError("baseline_invalid", "Mem0 wheel has invalid package paths")
        for name in members:
            if Path(str(distribution.locate_file(name))).read_bytes() != archive.read(name):
                raise AMLError(
                    "baseline_invalid", "Installed Mem0 differs from the registered wheel",
                )
    if any(importlib.util.find_spec(name) is not None for name in ("fastembed", "spacy")):
        raise AMLError(
            "baseline_invalid", "This method requires the registered minimal Mem0 runtime",
        )


@contextmanager
def open_mem0(
    registration: Mem0Registration, *, wheel: Path, dependency_lock: Path,
    environment: Mapping[str, str],
) -> Iterator[tuple[Mem0Memory, Mem0Ledger]]:
    """Open one fresh pinned optional store and explicit bounded compatible clients.

    Args:
        registration: Verified frozen method and input registration.
        wheel: Exact retained Mem0 wheel installed in this task-owned runtime.
        dependency_lock: Exact retained runtime lock, whose bytes are verified here.
        environment: Credential source; secrets never enter serialized configuration.

    Yields:
        The owned memory and HTTP ledger. Provider calls require explicit caller Add/Search.

    Raises:
        AMLError: Provenance, freshness, credentials or supported runtime conditions differ.
        ImportError: The optional pinned runtime has not been installed.
        OSError: A fresh task-owned path or retained artifact cannot be accessed.
    """
    if (
        "mem0" in sys.modules or environment.get("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    ):
        raise AMLError(
            "baseline_invalid", "Mem0 requires a fresh process without provider overrides",
        )
    _verify_install(registration, wheel, dependency_lock)
    config = registration.method.configuration
    credentials = [environment.get(endpoint.credential_env, "").strip()
                   for endpoint in (config.extraction, config.embedding)]
    if not all(credentials):
        raise AMLError("provider_unavailable", "Mem0 requires its explicitly named credentials")
    proxy = environment.get(config.proxy_env) if config.proxy_env is not None else None
    if config.proxy_env is not None:
        parts = urlsplit(proxy or "")
        if (
            parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment
        ):
            raise AMLError("baseline_invalid", "Mem0 requires its explicit credential-free proxy")
    workspace = Path(config.workspace)
    if not workspace.is_absolute():
        raise AMLError("baseline_invalid", "Mem0 workspace must be an absolute fresh path")
    workspace.mkdir(parents=False, exist_ok=False)
    previous = {key: os.environ.get(key) for key in ("MEM0_DIR", "MEM0_TELEMETRY")}
    os.environ.update(MEM0_DIR=str(workspace / "home"), MEM0_TELEMETRY="false")
    memory: Mem0Memory | None = None
    clients: list[_SDKClient] = []
    ledger = Mem0Ledger(registration)
    try:
        module = importlib.import_module("mem0")
        installed = importlib.metadata.distribution("mem0ai").locate_file("mem0/__init__.py")
        if module.__file__ is None or (
            Path(module.__file__).resolve() != Path(str(installed)).resolve()
        ):
            raise AMLError("baseline_invalid", "Mem0 import is outside the verified installation")
        factory = cast(_MemoryFactory, module.Memory)
        memory = factory.from_config({
            "version": "v1.1", "history_db_path": str(workspace / "history.sqlite3"),
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": config.collection_name, "path": str(workspace / "qdrant"),
                "embedding_model_dims": config.embedding_dimensions, "on_disk": True,
            }},
            "llm": {"provider": "openai", "config": {
                "model": config.extraction.requested_model, "api_key": credentials[0],
                "openai_base_url": config.extraction.base_url, "temperature": config.temperature,
                "top_p": config.top_p, "max_tokens": config.max_output_tokens_per_call,
                "is_reasoning_model": False, "enable_vision": False,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": config.embedding.requested_model, "api_key": credentials[1],
                "openai_base_url": config.embedding.base_url,
                "embedding_dims": config.embedding_dimensions,
            }},
            "reranker": None, "custom_instructions": None,
        })
        for owner_name, purpose in (("llm", "extraction"), ("embedding_model", "embedding")):
            owner = getattr(memory, owner_name)
            original = cast(_SDKClient, owner.client)
            endpoint = getattr(config, purpose)
            transport = Mem0Transport(
                ledger, cast(Literal["extraction", "embedding"], purpose),
                httpx.HTTPTransport(proxy=proxy, trust_env=False),
            )
            client = original.with_options(
                max_retries=0, timeout=endpoint.timeout_ms / 1000,
                http_client=httpx.Client(
                    transport=transport, follow_redirects=False, trust_env=False,
                ),
            )
            owner.client = client
            clients.append(client)
            original.close()
        yield memory, ledger
    finally:
        close_error = False
        if memory is not None:
            try:
                memory.close()
            except Exception:
                close_error = True
            seen: set[int] = set()
            for name in ("vector_store", "_entity_store"):
                store = getattr(memory, name, None)
                client = getattr(store, "client", None)
                if client is not None and id(client) not in seen:
                    seen.add(id(client))
                    try:
                        client.close()
                    except Exception:
                        close_error = True
        for client in clients:
            try:
                client.close()
            except Exception:
                close_error = True
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if close_error:
            raise AMLError("baseline_incomplete", "Mem0 resource closure failed") from None
