"""Server smoke test for CI: start `quotemem serve` against a local stub provider.

The stub answers the OpenAI-compatible `/embeddings` and `/chat/completions` calls
that the startup probe makes, so the test needs no network and no API key. It checks:

- `quotemem check` passes against the stub;
- `quotemem serve` binds, probes, opens a new store and reports `serving`;
- `/agent/ready`, `/agent/status`, `/health` and `/memory/search` (the AMLB `/search`
  body) answer;
- a request without the token is refused;
- an asynchronous `/memory/add` is accepted with 202;
- the service drains and exits on SIGINT. On Windows, where a console program cannot
  be sent a Ctrl+C on its own, it is stopped with Ctrl+Break in its own process group,
  and the check is that it stops.

Usage: python .github/ci/server_smoke.py   (run inside the project environment)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

CHAT_MODEL = "stub-chat"
EMBED_MODEL = "stub-embed"
START_TIMEOUT = 180.0
STOP_TIMEOUT = 90.0
WINDOWS = os.name == "nt"
# How Windows ends a console program that a Ctrl+Break stops (STATUS_CONTROL_C_EXIT).
WINDOWS_BREAK_EXIT = 0xC000013A


def _vector(text: str, dimensions: int) -> list[float]:
    """A deterministic unit vector for one text."""
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    values = values[:dimensions]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class StubProvider(BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible provider for the startup probe."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _reply(self, status: int, document: dict[str, Any]) -> None:
        body = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        model = request.get("model")
        if self.path.endswith("/embeddings"):
            texts = request.get("input")
            texts = [texts] if isinstance(texts, str) else list(texts or [])
            dimensions = int(request.get("dimensions") or 1024)
            self._reply(200, {
                "object": "list", "model": model,
                "data": [
                    {"object": "embedding", "index": index,
                     "embedding": _vector(str(text), dimensions)}
                    for index, text in enumerate(texts)
                ],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            })
        elif self.path.endswith("/chat/completions"):
            self._reply(200, {
                "id": "stub", "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        else:
            self._reply(404, {"error": {"message": "not found"}})


def _request(
    url: str, *, token: str | None = None, body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | None, dict[str, str]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            status, reply_headers = response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw, status, reply_headers = error.read(), error.code, dict(error.headers)
    try:
        document = json.loads(raw) if raw else None
    except ValueError:
        document = None
    return status, document if isinstance(document, dict) else None, reply_headers


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"SMOKE FAIL: {message}")
    print(f"ok: {message}", flush=True)


def _environment(stub_url: str, token: str) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("QUOTEMEM_", "OPENAI_", "DASHSCOPE_"))
    }
    environment.update({
        "QUOTEMEM_CHAT_BASE_URL": stub_url, "QUOTEMEM_CHAT_MODEL": CHAT_MODEL,
        "QUOTEMEM_CHAT_API_KEY": "stub-key", "QUOTEMEM_CHAT_STRUCTURED_OUTPUT": "json_object",
        "QUOTEMEM_EMBED_MODEL": EMBED_MODEL, "QUOTEMEM_TOKEN": token,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    })
    return environment


def main() -> None:
    quotemem = shutil.which("quotemem")
    _check(quotemem is not None, "the quotemem command is installed")
    assert quotemem is not None
    stub = ThreadingHTTPServer(("127.0.0.1", 0), StubProvider)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    stub_url = f"http://127.0.0.1:{stub.server_address[1]}/v1"
    token = secrets.token_hex(24)
    environment = _environment(stub_url, token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)
    root = Path(tempfile.mkdtemp(prefix="qm-smoke-"))
    data_dir = root / "data"
    try:
        for command in (["--help"], ["bench", "serve", "--help"], ["serve", "--help"]):
            result = subprocess.run([quotemem, *command], env=environment, capture_output=True,
                                    text=True, timeout=120)
            _check(result.returncode == 0, f"quotemem {' '.join(command)} exits 0")
        check = subprocess.run([quotemem, "check", "--data-dir", str(data_dir)],
                               env=environment, capture_output=True, text=True, timeout=180)
        _check(check.returncode == 0, "quotemem check passes against the stub provider")
        _check(not data_dir.exists(), "quotemem check creates nothing")

        server = subprocess.Popen(
            [quotemem, "serve", "--data-dir", str(data_dir), "--port", "0"],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
        )
        lines: list[str] = []
        port: int | None = None
        stdout = server.stdout
        assert stdout is not None

        def read_output() -> None:
            for line in stdout:
                lines.append(line.rstrip("\n"))

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + START_TIMEOUT
            while port is None and time.monotonic() < deadline and server.poll() is None:
                for line in list(lines):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and isinstance(event.get("port"), int) \
                            and event["port"] > 0:
                        port = event["port"]
                time.sleep(0.2)
            _check(port is not None, "quotemem serve prints its port")
            base = f"http://127.0.0.1:{port}"
            ready = 0
            while time.monotonic() < deadline and server.poll() is None:
                ready, _, _ = _request(base + "/agent/ready")
                if ready == 200:
                    break
                time.sleep(0.5)
            _check(ready == 200, "/agent/ready answers 200")

            status, document, _ = _request(base + "/agent/status", token=token)
            _check(status == 200 and document is not None
                   and document.get("phase") == "serving", "/agent/status reports serving")
            status, _, _ = _request(base + "/health", token=token)
            _check(status == 200, "/health answers 200")
            status, _, _ = _request(base + "/memory/search",
                                    body={"query": "where do I live?"})
            _check(status in {401, 403}, "a request without the token is refused")
            status, document, _ = _request(
                base + "/memory/search", token=token,
                body={"user_id": "smoke", "query": "Where do I live?", "top_k": 5},
            )
            _check(status == 200 and document is not None,
                   "/memory/search answers 200 on a new store")
            status, document, _ = _request(
                base + "/memory/add", token=token, headers={"Prefer": "respond-async"},
                body={"user_id": "smoke", "session_id": "s1", "request_id": "smoke-1",
                      "messages": [{"role": "user", "content": "I moved to Lisbon last spring.",
                                    "timestamp": "2026-01-01T00:00:00Z"}]},
            )
            _check(status == 202 and document is not None
                   and document.get("status") == "pending",
                   "an asynchronous /memory/add answers 202")
        finally:
            stop = "Ctrl+Break" if WINDOWS else "SIGINT"
            stopped = {0, WINDOWS_BREAK_EXIT} if WINDOWS else {0, -signal.SIGINT, 130}
            if server.poll() is None:
                server.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
            try:
                code = server.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                server.kill()
                code = server.wait()
                print("\n".join(lines[-40:]), file=sys.stderr)
                raise SystemExit(f"SMOKE FAIL: the service did not stop after {stop}") from None
            reader.join(timeout=5)
            if code not in stopped:
                print("\n".join(lines[-40:]), file=sys.stderr)
            _check(code in stopped, f"the service stops on {stop} (exit {code})")
    finally:
        stub.shutdown()
        shutil.rmtree(root, ignore_errors=True)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
