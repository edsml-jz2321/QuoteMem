# Run the memory service

This page covers installing the Python package, running `quotemem serve` and its HTTP routes.

## Install

Requirements:

- Python 3.12 or newer. Tested on Python 3.12, 3.13 and 3.14.
- [uv](https://docs.astral.sh/uv/) 0.11.21 or later for `uvx` or a clone.
- HeavenBase (MIT), which stores the data, is included (the published 0.1.2.3
  release, in `src/quotemem/_vendor`); you do not install it yourself. Do not
  import another HeavenBase in the same Python process. The store is two
  SQLite files in the `workspace/` folder of the data directory;
  [How QuoteMem works](architecture.md) describes them.

Run the service without installing it, or install the `quotemem` command:

```bash
uvx quotemem serve       # run it straight from PyPI
pip install "quotemem>=0.3"   # or install it, then run `quotemem serve`
```

Or install from a clone:

```bash
git clone https://github.com/edsml-jz2321/QuoteMem.git
cd QuoteMem
uv sync                  # core service
uv sync --extra bench    # also the benchmark tools (pyarrow, tiktoken)
```

From a clone, run each `quotemem` command below as `uv run quotemem ...`.

`pyproject.toml` pins only the direct
dependencies, so `uv sync` resolves the newest allowed versions of the
transitive ones. CI also pins those, with
[`.github/ci/constraints.txt`](../.github/ci/constraints.txt); to install that
exact set, run `uv venv` and then
`uv pip install --constraints .github/ci/constraints.txt -e ".[bench]"`.

## Run the HTTP service

There are two ways to run the service:

- `quotemem serve` is the long-lived memory service for agents and the
  plugins below. Start here.
- `quotemem bench serve` is the benchmark service with the Agent Memory
  Leaderboard (AMLB) `/add` and `/search` routes (see
  [Benchmark service](benchmarks.md#benchmark-service-quotemem-bench-serve)).

## Quickstart: `quotemem serve`

The service works with any OpenAI-compatible chat and embedding endpoints.
Keys are read only from environment variables; never put them on the command
line. `DASHSCOPE_API_KEY` is an Alibaba Cloud Model Studio key, from
[bailian.console.aliyun.com](https://bailian.console.aliyun.com) (China region).

```bash
export QUOTEMEM_CHAT_BASE_URL=https://api.deepseek.com/v1   # any OpenAI-compatible chat API
export QUOTEMEM_CHAT_MODEL=deepseek-chat
export QUOTEMEM_CHAT_API_KEY=...
export QUOTEMEM_CHAT_STRUCTURED_OUTPUT=json_object          # for providers without strict JSON schemas
export QUOTEMEM_EMBED_BASE_URL=https://api.openai.com/v1    # defaults to the chat URL
export QUOTEMEM_EMBED_MODEL=text-embedding-3-small          # bound to the store on first start
export QUOTEMEM_EMBED_API_KEY=...                           # else DASHSCOPE_API_KEY, else the chat key

quotemem check   # read-only: checks the settings and probes both providers
quotemem serve   # http://127.0.0.1:8765; the bearer token goes to <data-dir>/token
```

- The service URL is `http://127.0.0.1:8765` by default; `--port` changes
  the port.
- The service listens on loopback only, unless `--allow-remote` is given.
- Data lives in `QUOTEMEM_DATA_DIR` or `--data-dir` (default
  `$XDG_DATA_HOME/quotemem` or `~/.local/share/quotemem`). One store serves
  every `user_id`, each in its own scope.
- To delete a whole data folder, stop the service first. The store keeps
  some of its folders read-only, so a plain `rm -rf` stops with "Permission
  denied"; make them writable first:
  `chmod -R u+w <data-dir> && rm -rf <data-dir>`. On Windows,
  `Remove-Item -Recurse -Force <data-dir>` removes it as it is. To remove one
  user's memories, use `quotemem delete-user` instead.
- Without a token in the environment, one is generated into
  `<data-dir>/token` (mode 0600) on the first start. Clients and the commands
  below read it from there. Do not export the token in the shell that starts
  an agent: the agent's own shell tool would inherit it.
- The port is bound before the providers are probed. Until the store is open,
  the memory routes and `GET /health` answer 503 with `Retry-After`, and
  `GET /agent/status` says what is failing and how to fix it.
- Stop it with Ctrl-C; it drains in-flight requests before it exits.

Write a conversation turn, then search it. The token file is `token` in the
data directory; with `--data-dir`, use that directory instead. If
`QUOTEMEM_TOKEN` or `QUOTEMEM_HTTP_AUTH_SECRET` is set, the service uses that
token and writes no file; send that value instead.

```bash
TOKEN_FILE="${QUOTEMEM_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/quotemem}/token"

curl -s http://127.0.0.1:8765/memory/add \
  -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "session_id": "s1", "request_id": "r1",
       "messages": [{"role": "user", "content": "I moved to Lisbon last spring."}]}'

curl -s http://127.0.0.1:8765/memory/search \
  -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "query": "Where do I live?"}'
```

- `/memory/add` returns after the new memories are stored and searchable.
  With the header `Prefer: respond-async` it answers `202` as soon as the
  turns are durably saved; send the same request again to read the result.
- `/memory/add` and `/memory/search` take the AMLB `/add` and `/search`
  bodies, relaxed for ordinary chat: `user_id`, `session_id`, `request_id`
  and `top_k` may be omitted.
- In the `agent` Add mode (the default), a `user` message is the user's own
  words. A `speaker` or `source_participant` set on it is ignored, so
  first-person facts and updates always bind to the user. Put another person's
  words in the content with a line label (`Alice: I moved to Lisbon.`). An
  `assistant` message keeps its `speaker`. The `lean-local` and `full` Add
  modes keep the `speaker` of every message. `quotemem rebuild` applies the
  rule of the new store's Add mode. The MCP tool `quotemem_add` sends a
  `speaker` only on `assistant` messages, in every Add mode.
- A successful Add also carries an `agent` object: `status` (`stored`, or
  `partial` when a fact was rejected or left unresolved), `reasons` (the
  reason codes of those facts, with counts), `items` (each fact's category and
  reason code), `added` (how many new memories a read still shows) and `forgets`
  (what each forget request in the chat did).
- Search makes no model call besides the query embedding and keeps to a
  2.5-second deadline. Each memory hit carries a `Status:` line (`current`,
  `past` or `possibly_outdated`).

Routes:

| Route | Purpose |
|---|---|
| `POST /memory/add`, `POST /memory/search` | Write turns; search memories (AMLB `/add` and `/search` bodies) |
| `POST /memory/history` | Version history of a memory |
| `POST /memory/provenance` | The quotes and the review verdict behind one version |
| `POST /memory/add-results` | Each fact of one Add, with its reason code and memories |
| `POST /memory/list` | One page of a user's memories, with their status and quotes ([List the memories](cli.md#list-the-memories)) |
| `POST /memory/forget/prepare` | Step 1 of a forget: the body names exactly one of `statement` (the user's words) or `memory_id` with `expected_revision_id` |
| `POST /memory/forget/confirm` | Step 2: send back the prepare's `confirmation` unchanged |
| `GET /agent/ready` | 200 once serving (no token) |
| `GET /agent/status` | Phase, store binding, provider health and counters, without content |
| `POST /agent/admin/delete-user` | Erase one user |
| `POST /agent/admin/repair` | Inspect, resume or skip a quarantined user |
| `GET /health` | 200 once the store is open and accepts traffic, 503 before |

- The bodies of three read routes:
  - history: `{"schema_version": 3, "user_id", "memory_id"}`;
  - provenance:
    `{"schema_version": 2, "user_id", "target": {"kind": "memory", "memory_id"}}`;
  - add-results: `{"schema_version": 1, "user_id", "session_id", "request_id"}`.
- History and provenance answer 404 `memory_not_found` for an id that names
  no memory of the user: an unknown id, a forgotten memory, or the id of a
  message hit from Search.
- A forget prepare that names both `statement` and `memory_id`, or neither,
  answers 422 `invalid_request`. A prepare by statement that matches no
  memory returns `"status": "nothing_matched"` and `"confirmation": null`.
- Forget bodies also carry `schema_version: 1`, `user_id`,
  `client_session_id` and `request_id`. Send the confirm before the
  prepare's `expires_at`.
- The bench route names (`/add`, `/search`, `/memory/why`,
  `/memory/add-items`, `/memory/forget`, `/memory/forget/statement/prepare`
  and `/memory/forget/statement`) answer 422 `invalid_request` here; there
  are no aliases. `quotemem bench serve` keeps them.

Other commands, all with `--help`:

- `quotemem export` writes a backup of the store's source messages and
  forgets; `quotemem rebuild --from OLD --to NEW` replays one into a new store.
  A store is never migrated in place.
- `quotemem delete-user --user-id X` erases one user.
- `quotemem repair --user-id X` inspects (`--inspect`), resumes or skips
  (`--skip`) a quarantined user. A forget that could not finish holds its user
  until it does; the service log names its scope and `/agent/status` lists the
  user under `forget_attention_users`. `repair` (with or without `--skip`) runs
  that forget again first; it is never skipped.
- `delete-user` and `repair` call the running service
  ([Reaching the service](cli.md#reaching-the-service)).
- `quotemem memories` lists one user's memories ([List the memories](cli.md#list-the-memories)).
- Each command exits 0 on success, 1 on a failure and 2 on a usage error.
- `install`, `mcp`, `hook`, `doctor`, `setup`, `status` and `stop` are commands
  of the npm package `quotemem`. The Python command answers them with
  ``that is a plugin command: run `npx quotemem <command>` `` on stderr and
  exits 2.
