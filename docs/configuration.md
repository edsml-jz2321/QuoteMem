# Configuration

## Review

The review decides whether a new memory repeats, updates or corrects a stored
one. Set `QUOTEMEM_REVIEW` (or pass `--review`) to choose it for a new store:

| Review | What decides | Judge model calls per Add |
|---|---|---|
| `fast` | Local rules only. A change of state that the user says outright ("Hangzhou, not Beijing") retires the old memory. | none |
| `standard` (default) | The judge model reviews a new memory that may change a stored one ([which](architecture.md#review)); local rules review the rest. | usually one per review round, plus votes |
| `full` | The judge model reviews every new memory. | about one per four new memories, plus votes |

- A judge verdict that would end a current memory, or deny a new memory as
  a forgotten fact, is confirmed by votes: that memory is sent alone again,
  up to two more times, and the verdict that two answers share decides.
- Every review keeps the other rules of the `agent` Add mode: forgetting,
  history and status work the same.
- Only the `agent` Add mode lets you choose. The `lean-local` Add mode always
  reviews `fast`, and the `full` Add mode always reviews `full`. In those
  modes, a new store given any other `--review` or `QUOTEMEM_REVIEW` value is
  refused with `schema_invalid`.
- The review is fixed per store. Starting an existing store, in any Add mode,
  with another review is refused with `workspace_binding_mismatch`, which
  names the store's review.
  Leave `QUOTEMEM_REVIEW` unset (or set it to the store's review) to keep it.
- To switch, stop the service and rebuild the store into a new data
  directory:
  `quotemem rebuild --from <old data dir> --to <new data dir> --review full`.
  The rebuild takes `--review`, else `QUOTEMEM_REVIEW`, else `standard`; it
  never copies the old store's review.
- `GET /agent/status` and `quotemem check` report the store's review as
  `store.review`.

Secrets are read only from environment variables. Command-line flags, and the
optional `--provider-config` JSON file of `quotemem bench serve`, hold
non-secret settings only.

## `quotemem serve`

| Variable | Meaning | Default |
|---|---|---|
| `QUOTEMEM_CHAT_BASE_URL` | Chat service root (`https`, or `http` on loopback) | `OPENAI_BASE_URL` |
| `QUOTEMEM_CHAT_MODEL` | Chat model name | `OPENAI_MODEL` |
| `QUOTEMEM_CHAT_API_KEY` | Chat key | `OPENAI_API_KEY` |
| `QUOTEMEM_CHAT_STRUCTURED_OUTPUT` | `json_schema` (strict schemas) or `json_object` (JSON mode) | `json_schema` |
| `QUOTEMEM_CHAT_REASONING` | `off` asks every call for no extended thinking; `high` asks every call for thinking (see below); `none` sends no reasoning parameters, so the provider's default applies; `effort` sends only `reasoning_effort`; `passthrough` sends the service's own reasoning parameters | `off` |
| `QUOTEMEM_CHAT_REASONING_TOKENS` | Under `high`: the reasoning tokens added to every call's output bound (1024 to 131072) | `24576` |
| `QUOTEMEM_CHAT_REASONING_SECONDS` | Under `high`: the longest wait for one reasoning reply, in seconds (10 to 900) | `120` |
| `QUOTEMEM_CHAT_MODEL_PATTERN` | A regular expression that the model id of every chat reply must fully match | the model, optionally with a date suffix |
| `QUOTEMEM_EMBED_BASE_URL` | Embedding service root; local servers such as Ollama work on loopback `http` | the chat URL |
| `QUOTEMEM_EMBED_MODEL` | Embedding model name; bound to the store on first start | required |
| `QUOTEMEM_EMBED_API_KEY` | Embedding key | `DASHSCOPE_API_KEY`, then the chat key |
| `QUOTEMEM_EMBED_SEND_DIMENSIONS` | Set `false` for embedding models without a `dimensions` parameter | `true` |
| `QUOTEMEM_EMBED_MODEL_PATTERN` | A regular expression that the model id of every embedding reply must fully match | the model, optionally with a date suffix |
| `QUOTEMEM_TOKEN` | HTTP bearer token (`QUOTEMEM_HTTP_AUTH_SECRET` also works) | generated into `<data-dir>/token` |
| `QUOTEMEM_DATA_DIR` | Data directory | `$XDG_DATA_HOME/quotemem` or `~/.local/share/quotemem` |
| `QUOTEMEM_DAILY_BUDGET_USD` | Most one UTC day may spend on provider calls; needs `QUOTEMEM_CHAT_PRICE_INPUT_PER_MTOK`, `QUOTEMEM_CHAT_PRICE_OUTPUT_PER_MTOK` and `QUOTEMEM_EMBED_PRICE_PER_MTOK` (US dollars per million tokens) | uncapped |
| `QUOTEMEM_DAILY_CHAT_CALLS`, `QUOTEMEM_DAILY_EMBED_CALLS` | Most calls one role may make in one UTC day | uncapped |
| `QUOTEMEM_SCRUB_PII` | `1` replaces email addresses and phone numbers in every text sent to a provider | off |
| `QUOTEMEM_AUTO_REPAIR` | `resume` or `skip`: repair quarantined users automatically | off |
| `QUOTEMEM_REVIEW` | The review of a new store: `fast`, `standard` or `full` ([Review](#review)); fixed per store; `--review` overrides it | `standard` |

Thinking (`QUOTEMEM_CHAT_REASONING`):

- `off` sends `thinking: {"type": "disabled"}` on every call.
- `high` sends `thinking: {"type": "enabled"}` and
  `reasoning_effort: "high"` on every call. Thinking counts toward the output
  bound, so `QUOTEMEM_CHAT_REASONING_TOKENS` is added to it.
- Under `off` and `high`, once the provider answers 400 naming `thinking`,
  later calls leave it out and that call is sent again without it. Under
  `high`, `reasoning_effort` is still sent.
- Under `high`, a reasoning reply that its bound cuts off, or that takes longer
  than `QUOTEMEM_CHAT_REASONING_SECONDS`, is asked again without thinking.
  Thinking thus never truncates a reply or loses a fact.
- A reasoning attempt also waits at most half of the call's remaining time,
  and leaves at least 30 seconds for the resend. A call with too little time
  for that goes without thinking.

Flags: `--data-dir`, `--host`, `--port` (0 picks a free port and prints it),
`--allow-remote`, `--user-id` (the user of requests that name none,
`default`), `--default-top-k` (10), `--request-timeout-seconds` (the budget
of one Add or Search, 600), `--search-deadline-seconds` (2.5),
`--search-profile {interactive,bench}` (`interactive`), `--concurrency
{interactive,bench}` (`interactive`: Search never waits for an Add) and, for a
new store only, `--embedding-dimensions {1024,2048}`,
`--add-mode {agent,lean-local,full}` (`agent`) and
`--review {fast,standard,full}` (`QUOTEMEM_REVIEW`, else `standard`).
Starting an existing store with another embedding model is refused; use a new
data directory or `quotemem rebuild`.

## `quotemem bench serve`

| Variable | Needed for | Notes |
|---|---|---|
| `DASHSCOPE_API_KEY` | Embeddings | Model `qwen3.7-text-embedding` through the Model Studio OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | Extraction and review | Chat model `deepseek-v4.1-flash`, the gateway's alias for its current dated snapshot (now `deepseek-v4-1-flash-261017`, the model DashScope names `deepseek-v4.1-flash-1017`). A reply must name a dated `deepseek-v4-1-flash-YYMMDD` snapshot or `deepseek-v4.1-flash-1017`. |
| `OPENAI_BASE_URL` | Extraction and review | HTTPS root of an OpenAI-compatible gateway that serves that model. `api.openai.com` is refused for it. |
| `QUOTEMEM_HTTP_AUTH_SECRET` | The service | Non-empty printable ASCII |
| `QUOTEMEM_HTTP_AUTH_MODE` | The service | `bearer` (default), `token` or `x-api-key` |
| `QUOTEMEM_JOB_ID` | The service | Non-empty identifier |

Useful flags:

- `--add-mode {lean-local,full,agent}` sets the Add mode of a new
  workspace (default `lean-local`; `agent` is the default of
  `quotemem serve`). The mode is frozen when the workspace is created;
  reopening it with a different mode is refused.
- `--embedding-dimensions {1024,2048}` sets the vector size of a new workspace.
