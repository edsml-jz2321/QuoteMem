# Changelog

All notable changes to QuoteMem are listed here. Dates are UTC.

## 0.3.0 - Unreleased

Version 0.2.0 was not released.

### Memory service (Python package and command `quotemem`)

- **HeavenBase is included.** The store, HeavenBase (MIT; a pre-release build of
  0.1.2.3), comes with the package and is not installed separately.
- **One name: `quotemem`.** In 0.1.0 the Python package, its import name and
  its command were `aml`, and the service's variables used the `AML_` prefix;
  they are now `quotemem` and `QUOTEMEM_`. The memory service's commands are
  `quotemem serve`, `check`, `memories`, `export`, `rebuild`, `delete-user`
  and `repair`. The benchmark commands are `quotemem bench serve`, `init`,
  `route`, `status` and `purge`, and the benchmark tools run as
  `python -m quotemem.bench.cli`.
  Every environment variable that the service defines uses the `QUOTEMEM_`
  prefix, for example `QUOTEMEM_HTTP_AUTH_SECRET`, `QUOTEMEM_HTTP_AUTH_MODE`
  and `QUOTEMEM_JOB_ID`. The provider variables it also reads keep their
  names: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` and
  `DASHSCOPE_API_KEY`. For `install`, `mcp`, `hook`, `doctor`, `setup`, `status`
  and `stop`, the Python command prints a hint to run `npx quotemem <command>`
  and exits 2. A store written by 0.1.0 is not opened: start with a new data
  directory.
- **`quotemem serve`.** A long-lived memory service for agents and the plugins.
  It works with any OpenAI-compatible chat and embedding endpoints, configured
  through `QUOTEMEM_*` environment variables, and keys are read only from the
  environment. It listens on loopback unless `--allow-remote` is given. Without
  a token in the environment it generates one into `<data-dir>/token`.
  `GET /agent/ready` and `GET /agent/status` report readiness and health; until
  the store is open the memory routes answer 503 with `Retry-After`.
  `quotemem check` checks the settings and probes both providers read-only.
- **Memory routes under `/memory/*`.** `quotemem serve` serves
  `/memory/add`, `/memory/search`, `/memory/history`, `/memory/provenance`,
  `/memory/add-results`, `/memory/list` and one forget pair,
  `/memory/forget/prepare` and `/memory/forget/confirm`. A forget prepare
  names exactly one of `statement` or `memory_id`. The benchmark route names
  answer 422 `invalid_request` there; `quotemem bench serve` keeps them.
- **Agent Add mode.** The new Add mode `agent`, the default of
  `quotemem serve`, sends a new memory to a judge model only when it may
  repeat, update or correct a stored one, and reviews the rest with local rules.
  `quotemem bench serve` keeps `lean-local` as its default and still accepts `full`.
- **Review choice.** `QUOTEMEM_REVIEW` (or `--review`) sets how a new `agent`
  store reviews: `fast` (local rules, no judge model call), `standard` (the
  default, as above) or `full` (the judge model for every new memory). The
  review is fixed per store; `quotemem rebuild --review` moves a store to
  another one. `GET /agent/status` and `quotemem check` report it.
- **Thinking.** `QUOTEMEM_CHAT_REASONING` takes `high`: every chat call asks
  for thinking at high effort. `QUOTEMEM_CHAT_REASONING_TOKENS` (default 24576)
  is added to each call's output bound, and a reasoning reply that is cut off
  or slower than `QUOTEMEM_CHAT_REASONING_SECONDS` (default 120) is asked again
  without thinking. The default stays `off`.
- **Memory list.** `POST /memory/list` returns one page of a user's memories
  with their status and quotes (`status`, `limit` up to 200, `cursor`).
  `quotemem memories` prints the list as text, `--json` or one read-only
  `--html` page, from the running service or, when it is stopped, read-only
  from its store. Nothing forgotten is listed.
- **Interactive Search.** Search makes no model call besides the query embedding
  and keeps to a deadline (2.5 seconds by default, `--search-deadline-seconds`).
  Each memory hit carries a `Status:` line (`current`, `past` or
  `possibly_outdated`). Search never waits for an Add. `--search-profile bench`
  selects the benchmark Search instead.
- **Asynchronous Add.** With `Prefer: respond-async`, `/memory/add` (`/add` on
  `quotemem bench serve`) answers `202` as soon as the turns are durably saved;
  sending the same request again reads the result. `/memory/add` and
  `/memory/search` take the AMLB `/add` and `/search` bodies, relaxed for
  ordinary chat: `user_id`, `session_id`, `request_id` and `top_k` may be
  omitted.
- **A forget takes what depends on it.** In the `agent` Add mode, a memory left
  without readable support can no longer be recalled, and the prepare lists it.
  An in-chat forget request also takes the memories that record the request and
  the other memories stating the same fact, and withholds the sentences that
  restate it.
- **Forgetting one detail.** In the `agent` Add mode, forgetting one detail of a
  sentence keeps the other facts it states readable, in Chinese and
  mixed-script sentences too; when the detail cannot be cut out cleanly, the
  whole quote is withheld.
- **Forgetting a name.** In the `agent` Add mode, "forget my brother's name",
  "forget my friend Alice's name" (or "忘掉我弟弟的名字", "忘掉我朋友小雪的名字")
  takes the memories that give that name, and rewords the other memories about
  that person that show it, or takes those that cannot be reworded, listed in the
  prepare's `affected`. A nickname request is handled like any other forget.
- **Relearning after a forget.** In the `agent` Add mode, a forgotten fact is
  learned again only from the user's own later statement of it: a user
  message received and dated after the forget, which is no question or forget
  request and states the fact. The assistant's messages, the request itself,
  questions and older or undated messages never bring it back.
- **Store operations.** `quotemem export` writes a backup of a store's source
  messages and forgets, and `quotemem rebuild --from OLD --to NEW` replays an
  old store or a backup into a new store; a store is never migrated in place.
  `quotemem delete-user` erases one user (`POST /agent/admin/delete-user`), and
  `quotemem repair` inspects, resumes or skips a quarantined user
  (`POST /agent/admin/repair`; `QUOTEMEM_AUTO_REPAIR` does it automatically).
- **Spending and privacy settings.** `QUOTEMEM_DAILY_BUDGET_USD` caps one UTC
  day of provider spending, and `QUOTEMEM_SCRUB_PII` replaces email addresses
  and phone numbers in every text sent to a provider.
- **Dependencies.** The new direct dependency `cryptography==50.0.1` seals the
  forgotten statements in a backup written by `quotemem export` (AES-256-GCM).
- **Packaging.** The source distribution includes `src/quotemem`,
  `benchmarks/sources.json`, `benchmarks/sources-additional.json`, `README.md`,
  `CHANGELOG.md`, `LICENSE`, `NOTICE` and `pyproject.toml`. The build backend
  also adds `.gitignore` and the generated `PKG-INFO`.

### Repository

- **CI.** A GitHub Actions workflow runs ruff, a strict pyright check, an import
  check of every module, a server smoke test, and a build whose source
  distribution and wheel are checked file by file. It also runs the Node tests
  and the npm package check of the agent plugins on Node 20 and 24.
- **Release workflow.** A `vX.Y.Z` tag builds the wheel, sdist and npm tarball
  once, runs the Node tests, the package contents checks, the server smoke test
  and an install of the wheel with its dependencies from PyPI alone, and only
  then publishes the same files to PyPI, npm and a GitHub Release.

### Agent plugins (npm package `quotemem`)

- **DSH plugin.** A DeepSeek Harness (DSH) plugin bundle with an MCP server and a
  skill, then automatic capture and recall of memories in the background. The
  supported DSH range starts at 0.1.2-rc.1, with no upper bound.
- **Seamless memory for more agents.** One shared, zero-dependency core
  (`integrations/core`) is used by adapters for Claude Code, Codex CLI, OpenCode,
  Gemini CLI, Cline, goose, Devin, OpenClaw, Cursor and Continue. Where the agent's
  hooks allow it, the conversation is captured silently, relevant quoted memories
  are recalled as hidden context, and "forget …" requests are carried out and
  confirmed through one note to the model.
- **One npm package for every agent.** `quotemem` ships the bins `quotemem`,
  `quotemem-mcp` and `quotemem-hook`, and the subpath exports `quotemem/dsh`,
  `quotemem/core`, `quotemem/opencode` and `quotemem/openclaw`.
  `quotemem install <tool>` prints what it would change and writes only with
  `--write`; `quotemem doctor` checks the settings, the token and the service.
  `QUOTEMEM_AUTO=1` tells the MCP server that an adapter saves the conversation,
  which hides its `quotemem_add` tool; every adapter shares one default memory owner, and a guard stops Cursor and Devin from saving a turn twice.
- **Safer silent forget and capture.** The plugins recognize more everyday
  forget requests in English and Chinese, including names without "my", shared
  facts like "forget our wedding anniversary date", "from now on …",
  "I don't want you to remember …", "could you please not remember …?", and
  topics like "forget the things I told you about my ex".
  The plugins skip recall questions, corrections, generic requests to start
  over, and coding instructions such as "forget my implementation".
  When a request ends with a clear replacement, such as a new street address,
  the plugin asks to forget the old detail and saves the new value with the chat.
  Notes about a forget say what was removed, and an outcome that
  arrives late is reported in any session. Turns are not sent twice, sessions
  can be resumed, and the plugins stay quiet when the service is down, apart
  from one outage note.
  The OpenClaw memory is readable only by its owner, and the installer writes
  files safely, keeping a backup of each file it changes.
- **The forget note counts what went with it.** The note after a forget also
  counts the other memories that went with it, including those that stay
  stored but can no longer be recalled, without quoting them, and asks the
  model to tell the user. A follow-up note reports any extra removals the
  in-chat forget made while saving the request turn. Retrying a confirmation
  after a lost reply counts each forgotten memory once. The note counts
  reworded memories separately from forgotten ones, for example, "1 related
  memory was reworded to leave out what the user asked to forget, and is still
  stored."
- **OpenCode.** One exit watch is shared per process and a stop signal is raised
  again, so `opencode serve` still stops.
- **Asynchronous capture and status-aware recall.** The background flusher sends
  Add requests asynchronously. Recall reads each memory's status and tells a past
  memory in the past tense, never as how things are now. `memory_not_found`
  answers show the service's message, and the recall search asks for at most 20
  hits. Right after a forget, recall waits briefly while the index rebuilds, so
  a new chat can still find your other memories instead of reporting that memory
  is unavailable.
- **Restart-safe batches.** Batches the service has acknowledged survive a
  restart of the service. A late "nothing matched" note asks the user instead of
  acting. A saved batch that the service has not confirmed within 30 minutes is
  let go.
- **Local service, set up once.** `npx quotemem setup` asks for the models
  (`default`, the competition configuration with thinking at high effort;
  `fast`, the same models with thinking off; or `custom`, any OpenAI-compatible
  endpoints with thinking off, high or the provider's default) and the review
  (`fast`, `standard` or `full`), and saves them to `~/.quotemem/service.env`
  (mode 600). Keys are typed without echo.
- **Started on demand.** When an agent first needs memory, its plugin starts
  `quotemem serve` in the background through `uvx`. Only one instance is
  started, and every agent shares it. `npx quotemem status` and
  `npx quotemem stop` show and stop it; `QUOTEMEM_AUTOSTART=0` turns the
  automatic start off.
- **No token to copy.** The plugins read the service's own `<data-dir>/token`.
  For a service on another computer they read the file that
  `QUOTEMEM_TOKEN_FILE` names, which holds only the token, so the token never
  has to be exported where an agent starts.
- **The `/memory/*` routes.** The plugins call `quotemem serve`'s
  `/memory/add`, `/memory/search`, `/memory/history`, `/memory/provenance`,
  `/memory/list`, `/memory/forget/prepare` and `/memory/forget/confirm`.
- **See what is remembered.** The MCP tool `quotemem_memories` and
  `npx quotemem memories` list the memories of the plugins' user
  (`QUOTEMEM_USER_ID`, else the OS user name); `npx quotemem memories --user`
  lists another user. The MCP tools act on the configured user only; the
  evidence tool is `quotemem_provenance`.
- **Service commands.** `npx quotemem serve`, `check`, `memories`, `export`,
  `rebuild`, `delete-user`, `repair` and `bench` run the matching `quotemem`
  service command through `uvx`, with the settings of
  `~/.quotemem/service.env`.
- **Installs.** Claude Code and Codex install the plugin from this repository's
  marketplaces; DSH from npm (`dsh plugin --profile quotemem add quotemem`).

## 0.1.0 - 2026-11-19

Initial public release of the source on GitHub; no package was published.

- **Memory service** (a Python package and its command): an HTTP service whose
  memories are short statements tied to the exact UTF-8 byte ranges of the
  messages they came from.
  - Updates and corrections write a new version and close the old one
    (`/memory/history`).
  - Forgetting is a two-step prepare/confirm call, by memory or by a
    natural-language statement. It leaves a tombstone, so a later restatement is
    not stored again.
  - A review step compares each new memory with the memories it may update and
    stores the decision with the neighbours it was made against
    (`/memory/why`, `/memory/add-items`; `quotemem serve` names them
  `/memory/provenance` and `/memory/add-results`).
  - Review modes: `lean-local` (default, deterministic local rules) and `full`
    (a judge model reviews every candidate).
  - The benchmark service (now `quotemem bench serve`) serves `/add` and
    `/search` in the Agent Memory Leaderboard (AMLB) wire format, and a router
    (now `quotemem bench route`) spreads users over several service processes.
- **Benchmark tools.** A command-line tool (now `python -m quotemem.bench.cli`)
  downloads public benchmark files from their publishers, replays them over
  HTTP and computes local diagnostic scores. `benchmarks/` lists each source
  by URL, pinned revision, SHA-256 and licence; no benchmark data is included.
- **Dependencies.** HeavenBase 0.1.2.3 stores the data; no HeavenBase code is bundled.
- **Not included.** Tests, a `uv.lock` file and container files.
- **Licence.** MIT. Third-party components and benchmark licences are listed in
  `NOTICE`.
