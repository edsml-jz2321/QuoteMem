# Changelog

All notable changes to QuoteMem are listed here. Dates are UTC.

## 0.3.3 - 2026-11-28

- **A day with a time of day is dated.** "next Tuesday at 3 pm", "at 9 am
  tomorrow" or "tomorrow morning", and the same Chinese forms, now date a
  memory to that day, as "next Tuesday" alone does.
- **Searching the past ranks like any Search.** With the default Search
  profile, a Search with `as_of` now scores its hits from 0 to 1 and gives
  each memory a status read at that time. A `world` Search returns as hits
  only the message parts said by then. docs/service.md documents `as_of`.
- **Every hit clears the relevance floor.** With the default Search profile,
  a Search no longer returns a hit scored below 0.25, nor a memory that
  came with such a hit.
- **A service on this computer must prove it holds the token.** Before the
  plugins, the MCP server and the `quotemem` commands send the token to a URL
  on `localhost`, `127.x`, `[::1]`, `0.0.0.0` or `[::]`, the program on its
  port must prove that it holds it. See docs/agents.md.
- **Recalled memories show how close they are.** Each hit in the plugins'
  memory block now shows its relevance, from 0 to 1, and the block calls a
  score below about 0.4 a weak match. With `QUOTEMEM_RECALL=always`, a hit
  needs a relevance of 0.3 (it was 0.5).
- **The smallest recall budget is 2500 characters.** A smaller
  `QUOTEMEM_MAX_INJECT_CHARS`, or OpenClaw `maxInjectChars`, is raised to
  2500, which leaves room for the block's notes.
- **Search answers faster on a long history.** What a Search reads from a
  user's whole store (each memory's status, what a forget hides) is now read
  once and reused until the store changes, and similar memories are found
  through an index instead of one by one. The results are the same.
- **On a terminal, setup shows a `*` for each key character and says plainly
  whether the key works.** Its menus and notes are one short line each. The
  check ends with `Key works.`, or with the model that failed, the provider's
  error code and a hint. `quotemem check` reports that code too (`provider_code`).
- **Windows paths stay as they are in hook commands.** On Windows the Codex
  installer now writes each path of a hook command in quotes with single
  backslashes, as `quotemem install` does, and so do the install commands
  `quotemem install` prints for Gemini CLI, OpenClaw and Cursor.
- **Background saves hold their files on Windows too.** When Windows refuses
  for a moment to replace a state file, to grant a lock that was just let
  go, or to remove a temporary file, the plugins now try again. A lock left
  by a process that died is broken by one process at a time.
- **Background starts on Windows stay quick with large PowerShell modules
  installed.** The Windows PowerShell that starts the background sender and
  the service, and the one `quotemem doctor` and `quotemem stop` use, now
  looks for its cmdlets in Windows PowerShell's own module folders first.
  The started program gets the module path of the agent that started it.
- **No background send waits for the next prompt.** A send asked for while
  the background sender was still starting, a forget included, now starts
  once that start is done, and a hook waits for that start before it exits.
- **The demo ends cleanly on Windows.** When the service never answers
  `/health` with 200, `examples/demo/demo.mjs` now ends by itself with exit
  code 1.
- **The package check runs on Windows.** `npm run check` now runs npm and the
  installed commands on Windows too, and CI runs the plugin tests, the package
  check and the server smoke test on Windows.
- **A memory that states no name can drop a corrected name from its quote.**
  After the correction "my cat is called Max, not Bob", "The user has a cat."
  quotes "My cat" instead of "My cat is called Bob". Other memories of the cat
  that state no name can be requoted too, when the shorter quote supports them.
- **A question for a name finds the memory that gives it first.** "What is
  the name of the user's cat?" now ranks "The user's cat is named Max, not
  Bob." above "The user has a cat.". So do "What is the user's cat
  called?" and the same question asked in Chinese.
- **A forgotten name is looked for when it comes up again.** In the agent
  mode, a later message that a memory quotes is checked for the name before
  the review.
- **A forget also takes what names the old value as ended.** In the agent
  mode, for a target not marked `bundled`, a memory or sentence that names its
  own value after ", not" or the user's "I no longer", or its number or e-mail
  after "no longer" or "previous", is reworded, forgotten or withheld. The
  prepare lists what changes.
- **A forget of an old value can keep the current one.** A memory that names the
  old value beside the current one (", not 14 Oak Street", "rather than Arkwright
  Media") is reworded without it, or forgotten with it; the prepare lists which.
  A memory marked sensitive (credentials, health) is forgotten whole, with every quote.
- **A confirmation says what it left.** When its checks find a forgotten name,
  value, number or code still readable, a forget by statement answers
  `incomplete` with a last outcome `left_readable`, and the plugins report it
  to the user.
- **The prepare lists more of what a forget changes.** It also lists the
  memories that the user's words hide when a forget withholds sentences alone,
  and a past memory that leaves the list with the memories that replaced it.
- **A forget splits sentences as the plugins do.** A dot inside an e-mail
  address or a word no longer ends a saved sentence.

## 0.3.2 - 2026-11-26

- **`quotemem_add` no longer sends a guessed time.** The MCP tool no longer
  takes a time for each message: a model does not know when words were said,
  and a time it estimates could rank an update before the fact it replaces.
  A message it adds has no message time of its own, so it is never dated
  after a forget and cannot bring a forgotten fact back; a plugin's prompt,
  or an HTTP message with a `timestamp`, can.
- **One background process waits for idle chats.** The process that saves a
  chat's last turns after its idle time now serves every user of a service
  in one state folder, so OpenClaw with `userScope: "sender"` runs one such
  process instead of one per active sender.
- **A forget still queued keeps recall back until it ends.** When a busy
  service holds a forget behind earlier saves for more than two minutes, the
  plugins no longer wait for it at every message; until it ends, messages
  get no recall, so nothing it removes is shown meanwhile.
- **`quotemem doctor` and `quotemem status` check the token without a
  search**, through `GET /agent/status`, in every auth mode (`x-api-key`
  included).
- **Forgets of one user that overlap wait for each other.** A forget that
  arrives while another forget of the same user is running waits on the
  service for its turn before the client is asked to retry: up to 5 seconds
  for the plugin's forget tool, up to 60 seconds for a forget written in the
  chat, and up to 30 seconds when the same request is already running.
  Search, History, Why and the memory list do not show what an accepted
  forget is erasing until it ends. Once a forget written in the chat has
  started erasing, it runs to the end even after its Add has answered; such
  an Add may ask to be retried while its forget is already complete, and the
  retry finds nothing left to do. One request is one forget: the same
  sentence sent by a plugin's forget tool and seen again in the chat is
  carried out once. The plugins now keep a final "+", "#" or "%", so
  requests told apart only by a symbol, such as "C++" and "C#", stay two
  requests.
- **The memory list shows a replaced fact as past even when its update had no
  date.** `quotemem memories` now lists a memory that an undated update
  replaced as past, with no end date, instead of leaving it out. Search is
  unchanged.
- **An update is recognized when its client sends a rounded or day-only
  time.** A change to a fact first stated without a date, stated up to a day
  before that fact arrived, is ordered by when each message arrived, so a
  client that stamps midnight or rounds to the minute no longer leaves the old
  fact current.
- **A name taught again after it was forgotten can be found after `quotemem
  rebuild`.** The rebuilt store dates each forget as the old store did, so a
  later restatement of the name is found by Search again.
- **A store whose first start was interrupted opens.** If the service stopped
  while it was first creating its keyword search tables, the next start
  finishes them instead of refusing the store.
- **A forget that cannot finish no longer blocks its user unnoticed.** It is
  marked for repair under a warning that names its scope, `/agent/status`
  lists its user, and the next start, a repeated confirmation or `quotemem
  repair` runs it again. Nothing it hides becomes readable meanwhile.
- **A forget keeps tidying related memories through a passing error.** If an
  error that clears on retry interrupts the forget, the forget finishes that
  work when it runs again instead of giving it up.
- **Forgetting a name or a detail keeps more of the rest.** A memory that tells
  something else beside a forgotten name or value is reworded without it in
  more cases: two people named after a shared tie ("my friends Tom and
  Lucas"), a relative told with the side of the family, a memory whose quote
  never shows the name, a Chinese request about a memory stored in English,
  and more coat colours of pets.
- **A new value said with a forget request is kept.** "Forget my old number,
  my new number is ..." keeps the new number said with the request.
- **A page on how QuoteMem works.** `docs/architecture.md` describes the data
  folder and the tables of a store, each step of an Add, the read routes,
  forgetting and its tombstone, export, rebuild and deleting a user. The
  README's "How it works" links to it.
- **The demo shows the review and the tombstone.** `examples/demo` now prints
  the link to the replaced memory, each quote's byte range, the review verdict
  from `/memory/provenance` and the forget receipt. It then sends a chat that
  names the forgotten fact again, which the tombstone refuses
  (`denied_by_forget`). `--relearn` adds the user's own later statement, which
  brings the fact back.
- **The included HeavenBase is the published 0.1.2.3 release.** The copy in
  `src/quotemem/_vendor` now holds the HeavenBase package of the 0.1.2.3 wheel
  on the Python Package Index, unmodified, in place of a pre-release build.
  QuoteMem still supports Python 3.12, 3.13 and 3.14.
- **Upgrading from 0.3.0 or 0.3.1.** If a forget said in a chat that began
  before the upgrade does not take effect, say it again in other words or
  start a new chat.

## 0.3.1 - 2026-11-24

- **A user message is the user's own words.** In the `agent` Add mode (the
  default) a speaker set on a user message is ignored, so first-person facts
  and updates always bind to the user. Over HTTP, the `lean-local` and `full`
  Add modes keep the speaker of every message. The MCP tool `quotemem_add`
  sends a speaker only on assistant messages, in every Add mode, and returns
  the store status and reasons.
- **Setup names the Model Studio key and where to get it.** For the default
  and fast presets, `quotemem setup` says that both models, DeepSeek
  included, are called through Alibaba Cloud Model Studio, so the key comes
  from bailian.console.aliyun.com (China region). When Model Studio refuses
  the key, setup says so and how to try again. Each numbered question also
  takes the option's name, such as `standard` or `fast`.
- **Python 3.13 and 3.14.** The memory service runs on Python 3.12, 3.13 and
  3.14, and CI tests all three.
- **Official MCP Registry.** QuoteMem is listed in the official MCP Registry
  as `io.github.edsml-jz2321/quotemem`.
- **Releases start from a version change on `main`.** When a commit on `main`
  sets a new version and passes CI, the tag `vX.Y.Z` is created and the
  release workflow publishes that version. The release workflow validates
  `server.json` before it publishes anything, and publishes to PyPI before npm.
- **SECURITY.md and CONTRIBUTING.md.** They describe how to report a security
  issue and how to send a change.
- **A chat's last turns are saved after 5 minutes idle.** In the Claude Code,
  Codex, Gemini CLI, Cursor, Cline, goose and Devin plugins and in OpenClaw, a
  background sender saves a chat's last turns once it has been idle for
  `QUOTEMEM_IDLE_FLUSH_MS` (5 minutes), with no further prompt.
- **`quotemem install dsh` prints the full command with its profile:**
  `dsh plugin --profile quotemem add quotemem`, then `dsh --profile quotemem`.
- **Setup hints name `service.env` and the environment the tools start in.**
  Model settings and keys go in `~/.quotemem/service.env`. A port of your own
  goes in `~/.quotemem/env` and in the environment Claude Code, Codex and DSH
  start in.
- **A short README with a demo and a quick start.** The full guides are in
  `docs/`. The demo runs from a clone of the repository and waits up to a
  minute for the service to answer, so it can start right after
  `quotemem serve &`.

## 0.3.0 - 2026-11-24

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
