# QuoteMem for DeepSeek Harness

QuoteMem is an auditable long-term memory service for conversational agents.
Each memory is a short statement tied to the exact words it came from.

**How it differs.** Many memory layers keep free-text summaries or embeddings
only. QuoteMem keeps each memory with its source quote (exact UTF-8 byte
ranges of the original message), its version history and the review decision
that admitted it. An update writes a new version instead of overwriting, and
forgetting is a two-step prepare and confirm that leaves an unreadable
tombstone. Nothing else brings a forgotten fact back: not the assistant's
replies, the forget request itself or earlier messages. Only if the user
states it again themselves later may it be stored again.

**Result.** In the Agent Memory Leaderboard (AMLB) Cycle 2 official results,
QuoteMem scored 59.83 overall and ranked 23rd of 125 entries in
*Textual Memory · Closed-source Methods*, where it is listed as "QuoteMem":
<https://agentmemoryleaderboard.ai/results/cycle-2/#industry-textual>.

This bundle adds to DSH:

- **automatic memory** (`automem.mjs`), a native DSH plugin. It saves the
  conversation in the background, quietly recalls memories when the user
  asks about the past, and forgets what the user asks it to forget. Nobody
  has to call a tool;
- a zero-dependency stdio MCP server (`server.mjs`) for explicit requests:
  `quotemem_search`, `quotemem_memories`, `quotemem_history`, `quotemem_provenance`,
  `quotemem_forget_prepare`, `quotemem_forget_confirm` and `quotemem_health`
  (`quotemem_add` too, when automatic capture is off);
- the `quotemem` skill, which tells the agent that memory is automatic, to
  cite quoted evidence, and to forget only after the user confirms.

Both are clients of a QuoteMem service that you run yourself. The bundle does
not install Python or the service.

The bundle is part of the one `quotemem` npm package, which also carries the
adapters for the other agents. DSH reads `dsh.bundle` from the package's root
`package.json`; the plugin uses the shared adapter core in
[`integrations/core`](../core).

## Quick install

DSH itself requires Node 24. The bundle needs DSH 0.1.2-rc.1 or later
(`engines.dsh` is `>=0.1.2-rc.1`, with no upper bound) and DSH's pnpm
prerequisite. If a DSH version lacks an optional API the automatic layer uses,
the plugin skips that part, with one warning where it can detect the gap; the
MCP tools keep working. The service is the Python package `quotemem`; it
needs Python 3.12, and `uvx` from [uv](https://docs.astral.sh/uv/) 0.11.21 or
later runs it without a separate install. The plugin is tested
against a stub service and a simulated DSH context
(`node --test integrations/dsh/test/*.test.mjs`; the shell expands the glob), and was tried in a real DSH 0.2.2-rc.5.

```console
dsh plugin --profile quotemem add quotemem
```

Or install it from a packed tarball of a clone (`npm pack` in the clone's root
writes `quotemem-<version>.tgz`):

```console
dsh plugin --profile quotemem add file:/path/to/quotemem-0.3.0.tgz
```

### 1. Set up the memory service, once

QuoteMem runs on your computer. Give its memory service the model settings
it needs (it asks for them, checks them, and saves them to
`~/.quotemem/service.env`, mode 600):

```bash
npx quotemem setup
```

Pressing Enter picks the recommended `default` preset, the competition
configuration: chat model `deepseek-v4.1-flash-1017` with thinking at high
effort and embedding model `qwen3.7-text-embedding` (1024 dimensions), both
through Alibaba Cloud Model Studio with one API key, billed to your account.
Thinking makes saving memories slower (a large batch of corrections can take
several minutes) and adds output tokens. The `fast` preset is the same with
thinking off: memories are saved in seconds, and it scored slightly lower in
our benchmark (within noise). `custom` takes any OpenAI-compatible chat and
embedding endpoints. Switch with `npx quotemem setup --preset fast` (or
`custom`), or edit `~/.quotemem/service.env`, where each `QUOTEMEM_*` line
overrides the preset (`QUOTEMEM_CHAT_REASONING=off` turns thinking off).

Setup also asks the review, which decides whether a new memory repeats,
updates or corrects a stored one (`--review` skips the question; saved as
`QUOTEMEM_REVIEW`): `fast` uses only the local review rules of the
competition submission (no review calls; the fastest and cheapest),
`standard`, the default, sends to the chat model only the new memories that
may repeat, update or correct a stored one, and `full` sends every new
memory to it (the slowest and most careful). The review is fixed when the
memory store is created; to change it later, run `npx quotemem stop`, then
`npx quotemem rebuild --from <data dir> --to <new data dir> --review full`,
and point `QUOTEMEM_DATA_DIR` in `~/.quotemem/service.env` at the new
directory (with `npx quotemem setup --review full`).

From then on the plugin starts the service by itself
(`uvx --from quotemem==0.3.0 quotemem serve`, on 127.0.0.1:8765) when DSH
first needs memory, so [uv](https://docs.astral.sh/uv/) must be installed.
The service keeps its data in `~/.local/share/quotemem` (a
`QUOTEMEM_DATA_DIR` line in `~/.quotemem/service.env` moves it), and creates
a token there on its first start, which the plugin reads. Use
your own provider keys; none are included. To run the service yourself
instead, start `uvx quotemem serve`; the plugin then starts nothing.

### 2. Start DSH with memory

```bash
export QUOTEMEM_USER_ID=alice                   # optional; default: your OS user name
dsh --profile quotemem
```

For the service on this computer, the plugin and the MCP server read its own
token, so nothing else is needed. For a service elsewhere, keep its token in
a file only you can read and set `QUOTEMEM_URL` and `QUOTEMEM_TOKEN_FILE`
(the file's path) before `dsh`. Do not export the token itself: DSH's shell
tool inherits that environment, so the model could print it.

Then just talk. Memory works on its own; see the next section.

## How memory works automatically

**Saving (capture).** After each turn the plugin keeps the user's own
messages for that turn and, when the turn completed, the assistant's final
reply. A turn the user interrupted, or one that ended in a model error or at
the token limit, still keeps what the user said. Every
`QUOTEMEM_CAPTURE_EVERY` user turns (default 3) they are queued as one batch,
and a small background sender posts it to QuoteMem `/memory/add`. A partial batch is
also queued when the session ends, when DSH quits, after
`QUOTEMEM_IDLE_FLUSH_MS` without activity (default 5 minutes), and before a
forget. A partial batch holds only new turns: the service reads the
session's previous batch as context by itself.

- The chat never waits for a write and never sees one fail.
- The sender is a separate process. It keeps sending after DSH quits, and
  exits when the queue is empty. Turns that DSH could not queue before a
  crash are queued at the next start, after the idle time.
- Not saved: tool calls and tool output, reasoning, attachments, harness and
  plugin context (the recall block and the forget note too), slash commands,
  acknowledgements such as "ok" or "thanks", blocked turns and subagent
  sessions. Short answers such as "no" or "对" are saved; when one answers
  the assistant's question, the question is saved with it.
- Each batch has a deterministic `request_id` (session, turn range and a
  content hash), so a resend never stores it twice. It is written to the
  queue folder before it is sent. A failed write is retried with backoff
  (honouring `Retry-After`); a batch is dropped after
  `QUOTEMEM_CAPTURE_MAX_ATTEMPTS` attempts or `QUOTEMEM_CAPTURE_MAX_AGE_MS`.
- The service may hold a batch or a forget back for now and answer
  `429 delivery_busy`: for example after a forget, it holds a session's next
  batch while an earlier batch of that session completes (up to about two
  hours). It also gives this answer for its internal `finalize_pending`,
  `reception_stale`, `authorization_expired`, `budget_exceeded`,
  `extraction_busy` and `lease_lost`, and when its receiver is not
  available. In every case the job waits, as the service's `Retry-After`
  says, without using its attempts, and the jobs behind it wait for it. It
  is dropped only after `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS` of such answers
  (default 2.5 hours) or at the age limit. Any other answer ends the wait
  (only a failure uses an attempt); the next `delivery_busy` starts the
  wait's clock again.
- A refused token pauses writes for 10 minutes per attempt. The batches are
  kept until the attempt or age limit above.

**Recall.** Before the model answers a question about the past or about the
user (for example "do you remember", "last time", "what's my", "where do I
live", "上次", "之前", "我说过", "我的生日是哪天"), the plugin searches
QuoteMem with that message. The check is a cheap local keyword test in
English and Chinese. A plain statement such as "I prefer Rust" or "我住在杭州"
does not search; `QUOTEMEM_RECALL_GATE=broad` makes it search too.

- The search has a short deadline (`QUOTEMEM_RECALL_TIMEOUT_MS`, default
  3000 ms). If QuoteMem is slow or down, the turn goes on without memory,
  and the model gets one short note that memory could not be checked, so it
  does not tell the user that nothing is stored. When it is down, or misses
  the deadline twice in a row, later turns skip the search for a while
  instead of waiting again.
- The search asks for at least 10 hits (more when `QUOTEMEM_TOP_K` is
  larger). At most `QUOTEMEM_TOP_K` hits (default 5) are inserted, whole, in
  the service's order, within `QUOTEMEM_MAX_INJECT_CHARS` (default 4000); a
  lower hit fills in when a higher one is too long. The agent service ranks
  to a fixed depth, so its first 5 hits are the same with either size.
- The agent service scores each hit as a relevance from 0 to 1 and returns
  nothing below its own floor, so a question it cannot answer, or a message
  the keyword test picked by mistake, gets no hits and no block. Gated recall
  adds no threshold of its own; set `QUOTEMEM_RECALL_MIN_SCORE` for one.
- Each hit keeps its statement and quoted source lines. The block is marked
  as retrieved memory and as untrusted data that the model must not follow
  as instructions.
- A memory the service marks `past` no longer holds: it was replaced,
  corrected or has ended. One marked `possibly_outdated` conflicts with
  something the user said later, and nothing settled which holds. A past
  event that stays true (a move last week) is current and carries no mark.
  The block tells the model to tell a past memory as how things were and go
  by its replacement, to say that a possibly outdated one may have changed,
  and never to mention the labels.
- A memory already inserted in a session is not inserted again for
  `QUOTEMEM_RECALL_REPEAT_TURNS` user turns (default 10). After that it may
  be inserted again, since DSH compaction may have removed the earlier block.
- In DSH's web chat the inserted block appears as a collapsed context row
  labelled `plugin:quotemem`, like other plugin context. It is part of the
  session history until compaction.
- `QUOTEMEM_RECALL=always` searches on every user message instead, and
  inserts only hits with a relevance of 0.5 or more. `QUOTEMEM_RECALL=off`
  turns recall off.

**Forget.** When the user asks to forget something ("please forget that I
…", "please don't remember …", "忘掉…", "别记住…", "把…忘掉"), the plugin does it
before the model answers:

1. The turns not yet saved are queued, with their context.
2. The forget waits in the same queue, so everything said before it is
   sent first. It waits at most 2 s for those turns to be stored when the
   service also reads the request when the turn is saved; a request only the
   plugin reads (such as "forget that I asked you for …") waits until they
   are stored. Turns the service has not saved yet always go first: when
   the service holds them back as busy (see "Saving" above), the forget waits
   behind them for up to `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS` (default
   2.5 hours).
3. It runs QuoteMem's forget by statement: prepare, then confirm. The user's
   own request is the consent.
4. The model gets one short note, as a collapsed context row like the recall
   block. It says what was forgotten, and names no tool and no id. The model
   then confirms in its own words.
   - Only what the service verified counts: the memories forgotten, and the
     sentences forgotten when no memory stated the fact. What was gone
     already, or changed meanwhile, is never counted.
   - A memory that also held other facts goes whole; the note quotes it, so
     the model can tell the user what else went.
   - A look-alike the service kept (about another person, or saying the
     opposite) is named, and never forgotten.
   - When the forget cannot be confirmed, the note says so, and neither
     "forgotten" nor "nothing deleted".

The turn waits at most `QUOTEMEM_FORGET_TIMEOUT_MS` (default 1.8 s, at most
2.5 s). When the forget takes longer, the note says it finishes in the
background, and it does. How it ended is reported to the model on the next
message of the same user in any session of the profile, also after the
session that asked has ended; another user of the profile never sees it.
"Forget that I asked you for …" and "forget that I said …" are requests;
"forget it" as a dismissal, a bare "forget that I asked", a change of
instructions ("forget that I asked you to use tabs; use spaces"), "don't
forget that …" and questions about forgetting are not. `QUOTEMEM_FORGET=off` leaves
forgetting to the tools.

When the memory service itself finds a forget request while saving the
conversation (one the plugin did not carry out), the next message of the same
user gets a short note saying what it did.

**Explicit requests** still go through the tools: searching again with other
words, a memory's history, its provenance, a list of what is remembered
(`quotemem_memories`), and a forget the user asks the
agent to carry out with the tools. That forget is a prepare step, the user's
yes, then a confirm step. Before prepare runs, the plugin queues the
unsaved turns and waits up to 20 s for them to be stored. If they are still
not stored then, prepare is refused with a short reason ("still being
saved"), and the model tells the user to try again shortly: a forget run then
could miss them for good.

To turn saving off, set `QUOTEMEM_CAPTURE_EVERY=off` in the environment that
starts DSH; `quotemem_add` then comes back as a tool. Do the same if you
disable the `quotemem-automem` row. The MCP server sees only the
environment, so turning capture off in the row config stops saving but keeps
`quotemem_add` hidden.

The settings, gating, redaction, recall formatting, forget, HTTP client and
capture queue come from the shared adapter core in [`integrations/core`](../core),
which the bundle imports from the package's own `integrations/core` folder.
Other agents' adapters use the same core, so every adapter saves, recalls and
redacts the same way.

A profile can override the behaviour settings in its own `cordis.patch.yml`
row `quotemem-automem`, for example `config: { recall: always, captureEvery: 5 }`.
`recall` takes `gated`, `always` or `off`; `recallGate` takes `questions` or
`broad`; `forget` takes `auto` or `off`; `captureEvery` and `idleFlushMs`
take a number or `off`. The service URL, user id, auth mode and token are
read only from the `QUOTEMEM_*` environment variables, so the automatic layer
and the MCP tools always act on the same memory; the row config ignores them.

## Privacy

- Only the user's own messages and the final assistant replies are sent, and
  only to `QUOTEMEM_URL`. Secrets are redacted on the client before anything is
  sent or written: private keys, cloud and API keys, tokens, JWTs,
  `password: …` style values, credentials in URLs, Stripe, Hugging Face and
  npm tokens, card numbers (Luhn checked), US social security numbers and
  Chinese resident ID numbers. Redaction is best-effort and
  pattern-based: it can miss a quoted multi-word password, a short PIN or a
  secret in free phrasing, and it can redact a harmless word after
  "password is". A message that is mostly a secret is skipped. The service
  refuses to make a memory of a secret as a second line of defence; do not
  paste secrets into the chat.
- The state folder is `<profile>/quotemem/`, or `~/.quotemem/dsh/` when DSH
  gives no profile folder, or `QUOTEMEM_STATE_DIR`. Its files are readable
  only by you (mode 0600, folder 0700). `sessions/` holds the redacted turns
  of each session until they are queued, `queue/` holds the batches until the
  service has stored them, and `quotemem.log` holds the sender's warnings. A
  batch left by a profile that is never started again stays on disk until
  you delete the folder.
- By default the memory owner (`user_id`) is your OS account name, and it is
  sent to the service with every request. Set `QUOTEMEM_USER_ID` to use
  another id. The MCP tools take no user argument and refuse a call that
  names one, so the model cannot read or change another user's memory.
- Logs and warnings carry only counts, HTTP status, error codes and request
  ids, never the token, a query or message text. Each kind of warning is
  printed once.
- All profiles and clients that use the same `QUOTEMEM_USER_ID` share one
  memory. Set a stable id per person.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `QUOTEMEM_URL` | `http://127.0.0.1:8765` | Root URL of the QuoteMem service (`quotemem serve`) |
| `QUOTEMEM_TOKEN_FILE` | for a service on this computer, its own token | A file holding only the service's token. `QUOTEMEM_TOKEN` (then `QUOTEMEM_HTTP_AUTH_SECRET`) still wins when set; do not export it where DSH starts |
| `QUOTEMEM_AUTOSTART` | on | `0` keeps the plugins from starting the service on this computer (`npx quotemem setup` turns starting on) |
| `QUOTEMEM_HTTP_AUTH_MODE` | `bearer` | `bearer`, `token` or `x-api-key`; must match the service |
| `QUOTEMEM_USER_ID` | your OS user name | Whose memory is read and written |
| `QUOTEMEM_CAPTURE_EVERY` | `3` | User turns per saved batch; `off` or `0` turns saving off |
| `QUOTEMEM_IDLE_FLUSH_MS` | `300000` | Idle time before a partial batch is saved; `0` waits for the session end (or a week without activity) |
| `QUOTEMEM_RECALL` | `gated` | `gated` (memory questions), `always` or `off` |
| `QUOTEMEM_RECALL_GATE` | `questions` | `questions`, or `broad` to also search on plain statements about the user |
| `QUOTEMEM_FORGET` | `auto` | `auto` (silent forget) or `off` (the tools only) |
| `QUOTEMEM_FORGET_TIMEOUT_MS` | `1800` | How long a forget turn waits (from the message's arrival, 200 to 2500) before the note says it finishes in the background |
| `QUOTEMEM_TOP_K` | `5` | Hits inserted at most (1–20); the search asks for at least 10 |
| `QUOTEMEM_MAX_INJECT_CHARS` | `4000` | Size limit of the inserted memory block (500–50000) |
| `QUOTEMEM_RECALL_TIMEOUT_MS` | `3000` | Recall deadline; the turn goes on without memory after it |
| `QUOTEMEM_RECALL_MIN_SCORE` | none (the service's floor); `0.5` with `always` | Hits below this relevance (0 to 1) are not inserted |
| `QUOTEMEM_RECALL_REPEAT_TURNS` | `10` | User turns before an inserted memory may be inserted again |
| `QUOTEMEM_SEARCH_OPTIONS` | none | Comma-separated `options` sent with each automatic search, for a service-side search mode |
| `QUOTEMEM_CAPTURE_ASSISTANT` | `final` | `final` reply of each turn, or `none` for user messages only |
| `QUOTEMEM_MAX_MESSAGE_CHARS` | `12000` | Longer saved messages are cut |
| `QUOTEMEM_CAPTURE_TIMEOUT_MS` | `1860000` | Time limit of one background write |
| `QUOTEMEM_CAPTURE_MAX_ATTEMPTS` | `8` | Failed attempts before a batch or forget is dropped; a `429 delivery_busy` answer is not counted |
| `QUOTEMEM_CAPTURE_MAX_AGE_MS` | `86400000` | Unsent batches older than this are dropped |
| `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS` | `9000000` | How long a batch or forget the service holds back as busy (`429 delivery_busy`) keeps waiting before it is dropped (60000 to 30 days) |
| `QUOTEMEM_CAPTURE_QUEUE_MAX` | `200` | Batches kept at most; the oldest go first |
| `QUOTEMEM_CAPTURE_RETRY_BASE_MS` | `5000` | First retry delay; it triples up to 5 minutes |
| `QUOTEMEM_FLUSHER_MAX_WAIT_MS` | `120000` | Longest retry wait the background sender sleeps through; a later turn starts it again |
| `QUOTEMEM_STATE_DIR` | `<profile>/quotemem`, else `~/.quotemem/dsh` | Folder of the session files and the queue |
| `QUOTEMEM_INCLUDE_SUBAGENTS` | off | Also recall and save in subagent sessions |
| `QUOTEMEM_DEBUG` | off | Debug lines (counts and ids only) |
| `QUOTEMEM_SESSION_ID` | one id per DSH session | Session id of explicit `quotemem_add` calls |
| `QUOTEMEM_TIMEOUT_MS` | `1800000` | Time budget of one tool call, retries included |
| `QUOTEMEM_NODE` | `node` | Node executable that runs the server |
| `QUOTEMEM_MCP_SERVER`, `QUOTEMEM_SKILL_ROOT` | the bundle's own files | Override the server script or the skills folder |

The token is sent only to `QUOTEMEM_URL` and never written to output or logs.
Use `https://` or a loopback address: over plain `http` to another host the
token travels unencrypted.

DSH starts the MCP server with a scrubbed environment that drops names such
as `*_TOKEN`; the bundle passes every `QUOTEMEM_*` variable to it explicitly
(`QUOTEMEM_TOKEN_FILE` included), so the server and the plugin see the same
settings and read the same token file.

A write through `quotemem_add` returns only after the new memories are
stored and searchable, which usually takes tens of seconds, sometimes
minutes. Automatic saving runs in the background instead: it asks for an
early acknowledgement (`Prefer: respond-async`) and reads each batch's result
later with the identical request. `quotemem_history` and `quotemem_provenance` take
the id of a memory hit; for an Evidence hit's id the tool shows the service's
own explanation. The MCP server retries `429` and `503` answers with
the same request, honouring `Retry-After`.

## See what is remembered

Ask the agent, for example "what do you remember about me?": it lists the
memories with the `quotemem_memories` tool. In a terminal,
`quotemem memories` (or `npx quotemem memories`) prints each memory's status,
statement, dates and quotes; `--json` prints the same list as JSON, and
`--status current|past|possibly_outdated|all` and `--user` choose what to list.
`quotemem memories --html memories.html` writes one local, read-only page
with a search box: it holds no token, loads nothing from the network and only
you can read it. These only read the memory, from the running service or,
when it is stopped, from its data folder. Forgotten memories never appear.

## Check the installation

This command makes no model requests:

```console
dsh plugin --profile quotemem list
```

Then start DSH and ask the agent to call `quotemem_health`. If the tool is
missing, the MCP server did not start: the memory tools are optional, so DSH
starts without them. The usual cause is that `node` is not on `PATH` in the
shell that started DSH; set `QUOTEMEM_NODE` to the full path of a Node 20.3 or
later executable.

If automatic memory cannot start, DSH prints one
`[quotemem] automatic memory is off: …` warning. The usual causes are no
token (no `npx quotemem setup`, no service token on this computer, and no
readable `QUOTEMEM_TOKEN_FILE`), a `QUOTEMEM_URL` that is not an `http(s)://`
URL, or an unknown `QUOTEMEM_HTTP_AUTH_MODE`. If an old setup still exports
`QUOTEMEM_TOKEN`, do not share `--dump-config` output: the resolved
configuration then contains it.

To keep DSH's state out of your home directory, set `DSH_HOME` and
`DSH_AGENTS_HOME` to private folders and disable telemetry with
`DSH_TELEMETRY_DISABLED=1` before these commands. On the day a version is
published, DSH's pnpm may refuse it as too new; pin it, for example
`dsh plugin --profile quotemem add quotemem@0.3.0`.

## Other MCP clients

The same server works with any stdio MCP client; there `quotemem_add` stays a
normal tool unless the client's QuoteMem adapter saves automatically (the
[repository README](../../README.md) lists the agents with adapters). On its
own it needs Node 20.3 or later:

```json
{
  "mcpServers": {
    "quotemem": {
      "command": "npx",
      "args": ["-y", "--package=quotemem", "quotemem-mcp"],
      "env": { "QUOTEMEM_URL": "https://memory.example.org", "QUOTEMEM_TOKEN_FILE": "~/.quotemem/token" }
    }
  }
}
```

A client's `env` block reaches only the MCP server process, not the model's
shell tool. Leave it out for a service on this computer: the server reads the
service's own token file. Even here, prefer `QUOTEMEM_TOKEN_FILE` to the token
itself, so the token stays out of the client's config file.

## License

MIT, see [LICENSE](LICENSE).
