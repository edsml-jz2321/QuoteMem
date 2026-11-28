# QuoteMem for OpenCode

This plugin gives [OpenCode](https://opencode.ai) automatic long-term memory
from a running QuoteMem service. The user never has to ask for it:

- **Silent capture.** The conversation is saved every few user turns, when a
  session is deleted, when OpenCode shuts down, and after an idle time.
- **Silent recall.** When a message asks about the past or about the user,
  the relevant memories and their quoted source lines go into the model
  request only. They are marked as retrieved, untrusted data.
- **Silent forget.** When the user asks to forget something, it is forgotten
  after the earlier turns are stored. The model gets one short note so it can
  confirm.
- **Explicit tools.** The QuoteMem MCP server offers search, history, provenance, a
  list of what is remembered (`quotemem_memories`) and a two-step forget
  (prepare, then confirm).

One plugin directory serves both OpenCode lines:

- **OpenCode 1.x** (npm `opencode-ai`, 1.18.29 or later) calls `server()`.
- **OpenCode 2.x** (npm `@opencode/cli`) calls `setup()`.

## Requirements

- The QuoteMem memory service on your computer: run `npx quotemem setup`
  once, with [uv](https://docs.astral.sh/uv/) installed. The plugin then
  starts the service by itself when it is first needed and reads its token
  from its data folder (see
  [Use with your agents](https://github.com/edsml-jz2321/QuoteMem/blob/main/docs/agents.md)). A service
  you start yourself, or one on another computer, works too.
- Node 20.3 or later on `PATH`. It runs the MCP server and the background
  sender. Without it, the plugin sends batches from inside OpenCode, and a
  batch left at shutdown waits for the next start.

## Install

1. Get the code:

   ```console
   git clone https://github.com/edsml-jz2321/QuoteMem.git
   ```

2. With the service on your computer (`npx quotemem setup`, run once),
   OpenCode needs no settings: the plugin and the MCP server read the
   service's own token. For a service elsewhere, keep its token in a file
   only you can read, and put `QUOTEMEM_URL=https://memory.example.org` and
   `QUOTEMEM_TOKEN_FILE=~/.quotemem/token` (the file's path) in
   `~/.quotemem/env`, then run `chmod 600 ~/.quotemem/env`. The file also
   works for OpenCode 2.x, which runs as a background service, and for the
   desktop app. A variable set in the environment wins over the file. Do not
   export the token itself: OpenCode's shell tool inherits that environment,
   so the model could print it. `QUOTEMEM_USER_ID` (default: your OS user
   name) selects whose memory it is. Every QuoteMem adapter uses the same
   default, so Claude Code, Codex, DSH and OpenCode share one memory.

3. Check your OpenCode line with `opencode --version`. Then merge the
   matching snippet into `~/.config/opencode/opencode.jsonc`, or into a
   project's `opencode.jsonc`. Replace `/path/to/QuoteMem` with the absolute
   path of the clone.
   - 1.x: [`install/opencode.v1.jsonc`](install/opencode.v1.jsonc)
     (`"plugin"` and `"mcp": {"quotemem": …}`).
   - 2.x: [`install/opencode.v2.jsonc`](install/opencode.v2.jsonc)
     (`"plugins"` and `"mcp": {"servers": {"quotemem": …}}`).

4. Restart OpenCode (2.x: `opencode service restart`). `opencode mcp list`
   shows `quotemem`. Tell OpenCode a few facts about yourself, start a new
   session, and ask what it remembers.

With the npm package installed (`npm install -g quotemem`),
`quotemem install opencode` prints both blocks with the installed package's
absolute paths, ready to merge. It does not edit `opencode.jsonc`, because
that file keeps your comments.

The plugin entry must be the directory, not a file: OpenCode 2.x skips a
configured single file, and the plugin imports its own `core/` folder.

To turn the plugin off, remove the entry, or set `QUOTEMEM_CAPTURE_EVERY=off`,
`QUOTEMEM_RECALL=off` and `QUOTEMEM_FORGET=off` one by one. `opencode --pure`
starts without any external plugin.

## How it maps onto OpenCode

| What | OpenCode 1.x | OpenCode 2.x |
|---|---|---|
| The user's message | `chat.message` (non-synthetic text parts) | `session.hook("prompt")` (read only) |
| Recall or forget note into the request | `experimental.chat.messages.transform`: a synthetic part on that user message, never stored. Fallback: `experimental.chat.system.transform` | `session.hook("context")`: one `system` part |
| The final reply | `experimental.text.complete` and `message.part.updated`. Fallback: `client.session.messages` | `session.text.ended` events. Fallback: `session.context` |
| The turn ends | `session.idle`. An aborted turn (`session.error`, an assistant `error`) keeps the user's words only. | `session.execution.succeeded`, `failed` or `interrupted`, and `session.idle` |
| Slash commands | `command.execute.before`: the expanded prompt is not captured | not detected |
| Subagent sessions (with a parent) | skipped (`session.created`, else `client.session.get`) | skipped (`session.created`, else `session.get`) |
| Explicit forget tool | `tool.execute.before` stores the pending turns first (up to 20 s); prepare fails with "still being saved" if they are not stored by then | not hooked |
| Session deleted | `session.deleted`: the rest is queued | `session.deleted` |
| Shutdown | `dispose` queues every session's rest | the cleanup that `setup` returns |

The block for one turn is the same on every model request of that turn, so
the prompt prefix stays stable within the turn. In 2.x the block is a system
part, so a turn with recall misses the provider's prompt cache for the system
prefix. Gated recall keeps such turns rare.

**Waits on the chat path.**
- A memory question waits for the search, at most `QUOTEMEM_RECALL_TIMEOUT_MS`
  (default 3 s).
- A forget request waits for its outcome, at most `QUOTEMEM_FORGET_TIMEOUT_MS`
  (default 1.8 s, at most 2.5 s); a longer one is reported on the next message.
- Nothing else waits on the service. Batches go to a disk queue, and a
  detached `node` process sends them in order, with retries.

When the service is down, a circuit breaker skips recall for a while. The
batches stay queued and are retried. Every hook catches its own errors, so
OpenCode never sees one. The plugin writes nothing to the terminal.

## Settings

The service URL, token, auth mode and user id come from the environment (or
`~/.quotemem/env`) only, so the plugin and the MCP tools always act on the
same memory. The behaviour settings may also come from the plugin options:

```jsonc
// 1.x
"plugin": [["/path/to/QuoteMem/integrations/opencode", { "captureEvery": 5, "recall": "always" }]]
// 2.x
"plugins": [{ "package": "/path/to/QuoteMem/integrations/opencode", "options": { "captureEvery": 5 } }]
```

| Variable | Option | Default | Purpose |
|---|---|---|---|
| `QUOTEMEM_URL` | (env only) | `http://127.0.0.1:8765` | The service |
| `QUOTEMEM_TOKEN_FILE` | (env only) | for a service on this computer, its own token | A file holding only the service's token; without a token, and without `npx quotemem setup`, the plugin stays off. `QUOTEMEM_TOKEN` still wins when set, but never export it where OpenCode starts |
| `QUOTEMEM_AUTOSTART` | (env only) | on | `0` keeps the plugin from starting the service on this computer |
| `QUOTEMEM_USER_ID` | (env only) | OS user name | Whose memory |
| `QUOTEMEM_CAPTURE_EVERY` | `captureEvery` | `3` | User turns per batch; `off` disables capture |
| `QUOTEMEM_IDLE_FLUSH_MS` | `idleFlushMs` | `300000` | Idle time before a partial batch is queued |
| `QUOTEMEM_RECALL` | `recall` | `gated` | `gated`, `always` or `off` |
| `QUOTEMEM_RECALL_GATE` | `recallGate` | `questions` | `questions` ignores plain statements such as "I prefer Rust"; `broad` searches on them too |
| `QUOTEMEM_RECALL_TIMEOUT_MS` | `recallTimeoutMs` | `3000` | Search deadline |
| `QUOTEMEM_TOP_K`, `QUOTEMEM_MAX_INJECT_CHARS`, `QUOTEMEM_RECALL_MIN_SCORE` | `topK`, `maxInjectChars`, `recallMinScore` | `5`, `4000`, none (`0.3` with `always`) | Recall limits |
| `QUOTEMEM_FORGET` | `forget` | `auto` | `auto` (silent forget) or `off` (the explicit tools only) |
| `QUOTEMEM_FORGET_TIMEOUT_MS` | `forgetTimeoutMs` | `1800` | How long a forget turn waits for the outcome (200 to 2500) |
| `QUOTEMEM_INCLUDE_SUBAGENTS` | `includeSubagents` | off | Also recall and capture in subagent sessions |
| `QUOTEMEM_STATE_DIR` | `stateDir` | `~/.quotemem/opencode` | Queue, session files and `quotemem.log` |
| `QUOTEMEM_NODE` | (env only) | `node` on `PATH` | The runtime of the background sender; `off` sends from inside OpenCode |

The [core README](../core/README.md#settings) lists the remaining capture
limits. Warnings go to `<state dir>/quotemem.log`. They hold counts, status
codes and request ids, never message text or the token.

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

## Tests

```console
node --test integrations/opencode/test/opencode.test.mjs
npm test        # every adapter's tests, from the package root
```

The tests drive both plugin APIs with OpenCode's own hook and event payload
shapes against a stub QuoteMem. Each test uses a temporary state folder and
`HOME`.

## Notes

- **Versions.** OpenCode 1.18.29 and later 1.x releases, and 2.x. The 2.x
  payload shapes follow `@opencode/plugin` 2.3.2 and OpenCode's source at
  that tag. OpenCode 1.x also calls the 2.x `setup`; without session hooks it
  does nothing there, and `server` serves 1.x.
- **2.x session hooks.** The plugin uses the `prompt` and `context` session
  hooks. A hook that a release does not offer is logged once, and the rest
  keeps working.
- **Exit.** When `opencode serve` exits on SIGINT or SIGTERM, the plugin
  queues every session's rest itself, on the signal or at process exit, and
  the detached sender sends it. Turns left in the session file by a process
  killed with SIGKILL are queued by the next OpenCode start once they have
  been idle for `QUOTEMEM_IDLE_FLUSH_MS`.
- **Steered and queued prompts (2.x).** A message sent while a turn is still
  running closes the earlier turn with the user's words.
- **Slash commands (2.x)** are captured with their expanded text.
- **Recall in 1.x without the messages hook.** In the system-prompt
  fallback, the block goes into the system prompt of the turn's requests,
  including a title request made at the start of a turn.
- **Forgotten text and search.** Search follows the service's forget rules.
