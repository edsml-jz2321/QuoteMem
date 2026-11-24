# QuoteMem for the OpenAI Codex CLI

This plugin gives Codex the same automatic memory as the DSH plugin, through
Codex's lifecycle hooks. The user does nothing and sees nothing:

- **Silent capture.** Every 3 user turns (`QUOTEMEM_CAPTURE_EVERY`), the
  user's messages and Codex's final replies are saved to QuoteMem in the
  background. Whatever is left is saved when the session ends.
- **Silent recall.** When a message asks about the past or about the user
  ("where do I live?", "上次我们说的…"), the relevant memories, with their
  quoted source lines, are added to the model's context. They are not shown
  in the terminal.
- **Silent forget.** When the user asks to forget something ("please forget
  that I live in Lisbon", "忘掉我的电话号码"), the conversation so far is
  saved first, then the memory is forgotten. Codex gets one short note so it
  can confirm.
- **Explicit tools.** The QuoteMem MCP tools (search, forget with
  confirmation, history, provenance, and a list of what is remembered with
  `quotemem_memories`) stay available.

It needs Codex CLI 0.129 or later (hooks on by default, hook trust), Node 20.3
or later on `PATH`, and the QuoteMem memory service on your computer: run
`npx quotemem setup` once, with [uv](https://docs.astral.sh/uv/) installed,
and the hooks start the service by themselves when it is first needed. The
behaviour is the shared adapter core's; see the
[core README](../core/README.md).

## Install

With the service on your computer (`npx quotemem setup`, run once), nothing
needs to be set: the hooks and the MCP server use `http://127.0.0.1:8765` and
read the service's own token.

For a service on another computer, keep its token in a file only you can
read, and set these in the shell that starts `codex` (for example in
`~/.zshrc`). Codex hands them to the hooks and to the MCP server.

```console
export QUOTEMEM_URL=https://memory.example.org  # the service
export QUOTEMEM_TOKEN_FILE=~/.quotemem/token    # the file's path, not the token
export QUOTEMEM_USER_ID=quei                    # optional; default: the OS user name
```

Do not export the token itself (`QUOTEMEM_TOKEN`): Codex's shell tool
inherits that environment, so the model could print it with `env`.

Use the same `QUOTEMEM_USER_ID` in every tool (DSH, Claude Code, Codex, …),
so they share one memory.

### As a Codex plugin (recommended)

```console
codex plugin marketplace add edsml-jz2321/QuoteMem --sparse .agents --sparse integrations/codex
codex plugin add quotemem@quotemem
```

From a local clone, use the clone's path instead:
`codex plugin marketplace add /path/to/QuoteMem`.

Then start `codex`, run `/hooks`, and trust the five QuoteMem hooks. Codex
skips a hook until you trust it, and asks again after the plugin changes.
`codex mcp list` shows the `quotemem` MCP server.

### Without the plugin system

`install.mjs` adds the hooks to `$CODEX_HOME/hooks.json` and the MCP server
to `$CODEX_HOME/config.toml`. It keeps your other hooks and settings, and
running it again replaces only its own entries. Each file keeps its
permissions and symlink, and its old text is kept as `<file>.quotemem.bak`.

```console
node integrations/codex/install.mjs            # dry run: shows the changes
node integrations/codex/install.mjs --write    # install
node integrations/codex/install.mjs --uninstall --write
```

With the npm package, `quotemem install codex` does the same dry run and
`quotemem install codex --write` installs. The hooks then point at the
installed package, so install it with `npm install -g quotemem`, not
through `npx`.

Options:
- `--codex-home DIR`: another Codex home (default `$CODEX_HOME` or `~/.codex`).
- `--node PATH`: the Node binary the hooks run (default: the one running the
  installer).
- `--notify`: also set the legacy `notify` program, for a Codex older than
  0.116 or a setup where hooks stay untrusted. It only captures; it cannot
  recall. It is added only when `config.toml` sets no `notify` of its own,
  and it stays out of any session where the hooks run.

Install either the plugin or `install.mjs`, not both: both define an MCP
server named `quotemem`. Duplicate hooks would do no harm, since a turn is
recorded once.

Then run `/hooks` in `codex` and trust the hooks, as above.

### What gets installed

| File | Purpose |
|---|---|
| `.codex-plugin/plugin.json` | The plugin manifest |
| `hooks/hooks.json` | `SessionStart`, `UserPromptSubmit`, `Stop`, `Interrupt` and `SessionEnd` all run `hook.mjs` |
| `.mcp.json` | The MCP server (`server.mjs`, a copy of the DSH bundle's) |
| `hook.mjs` | The hook and notify entry point |
| `core/` | A copy of `integrations/core` |
| `../../.agents/plugins/marketplace.json` | The marketplace entry Codex reads in the repository |

Codex starts a stdio MCP server with a minimal environment. `.mcp.json`
therefore lists the `QUOTEMEM_*` variables to forward in `env_vars`. It sets
`QUOTEMEM_AUTO=1`, which hides `quotemem_add` while automatic capture is
on. Set `QUOTEMEM_CAPTURE_EVERY=off` to turn capture off and bring the tool
back.

## How it maps onto Codex

| Codex event | What the hook does | Timeout |
|---|---|---|
| `SessionStart` | Queues what idle or crashed sessions left; starts the sender | 3 s |
| `UserPromptSubmit` | Records the message. On a forget request it runs the silent forget; otherwise it runs the gated recall. The result goes back as `hookSpecificOutput.additionalContext` | 15 s |
| `Stop` | Records `last_assistant_message`; queues a batch every N user turns | 3 s |
| `Interrupt` | Closes the turn without a reply; the user's words are kept | 2 s |
| `SessionEnd` | Queues what is left | 3 s |

- **Session ids** are `codex:<session_id>`. The core turns them into
  deterministic request ids, so a repeated hook stores nothing twice.
- **Subagents.** A prompt with an `agent_id` comes from a subagent and is
  skipped; set `QUOTEMEM_INCLUDE_SUBAGENTS=1` to keep it.
- **Hook output.** A hook always exits 0. It prints nothing, or one JSON
  object with only `hookSpecificOutput`. It never sets `systemMessage`,
  `decision` or `statusMessage`, so Codex shows nothing.
- **Sending.** Batches go to a disk queue under `$CODEX_HOME/quotemem`
  (`QUOTEMEM_STATE_DIR` moves it). A detached background sender posts them
  to `/memory/add`. Codex kills a hook's process group when the hook exits; the
  sender runs in its own session, so it survives. The plugin hooks, a direct
  install and the notify program all share this folder.
- **Context.** Codex keeps injected context as developer messages in the
  thread. The core does not inject the same memory again within
  `QUOTEMEM_RECALL_REPEAT_TURNS` turns.
- **Logs.** Warnings go to `$CODEX_HOME/quotemem/quotemem.log`. It holds
  counts, codes and request ids, never message text or the token.
  `QUOTEMEM_DEBUG=1` adds debug lines.

Every setting of the core applies (`QUOTEMEM_CAPTURE_EVERY`, `QUOTEMEM_RECALL`,
`QUOTEMEM_RECALL_GATE`, `QUOTEMEM_TOP_K`, `QUOTEMEM_MAX_INJECT_CHARS`,
`QUOTEMEM_RECALL_TIMEOUT_MS`, `QUOTEMEM_FORGET`, `QUOTEMEM_FORGET_TIMEOUT_MS`, …);
see the [core settings](../core/README.md#settings).

## See what is remembered

Ask Codex, for example "what do you remember about me?": it lists the
memories with the `quotemem_memories` tool. In a terminal,
`quotemem memories` (or `npx quotemem memories`) prints each memory's status,
statement, dates and quotes; `--json` prints the same list as JSON, and
`--status current|past|possibly_outdated|all` and `--user` choose what to list.
`quotemem memories --html memories.html` writes one local, read-only page
with a search box: it holds no token, loads nothing from the network and only
you can read it. These only read the memory, from the running service or,
when it is stopped, from its data folder. Forgotten memories never appear.

## Notes

- **Trust.** Hooks need one manual trust in `/hooks`, and again after every
  plugin update. `codex exec --dangerously-bypass-hook-trust` skips the check
  for automation and tests only.
- **Visible traces.** A hook still running after 300 ms shows a brief
  "running hook" line in the terminal. A memory question that waits on a slow
  service, and a forget that waits for the service, can show it once.
- **Text.** The plugin saves the text of each message, which is what Codex
  passes to its hooks.
- **Queued batches.** Batches stay queued on disk until they are stored.
  Each hook starts a background sender when none runs.
- **Without hooks** (hooks off or untrusted), only the explicit MCP tools
  work, plus capture through `--notify`. `AGENTS.snippet.md` holds an
  instruction for that case.

## Tests

```console
node integrations/codex/check_plugin.mjs           # copies and manifests
node --test integrations/codex/test/codex.test.mjs # hooks, installer, MCP server
QUOTEMEM_CODEX_BIN=$(command -v codex) node --test integrations/codex/test/codex.test.mjs
npm test                                           # every adapter's tests, from the package root
```

The tests run every hook as its own process with the exact JSON Codex sends,
under a temporary `HOME` and `CODEX_HOME`, against a stub QuoteMem. Like
Codex, they kill the hook's process group as soon as it exits. With
`QUOTEMEM_CODEX_BIN`, two more tests drive the real Codex CLI:
- They install the plugin with `codex plugin marketplace add` and
  `codex plugin add`, and separately with `install.mjs`, into a temporary
  `CODEX_HOME`.
- Codex talks to a local fake model.
- They check what reaches the model, and that nothing shows in the output.

After changing `integrations/core` or `integrations/dsh/server.mjs`, run
`node integrations/codex/check_plugin.mjs --write` to refresh the copies.
