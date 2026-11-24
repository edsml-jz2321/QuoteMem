# QuoteMem for Claude Code

This Claude Code plugin gives the agent a long-term memory that works without
anyone calling a tool:

- **Silent capture.** Every few user turns (3 by default) the new user
  messages and the final assistant replies are saved to QuoteMem in the
  background. What is left is saved when the session ends.
- **Silent recall.** When you ask about the past or about yourself ("what did
  I decide about the database?", "我上次说的截止日期是哪天？"), the most relevant
  memories, with their quoted source lines, are added to the model's context
  for that turn. You do not see the call, and the model is told that the
  block is untrusted background data.
- **Silent forget.** When you ask to forget something ("please forget that my
  sister is called Ana", "把我住在里斯本这件事忘掉"), the turns before it are saved
  first, then QuoteMem forgets the matching memories. The model gets one short
  note so it can confirm.
- **Explicit tools.** The QuoteMem MCP server is included for explicit
  requests: search again, history, provenance, a list of what is remembered
  (`quotemem_memories`), and forget through prepare and confirm.

A slow or stopped QuoteMem service never blocks a prompt: the turn goes on
without memory, and the saved turns wait in a local queue.

The plugin and the MCP server are zero-dependency Node scripts. They are
clients of a QuoteMem service that you run yourself.

## Requirements

- Claude Code with plugin support. `claude plugin validate` of Claude Code
  2.1.304 accepts this plugin.
- Node.js 20.3 or later, as `node` on the `PATH` that Claude Code sees. The
  hooks and the MCP server run `node`.
- The QuoteMem memory service on your computer: run `npx quotemem setup`
  once (it asks for the models, keys, thinking and review), and [uv](https://docs.astral.sh/uv/).
  The plugin then starts the service by itself when it is first needed, and
  reads its token from the service's data folder; see
  [Use with your agents](../../README.md#use-with-your-agents). A service you
  start yourself, or one on another computer, works too.

## Install

From your shell:

```console
claude plugin marketplace add edsml-jz2321/QuoteMem --sparse .claude-plugin integrations
claude plugin install quotemem@quotemem
```

Or inside a Claude Code session:

```text
/plugin marketplace add edsml-jz2321/QuoteMem
/plugin install quotemem@quotemem
/reload-plugins
```

Claude Code 2.1.275 or later can do it in one step:
`/plugin install quotemem --marketplace edsml-jz2321/QuoteMem`.

When the plugin is enabled, Claude Code asks for three optional values:

| Option | Meaning | When empty |
|---|---|---|
| QuoteMem URL | Root URL of the service | `QUOTEMEM_URL`, else `http://127.0.0.1:8765` |
| QuoteMem token | The service's token; kept in the system credential store and given only to the plugin's hooks and MCP server | The file `QUOTEMEM_TOKEN_FILE` names, else, for the service on this computer, its own token |
| Memory owner | Whose memory to use | `QUOTEMEM_USER_ID`, else your OS user name |

For the service on your computer, leave them all empty: the plugin reads the
service's own token. For a service elsewhere, enter the token in the plugin
option, or keep it in a file only you can read and set the file's path in the
environment that starts Claude Code:

```bash
export QUOTEMEM_URL=https://memory.example.org
export QUOTEMEM_TOKEN_FILE=~/.quotemem/token   # the file's path, not the token
claude
```

Do not export the token itself: Claude Code's Bash tool inherits that
environment, so the model could print it.

To see or change the options later: `claude plugin configure quotemem@quotemem`
lists them, and `--values-stdin` saves new values from a JSON object, for
example `echo '{"url":"https://memory.example.org"}' | claude plugin configure quotemem@quotemem --values-stdin`.

Use the same memory owner in every tool (DSH, Codex, OpenCode, …): they then
share one memory, and a fact learned in one is recalled in the others.

### From a local checkout

To try the plugin for one session without installing it:

```console
claude --plugin-dir /path/to/QuoteMem/integrations/claude-code
```

### Without the plugin system

Add the hooks to `~/.claude/settings.json` (or a project's
`.claude/settings.json`) and register the MCP server. Replace
`/path/to/QuoteMem` with your checkout. For a service elsewhere, set
`QUOTEMEM_URL`, `QUOTEMEM_TOKEN_FILE` and, if wanted, `QUOTEMEM_USER_ID` in
the environment that starts Claude Code; a service on this computer needs
none of them.

```json
{
  "hooks": {
    "SessionStart":     [{ "hooks": [{ "type": "command", "command": "node", "args": ["/path/to/QuoteMem/integrations/claude-code/scripts/hook.mjs", "SessionStart"], "timeout": 10 }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "node", "args": ["/path/to/QuoteMem/integrations/claude-code/scripts/hook.mjs", "UserPromptSubmit"], "timeout": 30 }] }],
    "Stop":             [{ "hooks": [{ "type": "command", "command": "node", "args": ["/path/to/QuoteMem/integrations/claude-code/scripts/hook.mjs", "Stop"], "timeout": 10 }] }],
    "StopFailure":      [{ "hooks": [{ "type": "command", "command": "node", "args": ["/path/to/QuoteMem/integrations/claude-code/scripts/hook.mjs", "StopFailure"], "timeout": 10 }] }],
    "SessionEnd":       [{ "hooks": [{ "type": "command", "command": "node", "args": ["/path/to/QuoteMem/integrations/claude-code/scripts/hook.mjs", "SessionEnd"] }] }]
  }
}
```

```console
claude mcp add quotemem --scope user -e QUOTEMEM_AUTO=1 -- node /path/to/QuoteMem/integrations/claude-code/scripts/mcp.mjs
```

`QUOTEMEM_AUTO=1` hides the `quotemem_add` tool, because the hooks already
save the conversation. Leave it out if you register only the MCP server. In
this setup the local queue lives in `~/.quotemem/claude-code`. Do not install
both the plugin and these hooks: a turn would then be handled twice. Repeated
calls for the same prompt are recorded once, but each install keeps its own
queue.

With the npm package installed, `quotemem hook claude-code <Event>` runs the
same hook, so the settings can name `quotemem` instead of a checkout path.

Cursor and Devin also run the hooks in Claude Code's settings files. When
their own QuoteMem hooks are installed (`quotemem install cursor` or
`quotemem install devin`), these hooks see that agent's input and stay silent
there, so a turn is saved once. Cursor's input carries `cursor_version`;
Devin's has `prompt_id` but no `transcript_path`, and Devin sets
`DEVIN_PROJECT_DIR`. Devin's hooks count only for the project they are in.
Without their own hooks, these hooks do the work there. In Cursor 3.20.17
and later, the prompt hook then answers with Cursor's `additional_context`
field instead of Claude Code's. Devin's payload shape comes from its docs and
has not been seen on a real Devin, so for now do not set up both these hooks
and the Devin hooks where Devin reads them.

When `Stop` has no `last_assistant_message`, the reply is read from the
transcript, but only when it answers this turn's own message: a transcript
that is not yet written up to this turn gives no reply.

## How it works

Claude Code runs a hook as a new `node` process for each event, with the
event's JSON on stdin. The hooks use the shared
[adapter core](../core/README.md) (`HookMemory`); this plugin ships a copy of
it in `core/`.

| Event | What the hook does | Output |
|---|---|---|
| `SessionStart` | Queues what idle sessions left and starts the background sender | none |
| `UserPromptSubmit` | Records the message. A forget request runs the silent forget; otherwise a memory-related message is searched (3 s deadline by default) | `hookSpecificOutput.additionalContext`: the recall block or the forget note, or nothing |
| `Stop` | Records `last_assistant_message`; every 3 user turns a batch is queued | none |
| `StopFailure` | The turn failed: keeps the user's words without a reply | none |
| `SessionEnd` | Queues what is left; a detached sender stores it after Claude Code exits | none |

- **Background sender.** Batches go to a file queue. A detached Node process
  sends them one at a time, in order, with retries, and exits when the queue
  is empty. No hook waits for `/memory/add`.
- **What is saved.** Your messages and the assistant's final reply. Never:
  tool calls or tool output, reasoning, the recall block or forget note, slash
  commands (`/…`), `!` shell escapes, acknowledgements such as "ok", and
  prompts that Claude Code submits itself (background task reports and other
  messages wrapped in one element, such as `<task-notification>`). Secrets
  are redacted before anything is written or sent.
- **Subagents.** Hooks that fire inside a subagent (their input has an
  `agent_id`) are ignored unless `QUOTEMEM_INCLUDE_SUBAGENTS=1`.
- **Interrupted turns.** When you interrupt a turn, `Stop` does not run; the
  next prompt keeps your words without a reply.
- **Recall gate.** A local keyword test in English and Chinese. Plain
  statements such as "I prefer Rust" or "remember that I prefer tabs" do not
  search unless the message also asks something. `QUOTEMEM_RECALL=always` searches on every message with a
  stricter score threshold; `off` disables recall.
- **Recall size.** At most `QUOTEMEM_MAX_INJECT_CHARS` (4,000 by default)
  and never more than 9,000 characters, below Claude Code's 10,000-character
  limit for hook context.
- **Forget.** The note names no tool or id. When nothing stored matches, the
  note says so; it never reports a failure that did not happen. If the
  forget takes longer than `QUOTEMEM_FORGET_TIMEOUT_MS` (1.8 s by default,
  at most 2.5 s), the note says it will finish in the background, and the
  next message says how it ended.
  `QUOTEMEM_FORGET=off` leaves forgetting to the explicit tools.
- **Session ids.** The service sees `claude-code:<session_id>` for the
  automatic layer and `claude-code-mcp-…` for explicit tool calls.

## Settings

The three plugin options above, plus the variables of the
[adapter core](../core/README.md#settings), read from the environment that
starts Claude Code. The most useful ones:

| Variable | Default | Purpose |
|---|---|---|
| `QUOTEMEM_CAPTURE_EVERY` | `3` | User turns per batch; `off` disables capture and shows `quotemem_add` again |
| `QUOTEMEM_RECALL` | `gated` | `gated`, `always` or `off` |
| `QUOTEMEM_RECALL_TIMEOUT_MS` | `3000` | Recall deadline (at most 10,000 here) |
| `QUOTEMEM_TOP_K`, `QUOTEMEM_MAX_INJECT_CHARS` | `5`, `4000` | Recall size |
| `QUOTEMEM_FORGET` | `auto` | `auto` (silent forget) or `off` |
| `QUOTEMEM_IDLE_FLUSH_MS` | `300000` | A session idle this long has its pending turns saved |
| `QUOTEMEM_INCLUDE_SUBAGENTS` | off | Also capture and recall in subagents |
| `QUOTEMEM_STATE_DIR` | `${CLAUDE_PLUGIN_DATA}/state` | The local queue and logs |
| `QUOTEMEM_DEBUG` | off | More lines in the log |

## See what is remembered

Ask Claude, for example "what do you remember about me?": it lists the
memories with the `quotemem_memories` tool. In a terminal,
`quotemem memories` (or `npx quotemem memories`) prints each memory's status,
statement, dates and quotes; `--json` prints the same list as JSON, and
`--status current|past|possibly_outdated|all` and `--user` choose what to list.
`quotemem memories --html memories.html` writes one local, read-only page
with a search box: it holds no token, loads nothing from the network and only
you can read it. These only read the memory, from the running service or,
when it is stopped, from its data folder. Forgotten memories never appear.

## Check the installation

```console
claude plugin details quotemem@quotemem
```

It should list five hooks and the `quotemem` MCP server; `/mcp` inside a
session shows the server's tools. The plugin keeps its queue and its log in
`~/.claude/plugins/data/quotemem-quotemem/state/`. `quotemem.log` there holds
warnings only (status codes, counts and request ids, never message text or
the token). A line such as `automatic memory is off: no token` means no
token was found and `npx quotemem setup` was not run. `npx quotemem status`
shows whether the memory service runs; its own log is
`~/.quotemem/service.log`. Uninstalling the plugin deletes this folder, including
batches not yet sent, unless you pass `--keep-data`.

## Development

```console
node integrations/claude-code/check_plugin.mjs --write   # refresh core/ and server.mjs, then check
node --test integrations/claude-code/test/plugin.test.mjs
npm test                                                 # every adapter's tests, from the package root
claude plugin validate integrations/claude-code
```

`core/` is a copy of `integrations/core` and `server.mjs` a copy of
`integrations/dsh/server.mjs`; edit those, never the copies. The tests run
every hook as Claude Code does (`node scripts/hook.mjs <Event>` with the
event's JSON on stdin) against a stub QuoteMem, in a temporary HOME.

## Limits

- Prompts that Claude Code submits itself are recognised by their wrapper
  element only; the hook input has no field that marks them.
- If a `Stop` input has no `last_assistant_message`, the reply is read from
  the end of the transcript file, whose format is not documented.
- `SessionEnd` hooks share a 1.5 s budget, and a plugin cannot raise it. The
  hook only queues the batch; the detached sender stores it later. If the
  machine shuts down first, the batch is sent by the next session.
- On Windows, whether the detached sender outlives Claude Code has not been
  tested. Batches stay queued, and the next hook starts a new sender.
- `claude --bare` skips all hooks, so nothing is saved or recalled there.

## License

MIT.
