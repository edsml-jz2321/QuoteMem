# QuoteMem for other agents: Gemini CLI, Cursor, Cline, goose, Devin, OpenClaw

These adapters give more agents the same automatic memory as the DSH plugin:

- **Silent capture.** The conversation is saved every few user turns, and
  the rest when the session ends.
- **Silent recall.** When the user asks about the past or about themselves,
  the relevant memories, with their quoted source lines, are added to the
  model's context.
- **Silent forget.** When the user asks to forget something, it is forgotten
  and the model gets one short note so it can confirm.

The QuoteMem MCP tools stay available for explicit requests. All behaviour
comes from the shared [adapter core](../core/README.md). Every agent uses the
same memory, the one named by `QUOTEMEM_USER_ID` (default: the OS user name),
so something learned in one agent is recalled in the others.

| Agent | Mechanism | Capture | Recall | Forget |
|---|---|---|---|---|
| [Gemini CLI](#gemini-cli) | Extension: hooks, MCP and `GEMINI.md` | yes | yes, each turn | yes, with a note |
| [Cursor](#cursor) | User hooks, MCP and a rule | app and CLI interactive mode: yes, by the turn hooks; CLI print mode: at the end of each run, from the chat's transcript | app: yes, each turn (3.20.17 and later); CLI interactive mode: yes, each turn (2026.11.19 and later); CLI print mode: a session-start digest and the tools | app: yes, with a note (3.20.17 and later); CLI interactive mode: yes, with a note; CLI print mode: when the run ends, without a note, or through the tools |
| [Cline](#cline) (VS Code) | Hook files | yes | yes, each turn | yes, with a note |
| [goose](#goose) | Plugin hooks and the context file | yes | yes, each turn | yes, with a note |
| [Devin](#devin-cli-and-devin-desktop) (CLI, Desktop) | Hooks | yes (CLI 3000.5.20 and later) | yes, each turn | yes, with a note |
| [OpenClaw](#openclaw) | In-process plugin and MCP | yes | yes, each turn | yes, with a note |
| [Continue](#continue) | MCP and a rule | by the model | by the model | by the model |

Claude Code, Codex and OpenCode have their own adapters in
[`integrations/`](..).

## Before you start

1. **Set up the QuoteMem service, once:** `npx quotemem setup`. It asks for
   the models, keys, thinking and review, and saves them to `~/.quotemem/service.env`. The
   hooks then start the service on your computer by themselves when it is
   first needed (this needs [uv](https://docs.astral.sh/uv/)), and read its
   token from its data folder. A service you start yourself works too; see
   [Use with your agents](https://github.com/edsml-jz2321/QuoteMem/blob/main/docs/agents.md).
2. **Install Node 20.3 or later** and make sure it is on `PATH`.
3. **Install the hooks.** The shortest way is the npm package:

   ```bash
   npm install -g quotemem
   quotemem install <agent>           # prints what it would write
   quotemem install <agent> --write   # writes it
   ```

   `<agent>` is `gemini`, `cursor`, `cline`, `goose`, `devin`, `continue` or
   `openclaw`. The files it writes call the installed hook runner by its
   absolute path, so they work even when an editor started from the dock
   has no shell `PATH`. For agents that install a folder themselves (Gemini
   CLI, OpenClaw) it prints the agent's own install command. It never writes
   the token. A file it changes keeps its permissions, is written through a
   symlink, and its old text is kept as `<file>.quotemem.bak`.

   From a clone instead, put `quotemem-hook` on `PATH` and copy the files as
   shown in each section below:

   ```bash
   mkdir -p ~/.local/bin
   ln -sf "$PWD/integrations/hooks/quotemem-hook.mjs" ~/.local/bin/quotemem-hook
   quotemem-hook --help
   ```

   `quotemem-hook <agent> <event>` is the hook program.
   `quotemem-hook mcp` runs the MCP server.
4. **Only for a service on another computer, or other settings:** store
   them in `~/.quotemem/env`. With the service on your computer nothing is
   needed here. Some agents start hooks without your shell environment.
   Gemini CLI strips any variable whose name contains `TOKEN`.

   ```bash
   mkdir -p ~/.quotemem && chmod 700 ~/.quotemem
   cat > ~/.quotemem/env <<'EOF'
   QUOTEMEM_URL=http://127.0.0.1:8765
   QUOTEMEM_TOKEN_FILE=~/.local/share/quotemem/token
   EOF
   chmod 600 ~/.quotemem/env
   ```

   - Only `QUOTEMEM_*` lines are read.
   - A variable set in the environment always wins over the file.
   - `QUOTEMEM_ENV_FILE` moves the file.
   - `QUOTEMEM_TOKEN_FILE` names a file holding only the token; here it is
     the token file `quotemem serve` creates in its data directory. For a
     service on this computer that file is read even without the line; for
     one elsewhere, point the line at a file holding its token (chmod 600).
   - Never export the token itself (`QUOTEMEM_TOKEN`) where an agent
     starts: the agent's shell tool inherits that environment, so the model
     could print it.

   Every [core setting](../core/README.md#settings) works here, for example
   `QUOTEMEM_CAPTURE_EVERY=3` or `QUOTEMEM_RECALL=off`.

## Gemini CLI

The [`gemini-cli`](../gemini-cli) folder is a self-contained extension. It
holds:
- hooks on `BeforeAgent` (recall and forget), `AfterAgent` (capture) and
  `SessionEnd` (flush);
- the MCP server;
- a short `GEMINI.md` that explains the memory blocks to the model.

```bash
gemini extensions install /path/to/QuoteMem/integrations/gemini-cli
# or, to follow a checkout: gemini extensions link /path/to/QuoteMem/integrations/gemini-cli
```

Gemini asks for the URL, the token and the user id; an empty answer falls
back to `~/.quotemem/env`. Gemini stores the token in the system keychain.

Gemini shows an indicator while a hook runs. To hide it, set
`"hooksConfig": {"notifications": false}` in `~/.gemini/settings.json`; that
change is yours to make.

## Cursor

Capture and forget are fully automatic. How, and how recall works, depend
on whether it is the desktop app or Cursor's CLI, and on the app's version:

- **Every version.** At session start, a short digest of what is known
  about the user goes into the conversation's system context. Cursor does
  not wait for that hook, so a first prompt sent quickly may go without it.
- **Settings: in `~/.quotemem/env`.** Cursor's CLI does not pass its own
  environment to the MCP server it starts, and the MCP tools read
  `~/.quotemem/env`. So keep every `QUOTEMEM_*` setting for Cursor in that
  file (step 4 above), and none only in the shell. `quotemem install cursor`
  warns about a setting set only in the shell.
- **Cursor's CLI in interactive mode** (`cursor-agent`; its versions are
  dates, such as 2026.11.19). Version 2026.11.19 was run on Windows: it runs
  `beforeSubmitPrompt`, `afterAgentResponse` and `stop` for each turn, and
  the prompt hook's `additional_context` reaches the model. So it works as
  the desktop app below: recall and the forget note in each turn, from
  2026.11.19 on. `/quit` runs `sessionEnd`, which finds those turns saved
  and saves none again.
- **Cursor's CLI in print mode** (`-p`). Version 2026.11.19 was run on
  Windows: it runs only `sessionStart` (for a new chat) and `sessionEnd` (at
  the end of every run). `beforeSubmitPrompt`, `afterAgentResponse` and
  `stop` never run. So:
  - recall is the session-start digest plus the tools (`quotemem_search`);
  - the conversation is saved at the end of each run. The hook reads what
    the chat's transcript gained since the last run (the `transcript_path`
    of `sessionEnd`, a file named `<chat id>.jsonl`) and saves the turns no
    prompt hook saw: the user's words, without Cursor's `<timestamp>` and
    `<user_query>` wrappers, and the final reply. Each turn is saved once:
    a resumed chat's later run saves only its new turns, and a turn sent
    again keeps its request id and body. A run's last turn is saved when
    the run ended completed, or once its `turn_ended` line is written; the
    reply is the turn's last text, not the text before a tool call, and a
    reply added later is not saved again;
  - a forget request in those turns is carried out then, in the background,
    without a note to the model. So the rule tells the model, when a forget
    request comes without a `[QuoteMem: automatic forget]` note, to forget
    through the tools itself (`quotemem_forget_prepare`, then
    `quotemem_forget_confirm`) and never to say something was forgotten
    unless a note or a tool result says so. Such a forget is kept for a
    week (the statement only, in `forgotten/` under `~/.quotemem` and
    under `QUOTEMEM_STATE_DIR` when set, mode 600) and applied again behind the run's words
    when they are saved at its end, so a fact said earlier in the run gets
    the forget too. Words of a later run are kept; a turn of the same run and
    minute is forgotten too, since Cursor's times have minutes only;
  - in print mode, Cursor runs the MCP tools only with
    `--approve-mcps --force`; with `--approve-mcps` alone it rejects the
    tool call, and recall and forget through the tools do not happen.
- **The desktop app, 3.20.17 and later.** The prompt hook
  (`beforeSubmitPrompt`) answers with `additional_context`, which Cursor
  adds to that turn. So a memory question gets the relevant memories, and a
  forget gets one note, as in the other agents. The app 3.31.15 was run on
  Windows: the recall the prompt hook added reached the model. If the
  first prompt went before the session digest was ready, the next prompt
  carries it once. The prompt hook never searches for the digest itself,
  so a prompt that is not about memory never waits for the service.
- **The turn's reply.** Cursor runs `afterAgentResponse` (each assistant
  message) and `stop` (the end of the turn) as separate hooks, started at
  the same time, in either order, so the stop may run before the last
  message is kept. The stop takes a message kept for its own turn
  (`generation_id`), or the turn's reply in the transcript `stop` names,
  waiting at most 1 s for either. With neither, the turn stays open: the
  reply completes it when it comes. Else the chat's next prompt, its
  session end, a new chat's start or first prompt, or, 3 minutes later, any
  later hook of this user (the app runs no `sessionEnd` when a tab is
  closed) saves it, with the reply the
  transcript has by then, or with the user's words when it has none. A
  turn is saved once, and a message kept for another turn is never read
  for it: a reply is never saved with a later user message, nor twice. A
  follow-up sent while the earlier turn still works leaves that turn open:
  turns may overlap, and each stop and message completes its own turn only.
  Cursor runs a chat's turns one at a time, so once a later turn has ended,
  an earlier one whose stop never came is saved a minute later (at the
  next hook), with the transcript's reply if it has one; one with no later
  turn, after 10 minutes. Such a turn reaches the service after the later
  turns; each batch carries its messages' times.
- **Closing the console.** Turns queued when Cursor's CLI console is closed
  stay on disk, and a later hook sends them.
- **The app saves through its turn hooks.** It runs no `sessionEnd` when a
  chat tab is closed (and the CLI may run none when its console is closed
  without `/quit`), so its turns are saved by the turn hooks: every few
  turns; the last ones when another chat starts:
  - at the new chat's `sessionStart` and at its first prompt, at once, from
    every other chat with no turn running (its last `stop` came). A turn
    whose reply had not come by its stop goes with the reply the transcript
    has by then, else with the user's words. The new chat's first memory
    question waits up to 3 s for them to be stored, so a fact said in a tab
    closed just before is found when the service stores it in that time;
  - a chat whose turn still runs keeps its batch; the new chat's prompts in
    its first 10 minutes send it once it has been idle a minute;
  - else a chat goes once it has been idle for `QUOTEMEM_IDLE_FLUSH_MS`
    (5 minutes), with no further hook: after a hook that leaves unsent
    turns, the background sender waits for that time, saves them, and
    exits.

  So a new chat cuts the batch of every other chat between turns, in use or
  not: one more Add for each, never a turn lost or sent twice. Every hook
  prints its answer first, then sends and starts the background sender,
  within Cursor's timeout for that hook.
- **Older app versions.** The prompt hook cannot add context.
  The rule asks the model to call `quotemem_search` on memory questions and,
  since no forget note comes, to forget through the tools itself. The forget
  also runs before the prompt is submitted, so the tools may find nothing
  left; the model then says it is not in memory.
- **Both.** At `sessionEnd`, the transcript's new turns are matched in
  order with the messages the prompt hooks saw. A matched turn is never
  saved again; one whose `stop` hook never came takes its reply from the
  transcript. A turn no hook saw is saved, and every forget asked after it
  was said (by a later turn's hooks, or through the tools) runs again
  behind it. When prompt hooks ran but none of their messages is in the
  transcript, or their record is incomplete, nothing is saved from it that
  time and the record starts again, so a turn is never saved twice.

Per-turn recall reaches the model through `additional_context` in the
prompt hook's (`beforeSubmitPrompt`) answer, for that turn.
`QUOTEMEM_CURSOR_PROMPT_CONTEXT` sets when the hook sends it:
- `auto` (the default): on the app 3.20.17 and later, read from the
  payload's `cursor_version`, and on the CLI 2026.11.19 and later (its
  interactive mode; print mode runs no prompt hook);
- `on`: on every version;
- `off`: never. Recall is then the session digest and the tools.

```bash
mkdir -p ~/.cursor
cp integrations/cursor/hooks.json ~/.cursor/hooks.json   # merge by hand if you already have one
cp integrations/cursor/mcp.json ~/.cursor/mcp.json       # or add its "quotemem" entry
# in each project:
mkdir -p .cursor/rules && cp /path/to/QuoteMem/integrations/cursor/rules/quotemem.mdc .cursor/rules/
```

- `QUOTEMEM_SESSION_DIGEST=off` turns the digest off.
- What the hook adds to one prompt stays under 9,500 characters; Cursor
  moves longer text to a file. When notes and recall would be longer, the
  oldest notes are dropped; a block is never cut.
- Recall waits at most 10 s in Cursor's and Devin's prompt hooks, so the
  hook answers before their 15 s timeout.
- **A timeout you edit holds.** Cursor's and Devin's hooks read their own
  timeout from the hooks file that runs them: for Cursor,
  `~/.cursor/hooks.json`, then each workspace's `.cursor/hooks.json`; for
  Devin, the project's `.devin/hooks.v1.json`, then the `"hooks"` of
  `~/.config/devin/config.json`. The first entry whose command runs
  QuoteMem's hook for that event counts. The hook then answers ("carry on"
  at the latest) and ends the work after its answer 1 s before that
  timeout. Without such an entry it uses the timeouts `quotemem install`
  writes: 15 s for the prompt hook, 5 s for the others. When a shorter
  timeout ends the hook first, a later hook starts the background sender.
- Background agents are not remembered unless
  `QUOTEMEM_INCLUDE_SUBAGENTS=on`.

## Cline

Cline runs executable files named after the event:
- `UserPromptSubmit` does recall and forget, and adds the result as
  `contextModification`;
- `TaskComplete` captures the turn;
- `TaskCancel` closes the turn.

Cline runs no hook when a task ends or is left. So a new task's first
prompt sends, at once, the unsent turns of every other task with no turn
running, and its memory question waits up to 3 s for them to be stored.
A task whose turn still runs keeps its batch; the new task's prompts in
its first 10 minutes send it once it has been idle a minute.

```bash
mkdir -p ~/Documents/Cline/Hooks          # or ~/.cline/hooks
cp integrations/cline/hooks/* ~/Documents/Cline/Hooks/
chmod +x ~/Documents/Cline/Hooks/UserPromptSubmit ~/Documents/Cline/Hooks/TaskComplete ~/Documents/Cline/Hooks/TaskCancel
```

Then, in Cline's settings, turn on **Enable Hooks**.

- **MCP tools:** add this server in Cline's MCP settings:
  `"quotemem": {"command": "quotemem-hook", "args": ["mcp"]}`.
- **Without `quotemem-hook`:** if it is not on `PATH`, set `QUOTEMEM_HOOK` to
  its absolute path, or edit the hook files. A hook that cannot find it lets
  the turn go on.

Cline runs these hooks on macOS and Linux.

## goose

goose hooks cannot add context. goose's built-in `tom` extension, though,
adds the file named by `GOOSE_MOIM_MESSAGE_FILE` to every model call. So the
hooks work like this:
- The `UserPromptSubmit` hook writes the recall block or the forget note to
  `~/.quotemem/goose/moim.md`.
- `Stop` captures the turn and clears the file.
- `SessionEnd` flushes the session.

```bash
mkdir -p ~/.agents/plugins
cp -R integrations/goose/quotemem ~/.agents/plugins/quotemem
# goose must read the context file; put this in your shell profile:
export GOOSE_MOIM_MESSAGE_FILE=~/.quotemem/goose/moim.md
```

For the MCP tools, add an extension in `~/.config/goose/config.yaml`, or run
`goose configure`, choose "Add Extension" and then "Command-line Extension":

```yaml
extensions:
  quotemem:
    type: stdio
    name: quotemem
    cmd: quotemem-hook
    args: [mcp]
    enabled: true
    timeout: 300
```

The hook only ever writes its own file. Set `QUOTEMEM_GOOSE_MOIM_FILE` to move
it, and point `GOOSE_MOIM_MESSAGE_FILE` at the same path.

## Devin CLI and Devin Desktop

Devin uses Claude Code's hook format.
- `UserPromptSubmit` does recall and forget.
- `Stop` saves the final reply and closes the turn.
- `SessionEnd` flushes the session.

```bash
mkdir -p .devin && cp /path/to/QuoteMem/integrations/devin/hooks.v1.json .devin/hooks.v1.json
```

To install for every project, put the same object under `"hooks"` in
`~/.config/devin/config.json`. For the MCP tools, add a stdio server with
command `quotemem-hook` and argument `mcp`.

**Where the reply comes from.**
- Devin CLI 3000.5.20 and later send the final reply to `Stop` as
  `last_assistant_message` (Devin's changelog). Devin Desktop's local agent
  shares that harness.
- Without that field, a `transcript_path` in the payload is read instead, if
  the payload has one. The reader reads
  Claude Code's JSONL or an ATIF file. It takes the last reply only when the
  user message it answers is this turn's own message. A transcript that is
  not yet written up to this turn gives no reply, not an earlier turn's. An
  ATIF file over 2 MiB gives no reply.
- Without either, the turn keeps the user's messages only. The hook never
  waits for or fails on this.
- If another `Stop` hook makes Devin go on, the reply kept is the one at the
  first `Stop`.

Hooks do not run in Restricted Mode.

## OpenClaw

The [`openclaw`](../openclaw) folder is a native plugin. It runs in the
Gateway process:
- `before_prompt_build` does recall and forget and returns `prependContext`;
- `agent_end` captures the turn;
- `session_end` flushes the session.

The manifest also adds the MCP server.

```bash
openclaw plugins install /path/to/QuoteMem/integrations/openclaw
openclaw plugins inspect quotemem --runtime --json   # check that it loaded
```

**Settings.**
- Set the `QUOTEMEM_*` variables in the Gateway's environment or in
  `~/.quotemem/env`.
- Behaviour settings can also go in `plugins.entries.quotemem.config`:
  `captureEvery`, `recall`, `recallGate`, `topK`, `maxInjectChars`, `forget`
  and others; see `openclaw.plugin.json`. `maxInjectChars` takes 500 to
  50000, and a value below 2500 is raised to 2500.

**Whose memory.** By default all conversations share the owner's memory,
and only the owner uses it.
- A message without a channel sender (the local CLI or UI) is the owner's.
- A channel message is the owner's only when its sender is listed in
  `"ownerSenders"` (`"<sender id>"` or `"<channel>:<sender id>"`). Messages
  from anyone else get no recall, no forget and no capture, and one warning
  per sender is logged. Without this, anyone in a group or a DM could read
  and forget the owner's memories.
- With `"userScope": "sender"`, each channel sender (`<channel>:<sender id>`)
  gets their own memory.

**What is skipped.** Heartbeat, cron and system turns are never remembered.
Turns from another agent session are skipped unless `includeSubagents` is
on.

## Continue

Continue's hooks are not called from its chat loop, so the model does the
memory work itself through the MCP tools, as the rule asks. The tool
`quotemem_add` stays visible here (`QUOTEMEM_AUTO=0`).

```bash
mkdir -p .continue/mcpServers .continue/rules
cp /path/to/QuoteMem/integrations/continue/quotemem.yaml .continue/mcpServers/quotemem.yaml
cp /path/to/QuoteMem/integrations/continue/quotemem-rule.md .continue/rules/quotemem.md
```

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

## How the hooks behave

**Answers and failures.**
- Each hook reads its agent's JSON payload on stdin. It prints exactly the
  answer that agent expects, or nothing, and always exits 0.
- When QuoteMem is down, slow or not configured, the hook gives the agent's
  "carry on" answer at once.
- Recall waits at most `QUOTEMEM_RECALL_TIMEOUT_MS` (default 3 s). A
  forget waits at most `QUOTEMEM_FORGET_TIMEOUT_MS` (default 1.8 s, at most
  2.5 s); after that, it finishes in the background, and the next message
  says how it ended.

**Background sending.** Batches go to a disk queue. A detached background
process sends them, so no hook waits for `/memory/add`, and nothing is lost when
the agent quits.

**Logs and state.**
- Warnings go to `~/.quotemem/<agent>/quotemem.log`, never to stdout.
- The log holds counts, status codes and request ids. It never holds the
  token or message text.
- A configuration problem is logged once, in `problem.txt` in the same
  folder.

**Session ids** sent to QuoteMem are `<agent>:<agent session id>`, for
example `gemini-cli:…` or `cursor:…`.

**Hooks written for another agent.** Cursor and Devin also run the hooks in
Claude Code's settings files. If you set up both the Claude Code hooks and
that agent's own hooks, each turn would reach QuoteMem twice. So a Claude
Code hook stays silent when it runs inside another agent whose own QuoteMem
hooks are installed:
- Cursor: its input carries `cursor_version`. Cursor's hooks are per user,
  so they count as installed when `~/.cursor/hooks.json` or a workspace
  root's `.cursor/hooks.json` runs `quotemem-hook`, or when the Cursor hook
  left its marker, `~/.quotemem/native/cursor.json` (`native/cursor.json`
  under `QUOTEMEM_STATE_DIR` when that is set), in the last 30 days.
- Devin: its input has `prompt_id` but no `transcript_path` (Claude Code
  always sends one), and `DEVIN_PROJECT_DIR` is set. Devin's hooks are per
  project, so only the files count: `.devin/hooks.v1.json` in the project
  or a folder above it, up to the repository root, and
  `~/.config/devin/config.json`.

Without the agent's own hooks, the Claude Code hooks still do the work.
Where Devin reads the Claude Code hooks, set up either those hooks or the
Devin hooks, not both. Inside a Cursor that reads
`additional_context` from the prompt hook, the Claude Code prompt hook
answers with that field only.

## Tests

```console
node --test integrations/hooks/test/hosts.test.mjs
npm test                                     # every adapter's tests, from the package root
node integrations/hooks/bundle.mjs --check   # the bundles' copies of the runner and server
node integrations/core/sync.mjs --check      # the bundles' copies of the core
```

**How the tests run.**
- Each hook runs exactly as configured: the command from the shipped
  `hooks.json`, hook file or plugin script, through `sh -c`.
- Each gets the agent's own payload shape, a temporary `HOME` and a stub
  QuoteMem service.
- The OpenClaw plugin runs in a test host that offers OpenClaw's plugin API.

**What they cover.**
- capture batches and their session ids;
- recall in each agent's output format, Cursor's version gate, its session
  digest when the first prompt comes early, and the 10 s recall bound;
- Cursor CLI turns and forgets saved from the transcript at each run's end
  (recorded payload shapes), once each, also after the plugin's state is
  lost, a rewrite or a failed queue write, and never a turn the app's hooks
  saved; forgets applied again behind words saved later;
- Devin's reply from `last_assistant_message`, from a transcript, or none,
  and never an earlier turn's reply from a transcript that lags;
- forget order and notes;
- session-end flushes;
- the services being down, slow or unconfigured;
- the token sources;
- the MCP launchers;
- that no token or message text reaches a log.

**Editing the code.** `gemini-cli` and `openclaw` carry copies of the hook
runner and the MCP server. Edit the sources here and in `../dsh/server.mjs`,
then run `node integrations/hooks/bundle.mjs` and
`node integrations/core/sync.mjs`.
