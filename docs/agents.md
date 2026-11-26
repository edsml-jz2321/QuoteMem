# Use with your agents

One npm package, `quotemem`, carries every agent adapter, the MCP server and
a small command, `quotemem`. Everything runs on your own computer: the agents'
plugins share one memory service (`quotemem serve`, on `127.0.0.1:8765`), so a
fact learned in one agent is recalled in the others. Every adapter uses the
same memory by default: `QUOTEMEM_USER_ID`, which defaults to your OS user name.

Quick start (to run a clone instead, follow "From a clone" below):

1. Install the plugin for your agent (see the [install matrix](#install-matrix)).
2. Run this once:

   ```bash
   npx quotemem setup
   ```

   It asks which models the service calls, saves the settings to
   `~/.quotemem/service.env` (mode 600), and checks them with
   `quotemem check`. Keys are typed without echo and never put on a command
   line. Three choices are offered:

   | Preset | Models | Trade-off |
   |---|---|---|
   | `default` (recommended) | The competition configuration: chat `deepseek-v4.1-flash-1017` with thinking at high effort, embeddings `qwen3.7-text-embedding` (1024 dimensions), both through Alibaba Cloud Model Studio (DashScope), one API key | Saving memories is slower: a large batch of corrections can take several minutes. Thinking tokens are billed as output tokens. |
   | `fast` | The same models and key, thinking off | Memories are saved in seconds. |
   | `custom` | Any OpenAI-compatible chat and embedding endpoints, models and keys; structured output `json_schema` or `json_object`; thinking off, high or the provider's default | Yours to choose |

   Then it asks the review, which decides whether a new memory repeats,
   updates or corrects a stored one (`--review` sets it without asking;
   saved as `QUOTEMEM_REVIEW`):

   | Review | What decides | Trade-off |
   |---|---|---|
   | `fast` | The local review rules of the competition submission; no review calls | The fastest and cheapest. A change of state said outright ("Hangzhou, not Beijing") retires the old memory. |
   | `standard` (default) | The chat model reviews only the new memories that may repeat, update or correct a stored one | Review calls only where they matter |
   | `full` | The chat model reviews every new memory | The slowest and most careful, with the most model calls |

   The review is fixed when the memory store is created: the service refuses
   to open a store with another one. To change it later, stop the service and
   rebuild the store into a new data directory
   (`npx quotemem stop`, then
   `npx quotemem rebuild --from ~/.local/share/quotemem --to ~/.local/share/quotemem-full --review full`),
   then run `npx quotemem setup --review full` and put
   `QUOTEMEM_DATA_DIR=~/.local/share/quotemem-full` in
   `~/.quotemem/service.env`. `npx quotemem status` shows the review.

   Both presets take a Model Studio API key from
   [bailian.console.aliyun.com](https://bailian.console.aliyun.com) (China
   region), and bill your own Model Studio account at its usual prices. To
   switch later, run `npx quotemem setup` again (`--preset fast` skips the
   question), or edit a line of `~/.quotemem/service.env`: each
   `QUOTEMEM_*` value there overrides the preset, for example
   `QUOTEMEM_CHAT_REASONING=off` turns thinking off. For scripts,
   `npx quotemem setup --yes --preset default|fast|custom --review fast|standard|full` asks nothing and
   takes the keys from `QUOTEMEM_CHAT_API_KEY` and `QUOTEMEM_EMBED_API_KEY`
   (or `DASHSCOPE_API_KEY` for the presets); `npx quotemem setup --help`
   lists the options.
3. Done. When an agent first needs memory, its plugin starts the service in
   the background (`uvx --from quotemem==<plugin version> quotemem serve`,
   so [uv](https://docs.astral.sh/uv/) must be installed). That turn goes on
   without memory; the next ones use it. The first start downloads the
   service and can take a few minutes; `~/.quotemem/service.log` shows its
   progress.

   **Behind a proxy.** The started service keeps the proxy settings
   (`HTTPS_PROXY`, `NO_PROXY`) of the tool that starts it, and so do
   `quotemem setup` and `quotemem check`. If your proxy is only for PyPI and
   the model host must be reached directly, add that host to `NO_PROXY`, for
   example `NO_PROXY=127.0.0.1,localhost,.aliyuncs.com` for Model Studio.

The service keeps running after the agents exit, because they all share it.
`npx quotemem status` shows whether it runs, its pid, URL and log;
`npx quotemem stop` stops it. It starts again when an agent needs it, unless
`QUOTEMEM_AUTOSTART=0` is set where the agents start. Only one instance is
ever started, and only for a service URL on this computer (`QUOTEMEM_URL`
unset, or `localhost`, `127.0.0.1` or `[::1]`).

**Two people signed in at once** on one computer (fast user switching,
remote desktop, a shared server) cannot both use port 8765: the second
person's plugins reach the first person's service, which refuses their
token, so they get no memory. `npx quotemem doctor`, `npx quotemem status`
and the plugins' log say so, and name a port of that person's own
(20000–25999, from the user name and home folder). Put it in that person's
`~/.quotemem/env` (`QUOTEMEM_URL=http://127.0.0.1:<port>`) and in the
environment their Claude Code, Codex and DSH start in, and their plugins start
their own service there. Claude Code can also take the URL in its plugin's URL
option.
The first person keeps 8765.

**If several people use the computer at once**, give yourself a port of your
own as above, start the service before you open your tools, and use QuoteMem
there only with people you trust.

No token needs to be copied: the service creates one in `<data-dir>/token` on
its first start, and every plugin reads it from there. The data folder is
`~/.local/share/quotemem`; a `QUOTEMEM_DATA_DIR` line in
`~/.quotemem/service.env` moves it. The started service takes its settings,
token and data folder from that file only, never from the environment of the
tool that happened to start it, so every tool gets the same service. Leave
`QUOTEMEM_TOKEN` unset for it (or put the same token in `service.env`), and
use the same `QUOTEMEM_URL`, or none, in every tool.

**Running the service yourself** still works: start
`uvx quotemem serve` (or `quotemem serve`) with the model settings of
[Run the HTTP service](service.md#run-the-http-service) in its environment. Only the
npm command reads `~/.quotemem/service.env`: `npx quotemem serve` and
`npx quotemem check` add its settings, while a plain `uvx quotemem serve` or
`quotemem serve` from pip does not. The plugins find a running service and start
nothing. Started by hand, it keeps its data in `~/.local/share/quotemem`, or
in `$XDG_DATA_HOME/quotemem` when that is set (`QUOTEMEM_DATA_DIR` moves it),
and the agents read its `<data-dir>/token` the same way, so they need no token
setting either.

**From a clone.** The automatic start and setup's check run
`uvx --from quotemem==<plugin version>`, the published package. To run a clone
instead (after `uv sync`), write the settings without the check, then check and
start the service from the clone with the same file:

```bash
node integrations/cli/quotemem.mjs setup --no-check
uv run --env-file ~/.quotemem/service.env quotemem check
uv run --env-file ~/.quotemem/service.env quotemem serve
```

Set `QUOTEMEM_AUTOSTART=0` where the agents start, so no plugin tries to
start the published service while yours is stopped.

**Do not export the token in the shell that starts an agent.** The agent's
own shell tool inherits that environment, so the model could print the token
with `env`. For a service on another computer, keep its token in a file that
only you can read, and point the agents at the file:

- **DSH, Claude Code and Codex** read `QUOTEMEM_URL` and
  `QUOTEMEM_TOKEN_FILE` (the file's path, not the token) from the environment
  they start in. Claude Code can instead keep the token in its plugin option,
  which goes only to the plugin's hooks and MCP server.
- **The other agents** often start hooks without your shell environment.
  They also read `~/.quotemem/env`:

  ```bash
  mkdir -p ~/.quotemem && chmod 700 ~/.quotemem
  # save the service's token, and nothing else, in ~/.quotemem/token; then:
  chmod 600 ~/.quotemem/token
  printf 'QUOTEMEM_URL=https://memory.example.org\nQUOTEMEM_TOKEN_FILE=~/.quotemem/token\n' > ~/.quotemem/env
  chmod 600 ~/.quotemem/env
  ```

A variable set in the environment wins over the file. `QUOTEMEM_TOKEN` itself
is still read when it is set, so unset it if an old setup exported it. The
adapters need Node 20.3 or later on `PATH`.

## Install matrix

In the tools marked "seamless" below, conversations are saved and recalled by
themselves, and "forget my ..." is carried out for you. Ask "what do you
remember about me?" in any of them.

For the rows that use `quotemem`, install the package once with
`npm install -g quotemem`. `quotemem install <tool>` only prints what it
would do; add `--write` to write the files. It never writes the token.
A file it changes keeps its permissions, is written through a symlink, and
its old text is kept as `<file>.quotemem.bak`.
`quotemem doctor` checks the settings, the token, the service, its
autostart and `uvx`. The service's own commands also run through this
command: `quotemem serve`, `check`, `memories`, `export`, `rebuild`, `delete-user`,
`repair` and `bench` are handed to `uvx quotemem` with their arguments and
the settings of `~/.quotemem/service.env`. `memories`, `delete-user` and
`repair` call the running service as the plugins do: with their
`QUOTEMEM_URL`, user and token, from the environment or `~/.quotemem/env`.
The token of `service.env`, or of the service's data folder, goes only to a
service on this computer, never to another one. The Python package installs a
command named `quotemem` too, which runs only the service commands. With both
installed, the first one on `PATH` answers; `npx quotemem` always runs the npm one.

| Agent | Install | Memory |
|---|---|---|
| DSH | `dsh plugin --profile quotemem add quotemem` (DSH installs plugins with [pnpm](https://pnpm.io), so `pnpm` must be on `PATH`), then start `dsh --profile quotemem` | seamless |
| Claude Code | `claude plugin marketplace add edsml-jz2321/QuoteMem --sparse .claude-plugin integrations`, then `claude plugin install quotemem@quotemem` | seamless |
| Codex CLI | `codex plugin marketplace add edsml-jz2321/QuoteMem --sparse .agents --sparse integrations/codex`, then `codex plugin add quotemem@quotemem`, then trust the hooks once with `/hooks` (without the plugin system: `quotemem install codex --write`) | seamless after the one-time trust |
| OpenCode 1.x, 2.x | `quotemem install opencode` prints the block to merge into `opencode.jsonc` | seamless |
| Gemini CLI | `quotemem install gemini` prints the `gemini extensions install` command | seamless |
| Cline (VS Code) | `quotemem install cline --write`, then turn on Enable Hooks | seamless |
| goose | `quotemem install goose --write`, then set `GOOSE_MOIM_MESSAGE_FILE` as it prints | seamless |
| OpenClaw | `quotemem install openclaw` prints the `openclaw plugins install` command | seamless |
| Cursor | `quotemem install cursor --write`, plus the rule it prints, in each project | Cursor CLI in interactive mode, 2026.11.19 and later: seamless through the turn hooks. Cursor CLI in print mode (`-p`): recall is a session-start digest plus the tools; each run's turns are saved at its end, from the chat's transcript that `sessionEnd` names; forget through the tools or when the conversation is saved; the MCP tools need `--approve-mcps --force`. Keep `QUOTEMEM_*` settings in `~/.quotemem/env`, which both the hooks and the MCP tools read. App 3.20.17 and later: per-turn recall through the prompt hook, and a note to the model for each forget. Every app version saves its turns through the turn hooks; on versions before 3.20.17, recall is a session digest plus the tools |
| Devin CLI and Desktop | `quotemem install devin --write` in each project | seamless; with Devin CLI 3000.5.20 and later, each turn is saved with the agent's reply |
| Continue | `quotemem install continue --write` in each project | through the tools: the model saves and searches, guided by a rule |
| Any other MCP client | command `quotemem-mcp` (stdio) | tools only |

**From a clone** (`git clone https://github.com/edsml-jz2321/QuoteMem.git`),
run `node integrations/cli/quotemem.mjs install <tool>` in place of
`quotemem install <tool>`. The files it writes then point at the clone. The
Claude Code and Codex marketplace commands read this GitHub repository, so
they need no npm package. DSH installs the bundle from npm, or from a packed
tarball of the clone (see the [DSH guide](../integrations/dsh/README.md)).

"Seamless" means the agent's own hooks or plugin do the work: the
conversation is saved every few turns, memory questions get the relevant
quoted memories as hidden context, and "forget …" requests are carried out
and confirmed through a note to the model. The plugins read everyday
requests such as "forget Ben's name" and "could you please not remember my
phone number?" The done note counts related memories without quoting them
and counts reworded memories separately. A follow-up note reports any extra
removals from saving the request turn. The user calls no tool. When a
new chat starts, the same agent's other chats' unsent turns go out, and the
new chat's first question about memory waits up to 3 s for them to be
stored. When depends on whether the agent reports a chat's end:
- **Cursor and Cline** (a closed Cursor tab runs no session end, a closed
  console may run none; Cline has none): at the new chat's start (Cursor)
  and at its first prompt,
  every other chat with no turn running goes out at once. A fact said seconds
  before in a chat closed just now is then found by the new chat's first
  question, when the service stores it within those 3 s. A chat whose turn
  still runs keeps its batch, and goes once it has been idle a minute.
- **Claude Code, Codex, Gemini CLI, Devin, goose, OpenCode, OpenClaw and
  DSH**: a closed chat goes out at its own session end. A new chat sends the
  other chats only once they have been idle a minute, at its start (Claude
  Code, Codex) or at any of its prompts in its first 10 minutes, so chats in
  use (many workers at once) keep their batches. A fact said seconds before
  in a chat left open goes out at the new chat's first hook after that chat
  has been idle a minute.

Otherwise a chat's last turns go out when it ends, or before a forget. A
chat's last turns are saved once it has been idle for 5 minutes
(`QUOTEMEM_IDLE_FLUSH_MS`), with no further prompt. (On Windows, where
PowerShell cannot start background processes, they are saved at the next
prompt or session start instead.) The MCP tools stay available for explicit requests: search, history,
provenance (the quoted evidence behind a memory), a list of what is
remembered (`quotemem_memories`) and a two-step forget.
When the service cannot be reached, the turns wait on disk and go out on a
later hook, after a retry delay of up to 5 minutes, or at once when a hook's
search reaches the service again. Turns queued when an agent's console is
closed also go out on a later hook.

To keep a forget in force, the plugins keep each forget request locally:
the user's statement, secrets redacted, in `forgotten/` under `~/.quotemem`,
and also under `QUOTEMEM_STATE_DIR` when that is set (one file per user and
service; mode 600 in a 700 folder on macOS and Linux, the user profile's
own access rights on Windows). So the statement text of a kept forget is
also stored under `~/.quotemem/forgotten` (mode 600), even when
`QUOTEMEM_STATE_DIR` points elsewhere, such as another volume. Set
`QUOTEMEM_STATE_DIR` only in `~/.quotemem/env`, never in one agent's or MCP
server's own settings: the hooks and the MCP server then use one folder,
and `quotemem doctor` warns about a server it finds with another. A forget
is kept in force for a week, at most 200 per user; its text stays in the file
until a later forget rewrites it, so delete the file to remove the text sooner.
Words said before a forget and saved later normally get it again; words known
to be said after it are kept.
A forget through the MCP tools is kept only next to an automatic-memory
plugin, once confirmed or when nothing matched.

## Agent support at a glance

| Agent | Silent capture | Silent recall | Silent forget | MCP tools |
|---|---|---|---|---|
| DSH | yes | yes | yes, with a note to the model | yes |
| Claude Code | yes | yes | yes, with a note to the model | yes |
| Codex CLI | yes, after a one-time `/hooks` trust | yes | yes, with a note | yes |
| OpenCode 1.x and 2.x | yes | yes | yes, with a note | yes |
| Gemini CLI | yes | yes | yes, with a note | yes |
| Cline (VS Code) | yes, once hooks are turned on | yes | yes, with a note | yes |
| goose | yes | yes, through its context file | yes, with a note | yes |
| Devin CLI and Desktop | yes; with Devin CLI 3000.5.20 and later, with the agent's reply | yes | yes, with a note | yes |
| OpenClaw | yes | yes | yes, with a note | yes |
| Cursor | CLI print mode: at the end of each run, from the chat's transcript; CLI interactive mode and app: each turn | CLI interactive mode: yes, 2026.11.19 and later; app: yes, 3.20.17 and later; everywhere, a session-start digest, and the model can search | CLI print mode: when the conversation is saved, without a note, or through the tools; CLI interactive mode: with a note; app 3.20.17 and later: with a note | yes |
| Continue | no | no | no | yes, with a rule (the model decides) |
| Any other MCP client | no | no | no | yes |

Cursor's per-turn recall uses `additional_context` on its prompt hook.
Cursor and Devin also run the hooks in Claude Code's settings files; when
their own QuoteMem hooks are installed, those Claude Code hooks stay silent
inside them, so a turn is saved once. Where Devin reads the Claude Code
hooks, set up either those hooks or the Devin hooks, not both.

## Where each adapter lives

| Folder | What it is | Guide |
|---|---|---|
| [`integrations/dsh`](../integrations/dsh) | DSH bundle: automatic memory plugin, MCP server and skill | [README](../integrations/dsh/README.md) |
| [`integrations/claude-code`](../integrations/claude-code) | Claude Code plugin: hooks and MCP server | [README](../integrations/claude-code/README.md) |
| [`integrations/codex`](../integrations/codex) | Codex plugin, and `install.mjs` for a setup without plugins | [README](../integrations/codex/README.md) |
| [`integrations/opencode`](../integrations/opencode) | OpenCode plugin for 1.x and 2.x | [README](../integrations/opencode/README.md) |
| [`integrations/hooks`](../integrations/hooks) | One hook program for Gemini CLI, Cursor, Cline, goose and Devin; the OpenClaw plugin; Continue | [README](../integrations/hooks/README.md) |
| [`integrations/core`](../integrations/core) | The zero-dependency code every adapter shares, and the guide to writing a new adapter | [README](../integrations/core/README.md) |
| [`integrations/cli`](../integrations/cli) | The `quotemem` command: `setup`, `status` and `stop` of the local service, plugin setup, and the service commands through `uvx quotemem` | `quotemem --help` |

The package's entry points:

- bins: `quotemem`, `quotemem-mcp` (the MCP server) and `quotemem-hook`
  (the hook program for Gemini CLI, Cursor, Cline, goose and Devin);
- `quotemem hook <agent> <event>` runs the hook for `claude-code`, `codex`,
  `gemini`, `cursor`, `cline`, `goose` or `devin`;
- subpath exports: `quotemem/dsh` (also the package's main entry, for DSH),
  `quotemem/core`, `quotemem/opencode` and `quotemem/openclaw`.

The adapters and the MCP server run on Node 20.3 or later and have no
dependencies. DSH itself needs DSH 0.1.2-rc.1 or later (no upper bound), with
its own prerequisites (Node 24 and pnpm).
