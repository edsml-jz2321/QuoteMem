# QuoteMem adapter core

Shared code for every QuoteMem host adapter: DeepSeek Harness (DSH) today,
and hook-based agents such as Claude Code and Codex. It makes memory work
without the user noticing:

- **Silent capture.** The conversation is saved every few user turns.
- **Silent recall.** When the user asks about the past or about themselves,
  the relevant memories are inserted into the model's context.
- **Silent forget.** When the user asks to forget something, it is forgotten
  and the model gets one short note so it can confirm.

These are zero-dependency ES modules for Node 20.3 or later. Nothing here
throws into the host: when something fails, the turn goes on without memory
and one warning is logged.

This folder is the source of truth. The npm package imports it as
`quotemem/core`, and the DSH plugin imports it directly. Some hosts install
only one adapter folder (a Claude Code or Codex plugin cache, a Gemini CLI
extension, an OpenClaw or OpenCode plugin folder), so those adapters ship a
byte-identical copy in their own `core/` folder. The hook runner in
`integrations/hooks` has one too, because it is copied into the Gemini CLI
and OpenClaw bundles.

```console
node integrations/core/sync.mjs           # write the copies (targets in sync.mjs)
node integrations/core/sync.mjs --check   # fail when a copy differs
node --test integrations/core/test/*.test.mjs   # the core tests (the shell expands the glob)
npm test                                       # every adapter's tests, from the package root
```

Edit only `integrations/core/`, then sync. `node integrations/check_package.mjs`
checks every copy, the MCP server copies and the package manifest.

## The memory service on this computer

QuoteMem runs on the user's own computer. After a one-time
`npx quotemem setup` (model settings in `~/.quotemem/service.env`, mode 600;
the recommended `default` preset is the competition configuration,
`deepseek-v4.1-flash-1017` with thinking at high effort and
`qwen3.7-text-embedding` through Model Studio with one key, so writes are
slower; `fast` turns thinking off; `custom` takes any OpenAI-compatible
endpoints; each `QUOTEMEM_*` line of the file overrides the preset; the
review, `QUOTEMEM_REVIEW` = `fast`, `standard` (the default) or `full`, is
asked too and is fixed when a store is created, so changing it later means
`quotemem rebuild --review` into a new data directory),
`autostart.mjs` starts the service when a request to a loopback service URL
(`QUOTEMEM_URL` unset, or `localhost`, `127.0.0.1`, `[::1]`) finds nothing
listening:

- `http.mjs` calls `ensureService` on a refused connection. It is synchronous
  and quick: it starts `uvx --from quotemem==<release> quotemem serve` fully
  detached, holding none of the hook's handles (its own session; on
  Windows hidden, through `Start-Process`: see Flusher lifetime), with
  `service.env` over only the base of this environment (paths, locale,
  proxies and `UV_*`; never a token, data folder or model setting, so the
  service is the same whichever tool starts it), and output to
  `~/.quotemem/service.log` (moved to `service.log.1` at each start, cut
  when it passes 4 MiB). The request itself still fails, so recall is skipped
  with the usual note and captures stay in the disk queue; later requests
  find the service. A start in progress does not open the recall breaker.
- One instance only, across every tool and process: the starter takes
  `~/.quotemem/service.lock` with `O_EXCL` and records the service's pid and
  start time. A lock whose process is gone is stale; a recorded process that
  is no longer the service (a reused pid) is stale too (on Windows a
  detached background check finds that, so no hook waits for PowerShell).
  The host name is not compared, only the boot time and pid. A start whose
  process died within a minute is not repeated before that minute passes.
  The lock serves every port: a tool whose `QUOTEMEM_URL` names another
  port than the running instance gets one note to use the same URL
  everywhere.
- No token needs configuring: `resolveSettings` reads the service's own
  `<data-dir>/token` (data folder from `QUOTEMEM_DATA_DIR` or `XDG_DATA_HOME`
  in `service.env`, else `~/.local/share/quotemem`, as the started service
  decides it), and before the first
  start, a set-up service counts as configured; the token is read once the
  service has created it.
- Without `service.env` or without `uvx` nothing starts; the first memory
  question of a session that cannot be searched also gets one note saying
  how to set it up. `QUOTEMEM_AUTOSTART=0` turns starting off. A non-loopback
  URL never starts anything.
- The service keeps running after the tools exit; `quotemem stop` stops the
  recorded instance only, after checking its command line.

## How an adapter uses it

Every adapter maps its host's events onto `HookMemory`. It keeps each
session's turns in a file, and a detached background flusher sends the
batches. So a hook returns in milliseconds, a batch survives the host quitting,
and one queue orders every batch and forget.

- **Short-lived hook processes** (each event starts a new process, as in
  Claude Code, Codex or Gemini CLI) call it from each hook.
- **Long-lived host processes** (the plugin lives as long as the session, as
  in DSH or OpenCode) keep one `HookMemory` and also queue what is left when
  the host shuts down. The DSH plugin (`integrations/dsh/automem.mjs`) and
  the OpenCode plugin are examples.

```js
import path from 'node:path';
import { HookMemory, fileSink, makeLog, resolveSettings } from './core/index.mjs';

const settings = resolveSettings({}, process.env, { defaultStateDir: path.join(dataDir, 'quotemem') });
if (settings.problem) process.exit(0); // automatic memory is off; stay silent
const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
const memory = new HookMemory({ settings, log, host: 'claude-code' });

// The user sent a message:
const { context } = await memory.prompt({ session, turn: promptId, text: prompt });
if (context) printAdditionalContext(context);   // the host's own output format

// The turn ended; text is the final assistant reply:
await memory.reply({ session, turn: promptId, text: lastAssistantMessage });

// The user interrupted the turn, or it failed:
await memory.reply({ session, turn: promptId, completed: false });

// The session ended:
await memory.end({ session });
```

A hook must never print anything except its host's output format. Log to a
file sink, as above.

## What HookMemory does

**`prompt({session, turn, text, time})`** returns
`{context, kind: 'recall' | 'forget' | null}`.

1. **Idle sweep.** Sessions idle for `QUOTEMEM_IDLE_FLUSH_MS` have their
   pending turns queued. A turn held open for its reply (`hold`) is saved
   after 3 minutes (`HELD_TURN_MS`), with the reply its transcript has by
   then, else with the user's words. Sessions untouched for a week are
   closed, their turns queued and their files removed. This also runs when
   the idle flush is off.
2. **A new session.** In a session's first 10 minutes (`NEW_SESSION_MS`),
   each prompt queues the pending turns of this user's other sessions in
   the state folder that have been idle a minute (`OTHER_IDLE_MS`): a chat
   closed without a session end, or left for this one. A chat still in use
   keeps its batch. On a host whose chats can close without a session end
   (`reliableEnd: false`: Cursor and Cline in the hook runner), the
   session's start and its first prompt also queue, at once, every other
   session of this host with no turn in flight (none open or running, other
   than one held for its reply after its end), a held turn with the reply
   its transcript has, else the user's words: a tab closed seconds before
   is found. A session with a turn in flight keeps its batch. The
   session's first prompt, when it asks about memory (the recall gate),
   waits for the batches queued for it (by its start or by itself) to be
   stored, not for other sessions' own batches. A session whose lock
   another process holds (its start, queueing that session) is looked at
   after the others, whose batches are sent first, and its lock is waited
   for. All of it takes at most 3 s (`FIRST_PROMPT_WAIT_MS`) and stays
   within the recall deadline, which then gets what is left, at least 1 s.
   A prompt that searches nothing waits for nothing. A host's
   session-start hook queues them earlier, after its answer
   (`sessionStarted`). Within a session,
   turns still go in batches of `QUOTEMEM_CAPTURE_EVERY`; this costs at
   most one more Add for each other session with unsent turns, once per new
   session, and, where the end is reliable, none for chats in use (many
   workers running at once). A real service often needs longer than 3 s to
   extract a batch: the wait helps a fast store, and a later question finds
   the rest.
3. **Records the message** under the session lock.
   - A turn still open without a reply is closed first, keeping the user's
     words.
   - A second call with the same `turn` records nothing; a duplicate install
     of the same hooks is harmless.
   - A host with no turn ids (`turn: null`) gets a new turn number for every
     prompt. The same text again within 5 s counts as the same prompt.
4. **Forget request.** When the message asks to forget something (see
   [Silent forget](#silent-forget)), the forget runs and `context` is its
   note. No recall runs on that turn.
5. **Gated recall** otherwise:
   - The message is searched under `QUOTEMEM_RECALL_TIMEOUT_MS`.
   - Right after a forget the service rebuilds the user's search index for a
     second or two and answers `429 index_rebuilding`. The search is then
     sent again every 250 ms (sooner when Retry-After says so) while the
     deadline allows. No other answer is sent again, and the wait never
     opens the breaker.
   - Repeat suppression is per session and kept in the session file.
   - The circuit breaker is kept in `recall-breaker.json`. A refused
     connection, or two missed deadlines in a row, skips recall for a while
     (30 s, doubling up to 5 minutes). One slow search does not.
   - A forget that was still running when its note was given, and that then
     forgot nothing, is reported in a short note on the next message.
   - `context` is the recall block, or null.

**`reply({session, turn, text, completed})`** closes the turn and keeps the
final reply when the turn completed. Every `QUOTEMEM_CAPTURE_EVERY` user turns
it queues a batch and starts the flusher.

**`end({session})`** queues what is left and starts the flusher. The
session file stays: a resumed session (`codex exec resume`, `claude
--resume`) numbers its turns on and keeps its recall memory. A session file
untouched for a week is removed.

**`endSync({session})`** is `end` for a host process that is exiting. It is
synchronous, so it can run in an `exit` handler.

**`catchUp({session, file})`** saves the turns of a whole conversation
that no prompt hook recorded. Cursor's CLI in print mode runs no prompt or
stop hook; its `sessionEnd` names the chat's transcript, which holds every turn so far
(`readConversation` in `transcript.mjs` reads it). Call it before `end`,
with `completed` when the run ended completed; `begin({session, file})` at
the chat's start notes when it began and how much its transcript held then.
- Only what was added since the last call is read: a byte offset, with the
  line before it as a guard. A transcript rewritten or shortened since (a
  checkpoint restore) is read again whole; a turn saved before has the same
  id and body, so the service stores it once. When more than 32 MiB of new
  text waits, only its last 32 MiB are read, the turns before them are not
  saved, the place moves on, and the log says so.
- A turn is complete once a later user message, or (Cursor) its
  `{"type":"turn_ended"}` line, follows it, or, as the last turn of a run
  that ended completed, once it has a reply; another host's last turn once
  it has a reply. A turn not complete yet waits for the next call. A reply
  is the turn's last assistant text: the text before a tool call ("I'll
  look up…") is not it. A reply added to a turn after it was saved is not
  saved again: the first saved reply stands. Cursor rewrites the transcript
  each run with one `turn_ended` at the end, so the place sits after the
  last complete turn's last line, never after that marker.
- The new turns are matched, in order, with the user messages the prompt
  hooks recorded (the longest in-order match, `alignTurns`). A matched turn
  was saved by the hooks; the open one among them, which no stop hook
  closed, takes its reply from here. When prompt hooks ran but none of
  their messages is among the new turns, or their record is incomplete (a
  chat begun before it, or more than 512 prompts), none is saved, the
  record starts again, and the log says so: a turn is never saved twice.
- Every other turn is its own batch, with a request id from its place in
  the conversation and its words (`:x<n>-<n>:`, no message time; a turn the
  transcript gives no readable time carries none), so a turn sent again
  (after the session file was lost, say) has the same id and body. Each
  batch also knows when its turn was said at the earliest: after the
  previous call for text added since, after the chat began for text added
  after what the transcript held at `begin`, else not known.
- A forget request among them is queued as on a prompt, after the turns
  before it and before its own turn. Nothing waits for it; the next prompt
  of this user reports how it ended.
- The place moves only after the batches are queued, so a failed queue
  write loses nothing.

**Forgets applied again.** Every forget the user asked for is kept: one a
hook ran, and one the model carried out through the MCP tools next to an
adapter (confirmed, or one that found nothing because the words were not
saved yet; never a plan prepared and left). It is the user's statement,
redacted of secrets but in their words, kept locally in
`~/.quotemem/forgotten/<owner>.json` (mode 600 in a 700 folder, one file per
user and service) only to re-apply the forget. It is kept for a week: the
longest a turn said before it can still wait unsaved, since a session file
is closed, and its turns queued, a week after it was last touched. A batch
queued later that holds the user's words from before a forget gets that
forget again right behind it, waiting until the batch is stored. So words
saved after a forget never bring the fact back: a turn read late from a
transcript, or a run of Cursor's CLI, which is saved only when it ends.
The same holds for the request's own message when the service's in-chat
check does not read the request (a retraction, 把…忘掉, one after a list
marker) and the message says something before it ("1. My locker code is
HERON. 2) Please forget my locker code."): the forget follows that
message's batch too. A message that says something only after such a
request keeps it.
Words known to be said after it are kept: a batch whose turn was said after
the previous transcript reading, or after the chat began, when the forget
came before that. A turn whose time shows it was said a minute or more
before the forget gets it whatever its bound says (a clock set back). Where
the order is unknown the forget applies: within one run and one minute
(Cursor's times have minutes only) whether a turn came after the forget
cannot be told; a
chat whose plugin state was lost, or resumed where its sessionStart was
not seen, has no bound; nor has one whose transcript `begin` could not
read, or one whose sessionEnd names another file than its sessionStart. A
missing transcript at the start means a new chat only for Cursor's CLI,
whose sessionStart runs for new chats only. The start's bound holds for
turns after the last turn the transcript held then, so a `turn_ended` line
Cursor moves on its next rewrite changes nothing. A forget through the MCP
tools also queues this
user's pending turns in every state folder at once (`QUOTEMEM_STATE_DIR`
and each agent's folder in `~/.quotemem`), as a prompt hook's forget does
in its own. The file is kept in `~/.quotemem/forgotten`, and also in
`QUOTEMEM_STATE_DIR` when the environment sets it; both are read. So an MCP
server given its own `QUOTEMEM_STATE_DIR` and hooks without it still share
the forgets. At most 200 forgets are kept. On Windows the file has
no POSIX mode: it takes the user profile's access rights.

**No memory text on disk.** A forget's outcome keeps memory text (closest
statements, the matched or bundled memory) only while the hook that asked
may still be waiting to quote it in its note; past its forget deadline it
is written as counts only (`whole` counts the memories forgotten whole),
and an outcome no hook read in time is reduced to counts on the next
flusher run. A forget applied again, or read from a transcript, is written
as counts at once. A retry's carry in the queue file holds the plan (ids
and hashes), ids and counts, but no statements; only a forget of a stored
statement found by search keeps that statement while it waits, since the
retry prepares it again.

**`sweep()`** queues what idle sessions left; `prompt` already calls it.

**`sessionStarted({session})`** queues the unsent turns of this user's
other sessions idle a minute (with `reliableEnd: false`, also those of this
host between turns, at once) and starts the flusher; the new session's
first memory question waits for them (see `prompt`). The Cursor, Claude
Code and Codex adapters call it from their session-start hook.

**`new HookMemory({..., overlap, reliableEnd})`**: `overlap: true` lets a
prompt leave a turn that still runs open (Cursor runs a follow-up while the
earlier turn works). `reliableEnd: false` is for a host whose chats can
close without a session end (a Cursor tab, a Cline task); the default
`true` keeps a new session from cutting the batches of chats in use.

**`hold({session, turn, transcript})`** keeps a turn open for a reply that
has not come by its end (Cursor's stop and reply hooks run at the same
time). `reply({..., exact: true})` completes it; else the next prompt, the
session end or the idle sweep after 3 minutes saves it, with the
transcript's reply when there is one.

## Rules every adapter keeps

**Messages that are captured.** Only the user's own messages and the final
assistant reply go into a batch. These never do:
- tool calls and tool output;
- reasoning;
- harness or plugin context, including QuoteMem's own recall block and note;
- slash commands;
- acknowledgements such as "ok" and "thanks".

Short answers such as "no" or "对" are kept, together with the question they
answer when that question was not sent yet. Subagent sessions are skipped unless `QUOTEMEM_INCLUDE_SUBAGENTS` is
on; the adapter decides what counts as a subagent.

**Batches and identity.**
- `buildJobs` gives each batch a deterministic `request_id`: the session, the
  turn range and a content hash. A resend is then never stored twice.
- Session ids sent to the service are `<host>:<host session id>`.
- A batch cut early has fewer than `captureEvery` turns, as before a forget,
  at a session end, when another session starts, or after idle time. It
  holds only turns not sent before. A turn whose forget runs again after
  its batch (`ownTurn`) is a batch of its own, so the turns after it are
  never in it.
  The service reads the session's previous batch as context by itself, so
  no earlier turn is sent again; a turn sent twice would be stored twice.

**Secrets.** `redactText` runs before anything is written to disk or sent:
- private keys, cloud and API keys, tokens and JWTs;
- `password: …` style values and credentials in URLs;
- Stripe, Hugging Face and npm tokens;
- card numbers (Luhn checked), US social security numbers and Chinese
  resident ID numbers.

It is pattern based and best effort. A message that is mostly a secret is
skipped.

**The recall gate** (`isMemoryRelated`) is a local keyword test in English and
Chinese.
- **`questions` (the default).** A plain first-person statement
  ("I prefer Rust", "我喜欢咖啡") searches only when the message also asks
  something. So do "remember that …", "I forgot my …" and a decision stated
  with its content ("we decided to use Postgres").
- Coding questions about "my bug" or "my config" do not search. Neither do
  "continue" or "resume" without a person ("continue from where the build
  failed"), or a Chinese time word without one (上个月的销售数据).
- **`broad`.** It also searches on such statements.

`QUOTEMEM_RECALL=always` searches on every message, with a stricter score
threshold.

**Injection.**
- `formatRecall` marks the block as retrieved long-term memory and as
  untrusted data, and keeps each hit's quoted source line.
- It adds whole hits until `QUOTEMEM_MAX_INJECT_CHARS` is reached.
- The agent service scores each hit as a relevance from 0 to 1 and returns
  nothing below its own relevance floor, so a question it cannot answer gets
  no hit and no block. Gated recall therefore adds no threshold of its own;
  `QUOTEMEM_RECALL_MIN_SCORE` sets one (`always` mode uses 0.5).
- A memory hit of the agent service carries a status read from the whole
  store:
  - `current`: it still holds. A past event that stays true, such as a move
    last week, is current.
  - `past`: it no longer holds. It was replaced, corrected or retracted, or
    it is a state that has ended.
  - `possibly_outdated`: something the user said later conflicts with it,
    and nothing settled which holds.

  A past or possibly outdated hit is marked `status="…"` on its tag. The
  block then says once what the marks mean: tell a past memory as how things
  were and go by its replacement; say that a possibly outdated one may have
  changed. An Evidence hit (one quoted conversation line, `role, date:
  text`) is marked `kind="said"`: it is what was said on that date, not a
  checked memory, and the notice says to go by a memory that says otherwise. The server's order is kept; it puts a replacement directly above
  what it replaces.
- Inject the block into the model context only, never into the user's
  visible transcript where the host offers the choice.

## Silent forget

**Detection.** `detectForget(text, {previous})` follows the service's own
in-chat rule, so the plugin and the service read the same sentence the same
way.
- Detection is conservative: a sentence read as a request by mistake deletes
  the user's data without a word, while a missed one is simply said again
  more plainly. Complaints, questions, recall requests and instructions about
  code, files or apps are never requests.
- English requests are found: "please forget that I …", "don't remember
  that I …", "stop remembering …", "delete … from your memory". A bare
  "don't remember …" whose object is a plain noun phrase counts only when
  something marks it as a request: "please", "do not" written out, emphasis
  ("Do NOT"; not when the whole sentence is in capitals), or a closing please
  or thanks. A bare "never remember …" counts only with please. Neither counts
  when it asks to be reminded ("help", "how", "can you", "tbh"): without "I"
  it is as often the user's own "(I) don't remember the dog's name, can you
  remind me". No "remember" form, polite or not, reads a correction or a
  complaint ("please don't remember my name wrong", "… my birthday as the
  5th", "… everything I say", "… that you got my name wrong"); a that-clause
  in the first person states the fact itself ("please don't remember that I
  work as a nurse") and a condition asks ("… my address until I'm settled").
- Chinese requests are found: 忘掉/忘记…, 别记住/不要记…, 删除…记忆/记录,
  把…忘掉. A sentence-final 了 is ordinary Chinese (忘掉我离婚了 is a request),
  but "(I/you) forgot …" is not: 忘记 + "…的<a noun>了" or "我家…了" without 请
  or 帮我 (忘记我的密码了吗), 忘记了/忘掉了 (忘记了我的生日), and 你 + 忘 unless
  after 请, 帮我 or 能不能 (你忘了我的生日吧). 记错 and 记混 are not 记
  (别记错我的生日). 删掉, 删除 and 清除 forget only what is named as a memory
  (把你记住的我的地址删掉), never work (把我的分支删掉, 删除我的提交记录).
- The plugin also reads everyday forms beyond the service's rule: another
  person's or a pet's name or detail without "my" ("forget Ben's name",
  "forget the dog's name", 忘掉小李的名字, 忘掉阿杰, 忘掉老王), a fact about
  "we" or "our" ("forget our wedding anniversary date"), a request after
  "from now on" or 以后 (以后不要记得我恐高了), "I don't want you to remember my
  …" (one clause, not "… my name wrong" or "… me as rude"), "could you please
  not remember …" (with please), "don't ever remember …", 你能不能别记住…,
  你可以…吗, 忘了…吧. After "the", the owner must be a person or an animal, or a
  capitalised name; software and machines are never owners ("forget the
  function's name", "forget Docker's address", 忘掉这个函数的名字,
  忘掉Redis的地址, 忘记公司的地址, 忘掉老歌, 忘掉阿里). The service's in-chat check
  does not read these, so their forget waits for the batches before it.
- A clear replacement said after the request, in the last clause of its
  sentence, is not forgotten: the statement ends before it, and the turn is
  stored unchanged. A missed cut is safe, a wrong one is not, so the clause
  must restate the same thing (an address, a name, a number, an email) with a
  new value and say nothing more. The value is words that start with a digit,
  hold an @ or are in title case, with no "not", then at most "now"; a
  sentence in capitals is never cut: "my new <the same
  thing> is …", "my <the same thing> is now …", "it's … now" after "my old
  <thing>" ("forget my number, my new number is 555", "forget my old address,
  it's 27 Birch Lane now"), a new place to live after an address or a past
  place ("forget my old address, I live at 27 Birch Lane now", "forget that I
  lived in Leeds, I live in York now"), or what a name is now ("forget my
  cat's name, it's called Max now"). In Chinese the clause is the last one
  and has no word that goes on (以前, 还, 因为, 和…): 忘掉我原来的手机号，新号码是123,
  忘掉我的旧地址，我现在住在银杏路6号, 忘掉我的猫的名字，它现在叫咪咪. Anything else
  goes with what is forgotten: a state ("my number is now unlisted", "I'm a
  patient now"), a clause that only describes the thing or the person
  ("forget that I hold a prior conviction, I'm currently on parole",
  忘掉我得了抑郁症，我目前在吃药), more after the value ("…, the old one was
  444-1234", 忘掉我的旧名字，我现在叫小明，以前叫小红), and one that points back ("I
  moved there in 2019").
- A shared list of these sentences, with what detection returns for each, is
  in `test/forget-intent-vectors.json`; the tests check it.
- Starting the work over is no forget: what we said, without "I", "me" or
  "my" after it ("forget the stuff we discussed", "forget the things from our
  last session", 忘掉我们聊过的东西), and a forget about no fact of the user's
  followed by a clause or a sentence that starts over ("…, let's start
  fresh", "Forget the stuff about the movie. Let's talk about work."). After
  a fact of the user's (I, me, my, a name, "our …"; 我) such a clause only
  closes the request and is not forgotten ("Please forget that I'm pregnant,
  let's move on.", 忘掉我怀孕了，说点别的吧), when it is the last clause and goes
  on with nothing; otherwise the whole object is forgotten ("Forget that I
  have HIV, let's move on, my partner doesn't know."). "Never mind" and 算了
  withdraw the request itself.
- These are never requests: a dismissal ("forget it, let's move on"), a
  reminder ("don't forget that I …"), a question about forgetting, "forget
  that I asked", letting a quarrel go ("forget our fight last night"), and a
  new name to use ("forget Ben's name, call him Benny", 改叫…).
- Instructions, plans and code are not facts: "forget my previous
  instructions", "forget the approach I suggested", "forget my
  implementation / branch / PR / schema", 忘掉我的代码 are not requests. Nor
  is what was said, as a whole ("forget my questions", "forget that stuff",
  "forget our chats"); with what it was about it is ("forget the things I
  told you about my ex", "forget my messages to my ex").
- A request that refers back ("forget what I just said", "forget the last
  thing I said", "别记住我刚才说的") forgets the earlier sentences of the
  message, else the previous user message. Fillers such as "Ignore that;"
  and other requests are not what it refers to.

Sentences end at `.`, `!`, `?` or `;` followed by a space or the end, and at
`。！？；`. A period after a title ("Dr. Sato") or an initial ("J. Smith")
never ends one, so "forget that my dentist is Dr. Sato" keeps its last words.
Periods inside a word ("v2.3", "a.b@example.com") never split.

**The flow, all in the background:**
1. The pending turns are queued first: this session's, and those of every
   other session in the same state folder.
2. A forget job is queued after them.
3. The flusher sends jobs one at a time in queue order. A batch the service
   acknowledged early (see "Async Add" below) is read again until it is
   stored, and a forget waits for every batch before it. So every batch queued
   before the forget is stored (or dropped after its retries) before the
   forget starts. The fact the user said just before "forget …" is therefore
   in memory when the forget looks for it.
4. The forget job calls `/memory/forget/prepare` with the user's
   phrase (redacted) as its `statement`, the field that names a forget by
   the user's words (a forget by id names `memory_id` instead). It then sends
   the returned `confirmation` to `/memory/forget/confirm` unchanged; the
   user's own request is the consent.
   - Only a `verified` outcome is counted as forgotten. An outcome with a
     memory id forgets that memory. One without a memory id, and with
     `withheld_spans` n, forgets n sentences: the fact was in the saved
     conversation, but no memory stated it. `forgotten_earlier` was gone
     before, and `changed` was kept.
   - A plan confirmed again (its first confirmation failed or its answer was
     lost) counts a `forgotten_earlier` target as forgotten by this forget:
     every target was current when the plan was prepared, so the first
     confirmation (or the service's own forget of the same request) removed
     it. A bundled one is named, as below. Of the other memories the prepare
     said would go (`affected`), it counts only the candidates the plan kept:
     the list also names targets and memories the forget's cascade removes,
     which are counted already.
   - A target marked `bundled` also held other facts, so it went whole. The
     note names it, so the model can tell the user what else went.
   - A candidate the service kept (`excluded`: about another person or
     animal, or stating the opposite) is never forgotten, offered, or used
     in place of the user's words.
   - The service may ask for a new plan: `operation_conflict` (also an
     expired plan, a scope that changed, or a spent erasure budget). The
     plan is then prepared once more. One more conflict is tried again later;
     after three plans in conflict the forget is given up and reported as not
     confirmed. A target that changed meanwhile is prepared once more too.
   - A busy or unreachable service is retried with backoff, like a batch:
     `erasure_in_progress`, `index_rebuilding`, `provider_unavailable` and
     `service_unhealthy` (an embedding-space mismatch also answers that).
     `429 delivery_busy` is a wait, as for a batch (see "Writers and the
     flusher" below): it uses no attempts, for at most
     `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS`. A plan whose confirmation met one of
     these is confirmed again unchanged, so the service resumes it.
   - When nothing matches the phrase (it was cut, or the memory is stored in
     another language), the phrase is searched. A stored statement that
     says the same thing is then forgotten by its own text. "The same thing"
     is strict: most of the phrase's content words are in the statement and
     most of the statement's are in the phrase; both name the same people
     and animals; both are negated or neither is. Failing that, a memory
     whose quoted user message holds the phrase's content words (75% of them,
     at least two; same people and animals; same negation; not a question)
     is forgotten by its statement, but only when exactly one memory among
     the hits matches this way. That finds a memory stored in English for a
     request in Chinese. Otherwise the note lists the closest statements and
     claims nothing.
     A forget of such a statement that must be retried keeps that
     statement's own request: it resumes, or prepares again, for the
     statement, never for the user's original words.
   - Once the hook has given the note, the outcome on disk keeps only its
     status and counts, no memory text.
   - After an attempt whose confirmation may have reached the service, no
     statement is forgotten in place of the phrase: "nothing matched" then
     means it is already gone.
5. The hook waits for the outcome until `QUOTEMEM_FORGET_TIMEOUT_MS`
   (default 1800, at most 2500) after the message arrived, so a forget turn
   stays under a host's 3 s hook budget; then it returns one note. A forget
   not done by then finishes in the background, and the next message of the
   same user says how it ended. In the queue, how long a forget waits for the
   batches before it to be stored depends on whether the service's own
   in-chat forget reads the request too:
   - It does (the service's own forms): the forget waits at most 2 s
     (`FORGET_EARLIER_WAIT_MS`), then runs. A batch the service stores later
     is checked for the request by the service when the request's own turn
     is stored (see below).
   - It does not (a retraction such as "forget that I asked you for …", a
     reference back to an earlier message, "forget that I said that", and the
     Chinese 把…忘掉 and bare 不要记… forms): nothing would check a batch
     stored after the forget, so the forget waits until every batch before it
     is stored. The turn's note then says it is still running.
   - Either way, a batch before it that the service has not saved yet holds
     it. When the service holds that batch back as busy (`429
     delivery_busy`, see below), the forget waits behind it until it is
     saved, or dropped after `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS` (default
     2.5 hours). The
     order keeps the forget from missing what that batch holds.

   The forget reads the service's `earlier` record of what either path did
   for the same request. What the Add's in-chat forget, or a forget of the
   same words before this one, already removed counts as already gone
   (memories and sentences apart); what this forget's own confirmations
   removed, a lost answer's or the cascade's included, counts as forgotten
   now.

**The note** names no tool and no id. It is one of these:

| Outcome | The note says |
|---|---|
| Forgotten | Done, with the number of memories and of sentences forgotten (and the memory's text after a search match). A memory forgotten whole because it held other facts too is quoted, so the model can say what else went. Other memories that relied only on what was forgotten, and went with it, are counted with the memories and said apart ("2 stored memories were forgotten (1 of them because it relied only on what the user asked to forget)"); one that stays stored but can no longer be read until the user says it again is counted too. A memory the service reworded to leave out what was forgotten stays stored: it is counted apart, never as forgotten ("1 related memory was reworded to leave out what the user asked to forget"), and the model may tell the user that it was kept but reworded. These are never quoted. A memory the service quoted anew, which stays readable, is not counted |
| Already gone | It is not in memory: what matched was forgotten before; never "done" with a count |
| Nothing matched | Nothing was forgotten now; when the service's in-chat forget reads the request too, that it may still find it when the turn is saved; never "done" |
| Nothing matched, close memories found | Nothing was forgotten yet; the closest memories, which the model may forget by their exact text if one is clearly meant |
| Incomplete | Partly done: the memories and sentences forgotten, worded as for Forgotten (a memory forgotten whole is quoted too), and how many others changed meanwhile and were kept; with none forgotten, that nothing was |
| Still running | It is being forgotten in the background, never "already done"; how it ended is reported on the next message of any session of the same user (a plain success: of the session that asked) |
| Not confirmed | It could not be completed or confirmed; neither "forgotten" nor "nothing deleted" is said; the model asks the user to send the request again if they still want it forgotten (a new request is detected again) |

Every note also names the look-alike memories the service kept, and says not
to forget them. No note mentions tools or tool calls, so the model has nothing
internal to repeat.

**A forget reported later.** The note for a forget that finished in the
background comes with the next message of the same user, perhaps in another
conversation. A plain success (something was forgotten, and no related memory
went with it, was forgotten whole or was reworded) is the exception: only the
conversation that asked for it is told, at its next message. It says only
that an earlier request to forget something has finished, with the same
outcome and counts as above. It never quotes the request, the forgotten
memory, a memory forgotten whole, the closest or the kept memories:
otherwise the plugin itself would give the forgotten fact back. If the
request is in the model's own conversation, the note points to it;
otherwise the model is told not to say or guess what it was about, and not
to look it up.
With nothing matched, the user is told that they can ask again in other
words. The list of running forgets keeps no word of the request either.

**In-chat forgets of the service.** An agent-mode service carries out a
forget request it finds in an /memory/add, and its `200` answer reports it in
`agent.forgets` (`status` forgotten, nothing_matched or pending; counts of
memories, sentences and bundled memories, and of other memories it left
unreadable or reworded (`affected`); `earlier` when the plugin's own forget of the same
request already did it). The flusher keeps them for the batch's user, counts
only: the request's words are never written to disk. The next message of
that user gets one note saying what was done for "a request to forget in an
earlier message"; another user of the same state folder never sees it.
One forget is counted once. An `earlier` entry counts both paths' memories
and sentences, so it is reported only for what it holds beyond what the
plugin's own forget of that request reported. The two are matched by a hash
of the request's words, kept with that forget's result, and only the latest such
forget that started before the entry counts: the same words asked again later
are another request. A forget that failed counts nothing, so the entry is
reported in full. The other memories either path affected are matched by a
hash of their ids, so none is counted twice. An `earlier` entry
that removed nothing more, or whose request no plugin forget ran (the forget
tool told the user), is not reported.

**Requests.** Besides the service's own forms, a retraction of something
the user asked for is a request: "please forget that I asked you for a shell
command to count the lines in data.csv" forgets "I asked you for a shell
command …". A bare "forget that I asked" (never mind the question) is not,
and neither is a change of instructions: "forget that I asked you to write
it in Python, do it in Rust instead", "forget I asked you to rename the
variables". Taking back an instruction ("asked you to …") is a request only
when it is polite or names memory and is about the user ("please forget that
I asked you to look up my test results"). "Forget that I said that" refers
back like "forget what I just said". Any request that names what to use in
its place changes the instructions too: "forget my branch name, use main
instead", "忘掉我的分支名，改用main" or "…，用main代替". A request after a
list marker is read: "③", "1)", "1）", "1．", "Q1)", "a)", "iv)", "（一）",
"一、", "第一，" or a bullet. A word of roman letters that is not a numeral
from i to xxxix in one case is not a marker ("Mix) forget my number").

**Request identity.**
- Each forget's `request_id` is derived from the session, the turn, a random
  nonce saved with the turn and the statement. A repeated hook for the same
  turn records nothing, and a retry of the queued job replays the same
  operation. A later message that asks the same again is a new operation.
- The service also carries out forget requests it finds in a /memory/add (its
  post-Add in-chat forget). The request's own turn is captured after the
  plugin's forget, so the service then finds the memory already gone
  (`forgotten_earlier`); it may still mark the evidence sentences that state
  it. Nothing is forgotten twice.
- That later batch also lets the service record that the user asked to
  forget.

Set `QUOTEMEM_FORGET=off` to leave forgetting to the explicit MCP tools.

**Other agents.** Each agent keeps its own state folder by default, while all
agents share one memory owner. A forget saves first only the pending turns of
its own state folder. A fact said in another agent and not yet sent can
still be stored after the forget. Give the agents the same
`QUOTEMEM_STATE_DIR` to share one queue, or ask again once that agent has
saved its turns (its session ended, a new session of it started, or it was
idle for `QUOTEMEM_IDLE_FLUSH_MS`). The same holds for recall: a new
session sends and waits for its own agent's other sessions only.

## Files in the state folder

The folder is `QUOTEMEM_STATE_DIR`, or the adapter's default. The folder is
mode 0700 and each file 0600.

| Path | Content |
|---|---|
| `queue/<stamp>-<pid>-<seq>-<owner>-<key>.json` | One job (a /memory/add batch or a forget), written atomically; FIFO by name. `owner` is one user (`QUOTEMEM_USER_ID`) at one service (`QUOTEMEM_URL`) |
| `owners/<owner>/state.json` | When this owner's writes are paused after a refused token; when a batch was last stored |
| `owners/<owner>/flusher.lock` | This owner's one running flusher (host, boot time, pid), refreshed every 10 s |
| `owners/<owner>/forgets.json` | This owner's forgets still running when their note was given |
| `owners/<owner>/results/<key>.json` | A forget's outcome (counts only once its note was given), kept for a day |
| `owners/<owner>/chat-forgets.json` | What the service's in-chat forget did, counts and request-word hashes only, until the next message reports it |
| `sessions/<hash>.json` (+ `.lock`) | A session's open turn, pending turns and recall memory; the hash covers the owner (the user and the service URL), so a session is never sent as another user's, nor to another service |
| `recall-breaker.json` | When recall may be tried again |
| `quotemem.log` | Warnings: counts, status, codes and request ids; never text or the token |

Two kinds of file are shared by every agent, so by default they live in
`~/.quotemem`, not in one adapter's folder: the forgets kept to apply
again (`forgotten/`) and the native-adapter markers (`native/<host>.json`).
When the environment sets `QUOTEMEM_STATE_DIR`, the markers go there
instead, and a kept forget goes to both places (an MCP server and hooks
that do not share the setting still share the forgets). Set
`QUOTEMEM_STATE_DIR` only in `~/.quotemem/env`, so the hooks and the MCP
server use one folder; `quotemem doctor` warns about an MCP server it
finds with another. A run given a state folder of its own writes nothing
else under the home, except the local service's own files
(`~/.quotemem/service.*`, when `quotemem setup` made one and
`QUOTEMEM_URL` points at it), which every agent shares.

Files of the earlier layout (before owners: job files whose name names no
owner, `queue/state.json`, `flusher.lock`, `forgets.json`, `results/`, and
session files that name no user) are never sent or read, since that layout
could label one user's turns with another user's id. They are removed once
older than a current file of their kind is kept: a job, the queue state, the
lock, the forget list and outcomes after the capture age limit (a day at
least), a session after a week.

**Writers and the flusher.**
- Several users may share one state folder (one host profile, several
  `QUOTEMEM_USER_ID` values). Each owner has its own flusher, which sends only
  that owner's jobs with that owner's settings; a hook reads only its own
  owner's outcomes and notes.
- Hooks only create new job files; only the flusher changes or removes them.
- A stale lock is broken by renaming it aside. A lock is stale when its owner
  is gone (another boot, a dead pid), or when a file from another host has
  not been written for hours.
- A batch is dropped after `QUOTEMEM_CAPTURE_MAX_ATTEMPTS` attempts or
  `QUOTEMEM_CAPTURE_MAX_AGE_MS`.
- A batch the service has not saved, or a forget, may get
  `429 delivery_busy`: the service holds it back for now. The service gives
  this answer for several causes: a session's new batch held while an
  earlier batch of that session completes after a forget (up to about two
  hours), and its internal `finalize_pending`, `reception_stale`,
  `authorization_expired`, `budget_exceeded`, `extraction_busy` and
  `lease_lost`, and a receiver that is not available. Every one of them is a
  wait, not a failure. The job is sent again after the service's Retry-After
  (else with backoff), does not use its attempts and keeps its place, so the
  jobs behind it (forgets too) wait for it. It is dropped (`delivery_busy
  for <n> s` in the log; a dropped forget is reported as not confirmed) only
  when such answers last longer than `QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS`
  (default 9000000, 2.5 hours, from 60000 to 30 days), or at
  `QUOTEMEM_CAPTURE_MAX_AGE_MS`. Any other answer ends the wait, and the
  wait's clock starts again at the next `delivery_busy`. Only a failure
  among those other answers uses an attempt (a success uses none), so the
  attempt and age limits still bound the total.
- An attempt that could not reach the service (refused, reset, a network
  error) waits its retry delay, which grows to minutes. When a hook reaches
  the service again first (a recall search or a session digest that it
  answers), the hook notes that in `owners/<owner>/contact.json` and starts
  the flusher, and those jobs are sent at once. While such a job holds the
  queue, the flusher also asks `GET /health` (no token, 3 s), at most every
  30 s (`owners/<owner>/probe.json`), and a 2xx counts the same; a hook
  starts a flusher for that probe. So after an outage, a short session's
  turns do not stay on disk until the next one. A probe uses no attempt,
  and the attempts count as before. A job the service held back (busy, its
  spend cap) keeps the time the service gave and is never probed for.
- An attempt that may have reached the service and then met
  `operation_conflict` is sent once more under a new id.

**Async Add.**
- Every `/memory/add` carries `Prefer: respond-async`. A service that supports it
  answers `202 {"status":"pending"}` once the batch is durably saved and
  completes it by itself; an older one ignores the header and answers `200`
  when the batch is stored, as before.
- An acknowledged batch stays in the queue, marked `accepted`. The flusher
  sends the identical request again (same request id, same body) to read its
  state: after the service's Retry-After when the `202` carries one, else
  after 1 s, then 2 s, then every 3 s. `200` means stored, `202`
  still pending; the service never extracts it twice. A read that gets no
  answer within 15 s is taken as still pending and sent again later, so one
  slow read does not hold the queue for the full `/memory/add` timeout. A read that
  fails (the service is restarting, busy or unreachable) is retried with
  backoff and never counts toward `QUOTEMEM_CAPTURE_MAX_ATTEMPTS`: the
  service holds the batch, so it is not dropped for that.
- Meanwhile the batches behind it are sent, so the service can work on
  several. A batch not yet acknowledged still holds everything behind it,
  and a forget waits for the batches before it to be stored: at most 2 s when
  the service's in-chat forget reads the request too, else until they are.
- An acknowledged batch is never resent under a new request id. The service
  completes one user's batches one at a time, so a batch may wait while the
  ones before it complete. One the service has not confirmed while no batch
  was stored for 30 minutes (or by `QUOTEMEM_CAPTURE_MAX_AGE_MS`) is dropped
  from the queue with a warning; the service still has it, and the forgets
  behind it no longer wait. The 15 s read limit and this drop are safety
  nets: a service that answers a resend at once (202 while it works, 200
  once stored) never reaches them.
- A resend may answer a failure of the background completion once (for
  example `429 delivery_busy`). It is read again after the Retry-After; the
  next resend starts the completion again.
- While the service's daily spend cap is reached, it answers a new batch,
  and the resend of a saved batch not yet stored, `429 spend_cap_reached`
  with a Retry-After at its reset. The batch waits for the reset (at least a
  minute, at most a day) without using its attempts. A batch not yet saved
  keeps its age limit. A saved one stays held (the forgets behind it that the
  service also reads wait for it only 2 s; the others wait for it), and it is
  resent after the reset (the resend starts its
  completion). Its time under the cap counts neither toward its age nor
  toward the 30-minute drop, which runs from the reset.

**Flusher lifetime.**
- It is started by `spawnDetached` (`proc.mjs`), as the service and the
  Windows background check are. On Unix that is `spawn(process.execPath,
  [flusher.mjs], {detached: true, stdio: 'ignore'})`: it gets its own
  session, so a host that kills the hook's process group does not kill it.
  Codex does that.
- **It holds none of the hook's handles.** On Windows a process inherits
  every inheritable handle of its parent, whatever its stdio; a caller that
  reads an agent tool's output through a pipe (`… > file` in PowerShell,
  `$(…)`, CI) hands that pipe down to the hook, through cmd and PowerShell.
  A child started by plain `spawn` kept the pipe open until it exited, so
  the caller waited for the flusher (minutes with the service down or slow)
  or for the service (until `quotemem stop`). So on Windows `spawnDetached`
  has PowerShell start it through WMI (`Win32_Process.Create`, outside the
  host's job), else through the shell (`Start-Process`); neither passes a
  handle. What starts is a hidden `cmd.exe` that runs it with NUL (or
  the service log) as its output, its words passed through the
  environment. A pid from WMI counts only once that cmd has written its
  `started` file (within 3 s): an endpoint rule that stops what WMI starts
  (Defender's ASR rule, an EDR) leaves none, and the shell starts it
  instead; the debug log, and the service's lock (`via`), say which. The
  WMI cmd starts the program only once it has made a `claim` folder, which
  one process alone can make. When it is slower than 3 s, PowerShell makes
  that folder first, so the WMI cmd starts nothing, and stops it before
  the shell starts the program: one copy runs, never two on one port. A
  WMI cmd that made the folder in time keeps its pid. A hook
  starts the flusher while it goes on (spawnDetachedAsync) and prints its
  answer first; before it exits it waits for the start (settleLaunches),
  within its host's timeout, so a slow PowerShell never costs the answer.
  An exit handler, which cannot wait, starts it at once. The service's
  start waits for PowerShell (0.2 to 0.5 s) for the pid it records; that
  pid is cmd's, and `taskkill /T` stops both. The script works in ConstrainedLanguage mode
  (AppLocker, WDAC). The pid comes from a file PowerShell writes, and a
  file cmd writes first says it started, so a PowerShell stopped at its
  10 s limit after the start leaves no second instance (the service is
  then recorded without a pid, as starting). When PowerShell cannot run
  (missing, blocked) and nothing started, the program is started directly,
  holding the hook's handles again: the log says so once, a
  `direct-start.json` in the state folder records it, and `quotemem
  doctor` reports both. A test runs a hook under cmd and PowerShell with a
  piped caller and checks that the pipe ends while the flusher and a
  started program still run.
- It exits when the queue is empty, when the next retry is further off than
  `QUOTEMEM_FLUSHER_MAX_WAIT_MS` (default 120000), or after an hour. A later
  hook starts a new one.
- **Windows: job objects.** Started through WMI, the detached tree is in
  no job of the host: a host that stops its hooks' job or tree when it
  exits (Cursor's CLI on Ctrl+C) does not stop it. When WMI refuses and it
  is started through the shell, it escapes Node's own job only, and such a
  host kills the flusher too; the batches stay queued and the next hook
  sends them.

## Settings

`resolveSettings(config, env, {defaultStateDir, configName})` reads these
variables. The host config can override the behaviour settings only. The
URL, user id, auth mode and token come from the environment only (the token
from a file it names), so the automatic layer and the MCP tools act on the
same memory.

**The token** (`token.mjs`, the one token reader: the adapters, the doctor
and the MCP server all use it): `QUOTEMEM_TOKEN`, then
`QUOTEMEM_HTTP_AUTH_SECRET`, then the file `QUOTEMEM_TOKEN_FILE` names, then,
for a service on this computer (a loopback `QUOTEMEM_URL`), the service's own
token. When `npx quotemem setup` wrote `~/.quotemem/service.env`, that is the
service the plugins start, so its token is the `QUOTEMEM_TOKEN` of
`service.env`, else its `<data-dir>/token` with the data folder that file
gives. Otherwise it is `<data-dir>/token` with `QUOTEMEM_DATA_DIR`, else
`$XDG_DATA_HOME/quotemem`, else `~/.local/share/quotemem`. No setup needs the token in an agent's
environment, and none should put it there: the agent's shell tool inherits
that environment, so the model could print it.

| Variable | Default | Purpose |
|---|---|---|
| `QUOTEMEM_URL` | `http://127.0.0.1:8765` | The QuoteMem service |
| `QUOTEMEM_TOKEN_FILE` | for a service on this computer, its own token (`service.env`, else its `<data-dir>/token`) | A file holding only the service's token; without a token, and without `npx quotemem setup`, automatic memory stays off |
| `QUOTEMEM_AUTOSTART` | on | `0` keeps the adapters from starting the service on this computer |
| `QUOTEMEM_HTTP_AUTH_MODE` | `bearer` | `bearer`, `token` or `x-api-key` |
| `QUOTEMEM_USER_ID` | OS user name | Whose memory |
| `QUOTEMEM_CAPTURE_EVERY` | `3` | User turns per batch; `off` disables capture |
| `QUOTEMEM_IDLE_FLUSH_MS` | `300000` | Idle time before a partial batch is queued |
| `QUOTEMEM_RECALL` | `gated` | `gated`, `always` or `off` |
| `QUOTEMEM_RECALL_GATE` | `questions` | `questions` or `broad` (see above) |
| `QUOTEMEM_TOP_K`, `QUOTEMEM_MAX_INJECT_CHARS`, `QUOTEMEM_RECALL_MIN_SCORE`, `QUOTEMEM_RECALL_TIMEOUT_MS`, `QUOTEMEM_RECALL_REPEAT_TURNS`, `QUOTEMEM_SEARCH_OPTIONS` | as in DSH | Recall limits |
| `QUOTEMEM_FORGET` | `auto` | `auto` (silent forget) or `off` |
| `QUOTEMEM_FORGET_TIMEOUT_MS` | `1800` | How long a hook waits for the forget's outcome, from the message's arrival (200 to 2500) |
| `QUOTEMEM_FLUSHER_MAX_WAIT_MS` | `120000` | Longest retry wait the flusher sleeps through |
| `QUOTEMEM_CURSOR_PROMPT_CONTEXT` | `auto` | Whether a hook in Cursor adds recall and forget notes to each prompt. `auto`: on the Cursor app 3.20.17 and later, where the prompt hook's `additional_context` was seen to reach the model (the field is not in Cursor's docs), and on Cursor's CLI 2026.11.19 and later (date versions), whose interactive mode was seen to read it too (print mode runs no prompt hook; earlier CLI dates get none); `on`: always; `off`: never (recall is then a session digest plus the MCP tools) |
| `QUOTEMEM_CAPTURE_*`, `QUOTEMEM_MAX_MESSAGE_CHARS`, `QUOTEMEM_STATE_DIR`, `QUOTEMEM_INCLUDE_SUBAGENTS`, `QUOTEMEM_DEBUG` | as in DSH | See the [DSH README](../dsh/README.md#environment) |

## API

All functions are exported from `index.mjs`.

| Module | Exports |
|---|---|
| `config.mjs` | `resolveSettings(config, env, {defaultStateDir, configName})`, `publicSettings(settings)` |
| `gate.mjs` | `isMemoryRelated(text, {statements})`, `sentences`, `isQuestion`, `isTrivial`, `isCommand`, `endsWithQuestion` |
| `redact.mjs` | `redactText(text) → {text, secretOnly, kinds}` |
| `format.mjs` | `formatRecall(hits, {maxChars, maxHits, minScore, keepUnscored, skipIds}) → {text, ids} \| null`, `hitMemory(content) → {statement, status, quotes} \| null`, `forgetNote(statement, result, {earlier})`, `unavailableNote()`, `fitContext(text, maxChars)` (cut to a size without cutting a block: the oldest notes go first), `RECALL_HEADER`, `FORGET_HEADER`, `UNAVAILABLE_HEADER` |
| `http.mjs` | `healthOk(settings, timeoutMs)`, `postJson(settings, route, body, signal, headers)`, `callJson(settings, route, body, {deadlineMs, signal, attempts})`, `authHeaders`, `errKind`, `codeOf`, `retryable`, `retryAfterMs` |
| `capture.mjs` | `newCaptureState()`, `recordTurn(settings, state, {users, assistant, turn, completed}, {label}) → due`, `cutBatch(settings, state, sessionId, {untimed}) → jobs`, `buildJobs(settings, sessionId, first, last, messages, now, {untimed})` (`untimed`: a request id from the words only, for turns read again from a transcript), `wireMessage`, `textOf`, `sha16` |
| `recall.mjs` | `recallQuery(settings, text, {statements}) → query \| null`, `searchRecall(settings, query, {signal, skipIds, breaker, log, report}) → block \| null` (`report.unavailable` when the service could not answer), `searchBody` (asks for at least `SEARCH_MIN_HITS` = 10), `recallBlock`, `RecallBreaker`, `newRecallMemo`, `recallTurn`, `recentIds`, `rememberIds` |
| `forget.mjs` | `detectForget(text, {previous}) → statements`, `isForgetRequest`, `forgetBody(settings, sessionId, turnKey, statement)`, `forgetRequestId`, `requestKey(words)` (a hash of a request's words that matches the service's in-chat outcome of it), `noteForgotten(env, {url, userId}, statement, at)`, `forgottenOf(env, {url, userId})`, `forgottenFile`, `FORGOTTEN_KEEP_MS` (forgets to apply again, kept a week), `carryOnDisk(carry)`, `runForget(settings, body, {signal, substitute, carry}) → {status: 'forgotten' \| 'already' \| 'incomplete' \| 'nothing', forgotten, sentences, earlier, changed, bundled, excluded, matched?, closest?} \| {status: 'retry', code, carry} \| {status: 'auth' \| 'failed', code}`, `statementsOf(hits)`, `sameFact(words, statement)`, `saidIn(words, quote)`, `quotedMatch(words, hits)` |
| `send.mjs` | `sendAdd(settings, job, signal) → stored \| pending \| deferred \| retry \| …`, `deferMs`, `pollDelayMs`, `ADD_HEADERS`, `rotateJob`, `backoffMs`, `validJob` |
| `diskqueue.mjs` | `DiskQueue` (`enqueue`, `entries`, `has`, `result`, `markResultRead`, `waitResult`, `startFlusher({env})`, `flusherRunning`, `noteContact`, `contactAt`, `probedAt`), `PROBE_EVERY_MS`, `ACCEPTED_READ_TIMEOUT_MS`, `ACCEPTED_MAX_WAIT_MS`, `FORGET_CONFLICT_ATTEMPTS`, `runFlusher(settings, log, {signal})`, `FLUSHER_PATH` |
| `lock.mjs` | `tryLock`, `releaseLock`, `touchLock`, `lockHeld`, `withLock(file, fn, {timeoutMs, staleMs})`, `writeFileAtomic`, `readJson` |
| `proc.mjs` | `homeDir(env)` (the one home every part uses, as Python's Path.home(): USERPROFILE, then HOMEDRIVE+HOMEPATH on Windows; HOME elsewhere; then the OS's), `spawnDetached(command, args, {env, cwd, logFile, label})` → `{pid, via, on}` (a detached process that holds none of this process's handles; on Windows started through WMI, else `Start-Process`), `spawnDetachedAsync` and `settleLaunches(ms)` (the same without holding the caller up; wait before exiting), `launcherCheck`, `userPort`, `portOwner`, `alive(pid)`, `ownerRecord`, `ownerGone`, `bootTime`, `hostName` |
| `log.mjs` | `makeLog(settings, write)`, `fileSink(file)` |
| `hooks.mjs` | `HookMemory` (`prompt`, `reply`, `begin`, `catchUp`, `end`, `endSync`, `sweep`, `flush`, `reached`, `withForgetsAgain`, `openTurn`), `alignTurns(heads, seen)`, `queueOwnerSessions(env)` |
| `native.mjs` | `markNative(env, host)`, `nativeInstalled(env, host)`, `nativeMarkers(env)` (markers in `QUOTEMEM_STATE_DIR` when set, else `~/.quotemem/native`), `foreignHost(payload, env)` (`cursor` or `devin`), `nativeAdapterFor(host, env, payload)`: keep a turn from being saved twice when one host also runs another host's hooks (Cursor: hook files or a recent marker; Devin: hook files in the project, the folders above it up to the repository root, or the user config); `cursorPromptContext(mode, version)`, `CURSOR_CONTEXT_VERSION` (the app), `CURSOR_CLI_CONTEXT_VERSION` (the CLI): whether a Cursor prompt hook may add context; `cursorCliVersion(version)`: a version of Cursor's CLI (numbered by date) |
| `transcript.mjs` | `lastAssistantReply(file, {head, after}) → {text, time} \| null`, `lastAssistantText(file, {head, after})`, `replyFromLines(lines, {head, after})`, `replyFromSteps(document, {head, after})`, `promptHead(text)`: the turn's last assistant reply from a JSONL (Claude Code) or ATIF transcript. With `head` (the `promptHead` of the turn's user message), a reply is returned only when it answers that message, so a transcript that lags gives null, not an earlier turn's reply; a reply time before `after` is dropped. `readConversation(file, {from, guard})` (the turns added since a byte offset, `CONVERSATION_MAX_BYTES` at most), `conversationTurns(file)`, `turnsFromLines(lines)`: every user turn of a whole conversation with its final reply, from Cursor's agent transcript (the `<timestamp>` and `<user_query>` wrappers stripped; `cursorStampTime(text)` reads the local time) or Claude Code's JSONL |
| `flusher.mjs` | The background flusher's entry point (not imported) |
