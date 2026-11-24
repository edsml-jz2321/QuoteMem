#!/usr/bin/env node
// quotemem-hook: automatic QuoteMem memory for hook-based agents.
//
//   quotemem-hook <host> [event] < payload.json
//   quotemem-hook mcp                 the QuoteMem MCP server (stdio)
//
// <host> is gemini, cursor, cline, goose or devin; [event] is the host's
// event name (taken from the payload when omitted). The hook reads the
// host's JSON payload on stdin and prints exactly what the host expects on
// stdout (or nothing). It
//   - saves the conversation every QUOTEMEM_CAPTURE_EVERY user turns through
//     a detached background flusher (the hook never waits for /memory/add);
//   - adds retrieved memories to the model context when the user asks about
//     the past or about themselves;
//   - forgets what the user asks to forget and tells the model in one note.
// On Cursor versions before CURSOR_CONTEXT_VERSION (core/native.mjs) the
// prompt hook cannot add context: recall is the session digest plus the MCP
// tools, and a forget gets no note. Cursor's CLI runs no prompt or stop
// hook: its turns are saved, and their forgets carried out, at each run's
// sessionEnd from the chat's transcript (HookMemory.catchUp).
// It always exits 0. When QuoteMem is down, misconfigured or slow, the hook
// prints the host's "carry on" answer and the turn goes on without memory.
// Warnings go to <state dir>/quotemem.log, never to stdout, and never carry
// the token or message text.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HookMemory, RecallBreaker, cursorCliVersion, cursorPromptContext, fileSink, fitContext, flag, forgetWaitWithin, homeDir, lastAssistantReply, makeLog, readJson, resolveSettings, searchRecall, settleLaunches, sha16, turnReply, withLock, writeFileAtomic } from './core/index.mjs';
import { hostEnv, markNative } from './env.mjs';
import { HOSTS, fallback, hostOf, normalize, render } from './hosts.mjs';

const MAX_STDIN_BYTES = 1024 * 1024;
const STDIN_WAIT_MS = 3000;
const REPLY_KEEP_MS = 86_400_000;
// Cursor's stop waits this long at most for the turn's last reply.
const REPLY_WAIT_MS = 1000;
const REPLY_POLL_MS = 25;
const TRANSCRIPT_POLL_MS = 250;
// A reply buffered this long before the stop hook started may be an earlier
// message of the turn (text before a tool call): a later one may still come.
const REPLY_FRESH_MS = 500;
const DIGEST_QUERY = 'What do I prefer, what am I working on, and what should you remember about me?';
// Cursor keeps a prompt hook's additional_context inline up to 10,000
// characters (longer text goes to a file the model must read), so what the
// hook adds stays under that; recall itself is capped lower, since notes on
// earlier forgets and the first turn's digest come with it.
const CURSOR_CONTEXT_MAX_CHARS = 9500;
const CURSOR_RECALL_MAX_CHARS = 6000;
const DIGEST_KEEP_MS = 7 * 86_400_000;
// Cursor's and Devin's prompt hooks are set to time out after 15 s
// (cursor/hooks.json, devin/hooks.v1.json). Recall waits at most this long
// there, so a forget (at most 2.5 s) plus recall stays under 15 s.
const PROMPT_RECALL_MAX_MS = { cursor: 10_000, devin: 10_000 };
// Cursor's sessionStart hook times out after 5 s.
const DIGEST_WAIT_MAX_MS = 3500;
// A chat that starts while a forget of this user is still queued waits for
// it, reading only the local queue, at most DIGEST_FORGET_WAIT_MS before its
// digest. The wait and the digest's search share what the session-start
// hook's timeout leaves after START_MARGIN_MS (start-up, the answer and
// AFTER_MARGIN_MS), never more than the search alone had before, and the
// search keeps at least DIGEST_RECALL_MIN_MS. At Cursor's 5 s that is a
// wait of at most 2.5 s.
const DIGEST_FORGET_WAIT_MS = 3000;
const DIGEST_RECALL_MIN_MS = 1000;
const START_MARGIN_MS = 1500;

function readStdin() {
  return new Promise((resolve) => {
    const chunks = [];
    let size = 0;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(Buffer.concat(chunks).toString('utf8'));
    };
    const timer = setTimeout(finish, STDIN_WAIT_MS);
    if (process.stdin.isTTY) {
      finish();
      return;
    }
    process.stdin.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_STDIN_BYTES) {
        chunks.length = 0;
        finish();
        return;
      }
      chunks.push(chunk);
    });
    process.stdin.on('end', finish);
    process.stdin.on('error', finish);
  });
}

function parse(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function expandHome(file, env) {
  const home = homeDir(env);
  return path.resolve(home, String(file).replace(/^~(?=$|[\\/])/, home));
}

// A problem that keeps memory off is logged once, not on every turn.
function logProblem(settings, log, problem) {
  try {
    const file = path.join(settings.stateDir, 'problem.txt');
    let last = '';
    try { last = fs.readFileSync(file, 'utf8'); } catch { /* none yet */ }
    if (last === problem) return;
    writeFileAtomic(file, problem);
    log.warn(`automatic memory is off: ${problem}`);
  } catch {
    // logging must never fail a turn
  }
}

// --------------------------------------------- Cursor: the turn's last reply
// Cursor reports each assistant message (afterAgentResponse) apart from the
// end of the turn (stop), so the last one is kept until the turn ends. The
// two hooks are separate processes that Cursor starts at the same time, in
// either order: the stop may run before the last message is kept. So:
// - The stop takes a message kept for its own turn (generation_id), or the
//   turn's reply in the transcript the stop names, waiting for either at
//   most REPLY_WAIT_MS. A message kept for another turn is never read for
//   it: turns may overlap (a follow-up sent while the agent works), and
//   each stop and message completes its own generation only.
// - With neither, the turn stays open, held (its `.done` file says so): the
//   reply completes it when it comes; else the next prompt (with the
//   transcript's reply, if it has one by then), the session end (the same)
//   or, after 3 minutes, the idle sweep of any later hook (the same) saves
//   it; with no reply anywhere, with the user's words. The turn is saved
//   once, whichever comes first.
// - A message that comes after its turn was saved is not kept.

// One chat's files are `<chat>.<turn>.json` (the kept message of that turn)
// and `<chat>.<turn>.done.json` (its stop ran; `held`: waiting for its
// reply), so turns that overlap keep theirs apart.
const chatKey = (session) => sha16(session);
const turnKey = (turn) => sha16(String(turn ?? ''));
const replyFile = (settings, session, turn) => path.join(settings.stateDir, 'replies', `${chatKey(session)}.${turnKey(turn)}.json`);
const doneFile = (settings, session, turn) => path.join(settings.stateDir, 'replies', `${chatKey(session)}.${turnKey(turn)}.done.json`);
const known = (turn) => turn !== null && turn !== undefined;
const sameTurn = (a, b) => known(a) && known(b) && String(a) === String(b);

function withReplies(settings, session, fn) {
  return withLock(path.join(settings.stateDir, 'replies', `${chatKey(session)}.lock`), fn, { timeoutMs: 2000, staleMs: 10_000 });
}

// The turn whose stop ran: `held` while it waits for its reply.
function markDone(settings, session, turn, held) {
  if (known(turn)) writeFileAtomic(doneFile(settings, session, turn), JSON.stringify({ turn: String(turn), held, at: Date.now() }));
}

/**
 * Keep an assistant message until its turn's stop. For a turn held open by
 * its stop, return true: the message completes that turn now. For a turn
 * already saved, drop it.
 */
async function bufferReply(settings, action) {
  if (!action.text) return false;
  return withReplies(settings, action.session, () => {
    const done = known(action.turn) ? readJson(doneFile(settings, action.session, action.turn)) : null;
    if (done) {
      if (done.held !== true) return false;
      markDone(settings, action.session, action.turn, false);
      return true;
    }
    writeFileAtomic(replyFile(settings, action.session, action.turn), JSON.stringify({ turn: action.turn ?? null, text: action.text, time: action.time ?? Date.now(), at: Date.now() }));
    return false;
  });
}

// The kept message of this turn, taken; null when there is none. Without
// turn ids, only one kept since this turn's prompt (an older one is dropped).
async function takeReply(settings, session, turn, { open }) {
  const file = replyFile(settings, session, turn);
  return withReplies(settings, session, () => {
    const kept = readJson(file);
    if (!kept) return null;
    const usable = typeof kept.text === 'string' && Date.now() - Number(kept.at) <= REPLY_KEEP_MS
      && (known(turn) || Number(kept.at) >= (open?.time ?? Infinity));
    try { fs.unlinkSync(file); } catch { /* gone */ }
    return usable ? kept : null;
  });
}

// The turn's stop found no reply: hold the turn open, unless its message
// was kept just now (then return it).
async function holdTurn(settings, session, turn) {
  return withReplies(settings, session, () => {
    const kept = readJson(replyFile(settings, session, turn));
    if (kept && typeof kept.text === 'string') {
      try { fs.unlinkSync(replyFile(settings, session, turn)); } catch { /* gone */ }
      markDone(settings, session, turn, false);
      return kept;
    }
    markDone(settings, session, turn, true);
    return null;
  });
}

// The turn is saved: a later message of it is not kept.
async function closeReplies(settings, session, turn) {
  await withReplies(settings, session, () => {
    try { fs.unlinkSync(replyFile(settings, session, turn)); } catch { /* none */ }
    markDone(settings, session, turn, false);
  });
}

// The turns of a chat held open for their reply (their stop ran first).
function heldTurns(settings, session) {
  const dir = path.join(settings.stateDir, 'replies');
  try {
    return fs.readdirSync(dir).filter((name) => name.startsWith(`${chatKey(session)}.`) && name.endsWith('.done.json'))
      .map((name) => readJson(path.join(dir, name))).filter((done) => done?.held === true && known(done.turn)).map((done) => done.turn);
  } catch {
    return [];
  }
}

// Remove a chat's kept-reply files when it ends, and any older than
// REPLY_KEEP_MS (a chat whose console was killed runs no sessionEnd). A
// kept reply is in its own chat's file: it is never read for another chat.
function dropReplies(settings, session = null) {
  try {
    const dir = path.join(settings.stateDir, 'replies');
    for (const name of fs.readdirSync(dir)) {
      const file = path.join(dir, name);
      const mine = session !== null && name.startsWith(`${chatKey(session)}.`) && !name.endsWith('.lock');
      if (mine || (!name.endsWith('.lock') && Date.now() - fs.statSync(file).mtimeMs > REPLY_KEEP_MS)) fs.rmSync(file, { force: true });
    }
  } catch {
    // a file left behind is dropped later
  }
}

// The reply to the open turn's user message in Cursor's transcript, and
// whether the turn has ended there (a turn_ended line or a later message).
function transcriptReply(file, open) {
  return file && open?.head ? turnReply(file, open.head) : null;
}

/**
 * The last reply of the turn a Cursor stop ends, whatever order its hooks
 * ran in: a message kept for this turn since shortly before the stop (the
 * last one comes with the stop), or the transcript's reply once the turn has
 * ended there; waiting at most REPLY_WAIT_MS. Then an earlier kept message
 * of the turn, or the transcript's reply as it is; else null.
 */
async function stopReply(settings, action, open) {
  const since = Date.now();
  const deadline = since + REPLY_WAIT_MS;
  let early = null;
  let said = null;
  let readAt = 0;
  for (;;) {
    const kept = await takeReply(settings, action.session, action.turn, { open });
    if (kept && Number(kept.at) >= since - REPLY_FRESH_MS) return kept;
    if (kept) early = kept;
    if (Date.now() - readAt >= TRANSCRIPT_POLL_MS) {
      said = transcriptReply(action.transcript, open);
      readAt = Date.now();
      if (said?.ended) return said;
    }
    if (Date.now() >= deadline) return early ?? said;
    await new Promise((resolve) => setTimeout(resolve, REPLY_POLL_MS));
  }
}

/**
 * A Cursor stop: save its own turn (open, or still running beside a later
 * one) with its reply, or hold it open for a reply still to come. Returns
 * whether the turn was saved.
 */
async function cursorStop(settings, action, memory) {
  const { session, turn } = action;
  const open = memory.openTurn(session, turn);
  const kept = action.completed ? await stopReply(settings, action, open) : null;
  if (!kept && action.completed && sameTurn(open?.turn, turn)) {
    const late = await holdTurn(settings, session, turn);
    if (!late) {
      // The idle sweep saves it after a few minutes if nothing else does.
      await memory.hold({ session, turn, transcript: action.transcript ?? null });
      return false;
    }
    await memory.reply({ session, turn, text: late.text, time: late.time, completed: true, transcript: action.transcript ?? null });
    return true;
  }
  await memory.reply({ session, turn, text: kept?.text ?? null, time: Number.isFinite(kept?.time) ? kept.time : action.time, completed: action.completed, transcript: action.transcript ?? null });
  await closeReplies(settings, session, turn);
  return true;
}

// Before a Cursor prompt: each earlier turn held open by its stop takes its
// reply, if it is kept or in the transcript by now; the prompt then saves
// it as it is.
async function completeHeld(settings, action, memory) {
  for (const turn of heldTurns(settings, action.session)) {
    if (sameTurn(turn, action.turn)) continue;
    const open = memory.openTurn(action.session, turn);
    if (sameTurn(open?.turn, turn)) {
      const kept = await takeReply(settings, action.session, turn, { open }) ?? transcriptReply(action.transcript, open);
      if (kept) await memory.reply({ session: action.session, turn, text: kept.text, time: kept.time, completed: true, exact: true });
    }
    await closeReplies(settings, action.session, turn);
  }
}

// ----------------------------------------------------- goose: the MOIM file
// goose's hooks cannot add context, but its built-in `tom` extension adds the
// file named by GOOSE_MOIM_MESSAGE_FILE to every model call. The hook writes
// the recall block or forget note there and clears it when the turn ends.
// It only ever writes its own file (QUOTEMEM_GOOSE_MOIM_FILE, default
// <state dir>/moim.md), never a file the user pointed goose at for
// something else.

function moimFile(settings, env) {
  return env.QUOTEMEM_GOOSE_MOIM_FILE ? expandHome(env.QUOTEMEM_GOOSE_MOIM_FILE, env) : path.join(settings.stateDir, 'moim.md');
}

function writeMoim(settings, env, log, context) {
  const file = moimFile(settings, env);
  try {
    if (context) writeFileAtomic(file, `${context}\n`);
    else if (fs.existsSync(file) && fs.statSync(file).size > 0) writeFileAtomic(file, '');
  } catch {
    log.warnOnce('moim', 'could not write the goose context file');
  }
  const configured = env.GOOSE_MOIM_MESSAGE_FILE;
  if (context && (!configured || expandHome(configured, env) !== file)) {
    log.warnOnce('moim-unset', 'goose does not read the QuoteMem context file; set GOOSE_MOIM_MESSAGE_FILE to it (see the README)');
  }
}

// ------------------------------------------- Cursor: the chat's transcript
// Cursor keeps a chat's transcript at
// ~/.cursor/projects/<workspace>/agent-transcripts/<chat id>/<chat id>.jsonl.
// Its sessionStart may name no path (the CLI's does not): the file is looked
// for there, to tell a chat begun now from one reopened.

function cursorTranscript(env, chat) {
  try {
    const projects = path.join(homeDir(env), '.cursor', 'projects');
    for (const name of fs.readdirSync(projects)) {
      const file = path.join(projects, name, 'agent-transcripts', chat, `${chat}.jsonl`);
      if (fs.existsSync(file)) return file;
    }
  } catch {
    // no projects folder: nothing was written for the chat
  }
  return null;
}

// ------------------------------------------------- Cursor: session digest
// A short digest of what is known about the user, found by sessionStart on
// every version. Cursor keeps sessionStart's context for the whole
// conversation, but does not wait for that hook: a first prompt sent before
// it finishes may go without the digest. So, where the prompt hook adds
// context (cursorPromptContext), sessionStart also keeps the digest in
// <state>/digests/<id>.txt, and the first prompt notes whether it was there:
//   - there: sessionStart finished first and gave it; nothing more is added;
//   - not yet: the first later prompt that finds it adds it once, unless
//     that turn already recalled (it then waits for the next turn).
// No prompt searches for the digest itself, so no prompt waits for it.
// QUOTEMEM_SESSION_DIGEST=off turns it off.

function digestFiles(settings, session) {
  const base = path.join(settings.stateDir, 'digests', sha16(session));
  return { mark: base, text: `${base}.txt` };
}

function saveDigest(settings, session, text) {
  try {
    writeFileAtomic(digestFiles(settings, session).text, text);
  } catch {
    // without the file, a late digest is not added to a prompt
  }
}

// The digest a prompt should add, or null. The conversation's first prompt
// only records whether sessionStart had finished (`given`) or not (`late`).
function lateDigest(settings, session) {
  const { mark, text } = digestFiles(settings, session);
  try {
    let state = null;
    try { state = fs.readFileSync(mark, 'utf8'); } catch { /* the first prompt */ }
    if (state === null) {
      fs.mkdirSync(path.dirname(mark), { recursive: true, mode: 0o700 });
      fs.writeFileSync(mark, fs.existsSync(text) ? 'given' : 'late', { flag: 'wx', mode: 0o600 });
      return null;
    }
    if (state !== 'late') return null;
    return fs.readFileSync(text, 'utf8') || null;
  } catch {
    return null;
  }
}

function digestAdded(settings, session) {
  try {
    writeFileAtomic(digestFiles(settings, session).mark, 'given');
  } catch {
    // the digest may be added once more
  }
}

// Forget the digests of ended conversations (`session`) and of old ones.
function dropDigests(settings, session = null) {
  try {
    const dir = path.join(settings.stateDir, 'digests');
    if (session !== null) {
      const { mark, text } = digestFiles(settings, session);
      fs.rmSync(mark, { force: true });
      fs.rmSync(text, { force: true });
    }
    for (const name of fs.readdirSync(dir)) {
      const file = path.join(dir, name);
      if (Date.now() - fs.statSync(file).mtimeMs > DIGEST_KEEP_MS) fs.rmSync(file, { force: true });
    }
  } catch {
    // a file left behind only skips a digest
  }
}

// The context a Cursor prompt hook adds: the turn's recall block or forget
// notes, and a late digest unless the turn already recalled. Kept under
// CURSOR_CONTEXT_MAX_CHARS without cutting a block (fitContext).
function cursorContext(result, digest) {
  const text = fitContext(result?.context ?? null, CURSOR_CONTEXT_MAX_CHARS);
  if (digest && result?.kind !== 'recall') {
    const joined = text ? `${text}\n${digest}` : digest;
    if (joined.length <= CURSOR_CONTEXT_MAX_CHARS) return { text: joined, digest: true };
  }
  return { text, digest: false };
}

async function sessionDigest(settings, log, memory) {
  const breakerFile = path.join(settings.stateDir, 'recall-breaker.json');
  const breaker = RecallBreaker.from(readJson(breakerFile));
  const before = JSON.stringify(breaker);
  const report = {};
  const block = await searchRecall(settings, DIGEST_QUERY, { breaker, log, report });
  if (JSON.stringify(breaker) !== before) {
    try { writeFileAtomic(breakerFile, JSON.stringify(breaker)); } catch { /* best effort */ }
  }
  // The service answered: what an outage left queued is sent now.
  if (report.reached) memory.reached();
  return block?.text ?? null;
}

/**
 * Run one hook. Returns the stdout text; never throws.
 * @param {{host: string, event?: string, input: object|null, env: object}} options
 */
export async function runHook({ host, event = '', input, env }) {
  const action = normalize(host, input, event);
  const label = HOSTS[host].label;
  const { env: full, notes } = hostEnv(env);
  const settings = resolveSettings({}, full, {
    defaultStateDir: path.join(homeDir(full), '.quotemem', label),
    configName: `${HOSTS[host].name} hook settings`,
  });
  const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
  for (const note of notes) log.warnOnce(note, note);
  markNative(full, label);
  if (settings.problem) {
    logProblem(settings, log, settings.problem);
    if (host === 'goose' && action.action === 'prompt') writeMoim(settings, full, log, null);
    return render(host, action, null);
  }
  if (action.action === 'ignore') {
    log.debug(`${label}: ignored (${action.reason})`);
    return render(host, action, null);
  }
  if (action.subagent && !settings.includeSubagents) return render(host, action, null);

  // Cursor's prompt hook adds context only on versions that read it there.
  // On the others it never searches (the session digest and the MCP tools
  // cover recall), and a forget still runs but tells the model nothing.
  const promptContext = host === 'cursor' && cursorPromptContext(settings.cursorPromptContext, action.version);
  let turnSettings = hostSettings(host, settings);
  if (host === 'cursor') {
    turnSettings = promptContext
      ? { ...turnSettings, maxInjectChars: Math.min(settings.maxInjectChars, CURSOR_RECALL_MAX_CHARS) }
      : { ...turnSettings, recall: 'off' };
  }
  const digestOn = host === 'cursor' && settings.recall !== 'off' && flag(full.QUOTEMEM_SESSION_DIGEST, true);
  // Cursor runs a follow-up while the earlier turn works: its turns overlap.
  let forgetWait = 0;
  if (action.action === 'prompt') forgetWait = forgetWaitMs(host, turnSettings, configuredTimeoutMs(host, event, env, input));
  const digestTime = action.action === 'start' && digestOn ? digestTimes(configuredTimeoutMs(host, event, env, input) ?? hookTimeoutMs(host, event), settings) : null;
  if (digestTime) forgetWait = digestTime.wait;
  const memory = new HookMemory({ settings: turnSettings, log, host: label, label: `QuoteMem ${HOSTS[host].name} hook`, env: full, overlap: host === 'cursor', reliableEnd: HOSTS[host].reliableEnd !== false, forgetWaitMs: forgetWait });
  switch (action.action) {
    case 'prompt': {
      if (host === 'cursor') {
        // Read before the turn runs: whether sessionStart had finished.
        const digest = promptContext && digestOn ? lateDigest(settings, action.session) : null;
        try { await completeHeld(settings, action, memory); } catch { /* the prompt saves that turn as it is */ }
        const result = await memory.prompt({ session: action.session, turn: action.turn, text: action.text, time: action.time });
        if (!promptContext) return render(host, action, null);
        if (result.kind) log.debug(`${label}: ${result.kind}`);
        // A turn held back by a running forget gets no digest either.
        const context = cursorContext(result, result.held ? null : digest);
        if (context.digest) digestAdded(settings, action.session);
        return render(host, action, context.text);
      }
      const result = await memory.prompt({ session: action.session, turn: action.turn, text: action.text, time: action.time });
      if (host === 'goose') writeMoim(settings, full, log, result.context);
      if (result.kind) log.debug(`${label}: ${result.kind}`);
      return render(host, action, result.context);
    }
    case 'buffer':
      try {
        // A reply whose turn its stop held open completes that turn now.
        if (await bufferReply(settings, action)) {
          await memory.reply({ session: action.session, turn: action.turn, text: action.text, time: action.time, completed: true, exact: true });
        }
      } catch {
        log.warnOnce('reply-buffer', 'could not keep the assistant reply');
      }
      return render(host, action, null);
    case 'reply': {
      let { text, time } = action;
      const { turn } = action;
      if (host === 'cursor') {
        try {
          if (!await cursorStop(settings, action, memory)) memory.flush();
        } catch {
          log.warnOnce('reply-take', 'could not read the kept assistant reply');
          await memory.reply({ session: action.session, turn, text: null, time, completed: action.completed });
        }
        return render(host, action, null);
      }
      if (text === null && action.transcript) {
        // Devin without last_assistant_message: the transcript, if it holds
        // the reply to this turn's own user message; otherwise the turn
        // keeps the user's words.
        const open = memory.openTurn(action.session);
        const same = open?.head && (turn === null || String(turn) === open.turn);
        const kept = same ? lastAssistantReply(action.transcript, { head: open.head, after: open.time }) : null;
        if (kept) {
          text = kept.text;
          time = kept.time ?? time;
        }
      }
      if (host === 'goose') writeMoim(settings, full, log, null);
      await memory.reply({ session: action.session, turn, text, time, completed: action.completed });
      return render(host, action, null);
    }
    case 'end':
      if (host === 'goose') writeMoim(settings, full, log, null);
      if (host === 'cursor') {
        // The turns no prompt hook saw (all of them in Cursor's CLI print mode), from
        // the chat's transcript; before end(), so an open turn that got no
        // stop hook takes its reply from there.
        if (action.transcript) log.debug(`${label}: ${await memory.catchUp({ session: action.session, file: action.transcript, completed: action.completed === true })} turn(s) queued from the transcript`);
        dropReplies(settings, action.session);
        dropDigests(settings, action.session);
      }
      await memory.end({ session: action.session });
      return render(host, action, null);
    case 'start': {
      // After the digest is printed: the other sessions' unsent turns go
      // (sessionStarted), and the background sender starts.
      afterWork.push(async () => {
        await memory.sweep();
        await memory.sessionStarted({ session: action.session });
      });
      if (host === 'cursor') {
        dropDigests(settings);
        dropReplies(settings);
        // What its transcript holds now was said before now; what is added
        // later, after (catchUp's bound).
        // Cursor's CLI runs sessionStart for a new chat only (a resumed one
        // has none), so there no transcript means none yet.
        await memory.begin({ session: action.session, file: action.transcript ?? cursorTranscript(full, action.session), fresh: cursorCliVersion(action.version) });
      }
      let digest = null;
      if (digestTime) {
        // A forget of this user still queued is waited for first. One still
        // running after the wait leaves this chat without a digest, and
        // nothing is said about it: what it removes must not come back.
        const began = Date.now();
        if (!await memory.forgetHeld()) {
          const recallTimeoutMs = Math.min(digestTime.recall, digestTime.shared - (Date.now() - began));
          digest = await sessionDigest({ ...settings, recallTimeoutMs }, log, memory);
        }
      }
      // Kept for a first prompt that went before this hook finished.
      if (digest && promptContext) saveDigest(settings, action.session, digest);
      return render(host, action, digest);
    }
    default:
      return render(host, action, null);
  }
}

// The settings of one host: recall bounded by the host's prompt-hook timeout.
function hostSettings(host, settings) {
  const most = PROMPT_RECALL_MAX_MS[host];
  return most && settings.recallTimeoutMs > most ? { ...settings, recallTimeoutMs: most } : settings;
}

// Each host's timeout for a hook, in ms (what `quotemem install` writes for
// Cursor and Devin; 5 s for the others): the work after the answer ends
// AFTER_MARGIN_MS before it. For Cursor and Devin the timeout the hooks
// file gives this hook is read (configuredTimeoutMs), so one the user
// edited holds.
const HOOK_TIMEOUT_S = {
  cursor: { sessionStart: 5, beforeSubmitPrompt: 15, afterAgentResponse: 5, stop: 5, sessionEnd: 5 },
  devin: { UserPromptSubmit: 15, Stop: 5, SessionEnd: 5 },
};
const AFTER_MARGIN_MS = 1000;
const hookTimeoutMs = (host, event) => (HOOK_TIMEOUT_S[host]?.[event] ?? 5) * 1000;

// Each host's timeout for its prompt hook, in ms, in the hooks files QuoteMem
// ships or installs (for Cursor and Devin the file's own, when it names one).
// Cline's is not known: there a prompt waits for a running forget only as
// long as a forget may take, which the hook's budget already holds.
const PROMPT_HOOK_TIMEOUT_MS = { cursor: 15_000, devin: 15_000, gemini: 15_000, goose: 15_000 };

// The session digest's times within a session-start hook timeout of
// `limitMs`: its search's (`recall`), the longest wait for a queued forget
// (`wait`), and what the two share (`shared`).
function digestTimes(limitMs, settings) {
  const recall = Math.min(settings.recallTimeoutMs, DIGEST_WAIT_MAX_MS);
  const shared = Math.max(recall, limitMs - START_MARGIN_MS);
  return { recall, shared, wait: Math.max(0, Math.min(DIGEST_FORGET_WAIT_MS, shared - DIGEST_RECALL_MIN_MS)) };
}

// How long a prompt may wait for a queued forget, within the prompt hook's timeout.
function forgetWaitMs(host, settings, configured = null) {
  const limit = configured ?? PROMPT_HOOK_TIMEOUT_MS[host];
  return limit ? forgetWaitWithin(limit - AFTER_MARGIN_MS, settings) : settings.forgetTimeoutMs;
}

// The hooks files that may run this hook, in the order they are read:
// Cursor's user file (where `quotemem install cursor` writes) and each
// workspace's; Devin's project file (where `quotemem install devin` writes)
// and its user config.
function hookFiles(host, env, input) {
  const home = homeDir(env);
  if (host === 'cursor') {
    const roots = Array.isArray(input?.workspace_roots) ? input.workspace_roots.filter((root) => typeof root === 'string' && root) : [];
    return [path.join(home, '.cursor', 'hooks.json'), ...roots.map((root) => path.join(root, '.cursor', 'hooks.json'))].map((file) => ({ file, grouped: false }));
  }
  if (host === 'devin') {
    const projects = [...new Set([input?.cwd, env.DEVIN_PROJECT_DIR, process.cwd()].filter((dir) => typeof dir === 'string' && dir))];
    return [...projects.map((dir) => ({ file: path.join(dir, '.devin', 'hooks.v1.json'), grouped: true })), { file: path.join(home, '.config', 'devin', 'config.json'), grouped: true, under: 'hooks' }];
  }
  return [];
}

// The timeout, in ms, that the first hooks file naming this hook (a command
// that runs QuoteMem's hook for this host and event) gives it; null when no
// file does, or its timeout is not a number of seconds.
function configuredTimeoutMs(host, event, env, input) {
  const runs = new RegExp(`(^|\\s)${host}\\s+${String(event).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s|$)`);
  for (const { file, grouped, under } of hookFiles(host, env, input)) {
    let data = readJson(file);
    if (under) data = data?.[under];
    const entries = data && typeof data === 'object' ? (grouped ? data[event] : data.hooks?.[event]) : null;
    if (!Array.isArray(entries)) continue;
    const hooks = grouped ? entries.flatMap((group) => (Array.isArray(group?.hooks) ? group.hooks : [])) : entries;
    const ours = hooks.find((hook) => typeof hook?.command === 'string' && /quotemem/i.test(hook.command) && runs.test(hook.command));
    if (!ours) continue;
    const seconds = Number(ours.timeout);
    return Number.isFinite(seconds) && seconds > 0 ? Math.min(seconds, 3600) * 1000 : null;
  }
  return null;
}

// Work a hook does after its answer is printed (sessionStart's other chats).
const afterWork = [];

async function afterAnswer(untilMs) {
  const left = () => untilMs - Date.now();
  for (const work of afterWork.splice(0)) {
    if (left() <= 0) break;
    let timer;
    await Promise.race([work().catch(() => {}), new Promise((resolve) => { timer = setTimeout(resolve, left()); })]);
    clearTimeout(timer);
  }
  await settleLaunches(left());
}

// The longest a hook may run: under every host's own timeout, so the host
// never kills it mid-answer. Past it, the hook answers "carry on".
function budgetMs(env, host) {
  const { env: full } = hostEnv(env);
  const settings = hostSettings(host, resolveSettings({}, full, {}));
  // A prompt either runs a forget or may wait for one, then recalls.
  return Math.max(settings.forgetTimeoutMs, forgetWaitMs(host, settings)) + settings.recallTimeoutMs + 1500;
}

async function main() {
  if (process.argv[2] === 'mcp') {
    // `quotemem-hook mcp`: the QuoteMem MCP server, with the same settings sources.
    await import('./mcp.mjs');
    return;
  }
  const usage = `usage: quotemem-hook <${Object.keys(HOSTS).join('|')}> [event] < payload.json\n       quotemem-hook mcp\n`;
  if (process.argv[2] === '--help' || process.argv[2] === '-h') {
    process.stdout.write(usage);
    process.exit(0);
  }
  const host = hostOf(process.argv[2]);
  const event = process.argv[3] ?? '';
  if (!host) {
    process.stderr.write(usage);
    process.exit(0);
  }
  let answered = false;
  const started = Date.now();
  // The answer first; then the work that need not hold it up (the other
  // chats' turns, the background sender's start, which on Windows waits for
  // PowerShell), within the host's timeout for this hook.
  let limitMs = hookTimeoutMs(host, event);
  const answer = (text) => {
    if (answered) return;
    answered = true;
    const done = () => afterAnswer(started + limitMs - AFTER_MARGIN_MS).finally(() => process.exit(0));
    if (!text) done();
    else process.stdout.write(`${text}\n`, done);
  };
  let input = null;
  const carryOn = () => answer(fallback(host, event, input));
  let watchdog = setTimeout(carryOn, budgetMs(process.env, host));
  try {
    input = parse(await readStdin());
    // The timeout the hooks file gives this hook, when it names one: the
    // answer comes before it too.
    const configured = configuredTimeoutMs(host, event, process.env, input);
    if (configured !== null) {
      limitMs = configured;
      clearTimeout(watchdog);
      watchdog = setTimeout(carryOn, Math.max(0, started + Math.min(budgetMs(process.env, host), configured - AFTER_MARGIN_MS) - Date.now()));
    }
    answer(await runHook({ host, event, input, env: process.env }));
  } catch {
    answer(fallback(host, event, input));
  } finally {
    clearTimeout(watchdog);
  }
}

if (process.argv[1] && fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url))) {
  main();
}
