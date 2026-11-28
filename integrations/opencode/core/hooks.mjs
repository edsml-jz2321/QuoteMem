// Turn-level automatic memory for hook-based hosts.
//
// A host adapter maps its own events onto four calls:
//   prompt(): the user sent a message. Records it, runs a silent forget when
//             the message asks for one, else a gated recall. Returns the text
//             to add to the model context (or null).
//   reply():  the turn finished; records the final assistant text and queues
//             a batch every `captureEvery` user turns.
//   catchUp(): the session ended and its host named the conversation's
//             transcript: queues the turns no prompt hook recorded.
//   end():    the session ended; queues what is left. The session file stays
//             (turn numbers, recall memory), so a resumed session goes on
//             where it stopped; it is removed once abandoned for a week.
//   endSync(): end() for a host that is exiting: synchronous, no waiting.
//   sweep():  queues what idle sessions left (prompt() runs it too).
//   armIdle(): with `idleWait`, makes sure one background flusher waits to
//             run that sweep when the oldest chat's idle time is over
//             (runIdleWaiter), so the last turns go without a later hook.
//
// State lives in files under `settings.stateDir`, so every call may come from
// a new short-lived process: sessions/<sha16([owner, session])>.json holds the
// turn being collected, the pending batch and the recall memory, guarded by a
// per-session lock. Batches go to the disk queue and the detached flusher
// sends them; no call waits for the service except recall (bounded by
// `recallTimeoutMs`) and forget (bounded by `forgetTimeoutMs` from the
// message's arrival). No call throws: every failure is logged and the turn
// goes on without memory.
//
// One state folder may serve several users (QUOTEMEM_USER_ID) and services
// (QUOTEMEM_URL): a session belongs to its owner (a user at a service,
// DiskQueue's ownerKey), and the queue keeps jobs, forget outcomes and notes
// per owner. So a user's turns are sent only as that user and only to the
// service they were said to, and a forget outcome is told only to the user
// who asked for it.
import { randomBytes } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { SESSION_KEEP_MS, cutBatch, newCaptureState, ownerKey, recordTurn, sha16, timeOf } from './capture.mjs';
import { replaceOlder, setupNote } from './autostart.mjs';
import { publicSettings, resolveSettings } from './config.mjs';
import { DiskQueue, IDLE_LOCK_STALE_MS, runFlusher } from './diskqueue.mjs';
import { chatForgetNote, forgetNote, plainForgetDone, setupHeader, unavailableNote } from './format.mjs';
import { detectForgetRequests, forgetBody, forgottenOf, noteForgotten } from './forget.mjs';
import { isMemoryRelated } from './gate.mjs';
import { errKind } from './http.mjs';
import { checkListener, onThisComputer } from './listener.mjs';
import { adoptLock, readJson, releaseLock, touchLock, tryLock, withLock, writeFileAtomic } from './lock.mjs';
import { homeDir, settleLaunches } from './proc.mjs';
import { RecallBreaker, newRecallMemo, recallQuery, recallTurn, rememberIds, searchRecall } from './recall.mjs';
import { REDACT_MARGIN, redactText } from './redact.mjs';
import { serviceToken } from './token.mjs';
import { promptHead, readConversation, turnReply } from './transcript.mjs';

const SESSION_LOCK_STALE_MS = 15_000;
const SESSION_LOCK_WAIT_MS = 3000;
const LAST_USER_CHARS = 4000;
const MAX_PENDING_FORGETS = 50;
// The user messages prompt hooks recorded, kept so a transcript read at the
// session's end skips them (catchUp): at most this many, newest last. Past
// it (or for a chat begun before the record existed) the transcript is not
// matched once: its turns so far are taken as saved.
const MAX_SEEN = 512;
// A host without turn ids: the same text again within this time is the same
// prompt (a second install of the same hooks), not a new message.
const SAME_PROMPT_MS = 5000;
// A turn held open for its reply (hold) is saved by the idle sweep after
// this long: with the reply its transcript has by then, else without one.
export const HELD_TURN_MS = 3 * 60_000;
// The first prompt of a new session that asks about memory waits at most
// this long (and within its recall deadline) for the turns of this user's
// other sessions it queued.
export const FIRST_PROMPT_WAIT_MS = 3000;
// For NEW_SESSION_MS after it starts, a session sends the unsent turns of
// the other sessions idle for OTHER_IDLE_MS (a chat closed without its end,
// a switch), not those of a chat still in use. On a host whose closed chats
// send no end (reliableEnd false), its start and first prompt also send at
// once those of this host's chats with no turn in flight.
export const NEW_SESSION_MS = 10 * 60_000;
export const OTHER_IDLE_MS = 60_000;
// The batches a new session's first prompt waits for were queued at most
// this long before the session's start was recorded.
const STARTED_MARGIN_MS = 1000;
// With turn ids, an earlier turn still running when the next prompt comes
// (a follow-up sent while the agent works) stays open beside it for this
// long, at most MAX_RUNNING of them; its own end then saves it.
export const RUNNING_TURN_MS = 10 * 60_000;
// A later turn of the same chat has ended (Cursor runs a chat's turns one at a
// time): an earlier one still running is saved this much later, with the
// reply the transcript has by then, if its own end has not come.
export const SUPERSEDED_TURN_MS = 60_000;
const MAX_RUNNING = 8;
// The flusher that waits for idle chats (runIdleWaiter) sets one timer at a
// time, at most this long, and reads the wall clock when it fires: after a
// sleep of the computer it acts within this time of waking.
export const IDLE_TICK_MS = 60_000;
// The least time between two of its sweeps (a chat whose lock another
// process held is looked at again after this).
export const IDLE_PASS_MS = 30_000;
// How long one such flusher lives beyond the idle time. Chats still in use
// then are handed to a new flusher.
export const IDLE_EXTRA_LIFE_MS = 3_600_000;
// A prompt that comes while a forget of this user is still queued waits for
// it, reading only the local queue, at most FORGET_WAIT_MS (a host with a
// short hook timeout gives less: forgetWaitWithin), and has no recall when
// it is still queued then. A forget queued more than FORGET_HOLD_MS ago (a
// busy service holds it behind earlier batches) is not waited for any more,
// but still holds recall back until it ends: what it removes must not be
// shown meanwhile.
export const FORGET_WAIT_MS = 8000;
export const FORGET_HOLD_MS = 120_000;
const FORGET_POLL_MS = 100;
// What a prompt hook needs beside its wait and its recall: start-up, the
// session files and the answer.
const PROMPT_MARGIN_MS = 2000;
// The longest a session's first prompt waits for the listener check that
// tells the user memory is off (listenerNote). A service on this computer
// answers it in milliseconds; a busy or paused one is not waited for.
export const LISTENER_NOTE_MS = 300;
// The setup states of a listener that answered the check without proving
// the token (http.mjs listenerRefusal): the user is told why memory is off.
const LISTENER_STATES = new Set(['listener', 'older', 'older-hand', 'restarting']);

/**
 * The longest a prompt may wait for a queued forget within a hook time
 * limit of `limitMs`, leaving room for the recall after it.
 */
export function forgetWaitWithin(limitMs, settings) {
  return Math.max(0, Math.min(FORGET_WAIT_MS, limitMs - settings.recallTimeoutMs - PROMPT_MARGIN_MS));
}

// Match a transcript's user messages (`heads`, promptHead of each) with
// those prompt hooks recorded (`seen`), both in the order said: the longest
// in-order matching, the earliest where several fit, so a repeated message
// ("run it again") never matches past one in between. Returns, for each
// head, the index in `seen` it matches, or -1.
export function alignTurns(heads, seen) {
  const n = heads.length;
  const m = seen.length;
  const out = new Array(n).fill(-1);
  if (n === 0 || m === 0) return out;
  const longest = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      longest[i][j] = heads[i] !== null && heads[i] === seen[j] ? longest[i + 1][j + 1] + 1 : Math.max(longest[i + 1][j], longest[i][j + 1]);
    }
  }
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (heads[i] !== null && heads[i] === seen[j] && longest[i][j] === longest[i + 1][j + 1] + 1) {
      out[i] = j;
      i += 1;
      j += 1;
    } else if (longest[i + 1][j] >= longest[i][j + 1]) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return out;
}

function newSession(sessionId, user) {
  return { version: 1, sessionId, user, seq: 0, prompts: 0, open: null, lastTurn: null, lastUser: '', capture: newCaptureState(), recall: newRecallMemo() };
}

// The earlier turns of a session still running beside the open one.
function runningOf(state) {
  return Array.isArray(state.running) ? state.running : [];
}

// Whether a session has a turn in flight: open or running, not held for its
// reply after its end, and not superseded by a later turn's end.
function inFlight(state) {
  const live = (turn) => Boolean(turn) && typeof turn === 'object' && !turn.held && !turn.superseded;
  return live(state.open) || runningOf(state).some(live);
}

// Whether a turn waits no longer: held for its reply for HELD_TURN_MS, or
// (`running`, beside the open one) still running after RUNNING_TURN_MS.
function turnDue(turn, running, now = Date.now()) {
  if (!turn || typeof turn !== 'object') return false;
  const held = Number(turn.held?.at);
  if (Number.isFinite(held) && now - held >= HELD_TURN_MS) return true;
  if (!running) return false;
  const superseded = Number(turn.superseded?.at);
  if (Number.isFinite(superseded) && now - superseded >= SUPERSEDED_TURN_MS) return true;
  return now - Number(turn.at ?? 0) >= RUNNING_TURN_MS;
}

// When the idle sweep (queueSessions without `all`) would queue something of
// this session, last written at `mtime`, or null: its pending turns once it
// has been idle for `idleMs`; a turn held for its reply, or still running
// beside the open one, when turnDue says so (and the session has been idle
// a little). A turn in flight alone is left for its own end.
function idleDue(state, mtime, idleMs) {
  let due = Infinity;
  if ((state.capture?.pending?.length ?? 0) > 0) due = mtime + idleMs;
  const early = mtime + Math.min(idleMs, HELD_TURN_MS);
  const dueOf = (turn, running) => {
    if (!turn || typeof turn !== 'object') return Infinity;
    const times = [Number(turn.held?.at) + HELD_TURN_MS];
    if (running) times.push(Number(turn.superseded?.at) + SUPERSEDED_TURN_MS, Number(turn.at ?? 0) + RUNNING_TURN_MS);
    return Math.min(...times.filter(Number.isFinite));
  };
  for (const at of [dueOf(state.open, false), ...runningOf(state).map((t) => dueOf(t, true))]) {
    if (Number.isFinite(at)) due = Math.min(due, Math.max(early, at));
  }
  return Number.isFinite(due) ? due : null;
}

// Whether a session holds turns the idle sweep would queue some time.
function waitsIdle(state) {
  return idleDue(state, 0, 0) !== null;
}

// The open or running turn with this id, or null.
function turnOf(state, turn) {
  if (turn === null || turn === undefined) return state.open ?? null;
  if (state.open?.turn === String(turn)) return state.open;
  return runningOf(state).find((t) => t.turn === String(turn)) ?? null;
}

/**
 * Automatic memory for one host.
 * @param {object} options
 * @param {object} options.settings - from resolveSettings (problem must be null).
 * @param {object} options.log - from makeLog; use a file sink when stdout is parsed.
 * @param {string} options.host - prefix of session ids sent to the service, e.g. `claude-code`.
 * @param {string} [options.label] - names the plugin in truncation markers.
 * @param {object} [options.env] - environment of the background flusher (default process.env).
 * @param {boolean} [options.overlap] - turns may overlap (Cursor).
 * @param {boolean} [options.reliableEnd] - false for a host whose chats can
 *   close without a session end (Cursor, Cline).
 * @param {number} [options.forgetWaitMs] - how long a prompt waits for a
 *   queued forget of this user before it recalls (default FORGET_WAIT_MS).
 * @param {boolean} [options.idleWait] - the background flusher keeps the
 *   idle rule (armIdle, runIdleWaiter): for a host whose plugin keeps no
 *   timer of its own (hook processes, OpenClaw).
 */
export class HookMemory {
  constructor({ settings, log, host, label, env = process.env, overlap = false, reliableEnd = true, forgetWaitMs = FORGET_WAIT_MS, idleWait = false }) {
    this.settings = settings;
    this.idleWait = idleWait;
    // The sessions this object wrote with turns the idle sweep would queue
    // (armIdle).
    this.idleSessions = new Set();
    this.forgetWaitMs = Math.max(0, Number(forgetWaitMs) || 0);
    this.log = log;
    this.host = host;
    this.label = label ?? `${host} plugin`;
    this.env = env;
    // Whether turns may overlap (Cursor runs a follow-up while the earlier
    // turn works): a new prompt then leaves a turn that still runs open.
    this.overlap = overlap;
    // Whether every chat of this host ends with a session end. Where a chat
    // can close without one (a Cursor tab, a Cline task), a new session's
    // start and first prompt send this host's chats between turns at once.
    this.reliableEnd = reliableEnd;
    // Where a start of the local service on demand logs (never serialized).
    if (!settings.autostartLog) Object.defineProperty(settings, 'autostartLog', { value: log, enumerable: false, configurable: true });
    this.queue = new DiskQueue(settings, log);
    this.sessionsDir = path.join(settings.stateDir, 'sessions');
    // Whose sessions these are: the owner (one user at one service URL). A
    // session file of another user, or of the same user at another service,
    // is never read or sent as this one's.
    this.user = this.queue.owner;
    this.breakerFile = path.join(settings.stateDir, 'recall-breaker.json');
    // Forgets still running when their turn's note was given, for every
    // session of this owner: the next prompt of any of its sessions reports
    // how each ended, so a session that ends first does not lose it (a plain
    // success only to the session that asked, finishedForgets).
    this.forgetsFile = path.join(this.queue.ownerDir, 'forgets.json');
    if (idleWait) this.registerOwner();
  }

  // The waiting flusher of this service saves the idle chats of every owner
  // registered here with the same credential: it reads who they are and
  // their own settings from here (never the token: only a hash of it).
  registerOwner() {
    const record = { version: 2, url: this.settings.url, userId: this.settings.userId, credential: this.queue.credential, settings: publicSettings(this.settings) };
    try {
      if (JSON.stringify(readJson(this.queue.ownerFile)) !== JSON.stringify(record)) writeFileAtomic(this.queue.ownerFile, JSON.stringify(record));
      // In use now: a waiting flusher unregisters only an owner untouched for a week.
      else fs.utimesSync(this.queue.ownerFile, new Date(), new Date());
    } catch {
      // without it only this owner's own waiting flusher saves its idle chats
    }
  }

  /** The session id sent to the service for a host session id. */
  sessionId(session) {
    return `${this.host}:${session}`;
  }

  fileOf(sessionId) {
    return path.join(this.sessionsDir, `${sha16([this.user, sessionId])}.json`);
  }

  // `lockMs`: how long to wait for the session's lock (default SESSION_LOCK_WAIT_MS).
  async update(sessionId, fn, { lockMs = SESSION_LOCK_WAIT_MS } = {}) {
    const file = this.fileOf(sessionId);
    return withLock(`${file}.lock`, async () => {
      const state = readJson(file) ?? newSession(sessionId, this.user);
      const result = await fn(state);
      if (state.remove) {
        try { fs.unlinkSync(file); } catch { /* gone */ }
        this.idleSessions.delete(sessionId);
      } else {
        state.updated = Date.now();
        writeFileAtomic(file, JSON.stringify(state));
        if (waitsIdle(state)) this.idleSessions.add(sessionId); else this.idleSessions.delete(sessionId);
      }
      return result;
    }, { timeoutMs: lockMs, staleMs: SESSION_LOCK_STALE_MS });
  }

  // A user message as kept on disk: redacted, capped; '' when it is only a secret.
  stored(text) {
    const raw = String(text ?? '');
    const redacted = redactText(raw.length > this.settings.maxMessageChars + REDACT_MARGIN ? raw.slice(0, this.settings.maxMessageChars + REDACT_MARGIN) : raw);
    return redacted.secretOnly ? '' : redacted.text;
  }

  // Close the open turn of a state; returns jobs when a batch is due.
  commit(state, { reply = null, time, completed }) {
    const open = state.open;
    state.open = null;
    if (!open) return [];
    state.lastTurn = open.turn;
    if (this.settings.captureEvery <= 0) return [];
    state.seq += 1;
    const assistant = reply ? { text: reply, time: time ?? Date.now() } : null;
    const due = recordTurn(this.settings, state.capture, { users: open.users, assistant, turn: state.seq, completed }, { label: this.label });
    return due || open.alone ? cutBatch(this.settings, state.capture, state.sessionId) : [];
  }

  // Before a new prompt (turn: its id, or null): the earlier open turn.
  // With turn ids, one still running (no end yet, for less than
  // RUNNING_TURN_MS, and not held for a reply) stays open beside the new one:
  // turns may overlap (a follow-up sent while the agent works), and each
  // turn's own end saves it. Otherwise it is saved as it is.
  closeEarlier(state, turn) {
    // First the earlier turns that wait no longer (turnDue).
    const out = this.commitDue(state);
    const open = state.open;
    if (!open) return out;
    if (this.overlap && turn !== null && open.turn !== turn && !open.held && Date.now() - Number(open.at ?? 0) < RUNNING_TURN_MS) {
      state.running = [...runningOf(state), open];
      state.open = null;
      while (state.running.length > MAX_RUNNING) out.push(...this.commitRunning(state, 0));
      return out;
    }
    out.push(...(open.held ? this.commitHeld(state) : this.commit(state, { completed: false })));
    return out;
  }

  // Save the running turn at `index` (as reply would, with `args`; else as
  // it is), leaving the open one open.
  commitRunning(state, index, args = null) {
    const [turn] = runningOf(state).splice(index, 1);
    const open = state.open;
    state.open = turn;
    const out = args ? this.commit(state, args) : turn.held || turn.superseded ? this.commitHeld(state) : this.commit(state, { completed: false });
    state.open = open;
    return out;
  }

  // A turn ended: the running turns before it (the first `before`) were
  // sent earlier, so they have ended too in a host that runs a chat's turns
  // one at a time (Cursor). Each is saved SUPERSEDED_TURN_MS later unless
  // its own end comes first, with `transcript`'s reply if it has one.
  supersede(state, before, transcript) {
    for (const turn of runningOf(state).slice(0, before)) {
      if (!turn.superseded) turn.superseded = { at: Date.now(), ...(transcript ? { transcript } : {}) };
    }
  }

  // Save the turns that wait no longer (turnDue), in their order.
  commitDue(state) {
    const out = [];
    for (let index = runningOf(state).findIndex((t) => turnDue(t, true)); index >= 0; index = runningOf(state).findIndex((t) => turnDue(t, true))) {
      out.push(...this.commitRunning(state, index));
    }
    if (turnDue(state.open, false)) out.push(...this.commitHeld(state));
    return out;
  }

  // Save every open and running turn (a session that ends, or is abandoned).
  commitAll(state) {
    const out = [];
    while (runningOf(state).length > 0) out.push(...this.commitRunning(state, 0));
    out.push(...(state.open?.held ? this.commitHeld(state) : this.commit(state, { completed: false })));
    return out;
  }

  // Cut what is pending early (before a forget, at the end, after idle time).
  // The service reads the session's previous batch as context by itself, so
  // a short batch is sent as it is: no earlier turn is sent twice.
  cutEarly(state) {
    if (this.settings.captureEvery <= 0) return [];
    return cutBatch(this.settings, state.capture, state.sessionId);
  }

  enqueue(jobs) {
    if (jobs.length === 0) return;
    try {
      this.queue.enqueue(this.withForgetsAgain(jobs));
      this.log.debug(`queued ${jobs.length} job(s)`);
    } catch (error) {
      this.log.warnOnce(`queue-write:${errKind(error)}`, `could not write the capture queue (${errKind(error)})`);
    }
  }

  /**
   * The jobs, with every forget asked since some batch's user messages were
   * said (forgottenOf) after the last such batch: words said before a
   * forget and saved after it get that forget too. The forget waits until the batches
   * before it are stored (not `inChat`); a message without a time counts as
   * said before.
   */
  withForgetsAgain(jobs) {
    const owner = { url: this.settings.url, userId: this.settings.userId };
    const forgets = jobs.some((job) => job.kind !== 'forget') ? forgottenOf(this.env, owner) : [];
    if (forgets.length === 0) return jobs;
    const firstSaid = jobs.map((job) => {
      const messages = job.kind === 'forget' ? [] : (job.body?.messages ?? []).filter((m) => m?.role === 'user');
      return messages.length === 0 ? Infinity : Math.min(...messages.map((m) => (Number.isFinite(m.timestamp) ? m.timestamp : -Infinity)));
    });
    const after = new Map();
    for (const item of forgets) {
      // Said before the forget, as far as its time tells: a turn known to
      // be said after it (saidAfter, the reading before) is not.
      // A stamp whose minute ended before the forget proves it was said
      // before, whatever the bound says (a clock set back between runs).
      const last = firstSaid.findLastIndex((said, index) => said < item.at
        && ((Number.isFinite(said) && said + 60_000 <= item.at) || !(Number(jobs[index].saidAfter) >= item.at)));
      if (last < 0) continue;
      const job = jobs[last];
      const body = forgetBody(this.settings, job.body.session_id, `again:${item.at}:${sha16(job.request_id)}`, item.statement);
      if (!body || this.queue.result(body.request_id)) continue;
      after.set(last, [...(after.get(last) ?? []), { kind: 'forget', request_id: body.request_id, body, inChat: false, quiet: true, url: this.settings.url, created: Date.now(), attempts: 0, nextAt: 0 }]);
    }
    return jobs.flatMap((job, index) => [job, ...(after.get(index) ?? [])]);
  }

  /** Start the background flusher if anything is queued and none runs. */
  // `sync`: for an exit handler, which cannot wait (startFlusher).
  flush({ sync = false } = {}) {
    return this.queue.startFlusher({ env: this.env, sync, idle: this.idleWait });
  }

  /**
   * With `idleWait`, when a session this object wrote holds turns the idle
   * sweep would queue: start a background flusher unless one already waits
   * for this owner's idle chats. It wakes when the oldest chat's idle time
   * is over, queues its turns as sweep() does, and sends them
   * (runIdleWaiter). A hook calls this after its answer.
   * @returns {boolean} whether a flusher was started.
   */
  armIdle({ sync = false } = {}) {
    if (!this.idleWait || this.idleSessions.size === 0) return false;
    return this.queue.startFlusher({ env: this.env, sync, idle: 'pending' });
  }

  /**
   * When the idle sweep would next queue something of this owner's sessions
   * (idleDue), or null when nothing waits, the idle time is off or capture
   * is off. Another user's or service's sessions are not counted.
   */
  idleDueAt(sessions = readSessions(this.sessionsDir)) {
    if (!(this.settings.idleFlushMs > 0) || !(this.settings.captureEvery > 0)) return null;
    let next = null;
    for (const { data, mtime } of sessions) {
      if (data.user !== this.user) continue;
      const due = idleDue(data, mtime, this.settings.idleFlushMs);
      if (due !== null && (next === null || due < next)) next = due;
    }
    return next;
  }

  /**
   * A call reached the service (a search answered): the jobs that could not
   * reach it are sent now, not after their retry delay, and the flusher starts.
   */
  reached() {
    this.queue.noteContact();
    this.flush();
  }

  /**
   * The user sent a message.
   * @param {{session: string, turn?: string|number|null, text: string, time?: any}} input -
   *   `turn` is the host's id of this turn (a repeated call with the same id is recorded once).
   * @returns {Promise<{context: string|null, kind: 'recall'|'forget'|'unavailable'|null, ids?: string[], forgets?: object[]}>}
   *   The context also reports how an earlier forget ended in the
   *   background: a plain success only in the session that asked for it,
   *   any other outcome in whichever session prompts first. `unavailable`: a
   *   question about the user or the past while the service could not be
   *   searched.
   */
  async prompt({ session, turn = null, text, time }) {
    const started = Date.now();
    const result = await this.promptTurn({ session, turn, text, time, started });
    this.log.debug(`prompt ${result.kind ?? 'none'} in ${Date.now() - started} ms`);
    return result;
  }

  async promptTurn({ session, turn = null, text, time, started = Date.now() }) {
    const none = { context: null, kind: null };
    try {
      await this.sweep();
      const sessionId = this.sessionId(session);
      const raw = String(text ?? '');
      const stored = this.stored(raw);
      const saidAt = timeOf(time ?? Date.now());
      let statements = [];
      let skip = new Set();
      let first = false;
      let fresh = false;
      let startedAt = Date.now();
      const outcome = await this.update(sessionId, (state) => {
        // Without a host turn id, every prompt gets its own number: a prompt
        // whose turn got no reply must not make the next one a duplicate.
        const auto = turn === null || turn === undefined;
        const said = auto ? sha16(raw) : null;
        if (auto) {
          if (state.open?.said === said && Date.now() - (state.open.at ?? 0) < SAME_PROMPT_MS) return { duplicate: true };
        } else if ((state.open && state.open.turn === String(turn)) || runningOf(state).some((t) => t.turn === String(turn))) {
          return { duplicate: true };
        }
        // The first prompt of a new session waits for the other sessions'
        // turns queued for it (by its start or by itself).
        first = !(Number(state.prompts) > 0) && !(Number(state.seq) > 0);
        if (!Number.isFinite(state.startedAt)) state.startedAt = Date.now();
        startedAt = state.startedAt;
        fresh = Date.now() - state.startedAt < NEW_SESSION_MS;
        state.prompts = Math.max(Number(state.prompts) || 0, Number(state.seq) || 0) + 1;
        const turnKey = auto ? `n${state.prompts}` : String(turn);
        // An earlier turn that got no reply: saved with the user's words (a held
        // one: with its transcript's reply), unless it still runs (closeEarlier).
        const out = this.closeEarlier(state, auto ? null : String(turn));
        // The nonce makes each prompt's forget a new operation, even when a
        // host repeats its turn ids; a retry of the same job keeps its id.
        const nonce = randomBytes(6).toString('hex');
        // head: tells a transcript's user message apart from an earlier one
        // (openTurn), without keeping the raw text.
        state.open = { turn: turnKey, at: Date.now(), said, head: promptHead(raw), users: stored ? [{ text: stored, time: saidAt }] : [] };
        if (state.open.head) {
          // A chat whose earlier prompts were not recorded, or more prompts
          // than are kept: the next transcript reading cannot be matched.
          if ((!Array.isArray(state.seen) && state.prompts > 1) || (state.seen?.length ?? 0) >= MAX_SEEN) state.seenBlind = true;
          state.seen = [...(Array.isArray(state.seen) ? state.seen : []), state.open.head].slice(-MAX_SEEN);
        }
        if (this.settings.forget === 'auto') {
          statements = detectForgetRequests(raw, { previous: state.lastUser })
            .map(({ statement, inChat, ownTurn }) => ({ statement, inChat, ownTurn, body: forgetBody(this.settings, sessionId, `${turnKey}:${nonce}`, statement) }))
            .filter((item) => item.body);
        }
        if (stored) state.lastUser = stored.slice(0, LAST_USER_CHARS);
        // A forget applied again after this turn's batch (ownTurn) must not
        // reach the turns after it: the turn is a batch of its own.
        if (statements.some((item) => item.ownTurn)) state.open.alone = true;
        if (statements.length > 0) {
          // Everything said before the request is queued before it, so the
          // forget can find it.
          out.push(...this.cutEarly(state));
        } else {
          skip = recallTurn(this.settings, state.recall);
        }
        // Queued before the state that no longer holds these turns is saved.
        this.enqueue(out);
        return { duplicate: false };
      });
      // A forget of this user still running (asked in this chat or another)
      // is waited for first when this prompt would search, so its outcome is
      // reported below; one still running after the wait leaves this turn
      // without recall.
      const held = !outcome.duplicate && statements.length === 0 && Boolean(recallQuery(this.settings, raw)) && await this.forgetHeld();
      const updates = outcome.duplicate ? [] : [...await this.finishedForgets(sessionId), ...await this.chatForgets()];
      const earlier = updates.length > 0 ? updates.join('\n') : null;
      // A new session (its first NEW_SESSION_MS): the unsent turns of this
      // user's other sessions idle for OTHER_IDLE_MS go now (a chat closed
      // without its end, a switch); at its first prompt, on a host without a
      // reliable session end, also those of this host's chats between turns.
      // The first prompt, when it asks about memory, waits for the batches
      // queued for this session (queuedFor: by itself, or by its start) to
      // be stored. A chat whose lock its start holds is waited for while it
      // queues that chat's batch. All of it takes at most
      // FIRST_PROMPT_WAIT_MS and stays within its own recall deadline. A
      // prompt that searches nothing waits for nothing.
      let waited = 0;
      if (fresh && !outcome.duplicate && statements.length === 0) {
        const gated = first && recallQuery(this.settings, raw);
        const began = Date.now();
        const until = began + Math.min(FIRST_PROMPT_WAIT_MS, this.settings.recallTimeoutMs);
        const sent = await this.queueSessions({ all: true, except: sessionId, minIdleMs: OTHER_IDLE_MS, betweenTurns: first && !this.reliableEnd, ...(gated ? { lockUntil: until } : {}) });
        if (sent > 0) this.flush();
        if (gated) {
          await this.waitQueued(sessionId, startedAt - STARTED_MARGIN_MS, until, sent > 0);
          waited = Date.now() - began;
        }
      }
      if (statements.length > 0) {
        // So are the pending turns of this user's other sessions.
        await this.queueSessions({ all: true, except: sessionId });
        const result = await this.forget(sessionId, statements, started + this.settings.forgetTimeoutMs, saidAt);
        return earlier ? { ...result, context: `${earlier}\n${result.context}` } : result;
      }
      if (outcome.duplicate) {
        this.flush();
        return none;
      }
      if (held) {
        // Nothing is recalled and nothing is said about the forget: what it
        // removes must not come back while it runs.
        this.flush();
        return earlier ? { context: earlier, kind: 'forget', held: true } : { ...none, held: true };
      }
      // The background sender starts after the recall, off its way (on
      // Windows its start waits for PowerShell). A first question's wait is
      // taken from its recall deadline, which keeps at least 1 s (or all of
      // a shorter deadline).
      const limit = this.settings.recallTimeoutMs;
      const { searched, listener, ...answer } = await this.recall(sessionId, raw, skip, waited > 0 ? Math.max(Math.min(1000, limit), limit - waited) : null);
      let recalled = answer;
      // The first prompt of a session that recalled nothing: a service that
      // gets no token (it does not prove it holds it) is told at once, whatever
      // was asked, so the user knows memory is off and why. A search that
      // went out already checked the listener (`listener`: its setup state);
      // else one check is made (listenerNote).
      if (first && recalled.kind === null) {
        const note = searched ? (listener ? await this.setupNoteOnce(sessionId, listener) : null) : await this.listenerNote(sessionId);
        if (note) recalled = { context: note, kind: 'setup' };
      }
      this.flush();
      if (!earlier) return recalled;
      return { ...recalled, context: recalled.context ? `${earlier}\n${recalled.context}` : earlier, kind: recalled.kind ?? 'forget' };
    } catch (error) {
      this.log.warnOnce(`prompt:${errKind(error)}`, `automatic memory skipped this turn (${errKind(error)})`);
      return none;
    }
  }

  // Whether a forget of this user is still queued, after waiting for it:
  // up to forgetWaitMs while every queued forget is younger than
  // FORGET_HOLD_MS, not at all once one is older. Recall is held back for as
  // long as any forget is queued; only a queue with none lets it through.
  // Only the local queue is read; a queue that cannot be read holds nothing.
  async forgetHeld() {
    const pending = (since = 0) => {
      try {
        return this.queue.forgetPending(since);
      } catch {
        return false;
      }
    };
    const end = Date.now() + this.forgetWaitMs;
    let flushed = false;
    for (;;) {
      if (!pending()) return false;
      // One queued for a while (a busy service holds it behind earlier
      // batches): no wait, but no recall either.
      const old = !pending(Date.now() - FORGET_HOLD_MS);
      if (old || Date.now() >= end) {
        this.log.debug(old ? 'recall held back: a forget queued earlier is still waiting' : 'recall held back: a forget is still running');
        if (!flushed) this.flush();
        return true;
      }
      if (!flushed) {
        // The background sender carries the forget out: make sure it runs.
        this.flush();
        flushed = true;
      }
      await new Promise((resolve) => setTimeout(resolve, Math.min(FORGET_POLL_MS, end - Date.now())));
    }
  }

  // Run fn on the list of pending forgets under its lock, and save the list.
  async withForgets(fn) {
    return withLock(`${this.forgetsFile}.lock`, async () => {
      const list = readJson(this.forgetsFile)?.items;
      const items = Array.isArray(list) ? list : [];
      const result = await fn(items);
      writeFileAtomic(this.forgetsFile, JSON.stringify({ version: 1, items: items.slice(-MAX_PENDING_FORGETS) }));
      return result;
    }, { timeoutMs: SESSION_LOCK_WAIT_MS, staleMs: SESSION_LOCK_STALE_MS });
  }

  // What the service's in-chat forget did while storing this user's batches
  // (a request the plugin's own forget did not carry out, or what it removed
  // beyond that forget's counts): one note, once.
  async chatForgets() {
    try {
      const items = await this.queue.takeChatForgets();
      return items.length > 0 ? [chatForgetNote(items)] : [];
    } catch (error) {
      this.log.warnOnce(`chat-forgets:${errKind(error)}`, `could not read in-chat forget outcomes (${errKind(error)})`);
      return [];
    }
  }

  // This user's forgets that were still running when their note was given,
  // in any session: each that has finished leaves the list and gets a note
  // now, so the user learns how it ended. That note may come in another
  // conversation, so it gives counts only: no word of the request or of a
  // memory it matched. Its result keeps no memory text after that. A plain
  // success (plainForgetDone) is told only in the session that asked for it
  // (`sessionId` here): another session says nothing of it, and it waits for
  // that session's next prompt until the age limit.
  async finishedForgets(sessionId) {
    if (!fs.existsSync(this.forgetsFile)) return [];
    try {
      return await this.withForgets((items) => {
        const notes = [];
        const left = [];
        for (const item of items) {
          const result = item?.requestId ? this.queue.result(item.requestId) : null;
          if (!result) {
            if (item?.requestId && Date.now() - (item.at ?? 0) < this.settings.captureMaxAgeMs) left.push(item);
            continue;
          }
          if (item.session && item.session !== sessionId && plainForgetDone(result)) {
            if (Date.now() - (item.at ?? 0) < this.settings.captureMaxAgeMs) left.push(item);
            else this.queue.markResultRead(item.requestId);
            continue;
          }
          notes.push(forgetNote('', result, { earlier: true }));
          this.queue.markResultRead(item.requestId);
        }
        items.splice(0, items.length, ...left);
        return notes;
      });
    } catch (error) {
      this.log.warnOnce(`forgets:${errKind(error)}`, `could not read earlier forgets (${errKind(error)})`);
      return [];
    }
  }

  // Queue the forgets and wait for them until `deadline` (forgetTimeoutMs
  // after the message arrived); one still running finishes in the background
  // and the next prompt of this user reports it (a plain success: the next
  // prompt of this session).
  async forget(sessionId, statements, deadline, at = Date.now()) {
    const forgets = [];
    for (const { statement, inChat, ownTurn, body } of statements) {
      // Kept for the batches saved later with words from before it. A
      // request whose own message says the fact first (ownTurn) is kept as
      // asked just after that message: the message's batch, saved after
      // this forget ran, is followed by it again (withForgetsAgain).
      try {
        await noteForgotten(this.env, { url: this.settings.url, userId: this.settings.userId }, statement, ownTurn ? at + 1 : at);
      } catch (error) {
        this.log.warnOnce(`forgotten-write:${errKind(error)}`, `could not keep a forget to apply again (${errKind(error)})`);
      }
      if (!this.queue.result(body.request_id) && !this.queue.has(body.request_id)) {
        // inChat: the service checks the request again when its turn is
        // stored, so the forget waits only briefly for the saved batches
        // before it. A batch before it that the service has not saved yet
        // (held back as busy) holds it until saved, for at most
        // captureBusyMaxWaitMs: a forget never runs ahead of an unsent batch.
        this.enqueue([{ kind: 'forget', request_id: body.request_id, body, inChat: inChat === true, url: this.settings.url, created: Date.now(), attempts: 0, nextAt: 0 }]);
      }
      forgets.push({ statement: redactText(statement).text, requestId: body.request_id });
    }
    // Starting the flusher is not waiting for the forget: on Windows it takes
    // PowerShell up to half a second, which the wait gets back (at most 1 s).
    const flushing = Date.now();
    this.flush();
    await settleLaunches(1000);
    const until = deadline + Math.min(1000, Date.now() - flushing);
    const notes = [];
    for (const item of forgets) {
      const result = await this.queue.waitResult(item.requestId, Math.max(0, until - Date.now()));
      item.status = result?.status ?? 'pending';
      notes.push(forgetNote(item.statement, result ?? { status: 'pending' }));
      if (result) this.queue.markResultRead(item.requestId);
      this.log.debug(`forget ${item.requestId} ${item.status}`);
    }
    const pending = forgets.filter((item) => item.status === 'pending');
    if (pending.length > 0) {
      try {
        await this.withForgets((items) => {
          for (const item of pending) {
            // No words of the request are kept: its later note names none.
            if (!items.some((known) => known.requestId === item.requestId)) items.push({ requestId: item.requestId, session: sessionId, at: Date.now() });
          }
        });
      } catch {
        // without the entry, a forget that fails later is only logged
      }
    }
    return { context: notes.join('\n'), kind: 'forget', forgets: forgets.map(({ requestId, status }) => ({ requestId, status })) };
  }

  // `timeoutMs`: what is left of the recall deadline (the first prompt of a
  // session may have waited for other sessions' turns).
  async recall(sessionId, raw, skip, timeoutMs = null) {
    const query = recallQuery(this.settings, raw);
    if (!query) return { context: null, kind: null };
    const breaker = RecallBreaker.from(readJson(this.breakerFile));
    const before = JSON.stringify(breaker);
    const report = {};
    const settings = timeoutMs ? { ...this.settings, recallTimeoutMs: timeoutMs } : this.settings;
    // The copy keeps where a start of the local service logs (not enumerable).
    if (settings !== this.settings) Object.defineProperty(settings, 'autostartLog', { value: this.settings.autostartLog, enumerable: false, configurable: true });
    const block = await searchRecall(settings, query, { skipIds: skip, breaker, log: this.log, report });
    if (JSON.stringify(breaker) !== before) {
      try { writeFileAtomic(this.breakerFile, JSON.stringify(breaker)); } catch { /* best effort */ }
    }
    if (report.reached) this.reached();
    if (!block) {
      // A question about the user or the past that could not be searched:
      // the model must not take "nothing retrieved" for "nothing stored".
      // When the service is not running and cannot start by itself, the user
      // is also told, once per session, how to set it up.
      if (report.unavailable && isMemoryRelated(raw, { statements: false })) {
        const setup = await this.setupNoteOnce(sessionId, report.autostart);
        return { context: setup ? `${unavailableNote()}\n${setup}` : unavailableNote(), kind: 'unavailable' };
      }
      // `searched`: the breaker was asked, and any search went through the
      // listener check; `listener`: the setup state of a listener that
      // answered it without proving the token.
      return { context: null, kind: null, searched: true, listener: LISTENER_STATES.has(report.autostart) ? report.autostart : null };
    }
    try {
      await this.update(sessionId, (state) => { rememberIds(state.recall, block.ids); });
    } catch {
      // a lost dedupe entry only allows a repeat later
    }
    return { context: block.text, kind: 'recall', ids: block.ids };
  }

  // The setup note when the program on the service's port does not prove
  // the token (an older service, or another program), once per session; null
  // otherwise. One token-free check within LISTENER_NOTE_MS; nothing is
  // sent. None while recall's breaker is open (the service failed lately);
  // a check that is not answered in time counts as a timed-out search, so a
  // stalled service soon costs new sessions nothing.
  async listenerNote(sessionId) {
    if (!onThisComputer(this.settings.url)) return null;
    try {
      const breaker = RecallBreaker.from(readJson(this.breakerFile));
      if (breaker.open) return null;
      const settings = { ...this.settings };
      if (!settings.token && settings.localService) settings.token = serviceToken(process.env, settings.localService);
      const verdict = await checkListener(settings, { timeoutMs: Math.min(LISTENER_NOTE_MS, this.settings.recallTimeoutMs) });
      if (verdict.kind === 'no-answer') {
        breaker.fail('timeout');
        try { writeFileAtomic(this.breakerFile, JSON.stringify(breaker)); } catch { /* best effort */ }
      }
      if (verdict.ok || verdict.refused || (verdict.kind !== 'older' && verdict.kind !== 'foreign')) return null;
      const state = verdict.kind === 'older' ? replaceOlder(this.settings, { log: this.settings.autostartLog }).state : 'listener';
      return await this.setupNoteOnce(sessionId, state);
    } catch {
      return null;
    }
  }

  // The setup note for an autostart state, at most once per session.
  async setupNoteOnce(sessionId, state) {
    const text = setupNote(state);
    if (!text) return null;
    try {
      const first = await this.update(sessionId, (session) => {
        if (session.setupNoted) return false;
        session.setupNoted = true;
        return true;
      });
      return first ? `${setupHeader(state)} This note is from the memory plugin, not from the user. Tell the user once, briefly: ${text}` : null;
    } catch {
      return null;
    }
  }

  /**
   * The open turn of a session, for reading its reply from a transcript:
   * its id, the promptHead() of its user message and its time. Null when no
   * turn is open.
   * @returns {{turn: string, head: string|null, time: number|null}|null}
   */
  openTurn(session, turn = null) {
    try {
      const state = readJson(this.fileOf(this.sessionId(session)));
      // `turn`: that turn, open or still running; else the open one.
      const open = state && typeof state === 'object' ? turnOf(state, turn) : null;
      if (!open || typeof open !== 'object') return null;
      const time = Number(open.users?.[0]?.time ?? open.at);
      return { turn: String(open.turn), head: typeof open.head === 'string' ? open.head : null, time: Number.isFinite(time) ? time : null };
    } catch {
      return null;
    }
  }

  /**
   * The turn finished.
   * @param {{session: string, turn?: string|number|null, text?: string|null, time?: any, completed?: boolean, exact?: boolean}} input -
   *   `text` is the final assistant reply; `completed: false` for a turn the
   *   user interrupted or that failed (its user messages are still kept).
   *   `exact`: only when the open turn is `turn` (a reply that came after
   *   its turn's end, which leaves any other open turn as it is).
   */
  async reply({ session, turn = null, text = null, time, completed = true, exact = false, transcript = null }) {
    try {
      const sessionId = this.sessionId(session);
      await this.update(sessionId, (state) => {
        // An earlier turn superseded long enough ago is saved first.
        this.enqueue(this.commitDue(state));
        // An earlier turn still running beside the open one: only it is saved.
        const index = turn === null || turn === undefined ? -1 : runningOf(state).findIndex((t) => t.turn === String(turn));
        if (index >= 0) {
          this.supersede(state, index, transcript);
          this.enqueue(this.commitRunning(state, index, { reply: completed ? text : null, time, completed }));
          return;
        }
        if (!state.open) return;
        const known = turn !== null && turn !== undefined;
        if (exact && !known) return;
        // The end of another turn (one already saved) leaves the open one where
        // turns overlap; elsewhere it closes it, keeping the user's words.
        if (known && String(turn) !== state.open.turn) {
          if (!this.overlap && !exact) this.enqueue(this.commit(state, { completed: false }));
          return;
        }
        this.supersede(state, runningOf(state).length, transcript);
        this.enqueue(this.commit(state, { reply: completed ? text : null, time, completed }));
      });
      this.flush();
    } catch (error) {
      this.log.warnOnce(`reply:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    }
  }

  /**
   * The turn ended before its reply came (Cursor runs its stop and its
   * reply hooks at the same time): it stays open for the reply, which
   * reply({exact}) gives it. The next prompt, the session end, or the idle
   * sweep after HELD_TURN_MS saves it, with the reply its `transcript` has
   * by then (the sweep), else with the user's words.
   * @param {{session: string, turn: string|number, transcript?: string|null}} input
   */
  async hold({ session, turn, transcript = null }) {
    try {
      await this.update(this.sessionId(session), (state) => {
        const open = turn === null || turn === undefined ? null : turnOf(state, turn);
        if (open) open.held = { at: Date.now(), ...(transcript ? { transcript } : {}) };
      });
    } catch (error) {
      this.log.warnOnce(`hold:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    }
  }

  // Save a held turn whose reply did not come: with the transcript's reply.
  commitHeld(state) {
    const reply = turnReply(state.open?.held?.transcript ?? state.open?.superseded?.transcript, state.open?.head);
    return this.commit(state, { reply: reply?.text ?? null, time: reply?.time ?? Date.now(), completed: Boolean(reply) });
  }

  /**
   * Queue the turns of a whole conversation that no prompt hook recorded:
   * Cursor's CLI in print mode runs no prompt or stop hook, and its session end names the
   * chat's transcript, which holds every turn so far. A later call (the next
   * run of a resumed chat) finds the same turns again, and more.
   * - Only what was added since the last call is read (a byte offset, with
   *   the line before it as a guard); the turns before it count as done. A
   *   transcript rewritten or shortened since (a checkpoint restore) is read
   *   again whole: a turn saved before has the same id and body, so the
   *   service stores it once, and only new words are new.
   * - A last turn without its reply yet is left for the next call.
   * - The new turns are matched, in order, with the user messages prompt
   *   hooks recorded (`state.seen`): a matched turn was saved by those
   *   hooks; the open turn among them, which no stop hook closed, takes its
   *   reply from here. When prompt hooks ran but none of their messages is
   *   among the new turns, or they could not all be kept (seenBlind), none
   *   is saved: a turn is never saved twice.
   * - Every other turn is its own batch, with a request id from its place in
   *   the conversation and its words (no time), so a turn sent again (after
   *   the session file was lost, say) has the same id and body. A forget the
   *   user asked for in such a turn is queued as on a prompt, after the turns
   *   before it and before its own turn; a forget asked later (a later
   *   turn's hooks, or the tools) follows the batch (withForgetsAgain).
   * - The place is kept only once the batches are queued.
   * @param {{session: string, file: string}} input - the transcript path.
   * @returns {Promise<number>} turns queued.
   */
  async catchUp({ session, file, completed = false }) {
    try {
      const sessionId = this.sessionId(session);
      const owner = { url: this.settings.url, userId: this.settings.userId };
      const asked = [];
      const queued = await this.update(sessionId, async (state) => {
        const now = Date.now();
        let mark = state.transcript ?? {};
        let read = readConversation(file, { from: Number.isInteger(mark.offset) ? mark.offset : 0, guard: mark.guard ?? null, tail: true });
        if (read?.changed) {
          // Rewritten or shortened (a checkpoint restore): read whole again.
          // A turn saved before has the same id and body, so the service
          // stores it once; only new words are new.
          this.log.warnOnce('transcript-changed', 'the conversation transcript was rewritten; it is read again from the start');
          mark = Number.isFinite(mark.begun) ? { begun: mark.begun } : {};
          read = readConversation(file, { from: 0, tail: true });
        }
        if (!read?.turns) return 0;
        if (read.skipped) {
          this.log.warnOnce('transcript-large', 'the conversation transcript had more new text than the plugin reads at once (32 MiB); only the last 32 MiB were read, and the turns before them are not saved');
        }
        // `byPlace`: the place is a byte offset, and every turn read is new.
        // Otherwise (the first reading, or one again from the start) the
        // turns before the count are done.
        const byPlace = Number.isInteger(mark.offset) || read.skipped;
        if (!byPlace && read.turns.length < (Number(mark.turns) || 0)) {
          this.log.warnOnce('transcript-changed', 'the conversation transcript was rewritten; it is read again from the start');
          mark = Number.isFinite(mark.begun) ? { begun: mark.begun } : {};
        }
        const base = Math.max(0, Number(mark.turns) || 0);
        let fresh = byPlace ? read.turns : read.turns.slice(base);
        // When each turn was said at the earliest: after the last reading for
        // a turn not there then, after the chat began for one read again.
        // `begun` holds only for text after what the transcript held then.
        // ... and only for the file it was taken from (none: a new chat).
        const sameFile = mark.beginPath ? mark.beginPath === path.resolve(file) : mark.beginOffset === 0;
        const begun = (turn) => (Number.isFinite(mark.begun) && Number.isInteger(mark.beginOffset) && sameFile && Number.isInteger(turn?.start) && turn.start >= mark.beginOffset ? mark.begun : 0);
        const boundOf = (index) => {
          if (!Number.isInteger(mark.offset)) return begun(fresh[index]);
          if (index === 0 && Number.isFinite(mark.waitingBound)) return mark.waitingBound;
          return mark.at ?? 0;
        };
        // A turn is complete once a later user message, or (Cursor) its
        // {"type":"turn_ended"} line, follows it; another host's last turn
        // once it has a reply. A turn not complete yet waits for the next
        // call, with what follows it.
        // A run that ended as it should (`completed`) has written its last
        // reply: the last turn with a reply is complete then too. A reply
        // appended to it later is not saved again (the place is past it).
        const open = fresh.findIndex((turn, index) => !(turn.ended || (turn.assistant && (!turn.cursor || (completed && index === fresh.length - 1)))));
        const waiting = open >= 0 ? fresh[open] : null;
        if (waiting) fresh = fresh.slice(0, open);
        // The place: after the last complete turn's last line, never after
        // a turn_ended line (Cursor rewrites the file each run and moves its
        // single turn_ended to the end); with nothing new complete, where it
        // was. A turn waiting is read again from there.
        const lastDone = byPlace ? null : read.turns[base - 1] ?? null;
        const stop = fresh.length > 0 ? fresh.at(-1).end
          : read.skipped ? (waiting ? waiting.start : read.end)
            : Number.isInteger(mark.offset) ? mark.offset : (lastDone?.end ?? 0);
        const next = {
          turns: base + fresh.length,
          offset: stop,
          guard: read.markAt(stop) ?? (stop === mark.offset ? mark.guard ?? null : null),
          at: now,
          ...(Number.isFinite(mark.begun) ? { begun: mark.begun, beginPath: mark.beginPath ?? null, ...(Number.isInteger(mark.beginOffset) ? { beginOffset: mark.beginOffset } : {}) } : {}),
          // Text after the place that was there already (a turn waiting, a
          // line being written) keeps the bound it has now.
          ...(waiting || read.end < read.size ? { waitingBound: boundOf(fresh.length) } : {}),
        };
        const seen = Array.isArray(state.seen) ? state.seen : null;
        const blind = state.seenBlind === true || (seen === null && Number(state.prompts) > 0);
        const heads = fresh.map((turn) => promptHead(turn.user.text));
        const matches = alignTurns(heads, seen ?? []);
        if (fresh.length > 0 && (blind || ((seen?.length ?? 0) > 0 && matches.every((at) => at < 0)))) {
          this.log.warnOnce('transcript-unmatched', blind
            ? 'the prompt hooks of this chat were not all recorded; the transcript\'s turns so far are taken as saved'
            : 'the conversation transcript does not match the prompts the hooks saw; its turns so far are taken as saved');
          state.seen = [];
          delete state.seenBlind;
          state.transcript = { ...next, ...(mark.capture ? { capture: mark.capture } : {}) };
          return 0;
        }
        const capture = mark.capture ?? newCaptureState();
        const out = [];
        let cut = false;
        let count = 0;
        for (const [index, turn] of fresh.entries()) {
          const number = base + index + 1;
          if (matches[index] >= 0) {
            // A turn the hooks saw, open or still running, with no end yet: its reply is here.
            const reply = turn.assistant?.text ? { reply: turn.assistant.text, time: turn.assistant.time ?? Date.now(), completed: true } : null;
            const running = reply ? runningOf(state).findIndex((t) => t.head === heads[index]) : -1;
            if (running >= 0) out.push(...this.commitRunning(state, running, reply));
            else if (reply && state.open?.head === heads[index]) out.push(...this.commit(state, reply));
            continue;
          }
          // What the hooks collected comes first: it was said before.
          if (!cut) {
            out.push(...this.cutEarly(state));
            cut = true;
          }
          const raw = turn.user.text;
          // Forgets that follow this turn's own batch (ownTurn).
          const own = [];
          if (this.settings.forget === 'auto') {
            for (const { statement, inChat, ownTurn } of detectForgetRequests(raw, { previous: state.lastUser })) {
              const body = forgetBody(this.settings, sessionId, `x${number}`, statement);
              if (!body || this.queue.result(body.request_id)) continue;
              // quiet: its outcome is told by counts only (finishedForgets).
              out.push({ kind: 'forget', request_id: body.request_id, body, inChat: inChat === true, quiet: true, url: this.settings.url, created: Date.now(), attempts: 0, nextAt: 0 });
              asked.push({ requestId: body.request_id, statement, at: turn.user.time ?? Date.now() });
              const again = ownTurn ? forgetBody(this.settings, sessionId, `x${number}:own`, statement) : null;
              if (again && !this.queue.result(again.request_id)) own.push({ kind: 'forget', request_id: again.request_id, body: again, inChat: false, quiet: true, url: this.settings.url, created: Date.now(), attempts: 0, nextAt: 0 });
            }
          }
          const stored = this.stored(raw);
          if (stored) state.lastUser = stored.slice(0, LAST_USER_CHARS);
          if (this.settings.captureEvery <= 0) continue;
          // No time when the transcript gives none: the same words then
          // make the same batch every time.
          const time = turn.user.time ?? null;
          const assistant = turn.assistant?.text ? { text: turn.assistant.text, time: turn.assistant.time ?? time } : null;
          recordTurn(this.settings, capture, { users: stored ? [{ text: stored, time }] : [], assistant, turn: number, completed: assistant !== null }, { label: this.label });
          const jobs = cutBatch(this.settings, capture, sessionId, { untimed: true });
          if (jobs.length > 0) count += 1;
          // Said after this time (withForgetsAgain): not a wire field.
          for (const job of jobs) job.saidAfter = boundOf(index);
          out.push(...jobs);
          // Every turn read here is cut at once, so its batch is the last one.
          if (jobs.length > 0) out.push(...own);
        }
        const last = matches.reduce((most, at) => Math.max(most, at), -1);
        state.seen = (seen ?? []).slice(last + 1);
        // Queued for good first (an error leaves the place, so the next call
        // reads these turns again under the same ids); then the place moves.
        if (out.length > 0) this.queue.enqueue(this.withForgetsAgain(out));
        state.transcript = { ...next, capture };
        return count;
      });
      for (const { statement, at } of asked) {
        try {
          await noteForgotten(this.env, owner, statement, at);
        } catch (error) {
          this.log.warnOnce(`forgotten-write:${errKind(error)}`, `could not keep a forget to apply again (${errKind(error)})`);
        }
      }
      if (asked.length > 0) {
        try {
          await this.withForgets((items) => {
            for (const { requestId } of asked) if (!items.some((item) => item.requestId === requestId)) items.push({ requestId, session: sessionId, at: Date.now() });
          });
        } catch {
          // without the entry, a forget that fails later is only logged
        }
      }
      this.flush();
      return queued;
    } catch (error) {
      this.log.warnOnce(`catch-up:${errKind(error)}`, `capture from the transcript skipped (${errKind(error)})`);
      return 0;
    }
  }

  /**
   * A conversation began or was reopened (Cursor's sessionStart): what its
   * transcript (`file`) holds now was said before now, what is added later
   * after. A forget asked before now is then not applied again to a turn
   * added later, even one whose transcript time cannot be read; it is to a
   * turn already there.
   */
  async begin({ session, file = null, fresh = false }) {
    try {
      await this.update(this.sessionId(session), (state) => {
        if (state.transcript || Number(state.prompts) > 0) return;
        // What the transcript holds already was said before now: only a
        // turn after its last turn takes this time as its bound (the same
        // place catchUp keeps, so a turn_ended line Cursor moves on the next
        // rewrite does not count). No file: a new chat only where a start
        // means one (`fresh`: Cursor's CLI); elsewhere nothing is known and
        // no bound is kept.
        const read = file ? readConversation(file, { tail: true }) : null;
        let offset = null;
        if (read?.turns) offset = read.turns.at(-1)?.end ?? 0;
        else if (fresh && (!file || !fs.existsSync(file))) offset = 0;
        state.transcript = { begun: Date.now(), beginPath: file ? path.resolve(file) : null, ...(Number.isInteger(offset) ? { beginOffset: offset } : {}) };
      });
    } catch (error) {
      this.log.warnOnce(`begin:${errKind(error)}`, `could not note the conversation's start (${errKind(error)})`);
    }
  }

  /** The session ended: queue what is left and start the flusher. */
  async end({ session }) {
    try {
      const sessionId = this.sessionId(session);
      await this.update(sessionId, (state) => { this.enqueue(this.closeSession(state)); });
      this.flush();
    } catch (error) {
      this.log.warnOnce(`end:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    }
  }

  // Close the open turn and cut what is pending. The session state stays,
  // so a resumed session numbers its turns on and keeps its recall memory.
  closeSession(state) {
    const out = this.commitAll(state);
    out.push(...this.cutEarly(state));
    state.ended = Date.now();
    return out;
  }

  /**
   * end() for a host process that is exiting: synchronous, so it runs in an
   * `exit` handler. A session another process holds is left; the next start
   * queues it after the idle time.
   * @returns {number} jobs queued.
   */
  endSync({ session }) {
    const sessionId = this.sessionId(session);
    const file = this.fileOf(sessionId);
    let handle = null;
    try {
      handle = tryLock(`${file}.lock`, { staleMs: SESSION_LOCK_STALE_MS });
      if (!handle) return 0;
      const state = readJson(file);
      if (!state) return 0;
      const jobs = this.closeSession(state);
      this.enqueue(jobs);
      state.updated = Date.now();
      writeFileAtomic(file, JSON.stringify(state));
      return jobs.length;
    } catch (error) {
      this.log.warnOnce(`end:${errKind(error)}`, `capture skipped (${errKind(error)})`);
      return 0;
    } finally {
      releaseLock(handle);
    }
  }

  /**
   * Queue the pending turns of sessions idle for `idleFlushMs`, and close
   * and remove session files untouched for a week. With `idleFlushMs` off,
   * only the week-old sessions are closed. Old files of the earlier state
   * layout are removed too (DiskQueue.cleanLegacy, queueSessions).
   * @returns {Promise<number>} sessions whose turns were queued.
   */
  async sweep() {
    this.queue.cleanLegacy();
    const swept = await this.queueSessions({ all: false });
    if (swept > 0) this.flush();
    return swept;
  }

  /**
   * A session started (the host's session-start hook). The unsent turns of
   * this user's other sessions are queued now, not after the idle sweep's
   * time: a chat closed without a session end, or a switch, is then found by
   * the new chat's recall. Which ones:
   * - on a host whose chats can close without a session end (reliableEnd
   *   false: a Cursor tab), every chat of this host with no turn in flight,
   *   at once, and the others idle for OTHER_IDLE_MS;
   * - elsewhere, the chats idle for OTHER_IDLE_MS: one in use keeps its
   *   batch, and a closed one is sent by its own session end.
   * The new session's first prompt, when it asks about memory, waits briefly
   * for them (FIRST_PROMPT_WAIT_MS), and its prompts of the first
   * NEW_SESSION_MS send the chats idle by then. Within a session, turns
   * still go in batches of `captureEvery`.
   * @param {{session?: string}} input - the new session, when the host names it.
   * @returns {Promise<number>} sessions whose turns were queued.
   */
  async sessionStarted({ session = null } = {}) {
    try {
      this.queue.cleanLegacy();
      const sessionId = session ? this.sessionId(session) : null;
      // Recorded first: the first prompt waits for what is queued from now.
      if (sessionId) {
        await this.update(sessionId, (state) => {
          if (!Number.isFinite(state.startedAt)) state.startedAt = Date.now();
        });
      }
      const sent = await this.queueSessions({ all: true, except: sessionId, minIdleMs: OTHER_IDLE_MS, betweenTurns: !this.reliableEnd });
      this.flush();
      return sent;
    } catch (error) {
      this.log.warnOnce(`start:${errKind(error)}`, `could not queue the other sessions' turns (${errKind(error)})`);
      this.flush();
      return 0;
    }
  }

  // The queue keys of the batches queued for a new session (forSession: by
  // its start or by its first prompt, whichever came first) since `since`,
  // not yet stored. Other sessions' own batches are not waited for.
  queuedFor(sessionId, since) {
    const keys = [];
    for (const entry of this.queue.entries()) {
      if (Number(entry.name.slice(0, 15)) < since) continue;
      const job = readJson(entry.file);
      if (job && typeof job === 'object' && job.kind !== 'forget' && job.forSession === sessionId) keys.push(entry.key);
    }
    return keys;
  }

  // Wait, until `until`, while batches queued for this new session are not
  // stored yet: each look reads the queue again, so one its start queued
  // meanwhile is waited for too. Stops when no background sender runs to
  // send them; starts one when they appear and none was started (`flushed`).
  async waitQueued(sessionId, since, until, flushed) {
    const started = Date.now();
    while (Date.now() < until && this.queuedFor(sessionId, since).length > 0) {
      if (!flushed) {
        this.flush();
        flushed = true;
      }
      if (Date.now() - started > 1000 && !this.queue.flusherRunning()) break;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }

  /**
   * Queue the pending turns of this user's sessions in the state folder:
   * every session (`all`, before a forget or in a new session) or the idle
   * ones. Session files abandoned for a week are closed and removed. A
   * session another process holds is skipped, and so is every session of
   * another user, or of the earlier layout (no user named), which is only
   * removed once abandoned. `minIdleMs` (with `all`): only the sessions idle
   * this long; with `betweenTurns`, also this host's sessions with no turn
   * in flight, at once, their held turns saved too. `lockUntil`: wait for a
   * session's lock until then at most.
   * @returns {Promise<number>} sessions whose turns were queued.
   */
  async queueSessions({ all, except = null, minIdleMs = 0, betweenTurns = false, lockUntil = null }) {
    let names = [];
    try {
      names = fs.readdirSync(this.sessionsDir).filter((name) => /^[0-9a-f]{16}\.json$/.test(name));
    } catch {
      return 0;
    }
    let queued = 0;
    // With `lockUntil`, a session whose lock another process holds is looked
    // at again after the others, whose batches are then sent first, and its
    // lock waited for until then.
    const order = names.map((name) => ({ name, again: false }));
    let flushed = false;
    for (const { name, again } of order) {
      if (again && !flushed) {
        if (queued > 0) this.flush();
        flushed = true;
      }
      const file = path.join(this.sessionsDir, name);
      let mtime = 0;
      try {
        mtime = fs.statSync(file).mtimeMs;
      } catch {
        continue;
      }
      const idle = Date.now() - mtime;
      const idleBound = this.settings.idleFlushMs > 0 ? this.settings.idleFlushMs : SESSION_KEEP_MS;
      if (!all && idle < Math.min(idleBound, HELD_TURN_MS)) continue;
      // A new session sends only the sessions idle this long (minIdleMs): a
      // chat still in use keeps its batch of `captureEvery` turns. On a host
      // whose chats close without an end (betweenTurns), one of its chats
      // between turns goes at once: it may have been closed.
      const recent = all && idle < minIdleMs;
      if (recent && !betweenTurns) continue;
      const data = readJson(file);
      const ended = all && betweenTurns && Boolean(data) && typeof data === 'object' && String(data.sessionId ?? '').startsWith(`${this.host}:`) && !inFlight(data);
      if (recent && !ended) continue;
      // A turn held open for its reply is saved after HELD_TURN_MS, one still
      // running beside the open one after RUNNING_TURN_MS (commitDue).
      const heldDue = data && typeof data === 'object' && (turnDue(data.open, false) || runningOf(data).some((t) => turnDue(t, true)));
      if (!all && idle < idleBound && !heldDue) continue;
      const abandoned = idle >= SESSION_KEEP_MS;
      // A session of the earlier layout names no user: it is never sent
      // (it may be another user's), and is removed once abandoned.
      if (data && typeof data === 'object' && !('user' in data)) {
        if (abandoned) {
          try { fs.unlinkSync(file); } catch { /* gone */ }
        }
        continue;
      }
      // Another user's session is never sent as this one.
      if (!data?.sessionId || data.user !== this.user || data.sessionId === except) continue;
      // An open turn may still be running (a long agent turn): it is closed
      // only when the session file is abandoned, or when it is held (at once
      // in a chat sent between turns).
      const closable = ended && (Boolean(data.open) || runningOf(data).length > 0);
      if (!abandoned && !heldDue && !closable && (data.capture?.pending?.length ?? 0) === 0) continue;
      try {
        const jobs = await this.update(data.sessionId, (state) => {
          // The chat may have begun a turn since it was read: then it is
          // treated as in use.
          const between = ended && !inFlight(state);
          if (recent && !between) return [];
          const out = abandoned || between ? this.commitAll(state) : this.commitDue(state);
          out.push(...this.cutEarly(state));
          // Queued for that session (a new one, or one that forgets): its
          // first prompt waits for these (queuedFor).
          if (except) for (const job of out) job.forSession = except;
          if (abandoned) state.remove = true;
          this.enqueue(out);
          return out;
        }, lockUntil === null ? {} : { lockMs: again ? Math.max(0, lockUntil - Date.now()) : 0 });
        if (jobs.length > 0) queued += 1;
      } catch {
        // another process holds it; a later sweep gets it
        if (lockUntil !== null && !again) order.push({ name, again: true });
      }
    }
    return queued;
  }
}

/**
 * Queue this owner's pending turns in every state folder of this computer:
 * QUOTEMEM_STATE_DIR (or the default) and each folder in ~/.quotemem that
 * holds sessions (one per host). A forget the model made through the MCP
 * tools does this, as a prompt hook's forget does in its own folder: words
 * said before the forget are queued now, while the forget is kept, and get
 * it again behind them. Another user's sessions are never touched. Never
 * throws.
 * @returns {Promise<number>} sessions whose turns were queued.
 */
export async function queueOwnerSessions(env) {
  const quiet = { warn() {}, warnOnce() {}, debug() {} };
  try {
    const settings = resolveSettings({}, env, {});
    if (settings.problem) return 0;
    // QUOTEMEM_STATE_DIR (or the default) and every host folder in
    // ~/.quotemem: hooks without the MCP server's QUOTEMEM_STATE_DIR keep
    // their sessions there. Only this owner's sessions are sent.
    const dirs = new Set([settings.stateDir]);
    const root = path.join(homeDir(env), '.quotemem');
    let names = [];
    try { names = fs.readdirSync(root); } catch { /* none */ }
    for (const name of names) if (fs.existsSync(path.join(root, name, 'sessions'))) dirs.add(path.join(root, name));
    let queued = 0;
    for (const stateDir of dirs) {
      if (!fs.existsSync(path.join(stateDir, 'sessions'))) continue;
      const memory = new HookMemory({ settings: { ...settings, stateDir }, log: quiet, host: 'mcp', env });
      const count = await memory.queueSessions({ all: true });
      if (count > 0) memory.flush();
      queued += count;
    }
    return queued;
  } catch {
    return 0;
  }
}

/**
 * The session files of a state folder used in the last SESSION_KEEP_MS,
 * read once: [{data, mtime}] (each names its owner, `data.user`).
 */
function readSessions(dir) {
  let names = [];
  try {
    names = fs.readdirSync(dir).filter((name) => /^[0-9a-f]{16}\.json$/.test(name));
  } catch {
    return [];
  }
  const out = [];
  for (const name of names) {
    const file = path.join(dir, name);
    let mtime = 0;
    try {
      mtime = fs.statSync(file).mtimeMs;
    } catch {
      continue;
    }
    if (Date.now() - mtime >= SESSION_KEEP_MS) continue;
    const data = readJson(file);
    if (data && typeof data === 'object' && typeof data.user === 'string' && data.sessionId) out.push({ data, mtime });
  }
  return out;
}

// Wait `ms` (at most IDLE_TICK_MS), or until `signal` aborts.
function idleSleep(ms, signal) {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const done = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', done);
      resolve();
    };
    const timer = setTimeout(done, Math.max(0, Math.min(ms, IDLE_TICK_MS)));
    signal?.addEventListener('abort', done, { once: true });
  });
}

/**
 * The background flusher of a hook host (settings.idleWait): it sends the
 * queue (runFlusher), then keeps the idle rule. One waits for all the owners
 * of one service in a state folder (each OpenClaw channel sender is one):
 * the one that holds the idle lock. While their sessions hold turns the idle
 * sweep would queue (HookMemory.idleDueAt), it sleeps until the oldest is
 * due, by the wall clock, then queues them exactly as a later hook's sweep()
 * would (a chat whose turn is in flight keeps that turn; another user's or
 * service's sessions are not touched; kept forgets go again behind the
 * batches) and sends them. A turn written meanwhile moves the time: the time
 * is read again at every wake. It exits when nothing waits, when the idle
 * time or capture is off, when its lock is taken over (a sleep past
 * IDLE_LOCK_STALE_MS), or once it has lived IDLE_EXTRA_LIFE_MS past the idle
 * time; then chats still in use are handed to a new flusher.
 * @param {object} settings - the flusher's settings (with the token).
 * @param {object} log - from makeLog.
 * @param {{signal?: AbortSignal, env?: object}} options - `env`: the
 *   environment the hook gave the flusher (kept forgets, a successor).
 * @returns {Promise<string>} the outcome, for the log.
 */
export async function runIdleWaiter(settings, log, { signal, env = process.env } = {}) {
  const started = Date.now();
  const memory = new HookMemory({ settings, log, host: 'idle', env, idleWait: true });
  // This process sends what it queues.
  memory.flush = () => false;
  const others = new Map();
  // The flusher's credential tells which owners it may serve.
  const { credential } = memory.queue;
  if (!settings.token && settings.localService) settings.token = serviceToken(env, settings.localService);
  // The owners whose idle chats this flusher keeps, with the sessions of the
  // state folder read once: its own, and every owner of the same service and
  // credential registered in this state folder (owner.json) with the idle
  // rule on and a session in use, each with its own settings. An owner
  // registered long ago with no session left is unregistered.
  const owners = () => {
    const sessions = readSessions(memory.sessionsDir);
    const users = new Set(sessions.map(({ data }) => data.user));
    const list = [memory];
    const dir = path.join(settings.stateDir, 'owners');
    let names = [];
    try { names = fs.readdirSync(dir); } catch { /* none yet */ }
    for (const name of names) {
      if (name === memory.queue.owner || !/^[0-9a-f]{16}$/.test(name)) continue;
      const file = path.join(dir, name, 'owner.json');
      if (!users.has(name)) {
        try {
          // Unregistered only when untouched for a week and still without a session
          // (its hook may have registered and written one since the sessions were read).
          if (Date.now() - fs.statSync(file).mtimeMs >= SESSION_KEEP_MS
            && !readSessions(memory.sessionsDir).some(({ data }) => data.user === name)
            && Date.now() - fs.statSync(file).mtimeMs >= SESSION_KEEP_MS) fs.rmSync(file, { force: true });
        } catch {
          // none
        }
        continue;
      }
      const data = readJson(file);
      const own = data?.settings;
      if (!data || data.version !== 2 || data.url !== settings.url || typeof data.userId !== 'string' || ownerKey(data.url, data.userId) !== name) continue;
      if (data.credential !== credential || !own || typeof own !== 'object' || own.url !== settings.url || own.userId !== data.userId || own.stateDir !== settings.stateDir) continue;
      if (!(own.idleFlushMs > 0) || !(own.captureEvery > 0)) continue;
      const text = JSON.stringify(own);
      let other = others.get(name);
      if (!other || other.record !== text) {
        other = new HookMemory({ settings: { ...own, token: settings.token }, log, host: 'idle', env });
        other.flush = () => false;
        other.record = text;
        others.set(name, other);
      }
      list.push(other);
    }
    return { list, sessions };
  };
  const dueOf = ({ list, sessions }) => {
    let next = null;
    for (const one of list) {
      const due = one.idleDueAt(sessions);
      if (due !== null && (next === null || due < next)) next = due;
    }
    return next;
  };
  const on = settings.idleFlushMs > 0 && settings.captureEvery > 0;
  // The lock the hook that started this process took for it, else a free one.
  let handle = adoptLock(memory.queue.idleLockFile, env.QUOTEMEM_IDLE_LOCK);
  if (!on) {
    releaseLock(handle);
    handle = null;
  } else {
    handle ??= tryLock(memory.queue.idleLockFile, { staleMs: IDLE_LOCK_STALE_MS });
  }
  // The lock stays fresh while the queue is sent (which may take minutes).
  const send = async (list = [memory]) => {
    const beat = handle ? setInterval(() => { if (handle) touchLock(handle); }, IDLE_PASS_MS) : null;
    beat?.unref?.();
    try {
      let outcome = '';
      for (const one of list) {
        if (signal?.aborted) break;
        outcome = await runFlusher(one.settings, log, { signal });
      }
      return outcome;
    } finally {
      clearInterval(beat);
    }
  };
  let outcome = await send();
  if (!handle) return outcome;
  // QUOTEMEM_TEST_IDLE_LIFE_MS (tests) shortens the life past the idle time.
  const extra = Number(env.QUOTEMEM_TEST_IDLE_LIFE_MS);
  const deadline = started + settings.idleFlushMs + (Number.isFinite(extra) && extra >= 0 ? extra : IDLE_EXTRA_LIFE_MS);
  let lastPass = 0;
  try {
    for (;;) {
      if (signal?.aborted) return 'stopped';
      if (!touchLock(handle)) {
        handle = null;
        log.warnOnce('idle-lock-lost', 'the background flusher that waits for idle chats lost its lock to another one and stopped');
        return 'idle-busy';
      }
      const now = Date.now();
      let due = dueOf(owners());
      if (due === null) {
        // Release, then look again: a hook that saw the lock held started
        // no flusher, and its turns were written before this release.
        releaseLock(handle);
        handle = null;
        if (dueOf(owners()) === null) return `${outcome}, no idle chat left`;
        handle = tryLock(memory.queue.idleLockFile, { staleMs: IDLE_LOCK_STALE_MS });
        if (!handle) return 'idle-busy';
        continue;
      }
      // A chat that was due at the last sweep and still waits (another
      // process held it) is looked at again IDLE_PASS_MS after it.
      const retry = due <= lastPass && now - lastPass < IDLE_PASS_MS;
      if (retry) due = lastPass + IDLE_PASS_MS;
      if (due > now) {
        if (now >= deadline) {
          releaseLock(handle);
          handle = null;
          // Chats still in use go to a new flusher; one a sweep keeps
          // failing on is left to the next hook.
          if (retry) return 'idle life over';
          memory.queue.startFlusher({ env, sync: true, idle: 'pending' });
          return 'idle life over, handed over';
        }
        await idleSleep(Math.min(due, deadline) - now, signal);
        continue;
      }
      lastPass = Date.now();
      const queued = [];
      let swept = 0;
      const seen = owners();
      for (const one of seen.list) {
        const at = one.idleDueAt(seen.sessions);
        if (at === null || at > lastPass) continue;
        const count = await one.sweep();
        if (count > 0) queued.push(one);
        swept += count;
      }
      if (swept > 0) {
        log.debug(`idle: queued the turns of ${swept} chat(s)`);
        outcome = await send(queued);
      }
    }
  } finally {
    releaseLock(handle);
  }
}
