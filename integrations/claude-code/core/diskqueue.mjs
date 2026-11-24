// The cross-process capture queue, for hosts whose plugin code runs in
// short-lived hook processes (Claude Code, Codex, Gemini CLI, …).
//
// Every job belongs to one owner: one user (QUOTEMEM_USER_ID) of one service
// (QUOTEMEM_URL). Several users may share a state folder (one host profile,
// several QUOTEMEM_USER_ID values), so everything but the job files is kept
// per owner, and a flusher sends, and a hook reads, only its own owner's.
//
// Layout under the state folder (0700; every file 0600):
//   queue/<stamp>-<pid>-<seq>-<owner>-<key>.json   one job per file, FIFO by name;
//                                                  key = sha16([owner, request_id])
//   owners/<owner>/state.json                      {pausedUntil} after a refused token, {storedAt},
//                                                  {cappedUntil} the spend cap's last reset
//   owners/<owner>/flusher.lock                    the one running flusher of this owner
//   owners/<owner>/idle.lock                       the one flusher that waits for idle chats
//   owners/<owner>/results/<sha16(request_id)>.json   outcome of a forget job
//   owners/<owner>/chat-forgets.json               what the service's in-chat forget did while
//                                                  storing a batch: counts and a hash of a request's
//                                                  words (requestKey), never the words
//   owners/<owner>/contact.json                    {at}: when a hook last reached the service
//   owners/<owner>/probe.json                      {at}: the flusher's last /health probe
//   quotemem.log                                   warnings (no content, no token)
//
// A hook only writes new job files (atomically) and starts the flusher; it
// never waits for the service. The flusher is a detached Node process: it
// takes its owner's flusher.lock, sends that owner's jobs one at a time in
// order, and exits when they are done or the next retry is far off. Only the
// flusher changes or removes job files. A /memory/add batch asks for an early
// acknowledgement: a batch the service saved (202) stays queued, marked
// `accepted`, and is read again with the identical request (backing off from
// 1 s to 3 s, or as the service's Retry-After says) until the service answers
// it stored, while the batches behind it are sent. A batch is retried with
// backoff (honouring Retry-After) and dropped after the attempt or age limit;
// a batch or forget the service refuses as busy (`429 delivery_busy`: for
// example it holds a session's new batch while an earlier one completes)
// waits without using its attempts, for at most `captureBusyMaxWaitMs`, and
// the jobs behind it wait for it; a forget job runs prepare and confirm, and
// writes its outcome for the hook that waits on it. One flusher sends in
// order, so every batch queued before a forget has been sent before it runs.
// A forget the service's in-chat check also reads (`inChat`) waits for those
// batches to be stored for at most FORGET_EARLIER_WAIT_MS, then runs: the
// service forgets the request again from a batch stored later, when the
// request's own turn is stored, and the plugin reports that from the batch's
// answer. Any other forget (a request only the plugin reads, such as a
// retraction) waits until they are stored, since nothing would cover a batch
// stored after it. A job whose last attempt could not reach the service
// (`unreachableAt`) is due at once when a hook has reached the service since
// (contact.json): after an outage, the queue does not wait out a retry delay
// of minutes. While such a job waits, the flusher also asks `GET /health`
// itself, at most every PROBE_EVERY_MS, and a 2xx counts as that contact; a
// hook starts a flusher for it. A probe uses no attempt, and a job the
// service held back (busy, its spend cap) is never probed for.
//
// A hook host's flusher also keeps the idle rule (`idleWait`): while this
// owner's chats hold unsent turns, one flusher per owner takes idle.lock and
// stays until the oldest chat has been idle for QUOTEMEM_IDLE_FLUSH_MS, then
// queues those turns as a later hook's sweep would and sends them (see
// runIdleWaiter in hooks.mjs). It does not hold flusher.lock while it
// waits: a hook still starts a sender at once for a new batch or forget.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ownerKey, sha16 } from './capture.mjs';
import { CAPTURE_BUSY_MAX_WAIT_MS, publicSettings } from './config.mjs';
import { carryOnDisk, effectCounts, requestKey, runForget } from './forget.mjs';
import { errKind, healthOk } from './http.mjs';
import { handLock, lockHeld, readJson, releaseLock, touchLock, tryLock, withLock, writeFileAtomic } from './lock.mjs';
import { DIRECT_START_FILE, spawnDetached, spawnDetachedAsync } from './proc.mjs';
import { AUTH_PAUSE_MS, backoffMs, pollDelayMs, rotateJob, sendAdd, validJob } from './send.mjs';

export const FLUSHER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'flusher.mjs');
const JOB_FILE = /^(\d{15})-(\d+)-(\d{6})-([0-9a-f]{16})-([0-9a-f]{16})\.json$/;
// A job file of the earlier layout, whose name names no owner.
const LEGACY_JOB_FILE = /^\d{15}-\d+-\d{6}-[0-9a-f]{16}\.json$/;
// How often one queue object looks for files of the earlier layout.
const LEGACY_CHECK_MS = 3_600_000;
const FLUSHER_STALE_MS = 60_000;
// The waiting flusher refreshes idle.lock at every wake (at most a minute
// apart): a lock older than this is taken as left by a crash or a sleep.
export const IDLE_LOCK_STALE_MS = 180_000;
const HEARTBEAT_MS = 10_000;
const RESULT_KEEP_MS = 86_400_000;
// How long past its forget deadline an outcome may keep memory text for the
// hook that waits on it (a hook's clock and the flusher's run together).
const RESULT_QUOTE_GRACE_MS = 2000;
const MAX_LIFETIME_MS = 3_600_000;
// Reading an acknowledged batch again should answer at once (202 or 200);
// an answer slower than this is taken as "still pending", so one slow read
// does not hold the queue, and the read is sent again later.
export const ACCEPTED_READ_TIMEOUT_MS = 15_000;
// How long a saved batch may stay unconfirmed before the queue lets it go.
export const ACCEPTED_MAX_WAIT_MS = 1_800_000;
// Plans a forget may prepare in conflict (the scope kept changing under it)
// before it is given up; each prepare asks the service's model.
export const FORGET_CONFLICT_ATTEMPTS = 3;
// How long a forget the service's in-chat check also reads waits for the
// batches queued before it to be stored (it runs at once when they are). A
// batch the service completes later is checked for the request by the service
// itself when the request's own turn is stored. A forget only the plugin reads
// waits until they are stored.
export const FORGET_EARLIER_WAIT_MS = 2000;
// How long an explicit forget tool (prepare) waits for this owner's queued
// batches to be stored before it is refused as "still being saved". The
// service's in-chat check may not read the user's words, so a forget that
// ran before an earlier batch is stored could miss what it holds for good.
// About the longest a batch took from saved (202) to stored in live runs.
export const FORGET_TOOL_WAIT_MS = 20_000;
const MAX_CHAT_FORGETS = 20;
// How often the flusher may ask GET /health whether a service it could not
// reach answers again, and how long it waits for the answer.
export const PROBE_EVERY_MS = 30_000;
const PROBE_TIMEOUT_MS = 3000;

let lastStamp = 0;
let seq = 0;

const sleep = (ms, signal) => new Promise((resolve) => {
  const timer = setTimeout(done, Math.max(0, ms));
  function done() {
    clearTimeout(timer);
    signal?.removeEventListener('abort', done);
    resolve();
  }
  signal?.addEventListener('abort', done, { once: true });
});

export { ownerKey };

/** The file name key of a request id of one owner. */
export function jobKey(owner, requestId) {
  return sha16([owner, requestId]);
}

// The owner of a queued job: the service it was captured for and the user its body names.
function ownerOfJob(job, url) {
  return ownerKey(typeof job?.url === 'string' ? job.url : url, job?.body?.user_id);
}

/**
 * The queue of one state folder, as one owner (the settings' user at the
 * settings' service) sees it. All methods are synchronous, small and never
 * throw on a missing folder.
 */
export class DiskQueue {
  constructor(settings, log) {
    this.settings = settings;
    this.log = log;
    this.owner = ownerKey(settings.url, settings.userId);
    this.dir = path.join(settings.stateDir, 'queue');
    this.ownerDir = path.join(settings.stateDir, 'owners', this.owner);
    this.lockFile = path.join(this.ownerDir, 'flusher.lock');
    this.idleLockFile = path.join(this.ownerDir, 'idle.lock');
    this.resultsDir = path.join(this.ownerDir, 'results');
    this.chatForgetsFile = path.join(this.ownerDir, 'chat-forgets.json');
    this.contactFile = path.join(this.ownerDir, 'contact.json');
    this.probeFile = path.join(this.ownerDir, 'probe.json');
  }

  /** When the flusher last probed /health, in ms; 0 when never. */
  probedAt() {
    return Number(readJson(this.probeFile)?.at) || 0;
  }

  noteProbe() {
    try {
      writeFileAtomic(this.probeFile, JSON.stringify({ at: Date.now() }));
    } catch (error) {
      this.log.warnOnce(`probe-write:${errKind(error)}`, `could not note a health probe (${errKind(error)})`);
    }
  }

  /** Whether a job that could not reach the service may have the flusher probe for it now. */
  mayProbe(job, paused) {
    const now = Date.now();
    return Number(job?.unreachableAt) > 0 && paused <= now && now - this.probedAt() >= PROBE_EVERY_MS;
  }

  /** A hook reached the service (a search answered): jobs waiting out an outage are due now. */
  noteContact() {
    try {
      writeFileAtomic(this.contactFile, JSON.stringify({ at: Date.now() }));
    } catch (error) {
      this.log.warnOnce(`contact-write:${errKind(error)}`, `could not note that the service answered (${errKind(error)})`);
    }
  }

  /** When a hook last reached the service, in ms; 0 when never. */
  contactAt() {
    return Number(readJson(this.contactFile)?.at) || 0;
  }

  /**
   * Jobs in queue order: [{name, file, key, owner}]: this owner's, or with
   * `every` those of every owner.
   */
  entries({ every = false } = {}) {
    let names = [];
    try {
      names = fs.readdirSync(this.dir);
    } catch {
      return [];
    }
    return names.filter((name) => JOB_FILE.test(name)).sort()
      .map((name) => {
        const match = JOB_FILE.exec(name);
        return { name, file: path.join(this.dir, name), owner: match[4], key: match[5] };
      })
      .filter((entry) => every || entry.owner === this.owner);
  }

  /** Whether a job of this owner with this request id is queued. */
  has(requestId) {
    const key = jobKey(this.owner, requestId);
    return this.entries().some((entry) => entry.key === key);
  }

  /**
   * Whether a forget of this owner queued at or after `since` (ms) is still
   * queued with no outcome. Reads only the job files named as queued since
   * then; a file that cannot be read is no pending forget.
   */
  forgetPending(since) {
    for (const entry of this.entries()) {
      if (Number(JOB_FILE.exec(entry.name)[1]) < since) continue;
      const job = readJson(entry.file);
      if (job?.kind === 'forget' && typeof job.request_id === 'string' && !this.result(job.request_id)) return true;
    }
    return false;
  }

  /**
   * Append jobs in order; a job whose request id is already queued is skipped.
   * @returns {number} jobs written.
   */
  enqueue(jobs) {
    const queued = new Set(this.entries({ every: true }).map((entry) => `${entry.owner}-${entry.key}`));
    let written = 0;
    for (const job of jobs) {
      const owner = ownerOfJob(job, this.settings.url);
      const key = jobKey(owner, job.request_id);
      if (queued.has(`${owner}-${key}`)) continue;
      const stamp = Math.max(Date.now(), lastStamp);
      lastStamp = stamp;
      seq = (seq + 1) % 1_000_000;
      const name = `${String(stamp).padStart(15, '0')}-${process.pid}-${String(seq).padStart(6, '0')}-${owner}-${key}.json`;
      writeFileAtomic(path.join(this.dir, name), JSON.stringify({ version: 1, ...job, key }));
      queued.add(`${owner}-${key}`);
      written += 1;
    }
    return written;
  }

  resultFile(requestId) {
    return path.join(this.resultsDir, `${sha16(requestId)}.json`);
  }

  /** The outcome of this owner's finished forget job, or null. */
  result(requestId) {
    return readJson(this.resultFile(requestId));
  }

  /**
   * A hook has given the note for this outcome: the result keeps only its
   * status and counts, no memory text (`closest`, `matched`, `bundled`,
   * `excluded`), until it ages out.
   */
  markResultRead(requestId) {
    const file = this.resultFile(requestId);
    const result = readJson(file);
    if (!result || result.read) return;
    try {
      writeFileAtomic(file, JSON.stringify(countsOf(result)));
    } catch (error) {
      this.log.warnOnce(`result-write:${errKind(error)}`, `could not write a forget result (${errKind(error)})`);
    }
  }

  writeResult(requestId, result) {
    try {
      writeFileAtomic(this.resultFile(requestId), JSON.stringify({ ...result, at: Date.now() }));
    } catch (error) {
      this.log.warnOnce(`result-write:${errKind(error)}`, `could not write a forget result (${errKind(error)})`);
    }
  }

  state() {
    return readJson(path.join(this.ownerDir, 'state.json')) ?? {};
  }

  setState(state) {
    try {
      writeFileAtomic(path.join(this.ownerDir, 'state.json'), JSON.stringify(state));
    } catch (error) {
      this.log.warnOnce(`queue-write:${errKind(error)}`, `could not write the capture queue (${errKind(error)})`);
    }
  }

  /** Whether a live flusher holds the lock. */
  flusherRunning() {
    return lockHeld(this.lockFile, { staleMs: FLUSHER_STALE_MS });
  }

  /** Whether a live flusher waits for this owner's idle chats (idle.lock). */
  idleWaiting() {
    return lockHeld(this.idleLockFile, { staleMs: IDLE_LOCK_STALE_MS });
  }

  /**
   * Start the background flusher unless one is running. It runs detached
   * with no stdio and none of the hook's handles (spawnDetached), so it
   * outlives the hook process and never blocks it, nor a caller that reads
   * the agent tool's output through a pipe.
   * @param {{env?: object, execPath?: string}} options - the child's
   *   environment (default this process's; the settings and the token are
   *   added to it), and the JavaScript runtime that runs it (default this
   *   process's). A host compiled into one executable (OpenCode is a Bun
   *   binary) must pass a real `node` or `bun`: its own execPath would start
   *   the host itself. `sync`: start it before returning (an exit handler,
   *   which cannot wait); else it starts while the caller goes on.
   *   `idle`: the flusher keeps the idle rule (runIdleWaiter), and one is
   *   started when chats of this owner hold unsent turns (`idle` 'pending')
   *   and no flusher waits for them yet, even with nothing to send.
   *   `launch` (tests): spawnDetached's `platform` and `powershell`.
   * @returns {boolean} whether a flusher was started.
   */
  startFlusher({ env = process.env, execPath = process.execPath, sync = false, idle = false, launch = {} } = {}) {
    try {
      // One start at a time; an exit handler starts one anyway (the other dies with the process).
      if (this.launching && !sync) return false;
      // Chats wait for their idle time and no flusher waits for them: one
      // starts, whatever the queue holds (it sends that first). This process
      // takes idle.lock for it first and hands it over (handLock, adoptLock),
      // so two hooks at once start one.
      const handle = idle === 'pending' && this.settings.idleFlushMs > 0 && this.settings.captureEvery > 0
        ? tryLock(this.idleLockFile, { staleMs: IDLE_LOCK_STALE_MS }) : null;
      if (!handle) {
        const entries = this.entries();
        if (entries.length === 0 || this.flusherRunning()) return false;
        // Nothing is due soon (a paused queue, a far retry): a flusher would
        // exit at once, so none is started; a later hook tries again.
        const head = readJson(entries[0].file);
        const paused = Number(this.state().pausedUntil) || 0;
        const due = dueAt(head, paused, this.contactAt());
        // Unless the head could not reach the service and a /health probe is due.
        if (due - Date.now() > (this.settings.flusherMaxWaitMs ?? 120_000) && !this.mayProbe(head, paused)) return false;
      }
      // It starts while the hook goes on (on Windows PowerShell takes up to
      // half a second); the hook waits for it before it exits (settleLaunches).
      const options = {
        label: 'quotemem flusher',
        env: {
          ...env,
          QUOTEMEM_FLUSHER_SETTINGS: JSON.stringify({ ...publicSettings(this.settings), idleWait: Boolean(idle) }),
          QUOTEMEM_TOKEN: this.settings.token,
          QUOTEMEM_IDLE_LOCK: handle ? handle.nonce : '',
        },
        log: this.log,
        mark: path.join(this.settings.stateDir, DIRECT_START_FILE),
        ...launch,
      };
      const failed = (error) => this.log.warnOnce(`flusher-spawn:${errKind(error)}`, `could not start the background flusher (${errKind(error)})`);
      // The lock lives with the started process until it takes it over; a
      // start that failed leaves it to the next hook.
      const started = (child) => {
        if (handle && !(Number.isInteger(child?.pid) && handLock(handle, child.pid)) && !child?.started) releaseLock(handle);
        return child;
      };
      if (sync) {
        try {
          started(spawnDetached(execPath, [FLUSHER_PATH], options)).on('error', failed);
        } catch (error) {
          releaseLock(handle);
          throw error;
        }
        return true;
      }
      this.launching = spawnDetachedAsync(execPath, [FLUSHER_PATH], options)
        .then((child) => started(child).on('error', failed), (error) => { releaseLock(handle); failed(error); })
        .finally(() => { this.launching = null; });
      return true;
    } catch (error) {
      this.log.warnOnce(`flusher-spawn:${errKind(error)}`, `could not start the background flusher (${errKind(error)})`);
      return false;
    }
  }

  /**
   * Wait for a forget job's outcome.
   * @returns {Promise<object|null>} the outcome, or null when it is not done in time.
   */
  async waitResult(requestId, timeoutMs) {
    const end = Date.now() + timeoutMs;
    for (;;) {
      const result = this.result(requestId);
      if (result) return result;
      if (Date.now() >= end) return null;
      await sleep(Math.min(50, end - Date.now()));
    }
  }

  /**
   * Keep what the service's in-chat forget did while storing a batch, for
   * this owner's next prompt (chatForgetsOf: counts only). An outcome the
   * plugin's own forget of the same request also gave (`earlier`) is kept
   * only when it removed something and names its request's key:
   * takeChatForgets then tells what that forget did not.
   */
  async addChatForgets(items) {
    const fresh = items.filter((item) => !item.earlier || (item.key && removedBy(item) > 0))
      .map(({ earlier, key, ...item }) => ({ ...item, ...(earlier ? { earlier, key } : {}), at: Date.now() }));
    if (fresh.length === 0) return;
    try {
      await withLock(`${this.chatForgetsFile}.lock`, () => {
        const kept = readJson(this.chatForgetsFile);
        const list = [...(Array.isArray(kept?.items) ? kept.items : []), ...fresh].slice(-MAX_CHAT_FORGETS);
        writeFileAtomic(this.chatForgetsFile, JSON.stringify({ version: 1, items: list, told: Array.isArray(kept?.told) ? kept.told : [] }));
      }, { timeoutMs: 3000, staleMs: 15_000 });
    } catch (error) {
      this.log.warnOnce(`chat-forgets:${errKind(error)}`, `could not keep an in-chat forget outcome (${errKind(error)})`);
    }
  }

  /**
   * Take this owner's in-chat forget outcomes not yet reported: each is
   * returned once, and outcomes older than the capture age limit are dropped.
   *
   * One forget is counted once. An outcome of a request the plugin's own
   * forget by statement also handled (`earlier`) counts both paths' memories
   * and sentences, so only what it holds beyond that forget's counts (its
   * result, found by the request's key, plus what an earlier outcome of the
   * same request added; kept in `told`) is reported, as `more`: its
   * memories, sentences and memories forgotten whole (`bundled`), and the
   * other memories it affected that neither of those reported (by
   * memoryKey). The same words asked again later are another request: only
   * the latest forget of them that started before the outcome, and what was
   * told since it started, are taken from it. A forget that failed counts
   * nothing, so the outcome is told in full. While that forget still runs,
   * the outcome waits for a later prompt. One with no such forget was
   * handled by another path that told the user (the forget tool), and is
   * not reported.
   */
  async takeChatForgets() {
    if (!fs.existsSync(this.chatForgetsFile)) return [];
    return withLock(`${this.chatForgetsFile}.lock`, () => {
      const file = readJson(this.chatForgetsFile);
      const maxAge = this.settings.captureMaxAgeMs ?? 86_400_000;
      const live = (list) => (Array.isArray(list) ? list : []).filter((item) => item && Date.now() - (Number(item.at) || 0) < maxAge);
      const told = live(file?.told);
      const waiting = [];
      const out = [];
      for (const item of live(file?.items)) {
        if (!item.earlier) {
          out.push(item);
          continue;
        }
        const own = this.statementForgets(item.key, Number(item.at) || 0);
        if (own.queued) {
          waiting.push(item);
          continue;
        }
        if (own.results.length === 0) continue;
        const before = [...own.results, ...told.filter((entry) => entry.key === item.key && (Number(entry.at) || 0) >= own.since)];
        const memories = Math.max(0, item.forgotten - sumOf(before, 'memories'));
        const sentences = Math.max(0, item.withheld - sumOf(before, 'sentences'));
        const bundled = Math.min(memories, Math.max(0, (Number(item.bundled) || 0) - sumOf(before, 'whole')));
        const known = new Set(before.flatMap((entry) => (Array.isArray(entry.affectedKeys) ? entry.affectedKeys : []).map(([key]) => key)));
        const affected = (Array.isArray(item.affectedKeys) ? item.affectedKeys : []).filter(([key]) => key === null || !known.has(key));
        const others = effectCounts(affected.map(([, effect]) => effect));
        if (memories + sentences + affected.length === 0) continue;
        told.push({ key: item.key, memories, sentences, whole: bundled, affectedKeys: affected, at: item.at });
        out.push({ status: 'forgotten', forgotten: memories, withheld: sentences, bundled, ...others, more: true, at: item.at });
      }
      if (waiting.length === 0 && told.length === 0) {
        try { fs.unlinkSync(this.chatForgetsFile); } catch { /* gone */ }
      } else {
        writeFileAtomic(this.chatForgetsFile, JSON.stringify({ version: 1, items: waiting, told: told.slice(-MAX_CHAT_FORGETS) }));
      }
      return out;
    }, { timeoutMs: 3000, staleMs: 15_000 });
  }

  /**
   * This owner's forgets by statement of a request the service's in-chat
   * check reads too, by the request's key (requestKey), for an in-chat
   * outcome given at `at`: the counts the latest one started before `at`
   * gave the user (`results`: memories and sentences, those already gone
   * included, memories forgotten whole, and the other memories it affected
   * as `affectedKeys`), when it started (`since`; 0 when not known), and
   * whether one started before `at` is still queued. A forget of the same
   * words started earlier was another request; one started after `at` is a
   * later one.
   */
  statementForgets(key, at = Infinity) {
    const found = [];
    let names = [];
    try {
      names = fs.readdirSync(this.resultsDir);
    } catch {
      // no results yet
    }
    for (const name of names) {
      const result = readJson(path.join(this.resultsDir, name));
      if (!result || result.key !== key || (Number(result.created) || 0) > at) continue;
      found.push({
        created: Number(result.created) || 0,
        memories: (Number(result.forgotten) || 0) + (Number(result.earlier) || 0),
        sentences: (Number(result.sentences) || 0) + (Number(result.earlierSentences) || 0),
        whole: wholeOf(result),
        affectedKeys: Array.isArray(result.affectedKeys) ? result.affectedKeys : [],
      });
    }
    const since = Math.max(0, ...found.map((result) => result.created));
    const queued = this.entries().some((entry) => {
      const job = readJson(entry.file);
      return job?.kind === 'forget' && job.inChat === true && requestKey(job.body?.statement) === key && Number(job.created) <= at;
    });
    return { results: found.filter((result) => result.created >= since), since, queued };
  }

  /**
   * Remove what the earlier layout (before owners) left in the state folder:
   * job files whose name names no owner, the queue's state, the flusher lock,
   * the list of running forgets and the forget outcomes. They are never sent
   * or read: that layout could label one user's turns with another user's id.
   * Each goes once it is older than a current one would be kept (a batch's
   * age limit, an outcome's day), so an earlier version still running on the
   * folder finishes with its own first. Runs at most once an hour per queue.
   */
  cleanLegacy() {
    const now = Date.now();
    if (now - (this.legacyCheckedAt ?? 0) < LEGACY_CHECK_MS) return;
    this.legacyCheckedAt = now;
    const maxAge = Math.max(this.settings.captureMaxAgeMs ?? 86_400_000, RESULT_KEEP_MS);
    const removeOld = (file) => {
      try {
        if (now - fs.statSync(file).mtimeMs > maxAge) fs.unlinkSync(file);
      } catch {
        // gone, or not a file
      }
    };
    const list = (dir) => {
      try { return fs.readdirSync(dir); } catch { return []; }
    };
    for (const name of list(this.dir)) {
      if (LEGACY_JOB_FILE.test(name) || name === 'state.json') removeOld(path.join(this.dir, name));
    }
    const root = this.settings.stateDir;
    for (const name of ['flusher.lock', 'forgets.json', 'forgets.json.lock']) removeOld(path.join(root, name));
    const results = path.join(root, 'results');
    for (const name of list(results)) removeOld(path.join(results, name));
    try { fs.rmdirSync(results); } catch { /* not empty, or none */ }
  }

  cleanResults() {
    let names = [];
    try {
      names = fs.readdirSync(this.resultsDir);
    } catch {
      return;
    }
    for (const name of names) {
      const file = path.join(this.resultsDir, name);
      try {
        const age = Date.now() - fs.statSync(file).mtimeMs;
        if (age > RESULT_KEEP_MS) fs.unlinkSync(file);
        // An outcome no hook read in time (it crashed, say): counts only.
        else if (age > RESULT_QUOTE_GRACE_MS * 30 && readJson(file)?.read !== true) writeFileAtomic(file, JSON.stringify(countsOf(readJson(file) ?? {})));
      } catch {
        // gone
      }
    }
  }
}

// What an in-chat forget outcome did: memories and sentences it removed, and
// the other memories it left unreadable or reworded.
function removedBy(item) {
  return (Number(item.forgotten) || 0) + (Number(item.withheld) || 0) + (Number(item.dependents) || 0)
    + (Number(item.unreadable) || 0) + (Number(item.reworded) || 0);
}

const sumOf = (list, field) => list.reduce((total, entry) => total + (Number(entry[field]) || 0), 0);

// The memories a forget outcome forgot whole: named (`bundled`) or counted (`whole`).
function wholeOf(result) {
  return (Array.isArray(result.bundled) ? result.bundled.filter(Boolean).length : 0) + (Number(result.whole) || 0);
}

/**
 * A forget outcome without any memory text: its status and counts (`whole`:
 * memories forgotten whole; `dependents`, `unreadable`, `reworded`: other
 * memories the forget affected, and `affectedKeys`, those memories by
 * memoryKey), and for a request the in-chat check reads too, its `key` and
 * when its forget started (`created`).
 */
function countsOf(result) {
  const { status, forgotten, sentences, earlier, earlierSentences, changed, code, at, created, dependents, unreadable, reworded, affectedKeys, key } = result;
  const whole = wholeOf(result);
  return {
    status, forgotten, sentences, earlier, earlierSentences, changed, code, at, created, key, affectedKeys,
    ...(whole > 0 ? { whole } : {}), ...(dependents > 0 ? { dependents } : {}), ...(unreadable > 0 ? { unreadable } : {}),
    ...(reworded > 0 ? { reworded } : {}), read: true,
  };
}

// A forget of a request the service's in-chat check reads too: its
// request's key and when it started, kept with its outcome so the in-chat
// outcome of the same request is told only beyond it (takeChatForgets).
function inChatTag(job) {
  return job?.inChat === true && typeof job.body?.statement === 'string'
    ? { inChat: true, key: requestKey(job.body.statement), created: job.created }
    : {};
}

// ---------------------------------------------------------------- flusher

// A dropped forget leaves a failed outcome, which counts nothing.
function drop(queue, entry, job, why) {
  try { fs.unlinkSync(entry.file); } catch { /* gone */ }
  queue.log.warn(`memory ${job?.kind === 'forget' ? 'forget' : 'batch'} ${job?.request_id ?? entry.name} dropped: ${why}`);
  if (job?.kind === 'forget') queue.writeResult(job.request_id, { status: 'failed', code: why.split(' ')[0], ...inChatTag(job) });
}

function save(queue, entry, job) {
  try {
    writeFileAtomic(entry.file, JSON.stringify(job));
  } catch (error) {
    queue.log.warnOnce(`queue-write:${errKind(error)}`, `could not write the capture queue (${errKind(error)})`);
  }
}

function retryOrDrop(queue, entry, job, why, delay) {
  if (job.attempts >= queue.settings.captureMaxAttempts) {
    drop(queue, entry, job, `${why} after ${job.attempts} attempts`);
    return;
  }
  job.nextAt = Date.now() + delay;
  queue.log.debug(`retry ${job.request_id} in ${delay} ms (${why})`);
  save(queue, entry, job);
}

// The service answered `429 delivery_busy`: it holds the job back for now
// (a session's new batch while an earlier one completes, a forget or a
// completion of the same user in progress, a receiver not ready). That is a
// wait, not a failure: the job keeps its attempts and its place, so the jobs
// behind it wait and the service receives them in order. Only such answers
// for longer than `captureBusyMaxWaitMs` (or the age limit) drop it; any
// other answer starts that clock again (endBusy).
function waitBusy(queue, entry, job, delay) {
  const at = Date.now();
  job.attempts = Math.max(0, job.attempts - 1);
  job.busySince ??= at;
  job.busyTries = (job.busyTries || 0) + 1;
  const waited = at - job.busySince;
  if (waited > (queue.settings.captureBusyMaxWaitMs ?? CAPTURE_BUSY_MAX_WAIT_MS)) {
    drop(queue, entry, job, `delivery_busy for ${Math.round(waited / 1000)} s`);
    return;
  }
  const wait = delay ?? backoffMs(queue.settings, { attempts: job.busyTries });
  job.nextAt = at + wait;
  save(queue, entry, job);
  queue.log.debug(`busy ${job.request_id}; the service holds it back for now, send again in ${wait} ms`);
}

function endBusy(job) {
  delete job.busySince;
  delete job.busyTries;
}

// When a job is due: its retry time, not before a pause (a refused token).
// A job whose last attempt could not reach the service is due at once when a
// hook has reached it since; a job the service held back (busy, its spend
// cap) keeps the time the service gave.
function dueAt(job, paused, contact) {
  const next = Number(job?.nextAt) || 0;
  const unreachable = Number(job?.unreachableAt) || 0;
  return Math.max(paused, unreachable > 0 && contact > unreachable ? Math.min(next, contact) : next);
}

function capQueue(queue, entries) {
  const adds = entries.filter((entry) => !(readJson(entry.file)?.kind === 'forget'));
  const over = adds.length - queue.settings.captureQueueMax;
  if (over <= 0) return;
  for (const entry of adds.slice(0, over)) {
    try { fs.unlinkSync(entry.file); } catch { /* gone */ }
  }
  queue.log.warn(`capture queue full; dropped ${over} oldest batch(es)`);
}

// Whether this user's jobs for another service were left in the folder: a
// changed QUOTEMEM_URL. They stay for that service's flusher.
function leftElsewhere(queue, settings, log) {
  const others = queue.entries({ every: true });
  if (others.length === 0) return false;
  const mine = others.some((entry) => {
    const data = readJson(entry.file);
    return validJob(data) && data.body.user_id === settings.userId && typeof data.url === 'string' && data.url !== settings.url;
  });
  if (mine) log.warnOnce('queue-elsewhere', 'saved memory batches for another QUOTEMEM_URL were left in the queue folder');
  return mine;
}

/**
 * Send the settings owner's queued jobs in order until none is left, the
 * next retry is further off than `settings.flusherMaxWaitMs`, or `signal`
 * aborts. Only one flusher runs per owner; a second returns `busy` at once,
 * and one that finds its lock taken over returns `busy` before its next job.
 * Other owners' jobs (another user, another service) are never sent: each
 * owner's own flusher sends them, with that owner's settings.
 * @returns {Promise<'empty'|'busy'|'waiting'|'stopped'|'foreign'>} `foreign`:
 *   nothing of this owner is left, but this user's jobs for another service are.
 */
export async function runFlusher(settings, log, { signal, maxLifetimeMs = MAX_LIFETIME_MS } = {}) {
  const queue = new DiskQueue(settings, log);
  let handle = tryLock(queue.lockFile, { staleMs: FLUSHER_STALE_MS });
  if (!handle) return 'busy';
  const heartbeat = setInterval(() => { if (handle) touchLock(handle); }, HEARTBEAT_MS);
  heartbeat.unref?.();
  const started = Date.now();
  queue.cleanResults();
  queue.cleanLegacy();
  try {
    for (;;) {
      if (signal?.aborted) return 'stopped';
      // A flusher whose lock was broken (the machine slept past the stale
      // time, and a second flusher took over) stops at once, so two never
      // send the same queue.
      if (handle && !touchLock(handle)) {
        handle = null;
        log.warnOnce('flusher-lock-lost', 'the background flusher lost its lock to another one and stopped');
        return 'busy';
      }
      let entries = queue.entries();
      if (entries.length === 0) {
        // Release, then look again: a hook that saw the lock held did not
        // start a flusher, and its job was written before this release.
        releaseLock(handle);
        handle = null;
        if (queue.entries().length === 0) return leftElsewhere(queue, settings, log) ? 'foreign' : 'empty';
        handle = tryLock(queue.lockFile, { staleMs: FLUSHER_STALE_MS });
        if (!handle) return 'busy';
        continue;
      }
      if (entries.length > settings.captureQueueMax) {
        capQueue(queue, entries);
        entries = queue.entries();
      }
      // The next job to act on, in queue order. A batch the service has
      // acknowledged (202, saved and completing by itself) is only read again
      // when its poll is due, and the batches behind it are sent meanwhile; a
      // job not yet acknowledged holds everything behind it, and a forget
      // waits until every batch before it is stored, so it finds what was
      // said before it.
      const now = Date.now();
      const queueState = queue.state();
      const paused = Number(queueState.pausedUntil) || 0;
      // The service completes one user's batches one at a time, in order: a
      // saved batch waits while the ones before it complete. So the time a
      // saved batch may stay unconfirmed runs from the last batch stored, and
      // never while the service's daily spend cap holds its batches (it
      // starts no completion until the reset): it runs from the reset.
      const progress = Math.max(Number(queueState.storedAt) || 0, Number(queueState.cappedUntil) || 0);
      const contact = queue.contactAt();
      let entry = null;
      let job = null;
      let wake = Infinity;
      let held = 0;
      let own = 0;
      // The job that holds the queue until its retry, when it is one that
      // could not reach the service.
      let blocking = null;
      for (const candidate of entries) {
        const data = readJson(candidate.file);
        if (!validJob(data)) {
          drop(queue, candidate, data, 'unreadable');
          continue;
        }
        if (ownerOfJob(data, settings.url) !== queue.owner) continue; // another user's or service's
        own += 1;
        // A saved batch's time held by the spend cap does not age it: the
        // service starts its completion only on a resend after the reset.
        const capped = data.accepted ? Number(data.cappedMs) || 0 : 0;
        if (now - data.created - capped > settings.captureMaxAgeMs) {
          drop(queue, candidate, data, data.accepted ? 'saved by the service but not confirmed stored in time' : 'too old');
          continue;
        }
        // A batch the service saved but has not confirmed while no batch was
        // stored for a long time (its reads keep hanging, or the service is
        // held) is let go: the service still has it, and the forgets and
        // batches behind it must not wait for a day. A safety net only: a
        // service that answers a resend at once never gets here.
        if (data.accepted && now - Math.max(Number(data.acceptedAt) || data.created, progress) > (settings.acceptedMaxWaitMs ?? ACCEPTED_MAX_WAIT_MS)) {
          drop(queue, candidate, data, 'saved by the service but not confirmed stored in time');
          continue;
        }
        const due = dueAt(data, paused, contact);
        if (data.kind !== 'forget' && data.accepted) {
          if (due <= now) {
            entry = candidate;
            job = data;
            break;
          }
          wake = Math.min(wake, due);
          held += 1;
          continue;
        }
        // A forget waits for the saved batches before it to be stored, so it
        // finds what they hold: until they are, or for a short while when
        // the service checks the request again as its turn is stored.
        if (data.kind === 'forget' && held > 0) {
          if (data.inChat !== true) break;
          if (now - data.created < FORGET_EARLIER_WAIT_MS) {
            wake = Math.min(wake, Math.max(due, data.created + FORGET_EARLIER_WAIT_MS));
            break;
          }
        }
        if (due <= now) {
          entry = candidate;
          job = data;
        } else {
          wake = Math.min(wake, due);
          blocking = data;
        }
        break;
      }
      if (!entry) {
        if (Number.isFinite(wake)) {
          const wait = wake - now;
          if (wait > settings.flusherMaxWaitMs || now + wait - started > maxLifetimeMs) {
            // A service that could not be reached may answer again: one
            // probe, at most every PROBE_EVERY_MS; a 2xx makes that job due.
            if (queue.mayProbe(blocking, paused)) {
              queue.noteProbe();
              if (await healthOk(settings, PROBE_TIMEOUT_MS)) {
                queue.noteContact();
                log.debug('the service answers /health again; sending what waited');
                continue;
              }
            }
            return 'waiting';
          }
          await sleep(wait, signal);
          continue;
        }
        continue;
      }
      if (now - started > maxLifetimeMs) return 'waiting';
      // An attempt that started and never recorded its outcome may have
      // reached the service.
      if (job.sending) job.maybeSent = true;
      job.attempts = (job.attempts || 0) + 1;
      job.sending = true;
      save(queue, entry, job);
      const reading = job.kind !== 'forget' && job.accepted;
      const readLimit = Math.min(settings.captureTimeoutMs, settings.acceptedReadTimeoutMs ?? ACCEPTED_READ_TIMEOUT_MS);
      const attempt = AbortSignal.timeout(reading ? readLimit : settings.captureTimeoutMs);
      const all = signal ? AbortSignal.any([attempt, signal]) : attempt;
      job.sending = false;
      if (job.kind === 'forget') {
        // After an attempt that may have confirmed, "nothing matched" means
        // done: no stored statement is forgotten in its place.
        const result = await runForget(settings, job.body, { signal: all, substitute: !job.maybeSent, carry: job.carry ?? null });
        if (signal?.aborted) return 'stopped';
        const busy = result.status === 'retry' && result.code === 'delivery_busy';
        if (!busy) endBusy(job);
        if (result.status === 'retry' && result.transport) job.unreachableAt = Date.now();
        else delete job.unreachableAt;
        if (['forgotten', 'incomplete', 'already', 'nothing'].includes(result.status)) {
          // An outcome keeps memory text only while the hook that asked may
          // still be waiting to quote it (until its forget deadline); one no
          // note quotes (applied again, from a transcript, too late) is kept
          // as counts only.
          const waited = job.quiet !== true && Date.now() - Number(job.created) < (settings.forgetTimeoutMs ?? 1800) + RESULT_QUOTE_GRACE_MS;
          // The request's key: the service's in-chat outcome of the same
          // request is told only beyond these counts (takeChatForgets).
          const keyed = { ...result, ...inChatTag(job) };
          queue.writeResult(job.request_id, !waited ? countsOf(keyed) : keyed);
          try { fs.unlinkSync(entry.file); } catch { /* gone */ }
          log.debug(`forget ${job.request_id} ${result.status} forgotten=${result.forgotten} sentences=${result.sentences} earlier=${result.earlier}+${result.earlierSentences ?? 0}`);
        } else if (result.status === 'retry') {
          if (result.maybeDone) job.maybeSent = true;
          // A plan whose confirmation failed retryably is confirmed again
          // (the service resumes it); the counts so far are kept with it.
          job.carry = carryOnDisk(result.carry ?? null);
          job.conflicts = (job.conflicts || 0) + (result.conflict ? 1 : 0);
          if (job.conflicts >= FORGET_CONFLICT_ATTEMPTS) {
            drop(queue, entry, job, `${result.code} after ${job.conflicts} plans`);
            continue;
          }
          if (busy) waitBusy(queue, entry, job, result.delay);
          else retryOrDrop(queue, entry, job, result.code, result.delay ?? backoffMs(settings, job));
        } else if (result.status === 'auth') {
          queue.setState({ ...queue.state(), pausedUntil: Date.now() + AUTH_PAUSE_MS });
          log.warnOnce('capture-auth', 'the service refused the token; memory writes pause for 10 minutes');
          retryOrDrop(queue, entry, job, result.code, 0);
        } else {
          drop(queue, entry, job, result.code);
        }
        continue;
      }
      const result = await sendAdd(settings, job, all);
      // A batch the service has not saved and holds back as busy waits
      // (waitBusy); a saved one being read is never dropped anyway.
      const busy = result.kind === 'retry' && result.why === 'delivery_busy' && !reading;
      if (!busy) endBusy(job);
      // Not reached (refused, reset, a network error): the next hook that
      // reaches the service makes it due at once (dueAt). Any answer clears it.
      if (result.kind === 'error' && !(reading && attempt.aborted)) job.unreachableAt = Date.now();
      else delete job.unreachableAt;
      if (result.kind === 'error' && reading && attempt.aborted && !signal?.aborted) {
        // A slow read of a batch the service already holds: still pending.
        const delay = pollDelayMs((job.polls = (job.polls || 0) + 1));
        job.attempts = 0;
        job.nextAt = Date.now() + delay;
        save(queue, entry, job);
        log.debug(`pending ${job.request_id}; no answer within ${readLimit} ms, read again in ${delay} ms`);
      } else if (result.kind === 'error') {
        if (signal?.aborted) return 'stopped'; // the job stays queued, marked as maybe sent
        job.maybeSent = true;
        if (reading) {
          // The service holds the batch: a failed read (it is restarting, say)
          // is retried with backoff and never dropped for it; the age limit
          // still applies.
          const delay = backoffMs(settings, job);
          job.nextAt = Date.now() + delay;
          save(queue, entry, job);
          log.debug(`retry reading ${job.request_id} in ${delay} ms (${result.why})`);
        } else {
          retryOrDrop(queue, entry, job, result.why, backoffMs(settings, job));
        }
      } else if (result.kind === 'stored') {
        try { fs.unlinkSync(entry.file); } catch { /* gone */ }
        queue.setState({ ...queue.state(), storedAt: Date.now() });
        // The service's in-chat forget of a request in this batch: the next
        // prompt of this user tells the user what it did.
        if (result.forgets.length > 0) await queue.addChatForgets(result.forgets);
        log.debug(`stored ${job.request_id} ${result.outcome}${result.forgets.length > 0 ? ` forgets=${result.forgets.map((f) => f.status).join(',')}` : ''}`.trim());
      } else if (result.kind === 'pending') {
        // Saved by the service, which completes it by itself; the identical
        // request is sent again later to read the result, never a new one,
        // when the service's Retry-After says, else on the poll schedule.
        const delay = result.delay ?? pollDelayMs(job.accepted ? (job.polls = (job.polls || 0) + 1) : 0);
        if (!job.accepted) job.acceptedAt = Date.now();
        job.accepted = true;
        job.attempts = 0;
        job.nextAt = Date.now() + delay;
        save(queue, entry, job);
        log.debug(`pending ${job.request_id}; read again in ${delay} ms`);
      } else if (result.kind === 'deferred') {
        // The service's daily spend cap is reached (it answers so to a new
        // batch and to the resend of a saved one not yet stored): the batch
        // waits for the reset without using up its attempts. A batch not yet
        // saved keeps its age limit. A saved one stays held, so the forgets
        // behind it keep waiting; its time held is not counted toward its age
        // or its unconfirmed time, which starts again at the reset.
        job.attempts = Math.max(0, job.attempts - 1);
        job.nextAt = Date.now() + result.delay;
        if (job.accepted) {
          job.acceptedAt = Math.max(Number(job.acceptedAt) || 0, job.nextAt);
          job.cappedMs = (Number(job.cappedMs) || 0) + result.delay;
        }
        save(queue, entry, job);
        const capped = Number(queue.state().cappedUntil) || 0;
        if (job.nextAt > capped) queue.setState({ ...queue.state(), cappedUntil: job.nextAt });
        log.warnOnce('capture-deferred', 'the memory service reached its daily spend cap; memory writes wait for its reset');
      } else if (result.kind === 'retry' && reading) {
        const delay = result.delay ?? backoffMs(settings, job);
        job.nextAt = Date.now() + delay;
        save(queue, entry, job);
        log.debug(`retry reading ${job.request_id} in ${delay} ms (${result.why})`);
      } else if (busy) {
        waitBusy(queue, entry, job, result.delay);
      } else if (result.kind === 'retry') {
        retryOrDrop(queue, entry, job, result.why, result.delay ?? backoffMs(settings, job));
      } else if (result.kind === 'auth') {
        queue.setState({ ...queue.state(), pausedUntil: Date.now() + AUTH_PAUSE_MS });
        log.warnOnce('capture-auth', `the service refused the token (${result.status}); memory writes pause for 10 minutes`);
        retryOrDrop(queue, entry, job, result.why, 0);
      } else if (result.kind === 'rotate') {
        rotateJob(job);
        save(queue, entry, job);
      } else {
        drop(queue, entry, job, result.why);
      }
    }
  } finally {
    clearInterval(heartbeat);
    releaseLock(handle);
  }
}
