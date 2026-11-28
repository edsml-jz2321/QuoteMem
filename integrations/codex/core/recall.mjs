// Gated recall: decide whether a user message is worth a search, search
// under a short deadline, and build the block to inject. Every failure means
// "no memory this turn"; nothing here throws into the host.
import { formatRecall } from './format.mjs';
import { isCommand, isMemoryRelated } from './gate.mjs';
import { codeOf, errKind, postJson, retryAfterMs, sleep } from './http.mjs';
import { userPort } from './proc.mjs';
import { REDACT_MARGIN, redactText } from './redact.mjs';
import { AUTH_PAUSE_MS } from './send.mjs';

const BREAKER_MIN_MS = 30_000;
const BREAKER_MAX_MS = 300_000;
const MAX_MEMO_IDS = 2000;

/**
 * The search query for a user message, or null when recall should not run:
 * too short, a slash command, not memory related (gated mode), or only a
 * secret. Only the head of a long paste is gated and redacted.
 * @param {{statements?: boolean}} options - passed to isMemoryRelated; the
 *   default `settings.recallGate === 'broad'` decides.
 */
export function recallQuery(settings, text, { statements = settings.recallGate === 'broad' } = {}) {
  if (settings.recall === 'off') return null;
  const raw = String(text ?? '').trim().slice(0, settings.queryMaxChars + REDACT_MARGIN);
  if (raw.length < settings.recallMinChars || isCommand(raw)) return null;
  if (settings.recall === 'gated' && !isMemoryRelated(raw, { statements })) return null;
  const redacted = redactText(raw);
  if (redacted.secretOnly) return null;
  return redacted.text.slice(0, settings.queryMaxChars);
}

// Recall asks the service once for its largest list and inserts the first
// `topK` hits that fit. The agent service's ranking and floors do not depend
// on `top_k`, so a shorter list is the head of this one; the hits after
// `topK` fill the block when a higher hit is too long for the character
// budget or was inserted a few turns ago.
export const SEARCH_HITS = 20;

/** The /memory/search body for a query: always SEARCH_HITS hits are asked for. */
export function searchBody(settings, query) {
  const body = { user_id: settings.userId, query, top_k: SEARCH_HITS };
  if (settings.searchOptions.length > 0) body.options = settings.searchOptions;
  return body;
}

/**
 * A circuit breaker. A service that cannot be reached, or that misses the
 * deadline twice in a row, is skipped for a while (30 s, doubling up to 5
 * minutes); a refused token pauses recall for 10 minutes. One slow search
 * does not turn recall off: the next memory question searches again. So a
 * down service adds no delay and a busy one costs at most two deadlines.
 * JSON-safe (`toJSON`/`from`), so a hook process can keep it in a file.
 */
export class RecallBreaker {
  constructor(state = {}) {
    this.blockedUntil = Number(state.blockedUntil) || 0;
    this.failures = Number(state.failures) || 0;
    this.timeouts = Number(state.timeouts) || 0;
  }

  static from(state) {
    return new RecallBreaker(state ?? {});
  }

  get open() {
    return Date.now() < this.blockedUntil;
  }

  /** A failed search: `timeout` (the deadline passed) or any other failure (no connection). */
  fail(kind = 'error') {
    if (kind === 'timeout') {
      this.timeouts += 1;
      if (this.timeouts < 2) return;
    }
    this.timeouts = 0;
    this.failures += 1;
    this.blockedUntil = Date.now() + Math.min(BREAKER_MAX_MS, BREAKER_MIN_MS * 2 ** (this.failures - 1));
  }

  authPause() {
    this.blockedUntil = Date.now() + AUTH_PAUSE_MS;
  }

  ok() {
    this.failures = 0;
    this.timeouts = 0;
  }

  toJSON() {
    return { blockedUntil: this.blockedUntil, failures: this.failures, timeouts: this.timeouts };
  }
}

/**
 * Per-session memory of what was injected, so a memory is not injected again
 * within `recallRepeatTurns` user turns. `{turns, ids: [[id, turn], ...]}`,
 * JSON-safe.
 */
export function newRecallMemo() {
  return { turns: 0, ids: [] };
}

/** Count one user turn and return the ids injected recently enough to skip. */
export function recallTurn(settings, memo) {
  memo.turns += 1;
  return recentIds(settings, memo);
}

/** The ids injected within the last `recallRepeatTurns` turns; older ones are forgotten. */
export function recentIds(settings, memo) {
  memo.ids = memo.ids.filter(([, at]) => memo.turns - at < settings.recallRepeatTurns);
  return new Set(memo.ids.map(([id]) => id));
}

/** Remember the ids just injected. */
export function rememberIds(memo, ids) {
  const fresh = new Set(ids);
  memo.ids = memo.ids.filter(([id]) => !fresh.has(id));
  for (const id of ids) memo.ids.push([id, memo.turns]);
  if (memo.ids.length > MAX_MEMO_IDS) memo.ids.splice(0, memo.ids.length - MAX_MEMO_IDS);
}

/** Build the recall block from a /memory/search answer's hits with the settings' limits: the first `topK` that fit. */
export function recallBlock(settings, hits, skipIds = new Set()) {
  return formatRecall(hits, {
    maxChars: settings.maxInjectChars,
    maxHits: settings.topK,
    minScore: settings.recallMinScore,
    keepUnscored: settings.recall === 'gated',
    skipIds,
  });
}

// Right after a forget the service rebuilds the user's search index, and a
// search answers 429 `index_rebuilding` until it is done: about one to three
// seconds (measured), though its Retry-After says 60 s, a bulk client's
// hint. Recall asks again every REBUILD_POLL_MS (sooner when Retry-After
// says so) while its own deadline allows, so a question asked just after a
// forget, often the first of a new chat, is still answered. Only this
// answer is asked again: any other error ends the search at once.
const REBUILD_POLL_MS = 250;
// The shortest wait between two asks: a Retry-After of 0 must not make a tight loop.
const REBUILD_MIN_WAIT_MS = 50;
// A request needs about this long to be answered; a later retry would only miss the deadline.
const REBUILD_MIN_LEFT_MS = 150;

/**
 * How long to wait before asking again after a search answer, or null when
 * it is not asked again: it is not 429 `index_rebuilding`, or the deadline
 * leaves too little time (`leftMs`).
 */
function rebuildWait(response, leftMs) {
  if (response?.status !== 429 || codeOf(response) !== 'index_rebuilding') return null;
  const hint = retryAfterMs(response);
  const wait = hint === null ? REBUILD_POLL_MS : Math.max(REBUILD_MIN_WAIT_MS, Math.min(REBUILD_POLL_MS, hint));
  return leftMs - wait >= REBUILD_MIN_LEFT_MS ? wait : null;
}

/**
 * Search for one query under the recall deadline and build the block.
 * @param {object} options - `signal` (an extra abort, e.g. the host's turn),
 *   `skipIds` (recently injected), `breaker` (a RecallBreaker), `log`,
 *   `report` (an object; `report.unavailable` is set true when the service
 *   could not answer: skipped by the breaker, unreachable, too slow, or an
 *   error answer).
 * @returns {Promise<{text: string, ids: string[]} | null>} null when nothing qualifies or anything fails.
 */
export async function searchRecall(settings, query, { signal, skipIds = new Set(), breaker = new RecallBreaker(), log, report = {} } = {}) {
  if (!query) return null;
  if (breaker.open) {
    report.unavailable = true;
    return null;
  }
  const deadline = AbortSignal.timeout(settings.recallTimeoutMs);
  const all = signal ? AbortSignal.any([deadline, signal]) : deadline;
  let response;
  const started = Date.now();
  const end = started + settings.recallTimeoutMs;
  let rebuilding = false;
  try {
    for (;;) {
      response = await postJson(settings, '/memory/search', searchBody(settings, query), all);
      // Asked again while the deadline allows (rebuildWait); any other answer is final.
      const wait = rebuildWait(response, end - Date.now());
      if (wait === null) break;
      rebuilding = true;
      await sleep(wait, all);
    }
  } catch (error) {
    if (signal?.aborted) return null;
    report.unavailable = true;
    if (error?.autostart) report.autostart = error.autostart;
    // A deadline spent waiting for a rebuilt index: the service answered.
    const why = deadline.aborted ? (rebuilding ? 'index_rebuilding' : 'timeout') : errKind(error);
    // A local service that is being started on demand is not a failure: a
    // refused connection costs nothing, and the next question finds it. Nor
    // is a listener that answered the check but was not sent the question
    // (ELISTENER): its check takes milliseconds, and the user is told why
    // (setupNote). A listener that stalled or broke off on the check
    // (`no-answer`) is: a busy or paused service, as for any request.
    const answered = error?.code === 'ELISTENER' && error.kind !== 'no-answer';
    if (why !== 'index_rebuilding' && !answered && error?.autostart !== 'started' && error?.autostart !== 'starting') breaker.fail(why === 'timeout' || error?.kind === 'no-answer' ? 'timeout' : 'error');
    log?.warnOnce(`recall:${why}`, `recall skipped: ${why}`);
    return null;
  }
  if (response.status < 200 || response.status >= 300) report.unavailable = true;
  if (response.status === 401 || response.status === 403) {
    breaker.authPause();
    // A local service that refuses this user's token may be another OS
    // user's (two people signed in at once share 127.0.0.1:8765).
    log?.warnOnce('recall-auth', settings.localService
      ? `recall skipped: the memory service at ${settings.url} refused this user's token (${response.status}); it may be another user's QuoteMem on this computer (two people signed in at once). Give this user a port of its own: QUOTEMEM_URL=http://127.0.0.1:${userPort()} in ~/.quotemem/env and in the environment Claude Code, Codex and DSH start in (see \`quotemem doctor\`)`
      : `recall skipped: the service refused the token (${response.status})`);
    return null;
  }
  if (response.status < 200 || response.status >= 300) {
    const code = codeOf(response);
    log?.warnOnce(`recall:${code}`, `recall skipped: ${response.status} ${code}`);
    return null;
  }
  breaker.ok();
  report.reached = true;
  const block = recallBlock(settings, response.data?.data, skipIds);
  log?.debug(`recall ${block ? block.ids.length : 0} hit(s) in ${Date.now() - started} ms`);
  return block;
}
