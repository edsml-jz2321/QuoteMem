// Sending one queued /memory/add batch and classifying the answer. Used by the
// background flusher (diskqueue.mjs), which every adapter runs.
//
// Every /memory/add asks for an early acknowledgement (`Prefer: respond-async`): a
// service that supports it answers 202 {"status":"pending"} as soon as the
// batch is durably saved, and completes it without the client. Sending the
// identical request again (same request id, same body) reads its state
// without a second extraction: 200 with the result once it is stored, 202
// while it is still pending. An older service ignores the preference and
// answers 200 when the batch is stored, as before.
import { MAX_ID_CHARS } from './config.mjs';
import { LOST_EFFECTS, effectCounts, memoryKey, requestKey } from './forget.mjs';
import { codeOf, errKind, postJson, retryAfterMs, retryable } from './http.mjs';

export const AUTH_PAUSE_MS = 600_000;
const MAX_BACKOFF_MS = 300_000;
export const ADD_HEADERS = Object.freeze({ prefer: 'respond-async' });
const POLL_BASE_MS = 1000;
// A batch is usually stored within seconds of the last read: a longer gap
// only adds to the time until the next turn can find it.
const POLL_MAX_MS = 3000;
// The longest wait a service's Retry-After may set for reading a saved batch again.
const POLL_HINT_MAX_MS = 60_000;

/**
 * POST one /memory/add job and say what to do next.
 * @returns {Promise<{kind: 'stored', outcome: string, forgets: object[]} | {kind: 'pending', delay: number|null}
 *   | {kind: 'deferred', delay: number}
 *   | {kind: 'error', why: string}
 *   | {kind: 'retry', why: string, delay: number|null} | {kind: 'held', why: string} | {kind: 'auth', status: number, why: string}
 *   | {kind: 'rotate'} | {kind: 'drop', why: string}>}
 *   `stored`: `forgets` is what the service's in-chat forget did for a
 *   request in the batch (chatForgetsOf: counts only, no words).
 *   `pending`: the service saved the batch and completes it by itself; send
 *   the job again unchanged after `delay` (the service's Retry-After), else
 *   pollDelayMs later, to read the result.
 *   `deferred`: the service's daily spend cap is reached; send it again after `delay` ms.
 *   `error` is a transport failure (the request may have reached the service);
 *   `held`: not sent, the listener on the service's port failed the check;
 *   `retry` carries the Retry-After hint, if any; `rotate` means the first
 *   attempt ended without an answer and the service closed that request id.
 */
export async function sendAdd(settings, job, signal) {
  let response;
  try {
    response = await postJson(settings, '/memory/add', { ...job.body, request_id: job.request_id }, signal, ADD_HEADERS);
  } catch (error) {
    // Not sent: the program on the service's port did not prove it holds the token.
    if (error?.code === 'ELISTENER') return { kind: 'held', why: `listener ${error.kind ?? ''}`.trim() };
    return { kind: 'error', why: errKind(error) };
  }
  const { status, data } = response;
  const code = codeOf(response);
  if (status === 202) {
    const hint = retryAfterMs(response);
    return { kind: 'pending', delay: hint === null ? null : Math.min(POLL_HINT_MAX_MS, Math.max(100, hint)) };
  }
  if (status >= 200 && status < 300) {
    const outcome = data?.outcome && typeof data.outcome === 'object'
      ? Object.entries(data.outcome).map(([k, v]) => `${k}=${Number(v) || 0}`).join(' ')
      : '';
    return { kind: 'stored', outcome, forgets: chatForgetsOf(data) };
  }
  // The agent service's daily spend cap: the batch is kept until the reset.
  if (status === 429 && code === 'spend_cap_reached') return { kind: 'deferred', delay: deferMs(response) };
  if (retryable(response)) return { kind: 'retry', why: code, delay: retryAfterMs(response) };
  // A refused token is fixed by the user; keep the batches and pause.
  if (status === 401 || status === 403) return { kind: 'auth', status, why: code };
  // A batch the service acknowledged is never sent under a new id: that
  // would extract it twice.
  if (status === 422 && code === 'operation_conflict' && job.maybeSent && !job.rotated && !job.accepted) return { kind: 'rotate' };
  return { kind: 'drop', why: `${status} ${code}` };
}

const DEFER_MIN_MS = 60_000;
const DEFER_MAX_MS = 86_400_000;

/** How long a batch deferred by the spend cap waits: the service's Retry-After, from 1 minute to 1 day. */
export function deferMs(response) {
  const header = Number.parseInt(String(response?.retryAfter ?? ''), 10);
  const body = Number(response?.data?.retry_after_seconds);
  const seconds = Number.isFinite(header) && header >= 0 ? header : Number.isFinite(body) && body >= 0 ? body : 0;
  return Math.min(DEFER_MAX_MS, Math.max(DEFER_MIN_MS, seconds * 1000));
}

/** Delay before reading an acknowledged batch again, after `polls` reads: doubling from 1 s, capped at 3 s. */
export function pollDelayMs(polls) {
  return Math.min(POLL_MAX_MS, POLL_BASE_MS * 2 ** Math.max(0, polls));
}

/** Resend under a fresh request id once: the service closed the first one unanswered. */
export function rotateJob(job) {
  job.rotated = true;
  job.request_id = `${job.request_id.slice(0, MAX_ID_CHARS - 3)}:r1`;
  job.nextAt = 0;
}

/** Retry delay of a job after `job.attempts` attempts: tripling, jittered, capped. */
export function backoffMs(settings, job) {
  const base = settings.retryBaseMs * 3 ** Math.max(0, job.attempts - 1);
  const jitter = 0.8 + Math.random() * 0.4;
  return Math.min(MAX_BACKOFF_MS, Math.round(base * jitter));
}

/** Whether a value looks like a queued job (a /memory/add batch, or a forget for the disk queue). */
export function validJob(job) {
  return job && typeof job === 'object' && typeof job.request_id === 'string' && job.body
    && typeof job.body === 'object' && Number.isFinite(job.created)
    && (job.kind === 'forget' ? typeof job.body.statement === 'string' : Array.isArray(job.body.messages));
}

const count = (value) => (Number.isInteger(value) && value > 0 ? value : 0);

/**
 * Whether the plugin's own forget by statement already handled an in-chat
 * request (so its own note reports it): `earlier: true`, or an `earlier`
 * record whose `statement` is `forgotten` or `prepared`. A record's `in_chat`
 * alone is the in-chat forget's own outcome, which this report is.
 */
function handledEarlier(earlier) {
  if (earlier === true) return true;
  return Boolean(earlier && typeof earlier === 'object' && (earlier.statement === 'forgotten' || earlier.statement === 'prepared'));
}

// The other memories an in-chat forget affected (`affected`, as the forget
// answers list them): `dependents` went with it, `unreadable` stay stored but
// cannot be read, `reworded` stay stored, rewritten to leave out what was
// forgotten (effectCounts; each given when not 0). With `keyed`, also
// `affectedKeys`: each such memory as [memoryKey, effect] (null for an entry
// without an id), so takeChatForgets counts only those the plugin's own
// forget of the same request did not report.
function lostOf(affected, keyed) {
  const known = (Array.isArray(affected) ? affected : []).filter((entry) => LOST_EFFECTS.has(entry?.effect));
  const counts = effectCounts(known.map((entry) => entry.effect));
  if (!keyed || known.length === 0) return counts;
  return { ...counts, affectedKeys: known.map((entry) => [typeof entry.memory_id === 'string' ? memoryKey(entry.memory_id) : null, entry.effect]) };
}

/**
 * What the service's in-chat forget did while it stored a batch, from the
 * /memory/add answer's `agent.forgets`: status and counts only. The request's words
 * are never kept: an outcome the plugin's own forget by statement also
 * handled (`earlier`) gets their `key` (requestKey), so the two outcomes of
 * one request are counted once (DiskQueue.takeChatForgets).
 * @returns {{status: 'forgotten'|'nothing_matched'|'pending', forgotten: number, withheld: number, bundled: number,
 *   dependents?: number, unreadable?: number, reworded?: number, affectedKeys?: [string|null, string][],
 *   earlier: boolean, key?: string}[]}
 */
export function chatForgetsOf(data) {
  const list = Array.isArray(data?.agent?.forgets) ? data.agent.forgets : [];
  return list
    .filter((item) => item && typeof item === 'object' && ['forgotten', 'nothing_matched', 'pending'].includes(item.status))
    .map((item) => {
      const earlier = handledEarlier(item.earlier);
      const key = earlier ? requestKey(item.request) : null;
      return {
        status: item.status,
        forgotten: count(item.forgotten),
        withheld: count(item.withheld_spans),
        bundled: count(item.bundled),
        ...lostOf(item.affected, earlier),
        earlier,
        ...(key ? { key } : {}),
      };
    });
}
