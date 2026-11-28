// A zero-dependency JSON client for the QuoteMem HTTP service.
//
// node:http rather than fetch: /memory/add answers only when the memories are
// searchable, which can outlast fetch's default header timeout.
import http from 'node:http';
import https from 'node:https';

import { ensureService, refused, replaceOlder } from './autostart.mjs';
import { checkListener, listenerConnection, onThisComputer, pinnedUrl, requestOn } from './listener.mjs';
import { serviceToken } from './token.mjs';

export { checkListener, listenerConnection, listenerProof, onThisComputer, pinnedUrl, requestOn } from './listener.mjs';

const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
export const MAX_RETRY_AFTER_MS = 600_000;

/** The auth header for the configured mode; the token is never logged. */
export function authHeaders(settings) {
  if (!settings.token) return {};
  if (settings.authMode === 'x-api-key') return { 'x-api-key': settings.token };
  return { authorization: `${settings.authMode === 'token' ? 'Token' : 'Bearer'} ${settings.token}` };
}

/**
 * The error of a request that was not sent because the listener failed
 * checkListener (`code` ELISTENER, `kind` the verdict's). `autostart`, for
 * the hooks to tell the user once (setupNote): `restarting` when an older
 * service the plugins started is being replaced (replaceOlder), `older` for
 * one they started but cannot replace, `older-hand` for one the user
 * started, `listener` when another program answered without the token
 * (`foreign`). A listener that stalled or broke off (`no-answer`), as a
 * busy or paused service does, and one that cannot be checked yet
 * (`no-token`) get none: memory is just unavailable for now.
 */
function listenerRefusal(settings, verdict) {
  const replaced = verdict.kind === 'older' ? replaceOlder(settings, { log: settings.autostartLog }).state : null;
  const message = `no token and no memory data were sent to ${settings.url}: the program on its port is not shown to be this user's QuoteMem service: ${verdict.why}${replaced === 'restarting' ? '; it is the service the plugins started, so it is replaced by the current release' : ''} (run \`quotemem doctor\`)`;
  settings.autostartLog?.warnOnce?.(`listener:${verdict.kind}`, message);
  const autostart = replaced ?? (verdict.kind === 'foreign' ? 'listener' : undefined);
  return Object.assign(new Error(message), { code: 'ELISTENER', kind: verdict.kind, ...(autostart ? { autostart } : {}) });
}

/**
 * Nothing listens at a loopback URL: start the local service if it may be
 * started (`error.autostart` says how it went). This call still fails;
 * later calls find it. Returns `error`.
 */
function startOnRefusal(settings, error) {
  if (refused(error) && settings.localService) {
    try {
      error.autostart = ensureService(settings, { log: settings.autostartLog }).state;
    } catch {
      // starting it is best effort
    }
  }
  return error;
}

/**
 * POST a JSON body to `settings.url + route`.
 * @param {object} [headers] - extra request headers (e.g. `prefer`).
 * @returns {Promise<{status: number, data: any, retryAfter: string|undefined}>}
 *   rejects on a transport error or when `signal` aborts.
 */
export async function postJson(settings, route, body, signal, headers = {}) {
  // A service on this computer that was started on demand creates its
  // token on its first start: read it once it exists.
  if (!settings.token && settings.localService) settings.token = serviceToken(process.env, settings.localService);
  try {
    return await postOnce(settings, route, body, signal, headers);
  } catch (error) {
    // The connection the check proved closed before the request went (a
    // service that is stopping): the request was not sent. Once more, checked anew
    // on a new connection.
    if (error?.code !== 'ELISTENER_CLOSED') throw error;
    return postOnce(settings, route, body, signal, headers);
  }
}

/**
 * GET `route` with the token, as `quotemem doctor` and `status` read
 * /agent/status: from a service on this computer only once the program on
 * its port proved it holds the token, on the connection the check went on.
 * @returns {Promise<{verdict: object, status?: number, error?: string}>}
 *   `verdict`: checkListener's; `status` (0 with `error` on a transport
 *   failure) only when it passed.
 */
export async function checkedGet(settings, route, { timeoutMs = 3000 } = {}) {
  const connection = onThisComputer(settings.url) ? listenerConnection(settings.url) : null;
  try {
    const verdict = await checkListener(settings, { connection });
    if (!verdict.ok) return { verdict };
    try {
      const answer = await requestOn(connection, verdict.url + route, {
        method: 'GET',
        headers: { accept: 'application/json', ...authHeaders(settings) },
        signal: AbortSignal.timeout(timeoutMs),
      });
      return { verdict, status: answer.status };
    } catch (error) {
      return { verdict, status: 0, error: error?.code ?? error?.name ?? 'error' };
    }
  } finally {
    connection?.close();
  }
}

async function postOnce(settings, route, body, signal, headers) {
  let base = settings.url;
  // A service on this computer gets a request only once the program on its
  // port proved it holds the token, at the address it checked (pinnedUrl:
  // `localhost` is 127.0.0.1), and on the connection the check went on
  // (listenerConnection). Without a token only the check goes there, and it
  // carries no token.
  const connection = onThisComputer(settings.url) ? listenerConnection(settings.url) : null;
  try {
    if (connection) {
      const verdict = await checkListener(settings, { signal, connection });
      // Nothing listens: no request is sent; the refusal is this call's own.
      if (verdict.refused) throw startOnRefusal(settings, verdict.refused);
      if (!verdict.ok) throw listenerRefusal(settings, verdict);
      base = verdict.url;
    }
    let answer;
    try {
      answer = await requestOn(connection, base + route, {
        method: 'POST',
        signal,
        body: JSON.stringify(body),
        headers: { accept: 'application/json', 'content-type': 'application/json', ...headers, ...authHeaders(settings) },
        maxBytes: MAX_RESPONSE_BYTES,
      });
    } catch (error) {
      throw startOnRefusal(settings, error);
    }
    let data = null;
    try {
      const text = answer.body.toString('utf8');
      data = text ? JSON.parse(text) : null;
    } catch {
      data = null;
    }
    return { status: answer.status, data, retryAfter: answer.headers['retry-after'] };
  } finally {
    connection?.close();
  }
}

/**
 * Whether the service answers `GET /health` with 2xx within `timeoutMs`. No
 * token is sent, and nothing is started. Never throws.
 */
export function healthOk(settings, timeoutMs) {
  return new Promise((resolve) => {
    let url;
    try {
      url = new URL(`${onThisComputer(settings.url) ? pinnedUrl(settings.url) : settings.url}/health`);
    } catch {
      resolve(false);
      return;
    }
    const client = url.protocol === 'https:' ? https : http;
    const request = client.request(url, { method: 'GET', signal: AbortSignal.timeout(timeoutMs), headers: { accept: 'application/json' } }, (response) => {
      response.resume();
      resolve(response.statusCode >= 200 && response.statusCode < 300);
    });
    request.on('error', () => resolve(false));
    request.end();
  });
}

/** A short, content-free name for an error, for logs. */
export function errKind(error) {
  if (!error) return 'unknown';
  if (typeof error.code === 'string') return error.code;
  if (error.name === 'AbortError' || error.name === 'TimeoutError') return 'timeout';
  return error.name || 'error';
}

/** The error code of a response: the service's `code`, else `http_<status>`. */
export function codeOf(response) {
  return typeof response?.data?.code === 'string' ? response.data.code : `http_${response?.status ?? 0}`;
}

/**
 * Whether a response may succeed if sent again: 429 and 503 (busy, a forget
 * or an index rebuild in progress, a provider outage), another 5xx unless
 * the service says otherwise, and any answer the service marks retryable.
 */
export function retryable(response) {
  const { status, data } = response;
  return status === 429 || status === 503 || (status >= 500 && data?.retryable !== false) || data?.retryable === true;
}

/** The Retry-After hint in ms (capped), or null. */
export function retryAfterMs(response) {
  const seconds = Number.parseInt(String(response?.retryAfter ?? ''), 10);
  return Number.isFinite(seconds) && seconds >= 0 ? Math.min(MAX_RETRY_AFTER_MS, seconds * 1000) : null;
}

/** Wait `ms`; rejects with the signal's reason when it aborts first. */
export function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * POST with a deadline and bounded retries. Transport errors, 429, 503 and
 * retryable 5xx answers are sent again with the same body, honouring
 * Retry-After, while the deadline allows; otherwise the last answer is
 * returned. Use it only for requests that are safe to repeat.
 * @param {object} options - `deadlineMs` (total budget), `signal` (extra abort),
 *   `attempts` (default 3), `retryBaseMs` (default 250).
 * @returns {Promise<{status, data, retryAfter}>} rejects when no answer came in time.
 */
export async function callJson(settings, route, body, { deadlineMs = 10_000, signal, attempts = 3, retryBaseMs = 250 } = {}) {
  const deadline = AbortSignal.timeout(Math.max(1, deadlineMs));
  const all = signal ? AbortSignal.any([deadline, signal]) : deadline;
  const end = Date.now() + deadlineMs;
  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    let response = null;
    try {
      response = await postJson(settings, route, body, all);
    } catch (error) {
      if (all.aborted) throw error;
      lastError = error;
    }
    if (response && !retryable(response)) return response;
    if (attempt === attempts) {
      if (response) return response;
      throw lastError;
    }
    const wait = (response && retryAfterMs(response)) ?? retryBaseMs * 2 ** (attempt - 1);
    if (Date.now() + wait >= end) {
      if (response) return response;
      throw lastError;
    }
    await sleep(wait, all);
  }
  throw lastError ?? new Error('unreachable');
}
