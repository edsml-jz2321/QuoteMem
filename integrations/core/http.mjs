// A zero-dependency JSON client for the QuoteMem HTTP service.
//
// node:http rather than fetch: /memory/add answers only when the memories are
// searchable, which can outlast fetch's default header timeout.
import http from 'node:http';
import https from 'node:https';

import { ensureService, refused } from './autostart.mjs';
import { serviceToken } from './token.mjs';

const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
export const MAX_RETRY_AFTER_MS = 600_000;

/** The auth header for the configured mode; the token is never logged. */
export function authHeaders(settings) {
  if (!settings.token) return {};
  if (settings.authMode === 'x-api-key') return { 'x-api-key': settings.token };
  return { authorization: `${settings.authMode === 'token' ? 'Token' : 'Bearer'} ${settings.token}` };
}

/**
 * POST a JSON body to `settings.url + route`.
 * @param {object} [headers] - extra request headers (e.g. `prefer`).
 * @returns {Promise<{status: number, data: any, retryAfter: string|undefined}>}
 *   rejects on a transport error or when `signal` aborts.
 */
export function postJson(settings, route, body, signal, headers = {}) {
  return new Promise((resolve, reject) => {
    let url;
    try {
      url = new URL(settings.url + route);
    } catch (error) {
      reject(error);
      return;
    }
    // A service on this computer that was started on demand creates its
    // token on its first start: read it once it exists.
    if (!settings.token && settings.localService) settings.token = serviceToken(process.env, settings.localService);
    const payload = Buffer.from(JSON.stringify(body), 'utf8');
    const client = url.protocol === 'https:' ? https : http;
    const request = client.request(url, {
      method: 'POST',
      signal,
      headers: {
        accept: 'application/json',
        'content-type': 'application/json',
        'content-length': payload.length,
        ...headers,
        ...authHeaders(settings),
      },
    }, (response) => {
      const chunks = [];
      let size = 0;
      response.on('data', (chunk) => {
        size += chunk.length;
        if (size > MAX_RESPONSE_BYTES) request.destroy(Object.assign(new Error('response too large'), { code: 'ERESPONSE_TOO_LARGE' }));
        else chunks.push(chunk);
      });
      response.on('error', reject);
      response.on('end', () => {
        let data = null;
        try {
          const text = Buffer.concat(chunks).toString('utf8');
          data = text ? JSON.parse(text) : null;
        } catch {
          data = null;
        }
        resolve({ status: response.statusCode ?? 0, data, retryAfter: response.headers['retry-after'] });
      });
    });
    request.on('error', (error) => {
      // Nothing listens at a loopback URL: start the local service if it
      // may be started. This call still fails; later calls find it.
      if (refused(error) && settings.localService) {
        try {
          error.autostart = ensureService(settings, { log: settings.autostartLog }).state;
        } catch {
          // starting it is best effort
        }
      }
      reject(error);
    });
    request.end(payload);
  });
}

/**
 * Whether the service answers `GET /health` with 2xx within `timeoutMs`. No
 * token is sent, and nothing is started. Never throws.
 */
export function healthOk(settings, timeoutMs) {
  return new Promise((resolve) => {
    let url;
    try {
      url = new URL(`${settings.url}/health`);
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
