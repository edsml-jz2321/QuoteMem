// Who listens on the port of a QuoteMem service on this computer, checked
// before the token or any content goes there. Another user of the computer
// may have taken the port first (two people signed in at once): such a
// program must not receive the token or what the tools send.
//
// The service answers GET /agent/listener with a fresh challenge by
// listenerProof: an HMAC under its token of the address and port it listens
// on and the challenge, which only the token's holder can compute. The check
// carries no token and no content. The check and the request it guards go to
// one address (pinnedUrl): `localhost` and 0.0.0.0 mean 127.0.0.1, `[::]`
// means ::1.
//
// Only a listener that proves the token gets the token and the requests it
// guards. Anything else gets no token and no content: a program that
// answers with a wrong proof or refuses the check, and a service from
// before the check (QuoteMem 0.3.2 or older answers the
// route with 503 while it starts and 422 once it serves). The caller decides what to tell
// the user; the plugins replace an older service that they started
// themselves (autostart.mjs replaceOlder). No owner of a socket is read
// here: that takes seconds on macOS and Windows, and cannot be read there
// for another user's program. `quotemem doctor` reads it to explain a
// refusal (portOwnerAsync).
//
// Every request that carries the token or content is checked just before
// it is sent, and goes on the very connection the check proved
// (listenerConnection): no verdict is kept from one request to the next,
// and no second connection is opened after a check, so a program that takes
// the port after the service stopped, even between a check and its
// request, gets no token and no content either.
//
// When nothing listens, no request is sent: the caller gets the refusal, as
// if its own connection had been refused, and may start the service.
import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import http from 'node:http';
import https from 'node:https';

import { olderState, refused } from './autostart.mjs';
import { portOwnerAsync, userPort } from './proc.mjs';

export const LISTENER_ROUTE = '/agent/listener';
const LISTENER_TIMEOUT_MS = 3000;
/** The time doctor and status give an owner lookup (PowerShell can take seconds). */
export const OWNER_LOOKUP_MS = 10_000;

const errKind = (error) => (typeof error?.code === 'string' ? error.code : error?.name || 'error');

/** What only a holder of `token` answers to `challenge` at `address` and `port` (the service's listener_proof). */
export function listenerProof(token, address, port, challenge) {
  return createHmac('sha256', String(token)).update(`quotemem-listener-proof\0${address}\0${port}\0${challenge}`).digest('hex');
}

/**
 * An address as the service names it: lower case, no brackets, an
 * IPv4-mapped IPv6 address as IPv4 (`::ffff:127.0.0.1`, and `::ffff:7f00:1`
 * as a URL parser writes it, are 127.0.0.1).
 */
export function plainAddress(address) {
  const text = String(address).replace(/^\[|\]$/g, '').toLowerCase();
  const dotted = /^(?:0{0,4}:){0,4}:?:ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(text);
  if (dotted) return dotted[1];
  const hex = /^(?:0{0,4}:){0,4}:?:ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(text);
  if (!hex) return text;
  const [high, low] = [Number.parseInt(hex[1], 16), Number.parseInt(hex[2], 16)];
  return `${high >> 8}.${high & 255}.${low >> 8}.${low & 255}`;
}

// `localhost`, `localhost.` and any name under it (`app.localhost`): names
// that always mean this computer.
const LOCALHOST_NAME = /^(?:[a-z0-9_-]+\.)*localhost\.?$/;

/**
 * Whether a URL host names this computer: `localhost` (also `localhost.` and
 * a name under it, such as `app.localhost`), an IPv4 loopback address
 * (127.x.y.z, also IPv4-mapped as `[::ffff:7f00:1]`), ::1, or the
 * unspecified 0.0.0.0 and `[::]`, which connect to this computer. For this
 * computer's own host name, config.mjs warns to use 127.0.0.1.
 */
export function thisComputerHost(hostname) {
  const host = plainAddress(hostname);
  return LOCALHOST_NAME.test(host) || host === '::1' || host === '::' || host === '0.0.0.0' || /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}

/** The address a check and a request reach for such a host: a `localhost` name and 0.0.0.0 are 127.0.0.1, `[::]` is ::1. */
export function listenerAddress(hostname) {
  const host = plainAddress(hostname);
  if (LOCALHOST_NAME.test(host) || host === '0.0.0.0') return '127.0.0.1';
  return host === '::' ? '::1' : host;
}

/** Whether a service URL is checked before the token or content is sent: an http(s) URL whose host is this computer. */
export function onThisComputer(url) {
  try {
    const parsed = new URL(url);
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:') && thisComputerHost(parsed.hostname);
  } catch {
    return false;
  }
}

/**
 * `url` as every request to a service on this computer is sent: a
 * `localhost` name, 0.0.0.0, `[::]` or IPv4-mapped host replaced by its address
 * (listenerAddress), so the check and the request reach one address. Other
 * URLs are returned unchanged.
 */
export function pinnedUrl(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (!thisComputerHost(parsed.hostname)) return url;
  const address = listenerAddress(parsed.hostname);
  // Built from the parsed URL (the host as written may be `0`, `127.1` or
  // `[::ffff:127.0.0.1]`); a URL written without a path gets none.
  const pathWritten = /^[^:]*:\/\/[^/?#]*[/?#]/.test(String(url).trim());
  const rest = pathWritten ? `${parsed.pathname}${parsed.search}${parsed.hash}` : '';
  return `${parsed.protocol}//${address.includes(':') ? `[${address}]` : address}${parsed.port ? `:${parsed.port}` : ''}${rest}`;
}

/**
 * One connection for a check and the requests it guards (checkListener's
 * `connection`, then requestOn): its agent opens one socket and never
 * another, so a request goes only on the connection the check proved. A
 * request after that socket closed fails with `ELISTENER_CLOSED` and sends
 * nothing; the caller checks again on a new connection. `close()` ends it.
 */
export function listenerConnection(url) {
  const secure = String(url).startsWith('https:');
  const agent = new (secure ? https.Agent : http.Agent)({ keepAlive: true, maxSockets: 1, maxFreeSockets: 1 });
  const open = agent.createConnection.bind(agent);
  let opened = false;
  agent.createConnection = (options, callback) => {
    if (opened) {
      callback(Object.assign(new Error('the connection the listener check proved has closed, so nothing was sent on another'), { code: 'ELISTENER_CLOSED' }));
      return undefined;
    }
    opened = true;
    return open(options, callback);
  };
  return { agent, close: () => agent.destroy() };
}

/**
 * A request on `connection` (listenerConnection; null: a connection of its
 * own). Resolves `{status, headers, body}` (body a Buffer); rejects on a
 * transport error, when `signal` aborts, and beyond `maxBytes` of answer
 * (`ERESPONSE_TOO_LARGE`).
 */
export function requestOn(connection, url, { method = 'GET', headers = {}, body, signal, maxBytes = 16 * 1024 * 1024 } = {}) {
  return new Promise((resolve, reject) => {
    let target;
    try {
      target = new URL(url);
    } catch (error) {
      reject(error);
      return;
    }
    const client = target.protocol === 'https:' ? https : http;
    const payload = body === undefined || body === null ? null : Buffer.from(body);
    const request = client.request(target, {
      method,
      signal,
      ...(connection ? { agent: connection.agent } : {}),
      headers: payload ? { ...headers, 'content-length': payload.length } : headers,
    }, (response) => {
      const chunks = [];
      let size = 0;
      response.on('data', (chunk) => {
        size += chunk.length;
        if (size > maxBytes) request.destroy(Object.assign(new Error('response too large'), { code: 'ERESPONSE_TOO_LARGE' }));
        else chunks.push(chunk);
      });
      response.on('error', reject);
      response.on('end', () => resolve({ status: response.statusCode ?? 0, headers: response.headers, body: Buffer.concat(chunks) }));
    });
    request.on('error', reject);
    if (payload) request.end(payload);
    else request.end();
  });
}

// GET /agent/listener with a fresh challenge: {status, proof} or {error}.
function askListener(url, challenge, timeoutMs, signal, agent) {
  return new Promise((resolve) => {
    const client = url.protocol === 'https:' ? https : http;
    const limit = AbortSignal.timeout(Math.max(1, timeoutMs));
    const request = client.request(url, {
      method: 'GET',
      ...(agent ? { agent } : {}),
      signal: signal ? AbortSignal.any([limit, signal]) : limit,
      headers: { accept: 'application/json', 'x-quotemem-challenge': challenge },
    }, (response) => {
      const chunks = [];
      let size = 0;
      response.on('data', (chunk) => {
        size += chunk.length;
        if (size <= 4096) chunks.push(chunk);
      });
      response.on('error', (error) => resolve({ error }));
      response.on('end', () => {
        let proof = null;
        try { proof = JSON.parse(Buffer.concat(chunks).toString('utf8'))?.proof ?? null; } catch { proof = null; }
        resolve({ status: response.statusCode ?? 0, proof: typeof proof === 'string' ? proof : null });
      });
    });
    request.on('error', (error) => resolve({ error }));
    request.end();
  });
}

/**
 * Before the token or content goes to a service on this computer (onThisComputer):
 * whether the program listening on its port holds the token (see the top of
 * this file), checked anew for every request. The check keeps to
 * `timeoutMs` and stops when `signal` aborts. With `connection`
 * (listenerConnection) it goes on that connection, and the request it
 * guards must go there too (requestOn). A URL for another computer is not
 * checked.
 * @returns {Promise<{ok: true, url: string}
 *   | {ok: false, kind: 'older'|'foreign'|'no-token'|'no-answer', why: string}
 *   | {ok: false, refused: Error, why: string}>}
 *   never rejects. `url`: the base URL the request must use (pinnedUrl).
 *   `kind`: `older`, it does not answer the check (a service from before it,
 *   or another program); `foreign`, it answered without this user's token
 *   (another user's service or program, or a service with another token);
 *   `no-token`, no token is known for it yet; `no-answer`, it stalled or
 *   broke off. `refused`: nothing listens; no request may be sent, and the
 *   caller treats it as its own refused connection.
 */
export async function checkListener(settings, { timeoutMs = LISTENER_TIMEOUT_MS, signal, connection = null } = {}) {
  if (!settings?.url || !onThisComputer(settings.url)) return { ok: true, url: settings?.url };
  const url = pinnedUrl(settings.url);
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: true, url };
  }
  const port = Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80));
  const address = listenerAddress(parsed.hostname);
  const challenge = randomBytes(16).toString('hex');
  const answer = await askListener(new URL(url + LISTENER_ROUTE), challenge, timeoutMs, signal, connection?.agent);
  if (answer.error) {
    if (refused(answer.error)) return { ok: false, refused: answer.error, why: 'nothing listens on its port' };
    return { ok: false, kind: 'no-answer', why: `it did not answer the check (${errKind(answer.error)})` };
  }
  if (!settings.token) {
    return { ok: false, kind: 'no-token', why: 'no token is known for it yet (the service creates <data-dir>/token on its first start), so no program on its port can be checked' };
  }
  if (answer.status === 200) {
    const want = Buffer.from(listenerProof(settings.token, address, port, challenge), 'utf8');
    const got = Buffer.from(String(answer.proof ?? ''), 'utf8');
    if (got.length === want.length && timingSafeEqual(got, want)) return { ok: true, url };
  }
  if (answer.status === 200 || answer.status === 400) {
    return { ok: false, kind: 'foreign', why: 'it answered the check without this user\'s token: another user\'s QuoteMem or program, or a service with another token, holds the port' };
  }
  return { ok: false, kind: 'older', why: `it does not answer the check (${answer.status}): a QuoteMem service from before the check (0.3.2 or older) answers this way, as may another program` };
}

/**
 * For `quotemem doctor` and `status`: a refusal of checkListener in words,
 * with what to do. It reads the owner of the port (portOwnerAsync; seconds
 * on Windows, which a hook never waits for) to tell this user's own older
 * service or another token from another user's program, and service.lock
 * (olderState) to tell one the plugins started (`npx quotemem stop`) from
 * one the user started. `ownerOf` is for tests.
 */
export async function describeRefusal(settings, verdict, { env = process.env, ownerOf = portOwnerAsync } = {}) {
  const head = `no token and no memory data were sent to ${settings.url}: the program on its port is not shown to be this user's QuoteMem service: ${verdict.why}.`;
  const ownPort = `QUOTEMEM_URL=http://127.0.0.1:${userPort(env)} in ~/.quotemem/env and in the environment Claude Code, Codex and DSH start in (its plugins then start its own service there)`;
  if (verdict.kind === 'no-token' || verdict.kind === 'no-answer') return head;
  let owner = null;
  try {
    const parsed = new URL(pinnedUrl(settings.url));
    owner = await ownerOf(Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80)), listenerAddress(parsed.hostname), { timeoutMs: OWNER_LOOKUP_MS });
  } catch {
    owner = null;
  }
  // Who can stop it: the plugins (service.lock records it), else the user.
  const restart = olderState(settings) === 'older'
    ? 'restart it once with the current release: `npx quotemem stop` (the plugins then start the current one)'
    : 'stop the `quotemem serve` you started (in its terminal or service manager; no plugin started it, so `npx quotemem stop` cannot) and start the current release, for example `npx quotemem serve` or `uvx quotemem@latest serve`';
  if (owner?.mine) {
    return verdict.kind === 'older'
      ? `${head} It runs as this user: if it is a QuoteMem service from before an upgrade, ${restart}.`
      : `${head} It runs as this user, so its token differs from this user's: use the service's own token (its <data-dir>/token, or the QUOTEMEM_TOKEN it was started with) in QUOTEMEM_TOKEN_FILE or QUOTEMEM_TOKEN.`;
  }
  if (owner) return `${head} It runs as another user (${owner.owner}): two people signed in at once need a port each. Give this user its own: ${ownPort}.`;
  return `${head} Its owner could not be read. If it is your own QuoteMem service from before an upgrade, ${restart}. If it is another user's (two people signed in at once), give this user a port of its own: ${ownPort}.`;
}
