#!/usr/bin/env node
// No token and no memory data go to a service on this computer unless the program on its port
// proves it holds the token (core/listener.mjs, GET /agent/listener).
// Another user's program that took the port first gets no token and no data,
// and so does a service from before the check.
//
//   node --test integrations/core/test/listener.test.mjs
import assert from 'node:assert/strict';
import dnsPromises from 'node:dns/promises';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, TOKEN, stubServer, tempDir, testSettings } from './helpers.mjs';

const core = await import(pathToFileURL(path.join(CORE, 'index.mjs')).href);
const { checkListener, describeRefusal, listenerProof, onThisComputer, pinnedUrl, postJson } = core;
const servers = [];
after(async () => { await Promise.all(servers.map((s) => s.close())); });

/** A program on a loopback port: it records what reaches it; `listener(req, port)` answers. */
async function program(listener, host = '127.0.0.1', port = 0) {
  const seen = [];
  const server = http.createServer((req, res) => {
    let size = 0;
    req.on('data', (chunk) => { size += chunk.length; });
    req.on('end', async () => {
      seen.push({ url: req.url, headers: req.headers, size });
      const answer = await listener(req, server.address().port);
      if (answer === null) return; // never answers
      res.writeHead(answer.status, { 'content-type': 'application/json', ...(req.url === '/agent/listener' ? {} : { connection: 'close' }) });
      res.end(JSON.stringify(answer.body ?? {}));
    });
  });
  await new Promise((resolve) => server.listen(port, host, resolve));
  const handle = {
    url: `http://${host.includes(':') ? `[${host}]` : host}:${server.address().port}`,
    port: server.address().port,
    seen,
    tokenSeen: () => seen.some((r) => JSON.stringify(r.headers).includes(TOKEN)),
    onlyChecks: () => seen.every((r) => r.url === '/agent/listener' && r.size === 0 && r.headers.authorization === undefined),
    close: () => new Promise((resolve) => { server.closeAllConnections?.(); server.close(() => resolve()); }),
  };
  servers.push(handle);
  return handle;
}

const local = async (url, extra = {}, env = {}) => testSettings(url, tempDir(), extra, env);
const challengeOf = (req) => String(req.headers['x-quotemem-challenge']);
/** A current service's answer: its listener_proof for the address and port it listens on. */
const proving = (address) => (req, port) => ({ status: 200, body: { proof: listenerProof(TOKEN, address, port, challengeOf(req)) } });
const quiet = () => {
  const lines = [];
  return { lines, log: { warnOnce: (_key, line) => lines.push(line), warn: (line) => lines.push(line), debug() {} } };
};
const add = { user_id: 'quei', messages: [{ role: 'user', content: 'My PIN is 4711.' }] };

test('the user\'s own service proves the token for its address and port; the request goes, once checked', async () => {
  const server = await stubServer({ '/memory/search': () => ({ body: { data: [] } }) });
  servers.push(server);
  const settings = await local(server.url);
  assert.ok(settings.localService, 'a loopback URL is a service on this computer');
  const answer = await postJson(settings, '/memory/search', { user_id: 'quei', query: 'x' });
  assert.equal(answer.status, 200);
  assert.equal(server.of('/memory/search')[0].headers.authorization, `Bearer ${TOKEN}`);
  // The service's listener_proof (gateway.py): HMAC-SHA256 under the token.
  assert.equal(listenerProof('t', '127.0.0.1', 1, 'ab'), '72352e279e06dcf5659876e33f053d90a5564135edc9e37941c04bbef440cf1f');
});

test('a program on the port that does not prove the token gets no token and no content', async () => {
  // The real service on 127.0.0.2, at the port another program took on 127.0.0.1.
  let real = null;
  const cases = {
    'a wrong proof': ['foreign', () => ({ status: 200, body: { proof: '0'.repeat(64) } })],
    'a proof for another port': ['foreign', (req, port) => ({ status: 200, body: { proof: listenerProof(TOKEN, '127.0.0.1', port + 1, challengeOf(req)) } })],
    // The real service elsewhere answers the challenge it is handed: its proof names its own address.
    'a relayed proof': ['foreign', async (req) => {
      const relayed = await fetch(`http://127.0.0.2:${real.port}/agent/listener`, { headers: { 'x-quotemem-challenge': challengeOf(req) } });
      return { status: 200, body: await relayed.json() };
    }],
    'a refusal of the check': ['foreign', () => ({ status: 400, body: { code: 'invalid_request' } })],
    // A service from before the check answers 404, 401 or 503; so may anything else.
    'a 404': ['older', () => ({ status: 404, body: { code: 'not_found' } })],
    'a 401': ['older', () => ({ status: 401, body: { code: 'authentication_failed' } })],
    'a 503': ['older', () => ({ status: 503, body: { code: 'service_unhealthy' } })],
  };
  for (const [name, [kind, listener]] of Object.entries(cases)) {
    const squat = await program(listener);
    if (name === 'a relayed proof') real = await program(proving('127.0.0.2'), '127.0.0.2', squat.port);
    const settings = await local(squat.url);
    const { lines, log } = quiet();
    Object.defineProperty(settings, 'autostartLog', { value: log });
    const verdict = await checkListener(settings);
    assert.equal(verdict.ok, false, name);
    assert.equal(verdict.kind, kind, name);
    await assert.rejects(postJson(settings, '/memory/add', add), (error) => error.code === 'ELISTENER' && error.kind === kind
      && error.autostart === (kind === 'older' ? 'older-hand' : 'listener'), name);
    // A queued batch is held, not counted as sent.
    assert.deepEqual(await core.sendAdd(settings, { request_id: 'r1', body: add }), { kind: 'held', why: `listener ${kind}` }, name);
    assert.match(lines.join('\n'), /no token and no memory data were sent to .*is not shown to be this user's QuoteMem service/, name);
    assert.equal(squat.tokenSeen(), false, `${name}: the token never reached it`);
    assert.ok(squat.onlyChecks(), `${name}: only checks reached it, ${JSON.stringify(squat.seen.map((r) => r.url))}`);
    await squat.close();
  }
  assert.equal(real.tokenSeen(), false);
  assert.ok(real.seen.length >= 1 && real.onlyChecks(), 'only relayed checks reached the real service');
});

test('a program that stalls on the check gets no token and no content, and the caller does not wait past its time', async () => {
  const stall = await program(() => null);
  const settings = await local(stall.url);
  const t0 = Date.now();
  const verdict = await checkListener(settings, { timeoutMs: 300 });
  assert.equal(verdict.kind, 'no-answer');
  assert.ok(Date.now() - t0 < 2000);
  await assert.rejects(postJson(settings, '/memory/add', add, AbortSignal.timeout(300)), (error) => error.code === 'ELISTENER');
  assert.ok(stall.onlyChecks());
});

test('without a token only the check reaches a program on the port; with nothing listening the service may start', async () => {
  const squat = await program(() => ({ status: 200, body: { data: [] } }));
  // No token exists yet (a service set up but never started); no real profile's token is read here.
  const settings = await local(squat.url, { token: '', localService: null });
  const { log } = quiet();
  Object.defineProperty(settings, 'autostartLog', { value: log });
  await assert.rejects(postJson(settings, '/memory/add', add), (error) => error.code === 'ELISTENER' && error.kind === 'no-token');
  await assert.rejects(postJson(settings, '/memory/search', { user_id: 'quei', query: 'my passport number' }), (error) => error.code === 'ELISTENER');
  assert.ok(squat.seen.length >= 1 && squat.onlyChecks(), JSON.stringify(squat.seen.map((r) => [r.url, r.size])));
  const { url } = squat;
  await squat.close();
  await assert.rejects(postJson(await local(url, { token: '', localService: null }), '/memory/add', add), (error) => error.code === 'ECONNREFUSED');
});

test('every URL that reaches this computer is checked: 127.x, 0.0.0.0, [::], localhost, [::1], IPv4-mapped', async () => {
  for (const url of ['http://localhost:8765', 'http://localhost.:8765', 'http://app.localhost:8765', 'http://LOCALHOST:8765', 'http://127.0.0.1:8765', 'http://127.0.0.2:8765', 'http://0.0.0.0:8765', 'http://[::]:8765', 'http://[::1]:8765',
    'http://[::ffff:7f00:1]:8765', 'http://[::ffff:127.0.0.1]:8765', 'http://[0:0:0:0:0:ffff:7f00:2]:8765', 'http://[::ffff:0:0]:8765']) {
    assert.equal(onThisComputer(url), true, url);
  }
  for (const url of ['http://memory.example:8765', 'https://memory.example.com', 'http://localhost.example:8765', 'http://[::ffff:a00:1]:8765', 'not a url']) {
    assert.equal(onThisComputer(url), false, url);
  }
  assert.equal(pinnedUrl('http://0.0.0.0:8765/x'), 'http://127.0.0.1:8765/x');
  assert.equal(pinnedUrl('http://localhost.:8765/x'), 'http://127.0.0.1:8765/x');
  assert.equal(pinnedUrl('http://app.localhost:8765'), 'http://127.0.0.1:8765');
  assert.equal(pinnedUrl('http://[::]:8765'), 'http://[::1]:8765');
  assert.equal(pinnedUrl('http://127.0.0.2:8765'), 'http://127.0.0.2:8765');
  assert.equal(pinnedUrl('http://[::ffff:7f00:1]:8765/x'), 'http://127.0.0.1:8765/x');
  assert.equal(pinnedUrl('http://[::ffff:0:0]:8765'), 'http://127.0.0.1:8765');
  // A host not written in its normal form is rebuilt from the parsed URL.
  assert.equal(pinnedUrl('http://[::ffff:127.0.0.1]:23460'), 'http://127.0.0.1:23460');
  assert.equal(pinnedUrl('http://[::FFFF:127.0.0.1]:23460/base?x=1#y'), 'http://127.0.0.1:23460/base?x=1#y');
  assert.equal(pinnedUrl('http://0:23460'), 'http://127.0.0.1:23460');
  assert.equal(pinnedUrl('http://127.1:23460'), 'http://127.0.0.1:23460');
  assert.equal(pinnedUrl('http://127.1:23460/'), 'http://127.0.0.1:23460/');
  assert.equal(pinnedUrl('http://localhost:23460?x=1'), 'http://127.0.0.1:23460/?x=1');
  // An IPv4-mapped URL is checked at the IPv4 address it reaches.
  const mappedSquat = await program(() => ({ status: 404, body: {} }));
  const mapped = await local(`http://[::ffff:7f00:1]:${mappedSquat.port}`, {}, { QUOTEMEM_TOKEN: TOKEN });
  await assert.rejects(postJson(mapped, '/memory/add', add), (error) => error.code === 'ELISTENER');
  assert.equal(mappedSquat.tokenSeen(), false);
  assert.ok(mappedSquat.onlyChecks() && mappedSquat.seen.length >= 1);
  // A program on 127.0.0.2 (not a "local service" to autostart) that cannot prove the token.
  const squat = await program(() => ({ status: 404, body: {} }), '127.0.0.2');
  const settings = await local(squat.url, {}, { QUOTEMEM_TOKEN: TOKEN });
  assert.equal(settings.localService, null);
  await assert.rejects(postJson(settings, '/memory/add', add), (error) => error.code === 'ELISTENER');
  assert.equal(squat.tokenSeen(), false);
  assert.ok(squat.onlyChecks());
  // 0.0.0.0 reaches the user's service at 127.0.0.1, where the check went.
  const server = await stubServer({ '/memory/search': () => ({ body: { data: [] } }) });
  servers.push(server);
  const any = await local(`http://0.0.0.0:${server.port}`, {}, { QUOTEMEM_TOKEN: TOKEN });
  assert.equal((await postJson(any, '/memory/search', {})).status, 200);
  assert.equal(server.of('/memory/search')[0].headers.host, `127.0.0.1:${server.port}`);
  // So does an IPv4-mapped URL as written by a caller (resolveSettings would normalize it).
  assert.equal((await postJson({ ...any, url: `http://[::ffff:127.0.0.1]:${server.port}` }, '/memory/search', {})).status, 200);
  assert.equal(server.of('/memory/search')[1].headers.host, `127.0.0.1:${server.port}`);
  // A service elsewhere is not checked: nothing is asked of it first.
  assert.deepEqual(await checkListener({ ...settings, url: 'http://memory.example:1' }), { ok: true, url: 'http://memory.example:1' });
});

test('localhost is 127.0.0.1 for the check and the request: a program on ::1 gets nothing', async () => {
  assert.equal(pinnedUrl('http://localhost:8765'), 'http://127.0.0.1:8765');
  assert.equal(pinnedUrl('http://LOCALHOST:8765/base'), 'http://127.0.0.1:8765/base');
  assert.equal(pinnedUrl('http://127.0.0.1:8765'), 'http://127.0.0.1:8765');
  assert.equal(pinnedUrl('http://[::1]:8765'), 'http://[::1]:8765');
  assert.equal(core.listenerAddress('localhost'), '127.0.0.1');
  const server = await stubServer({ '/memory/search': () => ({ body: { data: [] } }) });
  servers.push(server);
  // Another program on the other address family, at the same port, where a resolver may say localhost is.
  let other = null;
  try {
    other = await program(() => ({ status: 200, body: { proof: '0'.repeat(64), data: [] } }), '::1', server.port);
  } catch {
    // no IPv6 loopback here
  }
  const settings = await local(`http://localhost:${server.port}`);
  // A resolver that lists ::1 first, as macOS and Windows do.
  const lookup = dnsPromises.lookup;
  dnsPromises.lookup = async (host, options) => (host === 'localhost' ? (options?.all ? [{ address: '::1', family: 6 }, { address: '127.0.0.1', family: 4 }] : { address: '::1', family: 6 }) : lookup(host, options));
  let verdict;
  try {
    verdict = await checkListener(settings);
    assert.equal((await postJson(settings, '/memory/search', {})).status, 200);
  } finally {
    dnsPromises.lookup = lookup;
  }
  assert.deepEqual(verdict, { ok: true, url: `http://127.0.0.1:${server.port}` });
  assert.equal(server.of('/memory/search')[0].headers.host, `127.0.0.1:${server.port}`, 'the request went where the check went');
  if (other) assert.equal(other.seen.length, 0, 'nothing reached ::1');
});

test('nothing listening: no request is sent; the call fails as refused', async () => {
  const gone = await program(() => ({ status: 200, body: {} }));
  const { url } = gone;
  await gone.close();
  const settings = await local(url);
  const verdict = await checkListener(settings);
  assert.equal(verdict.ok, false);
  assert.equal(verdict.refused?.code, 'ECONNREFUSED');
  await assert.rejects(postJson(settings, '/memory/search', {}), (error) => error.code === 'ECONNREFUSED');
});

test('every request is checked: a program that takes the port after the service stopped gets no token and no content', async () => {
  let checks = 0;
  const service = await program((req, port) => {
    if (req.url === '/agent/listener') {
      checks += 1;
      return { status: 200, body: { proof: listenerProof(TOKEN, '127.0.0.1', port, challengeOf(req)) } };
    }
    return { status: 200, body: { data: [] } };
  });
  const settings = await local(service.url, {}, { QUOTEMEM_TOKEN: TOKEN });
  assert.equal((await postJson(settings, '/memory/search', {})).status, 200);
  assert.equal((await postJson(settings, '/memory/search', {})).status, 200);
  assert.equal(checks, 2, 'one check per request: no verdict is kept');
  // The service stops; another program takes its port at once.
  const port = service.port;
  await service.close();
  const squat = await program(() => ({ status: 404, body: {} }), '127.0.0.1', port);
  await assert.rejects(postJson(settings, '/memory/add', add), (error) => error.code === 'ELISTENER');
  assert.equal(squat.tokenSeen(), false);
  assert.ok(squat.onlyChecks());
});

test('a forget is held, not sent and not marked as maybe done, while the listener fails the check', async () => {
  const squat = await program(() => ({ status: 404, body: {} }));
  const settings = await local(squat.url);
  const result = await core.runForget(settings, { user_id: 'quei', statement: 'my PIN' });
  assert.equal(result.status, 'retry');
  assert.equal(result.held, true);
  assert.equal(result.maybeDone, undefined);
  assert.ok(squat.onlyChecks());
});

test('doctor\'s explanation of a refusal reads the owner of the port: this user\'s older service, another token, another user', async () => {
  const home = tempDir();
  const settings = await local('http://127.0.0.1:8765', {}, { HOME: home, USERPROFILE: home });
  const older = { ok: false, kind: 'older', why: 'it does not answer the check (404)' };
  const foreign = { ok: false, kind: 'foreign', why: 'it answered the check without this user\'s token' };
  const mine = async () => ({ mine: true, owner: 'uid 1000' });
  const theirs = async () => ({ mine: false, owner: 'uid 1001' });
  const unread = async () => null;
  // Started by hand (service.lock records none): only the user can stop it.
  const byHand = await describeRefusal(settings, older, { ownerOf: mine });
  assert.match(byHand, /^no token and no memory data were sent to http:\/\/127\.0\.0\.1:8765: .*It runs as this user: if it is a QuoteMem service from before an upgrade, stop the `quotemem serve` you started .* start the current release, for example `npx quotemem serve` or `uvx quotemem@latest serve`/);
  // Started by a plugin (service.lock records it at this URL): `npx quotemem stop` stops it.
  const { bootTime } = await import(pathToFileURL(path.join(CORE, 'proc.mjs')).href);
  fs.mkdirSync(settings.localService.dir, { recursive: true });
  fs.writeFileSync(path.join(settings.localService.dir, 'service.lock'), JSON.stringify({ pid: process.pid, startedAt: Date.now() - 60_000, nonce: 'n-listener', url: settings.url, release: '0.3.1', boot: bootTime() }), { mode: 0o600 });
  assert.match(await describeRefusal(settings, older, { ownerOf: mine }), /It runs as this user: if it is a QuoteMem service from before an upgrade, restart it once with the current release: `npx quotemem stop`/);
  fs.rmSync(path.join(settings.localService.dir, 'service.lock'));
  assert.match(await describeRefusal(settings, foreign, { ownerOf: mine }), /It runs as this user, so its token differs from this user's/);
  assert.match(await describeRefusal(settings, foreign, { ownerOf: theirs }), /It runs as another user \(uid 1001\): two people signed in at once need a port each\. Give this user its own: QUOTEMEM_URL=http:\/\/127\.0\.0\.1:2\d{4} /);
  const both = await describeRefusal(settings, older, { ownerOf: unread });
  assert.match(both, /Its owner could not be read\. If it is your own QuoteMem service from before an upgrade, stop the `quotemem serve`.*If it is another user's .* give this user a port of its own/);
  // The lookup gets doctor's own time, not a hook's.
  let given = null;
  await describeRefusal(settings, older, { ownerOf: async (_port, _address, options) => { given = options.timeoutMs; return null; } });
  assert.equal(given, core.OWNER_LOOKUP_MS);
});

test('a held batch keeps its attempts and is not marked as maybe sent', async () => {
  const handlers = { '/agent/listener': () => ({ status: 404, body: {} }), '/memory/add': () => ({ body: { success: true } }) };
  const server = await stubServer(handlers);
  servers.push(server);
  const dir = tempDir();
  const { HookMemory, resolveSettings } = core;
  const settings = resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: dir, QUOTEMEM_CAPTURE_EVERY: '1' });
  const memory = new HookMemory({ settings, log: { warn() {}, warnOnce() {}, debug() {} }, host: 'test-host' });
  memory.flush = () => false;
  await memory.prompt({ session: 'L', turn: 'l1', text: 'I live in Porto.' });
  await memory.reply({ session: 'L', turn: 'l1', text: 'Noted.' });
  await core.runFlusher(settings, { warn() {}, warnOnce() {}, debug() {} }, { signal: AbortSignal.timeout(1500) });
  const queued = (folder) => fs.readdirSync(folder, { recursive: true }).filter((name) => /queue[\\/].*\.json$/.test(name) && !/state\.json$/.test(name));
  const files = queued(dir);
  assert.ok(files.length >= 1, 'the batch is still queued');
  const job = JSON.parse(fs.readFileSync(path.join(dir, files[0]), 'utf8'));
  assert.equal(job.attempts, 0, JSON.stringify(job));
  assert.ok(!job.maybeSent, 'not marked as maybe sent');
  assert.ok(job.held >= 1);
  assert.equal(server.of('/memory/add').length, 0);
  // The service proves itself: the batch goes.
  delete handlers['/agent/listener'];
  assert.equal(await core.runFlusher(settings, { warn() {}, warnOnce() {}, debug() {} }, { signal: AbortSignal.timeout(20_000) }), 'empty');
  assert.equal(server.of('/memory/add').length, 1);
});

test('a session that asks nothing of memory is told on its first prompt; its batch outlives the age limit while held and goes once the service proves itself', async () => {
  // An older service the user started by hand (no service.lock): it answers 404 to the check.
  const handlers = { '/agent/listener': () => ({ status: 404, body: {} }), '/memory/add': () => ({ body: { success: true } }) };
  const server = await stubServer(handlers);
  servers.push(server);
  const dir = tempDir();
  const home = tempDir();
  const { HookMemory, MEMORY_OFF_HEADER, resolveSettings } = core;
  const settings = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: dir, QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '50' });
  // The setting's floor is a minute; the queue honours any value.
  settings.captureMaxAgeMs = 1500;
  const log = { warn() {}, warnOnce() {}, debug() {} };
  const memory = new HookMemory({ settings, log, host: 'test-host' });
  memory.flush = () => false;
  const answers = [];
  for (const [turn, text] of [['n1', 'I moved to Porto last week.'], ['n2', 'My sister is called Ana.']]) {
    answers.push(await memory.prompt({ session: 'N', turn, text }));
    await memory.reply({ session: 'N', turn, text: 'Noted.' });
  }
  assert.equal(answers[0].kind, 'setup', JSON.stringify(answers[0]));
  assert.ok(answers[0].context.startsWith(MEMORY_OFF_HEADER), answers[0].context);
  assert.match(answers[0].context, /not started by the agents.*started yourself from an older release, stop it .*`npx quotemem stop` cannot.*`uvx quotemem@latest serve`/s);
  assert.equal(answers[1].context, null, 'told once per session');
  const queued = () => fs.readdirSync(dir, { recursive: true }).filter((name) => /queue[\\/].*\.json$/.test(name) && !/state\.json$/.test(name));
  // Held past the age limit: still queued, no batch sent.
  const until = Date.now() + 3000;
  while (Date.now() < until) await core.runFlusher(settings, log, { signal: AbortSignal.timeout(Math.max(1, until - Date.now())) });
  const files = queued();
  assert.ok(files.length >= 1, 'the batch is kept');
  const job = JSON.parse(fs.readFileSync(path.join(dir, files[0]), 'utf8'));
  assert.ok(Date.now() - job.created > settings.captureMaxAgeMs && job.held >= 2, JSON.stringify(job));
  assert.equal(server.of('/memory/add').length, 0);
  assert.ok(server.requests.length >= 2 && server.requests.every((r) => r.route === '/agent/listener' && r.headers.authorization === undefined && r.body === null), JSON.stringify(server.requests));
  // The user restarts it with the current release, which proves itself: the batch goes.
  delete handlers['/agent/listener'];
  assert.equal(await core.runFlusher(settings, log, { signal: AbortSignal.timeout(20_000) }), 'empty');
  assert.ok(server.of('/memory/add').length >= 1);
  assert.deepEqual(queued(), []);
});

test('the request goes on the connection the check proved: a program that takes the port between the check and the request gets nothing', async () => {
  // The service answers the check, then stops listening, and another program
  // takes the port at once, before the request goes (as when the service
  // stops or is replaced while a request is under way).
  for (const drop of [false, true]) {
    const seen = [];
    let squat = null;
    const service = http.createServer((req, res) => {
      seen.push(req.url);
      if (req.url !== '/agent/listener') {
        req.resume();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ data: [] }));
        return;
      }
      const port = front.address().port;
      res.on('finish', () => {
        // Stops listening; its open connections stay (as a service that is
        // shutting down still answers what it has).
        front.close();
        squat = program(() => ({ status: 422, body: {} }), '127.0.0.1', port);
        // `drop`: the service also closes the connection the check went on.
        if (drop) req.socket.destroy();
      });
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ proof: listenerProof(TOKEN, '127.0.0.1', port, challengeOf(req)) }));
    });
    const sockets = new Set();
    const front = net.createServer((socket) => {
      sockets.add(socket);
      service.emit('connection', socket);
    });
    await new Promise((resolve) => front.listen(0, '127.0.0.1', resolve));
    servers.push({ close: async () => { for (const socket of sockets) socket.destroy(); front.close(); } });
    const settings = await local(`http://127.0.0.1:${front.address().port}`, {}, { QUOTEMEM_TOKEN: TOKEN });
    if (drop) {
      // Nothing goes on another connection: either the closed connection is
      // seen first, and on a new one the program fails the check (ELISTENER),
      // or the request is written to the closing connection of the service
      // that proved itself, which drops it (a reset; Node 20 is quick to
      // reuse the socket).
      await assert.rejects(postJson(settings, '/memory/search', { query: 'pin' }), (error) => error.code === 'ELISTENER' || error.code === 'ECONNRESET');
    } else {
      assert.equal((await postJson(settings, '/memory/search', { query: 'pin' })).status, 200);
      assert.deepEqual(seen, ['/agent/listener', '/memory/search'], 'the service that proved itself got the request');
    }
    const squatter = await squat;
    assert.equal(squatter.tokenSeen(), false, `drop=${drop}`);
    assert.ok(squatter.onlyChecks(), JSON.stringify(squatter.seen));
  }
});

test('the user\'s own service that stalls, or that cannot be checked yet, is "memory unavailable", not another program; a stalled check counts against recall', async () => {
  const { HookMemory, MEMORY_OFF_HEADER, UNAVAILABLE_HEADER, resolveSettings } = core;
  const log = { warn() {}, warnOnce() {}, debug() {} };
  const session = (url, env = {}) => {
    const home = tempDir();
    const settings = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: tempDir(), QUOTEMEM_AUTOSTART: '0', QUOTEMEM_RECALL_TIMEOUT_MS: '400', ...env });
    const memory = new HookMemory({ settings, log, host: 'test-host' });
    memory.flush = () => false;
    return { settings, memory };
  };
  const question = 'What do you remember about my sister?';
  // A paused or busy service: it takes the connection and never answers.
  const stall = await program(() => null);
  const paused = session(stall.url);
  const answer = await paused.memory.prompt({ session: 'S', turn: 's1', text: question });
  assert.equal(answer.kind, 'unavailable', JSON.stringify(answer));
  assert.ok(answer.context.startsWith(UNAVAILABLE_HEADER), answer.context);
  assert.ok(!answer.context.includes(MEMORY_OFF_HEADER) && !/Another program/.test(answer.context), answer.context);
  assert.ok(stall.onlyChecks());
  // A new session's first prompt that asks nothing of memory waits at most
  // LISTENER_NOTE_MS for its check, which counts as the second timeout in a
  // row: recall's breaker opens.
  const plainPrompt = 'Please rename the variable foo to bar.';
  const later = (name) => {
    const memory = new HookMemory({ settings: paused.settings, log, host: 'test-host' });
    memory.flush = () => false;
    return memory;
  };
  const t1 = Date.now();
  const plain = await later('S2').prompt({ session: 'S2', turn: 't1', text: plainPrompt });
  assert.equal(plain.context, null, JSON.stringify(plain));
  assert.ok(Date.now() - t1 < core.LISTENER_NOTE_MS + 700, `${Date.now() - t1} ms`);
  // While the breaker is open, a new session's first prompt checks nothing.
  const checks = stall.seen.length;
  const third = await later('S3').prompt({ session: 'S3', turn: 't1', text: plainPrompt });
  assert.equal(third.context, null, JSON.stringify(third));
  assert.equal(stall.seen.length, checks, 'no check while recall\'s breaker is open');
  // No token yet for the user's own service: no token and no memory data are sent, and memory is just unavailable.
  const service = await program(proving('127.0.0.1'));
  const tokenless = session(service.url);
  tokenless.settings.token = null;
  const none = await tokenless.memory.prompt({ session: 'T', turn: 't1', text: question });
  assert.ok(!none.context?.includes(MEMORY_OFF_HEADER) && !/Another program/.test(none.context ?? ''), JSON.stringify(none));
  assert.equal(service.tokenSeen(), false);
  assert.ok(service.onlyChecks());
});

test('the owner lookup keeps its time limit when the lookup program leaves a child holding its output', { skip: process.platform === 'win32' }, async () => {
  // A stand-in lsof whose shell waits on a child: stopping the shell leaves
  // the child holding the output pipe, as netstat under PowerShell may.
  const bin = tempDir('qm-owner-bin-');
  fs.writeFileSync(path.join(bin, 'lsof'), '#!/bin/sh\nsleep 8\necho u1\n', { mode: 0o755 });
  const saved = process.env.PATH;
  process.env.PATH = `${bin}${path.delimiter}${saved}`;
  try {
    const started = Date.now();
    assert.equal(await core.portOwnerAsync(23401, '127.0.0.1', { platform: 'darwin', timeoutMs: 300 }), null);
    assert.ok(Date.now() - started < 3000, `took ${Date.now() - started} ms`);
    const aborted = new AbortController();
    const begun = Date.now();
    setTimeout(() => aborted.abort(), 200);
    assert.equal(await core.portOwnerAsync(23401, '127.0.0.1', { platform: 'darwin', timeoutMs: 10_000, signal: aborted.signal }), null);
    assert.ok(Date.now() - begun < 3000, `took ${Date.now() - begun} ms`);
  } finally {
    process.env.PATH = saved;
  }
  // A lookup that finishes in time still gives its answer.
  fs.writeFileSync(path.join(bin, 'lsof'), `#!/bin/sh\necho u${process.getuid()}\n`, { mode: 0o755 });
  process.env.PATH = `${bin}${path.delimiter}${saved}`;
  try {
    assert.deepEqual(await core.portOwnerAsync(23401, '127.0.0.1', { platform: 'darwin', timeoutMs: 3000 }), { mine: true, owner: `uid ${process.getuid()}` });
  } finally {
    process.env.PATH = saved;
  }
});
