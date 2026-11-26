#!/usr/bin/env node
// The idle rule kept by the background flusher: after a hook leaves turns
// that wait for QUOTEMEM_IDLE_FLUSH_MS, one detached flusher per state
// folder wakes when the oldest chat's idle time is over and saves its last
// turns, with no later hook. Every hook call runs in its own short-lived
// Node process, as a Cursor or Claude Code hook does, against a stub QuoteMem.
//
//   node --test integrations/core/test/idle.test.mjs
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, TOKEN, plan, sleep, stubServer, tempDir, testSettings, waitFor } from './helpers.mjs';

const CORE_URL = pathToFileURL(path.join(CORE, 'index.mjs')).href;
const core = await import(CORE_URL);
const { DIRECT_START_FILE, DiskQueue, HookMemory, readJson, runIdleWaiter } = core;
const windows = process.platform === 'win32';
const quiet = { warn() {}, warnOnce() {}, debug() {} };

const root = tempDir('qm-core-idle-');
const servers = new Set();
const stateDirs = new Set();
after(async () => {
  // Stop a flusher a failed test left waiting.
  for (const dir of stateDirs) {
    for (const lock of idleLocks(dir)) {
      const pid = readJson(lock)?.pid;
      if (Number.isInteger(pid) && pid !== process.pid) kill(pid);
    }
  }
  await Promise.all([...servers].map((s) => s.close()));
});
async function serve(handlers, port) {
  const server = await stubServer(handlers, port);
  servers.add(server);
  return server;
}
const stored = () => ({ body: { success: true, outcome: { added: 1 } } });

function kill(pid) {
  try {
    if (windows) execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    else process.kill(pid, 'SIGKILL');
  } catch {
    // gone
  }
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === 'EPERM';
  }
}

// A hook entry point as the hook runner writes it: one call per process,
// then the waiting flusher's start after the answer (armIdle).
const HOOK = path.join(root, 'hook.mjs');
fs.writeFileSync(HOOK, `
import path from 'node:path';
import { HookMemory, fileSink, makeLog, resolveSettings, settleLaunches } from ${JSON.stringify(CORE_URL)};
const settings = resolveSettings({}, process.env);
const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
const memory = new HookMemory({ settings, log, host: 'test-host', idleWait: true });
const [method, input] = process.argv.slice(2);
const out = await memory[method](JSON.parse(input));
process.stdout.write(JSON.stringify(out ?? null));
memory.armIdle();
await settleLaunches(5000);
if (process.env.TEST_HANG === '1') setInterval(() => {}, 1000);
`);

let dirSeq = 0;
function setup(server, env = {}) {
  dirSeq += 1;
  const stateDir = env.QUOTEMEM_STATE_DIR ?? path.join(root, `state-${dirSeq}`);
  stateDirs.add(stateDir);
  const base = {
    PATH: process.env.PATH,
    SystemRoot: process.env.SystemRoot,
    HOME: root, USERPROFILE: root,
    QUOTEMEM_URL: server.url,
    QUOTEMEM_TOKEN: TOKEN,
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_STATE_DIR: stateDir,
    QUOTEMEM_AUTOSTART: '0',
    QUOTEMEM_DEBUG: '1',
    QUOTEMEM_CAPTURE_RETRY_BASE_MS: '10',
    ...env,
  };
  const call = (method, input, extra = {}) => new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [HOOK, method, JSON.stringify(input)], { env: { ...base, ...extra }, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) reject(new Error(`hook exited ${code}: ${err}`));
      else resolve(JSON.parse(out || 'null'));
    });
  });
  // Returns when the reply hook started: the chat's idle time runs from its
  // write, a little later.
  const turn = async (session, n, user, reply = `Noted ${n}.`) => {
    await call('prompt', { session, turn: `p${n}`, text: user });
    const at = Date.now();
    await call('reply', { session, turn: `p${n}`, text: reply });
    return at;
  };
  const queue = new DiskQueue({ stateDir, url: server.url, userId: base.QUOTEMEM_USER_ID, token: base.QUOTEMEM_TOKEN, authMode: 'bearer' }, quiet);
  const log = () => { try { return fs.readFileSync(path.join(stateDir, 'quotemem.log'), 'utf8'); } catch { return ''; } };
  return { stateDir, env: base, call, turn, queue, log, lock: () => readJson(queue.idleLockFile) };
}

function idleLocks(stateDir) {
  const dir = path.join(stateDir, 'idle');
  let names = [];
  try { names = fs.readdirSync(dir); } catch { /* none */ }
  return names.filter((name) => name.endsWith('.lock')).map((name) => path.join(dir, name));
}

const users = (request) => request.body.messages.filter((m) => m.role === 'user').map((m) => m.content);
const exits = (log) => log.split('\n').filter((line) => line.includes('flusher exit'));

// The flusher that waited is gone and left no lock. Without its pid: the
// lock is gone and `count` flushers wrote their last log line (it comes
// after the lock is released).
async function exited(hooks, pid, count = 1) {
  const gone = () => (pid ? !alive(pid) : exits(hooks.log()).length >= count);
  await waitFor(() => !fs.existsSync(hooks.queue.idleLockFile) && gone(), 8000, 'the waiting flusher to exit');
}

test('idle: one flusher wakes after the idle time and saves exactly the pending turns, once, then exits', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  await hooks.turn('A', 1, 'My kite is orange.');
  const last = await hooks.turn('A', 2, 'My bike is blue.');
  const pid = hooks.lock()?.pid;
  assert.ok(Number.isInteger(pid) && alive(pid), 'a flusher waits');
  assert.equal(server.of('/memory/add').length, 0, 'nothing is saved before the idle time');
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  const [add] = server.of('/memory/add');
  assert.ok(add.at - last >= 1200, `saved after the idle time (${add.at - last} ms)`);
  assert.deepEqual(users(add), ['My kite is orange.', 'My bike is blue.']);
  assert.equal(add.body.messages.length, 4, 'the two turns with their replies');
  await exited(hooks, pid);
  await sleep(1300);
  assert.equal(server.of('/memory/add').length, 1, 'nothing is sent twice');
  assert.equal(hooks.queue.entries().length, 0);
  assert.match(hooks.log(), /flusher exit: .*no idle chat left/);
  assert.equal(exits(hooks.log()).length, 1, 'one flusher ran');
});

test('idle: a new turn while the flusher waits moves the time; the same flusher saves both', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  await hooks.turn('A', 1, 'My kite is orange.');
  const pid = hooks.lock()?.pid;
  await sleep(900);
  const last = await hooks.turn('A', 2, 'My bike is blue.');
  assert.equal(hooks.lock()?.pid, pid, 'the flusher that waited still waits; no second one');
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  const [add] = server.of('/memory/add');
  assert.ok(add.at - last >= 1500, `the idle time counts from the last turn (${add.at - last} ms)`);
  assert.deepEqual(users(add), ['My kite is orange.', 'My bike is blue.']);
  await exited(hooks, pid);
  assert.equal(exits(hooks.log()).length, 1);
});

test('idle: three turns before the time go as a batch; the fourth is saved at the idle time; nothing twice', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  for (const n of [1, 2, 3]) await hooks.turn('A', n, `Fact number ${n}.`);
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the batch of three');
  assert.deepEqual(users(server.of('/memory/add')[0]), ['Fact number 1.', 'Fact number 2.', 'Fact number 3.']);
  await hooks.turn('A', 4, 'Fact number 4.');
  await waitFor(() => server.of('/memory/add').length === 2, 6000, 'the idle save');
  assert.deepEqual(users(server.of('/memory/add')[1]), ['Fact number 4.']);
  await exited(hooks, null);
  await sleep(500);
  assert.equal(server.of('/memory/add').length, 2, 'nothing is sent twice');
});

test('idle: three turns and no more: the batch rule sends them, and the waiting flusher exits with nothing to save', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  await hooks.turn('A', 1, 'Fact number 1.');
  const pid = hooks.lock()?.pid;
  assert.ok(pid, 'a flusher waits after the first turn');
  for (const n of [2, 3]) await hooks.turn('A', n, `Fact number ${n}.`);
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the batch of three');
  await exited(hooks, pid);
  await sleep(1000);
  assert.equal(server.of('/memory/add').length, 1);
  assert.deepEqual(users(server.of('/memory/add')[0]), ['Fact number 1.', 'Fact number 2.', 'Fact number 3.']);
});

test('idle: two chats with different idle times are saved each at its own time by one flusher', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  const aLast = await hooks.turn('A', 1, 'Chat A says the kite is orange.');
  const pid = hooks.lock()?.pid;
  await sleep(1000);
  const bLast = await hooks.turn('B', 1, 'Chat B says the bike is blue.');
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'both chats');
  const [first, second] = server.of('/memory/add');
  assert.deepEqual(users(first), ['Chat A says the kite is orange.']);
  assert.deepEqual(users(second), ['Chat B says the bike is blue.']);
  assert.ok(first.at - aLast >= 1500 && first.at < bLast + 1500, `chat A at its own time (${first.at - aLast} ms)`);
  assert.ok(second.at - bLast >= 1500, `chat B at its own time (${second.at - bLast} ms)`);
  await exited(hooks, pid);
  assert.equal(exits(hooks.log()).length, 1, 'one flusher saved both');
});

test('idle: the waiting flusher saves only its own user\'s chats of its own service', async () => {
  const server = await serve({ '/memory/add': stored });
  const elsewhere = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  // In the same state folder, before: a chat of another user and one of the
  // same user on another service, both pending, both with the idle time off
  // in their own hooks (so no flusher of theirs waits).
  await hooks.call('prompt', { session: 'O', turn: 'p1', text: 'Another user owns a red car.' }, { QUOTEMEM_USER_ID: 'someone-else', QUOTEMEM_IDLE_FLUSH_MS: '0' });
  await hooks.call('reply', { session: 'O', turn: 'p1', text: 'Noted.' }, { QUOTEMEM_USER_ID: 'someone-else', QUOTEMEM_IDLE_FLUSH_MS: '0' });
  await hooks.call('prompt', { session: 'S', turn: 'p1', text: 'The other service hears of a green boat.' }, { QUOTEMEM_URL: elsewhere.url, QUOTEMEM_IDLE_FLUSH_MS: '0' });
  await hooks.call('reply', { session: 'S', turn: 'p1', text: 'Noted.' }, { QUOTEMEM_URL: elsewhere.url, QUOTEMEM_IDLE_FLUSH_MS: '0' });
  await hooks.turn('A', 1, 'My kite is orange.');
  const pid = hooks.lock()?.pid;
  assert.ok(Number.isInteger(pid), 'a flusher waits for this user');
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  await exited(hooks, pid);
  const [add] = server.of('/memory/add');
  assert.deepEqual(users(add), ['My kite is orange.']);
  assert.equal(add.body.user_id, 'quei');
  assert.equal(elsewhere.of('/memory/add').length, 0, 'nothing to the other service');
  // Both other chats still hold their turn: their own owners' hooks save it.
  const sessions = path.join(hooks.stateDir, 'sessions');
  const states = fs.readdirSync(sessions).filter((name) => name.endsWith('.json')).map((name) => readJson(path.join(sessions, name)));
  const waiting = states.filter((state) => state?.capture?.pending?.length > 0).map((state) => state.capture.pending.map((m) => m.content).join(' '));
  assert.ok(waiting.some((text) => text.includes('red car')), 'the other user\'s turn waits');
  assert.ok(waiting.some((text) => text.includes('green boat')), 'the other service\'s turn waits');
});

test('idle: one waiting flusher serves the owners of its credential, each with its own settings; another credential gets its own', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  // Same token, another user, who keeps no assistant text; then another user with a token of its own.
  const plain = { QUOTEMEM_USER_ID: 'plain', QUOTEMEM_CAPTURE_ASSISTANT: 'none' };
  const keyed = { QUOTEMEM_USER_ID: 'keyed', QUOTEMEM_TOKEN: 'another-token-2' };
  await hooks.turn('A', 1, 'My kite is orange.');
  const pid = hooks.lock()?.pid;
  assert.ok(Number.isInteger(pid), 'a flusher waits');
  for (const [env, words] of [[plain, 'I keep bees.'], [keyed, 'I sail on Sundays.']]) {
    await hooks.call('prompt', { session: env.QUOTEMEM_USER_ID, turn: 'p1', text: words }, env);
    await hooks.call('reply', { session: env.QUOTEMEM_USER_ID, turn: 'p1', text: `Noted, ${env.QUOTEMEM_USER_ID}.` }, env);
  }
  assert.equal(idleLocks(hooks.stateDir).length, 2, 'one waiting flusher per credential');
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'every idle save');
  const byUser = Object.fromEntries(server.of('/memory/add').map((r) => [r.body.user_id, r]));
  assert.equal(byUser.quei.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(byUser.plain.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(byUser.keyed.headers.authorization, 'Bearer another-token-2', 'its own token, from its own flusher');
  assert.deepEqual(byUser.plain.body.messages.map((m) => m.role), ['user'], 'its own setting: no assistant text');
  assert.deepEqual(byUser.quei.body.messages.map((m) => m.role), ['user', 'assistant']);
  // owner.json keeps an HMAC of the credential under the state folder's own key (0600),
  // never the token nor a bare hash a copy of the folder could test guesses against.
  const owners = path.join(hooks.stateDir, 'owners');
  const bare = createHash('sha256').update(`quotemem-owner\0bearer\0${TOKEN}`).digest('hex');
  for (const name of fs.readdirSync(owners)) {
    const text = fs.readFileSync(path.join(owners, name, 'owner.json'), 'utf8');
    assert.ok(!text.includes(TOKEN) && !text.includes('another-token-2') && !text.includes(bare));
  }
  const key = path.join(hooks.stateDir, 'state.key');
  assert.equal(fs.readFileSync(key).length, 32);
  if (!windows) assert.equal(fs.statSync(key).mode & 0o777, 0o600);
});

test('idle: an owner registered again is not unregistered as idle for a week', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  await hooks.call('prompt', { session: 'W', turn: 'p1', text: 'My kite is orange.' });
  const file = path.join(hooks.queue.ownerDir, 'owner.json');
  const weekAgo = new Date(Date.now() - 8 * 86_400_000);
  fs.utimesSync(file, weekAgo, weekAgo);
  // The same registration again (nothing changed): it still marks the owner in use.
  await hooks.call('reply', { session: 'W', turn: 'p1', text: 'Noted.' });
  assert.ok(Date.now() - fs.statSync(file).mtimeMs < 60_000, 'touched by its hook');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the idle save');
  assert.ok(fs.existsSync(file));
});

test('idle: a chat with a turn in flight keeps that turn; its closed turns go as sweep sends them; the turn goes after its reply', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1000' });
  await hooks.turn('A', 1, 'My kite is orange.');
  // A long agent turn: its prompt, no reply yet.
  await hooks.call('prompt', { session: 'A', turn: 'p2', text: 'Plan my trip to Lisbon.' });
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the closed turn');
  assert.deepEqual(users(server.of('/memory/add')[0]), ['My kite is orange.']);
  await exited(hooks, null);
  await sleep(500);
  assert.equal(server.of('/memory/add').length, 1, 'the turn in flight is left alone');
  await hooks.call('reply', { session: 'A', turn: 'p2', text: 'Here is a plan.' });
  await waitFor(() => server.of('/memory/add').length === 2, 6000, 'the turn after its reply');
  assert.deepEqual(users(server.of('/memory/add')[1]), ['Plan my trip to Lisbon.']);
});

test('idle: the host app closed (its hook process killed) does not stop the waiting flusher', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  await hooks.call('prompt', { session: 'A', turn: 'p1', text: 'My kite is orange.' });
  // The last hook stays up (an app's process tree), then is killed.
  const child = spawn(process.execPath, [HOOK, 'reply', JSON.stringify({ session: 'A', turn: 'p1', text: 'Noted.' })], { env: { ...hooks.env, TEST_HANG: '1' }, stdio: 'ignore' });
  await waitFor(() => Number.isInteger(hooks.lock()?.pid) && hooks.lock().pid !== child.pid, 8000, 'the flusher to take its lock');
  child.kill('SIGKILL');
  await new Promise((resolve) => child.once('exit', resolve));
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  assert.deepEqual(users(server.of('/memory/add')[0]), ['My kite is orange.']);
});

test('idle: hooks at once in five chats start one waiting flusher', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  const chats = ['A', 'B', 'C', 'D', 'E'];
  await Promise.all(chats.map((s) => hooks.call('prompt', { session: s, turn: 'p1', text: `Chat ${s} fact.` })));
  await Promise.all(chats.map((s) => hooks.call('reply', { session: s, turn: 'p1', text: 'Noted.' })));
  await waitFor(() => server.of('/memory/add').length === 5, 8000, 'every chat');
  await exited(hooks, null);
  assert.deepEqual(server.of('/memory/add').map((r) => users(r)[0]).sort(), chats.map((s) => `Chat ${s} fact.`));
  assert.equal(exits(hooks.log()).length, 1, 'one flusher started');
});

test('idle: with QUOTEMEM_IDLE_FLUSH_MS off, or capture off, no flusher waits', async () => {
  const server = await serve({ '/memory/add': stored });
  for (const env of [{ QUOTEMEM_IDLE_FLUSH_MS: 'off' }, { QUOTEMEM_IDLE_FLUSH_MS: '500', QUOTEMEM_CAPTURE_EVERY: 'off' }]) {
    const hooks = setup(server, env);
    await hooks.turn('A', 1, 'My kite is orange.');
    assert.equal(fs.existsSync(hooks.queue.idleLockFile), false, JSON.stringify(env));
    await sleep(1200);
    assert.equal(exits(hooks.log()).length, 0, `no flusher started: ${JSON.stringify(env)}`);
  }
  assert.equal(server.of('/memory/add').length, 0);
  // A flusher started with the idle rule but these settings exits at once.
  for (const extra of [{ idleFlushMs: 0 }, { captureEvery: 0 }]) {
    const settings = await testSettings(server.url, tempDir(), { ...extra, idleWait: true });
    const t0 = Date.now();
    const outcome = await runIdleWaiter(settings, quiet, { env: {} });
    assert.ok(Date.now() - t0 < 2000, outcome);
    assert.equal(fs.existsSync(new DiskQueue(settings, quiet).idleLockFile), false);
  }
});

test('idle: a service that is down: the turns stay queued and are retried as before; one that answers late gets them once', async () => {
  // Busy for the first two tries, then stored: the flusher retries.
  let calls = 0;
  const flaky = await serve({ '/memory/add': () => { calls += 1; return calls <= 2 ? { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy' } } : stored(); } });
  const hooks = setup(flaky, { QUOTEMEM_IDLE_FLUSH_MS: '1000' });
  await hooks.turn('A', 1, 'My kite is orange.');
  await waitFor(() => flaky.of('/memory/add').length === 3, 8000, 'the retries');
  assert.equal(new Set(flaky.of('/memory/add').map((r) => r.headers['idempotency-key'] ?? JSON.stringify(r.body))).size, 1, 'one batch, tried again');
  await exited(hooks, null);
  assert.equal(hooks.queue.entries().length, 0);
  // Nothing listening: the batch stays in the queue, the chat is cut.
  const down = await serve({ '/memory/add': stored });
  const port = down.port;
  await down.close();
  servers.delete(down);
  // Its next try is a minute away: the flusher leaves it to a later hook.
  const offline = setup(down, { QUOTEMEM_IDLE_FLUSH_MS: '1000', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '60000', QUOTEMEM_FLUSHER_MAX_WAIT_MS: '0' });
  await offline.turn('B', 1, 'My bike is blue.');
  await waitFor(() => offline.queue.entries().length === 1, 6000, 'the batch queued');
  await exited(offline, null);
  assert.equal(offline.queue.entries().length, 1, 'the batch waits in the queue');
  // The service is back; a later hook's flusher sends it once its retry is due.
  const back = await serve({ '/memory/add': stored }, port);
  const job = readJson(offline.queue.entries()[0].file);
  fs.writeFileSync(offline.queue.entries()[0].file, JSON.stringify({ ...job, nextAt: 0 }));
  await offline.call('prompt', { session: 'C', turn: 'p1', text: 'Hello.' });
  await waitFor(() => back.of('/memory/add').length === 1, 8000, 'the batch after the service is back');
  assert.deepEqual(users(back.of('/memory/add')[0]), ['My bike is blue.']);
});

test('idle: a forget kept while a chat\'s turn was in flight runs again after that turn is saved at the idle time', async () => {
  const server = await serve({
    '/memory/add': stored,
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: [{ memory_id: '01JX0000000000000000000F0', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  // Chat B states the fact; its turn is still running when chat A forgets it.
  await hooks.call('prompt', { session: 'B', turn: 'p1', text: 'My locker code is HERON.' });
  await hooks.call('prompt', { session: 'A', turn: 'p1', text: 'Please forget my locker code.' });
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'the forget');
  await hooks.call('reply', { session: 'B', turn: 'p1', text: 'Noted.' });
  await hooks.call('reply', { session: 'A', turn: 'p1', text: 'Done.' });
  await waitFor(() => server.of('/memory/add').some((r) => JSON.stringify(r.body).includes('HERON')), 8000, 'chat B saved at the idle time');
  const add = server.of('/memory/add').find((r) => JSON.stringify(r.body).includes('HERON'));
  await waitFor(() => server.of('/memory/forget/prepare').some((r) => r.at >= add.at), 8000, 'the kept forget again');
  const again = server.of('/memory/forget/prepare').filter((r) => r.at >= add.at);
  assert.ok(again.every((r) => r.body.statement === 'my locker code'));
  await exited(hooks, null);
});

test('idle: a flusher killed while it waited leaves a stale lock; the next hook starts a new one that saves every turn', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200' });
  await hooks.turn('A', 1, 'My kite is orange.');
  const pid = hooks.lock()?.pid;
  assert.ok(pid);
  kill(pid);
  await waitFor(() => !alive(pid), 5000, 'the flusher to die');
  assert.ok(fs.existsSync(hooks.queue.idleLockFile), 'its lock stays');
  await hooks.turn('A', 2, 'My bike is blue.');
  const next = hooks.lock()?.pid;
  assert.ok(next && next !== pid, 'a new flusher took the lock');
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  assert.deepEqual(users(server.of('/memory/add')[0]), ['My kite is orange.', 'My bike is blue.']);
  await exited(hooks, next);
});

test('idle: a flusher lives a bounded time past the idle time, then hands a chat still in use to a new one', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200', QUOTEMEM_TEST_IDLE_LIFE_MS: '200' });
  await hooks.turn('A', 1, 'My kite is orange.');
  const pid = hooks.lock()?.pid;
  await sleep(700);
  await hooks.turn('A', 2, 'My bike is blue.');
  await waitFor(() => !alive(pid), 6000, 'the first flusher to end its life');
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save by the next one');
  assert.deepEqual(users(server.of('/memory/add')[0]), ['My kite is orange.', 'My bike is blue.']);
  await exited(hooks, null, 2);
  assert.match(hooks.log(), /flusher exit: .*handed over/);
  assert.equal(exits(hooks.log()).length, 2);
});

test('idle: the token is in no argument and no log line; a piped caller sees the hook end while the flusher waits', async () => {
  const server = await serve({ '/memory/add': stored });
  const hooks = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1500' });
  await hooks.call('prompt', { session: 'A', turn: 'p1', text: 'My kite is orange.' });
  const t0 = Date.now();
  await hooks.call('reply', { session: 'A', turn: 'p1', text: 'Noted.' });
  const ms = Date.now() - t0;
  const pid = hooks.lock()?.pid;
  assert.ok(pid && alive(pid), 'the flusher waits');
  assert.ok(ms < 1500, `the hook's pipes closed before the idle time (${ms} ms)`);
  if (!windows) {
    const args = execFileSync('ps', ['-o', 'args=', '-p', String(pid)], { encoding: 'utf8' });
    assert.match(args, /flusher\.mjs/);
    assert.ok(!args.includes(TOKEN), 'no token in its arguments');
  }
  await waitFor(() => server.of('/memory/add').length === 1, 6000, 'the idle save');
  assert.equal(server.of('/memory/add')[0].headers.authorization, `Bearer ${TOKEN}`);
  await exited(hooks, pid);
  const files = fs.readdirSync(hooks.stateDir, { recursive: true }).map((name) => path.join(hooks.stateDir, String(name))).filter((file) => fs.statSync(file).isFile());
  for (const file of files) assert.ok(!fs.readFileSync(file, 'utf8').includes(TOKEN), `no token in ${path.relative(hooks.stateDir, file)}`);
});

test('idle: the Windows way of starting is chosen for win32; without PowerShell a direct start does not wait for idle chats', { skip: windows && 'not on Windows (it would start through PowerShell)' }, async () => {
  const server = await serve({ '/memory/add': stored });
  const stateDir = tempDir();
  stateDirs.add(stateDir);
  const IDLE_MS = 3000;
  const settings = await testSettings(server.url, stateDir, {}, { QUOTEMEM_IDLE_FLUSH_MS: String(IDLE_MS), QUOTEMEM_DEBUG: '1' });
  const logOf = () => { try { return fs.readFileSync(path.join(stateDir, 'quotemem.log'), 'utf8'); } catch { return ''; } };
  const said = (sync) => `Started ${sync ? 'at once' : 'later'}.`;
  const saved = (text) => server.of('/memory/add').some((r) => users(r)[0] === text);
  for (const sync of [true, false]) {
    const lines = [];
    const log = { warn() {}, warnOnce(_key, message) { lines.push(message); }, debug() {} };
    const memory = new HookMemory({ settings, log, host: 'test-host', idleWait: true });
    await memory.prompt({ session: `W${sync}`, turn: 'p1', text: said(sync) });
    await memory.reply({ session: `W${sync}`, turn: 'p1', text: 'Noted.' });
    const at = Date.now();
    fs.rmSync(path.join(stateDir, DIRECT_START_FILE), { force: true });
    const missing = path.join(root, 'no-powershell.exe');
    const launch = { platform: 'win32', powershell: missing };
    assert.equal(memory.queue.startFlusher({ env: process.env, sync, idle: 'pending', launch }), true);
    await core.settleLaunches(5000);
    assert.ok(lines.some((line) => /PowerShell could not start a background process/.test(line)), 'PowerShell was tried');
    assert.ok(fs.existsSync(path.join(stateDir, DIRECT_START_FILE)), 'the direct start is recorded');
    // It holds this process's handles: it sends the queue and exits, and
    // gives the idle lock back, long before the idle time is over.
    const given = () => !fs.existsSync(memory.queue.idleLockFile) && /started directly: sends the queue only/.test(logOf());
    await waitFor(given, IDLE_MS - 1000, 'the direct start to give the idle wait back');
    assert.ok(Date.now() - at < IDLE_MS, 'no process waits for the idle time');
    assert.equal(saved(said(sync)), false, 'the idle chat is not saved by the direct start');
    // The next hook after the idle time saves it.
    await sleep(Math.max(0, at + IDLE_MS + 200 - Date.now()));
    assert.equal(await memory.sweep(), 1);
    await core.runFlusher(settings, quiet);
    assert.equal(saved(said(sync)), true, 'a later hook saves the idle chat');
  }
  // On other platforms the plain detached start is used: no PowerShell, nothing recorded,
  // and the started flusher waits for the idle chat and saves it.
  fs.rmSync(path.join(stateDir, DIRECT_START_FILE), { force: true });
  const memory = new HookMemory({ settings, log: quiet, host: 'test-host', idleWait: true });
  await memory.prompt({ session: 'P', turn: 'p1', text: 'Started plainly.' });
  await memory.reply({ session: 'P', turn: 'p1', text: 'Noted.' });
  const launch = { platform: 'linux', powershell: path.join(root, 'no-powershell.exe') };
  // A mark of a direct start in this process's environment is not passed on.
  const env = { ...process.env, [core.STARTED_DIRECTLY_VAR]: '1' };
  assert.equal(memory.queue.startFlusher({ env, sync: true, idle: 'pending', launch }), true);
  await waitFor(() => saved('Started plainly.'), IDLE_MS + 6000, 'the idle save');
  assert.equal(fs.existsSync(path.join(stateDir, DIRECT_START_FILE)), false);
});
