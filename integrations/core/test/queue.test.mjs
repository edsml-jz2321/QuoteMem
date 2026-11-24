#!/usr/bin/env node
// The cross-process disk queue, its locks and the background flusher.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, TOKEN, forgetTargetRefusal, memoryLog, plan, sleep, stubServer, tempDir, testSettings, waitFor } from './helpers.mjs';

const core = await import(pathToFileURL(path.join(CORE, 'index.mjs')).href);
const { DiskQueue, PROBE_EVERY_MS, buildJobs, lockHeld, releaseLock, runFlusher, tryLock } = core;

const root = tempDir('qm-core-queue-');
let dirSeq = 0;
const newDir = () => path.join(root, `s${(dirSeq += 1)}`);
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });
async function serve(handlers, port) {
  const server = await stubServer(handlers, port);
  servers.add(server);
  return server;
}

const addJob = (settings, session, n, text = `fact ${n}`) => buildJobs(settings, `test:${session}`, n, n, [{ role: 'user', content: text, timestamp: 1_763_500_000_000 + n }])[0];
// inChat: the service's in-chat check reads the request too (detectForgetRequests).
const forgetJob = (settings, statement, requestId = `test:S:forget:${statement.length}`, { inChat = false } = {}) => ({
  kind: 'forget',
  request_id: requestId,
  body: { schema_version: 1, user_id: settings.userId, client_session_id: 'test:S', request_id: requestId, statement },
  inChat,
  url: settings.url,
  created: Date.now(),
  attempts: 0,
  nextAt: 0,
});

/** Run a small ESM script in a child process; resolves with its exit code. */
function runChild(source, env = {}) {
  const file = path.join(root, `child-${Math.random().toString(16).slice(2)}.mjs`);
  fs.writeFileSync(file, source);
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [file], { stdio: ['ignore', 'inherit', 'inherit'], env: { PATH: process.env.PATH, ...env } });
    child.on('error', reject);
    child.on('exit', (code) => resolve(code));
  });
}
const coreUrl = pathToFileURL(path.join(CORE, 'index.mjs')).href;

// ------------------------------------------------------------------ locks

test('locks: exclusive, stale owners broken, released only by the owner', async () => {
  const file = path.join(newDir(), 'x.lock');
  const a = tryLock(file);
  assert.ok(a);
  assert.equal(tryLock(file), null, 'held');
  assert.ok(lockHeld(file));
  if (process.platform !== 'win32') assert.equal(fs.statSync(file).mode & 0o777, 0o600);
  releaseLock({ file, nonce: 'not-the-owner' });
  assert.ok(fs.existsSync(file), 'a stranger cannot release it');
  releaseLock(a);
  assert.ok(!fs.existsSync(file));

  // A lock whose process is gone is broken.
  fs.writeFileSync(file, JSON.stringify({ host: os.hostname(), boot: Date.now() - os.uptime() * 1000, pid: 999_999, nonce: 'dead' }));
  const b = tryLock(file);
  assert.ok(b, 'dead owner: taken over');
  releaseLock(b);

  // A lock held by another live process is respected until it lets go.
  const holder = runChild(`
    import { tryLock, releaseLock } from ${JSON.stringify(coreUrl)};
    const h = tryLock(${JSON.stringify(file)});
    if (!h) process.exit(3);
    await new Promise((r) => setTimeout(r, 600));
    releaseLock(h);
  `);
  await waitFor(() => fs.existsSync(file), 3000, 'child lock');
  assert.equal(tryLock(file), null, 'live owner in another process');
  assert.equal(await holder, 0);
  const c = tryLock(file);
  assert.ok(c, 'free after the other process released it');
  releaseLock(c);

  // withLock serialises read-modify-write across processes.
  const counter = path.join(newDir(), 'counter.json');
  fs.mkdirSync(path.dirname(counter), { recursive: true });
  fs.writeFileSync(counter, '0');
  const bump = `
    import fs from 'node:fs';
    import { withLock } from ${JSON.stringify(coreUrl)};
    for (let i = 0; i < 25; i += 1) {
      await withLock(${JSON.stringify(`${counter}.lock`)}, async () => {
        const n = Number(fs.readFileSync(${JSON.stringify(counter)}, 'utf8'));
        await new Promise((r) => setImmediate(r));
        fs.writeFileSync(${JSON.stringify(counter)}, String(n + 1));
      }, { timeoutMs: 20000 });
    }
  `;
  const codes = await Promise.all([1, 2, 3, 4].map(() => runChild(bump)));
  assert.deepEqual(codes, [0, 0, 0, 0]);
  assert.equal(fs.readFileSync(counter, 'utf8'), '100', 'no lost update');
});

// ------------------------------------------------------------ the queue

test('queue: concurrent writers in several processes, FIFO per writer, dedupe by request id', async () => {
  const stateDir = newDir();
  const writer = (n) => `
    import { DiskQueue, buildJobs } from ${JSON.stringify(coreUrl)};
    const settings = { stateDir: ${JSON.stringify(stateDir)}, userId: 'quei', url: 'http://127.0.0.1:9' };
    const q = new DiskQueue(settings, { warn() {}, warnOnce() {}, debug() {} });
    for (let i = 0; i < 20; i += 1) {
      q.enqueue(buildJobs(settings, 'test:W${n}', i, i, [{ role: 'user', content: 'w${n} ' + i, timestamp: 1 }]));
    }
  `;
  const codes = await Promise.all([0, 1, 2, 3, 4, 5].map((n) => runChild(writer(n))));
  assert.deepEqual(codes, [0, 0, 0, 0, 0, 0]);
  const settings = { stateDir, userId: 'quei', url: 'http://127.0.0.1:9' };
  const queue = new DiskQueue(settings, memoryLog());
  const entries = queue.entries();
  assert.equal(entries.length, 120, 'every job landed');
  const jobs = entries.map((e) => JSON.parse(fs.readFileSync(e.file, 'utf8')));
  for (let n = 0; n < 6; n += 1) {
    const mine = jobs.filter((j) => j.body.session_id === `test:W${n}`).map((j) => j.body.messages[0].content);
    assert.deepEqual(mine, Array.from({ length: 20 }, (_, i) => `w${n} ${i}`), `writer ${n} in order`);
  }
  assert.equal(fs.readdirSync(path.join(stateDir, 'queue')).filter((n) => n.endsWith('.tmp')).length, 0, 'no temporary files left');
  // The same request id is never queued twice.
  assert.equal(queue.enqueue([jobs[0]]), 0);
  assert.ok(queue.has(jobs[0].request_id));
  if (process.platform !== 'win32') {
    assert.equal(fs.statSync(path.join(stateDir, 'queue')).mode & 0o777, 0o700);
    assert.equal(fs.statSync(entries[0].file).mode & 0o777, 0o600);
  }
});

test('flusher: in order, one at a time, retries with Retry-After, then exits', async () => {
  let calls = 0;
  const server = await serve({
    '/memory/add': () => {
      calls += 1;
      if (calls === 1) return { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy' } };
      return { delay: 30, body: { success: true, outcome: { added: 1 } } };
    },
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2), addJob(settings, 'S', 3)]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.messages[0].content), ['fact 1', 'fact 1', 'fact 2', 'fact 3']);
  // A retry resends the batch as it was queued: each message keeps the time it
  // was said, never the time of the retry (a later dated restatement is what
  // lifts a forget, so a resend must not look newer than it is).
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.messages[0].timestamp), [1_763_500_000_001, 1_763_500_000_001, 1_763_500_000_002, 1_763_500_000_003]);
  assert.deepEqual(server.of('/memory/add')[1].body, server.of('/memory/add')[0].body, 'the retry sends the same body');
  assert.equal(server.stats.maxInflight, 1);
  assert.equal(server.of('/memory/add')[0].headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(server.of('/memory/add')[0].headers.prefer, 'respond-async', 'asks for the early acknowledgement; a 200 (an older service) is stored as before');
  assert.equal(queue.entries().length, 0);
  assert.ok(!fs.existsSync(queue.lockFile), 'lock released');
});

test('flusher: a forget runs only after every batch queued before it is stored', async () => {
  let prepared = null;
  const server = await serve({
    '/memory/add': () => ({ delay: 120, body: { success: true } }),
    '/memory/forget/prepare': (body) => {
      prepared = plan(body);
      return { body: prepared };
    },
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: [{ memory_id: '01JX00000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const job = forgetJob(settings, 'my address');
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2), job]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual(server.requests.map((r) => r.route), ['/memory/add', '/memory/add', '/memory/forget/prepare', '/memory/forget/confirm']);
  const prepare = server.of('/memory/forget/prepare')[0];
  assert.ok(prepare.at >= server.of('/memory/add')[1].at + 100, 'prepare only after the second /memory/add answered');
  assert.deepEqual(prepare.body, job.body);
  assert.deepEqual(server.of('/memory/forget/confirm')[0].body, prepared.confirmation, 'the plan is confirmed exactly as prepared');
  // The single forget pair routes on the one target field each body names:
  // `statement` for the prepare, the plan's `plan_binding` for the confirm.
  assert.deepEqual(Object.keys(prepare.body).sort(), ['client_session_id', 'request_id', 'schema_version', 'statement', 'user_id']);
  const confirm = server.of('/memory/forget/confirm')[0].body;
  assert.ok(Object.hasOwn(confirm, 'plan_binding') && !Object.hasOwn(confirm, 'memory_id'), 'a statement confirmation names no memory_id');
  const result = queue.result(job.request_id);
  assert.equal(result.status, 'forgotten');
  assert.equal(result.forgotten, 1);
});

test('flusher: async Add: acknowledged batches are read again unchanged, the next is sent meanwhile, an in-chat forget waits for them at most 2 s', async () => {
  // The service: the first send of a request id saves it (202); a resend
  // reads it, pending for the first batch's first read, stored after.
  const reads = new Map();
  const stored = [];
  const server = await serve({
    '/memory/add': (body, entry) => {
      const n = (reads.get(body.request_id) ?? 0) + 1;
      reads.set(body.request_id, n);
      if (entry.headers.prefer !== 'respond-async') return { status: 400, body: { code: 'invalid_request' } };
      const waits = body.messages[0].content === 'fact 1' ? 3 : 2;
      if (n < waits) return { status: 202, body: { success: true, status: 'pending', request_id: body.request_id, user_id: body.user_id, session_id: body.session_id } };
      if (n === waits) stored.push(body.request_id);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', outcomes: [{ memory_id: '01JX00000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const jobs = [addJob(settings, 'S', 1), addJob(settings, 'S', 2)];
  const job = forgetJob(settings, 'my address', undefined, { inChat: true });
  queue.enqueue([...jobs, job]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  const adds = server.of('/memory/add');
  assert.deepEqual(adds.slice(0, 2).map((r) => r.body.messages[0].content), ['fact 1', 'fact 2'], 'the second batch went out while the first was pending');
  assert.equal(adds.length, 5, 'fact 1: saved, pending, stored; fact 2: saved, stored');
  for (const add of adds) {
    const original = jobs.find((j) => j.request_id === add.body.request_id);
    assert.deepEqual(add.body, { ...original.body, request_id: original.request_id }, 'every read is the identical request');
  }
  const firstReads = adds.filter((r) => r.body.request_id === jobs[0].request_id).map((r) => r.at);
  assert.ok(firstReads[1] - firstReads[0] >= 900 && firstReads[2] - firstReads[1] >= 1900, 'reads back off from 1 s');
  const prepare = server.of('/memory/forget/prepare')[0];
  const fact2Stored = adds.filter((r) => r.body.request_id === jobs[1].request_id).at(-1).at;
  assert.ok(prepare.at >= fact2Stored, 'the forget waited while batches were completing');
  assert.ok(prepare.at >= job.created + core.FORGET_EARLIER_WAIT_MS - 50, 'for FORGET_EARLIER_WAIT_MS');
  assert.ok(prepare.at < firstReads[2], 'and no longer: fact 1 was stored after it (the service checks it for the request itself)');
  assert.deepEqual(stored.sort(), jobs.map((j) => j.request_id).sort());
  assert.equal(queue.result(job.request_id).status, 'forgotten');
  assert.equal(queue.result(job.request_id).inChat, true, 'its note may say the service still checks the request');
  assert.equal(queue.entries().length, 0);
  assert.ok(lines.some((l) => /pending .* read again in 1000 ms/.test(l)));
});

test('flusher: async Add: a forget only the plugin reads (a retraction) waits until the batches before it are stored', async () => {
  // The ask is saved (202) and stored about 6 s later, as in live runs; the
  // service's in-chat check never reads "forget that I asked you for …", so
  // a forget run before the ask is stored would leave it stored for good.
  let storedAt = 0;
  const server = await serve({
    '/memory/add': (body) => {
      const reads = server.of('/memory/add').filter((r) => r.body.request_id === body.request_id).length;
      if (reads < 4) return { status: 202, body: { success: true, status: 'pending', request_id: body.request_id } };
      storedAt = Date.now();
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', outcomes: [{ memory_id: '01JX00000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const ask = addJob(settings, 'S', 1, 'Can you give me a shell command to count the lines in data.csv?');
  const [request] = core.detectForgetRequests('Please forget that I asked you for a shell command to count the lines in data.csv.');
  assert.deepEqual(request, { statement: 'I asked you for a shell command to count the lines in data.csv', inChat: false, ownTurn: false });
  const job = forgetJob(settings, request.statement, 'r-retract', { inChat: request.inChat });
  queue.enqueue([ask, job]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const prepare = server.of('/memory/forget/prepare');
  assert.equal(prepare.length, 1);
  assert.ok(storedAt > 0 && prepare[0].at >= storedAt, `prepare ran after the ask was stored (${prepare[0].at - job.created} ms vs ${storedAt - job.created} ms)`);
  assert.ok(storedAt - job.created > core.FORGET_EARLIER_WAIT_MS + 1000, 'past the in-chat wait');
  const result = queue.result('r-retract');
  assert.equal(result.status, 'forgotten');
  assert.ok(!result.inChat, 'its note does not say the service may still check it');
});

test('flusher: async Add: an acknowledged batch is never sent under a new id, and expires unconfirmed', async () => {
  let n = 0;
  const server = await serve({
    '/memory/add': () => {
      n += 1;
      if (n === 1) return { status: 202, body: { success: true, status: 'pending' } };
      if (n === 2) return { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy' } };
      return { status: 422, body: { code: 'operation_conflict', retryable: false } };
    },
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  const job = addJob(settings, 'S', 1);
  job.maybeSent = true;
  queue.enqueue([job]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.request_id), [job.request_id, job.request_id, job.request_id], 'no rotated id after the acknowledgement');
  assert.ok(lines.some((l) => /dropped: 422 operation_conflict/.test(l)));

  // An acknowledged batch the service never confirms is dropped at the age limit.
  const silent = await serve({ '/memory/add': () => ({ status: 202, body: { success: true, status: 'pending' } }) });
  const settings2 = await testSettings(silent.url, newDir());
  const queue2 = new DiskQueue(settings2, memoryLog());
  const old = { ...addJob(settings2, 'S', 2), accepted: true, created: Date.now() - settings2.captureMaxAgeMs - 1000 };
  queue2.enqueue([old]);
  const lines2 = [];
  assert.equal(await runFlusher(settings2, memoryLog(lines2)), 'empty');
  assert.equal(silent.of('/memory/add').length, 0);
  assert.ok(lines2.some((l) => /saved by the service but not confirmed stored in time/.test(l)));

  // One whose read keeps hanging is let go after acceptedMaxWaitMs, so the
  // forget behind it runs.
  const hung = await serve({
    '/memory/add': () => ({ delay: 3000, body: { success: true } }),
    '/memory/forget/prepare': () => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }),
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings3 = await testSettings(hung.url, newDir(), { acceptedReadTimeoutMs: 100, acceptedMaxWaitMs: 1500 });
  const queue3 = new DiskQueue(settings3, memoryLog());
  const stuck = { ...addJob(settings3, 'S', 3), accepted: true, acceptedAt: Date.now() };
  const behind = forgetJob(settings3, 'my address');
  queue3.enqueue([stuck, behind]);
  const lines3 = [];
  assert.equal(await runFlusher(settings3, memoryLog(lines3)), 'empty');
  assert.ok(lines3.some((l) => /dropped: saved by the service but not confirmed stored in time/.test(l)));
  assert.equal(queue3.result(behind.request_id).status, 'nothing', 'the forget behind it ran');
});

test('flusher: async Add: a slow read of an acknowledged batch is still pending, not an error', async () => {
  // The service answers the first resend only when the batch is stored, long
  // after; the flusher stops waiting, reads again later, and a forget queued
  // behind (one the service's in-chat check reads too) waits for the batch
  // at most FORGET_EARLIER_WAIT_MS.
  let n = 0;
  const server = await serve({
    '/memory/add': () => {
      n += 1;
      if (n === 1) return { status: 202, body: { success: true, status: 'pending' } };
      if (n === 2) return { delay: 1500, body: { success: true, outcome: { added: 1 } } };
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', outcomes: [{ memory_id: '01JX00000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const settings = await testSettings(server.url, newDir(), { acceptedReadTimeoutMs: 300 });
  const queue = new DiskQueue(settings, memoryLog());
  const job = addJob(settings, 'S', 1);
  const forget = forgetJob(settings, 'my address', undefined, { inChat: true });
  queue.enqueue([job, forget]);
  const lines = [];
  const started = Date.now();
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  const adds = server.of('/memory/add');
  assert.equal(adds.length, 3, 'saved, a slow read given up, then read again');
  assert.ok(adds.every((r) => r.body.request_id === job.request_id), 'never under a new id');
  assert.ok(lines.some((l) => /pending .*; no answer within 300 ms, read again in 2000 ms/.test(l)), lines.join('\n'));
  assert.ok(!lines.some((l) => /retry|dropped/.test(l)), 'not an error');
  const prepareAt = server.of('/memory/forget/prepare')[0].at;
  assert.ok(prepareAt >= forget.created + core.FORGET_EARLIER_WAIT_MS - 50 && prepareAt < adds[2].at, 'the forget waited 2 s, then ran');
  assert.equal(queue.result(forget.request_id).status, 'forgotten');
  assert.ok(Date.now() - started < 8000);
});

test('flusher: async Add: an acknowledged batch is not dropped while the service cannot be read', async () => {
  // Saved (202), then the service restarts: busy, refused connections; it is
  // read until stored, past the attempt limit, and the forget waits for it.
  let n = 0;
  const server = await serve({
    '/memory/add': () => {
      n += 1;
      if (n === 1) return { status: 202, body: { success: true, status: 'pending' } };
      if (n <= 4) return { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_starting' } };
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', outcomes: [{ memory_id: '01JX00000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
  const queue = new DiskQueue(settings, memoryLog());
  const job = addJob(settings, 'S', 1);
  const forget = forgetJob(settings, 'my address');
  queue.enqueue([job, forget]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.equal(server.of('/memory/add').length, 5);
  assert.ok(!lines.some((l) => /dropped/.test(l)), lines.join('\n'));
  assert.ok(lines.some((l) => /retry reading .* \(service_starting\)/.test(l)));
  assert.ok(server.of('/memory/forget/prepare')[0].at >= server.of('/memory/add')[4].at);
  assert.equal(queue.result(forget.request_id).status, 'forgotten');
});

test('flusher: async Add: a saved batch waits while the batches before it complete, and is not dropped meanwhile', async () => {
  // One user's batches complete one at a time: the second is confirmed only
  // after the first. The unconfirmed time counts from the last batch stored.
  const t0 = Date.now();
  const server = await serve({
    '/memory/add': (body) => {
      const ready = body.messages[0].content === 'fact 1' ? 900 : 2600;
      return Date.now() - t0 < ready ? { status: 202, body: { success: true, status: 'pending' } } : { body: { success: true, outcome: { added: 1 } } };
    },
  });
  const settings = await testSettings(server.url, newDir(), { acceptedMaxWaitMs: 2500 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2)]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.ok(!lines.some((l) => /dropped/.test(l)), lines.join('\n'));
  const last = server.of('/memory/add').at(-1);
  assert.equal(last.body.messages[0].content, 'fact 2');
  assert.ok(last.at - t0 > 2500, 'confirmed later than the wait limit after it was saved');
  assert.ok(Number(queue.state().storedAt) >= last.at - 50, 'the last batch stored is recorded');
  for (const add of server.of('/memory/add')) assert.equal(add.headers.prefer, 'respond-async');
});

test('flusher: the service\'s daily spend cap defers a batch until the reset, without using its attempts', async () => {
  const server = await serve({
    '/memory/add': () => ({ status: 429, headers: { 'retry-after': '7200' }, body: { code: 'spend_cap_reached', status: 'deferred', retry_after_seconds: 7200 } }),
  });
  const settings = await testSettings(server.url, newDir(), { captureMaxAttempts: 1, flusherMaxWaitMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1)]);
  const lines = [];
  const before = Date.now();
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'waiting');
  const entries = queue.entries();
  assert.equal(entries.length, 1, 'kept, not dropped after its one attempt');
  const job = core.readJson(entries[0].file);
  assert.equal(job.attempts, 0);
  assert.ok(job.nextAt >= before + 7_200_000 - 1000 && job.nextAt <= Date.now() + 7_200_000, 'sent again at the reset');
  assert.ok(lines.some((l) => /daily spend cap; memory writes wait for its reset/.test(l)));
  assert.equal(queue.startFlusher(), false, 'no flusher is started for a batch due hours later');
  assert.equal(core.deferMs({ retryAfter: '30' }), 60_000, 'at least a minute');
  assert.equal(core.deferMs({ data: { retry_after_seconds: 999_999 } }), 86_400_000, 'at most a day');
});

test('flusher: a batch the service holds back as busy waits without using its attempts; the batches behind it wait for it', async () => {
  // After a forget, the service holds a session's new batch while an earlier
  // one completes (429 delivery_busy). Ten such answers with an attempt limit
  // of 2: the batch is still sent, saved and stored under its request id.
  let busy = 0;
  let saved = false;
  const server = await serve({
    '/memory/add': (body) => {
      if (body.messages[0].content !== 'fact 2') return { body: { success: true, outcome: { added: 1 } } };
      if (busy < 10) {
        busy += 1;
        return { status: 429, headers: { 'retry-after': '0' }, body: { code: 'delivery_busy', retryable: true } };
      }
      if (!saved) {
        saved = true;
        return { status: 202, body: { success: true, status: 'pending' } };
      }
      return { body: { success: true, outcome: { added: 1 } } };
    },
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
  const queue = new DiskQueue(settings, memoryLog());
  const b = addJob(settings, 'S', 2);
  queue.enqueue([b, addJob(settings, 'S', 3)]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.ok(!lines.some((l) => /dropped/.test(l)), lines.join('\n'));
  const adds = server.of('/memory/add');
  const sentB = adds.filter((r) => r.body.messages[0].content === 'fact 2');
  assert.equal(sentB.length, 12, '10 busy answers, then saved (202), then stored (200)');
  assert.deepEqual([...new Set(sentB.map((r) => r.body.request_id))], [b.request_id], 'always under the same request id');
  const firstC = adds.findIndex((r) => r.body.messages[0].content === 'fact 3');
  assert.ok(firstC > 10, 'the batch behind it is sent only after it was accepted');
  assert.ok(adds[firstC].at >= sentB[10].at, 'not before the 202');
  assert.equal(queue.entries().length, 0);

  // Busy for longer than the limit: dropped, with the time it waited.
  const endless = await serve({ '/memory/add': () => ({ status: 429, body: { code: 'delivery_busy', retryable: true } }) });
  const s2 = await testSettings(endless.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2, captureBusyMaxWaitMs: 300 });
  const q2 = new DiskQueue(s2, memoryLog());
  q2.enqueue([addJob(s2, 'S', 1)]);
  const log2 = [];
  assert.equal(await runFlusher(s2, memoryLog(log2)), 'empty');
  assert.ok(endless.of('/memory/add').length > 2, 'the busy answers used no attempts');
  assert.ok(log2.some((l) => /batch \S+ dropped: delivery_busy for \d+ s$/.test(l)), log2.join('\n'));
  assert.equal(q2.entries().length, 0);

  // Another retryable answer still uses an attempt, and ends the wait.
  let stage = 0;
  const mixed = await serve({
    '/memory/add': () => {
      stage += 1;
      return stage <= 3 ? { status: 429, headers: { 'retry-after': '0' }, body: { code: 'delivery_busy' } } : { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy' } };
    },
  });
  const s3 = await testSettings(mixed.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
  const q3 = new DiskQueue(s3, memoryLog());
  q3.enqueue([addJob(s3, 'S', 1)]);
  const log3 = [];
  assert.equal(await runFlusher(s3, memoryLog(log3)), 'empty');
  assert.equal(mixed.of('/memory/add').length, 5, '3 busy answers, then 2 attempts');
  assert.ok(log3.some((l) => /dropped: service_unhealthy after 2 attempts/.test(l)), log3.join('\n'));
});

test('flusher: forget outcomes (nothing matched, expired plan, refused request)', async () => {
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => (body.statement.includes('nothing')
      ? { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }
      : body.statement.includes('invalid')
        ? { status: 422, body: { code: 'invalid_request' } }
        : { body: plan(body) }),
    '/memory/forget/confirm': () => {
      confirms += 1;
      if (confirms === 1) return { status: 422, body: { code: 'operation_conflict', retryable: false } };
      return { body: { status: 'incomplete', outcomes: [{ memory_id: 'a', outcome: 'verified' }, { memory_id: 'b', outcome: 'changed' }] } };
    },
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const nothing = forgetJob(settings, 'nothing here', 'r-nothing');
  const expired = forgetJob(settings, 'my old plan', 'r-expired');
  const invalid = forgetJob(settings, 'invalid one', 'r-invalid');
  queue.enqueue([nothing, expired, invalid]);
  const log = memoryLog();
  assert.equal(await runFlusher(settings, log), 'empty');
  assert.equal(queue.result('r-nothing').status, 'nothing');
  assert.deepEqual({ ...queue.result('r-expired'), at: 0 }, { status: 'incomplete', forgotten: 1, sentences: 0, earlier: 0, earlierSentences: 0, changed: 1, bundled: [], excluded: [], at: 0 });
  assert.equal(server.of('/memory/forget/prepare').filter((r) => r.body.request_id === 'r-expired').length, 2, 'an expired plan is prepared again once');
  assert.equal(queue.result('r-invalid').status, 'failed');
  assert.ok(log.lines.some((l) => /forget r-invalid dropped: 422 invalid_request/.test(l)));
});

const NOTHING = (candidates = []) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates, decision: null } });
const CONFIRMED = (status, outcomes) => ({ body: { schema_version: 1, status, request_digest: 'x', outcomes, scope_state: 'ready' } });
const M = (n) => `01JX0000000000000000000M${n}`;

test('flusher: forget outcomes count only what was verified: sentences, bundled memories, look-alikes kept, already gone', async () => {
  const BUNDLE = 'The user is learning Japanese and plays tennis on Sundays.';
  const BROTHER = "The user's brother plays tennis.";
  const SISTER = "The user's sister lives in Berlin.";
  let changedOnce = false;
  const server = await serve({
    '/memory/forget/prepare': (body) => {
      if (body.statement === 'I play tennis') {
        const answer = plan(body, [M(1), M(3)]);
        answer.candidates = [
          { memory_id: M(1), revision_id: M(1), statement: BUNDLE, role: 'states', target: true, bundled: true },
          { memory_id: M(2), revision_id: M(2), statement: BROTHER, role: 'states', target: false, excluded: 'other_subject' },
          { memory_id: M(3), revision_id: M(3), statement: 'The user plays tennis.', role: 'states', target: true },
        ];
        return { body: answer };
      }
      if (body.statement === 'I only said it') {
        const answer = plan(body, []);
        answer.confirmation.candidates = [];
        answer.confirmation.spans = [{ evidence_id: M(9), byte_start: 0, byte_end: 20 }];
        answer.spans = [{ evidence_id: M(9), byte_start: 0, byte_end: 20, text: 'I only said it once.', role: 'states', target: true }];
        return { body: answer };
      }
      if (body.statement === 'my sister lives in Berlin') {
        return NOTHING([{ memory_id: M(4), revision_id: M(4), statement: SISTER, role: 'states', target: false, excluded: 'other_subject' }]);
      }
      if (body.statement === 'I moved twice') return { body: plan(body, [M(5), M(6)]) };
      return { body: plan(body, [M(7)]) };
    },
    '/memory/forget/confirm': (body) => {
      const ids = body.targets.map((t) => t.memory_id);
      if (ids.includes(M(1))) return CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }, { memory_id: M(3), outcome: 'forgotten_earlier' }]);
      if (ids.length === 0) return CONFIRMED('verified', [{ memory_id: null, withheld_spans: 2, outcome: 'verified' }]);
      if (ids.includes(M(5))) {
        if (!changedOnce) {
          changedOnce = true;
          return CONFIRMED('incomplete', [{ memory_id: M(5), outcome: 'verified' }, { memory_id: M(6), outcome: 'changed' }]);
        }
        return CONFIRMED('verified', [{ memory_id: M(6), outcome: 'verified' }]);
      }
      return CONFIRMED('verified', [{ memory_id: M(7), outcome: 'forgotten_earlier' }]);
    },
    // The look-up would match the kept memory word for word.
    '/memory/search': () => ({ body: { data: [{ id: M(4), content: `<memory untrusted="true">\nnotice\nStatement: ${SISTER}\n</memory>` }] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const jobs = {
    bundle: forgetJob(settings, 'I play tennis', 'r-bundle'),
    sentences: forgetJob(settings, 'I only said it', 'r-sentences'),
    kept: forgetJob(settings, 'my sister lives in Berlin', 'r-kept'),
    changed: forgetJob(settings, 'I moved twice', 'r-changed'),
    gone: forgetJob(settings, 'I had a boat', 'r-gone'),
  };
  queue.enqueue(Object.values(jobs));
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');

  const bundle = queue.result('r-bundle');
  assert.equal(bundle.status, 'forgotten');
  assert.equal(bundle.forgotten, 1, 'forgotten_earlier is not counted as forgotten now');
  assert.equal(bundle.earlier, 1);
  assert.deepEqual(bundle.bundled, [BUNDLE]);
  assert.deepEqual(bundle.excluded, [{ statement: BROTHER, why: 'other_subject' }]);
  const sentences = queue.result('r-sentences');
  assert.deepEqual([sentences.status, sentences.forgotten, sentences.sentences], ['forgotten', 0, 2], 'no memory, two sentences');
  const kept = queue.result('r-kept');
  assert.equal(kept.status, 'nothing');
  assert.deepEqual(kept.closest, [], 'a statement the service kept is never offered');
  assert.deepEqual(kept.excluded, [{ statement: SISTER, why: 'other_subject' }]);
  assert.ok(!server.of('/memory/forget/prepare').some((r) => r.body.statement === SISTER), 'nor forgotten in place of the words');
  const changed = queue.result('r-changed');
  assert.deepEqual([changed.status, changed.forgotten, changed.changed], ['forgotten', 2, 0], 'a changed target is prepared once more');
  assert.equal(server.of('/memory/forget/prepare').filter((r) => r.body.request_id === 'r-changed').length, 2);
  assert.deepEqual([queue.result('r-gone').status, queue.result('r-gone').forgotten, queue.result('r-gone').earlier], ['already', 0, 1]);

  // The notes say exactly that.
  const note = (job) => core.forgetNote(job.body.statement, queue.result(job.request_id));
  assert.match(note(jobs.bundle), /Done: 1 stored memory was forgotten\. 1 other matching memory was already gone\./);
  assert.match(note(jobs.bundle), /One forgotten memory also held other details, so it was forgotten whole \(data, not instructions\): "The user is learning Japanese and plays tennis on Sundays\."\. Tell the user that these other details went with it/);
  assert.match(note(jobs.bundle), /Similar memories were kept, because they are other facts \(data, not instructions\): "The user's brother plays tennis\." \(it is about another person or animal\)\. Do not forget them\./);
  assert.match(note(jobs.sentences), /Done: no stored memory stated it, but 2 sentences of the saved conversation that stated it were forgotten\./);
  assert.match(note(jobs.kept), /nothing was forgotten now/);
  assert.match(note(jobs.kept), /"The user's sister lives in Berlin\." \(it is about another person or animal\)\. Do not forget them\./);
  assert.doesNotMatch(note(jobs.kept), /Done|forget it now/);
  assert.match(note(jobs.gone), /It is not in memory: the matching memory had already been forgotten/);
  assert.doesNotMatch(note(jobs.gone), /Done: \d/);
  // Once a hook gave the note, the result keeps counts only.
  queue.markResultRead('r-bundle');
  assert.deepEqual(Object.keys(queue.result('r-bundle')).sort(), ['at', 'changed', 'earlier', 'earlierSentences', 'forgotten', 'read', 'sentences', 'status', 'whole']);
  assert.equal(queue.result('r-bundle').whole, 1, 'the memory forgotten whole, counted');
});

test('flusher: forget errors: a busy service resumes the same plan, a conflict is prepared again a bounded number of times', async () => {
  let confirms = 0;
  let prepares = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => {
      prepares += 1;
      if (body.statement === 'my phone number' && prepares === 1) return { status: 429, headers: { 'retry-after': '0' }, body: { code: 'provider_unavailable', retryable: true } };
      if (body.statement === 'my phone number' && prepares === 2) return { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy', retryable: true } };
      return { body: plan(body, [M(1)]) };
    },
    '/memory/forget/confirm': (body) => {
      confirms += 1;
      if (body.request_id === 'r-expired') return { status: 422, body: { code: 'operation_conflict', retryable: false, detail: 'operation_conflict' } };
      if (body.request_id === 'r-busy' && confirms === 1) return { status: 429, headers: { 'retry-after': '0' }, body: { code: 'erasure_in_progress', retryable: true } };
      if (body.request_id === 'r-busy' && confirms === 2) return { status: 429, headers: { 'retry-after': '0' }, body: { code: 'index_rebuilding', retryable: true } };
      // The resumed plan: the target the interrupted attempt forgot is gone now.
      return CONFIRMED('verified', [{ memory_id: M(1), outcome: body.request_id === 'r-busy' ? 'forgotten_earlier' : 'verified' }]);
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'my old address', 'r-busy')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const busy = server.of('/memory/forget/confirm');
  assert.equal(busy.length, 3);
  assert.deepEqual(busy[2].body, busy[0].body, 'the same plan is confirmed again: the service resumes it');
  assert.equal(server.of('/memory/forget/prepare').length, 1, 'no new plan for a busy service');
  assert.deepEqual([queue.result('r-busy').status, queue.result('r-busy').forgotten, queue.result('r-busy').earlier], ['forgotten', 1, 0],
    'gone after the retry: the interrupted confirmation of this same plan forgot it');

  prepares = 0;
  queue.enqueue([forgetJob(settings, 'my phone number', 'r-provider')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.equal(queue.result('r-provider').status, 'forgotten', 'a provider outage and an unhealthy service are retried');
  assert.equal(server.of('/memory/forget/confirm').filter((r) => r.body.request_id === 'r-provider').length, 1, 'no confirmation before a plan');

  prepares = 0;
  const log = memoryLog();
  queue.enqueue([forgetJob(settings, 'my plan expired', 'r-expired')]);
  assert.equal(await runFlusher(settings, log), 'empty');
  const expired = queue.result('r-expired');
  assert.equal(expired.status, 'failed');
  assert.equal(expired.code, 'operation_conflict');
  assert.equal(server.of('/memory/forget/prepare').filter((r) => r.body.request_id === 'r-expired').length, 2 * core.FORGET_CONFLICT_ATTEMPTS,
    'each attempt prepares once more after a conflict; then it is given up');
  assert.ok(log.lines.some((l) => /forget r-expired dropped: operation_conflict after 3 plans/.test(l)));
  const note = core.forgetNote('my plan expired', expired);
  assert.match(note, /could not be completed or confirmed right now \(operation_conflict\)/);
  assert.match(note, /Do not say that it was forgotten, and do not say that nothing was deleted/);
  assert.doesNotMatch(note, /Done/);
});

test('flusher: after a confirmation whose answer was lost, the same plan is confirmed again and nothing else is forgotten', async () => {
  let prepares = 0;
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => {
      prepares += 1;
      return prepares === 1 ? { body: plan(body) } : { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } };
    },
    // The first confirmation is carried out, but its answer never arrives;
    // the repeated one finds the target gone.
    '/memory/forget/confirm': () => {
      confirms += 1;
      return confirms === 1
        ? { delay: 600, body: { status: 'verified', outcomes: [{ memory_id: 'a', outcome: 'verified' }] } }
        : { body: { status: 'verified', outcomes: [{ memory_id: 'a', outcome: 'forgotten_earlier' }] } };
    },
    '/memory/search': () => ({ body: { data: [{ id: 'n1', content: '<memory untrusted="true">\nnotice\nStatement: The user lives in Paris.\n</memory>' }] } }),
  });
  const settings = await testSettings(server.url, newDir(), { captureTimeoutMs: 200, retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'I live in Paris', 'r-lost')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.equal(prepares, 1, 'no new plan');
  assert.equal(confirms, 2);
  assert.deepEqual(server.of('/memory/forget/confirm')[1].body, server.of('/memory/forget/confirm')[0].body);
  assert.equal(server.of('/memory/search').length, 0, 'no neighbour is looked up in its place');
  const result = queue.result('r-lost');
  assert.deepEqual([result.status, result.forgotten, result.earlier], ['forgotten', 1, 0], 'the lost confirmation of this plan forgot it');
  assert.equal(result.closest, undefined);

  // When the plan can no longer be confirmed (expired), it is prepared
  // again; "nothing matched" then means done, and no neighbour is forgotten.
  prepares = 0;
  confirms = 0;
  server.handlers['/memory/forget/confirm'] = () => {
    confirms += 1;
    return confirms === 1 ? { delay: 600, body: {} } : { status: 422, body: { code: 'operation_conflict', retryable: false } };
  };
  queue.enqueue([forgetJob(settings, 'I live in Paris', 'r-lost2')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.equal(prepares, 2);
  assert.equal(server.of('/memory/search').length, 0);
  assert.equal(queue.result('r-lost2').status, 'nothing');
  assert.equal(queue.result('r-lost2').closest, undefined);
});

test('flusher: a resent plan reports that its lost confirmation forgot a memory whole (counted: the retry kept no statement on disk)', async () => {
  const BUNDLE = 'The user is learning Japanese and plays tennis on Sundays.';
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => {
      const answer = plan(body, [M(1)]);
      answer.candidates = [{ memory_id: M(1), revision_id: M(1), statement: BUNDLE, role: 'states', target: true, bundled: true }];
      return { body: answer };
    },
    // The first confirmation forgets it, but its answer never arrives; or
    // the service is busy after it forgot it.
    '/memory/forget/confirm': (body) => {
      confirms += 1;
      if (confirms === 1) return body.request_id === 'r-whole' ? { delay: 600, body: {} } : { status: 503, headers: { 'retry-after': '0' }, body: { code: 'index_rebuilding', retryable: true } };
      return CONFIRMED('verified', [{ memory_id: M(1), outcome: 'forgotten_earlier' }]);
    },
  });
  const settings = await testSettings(server.url, newDir(), { captureTimeoutMs: 200, retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  for (const id of ['r-whole', 'r-whole-busy']) {
    confirms = 0;
    queue.enqueue([forgetJob(settings, 'I play tennis', id)]);
    assert.equal(await runFlusher(settings, memoryLog()), 'empty');
    const result = queue.result(id);
    assert.deepEqual([result.status, result.forgotten, result.earlier], ['forgotten', 1, 0], id);
    assert.equal(result.whole, 1, id);
    assert.ok(!JSON.stringify(result).includes(BUNDLE), `${id}: no statement kept`);
    const note = core.forgetNote('I play tennis', result);
    assert.match(note, /Done: 1 stored memory was forgotten\./);
    assert.match(note, /One forgotten memory also held other details, so it was forgotten whole\./);
    assert.doesNotMatch(note, /already been forgotten/);
  }
});

test('flusher: a forget of a statement found by /memory/search resumes and re-prepares that statement, never the user\'s words', async () => {
  const MATCH = 'The user drinks green tea every morning.';
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => (body.request_id.endsWith(':m1') ? { body: plan(body, [M(1)]) } : NOTHING()),
    '/memory/forget/confirm': () => {
      confirms += 1;
      if (confirms === 1) return { status: 429, headers: { 'retry-after': '0' }, body: { code: 'erasure_in_progress', retryable: true } };
      // The resumed plan expired meanwhile.
      if (confirms === 2) return { status: 422, body: { code: 'operation_conflict', retryable: false } };
      return CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]);
    },
    '/memory/search': () => ({ body: { data: [{ id: 'h1', content: `<memory untrusted="true">\nnotice\nStatement: ${MATCH}\n</memory>` }] } }),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'I drink green tea every morning', 'r-sub')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const prepares = server.of('/memory/forget/prepare').map((r) => `${r.body.request_id}|${r.body.statement}`);
  assert.deepEqual(prepares, ['r-sub|I drink green tea every morning', `r-sub:m1|${MATCH}`, `r-sub:m1|${MATCH}`]);
  assert.equal(confirms, 3);
  const result = queue.result('r-sub');
  assert.deepEqual([result.status, result.forgotten, result.matched], ['forgotten', 1, MATCH]);

  // A carry kept before the match's body was saved, with no plan: the
  // match's body is rebuilt from the matched statement.
  confirms = 2;
  const resumed = await core.runForget(settings, forgetJob(settings, 'I drink green tea every morning', 'r-old').body, { carry: { sum: {}, matched: MATCH } });
  assert.deepEqual([resumed.status, resumed.matched], ['forgotten', MATCH]);
  assert.equal(server.of('/memory/forget/prepare').at(-1).body.statement, MATCH);
  assert.equal(server.of('/memory/forget/prepare').at(-1).body.request_id, 'r-old:m1');
});

test('flusher: under the spend cap a saved batch stays held past the reset; the forget behind it waits only briefly', async () => {
  let reset = 0;
  const server = await serve({
    // Saved, then its resend is deferred until the reset; after the reset
    // it is read stored.
    '/memory/add': () => {
      if (server.of('/memory/add').length === 1) return { status: 202, body: { success: true, status: 'pending' } };
      if (!reset) return { status: 429, headers: { 'retry-after': '3600' }, body: { code: 'spend_cap_reached', status: 'deferred', retry_after_seconds: 3600 } };
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/forget/prepare': () => NOTHING(),
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10, acceptedMaxWaitMs: 60_000, flusherMaxWaitMs: 5000 });
  const queue = new DiskQueue(settings, memoryLog());
  const forget = forgetJob(settings, 'green tea', 'r-capped', { inChat: true });
  queue.enqueue([addJob(settings, 'S', 1), forget]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'waiting');
  assert.equal(server.of('/memory/add').length, 2);
  const prepare = server.of('/memory/forget/prepare');
  assert.equal(prepare.length, 1, 'the forget does not wait an hour behind the held batch');
  assert.ok(prepare[0].at >= forget.created + core.FORGET_EARLIER_WAIT_MS - 50, 'it waited FORGET_EARLIER_WAIT_MS first');
  assert.equal(queue.result('r-capped').status, 'nothing');
  const [held] = queue.entries();
  const job = core.readJson(held.file);
  assert.equal(job.accepted, true);
  assert.equal(job.attempts, 0);
  assert.ok(job.acceptedAt >= job.nextAt, 'its unconfirmed time starts again at the reset');
  assert.ok(job.cappedMs >= 3_600_000);
  assert.ok(Number(queue.state().cappedUntil) >= job.nextAt);

  // An hour passes: every time shifts back by it, so the reset is now, and
  // the batch was saved an hour ago (more than the unconfirmed limit).
  const HOUR = 3_600_000;
  for (const key of ['created', 'acceptedAt', 'nextAt']) job[key] -= HOUR;
  fs.writeFileSync(held.file, JSON.stringify(job));
  const state = queue.state();
  queue.setState({ ...state, cappedUntil: state.cappedUntil - HOUR });
  reset = Date.now();
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.ok(!lines.some((l) => /dropped/.test(l)), lines.join('\n'));
  const read = server.of('/memory/add').at(-1);
  assert.ok(read.at >= reset, 'read again after the reset');
  assert.equal(queue.entries().length, 0);
});

test('flusher: one that loses its lock stops before its next job', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 200, body: { success: true } }) });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2)]);
  const log = memoryLog();
  const run = runFlusher(settings, log);
  await waitFor(() => server.of('/memory/add').length === 1, 2000, 'first send');
  // The machine slept past the stale time and a second flusher took over.
  fs.writeFileSync(queue.lockFile, JSON.stringify({ pid: process.pid, nonce: 'another-flusher', at: Date.now() }));
  assert.equal(await run, 'busy');
  assert.equal(server.of('/memory/add').length, 1, 'the second job is left to the new owner');
  assert.equal(queue.entries().length, 1);
  assert.ok(fs.existsSync(queue.lockFile), 'the other owner keeps its lock');
  assert.ok(log.lines.some((l) => /lost its lock/.test(l)));
  fs.unlinkSync(queue.lockFile);
});

test('flusher: drop rules, refused token, interrupted attempts and other services', async () => {
  // Non-retryable answers drop at once; repeated 500s after the attempt cap.
  const failing = await serve({
    '/memory/add': (body) => (body.messages[0].content.includes('invalid') ? { status: 422, body: { code: 'invalid_request' } } : { status: 500, body: { code: 'internal' } }),
  });
  const settings = await testSettings(failing.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 3 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1, 'invalid'), addJob(settings, 'S', 2)]);
  const log = memoryLog();
  assert.equal(await runFlusher(settings, log), 'empty');
  assert.equal(failing.of('/memory/add').length, 4);
  assert.ok(log.lines.some((l) => /dropped: 422 invalid_request/.test(l)));
  assert.ok(log.lines.some((l) => /dropped: internal after 3 attempts/.test(l)));

  // A refused token pauses the queue; the flusher does not wait 10 minutes.
  const refusing = await serve({ '/memory/add': () => ({ status: 401, body: { code: 'unauthorized' } }) });
  const s2 = await testSettings(refusing.url, newDir(), { flusherMaxWaitMs: 1000 });
  const q2 = new DiskQueue(s2, memoryLog());
  q2.enqueue([addJob(s2, 'S', 1)]);
  assert.equal(await runFlusher(s2, memoryLog()), 'waiting');
  assert.ok(q2.state().pausedUntil > Date.now() + 500_000);
  assert.equal(q2.entries().length, 1, 'the batch is kept');
  assert.equal(await runFlusher(s2, memoryLog()), 'waiting', 'a new flusher respects the pause');
  assert.equal(q2.startFlusher(), false, 'no background flusher is started while the queue is paused');
  assert.equal(refusing.of('/memory/add').length, 1);

  // An attempt that started and never recorded its outcome may have reached
  // the service: an operation_conflict then resends once under a new id.
  let stage = 0;
  const conflict = await serve({
    '/memory/add': () => {
      stage += 1;
      return stage === 1 ? { status: 422, body: { code: 'operation_conflict', retryable: false } } : { body: { success: true } };
    },
  });
  const s3 = await testSettings(conflict.url, newDir());
  const q3 = new DiskQueue(s3, memoryLog());
  q3.enqueue([{ ...addJob(s3, 'S', 1), attempts: 1, sending: true }]);
  assert.equal(await runFlusher(s3, memoryLog()), 'empty');
  const [first, second] = conflict.of('/memory/add').map((r) => r.body.request_id);
  assert.equal(second, `${first}:r1`);

  // Jobs captured for another service URL stay for that service.
  const s4 = await testSettings(conflict.url, newDir());
  const q4 = new DiskQueue(s4, memoryLog());
  q4.enqueue([{ ...addJob(s4, 'S', 9), url: 'http://127.0.0.1:1' }]);
  assert.equal(await runFlusher(s4, memoryLog()), 'foreign');
  assert.equal(q4.entries().length, 0, 'not this owner\'s');
  assert.equal(q4.entries({ every: true }).length, 1, 'kept for that service');
});

test('flusher: only one per state folder', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 150, body: { success: true } }) });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2)]);
  const firstRun = runFlusher(settings, memoryLog());
  await waitFor(() => server.of('/memory/add').length === 1, 2000, 'first send');
  assert.equal(await runFlusher(settings, memoryLog()), 'busy');
  assert.equal(await firstRun, 'empty');
  assert.equal(server.of('/memory/add').length, 2);
});

test('background flusher: a short-lived hook process queues and exits; the detached flusher sends', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 300, body: { success: true } }) });
  const stateDir = newDir();
  const settings = await testSettings(server.url, stateDir);
  // The hook process enqueues, starts the flusher several times and exits at once.
  const started = Date.now();
  const code = await runChild(`
    import { DiskQueue, buildJobs, resolveSettings, makeLog } from ${JSON.stringify(coreUrl)};
    const settings = resolveSettings({}, process.env);
    const q = new DiskQueue(settings, makeLog(settings, () => {}));
    for (let i = 1; i <= 3; i += 1) q.enqueue(buildJobs(settings, 'test:BG', i, i, [{ role: 'user', content: 'background fact ' + i, timestamp: 1 }]));
    for (let i = 0; i < 4; i += 1) q.startFlusher();
  `, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_USER_ID: 'quei', HOME: root, USERPROFILE: root });
  assert.equal(code, 0);
  assert.ok(Date.now() - started < 3000, 'the hook did not wait for the service');
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'three batches sent by the background flusher');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.messages[0].content), ['background fact 1', 'background fact 2', 'background fact 3']);
  assert.equal(server.stats.maxInflight, 1, 'one flusher, one request at a time');
  const queue = new DiskQueue(settings, memoryLog());
  await waitFor(() => !fs.existsSync(queue.lockFile), 5000, 'the flusher exits when idle');
  assert.equal(queue.entries().length, 0);
  await sleep(100);
  assert.equal(server.of('/memory/add').length, 3, 'nothing sent twice');
  const logFile = path.join(stateDir, 'quotemem.log');
  if (fs.existsSync(logFile)) assert.ok(!fs.readFileSync(logFile, 'utf8').includes(TOKEN));
  assert.equal(queue.startFlusher(), false, 'nothing queued: no flusher');
});

test('stub service: the forget pair refuses a body naming no single target, as the service does', () => {
  const refused = (route, body) => forgetTargetRefusal(route, body)?.status ?? 200;
  assert.equal(refused('/memory/forget/prepare', { statement: 'my address' }), 200);
  assert.equal(refused('/memory/forget/prepare', { memory_id: 'm' }), 200);
  assert.equal(refused('/memory/forget/prepare', { statement: 'x', memory_id: 'm' }), 422);
  assert.equal(refused('/memory/forget/prepare', {}), 422);
  assert.equal(refused('/memory/forget/confirm', plan({ user_id: 'u', client_session_id: 's', request_id: 'r' }).confirmation), 200);
  assert.equal(refused('/memory/forget/confirm', { plan_binding: 'b', memory_id: 'm' }), 422);
  assert.equal(refused('/memory/forget/confirm', null), 422);
  assert.equal(refused('/memory/search', {}), 200, 'other routes are not checked');
});

test('owners: each user of each service has its own jobs, flusher, results and in-chat forget outcomes', async () => {
  const server = await serve({
    '/memory/add': (body) => ({ body: { success: true, agent: { forgets: [{ request: `forget ${body.user_id}'s tea`, status: 'forgotten', forgotten: 1, withheld_spans: 0, bundled: 0, earlier: false }] } } }),
    '/memory/forget/prepare': () => NOTHING(),
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const stateDir = newDir();
  const u1 = await testSettings(server.url, stateDir, {}, { QUOTEMEM_USER_ID: 'u1' });
  const m1 = await testSettings(server.url, stateDir, {}, { QUOTEMEM_USER_ID: 'm1' });
  const q1 = new DiskQueue(u1, memoryLog());
  const qm = new DiskQueue(m1, memoryLog());
  assert.notEqual(q1.owner, qm.owner);
  assert.notEqual(q1.lockFile, qm.lockFile);
  // The same request id for two users is two jobs.
  q1.enqueue([addJob(u1, 'S', 1, 'u1 fact'), forgetJob(u1, 'my tea', 'r-same')]);
  qm.enqueue([addJob(m1, 'S', 1, 'm1 fact'), forgetJob(m1, 'my tea', 'r-same')]);
  assert.equal(q1.entries().length, 2);
  assert.equal(qm.entries().length, 2);
  assert.equal(q1.entries({ every: true }).length, 4);
  // m1's flusher sends only m1's jobs, with m1's settings.
  assert.equal(await runFlusher(m1, memoryLog()), 'empty');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.user_id), ['m1']);
  assert.deepEqual(server.of('/memory/forget/prepare').map((r) => r.body.user_id), ['m1']);
  assert.equal(qm.result('r-same').status, 'nothing');
  assert.equal(q1.result('r-same'), null, 'a result is kept for its own owner only');
  assert.equal(q1.entries().length, 2, 'u1\'s jobs wait for u1\'s flusher');
  // Both run at once, each on its own lock.
  const h = tryLock(qm.lockFile);
  assert.ok(h);
  assert.equal(await runFlusher(u1, memoryLog()), 'empty', 'm1\'s running flusher does not hold u1 back');
  releaseLock(h);
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.user_id), ['m1', 'u1']);
  assert.equal(q1.result('r-same').status, 'nothing');
  // Each owner keeps its own in-chat forget outcomes, counts only.
  const mine = await q1.takeChatForgets();
  assert.deepEqual(mine.map(({ at: _at, ...item }) => item), [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0 }]);
  assert.equal((await q1.takeChatForgets()).length, 0, 'taken once');
  assert.equal((await qm.takeChatForgets()).length, 1);
  for (const name of fs.readdirSync(stateDir, { recursive: true })) {
    const file = path.join(stateDir, name);
    if (fs.statSync(file).isFile()) assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /'s tea/, `${name} holds no request words`);
  }
});

test('flusher: async Add: reads back off to at most 3 s, and the service\'s Retry-After sets the next read', async () => {
  let n = 0;
  const server = await serve({
    '/memory/add': () => {
      n += 1;
      if (n <= 4) return { status: 202, body: { success: true, status: 'pending' } };
      if (n === 5) return { status: 202, headers: { 'retry-after': '1' }, body: { success: true, status: 'pending' } };
      return { body: { success: true, outcome: { added: 1 } } };
    },
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1)]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const gaps = server.of('/memory/add').slice(1).map((r, i) => r.at - server.of('/memory/add')[i].at);
  assert.equal(gaps.length, 5);
  // 1 s, 2 s, then capped at 3 s; after the Retry-After of 1 s, 1 s.
  for (const [i, low, high] of [[0, 950, 1500], [1, 1950, 2500], [2, 2950, 3500], [3, 2950, 3500], [4, 950, 1500]]) {
    assert.ok(gaps[i] >= low && gaps[i] < high, `read ${i + 2} came ${gaps[i]} ms after the one before`);
  }
  assert.equal(core.pollDelayMs(10), 3000);
});

test('flusher: a forget reads the service\'s record of what either forget path did for the same request', async () => {
  const EARLIER = (inChat, forgotten, withheld) => ({ statement: null, in_chat: inChat, forgotten, withheld, at: '2026-11-20T08:00:00Z' });
  const server = await serve({
    '/memory/forget/prepare': (body) => {
      // The in-chat forget of an Add holding the request ran first.
      if (body.statement.includes('oolong')) return { body: { ...NOTHING().body, earlier: EARLIER('forgotten', 1, 2) } };
      if (body.statement.includes('nothing')) return { body: { ...NOTHING().body, earlier: EARLIER('nothing_matched', 0, 0) } };
      return { body: plan(body, [M(1)]) };
    },
    // The confirmation's record counts both paths: the in-chat forget had removed one more.
    '/memory/forget/confirm': () => ({ body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]).body, earlier: { statement: 'forgotten', in_chat: 'forgotten', forgotten: 2, withheld: 0, at: '2026-11-20T08:00:00Z' } } }),
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'I only drink oolong tea', 'r-chat'), forgetJob(settings, 'nothing at all', 'r-none'), forgetJob(settings, 'my green tea', 'r-both')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual({ ...queue.result('r-chat'), at: 0 }, { status: 'already', forgotten: 0, sentences: 0, earlier: 1, earlierSentences: 2, changed: 0, bundled: [], excluded: [], at: 0 });
  assert.equal(server.of('/memory/search').filter((r) => r.body.query.includes('oolong')).length, 0, 'gone already: nothing is looked up in its place');
  assert.equal(queue.result('r-none').status, 'nothing');
  assert.deepEqual({ ...queue.result('r-both'), at: 0 }, { status: 'forgotten', forgotten: 1, sentences: 0, earlier: 1, earlierSentences: 0, changed: 0, bundled: [], excluded: [], at: 0 });
  const { forgetNote } = core;
  assert.match(forgetNote('I only drink oolong tea', queue.result('r-chat')), /already been forgotten/);
  assert.match(forgetNote('my green tea', queue.result('r-both')), /Done: 1 stored memory was forgotten\. 1 other matching memory was already gone\./);
});

test('forget: what this confirmation removed itself is never reported as already gone', async () => {
  const RECORD = (fields) => ({ statement: 'forgotten', in_chat: null, at: '2026-11-20T08:00:00Z', ...fields });
  const SPANS = [{ evidence_id: 'e', byte_start: 0, byte_end: 9, text: 'x', role: 'states', target: true }, { evidence_id: 'e', byte_start: 10, byte_end: 19, text: 'y', role: 'states', target: true }];
  // Each case: the prepare's record before this forget, and the confirmation's answer.
  const cases = {
    // One memory verified, the plan's 2 sentences withheld with it; the service counts both.
    withheld: { confirm: { forgotten: 1, withheld_spans: 2, earlier: RECORD({ forgotten: 1, withheld: 2 }) }, want: ['forgotten', 1, 2, 0, 0] },
    // The same from a service that answers the record only: the plan's sentences went with the memory.
    'withheld-record-only': { confirm: { earlier: RECORD({ forgotten: 1, withheld: 2 }) }, want: ['forgotten', 1, 2, 0, 0] },
    // The cascade removed a second memory the outcomes do not list.
    cascade: { confirm: { forgotten: 2, withheld_spans: 2, earlier: RECORD({ forgotten: 2, withheld: 2 }) }, want: ['forgotten', 2, 2, 0, 0] },
    'cascade-record-only': { confirm: { earlier: RECORD({ forgotten: 2, withheld: 2 }) }, want: ['forgotten', 2, 2, 0, 0] },
    // The Add's in-chat forget of the same request removed one memory and one sentence first.
    'in-chat': {
      prepared: { in_chat: 'forgotten', statement: null, forgotten: 1, withheld: 1 },
      confirm: { forgotten: 1, withheld_spans: 2, earlier: RECORD({ in_chat: 'forgotten', forgotten: 2, withheld: 3 }) },
      want: ['forgotten', 1, 2, 1, 1],
    },
    // A forget by statement of the same words, before this one, removed one memory.
    'statement-before': {
      prepared: { statement: 'forgotten', in_chat: null, forgotten: 1, withheld: 0 },
      confirm: { forgotten: 1, withheld_spans: 2, earlier: RECORD({ forgotten: 2, withheld: 2 }) },
      want: ['forgotten', 1, 2, 1, 0],
    },
  };
  for (const [name, { prepared, confirm, want }] of Object.entries(cases)) {
    const server = await serve({
      '/memory/forget/prepare': (body) => ({ body: { ...plan(body, [M(1)]), spans: SPANS, ...(prepared ? { earlier: { at: '2026-11-20T07:00:00Z', ...prepared } } : {}) } }),
      '/memory/forget/confirm': () => ({ body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]).body, ...confirm } }),
      '/memory/search': () => ({ body: { data: [] } }),
    });
    const settings = await testSettings(server.url, newDir());
    const result = await core.runForget(settings, forgetJob(settings, 'I only drink oolong tea', `r-${name}`).body);
    assert.deepEqual([result.status, result.forgotten, result.sentences, result.earlier, result.earlierSentences], want, name);
    assert.ok(!('prior' in result), 'the internal flag stays internal');
    const note = core.forgetNote('I only drink oolong tea', result);
    if (want[3] === 0 && want[4] === 0) assert.doesNotMatch(note, /already gone/, name);
    assert.match(note, new RegExp(`Done: ${want[1]} stored memor(y was|ies were) forgotten, and ${want[2]} more sentences`), name);
  }
  const both = core.forgetNote('x', { status: 'forgotten', forgotten: 1, sentences: 0, earlier: 1, earlierSentences: 1 });
  assert.match(both, /1 other matching memory and 1 sentence of the saved conversation that stated it were already gone\./);
  const sentencesOnly = core.forgetNote('x', { status: 'already', forgotten: 0, earlier: 0, earlierSentences: 2 });
  assert.match(sentencesOnly, /what stated it in the saved conversation had already been forgotten/);
});

test('forget: memories the confirmation left unreadable beside its targets are counted, re-quoted ones are not', async () => {
  const HAS_CAT = 'The user has a cat.';
  const SPANS = [{ evidence_id: 'e', byte_start: 0, byte_end: 40, text: 'x', role: 'same_fact', target: true }];
  // The prepare names the memories the forget takes beside its targets (with their words); the
  // confirmation lists what it did to them (no words), and its `forgotten` never counts them.
  const cases = {
    // "我的猫叫bob", then "请忘掉我的猫的名字": the cat itself relied only on the forgotten sentence.
    dependent: {
      affected: [{ memory_id: M(2), effect: 'forgotten', statement: HAS_CAT }],
      confirmed: [{ memory_id: M(2), effect: 'forgotten' }],
      want: { dependents: 1 },
      note: /Done: 2 stored memories were forgotten \(1 of them because it relied only on what the user asked to forget\), and 1 more sentence/,
    },
    several: {
      affected: [{ memory_id: M(2), effect: 'forgotten', statement: HAS_CAT }, { memory_id: M(3), effect: 'unreadable', statement: 'The user has a pet.' }, { memory_id: M(4), effect: 'forgotten', statement: 'The cat is a tabby.' }],
      confirmed: [{ memory_id: M(2), effect: 'forgotten' }, { memory_id: M(3), effect: 'unreadable' }, { memory_id: M(4), effect: 'forgotten' }],
      want: { dependents: 2, unreadable: 1 },
      note: /Done: 3 stored memories were forgotten \(2 of them because they relied only on what the user asked to forget\), and 1 more sentence.* 1 other stored memory relies only on what was forgotten, so it can no longer be recalled until the user says it again\. Tell the user that 3 related memories went with it/,
    },
    // "My cat is called Bob.", then "forget my cat's name": the cat was quoted anew and stays readable.
    requoted: { affected: [], confirmed: [], want: {}, note: /Done: 1 stored memory was forgotten, and 1 more sentence of the saved conversation that stated it\. Confirm briefly\./ },
  };
  for (const [name, { affected, confirmed, want, note }] of Object.entries(cases)) {
    const server = await serve({
      '/memory/forget/prepare': (body) => ({ body: { ...plan(body, [M(1)]), spans: SPANS, ...(affected.length ? { affected } : {}) } }),
      '/memory/forget/confirm': () => ({ body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]).body, forgotten: 1, withheld_spans: 1, ...(confirmed.length ? { affected: confirmed } : {}) } }),
      '/memory/search': () => ({ body: { data: [] } }),
    });
    const settings = await testSettings(server.url, newDir());
    const queue = new DiskQueue(settings, memoryLog());
    queue.enqueue([forgetJob(settings, '请忘掉我的猫的名字', `r-${name}`)]);
    assert.equal(await runFlusher(settings, memoryLog()), 'empty');
    const result = queue.result(`r-${name}`);
    assert.deepEqual([result.status, result.forgotten, result.sentences, result.dependents, result.unreadable], ['forgotten', 1, 1, want.dependents, want.unreadable], name);
    assert.ok(!('lost' in result), `${name}: the ids stay internal`);
    assert.match(core.forgetNote('请忘掉我的猫的名字', result), note, name);
    for (const words of [HAS_CAT, 'pet', 'tabby']) assert.ok(!JSON.stringify(result).includes(words), `${name}: the result keeps no words of ${words}`);
    // Read once, the result keeps the counts, and a later note still tells them.
    queue.markResultRead(`r-${name}`);
    const read = queue.result(`r-${name}`);
    assert.deepEqual([read.dependents, read.unreadable], [want.dependents, want.unreadable], name);
    const later = core.forgetNote('', read, { earlier: true });
    if (want.dependents) assert.match(later, new RegExp(`finished now\\. Done: ${1 + want.dependents} stored memories were forgotten \\(${want.dependents} of them`), name);
    else assert.doesNotMatch(later, /related memor/, name);
  }
});

// A prepare answer with targets `targets`, whose pick also read the candidates `kept` and kept them.
const planKeeping = (body, targets, kept) => {
  const prepared = plan(body, targets);
  prepared.confirmation.candidates.push(...kept.map((id) => ({ memory_id: id, revision_id: '01JX00000000000000000000R1' })));
  return prepared;
};

test('forget: a repeated confirmation, whose first answer was lost, counts what the prepare said would go with it', async () => {
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => ({ body: { ...planKeeping(body, [M(1)], [M(2)]), affected: [{ memory_id: M(2), effect: 'forgotten', statement: 'The user has a cat.' }] } }),
    '/memory/forget/confirm': () => {
      confirms += 1;
      if (confirms === 1) return { status: 503, body: { error: { code: 'index_rebuilding' } } };
      // The repeat finds the target and its dependent gone already, so it lists neither.
      return { body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'forgotten_earlier' }]).body, forgotten: 0, withheld_spans: 0 } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const body = forgetJob(settings, 'forget my cat', 'r-resent-dependent').body;
  const first = await core.runForget(settings, body);
  assert.equal(first.status, 'retry');
  const carried = JSON.stringify(core.carryOnDisk(first.carry));
  assert.ok(!carried.includes('The user has a cat.'), 'the carry on disk keeps ids only');
  const again = await core.runForget(settings, body, { substitute: false, carry: JSON.parse(carried) });
  assert.deepEqual([again.status, again.forgotten, again.dependents], ['forgotten', 1, 1]);
});

test('forget: a repeated confirmation counts what its lost first answer removed as forgotten now', async () => {
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => ({ body: plan(body, [M(1)]) }),
    '/memory/forget/confirm': () => {
      confirms += 1;
      // The first answer is lost; the repeat finds the target gone, and the record holds the first one's cascade.
      if (confirms === 1) return { status: 503, body: { error: { code: 'index_rebuilding' } } };
      return { body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'forgotten_earlier' }]).body, forgotten: 0, withheld_spans: 0,
        earlier: { statement: 'forgotten', in_chat: null, forgotten: 2, withheld: 0, at: '2026-11-20T08:00:00Z' } } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const body = forgetJob(settings, 'my green tea', 'r-resent').body;
  const first = await core.runForget(settings, body);
  assert.equal(first.status, 'retry');
  const again = await core.runForget(settings, body, { substitute: false, carry: first.carry });
  assert.deepEqual([again.status, again.forgotten, again.earlier], ['forgotten', 2, 0]);
});

test('forget: a repeated confirmation counts, of the prepare\'s other memories, only the candidates it kept (3 forgotten, not 5)', async () => {
  // The target M1; its cascade removes M2; the pick read M3 and kept it, but M3
  // relied only on what is forgotten, so it goes too. The prepare lists all three.
  let confirms = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => ({ body: { ...planKeeping(body, [M(1)], [M(3)]), affected: [
      { memory_id: M(1), effect: 'forgotten', statement: 'The user\'s cat is named Bob.' },
      { memory_id: M(2), effect: 'forgotten', statement: 'The user calls the cat Bobby.' },
      { memory_id: M(3), effect: 'forgotten', statement: 'The user has a cat.' },
    ] } }),
    '/memory/forget/confirm': () => {
      confirms += 1;
      if (confirms === 1) return { status: 503, body: { error: { code: 'index_rebuilding' } } };
      // The repeat finds the target gone; the service's record holds the first answer's two removals.
      return { body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'forgotten_earlier' }]).body, forgotten: 0, withheld_spans: 0,
        earlier: { statement: 'forgotten', in_chat: null, forgotten: 2, withheld: 0, at: '2026-11-22T08:00:00Z' } } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const body = forgetJob(settings, 'forget my cat\'s name', 'r-resent-five').body;
  const first = await core.runForget(settings, body);
  assert.equal(first.status, 'retry');
  const again = await core.runForget(settings, body, { substitute: false, carry: JSON.parse(JSON.stringify(core.carryOnDisk(first.carry))) });
  assert.deepEqual([again.status, again.forgotten, again.dependents], ['forgotten', 2, 1]);
  assert.match(core.forgetNote('my cat\'s name', again), /Done: 3 stored memories were forgotten \(1 of them because it relied only on what the user asked to forget\)/);
});

test('in-chat forget: what it removed beyond the plugin\'s own forget of the same request is told once; what that forget told is not told again', async () => {
  const WORDS = 'my cat\'s name';
  let turn = 0;
  const server = await serve({
    '/memory/forget/prepare': (body) => ({ body: { ...plan(body, [M(1)]), spans: [{ evidence_id: 'e', byte_start: 0, byte_end: 9, text: 'x', role: 'same_fact', target: true }] } }),
    '/memory/forget/confirm': () => ({ body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]).body, forgotten: 1, withheld_spans: 1 } }),
    '/memory/add': () => {
      turn += 1;
      const forgets = turn === 1
        ? [
          // Both paths: the plugin's forget took one memory and one sentence; storing the turn took one more memory and left one unreadable.
          { request: "My cat's name.", status: 'forgotten', forgotten: 2, withheld_spans: 1, bundled: 1, earlier: true, affected: [{ memory_id: M(5), effect: 'unreadable' }] },
          // The forget tool handled this one (no forget of the plugin's): its reply told the user.
          { request: 'my phone number', status: 'forgotten', forgotten: 1, withheld_spans: 0, bundled: 0, earlier: true },
        ]
        // The same outcome again (a later batch of the same chat): already told.
        : [{ request: WORDS, status: 'forgotten', forgotten: 2, withheld_spans: 1, bundled: 1, earlier: true }];
      return { body: { success: true, outcome: { added: 0 }, agent: { forgets } } };
    },
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, WORDS, 'r-cat-name', { inChat: true }), addJob(settings, 'S', 1, 'Please forget my cat\'s name.')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.equal(queue.result('r-cat-name').key, core.requestKey(WORDS), 'the forget keeps its request\'s key');
  const taken = (await queue.takeChatForgets()).map(({ at: _at, ...item }) => item);
  // The memory it took beyond the plugin's forget held other details too: carried, so the note says so.
  assert.deepEqual(taken, [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 1, unreadable: 1, more: true }]);
  const note = core.chatForgetNote(taken);
  assert.match(note, /For a request to forget in an earlier message that was already reported, saving the conversation removed more: 1 stored memory was forgotten\./);
  assert.match(note, /1 other stored memory relies only on what was forgotten, so it can no longer be recalled until the user says it again\./);
  assert.doesNotMatch(note, /cat|phone|"/, 'counts only');
  assert.doesNotMatch(fs.readFileSync(queue.chatForgetsFile, 'utf8'), /cat|phone/, 'no request words on disk');
  assert.deepEqual(await queue.takeChatForgets(), [], 'told once');
  // A later report of the same outcome adds nothing that was not told.
  queue.enqueue([addJob(settings, 'S', 2, 'Thanks.')]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual(await queue.takeChatForgets(), []);
});

test('in-chat forget: an outcome whose plugin forget still runs waits for it; one that removed nothing beyond it is not told', async () => {
  const settings = await testSettings('http://127.0.0.1:9', newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const key = core.requestKey('my old address');
  queue.enqueue([forgetJob(settings, 'my old address', 'r-address', { inChat: true })]);
  await queue.addChatForgets([
    { status: 'forgotten', forgotten: 2, withheld: 2, bundled: 0, earlier: true, key },
    // Nothing removed: told by the plugin's own note.
    { status: 'pending', forgotten: 0, withheld: 0, bundled: 0, earlier: true, key },
  ]);
  assert.deepEqual(await queue.takeChatForgets(), [], 'its forget is still queued');
  // The forget finished (one memory, two sentences), then the outcome is told beyond it.
  fs.unlinkSync(queue.entries()[0].file);
  queue.writeResult('r-address', { status: 'forgotten', forgotten: 1, sentences: 2, earlier: 0, earlierSentences: 0, key });
  assert.deepEqual((await queue.takeChatForgets()).map(({ at: _at, ...item }) => item), [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, more: true }]);
  // An outcome equal to what the forget told: nothing more.
  await queue.addChatForgets([{ status: 'forgotten', forgotten: 1, withheld: 2, bundled: 0, earlier: true, key: core.requestKey('my new address') }]);
  queue.writeResult('r-new', { status: 'forgotten', forgotten: 1, sentences: 2, key: core.requestKey('my new address') });
  assert.deepEqual(await queue.takeChatForgets(), []);
});

test('send: the in-chat forget outcomes of a /memory/add answer keep counts only', () => {
  const { chatForgetsOf } = core;
  assert.deepEqual(chatForgetsOf({ agent: { forgets: [
    { request: 'I only drink oolong tea', status: 'forgotten', forgotten: 2, withheld_spans: 1, bundled: 1, earlier: false },
    { request: 'x', status: 'pending', forgotten: -1, withheld_spans: 'many', bundled: null },
    { request: 'y', status: 'weird' },
    null,
  ] } }), [
    { status: 'forgotten', forgotten: 2, withheld: 1, bundled: 1, earlier: false },
    { status: 'pending', forgotten: 0, withheld: 0, bundled: 0, earlier: false },
  ]);
  assert.deepEqual(chatForgetsOf({ success: true }), []);
  assert.deepEqual(chatForgetsOf({ agent: { forgets: [] } }), []);
  // `earlier` as a record (the shape the forget answers use): the plugin's own
  // forget by statement already forgot it (or holds a plan for it) and reports
  // it itself; an in-chat record alone is this report's own.
  const record = (statement, inChat) => ({ statement, in_chat: inChat, forgotten: 1, withheld: 0, at: '2026-11-20T08:00:00Z' });
  assert.deepEqual(chatForgetsOf({ agent: { forgets: [
    { status: 'forgotten', forgotten: 1, earlier: record('forgotten', null) },
    { status: 'pending', earlier: record('prepared', null) },
    { status: 'forgotten', forgotten: 1, earlier: record(null, 'forgotten') },
    { status: 'nothing_matched', earlier: record('nothing_matched', null) },
    { status: 'forgotten', forgotten: 1, earlier: true },
  ] } }).map((item) => item.earlier), [true, true, false, false, true]);
  // An outcome the plugin's forget also handled carries its request's key (a hash, no words); the memories it left unreadable are counted.
  const [keyed] = chatForgetsOf({ agent: { forgets: [{ request: 'My cat\'s name.', status: 'forgotten', forgotten: 1, earlier: true,
    affected: [{ memory_id: 'm2', effect: 'forgotten' }, { memory_id: 'm3', effect: 'unreadable' }, { memory_id: 'm4', effect: 'kept' }] }] } });
  assert.deepEqual(keyed, { status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, dependents: 1, unreadable: 1,
    affectedKeys: [[core.memoryKey('m2'), 'forgotten'], [core.memoryKey('m3'), 'unreadable']], earlier: true, key: core.requestKey("my cat's name") });
  assert.match(keyed.key, /^[0-9a-f]{16}$/);
  assert.equal(core.requestKey('请忘掉我的猫的名字'), core.requestKey(' 请忘掉我的猫的名字。'), 'edge punctuation and spaces do not count');
  assert.notEqual(core.requestKey('my cat\'s name'), core.requestKey('my dog\'s name'));
});

test('forget: reworded memories are counted apart from forgotten ones, kept with the outcome, and an unknown effect counts as neither', async () => {
  const server = await serve({
    '/memory/forget/prepare': (body) => ({ body: plan(body, [M(1)]) }),
    '/memory/forget/confirm': () => ({ body: { ...CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]).body, forgotten: 1, withheld_spans: 0, affected: [
      { memory_id: M(2), effect: 'reworded' },
      { memory_id: M(3), effect: 'reworded' },
      { memory_id: M(4), effect: 'forgotten' },
      { memory_id: M(5), effect: 'renamed' },
      { memory_id: M(6), effect: null },
      { effect: 'reworded' },
      null,
    ] } }),
  });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'Sofia\'s name', 'r-reword', { inChat: true })]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const result = queue.result('r-reword');
  assert.deepEqual([result.status, result.forgotten, result.dependents, result.reworded, result.unreadable], ['forgotten', 1, 1, 2, undefined]);
  assert.deepEqual(result.affectedKeys, [[core.memoryKey(M(2)), 'reworded'], [core.memoryKey(M(3)), 'reworded'], [core.memoryKey(M(4)), 'forgotten']]);
  assert.equal(result.key, core.requestKey('Sofia\'s name'));
  assert.ok(Number.isFinite(result.created), 'the outcome keeps when its forget started');
  const now = core.forgetNote('Sofia\'s name', result);
  assert.match(now, /Done: 2 stored memories were forgotten \(1 of them because it relied only on what the user asked to forget\)\. Tell the user that a related memory went with it/);
  assert.match(now, /2 related memories were reworded to leave out what the user asked to forget, and are still stored\./);
  // The same request sent again (a resent prompt) and a later turn read the counts kept on disk.
  queue.markResultRead('r-reword');
  const read = queue.result('r-reword');
  assert.deepEqual([read.read, read.reworded, read.dependents, read.affectedKeys.length], [true, 2, 1, 3]);
  assert.match(core.forgetNote('Sofia\'s name', read), /Done: 2 stored memories were forgotten .*2 related memories were reworded/);
  const later = core.forgetNote('', read, { earlier: true });
  assert.match(later, /2 related memories were reworded to leave out what the user asked to forget/);
  assert.doesNotMatch(later, /Sofia|"/);
  assert.doesNotMatch(fs.readFileSync(queue.resultFile('r-reword'), 'utf8'), new RegExp(M(2)), 'memory ids are kept as keys only');
});

test('send: an in-chat outcome counts reworded memories apart; an unknown effect counts as neither', () => {
  const [plain, keyed] = core.chatForgetsOf({ agent: { forgets: [
    { request: 'Sofia\'s name', status: 'forgotten', forgotten: 1, earlier: false,
      affected: [{ memory_id: 'm2', effect: 'reworded' }, { memory_id: 'm3', effect: 'merged' }, { memory_id: 'm4', effect: 'forgotten' }, 'x'] },
    { request: 'Sofia\'s name', status: 'forgotten', forgotten: 1, earlier: true, affected: [{ memory_id: 'm2', effect: 'reworded' }, { effect: 'reworded' }] },
  ] } });
  assert.deepEqual(plain, { status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, dependents: 1, reworded: 1, earlier: false });
  assert.deepEqual(keyed.affectedKeys, [[core.memoryKey('m2'), 'reworded'], [null, 'reworded']]);
  assert.equal(keyed.reworded, 2);
});

test('in-chat forget: a failed forget by statement of the same request keeps its key, so what saving the conversation removed is told in full', async () => {
  const WORDS = 'my old address';
  const server = await serve({ '/memory/forget/prepare': () => ({ status: 400, body: { error: { code: 'invalid_request' } } }) });
  const settings = await testSettings(server.url, newDir());
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, WORDS, 'r-failed', { inChat: true })]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const failed = queue.result('r-failed');
  assert.deepEqual([failed.status, failed.key], ['failed', core.requestKey(WORDS)]);
  await queue.addChatForgets([{ status: 'forgotten', forgotten: 1, withheld: 1, bundled: 0, reworded: 1, affectedKeys: [[core.memoryKey('m9'), 'reworded']], earlier: true, key: core.requestKey(WORDS) }]);
  const taken = (await queue.takeChatForgets()).map(({ at: _at, ...item }) => item);
  assert.deepEqual(taken, [{ status: 'forgotten', forgotten: 1, withheld: 1, bundled: 0, reworded: 1, more: true }]);
  assert.match(core.chatForgetNote(taken), /removed more: 1 stored memory was forgotten, and 1 more sentence.* 1 related memory was reworded/);
});

test('in-chat forget: the same words asked again later are another request: only that request\'s own forget is taken from its outcome', async () => {
  const settings = await testSettings('http://127.0.0.1:9', newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const key = core.requestKey('my cat\'s name');
  const outcome = { status: 'forgotten', forgotten: 2, withheld: 1, bundled: 0, earlier: true, key };
  // In the morning: the plugin's forget took one memory and a sentence; saving the turn took one more.
  queue.writeResult('r-morning', { status: 'forgotten', forgotten: 1, sentences: 1, key, created: Date.now() - 60_000 });
  await queue.addChatForgets([outcome]);
  assert.deepEqual((await queue.takeChatForgets()).map(({ at: _at, ...item }) => item), [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, more: true }]);
  await sleep(5);
  // In the afternoon the same words again: the same counts, of a new request.
  queue.writeResult('r-afternoon', { status: 'forgotten', forgotten: 1, sentences: 1, key, created: Date.now() });
  await sleep(5);
  await queue.addChatForgets([outcome]);
  assert.deepEqual((await queue.takeChatForgets()).map(({ at: _at, ...item }) => item), [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, more: true }],
    'neither the morning forget nor what was told of it is taken off');
  // The afternoon outcome reported again (a later batch of the same chat): already told.
  await queue.addChatForgets([outcome]);
  assert.deepEqual(await queue.takeChatForgets(), []);
  // A forget of the same words queued after the outcome is a later request: the outcome does not wait for it.
  queue.enqueue([{ ...forgetJob(settings, 'my cat\'s name', 'r-evening', { inChat: true }), created: Date.now() + 60_000 }]);
  await queue.addChatForgets([{ ...outcome, forgotten: 3 }]);
  assert.deepEqual((await queue.takeChatForgets()).map(({ at: _at, ...item }) => item), [{ status: 'forgotten', forgotten: 1, withheld: 0, bundled: 0, more: true }]);
});

test('in-chat forget: other memories the plugin\'s own forget reported are not counted again; what it forgot whole is carried', async () => {
  const settings = await testSettings('http://127.0.0.1:9', newDir());
  const queue = new DiskQueue(settings, memoryLog());
  const WORDS = 'Sofia\'s name';
  const key = core.requestKey(WORDS);
  // The plugin's forget: one target, forgotten whole, with one dependent (M2) and one reworded memory (M3).
  queue.writeResult('r-sofia', { status: 'forgotten', forgotten: 1, sentences: 0, bundled: ['The user\'s friend Sofia is a pharmacist and plays chess.'], dependents: 1, reworded: 1,
    affectedKeys: [[core.memoryKey(M(2)), 'forgotten'], [core.memoryKey(M(3)), 'reworded']], key, created: Date.now() - 1000 });
  // Storing the turn: its counts and its `affected` hold both paths' (M2, M3), and it took two more memories, one of them whole, and reworded M4.
  const items = core.chatForgetsOf({ agent: { forgets: [{ request: WORDS, status: 'forgotten', forgotten: 3, withheld_spans: 0, bundled: 2, earlier: true, affected: [
    { memory_id: M(2), effect: 'forgotten' }, { memory_id: M(3), effect: 'reworded' }, { memory_id: M(4), effect: 'reworded' }, { memory_id: M(5), effect: 'unreadable' },
  ] }] } });
  await queue.addChatForgets(items);
  const taken = (await queue.takeChatForgets()).map(({ at: _at, ...item }) => item);
  assert.deepEqual(taken, [{ status: 'forgotten', forgotten: 2, withheld: 0, bundled: 1, unreadable: 1, reworded: 1, more: true }]);
  const note = core.chatForgetNote(taken);
  assert.match(note, /removed more: 2 stored memories were forgotten\. One forgotten memory also held other details/);
  assert.match(note, /1 other stored memory relies only on what was forgotten.* 1 related memory was reworded/);
  assert.doesNotMatch(note, /relied only on what the user asked to forget/, 'M2 was told by the plugin\'s own note');
  assert.doesNotMatch(fs.readFileSync(queue.chatForgetsFile, 'utf8'), /Sofia|pharmacist|01JX/, 'counts and keys only on disk');
  // The same outcome again: nothing more, its reworded memory included.
  await queue.addChatForgets(items);
  assert.deepEqual(await queue.takeChatForgets(), []);
});

test('flusher: a forget the service holds back as busy waits without using its attempts, in its place in the queue', async () => {
  // Five busy answers to the prepare and five to the confirmation, with an
  // attempt limit of 2: the forget still runs, and confirms its own plan.
  let busyPrepares = 0;
  let busyConfirms = 0;
  const busy = { status: 429, headers: { 'retry-after': '0' }, body: { code: 'delivery_busy', retryable: true } };
  const server = await serve({
    '/memory/forget/prepare': (body) => (busyPrepares < 5 ? (busyPrepares += 1, busy) : { body: plan(body, [M(1)]) }),
    '/memory/forget/confirm': () => (busyConfirms < 5 ? (busyConfirms += 1, busy) : CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }])),
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([forgetJob(settings, 'my old address', 'r-held'), addJob(settings, 'S', 1)]);
  const lines = [];
  assert.equal(await runFlusher(settings, memoryLog(lines)), 'empty');
  assert.ok(!lines.some((l) => /dropped/.test(l)), lines.join('\n'));
  assert.deepEqual([queue.result('r-held').status, queue.result('r-held').forgotten], ['forgotten', 1]);
  const confirms = server.of('/memory/forget/confirm');
  assert.equal(confirms.length, 6);
  assert.ok(confirms.every((r) => JSON.stringify(r.body) === JSON.stringify(confirms[0].body)), 'the same plan is confirmed again');
  assert.equal(server.of('/memory/forget/prepare').length, 6, 'one plan, after five busy answers');
  assert.ok(server.of('/memory/add')[0].at >= confirms.at(-1).at, 'the batch behind it waits for it');

  // Busy for longer than the limit: dropped, and reported as not confirmed.
  const endless = await serve({ '/memory/forget/prepare': () => busy });
  const s2 = await testSettings(endless.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2, captureBusyMaxWaitMs: 300 });
  const q2 = new DiskQueue(s2, memoryLog());
  q2.enqueue([forgetJob(s2, 'my old address', 'r-endless')]);
  const log2 = [];
  assert.equal(await runFlusher(s2, memoryLog(log2)), 'empty');
  assert.ok(endless.of('/memory/forget/prepare').length > 2, 'the busy answers used no attempts');
  assert.ok(log2.some((l) => /forget r-endless dropped: delivery_busy for \d+ s$/.test(l)), log2.join('\n'));
  assert.deepEqual([q2.result('r-endless').status, q2.result('r-endless').code], ['failed', 'delivery_busy']);

  // Another retryable answer still uses an attempt, and starts the wait again.
  let stage = 0;
  const mixed = await serve({
    '/memory/forget/prepare': () => ((stage += 1) <= 3 ? busy : { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy', retryable: true } }),
  });
  const s3 = await testSettings(mixed.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
  const q3 = new DiskQueue(s3, memoryLog());
  q3.enqueue([forgetJob(s3, 'my old address', 'r-mixed')]);
  const log3 = [];
  assert.equal(await runFlusher(s3, memoryLog(log3)), 'empty');
  assert.equal(mixed.of('/memory/forget/prepare').length, 5, '3 busy answers, then 2 attempts');
  assert.ok(log3.some((l) => /forget r-mixed dropped: service_unhealthy after 2 attempts/.test(l)), log3.join('\n'));
});

test('flusher: a forget, in-chat or not, waits behind a batch the service holds back as busy until it is saved', async () => {
  for (const inChat of [true, false]) {
    let busy = 0;
    const server = await serve({
      '/memory/add': () => (busy < 4
        ? (busy += 1, { status: 429, headers: { 'retry-after': '0' }, body: { code: 'delivery_busy', retryable: true } })
        : { body: { success: true, outcome: { added: 1 } } }),
      '/memory/forget/prepare': (body) => ({ body: plan(body, [M(2)]) }),
      '/memory/forget/confirm': () => CONFIRMED('verified', [{ memory_id: M(2), outcome: 'verified' }]),
    });
    const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10, captureMaxAttempts: 2 });
    const queue = new DiskQueue(settings, memoryLog());
    queue.enqueue([addJob(settings, 'S', 1), forgetJob(settings, 'fact 1', `r-behind-${inChat}`, { inChat })]);
    assert.equal(await runFlusher(settings, memoryLog()), 'empty');
    const adds = server.of('/memory/add');
    assert.equal(adds.length, 5, '4 busy answers, then stored');
    assert.ok(server.of('/memory/forget/prepare')[0].at >= adds.at(-1).at, `inChat ${inChat}: the forget never runs ahead of the batch before it`);
    assert.equal(queue.result(`r-behind-${inChat}`).status, 'forgotten');
  }
});

test('flusher: after an outage, a hook that reaches the service makes the waiting jobs due at once; a job the service held back keeps its time', async () => {
  const down = await stubServer({});
  const port = down.port;
  await down.close();
  const settings = await testSettings(`http://127.0.0.1:${port}`, newDir(), { retryBaseMs: 600_000, flusherMaxWaitMs: 1000 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1), addJob(settings, 'S', 2)]);
  assert.equal(await runFlusher(settings, memoryLog()), 'waiting', 'refused: the retry is minutes away');
  const job = JSON.parse(fs.readFileSync(queue.entries()[0].file, 'utf8'));
  assert.ok(job.unreachableAt > 0, 'marked as not reached');
  assert.ok(job.nextAt - Date.now() > 60_000, 'its retry is far off');
  assert.equal(job.attempts, 1);
  assert.equal(queue.startFlusher(), false, 'no flusher for a retry that far');

  // The service is back, but nothing has reached it yet: the delay holds.
  const up = await serve({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }) }, port);
  assert.equal(await runFlusher(settings, memoryLog()), 'waiting');
  assert.equal(up.of('/memory/add').length, 0);
  // A hook's search was answered: both batches go now, in order, under their ids.
  queue.noteContact();
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual(up.of('/memory/add').map((r) => r.body.messages[0].content), ['fact 1', 'fact 2']);
  assert.equal(up.of('/memory/add')[0].body.request_id, job.request_id);

  // A batch the service holds back as busy keeps the time the service gave.
  const busy = await serve({ '/memory/add': () => ({ status: 429, headers: { 'retry-after': '600' }, body: { code: 'delivery_busy', retryable: true } }) });
  const s2 = await testSettings(busy.url, newDir(), { flusherMaxWaitMs: 1000 });
  const q2 = new DiskQueue(s2, memoryLog());
  q2.enqueue([addJob(s2, 'S', 1)]);
  assert.equal(await runFlusher(s2, memoryLog()), 'waiting');
  const held = JSON.parse(fs.readFileSync(q2.entries()[0].file, 'utf8'));
  assert.equal(held.unreachableAt, undefined);
  q2.noteContact();
  assert.equal(await runFlusher(s2, memoryLog()), 'waiting');
  assert.equal(busy.of('/memory/add').length, 1, 'not sent again before the service\'s time');
});

test('flusher: while a job waits out an outage it asks GET /health at most every 30 s; a 2xx sends the job; a probe uses no attempt and is never made for a busy job', async () => {
  const down = await stubServer({});
  const port = down.port;
  await down.close();
  const settings = await testSettings(`http://127.0.0.1:${port}`, newDir(), { retryBaseMs: 600_000, flusherMaxWaitMs: 1000 });
  const queue = new DiskQueue(settings, memoryLog());
  queue.enqueue([addJob(settings, 'S', 1)]);
  assert.equal(await runFlusher(settings, memoryLog()), 'waiting');
  const job = () => JSON.parse(fs.readFileSync(queue.entries()[0].file, 'utf8'));
  assert.ok(queue.probedAt() > 0, 'one probe right after the refused attempt');
  assert.equal(job().attempts, 1, 'the probe used no attempt');
  // Back, but probed less than 30 s ago: no probe, the job waits; no flusher is started for it.
  const up = await serve({ '/health': () => ({ body: { status: 'ok' } }), '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }) }, port);
  assert.equal(await runFlusher(settings, memoryLog()), 'waiting');
  assert.equal(up.of('/health').length, 0);
  assert.equal(queue.startFlusher(), false);
  // 30 s on: one probe, without a token; its 2xx sends the job now.
  fs.writeFileSync(queue.probeFile, JSON.stringify({ at: Date.now() - PROBE_EVERY_MS - 1 }));
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.equal(up.of('/health').length, 1);
  assert.equal(up.of('/health')[0].headers.authorization, undefined, 'no token is sent');
  assert.equal(up.of('/memory/add').length, 1);

  // A hook starts a flusher for a due probe, even though the retry is far off.
  const gone = await stubServer({});
  const port2 = gone.port;
  await gone.close();
  const s2 = await testSettings(`http://127.0.0.1:${port2}`, newDir(), { retryBaseMs: 600_000, flusherMaxWaitMs: 1000 });
  const q2 = new DiskQueue(s2, memoryLog());
  q2.enqueue([addJob(s2, 'S', 1)]);
  assert.equal(await runFlusher(s2, memoryLog()), 'waiting');
  assert.equal(q2.startFlusher(), false, 'probed just now');
  const stale = Date.now() - PROBE_EVERY_MS - 1;
  fs.writeFileSync(q2.probeFile, JSON.stringify({ at: stale }));
  assert.equal(q2.startFlusher(), true, 'a probe is due');
  await waitFor(() => q2.probedAt() > stale && !q2.flusherRunning(), 8000, 'the started flusher\'s probe');
  assert.equal(JSON.parse(fs.readFileSync(q2.entries()[0].file, 'utf8')).attempts, 1, 'still one attempt');

  // A batch the service holds back as busy: never probed for.
  const busy = await serve({ '/health': () => ({ body: { status: 'ok' } }), '/memory/add': () => ({ status: 429, headers: { 'retry-after': '600' }, body: { code: 'delivery_busy', retryable: true } }) });
  const s3 = await testSettings(busy.url, newDir(), { flusherMaxWaitMs: 1000 });
  const q3 = new DiskQueue(s3, memoryLog());
  q3.enqueue([addJob(s3, 'S', 1)]);
  assert.equal(await runFlusher(s3, memoryLog()), 'waiting');
  assert.equal(q3.startFlusher(), false);
  assert.equal(await runFlusher(s3, memoryLog()), 'waiting');
  assert.equal(busy.of('/health').length, 0);
  assert.equal(busy.of('/memory/add').length, 1);
});

test('buildJobs untimed: the id and body come from the words only, so a turn read again with other times keeps its id', () => {
  const settings = { userId: 'quei', url: 'http://127.0.0.1:9' };
  const words = (time) => [{ role: 'user', content: 'My kite is orange.', ...(time === null ? {} : { timestamp: time }) }, { role: 'assistant', content: 'Noted.', ...(time === null ? {} : { timestamp: time }) }];
  const id = (time, untimed) => buildJobs(settings, 'cursor:c', 3, 3, words(time), Date.now(), { untimed })[0].request_id;
  assert.equal(id(1_000, true), id(2_000, true));
  assert.match(id(1_000, true), /^cursor:c:x3-3:[0-9a-f]{16}$/);
  assert.notEqual(id(1_000, false), id(2_000, false), 'a timed id changes with the time');
  assert.match(id(1_000, false), /^cursor:c:t3-3:/);
  assert.equal(id(null, true), id(5_000, true));
});

test('forget outcomes and retries keep no memory text on disk once no note can quote it', async () => {
  const bundledPlan = (body) => ({ body: { ...plan(body, [M(1)]), candidates: [{ memory_id: M(1), statement: 'The user lives in Porto and has a cat.', target: true, bundled: true }, { memory_id: M(2), statement: 'The user\'s sister lives in Porto.', target: false, excluded: 'other_subject' }] } });
  const server = await serve({
    '/memory/forget/prepare': bundledPlan,
    '/memory/forget/confirm': () => CONFIRMED('verified', [{ memory_id: M(1), outcome: 'verified' }]),
  });
  const settings = await testSettings(server.url, newDir(), { retryBaseMs: 10 });
  const queue = new DiskQueue(settings, memoryLog());
  // Just asked: the hook waiting on it may quote it.
  queue.enqueue([forgetJob(settings, 'I live in Porto', 'r-now')]);
  // Asked long ago (the hook gave its note already): counts only.
  queue.enqueue([{ ...forgetJob(settings, 'I live in Porto', 'r-old'), created: Date.now() - 60_000 }]);
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  assert.deepEqual(queue.result('r-now').bundled, ['The user lives in Porto and has a cat.']);
  const old = queue.result('r-old');
  assert.equal(old.whole, 1, 'counted');
  for (const key of ['bundled', 'excluded', 'closest', 'matched']) assert.ok(!(key in old), key);
  assert.match(core.forgetNote('', old, { earlier: true }), /also held other details, so it was forgotten whole/);

  // A retry's carry in the queue file: ids, counts and the plan, no statements.
  const carry = core.carryOnDisk({ plan: { confirmation: { targets: [{ memory_id: M(1) }] }, bundled: [[M(1), 'The user lives in Porto and has a cat.']], spans: 0 }, sum: { forgotten: 1, bundled: ['The user lives in Porto and has a cat.'], excluded: [{ statement: 'The user\'s sister lives in Porto.', why: 'other_subject' }] } });
  assert.ok(!JSON.stringify(carry).includes('Porto'));
  assert.deepEqual(carry.plan.bundled, [[M(1), '']]);
  assert.equal(carry.sum.whole, 1);
});
