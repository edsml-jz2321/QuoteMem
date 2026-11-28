#!/usr/bin/env node
// The OpenCode adapter end to end against a stub QuoteMem. Both plugin APIs
// are driven with the payload shapes OpenCode itself sends:
//   1.x (`@opencode-ai/plugin` 1.19 Hooks): chat.message, the
//       experimental.chat.messages/system transforms, experimental.text.complete,
//       command.execute.before, tool.execute.before, event, dispose;
//   2.x (`@opencode/plugin` 2.3 Promise API): session.hook('prompt' | 'context'),
//       event.subscribe (session.text.ended, session.execution.*, session.idle).
// Every test uses its own temp state folder and HOME; no real config is read.
//
//   node --test integrations/opencode/test/opencode.test.mjs
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { HIT, TOKEN, WINDOWS_ENV, plan, sleep, stubServer, tempDir, waitFor } from '../../core/test/helpers.mjs';

const HERE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const adapter = await import(pathToFileURL(path.join(HERE, 'adapter.js')).href);
const { FORGET_HEADER, RECALL_HEADER } = await import(pathToFileURL(path.join(HERE, 'core', 'index.mjs')).href);
const { serverWith, setupWith, resolveRuntime, lastReplyV1, lastReplyV2 } = adapter;

const root = tempDir('qm-opencode-');
const home = path.join(root, 'home');
fs.mkdirSync(home, { recursive: true });
const servers = new Set();
const cleanups = [];
after(async () => {
  for (const cleanup of cleanups) await cleanup().catch(() => {});
  await Promise.all([...servers].map((s) => s.close()));
});

async function serve(handlers) {
  const server = await stubServer(handlers);
  servers.add(server);
  return server;
}

let dirSeq = 0;
function env(url, extra = {}) {
  const stateDir = path.join(root, `state-${++dirSeq}`);
  return {
    ...WINDOWS_ENV,
    PATH: process.env.PATH,
    HOME: home,
    USERPROFILE: home,
    QUOTEMEM_URL: url,
    QUOTEMEM_TOKEN: TOKEN,
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_STATE_DIR: stateDir,
    QUOTEMEM_CAPTURE_RETRY_BASE_MS: '50',
    ...extra,
  };
}

// How long a test waits for a background send: on Windows each start of the
// background sender goes through PowerShell, which a loaded machine can take
// seconds to start, so the wait there is three times longer.
const WAIT_MS = process.platform === 'win32' ? 24_000 : 8000;

const ok = () => ({ body: { success: true } });
const verified = (n) => ({ body: { status: 'verified', outcomes: Array.from({ length: n }, () => ({ outcome: 'verified' })) } });

// Nothing may reach stdout or stderr: it would corrupt OpenCode's TUI.
const written = [];
for (const stream of [process.stdout, process.stderr]) {
  const original = stream.write.bind(stream);
  stream.write = (chunk, ...rest) => {
    const text = String(chunk);
    if (text.includes('[quotemem]') || text.includes(TOKEN)) written.push(text);
    return original(chunk, ...rest);
  };
}

function logOf(e) {
  try {
    return fs.readFileSync(path.join(e.QUOTEMEM_STATE_DIR, 'quotemem.log'), 'utf8');
  } catch {
    return '';
  }
}

// ------------------------------------------------------------- 1.x driver

/** A stand-in for the OpenCode process, so tests add no listeners to the real one. */
function fakeProcess() {
  const proc = new EventEmitter();
  proc.pid = 4242;
  proc.kill = () => {};
  return proc;
}

let seq = 0;

/** A 1.x host: the hooks object and helpers that send what OpenCode sends. */
async function v1(e, { client, runtime = process.execPath, options = {}, drop = [], process: proc = fakeProcess(), overrides = {} } = {}) {
  const hooks = await serverWith({ client, project: { id: 'prj' }, directory: root, worktree: root, serverUrl: new URL('http://127.0.0.1:4096') }, options, { env: e, runtime, process: proc, ...overrides });
  for (const name of drop) delete hooks[name];
  cleanups.push(async () => hooks.dispose?.());
  const model = { providerID: 'stub', modelID: 'stub-model' };

  async function user(sessionID, text, { synthetic = [] } = {}) {
    seq += 1;
    const message = { id: `msg_u${seq}`, sessionID, role: 'user', time: { created: Date.now() }, agent: 'build', model };
    const parts = [
      { id: `prt_u${seq}`, sessionID, messageID: message.id, type: 'text', text },
      ...synthetic.map((s, i) => ({ id: `prt_s${seq}_${i}`, sessionID, messageID: message.id, type: 'text', text: s, synthetic: true })),
    ];
    const started = Date.now();
    await hooks['chat.message']?.({ sessionID, agent: 'build', model, messageID: undefined, variant: undefined }, { message, parts });
    return { message, parts, ms: Date.now() - started };
  }

  // One loop step: the transforms OpenCode runs before the model call.
  async function step(sessionID, user) {
    const messages = [{ info: user.message, parts: user.parts.map((p) => ({ ...p })) }];
    await hooks['experimental.chat.messages.transform']?.({}, { messages });
    const system = ['You are OpenCode.'];
    await hooks['experimental.chat.system.transform']?.({ sessionID, model }, { system });
    return { messages, system };
  }

  async function assistant(sessionID, user, text, { fail = false } = {}) {
    const info = {
      id: `msg_a${seq}`, sessionID, role: 'assistant', time: { created: Date.now() }, parentID: user.message.id,
      modelID: model.modelID, providerID: model.providerID, mode: 'build', path: { cwd: root, root }, cost: 0,
      tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
    };
    await hooks.event({ event: { type: 'message.updated', properties: { info } } });
    if (fail) {
      const error = { name: 'MessageAbortedError', data: { message: 'Aborted' } };
      await hooks.event({ event: { type: 'message.updated', properties: { info: { ...info, error, time: { ...info.time, completed: Date.now() } } } } });
      await hooks.event({ event: { type: 'session.error', properties: { sessionID, error } } });
    } else if (text !== null) {
      const part = { id: `prt_a${seq}`, sessionID, messageID: info.id, type: 'text', text, time: { start: Date.now(), end: Date.now() } };
      await hooks['experimental.text.complete']?.({ sessionID, messageID: info.id, partID: part.id }, { text });
      await hooks.event({ event: { type: 'message.part.updated', properties: { part } } });
      await hooks.event({ event: { type: 'message.updated', properties: { info: { ...info, finish: 'stop', time: { ...info.time, completed: Date.now() } } } } });
    }
    await hooks.event({ event: { type: 'session.idle', properties: { sessionID } } });
  }

  async function turn(sessionID, userText, replyText, options) {
    const u = await user(sessionID, userText);
    const s = await step(sessionID, u);
    await assistant(sessionID, u, replyText, options);
    return { ...u, ...s };
  }

  return { hooks, user, step, assistant, turn };
}

const injected = (messages) => messages.at(-1).parts.filter((p) => p.synthetic && String(p.text).startsWith('[QuoteMem'));

// --------------------------------------------------------------- tests

test('module: one default export that both OpenCode plugin APIs accept', async () => {
  const mod = await import(pathToFileURL(path.join(HERE, 'index.js')).href);
  assert.deepEqual(Object.keys(mod), ['default'], 'OpenCode 1.x reads every export; only the default is allowed');
  assert.equal(mod.default.id, 'quotemem');
  assert.equal(typeof mod.default.server, 'function', '1.x calls server(input, options)');
  assert.equal(typeof mod.default.setup, 'function', '2.x calls setup(ctx)');
  assert.ok(!('tui' in mod.default), '1.x refuses a module with both server and tui');
  const pkg = JSON.parse(fs.readFileSync(path.join(HERE, 'package.json'), 'utf8'));
  assert.equal(pkg.type, 'module');
  assert.equal(pkg.exports['./server'], './index.js', 'both 1.x and 2.x resolve the server entry through exports');
  assert.equal(pkg.main, 'index.js');
  assert.match(pkg.engines.opencode, /^>=1\.18\.29$/, 'a lower bound only');
  for (const file of fs.readdirSync(path.join(HERE, 'core'))) {
    const copy = fs.readFileSync(path.join(HERE, 'core', file));
    assert.ok(copy.equals(fs.readFileSync(path.join(HERE, '..', 'core', file))), `core/${file} matches integrations/core`);
  }
});

test('1.x: capture every 3 turns, gated recall injected into the request only, no token in logs', async () => {
  const server = await serve({
    '/memory/add': ok,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User lives in Lisbon', { score: 0.9 })] } }),
  });
  const e = env(server.url);
  const host = await v1(e);
  const a = await host.turn('ses_A', 'I live in Lisbon.', 'Lisbon, noted.');
  assert.equal(server.of('/memory/search').length, 0, 'a plain statement does not search');
  assert.equal(injected(a.messages).length, 0);
  const b = await host.turn('ses_A', 'Fix the memory leak in parser.ts', 'Fixed the leak.');
  assert.equal(server.of('/memory/search').length, 0, 'an ordinary coding request does not search');
  assert.equal(injected(b.messages).length, 0);
  assert.equal(server.of('/memory/add').length, 0, 'two turns: nothing sent yet');

  const c = await host.user('ses_A', 'Where do I live?');
  assert.equal(server.of('/memory/search').length, 1, 'a memory question searches');
  assert.deepEqual(server.of('/memory/search')[0].body, { user_id: 'quei', query: 'Where do I live?', top_k: 20 });
  assert.equal(server.of('/memory/search')[0].headers.authorization, `Bearer ${TOKEN}`);
  const s1 = await host.step('ses_A', c);
  const s2 = await host.step('ses_A', c);
  for (const s of [s1, s2]) {
    const parts = injected(s.messages);
    assert.equal(parts.length, 1, 'the same block on every step of the turn');
    assert.ok(parts[0].text.startsWith(RECALL_HEADER));
    assert.match(parts[0].text, /User lives in Lisbon/);
    assert.match(parts[0].text, /untrusted/);
    assert.equal(parts[0].messageID, c.message.id);
    assert.deepEqual(s.system, ['You are OpenCode.'], 'the messages hook is used; the system prompt stays stable');
  }
  assert.equal(c.parts.length, 1, 'the stored user message is untouched');
  await host.assistant('ses_A', c, 'You live in Lisbon.');
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the batch');
  const add = server.of('/memory/add')[0].body;
  assert.equal(add.user_id, 'quei');
  assert.equal(add.session_id, 'opencode:ses_A');
  assert.match(add.request_id, /^opencode:ses_A:t1-3:[0-9a-f]{16}$/);
  assert.deepEqual(add.messages.map((m) => [m.role, m.content]), [
    ['user', 'I live in Lisbon.'], ['assistant', 'Lisbon, noted.'],
    ['user', 'Fix the memory leak in parser.ts'], ['assistant', 'Fixed the leak.'],
    ['user', 'Where do I live?'], ['assistant', 'You live in Lisbon.'],
  ], 'the recall block is never captured');

  // The next turn gets no block: it was for the previous turn only.
  const d = await host.turn('ses_A', 'Thanks, now run the tests.', 'Tests pass.');
  assert.equal(injected(d.messages).length, 0);
  assert.ok(!logOf(e).includes(TOKEN));
  assert.deepEqual(written, [], 'nothing written to stdout or stderr');
});

test('1.x: system prompt fallback when the messages hook is missing', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000002', 'User drinks green tea', { score: 0.8 })] } }) });
  const host = await v1(env(server.url), { drop: ['experimental.chat.messages.transform'] });
  const u = await host.user('ses_S', 'What do I usually drink?');
  const { system } = await host.step('ses_S', u);
  assert.equal(system.length, 2);
  assert.ok(system[1].startsWith(RECALL_HEADER));
  assert.match(system[1], /green tea/);
  await host.assistant('ses_S', u, 'Green tea.');
  const next = await host.step('ses_S', await host.user('ses_S', 'ok'));
  assert.deepEqual(next.system, ['You are OpenCode.'], 'cleared once the turn ends');
});

test('1.x: slash commands, subagents and aborted turns', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const host = await v1(env(server.url));
  // A slash command's expanded prompt is not the user's words.
  await host.hooks['command.execute.before']({ command: 'review', sessionID: 'ses_C', arguments: '' }, { parts: [] });
  await host.turn('ses_C', 'Review the staged changes and list the risks you see in them.', 'Two risks.');
  // A subagent session (it has a parent) is skipped.
  await host.hooks.event({ event: { type: 'session.created', properties: { info: { id: 'ses_child', parentID: 'ses_C', projectID: 'prj', directory: root, title: 'sub', version: '1', time: { created: 1, updated: 1 } } } } });
  await host.turn('ses_child', 'Where did I say the config lives?', 'In ./config.');
  assert.equal(server.of('/memory/search').length, 0, 'no recall in a subagent session');
  // Agent-generated synthetic text in a user message is not the user's words.
  const u = await host.user('ses_C', 'I moved to Porto last month.', { synthetic: ['Use the above message and context to generate a prompt and call the task tool with subagent: general'] });
  await host.step('ses_C', u);
  await host.assistant('ses_C', u, null, { fail: true });
  await host.turn('ses_C', 'My editor is Helix.', 'Helix it is.');
  await host.turn('ses_C', 'My shell is fish.', 'Fish.');
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), [
    'I moved to Porto last month.',
    'My editor is Helix.', 'Helix it is.',
    'My shell is fish.', 'Fish.',
  ], 'the aborted turn keeps the user words only; the command and the subagent are not captured');
  assert.ok(server.of('/memory/add').every((r) => r.body.session_id === 'opencode:ses_C'));

  // Deleting a session queues what is left; no earlier turn is sent again.
  await host.turn('ses_C', 'I prefer tabs.', 'Tabs.');
  await host.hooks.event({ event: { type: 'session.deleted', properties: { info: { id: 'ses_C', projectID: 'prj', directory: root, title: 't', version: '1', time: { created: 1, updated: 1 } } } } });
  await waitFor(() => server.of('/memory/add').length === 2, WAIT_MS, 'the session-end batch');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['I prefer tabs.', 'Tabs.']);
});

test('1.x: silent forget runs after the pending turns are stored, then one note', async () => {
  let prepared = null;
  const server = await serve({
    '/memory/add': () => ({ delay: 150, body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      prepared = plan(body);
      return { delay: 30, body: prepared };
    },
    '/memory/forget/confirm': () => verified(1),
  });
  const host = await v1(env(server.url));
  await host.turn('ses_F', 'I live in Lisbon.', 'Lisbon, noted.');
  await host.turn('ses_F', 'My cat is Miso.', 'Miso!');
  const u = await host.user('ses_F', 'Please forget that I live in Lisbon.');
  assert.deepEqual(server.requests.map((r) => r.route), ['/memory/add', '/memory/forget/prepare', '/memory/forget/confirm'],
    'pending turns first, then prepare and confirm; no recall on a forget turn');
  assert.ok(server.of('/memory/forget/prepare')[0].at >= server.of('/memory/add')[0].at + 140, 'prepare waited for /memory/add');
  const prepare = server.of('/memory/forget/prepare')[0].body;
  assert.equal(prepare.statement, 'I live in Lisbon');
  assert.equal(prepare.user_id, 'quei');
  assert.equal(prepare.client_session_id, 'opencode:ses_F');
  assert.match(prepare.request_id, /^opencode:ses_F:forget:[0-9a-f]{16}$/);
  assert.deepEqual(server.of('/memory/forget/confirm')[0].body, prepared.confirmation);
  const { messages } = await host.step('ses_F', u);
  const note = injected(messages);
  assert.equal(note.length, 1);
  assert.ok(note[0].text.startsWith(FORGET_HEADER));
  assert.match(note[0].text, /Done: 1 stored memory was forgotten/);
  assert.ok(!/quotemem_|request_id|01JX/.test(note[0].text), 'no tool names or ids');
  await host.assistant('ses_F', u, 'Done, I forgot where you live.');
  await host.turn('ses_F', 'I work at a bakery.', 'Bakery.');
  await host.turn('ses_F', 'I play chess.', 'Chess.');
  await waitFor(() => server.of('/memory/add').length === 2, WAIT_MS, 'the next batch');
  assert.equal(server.of('/memory/add')[1].body.messages[0].content, 'Please forget that I live in Lisbon.', 'the forget turn is captured after the forget');
  assert.equal(server.of('/memory/forget/prepare').length, 1, 'one forget only');
});

test('1.x: the explicit forget tool waits until pending turns are stored', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 100, body: { success: true } }), '/memory/search': () => ({ body: { data: [] } }) });
  const host = await v1(env(server.url));
  await host.turn('ses_T', 'My bike is red.', 'Red bike.');
  await host.turn('ses_T', 'My car is blue.', 'Blue car.');
  await host.hooks['tool.execute.before']({ tool: 'quotemem_quotemem_forget_prepare', sessionID: 'ses_T', callID: 'call_1' }, { args: { statement: 'my bike is red' } });
  assert.equal(server.of('/memory/add').length, 1, 'stored before the tool runs');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['My bike is red.', 'Red bike.', 'My car is blue.', 'Blue car.']);
  await host.hooks['tool.execute.before']({ tool: 'bash', sessionID: 'ses_T', callID: 'call_2' }, { args: {} });
  assert.equal(server.of('/memory/add').length, 1, 'other tools do not flush');
});

test('1.x: the explicit forget prepare is refused while the earlier turns are still being saved', async () => {
  // The service cannot store the turns in time: prepare run now could miss
  // them for good (the service's in-chat check may not read the user's
  // words), so OpenCode fails that tool call with a short reason.
  const busy = await serve({ '/memory/add': () => ({ status: 503, headers: { 'retry-after': '1' }, body: { code: 'service_unhealthy' } }), '/memory/search': () => ({ body: { data: [] } }) });
  const host = await v1(env(busy.url), { overrides: { forgetToolWaitMs: 300 } });
  const { hooks } = host;
  await host.turn('ses_H', 'My locker code changed last week.', 'Noted.');
  const started = Date.now();
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'quotemem_quotemem_forget_prepare', sessionID: 'ses_H', callID: 'call_h' }, { args: { statement: 'my locker code' } }),
    (error) => /still being saved/.test(error.message) && !/quotemem_|request|token/i.test(error.message),
  );
  assert.ok(Date.now() - started >= 250, 'it waited first');
  // Confirm and other tools are never refused.
  await hooks['tool.execute.before']({ tool: 'quotemem_quotemem_forget_confirm', sessionID: 'ses_H', callID: 'call_c' }, { args: {} });
  await hooks['tool.execute.before']({ tool: 'bash', sessionID: 'ses_H', callID: 'call_b' }, { args: {} });
});

test('1.x: QuoteMem down, slow or misconfigured never blocks or breaks OpenCode', async () => {
  // Nothing listens on this port.
  const dead = await serve({});
  const deadUrl = dead.url;
  await dead.close();
  servers.delete(dead);
  const e = env(deadUrl);
  const down = await v1(e);
  const u = await down.user('ses_D', 'Do you remember my cat?');
  assert.ok(u.ms < 2500, `a down service costs at most the deadline (${u.ms} ms)`);
  const again = await down.user('ses_D', 'What is my cat called?');
  assert.ok(again.ms < 300, `the breaker skips the next search (${again.ms} ms)`);
  await down.step('ses_D', again);
  await down.assistant('ses_D', again, 'I do not know.');
  await down.turn('ses_D', 'My cat is Miso.', 'Miso.');
  await down.hooks.dispose();
  assert.ok(!logOf(e).includes(TOKEN));

  const slow = await serve({ '/memory/search': () => ({ delay: 3000, body: { data: [] } }), '/memory/add': ok });
  const s = await v1(env(slow.url, { QUOTEMEM_RECALL_TIMEOUT_MS: '300' }));
  const t = await s.user('ses_L', 'Do you remember my cat?');
  assert.ok(t.ms < 1500, `a slow search is cut at its deadline (${t.ms} ms)`);

  // Hooks never throw, whatever OpenCode passes.
  for (const [name, fn] of Object.entries(s.hooks)) {
    if (name === 'dispose') continue;
    await fn(undefined, undefined);
    await fn({}, {});
  }
  await s.hooks.event({ event: { type: 'session.idle', properties: {} } });

  // No token: automatic memory stays off and silent.
  const off = await serverWith({}, {}, { env: { ...env(slow.url), QUOTEMEM_TOKEN: '' }, runtime: process.execPath, process: fakeProcess() });
  assert.deepEqual(off, {});
  const offV2 = await setupWith({ session: { hook: () => { throw new Error('must not register'); } } }, { env: { ...env(slow.url), QUOTEMEM_TOKEN: '' } });
  assert.equal(offV2, undefined);
  // OpenCode 1.x also calls setup(), with no session hooks: no second adapter, no log noise.
  const v1Setup = env(slow.url);
  assert.equal(await setupWith({ client: {}, options: {} }, { env: v1Setup, runtime: process.execPath }), undefined);
  assert.ok(!fs.existsSync(path.join(v1Setup.QUOTEMEM_STATE_DIR, 'quotemem.log')), 'nothing logged by the 2.x path under 1.x');
  assert.deepEqual(written, []);
});

test('1.x: dispose hands what is left to the detached background sender', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 300, body: { success: true } }), '/memory/search': () => ({ body: { data: [] } }) });
  const e = env(server.url);
  const host = await v1(e);
  await host.turn('ses_X', 'I am learning Rust.', 'Nice.');
  const started = Date.now();
  await host.hooks.dispose();
  assert.ok(Date.now() - started < 1000, 'dispose does not wait for /memory/add');
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the batch from the detached sender');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I am learning Rust.', 'Nice.']);
  await waitFor(() => !fs.readdirSync(e.QUOTEMEM_STATE_DIR, { recursive: true }).some((name) => path.basename(name) === 'flusher.lock'), WAIT_MS, 'the sender exits when idle');
});

test('1.x: a stop signal without dispose still hands the rest to the background sender', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const e = env(server.url);
  // A stand-in for the OpenCode process. `opencode serve` 1.19.6 has its own
  // signal listener that stops the process only while it is the sole one,
  // and it exits without unloading the plugin.
  const proc = fakeProcess();
  const kills = [];
  proc.kill = (pid, signal) => kills.push([pid, signal]);
  const own = () => {};
  proc.on('SIGTERM', own);
  const host = await v1(e, { process: proc });
  const second = await v1(e, { process: proc });
  assert.equal(proc.listenerCount('SIGTERM'), 2, 'one shared listener for every adapter of the process');
  await host.turn('ses_S', 'My favourite band is Radiohead.', 'Good taste.');
  await second.turn('ses_T', 'I collect vinyl records.', 'Nice.');
  proc.emit('SIGTERM', 'SIGTERM');
  assert.deepEqual(proc.listeners('SIGTERM'), [own], 'the plugin stops watching at once');
  assert.equal(proc.listenerCount('exit'), 0);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(kills, [[4242, 'SIGTERM']], 'the signal is raised again, so OpenCode\'s own handler stops the process');
  await waitFor(() => server.of('/memory/add').length === 2, WAIT_MS, 'both batches, sent by the detached sender');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.messages[0].content).sort(), ['I collect vinyl records.', 'My favourite band is Radiohead.']);
  // `exit` alone also queues; dispose stops watching.
  const proc2 = fakeProcess();
  const other = await v1(e, { process: proc2 });
  await other.turn('ses_U', 'I play the cello.', 'Lovely.');
  proc2.emit('exit');
  await waitFor(() => server.of('/memory/add').length === 3, WAIT_MS, 'the exit batch');
  await other.hooks.dispose();
  assert.equal(proc2.listenerCount('exit'), 0, 'dispose stops watching');
});

test('1.x: a partial batch is queued once the session has been idle', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const host = await v1(env(server.url, { QUOTEMEM_IDLE_FLUSH_MS: '300' }));
  await host.turn('ses_Y', 'I am vegetarian.', 'Noted.');
  assert.equal(server.of('/memory/add').length, 0);
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the idle batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I am vegetarian.', 'Noted.']);
});

test('1.x: without node or bun the batches are sent from inside OpenCode', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const host = await v1(env(server.url), { runtime: null });
  await host.turn('ses_I', 'I like jazz.', 'Jazz.');
  await host.turn('ses_I', 'I like tea.', 'Tea.');
  await host.turn('ses_I', 'I like chess.', 'Chess.');
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the in-process batch');
  assert.equal(server.of('/memory/add')[0].body.messages.length, 6);
  // The final reply is read back from OpenCode when no text event came.
  const client = {
    session: {
      get: async () => ({ data: { id: 'ses_R' } }),
      messages: async () => ({ data: [
        { info: { id: 'msg_u', role: 'user', sessionID: 'ses_R' }, parts: [{ type: 'text', text: 'hi' }] },
        { info: { id: 'msg_a', role: 'assistant', sessionID: 'ses_R' }, parts: [{ type: 'reasoning', text: 'x' }, { type: 'text', text: 'Read back.' }] },
      ] }),
    },
  };
  const e = env(server.url);
  const other = await v1(e, { client, runtime: null });
  const u = await other.user('ses_R', 'I collect stamps.');
  await other.assistant('ses_R', u, null);
  await other.hooks.dispose();
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 1, 'unloaded without a sender: the batch waits on disk');
  assert.match(logOf(e), /stay queued until OpenCode starts again/);
  await v1(e, { runtime: null });
  await waitFor(() => server.of('/memory/add').length === 2, WAIT_MS, 'the dispose batch, sent at the next start');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['I collect stamps.', 'Read back.']);
});

// ------------------------------------------------------------- 2.x driver

function eventBus() {
  const queue = [];
  let wake = null;
  let idle = null;
  const settle = () => { if (queue.length === 0 && idle) { idle(); idle = null; } };
  return {
    subscribe({ types, signal } = {}) {
      return {
        async *[Symbol.asyncIterator]() {
          while (!signal?.aborted) {
            if (queue.length > 0) {
              const event = queue.shift();
              if (!types || types.includes(event.type)) yield event;
              continue;
            }
            settle();
            await new Promise((resolve) => {
              wake = resolve;
              signal?.addEventListener('abort', resolve, { once: true });
            });
          }
        },
      };
    },
    /** Publish events; resolves when the subscriber has handled them all. */
    emit(...events) {
      queue.push(...events);
      const done = new Promise((resolve) => { idle = resolve; });
      wake?.();
      wake = null;
      return done;
    },
  };
}

let eventSeq = 0;
const ev = (type, data) => ({ id: `evt_${++eventSeq}`, created: Date.now(), type, location: { directory: root }, data });

async function v2(e, { runtime = process.execPath, refuse = [], noEvents = false, parents = {}, context = null } = {}) {
  const hooks = new Map();
  const bus = eventBus();
  const ctx = {
    app: { name: 'opencode', version: '2.3.2', channel: 'latest' },
    location: { directory: root, project: { id: 'prj', directory: root } },
    options: {},
    session: {
      hook: async (name, callback) => {
        if (refuse.includes(name)) throw new Error(`unknown hook ${name}`);
        hooks.set(name, callback);
        return { dispose: async () => { hooks.delete(name); } };
      },
      get: async ({ sessionID }) => ({ id: sessionID, parentID: parents[sessionID], projectID: 'prj', time: { created: 1, updated: 1 } }),
      context: async () => context ?? [],
    },
    event: noEvents ? undefined : { subscribe: (options) => bus.subscribe(options) },
  };
  const cleanup = await setupWith(ctx, { env: e, runtime, process: fakeProcess() });
  cleanups.push(async () => cleanup?.());
  let n = 0;

  async function prompt(sessionID, text) {
    n += 1;
    const event = { sessionID, messageID: `msg_${sessionID}_${n}`, prompt: { text }, metadata: undefined, delivery: 'steer' };
    const before = JSON.stringify(event.prompt);
    const started = Date.now();
    await hooks.get('prompt')?.(event);
    assert.equal(JSON.stringify(event.prompt), before, 'the prompt is never edited');
    return { messageID: event.messageID, ms: Date.now() - started };
  }

  async function request(sessionID) {
    const event = {
      sessionID, agent: 'build', model: { providerID: 'stub', id: 'stub-model' },
      system: [{ type: 'text', text: 'You are OpenCode.' }], messages: [], options: {}, tools: {},
    };
    await hooks.get('context')?.(event);
    return event.system;
  }

  async function finish(sessionID, text, { outcome = 'succeeded' } = {}) {
    const events = [];
    if (text !== null) events.push(ev('session.text.ended', { sessionID, assistantMessageID: `msg_a_${sessionID}_${n}`, ordinal: 0, text }));
    if (outcome === 'succeeded') events.push(ev('session.execution.succeeded', { sessionID }));
    else if (outcome === 'failed') events.push(ev('session.execution.failed', { sessionID, error: { type: 'provider', message: 'boom' } }));
    else events.push(ev('session.execution.interrupted', { sessionID, reason: 'user' }));
    events.push(ev('session.idle', { sessionID }));
    await bus.emit(...events);
  }

  async function turn(sessionID, text, reply, options) {
    const p = await prompt(sessionID, text);
    const system = await request(sessionID);
    await finish(sessionID, reply, options);
    return { ...p, system };
  }

  return { ctx, hooks, bus, cleanup, prompt, request, finish, turn };
}

test('2.x: prompt, context and events give capture, recall and forget', async () => {
  let prepared = null;
  const server = await serve({
    '/memory/add': ok,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000003', 'User lives in Porto', { score: 0.9 })] } }),
    '/memory/forget/prepare': (body) => {
      prepared = plan(body);
      return { body: prepared };
    },
    '/memory/forget/confirm': () => verified(1),
  });
  const e = env(server.url);
  const host = await v2(e);
  assert.deepEqual([...host.hooks.keys()].sort(), ['context', 'prompt']);
  const a = await host.turn('ses_V', 'I live in Porto.', 'Porto, noted.');
  assert.equal(a.system.length, 1, 'no block on a statement');
  const b = await host.prompt('ses_V', 'Where do I live?');
  assert.equal(server.of('/memory/search').length, 1);
  const first = await host.request('ses_V');
  const second = await host.request('ses_V');
  for (const system of [first, second]) {
    assert.equal(system.length, 2, 'the same block on every model request of the turn');
    assert.equal(system[1].type, 'text');
    assert.ok(system[1].text.startsWith(RECALL_HEADER));
    assert.match(system[1].text, /User lives in Porto/);
  }
  await host.finish('ses_V', 'You live in Porto.');
  assert.equal((await host.request('ses_V')).length, 1, 'cleared when the turn ends');
  assert.ok(b.ms < 2500);
  // An interrupted turn keeps the user's words only.
  await host.turn('ses_V', 'My dog is Rex.', 'Rex is a', { outcome: 'interrupted' });
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => [m.role, m.content]), [
    ['user', 'I live in Porto.'], ['assistant', 'Porto, noted.'],
    ['user', 'Where do I live?'], ['assistant', 'You live in Porto.'],
    ['user', 'My dog is Rex.'],
  ]);
  assert.equal(server.of('/memory/add')[0].body.session_id, 'opencode:ses_V');

  // Silent forget.
  await host.turn('ses_V', 'My bike is green.', 'Green.');
  const f = await host.prompt('ses_V', 'Forget that my bike is green.');
  const forgetRoutes = ['/memory/add', '/memory/forget/prepare', '/memory/forget/confirm'];
  if (process.platform === 'win32') {
    // On a loaded Windows machine the start of the background sender (through
    // PowerShell) can take longer than the forget's wait: the note then says
    // that it goes on in the background, and it finishes there.
    const note = await host.request('ses_V');
    assert.ok(note[1].text.startsWith(FORGET_HEADER));
    if (!/Done: 1 stored memory was forgotten/.test(note[1].text)) {
      assert.match(note[1].text, /It is being forgotten in the background/);
      await waitFor(() => server.of('/memory/forget/confirm').length === 1, 15_000, 'the forget in the background');
    }
    assert.deepEqual(server.requests.slice(-3).map((r) => r.route), forgetRoutes);
  } else {
    assert.deepEqual(server.requests.slice(-3).map((r) => r.route), forgetRoutes);
    const note = await host.request('ses_V');
    assert.ok(note[1].text.startsWith(FORGET_HEADER));
    assert.match(note[1].text, /Done: 1 stored memory was forgotten/);
  }
  assert.equal(server.of('/memory/forget/prepare')[0].body.statement, 'my bike is green');
  assert.ok(f.ms < 8000);
  await host.finish('ses_V', 'Done.');

  // Cleanup (OpenCode unloads the plugin) queues the rest.
  await host.cleanup();
  await waitFor(() => server.of('/memory/add').length === 3, WAIT_MS, 'the unload batch');
  assert.deepEqual(server.of('/memory/add')[2].body.messages.slice(-2).map((m) => m.content), ['Forget that my bike is green.', 'Done.']);
  assert.ok(!logOf(e).includes(TOKEN));
  assert.deepEqual(written, []);
});

test('2.x: subagents skipped, final reply read back, missing APIs degrade', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const context = [
    { type: 'user', id: 'msg_1', text: 'I run marathons.', time: { created: 1 } },
    { type: 'assistant', id: 'msg_2', content: [{ type: 'reasoning', text: 'r' }, { type: 'text', text: 'Impressive.' }], time: { created: 2 } },
  ];
  const host = await v2(env(server.url), { parents: { ses_kid: 'ses_P' }, context });
  await host.bus.emit(ev('session.created', { sessionID: 'ses_kid2', parentID: 'ses_P', projectID: 'prj', slug: 'k', version: '2' }));
  await host.turn('ses_kid', 'Where did I put the keys?', 'Nowhere.');
  await host.turn('ses_kid2', 'Where did I put the keys?', 'Nowhere.');
  assert.equal(server.of('/memory/search').length, 0, 'subagent sessions are skipped');
  await host.prompt('ses_P', 'I run marathons.');
  await host.finish('ses_P', null);
  await host.bus.emit(ev('session.deleted', { sessionID: 'ses_P' }));
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'the session-end batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I run marathons.', 'Impressive.']);

  // A host without the context hook and without events still captures user words.
  const e = env(server.url);
  const bare = await v2(e, { refuse: ['context'], noEvents: true });
  assert.deepEqual([...bare.hooks.keys()], ['prompt']);
  await bare.prompt('ses_B', 'I teach maths.');
  await bare.prompt('ses_B', 'I live by the sea.');
  await bare.cleanup();
  await waitFor(() => server.of('/memory/add').length === 2, WAIT_MS, 'the bare batch');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['I teach maths.', 'I live by the sea.']);
  assert.match(logOf(e), /did not accept the "context" session hook/);
  assert.match(logOf(e), /no event stream/);
});

test('helpers: runtime choice and reply extraction', () => {
  const bin = path.join(root, 'bin');
  fs.mkdirSync(bin, { recursive: true });
  const fakeNode = path.join(bin, 'node');
  fs.writeFileSync(fakeNode, '#!/bin/sh\n', { mode: 0o755 });
  assert.equal(resolveRuntime({ PATH: bin }, { execPath: '/usr/bin/node', bun: undefined }), '/usr/bin/node', 'a Node host runs it itself');
  assert.equal(resolveRuntime({ PATH: bin }, { execPath: '/usr/local/bin/bun', bun: '1.3.0' }), '/usr/local/bin/bun', 'so does plain bun');
  assert.equal(resolveRuntime({ PATH: bin }, { execPath: '/opt/opencode/bin/opencode', bun: '1.3.0' }), fakeNode, 'the compiled OpenCode binary is never re-run');
  assert.equal(resolveRuntime({ PATH: path.join(root, 'none') }, { execPath: '/opt/opencode/bin/opencode', bun: '1.3.0' }), null);
  assert.equal(resolveRuntime({ PATH: bin, QUOTEMEM_NODE: 'off' }, { execPath: '/usr/bin/node' }), null);
  assert.equal(resolveRuntime({ PATH: '', QUOTEMEM_NODE: fakeNode }, { execPath: '/opt/opencode/bin/opencode', bun: '1.3.0' }), fakeNode);
  assert.equal(lastReplyV1([{ info: { role: 'user' }, parts: [] }]), '');
  assert.equal(lastReplyV1([
    { info: { role: 'user' }, parts: [{ type: 'text', text: 'q' }] },
    { info: { role: 'assistant' }, parts: [{ type: 'text', text: 'first' }] },
    { info: { role: 'assistant', error: { name: 'MessageAbortedError' } }, parts: [{ type: 'text', text: 'cut' }] },
  ]), 'first');
  assert.equal(lastReplyV2([
    { type: 'user', text: 'q' },
    { type: 'assistant', content: [{ type: 'text', text: 'a' }, { type: 'tool', name: 'x' }] },
    { type: 'assistant', content: [{ type: 'tool', name: 'y' }] },
  ]), 'a');
});

test('settings from ~/.quotemem/env reach the plugin and the MCP launcher', async () => {
  const server = await serve({ '/memory/add': ok, '/memory/search': () => ({ body: { data: [] } }) });
  const fileHome = path.join(root, `home-${++dirSeq}`);
  fs.mkdirSync(path.join(fileHome, '.quotemem'), { recursive: true });
  const envFile = path.join(fileHome, '.quotemem', 'env');
  fs.writeFileSync(envFile, `# QuoteMem\nexport QUOTEMEM_URL=${server.url}\nQUOTEMEM_TOKEN="${TOKEN}"\nQUOTEMEM_USER_ID=quei\nOTHER=ignored\n`, { mode: 0o600 });
  const e = { ...WINDOWS_ENV, PATH: process.env.PATH, HOME: fileHome, USERPROFILE: fileHome, QUOTEMEM_STATE_DIR: path.join(root, `state-${++dirSeq}`), QUOTEMEM_CAPTURE_EVERY: '1' };
  const host = await v1(e);
  await host.turn('ses_E', 'I speak Portuguese.', 'Noted.');
  await waitFor(() => server.of('/memory/add').length === 1, WAIT_MS, 'a batch with settings from the file');
  assert.equal(server.of('/memory/add')[0].headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(server.of('/memory/add')[0].body.user_id, 'quei');

  // The MCP launcher reads the same file and hides quotemem_add.
  const child = spawn(process.execPath, [path.join(HERE, 'mcp.js')], { env: { ...WINDOWS_ENV, PATH: process.env.PATH, HOME: fileHome, USERPROFILE: fileHome }, stdio: ['pipe', 'pipe', 'pipe'] });
  let out = '';
  let err = '';
  child.stdout.on('data', (chunk) => { out += chunk; });
  child.stderr.on('data', (chunk) => { err += chunk; });
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  const answer = async (id) => {
    await waitFor(() => out.split('\n').some((line) => line.includes(`"id":${id}`)), 8000, `answer ${id}`);
    return JSON.parse(out.split('\n').find((line) => line.includes(`"id":${id}`)));
  };
  try {
    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'opencode', version: '1.19.6' } } });
    await answer(1);
    send({ jsonrpc: '2.0', method: 'notifications/initialized' });
    send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
    const names = (await answer(2)).result.tools.map((t) => t.name);
    assert.ok(names.includes('quotemem_search') && names.includes('quotemem_forget_prepare') && names.includes('quotemem_forget_confirm'));
    assert.ok(!names.includes('quotemem_add'), 'capture is automatic, so quotemem_add is hidden');
    send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
    await answer(3);
    assert.equal(server.of('/memory/search').at(-1).headers.authorization, `Bearer ${TOKEN}`);
    assert.ok(!out.includes(TOKEN) && !err.includes(TOKEN), 'the token never reaches the output');
  } finally {
    child.kill();
  }
});

test('install snippets parse and point at real files', () => {
  const repo = path.resolve(HERE, '..', '..');
  for (const name of ['opencode.v1.jsonc', 'opencode.v2.jsonc']) {
    const text = fs.readFileSync(path.join(HERE, 'install', name), 'utf8')
      .split('\n').filter((line) => !line.trim().startsWith('//')).join('\n')
      .replaceAll('/path/to/QuoteMem', JSON.stringify(repo).slice(1, -1));
    const config = JSON.parse(text);
    const plugins = config.plugin ?? config.plugins;
    assert.equal(plugins.length, 1);
    assert.ok(fs.existsSync(path.join(plugins[0], 'package.json')), `${name}: the plugin directory`);
    const mcp = config.mcp.quotemem ?? config.mcp.servers.quotemem;
    assert.equal(mcp.type, 'local');
    assert.equal(mcp.command[0], 'node');
    assert.ok(fs.existsSync(mcp.command[1]), `${name}: the MCP launcher`);
    assert.ok(!text.includes('TOKEN'), `${name}: no token in the config`);
  }
});
