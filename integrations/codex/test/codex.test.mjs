#!/usr/bin/env node
// Tests for the Codex adapter (hook.mjs, install.mjs, the plugin files).
//
//   node --test integrations/codex/test/codex.test.mjs
//
// Every hook call runs in its own process with the exact stdin JSON Codex
// sends (the shapes of codex-rs/hooks/schema/generated/*.input.schema.json),
// under a temporary HOME and CODEX_HOME, against a stub QuoteMem. Like Codex,
// the test kills the hook's whole process group as soon as the hook exits,
// so the background flusher must survive that on its own.
//
// With QUOTEMEM_CODEX_BIN set to a Codex CLI binary (or `codex` on PATH), the
// last tests also drive the real Codex: they install the plugin into a
// temporary CODEX_HOME, point Codex at a local fake model (Responses API),
// and check what reaches the model. Outbound traffic is sent to a dead proxy.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { HIT, TOKEN, plan, sleep, stubServer, tempDir, waitFor } from '../../core/test/helpers.mjs';

const PLUGIN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REPO = path.resolve(PLUGIN, '..', '..');
const HOOK = path.join(PLUGIN, 'hook.mjs');
const { FORGET_HEADER, RECALL_HEADER } = await import(pathToFileURL(path.join(PLUGIN, 'core', 'index.mjs')).href);
const installer = await import(pathToFileURL(path.join(PLUGIN, 'install.mjs')).href);

const root = tempDir('qm-codex-');
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });
async function serve(handlers) {
  const server = await stubServer(handlers);
  servers.add(server);
  return server;
}

const verified = (n = 1) => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: Array.from({ length: n }, (_, i) => ({ memory_id: `01JX0000000000000000000F${i}`, outcome: 'verified' })), scope_state: 'ready' } });

// ------------------------------------------------------- Codex payloads

const WORK = path.join(root, 'work');
fs.mkdirSync(WORK, { recursive: true });
const common = (session, event) => ({ session_id: session, transcript_path: null, cwd: WORK, hook_event_name: event, model: 'gpt-5.1', permission_mode: 'default' });
const P = {
  start: (s, source = 'startup') => ({ ...common(s, 'SessionStart'), source }),
  prompt: (s, turn, prompt, extra = {}) => ({ ...common(s, 'UserPromptSubmit'), turn_id: turn, prompt, ...extra }),
  stop: (s, turn, message) => ({ ...common(s, 'Stop'), turn_id: turn, stop_hook_active: false, last_assistant_message: message }),
  interrupt: (s, turn) => ({ ...common(s, 'Interrupt'), turn_id: turn }),
  end: (s) => ({ session_id: s, transcript_path: null, cwd: WORK, hook_event_name: 'SessionEnd', reason: 'other' }),
};
const SID = (n) => `019a7c2e-${String(n).padStart(4, '0')}-7d4e-9b1a-5c3f2e1d0a9b`;
const TID = (n) => `turn-${String(n).padStart(4, '0')}`;

let homeSeq = 0;
function setup(server, env = {}) {
  homeSeq += 1;
  const home = path.join(root, `home-${homeSeq}`);
  const codexHome = path.join(home, '.codex');
  fs.mkdirSync(codexHome, { recursive: true });
  // No flusher waits for idle chats beyond a test, unless the test sets a time.
  const base = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, CODEX_HOME: codexHome, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_IDLE_FLUSH_MS: '0', ...env };
  const stateDir = path.join(codexHome, 'quotemem');
  // One hook call as Codex makes it: stdin JSON, its own process group,
  // SIGKILL to the group once the hook exits (ProcessTreeGuard).
  const run = (argv, stdin) => new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn(process.execPath, [HOOK, ...argv], { env: base, cwd: WORK, stdio: ['pipe', 'pipe', 'pipe'], detached: true });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('exit', (code) => {
      const ms = Date.now() - started;
      try { process.kill(-child.pid, 'SIGKILL'); } catch { /* group gone */ }
      resolve({ code, out, err, ms });
    });
    child.stdin.on('error', () => {});
    child.stdin.end(stdin === undefined ? '' : typeof stdin === 'string' ? stdin : JSON.stringify(stdin));
  });
  const hook = (payload) => run([payload.hook_event_name], payload);
  const turn = async (s, n, user, reply) => {
    const prompt = await hook(P.prompt(s, TID(n), user));
    const stop = await hook(P.stop(s, TID(n), reply));
    return { prompt, stop };
  };
  return { home, codexHome, stateDir, env: base, run, hook, turn };
}

/** The parsed hook output, checked against Codex's UserPromptSubmit output schema. */
function contextOf(result) {
  assert.equal(result.code, 0);
  assert.equal(result.err, '', 'nothing on stderr');
  if (result.out === '') return null;
  const parsed = JSON.parse(result.out);
  assert.deepEqual(Object.keys(parsed), ['hookSpecificOutput'], 'no systemMessage, decision or continue');
  assert.deepEqual(Object.keys(parsed.hookSpecificOutput).sort(), ['additionalContext', 'hookEventName']);
  assert.equal(parsed.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  return parsed.hookSpecificOutput.additionalContext;
}

function silent(result) {
  assert.equal(result.code, 0);
  assert.equal(result.out, '', 'nothing on stdout');
  assert.equal(result.err, '', 'nothing on stderr');
}

function files(dir) {
  const out = [];
  const walk = (d) => {
    for (const name of fs.readdirSync(d)) {
      const p = path.join(d, name);
      if (fs.statSync(p).isDirectory()) walk(p);
      else out.push(p);
    }
  };
  if (fs.existsSync(dir)) walk(dir);
  return out;
}

// ---------------------------------------------------------------- hooks

test('capture: every 3 user turns from UserPromptSubmit + Stop; the flusher outlives the killed process group', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 300, body: { success: true } }) });
  const c = setup(server);
  silent(await c.hook(P.start(SID(1))));
  const t1 = await c.turn(SID(1), 1, 'I moved to Lisbon last spring.', 'Noted: Lisbon.');
  silent(t1.prompt);
  silent(t1.stop);
  await c.turn(SID(1), 2, 'thanks', 'You are welcome.');
  const t3 = await c.turn(SID(1), 3, 'My daughter is called Ana.', 'Nice name.');
  silent(t3.prompt);
  silent(t3.stop);
  assert.ok(t3.stop.ms < 1500, `Stop returned without waiting for /memory/add (${t3.stop.ms} ms)`);
  assert.equal(server.of('/memory/add').length, 0, 'two real user turns: nothing yet');
  await c.turn(SID(1), 4, 'I prefer window seats on long flights.', 'Window seats, noted.');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the batch, sent after the hook group was killed');
  const add = server.of('/memory/add')[0];
  assert.equal(add.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(add.body.user_id, 'quei');
  assert.equal(add.body.session_id, `codex:${SID(1)}`);
  assert.match(add.body.request_id, new RegExp(`^codex:${SID(1)}:t1-4:[0-9a-f]{16}$`));
  assert.deepEqual(add.body.messages.map((m) => `${m.role}:${m.content}`), [
    'user:I moved to Lisbon last spring.', 'assistant:Noted: Lisbon.',
    'user:My daughter is called Ana.', 'assistant:Nice name.',
    'user:I prefer window seats on long flights.', 'assistant:Window seats, noted.',
  ]);
  assert.equal(server.of('/memory/search').length, 0, 'statements never search');

  // Interrupt keeps the user's words; SessionEnd sends the rest, and no
  // earlier turn again (the service reads the previous batch as context).
  await c.hook(P.prompt(SID(1), TID(5), 'My bike is red.'));
  silent(await c.hook(P.interrupt(SID(1), TID(5))));
  silent(await c.hook(P.end(SID(1))));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the session-end batch');
  const last = server.of('/memory/add')[1].body;
  assert.match(last.request_id, /:t5-5:/);
  assert.deepEqual(last.messages.map((m) => m.content), ['My bike is red.']);

  // `codex exec resume`: the same thread in a new process numbers its turns on.
  await c.hook(P.prompt(SID(1), TID(6), 'My helmet is green.'));
  silent(await c.hook(P.stop(SID(1), TID(6), 'Green helmet.')));
  silent(await c.hook(P.end(SID(1))));
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'the resumed batch');
  assert.match(server.of('/memory/add')[2].body.request_id, /:t6-6:/, 'not t1-1 again');
});

test('idle: a chat left open is saved after QUOTEMEM_IDLE_FLUSH_MS by a flusher that waits, with no later hook', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }) });
  const c = setup(server, { QUOTEMEM_IDLE_FLUSH_MS: '1200', QUOTEMEM_DEBUG: '1' });
  const s = SID(80);
  await c.hook(P.prompt(s, TID(1), 'My kite is orange.'));
  const at = Date.now();
  const stop = await c.hook(P.stop(s, TID(1), 'Noted.'));
  silent(stop);
  assert.ok(stop.ms < 1200, `the Stop hook did not wait for the idle time (${stop.ms} ms)`);
  assert.equal(server.of('/memory/add').length, 0, 'nothing is saved before the idle time');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the idle save');
  const [add] = server.of('/memory/add');
  assert.ok(add.at - at >= 1200, `saved after the idle time (${add.at - at} ms)`);
  assert.equal(add.body.session_id, `codex:${s}`);
  assert.deepEqual(add.body.messages.map((m) => [m.role, m.content]), [['user', 'My kite is orange.'], ['assistant', 'Noted.']]);
  const log = () => { try { return fs.readFileSync(path.join(c.stateDir, 'quotemem.log'), 'utf8'); } catch { return ''; } };
  await waitFor(() => /flusher exit: .*no idle chat left/.test(log()), 8000, 'the waiting flusher to exit');
  assert.equal(server.of('/memory/add').length, 1, 'nothing is sent twice');
});

test('capture: a repeated hook (plugin and direct install both active) records a turn once', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }) });
  const c = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1' });
  for (let i = 0; i < 2; i += 1) await c.hook(P.prompt(SID(2), TID(1), 'My dog is called Rex.'));
  for (let i = 0; i < 2; i += 1) await c.hook(P.stop(SID(2), TID(1), 'Rex, noted.'));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'one batch');
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(server.of('/memory/add').length, 1);
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['My dog is called Rex.', 'Rex, noted.']);
});

test('recall: a memory question gets additionalContext; statements, subagents and duplicates get nothing', async () => {
  const server = await serve({
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User lives in Lisbon', { score: 0.9 })] } }),
    '/memory/add': () => ({ body: { success: true } }),
  });
  const c = setup(server);
  const plain = await c.hook(P.prompt(SID(3), TID(1), 'Refactor the parser into two functions.'));
  silent(plain);
  assert.equal(server.of('/memory/search').length, 0, 'an ordinary coding request does not search');
  const statement = await c.hook(P.prompt(SID(3), TID(2), 'I prefer Rust for new services.'));
  silent(statement);
  assert.equal(server.of('/memory/search').length, 0, 'a plain first-person statement does not search');

  const asked = await c.hook(P.prompt(SID(3), TID(3), 'Where do I live? Do you remember?'));
  const context = contextOf(asked);
  assert.ok(context.startsWith(`${RECALL_HEADER}\n`));
  assert.match(context, /untrusted background data/);
  assert.match(context, /&gt; "User lives in Lisbon"/);
  // At least 10 hits are asked for; only top_k (5) are inserted.
  assert.deepEqual(server.of('/memory/search')[0].body, { user_id: 'quei', query: 'Where do I live? Do you remember?', top_k: 10 });

  const sub = await c.hook(P.prompt(SID(4), TID(1), 'Where do I live?', { agent_id: 'agent-7', agent_type: 'explorer' }));
  silent(sub);
  assert.equal(server.of('/memory/search').length, 1, 'a subagent prompt is ignored');
  const again = await c.hook(P.prompt(SID(3), TID(4), 'Do you remember where I live?'));
  silent(again);
  assert.equal(server.of('/memory/search').length, 2, 'searched, but the same memory is not injected twice in a row');
  console.log(`# hook latency: plain prompt ${plain.ms} ms, statement ${statement.ms} ms, recall ${asked.ms} ms`);
});

test('a fleet of concurrent sessions: a new session cuts no batch of a session in use; each session end sends its turns in one batch', async () => {
  const stored = [];
  const server = await serve({
    '/memory/add': (body) => {
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true } };
    },
    '/memory/search': () => ({ body: { data: stored.filter((t) => t.includes('parser')).map((t, i) => HIT(`01JX00000000000000000000A${i}`, t, { score: 0.9 })) } }),
  });
  const c = setup(server);
  // Workers 21 and 22 are between turns; worker 23 has a turn running.
  for (const n of [21, 22, 23]) {
    silent(await c.hook(P.start(SID(n))));
    await c.turn(SID(n), 1, `Worker ${n}: the parser goes in src/parse${n}.rs.`, 'On it.');
  }
  silent(await c.hook(P.prompt(SID(23), TID(2), 'Now add the tests.')));
  // A new session starts and asks at once: the others keep their batches
  // (Codex ends each session with SessionEnd), so nothing is sent or waited for.
  silent(await c.hook(P.start(SID(24))));
  const asked = await c.hook(P.prompt(SID(24), TID(1), 'Do you remember where the parser goes?'));
  assert.equal(asked.code, 0);
  await sleep(500);
  assert.equal(server.of('/memory/add').length, 0, 'no batch of a session in use was cut');
  silent(await c.hook(P.stop(SID(23), TID(2), 'Tests added.')));
  for (const n of [21, 22, 23]) silent(await c.hook(P.end(SID(n))));
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'one batch per session');
  for (const n of [21, 22]) {
    const batch = server.of('/memory/add').find((r) => r.body.session_id === `codex:${SID(n)}`).body;
    assert.deepEqual(batch.messages.map((m) => m.content), [`Worker ${n}: the parser goes in src/parse${n}.rs.`, 'On it.']);
  }
  const busy = server.of('/memory/add').find((r) => r.body.session_id === `codex:${SID(23)}`).body;
  assert.deepEqual(busy.messages.map((m) => m.content), ['Worker 23: the parser goes in src/parse23.rs.', 'On it.', 'Now add the tests.', 'Tests added.']);
});

test('forget: pending turns are stored first, then prepare and confirm; the model gets one note', async () => {
  let prepared = null;
  const server = await serve({
    '/memory/add': () => ({ delay: 150, body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      prepared = plan(body);
      return { delay: 50, body: prepared };
    },
    '/memory/forget/confirm': () => verified(1),
  });
  const c = setup(server);
  await c.turn(SID(5), 1, 'I live in Lisbon.', 'Lisbon, noted.');
  await c.turn(SID(5), 2, 'My cat is Miso.', 'Miso!');
  const result = await c.hook(P.prompt(SID(5), TID(3), 'Please forget that I live in Lisbon.'));
  const note = contextOf(result);
  assert.deepEqual(server.requests.map((r) => r.route), ['/memory/add', '/memory/forget/prepare', '/memory/forget/confirm']);
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I live in Lisbon.', 'Lisbon, noted.', 'My cat is Miso.', 'Miso!']);
  const prepare = server.of('/memory/forget/prepare')[0].body;
  assert.equal(prepare.statement, 'I live in Lisbon');
  assert.equal(prepare.client_session_id, `codex:${SID(5)}`);
  assert.deepEqual(server.of('/memory/forget/confirm')[0].body, prepared.confirmation);
  assert.ok(note.startsWith(FORGET_HEADER));
  assert.match(note, /Done: 1 stored memory was forgotten/);
  assert.ok(!/quotemem_|request_id|01JX/.test(note), 'no tool names or ids');
  assert.ok(result.ms < 9000);

  // Chinese, and a forget slower than the hook budget.
  const slow = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/forget/prepare': (body) => ({ delay: 1500, body: plan(body) }),
    '/memory/forget/confirm': () => verified(2),
  });
  const d = setup(slow, { QUOTEMEM_FORGET_TIMEOUT_MS: '400' });
  const pending = contextOf(await d.hook(P.prompt(SID(6), TID(1), '忘掉我的电话号码')));
  assert.match(pending, /being forgotten in the background/);
  assert.equal(slow.of('/memory/forget/prepare')[0].body.statement, '我的电话号码');
  await waitFor(() => slow.of('/memory/forget/confirm').length === 1, 8000, 'the flusher finished the forget');
});

test('degradation: service down, no token, bad input, unknown event: exit 0, silent, fast', async () => {
  const down = await serve({});
  const url = down.url;
  await down.close();
  servers.delete(down);
  const c = setup({ url }, { QUOTEMEM_FORGET_TIMEOUT_MS: '600' });
  // A memory question while the service is down: one short note, so the
  // model does not say that nothing is stored.
  const asked = await c.hook(P.prompt(SID(7), TID(1), 'What is my address? Do you remember?'));
  assert.match(contextOf(asked), /^\[QuoteMem: memory unavailable\]/);
  assert.ok(asked.ms < 4000, `returned near the recall deadline (${asked.ms} ms)`);
  silent(await c.hook(P.stop(SID(7), TID(1), 'I do not know.')));
  silent(await c.hook(P.end(SID(7))));
  const forget = await c.hook(P.prompt(SID(8), TID(1), 'Please forget my address.'));
  assert.equal(forget.code, 0);
  assert.match(contextOf(forget), /background|could not/);

  const server = await serve({ '/memory/search': () => ({ body: { data: [] } }) });
  const noToken = setup(server, { QUOTEMEM_TOKEN: '' });
  silent(await noToken.hook(P.start(SID(9))));
  silent(await noToken.hook(P.prompt(SID(9), TID(1), 'Where do I live?')));
  assert.equal(server.requests.length, 0, 'without a token nothing is sent');
  const e = setup(server);
  silent(await e.run(['UserPromptSubmit'], 'not json'));
  silent(await e.run(['UserPromptSubmit'], ''));
  silent(await e.run(['PreToolUse'], { ...common(SID(9), 'PreToolUse'), tool_name: 'shell' }));
  silent(await e.run(['UserPromptSubmit'], { hook_event_name: 'UserPromptSubmit', prompt: 'Where do I live?' }));
});

test('notify fallback: records turns when no hook runs, stays out when hooks run', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }), '/memory/search': () => ({ body: { data: [] } }) });
  const c = setup(server, { QUOTEMEM_CAPTURE_EVERY: '2' });
  const notify = (thread, turn, inputs, reply) => c.run(['notify', JSON.stringify({ type: 'agent-turn-complete', 'thread-id': thread, 'turn-id': turn, cwd: WORK, client: 'codex-tui', 'input-messages': inputs, 'last-assistant-message': reply })]);
  silent(await notify(SID(10), 't1', ['I work at a bakery.'], 'A bakery!'));
  silent(await notify(SID(10), 't2', ['Where do I work?'], 'At a bakery.'));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the notify batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I work at a bakery.', 'A bakery!', 'Where do I work?', 'At a bakery.']);
  assert.equal(server.of('/memory/search').length, 0, 'notify cannot add context, so it never searches');

  // The same session through hooks: the notify call for that turn is ignored.
  await c.turn(SID(11), 1, 'My cat is Miso.', 'Miso!');
  silent(await notify(SID(11), TID(1), ['My cat is Miso.'], 'Miso!'));
  await c.turn(SID(11), 2, 'I play chess.', 'Chess.');
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the hook batch');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['My cat is Miso.', 'Miso!', 'I play chess.', 'Chess.']);
});

test('privacy: secrets redacted; no token or message text in the log or the state folder', async () => {
  const server = await serve({ '/memory/add': () => ({ status: 500, body: { code: 'internal' } }), '/memory/search': () => ({ status: 503, body: { code: 'busy' } }) });
  const c = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '10', QUOTEMEM_CAPTURE_MAX_ATTEMPTS: '2', QUOTEMEM_DEBUG: '1' });
  await c.turn(SID(12), 1, 'My key is sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz012345 and I live near the river.', 'Never share keys.');
  const busy = contextOf(await c.hook(P.prompt(SID(12), TID(2), 'What is my email again? My password is hunter2xyz')));
  assert.match(busy, /memory unavailable/);
  assert.ok(!busy.includes('hunter2xyz'));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'two attempts');
  assert.ok(!JSON.stringify(server.of('/memory/add')[0].body).includes('sk-proj-AbCd'));
  await waitFor(() => !fs.readdirSync(c.stateDir, { recursive: true }).some((name) => path.basename(name) === 'flusher.lock'), 5000, 'flusher exit');
  const disk = files(c.stateDir).map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  for (const text of ['sk-proj-AbCd', 'hunter2xyz', TOKEN]) assert.ok(!disk.includes(text), `state folder contains ${text}`);
  const log = fs.readFileSync(path.join(c.stateDir, 'quotemem.log'), 'utf8');
  assert.match(log, /dropped: internal after 2 attempts/);
  for (const text of [TOKEN, 'river', 'password', 'Never share']) assert.ok(!log.includes(text), `log contains ${text}`);
});

// ------------------------------------------------------------ installer

test('install.mjs: dry run, merge with the user\'s files, idempotent, notify only when free, uninstall', () => {
  const home = path.join(root, 'install-home');
  fs.mkdirSync(home, { recursive: true });
  const userHooks = { hooks: { Stop: [{ hooks: [{ type: 'command', command: 'say done', timeout: 5 }] }] } };
  const userConfig = 'model = "gpt-5.1"\n\n[mcp_servers.other]\ncommand = "other-mcp"\n';
  fs.writeFileSync(path.join(home, 'hooks.json'), JSON.stringify(userHooks));
  fs.writeFileSync(path.join(home, 'config.toml'), userConfig);
  const node = '/usr/local/bin/node';

  const dry = installer.install({ codexHome: home, node });
  assert.ok(dry.ok && dry.lines[0].startsWith('[dry run]'));
  assert.equal(fs.readFileSync(path.join(home, 'config.toml'), 'utf8'), userConfig, 'a dry run writes nothing');

  for (let i = 0; i < 2; i += 1) assert.ok(installer.install({ codexHome: home, node, write: true, notify: true }).ok);
  const hooks = JSON.parse(fs.readFileSync(path.join(home, 'hooks.json'), 'utf8'));
  assert.deepEqual(Object.keys(hooks.hooks).sort(), ['Interrupt', 'SessionEnd', 'SessionStart', 'Stop', 'UserPromptSubmit']);
  assert.equal(hooks.hooks.Stop.length, 2, 'the user\'s Stop hook is kept, ours added once');
  assert.equal(hooks.hooks.Stop[0].hooks[0].command, 'say done');
  assert.equal(hooks.hooks.UserPromptSubmit[0].hooks[0].command, `"${node}" "${HOOK}" UserPromptSubmit`);
  const config = fs.readFileSync(path.join(home, 'config.toml'), 'utf8');
  assert.equal(config.match(/\[mcp_servers\.quotemem\]/g).length, 1);
  assert.ok(config.startsWith('# >>> quotemem notify'), 'notify is a top-level key, before any table');
  assert.match(config, /\[mcp_servers\.other\]\ncommand = "other-mcp"/);
  assert.match(config, /env_vars = \["QUOTEMEM_URL", "QUOTEMEM_TOKEN_FILE", "QUOTEMEM_DATA_DIR"/);
  assert.ok(!config.includes(TOKEN));

  // A user notify is never replaced.
  const busy = path.join(root, 'install-busy');
  fs.mkdirSync(busy);
  fs.writeFileSync(path.join(busy, 'config.toml'), 'notify = ["my-notifier"]\n');
  const kept = installer.install({ codexHome: busy, node, write: true, notify: true });
  assert.ok(kept.lines.some((l) => l.includes('already sets notify')));
  assert.equal(fs.readFileSync(path.join(busy, 'config.toml'), 'utf8').match(/^notify/gm).length, 1);

  // Refusals change nothing.
  const inline = path.join(root, 'install-inline');
  fs.mkdirSync(inline);
  fs.writeFileSync(path.join(inline, 'config.toml'), '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = "x"\n');
  assert.equal(installer.install({ codexHome: inline, node, write: true }).ok, false);
  assert.ok(!fs.existsSync(path.join(inline, 'hooks.json')));

  assert.ok(installer.install({ codexHome: home, node, write: true, remove: true }).ok);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(home, 'hooks.json'), 'utf8')), userHooks);
  assert.equal(fs.readFileSync(path.join(home, 'config.toml'), 'utf8').trim(), userConfig.trim());
});

test('install.mjs: a hook command never lets the shell expand a path', () => {
  const command = installer.hookCommand('/opt/n$de `x` "q"\\bin/node', 'Stop');
  assert.ok(command.startsWith('"/opt/n\\$de \\`x\\` \\"q\\"\\\\bin/node" '), command);
  const echoed = execFileSync('/bin/sh', ['-c', `printf %s ${command.split(' "')[0]}`]).toString();
  assert.equal(echoed, '/opt/n$de `x` "q"\\bin/node');
  assert.ok(installer.isOurs(command));
});

test('plugin files are consistent (check_plugin.mjs)', () => {
  const out = execFileSync(process.execPath, [path.join(PLUGIN, 'check_plugin.mjs')], { encoding: 'utf8' });
  assert.match(out, /^ok: Codex plugin quotemem@/);
});

test('MCP server starts with the minimal environment Codex gives a stdio server', async () => {
  const server = await serve({});
  // Codex passes only HOME, PATH, … plus the env_vars and env of .mcp.json.
  const mcp = JSON.parse(fs.readFileSync(path.join(PLUGIN, '.mcp.json'), 'utf8')).mcpServers.quotemem;
  const source = { PATH: process.env.PATH, HOME: root, USERPROFILE: root, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', UNRELATED: 'x' };
  const env = { PATH: source.PATH, HOME: source.HOME, ...mcp.env };
  for (const name of mcp.env_vars) if (source[name] !== undefined) env[name] = source[name];
  const child = spawn(mcp.command, mcp.args, { cwd: path.resolve(PLUGIN, mcp.cwd), env, stdio: ['pipe', 'pipe', 'pipe'] });
  const lines = [];
  let buffer = '';
  child.stdout.on('data', (d) => {
    buffer += d;
    let i;
    while ((i = buffer.indexOf('\n')) >= 0) { lines.push(JSON.parse(buffer.slice(0, i))); buffer = buffer.slice(i + 1); }
  });
  const send = (msg) => child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', ...msg })}\n`);
  send({ id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'codex-test', version: '0' } } });
  send({ method: 'notifications/initialized' });
  send({ id: 2, method: 'tools/list' });
  await waitFor(() => lines.some((l) => l.id === 2), 5000, 'tools/list');
  child.kill();
  const tools = lines.find((l) => l.id === 2).result.tools.map((t) => t.name);
  assert.ok(tools.includes('quotemem_search') && tools.includes('quotemem_forget_prepare'));
  assert.ok(!tools.includes('quotemem_add'), 'automatic capture is on, so quotemem_add is hidden');
});

// ---------------------------------------------------------- real Codex

function codexBin() {
  if (process.env.QUOTEMEM_CODEX_BIN) return process.env.QUOTEMEM_CODEX_BIN;
  for (const dir of (process.env.PATH ?? '').split(path.delimiter)) {
    const file = path.join(dir, process.platform === 'win32' ? 'codex.exe' : 'codex');
    if (fs.existsSync(file)) return file;
  }
  return null;
}

/** A fake model: answers every Responses API call with one assistant message and records the requests. */
async function fakeModel(reply) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      let body = null;
      try { body = JSON.parse(Buffer.concat(chunks).toString('utf8') || 'null'); } catch { body = null; }
      requests.push({ method: req.method, url: req.url, body });
      if (req.method === 'POST' && req.url.endsWith('/responses')) {
        const ev = (o) => `event: ${o.type}\ndata: ${JSON.stringify(o)}\n\n`;
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.end(ev({ type: 'response.created', response: { id: 'resp-1' } })
          + ev({ type: 'response.output_item.done', item: { type: 'message', role: 'assistant', id: 'msg-1', content: [{ type: 'output_text', text: reply(body) }] } })
          + ev({ type: 'response.completed', response: { id: 'resp-1', usage: { input_tokens: 0, input_tokens_details: null, output_tokens: 0, output_tokens_details: null, total_tokens: 0 } } }));
        return;
      }
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end('{}');
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const close = () => new Promise((resolve) => { server.closeAllConnections?.(); server.close(() => resolve()); });
  servers.add({ close });
  return { url: `http://127.0.0.1:${server.address().port}`, requests };
}

const MODEL_CONFIG = (modelUrl) => [
  'model = "gpt-5.1"',
  'model_provider = "fakemodel"',
  'approval_policy = "never"',
  'sandbox_mode = "read-only"',
  'check_for_update_on_startup = false',
  '',
  '[model_providers.fakemodel]',
  'name = "fakemodel"',
  `base_url = "${modelUrl}/v1"`,
  'wire_api = "responses"',
  'request_max_retries = 0',
  'stream_max_retries = 0',
  '',
  '[analytics]',
  'enabled = false',
  '',
].join('\n');

function realCodex(bin, home, quotemem) {
  const codexHome = path.join(home, '.codex');
  const env = {
    PATH: process.env.PATH, HOME: home, USERPROFILE: home, CODEX_HOME: codexHome, OPENAI_API_KEY: 'unused-test-key',
    HTTPS_PROXY: 'http://127.0.0.1:9', HTTP_PROXY: 'http://127.0.0.1:9', NO_PROXY: '127.0.0.1,localhost',
    QUOTEMEM_URL: quotemem.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_IDLE_FLUSH_MS: '0',
  };
  const cli = (args) => execFileSync(bin, args, { env, cwd: WORK, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 60_000 });
  const exec = (prompt) => new Promise((resolve) => {
    const started = Date.now();
    const child = spawn(bin, ['exec', '--dangerously-bypass-hook-trust', '--skip-git-repo-check', '-C', WORK, prompt], { env, cwd: WORK, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { out += d; });
    child.on('exit', (code) => resolve({ code, out, ms: Date.now() - started }));
  });
  return { codexHome, cli, exec };
}

/** The model-visible text Codex sent with the last request. */
function lastInput(model) {
  const request = model.requests.filter((r) => r.method === 'POST' && r.url.endsWith('/responses')).at(-1);
  return (request?.body?.input ?? []).map((item) => ({ role: item.role, text: JSON.stringify(item.content ?? '') }));
}

const BIN = codexBin();

test('real Codex: plugin install, silent capture, recall and forget', { skip: BIN ? false : 'set QUOTEMEM_CODEX_BIN to run against a real Codex CLI', timeout: 240_000 }, async () => {
  let prepared = null;
  const quotemem = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User lives in Lisbon', { score: 0.9 })] } }),
    '/memory/forget/prepare': (body) => {
      prepared = plan(body);
      return { body: prepared };
    },
    '/memory/forget/confirm': () => verified(1),
  });
  const model = await fakeModel(() => 'Understood.');
  const home = path.join(root, 'real-plugin');
  fs.mkdirSync(path.join(home, '.codex'), { recursive: true });
  const codex = realCodex(BIN, home, quotemem);
  console.log(`# ${codex.cli(['--version']).trim()}`);
  fs.writeFileSync(path.join(codex.codexHome, 'config.toml'), MODEL_CONFIG(model.url));
  assert.match(codex.cli(['plugin', 'marketplace', 'add', REPO]), /Added marketplace `quotemem`/);
  assert.match(codex.cli(['plugin', 'add', 'quotemem@quotemem']), /Added plugin `quotemem`/);
  assert.match(codex.cli(['mcp', 'list']), /quotemem\s+node\s+\.\/server\.mjs/);

  // 1. A statement: no search, the turn is captured after Codex exits.
  const first = await codex.exec('I live in Lisbon and my cat is called Miso.');
  assert.equal(first.code, 0, first.out);
  assert.ok(!/hook: \w+ (Failed|Blocked|Stopped)/.test(first.out), first.out);
  await waitFor(() => quotemem.of('/memory/add').length >= 1, 10_000, 'the turn captured by the background flusher');
  const add = quotemem.of('/memory/add')[0].body;
  assert.match(add.session_id, /^codex:[0-9a-f-]{36}$/);
  assert.deepEqual(add.messages.map((m) => `${m.role}:${m.content}`), ['user:I live in Lisbon and my cat is called Miso.', 'assistant:Understood.']);
  assert.equal(quotemem.of('/memory/search').length, 0);

  // 2. A memory question: the recall block reaches the model as a developer message, not the user.
  const second = await codex.exec('Where do I live? Do you remember?');
  assert.equal(second.code, 0, second.out);
  assert.equal(quotemem.of('/memory/search').length, 1);
  const recalled = lastInput(model).filter((item) => item.text.includes('QuoteMem: retrieved long-term memory'));
  assert.equal(recalled.length, 1, 'one recall block in the model input');
  assert.equal(recalled[0].role, 'developer');
  assert.match(recalled[0].text, /User lives in Lisbon/);
  assert.ok(!second.out.includes('retrieved long-term memory'), 'the user does not see the recall block');

  // 3. A forget request: prepare and confirm, one note to the model, nothing shown.
  const third = await codex.exec('Please forget that I live in Lisbon.');
  assert.equal(third.code, 0, third.out);
  assert.equal(quotemem.of('/memory/forget/prepare').length, 1);
  assert.deepEqual(quotemem.of('/memory/forget/confirm')[0].body, prepared.confirmation);
  const note = lastInput(model).filter((item) => item.text.includes('QuoteMem: automatic forget'));
  assert.equal(note.length, 1);
  assert.match(note[0].text, /Done: 1 stored memory was forgotten/);
  assert.ok(!third.out.includes('automatic forget'));
  const state = files(path.join(codex.codexHome, 'quotemem')).map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  assert.ok(!state.includes(TOKEN), 'no token in the state folder or log');
  assert.ok(!files(codex.codexHome).some((f) => fs.readFileSync(f).includes(TOKEN)), 'no token anywhere in the Codex home');
  console.log(`# real Codex exec wall time: ${first.ms} / ${second.ms} / ${third.ms} ms (statement / recall / forget)`);
});

test('real Codex: direct install with install.mjs', { skip: BIN ? false : 'set QUOTEMEM_CODEX_BIN to run against a real Codex CLI', timeout: 120_000 }, async () => {
  const quotemem = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000002', 'User drinks green tea', { score: 0.8 })] } }),
  });
  const model = await fakeModel(() => 'Green tea.');
  const home = path.join(root, 'real-direct');
  fs.mkdirSync(path.join(home, '.codex'), { recursive: true });
  const codex = realCodex(BIN, home, quotemem);
  fs.writeFileSync(path.join(codex.codexHome, 'config.toml'), MODEL_CONFIG(model.url));
  assert.ok(installer.install({ codexHome: codex.codexHome, write: true }).ok);
  assert.match(codex.cli(['mcp', 'list']), /quotemem/);
  const run = await codex.exec('What do I usually drink? You should remember.');
  assert.equal(run.code, 0, run.out);
  assert.equal(quotemem.of('/memory/search').length, 1);
  assert.equal(lastInput(model).filter((item) => item.role === 'developer' && item.text.includes('User drinks green tea')).length, 1);
  await waitFor(() => quotemem.of('/memory/add').length === 1, 10_000, 'capture');
});
