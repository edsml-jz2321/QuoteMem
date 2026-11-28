#!/usr/bin/env node
// HookMemory end to end: every hook call runs in its own short-lived Node
// process, as a Claude Code or Codex hook does, against a stub QuoteMem.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, HIT, TOKEN, plan, sleep, stubServer, tempDir, testSettings, treeTexts, waitFor } from './helpers.mjs';

const core = await import(pathToFileURL(path.join(CORE, 'index.mjs')).href);
const { DiskQueue, FORGET_HEADER, RECALL_HEADER } = core;

const root = tempDir('qm-core-hooks-');
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });
async function serve(handlers, port) {
  const server = await stubServer(handlers, port);
  servers.add(server);
  return server;
}

// A hook entry point, as an adapter would write it: one call per process.
const HOOK = path.join(root, 'hook.mjs');
fs.writeFileSync(HOOK, `
import path from 'node:path';
import { HookMemory, fileSink, makeLog, resolveSettings } from ${JSON.stringify(pathToFileURL(path.join(CORE, 'index.mjs')).href)};
const settings = resolveSettings({}, process.env);
const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
const forgetWaitMs = process.env.TEST_FORGET_WAIT_MS ? Number(process.env.TEST_FORGET_WAIT_MS) : undefined;
const memory = new HookMemory({ settings, log, host: 'test-host', overlap: process.env.TEST_OVERLAP === '1', forgetWaitMs });
const [method, input] = process.argv.slice(2);
const out = await memory[method](JSON.parse(input));
process.stdout.write(JSON.stringify(out ?? null));
`);

let dirSeq = 0;
function setup(server, env = {}) {
  dirSeq += 1;
  const stateDir = env.QUOTEMEM_STATE_DIR ?? path.join(root, `state-${dirSeq}`);
  const base = {
    PATH: process.env.PATH,
    HOME: root, USERPROFILE: root,
    QUOTEMEM_URL: server.url,
    QUOTEMEM_TOKEN: TOKEN,
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_STATE_DIR: stateDir,
    ...env,
  };
  const call = (method, input) => new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn(process.execPath, [HOOK, method, JSON.stringify(input)], { env: base, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code !== 0) reject(new Error(`hook exited ${code}: ${err}`));
      else resolve({ ...(JSON.parse(out || 'null') ?? {}), ms: Date.now() - started, stderr: err });
    });
  });
  const turn = async (session, n, user, reply) => {
    const prompt = await call('prompt', { session, turn: `p${n}`, text: user, time: 1_763_500_000_000 + n * 1000 });
    await call('reply', { session, turn: `p${n}`, text: reply, time: 1_763_500_000_500 + n * 1000 });
    return prompt;
  };
  return { stateDir, call, turn, queue: () => new DiskQueue({ stateDir, url: base.QUOTEMEM_URL, userId: base.QUOTEMEM_USER_ID }, { warn() {}, warnOnce() {}, debug() {} }) };
}

// A prompt that asks to forget returns by its forget deadline
// (QUOTEMEM_FORGET_TIMEOUT_MS), plus the time the flusher's start takes,
// which the wait gets back (at most 1 s), plus the hook process's own start.
const FORGET_RETURN_MS = core.resolveSettings({}, { QUOTEMEM_URL: 'http://127.0.0.1:9', QUOTEMEM_TOKEN: TOKEN }).forgetTimeoutMs + 1000 + 700;

// The forget outcomes kept in `dir`. An outcome is written to a `.tmp` file and renamed into
// place, so a listing taken meanwhile can name a file that is gone when it is read.
const outcomes = (dir) => (fs.existsSync(dir) ? fs.readdirSync(dir).filter((name) => name.endsWith('.json')) : []);
const outcomeTexts = (dir) => outcomes(dir).map((name) => fs.readFileSync(path.join(dir, name), 'utf8'));

const verified = (n = 1) => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: Array.from({ length: n }, (_, i) => ({ memory_id: `01JX0000000000000000000F${i}`, outcome: 'verified' })), scope_state: 'ready' } });

test('capture: every 3 user turns, sent by the background flusher, nothing waits', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 200, body: { success: true, outcome: { added: 2 } } }) });
  const hooks = setup(server);
  await hooks.turn('S1', 1, 'I moved to Lisbon last spring.', 'Noted, Lisbon.');
  await hooks.turn('S1', 2, 'thanks', 'You are welcome.');
  const third = await hooks.turn('S1', 3, 'My daughter is called Ana.', 'Nice name.');
  assert.equal(third.context, null, 'a plain statement injects nothing');
  await sleep(100);
  assert.equal(server.of('/memory/add').length, 0, 'two user turns: nothing sent yet');
  // An interrupted turn (no reply) keeps the user's words.
  await hooks.call('prompt', { session: 'S1', turn: 'p4', text: 'I prefer window seats. Stop, never mind.' });
  const replyStarted = Date.now();
  await hooks.call('prompt', { session: 'S1', turn: 'p5', text: 'My bike is red.' });
  await hooks.call('reply', { session: 'S1', turn: 'p5', text: 'Red bike.' });
  assert.ok(Date.now() - replyStarted < 3000, 'hooks returned without waiting for /memory/add');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'one batch');
  const add = server.of('/memory/add')[0].body;
  assert.equal(add.user_id, 'quei');
  assert.equal(add.session_id, 'test-host:S1');
  // Turn numbers count the host's turns, the skipped acknowledgement included (as in DSH).
  assert.match(add.request_id, /^test-host:S1:t1-4:[0-9a-f]{16}$/);
  assert.deepEqual(add.messages.map((m) => `${m.role}:${m.content}`), [
    'user:I moved to Lisbon last spring.',
    'assistant:Noted, Lisbon.',
    'user:My daughter is called Ana.',
    'assistant:Nice name.',
    'user:I prefer window seats. Stop, never mind.',
  ]);
  assert.equal(server.of('/memory/search').length, 0, 'no recall on statements');
  // A repeated hook for the same turn records nothing twice.
  await hooks.call('prompt', { session: 'S2', turn: 'q1', text: 'My dog is Rex.' });
  await hooks.call('prompt', { session: 'S2', turn: 'q1', text: 'My dog is Rex.' });
  await hooks.call('reply', { session: 'S2', turn: 'q1', text: 'Rex!' });
  await hooks.call('reply', { session: 'S2', turn: 'q1', text: 'Rex!' });
  await hooks.call('end', { session: 'S2' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'session end batch');
  // S1 is still in use (idle under a minute): the new session left its batch.
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['My dog is Rex.', 'Rex!']);
});

test('capture: session end sends only new turns; a resumed session numbers on', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }) });
  const hooks = setup(server);
  await hooks.turn('E', 1, 'I live in Lisbon.', 'Lisbon.');
  await hooks.turn('E', 2, 'My cat is Miso.', 'Miso.');
  await hooks.turn('E', 3, 'I drink green tea.', 'Tea.');
  await hooks.turn('E', 4, 'I moved to Porto.', 'Porto.');
  await hooks.call('end', { session: 'E' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'two batches');
  const [, early] = server.of('/memory/add').map((r) => r.body);
  assert.match(early.request_id, /:t4-4:/);
  assert.deepEqual(early.messages.map((m) => m.content), ['I moved to Porto.', 'Porto.'], 'no earlier turn is sent twice');
  // `exec resume` / `--resume`: the same session goes on in a new process.
  await hooks.turn('E', 5, 'I play chess.', 'Chess.');
  await hooks.call('end', { session: 'E' });
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'the resumed turn');
  const resumed = server.of('/memory/add')[2].body;
  assert.match(resumed.request_id, /:t5-5:/, 'turn numbers go on after a resume');
  assert.deepEqual(resumed.messages.map((m) => m.content), ['I play chess.', 'Chess.']);
});

test('recall: gated, deduplicated across processes, breaker kept across processes', async () => {
  const server = await serve({
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User lives in Lisbon', { score: 0.9 }), HIT('01JX0000000000000000000002', 'weak', { score: 0.1 })] } }),
  });
  const hooks = setup(server, { QUOTEMEM_RECALL_REPEAT_TURNS: '3', QUOTEMEM_RECALL_MIN_SCORE: '0.3' });
  const first = await hooks.call('prompt', { session: 'R', turn: 'a1', text: 'Where do I live? You should remember.' });
  assert.equal(first.kind, 'recall');
  assert.ok(first.context.startsWith(`${RECALL_HEADER}\n`));
  assert.match(first.context, /untrusted background data/);
  assert.match(first.context, /&gt; "User lives in Lisbon"/);
  assert.ok(!first.context.includes('01JX0000000000000000000002'), 'below the threshold QUOTEMEM_RECALL_MIN_SCORE sets');
  assert.deepEqual(Object.keys(server.of('/memory/search')[0].body).sort(), ['query', 'top_k', 'user_id']);
  assert.equal(server.of('/memory/search')[0].body.top_k, 20, 'one search for the service\'s largest list');
  assert.match(first.context, /<memory-hit id="01JX0000000000000000000001" rank="1" score="0.90"/);
  const again = await hooks.call('prompt', { session: 'R', turn: 'a2', text: 'Do you remember where I live?' });
  assert.equal(again.context, null, 'injected a turn ago: not again');
  assert.equal(server.of('/memory/search').length, 2);
  assert.equal((await hooks.call('prompt', { session: 'R', turn: 'a3', text: 'I prefer Rust.' })).context, null);
  assert.equal(server.of('/memory/search').length, 2, 'a plain statement does not search');
  assert.equal((await hooks.call('prompt', { session: 'R', turn: 'a4', text: 'Remind me where I live?' })).kind, 'recall', 'after the repeat window it may come back');

  // A slow service: each hook returns at the deadline; one timeout is
  // forgiven, after two in a row the breaker (kept in a file) skips searches.
  const slow = await serve({ '/memory/search': () => ({ delay: 3000, body: { data: [] } }) });
  const slowHooks = setup(slow, { QUOTEMEM_RECALL_TIMEOUT_MS: '250' });
  const late = await slowHooks.call('prompt', { session: 'T', turn: 'b1', text: 'What is my address?' });
  assert.equal(late.kind, 'unavailable', 'a memory question that could not be searched gets the short note');
  assert.ok(late.ms < 2500, `returned near the deadline (${late.ms} ms)`);
  await slowHooks.call('prompt', { session: 'T', turn: 'b2', text: 'What is my address?' });
  assert.equal(slow.of('/memory/search').length, 2, 'one timeout does not turn recall off');
  await slowHooks.call('prompt', { session: 'T', turn: 'b3', text: 'What is my address?' });
  assert.equal(slow.of('/memory/search').length, 2, 'after two timeouts in a row the breaker skipped the search');

  // A service that is down: the turn goes on, nothing is printed to stderr.
  const down = await serve({});
  const downUrl = down.url;
  await down.close();
  servers.delete(down);
  const downHooks = setup({ url: downUrl });
  const result = await downHooks.call('prompt', { session: 'D', turn: 'c1', text: 'What is my address?' });
  assert.equal(result.kind, 'unavailable');
  // Not set up to start by itself (no ~/.quotemem/service.env): the first
  // such note of the session also says how to set it up.
  assert.match(result.context, /^\[QuoteMem: memory unavailable\] .*cannot check your memory right now\.\n\[QuoteMem: memory service not running\] .*`npx quotemem setup`/);
  assert.equal(result.stderr, '');
  const next = await downHooks.call('prompt', { session: 'D', turn: 'c2', text: 'What is my address?' });
  assert.match(next.context, /^\[QuoteMem: memory unavailable\] .*cannot check your memory right now\.$/, 'once per session');
  // Broad gate: a plain statement searches, but gets no note when the search fails.
  const broad = setup({ url: downUrl }, { QUOTEMEM_RECALL_GATE: 'broad' });
  const said = await broad.call('prompt', { session: 'D2', turn: 'c1', text: 'I prefer Rust for new services.' });
  assert.equal(said.context, null);
});

test('recall: an empty answer inserts nothing, in gated and in always mode', async () => {
  const server = await serve({ '/memory/search': () => ({ body: { data: [] } }) });
  for (const mode of ['gated', 'always']) {
    const hooks = setup(server, { QUOTEMEM_RECALL: mode });
    const result = await hooks.call('prompt', { session: `E-${mode}`, turn: 'e1', text: 'What is my blood type?' });
    assert.equal(result.context, null, `${mode}: no block and no note`);
    assert.equal(result.kind, null);
  }
  assert.equal(server.of('/memory/search').length, 2);
});

test('forget: pending turns stored first, then prepare and confirm, then one note', async () => {
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
  const hooks = setup(server);
  await hooks.turn('F', 1, 'I live in Lisbon.', 'Lisbon, noted.');
  await hooks.turn('F', 2, 'My cat is Miso.', 'Miso!');
  const forget = await hooks.call('prompt', { session: 'F', turn: 'p3', text: 'Please forget that I live in Lisbon.' });
  assert.equal(forget.kind, 'forget');
  assert.deepEqual(server.requests.map((r) => r.route), ['/memory/add', '/memory/forget/prepare', '/memory/forget/confirm'],
    'the pending turns were stored before the forget; no recall search on a forget turn');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I live in Lisbon.', 'Lisbon, noted.', 'My cat is Miso.', 'Miso!']);
  assert.ok(server.of('/memory/forget/prepare')[0].at >= server.of('/memory/add')[0].at + 140, 'prepare waited for the /memory/add answer');
  const prepare = server.of('/memory/forget/prepare')[0].body;
  assert.equal(prepare.statement, 'I live in Lisbon');
  assert.equal(prepare.client_session_id, 'test-host:F');
  assert.match(prepare.request_id, /^test-host:F:forget:[0-9a-f]{16}$/);
  assert.deepEqual(server.of('/memory/forget/confirm')[0].body, prepared.confirmation);
  assert.ok(forget.context.startsWith(FORGET_HEADER));
  assert.match(forget.context, /"I live in Lisbon"/);
  assert.match(forget.context, /Done: 1 stored memory was forgotten/);
  assert.ok(!/quotemem_|request_id|01JX/.test(forget.context), 'no tool names or ids for the user to see');
  assert.doesNotMatch(forget.context, /\btools?\b/i, 'the note does not mention tools or tool calls');

  // The forget turn itself is captured later, after the forget.
  await hooks.call('reply', { session: 'F', turn: 'p3', text: 'Done, I forgot it.' });
  await hooks.call('end', { session: 'F' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'next batch');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['Please forget that I live in Lisbon.', 'Done, I forgot it.'],
    'the batch after a forget does not send the forgotten turn again');
  assert.equal(server.of('/memory/forget/prepare').length, 1, 'one forget only');
});

test('forget: a new fact said with the request is not in the forget, and its turn is stored whole', async () => {
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server);
  await hooks.turn('N', 1, 'I live at 14 Elm Street.', 'Got it.');
  const text = 'Forget my old address, I live at 27 Birch Lane now.';
  const forget = await hooks.call('prompt', { session: 'N', turn: 'n2', text });
  assert.equal(forget.kind, 'forget');
  assert.deepEqual(server.of('/memory/forget/prepare').map((r) => r.body.statement), ['my old address']);
  await hooks.call('reply', { session: 'N', turn: 'n2', text: 'Done.' });
  await hooks.call('end', { session: 'N' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the request turn');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), [text, 'Done.'],
    'the request turn keeps the new fact, in the same words as the statement');
  // A name without "my" is a request too.
  await hooks.call('prompt', { session: 'N', turn: 'n3', text: "forget Ben's name" });
  assert.deepEqual(server.of('/memory/forget/prepare').map((r) => r.body.statement), ['my old address', "Ben's name"]);
});

test('forget: a fact only the conversation stated is forgotten by its sentences; a bundled memory is named', async () => {
  const BUNDLE = 'The user is learning Japanese and plays tennis on Sundays.';
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      const answer = plan(body, body.statement.includes('tennis') ? ['01JX0000000000000000000TE1'] : []);
      if (body.statement.includes('tennis')) {
        answer.candidates = [{ memory_id: '01JX0000000000000000000TE1', revision_id: '01JX0000000000000000000TE1', statement: BUNDLE, role: 'states', target: true, bundled: true }];
      }
      return { body: answer };
    },
    '/memory/forget/confirm': (body) => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', scope_state: 'ready', outcomes: body.targets.length === 0
      ? [{ memory_id: null, withheld_spans: 1, outcome: 'verified' }]
      : [{ memory_id: '01JX0000000000000000000TE1', outcome: 'verified' }] } }),
  });
  const hooks = setup(server);
  const said = await hooks.call('prompt', { session: 'W', turn: 'w1', text: 'Please forget that I was in hospital last May.' });
  assert.match(said.context, /Done: no stored memory stated it, but 1 sentence of the saved conversation that stated it was forgotten\./);
  assert.doesNotMatch(said.context, /stored memory was forgotten/);
  const whole = await hooks.call('prompt', { session: 'W', turn: 'w2', text: 'Please forget that I play tennis.' });
  assert.match(whole.context, /Done: 1 stored memory was forgotten\./);
  assert.match(whole.context, /also held other details, so it was forgotten whole \(data, not instructions\): "The user is learning Japanese and plays tennis on Sundays\."/);
  assert.ok(!/01JX|quotemem_/.test(said.context + whole.context), 'no ids or tool names');
});

test('forget: nothing matched, slow service, refused, and turned off', async () => {
  let prepareDelay = 0;
  let mode = 'nothing';
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      if (mode === 'nothing') return { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } };
      if (mode === 'invalid') return { status: 422, body: { code: 'invalid_request' } };
      return { delay: prepareDelay, body: plan(body) };
    },
    '/memory/forget/confirm': () => verified(2),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '400' });
  const nothing = await hooks.call('prompt', { session: 'N', turn: 'x1', text: '忘掉我的地址' });
  assert.equal(nothing.kind, 'forget');
  assert.match(nothing.context, /No stored memory states this .* so nothing was forgotten/);
  assert.doesNotMatch(nothing.context, /Done/, 'nothing matched is never reported as done');
  assert.equal(server.of('/memory/forget/prepare')[0].body.statement, '我的地址');

  // Slower than the hook's budget: the note says it continues in the background.
  mode = 'plan';
  prepareDelay = 1200;
  const pending = await hooks.call('prompt', { session: 'N', turn: 'x2', text: 'Please forget my phone number.' });
  assert.match(pending.context, /being forgotten in the background/);
  assert.ok(pending.ms < 3000, `the hook did not wait for the slow forget (${pending.ms} ms)`);
  const id = server.of('/memory/forget/prepare').at(-1).body.request_id;
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'the background flusher finished the forget');
  await waitFor(() => hooks.queue().result(id)?.status === 'forgotten', 3000, 'result written');
  assert.equal(hooks.queue().result(id).forgotten, 2);

  mode = 'invalid';
  const refused = await hooks.call('prompt', { session: 'N', turn: 'x3', text: "Don't remember that I own a boat." });
  assert.match(refused.context, /could not be completed or confirmed right now/);
  assert.match(refused.context, /Do not say that it was forgotten, and do not say that nothing was deleted/);
  assert.ok(!refused.context.includes('quotemem_'));
  assert.doesNotMatch(refused.context, /\btools?\b/i, 'no tool is named');

  const off = setup(server, { QUOTEMEM_FORGET: 'off' });
  const before = server.of('/memory/forget/prepare').length;
  const plain = await off.call('prompt', { session: 'O', turn: 'y1', text: 'Please forget my address.' });
  assert.notEqual(plain.kind, 'forget');
  assert.equal(server.of('/memory/forget/prepare').length, before, 'no silent forget when it is off');
});

test('forget: when the words miss, a stored statement that says the same thing is forgotten', async () => {
  const stored = {
    "The user's dentist is Dr. Haruki Sato.": '01JX0000000000000000000DE1',
    'The user is learning Japanese and takes lessons twice a week.': '01JX0000000000000000000JA1',
  };
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': (body) => ({ body: { data: /dentist/.test(body.query) ? [HIT('01JX0000000000000000000DE1', "The user's dentist is Dr. Haruki Sato.", { score: 0.8 })]
      : [HIT('01JX0000000000000000000JA1', 'The user is learning Japanese and takes lessons twice a week.', { score: 0.7 })] } }),
    '/memory/forget/prepare': (body) => (stored[body.statement]
      ? { body: plan(body, [stored[body.statement]]) }
      : { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server);
  // The old splitter cut this request at "Dr." and reported success.
  const dentist = await hooks.call('prompt', { session: 'M', turn: 'm1', text: 'Please forget that my dentist is Dr. Sato.' });
  assert.equal(server.of('/memory/forget/prepare')[0].body.statement, 'my dentist is Dr. Sato');
  assert.equal(server.of('/memory/search')[0].body.query, 'my dentist is Dr. Sato', 'nothing matched: the words are looked up');
  const second = server.of('/memory/forget/prepare')[1].body;
  assert.equal(second.statement, "The user's dentist is Dr. Haruki Sato.");
  assert.match(second.request_id, /:forget:[0-9a-f]{16}:m1$/);
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  assert.match(dentist.context, /Done: 1 stored memory was forgotten\. It read: "The user's dentist is Dr\. Haruki Sato\."/);
  // Another language: the closest statement is shown, nothing is claimed.
  const japanese = await hooks.call('prompt', { session: 'M', turn: 'm2', text: '请忘掉我在学日语这件事。' });
  assert.equal(server.of('/memory/forget/confirm').length, 1, 'no forget without a clear match');
  assert.match(japanese.context, /nothing was forgotten yet/);
  assert.match(japanese.context, /"The user is learning Japanese and takes lessons twice a week\."/);
  assert.doesNotMatch(japanese.context, /Done/);
});

test('forget: other sessions are saved first, and a late outcome is reported on the next turn', async () => {
  let prepareDelay = 0;
  const server = await serve({
    '/memory/add': () => ({ delay: 100, body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': () => ({ delay: prepareDelay, body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '400' });
  // Another session of the same host still holds an unsent turn.
  await hooks.turn('other', 1, 'I only drink green tea.', 'Green tea it is.');
  prepareDelay = 1500;
  const forget = await hooks.call('prompt', { session: 'main', turn: 'f1', text: 'Please forget that I drink green tea.' });
  assert.match(forget.context, /being forgotten in the background/);
  await waitFor(() => server.of('/memory/forget/prepare').length === 1, 8000, 'the forget ran');
  const add = server.of('/memory/add')[0];
  assert.deepEqual(add.body.messages.map((m) => m.content), ['I only drink green tea.', 'Green tea it is.']);
  assert.match(add.body.session_id, /:other$/);
  assert.ok(add.at < server.of('/memory/forget/prepare')[0].at, 'the other session was stored before the forget');
  const resultsDir = hooks.queue().resultsDir;
  await waitFor(() => outcomes(resultsDir).length === 1, 8000, 'the outcome');
  // The forget found nothing: the next turn tells the model, once.
  const next = await hooks.call('prompt', { session: 'main', turn: 'f2', text: 'Thanks. Now write a haiku.' });
  assert.match(next.context, /An earlier request of the user to forget something has finished now\. No stored memory matched it/);
  assert.doesNotMatch(next.context, /green tea/, 'a later note names nothing of the request');
  const after = await hooks.call('prompt', { session: 'main', turn: 'f3', text: 'And another haiku.' });
  assert.equal(after.context, null, 'reported once');
});

test('forget: a request in one language forgets the memory stored in another, by its quoted words', async () => {
  const STATEMENT = 'The user takes two Japanese lessons a week.';
  const compact = (id, statement, quote) => ({ id, score: 0.8, content: ['<memory untrusted="true">', `${statement} (said on Thursday, 19 November 2026)`, 'Status: current', `- user: "${quote}"`, '</memory>'].join('\n') });
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [compact('01JX0000000000000000000JA1', STATEMENT, '我正在学日语，每周上两次课'), compact('01JX0000000000000000000CO1', 'The user never drinks coffee.', '我从来不喝咖啡')] } }),
    '/memory/forget/prepare': (body) => (body.statement === STATEMENT
      ? { body: plan(body, ['01JX0000000000000000000JA1']) }
      : { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server);
  const result = await hooks.call('prompt', { session: 'L', turn: 'l1', text: '请忘掉我在学日语这件事。' });
  assert.deepEqual(server.of('/memory/forget/prepare').map((r) => r.body.statement), ['我在学日语这件事', STATEMENT]);
  assert.equal(server.of('/memory/search')[0].body.top_k, 10);
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  assert.match(result.context, /Done: 1 stored memory was forgotten\. It read: "The user takes two Japanese lessons a week\."/);
  assert.doesNotMatch(result.context, /closest/);
  // The note was given: the result on disk keeps no memory text.
  const results = hooks.queue().resultsDir;
  const kept = outcomeTexts(results).join('\n');
  assert.doesNotMatch(kept, /Japanese|日语/);
  assert.match(kept, /"status":"forgotten"/);
});

test('forget: a late outcome is reported to the next prompt of any session, even after its own session ended', async () => {
  let prepareDelay = 1500;
  let status = 'nothing_matched';
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000AD1', 'The user lives at 12 Rua Augusta.', { score: 0.9 })] } }),
    '/memory/forget/prepare': (body) => ({ delay: prepareDelay, body: status === 'ready' ? plan(body) : { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null } }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '300' });
  const first = await hooks.call('prompt', { session: 'A', turn: 'a1', text: 'Please forget that I live at 12 Rua Augusta.' });
  assert.match(first.context, /being forgotten in the background/);
  const running = fs.readFileSync(path.join(hooks.queue().ownerDir, 'forgets.json'), 'utf8');
  assert.match(running, /"requestId"/);
  assert.doesNotMatch(running, /Rua Augusta/, 'the list of running forgets keeps no word of the request');
  await hooks.call('end', { session: 'A' });
  const results = hooks.queue().resultsDir;
  await waitFor(() => outcomes(results).length === 1, 8000, 'the outcome');
  const [before] = outcomeTexts(results);
  assert.doesNotMatch(before, /Rua Augusta/, 'an outcome that came after its turn\'s note keeps no memory text');
  // Session A is gone; the next prompt of session B gets the note, once.
  const other = await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'Write a haiku about rain.' });
  assert.match(other.context, /An earlier request of the user to forget something has finished now\. No stored memory matched it .*so nothing was forgotten\. .*they can ask again in other words/);
  assert.doesNotMatch(other.context, /Rua Augusta|12 Rua/, 'another session gets back no word of the request nor of the closest memories');
  assert.doesNotMatch(outcomeTexts(results).join('\n'), /Rua Augusta/, 'read: no memory text kept');
  assert.equal((await hooks.call('prompt', { session: 'A', turn: 'a2', text: 'And another haiku.' })).context, null, 'reported once');

  // A late success is reported too.
  status = 'ready';
  const second = await hooks.call('prompt', { session: 'C', turn: 'c1', text: 'Please forget that my dentist is Dr. Sato.' });
  assert.match(second.context, /being forgotten in the background/);
  await waitFor(() => outcomes(results).length === 2, 8000, 'the second outcome');
  prepareDelay = 0;
  const later = await hooks.call('prompt', { session: 'C', turn: 'c2', text: 'Thanks. Now write a limerick.' });
  assert.match(later.context, /something has finished now\. Done: 1 stored memory was forgotten\. It was reported as still in progress before/);
  assert.doesNotMatch(later.context, /dentist|Sato/, 'the later note quotes neither the request nor the forgotten memory');
});

test('forget: a plain success that finished in the background is told only in the chat that asked; another chat gets no note and recalls as usual', async () => {
  let prepareDelay = 1500;
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000TB1', 'The user prefers tabs over spaces.', { score: 0.9 })] } }),
    '/memory/forget/prepare': (body) => ({ delay: prepareDelay, body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '300' });
  const asked = await hooks.call('prompt', { session: 'A', turn: 'a1', text: 'Please forget that my dentist is Dr. Sato.' });
  assert.match(asked.context, /being forgotten in the background/, 'the requesting chat\'s note is unchanged');
  const id = server.of('/memory/forget/prepare')[0].body.request_id;
  await waitFor(() => hooks.queue().result(id)?.status === 'forgotten', 8000, 'the forget finished');

  // A new chat: no word of the forget, and recall as usual.
  const other = await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'What do I prefer, tabs or spaces?' });
  assert.equal(other.kind, 'recall');
  assert.ok(other.context.startsWith(RECALL_HEADER), other.context);
  assert.match(other.context, /tabs over spaces/);
  assert.ok(!other.context.includes(FORGET_HEADER), other.context);
  assert.doesNotMatch(other.context, /forg[eo]t/i);
  assert.equal((await hooks.call('prompt', { session: 'B', turn: 'b2', text: 'Write a haiku about rain.' })).context, null, 'not kept for that chat either');

  // The chat that asked is told at its next prompt, once.
  const own = await hooks.call('prompt', { session: 'A', turn: 'a2', text: 'Thanks. Now write a limerick.' });
  assert.match(own.context, /An earlier request of the user to forget something has finished now\. Done: 1 stored memory was forgotten\. It was reported as still in progress before/);
  assert.equal((await hooks.call('prompt', { session: 'A', turn: 'a3', text: 'Another one.' })).context, null, 'reported once');
  assert.equal((await hooks.call('prompt', { session: 'B', turn: 'b3', text: 'And a haiku about snow.' })).context, null);

  // Done within the turn: the note comes at once in the chat that asked, as before; another chat gets none.
  prepareDelay = 0;
  const quick = await hooks.call('prompt', { session: 'A', turn: 'a4', text: 'Please forget that my bike is red.' });
  assert.match(quick.context, /The user's message asks to forget: "my bike is red"\. Done: 1 stored memory was forgotten/);
  assert.equal((await hooks.call('prompt', { session: 'B', turn: 'b4', text: 'Write a limerick about a cat.' })).context, null);

  // A running forget kept by an earlier version, with no chat named: told in any chat, as before.
  const queue = hooks.queue();
  fs.writeFileSync(path.join(queue.ownerDir, 'forgets.json'), JSON.stringify({ version: 1, items: [{ requestId: 'unnamed-1', at: Date.now() }] }));
  queue.writeResult('unnamed-1', { status: 'forgotten', forgotten: 1 });
  const unnamed = await hooks.call('prompt', { session: 'B', turn: 'b5', text: 'One more limerick.' });
  assert.match(unnamed.context, /finished now\. Done: 1 stored memory was forgotten/);
});

test('forget: two chats that each asked to forget are each told only of their own', async () => {
  const locker = new Set();
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      if (/locker/.test(body.statement)) locker.add(body.request_id);
      return { delay: 1500, body: plan(body) };
    },
    // The locker code was stored twice.
    '/memory/forget/confirm': (body) => verified(locker.has(body.request_id) ? 2 : 1),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '300' });
  assert.match((await hooks.call('prompt', { session: 'A', turn: 'a1', text: 'Please forget my locker code.' })).context, /being forgotten in the background/);
  assert.match((await hooks.call('prompt', { session: 'C', turn: 'c1', text: 'Please forget that my bike is red.' })).context, /being forgotten in the background/);
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'both forgets sent');
  const ids = server.of('/memory/forget/prepare').map((r) => r.body.request_id);
  await waitFor(() => ids.every((id) => hooks.queue().result(id)?.status === 'forgotten'), 8000, 'both forgets finished');
  // A third chat hears of neither.
  assert.equal((await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'Write a haiku about rain.' })).context, null);
  const a = await hooks.call('prompt', { session: 'A', turn: 'a2', text: 'Thanks. Write a limerick.' });
  assert.equal(a.context.split('has finished now').length - 1, 1, a.context);
  assert.match(a.context, /Done: 2 stored memories were forgotten/);
  const c = await hooks.call('prompt', { session: 'C', turn: 'c2', text: 'Thanks. Write a haiku.' });
  assert.equal(c.context.split('has finished now').length - 1, 1, c.context);
  assert.match(c.context, /Done: 1 stored memory was forgotten/);
  for (const [session, turn] of [['A', 'a3'], ['B', 'b2'], ['C', 'c3']]) {
    assert.equal((await hooks.call('prompt', { session, turn, text: 'Another one.' })).context, null, `${session}: reported once`);
  }
});

test('forget: an outcome that failed, or took other memories with it, is still told to whichever chat prompts next', async () => {
  let mode = 'fail';
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => (mode === 'fail'
      ? { delay: 1500, status: 422, body: { code: 'invalid_request' } }
      : { delay: 1500, body: { ...plan(body), affected: [{ memory_id: '01JX0000000000000000000DP1', effect: 'forgotten', statement: 'The user has a cat.' }] } }),
    '/memory/forget/confirm': () => ({ body: { ...verified(1).body, affected: [{ memory_id: '01JX0000000000000000000DP1', effect: 'forgotten' }] } }),
  });
  const hooks = setup(server, { QUOTEMEM_FORGET_TIMEOUT_MS: '300' });
  await hooks.call('prompt', { session: 'A', turn: 'a1', text: 'Please forget my cat\'s name.' });
  const failed = server.of('/memory/forget/prepare')[0].body.request_id;
  await waitFor(() => hooks.queue().result(failed)?.status === 'failed', 8000, 'the forget failed');
  const told = await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'Write a haiku about rain.' });
  assert.match(told.context, /An earlier request of the user to forget something has finished now\..*could not be completed/);
  assert.doesNotMatch(told.context, /Done/);
  assert.equal((await hooks.call('prompt', { session: 'A', turn: 'a2', text: 'Another haiku.' })).context, null, 'reported once');

  mode = 'related';
  await hooks.call('prompt', { session: 'A', turn: 'a3', text: 'Please forget my cat\'s name.' });
  const related = server.of('/memory/forget/prepare').at(-1).body.request_id;
  await waitFor(() => hooks.queue().result(related)?.status === 'forgotten', 8000, 'the forget finished');
  assert.equal(hooks.queue().result(related).dependents, 1);
  const extra = await hooks.call('prompt', { session: 'B', turn: 'b2', text: 'Write a limerick.' });
  assert.match(extra.context, /finished now\. Done: 2 stored memories were forgotten \(1 of them because it relied only on what the user asked to forget\)/);
  assert.match(extra.context, /Tell the user that a related memory was affected/);
  assert.doesNotMatch(extra.context, /cat/);
  assert.equal((await hooks.call('prompt', { session: 'A', turn: 'a4', text: 'Another limerick.' })).context, null, 'reported once');
});

test('forget: the turn waits no longer than its forget deadline; a slow forget is reported on the next turn', async () => {
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    // Slower than the deadline with all the time the wait gets back.
    '/memory/forget/prepare': (body) => ({ delay: FORGET_RETURN_MS, body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  // The default wait (QUOTEMEM_FORGET_TIMEOUT_MS unset).
  const hooks = setup(server);
  await hooks.turn('W', 1, 'My locker code is 4411.', 'Noted.');
  const forget = await hooks.call('prompt', { session: 'W', turn: 'w2', text: 'Please forget my locker code.' });
  assert.match(forget.context, /being forgotten in the background/);
  assert.ok(forget.ms < FORGET_RETURN_MS, `the hook returned in ${forget.ms} ms`);
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'the forget finished in the background');
  const id = server.of('/memory/forget/prepare')[0].body.request_id;
  await waitFor(() => hooks.queue().result(id)?.status === 'forgotten', 3000, 'result written');
  const next = await hooks.call('prompt', { session: 'W', turn: 'w3', text: 'Thanks. Now a haiku.' });
  assert.match(next.context, /something has finished now\. Done: 1 stored memory was forgotten/);
  assert.doesNotMatch(next.context, /locker/);
  assert.ok(next.ms < 2500);
});

test('forget: a retraction waits until the batch that holds the ask is stored, then forgets it', async () => {
  // The service saves the ask (202) and stores it about 6 s later, as in live
  // runs. Its in-chat check never reads "forget that I asked you for …", so
  // the forget must not run before the ask is stored.
  const ASK = 'Can you give me a shell command to count the lines in data.csv?';
  let storedAt = 0;
  const server = await serve({
    '/memory/add': (body) => {
      const holdsAsk = body.messages.some((m) => m.content === ASK);
      const reads = server.of('/memory/add').filter((r) => r.body.request_id === body.request_id).length;
      if (holdsAsk && reads < 4) return { status: 202, body: { success: true, status: 'pending', request_id: body.request_id } };
      if (holdsAsk && !storedAt) storedAt = Date.now();
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server);
  await hooks.turn('R', 1, ASK, 'wc -l data.csv');
  const forget = await hooks.call('prompt', { session: 'R', turn: 'r2', text: 'Please forget that I asked you for a shell command to count the lines in data.csv.' });
  assert.match(forget.context, /being forgotten in the background/, 'pending, never "nothing matched"');
  assert.ok(forget.ms < FORGET_RETURN_MS, `the hook returned in ${forget.ms} ms`);
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 15_000, 'the forget ran');
  const prepare = server.of('/memory/forget/prepare');
  assert.equal(prepare.length, 1);
  assert.equal(prepare[0].body.statement, 'I asked you for a shell command to count the lines in data.csv');
  assert.ok(storedAt > 0 && prepare[0].at >= storedAt, 'prepare ran only after the ask was stored');
  await waitFor(() => hooks.queue().result(prepare[0].body.request_id)?.status === 'forgotten', 3000, 'result written');
  const next = await hooks.call('prompt', { session: 'R', turn: 'r3', text: 'Thanks. Now a haiku.' });
  assert.match(next.context, /something has finished now\. Done: 1 stored memory was forgotten/);
  assert.doesNotMatch(next.context, /data\.csv|shell command/);
});

test('forget: a request the service does not read is applied again after the batch of its own message, which says the fact first', async () => {
  // "2) Please forget …" follows a list marker, so the service's in-chat
  // check does not read it. The fact before it is in the same message, which
  // is stored after the forget ran: the forget must run again after it.
  const server = await serve({
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1' });
  // Said now: a forget is kept a week from when it was asked.
  const now = Date.now();
  await hooks.call('prompt', { session: 'L', turn: 'p1', text: '1. My locker code is HERON. 2) Please forget my locker code.', time: now });
  await hooks.call('reply', { session: 'L', turn: 'p1', text: 'Done.', time: now + 500 });
  await waitFor(() => server.of('/memory/forget/prepare').length >= 2, 15_000, 'the forget ran again');
  const add = server.of('/memory/add').find((r) => r.body.messages.some((m) => m.role === 'user' && m.content.includes('locker code')));
  assert.ok(add, 'the message was stored');
  const prepares = server.of('/memory/forget/prepare');
  assert.ok(prepares.every((p) => p.body.statement === 'my locker code'));
  assert.ok(prepares.at(-1).at >= add.at, 'a forget ran after the batch that holds the fact');
  // A fact said only after such a request is kept: nothing runs again.
  const later = await serve({
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const kept = setup(later, { QUOTEMEM_CAPTURE_EVERY: '1' });
  await kept.call('prompt', { session: 'K', turn: 'p1', text: '1) Please forget my locker code. 2) My new locker code is EAGLE.', time: Date.now() });
  await kept.call('reply', { session: 'K', turn: 'p1', text: 'Done.', time: Date.now() });
  await waitFor(() => later.of('/memory/add').length >= 1 && later.of('/memory/forget/confirm').length >= 1, 15_000, 'the turn and the forget');
  await sleep(1500);
  assert.equal(later.of('/memory/forget/prepare').length, 1);
});

test('forget: the batch a forget runs again after holds its own message only, so the next turns\' facts are kept', async () => {
  // Every 3 turns would put the request's message and the next ones in one
  // batch; a forget applied again after it would forget the new code too.
  const server = await serve({
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '3' });
  const now = Date.now();
  await hooks.call('prompt', { session: 'O', turn: 'p1', text: 'My locker code is HERON. 2) Please forget my locker code.', time: now });
  await hooks.call('reply', { session: 'O', turn: 'p1', text: 'Done.', time: now + 100 });
  await hooks.call('prompt', { session: 'O', turn: 'p2', text: 'My new locker code is OWL.', time: now + 1000 });
  await hooks.call('reply', { session: 'O', turn: 'p2', text: 'Noted.', time: now + 1100 });
  await hooks.call('end', { session: 'O' });
  const holds = (text) => server.of('/memory/add').find((r) => r.body.messages.some((m) => m.content.includes(text)));
  await waitFor(() => holds('OWL') && server.of('/memory/forget/confirm').length >= 2, 15_000, 'the turns and both forgets');
  await sleep(1500);
  const own = holds('is HERON');
  assert.ok(own && !own.body.messages.some((m) => m.content.includes('OWL')), 'the request\'s message is a batch of its own');
  const prepares = server.of('/memory/forget/prepare');
  assert.equal(prepares.length, 2, 'the forget, and once again after its own batch');
  assert.ok(prepares.every((p) => p.at < holds('OWL').at), 'no forget runs after the batch with the new code');
});

test('sessions: a new chat sends what other chats left unsent once they are idle a minute; its first memory question waits for it, other prompts do not', async () => {
  // The service stores a batch in `addDelay` ms; search finds what it stored.
  const stored = [];
  let addDelay = 400;
  const server = await serve({
    '/memory/add': async (body) => {
      await sleep(addDelay);
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': (body) => ({ body: { data: stored.filter((t) => t.includes(/cat/.test(body.query) ? 'cat' : 'dog')).map((t, i) => HIT(`01JX00000000000000000000C${i}`, t, { score: 0.9 })) } }),
  });
  const hooks = setup(server);
  const sessionFile = (id) => fs.readdirSync(path.join(hooks.stateDir, 'sessions')).map((name) => path.join(hooks.stateDir, 'sessions', name))
    .find((file) => fs.readFileSync(file, 'utf8').includes(`"sessionId":"test-host:${id}"`));
  const idle = (id, ms) => { const at = new Date(Date.now() - ms); fs.utimesSync(sessionFile(id), at, at); };
  // Chat A says one fact (every 3 turns: nothing sent yet).
  await hooks.turn('A', 1, 'My cat is called Bob.', 'Nice name.');
  // A new chat at once: A may still be in use, so its batch waits, and nothing is waited for.
  const soon = await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'What is my cat called?' });
  assert.doesNotMatch(soon.context ?? '', /Bob/);
  assert.equal(server.of('/memory/add').length, 0, 'a chat in use keeps its batch');
  // A minute later, the new chat's next prompt sends A's turn; it waits for nothing.
  idle('A', 70_000);
  const plain = await hooks.call('prompt', { session: 'B', turn: 'b2', text: 'Hello again.' });
  assert.ok(plain.ms < 2500, `a prompt that searches nothing does not wait (${plain.ms} ms)`);
  await waitFor(() => stored.includes('My cat is called Bob.'), 8000, 'chat A\'s turn stored');
  assert.match((await hooks.call('prompt', { session: 'B', turn: 'b3', text: 'What is my cat called?' })).context ?? '', /Bob/);
  assert.equal(server.of('/memory/add').length, 1, 'one batch for chat A');
  // Chat C, idle a minute when chat D starts: D's start sends it, and D's first memory question waits for it.
  await hooks.turn('C', 1, 'My dog is called Rex.', 'Good dog.');
  idle('C', 70_000);
  await hooks.call('sessionStarted', { session: 'D' });
  const later = await hooks.call('prompt', { session: 'D', turn: 'd1', text: 'What is my dog called?' });
  assert.match(later.context ?? '', /Rex/);
  // A first prompt that searches nothing never waits, even for a slow store.
  addDelay = 2500;
  await hooks.turn('E', 1, 'I like green tea.', 'Noted.');
  idle('E', 70_000);
  const thanks = await hooks.call('prompt', { session: 'F', turn: 'f1', text: 'Thanks.' });
  assert.ok(thanks.ms < 2000, `it did not wait for the store (${thanks.ms} ms)`);
  await waitFor(() => stored.includes('I like green tea.'), 8000, 'chat E\'s turn stored');
  // Within a session, turns still go every 3.
  const adds = server.of('/memory/add').length;
  await hooks.turn('D', 2, 'I like tea.', 'Noted.');
  await sleep(500);
  assert.equal(server.of('/memory/add').length, adds);
});

test('sessions: a new session\'s first question waits only for the batches queued for it, not for one another chat cut on its own', async () => {
  const server = await serve({
    '/memory/add': async () => {
      await sleep(2500);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1' });
  await hooks.call('sessionStarted', { session: 'K' });
  // Chat L, in use, sends its own batch meanwhile; the store is slow.
  await hooks.turn('L', 1, 'My cat is called Bob.', 'Nice name.');
  const asked = await hooks.call('prompt', { session: 'K', turn: 'k1', text: 'What is my cat called?' });
  assert.ok(asked.ms < 2000, `it did not wait for chat L's batch (${asked.ms} ms)`);
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'chat L\'s batch');
});

test('sessions: a new session\'s first question waits at most 3 s in all, even while another process holds a chat\'s lock', async () => {
  const stored = [];
  const server = await serve({
    '/memory/add': async (body) => {
      await sleep(2500);
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const hooks = setup(server);
  const sessionFile = (id) => fs.readdirSync(path.join(hooks.stateDir, 'sessions')).map((name) => path.join(hooks.stateDir, 'sessions', name))
    .find((file) => fs.readFileSync(file, 'utf8').includes(`"sessionId":"test-host:${id}"`));
  const idle = (id, ms) => { const at = new Date(Date.now() - ms); fs.utimesSync(sessionFile(id), at, at); };
  await hooks.turn('A', 1, 'My cat is called Bob.', 'Nice name.');
  await hooks.turn('C', 1, 'My dog is called Rex.', 'Good dog.');
  idle('A', 70_000);
  idle('C', 70_000);
  // Another process holds chat A's lock (a start queueing it, a slow disk).
  const held = core.tryLock(`${sessionFile('A')}.lock`);
  assert.ok(held);
  try {
    const asked = await hooks.call('prompt', { session: 'B', turn: 'b1', text: 'What is my dog called?' });
    assert.ok(asked.ms < 4500, `the lock and the store were waited for within 3 s in all (${asked.ms} ms)`);
  } finally {
    core.releaseLock(held);
  }
  await waitFor(() => stored.includes('My dog is called Rex.'), 8000, 'chat C\'s turn stored');
});

test('turns that overlap: a prompt sent while the earlier turn runs leaves it open; each end saves its own turn once, in either order', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }), '/memory/search': () => ({ body: { data: [] } }) });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1', TEST_OVERLAP: '1' });
  const batchOf = (user) => server.of('/memory/add').map((r) => r.body.messages).filter((m) => m.some((x) => x.role === 'user' && x.content === user));
  const run = async (order) => {
    for (const [turn, text] of order.prompts) await hooks.call('prompt', { session: 'V', turn, text });
    for (const [turn, reply] of order.replies) await hooks.call('reply', { session: 'V', turn, text: reply });
  };
  await run({ prompts: [['v1', 'My kite is orange.'], ['v2', 'My bike is blue.']], replies: [['v1', 'Orange kite.'], ['v2', 'Blue bike.']] });
  await run({ prompts: [['v3', 'My boat is green.'], ['v4', 'My owl is white.']], replies: [['v4', 'White owl.'], ['v3', 'Green boat.']] });
  await waitFor(() => server.of('/memory/add').length === 4, 8000, 'four batches');
  for (const [user, reply] of [['My kite is orange.', 'Orange kite.'], ['My bike is blue.', 'Blue bike.'], ['My boat is green.', 'Green boat.'], ['My owl is white.', 'White owl.']]) {
    const batches = batchOf(user);
    assert.equal(batches.length, 1, user);
    assert.deepEqual(batches[0].map((m) => m.content), [user, reply], user);
  }
  // A turn whose end never comes: once a later turn has ended (a chat runs
  // its turns one at a time), it is saved a minute later, at the next hook,
  // with the user's words; after that later turn.
  await run({ prompts: [['v5', 'My cat is grey.'], ['v6', 'My dog is brown.']], replies: [['v6', 'Brown dog.']] });
  await waitFor(() => batchOf('My dog is brown.').length === 1, 8000, 'the later turn');
  const file = fs.readdirSync(path.join(hooks.stateDir, 'sessions')).map((name) => path.join(hooks.stateDir, 'sessions', name))
    .find((f) => fs.readFileSync(f, 'utf8').includes('"sessionId":"test-host:V"'));
  const state = JSON.parse(fs.readFileSync(file, 'utf8'));
  assert.equal(state.running.length, 1);
  state.running[0].superseded.at -= 61_000;
  fs.writeFileSync(file, JSON.stringify(state));
  await hooks.call('prompt', { session: 'V', turn: 'v7', text: 'My fish is gold.' });
  await waitFor(() => batchOf('My cat is grey.').length === 1, 8000, 'the turn whose end never came');
  assert.deepEqual(batchOf('My cat is grey.')[0].map((m) => m.content), ['My cat is grey.']);
});

test('outage: turns queued while the service was down are sent as soon as a search reaches it again, not after the retry delay', async () => {
  const down = await stubServer({});
  const port = down.port;
  await down.close();
  const hooks = setup({ url: `http://127.0.0.1:${port}` }, { QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '600000', QUOTEMEM_FLUSHER_MAX_WAIT_MS: '1000' });
  await hooks.turn('S1', 1, 'I moved to Porto last spring.', 'Noted: Porto.');
  // The background flusher was refused; the batch waits minutes for its retry.
  await waitFor(() => hooks.queue().entries().some((entry) => fs.readFileSync(entry.file, 'utf8').includes('"unreachableAt"')), 8000, 'the refused attempt');
  const up = await serve({
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User lives in Porto', { score: 0.9 })] } }),
  }, port);
  // A prompt that searches nothing does not reach the service: the batch still waits.
  await hooks.call('prompt', { session: 'S1', turn: 'p2', text: 'Let us write a function.' });
  await sleep(800);
  assert.equal(up.of('/memory/add').length, 0);
  // A question about the past searches; the service answers, and the batch goes now.
  const asked = await hooks.call('prompt', { session: 'S1', turn: 'p3', text: 'Where do I live? You should remember.' });
  assert.equal(asked.kind, 'recall');
  await waitFor(() => up.of('/memory/add').length >= 1, 8000, 'the batch from the outage');
  assert.deepEqual(up.of('/memory/add')[0].body.messages.map((m) => m.content), ['I moved to Porto last spring.', 'Noted: Porto.']);

  // A second outage, and a session that searches nothing: the flusher's
  // own /health probe (at most every 30 s) finds the service again.
  await up.close();
  servers.delete(up);
  await hooks.turn('S2', 1, 'My bike is a red Brompton.', 'Nice.');
  await waitFor(() => hooks.queue().entries().some((entry) => fs.readFileSync(entry.file, 'utf8').includes('"unreachableAt"')), 8000, 'the second refused attempt');
  const back = await serve({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }), '/health': () => ({ body: { status: 'ok' } }) }, port);
  await waitFor(() => !hooks.queue().flusherRunning(), 8000, 'the flusher that was refused');
  fs.writeFileSync(hooks.queue().probeFile, JSON.stringify({ at: Date.now() - 31_000 }));
  await hooks.call('prompt', { session: 'S2', turn: 'p2', text: 'Let us write another function.' });
  await waitFor(() => back.of('/memory/add').some((r) => r.body.messages[0].content === 'My bike is a red Brompton.'), 8000, 'the batch, after the probe');
  assert.equal(back.of('/health').length, 1);
  assert.equal(back.of('/memory/search').length, 0, 'nothing searched');
});

test('users: two users of one state folder never send each other\'s turns nor see each other\'s forget outcomes', async () => {
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => ({ delay: 2200, body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
  });
  const stateDir = path.join(root, 'shared-users');
  // One host profile, two QUOTEMEM_USER_ID values (as two DSH users of one profile).
  const u1 = setup(server, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_USER_ID: 'u1', QUOTEMEM_IDLE_FLUSH_MS: '50' });
  const m1 = setup(server, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_USER_ID: 'm1', QUOTEMEM_IDLE_FLUSH_MS: '50' });
  await u1.turn('A', 1, 'I only drink oolong tea.', 'Oolong, noted.');
  const forget = await u1.call('prompt', { session: 'A', turn: 'a2', text: 'Please forget that I only drink oolong tea.' });
  assert.match(forget.context, /being forgotten in the background/);
  // u1 leaves an idle session with an unsent turn.
  await u1.turn('A2', 1, 'My locker code is 4411.', 'Noted.');
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'u1\'s forget finished');
  const id = server.of('/memory/forget/prepare')[0].body.request_id;
  await waitFor(() => u1.queue().result(id)?.status === 'forgotten', 3000, 'u1\'s result written');
  await sleep(100);

  // m1's next session: no word of u1's forget, and u1's idle turn is not sent as m1.
  const other = await m1.call('prompt', { session: 'B', turn: 'b1', text: 'Write a haiku about rain.' });
  assert.equal(other.context, null, 'another user\'s forget outcome is never shown');
  assert.equal(m1.queue().result(id), null, 'nor readable as m1');
  await m1.call('reply', { session: 'B', turn: 'b1', text: 'Rain on the roof.' });
  await m1.call('end', { session: 'B' });
  await waitFor(() => server.of('/memory/add').some((r) => r.body.user_id === 'm1'), 8000, 'm1\'s batch');
  await sleep(300);
  for (const add of server.of('/memory/add')) {
    const text = add.body.messages.map((m) => m.content).join(' ');
    if (/oolong|locker/.test(text)) assert.equal(add.body.user_id, 'u1', `u1's words were sent as ${add.body.user_id}`);
    if (/haiku/.test(text)) assert.equal(add.body.user_id, 'm1');
  }
  assert.ok(!server.of('/memory/add').some((r) => /locker/.test(JSON.stringify(r.body))), 'u1\'s idle turn waits for u1');

  // u1's other chat hears nothing of a plain success, and its idle turn is sent as u1.
  assert.equal((await u1.call('prompt', { session: 'C', turn: 'c1', text: 'Thanks. Now a limerick.' })).context, null, 'a plain success is told only to the chat that asked');
  // u1's chat that asked gets its own outcome, once.
  const own = await u1.call('prompt', { session: 'A', turn: 'a3', text: 'Thanks. Now a limerick.' });
  assert.match(own.context, /An earlier request of the user to forget something has finished now\. Done/);
  assert.doesNotMatch(own.context, /oolong/);
  assert.equal((await u1.call('prompt', { session: 'A', turn: 'a4', text: 'Another one.' })).context, null, 'reported once');
  await waitFor(() => server.of('/memory/add').some((r) => /locker/.test(JSON.stringify(r.body))), 8000, 'u1\'s idle turn');
  assert.equal(server.of('/memory/add').find((r) => /locker/.test(JSON.stringify(r.body))).body.user_id, 'u1');
  assert.equal(server.of('/memory/forget/prepare')[0].body.user_id, 'u1');
});

test('users: one user\'s turns said to one service are never sent to another service of the same state folder', async () => {
  const a = await serve({ '/memory/add': () => ({ body: { success: true } }), '/memory/search': () => ({ body: { data: [] } }) });
  const b = await serve({ '/memory/add': () => ({ body: { success: true } }), '/memory/search': () => ({ body: { data: [] } }) });
  const stateDir = path.join(root, 'shared-urls');
  // One host profile and user, a different QUOTEMEM_URL per project.
  const atA = setup(a, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_IDLE_FLUSH_MS: '50' });
  const atB = setup(b, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_IDLE_FLUSH_MS: '50' });
  await atA.turn('work-1', 1, 'Our Q3 acquisition target is Contoso, keep it confidential.', 'Understood.');
  await sleep(200);
  // B's prompt sweeps idle sessions: A's is not B's.
  await atB.call('prompt', { session: 'home-1', turn: 'h1', text: 'Write a haiku about rain.' });
  await sleep(1500);
  assert.ok(!b.of('/memory/add').some((r) => /Contoso/.test(JSON.stringify(r.body))), 'A\'s turn is never sent to B');
  // A's own sweep sends it to A.
  await atA.call('prompt', { session: 'work-2', turn: 'w1', text: 'Write a limerick.' });
  await waitFor(() => a.of('/memory/add').some((r) => /Contoso/.test(JSON.stringify(r.body))), 8000, 'A\'s idle turn sent to A');
  assert.ok(!b.of('/memory/add').some((r) => /Contoso/.test(JSON.stringify(r.body))));
});

test('in-chat forget: what the service forgot while storing a batch is told once, to that user only, without its words on disk', async () => {
  const server = await serve({
    '/memory/add': (body) => ({
      body: {
        success: true,
        outcome: { added: 0 },
        agent: {
          forgets: body.messages.some((m) => /oolong/.test(m.content))
            ? [
              { request: 'I only drink oolong tea', status: 'forgotten', forgotten: 1, withheld_spans: 2, bundled: 1, earlier: false },
              { request: 'my phone number', status: 'nothing_matched', forgotten: 0, withheld_spans: 0, bundled: 0, earlier: true },
            ]
            : [],
        },
      },
    }),
    '/memory/search': () => ({ body: { data: [] } }),
  });
  const stateDir = path.join(root, 'shared-chat');
  // The plugin's own forget is off for u1: only the service's in-chat forget runs.
  const u1 = setup(server, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_USER_ID: 'u1', QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_FORGET: 'off' });
  const m1 = setup(server, { QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_USER_ID: 'm1', QUOTEMEM_CAPTURE_EVERY: '1' });
  await u1.turn('A', 1, 'Please forget that I only drink oolong tea.', 'Sure.');
  const file = u1.queue().chatForgetsFile;
  await waitFor(() => fs.existsSync(file), 8000, 'the outcome kept');
  assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /oolong|phone|request/i, 'counts only: the request words are never written');

  assert.equal((await m1.call('prompt', { session: 'B', turn: 'b1', text: 'Write a haiku.' })).context, null, 'never told to another user');
  const next = await u1.call('prompt', { session: 'A', turn: 'p2', text: 'Write a haiku.' });
  assert.equal(next.kind, 'forget');
  assert.match(next.context, /For a request to forget in an earlier message, 1 stored memory was forgotten, and 2 more sentences of the saved conversation that stated it\./);
  assert.match(next.context, /also held other details/);
  assert.doesNotMatch(next.context, /nothing more was found/, 'an outcome the plugin\'s own forget already gave is not repeated');
  assert.doesNotMatch(next.context, /oolong|\btools?\b/i);
  assert.equal((await u1.call('prompt', { session: 'A', turn: 'p3', text: 'Another.' })).context, null, 'told once');
  assert.ok(!fs.existsSync(file));
});

test('hosts without turn ids: an unanswered prompt does not swallow the next; each forget is its own operation', async () => {
  const forgets = [];
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [] } }),
    '/memory/forget/prepare': (body) => {
      forgets.push(body.request_id);
      return { body: plan(body) };
    },
    '/memory/forget/confirm': () => verified(1),
  });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1' });
  // Gemini, Cline and goose pass no turn id. The first prompt gets no reply.
  await hooks.call('prompt', { session: 'G', text: 'I live in Paris now.' });
  const forget = await hooks.call('prompt', { session: 'G', text: 'Forget that I live in Paris please.' });
  assert.equal(forget.kind, 'forget', 'the second prompt is not taken for a repeat of the first');
  assert.match(forget.context, /Done: 1 stored memory was forgotten/);
  await hooks.call('reply', { session: 'G', text: 'Done.' });
  await hooks.call('end', { session: 'G' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'both turns');
  assert.deepEqual(server.of('/memory/add').map((r) => r.body.messages.map((m) => `${m.role}:${m.content}`)), [
    ['user:I live in Paris now.'],
    ['user:Forget that I live in Paris please.', 'assistant:Done.'],
  ], 'the reply is paired with the message it answers');

  // A second install of the same hooks: the same text at once is one prompt.
  await hooks.call('prompt', { session: 'D', text: 'My cat is Miso.' });
  await hooks.call('prompt', { session: 'D', text: 'My cat is Miso.' });
  await hooks.call('reply', { session: 'D', text: 'Miso!' });
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'one batch');
  assert.deepEqual(server.of('/memory/add')[2].body.messages.map((m) => m.content), ['My cat is Miso.', 'Miso!']);

  // With capture off the turn number never moves; a repeated forget is still sent.
  const off = setup(server, { QUOTEMEM_CAPTURE_EVERY: 'off' });
  await off.call('prompt', { session: 'R', text: 'Forget that I live in Paris.' });
  await off.call('reply', { session: 'R', text: 'ok' });
  await off.call('prompt', { session: 'R', text: 'I live in Paris again.' });
  await off.call('reply', { session: 'R', text: 'noted' });
  const again = await off.call('prompt', { session: 'R', text: 'Forget that I live in Paris.' });
  assert.match(again.context, /Done: 1 stored memory was forgotten/);
  const ours = forgets.filter((id) => id.startsWith('test-host:R:'));
  assert.equal(ours.length, 2, 'the second request reached the service');
  assert.notEqual(ours[0], ours[1]);
});

test('sessions: jobs are queued before the state that drops them is saved; week-old sessions close with idle flush off', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }) });
  const hooks = setup(server);
  const { HookMemory, resolveSettings } = core;
  const settings = resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: hooks.stateDir, QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_IDLE_FLUSH_MS: 'off' });
  const memory = new HookMemory({ settings, log: { warn() {}, warnOnce() {}, debug() {} }, host: 'test-host' });
  memory.flush = () => false;
  const seen = [];
  const enqueue = memory.enqueue.bind(memory);
  memory.enqueue = (jobs) => {
    if (jobs.length > 0) seen.push(JSON.parse(fs.readFileSync(memory.fileOf('test-host:K'), 'utf8')).open?.users.map((u) => u.text));
    enqueue(jobs);
  };
  await memory.prompt({ session: 'K', turn: 'k1', text: 'I live in Lisbon.' });
  await memory.reply({ session: 'K', turn: 'k1', text: 'Lisbon.' });
  assert.deepEqual(seen, [['I live in Lisbon.']], 'the saved session still held the turn when its batch was queued');
  assert.equal(hooks.queue().entries().length, 1);

  // An old session with an unsent turn and no clean end (Cline has none).
  await memory.prompt({ session: 'old', turn: 'o1', text: 'I drink green tea.' });
  const file = memory.fileOf('test-host:old');
  const weekAgo = new Date(Date.now() - 8 * 86_400_000);
  fs.utimesSync(file, weekAgo, weekAgo);
  await memory.prompt({ session: 'new', turn: 'n1', text: 'Hello there.' });
  assert.ok(!fs.existsSync(file), 'the abandoned session file is removed');
  const queued = hooks.queue().entries().map((e) => JSON.parse(fs.readFileSync(e.file, 'utf8')).body);
  assert.ok(queued.some((b) => b.session_id === 'test-host:old' && b.messages.some((m) => m.content === 'I drink green tea.')), 'its turn is queued');
});

test('earlier layout: its files are never sent, and are removed once past their age', async () => {
  // A state folder written before owners: job and session files that name no
  // owner or user (one user's turns may carry another user's id), the queue
  // state, the flusher lock, the running-forget list and outcomes.
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }) });
  const hooks = setup(server);
  const dir = hooks.stateDir;
  const old = new Date(Date.now() - 8 * 86_400_000);
  const write = (rel, data, time = old) => {
    const file = path.join(dir, rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(data));
    fs.utimesSync(file, time, time);
    return file;
  };
  const job = { version: 1, kind: 'add', request_id: 'legacy:1', url: server.url, created: old.getTime(), attempts: 0, nextAt: 0, key: 'aaaaaaaaaaaaaaaa',
    body: { schema_version: 1, user_id: 'quei', session_id: 'legacy', request_id: 'legacy:1', messages: [{ role: 'user', content: 'My locker code is 4411.' }] } };
  const oldFiles = [
    write('queue/000001763500000-123-000001-aaaaaaaaaaaaaaaa.json', job),
    write('queue/state.json', { storedAt: 1 }),
    write('flusher.lock', { pid: 1, nonce: 'x' }),
    write('forgets.json', { version: 1, items: [{ requestId: 'r', statement: 'my locker code' }] }),
    write('results/bbbbbbbbbbbbbbbb.json', { status: 'nothing', closest: ['The user\'s locker code is 4411.'] }),
    write('sessions/cccccccccccccccc.json', { version: 1, sessionId: 'test-host:legacy', capture: { pending: [{ role: 'user', content: 'My locker code is 4411.' }] } }),
  ];
  // Recent ones may still be an earlier version's: they stay, and are not sent.
  const recent = [
    write('queue/000001763500001-123-000002-dddddddddddddddd.json', { ...job, request_id: 'legacy:2' }, new Date()),
    write('sessions/eeeeeeeeeeeeeeee.json', { version: 1, sessionId: 'test-host:recent', capture: { pending: [{ role: 'user', content: 'My locker code is 4411.' }] } }, new Date()),
  ];
  const { HookMemory, resolveSettings } = core;
  const settings = resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: dir, QUOTEMEM_CAPTURE_EVERY: '1' });
  const memory = new HookMemory({ settings, log: { warn() {}, warnOnce() {}, debug() {} }, host: 'test-host' });
  memory.flush = () => false;
  await memory.prompt({ session: 'N', turn: 'n1', text: 'I live in Porto.' });
  for (const file of oldFiles) assert.ok(!fs.existsSync(file), `removed: ${path.relative(dir, file)}`);
  assert.ok(!fs.existsSync(path.join(dir, 'results')), 'the old outcomes folder is gone');
  for (const file of recent) assert.ok(fs.existsSync(file), `kept: ${path.relative(dir, file)}`);
  await memory.reply({ session: 'N', turn: 'n1', text: 'Sure.' });
  assert.equal(await core.runFlusher(settings, { warn() {}, warnOnce() {}, debug() {} }), 'empty');
  assert.ok(server.of('/memory/add').length >= 1);
  assert.ok(server.of('/memory/add').every((r) => r.body.messages.every((m) => !m.content.includes('4411'))), 'nothing of the earlier layout is sent');
  assert.ok(fs.existsSync(recent[0]), 'an earlier version\'s recent job is left to it');
});

test('endSync: a host that exits queues the rest at once and keeps the session', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true } }) });
  const hooks = setup(server);
  await hooks.turn('X', 1, 'I live in Lisbon.', 'Lisbon.');
  const { HookMemory, resolveSettings } = core;
  const settings = resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: hooks.stateDir });
  const memory = new HookMemory({ settings, log: { warn() {}, warnOnce() {}, debug() {} }, host: 'test-host' });
  assert.equal(memory.endSync({ session: 'X' }), 1);
  assert.equal(memory.endSync({ session: 'X' }), 0, 'nothing left');
  assert.equal(hooks.queue().entries().length, 1);
  const state = JSON.parse(fs.readFileSync(memory.fileOf('test-host:X'), 'utf8'));
  assert.equal(state.seq, 1, 'the session state stays for a resume');
  memory.flush();
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'sent');
});

test('privacy: secrets redacted on disk and on the wire; logs hold no token or content', async () => {
  const server = await serve({ '/memory/add': () => ({ status: 500, body: { code: 'internal' } }), '/memory/search': () => ({ status: 503, body: { code: 'busy' } }) });
  const hooks = setup(server, { QUOTEMEM_CAPTURE_EVERY: '1', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '10', QUOTEMEM_CAPTURE_MAX_ATTEMPTS: '2', QUOTEMEM_DEBUG: '1' });
  await hooks.turn('P', 1, 'My OpenAI key is sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz012345 and I live near the river.', 'Do not share keys.');
  await hooks.call('prompt', { session: 'P', turn: 'p2', text: 'What is my email again? My password is hunter2xyz' });
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'two attempts');
  const sent = JSON.stringify(server.of('/memory/add')[0].body);
  assert.ok(!sent.includes('sk-proj-AbCd') && sent.includes('[REDACTED:api_key]'));
  assert.ok(!JSON.stringify(server.of('/memory/search')).includes('hunter2xyz'));
  await waitFor(() => !fs.existsSync(hooks.queue().lockFile), 5000, 'flusher exit');
  const disk = treeTexts(hooks.stateDir).map(({ text }) => text).join('\n');
  assert.ok(!disk.includes('sk-proj-AbCd') && !disk.includes('hunter2xyz') && !disk.includes(TOKEN), 'no secret or token on disk');
  const log = fs.readFileSync(path.join(hooks.stateDir, 'quotemem.log'), 'utf8');
  assert.match(log, /dropped: internal after 2 attempts/);
  for (const text of [TOKEN, 'river', 'password', 'Do not share']) assert.ok(!log.includes(text), `log contains ${text}`);
});

test('fitContext: over the limit, the oldest notes go first; a block is never cut', () => {
  const { fitContext, UNAVAILABLE_HEADER } = core;
  const note = (n) => `${FORGET_HEADER}\nForget ${n}: ${'x'.repeat(300)}`;
  const recall = `${RECALL_HEADER}\nnotice\n<quotemem-recall count="1">\n<memory-hit id="a" rank="1">\n${'y'.repeat(600)}\n</memory-hit>\n</quotemem-recall>`;
  const text = [note(1), note(2), note(3), recall].join('\n');
  assert.equal(fitContext(text, 10_000), text, 'fits: unchanged');
  const cut = fitContext(text, recall.length + note(3).length + 50);
  assert.equal(cut, `${note(3)}\n${recall}`, 'the newest note and the whole block');
  assert.ok(cut.endsWith('</quotemem-recall>'));
  assert.equal(fitContext(text, recall.length + 10), recall);
  // Only notes: whole notes, newest first.
  assert.equal(fitContext([note(1), note(2)].join('\n'), note(2).length + 5), note(2));
  // A block alone too long: notes that fit are kept, the block is dropped, never cut.
  assert.equal(fitContext(`${UNAVAILABLE_HEADER}\nshort\n${recall}`, 100), `${UNAVAILABLE_HEADER}\nshort`);
  assert.equal(fitContext(recall, 100), null);
  assert.equal(fitContext('', 100), null);
  assert.equal(fitContext(null, 100), null);
});

test('openTurn: the open turn\'s id, the head of its message and its time; null when none', async () => {
  const server = await serve({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }), '/memory/search': () => ({ body: { data: [] } }) });
  const stateDir = path.join(root, 'open-turn');
  const settings = core.resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_CAPTURE_EVERY: '99', HOME: root, USERPROFILE: root });
  const memory = new core.HookMemory({ settings, log: core.makeLog(settings, () => {}), host: 'test' });
  assert.equal(memory.openTurn('s1'), null);
  const at = Date.now();
  await memory.prompt({ session: 's1', turn: 'p1', text: 'We picked Postgres.', time: at });
  assert.deepEqual(memory.openTurn('s1'), { turn: 'p1', head: core.promptHead('We picked Postgres.'), time: at });
  await memory.reply({ session: 's1', turn: 'p1', text: 'OK.' });
  assert.equal(memory.openTurn('s1'), null);
});

test('recall: a forget still queued holds recall back until it ends, for its own user only', async () => {
  const server = await serve({
    '/memory/add': () => ({ body: { success: true } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000CA1', 'The user\'s cat is called Miso.', { score: 0.9 })] } }),
  });
  const ask = "What is my cat's name?";
  const hooks = setup(server, { TEST_FORGET_WAIT_MS: '1500' });
  const queue = hooks.queue();
  // No background sender runs for this user: the queued forgets stay as written.
  const lock = core.tryLock(queue.lockFile);
  assert.ok(lock);
  const forgetJob = (id) => ({ kind: 'forget', request_id: id, body: { user_id: 'quei', request_id: id }, url: server.url, created: Date.now(), attempts: 0, nextAt: 0 });
  const jobFile = () => queue.entries().at(-1).file;
  const searches = () => server.of('/memory/search').length;
  // Each check asks in a new chat: a chat is not shown the same memory twice.
  const recalled = (out, what) => {
    assert.ok(out.context?.includes(RECALL_HEADER), `${what}: normal recall`);
    assert.match(out.context, /Miso/, what);
  };
  try {
    const plain = await hooks.call('prompt', { session: 'A', turn: 'a1', text: ask });
    recalled(plain, 'no forget queued');
    const saving = plain.context.split('saved automatically when their next message or a new chat starts').length - 1;
    assert.equal(saving, 1, 'the sentence on automatic saving appears once');
    assert.match(plain.context, /never say it was not recorded, and do not ask the user to save it/);

    // Pending past the wait: nothing is injected, and nothing about the forget.
    queue.enqueue([forgetJob('f1')]);
    const before = searches();
    const held = await hooks.call('prompt', { session: 'B', turn: 'b1', text: ask });
    assert.equal(held.context, null, 'no memory and no note while the forget runs');
    assert.equal(held.held, true);
    assert.ok(held.ms >= 1500, `waited for the forget first (${held.ms} ms)`);
    assert.equal(searches(), before, 'no search while the forget runs');

    // Another user of the same state folder neither waits nor loses recall.
    const other = setup(server, { QUOTEMEM_STATE_DIR: hooks.stateDir, QUOTEMEM_USER_ID: 'someone-else', TEST_FORGET_WAIT_MS: '5000' });
    const theirs = await other.call('prompt', { session: 'O', turn: 'o1', text: ask });
    recalled(theirs, 'another user');
    assert.ok(theirs.ms < 4000, `another user did not wait (${theirs.ms} ms)`);

    // Failed: normal recall again.
    queue.writeResult('f1', { status: 'failed', code: 'http_500' });
    recalled(await hooks.call('prompt', { session: 'F', turn: 'f1', text: ask }), 'failed forget');
    fs.unlinkSync(jobFile());

    // Dropped from the queue: normal recall.
    queue.enqueue([forgetJob('f2')]);
    fs.unlinkSync(jobFile());
    recalled(await hooks.call('prompt', { session: 'G', turn: 'g1', text: ask }), 'dropped forget');

    // Queued more than 120 s ago and still waiting (a busy service): not
    // waited for any more, but recall stays held back until it ends.
    queue.enqueue([forgetJob('f3')]);
    const name = path.basename(jobFile());
    const oldFile = path.join(queue.dir, `${String(Date.now() - 121_000).padStart(15, '0')}${name.slice(15)}`);
    fs.renameSync(jobFile(), oldFile);
    const searched = searches();
    const late = await hooks.call('prompt', { session: 'H', turn: 'h1', text: ask });
    assert.equal(late.context, null, 'an older forget still queued: no memory');
    assert.equal(late.held, true);
    assert.ok(late.ms < 1500, `not waited for again (${late.ms} ms)`);
    assert.equal(searches(), searched, 'no search while it waits');
    // Once it ends (here: dropped from the queue), recall is normal again.
    fs.unlinkSync(oldFile);
    recalled(await hooks.call('prompt', { session: 'H2', turn: 'h2', text: ask }), 'old forget gone');

    // A queue file that cannot be read means no pending forget.
    queue.enqueue([forgetJob('f4')]);
    fs.writeFileSync(jobFile(), '{not json');
    recalled(await hooks.call('prompt', { session: 'I', turn: 'i1', text: ask }), 'unreadable job');
    fs.unlinkSync(jobFile());

    // Done during the wait: the prompt recalls as usual once it is.
    const waiting = setup(server, { QUOTEMEM_STATE_DIR: hooks.stateDir });
    queue.enqueue([forgetJob('f5')]);
    const file = jobFile();
    setTimeout(() => {
      queue.writeResult('f5', { status: 'forgotten', forgotten: 1 });
      fs.unlinkSync(file);
    }, 1000);
    const done = await waiting.call('prompt', { session: 'C', turn: 'c1', text: ask });
    recalled(done, 'forget done within the wait');
    assert.ok(done.ms >= 900, `waited for the forget (${done.ms} ms)`);
    assert.doesNotMatch(done.context, /being forgotten/);
  } finally {
    core.releaseLock(lock);
  }
});

test('recall: held back while any forget is queued, one that turns old during the wait or an old one beside a new one', async () => {
  const quiet = { warn() {}, warnOnce() {}, debug() {} };
  const settings = await testSettings('http://127.0.0.1:9', tempDir(), {}, { QUOTEMEM_AUTOSTART: '0' });
  const forgetJob = (id) => ({ kind: 'forget', request_id: id, body: { user_id: 'quei', request_id: id }, url: settings.url, created: Date.now(), attempts: 0, nextAt: 0 });
  // Queue a forget whose file says it was queued `agoMs` ago.
  const queueAt = (memory, id, agoMs) => {
    memory.queue.enqueue([forgetJob(id)]);
    const entry = memory.queue.entries().find((e) => e.name.includes(core.jobKey(memory.queue.owner, id)));
    const moved = path.join(memory.queue.dir, `${String(Date.now() - agoMs).padStart(15, '0')}${entry.name.slice(15)}`);
    fs.renameSync(entry.file, moved);
    return moved;
  };
  const HOLD_MS = 120_000; // FORGET_HOLD_MS
  // A forget queued 119.6 s ago turns 120 s old during the wait: still held.
  const crossing = new core.HookMemory({ settings: { ...settings, stateDir: tempDir() }, log: quiet, host: 'test-host', forgetWaitMs: 3000 });
  crossing.flush = () => false;
  queueAt(crossing, 'f-cross', HOLD_MS - 400);
  const started = Date.now();
  assert.equal(await crossing.forgetHeld(), true, 'held while it is queued');
  assert.ok(Date.now() - started < 2500, 'no longer waited for once it is old');
  // An old forget beside a new one that ends during the wait: still held.
  const both = new core.HookMemory({ settings: { ...settings, stateDir: tempDir() }, log: quiet, host: 'test-host', forgetWaitMs: 3000 });
  both.flush = () => false;
  queueAt(both, 'f-old', HOLD_MS + 1000);
  const fresh = queueAt(both, 'f-new', 0);
  setTimeout(() => fs.unlinkSync(fresh), 300);
  assert.equal(await both.forgetHeld(), true, 'the old forget still holds recall back');
  // None queued: not held.
  const none = new core.HookMemory({ settings: { ...settings, stateDir: tempDir() }, log: quiet, host: 'test-host', forgetWaitMs: 3000 });
  assert.equal(await none.forgetHeld(), false);
});

test('recall: the wait for a queued forget fits in the hook time limit', () => {
  assert.equal(core.FORGET_WAIT_MS, 8000);
  // A 15 s hook (Cursor, Devin, Gemini CLI, goose) answering 1 s early, with the default 3 s recall.
  assert.equal(core.forgetWaitWithin(14_000, { recallTimeoutMs: 3000 }), 8000);
  // A longer recall leaves less: the wait, the recall and the margin stay within the limit.
  assert.equal(core.forgetWaitWithin(14_000, { recallTimeoutMs: 10_000 }), 2000);
  assert.equal(core.forgetWaitWithin(4000, { recallTimeoutMs: 3000 }), 0);
});
