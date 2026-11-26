#!/usr/bin/env node
// The host adapters end to end. Each hook runs as the host would run it:
// the command from the shipped config file (hooks.json, a Cline hook file, a
// goose plugin script), started through `sh -c` in its own process, with the
// host's own payload on stdin, under a temporary HOME, against a stub
// QuoteMem. The OpenClaw plugin is loaded in process with a fake plugin API.
//
//   node --test integrations/hooks/test/hosts.test.mjs
import assert from 'node:assert/strict';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { HIT, TOKEN, plan, sleep, stubServer, tempDir, waitFor } from '../../core/test/helpers.mjs';
import { DiskQueue, forgottenOf, ownerKey, releaseLock, sha16, tryLock } from '../../core/index.mjs';
import { hostEnv, nativeInstalled, parseEnvFile } from '../env.mjs';
import { fallback, hostTime, normalize, render } from '../hosts.mjs';

const HOOKS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const INTEGRATIONS = path.dirname(HOOKS);
const RECALL_HEADER = '[QuoteMem: retrieved long-term memory]';
const FORGET_HEADER = '[QuoteMem: automatic forget]';

const root = tempDir('qm-hosts-');
// `npm install -g quotemem` puts quotemem-hook on PATH; this does the same.
const BIN = path.join(root, 'bin');
fs.mkdirSync(BIN);
fs.symlinkSync(path.join(HOOKS, 'quotemem-hook.mjs'), path.join(BIN, 'quotemem-hook'));

const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });

const verified = (n = 1) => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: Array.from({ length: n }, (_, i) => ({ memory_id: `01JX0000000000000000000F${i}`, outcome: 'verified' })), scope_state: 'ready' } });

async function memoryServer(extra = {}) {
  const server = await stubServer({
    '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }),
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User prefers tabs over spaces', { score: 0.9 })] } }),
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => verified(1),
    ...extra,
  });
  servers.add(server);
  return server;
}

let homeSeq = 0;
function newHome() {
  homeSeq += 1;
  const home = path.join(root, `home-${homeSeq}`);
  fs.mkdirSync(home, { recursive: true, mode: 0o700 });
  return home;
}

function env(server, home, extra = {}) {
  return {
    PATH: `${BIN}${path.delimiter}${process.env.PATH}`,
    HOME: home, USERPROFILE: home,
    QUOTEMEM_URL: server.url,
    QUOTEMEM_TOKEN: TOKEN,
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_CAPTURE_EVERY: '2',
    QUOTEMEM_CAPTURE_RETRY_BASE_MS: '50',
    QUOTEMEM_FLUSHER_MAX_WAIT_MS: '2000',
    // No flusher waits for idle chats beyond a test, unless the test sets a time.
    QUOTEMEM_IDLE_FLUSH_MS: '0',
    ...extra,
  };
}

/** Run a configured hook command the way hosts do: `sh -c`, payload on stdin. */
function sh(command, payload, childEnv, cwd = root, { stdinDelayMs = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn('/bin/sh', ['-c', command], { env: childEnv, cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    // firstOutMs: when the hook's answer came, before it exits.
    let firstOutMs = null;
    child.stdout.on('data', (d) => { firstOutMs ??= Date.now() - started; stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('error', reject);
    child.on('exit', (code) => resolve({ code, stdout, stderr, ms: Date.now() - started, firstOutMs }));
    // stdinDelayMs: the hook starts now and reads its payload later, as a
    // hook Cursor started at the same time as another but that runs slower.
    const send = () => child.stdin.end(payload === undefined ? '' : JSON.stringify(payload));
    if (stdinDelayMs > 0) setTimeout(send, stdinDelayMs); else send();
  });
}

/** The hook printed exactly one JSON value (or nothing when `empty`), exit 0, nothing on stderr. */
function output(result, { empty = false } = {}) {
  assert.equal(result.code, 0, `exit ${result.code}: ${result.stderr}`);
  assert.equal(result.stderr, '', 'nothing on stderr');
  if (empty) {
    assert.equal(result.stdout, '', 'stdout stays empty');
    return null;
  }
  const lines = result.stdout.split('\n').filter(Boolean);
  assert.equal(lines.length, 1, `one line of JSON, got ${JSON.stringify(result.stdout)}`);
  return JSON.parse(lines[0]);
}

function walk(dir) {
  const out = [];
  let entries = [];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (error) {
    if (error.code === 'ENOENT') return out; // a folder a background flusher removed meanwhile
    throw error;
  }
  for (const entry of entries) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(file));
    else if (entry.isFile()) out.push(file);
  }
  return out;
}

/** No file under HOME holds the token (except the user's own env file), and logs hold no message text. */
function assertNoLeaks(home, words = []) {
  for (const file of walk(home)) {
    if (file === path.join(home, '.quotemem', 'env')) continue;
    let text;
    try {
      text = fs.readFileSync(file, 'utf8');
    } catch (error) {
      if (error.code === 'ENOENT') continue; // a lock or job file a background flusher removed meanwhile
      throw error;
    }
    assert.ok(!text.includes(TOKEN), `${path.relative(home, file)} holds the token`);
    if (/quotemem\.log/.test(file)) for (const word of words) assert.ok(!text.includes(word), `${path.relative(home, file)} holds message text "${word}"`);
  }
}

const config = (rel) => JSON.parse(fs.readFileSync(path.join(INTEGRATIONS, rel), 'utf8'));
const commandOf = (hooks, event) => hooks[event][0].hooks?.[0]?.command ?? hooks[event][0].command;

function copyDir(from, to) {
  fs.cpSync(from, to, { recursive: true });
}

// ---------------------------------------------------------------- Gemini CLI

test('Gemini CLI extension: capture, recall, forget and session end through the installed extension', async () => {
  const server = await memoryServer();
  const home = newHome();
  // `gemini extensions install <path>` copies the folder.
  const ext = path.join(home, '.gemini', 'extensions', 'quotemem');
  copyDir(path.join(INTEGRATIONS, 'gemini-cli'), ext);
  const hooks = JSON.parse(fs.readFileSync(path.join(ext, 'hooks', 'hooks.json'), 'utf8')).hooks;
  const cmd = (event) => commandOf(hooks, event).replaceAll('${extensionPath}', ext).replaceAll('${/}', path.sep);
  // Gemini passes an extension only safe variables plus declared settings,
  // and its redaction strips QUOTEMEM_TOKEN; the token comes from ~/.quotemem/env.
  fs.mkdirSync(path.join(home, '.quotemem'), { mode: 0o700 });
  fs.writeFileSync(path.join(home, '.quotemem', 'env'), `# QuoteMem\nexport QUOTEMEM_TOKEN="${TOKEN}"\nQUOTEMEM_CAPTURE_EVERY=2\nQUOTEMEM_CAPTURE_RETRY_BASE_MS=50\nIGNORED=1\n`, { mode: 0o600 });
  const childEnv = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_IDLE_FLUSH_MS: '0' };
  const base = { session_id: 'g-sess-1', transcript_path: path.join(home, 't.json'), cwd: home, timestamp: '2026-11-20T10:00:00Z' };
  const turn = async (prompt, reply) => {
    const before = output(await sh(cmd('BeforeAgent'), { ...base, hook_event_name: 'BeforeAgent', prompt }, childEnv));
    const afterOut = output(await sh(cmd('AfterAgent'), { ...base, hook_event_name: 'AfterAgent', prompt, prompt_response: reply, stop_hook_active: false }, childEnv));
    assert.deepEqual(afterOut, {});
    return before;
  };

  assert.deepEqual(await turn('I prefer tabs over spaces.', 'Noted: tabs.'), {}, 'a plain statement adds nothing');
  assert.equal(server.of('/memory/search').length, 0, 'and does not search');
  await turn('My dog is called Rex.', 'Nice name.');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  const first = server.of('/memory/add')[0];
  assert.equal(first.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(first.body.user_id, 'quei');
  assert.equal(first.body.session_id, 'gemini-cli:g-sess-1');
  assert.deepEqual(first.body.messages.map((m) => [m.role, m.content]), [
    ['user', 'I prefer tabs over spaces.'], ['assistant', 'Noted: tabs.'], ['user', 'My dog is called Rex.'], ['assistant', 'Nice name.'],
  ]);
  assert.ok(first.body.messages.every((m) => m.timestamp === Date.parse('2026-11-20T10:00:00Z')), 'each message carries the time Gemini gave');

  const recall = await turn('What do I prefer for indentation?', 'Tabs.');
  assert.equal(recall.suppressOutput, true);
  assert.equal(recall.hookSpecificOutput.hookEventName, 'BeforeAgent');
  assert.match(recall.hookSpecificOutput.additionalContext, /^\[QuoteMem: retrieved long-term memory\]/);
  assert.match(recall.hookSpecificOutput.additionalContext, /User prefers tabs over spaces/);
  assert.equal(server.of('/memory/search')[0].body.query, 'What do I prefer for indentation?');

  // A retry of AfterAgent (stop_hook_active) adds nothing.
  output(await sh(cmd('AfterAgent'), { ...base, hook_event_name: 'AfterAgent', prompt: 'x', prompt_response: 'again', stop_hook_active: true }, childEnv));

  const forget = await sh(cmd('BeforeAgent'), { ...base, hook_event_name: 'BeforeAgent', prompt: 'Please forget that my dog is called Rex.' }, childEnv);
  const note = output(forget).hookSpecificOutput.additionalContext;
  assert.ok(note.startsWith(FORGET_HEADER), note);
  assert.match(note, /1 stored memory was forgotten/);
  assert.doesNotMatch(note, /quotemem_|01JX/, 'the note names no tool and no id');
  const prepare = server.of('/memory/forget/prepare')[0];
  assert.equal(prepare.body.statement, 'my dog is called Rex');
  assert.match(prepare.body.request_id, /^gemini-cli:g-sess-1:forget:[0-9a-f]{16}$/);
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  // The turn said before the request was stored before the forget ran.
  const order = server.requests.map((r) => r.route);
  assert.ok(order.lastIndexOf('/memory/add') < order.indexOf('/memory/forget/prepare'), order.join(' '));
  output(await sh(cmd('AfterAgent'), { ...base, hook_event_name: 'AfterAgent', prompt: 'Please forget that my dog is called Rex.', prompt_response: 'Done.', stop_hook_active: false }, childEnv));

  output(await sh(cmd('SessionEnd'), { ...base, hook_event_name: 'SessionEnd', reason: 'exit' }, childEnv));
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'the session-end batch');
  const last = server.of('/memory/add')[2].body.messages.map((m) => m.content);
  assert.deepEqual(last.slice(-2), ['Please forget that my dog is called Rex.', 'Done.']);
  assert.ok(nativeInstalled({ HOME: home, USERPROFILE: home }, 'gemini-cli'), 'the native marker is written');
  assertNoLeaks(home, ['Rex', 'tabs']);
});

test('Gemini CLI: the token can also come from GEMINI_CLI_QUOTEMEM_TOKEN or QUOTEMEM_TOKEN_FILE', async () => {
  const server = await memoryServer();
  const home = newHome();
  const cmd = `node "${path.join(INTEGRATIONS, 'gemini-cli', 'quotemem-hook.mjs')}" gemini BeforeAgent`;
  const payload = { session_id: 's', hook_event_name: 'BeforeAgent', prompt: 'Do you remember my dog?' };
  const a = output(await sh(cmd, payload, { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, GEMINI_CLI_QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_IDLE_FLUSH_MS: '0' }));
  assert.match(a.hookSpecificOutput.additionalContext, /retrieved long-term memory/);
  fs.writeFileSync(path.join(home, 'qm-token'), `${TOKEN}\n`, { mode: 0o600 });
  const b = output(await sh(cmd, { ...payload, session_id: 's2' }, { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN_FILE: '~/qm-token', QUOTEMEM_IDLE_FLUSH_MS: '0' }));
  assert.match(b.hookSpecificOutput.additionalContext, /retrieved long-term memory/);
  assert.equal(server.of('/memory/search').every((r) => r.headers.authorization === `Bearer ${TOKEN}`), true);
});

// -------------------------------------------------------------------- Cursor

/** The shipped Cursor hooks in a temp ~/.cursor, run as Cursor runs them. */
function cursorHooks(server, home, { version, extraEnv = {} } = {}) {
  const cursorDir = path.join(home, '.cursor');
  fs.mkdirSync(cursorDir, { recursive: true });
  fs.copyFileSync(path.join(INTEGRATIONS, 'cursor', 'hooks.json'), path.join(cursorDir, 'hooks.json'));
  const hooks = JSON.parse(fs.readFileSync(path.join(cursorDir, 'hooks.json'), 'utf8'));
  assert.equal(hooks.version, 1);
  return (event, fields, options) => sh(commandOf(hooks.hooks, event), {
    conversation_id: 'c-1', model: 'some-model', hook_event_name: event, cursor_version: version,
    workspace_roots: ['/work/project'], user_email: 'user@example.com', transcript_path: null, ...fields,
  }, env(server, home, extraEnv), cursorDir, options); // user hooks run from ~/.cursor
}

// The reply stored with a user message, once its batch is sent (null: none).
async function storedReply(server, user) {
  const batchOf = () => server.of('/memory/add').map((r) => r.body.messages).find((m) => m.some((x) => x.role === 'user' && x.content === user));
  await waitFor(() => batchOf(), 8000, `the batch of "${user}"`);
  return batchOf().find((m) => m.role === 'assistant')?.content ?? null;
}

test('Cursor before 3.20.17: session digest, capture of the last reply per turn, silent forget, background agents skipped', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.19.0' });

  const start = output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false, composer_mode: 'agent' }));
  assert.match(start.additional_context, /^\[QuoteMem: retrieved long-term memory\]/);
  assert.equal(server.of('/memory/search').length, 1, 'one digest search');

  assert.deepEqual(output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'I prefer tabs over spaces.', attachments: [] })), { continue: true });
  assert.deepEqual(output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Let me check…' })), {});
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Noted: tabs.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'My dog is called Rex.', attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g2', text: 'Nice name.' }));
  output(await run('stop', { generation_id: 'g2', status: 'completed', loop_count: 0 }));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  const batch = server.of('/memory/add')[0].body;
  assert.equal(batch.session_id, 'cursor:c-1');
  assert.deepEqual(batch.messages.map((m) => m.content), ['I prefer tabs over spaces.', 'Noted: tabs.', 'My dog is called Rex.', 'Nice name.']);
  assert.ok(!JSON.stringify(server.requests.map((r) => r.body)).includes('user@example.com'), 'the email is never sent');

  // A memory question cannot get context in this Cursor's prompt hook: no search there.
  assert.deepEqual(output(await run('beforeSubmitPrompt', { generation_id: 'g3', prompt: 'What do I prefer for indentation?', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/search').length, 1);
  output(await run('stop', { generation_id: 'g3', status: 'aborted', loop_count: 0 }));

  // Forget runs silently before the prompt is submitted.
  assert.deepEqual(output(await run('beforeSubmitPrompt', { generation_id: 'g4', prompt: 'Please forget that my dog is called Rex.', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/forget/confirm').length, 1);

  // A background agent is not remembered.
  output(await run('beforeSubmitPrompt', { conversation_id: 'bg-1', generation_id: 'b1', prompt: 'I live in Lisbon.', is_background_agent: true, attachments: [] }));

  output(await run('sessionEnd', { session_id: 'c-1', reason: 'user_close', duration_ms: 1000, is_background_agent: false, final_status: 'completed' }));
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'forget-turn batch and session-end batch');
  assert.ok(!server.of('/memory/add').some((r) => JSON.stringify(r.body).includes('Lisbon')), 'the background agent turn was skipped');
  assertNoLeaks(home, ['Rex', 'tabs', 'Lisbon']);
});

test('Cursor 3.20.17 and later: per-turn recall and the forget note in additional_context; a late session digest on a later prompt', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  // sessionStart gives the digest, which Cursor keeps for the whole conversation.
  const start = output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false, composer_mode: 'agent' }));
  assert.match(start.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  assert.equal(server.of('/memory/search').length, 1, 'one digest search');

  // It finished before the first prompt: the prompt adds nothing and searches nothing.
  assert.deepEqual(output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'I prefer tabs over spaces.', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/search').length, 1);
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Noted: tabs.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));

  // Later plain statements add nothing and search nothing.
  assert.deepEqual(output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'My dog is called Rex.', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/search').length, 1);
  output(await run('afterAgentResponse', { generation_id: 'g2', text: 'Nice name.' }));
  output(await run('stop', { generation_id: 'g2', status: 'completed', loop_count: 0 }));

  // A memory question gets the relevant memories.
  const recall = output(await run('beforeSubmitPrompt', { generation_id: 'g3', prompt: 'What do I prefer for indentation?', attachments: [] }));
  assert.deepEqual(Object.keys(recall).sort(), ['additional_context', 'continue']);
  assert.equal(recall.continue, true);
  assert.match(recall.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*User prefers tabs over spaces[\s\S]*<\/quotemem-recall>$/);
  assert.equal(server.of('/memory/search').at(-1).body.query, 'What do I prefer for indentation?');
  output(await run('afterAgentResponse', { generation_id: 'g3', text: 'Tabs.' }));
  output(await run('stop', { generation_id: 'g3', status: 'completed', loop_count: 0 }));

  // A forget runs before the prompt is submitted, and the model gets one note.
  const forget = output(await run('beforeSubmitPrompt', { generation_id: 'g4', prompt: 'Please forget that my dog is called Rex.', attachments: [] }));
  assert.equal(forget.continue, true);
  assert.ok(forget.additional_context.startsWith(FORGET_HEADER), forget.additional_context);
  assert.match(forget.additional_context, /1 stored memory was forgotten/);
  assert.doesNotMatch(forget.additional_context, /quotemem_|01JX/);
  assert.equal(server.of('/memory/forget/prepare')[0].body.statement, 'my dog is called Rex');
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  output(await run('afterAgentResponse', { generation_id: 'g4', text: 'Done, I forgot it.' }));
  output(await run('stop', { generation_id: 'g4', status: 'completed', loop_count: 0 }));

  output(await run('sessionEnd', { session_id: 'c-1', reason: 'user_close', duration_ms: 1000, is_background_agent: false, final_status: 'completed' }));
  await waitFor(() => server.of('/memory/add').length >= 3, 8000, 'every batch');
  await sleep(300);
  const sent = server.of('/memory/add').flatMap((r) => r.body.messages.map((m) => [m.role, m.content]));
  assert.deepEqual(sent, [
    ['user', 'I prefer tabs over spaces.'], ['assistant', 'Noted: tabs.'], ['user', 'My dog is called Rex.'], ['assistant', 'Nice name.'],
    ['user', 'What do I prefer for indentation?'], ['assistant', 'Tabs.'],
    ['user', 'Please forget that my dog is called Rex.'], ['assistant', 'Done, I forgot it.'],
  ]);
  assert.ok(!JSON.stringify(server.of('/memory/add').map((r) => r.body)).includes('[QuoteMem'), 'no recall block or note is captured back');

  // A first prompt sent before sessionStart finished: the digest comes with
  // the next prompt, once. The prompt itself never searches for it.
  const searches = server.of('/memory/search').length;
  assert.deepEqual(output(await run('beforeSubmitPrompt', { conversation_id: 'c-2', generation_id: 'h1', prompt: 'Hello there.', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/search').length, searches, 'the prompt does not search for the digest');
  assert.match(output(await run('sessionStart', { conversation_id: 'c-2', session_id: 'c-2', is_background_agent: false })).additional_context, /^\[QuoteMem: retrieved long-term memory\]/);
  const late = output(await run('beforeSubmitPrompt', { conversation_id: 'c-2', generation_id: 'h2', prompt: 'Let us start on the parser.', attachments: [] }));
  assert.match(late.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  assert.deepEqual(output(await run('beforeSubmitPrompt', { conversation_id: 'c-2', generation_id: 'h3', prompt: 'Now the lexer.', attachments: [] })), { continue: true }, 'only once');

  // A late digest waits while a turn recalls, then comes with the next turn.
  output(await run('beforeSubmitPrompt', { conversation_id: 'c-3', generation_id: 'k1', prompt: 'Hi.', attachments: [] }));
  output(await run('sessionStart', { conversation_id: 'c-3', session_id: 'c-3', is_background_agent: false }));
  const asked = output(await run('beforeSubmitPrompt', { conversation_id: 'c-3', generation_id: 'k2', prompt: 'What do I prefer for indentation?', attachments: [] }));
  assert.equal(asked.additional_context.split('[QuoteMem: retrieved long-term memory]').length, 2, 'the recall block only');
  assert.match(output(await run('beforeSubmitPrompt', { conversation_id: 'c-3', generation_id: 'k3', prompt: 'Thanks.', attachments: [] })).additional_context, /^\[QuoteMem: retrieved long-term memory\]/);
  // The conversation's end removes its digest files.
  const digests = path.join(home, '.quotemem', 'cursor', 'digests');
  const kept = fs.readdirSync(digests).length;
  output(await run('sessionEnd', { conversation_id: 'c-3', session_id: 'c-3', reason: 'user_close', is_background_agent: false }));
  assert.equal(fs.readdirSync(digests).length, kept - 2);
  assertNoLeaks(home, ['Rex', 'tabs']);
});

test('Cursor 3.20.17 and later: a slow service never delays a prompt that is not about memory; recall waits at most 10 s', async () => {
  const server = await memoryServer({ '/memory/search': () => ({ delay: 12_000, body: { data: [HIT('01JX0000000000000000000001', 'User prefers tabs over spaces', { score: 0.9 })] } }) });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9', extraEnv: { QUOTEMEM_RECALL_TIMEOUT_MS: '15000' } });
  // The first prompt of a conversation, before sessionStart: no search, no wait.
  const first = await run('beforeSubmitPrompt', { conversation_id: 's-1', generation_id: 'g1', prompt: 'Let us refactor the parser.', attachments: [] });
  assert.deepEqual(output(first), { continue: true });
  assert.ok(first.ms < 2000, `the first prompt took ${first.ms} ms`);
  assert.equal(server.of('/memory/search').length, 0);
  // A memory question waits for recall, but not past Cursor's 15 s hook
  // timeout; the model is told memory could not be checked.
  const asked = await run('beforeSubmitPrompt', { conversation_id: 's-1', generation_id: 'g2', prompt: 'What do I prefer for indentation?', attachments: [] });
  assert.match(output(asked).additional_context, /^\[QuoteMem: memory unavailable\]/);
  assert.ok(asked.ms >= 9000 && asked.ms < 13_500, `the recall prompt took ${asked.ms} ms`);
});

test('Cursor: the version gate and QUOTEMEM_CURSOR_PROMPT_CONTEXT', async () => {
  const server = await memoryServer();
  const question = { generation_id: 'q1', prompt: 'What do I prefer for indentation?', attachments: [] };
  const cases = [
    // [version, setting, prompt hook adds context]
    ['3.20.17', undefined, true],
    ['3.20.16', undefined, false],
    ['2026.11.17', undefined, false], // the CLI's date versions
    [undefined, undefined, false],
    ['3.24.9', 'off', false],
    ['3.19.0', 'on', true],
  ];
  for (const [version, setting, adds] of cases) {
    const home = newHome();
    const run = cursorHooks(server, home, { version, extraEnv: setting ? { QUOTEMEM_CURSOR_PROMPT_CONTEXT: setting } : {} });
    const what = `${version} ${setting ?? 'auto'}`;
    const start = output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false }));
    const answer = output(await run('beforeSubmitPrompt', question));
    assert.match(start.additional_context, /^\[QuoteMem: retrieved long-term memory\]/, `${what}: the session digest on every version`);
    if (adds) {
      assert.match(answer.additional_context, /tabs over spaces/, what);
      assert.equal(answer.continue, true);
    } else {
      assert.deepEqual(answer, { continue: true }, what);
    }
  }
});

test('Cursor: what the prompt hook adds stays under 9,500 characters', async () => {
  const long = (i) => HIT(`01JX00000000000000000000${String(i).padStart(2, '0')}`, `User fact ${i}: ${'detail '.repeat(100)}`, { score: 0.9 });
  const server = await memoryServer({ '/memory/search': () => ({ body: { data: Array.from({ length: 20 }, (_, i) => long(i)) } }) });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9', extraEnv: { QUOTEMEM_MAX_INJECT_CHARS: '50000', QUOTEMEM_TOP_K: '20' } });
  // The first prompt asks to forget: the forget note and the digest together.
  const first = output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'Please forget that my dog is called Rex.', attachments: [] }));
  assert.ok(first.additional_context.startsWith(FORGET_HEADER));
  assert.ok(first.additional_context.length <= 9500, `${first.additional_context.length} characters`);
  const recall = output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'What facts do you remember about me?', attachments: [] }));
  assert.match(recall.additional_context, /^\[QuoteMem: retrieved long-term memory\]/);
  assert.ok(recall.additional_context.length <= 9500, `${recall.additional_context.length} characters`);
  assert.ok(recall.additional_context.endsWith('</quotemem-recall>'), 'the recall block is whole');
  assert.ok(20 * long(0).content.length > 2 * 9500, 'without the caps the hits would fill far more');
});

// --------------------------------------------------------------------- Cline

// What Cursor CLI 2026.11.19 sends in print mode (-p): sessionStart for a
// new chat, sessionEnd after every run, no prompt or stop hook. The chat's
// transcript holds every turn so far; a --resume run adds to it.
const CLI_VERSION = '2026.11.19-f005e9d';
const CHAT = 'd8ed2e97-45f8-4f30-8713-35473298976c';
const transcriptLine = (role, text) => `${JSON.stringify({ role, message: { content: [{ type: 'text', text }] } })}\n`;
// A turn as Cursor writes it. Cursor CLI 2026.11.19 rewrites the transcript
// each run with a single turn_ended line at the very end (appendRun).
const TURN_ENDED = `${JSON.stringify({ type: 'turn_ended', status: 'success' })}\n`;
const transcriptTurn = (stamp, user, reply) => transcriptLine('user', `<timestamp>${stamp}</timestamp>\n<user_query>\n${user}\n</user_query>`)
  + (reply === null ? '' : transcriptLine('assistant', reply));
/** A run that ends: the file again, its earlier turn_ended line gone, the run's lines and one turn_ended at the end. */
function appendRun(file, text) {
  const before = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  fs.writeFileSync(file, before.split('\n').filter((line) => line !== TURN_ENDED.trim()).join('\n') + text + TURN_ENDED);
}
function cliTranscript(home, chat = CHAT) {
  const file = path.join(home, '.cursor', 'projects', 'C-Users-me-proj', 'agent-transcripts', chat, `${chat}.jsonl`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, '');
  return file;
}
const cliFields = (chat = CHAT) => ({ conversation_id: chat, generation_id: chat, session_id: chat, model: 'unknown', is_background_agent: false });
const cliEnd = (file, chat = CHAT) => ({ ...cliFields(chat), reason: 'completed', duration_ms: 26632, final_status: 'completed', transcript_path: file });

test('Cursor CLI: only sessionStart and sessionEnd run; each run\'s new turns are saved from the transcript, once, with their forgets', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const start = output(await run('sessionStart', cliFields()));
  assert.match(start.additional_context, /^\[QuoteMem: retrieved long-term memory\]/, 'the session-start digest');
  const file = cliTranscript(home);

  // Run 1: one turn.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Note for later: my kite is orange and is called KITE-7DBF52. Reply in one short sentence.', 'Got it — your orange kite is called KITE-7DBF52.'));
  assert.deepEqual(output(await run('sessionEnd', cliEnd(file))), {});
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first run\'s turn');
  const first = server.of('/memory/add')[0].body;
  assert.equal(first.session_id, `cursor:${CHAT}`);
  assert.match(first.request_id, new RegExp(`^cursor:${CHAT}:x1-1:[0-9a-f]{16}$`));
  assert.deepEqual(first.messages.map((m) => [m.role, m.content]), [
    ['user', 'Note for later: my kite is orange and is called KITE-7DBF52. Reply in one short sentence.'],
    ['assistant', 'Got it — your orange kite is called KITE-7DBF52.'],
  ], 'the wrappers are stripped');
  assert.equal(first.messages[0].timestamp, Date.UTC(2025, 10, 20, 3, 4), 'the transcript\'s local time');

  // Run 2 (--resume): the transcript holds both turns; only the new one is sent.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:05 AM (UTC+8)', 'My locker code word is HERON.', 'Noted: HERON.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the second run\'s turn');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['My locker code word is HERON.', 'Noted: HERON.']);
  assert.match(server.of('/memory/add')[1].body.request_id, /:x2-2:/);
  // A run that added no turn sends nothing.
  output(await run('sessionEnd', cliEnd(file)));
  await sleep(700);
  assert.equal(server.of('/memory/add').length, 2);

  // The plugin's state is lost: the turns are sent again under the same ids,
  // which the service stores once.
  const ids = server.of('/memory/add').map((r) => r.body.request_id);
  fs.rmSync(path.join(home, '.quotemem', 'cursor', 'sessions'), { recursive: true, force: true });
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 4, 8000, 'the resend');
  assert.deepEqual(server.of('/memory/add').slice(2).map((r) => r.body.request_id), ids);

  // Run 3: a forget request in the chat is carried out, before its own turn is stored.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:06 AM (UTC+8)', 'Please forget that my locker code word is HERON.', 'Done, I will not keep that.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/confirm').length === 1 && server.of('/memory/add').length === 5, 8000, 'the forget and its turn');
  const prepare = server.of('/memory/forget/prepare')[0];
  assert.equal(prepare.body.statement, 'my locker code word is HERON');
  assert.match(prepare.body.request_id, new RegExp(`^cursor:${CHAT}:forget:[0-9a-f]{16}$`));
  assert.ok(prepare.at <= server.of('/memory/add')[4].at, 'the forget runs before its own turn is stored');
  // The same transcript again: the forget is not run twice.
  output(await run('sessionEnd', cliEnd(file)));
  await sleep(700);
  assert.equal(server.of('/memory/forget/prepare').length, 1);
  assert.equal(server.of('/memory/add').length, 5);

  // Another chat's transcript file is never read for a chat, even one with nothing saved yet.
  const other = cliTranscript(home, 'another-chat');
  appendRun(other, transcriptTurn('Thursday, Nov 20, 2025, 11:07 AM (UTC+8)', 'My boat is called WAVE.', 'Nice.'));
  output(await run('sessionEnd', cliEnd(other, 'chat-without-turns')));
  await sleep(700);
  assert.equal(server.of('/memory/add').length, 5);
  assertNoLeaks(home, ['KITE-7DBF52', 'HERON', 'WAVE']);
});

test('Cursor app with every hook: a turn the hooks saved is never saved again from the transcript; an unclosed turn takes its reply from it', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  const file = cliTranscript(home, 'c-1');
  output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false }));
  output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'I prefer tabs over spaces.', attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Noted: tabs.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  // A turn whose stop hook never came.
  output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'My dog is called Rex.', attachments: [] }));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', 'I prefer tabs over spaces.', 'Noted: tabs.'));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:01 AM (UTC)', 'My dog is called Rex.', 'Nice name.'));
  // A turn no hook saw.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:02 AM (UTC)', 'My cat is called Pixel.', 'Lovely.'));
  output(await run('sessionEnd', { session_id: 'c-1', reason: 'user_close', duration_ms: 1000, is_background_agent: false, final_status: 'completed', transcript_path: file }));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the hooks\' batch and the unseen turn');
  await sleep(500);
  const adds = server.of('/memory/add').map((r) => r.body);
  assert.equal(adds.length, 2);
  assert.deepEqual(adds[0].messages.map((m) => m.content), ['I prefer tabs over spaces.', 'Noted: tabs.', 'My dog is called Rex.', 'Nice name.'], 'the hooks\' turns, the open one with its reply from the transcript');
  assert.match(adds[0].request_id, /:t1-2:/);
  assert.deepEqual(adds[1].messages.map((m) => m.content), ['My cat is called Pixel.', 'Lovely.']);
  assert.match(adds[1].request_id, /:x3-3:/);
  const words = JSON.stringify(adds);
  for (const word of ['tabs over spaces', 'Rex.', 'Pixel']) assert.equal(words.split(word).length - 1, 1, `${word} is saved once`);

  // Prompt hooks that ran, and a transcript whose messages match none of
  // them: nothing is saved from it.
  const home2 = newHome();
  const run2 = cursorHooks(server, home2, { version: '3.24.9' });
  const file2 = cliTranscript(home2, 'c-1');
  output(await run2('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'I live in Lisbon.', attachments: [] }));
  output(await run2('afterAgentResponse', { generation_id: 'g1', text: 'Noted.' }));
  output(await run2('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  appendRun(file2, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', 'I live in Lisbon, in Portugal, by the sea.', 'Noted.'));
  output(await run2('sessionEnd', { session_id: 'c-1', reason: 'user_close', duration_ms: 1000, is_background_agent: false, final_status: 'completed', transcript_path: file2 }));
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'the hooks\' turn');
  await sleep(500);
  assert.equal(server.of('/memory/add').length, 3);
  assert.equal(JSON.stringify(server.of('/memory/add')[2].body).split('Lisbon').length - 1, 1);
  assert.match(fs.readFileSync(path.join(home2, '.quotemem', 'cursor', 'quotemem.log'), 'utf8'), /does not match the prompts the hooks saw/);
});

// A Cursor <timestamp> text for a time, to the minute, as UTC.
function stampAt(ms) {
  const d = new Date(ms);
  const hour = d.getUTCHours();
  const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getUTCMonth()];
  return `Someday, ${month} ${d.getUTCDate()}, ${d.getUTCFullYear()}, ${hour % 12 || 12}:${String(d.getUTCMinutes()).padStart(2, '0')} ${hour < 12 ? 'AM' : 'PM'} (UTC)`;
}
/** Every forget outcome kept under HOME holds counts only, never a memory's text. */
/** Whether a forget job of the Cursor hooks is still queued under HOME. */
function hostQueueBusy(home) {
  const dir = path.join(home, '.quotemem', 'cursor', 'queue');
  try {
    return fs.readdirSync(dir).some((name) => {
      try {
        return name.endsWith('.json') && fs.readFileSync(path.join(dir, name), 'utf8').includes('"kind":"forget"');
      } catch {
        return false; // sent and removed meanwhile
      }
    });
  } catch {
    return false;
  }
}

async function assertCountsOnly(home) {
  // `key`: a hash of the request's words (requestKey), no word of them;
  // `affectedKeys`: the other memories it affected, by a hash of their ids.
  const allowed = new Set(['status', 'forgotten', 'sentences', 'earlier', 'earlierSentences', 'changed', 'code', 'at', 'created', 'read', 'key',
    'whole', 'dependents', 'unreadable', 'reworded', 'affectedKeys']);
  const outcomes = () => walk(home).filter((file) => /[\\/]results[\\/][0-9a-f]{16}\.json$/.test(file));
  // The flusher writes an outcome once the forget has finished.
  await waitFor(() => outcomes().length > 0 && !hostQueueBusy(home), 8000, 'the forget outcomes');
  const files = outcomes();
  for (const file of files) {
    const result = JSON.parse(fs.readFileSync(file, 'utf8'));
    const extra = Object.keys(result).filter((key) => !allowed.has(key));
    assert.deepEqual(extra, [], `${path.relative(home, file)} keeps ${extra.join(', ')}`);
    if ('key' in result) assert.match(result.key, /^[0-9a-f]{16}$/);
    for (const [key] of result.affectedKeys ?? []) assert.match(key, /^[0-9a-f]{16}$/);
  }
}
const appEnd = (file, chat = 'c-1') => ({ conversation_id: chat, session_id: chat, reason: 'user_close', duration_ms: 1000, is_background_agent: false, final_status: 'completed', transcript_path: file });
const prepares = (server) => server.of('/memory/forget/prepare').map((r) => r.body.statement);

test('Cursor: a turn saved late from the transcript gets again every forget asked after it was said; a later word is kept', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  const file = cliTranscript(home, 'c-1');
  // Turn 1 got no prompt hook (a lock timeout, a crash); turn 2's hooks run its forget.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', 'My locker code word is HERON.', 'Noted.'));
  output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'Please forget that my locker code word is HERON.', attachments: [] }));
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'the hooks\' forget');
  output(await run('afterAgentResponse', { generation_id: 'g2', text: 'Done.' }));
  output(await run('stop', { generation_id: 'g2', status: 'completed', loop_count: 0 }));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:01 AM (UTC)', 'Please forget that my locker code word is HERON.', 'Done.'));
  output(await run('sessionEnd', appEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'the forget again, after the late turn');
  const late = server.of('/memory/add').find((r) => /:x1-1:/.test(r.body.request_id));
  assert.deepEqual(late.body.messages.map((m) => m.content), ['My locker code word is HERON.', 'Noted.']);
  const again = server.of('/memory/forget/prepare')[1];
  assert.equal(again.body.statement, 'my locker code word is HERON');
  assert.ok(again.at >= late.at, 'the forget runs again after the late turn is saved');
  // A turn said after the forget is kept: no forget follows it.
  appendRun(file, transcriptTurn(stampAt(Date.now() + 120_000), 'My new locker code word is EGRET.', 'Noted.'));
  output(await run('sessionEnd', appEnd(file)));
  await waitFor(() => server.of('/memory/add').some((r) => /EGRET/.test(JSON.stringify(r.body))), 8000, 'the later turn');
  await sleep(800);
  assert.equal(server.of('/memory/forget/prepare').length, 2);
  await assertCountsOnly(home);
});

test('Cursor CLI: a forget the model makes through the tools during a run is applied again to the run\'s words saved at its end', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': (body) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null, request_id: body.request_id } }) });
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My locker code is HERON.', 'Noted.'));
  // "Scrub" is no request the plugin or the service reads: the model forgets through the tools.
  const mcp = mcpSession('quotemem-hook', ['mcp'], env(server, home));
  try {
    await mcp.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
    const call = await mcp.request('tools/call', { name: 'quotemem_forget_prepare', arguments: { statement: 'the locker code' } });
    assert.equal(JSON.parse(call.result.content[0].text).status, 'nothing_matched', 'this run\'s words are not saved yet');
  } finally {
    mcp.close();
  }
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:05 AM (UTC+8)', 'Please scrub the locker code from your memory.', 'It is gone.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length >= 2, 8000, 'the forget applied again');
  const saved = server.of('/memory/add').find((r) => /HERON/.test(JSON.stringify(r.body)));
  await sleep(500);
  assert.equal(server.of('/memory/forget/prepare').length, 2, 'once, after the last batch that holds words from before it');
  const again = server.of('/memory/forget/prepare')[1];
  assert.equal(again.body.statement, 'the locker code');
  assert.ok(again.at >= saved.at && again.at >= server.of('/memory/add').at(-1).at, 'after the batches that hold it');
  const kept = JSON.parse(fs.readFileSync(path.join(home, '.quotemem', 'forgotten', `${ownerKey(server.url, 'quei')}.json`), 'utf8'));
  assert.deepEqual(kept.items.map((item) => item.statement), ['the locker code'], 'the statement only');
  assert.equal(fs.statSync(path.join(home, '.quotemem', 'forgotten', `${ownerKey(server.url, 'quei')}.json`)).mode & 0o777, 0o600);
});

test('Cursor: transcript turns are matched with the hooks\' prompts in order; a chat begun before the record, or a mismatch, saves nothing twice and resynchronizes', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  const file = cliTranscript(home, 'c-1');
  const again = 'Run the build once more, please.';
  // The transcript: `again` (no hook saw it), then C and `again`, both through the hooks.
  output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'My cat is called Pixel.', attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Lovely.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  output(await run('beforeSubmitPrompt', { generation_id: 'g2', prompt: again, attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g2', text: 'Built.' }));
  output(await run('stop', { generation_id: 'g2', status: 'completed', loop_count: 0 }));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', again, 'Built.'));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:01 AM (UTC)', 'My cat is called Pixel.', 'Lovely.'));
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 9:02 AM (UTC)', again, 'Built.'));
  output(await run('sessionEnd', appEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the hooks\' batch and the turn they missed');
  await sleep(500);
  const words = JSON.stringify(server.of('/memory/add').map((r) => r.body));
  assert.equal(words.split('Pixel').length - 1, 1, 'C is saved once');
  assert.ok(server.of('/memory/add').some((r) => /:x1-1:/.test(r.body.request_id)), 'the first turn, which no hook saw, is saved');

  // A chat whose prompts ran before the plugin recorded them: nothing from the transcript.
  const old = newHome();
  const runOld = cursorHooks(server, old, { version: '3.24.9' });
  const fileOld = cliTranscript(old, 'c-1');
  const sessionFile = path.join(old, '.quotemem', 'cursor', 'sessions', `${sha16([ownerKey(server.url, 'quei'), 'cursor:c-1'])}.json`);
  fs.mkdirSync(path.dirname(sessionFile), { recursive: true });
  fs.writeFileSync(sessionFile, JSON.stringify({ version: 1, sessionId: 'cursor:c-1', user: ownerKey(server.url, 'quei'), seq: 2, prompts: 2, open: null, lastTurn: 'g2', lastUser: '', capture: { pending: [], turns: 0, firstTurn: null, lastTurn: null, question: null, questionSent: false }, recall: {} }));
  appendRun(fileOld, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', 'My dog is called Rex.', 'Nice.'));
  appendRun(fileOld, transcriptTurn('Thursday, Nov 20, 2025, 9:01 AM (UTC)', 'I prefer tabs.', 'Noted.'));
  const before = server.of('/memory/add').length;
  output(await runOld('sessionEnd', appEnd(fileOld)));
  await sleep(800);
  assert.equal(server.of('/memory/add').length, before, 'saved by the hooks before');
  // A turn after that (the chat goes on in the CLI) is saved.
  appendRun(fileOld, transcriptTurn('Thursday, Nov 20, 2025, 9:02 AM (UTC)', 'My boat is called Wave.', 'Nice.'));
  output(await runOld('sessionEnd', appEnd(fileOld)));
  await waitFor(() => server.of('/memory/add').some((r) => /Wave/.test(JSON.stringify(r.body))), 8000, 'the turn after');

  // A transcript that matches none of the hooks' prompts: nothing from it now, and the record starts again.
  const odd = newHome();
  const runOdd = cursorHooks(server, odd, { version: '3.24.9' });
  const fileOdd = cliTranscript(odd, 'c-1');
  output(await runOdd('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'I live in Lisbon.', attachments: [] }));
  output(await runOdd('afterAgentResponse', { generation_id: 'g1', text: 'Noted.' }));
  output(await runOdd('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  appendRun(fileOdd, transcriptTurn('Thursday, Nov 20, 2025, 9:00 AM (UTC)', 'I live in Lisbon, in Portugal, by the sea.', 'Noted.'));
  output(await runOdd('sessionEnd', appEnd(fileOdd)));
  await sleep(800);
  assert.ok(!server.of('/memory/add').some((r) => /by the sea/.test(JSON.stringify(r.body))));
  assert.match(fs.readFileSync(path.join(odd, '.quotemem', 'cursor', 'quotemem.log'), 'utf8'), /does not match the prompts the hooks saw/);
  appendRun(fileOdd, transcriptTurn('Thursday, Nov 20, 2025, 9:05 AM (UTC)', 'My bike is a Brompton.', 'Nice.'));
  output(await runOdd('sessionEnd', appEnd(fileOdd)));
  await waitFor(() => server.of('/memory/add').some((r) => /Brompton/.test(JSON.stringify(r.body))), 8000, 'a later turn, after the record started again');
});

test('Cursor CLI: a turn whose reply is not written yet waits; a rewritten transcript is read again; a failed queue write keeps the place; no time means the same body', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  const adds = () => server.of('/memory/add').map((r) => r.body);
  // The reply of turn 2 is not written yet: turn 1 goes, turn 2 waits.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My kite is orange.', 'Noted.'));
  fs.appendFileSync(file, transcriptTurn('Thursday, Nov 20, 2025, 11:05 AM (UTC+8)', 'My boat is called Wave.', null));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => adds().length === 1, 8000, 'turn 1');
  await sleep(500);
  assert.equal(adds().length, 1);
  appendRun(file, transcriptLine('assistant', 'A fine name.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => adds().length === 2, 8000, 'turn 2 with its reply');
  assert.deepEqual(adds()[1].messages.map((m) => m.content), ['My boat is called Wave.', 'A fine name.']);
  assert.match(adds()[1].request_id, /:x2-2:/);

  // A checkpoint restore: the transcript keeps turn 1 and goes on differently.
  fs.writeFileSync(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My kite is orange.', 'Noted.')
    + transcriptTurn('Thursday, Nov 20, 2025, 11:06 AM (UTC+8)', 'My kite is called Swift.', 'Nice.') + TURN_ENDED);
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => adds().some((b) => /Swift/.test(JSON.stringify(b))), 8000, 'the new turn 2');
  const turnOne = adds().filter((b) => /orange/.test(JSON.stringify(b)));
  assert.equal(new Set(turnOne.map((b) => JSON.stringify(b))).size, 1, 'turn 1 sent again has the same id and body');

  // The queue cannot be written: nothing is lost; the next run sends it.
  const queueDir = path.join(home, '.quotemem', 'cursor', 'queue');
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:07 AM (UTC+8)', 'My bike is a Brompton.', 'Nice.'));
  fs.chmodSync(queueDir, 0o500);
  try {
    output(await run('sessionEnd', cliEnd(file)));
  } finally {
    fs.chmodSync(queueDir, 0o700);
  }
  await sleep(500);
  assert.ok(!adds().some((b) => /Brompton/.test(JSON.stringify(b))));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => adds().some((b) => /Brompton/.test(JSON.stringify(b))), 8000, 'the turn, on the next run');

  // A stamp in another language: no time, so a resend after the state is lost is the same batch.
  const home2 = newHome();
  const run2 = cursorHooks(server, home2, { version: CLI_VERSION });
  const file2 = cliTranscript(home2, 'chat-fr');
  appendRun(file2, transcriptTurn('samedi 21 novembre 2026, 11:00 (UTC+1)', 'Mon chat s\'appelle Pixel.', 'Joli nom.'));
  output(await run2('sessionEnd', cliEnd(file2, 'chat-fr')));
  await waitFor(() => adds().some((b) => /Pixel/.test(JSON.stringify(b))), 8000, 'the turn');
  fs.rmSync(path.join(home2, '.quotemem', 'cursor', 'sessions'), { recursive: true, force: true });
  await sleep(20);
  output(await run2('sessionEnd', cliEnd(file2, 'chat-fr')));
  await waitFor(() => adds().filter((b) => /Pixel/.test(JSON.stringify(b))).length === 2, 8000, 'the resend');
  const [first, resent] = adds().filter((b) => /Pixel/.test(JSON.stringify(b)));
  assert.equal(JSON.stringify(resent), JSON.stringify(first));
  assert.ok(first.messages.every((m) => !('timestamp' in m)), 'no time');
});

test('Cursor CLI: a forget made through the tools is not applied to words said in a later run of the chat, in the same minute; within its own run it is', async () => {
  const nothing = (body) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null, request_id: body.request_id } });
  const server = await memoryServer({ '/memory/forget/prepare': nothing });
  const toolForget = async (home, statement) => {
    const mcp = mcpSession('quotemem-hook', ['mcp'], env(server, home));
    try {
      await mcp.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
      await mcp.request('tools/call', { name: 'quotemem_forget_prepare', arguments: { statement } });
    } finally {
      mcp.close();
    }
  };
  // One minute for every turn: Cursor's stamps have minutes only.
  const minute = stampAt(Date.now());
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  output(await run('sessionStart', cliFields()));
  // Run 1: the fact, then a forget through the tools.
  appendRun(file, transcriptTurn(minute, 'My locker code is HERON.', 'Noted.'));
  await toolForget(home, 'the locker code');
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'the forget applied again to run 1');
  // Run 2 (--resume), the same minute: a new code, said after the forget.
  appendRun(file, transcriptTurn(minute, 'My new locker code is OWL.', 'Noted: OWL.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').some((r) => /OWL/.test(JSON.stringify(r.body))), 8000, 'run 2');
  await sleep(800);
  assert.equal(server.of('/memory/forget/prepare').length, 2, 'the new code is kept');

  // The same, within one run: whether the second turn came after the
  // forget cannot be told within its minute, so the forget covers it too.
  const home2 = newHome();
  const run2 = cursorHooks(server, home2, { version: CLI_VERSION });
  const file2 = cliTranscript(home2, 'chat-same-run');
  output(await run2('sessionStart', cliFields('chat-same-run')));
  appendRun(file2, transcriptTurn(minute, 'My bike lock code is 4711.', 'Noted.'));
  await toolForget(home2, 'the bike lock code');
  appendRun(file2, transcriptTurn(minute, 'My new bike lock code is 0815.', 'Noted.'));
  const before = server.of('/memory/forget/prepare').length;
  output(await run2('sessionEnd', cliEnd(file2, 'chat-same-run')));
  await waitFor(() => server.of('/memory/forget/prepare').length === before + 1, 8000, 'the forget after the run');
  const again = server.of('/memory/forget/prepare').at(-1);
  const last = server.of('/memory/add').find((r) => /0815/.test(JSON.stringify(r.body)));
  assert.equal(again.body.statement, 'the bike lock code');
  assert.ok(again.at >= last.at, 'after the run\'s last batch');
  await assertCountsOnly(home2);
});

test('Cursor CLI: a turn without a readable time is said after its chat began: a forget kept from before that is not applied to it', async () => {
  const nothing = (body) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null, request_id: body.request_id } });
  const server = await memoryServer({ '/memory/forget/prepare': nothing });
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const mcp = mcpSession('quotemem-hook', ['mcp'], env(server, home));
  try {
    await mcp.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
    await mcp.request('tools/call', { name: 'quotemem_forget_prepare', arguments: { statement: 'my old address' } });
  } finally {
    mcp.close();
  }
  await sleep(20);
  output(await run('sessionStart', cliFields('chat-fr')));
  const file = cliTranscript(home, 'chat-fr');
  appendRun(file, transcriptTurn('samedi 21 novembre 2026, 11:00 (UTC+1)', 'Mon chat s\'appelle Pixel.', 'Joli nom.'));
  output(await run('sessionEnd', cliEnd(file, 'chat-fr')));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the turn');
  await sleep(800);
  assert.equal(server.of('/memory/forget/prepare').length, 1, 'no forget applied again');
});

test('Cursor CLI: a run that did not complete leaves its last turn waiting until its turn_ended line, then saves its final reply', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  fs.appendFileSync(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Look at my notes, then tell me my locker code.', 'Let me look.'));
  output(await run('sessionEnd', { ...cliEnd(file), reason: 'error', final_status: 'error' }));
  await sleep(800);
  assert.equal(server.of('/memory/add').length, 0, 'not ended yet');
  appendRun(file, transcriptLine('assistant', 'Your locker code is HERON.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the turn');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['Look at my notes, then tell me my locker code.', 'Your locker code is HERON.']);
});

const assistantBlocks = (...blocks) => `${JSON.stringify({ role: 'assistant', message: { content: blocks } })}\n`;
const nothingMatched = (body) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null, request_id: body.request_id } });
async function toolForget(server, home, statement, extra = {}) {
  const mcp = mcpSession('quotemem-hook', ['mcp'], env(server, home, extra));
  try {
    await mcp.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
    return await mcp.request('tools/call', { name: 'quotemem_forget_prepare', arguments: { statement } });
  } finally {
    mcp.close();
  }
}

test('Cursor CLI, the recorded two-run shape: each run is saved at its end; the resumed run rewrites the file, and its turn with tool lines is saved with its final text', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  output(await run('sessionStart', cliFields()));
  // Run 1 (-p): user, assistant, turn_ended.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Note for later: my kite is orange.', 'Got it: an orange kite.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'run 1');
  // Run 2 (--resume): the file again, run 1's turn_ended gone; a turn whose
  // reply goes through two tool calls; one turn_ended at the end.
  appendRun(file, transcriptLine('user', '<timestamp>Thursday, Nov 20, 2025, 11:05 AM (UTC+8)</timestamp>\n<user_query>\nWhat is my favourite tea called?\n</user_query>')
    + assistantBlocks({ type: 'text', text: 'I\'ll look up your memory first.' }, { type: 'tool_use', name: 'GetDynamicTools', input: {} })
    + assistantBlocks({ type: 'tool_use', name: 'CallDynamicTool', input: { namespace: 'quotemem', toolName: 'quotemem_search', mcpDetails: {}, arguments: { query: 'favourite tea' } } })
    + assistantBlocks({ type: 'text', text: 'Your favourite tea is oolong.' }));
  assert.equal(fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).length, 7, 'the 7 lines of the recorded run 2');
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'run 2');
  await sleep(500);
  assert.equal(server.of('/memory/add').length, 2, 'run 1 is not sent again: the file was read on from the end of its reply');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['What is my favourite tea called?', 'Your favourite tea is oolong.'], 'the final text is the reply');

  // A one-shot run whose file has no turn_ended yet, which ended completed: saved.
  const file2 = cliTranscript(home, 'chat-one-shot');
  fs.writeFileSync(file2, transcriptTurn('Thursday, Nov 20, 2025, 11:06 AM (UTC+8)', 'My boat is called Wave.', 'A fine name.'));
  output(await run('sessionEnd', cliEnd(file2, 'chat-one-shot')));
  await waitFor(() => server.of('/memory/add').some((r) => /Wave/.test(JSON.stringify(r.body))), 8000, 'the one-shot run');
  // A longer reply written to that turn afterwards: the first saved reply stands.
  appendRun(file2, transcriptLine('assistant', 'A fine name, and a longer reply.'));
  output(await run('sessionEnd', cliEnd(file2, 'chat-one-shot')));
  await sleep(800);
  assert.equal(server.of('/memory/add').filter((r) => /Wave/.test(JSON.stringify(r.body))).length, 1, 'no second copy');
});

test('Cursor: a chat reopened with words already in its transcript: a forget kept before it is still applied to them', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  // The CLI: sessionStart names no transcript; the file is found.
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  // Said before the forget: its minute is read before the forget is kept (a
  // stamp read after it would name the next minute when the minute turns
  // between them, and so prove the words came after the forget).
  const said = stampAt(Date.now());
  await toolForget(server, home, 'the locker code');
  const file = cliTranscript(home);
  appendRun(file, transcriptTurn(said, 'My locker code is HERON.', 'Noted.'));
  output(await run('sessionStart', cliFields()));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'the forget applied to the older words');
  const saved = server.of('/memory/add').find((r) => /HERON/.test(JSON.stringify(r.body)));
  assert.ok(server.of('/memory/forget/prepare')[1].at >= saved.at);

  // The app (the reviewer's case): an hour-old turn, sessionStart naming the
  // file, one turn through the hooks, then sessionEnd.
  const home2 = newHome();
  const run2 = cursorHooks(server, home2, { version: '3.24.9' });
  const file2 = cliTranscript(home2, 'c-1');
  appendRun(file2, transcriptTurn(stampAt(Date.now() - 3_600_000), 'My locker code word is HERON.', 'Noted.'));
  const before = server.of('/memory/forget/prepare').length;
  await toolForget(server, home2, 'the locker code word');
  output(await run2('sessionStart', { session_id: 'c-1', is_background_agent: false, transcript_path: file2 }));
  output(await run2('beforeSubmitPrompt', { generation_id: 'g2', prompt: 'What day is it?', attachments: [] }));
  output(await run2('afterAgentResponse', { generation_id: 'g2', text: 'Friday.' }));
  output(await run2('stop', { generation_id: 'g2', status: 'completed', loop_count: 0 }));
  appendRun(file2, transcriptTurn(stampAt(Date.now()), 'What day is it?', 'Friday.'));
  output(await run2('sessionEnd', appEnd(file2)));
  await waitFor(() => server.of('/memory/forget/prepare').length === before + 2, 8000, 'the forget applied to the hour-old turn');
  const old = server.of('/memory/add').find((r) => /word is HERON/.test(JSON.stringify(r.body)));
  assert.match(old.body.request_id, /:x1-1:/);
  assert.ok(server.of('/memory/forget/prepare').at(-1).at >= old.at);
});

test('Cursor app: a chat whose transcript is missing at its start, or named otherwise at its end, keeps no bound: a forget kept before applies to its turns', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  // Missing at sessionStart (only Cursor's CLI starts a new chat there).
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  await toolForget(server, home, 'my cat\'s name');
  const file = cliTranscript(home, 'c-1');
  fs.rmSync(file);
  output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false, transcript_path: file }));
  appendRun(file, transcriptTurn('samedi 21 novembre 2026, 11:00 (UTC+1)', 'Mon chat s\'appelle Pixel.', 'Joli nom.'));
  output(await run('sessionEnd', appEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'the forget applied to the turn');

  // sessionStart names another file than sessionEnd.
  const home2 = newHome();
  const run2 = cursorHooks(server, home2, { version: '3.24.9' });
  const before = server.of('/memory/forget/prepare').length;
  await toolForget(server, home2, 'my dog\'s name');
  const file2 = cliTranscript(home2, 'c-1');
  appendRun(file2, transcriptTurn('samedi 21 novembre 2026, 11:00 (UTC+1)', 'Mon chien s\'appelle Rex.', 'Joli nom.'));
  const elsewhere = path.join(home2, 'elsewhere', 'c-1.jsonl');
  fs.mkdirSync(path.dirname(elsewhere), { recursive: true });
  fs.writeFileSync(elsewhere, '');
  output(await run2('sessionStart', { session_id: 'c-1', is_background_agent: false, transcript_path: elsewhere }));
  output(await run2('sessionEnd', appEnd(file2)));
  await waitFor(() => server.of('/memory/forget/prepare').length === before + 2, 8000, 'the forget applied to the other file\'s turn');
});

test('Cursor app: a chat reopened and rewritten: a turn added after its start keeps the bound, though Cursor moved its turn_ended line', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  const file = cliTranscript(home, 'c-1');
  appendRun(file, transcriptTurn(stampAt(Date.now() - 3_600_000), 'My locker code word is HERON.', 'Noted.'));
  const minute = stampAt(Date.now());
  await toolForget(server, home, 'the locker code word');
  output(await run('sessionStart', { session_id: 'c-1', is_background_agent: false, transcript_path: file }));
  // The next run rewrites the file: the turn_ended line goes, a new turn takes its place.
  appendRun(file, transcriptTurn(minute, 'My new locker code word is OWL.', 'Noted.'));
  output(await run('sessionEnd', appEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 2 && server.of('/memory/forget/prepare').length === 2, 8000, 'both turns and the forget');
  await sleep(500);
  const owl = server.of('/memory/add').find((r) => /OWL/.test(JSON.stringify(r.body)));
  const again = server.of('/memory/forget/prepare')[1];
  assert.equal(server.of('/memory/forget/prepare').length, 2);
  assert.ok(again.at <= owl.at, 'the forget follows the old turn only: the new code, said after it, is kept');
});

test('Cursor CLI: a turn whose time proves it was said before a forget gets the forget again, even when the clock was set back between runs', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My kite is orange.', 'Noted.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'run 1');
  // Run 1's reading happened, by the clock then, an hour ahead of now.
  const sessionFile = path.join(home, '.quotemem', 'cursor', 'sessions', `${sha16([ownerKey(server.url, 'quei'), `cursor:${CHAT}`])}.json`);
  const state = JSON.parse(fs.readFileSync(sessionFile, 'utf8'));
  state.transcript.at = Date.now() + 3_600_000;
  fs.writeFileSync(sessionFile, JSON.stringify(state));
  appendRun(file, transcriptTurn(stampAt(Date.now() - 300_000), 'My locker code is HERON.', 'Noted.'));
  await toolForget(server, home, 'the locker code');
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2, 8000, 'the forget applied again');
});

test('Cursor CLI: a request the service does not read, after a fact in its own turn, runs again after that turn is stored', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', '1. My locker code is HERON. 2) Please forget my locker code.', 'Done.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/forget/prepare').length === 2 && server.of('/memory/add').length === 1, 8000, 'the forget, the turn, the forget again');
  const [first, again] = server.of('/memory/forget/prepare');
  assert.deepEqual([first.body.statement, again.body.statement], ['my locker code', 'my locker code']);
  assert.ok(first.at <= server.of('/memory/add')[0].at, 'the forget runs before its own turn is stored');
  assert.ok(again.at >= server.of('/memory/add')[0].at, 'and again after it');
  // The same transcript again: nothing runs twice.
  output(await run('sessionEnd', cliEnd(file)));
  await sleep(700);
  assert.equal(server.of('/memory/forget/prepare').length, 2);
  assert.equal(server.of('/memory/add').length, 1);
  assertNoLeaks(home, ['HERON']);
});

// Cursor's interactive hooks for one turn, for the reply tests below.
function cursorTurns(server, home) {
  const run = cursorHooks(server, home, { version: '3.31.15', extraEnv: { QUOTEMEM_CAPTURE_EVERY: '1' } });
  const prompt = async (n, text) => output(await run('beforeSubmitPrompt', { generation_id: `g${n}`, prompt: text, attachments: [] }));
  const stop = (n, fields = {}, options) => run('stop', { generation_id: `g${n}`, status: 'completed', loop_count: 0, ...fields }, options);
  const kept = (n, text, options) => run('afterAgentResponse', { generation_id: `g${n}`, text }, options);
  const later = (ms, fn) => new Promise((resolve) => setTimeout(resolve, ms)).then(fn);
  // The batches that hold a user message.
  const batches = (user) => server.of('/memory/add').filter((r) => r.body.messages.some((m) => m.role === 'user' && m.content === user));
  return { run, prompt, stop, kept, later, batches };
}

test('Cursor app: a stop and its afterAgentResponse run at the same time, in either order: the turn keeps its reply', async () => {
  const server = await memoryServer();
  const { prompt, stop, kept, later, batches } = cursorTurns(server, newHome());
  // The stop starts first; the reply starts 17 ms later and is kept 100 ms after that.
  await prompt(1, 'My kite is orange.');
  for (const result of await Promise.all([stop(1), later(17, () => kept(1, 'Orange kite, noted.', { stdinDelayMs: 100 }))])) output(result);
  assert.equal(await storedReply(server, 'My kite is orange.'), 'Orange kite, noted.');
  // Both start together.
  await prompt(2, 'My bike is blue.');
  for (const result of await Promise.all([kept(2, 'Blue bike, noted.'), stop(2)])) output(result);
  assert.equal(await storedReply(server, 'My bike is blue.'), 'Blue bike, noted.');
  // The reply starts first but reads its input after the stop has begun.
  await prompt(3, 'My boat is green.');
  for (const result of await Promise.all([kept(3, 'Green boat, noted.', { stdinDelayMs: 300 }), stop(3)])) output(result);
  assert.equal(await storedReply(server, 'My boat is green.'), 'Green boat, noted.');
  // The ordinary order: the reply is kept before the stop runs.
  await prompt(4, 'My owl is white.');
  for (const result of await Promise.all([kept(4, 'White owl, noted.'), stop(4, {}, { stdinDelayMs: 300 })])) output(result);
  assert.equal(await storedReply(server, 'My owl is white.'), 'White owl, noted.');
  for (const user of ['My kite is orange.', 'My bike is blue.', 'My boat is green.', 'My owl is white.']) assert.equal(batches(user).length, 1, user);
});

test('Cursor app: the stop reads the reply from the transcript it names, there already or written late; never another chat\'s', async () => {
  const server = await memoryServer();
  const home = newHome();
  const { prompt, stop, later, batches } = cursorTurns(server, home);
  const file = cliTranscript(home, 'c-1');
  const write = (stamp, user, reply) => fs.appendFileSync(file, transcriptTurn(stamp, user, reply) + TURN_ENDED);
  await prompt(1, 'My kite is orange.');
  write('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My kite is orange.', 'Orange kite, noted.');
  const stopped = await stop(1, { transcript_path: file });
  output(stopped);
  assert.equal(await storedReply(server, 'My kite is orange.'), 'Orange kite, noted.');
  assert.ok(stopped.ms < 2500, `an ended turn in the transcript needs no wait (${stopped.ms} ms)`);
  // The transcript gets the turn 300 ms after the stop began.
  await prompt(2, 'My bike is blue.');
  for (const result of await Promise.all([stop(2, { transcript_path: file }), later(300, () => write('Thursday, Nov 20, 2025, 11:05 AM (UTC+8)', 'My bike is blue.', 'Blue bike, noted.'))])) if (result) output(result);
  assert.equal(await storedReply(server, 'My bike is blue.'), 'Blue bike, noted.');
  // Another chat's transcript is never read for this one: the turn is saved
  // with the user's words when the next prompt comes.
  const other = cliTranscript(home, 'another-chat');
  fs.appendFileSync(other, transcriptTurn('Thursday, Nov 20, 2025, 11:06 AM (UTC+8)', 'My boat is green.', 'Green boat, noted.') + TURN_ENDED);
  await prompt(3, 'My boat is green.');
  output(await stop(3, { transcript_path: other }));
  await prompt(4, 'Thanks.');
  assert.equal(await storedReply(server, 'My boat is green.'), null);
  for (const user of ['My kite is orange.', 'My bike is blue.', 'My boat is green.']) assert.equal(batches(user).length, 1, user);
});

test('Cursor app: a turn whose stop found no reply is held open, then saved once: with a late reply, at the next prompt, or at the session end', async () => {
  const server = await memoryServer();
  const home = newHome();
  const { run, prompt, stop, kept, batches } = cursorTurns(server, home);
  // Neither the reply nor the transcript: the turn waits, and its late reply completes it.
  await prompt(1, 'My kite is orange.');
  output(await stop(1));
  await sleep(500);
  assert.equal(batches('My kite is orange.').length, 0, 'held, not saved without its reply');
  output(await kept(1, 'Orange kite, noted.'));
  assert.equal(await storedReply(server, 'My kite is orange.'), 'Orange kite, noted.');
  // No reply ever comes: the next prompt saves the turn with the user's
  // words; a reply that comes after that is not saved again, nor kept for
  // the next turn.
  await prompt(2, 'My bike is blue.');
  output(await stop(2));
  await prompt(3, 'My boat is green.');
  assert.equal(await storedReply(server, 'My bike is blue.'), null);
  output(await kept(2, 'Blue bike, noted.'));
  for (const result of await Promise.all([kept(3, 'Green boat, noted.', { stdinDelayMs: 100 }), stop(3)])) output(result);
  assert.equal(await storedReply(server, 'My boat is green.'), 'Green boat, noted.');
  // A reply kept for an earlier turn and left on disk (a stop that never
  // came) is never given to another turn; the session end removes it.
  const stateDir = path.join(home, '.quotemem', 'cursor');
  const stale = path.join(stateDir, 'replies', `${sha16('c-1')}.${sha16('g0')}.json`);
  fs.writeFileSync(stale, JSON.stringify({ turn: 'g0', text: 'A reply from long ago.', time: Date.now() - 60_000, at: Date.now() - 60_000 }));
  await prompt(4, 'My cat is grey.');
  output(await stop(4));
  assert.ok(fs.existsSync(stale));
  // The session ends: the held turn takes its reply from the transcript.
  const file = cliTranscript(home, 'c-1');
  fs.appendFileSync(file, transcriptTurn('Thursday, Nov 20, 2025, 11:07 AM (UTC+8)', 'My cat is grey.', 'Grey cat, noted.') + TURN_ENDED);
  output(await run('sessionEnd', { reason: 'completed', final_status: 'completed', transcript_path: file }));
  assert.equal(await storedReply(server, 'My cat is grey.'), 'Grey cat, noted.');
  assert.equal(fs.existsSync(stale), false, 'the stale reply is gone with its session');
  // A chat whose console was killed (no sessionEnd) left its kept reply:
  // no other chat reads it, and a session start removes it once a day old.
  const killed = path.join(stateDir, 'replies', `${sha16('c-killed')}.${sha16('k1')}.json`);
  fs.writeFileSync(killed, JSON.stringify({ turn: 'k1', text: 'A killed chat\'s reply.', time: Date.now(), at: Date.now() }));
  const dayOld = new Date(Date.now() - 2 * 86_400_000);
  fs.utimesSync(killed, dayOld, dayOld);
  output(await run('sessionStart', { is_background_agent: false, composer_mode: 'agent' }));
  assert.equal(fs.existsSync(killed), false, 'removed once a day old');
  await sleep(500);
  for (const user of ['My kite is orange.', 'My bike is blue.', 'My boat is green.', 'My cat is grey.']) assert.equal(batches(user).length, 1, user);
  const all = server.of('/memory/add').flatMap((r) => r.body.messages.map((m) => m.content));
  for (const text of ['Blue bike, noted.', 'A reply from long ago.', 'A killed chat\'s reply.']) assert.ok(!all.includes(text), text);
});

test('Cursor app: a held turn of a tab closed without sessionEnd is saved by a later hook after 3 minutes, with the transcript\'s reply or the user\'s words', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.31.15', extraEnv: { QUOTEMEM_CAPTURE_EVERY: '1' } });
  const batches = (user) => server.of('/memory/add').filter((r) => r.body.messages.some((m) => m.role === 'user' && m.content === user));
  const file = cliTranscript(home, 'c-1');
  // Chats already begun, each with a turn still running: no first prompt of
  // a new chat comes later (that would send the held turns at once).
  output(await run('beforeSubmitPrompt', { conversation_id: 'c-2', generation_id: 'k0', prompt: 'Let us write a parser.', attachments: [] }));
  output(await run('beforeSubmitPrompt', { conversation_id: 'c-3', generation_id: 'h0', prompt: 'Let us write a lexer.', attachments: [] }));
  output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'My kite is orange.', attachments: [] }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0, transcript_path: file }));
  output(await run('beforeSubmitPrompt', { conversation_id: 'c-3', generation_id: 'h1', prompt: 'My bike is blue.', attachments: [] }));
  output(await run('stop', { conversation_id: 'c-3', generation_id: 'h1', status: 'completed', loop_count: 0 }));
  // Both tabs are closed (no sessionEnd); the first reply reaches its transcript.
  fs.appendFileSync(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'My kite is orange.', 'Orange kite, noted.') + TURN_ENDED);
  await sleep(300);
  assert.equal(batches('My kite is orange.').length + batches('My bike is blue.').length, 0, 'held');
  // Three minutes on (the session files made older), another chat's prompt saves them.
  for (const chat of ['c-1', 'c-3']) {
    const sessionFile = path.join(home, '.quotemem', 'cursor', 'sessions', `${sha16([ownerKey(server.url, 'quei'), `cursor:${chat}`])}.json`);
    const state = JSON.parse(fs.readFileSync(sessionFile, 'utf8'));
    state.open.held.at -= 4 * 60_000;
    fs.writeFileSync(sessionFile, JSON.stringify(state));
    const old = new Date(Date.now() - 4 * 60_000);
    fs.utimesSync(sessionFile, old, old);
  }
  output(await run('beforeSubmitPrompt', { conversation_id: 'c-2', generation_id: 'k1', prompt: 'Hello there.', attachments: [] }));
  assert.equal(await storedReply(server, 'My kite is orange.'), 'Orange kite, noted.');
  assert.equal(await storedReply(server, 'My bike is blue.'), null);
  // A reply that comes after that is not saved again.
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Orange kite, noted.' }));
  await sleep(500);
  assert.equal(batches('My kite is orange.').length, 1);
  assert.equal(batches('My bike is blue.').length, 1);
});

test('Cursor app: a fact said in a chat closed just now is recalled in a new chat, at its first prompt; the hook answers before the background start', async () => {
  // A batch every 2 turns, and no sessionEnd when a tab closes.
  const stored = [];
  const server = await memoryServer({
    '/memory/add': async (body) => {
      await sleep(300);
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': (body) => ({ body: { data: /猫/.test(body.query) ? stored.filter((t) => t.includes('猫')).map((t, i) => HIT(`01JX00000000000000000000D${i}`, t, { score: 0.9 })) : [] } }),
  });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.31.15' });
  output(await run('sessionStart', { conversation_id: 'chat-a', session_id: 'chat-a', is_background_agent: false, composer_mode: 'agent' }));
  output(await run('beforeSubmitPrompt', { conversation_id: 'chat-a', generation_id: 'a1', prompt: '我的猫叫bob', attachments: [] }));
  for (const result of await Promise.all([
    run('afterAgentResponse', { conversation_id: 'chat-a', generation_id: 'a1', text: '好的，记住了。' }),
    run('stop', { conversation_id: 'chat-a', generation_id: 'a1', status: 'completed', loop_count: 0 }, { stdinDelayMs: 100 }),
  ])) output(result);
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'one turn: not sent yet');
  // The tab is closed (no sessionEnd) and a new chat starts at once.
  // The background start is slow (a loaded machine, PowerShell): the digest is printed first.
  const slow = { QUOTEMEM_TEST_LAUNCH_DELAY_MS: '2500' };
  const started = await cursorHooks(server, home, { version: '3.31.15', extraEnv: slow })('sessionStart', { conversation_id: 'chat-b', session_id: 'chat-b', is_background_agent: false, composer_mode: 'agent' });
  output(started);
  assert.ok(started.firstOutMs < 2000, `the answer came at ${started.firstOutMs} ms`);
  assert.ok(started.ms >= 2500, `the start ran after it, before the hook exited (${started.ms} ms)`);
  const asked = await cursorHooks(server, home, { version: '3.31.15', extraEnv: slow })('beforeSubmitPrompt', { conversation_id: 'chat-b', generation_id: 'b1', prompt: '我的猫叫什么', attachments: [] });
  assert.match(output(asked).additional_context ?? '', /我的猫叫bob/);
  assert.equal(server.of('/memory/add').length, 1);
});

test('Cursor app: a new chat whose start and first prompt come together finds a chat closed just now; a chat with a turn running keeps its batch; a turn held for its reply goes with the user\'s words', async () => {
  const stored = [];
  const server = await memoryServer({
    '/memory/add': async (body) => {
      await sleep(300);
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': (body) => ({ body: { data: /猫/.test(body.query) ? stored.filter((t) => t.includes('猫')).map((t, i) => HIT(`01JX00000000000000000000E${i}`, t, { score: 0.9 })) : [] } }),
  });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.31.15' });
  const batches = (user) => server.of('/memory/add').filter((r) => r.body.messages.some((m) => m.role === 'user' && m.content === user));
  const chat = (id) => ({ conversation_id: id, session_id: id, is_background_agent: false, composer_mode: 'agent' });
  // Three chats begin while each other's first turn runs.
  for (const [id, turn, text] of [['chat-r', 'r1', 'My bike is blue.'], ['chat-h', 'h1', 'My kite is orange.'], ['chat-a', 'a1', '我的猫叫bob']]) {
    output(await run('sessionStart', chat(id)));
    output(await run('beforeSubmitPrompt', { conversation_id: id, generation_id: turn, prompt: text, attachments: [] }));
  }
  // Chat R: its turn ends and the next one runs.
  for (const result of await Promise.all([
    run('afterAgentResponse', { conversation_id: 'chat-r', generation_id: 'r1', text: 'Blue bike, noted.' }),
    run('stop', { conversation_id: 'chat-r', generation_id: 'r1', status: 'completed', loop_count: 0 }),
  ])) output(result);
  output(await run('beforeSubmitPrompt', { conversation_id: 'chat-r', generation_id: 'r2', prompt: 'Now refactor the parser.', attachments: [] }));
  // Chat H: its stop came, its reply not yet (held).
  output(await run('stop', { conversation_id: 'chat-h', generation_id: 'h1', status: 'completed', loop_count: 0 }));
  // Chat A: its turn ends; its tab is closed.
  for (const result of await Promise.all([
    run('afterAgentResponse', { conversation_id: 'chat-a', generation_id: 'a1', text: '好的，记住了。' }),
    run('stop', { conversation_id: 'chat-a', generation_id: 'a1', status: 'completed', loop_count: 0 }),
  ])) output(result);
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'one turn each: nothing sent yet');
  // Chat B at once: Cursor does not wait for its sessionStart before the first prompt.
  const [started, asked] = await Promise.all([
    run('sessionStart', chat('chat-b')),
    run('beforeSubmitPrompt', { conversation_id: 'chat-b', generation_id: 'b1', prompt: '我的猫叫什么', attachments: [] }),
  ]);
  output(started);
  assert.match(output(asked).additional_context ?? '', /我的猫叫bob/);
  assert.equal(await storedReply(server, '我的猫叫bob'), '好的，记住了。');
  assert.equal(await storedReply(server, 'My kite is orange.'), null);
  await sleep(500);
  assert.equal(batches('My bike is blue.').length, 0, 'the chat whose turn runs keeps its batch');
  assert.equal(server.of('/memory/add').length, 2);
});

test('Cline: a new task\'s first prompt finds what a task left between turns said just now', async () => {
  const stored = [];
  const server = await memoryServer({
    '/memory/add': async (body) => {
      await sleep(300);
      for (const m of body.messages) if (m.role === 'user') stored.push(m.content);
      return { body: { success: true, outcome: { added: 1 } } };
    },
    '/memory/search': (body) => ({ body: { data: /cat/.test(body.query) ? stored.filter((t) => t.includes('cat')).map((t, i) => HIT(`01JX00000000000000000000F${i}`, t, { score: 0.9 })) : [] } }),
  });
  const home = newHome();
  const dir = path.join(home, '.cline', 'hooks');
  copyDir(path.join(INTEGRATIONS, 'cline', 'hooks'), dir);
  const base = { clineVersion: '4.1.26', timestamp: String(Date.now()), workspaceRoots: ['/work/project'], userId: 'local-user', sessionContext: {} };
  const run = (taskId, hookName, fields) => sh(`"${path.join(dir, hookName)}"`, { ...base, taskId, hookName, ...fields }, env(server, home));
  output(await run('task-1', 'UserPromptSubmit', { userPromptSubmit: { prompt: 'My cat is called Bob.', attachments: [] } }));
  output(await run('task-1', 'TaskComplete', { taskComplete: { taskMetadata: { taskId: 'task-1', ulid: '', initialTask: '', result: 'Nice name.' } } }));
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'one turn: not sent yet');
  // Cline runs no hook when a task is left: the next task's first prompt sends it.
  const asked = output(await run('task-2', 'UserPromptSubmit', { userPromptSubmit: { prompt: 'What is my cat called?', attachments: [] } }));
  assert.match(asked.contextModification ?? '', /My cat is called Bob\./);
  assert.equal(server.of('/memory/add').length, 1);
});

test('Cursor: a forget that finished in the background is told in the chat that asked, not in a new chat, which recalls as usual', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': (body) => ({ delay: 1500, body: plan(body) }) });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9', extraEnv: { QUOTEMEM_FORGET_TIMEOUT_MS: '300' } });
  const queue = new DiskQueue({ stateDir: path.join(home, '.quotemem', 'cursor'), url: server.url, userId: 'quei' }, { warn() {}, warnOnce() {}, debug() {} });
  const prompt = (chat, generation, text) => run('beforeSubmitPrompt', { conversation_id: chat, generation_id: generation, prompt: text, attachments: [] });
  const answer = async (chat, generation, text) => {
    output(await run('afterAgentResponse', { conversation_id: chat, generation_id: generation, text }));
    output(await run('stop', { conversation_id: chat, generation_id: generation, status: 'completed', loop_count: 0 }));
  };

  output(await run('sessionStart', { conversation_id: 'c-1', session_id: 'c-1', is_background_agent: false, composer_mode: 'agent' }));
  const asked = output(await prompt('c-1', 'g1', 'Please forget that my dog is called Rex.'));
  assert.ok(asked.additional_context.startsWith(FORGET_HEADER), asked.additional_context);
  assert.match(asked.additional_context, /being forgotten in the background/);
  await answer('c-1', 'g1', 'I will forget it.');
  const id = server.of('/memory/forget/prepare')[0].body.request_id;
  await waitFor(() => queue.result(id)?.status === 'forgotten', 8000, 'the forget finished');

  // A new chat: its digest and its question get the memories, and no word of the forget.
  const start = output(await run('sessionStart', { conversation_id: 'c-2', session_id: 'c-2', is_background_agent: false, composer_mode: 'agent' }));
  assert.match(start.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  const recall = output(await prompt('c-2', 'h1', 'What do I prefer for indentation?'));
  assert.match(recall.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*User prefers tabs over spaces/);
  assert.doesNotMatch(recall.additional_context, /QuoteMem: automatic forget|forg[eo]t/i);
  await answer('c-2', 'h1', 'Tabs.');

  // The chat that asked is told at its next prompt, once.
  const told = output(await prompt('c-1', 'g2', 'Thanks. Now write a haiku.'));
  assert.ok(told.additional_context.startsWith(FORGET_HEADER), told.additional_context);
  assert.match(told.additional_context, /finished now\. Done: 1 stored memory was forgotten/);
  assert.doesNotMatch(told.additional_context, /Rex|dog/);
  await answer('c-1', 'g2', 'Rain on the roof.');
  assert.deepEqual(output(await prompt('c-1', 'g3', 'Another haiku.')), { continue: true }, 'reported once');
});

test('Cursor sessionStart: a forget of the same user still queued holds the digest back; done in time, another user\'s, or unreadable: the usual digest', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  const start = (id, runner = run) => runner('sessionStart', { conversation_id: id, session_id: id, is_background_agent: false, composer_mode: 'agent' });
  const searches = () => server.of('/memory/search').length;
  const queue = new DiskQueue({ stateDir: path.join(home, '.quotemem', 'cursor'), url: server.url, userId: 'quei' }, { warn() {}, warnOnce() {}, debug() {} });
  const forgetJob = (id) => ({ kind: 'forget', request_id: id, body: { user_id: 'quei', request_id: id }, url: server.url, created: Date.now(), attempts: 0, nextAt: 0 });
  const jobFile = () => queue.entries().at(-1).file;

  // No forget queued: the digest as before, with no wait.
  const plainRun = await start('s-plain');
  const plain = output(plainRun).additional_context;
  assert.match(plain, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  assert.ok(plainRun.firstOutMs < 2500, `no wait (${plainRun.firstOutMs} ms)`);
  const sameRun = await start('s-plain-2');
  assert.equal(output(sameRun).additional_context, plain, 'the same digest text');
  assert.ok(sameRun.firstOutMs < 2500, `no wait (${sameRun.firstOutMs} ms)`);

  // No background sender runs for this user: a queued forget stays as written.
  const lock = tryLock(queue.lockFile);
  assert.ok(lock);
  try {
    // Still pending after the wait: no digest, nothing about the forget, no search.
    queue.enqueue([forgetJob('f1')]);
    const before = searches();
    const heldRun = await start('s-held');
    assert.deepEqual(output(heldRun), {}, 'no digest and no text about the forget');
    assert.doesNotMatch(heldRun.stdout, /QuoteMem|forget/i);
    assert.equal(searches(), before, 'no digest search while the forget runs');
    assert.ok(heldRun.firstOutMs >= 2500, `waited for the forget (${heldRun.firstOutMs} ms)`);
    assert.ok(heldRun.firstOutMs < 4000, `answered within Cursor's 5 s sessionStart timeout, less its margin (${heldRun.firstOutMs} ms)`);
    // That chat's first prompt adds no late digest.
    assert.deepEqual(output(await run('beforeSubmitPrompt', { conversation_id: 's-held', generation_id: 'h1', prompt: 'Let us start.', attachments: [] })), { continue: true });

    // Another user of the same state folder neither waits nor loses the digest.
    const otherRun = await start('s-other', cursorHooks(server, home, { version: '3.24.9', extraEnv: { QUOTEMEM_USER_ID: 'someone-else' } }));
    assert.match(output(otherRun).additional_context, /tabs over spaces/);
    assert.ok(otherRun.firstOutMs < 2500, `another user did not wait (${otherRun.firstOutMs} ms)`);

    // Done: the next prompt that asks recalls as usual.
    queue.writeResult('f1', { status: 'forgotten', forgotten: 1 });
    fs.unlinkSync(jobFile());
    const asked = output(await run('beforeSubmitPrompt', { conversation_id: 's-held', generation_id: 'h2', prompt: 'What do I prefer, tabs or spaces?', attachments: [] }));
    assert.match(asked.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);

    // A queue file that cannot be read: no pending forget, the usual digest.
    queue.enqueue([forgetJob('f2')]);
    fs.writeFileSync(jobFile(), '{not json');
    const unreadable = await start('s-unreadable');
    assert.equal(output(unreadable).additional_context, plain);
    assert.ok(unreadable.firstOutMs < 2500, `no wait (${unreadable.firstOutMs} ms)`);
    fs.unlinkSync(jobFile());

    // Done within the wait: the usual digest once it is.
    queue.enqueue([forgetJob('f3')]);
    const file = jobFile();
    setTimeout(() => {
      queue.writeResult('f3', { status: 'forgotten', forgotten: 1 });
      fs.unlinkSync(file);
    }, 1000);
    const doneRun = await start('s-done');
    assert.equal(output(doneRun).additional_context, plain, 'the usual digest');
    assert.ok(doneRun.firstOutMs >= 900, `waited for the forget (${doneRun.firstOutMs} ms)`);
  } finally {
    releaseLock(lock);
  }
});

test('Cursor and Devin: a hook timeout the user edited in the hooks file holds; the work after the answer ends before it', async () => {
  const server = await memoryServer();
  // The background start takes 2.5 s; by default a 5 s hook waits for it.
  const slow = { QUOTEMEM_TEST_LAUNCH_DELAY_MS: '2500', QUOTEMEM_CAPTURE_EVERY: '1' };
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.31.15', extraEnv: slow });
  const cursorTurn = async (n) => {
    output(await run('beforeSubmitPrompt', { generation_id: `g${n}`, prompt: `My kite number ${n} is orange.`, attachments: [] }));
    output(await run('afterAgentResponse', { generation_id: `g${n}`, text: 'Noted.' }));
    const stop = await run('stop', { generation_id: `g${n}`, status: 'completed', loop_count: 0 });
    output(stop);
    return stop.ms;
  };
  const byDefault = await cursorTurn(1);
  assert.ok(byDefault >= 2500, `the 5 s stop hook waited for the start (${byDefault} ms)`);
  // Until that sender has sent the batch and exited (it waits 2 s for more).
  await sleep(3500);
  const file = path.join(home, '.cursor', 'hooks.json');
  const edited = JSON.parse(fs.readFileSync(file, 'utf8'));
  edited.hooks.stop[0].timeout = 2;
  fs.writeFileSync(file, JSON.stringify(edited));
  const short = await cursorTurn(2);
  assert.ok(short < 2300, `the stop hook ended within its edited 2 s (${short} ms)`);
  // Devin: the project's .devin/hooks.v1.json gives Stop 2 s.
  const project = path.join(home, 'project');
  fs.mkdirSync(path.join(project, '.devin'), { recursive: true });
  const devinHooks = config('devin/hooks.v1.json');
  devinHooks.Stop[0].hooks[0].timeout = 2;
  fs.writeFileSync(path.join(project, '.devin', 'hooks.v1.json'), JSON.stringify(devinHooks));
  const devin = (event, fields) => sh(commandOf(devinHooks, event), { hook_event_name: event, session_id: 'd-9', cwd: project, ...fields }, env(server, home, slow));
  output(await devin('UserPromptSubmit', { prompt_id: 'p1', prompt: 'My bike is blue.' }), { empty: true });
  const stop = await devin('Stop', { prompt_id: 'p1', stop_hook_active: false, last_assistant_message: 'Blue bike, noted.' });
  output(stop, { empty: true });
  assert.ok(stop.ms < 2300, `Devin's Stop ended within its edited 2 s (${stop.ms} ms)`);
});

test('Cursor: a follow-up sent while the earlier turn still runs: each turn is saved once with its own reply, whichever ends first', async () => {
  const server = await memoryServer();
  const { prompt, stop, kept, batches } = cursorTurns(server, newHome());
  // Turn 2 is sent while turn 1 works; turn 1 ends first.
  await prompt(1, 'My kite is orange.');
  await prompt(2, 'My bike is blue.');
  for (const result of await Promise.all([kept(1, 'Orange kite, noted.'), stop(1, {}, { stdinDelayMs: 100 })])) output(result);
  assert.equal(await storedReply(server, 'My kite is orange.'), 'Orange kite, noted.');
  for (const result of await Promise.all([kept(2, 'Blue bike, noted.'), stop(2, {}, { stdinDelayMs: 100 })])) output(result);
  assert.equal(await storedReply(server, 'My bike is blue.'), 'Blue bike, noted.');
  // Turn 4 is sent while turn 3 works; turn 4 ends first.
  await prompt(3, 'My boat is green.');
  await prompt(4, 'My owl is white.');
  for (const result of await Promise.all([kept(4, 'White owl, noted.'), stop(4, {}, { stdinDelayMs: 100 })])) output(result);
  assert.equal(await storedReply(server, 'My owl is white.'), 'White owl, noted.');
  for (const result of await Promise.all([kept(3, 'Green boat, noted.'), stop(3, {}, { stdinDelayMs: 100 })])) output(result);
  assert.equal(await storedReply(server, 'My boat is green.'), 'Green boat, noted.');
  await sleep(300);
  for (const user of ['My kite is orange.', 'My bike is blue.', 'My boat is green.', 'My owl is white.']) assert.equal(batches(user).length, 1, user);
});

test('Cursor CLI 2026.11.19 and later, interactive: per-turn recall and the forget note in additional_context; an earlier CLI gets none', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const fields = (n) => ({ ...cliFields(), generation_id: `g${n}` });
  const recall = output(await run('beforeSubmitPrompt', { ...fields(1), prompt: 'What do I prefer for indentation?', attachments: [] }));
  assert.equal(recall.continue, true);
  assert.match(recall.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*User prefers tabs over spaces/);
  output(await run('afterAgentResponse', { ...fields(1), text: 'Tabs.' }));
  output(await run('stop', { ...fields(1), status: 'completed', loop_count: 0 }));
  const forget = output(await run('beforeSubmitPrompt', { ...fields(2), prompt: 'Please forget that my dog is called Rex.', attachments: [] }));
  assert.ok(forget.additional_context.startsWith(FORGET_HEADER), forget.additional_context);
  assert.match(forget.additional_context, /1 stored memory was forgotten/);
  // Nothing showed that an earlier CLI reads the field: its recall stays the digest and the tools.
  const searches = server.of('/memory/search').length;
  const older = cursorHooks(server, newHome(), { version: '2026.11.17-0a1b2c3' });
  assert.deepEqual(output(await older('beforeSubmitPrompt', { ...cliFields(), generation_id: 'g9', prompt: 'What do I prefer for indentation?', attachments: [] })), { continue: true });
  assert.equal(server.of('/memory/search').length, searches);
});

test('a forget made through the tools also queues the pending turns of this user\'s sessions, with the forget behind them', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  const home = newHome();
  const run = cursorHooks(server, home, { version: '3.24.9' });
  // One turn through the hooks: pending until a second (QUOTEMEM_CAPTURE_EVERY=2).
  output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'My locker code is HERON.', attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Noted.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'pending');
  await toolForget(server, home, 'the locker code');
  await waitFor(() => server.of('/memory/add').length === 1 && server.of('/memory/forget/prepare').length === 2, 8000, 'the pending turn, then the forget');
  assert.ok(server.of('/memory/forget/prepare')[1].at >= server.of('/memory/add')[0].at);
});

test('a run given its own QUOTEMEM_STATE_DIR and QUOTEMEM_ENV_FILE writes nothing under the home but the kept forget: Devin, Cursor and a forget through the tools', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  const home = newHome();
  const devinHooks = config('devin/hooks.v1.json');
  const devin = (extra) => (event, fields) => sh(commandOf(devinHooks, event), { hook_event_name: event, cwd: '/work/project', ...fields }, env(server, home, extra));
  // The profile's own state from before, of another user: a Devin turn still pending (a batch every 2).
  // Its idle time is off, so no background flusher waits for that turn during the test.
  const before = devin({ QUOTEMEM_USER_ID: 'someone-else', QUOTEMEM_IDLE_FLUSH_MS: '0' });
  output(await before('UserPromptSubmit', { session_id: 'd-home', prompt_id: 'p1', prompt: 'My locker code is HERON.' }), { empty: true });
  output(await before('Stop', { session_id: 'd-home', prompt_id: 'p1', stop_hook_active: false, last_assistant_message: 'Noted.' }), { empty: true });
  // A run with its own state folder and env file, outside the home.
  const elsewhere = newHome();
  const state = path.join(elsewhere, 'state');
  const envFile = path.join(elsewhere, 'env');
  fs.writeFileSync(envFile, 'QUOTEMEM_SESSION_DIGEST=off\n', { mode: 0o600 });
  const own = { QUOTEMEM_STATE_DIR: state, QUOTEMEM_ENV_FILE: envFile, QUOTEMEM_CAPTURE_EVERY: '1' };
  const cursor = cursorHooks(server, home, { version: '3.31.15', extraEnv: own });
  const snapshot = () => walk(home).map((file) => `${path.relative(home, file)} ${fs.statSync(file).size} ${fs.statSync(file).mtimeMs}`).sort();
  const untouched = snapshot();
  const run = devin(own);
  output(await run('UserPromptSubmit', { session_id: 'd-own', prompt_id: 'q1', prompt: 'My bike is blue.' }), { empty: true });
  output(await run('Stop', { session_id: 'd-own', prompt_id: 'q1', stop_hook_active: false, last_assistant_message: 'Blue bike, noted.' }), { empty: true });
  output(await run('SessionEnd', { session_id: 'd-own' }), { empty: true });
  output(await cursor('sessionStart', { session_id: 'c-own', conversation_id: 'c-own', is_background_agent: false, composer_mode: 'agent' }));
  output(await cursor('beforeSubmitPrompt', { conversation_id: 'c-own', generation_id: 'g1', prompt: 'My kite is orange.', attachments: [] }));
  output(await cursor('afterAgentResponse', { conversation_id: 'c-own', generation_id: 'g1', text: 'Orange kite, noted.' }));
  output(await cursor('stop', { conversation_id: 'c-own', generation_id: 'g1', status: 'completed', loop_count: 0 }));
  await toolForget(server, home, 'the locker code', own);
  await waitFor(() => ['My bike is blue.', 'My kite is orange.'].every((text) => server.of('/memory/add').some((r) => JSON.stringify(r.body).includes(text))), 8000, 'the run\'s turns');
  await sleep(500);
  // Only the kept forget, which hooks without that state folder read there.
  const written = snapshot().filter((line) => !untouched.includes(line)).map((line) => line.split(' ')[0]);
  assert.deepEqual(written, [path.join('.quotemem', 'forgotten', `${ownerKey(server.url, 'quei')}.json`)], 'nothing else under the home was written');
  assert.ok(!server.of('/memory/add').some((r) => JSON.stringify(r.body).includes('HERON')), 'another user\'s pending turn was left as it was');
  for (const host of ['devin', 'cursor']) {
    assert.ok(fs.existsSync(path.join(state, 'native', `${host}.json`)), `the ${host} marker is in the state folder`);
    assert.ok(nativeInstalled({ HOME: home, USERPROFILE: home, QUOTEMEM_STATE_DIR: state }, host));
  }
});

test('QUOTEMEM_STATE_DIR set for the MCP server only: a forget through the tools still queues the hooks\' unsent turns first, and the hooks read the kept forget', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': nothingMatched });
  const home = newHome();
  // The hooks keep their state in ~/.quotemem/cursor: one turn pending (a batch every 2).
  const run = cursorHooks(server, home, { version: '3.31.15' });
  output(await run('beforeSubmitPrompt', { generation_id: 'g1', prompt: 'My locker code is HERON.', attachments: [] }));
  output(await run('afterAgentResponse', { generation_id: 'g1', text: 'Noted.' }));
  output(await run('stop', { generation_id: 'g1', status: 'completed', loop_count: 0 }));
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'pending');
  // The MCP server's own config gives it another state folder.
  const state = path.join(newHome(), 'mcp-state');
  await toolForget(server, home, 'the locker code', { QUOTEMEM_STATE_DIR: state });
  await waitFor(() => server.of('/memory/add').length === 1 && server.of('/memory/forget/prepare').length === 2, 8000, 'the hooks\' pending turn, then the forget');
  assert.match(JSON.stringify(server.of('/memory/add')[0].body), /HERON/);
  assert.ok(server.of('/memory/forget/prepare')[1].at >= server.of('/memory/add')[0].at);
  // The kept forget is in both places: the hooks, without QUOTEMEM_STATE_DIR, read ~/.quotemem/forgotten.
  const owner = { url: server.url, userId: 'quei' };
  for (const file of [path.join(state, 'forgotten', `${ownerKey(server.url, 'quei')}.json`), path.join(home, '.quotemem', 'forgotten', `${ownerKey(server.url, 'quei')}.json`)]) {
    assert.match(fs.readFileSync(file, 'utf8'), /the locker code/, file);
  }
  assert.deepEqual(forgottenOf({ HOME: home, USERPROFILE: home }, owner).map((item) => item.statement), ['the locker code']);
  // The other way round: hooks with their own folder still read a forget kept in ~/.quotemem.
  assert.deepEqual(forgottenOf({ HOME: home, USERPROFILE: home, QUOTEMEM_STATE_DIR: path.join(home, 'other') }, owner).map((item) => item.statement), ['the locker code']);
  // doctor names the MCP server it can read whose folder differs from the hooks'.
  fs.writeFileSync(path.join(home, '.cursor', 'mcp.json'), JSON.stringify({ mcpServers: { quotemem: { command: 'node', args: ['x', 'mcp'], env: { QUOTEMEM_STATE_DIR: state } } } }));
  const doctor = spawnSync(process.execPath, [path.join(INTEGRATIONS, 'cli', 'quotemem.mjs'), 'doctor'], { env: env(server, home), encoding: 'utf8' });
  assert.match(doctor.stdout, /FAIL {2}the QuoteMem MCP server in .*mcp\.json keeps its state in .*mcp-state, the hooks in their own folders .*: set QUOTEMEM_STATE_DIR only in .*env/);
  assert.match(doctor.stdout, /agents seen with their own adapter: cursor/);
  // A config doctor cannot parse (a comment, a trailing comma) is said to be unchecked.
  fs.writeFileSync(path.join(home, '.cursor', 'mcp.json'), `// mine\n${JSON.stringify({ mcpServers: { quotemem: { command: 'node', env: { QUOTEMEM_STATE_DIR: state } } } })}\n`);
  const unread = spawnSync(process.execPath, [path.join(INTEGRATIONS, 'cli', 'quotemem.mjs'), 'doctor'], { env: env(server, home), encoding: 'utf8' });
  assert.match(unread.stdout, /note {2}could not check the state folder of the QuoteMem MCP server in .*mcp\.json: it is not plain JSON/);
  assert.doesNotMatch(unread.stdout, /keeps its state in/);
});

test('Cursor CLI: a transcript with more new text than one reading holds is named in the log', async () => {
  const server = await memoryServer();
  const home = newHome();
  const run = cursorHooks(server, home, { version: CLI_VERSION });
  const file = cliTranscript(home);
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Read this file.', 'x'.repeat(33 * 1024 * 1024)));
  output(await run('sessionEnd', cliEnd(file)));
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0);
  assert.match(fs.readFileSync(path.join(home, '.quotemem', 'cursor', 'quotemem.log'), 'utf8'), /more new text than the plugin reads at once \(32 MiB\)/);
  // The place moved past it: the chat's next turn is saved.
  appendRun(file, transcriptTurn('Thursday, Nov 20, 2025, 11:06 AM (UTC+8)', 'My kite is orange.', 'Noted.'));
  output(await run('sessionEnd', cliEnd(file)));
  await waitFor(() => server.of('/memory/add').some((r) => /kite is orange/.test(JSON.stringify(r.body))), 8000, 'the next turn');
});

test('Cline: hook files in ~/.cline/hooks inject recall as contextModification and capture TaskComplete', async () => {
  const server = await memoryServer();
  const home = newHome();
  const dir = path.join(home, '.cline', 'hooks');
  copyDir(path.join(INTEGRATIONS, 'cline', 'hooks'), dir);
  const base = { clineVersion: '4.1.26', timestamp: '1763632800000', taskId: 'task-7', workspaceRoots: ['/work/project'], userId: 'local-user', sessionContext: {} };
  const run = (hookName, fields) => sh(`"${path.join(dir, hookName)}"`, { ...base, hookName, ...fields }, env(server, home));
  const prompt = (text) => run('UserPromptSubmit', { userPromptSubmit: { prompt: text, attachments: [] } });
  const done = (result) => run('TaskComplete', { taskComplete: { taskMetadata: { taskId: 'task-7', ulid: '', initialTask: '', result } } });

  assert.deepEqual(output(await prompt('I prefer tabs over spaces.')), { cancel: false });
  assert.deepEqual(output(await done('Noted: tabs.')), { cancel: false });
  output(await prompt('My dog is called Rex.'));
  output(await done('Nice name.'));
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  assert.equal(server.of('/memory/add')[0].body.session_id, 'cline:task-7');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I prefer tabs over spaces.', 'Noted: tabs.', 'My dog is called Rex.', 'Nice name.']);
  assert.ok(server.of('/memory/add')[0].body.messages.every((m) => m.timestamp === 1763632800000), 'each message carries the time Cline gave');

  const recall = output(await prompt('What do I prefer for indentation?'));
  assert.equal(recall.cancel, false);
  assert.match(recall.contextModification, /User prefers tabs over spaces/);
  output(await run('TaskCancel', { taskCancel: { taskMetadata: { taskId: 'task-7', ulid: '', initialTask: '', completionStatus: 'cancelled' } } }));

  const forget = output(await prompt('请忘掉我的狗叫 Rex。'));
  assert.ok(forget.contextModification.startsWith(FORGET_HEADER));
  assert.equal(server.of('/memory/forget/prepare')[0].body.statement, '我的狗叫 Rex');

  // Without quotemem-hook on PATH the hook lets the turn go on.
  const bare = await sh(`"${path.join(dir, 'UserPromptSubmit')}"`, { ...base, hookName: 'UserPromptSubmit', userPromptSubmit: { prompt: 'hi' } }, { PATH: '/usr/bin:/bin', HOME: home, USERPROFILE: home });
  assert.deepEqual(output(bare), { cancel: false });
  assertNoLeaks(home, ['Rex', 'tabs']);
});

// --------------------------------------------------------------------- goose

test('goose: plugin hooks write recall to the MOIM file, clear it at Stop, and capture', async () => {
  const server = await memoryServer();
  const home = newHome();
  const plugin = path.join(home, '.agents', 'plugins', 'quotemem');
  copyDir(path.join(INTEGRATIONS, 'goose', 'quotemem'), plugin);
  const hooks = JSON.parse(fs.readFileSync(path.join(plugin, 'hooks', 'hooks.json'), 'utf8')).hooks;
  const moim = path.join(home, '.quotemem', 'goose', 'moim.md');
  const childEnv = env(server, home, { PLUGIN_ROOT: plugin, GOOSE_MOIM_MESSAGE_FILE: '~/.quotemem/goose/moim.md' });
  const run = (event, fields) => sh(commandOf(hooks, event).replaceAll('${PLUGIN_ROOT}', plugin), { event, session_id: '20261120_1', ...fields }, childEnv);
  const prompt = (text) => run('UserPromptSubmit', { matcher_context: text, message: text, working_dir: '/work/project' });

  output(await prompt('I prefer tabs over spaces.'), { empty: true });
  output(await run('Stop', { last_assistant_message: 'Noted: tabs.' }), { empty: true });
  output(await prompt('My dog is called Rex.'), { empty: true });
  output(await run('Stop', { last_assistant_message: 'Nice name.' }), { empty: true });
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  assert.equal(server.of('/memory/add')[0].body.session_id, 'goose:20261120_1');

  output(await prompt('What do I prefer for indentation?'), { empty: true });
  assert.match(fs.readFileSync(moim, 'utf8'), /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  assert.equal(fs.statSync(moim).mode & 0o077, 0, 'the context file is private');
  output(await run('Stop', { last_assistant_message: 'Tabs.' }), { empty: true });
  assert.equal(fs.readFileSync(moim, 'utf8'), '', 'cleared when the turn ends');

  output(await prompt('Please forget that my dog is called Rex.'), { empty: true });
  assert.match(fs.readFileSync(moim, 'utf8'), /^\[QuoteMem: automatic forget\]/);
  output(await run('SessionEnd', {}), { empty: true });
  assert.equal(fs.readFileSync(moim, 'utf8'), '');
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'remaining batches');
  const log = fs.existsSync(path.join(home, '.quotemem', 'goose', 'quotemem.log')) ? fs.readFileSync(path.join(home, '.quotemem', 'goose', 'quotemem.log'), 'utf8') : '';
  assert.doesNotMatch(log, /GOOSE_MOIM_MESSAGE_FILE/, 'goose reads the file, so no warning');
  assertNoLeaks(home, ['Rex', 'tabs']);
});

// --------------------------------------------------------------------- Devin

test('Devin before 3000.5.20 (no reply in Stop): a repeated hook records once; user turns are captured', async () => {
  const server = await memoryServer();
  const home = newHome();
  const hooks = config('devin/hooks.v1.json');
  const run = (event, fields) => sh(commandOf(hooks, event), { hook_event_name: event, session_id: 'd-1', cwd: '/work/project', ...fields }, env(server, home));
  output(await run('UserPromptSubmit', { prompt_id: 'p1', prompt: 'I prefer tabs over spaces.' }), { empty: true });
  output(await run('UserPromptSubmit', { prompt_id: 'p1', prompt: 'I prefer tabs over spaces.' }), { empty: true });
  output(await run('Stop', { prompt_id: 'p1', stop_hook_active: false }), { empty: true });
  const recall = output(await run('UserPromptSubmit', { prompt_id: 'p2', prompt: 'What do I prefer for indentation?' }));
  assert.equal(recall.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  assert.match(recall.hookSpecificOutput.additionalContext, /tabs over spaces/);
  output(await run('Stop', { prompt_id: 'p2', stop_hook_active: false }), { empty: true });
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  assert.equal(server.of('/memory/add')[0].body.session_id, 'devin:d-1');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => [m.role, m.content]), [['user', 'I prefer tabs over spaces.'], ['user', 'What do I prefer for indentation?']]);
  output(await run('SessionEnd', { reason: 'exit' }), { empty: true });
});

test('Devin: the reply from last_assistant_message, else a transcript, else none; never twice, never another turn\'s', async () => {
  const server = await memoryServer();
  const home = newHome();
  const hooks = config('devin/hooks.v1.json');
  const childEnv = env(server, home, { QUOTEMEM_CAPTURE_EVERY: '1', DEVIN_PROJECT_DIR: '/work/project' });
  const run = (event, fields) => sh(commandOf(hooks, event), { hook_event_name: event, session_id: 'd-2', ...fields }, childEnv);
  const turn = async (id, prompt, stop) => {
    output(await run('UserPromptSubmit', { prompt_id: id, prompt }), { empty: !/indentation/.test(prompt) });
    return run('Stop', { prompt_id: id, stop_hook_active: false, ...stop });
  };
  const transcripts = path.join(home, 'transcripts');
  fs.mkdirSync(transcripts);
  const write = (name, content) => {
    const file = path.join(transcripts, name);
    fs.writeFileSync(file, content);
    return file;
  };

  // 1. Devin 3000.5.20 and later: the reply is in the Stop payload.
  output(await turn('p1', 'I prefer tabs over spaces.', { last_assistant_message: 'Noted: tabs.' }), { empty: true });
  // A continuation Stop of the same turn adds nothing.
  output(await run('Stop', { prompt_id: 'p1', stop_hook_active: true, last_assistant_message: 'Noted again.' }), { empty: true });
  // 2. A recall turn: the injected block is not captured back.
  const recall = await turn('p2', 'What do I prefer for indentation?', { last_assistant_message: 'Tabs.' });
  output(recall, { empty: true });
  // 3. No reply field: a Claude Code JSONL transcript, with the reply's own time.
  // Transcript times after the turn's prompt (they must not be earlier).
  const jsonlAt = new Date(Date.now() + 60_000).toISOString();
  const jsonl = write('t.jsonl', [
    { type: 'user', message: { role: 'user', content: 'We picked Postgres for the shop.' } },
    { type: 'assistant', timestamp: jsonlAt, message: { role: 'assistant', content: [{ type: 'text', text: 'Postgres it is.' }] } },
  ].map((r) => JSON.stringify(r)).join('\n'));
  output(await turn('p3', 'We picked Postgres for the shop.', { last_assistant_message: '', transcript_path: jsonl }), { empty: true });
  // 4. An ATIF transcript.
  const atifAt = new Date(Date.now() + 120_000).toISOString();
  const atif = write('t.json', JSON.stringify({ schema_version: 'ATIF-v1.7', steps: [
    { step_id: 1, source: 'user', message: 'My cat is called Miso.' },
    { step_id: 2, source: 'agent', message: [{ type: 'text', text: 'Miso is a lovely name.' }], timestamp: atifAt },
  ] }));
  output(await turn('p4', 'My cat is called Miso.', { transcript_path: atif }), { empty: true });
  // 5. The transcript holds only an earlier turn's reply: none is taken.
  const stale = write('stale.jsonl', [
    { type: 'user', message: { content: 'Old question.' } },
    { type: 'assistant', message: { content: 'Old answer.' } },
    { type: 'user', message: { content: 'I work in Porto.' } },
  ].map((r) => JSON.stringify(r)).join('\n'));
  output(await turn('p5', 'I work in Porto.', { transcript_path: stale }), { empty: true });
  // 5b. The host has not written this turn yet: the file ends with turn 4's
  // reply. It is not taken as this turn's.
  const lagging = write('lagging.jsonl', [
    { type: 'user', message: { content: 'what is 2+2?' } },
    { type: 'assistant', message: { content: 'FOUR (turn 1)' } },
  ].map((r) => JSON.stringify(r)).join('\n'));
  output(await turn('p5b', 'Now add three and three.', { transcript_path: lagging }), { empty: true });
  // 5c. A reply time before the turn's prompt is not used; the hook's time is.
  const early = write('early.jsonl', [
    { type: 'user', message: { content: 'I have two cats.' } },
    { type: 'assistant', timestamp: '2020-01-01T00:00:00Z', message: { content: 'Two cats, noted.' } },
  ].map((r) => JSON.stringify(r)).join('\n'));
  const beforeEarly = Date.now();
  output(await turn('p5c', 'I have two cats.', { transcript_path: early }), { empty: true });
  // 6. A transcript that is a FIFO, or missing: user words only, at once.
  let fifo = path.join(transcripts, 'missing.jsonl');
  if (process.platform !== 'win32') {
    fifo = path.join(transcripts, 'fifo');
    execFileSync('mkfifo', [fifo]);
  }
  const blocked = await turn('p6', 'I ride a red bike.', { transcript_path: fifo });
  output(blocked, { empty: true });
  assert.ok(blocked.ms < 3000, `Stop took ${blocked.ms} ms`);
  // 7. A Stop of another turn does not attach its reply to this one.
  output(await turn('p7', 'I drink green tea.', { prompt_id: 'other', last_assistant_message: 'A reply from elsewhere.' }), { empty: true });

  output(await run('SessionEnd', { reason: 'exit' }), { empty: true });
  await waitFor(() => server.of('/memory/add').length >= 9, 8000, 'one batch per turn');
  await sleep(300);
  const adds = server.of('/memory/add').map((r) => r.body);
  assert.ok(adds.every((b) => b.session_id === 'devin:d-2'));
  const sent = adds.flatMap((b) => b.messages.map((m) => [m.role, m.content]));
  assert.deepEqual(sent, [
    ['user', 'I prefer tabs over spaces.'], ['assistant', 'Noted: tabs.'],
    ['user', 'What do I prefer for indentation?'], ['assistant', 'Tabs.'],
    ['user', 'We picked Postgres for the shop.'], ['assistant', 'Postgres it is.'],
    ['user', 'My cat is called Miso.'], ['assistant', 'Miso is a lovely name.'],
    ['user', 'I work in Porto.'],
    ['user', 'Now add three and three.'],
    ['user', 'I have two cats.'], ['assistant', 'Two cats, noted.'],
    ['user', 'I ride a red bike.'],
    ['user', 'I drink green tea.'],
  ]);
  const timeOf = (content) => adds.flatMap((b) => b.messages).find((m) => m.content === content).timestamp;
  assert.equal(timeOf('Postgres it is.'), Date.parse(jsonlAt), 'the transcript\'s time of the reply');
  assert.equal(timeOf('Miso is a lovely name.'), Date.parse(atifAt));
  assert.ok(timeOf('Two cats, noted.') >= beforeEarly, 'an earlier transcript time is not used');
  assert.ok(!JSON.stringify(adds).includes('[QuoteMem'), 'no recall block is captured back');
  assert.ok(nativeInstalled({ HOME: home, USERPROFILE: home }, 'devin'), 'the native marker is written');
  assertNoLeaks(home, ['Postgres', 'Miso', 'Porto', 'tabs']);
});

// ------------------------------------------------------- graceful degradation

test('every host carries on at once when QuoteMem is down, slow, or not configured', async () => {
  const down = await stubServer({});
  const downUrl = down.url;
  await down.close();
  const slow = await memoryServer({ '/memory/search': () => ({ delay: 4000, body: { data: [] } }) });
  servers.add(slow);
  const cases = [
    ['gemini', 'BeforeAgent', { session_id: 's', hook_event_name: 'BeforeAgent', prompt: 'Do you remember my dog?' }, '{}'],
    ['cursor', 'beforeSubmitPrompt', { conversation_id: 'c', generation_id: 'g', prompt: 'Do you remember my dog?' }, '{"continue":true}'],
    ['cursor', 'beforeSubmitPrompt', { conversation_id: 'c', generation_id: 'g', cursor_version: '3.24.9', prompt: 'Do you remember my dog?' }, '{"continue":true}'],
    ['devin', 'Stop', { hook_event_name: 'Stop', session_id: 's', prompt_id: 'p', last_assistant_message: 'Rex is a good dog.' }, ''],
    ['cline', 'UserPromptSubmit', { taskId: 't', hookName: 'UserPromptSubmit', userPromptSubmit: { prompt: 'Do you remember my dog?' } }, '{"cancel":false}'],
    ['goose', 'UserPromptSubmit', { event: 'UserPromptSubmit', session_id: 's', message: 'Do you remember my dog?' }, ''],
    ['devin', 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', session_id: 's', prompt_id: 'p', prompt: 'Do you remember my dog?' }, ''],
  ];
  // A memory question the service could not answer gets one short note on
  // hosts whose prompt hook adds context on stdout (Cursor from 3.20.17);
  // Goose's goes to its file, and an older Cursor's prompt hook never recalls.
  const NOTED = new Set(['gemini', 'cline', 'devin']);
  const carriesOn = (host, payload, out, expected, what) => {
    const noted = (NOTED.has(host) && /Prompt|Agent/.test(what)) || payload.cursor_version === '3.24.9';
    if (noted) {
      assert.match(out, /\[QuoteMem: memory unavailable\]/, what);
      if (host === 'cursor') assert.equal(JSON.parse(out).continue, true, what);
    } else {
      assert.equal(out, expected, what);
    }
  };
  for (const [host, event, payload, expected] of cases) {
    const home = newHome();
    const cmd = `quotemem-hook ${host} ${event}`;
    const offline = await sh(cmd, payload, env({ url: downUrl }, home));
    carriesOn(host, payload, offline.stdout.trim(), expected, `${host} ${event} offline`);
    assert.ok(offline.ms < 3000, `${host} offline took ${offline.ms} ms`);
    // Its own home: in the same one the same prompt again is a repeated hook.
    const lateHome = newHome();
    const late = await sh(cmd, payload, env(slow, lateHome, { QUOTEMEM_RECALL_TIMEOUT_MS: '300' }));
    carriesOn(host, payload, late.stdout.trim(), expected, `${host} ${event} slow`);
    assert.ok(late.ms < 3000, `${host} slow took ${late.ms} ms`);
    const unset = await sh(cmd, payload, { PATH: `${BIN}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home });
    assert.equal(unset.stdout.trim(), expected, `${host} without a token`);
    for (const result of [offline, late, unset]) assert.equal(result.code, 0);
    const problem = fs.readFileSync(path.join(home, '.quotemem', host === 'gemini' ? 'gemini-cli' : host, 'problem.txt'), 'utf8');
    assert.match(problem, /no token/);
    assertNoLeaks(home, ['dog']);
    assertNoLeaks(lateHome, ['dog']);
  }
  // Garbage on stdin, an unknown event: still the host's "carry on" answer.
  const home = newHome();
  assert.equal((await sh('quotemem-hook cursor beforeSubmitPrompt', undefined, env(slow, home))).stdout.trim(), '{"continue":true}');
  const junk = await new Promise((resolve) => {
    const child = spawn('/bin/sh', ['-c', 'quotemem-hook cline UserPromptSubmit'], { env: env(slow, home) });
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.on('exit', (code) => resolve({ code, out }));
    child.stdin.end('not json');
  });
  assert.deepEqual([junk.code, junk.out.trim()], [0, '{"cancel":false}']);
});

test('a forget that cannot finish in time tells the model it continues in the background', async () => {
  const server = await memoryServer({ '/memory/forget/prepare': (body) => ({ delay: 1500, body: plan(body) }) });
  const home = newHome();
  const result = output(await sh('quotemem-hook gemini BeforeAgent', { session_id: 'f', hook_event_name: 'BeforeAgent', prompt: 'Please forget that I live in Lisbon.' }, env(server, home, { QUOTEMEM_FORGET_TIMEOUT_MS: '300' })));
  assert.match(result.hookSpecificOutput.additionalContext, /in the background/);
  await waitFor(() => server.of('/memory/forget/confirm').length === 1, 8000, 'the background flusher finishes the forget');
});

// ------------------------------------------------------------------ OpenClaw

function fakeApi(pluginConfig = {}) {
  const handlers = {};
  const warnings = [];
  return {
    handlers,
    warnings,
    pluginConfig,
    logger: { warn: (m) => warnings.push(m), info() {}, debug() {} },
    on(name, handler, options) { handlers[name] = { handler, options }; },
  };
}

// The flushers that wait for idle chats under a state folder, by their locks
// (one per service now; one per owner before).
function idleWaiters(stateDir) {
  const locks = [];
  const add = (dir, pick) => {
    let names = [];
    try { names = fs.readdirSync(dir); } catch { return; }
    for (const name of names) {
      const file = pick(path.join(dir, name));
      if (fs.existsSync(file)) locks.push(file);
    }
  };
  add(path.join(stateDir, 'idle'), (file) => file);
  add(path.join(stateDir, 'owners'), (dir) => path.join(dir, 'idle.lock'));
  return locks.filter((file) => file.endsWith('.lock'));
}

test('OpenClaw userScope sender: one flusher waits for every sender\'s idle chat and saves each as its own user', async () => {
  const server = await memoryServer();
  const home = newHome();
  const mod = await import(pathToFileURL(path.join(INTEGRATIONS, 'openclaw', 'index.mjs')).href);
  const IDLE_MS = 1500;
  const group = fakeApi({ userScope: 'sender', captureEvery: 10, idleFlushMs: IDLE_MS });
  const childEnv = env(server, home);
  delete childEnv.QUOTEMEM_IDLE_FLUSH_MS;
  delete childEnv.QUOTEMEM_CAPTURE_EVERY;
  const registered = mod.registerQuoteMem(group, { env: childEnv });
  assert.ok(registered);
  const senders = ['111', '222', '333', '444'];
  for (const sender of senders) {
    const gctx = { sessionKey: 'agent:main:discord:group:9', runId: `g${sender}`, channel: 'discord', senderId: sender, inputProvenance: { kind: 'external_user' } };
    await group.handlers.before_prompt_build.handler({ currentUserMessage: `My lucky number is ${sender}.` }, gctx);
    await group.handlers.agent_end.handler({ runId: `g${sender}`, success: true, messages: [{ role: 'assistant', content: 'Noted.' }] }, gctx);
  }
  const { stateDir } = registered.settings;
  await waitFor(() => idleWaiters(stateDir).length > 0, 5000, 'a waiting flusher');
  assert.equal(idleWaiters(stateDir).length, 1, `one waiting flusher for ${senders.length} senders`);
  assert.equal(server.of('/memory/add').length, 0, 'nothing is saved before the idle time');
  await waitFor(() => server.of('/memory/add').length === senders.length, IDLE_MS + 8000, 'every sender\'s idle save');
  const adds = server.of('/memory/add').map((r) => r.body);
  assert.deepEqual(adds.map((b) => b.user_id).sort(), senders.map((s) => `discord:${s}`));
  for (const body of adds) {
    assert.deepEqual(body.messages.filter((m) => m.role === 'user').map((m) => m.content), [`My lucky number is ${body.user_id.slice(8)}.`]);
  }
  await waitFor(() => idleWaiters(stateDir).length === 0, 8000, 'the waiting flusher to exit');
  group.handlers.gateway_stop.handler({}, {});
});

test('OpenClaw plugin: prependContext recall, capture from agent_end, forget, provenance and per-sender memory', async () => {
  const server = await memoryServer();
  const home = newHome();
  const mod = await import(pathToFileURL(path.join(INTEGRATIONS, 'openclaw', 'index.mjs')).href);
  const manifest = config('openclaw/openclaw.plugin.json');
  assert.equal(mod.default.id, manifest.id);
  assert.equal(typeof mod.default.register, 'function');
  const pkg = config('openclaw/package.json');
  for (const entry of pkg.openclaw.extensions) assert.ok(fs.existsSync(path.join(INTEGRATIONS, 'openclaw', entry)));

  const api = fakeApi({ captureEvery: 2 });
  const childEnv = env(server, home);
  delete childEnv.QUOTEMEM_CAPTURE_EVERY;
  assert.ok(mod.registerQuoteMem(api, { env: childEnv }));
  assert.deepEqual(Object.keys(api.handlers).sort(), ['agent_end', 'before_prompt_build', 'gateway_stop', 'session_end']);
  // Room for a forget's wait and a recall search, both bounded well below it.
  assert.ok(api.handlers.before_prompt_build.options.timeoutMs >= 6000);
  const ctx = (runId, extra = {}) => ({ sessionKey: 'agent:main:main', sessionId: 'sid-1', runId, trigger: 'user', inputProvenance: { kind: 'external_user' }, ...extra });
  const prompt = (runId, text, extra) => api.handlers.before_prompt_build.handler(
    { prompt: `[history]\n${text}`, currentUserMessage: text, currentUserMessageId: `m-${runId}`, messages: [{ role: 'user', content: text }] },
    ctx(runId, extra),
  );
  const end = (runId, reply, success = true) => api.handlers.agent_end.handler(
    { runId, success, durationMs: 5, messages: [{ role: 'user', content: 'x' }, { role: 'assistant', content: [{ type: 'thinking', thinking: 'secret plan' }, { type: 'text', text: reply }] }] },
    ctx(runId),
  );

  assert.equal(await prompt('r1', 'I prefer tabs over spaces.'), undefined);
  await end('r1', 'Noted: tabs.');
  assert.equal(await prompt('r2', 'My dog is called Rex.'), undefined);
  await end('r2', 'Nice name.');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the first batch');
  const batch = server.of('/memory/add')[0].body;
  assert.equal(batch.session_id, 'openclaw:agent:main:main');
  assert.deepEqual(batch.messages.map((m) => m.content), ['I prefer tabs over spaces.', 'Noted: tabs.', 'My dog is called Rex.', 'Nice name.']);

  const recall = await prompt('r3', 'What do I prefer for indentation?');
  assert.match(recall.prependContext, /^\[QuoteMem: retrieved long-term memory\][\s\S]*tabs over spaces/);
  await end('r3', 'Tabs.');
  // Heartbeat and system turns are not remembered or searched.
  const searches = server.of('/memory/search').length;
  assert.equal(await prompt('r4', 'What do I prefer for lunch?', { inputProvenance: { kind: 'internal_system' } }), undefined);
  assert.equal(await prompt('r5', 'What do I prefer for lunch?', { trigger: 'heartbeat' }), undefined);
  assert.equal(server.of('/memory/search').length, searches);

  const forget = await prompt('r6', 'Please forget that my dog is called Rex.');
  assert.ok(forget.prependContext.startsWith(FORGET_HEADER));
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  await end('r6', 'Done.');
  await api.handlers.session_end.handler({ sessionId: 'sid-1', sessionKey: 'agent:main:main', reason: 'reset' }, { sessionKey: 'agent:main:main' });
  api.handlers.gateway_stop.handler({}, {});
  await waitFor(() => server.of('/memory/add').length === 3, 8000, 'remaining batches');
  assert.ok(!JSON.stringify(server.of('/memory/add').map((r) => r.body)).includes('secret plan'), 'reasoning is never captured');

  // userScope sender: each sender of a group chat has its own memory.
  const group = fakeApi({ userScope: 'sender', captureEvery: 1 });
  mod.registerQuoteMem(group, { env: childEnv });
  for (const sender of ['111', '222']) {
    const gctx = { sessionKey: 'agent:main:discord:group:9', runId: `g${sender}`, channel: 'discord', senderId: sender, inputProvenance: { kind: 'external_user' } };
    await group.handlers.before_prompt_build.handler({ currentUserMessage: `I am sender ${sender}.` }, gctx);
    await group.handlers.agent_end.handler({ runId: `g${sender}`, success: true, messages: [{ role: 'assistant', content: 'Hi.' }] }, gctx);
  }
  await waitFor(() => server.of('/memory/add').length === 5, 8000, 'one batch per sender');
  const groupAdds = server.of('/memory/add').slice(3).map((r) => r.body);
  assert.deepEqual(groupAdds.map((b) => b.user_id).sort(), ['discord:111', 'discord:222']);
  assert.notEqual(groupAdds[0].session_id, groupAdds[1].session_id);

  // userScope owner: a channel sender who is not the owner gets no memory:
  // no recall of the owner's memories, no forget, no capture.
  const shared = fakeApi({ captureEvery: 1, ownerSenders: ['telegram:42'] });
  mod.registerQuoteMem(shared, { env: childEnv });
  const before = { search: server.of('/memory/search').length, prepare: server.of('/memory/forget/prepare').length, add: server.of('/memory/add').length };
  const stranger = { sessionKey: 'agent:main:telegram:group:7', runId: 's1', channel: 'telegram', senderId: '99', inputProvenance: { kind: 'external_user' } };
  assert.equal(await shared.handlers.before_prompt_build.handler({ currentUserMessage: 'What do you remember about me? What do I prefer?' }, stranger), undefined);
  assert.equal(await shared.handlers.before_prompt_build.handler({ currentUserMessage: 'Please forget that my dog is called Rex.' }, { ...stranger, runId: 's2' }), undefined);
  await shared.handlers.agent_end.handler({ runId: 's2', success: true, messages: [{ role: 'assistant', content: 'Done.' }] }, { ...stranger, runId: 's2' });
  await shared.handlers.session_end.handler({ sessionKey: stranger.sessionKey }, { sessionKey: stranger.sessionKey });
  assert.deepEqual({ search: server.of('/memory/search').length, prepare: server.of('/memory/forget/prepare').length, add: server.of('/memory/add').length }, before);
  assert.match(shared.warnings.join('\n'), /no memory for channel sender telegram:99/);
  // The listed owner is served as usual.
  const owner = { ...stranger, runId: 'o1', senderId: '42' };
  const ownerRecall = await shared.handlers.before_prompt_build.handler({ currentUserMessage: 'What do I prefer for indentation?' }, owner);
  assert.match(ownerRecall.prependContext, /tabs over spaces/);

  // Not configured: nothing is registered, one warning, no throw.
  const off = fakeApi();
  assert.equal(mod.registerQuoteMem(off, { env: { HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url } }), null);
  assert.deepEqual(Object.keys(off.handlers), []);
  assert.match(off.warnings.join('\n'), /automatic memory is off: no token/);
  assert.ok(!off.warnings.join('\n').includes(TOKEN));
  assertNoLeaks(home, ['Rex', 'tabs']);
});

// ----------------------------------------------------------------------- MCP

function mcpSession(command, args, childEnv) {
  const child = spawn(command, args, { env: childEnv, stdio: ['pipe', 'pipe', 'pipe'] });
  let buffer = '';
  const waiting = new Map();
  child.stdout.on('data', (d) => {
    buffer += d;
    let i;
    while ((i = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, i);
      buffer = buffer.slice(i + 1);
      if (!line.trim()) continue;
      const message = JSON.parse(line);
      waiting.get(message.id)?.(message);
    }
  });
  let seq = 0;
  return {
    request(method, params) {
      seq += 1;
      const id = seq;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`no answer to ${method}`)), 8000);
        waiting.set(id, (message) => { clearTimeout(timer); resolve(message); });
        child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
      });
    },
    notify(method, params) { child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`); },
    close() { child.stdin.end(); child.kill(); },
  };
}

test('MCP: `quotemem-hook mcp` and the bundled launchers serve the tools with the env-file token; quotemem_add is hidden', async () => {
  const server = await memoryServer({ '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'User prefers tabs over spaces', { score: 0.9 })] } }) });
  const home = newHome();
  fs.mkdirSync(path.join(home, '.quotemem'), { mode: 0o700 });
  fs.writeFileSync(path.join(home, '.quotemem', 'env'), `QUOTEMEM_TOKEN=${TOKEN}\nQUOTEMEM_URL=${server.url}\nQUOTEMEM_USER_ID=quei\n`, { mode: 0o600 });
  const childEnv = { PATH: `${BIN}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home };
  const launchers = [
    ['quotemem-hook', ['mcp']],
    [process.execPath, [path.join(INTEGRATIONS, 'gemini-cli', 'mcp.mjs')]],
    [process.execPath, [path.join(INTEGRATIONS, 'openclaw', 'mcp.mjs')]],
  ];
  for (const [command, args] of launchers) {
    const session = mcpSession(command, args, childEnv);
    try {
      const init = await session.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
      assert.equal(init.result.serverInfo.name, 'quotemem');
      session.notify('notifications/initialized', {});
      const names = (await session.request('tools/list', {})).result.tools.map((t) => t.name);
      assert.ok(names.includes('quotemem_search') && names.includes('quotemem_forget_prepare'), names.join(','));
      assert.ok(!names.includes('quotemem_add'), 'capture is automatic, so quotemem_add is hidden');
      const before = server.of('/memory/search').length;
      const call = await session.request('tools/call', { name: 'quotemem_search', arguments: { query: 'indentation' } });
      assert.ok(!call.result.isError, JSON.stringify(call.result));
      assert.equal(server.of('/memory/search').length, before + 1);
      assert.equal(server.of('/memory/search').at(-1).headers.authorization, `Bearer ${TOKEN}`);
      assert.ok(!JSON.stringify(call).includes(TOKEN));
    } finally {
      session.close();
    }
  }
});

// ------------------------------------------------------------ units and files

test('env file and token sources: the environment wins, only QUOTEMEM_* names are read', () => {
  assert.deepEqual(parseEnvFile('# c\nexport QUOTEMEM_URL="http://x"\nQUOTEMEM_TOKEN=\'t\'\nPATH=/evil\nquotemem_x=1\n'), { QUOTEMEM_URL: 'http://x', QUOTEMEM_TOKEN: 't' });
  const home = newHome();
  fs.mkdirSync(path.join(home, '.quotemem'));
  fs.writeFileSync(path.join(home, '.quotemem', 'env'), 'QUOTEMEM_URL=http://file\nQUOTEMEM_TOKEN=from-file\n', { mode: 0o644 });
  const { env: merged, notes } = hostEnv({ HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'http://env' });
  assert.equal(merged.QUOTEMEM_URL, 'http://env');
  assert.equal(merged.QUOTEMEM_TOKEN, 'from-file');
  assert.match(notes.join(' '), /chmod 600/);
  assert.ok(!notes.join(' ').includes('from-file'));
  assert.equal(hostEnv({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN: 'e', GEMINI_CLI_QUOTEMEM_TOKEN: 'g' }).env.QUOTEMEM_TOKEN, 'e');
});

test('payload mapping and fallbacks per host', () => {
  assert.equal(normalize('gemini', { session_id: 's', hook_event_name: 'AfterAgent', prompt_response: 'r', stop_hook_active: true }).action, 'ignore');
  assert.equal(normalize('cursor', { conversation_id: 'c', hook_event_name: 'stop', status: 'error' }).completed, false);
  assert.equal(normalize('cline', { taskId: 't', hookName: 'TaskComplete', taskComplete: { taskMetadata: { result: 'ok' } } }).text, 'ok');
  assert.equal(normalize('goose', { event: 'Stop', session_id: 's' }).text, null);
  assert.equal(normalize('devin', { hook_event_name: 'UserPromptSubmit', prompt_id: 'p' }).action, 'ignore', 'no session');
  // Devin's Stop: the reply text, else a transcript path to read it from.
  const devinStop = (fields) => normalize('devin', { hook_event_name: 'Stop', session_id: 's', prompt_id: 'p', stop_hook_active: false, ...fields });
  assert.deepEqual(devinStop({ last_assistant_message: 'Done.' }), { action: 'reply', event: 'Stop', session: 's', turn: 'p', text: 'Done.', completed: true });
  assert.equal(devinStop({}).text, null);
  for (const bad of ['', '   ', 42, null, ['x'], { text: 'x' }]) assert.equal(devinStop({ last_assistant_message: bad }).text, null, JSON.stringify(bad));
  assert.equal(devinStop({ last_assistant_message: '', transcript_path: '/t/a.jsonl' }).transcript, '/t/a.jsonl');
  assert.equal(devinStop({ last_assistant_message: 'Done.', transcript_path: '/t/a.jsonl' }).transcript, undefined, 'the reply field wins');
  for (const bad of [42, '', 'x'.repeat(5000), {}]) assert.equal(devinStop({ transcript_path: bad }).transcript, undefined, String(bad).slice(0, 20));
  assert.equal(devinStop({ stop_hook_active: true, last_assistant_message: 'Again.' }).action, 'ignore');
  assert.equal(normalize('devin', { hook_event_name: 'Stop', prompt_id: 'p', last_assistant_message: 'x' }).action, 'ignore', 'no session');
  // Cursor actions carry the version; the prompt answer carries context only when given.
  assert.equal(normalize('cursor', { conversation_id: 'c', hook_event_name: 'beforeSubmitPrompt', prompt: 'p', cursor_version: '3.24.9' }).version, '3.24.9');
  assert.equal(normalize('cursor', { conversation_id: 'c', hook_event_name: 'beforeSubmitPrompt', prompt: 'p' }).version, null);
  // sessionEnd keeps the transcript path only for the chat's own file (<conversation id>.jsonl).
  const end = (transcript) => normalize('cursor', { conversation_id: 'c-9', session_id: 'c-9', hook_event_name: 'sessionEnd', transcript_path: transcript }).transcript;
  assert.equal(end('C:\\Users\\me\\.cursor\\projects\\p\\agent-transcripts\\c-9\\c-9.jsonl'), 'C:\\Users\\me\\.cursor\\projects\\p\\agent-transcripts\\c-9\\c-9.jsonl');
  assert.equal(end('/home/me/.cursor/projects/p/agent-transcripts/c-9/c-9.jsonl'), '/home/me/.cursor/projects/p/agent-transcripts/c-9/c-9.jsonl');
  for (const other of [null, '', '/tmp/c-8.jsonl', '/tmp/c-9.json', '/tmp/xc-9.jsonl', 42]) assert.equal(end(other), undefined, String(other));
  assert.equal(render('cursor', { event: 'beforeSubmitPrompt' }, '[QuoteMem: x] y'), '{"continue":true,"additional_context":"[QuoteMem: x] y"}');
  assert.equal(render('cursor', { event: 'beforeSubmitPrompt' }, null), '{"continue":true}');
  assert.equal(render('cursor', { event: 'sessionStart' }, 'd'), '{"additional_context":"d"}');
  assert.equal(render('cursor', { event: 'stop' }, 'x'), '{}');
  assert.equal(normalize('gemini', null).action, 'ignore');
  // The host's own time of a message, when its payload gives one.
  const at = Date.parse('2026-11-20T10:00:00Z');
  assert.equal(normalize('gemini', { session_id: 's', hook_event_name: 'BeforeAgent', prompt: 'p', timestamp: '2026-11-20T10:00:00Z' }).time, at);
  assert.equal(normalize('cline', { taskId: 't', hookName: 'UserPromptSubmit', timestamp: String(at), userPromptSubmit: { prompt: 'p' } }).time, at);
  assert.equal(normalize('cursor', { conversation_id: 'c', hook_event_name: 'afterAgentResponse', text: 'r', timestamp: at / 1000 }).time, at, 'seconds');
  assert.equal(normalize('devin', { hook_event_name: 'UserPromptSubmit', session_id: 's', prompt: 'p' }).time, undefined, 'none given: the hook time applies');
  for (const bad of ['yesterday', '', -1, '1999-12-31T00:00:00Z', Date.now() + 3 * 86_400_000, null, {}]) assert.equal(hostTime(bad), null, String(bad));
  assert.equal(normalize('gemini', { session_id: 's', hook_event_name: 'SessionEnd', timestamp: '2026-11-20T10:00:00Z' }).time, undefined, 'only messages carry a time');
  assert.equal(fallback('cursor', 'beforeSubmitPrompt'), '{"continue":true}');
  assert.equal(fallback('cline', '', { hookName: 'TaskComplete' }), '{"cancel":false}');
  assert.equal(fallback('goose', 'Stop'), '');
});

test('shipped install files are well formed and point at real files and events', () => {
  const gemini = config('gemini-cli/gemini-extension.json');
  assert.equal(gemini.name, 'quotemem');
  assert.ok(fs.existsSync(path.join(INTEGRATIONS, 'gemini-cli', gemini.contextFileName)));
  assert.ok(gemini.settings.some((s) => s.envVar === 'QUOTEMEM_TOKEN' && s.sensitive === true));
  const geminiHooks = config('gemini-cli/hooks/hooks.json').hooks;
  assert.deepEqual(Object.keys(geminiHooks).sort(), ['AfterAgent', 'BeforeAgent', 'SessionEnd']);
  for (const [event, groups] of Object.entries(geminiHooks)) {
    const hook = groups[0].hooks[0];
    assert.ok(hook.command.endsWith(` gemini ${event}`));
    assert.ok(hook.timeout >= 5000 && hook.timeout <= 60000, 'milliseconds');
  }
  assert.ok(config('gemini-cli/hooks/hooks.json').hooks.BeforeAgent[0].hooks[0].timeout > 11_500, 'longer than the hook budget');
  const cursor = config('cursor/hooks.json');
  assert.deepEqual(Object.keys(cursor.hooks).sort(), ['afterAgentResponse', 'beforeSubmitPrompt', 'sessionEnd', 'sessionStart', 'stop']);
  assert.equal(config('cursor/mcp.json').mcpServers.quotemem.args[0], 'mcp');
  assert.match(fs.readFileSync(path.join(INTEGRATIONS, 'cursor', 'rules', 'quotemem.mdc'), 'utf8'), /^---\n[\s\S]*alwaysApply: true/);
  for (const name of ['UserPromptSubmit', 'TaskComplete', 'TaskCancel']) {
    const file = path.join(INTEGRATIONS, 'cline', 'hooks', name);
    assert.ok(fs.statSync(file).mode & 0o111, `${name} is executable`);
    assert.match(fs.readFileSync(file, 'utf8'), new RegExp(`cline ${name}\\n`));
  }
  assert.ok(fs.statSync(path.join(INTEGRATIONS, 'goose', 'quotemem', 'scripts', 'quotemem-hook.sh')).mode & 0o111);
  assert.equal(config('goose/quotemem/plugin.json').name, 'quotemem');
  assert.deepEqual(Object.keys(config('devin/hooks.v1.json')).sort(), ['SessionEnd', 'Stop', 'UserPromptSubmit']);
  assert.ok(fs.statSync(path.join(HOOKS, 'quotemem-hook.mjs')).mode & 0o111, 'the bin is executable');
});

test('model instructions: a forget request without a note is carried out through the tools, and a forget is claimed only when a note or a tool says so', () => {
  const read = (rel) => fs.readFileSync(path.join(INTEGRATIONS, rel), 'utf8');
  // Every text that tells a model how QuoteMem forgets: none promises a
  // forget that no hook or tool reported, and each names the tools.
  const texts = ['cursor/rules/quotemem.mdc', 'gemini-cli/GEMINI.md', 'dsh/skills/quotemem/SKILL.md', 'continue/quotemem-rule.md', 'codex/AGENTS.snippet.md'];
  for (const rel of texts) {
    const text = read(rel);
    assert.match(text, /quotemem_forget_prepare[\s\S]*quotemem_forget_confirm/, `${rel}: the tools`);
    assert.doesNotMatch(text, /without a note, confirm|carried out automatically before you see/i, `${rel}: no promise without a note`);
  }
  // Where a hook can be missing (Cursor's CLI in print mode runs no prompt hook), the model forgets itself.
  const cursor = read('cursor/rules/quotemem.mdc');
  assert.match(cursor, /If no such note came \(Cursor's CLI in print mode never sends one\),\s+forget it yourself: call `quotemem_forget_prepare`[\s\S]*then\s+`quotemem_forget_confirm` with the token/);
  assert.match(cursor, /Never say that something was forgotten unless that note or a forget\s+tool's result says so/);
  const gemini = read('gemini-cli/GEMINI.md');
  assert.match(gemini, /no such note came with the\s+message \(the extension's hooks did not run\), forget it yourself/);
  assert.match(gemini, /Never say that something was forgotten unless that\s+note or a forget tool's result says so/);
  assert.match(read('dsh/skills/quotemem/SKILL.md'), /Never say that something\s+was forgotten unless an automatic-forget note or the confirm result says so/);
  // These hosts run their MCP server next to an adapter (QUOTEMEM_AUTO=1): the
  // request to forget is the agreement, as the server's tool says.
  for (const rel of ['cursor/rules/quotemem.mdc', 'gemini-cli/GEMINI.md', 'dsh/skills/quotemem/SKILL.md']) {
    assert.match(read(rel), /request\s+(?:to forget\s+)?is\s+their\s+agreement/, rel);
    assert.doesNotMatch(read(rel), /get a clear yes|only after the user agrees|after they agree/, rel);
  }
  // Cursor's CLI print mode saves a run only at its end: the model is told the forget is applied to it again then.
  assert.match(cursor, /In Cursor's CLI\s+print mode, the words of this run are saved only when the run ends; a\s+forget you ask for is applied to them again then/);
});

test('bundled copies match their sources', () => {
  execFileSync(process.execPath, [path.join(HOOKS, 'bundle.mjs'), '--check'], { stdio: 'pipe' });
  execFileSync(process.execPath, [path.join(INTEGRATIONS, 'core', 'sync.mjs'), '--check'], { stdio: 'pipe' });
});

await sleep(0);
