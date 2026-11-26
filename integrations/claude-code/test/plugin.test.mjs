#!/usr/bin/env node
// The Claude Code plugin end to end, without Claude Code: every hook runs as
// Claude Code runs it (`node scripts/hook.mjs <Event>` in its own process,
// the event's JSON on stdin, the plugin environment set), against a stub
// QuoteMem service. The inputs follow the documented hook input shapes.
// HOME, the plugin data folder and every state file live in a temp folder.
//
//   node --test integrations/claude-code/test/plugin.test.mjs
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { HIT, TOKEN, plan, sleep, stubServer, tempDir, treeTexts, waitFor } from '../../core/test/helpers.mjs';

const PLUGIN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const HOOK = path.join(PLUGIN, 'scripts', 'hook.mjs');
const MCP = path.join(PLUGIN, 'scripts', 'mcp.mjs');
const pluginCore = await import(pathToFileURL(path.join(PLUGIN, 'core', 'index.mjs')).href);
const { FORGET_HEADER, RECALL_HEADER } = pluginCore;
const { cleanPrompt, isHostInjected, lastAssistantText } = await import(pathToFileURL(path.join(PLUGIN, 'scripts', 'adapter.mjs')).href);

const root = tempDir('qm-claude-code-');
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });
async function serve(handlers) {
  const server = await stubServer(handlers);
  servers.add(server);
  return server;
}

const ADDED = () => ({ body: { success: true, outcome: { added: 1 } } });
const verified = (n = 1) => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: Array.from({ length: n }, (_, i) => ({ memory_id: `01JX0000000000000000000F${i}`, outcome: 'verified' })), scope_state: 'ready' } });

let seq = 0;
/** A Claude Code install in a temp HOME: plugin options, data folder, a project and a transcript. */
function install(server, { options = {}, env = {} } = {}) {
  seq += 1;
  const home = path.join(root, `home-${seq}`);
  const data = path.join(home, '.claude', 'plugins', 'data', 'quotemem-quotemem');
  const project = path.join(home, 'project');
  fs.mkdirSync(project, { recursive: true });
  const base = {
    PATH: process.env.PATH,
    HOME: home, USERPROFILE: home,
    CLAUDE_PROJECT_DIR: project,
    CLAUDE_PLUGIN_ROOT: PLUGIN,
    CLAUDE_PLUGIN_DATA: data,
    CLAUDE_PLUGIN_OPTION_URL: server ? server.url : 'http://127.0.0.1:9',
    CLAUDE_PLUGIN_OPTION_TOKEN: TOKEN,
    CLAUDE_PLUGIN_OPTION_USER_ID: 'quei',
    // No flusher waits for idle chats beyond a test, unless the test sets a time.
    QUOTEMEM_IDLE_FLUSH_MS: '0',
    ...options,
    ...env,
  };
  for (const [key, value] of Object.entries(base)) if (value === undefined) delete base[key];
  const stateDir = path.join(data, 'state');
  const transcript = (session) => path.join(home, '.claude', 'projects', '-project', `${session}.jsonl`);

  /** Run one hook the way Claude Code does: exec form, input JSON on stdin. */
  const hook = (event, input, raw = null) => new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn('node', [HOOK, event], { env: base, cwd: project, stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('exit', (code) => resolve({ code, out, err, ms: Date.now() - started }));
    child.stdin.end(raw ?? JSON.stringify({ hook_event_name: event, ...input }));
  });
  const common = (session, extra = {}) => ({
    session_id: session,
    transcript_path: transcript(session),
    cwd: project,
    permission_mode: 'default',
    scratchpad_dir: path.join(root, 'scratch'),
    ...extra,
  });
  const start = (session, source = 'startup') => hook('SessionStart', { ...common(session), source, model: 'claude-opus-4' });
  const prompt = (session, promptId, text, extra = {}) => hook('UserPromptSubmit', { ...common(session, { prompt_id: promptId }), prompt: text, ...extra });
  const stop = (session, promptId, reply, extra = {}) => hook('Stop', {
    ...common(session, { prompt_id: promptId }),
    stop_hook_active: false,
    last_assistant_message: reply,
    background_tasks: [],
    session_crons: [],
    ...extra,
  });
  const turn = async (session, n, user, reply) => {
    const p = await prompt(session, `prompt-${n}`, user);
    const s = await stop(session, `prompt-${n}`, reply);
    return { prompt: p, stop: s };
  };
  const end = (session, reason = 'prompt_input_exit') => hook('SessionEnd', { ...common(session), reason });
  // Every file under the plugin data folder (background processes may still change it).
  const logText = () => treeTexts(data).map(({ text }) => text).join('\n');
  return { home, data, stateDir, env: base, hook, start, prompt, stop, turn, end, transcript, logText };
}

const context = (result) => {
  if (!result.out) return null;
  const parsed = JSON.parse(result.out);
  assert.equal(parsed.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  return parsed.hookSpecificOutput.additionalContext;
};

test('capture: one /memory/add per 3 user turns, sent in the background, hooks print nothing', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 300, ...ADDED() }) });
  const cc = install(server);
  const startResult = await cc.start('sess-a');
  assert.deepEqual([startResult.code, startResult.out, startResult.err], [0, '', '']);
  const results = [];
  results.push(await cc.turn('sess-a', 1, 'I moved to Lisbon last spring.', 'Noted, Lisbon.'));
  results.push(await cc.turn('sess-a', 2, 'My dog is called Biscuit.', 'Nice name.'));
  results.push(await cc.turn('sess-a', 3, 'I work on a Rust CLI called tidy.', 'Sounds good.'));
  for (const r of results) {
    for (const x of [r.prompt, r.stop]) assert.deepEqual([x.code, x.out, x.err], [0, '', '']);
  }
  // The /memory/add takes 300 ms; the third Stop hook did not wait for it.
  assert.ok(results[2].stop.ms < 1500, `Stop hook took ${results[2].stop.ms} ms`);
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the batch');
  const add = server.of('/memory/add')[0];
  assert.equal(add.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(add.body.user_id, 'quei');
  assert.equal(add.body.session_id, 'claude-code:sess-a');
  assert.deepEqual(add.body.messages.map((m) => [m.role, m.content]), [
    ['user', 'I moved to Lisbon last spring.'], ['assistant', 'Noted, Lisbon.'],
    ['user', 'My dog is called Biscuit.'], ['assistant', 'Nice name.'],
    ['user', 'I work on a Rust CLI called tidy.'], ['assistant', 'Sounds good.'],
  ]);
  assert.match(add.body.request_id, /\S/);
  // The fourth turn starts a new batch; nothing more is sent yet.
  await cc.turn('sess-a', 4, 'SQLite works well for storage.', 'OK.');
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 1);
  assert.equal(server.of('/memory/search').length, 0, 'no recall on plain statements');
});

test('recall: a memory question gets the quoted memories as hidden context; statements do not search', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'I moved to Lisbon last spring.', { score: 0.9 }), HIT('01JX0000000000000000000002', 'weak', { score: 0.05 })] } }),
  });
  const cc = install(server);
  const statement = await cc.prompt('sess-r', 'p1', 'I prefer Rust for command-line tools.');
  assert.deepEqual([statement.code, statement.out], [0, '']);
  await cc.stop('sess-r', 'p1', 'Rust is a good fit.');
  assert.equal(server.of('/memory/search').length, 0);

  const question = await cc.prompt('sess-r', 'p2', 'Where did I say I live now?');
  assert.equal(question.code, 0);
  const ctx = context(question);
  assert.ok(ctx.startsWith(RECALL_HEADER));
  assert.match(ctx, /I moved to Lisbon last spring\./);
  assert.match(ctx, /weak/, 'the service applies its relevance floor; the plugin adds none');
  assert.ok(ctx.length <= 9000);
  const search = server.of('/memory/search')[0];
  assert.equal(search.body.user_id, 'quei');
  assert.equal(search.body.query, 'Where did I say I live now?');
  assert.equal(search.headers.authorization, `Bearer ${TOKEN}`);

  // The same memory is not injected again a turn later.
  await cc.stop('sess-r', 'p2', 'Lisbon.');
  const again = await cc.prompt('sess-r', 'p3', 'Remind me where I live?');
  assert.equal(again.out, '');
});

const CAT_ID = '01JX0000000000000000000CA1';
test('recall: a forget still running after the 8 s wait leaves the turn without memory', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/search': () => ({ body: { data: [HIT(CAT_ID, 'The user\'s cat is called Miso.')] } }),
  });
  const cc = install(server);
  // A forget of this user is queued and no background sender runs it, so it is
  // still running when the prompt's wait (the default, FORGET_WAIT_MS) is over.
  const quiet = { warn() {}, warnOnce() {}, debug() {} };
  const owner = { stateDir: cc.stateDir, url: server.url, userId: 'quei' };
  const queue = new pluginCore.DiskQueue(owner, quiet);
  const lock = pluginCore.tryLock(queue.lockFile);
  assert.ok(lock);
  try {
    const body = { user_id: 'quei', request_id: 'f-slow' };
    queue.enqueue([{ kind: 'forget', request_id: 'f-slow', body, url: server.url,
      created: Date.now(), attempts: 0, nextAt: 0 }]);
    const held = await cc.prompt('sess-h', 'p1', "What is my cat's name?");
    assert.equal(held.code, 0);
    assert.equal(held.out, '', 'nothing is injected: no memory and no note about the forget');
    assert.equal(server.of('/memory/search').length, 0, 'no search while the forget runs');
    assert.ok(held.ms >= pluginCore.FORGET_WAIT_MS, `waited for the forget first (${held.ms} ms)`);
    assert.ok(held.ms < 30_000 - 2000, `within the UserPromptSubmit timeout (${held.ms} ms)`);
  } finally {
    pluginCore.releaseLock(lock);
  }
});

test('recall: QUOTEMEM_MAX_INJECT_CHARS cannot push the context past the 10,000-character hook cap', async () => {
  const big = (i) => HIT(`01JX00000000000000000000B${i}`, `${'x'.repeat(2500)} fact ${i}`, { score: 0.9 });
  const server = await serve({ '/memory/search': () => ({ body: { data: [big(1), big(2), big(3), big(4), big(5)] } }) });
  const cc = install(server, { env: { QUOTEMEM_MAX_INJECT_CHARS: '50000' } });
  const ctx = context(await cc.prompt('sess-big', 'p1', 'What do you remember about my projects?'));
  assert.ok(ctx.length <= 9000, `context is ${ctx.length} characters`);
  assert.match(ctx, /<\/quotemem-recall>$/, 'whole hits only');
});

test('forget: silent; earlier turns are stored first, the model gets one note naming no tool', async () => {
  const order = [];
  const server = await serve({
    '/memory/add': (body) => { order.push(`add:${body.messages.length}`); return { delay: 200, ...ADDED() }; },
    '/memory/forget/prepare': (body) => { order.push('prepare'); return { body: plan(body) }; },
    '/memory/forget/confirm': () => { order.push('confirm'); return verified(1); },
  });
  const cc = install(server);
  await cc.turn('sess-f', 1, 'I live in Lisbon.', 'Noted.');
  await cc.turn('sess-f', 2, 'My sister is called Ana.', 'Nice.');
  const forget = await cc.prompt('sess-f', 'prompt-3', 'Please forget that my sister is called Ana.');
  assert.equal(forget.code, 0);
  const note = context(forget);
  assert.ok(note.startsWith(FORGET_HEADER), note);
  assert.match(note, /Done: 1 stored memory was forgotten/);
  assert.match(note, /my sister is called Ana/);
  assert.doesNotMatch(note, /quotemem_|01JX|request_id/, 'no tool names or ids');
  assert.deepEqual(order.slice(0, 3), ['add:4', 'prepare', 'confirm'], 'the pending turns were stored before the forget');
  const prepare = server.of('/memory/forget/prepare')[0].body;
  assert.equal(prepare.user_id, 'quei');
  assert.equal(prepare.client_session_id, 'claude-code:sess-f');
  assert.equal(prepare.statement, 'my sister is called Ana');
  assert.equal(server.of('/memory/search').length, 0, 'no recall on a forget turn');

  // Claude Code may run the hook twice for one prompt (a plugin plus a manual
  // install): the same prompt id replays the same forget, nothing new is sent.
  const repeat = await cc.prompt('sess-f', 'prompt-3', 'Please forget that my sister is called Ana.');
  assert.equal(repeat.code, 0);
  await sleep(300);
  assert.equal(server.of('/memory/forget/confirm').length, 1);
  await cc.stop('sess-f', 'prompt-3', 'Done, I forgot that.');
});

test('forget: nothing matched is never reported as done; Chinese requests work', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/forget/prepare': () => ({ body: { status: 'nothing_matched', candidates: [] } }),
  });
  const cc = install(server);
  const note = context(await cc.prompt('sess-zh', 'p1', '请忘掉我住在里斯本这件事'));
  assert.ok(note.startsWith(FORGET_HEADER));
  assert.match(note, /so nothing was forgotten/);
  assert.doesNotMatch(note, /Done/);
  assert.match(server.of('/memory/forget/prepare')[0].body.statement, /里斯本/);
});

test('session end: the rest is handed to a detached flusher; the hook returns at once', async () => {
  const server = await serve({ '/memory/add': () => ({ delay: 400, ...ADDED() }) });
  const cc = install(server);
  await cc.turn('sess-e', 1, 'I am allergic to peanuts.', 'Noted.');
  await cc.turn('sess-e', 2, 'My birthday is on 3 May.', 'Noted.');
  const ended = await cc.end('sess-e');
  assert.deepEqual([ended.code, ended.out, ended.err], [0, '', '']);
  assert.ok(ended.ms < 1300, `SessionEnd took ${ended.ms} ms (budget 1.5 s)`);
  assert.equal(server.of('/memory/add').length, 0, 'the hook did not wait for the service');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the background flush');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['I am allergic to peanuts.', 'Noted.', 'My birthday is on 3 May.', 'Noted.']);
});

test('idle: a chat left open is saved after QUOTEMEM_IDLE_FLUSH_MS by a flusher that waits, with no later hook', async () => {
  const server = await serve({ '/memory/add': ADDED });
  const cc = install(server, { env: { QUOTEMEM_IDLE_FLUSH_MS: '1200', QUOTEMEM_DEBUG: '1' } });
  await cc.prompt('sess-i', 'prompt-1', 'My kite is orange.');
  const at = Date.now();
  const stopped = await cc.stop('sess-i', 'prompt-1', 'Noted.');
  assert.deepEqual([stopped.code, stopped.out, stopped.err], [0, '', '']);
  assert.ok(stopped.ms < 1200, `the Stop hook did not wait for the idle time (${stopped.ms} ms)`);
  assert.equal(server.of('/memory/add').length, 0, 'nothing is saved before the idle time');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the idle save');
  const [add] = server.of('/memory/add');
  assert.ok(add.at - at >= 1200, `saved after the idle time (${add.at - at} ms)`);
  assert.equal(add.body.session_id, 'claude-code:sess-i');
  assert.deepEqual(add.body.messages.map((m) => [m.role, m.content]), [['user', 'My kite is orange.'], ['assistant', 'Noted.']]);
  await waitFor(() => /flusher exit: .*no idle chat left/.test(cc.logText()), 8000, 'the waiting flusher to exit');
  assert.equal(server.of('/memory/add').length, 1, 'nothing is sent twice');
});

test('inside Cursor: silent when Cursor\'s own QuoteMem hooks are installed, so a turn is saved once', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'I moved to Lisbon last spring.', { score: 0.9 })] } }),
  });
  // Cursor runs Claude Code's hooks with its own payload fields.
  const cursorFields = (project) => ({ cursor_version: '2.1.0', conversation_id: 'conv-1', workspace_roots: [project] });
  const cursorTurns = async (cc, session) => {
    for (const n of [1, 2, 3]) {
      await cc.prompt(session, `prompt-${n}`, `Fact number ${n} about me.`, cursorFields(path.join(cc.home, 'project')));
      await cc.stop(session, `prompt-${n}`, 'Noted.', cursorFields(path.join(cc.home, 'project')));
    }
    return cc.prompt(session, 'prompt-4', 'Where did I say I live now?', cursorFields(path.join(cc.home, 'project')));
  };

  // Cursor's hooks file (here in the project) runs quotemem-hook: the Claude Code hooks stay silent.
  const withFile = install(server);
  fs.mkdirSync(path.join(withFile.home, 'project', '.cursor'), { recursive: true });
  fs.writeFileSync(path.join(withFile.home, 'project', '.cursor', 'hooks.json'), JSON.stringify({ version: 1, hooks: { stop: [{ command: 'quotemem-hook cursor stop' }] } }));
  const silent = await cursorTurns(withFile, 'sess-cursor-a');
  assert.deepEqual([silent.code, silent.out], [0, '']);

  // Only the marker the Cursor hook leaves: silent too.
  const withMarker = install(server);
  fs.mkdirSync(path.join(withMarker.home, '.quotemem', 'native'), { recursive: true });
  fs.writeFileSync(path.join(withMarker.home, '.quotemem', 'native', 'cursor.json'), '{"host":"cursor"}');
  assert.equal((await cursorTurns(withMarker, 'sess-cursor-b')).out, '');
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'nothing captured twice');
  assert.equal(server.of('/memory/search').length, 0);
  assert.ok(!fs.existsSync(withMarker.stateDir) || fs.readdirSync(withMarker.stateDir).length === 0, 'no queue was written');

  // The marker does not silence Claude Code itself (no cursor_version).
  await withMarker.turn('sess-cc', 1, 'I live in Lisbon.', 'Noted.');
  await withMarker.turn('sess-cc', 2, 'My dog is Biscuit.', 'Nice.');
  await withMarker.turn('sess-cc', 3, 'I write Rust.', 'Good.');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the Claude Code batch');
  assert.equal(server.of('/memory/add')[0].body.session_id, 'claude-code:sess-cc');

  // Inside Cursor without Cursor's QuoteMem hooks, these hooks are the only capture: they run.
  const alone = install(server);
  const answered = await cursorTurns(alone, 'sess-cursor-c');
  assert.match(context(answered), /Lisbon/);
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the batch from Cursor');
});

test('inside Cursor 3.20.17 and later without Cursor\'s hooks: the prompt answer uses Cursor\'s additional_context', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'I moved to Lisbon last spring.', { score: 0.9 })] } }),
  });
  const cc = install(server);
  const fields = { cursor_version: '3.24.9', conversation_id: 'conv-9', workspace_roots: [path.join(cc.home, 'project')] };
  const answer = await cc.prompt('sess-cursor-new', 'p1', 'Where did I say I live now?', fields);
  const parsed = JSON.parse(answer.out);
  // Cursor's field only: Cursor may read Claude Code's field too and add the text twice.
  assert.deepEqual(Object.keys(parsed).sort(), ['additional_context', 'continue']);
  assert.equal(parsed.continue, true);
  assert.match(parsed.additional_context, /^\[QuoteMem: retrieved long-term memory\][\s\S]*Lisbon/);
  // Turned off: Claude Code's own field only.
  const off = install(server, { env: { QUOTEMEM_CURSOR_PROMPT_CONTEXT: 'off' } });
  const plain = JSON.parse((await off.prompt('sess-cursor-off', 'p1', 'Where did I say I live now?', fields)).out);
  assert.deepEqual(Object.keys(plain), ['hookSpecificOutput']);
});

test('inside Devin: silent when Devin\'s own QuoteMem hooks are installed for the project, so a turn is saved once', async () => {
  const server = await serve({
    '/memory/add': ADDED,
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'I moved to Lisbon last spring.', { score: 0.9 })] } }),
  });
  // Devin runs Claude Code's hooks with its own payload (prompt_id, no
  // transcript_path) and DEVIN_PROJECT_DIR set.
  const inDevin = (project = null) => {
    const cc = install(server);
    cc.env.DEVIN_PROJECT_DIR = project ?? path.join(cc.home, 'project');
    return cc;
  };
  const devinTurns = async (cc, session) => {
    for (const n of [1, 2, 3]) {
      await cc.hook('UserPromptSubmit', { session_id: session, prompt_id: `p${n}`, prompt: `Fact number ${n} about me.` });
      await cc.hook('Stop', { session_id: session, prompt_id: `p${n}`, stop_hook_active: false, last_assistant_message: 'Noted.' });
    }
    return cc.hook('UserPromptSubmit', { session_id: session, prompt_id: 'p4', prompt: 'Where did I say I live now?' });
  };
  const ours = JSON.stringify({ Stop: [{ hooks: [{ type: 'command', command: 'quotemem-hook devin Stop' }] }] });

  // The project's .devin/hooks.v1.json runs quotemem-hook.
  const withFile = inDevin();
  fs.mkdirSync(path.join(withFile.home, 'project', '.devin'), { recursive: true });
  fs.writeFileSync(path.join(withFile.home, 'project', '.devin', 'hooks.v1.json'), ours);
  const silent = await devinTurns(withFile, 'sess-devin-a');
  assert.deepEqual([silent.code, silent.out], [0, '']);

  // A hooks file in a folder above the project, inside the repository: silent too.
  const nested = install(server);
  fs.mkdirSync(path.join(nested.home, 'project', '.git'), { recursive: true });
  fs.mkdirSync(path.join(nested.home, 'project', '.devin'), { recursive: true });
  fs.writeFileSync(path.join(nested.home, 'project', '.devin', 'hooks.v1.json'), ours);
  fs.mkdirSync(path.join(nested.home, 'project', 'app'), { recursive: true });
  nested.env.DEVIN_PROJECT_DIR = path.join(nested.home, 'project', 'app');
  assert.equal((await devinTurns(nested, 'sess-devin-b')).out, '');
  await sleep(300);
  assert.equal(server.of('/memory/add').length, 0, 'nothing captured twice');
  assert.equal(server.of('/memory/search').length, 0);

  // Claude Code itself (it sends transcript_path) is not silenced, even with
  // Devin's hooks in the project.
  await withFile.turn('sess-cc', 1, 'I live in Lisbon.', 'Noted.');
  await withFile.turn('sess-cc', 2, 'My dog is Biscuit.', 'Nice.');
  await withFile.turn('sess-cc', 3, 'I write Rust.', 'Good.');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the Claude Code batch');
  assert.equal(server.of('/memory/add')[0].body.session_id, 'claude-code:sess-cc');

  // Devin's hooks ran in another project (they left their marker), but not
  // in this one: these hooks do the work here, replies included.
  const elsewhere = inDevin();
  fs.mkdirSync(path.join(elsewhere.home, '.quotemem', 'native'), { recursive: true });
  fs.writeFileSync(path.join(elsewhere.home, '.quotemem', 'native', 'devin.json'), '{"host":"devin"}');
  const answered = await devinTurns(elsewhere, 'sess-devin-c');
  assert.match(context(answered), /Lisbon/);
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the batch from Devin');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => m.content), ['Fact number 1 about me.', 'Noted.', 'Fact number 2 about me.', 'Noted.', 'Fact number 3 about me.', 'Noted.']);
});

test('interrupted and failed turns keep the user words; Stop falls back to the transcript', async () => {
  const server = await serve({ '/memory/add': ADDED });
  const cc = install(server);
  // Interrupted: Stop never fires, the next prompt closes the turn.
  await cc.prompt('sess-i', 'p1', 'I drive a blue Volvo.');
  // StopFailure (an API error) closes the turn without a reply.
  await cc.prompt('sess-i', 'p2', 'My partner works as a nurse.');
  const failed = await cc.hook('StopFailure', { session_id: 'sess-i', prompt_id: 'p2', transcript_path: cc.transcript('sess-i'), cwd: '/', permission_mode: 'default', error: 'rate_limit', last_assistant_message: 'API Error: rate limited' });
  assert.deepEqual([failed.code, failed.out], [0, '']);
  // No last_assistant_message: the reply is read from the transcript tail.
  const file = cc.transcript('sess-i');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, [
    { type: 'user', promptId: 'p3', message: { role: 'user', content: 'We picked Postgres for the shop.' } },
    { type: 'assistant', message: { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }] } },
    { type: 'user', message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'secret tool output' }] } },
    { type: 'assistant', message: { role: 'assistant', content: [{ type: 'thinking', thinking: 'hmm' }, { type: 'text', text: 'Postgres it is.' }] } },
  ].map((r) => JSON.stringify(r)).join('\n') + '\n');
  await cc.prompt('sess-i', 'p3', 'We picked Postgres for the shop.');
  await cc.stop('sess-i', 'p3', undefined);
  // The transcript does not hold the next turn yet: its last reply is p3's,
  // which is not taken for p4.
  await cc.prompt('sess-i', 'p4', 'And the cache is Redis.');
  await cc.stop('sess-i', 'p4', undefined);
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => [m.role, m.content]), [
    ['user', 'I drive a blue Volvo.'],
    ['user', 'My partner works as a nurse.'],
    ['user', 'We picked Postgres for the shop.'], ['assistant', 'Postgres it is.'],
  ]);
  await cc.end('sess-i');
  await waitFor(() => server.of('/memory/add').length === 2, 8000, 'the rest at session end');
  assert.deepEqual(server.of('/memory/add')[1].body.messages.map((m) => [m.role, m.content]), [['user', 'And the cache is Redis.']]);
});

test('skipped: slash commands, host-submitted prompts, subagents; paste markers are stripped', async () => {
  const server = await serve({ '/memory/add': ADDED, '/memory/search': () => ({ body: { data: [] } }) });
  const cc = install(server);
  for (const [i, text] of [
    '/review src/main.rs',
    '<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\nWhat did I decide earlier?\n</task-notification>',
    '<teammate-message from="planner">Do you remember my deadline?</teammate-message>',
  ].entries()) {
    const r = await cc.prompt('sess-s', `skip-${i}`, text);
    assert.deepEqual([r.code, r.out], [0, '']);
    await cc.stop('sess-s', `skip-${i}`, 'ok then');
  }
  // A subagent's hooks carry its agent_id.
  await cc.prompt('sess-s', 'sub-1', 'What did I say about my deadline?', { agent_id: 'agent-7', agent_type: 'Explore' });
  await cc.stop('sess-s', 'sub-1', 'Subagent answer.', { agent_id: 'agent-7', agent_type: 'Explore' });
  await cc.prompt('sess-s', 'p1', 'Here is my config:\n<pasted_content id="1">\nport = 8080\n</pasted_content id="1">\nI always use port 8080.');
  await cc.stop('sess-s', 'p1', 'Got it.');
  await cc.end('sess-s');
  await waitFor(() => server.of('/memory/add').length === 1, 8000, 'the batch');
  assert.deepEqual(server.of('/memory/add')[0].body.messages.map((m) => m.content), ['Here is my config:\nport = 8080\nI always use port 8080.', 'Got it.']);
  assert.equal(server.of('/memory/search').length, 0);

  assert.equal(cleanPrompt('a\n<pasted_content id="x">\nb\n</pasted_content id="x">\nc'), 'a\nb\nc');
  assert.equal(isHostInjected('<my-snippet>hello</my-snippet>'), true);
  assert.equal(isHostInjected('<div>hello</div> what does this do?'), false);
  assert.equal(isHostInjected('Explain <task-notification> tags'), false);
});

test('degrades: service down, no token, bad input — every hook exits 0 quickly and prints nothing', async () => {
  // A closed port: recall and capture fail, the prompt goes on.
  const down = install(null, { options: { CLAUDE_PLUGIN_OPTION_URL: 'http://127.0.0.1:9' } });
  // A memory question gets only the short "memory unavailable" note.
  const p = await down.prompt('sess-d', 'p1', 'What did I tell you about my job?');
  assert.deepEqual([p.code, p.err], [0, '']);
  assert.match(JSON.parse(p.out).hookSpecificOutput.additionalContext, /^\[QuoteMem: memory unavailable\]/);
  assert.ok(p.ms < 4000, `prompt hook took ${p.ms} ms`);
  for (let i = 2; i <= 3; i += 1) await down.turn('sess-d', i, `fact ${i} about my work`, 'ok');
  const e = await down.end('sess-d');
  assert.deepEqual([e.code, e.out, e.err], [0, '', '']);
  // The breaker skips recall for a while: the next memory question costs nothing.
  const p2 = await down.prompt('sess-d2', 'p1', 'Do you remember my job?');
  assert.equal(p2.code, 0);
  assert.match(JSON.parse(p2.out).hookSpecificOutput.additionalContext, /memory unavailable/);
  const plain = await down.prompt('sess-d2', 'p2', 'Refactor the parser into two functions.');
  assert.deepEqual([plain.code, plain.out, plain.err], [0, '', ''], 'an ordinary request gets nothing');
  assert.ok(p2.ms < 1500, `breaker open, took ${p2.ms} ms`);

  // No token anywhere: automatic memory is off and says so once in the log.
  const server = await serve({ '/memory/add': ADDED, '/memory/search': () => ({ body: { data: [] } }) });
  const none = install(server, { options: { CLAUDE_PLUGIN_OPTION_TOKEN: undefined } });
  const s = await none.start('sess-n');
  assert.deepEqual([s.code, s.out, s.err], [0, '', '']);
  for (let i = 1; i <= 3; i += 1) {
    const r = await none.turn('sess-n', i, `What did I say about item ${i}?`, 'ok');
    assert.deepEqual([r.prompt.code, r.prompt.out, r.stop.code], [0, '', 0]);
  }
  assert.equal(server.requests.length, 0);
  assert.match(none.logText(), /automatic memory is off: no token/);

  // Malformed or empty stdin, an unknown event.
  for (const raw of ['', 'not json', '[1,2]']) {
    const r = await down.hook('UserPromptSubmit', null, raw);
    assert.deepEqual([r.code, r.out, r.err], [0, '', '']);
  }
  const unknown = await down.hook('PreToolUse', { session_id: 'x', tool_name: 'Bash' });
  assert.deepEqual([unknown.code, unknown.out], [0, '']);
});

test('options: a set plugin option wins over QUOTEMEM_*; an empty or unexpanded one does not', async () => {
  const server = await serve({ '/memory/search': () => ({ body: { data: [] } }) });
  const other = await serve({ '/memory/search': () => ({ body: { data: [] } }) });
  const a = install(server, { env: { QUOTEMEM_URL: other.url, QUOTEMEM_USER_ID: 'someone-else' } });
  await a.prompt('sess-o', 'p1', 'What is my name?');
  assert.equal(server.of('/memory/search').length, 1);
  assert.equal(other.of('/memory/search').length, 0);
  assert.equal(server.of('/memory/search')[0].body.user_id, 'quei');

  const b = install(null, {
    options: { CLAUDE_PLUGIN_OPTION_URL: '${user_config.url}', CLAUDE_PLUGIN_OPTION_TOKEN: '', CLAUDE_PLUGIN_OPTION_USER_ID: '' },
    env: { QUOTEMEM_URL: other.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei' },
  });
  await b.prompt('sess-o', 'p1', 'What is my name?');
  assert.equal(other.of('/memory/search').length, 1);
  assert.equal(other.of('/memory/search')[0].body.user_id, 'quei');
  assert.equal(other.of('/memory/search')[0].headers.authorization, `Bearer ${TOKEN}`);
});

test('the token never reaches a log, a state file, stdout or stderr', async () => {
  const server = await serve({
    '/memory/add': () => ({ status: 503, headers: { 'retry-after': '0' }, body: { code: 'busy' } }),
    '/memory/search': () => ({ status: 401, body: { code: 'unauthorized' } }),
  });
  const cc = install(server, { env: { QUOTEMEM_DEBUG: '1', QUOTEMEM_CAPTURE_RETRY_BASE_MS: '10' } });
  const outputs = [];
  outputs.push(await cc.start('sess-t'));
  for (let i = 1; i <= 3; i += 1) {
    const r = await cc.turn('sess-t', i, `What did I say about secret ${i}?`, 'ok');
    outputs.push(r.prompt, r.stop);
  }
  outputs.push(await cc.prompt('sess-t', 'p9', 'Forget that I like tea.'));
  outputs.push(await cc.end('sess-t'));
  await waitFor(() => server.of('/memory/add').length >= 2, 8000, 'retries');
  const everything = cc.logText() + outputs.map((o) => o.out + o.err).join('\n');
  assert.ok(cc.logText().length > 0, 'something was logged');
  assert.ok(!everything.includes(TOKEN), 'the token leaked');
});

test('MCP: the launcher applies the plugin options and serves the DSH bundle tools', async () => {
  const server = await serve({ '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000009', 'I live in Lisbon.')] } }) });
  const env = {
    PATH: process.env.PATH,
    HOME: path.join(root, 'mcp-home'),
    CLAUDE_PLUGIN_ROOT: PLUGIN,
    QUOTEMEM_URL: 'http://127.0.0.1:9',
    QUOTEMEM_PLUGIN_OPTION_URL: server.url,
    QUOTEMEM_PLUGIN_OPTION_TOKEN: TOKEN,
    QUOTEMEM_PLUGIN_OPTION_USER_ID: '${user_config.user_id}',
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_AUTO: '1',
  };
  const child = spawn('node', [MCP], { env, stdio: ['pipe', 'pipe', 'pipe'] });
  let out = '';
  let err = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { err += d; });
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  const reply = async (id) => {
    await waitFor(() => out.split('\n').some((line) => line.includes(`"id":${id}`)), 5000, `reply ${id}`);
    return JSON.parse(out.split('\n').find((line) => line.includes(`"id":${id}`)));
  };
  send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'claude-code', version: 'test' } } });
  assert.equal((await reply(1)).result.serverInfo.name, 'quotemem');
  send({ jsonrpc: '2.0', method: 'notifications/initialized' });
  send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
  const names = (await reply(2)).result.tools.map((t) => t.name);
  assert.ok(names.includes('quotemem_search') && names.includes('quotemem_forget_prepare') && names.includes('quotemem_forget_confirm'));
  assert.ok(!names.includes('quotemem_add'), 'add is hidden while capture is automatic');
  send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
  const result = await reply(3);
  assert.ok(!result.result.isError, JSON.stringify(result));
  child.stdin.end();
  await new Promise((resolve) => child.on('exit', resolve));
  const search = server.of('/memory/search')[0];
  assert.equal(search.body.user_id, 'quei', 'an unexpanded option falls back to QUOTEMEM_USER_ID');
  assert.equal(search.headers.authorization, `Bearer ${TOKEN}`);
  assert.ok(!(out + err).includes(TOKEN));
});

test('lastAssistantText reads only the tail and tolerates junk', () => {
  const file = path.join(root, 'tail.jsonl');
  const filler = JSON.stringify({ type: 'user', message: { content: 'x'.repeat(1000) } });
  fs.writeFileSync(file, `${Array(800).fill(filler).join('\n')}\nnot json\n${JSON.stringify({ type: 'assistant', message: { content: [{ type: 'text', text: 'final words' }] } })}\n`);
  assert.equal(lastAssistantText(file), 'final words');
  assert.equal(lastAssistantText(path.join(root, 'missing.jsonl')), null);
  assert.equal(lastAssistantText(undefined), null);
});
