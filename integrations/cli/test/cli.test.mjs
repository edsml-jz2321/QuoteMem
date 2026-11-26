#!/usr/bin/env node
// The `quotemem` command and the package-wide defaults, against a stub
// QuoteMem service. Every run uses a temp HOME, so nothing outside the temp
// folder is read or written.
//
//   node --test integrations/cli/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { HIT, TOKEN, stubServer, tempDir, waitFor } from '../../core/test/helpers.mjs';

const INTEGRATIONS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const CLI = path.join(INTEGRATIONS, 'cli', 'quotemem.mjs');
const OS_USER = os.userInfo().username;
const VERSION = JSON.parse(fs.readFileSync(path.join(INTEGRATIONS, '..', 'package.json'), 'utf8')).version;

const root = tempDir('qm-cli-');
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });
async function serve(handlers) {
  const server = await stubServer(handlers);
  servers.add(server);
  return server;
}

let seq = 0;
function freshHome() {
  seq += 1;
  const home = path.join(root, `home-${seq}`);
  fs.mkdirSync(path.join(home, 'project'), { recursive: true });
  return home;
}

/** Run a command with JSON (or nothing) on stdin. */
function run(args, { env, cwd, input = '' }) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { env, cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('exit', (code) => resolve({ code, out, err }));
    child.stdin.end(input);
  });
}

const quotemem = (args, options) => run([CLI, ...args], options);

test('one memory: every adapter and the MCP server default user_id to the OS user name', async () => {
  const server = await serve({
    '/memory/search': () => ({ body: { data: [HIT('01JX0000000000000000000001', 'I moved to Lisbon last spring.', { score: 0.9 })] } }),
  });
  const home = freshHome();
  const project = path.join(home, 'project');
  // No QUOTEMEM_USER_ID anywhere: not in the environment, not in ~/.quotemem/env.
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, CODEX_HOME: path.join(home, '.codex'), QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN };
  const question = 'Where did I say I live now?';

  // Claude Code, Codex and Gemini CLI hooks, each through `quotemem hook`.
  const claude = await quotemem(['hook', 'claude-code', 'UserPromptSubmit'], {
    env, cwd: project,
    input: JSON.stringify({ hook_event_name: 'UserPromptSubmit', session_id: 'cc-1', prompt_id: 'p1', prompt: question, cwd: project, transcript_path: null }),
  });
  const codex = await quotemem(['hook', 'codex', 'UserPromptSubmit'], {
    env, cwd: project,
    input: JSON.stringify({ hook_event_name: 'UserPromptSubmit', session_id: '019a7c2e-0001-7d4e-9b1a-5c3f2e1d0a9b', turn_id: 'turn-1', prompt: question, cwd: project, transcript_path: null, model: 'gpt-5.1', permission_mode: 'default' }),
  });
  const gemini = await quotemem(['hook', 'gemini', 'BeforeAgent'], {
    env, cwd: project,
    input: JSON.stringify({ hook_event_name: 'BeforeAgent', session_id: 'gm-1', prompt: question }),
  });
  for (const [name, result] of [['claude-code', claude], ['codex', codex], ['gemini', gemini]]) {
    assert.equal(result.code, 0, `${name}: ${result.err}`);
    assert.match(result.out, /Lisbon/, `${name} injected the recall`);
  }

  // The MCP server, started as `quotemem mcp`.
  const mcp = spawn(process.execPath, [CLI, 'mcp'], { env, cwd: project, stdio: ['pipe', 'pipe', 'pipe'] });
  let buffer = '';
  const replies = new Map();
  mcp.stdout.on('data', (d) => {
    buffer += d;
    let i;
    while ((i = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, i);
      buffer = buffer.slice(i + 1);
      if (line.trim()) { const msg = JSON.parse(line); replies.set(msg.id, msg); }
    }
  });
  const send = (msg) => mcp.stdin.write(`${JSON.stringify(msg)}\n`);
  send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '1' } } });
  send({ jsonrpc: '2.0', method: 'notifications/initialized' });
  send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: question } } });
  await waitFor(() => replies.has(2), 8000, 'the MCP search');
  assert.ok(!replies.get(2).result.isError, JSON.stringify(replies.get(2)));
  mcp.stdin.end();
  await new Promise((resolve) => mcp.on('exit', resolve));

  const searches = server.of('/memory/search');
  assert.equal(searches.length, 4, 'one search from each of the four');
  assert.deepEqual(searches.map((s) => s.body.user_id), [OS_USER, OS_USER, OS_USER, OS_USER]);
});

test('install: a dry run writes nothing; --write merges and keeps other entries; reruns do not duplicate; the token is never written', async () => {
  const home = freshHome();
  const project = path.join(home, 'project');
  const canary = 'qm-cli-canary-token-7f1e';
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, CODEX_HOME: path.join(home, '.codex'), QUOTEMEM_TOKEN: canary };
  const cursorDir = path.join(home, '.cursor');
  fs.mkdirSync(cursorDir, { recursive: true });
  fs.writeFileSync(path.join(cursorDir, 'hooks.json'), JSON.stringify({ version: 1, hooks: { stop: [{ command: './my-own-hook.sh' }] } }));
  fs.writeFileSync(path.join(cursorDir, 'mcp.json'), JSON.stringify({ mcpServers: { other: { command: 'other' } } }));
  const before = fs.readFileSync(path.join(cursorDir, 'hooks.json'), 'utf8');

  const dry = await quotemem(['install', 'cursor'], { env, cwd: project });
  assert.equal(dry.code, 0, dry.err);
  assert.match(dry.out, /\[dry run\]/);
  assert.equal(fs.readFileSync(path.join(cursorDir, 'hooks.json'), 'utf8'), before, 'the dry run changed nothing');

  for (let i = 0; i < 2; i += 1) {
    const written = await quotemem(['install', 'cursor', '--write'], { env, cwd: project });
    assert.equal(written.code, 0, written.err);
  }
  const hooks = JSON.parse(fs.readFileSync(path.join(cursorDir, 'hooks.json'), 'utf8'));
  assert.deepEqual(hooks.hooks.stop.map((h) => h.command.includes('quotemem-hook') ? 'quotemem' : h.command), ['./my-own-hook.sh', 'quotemem']);
  for (const event of ['sessionStart', 'beforeSubmitPrompt', 'afterAgentResponse', 'stop', 'sessionEnd']) {
    assert.equal(hooks.hooks[event].filter((h) => h.command.includes('quotemem-hook')).length, 1, `one ${event} hook`);
    assert.match(hooks.hooks[event].at(-1).command, new RegExp(`cursor ${event}$`));
  }
  const mcp = JSON.parse(fs.readFileSync(path.join(cursorDir, 'mcp.json'), 'utf8'));
  assert.deepEqual(Object.keys(mcp.mcpServers).sort(), ['other', 'quotemem']);
  assert.deepEqual(mcp.mcpServers.quotemem.args.slice(-1), ['mcp']);

  for (const tool of ['devin', 'continue', 'cline', 'goose']) {
    const result = await quotemem(['install', tool, '--write'], { env, cwd: project });
    assert.equal(result.code, 0, `${tool}: ${result.err}`);
  }
  assert.ok(fs.existsSync(path.join(project, '.devin', 'hooks.v1.json')));
  assert.ok(fs.existsSync(path.join(project, '.continue', 'mcpServers', 'quotemem.yaml')));
  assert.ok(fs.existsSync(path.join(project, '.continue', 'rules', 'quotemem.md')));
  const clineHook = path.join(home, 'Documents', 'Cline', 'Hooks', 'UserPromptSubmit');
  assert.ok(fs.statSync(clineHook).mode & 0o100, 'the Cline hook is executable');
  assert.ok(fs.existsSync(path.join(home, '.agents', 'plugins', 'quotemem', 'hooks', 'hooks.json')));

  const codex = await quotemem(['install', 'codex', '--write'], { env, cwd: project });
  assert.equal(codex.code, 0, codex.err);
  assert.match(fs.readFileSync(path.join(home, '.codex', 'config.toml'), 'utf8'), /\[mcp_servers\.quotemem\][\s\S]*QUOTEMEM_AUTO = "1"/);

  // The written Cline hook runs; with QuoteMem down it lets the turn go on.
  const cline = await new Promise((resolve) => {
    const child = spawn(clineHook, [], { env: { ...env, QUOTEMEM_URL: 'http://127.0.0.1:9' }, stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.on('exit', (code) => resolve({ code, out }));
    child.stdin.end(JSON.stringify({ taskId: 't1', hookName: 'UserPromptSubmit', userPromptSubmit: { prompt: 'hi' } }));
  });
  assert.deepEqual([cline.code, JSON.parse(cline.out)], [0, { cancel: false }]);

  // Never the token: not in a written file, not in the output.
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]));
  for (const file of walk(home)) assert.ok(!fs.readFileSync(file, 'utf8').includes(canary), `${file} holds the token`);

  // A Cline hook file that is not ours is never replaced.
  fs.writeFileSync(path.join(home, 'Documents', 'Cline', 'Hooks', 'TaskComplete'), '#!/bin/sh\necho mine\n');
  const refused = await quotemem(['install', 'cline', '--write'], { env, cwd: project });
  assert.equal(refused.code, 1);
  assert.match(refused.err, /is another hook/);
  assert.equal(fs.readFileSync(path.join(home, 'Documents', 'Cline', 'Hooks', 'TaskComplete'), 'utf8'), '#!/bin/sh\necho mine\n');
});

test('install cursor: a QUOTEMEM_* setting set only in the shell is named (the CLI\'s MCP server would not see it)', async () => {
  const home = freshHome();
  const project = path.join(home, 'project');
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.example.org', QUOTEMEM_USER_ID: 'quei', QUOTEMEM_TOKEN: 'qm-shell-token-31c9' };
  const run = await quotemem(['install', 'cursor'], { env, cwd: project });
  assert.equal(run.code, 0, run.err);
  assert.match(run.out, /warning: QUOTEMEM_TOKEN, QUOTEMEM_URL, QUOTEMEM_USER_ID are set in this shell, not in ~\/\.quotemem\/env\. Cursor's CLI\ndoes not pass the shell's environment to the MCP server/);
  assert.ok(!run.out.includes('qm-shell-token-31c9') && !run.out.includes('memory.example.org'), 'names only, never values');
  // The same values in ~/.quotemem/env: nothing to say; another value there is named.
  fs.mkdirSync(path.join(home, '.quotemem'), { recursive: true, mode: 0o700 });
  fs.writeFileSync(path.join(home, '.quotemem', 'env'), 'QUOTEMEM_URL=https://memory.example.org\nQUOTEMEM_USER_ID=someone-else\n', { mode: 0o600 });
  const second = await quotemem(['install', 'cursor'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.example.org', QUOTEMEM_USER_ID: 'quei' }, cwd: project });
  assert.match(second.out, /warning: QUOTEMEM_USER_ID is set in this shell, not in ~\/\.quotemem\/env\./);
  const same = await quotemem(['install', 'cursor'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.example.org' }, cwd: project });
  assert.doesNotMatch(same.out, /set in this shell/);
  // The service's model settings and keys are pointed to service.env, never to ~/.quotemem/env.
  const keys = await quotemem(['install', 'cursor'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.example.org', QUOTEMEM_CHAT_API_KEY: 'sk-shell-key-7f2e', QUOTEMEM_EMBED_MODEL: 'text-embedding-v4' }, cwd: project });
  assert.equal(keys.code, 0, keys.err);
  assert.match(keys.out, /note: QUOTEMEM_CHAT_API_KEY, QUOTEMEM_EMBED_MODEL are set in this shell\. The service's model settings and\nkeys belong in ~\/\.quotemem\/service\.env \(written by `quotemem setup`\), not in ~\/\.quotemem\/env\./);
  assert.doesNotMatch(keys.out, /warning: [^\n]*QUOTEMEM_CHAT_API_KEY/);
  assert.ok(!keys.out.includes('sk-shell-key-7f2e') && !keys.out.includes('text-embedding-v4'), 'names only, never values');
});

test('install: protected files keep their mode, get a backup, and stay symlinks', { skip: process.platform === 'win32' ? 'POSIX modes' : false }, async () => {
  const home = freshHome();
  const project = path.join(home, 'project');
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, CODEX_HOME: path.join(home, '.codex') };
  // ~/.cursor/mcp.json is a link into a dotfiles folder and holds another server's key.
  const dotfiles = path.join(home, 'dotfiles');
  fs.mkdirSync(dotfiles, { recursive: true });
  fs.mkdirSync(path.join(home, '.cursor'), { recursive: true });
  const real = path.join(dotfiles, 'mcp.json');
  const original = JSON.stringify({ mcpServers: { other: { command: 'other', env: { OTHER_KEY: 'k' } } } });
  fs.writeFileSync(real, original, { mode: 0o600 });
  fs.chmodSync(real, 0o600);
  fs.symlinkSync(real, path.join(home, '.cursor', 'mcp.json'));
  const codexHome = path.join(home, '.codex');
  fs.mkdirSync(codexHome, { recursive: true });
  fs.writeFileSync(path.join(codexHome, 'config.toml'), '[mcp_servers.other]\ncommand = "other"\nenv = { OTHER_KEY = "k" }\n', { mode: 0o600 });
  fs.chmodSync(path.join(codexHome, 'config.toml'), 0o600);

  for (const tool of ['cursor', 'codex']) {
    const result = await quotemem(['install', tool, '--write'], { env, cwd: project });
    assert.equal(result.code, 0, `${tool}: ${result.err}`);
    assert.match(result.out, /backup of the old file: /);
  }
  assert.ok(fs.lstatSync(path.join(home, '.cursor', 'mcp.json')).isSymbolicLink(), 'the link is kept');
  assert.ok('quotemem' in JSON.parse(fs.readFileSync(real, 'utf8')).mcpServers, 'written through the link');
  assert.equal(fs.statSync(real).mode & 0o777, 0o600);
  assert.equal(fs.readFileSync(`${real}.quotemem.bak`, 'utf8'), original);
  assert.equal(fs.statSync(`${real}.quotemem.bak`).mode & 0o777, 0o600);
  assert.equal(fs.statSync(path.join(codexHome, 'config.toml')).mode & 0o777, 0o600);
  assert.match(fs.readFileSync(path.join(codexHome, 'config.toml.quotemem.bak'), 'utf8'), /^\[mcp_servers\.other\]/);
  assert.equal(fs.statSync(path.join(codexHome, 'config.toml.quotemem.bak')).mode & 0o777, 0o600);
});

test('install: print-only tools, an unknown tool, and invalid JSON in a config file', async () => {
  const home = freshHome();
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home };
  const cwd = path.join(home, 'project');
  const expected = {
    'claude-code': /claude plugin marketplace add edsml-jz2321\/QuoteMem[\s\S]*claude plugin install quotemem@quotemem/,
    dsh: /needs Node 24 and pnpm[\s\S]*dsh plugin --profile quotemem add quotemem\n[\s\S]*dsh --profile quotemem/,
    gemini: /gemini extensions install ".*integrations\/gemini-cli"/,
    openclaw: /openclaw plugins install ".*integrations\/openclaw"/,
    opencode: /"plugin": \[[\s\S]*integrations\/opencode"[\s\S]*"plugins": \[/,
  };
  for (const [tool, pattern] of Object.entries(expected)) {
    const result = await quotemem(['install', tool, '--write'], { env, cwd });
    assert.equal(result.code, 0, `${tool}: ${result.err}`);
    assert.match(result.out, pattern, tool);
  }
  assert.deepEqual(fs.readdirSync(home), ['project'], 'print-only tools write nothing');
  const unknown = await quotemem(['install', 'vim'], { env, cwd });
  assert.equal(unknown.code, 2);
  fs.mkdirSync(path.join(home, '.cursor'));
  fs.writeFileSync(path.join(home, '.cursor', 'hooks.json'), '{ not json');
  const broken = await quotemem(['install', 'cursor', '--write'], { env, cwd });
  assert.equal(broken.code, 1);
  assert.match(broken.err, /not valid JSON; fix it first/);
});

// /agent/status as the service answers it: 200 with the right credential header, else 401.
const statusRoute = (header, value) => (_body, entry) => (entry.headers[header] === value
  ? { body: { phase: 'serving' } }
  : { status: 401, body: { code: 'authentication_failed' } });

test('doctor: reports the token check and the shared default owner, never the token', async () => {
  const server = await serve({
    '/health': () => ({ body: {} }),
    '/agent/status': statusRoute('authorization', `Bearer ${TOKEN}`),
    '/memory/search': () => ({ status: 400, body: { code: 'invalid_request' } }),
  });
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const good = await quotemem(['doctor'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN }, cwd });
  assert.equal(good.code, 0, good.out);
  assert.match(good.out, /the service accepts the token/);
  assert.match(good.out, new RegExp(`memory owner \\(QUOTEMEM_USER_ID\\): ${OS_USER} \\(the OS user name`));
  assert.ok(!good.out.includes(TOKEN));
  const bad = await quotemem(['doctor'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: 'wrong-token-1' }, cwd });
  assert.equal(bad.code, 1);
  assert.match(bad.out, /FAIL {2}the service refused the token/);
  // A service on this computer: it may be another user's, and a port of this user's own is named.
  assert.match(bad.out, /another user's QuoteMem on this computer[^\n]*QUOTEMEM_URL=http:\/\/127\.0\.0\.1:2\d{4} in ~\/\.quotemem\/env and in the environment Claude Code, Codex and DSH start in/);
  assert.ok(!bad.out.includes('wrong-token-1'));
  // The token is checked with GET /agent/status: no search request reaches the service.
  assert.equal(server.of('/memory/search').length, 0);
  assert.ok(server.of('/agent/status').length >= 2);
  // An exported token is flagged: an agent started from this shell hands it to its shell tool.
  assert.match(good.out, /note {2}the token is exported in this shell: an agent started from it hands it to its own shell tool/);

  // The documented setup: no token in the environment. The service's own
  // token file on this computer is read; the doctor names where, never what.
  const fileHome = freshHome();
  fs.mkdirSync(path.join(fileHome, '.local', 'share', 'quotemem'), { recursive: true });
  fs.writeFileSync(path.join(fileHome, '.local', 'share', 'quotemem', 'token'), `${TOKEN}\n`, { mode: 0o600 });
  const fromFile = await quotemem(['doctor'], { env: { PATH: process.env.PATH, HOME: fileHome, USERPROFILE: fileHome, QUOTEMEM_URL: server.url }, cwd: path.join(fileHome, 'project') });
  assert.equal(fromFile.code, 0, fromFile.out);
  assert.match(fromFile.out, /token found in the service's own token file \(.*\.local\/share\/quotemem\/token\) \(not shown\)/);
  assert.match(fromFile.out, /the service accepts the token/);
  assert.doesNotMatch(fromFile.out, /exported in this shell/);
  assert.ok(!fromFile.out.includes(TOKEN), 'the token is never printed');
  // A token file named by QUOTEMEM_TOKEN_FILE that cannot be read: the
  // doctor says so, and nothing else.
  const missing = await quotemem(['doctor'], { env: { PATH: process.env.PATH, HOME: freshHome(), QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN_FILE: '~/no-such-token' }, cwd });
  assert.equal(missing.code, 1);
  assert.match(missing.out, /FAIL {2}no token: QUOTEMEM_TOKEN_FILE cannot be read/);
});

test('doctor and status check the token in the configured header mode (x-api-key)', async () => {
  const server = await serve({
    '/health': () => ({ body: {} }),
    '/agent/status': statusRoute('x-api-key', TOKEN),
  });
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const env = { PATH: process.env.PATH, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_MODE: 'x-api-key' };
  const doctor = await quotemem(['doctor'], { env, cwd });
  assert.equal(doctor.code, 0, doctor.out);
  assert.match(doctor.out, /the service accepts the token/);
  const checked = server.of('/agent/status').length;
  const status = await quotemem(['status'], { env, cwd });
  assert.equal(status.code, 0, status.out);
  assert.match(status.out, /memory service: running/);
  assert.doesNotMatch(status.out, /problem:/);
  assert.equal(server.of('/agent/status').length, checked + 1, 'status checked the token too');
  assert.ok(server.of('/agent/status').every((r) => r.headers['x-api-key'] === TOKEN && !r.headers.authorization));
  assert.ok(!(doctor.out + status.out).includes(TOKEN));
});

/** A folder with a fake `uvx` that records its arguments and stdin, then exits with QM_FAKE_UVX_EXIT. */
function fakeUvx(home) {
  const bin = path.join(home, 'fake-bin');
  fs.mkdirSync(bin, { recursive: true });
  const script = [
    '#!/bin/sh',
    'printf "%s\\n" "$@" > "$QM_FAKE_UVX_ARGS"',
    'cat > "$QM_FAKE_UVX_STDIN"',
    'echo "fake uvx out"',
    'echo "fake uvx err" >&2',
    'exit "${QM_FAKE_UVX_EXIT:-0}"',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(bin, 'uvx'), script, { mode: 0o755 });
  return bin;
}

test('service commands: handed to `uvx --from quotemem==<this version> quotemem` with their arguments, stdio and exit code', { skip: process.platform === 'win32' ? 'POSIX shell fake' : false }, async () => {
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const bin = fakeUvx(home);
  const argsFile = path.join(home, 'uvx-args');
  const stdinFile = path.join(home, 'uvx-stdin');
  const env = { PATH: `${bin}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home, QM_FAKE_UVX_ARGS: argsFile, QM_FAKE_UVX_STDIN: stdinFile, QUOTEMEM_USER_ID: 'quei' };
  const cases = [
    ['serve', '--port', '9000', '--data-dir', 'my dir'],
    ['check'],
    ['memories', '--user', 'alice', '--status', 'all', '--limit', '5', '--json'],
    ['memories', '--data-dir', 'my dir', '--html', 'my memories.html'],
    ['export', '--out', 'journal.jsonl'],
    ['rebuild', '--from', 'old', '--dry-run'],
    ['delete-user', 'alice'],
    ['repair', 'alice', '--inspect'],
    ['bench', 'serve', '--workspace', './ws', '--help'],
  ];
  for (const [index, args] of cases.entries()) {
    const run = await quotemem(args, { env: { ...env, QM_FAKE_UVX_EXIT: String(index + 3) }, cwd, input: `stdin ${index}\n` });
    assert.equal(run.code, index + 3, `${args[0]}: the exit code is uvx's`);
    // `memories` without --user lists the plugins' user, not the service's `default`.
    const sent = args[0] === 'memories' && !args.includes('--user') ? [...args, '--user', 'quei'] : args;
    assert.deepEqual(fs.readFileSync(argsFile, 'utf8').split('\n').slice(0, -1), ['--from', `quotemem${args[0] === 'bench' ? '[bench]' : ''}==${VERSION}`, 'quotemem', ...sent], `${args[0]}: pinned to this version (bench with its extra), the arguments unchanged`);
    assert.equal(fs.readFileSync(stdinFile, 'utf8'), `stdin ${index}\n`, `${args[0]}: stdin is passed on`);
    assert.equal(run.out, 'fake uvx out\n');
    assert.equal(run.err, 'fake uvx err\n');
  }
  // The plugin's own commands stay here.
  const version = await quotemem(['--version'], { env, cwd });
  assert.equal(version.code, 0);
  assert.ok(!fs.readFileSync(argsFile, 'utf8').includes('--version'));
});

/** A fake `uvx` that writes its environment and arguments to files, and exits 0. */
function envUvx(home) {
  const bin = path.join(home, 'env-bin');
  fs.mkdirSync(bin, { recursive: true });
  fs.writeFileSync(path.join(bin, 'uvx'), '#!/bin/sh\nenv > "$QM_FAKE_UVX_ENV"\nprintf "%s\\n" "$@" > "$QM_FAKE_UVX_ARGS"\n', { mode: 0o755 });
  return bin;
}

test('service commands: memories, repair and delete-user get the plugins\' URL and token; service.env\'s token goes only to a service on this computer', { skip: process.platform === 'win32' ? 'POSIX shell fake' : false }, async () => {
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const bin = envUvx(home);
  const envOut = path.join(home, 'uvx-env');
  const argsOut = path.join(home, 'uvx-args');
  const dir = path.join(home, '.quotemem');
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const write = (file, text) => fs.writeFileSync(file, text, { mode: 0o600 });
  write(path.join(dir, 'service.env'), 'QUOTEMEM_TOKEN=service-env-token\nQUOTEMEM_HTTP_AUTH_SECRET=service-env-secret\nQUOTEMEM_DATA_DIR=~/store\nQUOTEMEM_CHAT_MODEL=m1\n');
  write(path.join(home, 'remote-token'), 'remote-file-token\n');
  const base = { PATH: `${bin}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home, QM_FAKE_UVX_ENV: envOut, QM_FAKE_UVX_ARGS: argsOut, QUOTEMEM_AUTOSTART: '0' };
  /** What one command hands to uvx, with `hostFile` as ~/.quotemem/env (none when null). */
  const forwarded = async (args, env = {}, hostFile = null) => {
    fs.rmSync(path.join(dir, 'env'), { force: true });
    if (hostFile !== null) write(path.join(dir, 'env'), hostFile);
    const run = await quotemem(args, { env: { ...base, ...env }, cwd });
    assert.equal(run.code, 0, run.err);
    const text = fs.readFileSync(envOut, 'utf8');
    const vars = Object.fromEntries(text.split('\n').filter((line) => line.includes('=')).map((line) => [line.slice(0, line.indexOf('=')), line.slice(line.indexOf('=') + 1)]));
    assert.equal(`${run.out}${run.err}`, '', 'nothing is printed, no token either');
    return { vars, text, args: fs.readFileSync(argsOut, 'utf8').split('\n').slice(0, -1) };
  };
  const CLIENT = [['memories'], ['repair', '--user-id', 'alice', '--inspect'], ['delete-user', '--user-id', 'alice', '--yes']];
  const REMOTE = 'QUOTEMEM_URL=https://memory.example.com\nQUOTEMEM_USER_ID=bob\n';
  const leaked = (run) => ['service-env-token', 'service-env-secret'].filter((token) => run.text.includes(token));

  for (const args of CLIENT) {
    // The service on this computer (the default URL): service.env's token,
    // as the plugins send it, and its data folder.
    const local = await forwarded(args);
    assert.equal(local.vars.QUOTEMEM_TOKEN, 'service-env-token', `${args[0]}: the local service's token`);
    assert.equal(local.vars.QUOTEMEM_HTTP_AUTH_SECRET, undefined);
    assert.equal(local.vars.QUOTEMEM_URL, 'http://127.0.0.1:8765', 'the default URL, as checked');
    assert.equal(local.vars.QUOTEMEM_DATA_DIR, '~/store');
    assert.equal(local.vars.QUOTEMEM_CHAT_MODEL, 'm1');
    // A service on another computer, named only in ~/.quotemem/env: its URL
    // and user reach the command, service.env's token never does.
    const remote = await forwarded(args, {}, REMOTE);
    assert.equal(remote.vars.QUOTEMEM_URL, 'https://memory.example.com', `${args[0]}: the URL of ~/.quotemem/env`);
    assert.equal(remote.vars.QUOTEMEM_USER_ID, 'bob');
    assert.deepEqual(leaked(remote), [], `${args[0]}: no local token for a remote service`);
    // Its token file, named only in ~/.quotemem/env.
    const named = await forwarded(args, {}, `${REMOTE}QUOTEMEM_TOKEN_FILE=~/remote-token\n`);
    assert.equal(named.vars.QUOTEMEM_TOKEN, 'remote-file-token', `${args[0]}: the token of QUOTEMEM_TOKEN_FILE`);
    assert.equal(named.vars.QUOTEMEM_TOKEN_FILE, '~/remote-token');
    assert.deepEqual(leaked(named), []);
  }
  const remoteList = await forwarded(['memories'], {}, REMOTE);
  assert.deepEqual(remoteList.args.slice(-2), ['--user', 'bob'], 'memories lists the plugins\' user');

  // The plugins' order: this environment first, then ~/.quotemem/env; other
  // QUOTEMEM_* values of ~/.quotemem/env fill in, and never replace service.env's.
  const shell = await forwarded(['memories'], { QUOTEMEM_URL: 'http://127.0.0.1:9100', QUOTEMEM_USER_ID: 'carol' }, `${REMOTE}QUOTEMEM_DEBUG=1\nQUOTEMEM_CHAT_MODEL=m2\n`);
  assert.equal(shell.vars.QUOTEMEM_URL, 'http://127.0.0.1:9100');
  assert.equal(shell.vars.QUOTEMEM_USER_ID, 'carol');
  assert.deepEqual(shell.args.slice(-2), ['--user', 'carol']);
  assert.equal(shell.vars.QUOTEMEM_TOKEN, 'service-env-token', 'a service on this computer');
  assert.equal(shell.vars.QUOTEMEM_DEBUG, '1');
  assert.equal(shell.vars.QUOTEMEM_CHAT_MODEL, 'm1');
  // A token in this environment wins, for any service, as in the plugins.
  assert.equal((await forwarded(['memories'], { QUOTEMEM_TOKEN: 'shell-token' })).vars.QUOTEMEM_TOKEN, 'shell-token');
  const shellRemote = await forwarded(['repair', '--user-id', 'alice'], { QUOTEMEM_HTTP_AUTH_SECRET: 'shell-secret' }, REMOTE);
  assert.equal(shellRemote.vars.QUOTEMEM_TOKEN, 'shell-secret');
  assert.deepEqual(leaked(shellRemote), []);
  // A token file that cannot be read is passed on, so the command says so.
  const missing = await forwarded(['memories'], {}, `${REMOTE}QUOTEMEM_TOKEN_FILE=~/no-such-token\n`);
  assert.equal(missing.vars.QUOTEMEM_TOKEN, undefined);
  assert.equal(missing.vars.QUOTEMEM_TOKEN_FILE, '~/no-such-token');
  assert.deepEqual(leaked(missing), []);

  // The URL on the command line decides where the token may go.
  for (const args of [['repair', '--user-id', 'alice', '--url', 'https://other.example.com'], ['memories', '--url=https://other.example.com'], ['delete-user', '--user-id', 'alice', '--ur', 'https://other.example.com']]) {
    const other = await forwarded(args);
    assert.deepEqual(leaked(other), [], `${args.join(' ')}: no local token for another computer`);
    assert.equal(other.vars.QUOTEMEM_TOKEN, undefined);
  }
  // --ur= and QUOTEMEM_URL: neither service.env's token nor its secret reaches a remote service.
  for (const [args, env] of [[['memories', '--ur=https://other.example.com'], {}], [['repair', '--user-id', 'alice'], { QUOTEMEM_URL: 'https://other.example.com' }],
    [['delete-user', '--user-id', 'alice', '--url', 'https://other.example.com'], {}]]) {
    const other = await forwarded(args, env);
    assert.deepEqual(leaked(other), [], `${args.join(' ')}`);
    assert.equal(other.vars.QUOTEMEM_TOKEN, undefined);
    assert.equal(other.vars.QUOTEMEM_HTTP_AUTH_SECRET, undefined);
  }
  assert.equal((await forwarded(['memories', '--ur=http://localhost:9200'])).vars.QUOTEMEM_TOKEN, 'service-env-token');
  // A URL Node reads as this computer and the service's commands as another
  // host (a backslash in the host): no token and no token file at all, not
  // even the shell's.
  const BACKSLASH = 'http://127.0.0.1:8765\\.evil.example.com';
  for (const [args, env, hostFile] of [
    [['memories', '--url', BACKSLASH], {}, null],
    [['delete-user', '--user-id', 'alice', `--ur=${BACKSLASH}`], { QUOTEMEM_TOKEN: 'shell-token' }, null],
    [['repair', '--user-id', 'alice'], {}, `QUOTEMEM_URL=${BACKSLASH}
QUOTEMEM_TOKEN_FILE=~/remote-token
`],
    [['memories'], { QUOTEMEM_URL: BACKSLASH, QUOTEMEM_HTTP_AUTH_SECRET: 'shell-secret' }, null],
  ]) {
    const bad = await forwarded(args, env, hostFile);
    assert.deepEqual(leaked(bad), [], args.join(' '));
    assert.ok(!['shell-token', 'shell-secret', 'remote-file-token'].some((token) => bad.text.includes(token)), args.join(' '));
    assert.equal(bad.vars.QUOTEMEM_TOKEN, undefined);
    assert.equal(bad.vars.QUOTEMEM_TOKEN_FILE, undefined);
  }
  // Hosts Node rewrites (to 127.0.0.1 or [::1]) while the service's commands
  // connect to them as written: no token at all, whichever way they come.
  for (const url of ['http://127.0.0.1.:8765', 'http://127.1.:8765', 'http://localhost.:8765', 'http://0x7f.0.0.1:8765', 'http://2130706433:8765', 'http://[0:0:0:0:0:0:0:1]:8765']) {
    for (const [args, env] of [[['memories', '--url', url], {}], [['repair', '--user-id', 'alice', `--ur=${url}`], {}], [['delete-user', '--user-id', 'alice'], { QUOTEMEM_URL: url }]]) {
      const rewritten = await forwarded(args, env);
      assert.deepEqual(leaked(rewritten), [], `${args.join(' ')} ${JSON.stringify(env)}`);
      assert.equal(rewritten.vars.QUOTEMEM_TOKEN, undefined, url);
    }
  }
  // The command gets the URL as it was checked, in QUOTEMEM_URL and in every --url.
  const checked = await forwarded(['memories', '--url', 'HTTP://LOCALHOST:9200/', '--ur=http://localhost:9200']);
  assert.deepEqual(checked.args.slice(3, 7), ['memories', '--url', 'http://localhost:9200', '--ur=http://localhost:9200']);
  assert.equal(checked.vars.QUOTEMEM_URL, 'http://localhost:9200');
  assert.equal(checked.vars.QUOTEMEM_TOKEN, 'service-env-token');
  // An empty --url= counts as none, as for the service's commands: QUOTEMEM_URL, else the default, decides.
  assert.equal((await forwarded(['memories', '--url='])).vars.QUOTEMEM_TOKEN, 'service-env-token', 'the default URL, on this computer');
  assert.equal((await forwarded(['memories', '--url', '  '])).vars.QUOTEMEM_TOKEN, 'service-env-token', 'a blank one too');
  const emptyRemote = await forwarded(['memories', '--ur='], {}, REMOTE);
  assert.deepEqual(leaked(emptyRemote), []);
  assert.equal(emptyRemote.vars.QUOTEMEM_URL, 'https://memory.example.com');
  assert.equal((await forwarded(['memories', '--url', 'http://localhost:9200'])).vars.QUOTEMEM_TOKEN, 'service-env-token');
  assert.equal((await forwarded(['memories', '--url', 'http://127.0.0.1:9200'], {}, REMOTE)).vars.QUOTEMEM_TOKEN, 'service-env-token', 'a --url on this computer gets its token, whatever QUOTEMEM_URL says');

  // Other service commands keep service.env's settings for the service itself.
  const served = await forwarded(['serve'], {}, REMOTE);
  assert.equal(served.vars.QUOTEMEM_TOKEN, 'service-env-token');
  assert.equal(served.vars.QUOTEMEM_URL, undefined);
});

test('service commands without uvx: one line saying how to get it', async () => {
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const empty = path.join(home, 'empty-bin');
  fs.mkdirSync(empty);
  for (const command of ['serve', 'bench']) {
    const run = await quotemem([command, '--port', '0'], { env: { PATH: empty, HOME: home, USERPROFILE: home }, cwd });
    assert.equal(run.code, 127);
    assert.equal(run.out, '');
    assert.equal(run.err.split('\n').filter(Boolean).length, 1, run.err);
    assert.match(run.err, new RegExp(`^quotemem: \`quotemem ${command}\` runs the Python service through uvx, which is not on PATH; install uv \\(https://docs\\.astral\\.sh/uv/\\) or run \`pipx run --spec 'quotemem${command === 'bench' ? '\\[bench\\]' : ''}==${VERSION.replaceAll('.', '\\.')}' quotemem ${command}\`\\n$`));
  }
});

test('help lists the plugin commands and the service commands; doctor reports uvx and the service', { skip: process.platform === 'win32' ? 'POSIX shell fake' : false }, async () => {
  const home = freshHome();
  const cwd = path.join(home, 'project');
  const help = await quotemem(['--help'], { env: { PATH: process.env.PATH, HOME: home, USERPROFILE: home }, cwd });
  assert.equal(help.code, 0);
  assert.match(help.out, /plugin commands:\n {2}quotemem install/);
  assert.match(help.out, /service commands \(run as `uvx quotemem <command> \.\.\.` with the settings of\n~\/\.quotemem\/service\.env; they need uv\):/);
  assert.match(help.out, /quotemem setup +give the memory service its models, keys,\n +thinking and review, once/);
  for (const command of ['setup', 'status', 'stop', 'serve', 'check', 'memories', 'export', 'rebuild', 'delete-user', 'repair', 'bench', 'install', 'mcp', 'hook', 'doctor']) {
    assert.match(help.out, new RegExp(`quotemem ${command}\\b|\\| ${command}\\b`), command);
  }

  const server = await serve({
    '/health': () => ({ body: {} }),
    '/agent/status': statusRoute('authorization', `Bearer ${TOKEN}`),
  });
  const bin = fakeUvx(home);
  const withUvx = await quotemem(['doctor'], { env: { PATH: `${bin}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_HTTP_AUTH_SECRET: TOKEN }, cwd });
  assert.equal(withUvx.code, 0, withUvx.out);
  assert.ok(withUvx.out.includes(` ok   uvx at ${path.join(bin, 'uvx')}`), withUvx.out);
  assert.match(withUvx.out, /service answers \/health with 200/);
  assert.match(withUvx.out, /the service accepts the token/);
  assert.ok(!withUvx.out.includes('token comes from a file'), 'QUOTEMEM_HTTP_AUTH_SECRET is an environment token');
  assert.ok(!withUvx.out.includes(TOKEN));

  const empty = path.join(home, 'empty-bin');
  fs.mkdirSync(empty, { recursive: true });
  const down = await serve({});
  const url = down.url;
  await down.close();
  const without = await quotemem(['doctor'], { env: { PATH: empty, HOME: home, USERPROFILE: home, QUOTEMEM_URL: url, QUOTEMEM_TOKEN: TOKEN }, cwd });
  assert.equal(without.code, 1, 'an unreachable service fails the check');
  assert.match(without.out, /note {2}uvx not on PATH: `quotemem serve` and the other service commands need uv/);
  const port = new URL(url).port;
  assert.ok(without.out.includes(`; start it with \`quotemem serve --port ${port}\``), `a URL off the default port gets --port: ${without.out}`);
  assert.match(help.out, /quotemem serve \[options\] +run the memory service on 127\.0\.0\.1:8765/);
  assert.match(help.out, /quotemem memories \[--user U\] \[--status S\] \[--json \| --html FILE\]\n +list what is remembered \(read-only\)/);
});

/** A fake `uvx` that counts the SIGINTs it gets and exits with 7 a while after the first. */
function slowUvx(home, { drainMs }) {
  const bin = path.join(home, 'slow-bin');
  fs.mkdirSync(bin, { recursive: true });
  const script = [
    `#!${process.execPath}`,
    "const fs = require('node:fs');",
    'let count = 0;',
    "process.on('SIGINT', () => {",
    '  count += 1;',
    "  fs.writeFileSync(process.env.QM_FAKE_UVX_INTS, String(count));",
    `  if (count === 1) setTimeout(() => process.exit(7), ${drainMs});`,
    '});',
    "fs.writeFileSync(process.env.QM_FAKE_UVX_READY, 'ready');",
    'setInterval(() => {}, 1000);',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(bin, 'uvx'), script, { mode: 0o755 });
  return bin;
}

test('service commands: a SIGINT sent to the wrapper alone reaches the service once', { skip: process.platform === 'win32' ? 'POSIX signals' : false }, async () => {
  const home = freshHome();
  const bin = slowUvx(home, { drainMs: 1500 });
  const ints = path.join(home, 'uvx-ints');
  const ready = path.join(home, 'uvx-ready');
  const env = { PATH: `${bin}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home, QM_FAKE_UVX_INTS: ints, QM_FAKE_UVX_READY: ready };
  const wrapper = spawn(process.execPath, [CLI, 'serve'], { env, cwd: path.join(home, 'project'), stdio: ['ignore', 'pipe', 'pipe'] });
  const exited = new Promise((resolve) => wrapper.on('exit', (code, signal) => resolve({ code, signal })));
  await waitFor(() => fs.existsSync(ready), 8000, 'the fake service started');
  // Only the wrapper is signalled (kill -INT <pid>, a supervisor): twice.
  wrapper.kill('SIGINT');
  await new Promise((resolve) => setTimeout(resolve, 100));
  wrapper.kill('SIGINT');
  await waitFor(() => fs.existsSync(ints), 8000, 'the SIGINT passed on');
  const { code, signal } = await exited;
  assert.equal(fs.readFileSync(ints, 'utf8'), '1', 'passed on exactly once');
  assert.equal(signal, null);
  assert.equal(code, 7, "the wrapper exits with the service's code");
});

test('service commands: a Ctrl-C the service got itself is not passed on again', { skip: process.platform === 'win32' ? 'POSIX signals' : false }, async () => {
  const home = freshHome();
  const bin = slowUvx(home, { drainMs: 100 });
  const ints = path.join(home, 'uvx-ints');
  const ready = path.join(home, 'uvx-ready');
  const env = { PATH: `${bin}${path.delimiter}${process.env.PATH}`, HOME: home, USERPROFILE: home, QM_FAKE_UVX_INTS: ints, QM_FAKE_UVX_READY: ready };
  // Its own process group, like a terminal's foreground job.
  const wrapper = spawn(process.execPath, [CLI, 'serve'], { env, cwd: path.join(home, 'project'), stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const exited = new Promise((resolve) => wrapper.on('exit', (code, signal) => resolve({ code, signal })));
  await waitFor(() => fs.existsSync(ready), 8000, 'the fake service started');
  process.kill(-wrapper.pid, 'SIGINT');
  const { code } = await exited;
  assert.equal(code, 7);
  assert.equal(fs.readFileSync(ints, 'utf8'), '1', 'the service drained within the grace, so nothing more was sent');
});
