#!/usr/bin/env node
// Where the adapters find the service's token without it in an agent's
// environment (core/token.mjs), and that no doc or install snippet exports
// the token where an agent starts: the agent's shell tool inherits that
// environment, so the model could print it.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { after, test } from 'node:test';

import { CORE, TOKEN, memoryLog, stubServer, tempDir } from './helpers.mjs';

const core = await import(pathToFileURL(path.join(CORE, 'index.mjs')).href);
const { HookMemory, resolveSettings, resolveToken, runFlusher, serviceTokenFile } = core;

const ROOT = path.resolve(CORE, '..', '..');
const servers = new Set();
after(async () => { await Promise.all([...servers].map((s) => s.close())); });

function homeWith(files) {
  const home = tempDir('qm-token-home-');
  for (const [rel, text] of Object.entries(files)) {
    const file = path.join(home, rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, text, { mode: 0o600 });
  }
  return home;
}

const LOCAL = 'http://127.0.0.1:8765';

test('token order: the environment, then QUOTEMEM_TOKEN_FILE, then the local service\'s own <data-dir>/token', () => {
  const home = homeWith({
    '.local/share/quotemem/token': 'default-token\n',
    'xdg/quotemem/token': 'xdg-token\n',
    'data/token': ' data-token \n',
    'named': 'named-token\n',
  });
  assert.deepEqual(resolveToken({ HOME: home, USERPROFILE: home }, LOCAL), { token: 'default-token', source: 'service', problem: null });
  assert.equal(serviceTokenFile({ HOME: home, USERPROFILE: home }), path.join(home, '.local', 'share', 'quotemem', 'token'));
  assert.equal(resolveToken({ HOME: home, USERPROFILE: home, XDG_DATA_HOME: path.join(home, 'xdg') }, LOCAL).token, 'xdg-token');
  assert.equal(resolveToken({ HOME: home, USERPROFILE: home, XDG_DATA_HOME: path.join(home, 'xdg'), QUOTEMEM_DATA_DIR: '~/data' }, LOCAL).token, 'data-token');
  for (const url of ['http://localhost:8765', 'http://[::1]:8765']) assert.equal(resolveToken({ HOME: home, USERPROFILE: home }, url).source, 'service', url);
  // A named file, `~` or relative to the home folder, wins over the service's.
  assert.deepEqual(resolveToken({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN_FILE: '~/named' }, LOCAL), { token: 'named-token', source: 'file', problem: null });
  assert.equal(resolveToken({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN_FILE: 'named' }, 'https://memory.example.org').token, 'named-token');
  // The environment still wins (the service's own order), never needed.
  assert.deepEqual(resolveToken({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN: 'env-token', QUOTEMEM_TOKEN_FILE: '~/named' }, LOCAL), { token: 'env-token', source: 'env', problem: null });
  assert.equal(resolveToken({ HOME: home, USERPROFILE: home, QUOTEMEM_HTTP_AUTH_SECRET: 'secret-token' }, LOCAL).token, 'secret-token');
  // A named file that cannot be read is a problem, not a fall back to the service's file.
  const missing = resolveToken({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN_FILE: '~/no-such-file' }, LOCAL);
  assert.equal(missing.token, '');
  assert.match(missing.problem, /^QUOTEMEM_TOKEN_FILE cannot be read/);
  // The service's own file is for a service on this computer only.
  assert.deepEqual(resolveToken({ HOME: home, USERPROFILE: home }, 'https://memory.example.org'), { token: '', source: null, problem: null });
  assert.deepEqual(resolveToken({ HOME: homeWith({}) }, LOCAL), { token: '', source: null, problem: null });
});

test('settings: the token source is kept, never the token in a message', () => {
  const home = homeWith({ '.local/share/quotemem/token': `${TOKEN}\n` });
  const local = resolveSettings({}, { HOME: home, USERPROFILE: home });
  assert.equal(local.problem, null);
  assert.equal(local.token, TOKEN);
  assert.equal(local.tokenSource, 'service');
  assert.ok(!JSON.stringify(core.publicSettings(local)).includes(TOKEN), 'a child process gets the source, not the token');
  const remote = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.example.org' });
  assert.match(remote.problem, /^no token; .*set QUOTEMEM_TOKEN_FILE to that file$/);
  const unreadable = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN_FILE: '/no/such/token' });
  assert.match(unreadable.problem, /^QUOTEMEM_TOKEN_FILE cannot be read/);
  for (const settings of [remote, unreadable]) assert.ok(!settings.problem.includes(TOKEN));
});

test('a hook with no token in its environment saves through the service\'s own token file', async () => {
  const server = await stubServer({ '/memory/add': () => ({ body: { success: true, outcome: { added: 1 } } }) });
  servers.add(server);
  const home = homeWith({ '.local/share/quotemem/token': `${TOKEN}\n` });
  const env = { HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: path.join(home, 'state'), QUOTEMEM_CAPTURE_EVERY: '1' };
  const settings = resolveSettings({}, env);
  assert.equal(settings.problem, null);
  const memory = new HookMemory({ settings, log: memoryLog(), host: 'test-host', env });
  memory.flush = () => false;
  await memory.prompt({ session: 'S', turn: 't1', text: 'I live in Lisbon.' });
  await memory.reply({ session: 'S', turn: 't1', text: 'Noted.' });
  assert.equal(await runFlusher(settings, memoryLog()), 'empty');
  const adds = server.of('/memory/add');
  assert.equal(adds.length, 1);
  assert.equal(adds[0].headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(adds[0].body.user_id, 'quei');
});

// ------------------------------------------------- docs and install snippets

// Shapes that put the token itself in the environment an agent starts with:
// a shell export or assignment from the token file, a host's `-e` flag, a
// PowerShell, fish or Windows assignment, and a literal JSON env entry.
const EXPORTS = [
  /\bexport\s+QUOTEMEM_TOKEN=/,
  /\bQUOTEMEM_TOKEN=["']?\$\(/,
  /(?:^|\s)-e\s+QUOTEMEM_TOKEN=/,
  /\$env:QUOTEMEM_TOKEN\s*=/i,
  /\bset\s+-x\s+QUOTEMEM_TOKEN\b/,
  /\bsetx\s+QUOTEMEM_TOKEN\b/i,
  /["']QUOTEMEM_TOKEN["']\s*:/,
];

function shippedTextFiles() {
  const out = [path.join(ROOT, 'README.md')];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'test' || entry.name === 'node_modules' || entry.name.startsWith('.git')) continue;
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (entry.isFile() && fs.statSync(file).size < 1_000_000) out.push(file);
    }
  };
  walk(path.join(ROOT, 'integrations'));
  return out;
}

test('no doc, manifest or install snippet exports the token where an agent starts', () => {
  const files = shippedTextFiles();
  assert.ok(files.length > 50, `scanned ${files.length} files`);
  const found = [];
  for (const file of files) {
    const text = fs.readFileSync(file, 'utf8');
    for (const pattern of EXPORTS) {
      const match = pattern.exec(text);
      if (match) found.push(`${path.relative(ROOT, file)}: ${match[0]}`);
    }
  }
  assert.deepEqual(found, []);
  // Each setup guide names the file setting instead.
  for (const guide of ['README.md', 'integrations/claude-code/README.md', 'integrations/codex/README.md', 'integrations/dsh/README.md', 'integrations/opencode/README.md', 'integrations/hooks/README.md']) {
    assert.match(fs.readFileSync(path.join(ROOT, guide), 'utf8'), /QUOTEMEM_TOKEN_FILE/, guide);
  }
});

test('the install hints point at the token file, never at the token', async () => {
  const { install, toolList } = await import(pathToFileURL(path.join(ROOT, 'integrations', 'cli', 'install.mjs')).href);
  const home = homeWith({});
  for (const line of toolList()) {
    const tool = line.trim().split(/\s+/)[0];
    const result = install(tool, { write: false, node: process.execPath, project: home, args: { 'codex-home': path.join(home, '.codex'), dir: path.join(home, 'cline') } });
    const text = result.lines.join('\n');
    assert.ok(!EXPORTS.some((pattern) => pattern.test(text)), `${tool}: ${text}`);
    assert.match(text, /QUOTEMEM_TOKEN_FILE/, tool);
    assert.doesNotMatch(text, /export QUOTEMEM_TOKEN\b(?!_FILE)/, tool);
  }
});
