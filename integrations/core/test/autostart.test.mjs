#!/usr/bin/env node
// Starting the local memory service on demand (core/autostart.mjs), with a
// stub `uvx` that records each start and plays the service. Every test uses a
// temp HOME, so nothing outside the temp folder is read or written, and no
// real uvx or service is ever started.
//
//   node --test integrations/core/test/autostart.test.mjs
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, HIT, memoryLog, sleep, tempDir, waitFor } from './helpers.mjs';

const auto = await import(pathToFileURL(path.join(CORE, 'autostart.mjs')).href);
const { resolveSettings } = await import(pathToFileURL(path.join(CORE, 'config.mjs')).href);
const { resolveToken, serviceToken } = await import(pathToFileURL(path.join(CORE, 'token.mjs')).href);
const { postJson } = await import(pathToFileURL(path.join(CORE, 'http.mjs')).href);
const { HookMemory } = await import(pathToFileURL(path.join(CORE, 'hooks.mjs')).href);
const { SETUP_HEADER, UNAVAILABLE_HEADER } = await import(pathToFileURL(path.join(CORE, 'format.mjs')).href);
// What a user should never see in a reply when the service is down: the
// notes the model relays name no command, variable, address or error code.
const NOISE = /ECONNREFUSED|not reachable|无法连接|连不上|拒绝连接|quotemem serve|aml (agent )?serve|QUOTEMEM_|quotemem_|127\.0\.0\.1:\d+|\bhook/i;
const CLI = path.resolve(CORE, '..', 'cli', 'quotemem.mjs');
const SERVER = path.resolve(CORE, '..', 'dsh', 'server.mjs');

const root = tempDir('qm-auto-');
const windows = process.platform === 'win32';
// On Windows a start waits for PowerShell (spawnDetached), so a call that
// starts the service returns within this, and the stub runs this much later.
const AT_ONCE_MS = windows ? 6000 : 1500;
const LAUNCH_MS = windows ? 10_000 : 3000;
const started = new Set();
const boxes = [];
// Every stub the tests started, tracked or not: a detached stub left running
// would outlive the test run.
after(() => {
  for (const box of boxes) track(box);
  for (const pid of started) {
    try { process.kill(-pid, 'SIGKILL'); } catch { try { process.kill(pid, 'SIGKILL'); } catch { /* gone */ } }
  }
});

// The stub `uvx`: records its pid and arguments, then (after STUB_DELAY_MS)
// plays the service on the port it was given: it writes its token to the
// data folder and answers /health and /memory/search for that token only.
const STUB_HIT = JSON.stringify({ data: [HIT('01JX0000000000000000000009', 'I live in Lisbon.', { score: 0.9 })] });
const STUB = `const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const args = process.argv.slice(2);
fs.appendFileSync(process.env.STUB_RECORD, JSON.stringify({ pid: process.pid, args }) + '\\n');
const port = args.includes('--port') ? Number(args[args.indexOf('--port') + 1]) : 8765;
setTimeout(() => {
  if (process.env.STUB_MODE === 'hang') { setInterval(() => {}, 1000); return; }
  const dataDir = process.env.QUOTEMEM_DATA_DIR;
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(path.join(dataDir, 'token'), 'stub-service-token-77\\n', { mode: 0o600 });
  http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      const ok = req.headers.authorization === 'Bearer stub-service-token-77';
      if (req.url === '/health') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"status":"ok"}'); return; }
      if (!ok) { res.writeHead(401, { 'content-type': 'application/json' }); res.end('{"code":"authentication_failed"}'); return; }
      if (req.url === '/memory/search') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(${JSON.stringify(STUB_HIT)});
        return;
      }
      res.writeHead(202, { 'content-type': 'application/json' }); res.end('{}');
    });
  }).listen(port, '127.0.0.1');
}, Number(process.env.STUB_DELAY_MS || 0));
`;

// Windows: QuoteMem runs uvx.exe only (a .cmd needs a shell), so the stub
// there is a small console program, built once with the C# compiler
// Windows ships (PowerShell's Add-Type). It runs Node on uvx.cjs, next to
// it, with its own arguments, and exits with Node's code.
const SHIM_SOURCE = `using System;
using System.Diagnostics;
using System.IO;
using System.Text;
public static class UvxStub {
  public static int Main(string[] args) {
    StringBuilder line = new StringBuilder(Quote(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "uvx.cjs")));
    foreach (string arg in args) { line.Append(' '); line.Append(Quote(arg)); }
    ProcessStartInfo info = new ProcessStartInfo(@"${process.execPath.replace(/"/g, '""')}", line.ToString());
    info.UseShellExecute = false;
    using (Process child = Process.Start(info)) { child.WaitForExit(); return child.ExitCode; }
  }
  static string Quote(string arg) {
    if (arg.Length > 0 && arg.IndexOfAny(new char[] { ' ', '\\t', '"' }) < 0) return arg;
    StringBuilder out_ = new StringBuilder("\\"");
    int slashes = 0;
    foreach (char c in arg) {
      if (c == '\\\\') { slashes++; continue; }
      out_.Append('\\\\', c == '"' ? slashes * 2 + 1 : slashes);
      slashes = 0;
      out_.Append(c);
    }
    out_.Append('\\\\', slashes * 2);
    out_.Append('"');
    return out_.ToString();
  }
}
`;
let shim = null;
function windowsShim() {
  if (shim) return shim;
  const source = path.join(root, 'uvx-stub.cs');
  const exe = path.join(root, 'uvx-stub.exe');
  fs.writeFileSync(source, SHIM_SOURCE);
  const build = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
    'Add-Type -TypeDefinition ([IO.File]::ReadAllText($env:SHIM_SOURCE)) -OutputAssembly $env:SHIM_EXE -OutputType ConsoleApplication'],
  { env: { ...process.env, SHIM_SOURCE: source, SHIM_EXE: exe }, encoding: 'utf8', windowsHide: true, timeout: 120_000 });
  if (build.status !== 0 || !fs.existsSync(exe)) throw new Error(`could not build the uvx stub: ${build.stderr || build.error?.code}`);
  shim = exe;
  return exe;
}

/** The stub uvx in `bin` that runs `script` (Node, CommonJS); returns its path. */
function writeUvx(bin, script) {
  if (!windows) {
    fs.writeFileSync(path.join(bin, 'uvx'), `#!${process.execPath}\n${script}`, { mode: 0o755 });
    return path.join(bin, 'uvx');
  }
  fs.copyFileSync(windowsShim(), path.join(bin, 'uvx.exe'));
  fs.writeFileSync(path.join(bin, 'uvx.cjs'), script);
  return path.join(bin, 'uvx.exe');
}

// Windows keeps no POSIX mode bits (setup restricts nothing by mode there).
function assertMode(file, mode, message) {
  if (!windows) assert.equal(fs.statSync(file).mode & 0o777, mode, message);
}

let seq = 0;
/** A temp HOME with a stub uvx; `setup` writes ~/.quotemem/service.env. */
function sandbox({ setup = true, serviceEnv = {} } = {}) {
  seq += 1;
  const home = path.join(root, `home-${seq}`);
  const bin = path.join(home, 'bin');
  fs.mkdirSync(bin, { recursive: true });
  const uvx = writeUvx(bin, STUB);
  const record = path.join(home, 'starts.jsonl');
  const dataDir = path.join(home, 'data');
  boxes.push({ record });
  if (setup) {
    fs.mkdirSync(path.join(home, '.quotemem'), { recursive: true, mode: 0o700 });
    const lines = Object.entries({ STUB_RECORD: record, QUOTEMEM_DATA_DIR: dataDir, QUOTEMEM_CHAT_API_KEY: 'sk-never-printed-1', ...serviceEnv })
      .map(([k, v]) => `${k}=${v}`);
    fs.writeFileSync(path.join(home, '.quotemem', 'service.env'), `${lines.join('\n')}\n`, { mode: 0o600 });
  }
  return { home, bin, uvx, record, dataDir };
}

function starts(box) {
  try {
    return fs.readFileSync(box.record, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
  } catch {
    return [];
  }
}

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

function envOf(box, port, extra = {}) {
  return {
    HOME: box.home, USERPROFILE: box.home,
    PATH: `${box.bin}${path.delimiter}${process.env.PATH}`,
    QUOTEMEM_URL: `http://127.0.0.1:${port}`,
    QUOTEMEM_USER_ID: 'quei',
    QUOTEMEM_STATE_DIR: path.join(box.home, 'state'),
    ...extra,
  };
}

function track(box) {
  for (const { pid } of starts(box)) started.add(pid);
}

/** One plugin call in its own process (a hook): POST /memory/search, print what happened. */
const CALLER = `
const [core, envJson] = process.argv.slice(1);
const env = JSON.parse(envJson);
Object.assign(process.env, env);
const { resolveSettings } = await import(core + '/config.mjs');
const { postJson } = await import(core + '/http.mjs');
const settings = resolveSettings({}, process.env, {});
const t0 = Date.now();
try {
  const r = await postJson(settings, '/memory/search', { user_id: 'quei', query: 'where do I live?' });
  console.log(JSON.stringify({ status: r.status, ms: Date.now() - t0 }));
} catch (e) {
  console.log(JSON.stringify({ error: e.code, autostart: e.autostart ?? null, ms: Date.now() - t0 }));
}
`;

function caller(env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--input-type=module', '-e', CALLER, pathToFileURL(CORE).href, JSON.stringify(env)], { env, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.on('error', reject);
    child.on('exit', () => {
      try { resolve(JSON.parse(out.trim().split('\n').at(-1))); } catch { resolve({ raw: out }); }
    });
  });
}

test('a refused call starts the service once, even with 5 callers at the same time; later calls work', async () => {
  const box = sandbox({ serviceEnv: { STUB_DELAY_MS: 1500 } });
  const port = await freePort();
  const env = envOf(box, port);
  const results = await Promise.all(Array.from({ length: 5 }, () => caller(env)));
  track(box);
  for (const r of results) {
    assert.equal(r.error, 'ECONNREFUSED', JSON.stringify(r));
    assert.ok(['started', 'starting'].includes(r.autostart), JSON.stringify(r));
    assert.ok(r.ms < AT_ONCE_MS, `a caller waited ${r.ms} ms`);
  }
  assert.equal(results.filter((r) => r.autostart === 'started').length, 1, JSON.stringify(results));
  if (windows) await waitFor(() => starts(box).length > 0, LAUNCH_MS, 'the stub to start');
  await sleep(300);
  const recorded = starts(box);
  assert.equal(recorded.length, 1, 'exactly one start');
  assert.deepEqual(recorded[0].args, ['--from', `quotemem==${auto.SERVICE_RELEASE}`, 'quotemem', 'serve', '--port', String(port)]);
  const lock = JSON.parse(fs.readFileSync(path.join(box.home, '.quotemem', 'service.lock'), 'utf8'));
  // The lock records the started process: the stub, or on Windows the cmd that runs it.
  if (!windows) assert.equal(lock.pid, recorded[0].pid);
  assert.ok(Number.isInteger(lock.pid));
  assert.equal(lock.url, `http://127.0.0.1:${port}`);
  const log = fs.readFileSync(path.join(box.home, '.quotemem', 'service.log'), 'utf8');
  assert.match(log, /first start downloads the service/);
  assert.doesNotMatch(log, /sk-never-printed/);
  // Once it answers, a new call (no token configured) reads the service's token.
  await waitFor(async () => (await caller(env)).status === 200, 8000, 'the service to answer');
  assert.equal(starts(box).length, 1, 'still one start');
  const status = spawnSync(process.execPath, [CLI, 'status'], { env, encoding: 'utf8' });
  assert.equal(status.status, 0, status.stdout + status.stderr);
  assert.match(status.stdout, /memory service: running/);
  assert.match(status.stdout, new RegExp(`pid: ${lock.pid}`));
  const stop = spawnSync(process.execPath, [CLI, 'stop'], { env, encoding: 'utf8' });
  assert.equal(stop.status, 0, stop.stdout + stop.stderr);
  assert.match(stop.stdout, /stopped the memory service/);
  assert.equal(fs.existsSync(path.join(box.home, '.quotemem', 'service.lock')), false);
});

test('a stale lock (its process is gone) is replaced; a start that just died is not repeated at once', async () => {
  const box = sandbox({ serviceEnv: { STUB_MODE: 'hang' } });
  const port = await freePort();
  const dead = spawnSync(process.execPath, ['-e', 'console.log(process.pid)'], { encoding: 'utf8' });
  const deadPid = Number(dead.stdout.trim());
  const lockFile = path.join(box.home, '.quotemem', 'service.lock');
  // Died seconds ago: the backoff holds.
  fs.writeFileSync(lockFile, JSON.stringify({ pid: deadPid, startedAt: Date.now() - 5000, nonce: 'x' }), { mode: 0o600 });
  const settings = resolveSettings({}, envOf(box, port), {});
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env: envOf(box, port) }).state, 'backoff');
  assert.equal(starts(box).length, 0);
  // Died long ago: replaced by a new start.
  fs.writeFileSync(lockFile, JSON.stringify({ pid: deadPid, startedAt: Date.now() - 10 * 60_000, nonce: 'x' }), { mode: 0o600 });
  auto.resetAutostartMemo();
  const result = auto.ensureService(settings, { env: envOf(box, port) });
  assert.equal(result.state, 'started');
  await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the stub to start');
  track(box);
  assert.equal(JSON.parse(fs.readFileSync(lockFile, 'utf8')).pid, result.pid);
  // While it runs (still starting), no second start.
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env: envOf(box, port) }).state, 'starting');
  await sleep(200);
  assert.equal(starts(box).length, 1);
  await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
});

test('without service.env nothing starts, and the session gets one note pointing to setup', async () => {
  const box = sandbox({ setup: false });
  const port = await freePort();
  // A token from an earlier hand-started service: the plugin is on, the service is down.
  fs.mkdirSync(path.join(box.home, '.local', 'share', 'quotemem'), { recursive: true });
  fs.writeFileSync(path.join(box.home, '.local', 'share', 'quotemem', 'token'), 'old-token-1\n', { mode: 0o600 });
  const env = envOf(box, port);
  const settings = resolveSettings({}, env, {});
  assert.equal(settings.problem, null);
  assert.equal(settings.token, 'old-token-1', 'the service token is read from its data folder');
  assert.equal(settings.localService.autostart, false);
  const lines = [];
  const memory = new HookMemory({ settings, log: memoryLog(lines), host: 'test' });
  auto.resetAutostartMemo();
  const first = await memory.prompt({ session: 's1', turn: 't1', text: 'Where do I live?' });
  assert.equal(first.kind, 'unavailable');
  assert.ok(first.context.includes(UNAVAILABLE_HEADER));
  assert.ok(first.context.includes(SETUP_HEADER));
  assert.match(first.context, /npx quotemem setup/);
  assert.doesNotMatch(first.context, NOISE);
  auto.resetAutostartMemo();
  const second = await memory.prompt({ session: 's1', turn: 't2', text: 'What did I say about my job?' });
  assert.ok(!String(second.context ?? '').includes(SETUP_HEADER), 'the note is given once per session');
  assert.equal(starts(box).length, 0);
  assert.equal(fs.existsSync(path.join(box.home, '.quotemem', 'service.lock')), false);
  assert.ok(lines.some((l) => /npx quotemem setup/.test(l)), 'the note is logged too');
});

test('no token and no setup: automatic memory stays off with a problem that names setup', () => {
  const box = sandbox({ setup: false });
  const settings = resolveSettings({}, envOf(box, 8765), {});
  assert.match(settings.problem, /npx quotemem setup/);
  const remote = resolveSettings({}, { HOME: box.home, USERPROFILE: box.home, QUOTEMEM_URL: 'https://memory.example.com' }, {});
  assert.match(remote.problem, /QUOTEMEM_TOKEN/);
  assert.equal(remote.localService, null);
});

test('set up but never started: no token is needed until the service creates one', () => {
  const box = sandbox();
  const settings = resolveSettings({}, envOf(box, 8765), {});
  assert.equal(settings.problem, null);
  assert.equal(settings.token, '');
  assert.equal(settings.localService.autostart, true);
  assert.equal(settings.localService.dataDir, box.dataDir, 'QUOTEMEM_DATA_DIR of service.env decides where the token is');
  fs.mkdirSync(box.dataDir, { recursive: true });
  fs.writeFileSync(path.join(box.dataDir, 'token'), 'later-token\n');
  assert.equal(serviceToken({ HOME: box.home, USERPROFILE: box.home }, settings.localService), 'later-token');
  // One token reader (token.mjs) for the started service too: its data folder
  // and service.env decide, not this environment's; QUOTEMEM_TOKEN_FILE and
  // QUOTEMEM_TOKEN still come first.
  const url = 'http://127.0.0.1:8765';
  const local = auto.localService(envOf(box, 8765), url);
  assert.deepEqual(resolveToken({ ...envOf(box, 8765), QUOTEMEM_DATA_DIR: path.join(box.home, 'elsewhere') }, url, local), { token: 'later-token', source: 'service', problem: null });
  fs.writeFileSync(path.join(box.home, 'named-token'), 'named-token\n');
  assert.equal(resolveToken({ ...envOf(box, 8765), QUOTEMEM_TOKEN_FILE: '~/named-token' }, url, local).token, 'named-token');
  assert.equal(resolveSettings({}, { ...envOf(box, 8765), QUOTEMEM_TOKEN: 'env-token' }, {}).token, 'env-token');
  fs.appendFileSync(path.join(box.home, '.quotemem', 'service.env'), 'QUOTEMEM_TOKEN=service-env-token\n');
  assert.equal(resolveSettings({}, envOf(box, 8765), {}).token, 'service-env-token', 'the token service.env gives the started service');
});

test('QUOTEMEM_AUTOSTART=0 never starts the service', async () => {
  const box = sandbox();
  const port = await freePort();
  const r = await caller(envOf(box, port, { QUOTEMEM_AUTOSTART: '0', QUOTEMEM_TOKEN: 't0' }));
  assert.equal(r.error, 'ECONNREFUSED');
  assert.equal(r.autostart, 'disabled');
  assert.equal(starts(box).length, 0);
});

test('a URL that is not on this computer never starts anything', async () => {
  const box = sandbox();
  const env = envOf(box, 8765, { QUOTEMEM_URL: 'http://192.0.2.10:8765', QUOTEMEM_TOKEN: 't1' });
  const settings = resolveSettings({}, env, {});
  assert.equal(settings.localService, null);
  assert.equal(auto.ensureService(settings, { env }).state, 'remote');
  assert.equal(auto.ensureService({ ...settings, localService: null }, { env }).state, 'remote');
  // A refused connection to such a URL (simulated through a non-loopback name) starts nothing.
  const lan = resolveSettings({}, { ...env, QUOTEMEM_URL: 'http://memory.invalid:1' }, {});
  await assert.rejects(postJson(lan, '/memory/search', {}));
  assert.equal(starts(box).length, 0);
  assert.equal(fs.existsSync(path.join(box.home, '.quotemem', 'service.lock')), false);
});

test('stop stops only the recorded process, after checking its command line', async () => {
  const box = sandbox();
  const lockFile = path.join(box.home, '.quotemem', 'service.lock');
  // An unrelated process whose pid ended up in the lock (a reused pid).
  const other = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { detached: true, stdio: 'ignore', windowsHide: true });
  started.add(other.pid);
  fs.writeFileSync(lockFile, JSON.stringify({ pid: other.pid, startedAt: Date.now(), nonce: 'n1' }), { mode: 0o600 });
  const refusedStop = await auto.stopService({ HOME: box.home, USERPROFILE: box.home }, { waitMs: 500 });
  assert.equal(refusedStop.stopped, false);
  assert.match(refusedStop.message, /not running/);
  assert.equal(auto.serviceState({ HOME: box.home, USERPROFILE: box.home }).record, null, 'the lock of a reused pid is dropped');
  process.kill(other.pid, 0); // still alive
  // The recorded service: a process whose command line is `… quotemem serve`.
  const service = spawn(box.uvx, ['--from', 'quotemem==0', 'quotemem', 'serve'], {
    detached: true, stdio: 'ignore', windowsHide: true, env: { ...process.env, STUB_RECORD: box.record, STUB_MODE: 'hang' },
  });
  started.add(service.pid);
  fs.writeFileSync(lockFile, JSON.stringify({ pid: service.pid, startedAt: Date.now(), nonce: 'n2' }), { mode: 0o600 });
  await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the stub service');
  const stop = await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
  assert.equal(stop.stopped, true, stop.message);
  await waitFor(() => { try { process.kill(service.pid, 0); return false; } catch { return true; } }, 3000, 'the service to exit');
  process.kill(other.pid, 0); // the unrelated process is untouched
  process.kill(other.pid, 'SIGKILL');
});

test('setup writes service.env 600 in a 700 folder and never prints a key', async () => {
  const box = sandbox({ setup: false });
  const env = { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH, QUOTEMEM_CHAT_API_KEY: 'sk-chat-secret-AAAA', QUOTEMEM_EMBED_API_KEY: 'sk-embed-secret-BBBB' };
  const run = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--no-check', '--chat-url', 'https://api.deepseek.com/v1', '--chat-model', 'deepseek-chat',
    '--chat-structured-output', 'json_object', '--embed-url', 'https://api.openai.com/v1', '--embed-model', 'text-embedding-3-small'], { env, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const all = run.stdout + run.stderr;
  assert.doesNotMatch(all, /sk-chat-secret|sk-embed-secret/);
  const file = path.join(box.home, '.quotemem', 'service.env');
  assertMode(file, 0o600);
  assertMode(path.dirname(file), 0o700);
  const saved = auto.parseServiceEnv(fs.readFileSync(file, 'utf8'));
  assert.equal(saved.QUOTEMEM_CHAT_API_KEY, 'sk-chat-secret-AAAA');
  assert.equal(saved.QUOTEMEM_EMBED_MODEL, 'text-embedding-3-small');
  assert.equal(saved.QUOTEMEM_CHAT_STRUCTURED_OUTPUT, 'json_object');

  // Asked interactively (answers on stdin): custom, keys typed, an extra line kept.
  fs.appendFileSync(file, 'QUOTEMEM_DAILY_CHAT_CALLS=50\nQUOTEMEM_CHAT_REASONING=effort\n');
  fs.chmodSync(path.dirname(file), 0o755);
  // Menu: 1 default, 2 fast, 3 custom, 4 keep. Custom: chat URL, model, structured
  // output (Enter: json_schema), thinking (Enter: the provider's default); embedding
  // URL (Enter: the chat URL), model (Enter keeps the saved one); the review (Enter:
  // standard); then the chat key and the embedding key (Enter keeps the saved one).
  const answers = `${['3', 'https://api.openai.com/v1', 'gpt-4o-mini', '', '', '', '', '', 'sk-openai-chat-CCCC', ''].join('\n')}\n`;
  const asked = spawnSync(process.execPath, [CLI, 'setup', '--no-check'], { env: { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH }, input: answers, encoding: 'utf8' });
  assert.equal(asked.status, 0, asked.stdout + asked.stderr);
  assert.doesNotMatch(asked.stdout + asked.stderr, /sk-openai-chat|sk-chat-secret|sk-embed-secret/);
  assert.match(asked.stdout, /recommended, the competition configuration/);
  const again = auto.parseServiceEnv(fs.readFileSync(file, 'utf8'));
  assert.equal(again.QUOTEMEM_CHAT_BASE_URL, 'https://api.openai.com/v1');
  assert.equal(again.QUOTEMEM_CHAT_MODEL, 'gpt-4o-mini');
  assert.equal(again.QUOTEMEM_CHAT_STRUCTURED_OUTPUT, 'json_schema');
  assert.equal(again.QUOTEMEM_CHAT_API_KEY, 'sk-openai-chat-CCCC');
  assert.equal(again.QUOTEMEM_EMBED_BASE_URL, 'https://api.openai.com/v1');
  assert.equal(again.QUOTEMEM_EMBED_MODEL, 'text-embedding-3-small');
  assert.equal(again.QUOTEMEM_EMBED_API_KEY, 'sk-embed-secret-BBBB', 'Enter keeps the saved key');
  assert.equal(again.QUOTEMEM_CHAT_REASONING, 'none', "the provider's default: no reasoning parameters, not the service's empty-means-off");
  assert.equal(again.QUOTEMEM_REVIEW, 'standard');
  assert.equal(again.QUOTEMEM_DAILY_CHAT_CALLS, '50', 'a line setup does not write is kept');
  assertMode(path.dirname(file), 0o700, 'the folder is tightened');

  // Missing values with --yes: refused, nothing printed but the names.
  const missing = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--no-check'], { env: { HOME: path.join(box.home, 'nowhere'), PATH: process.env.PATH }, encoding: 'utf8' });
  assert.equal(missing.status, 2);
  assert.match(missing.stderr, /QUOTEMEM_CHAT_API_KEY/);
});

test('setup checks the settings through uvx with the saved file, without printing keys', () => {
  const box = sandbox({ setup: false });
  // A stub uvx for `quotemem check`: it sees the key in its environment and says so without printing it.
  writeUvx(box.bin, `const ok = process.env.QUOTEMEM_CHAT_API_KEY === 'sk-chat-secret-DDDD';\nconsole.log('stub check ' + process.argv.slice(2).join(' ') + (ok ? ' key-seen' : ' no-key'));\nprocess.exit(ok ? 0 : 3);\n`);
  const env = { HOME: box.home, USERPROFILE: box.home, PATH: `${box.bin}${path.delimiter}${process.env.PATH}`, QUOTEMEM_CHAT_API_KEY: 'sk-chat-secret-DDDD' };
  const run = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--chat-url', 'https://api.deepseek.com/v1', '--chat-model', 'deepseek-chat', '--embed-model', 'm'], { env, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stdout + run.stderr);
  assert.match(run.stdout, new RegExp(`stub check --from quotemem==${auto.SERVICE_RELEASE} quotemem check key-seen`));
  assert.match(run.stdout, /ok: the settings work/);
  assert.doesNotMatch(run.stdout + run.stderr, /sk-chat-secret/);
  // `quotemem check` on its own runs with the saved settings (the key is not in this environment).
  const check = spawnSync(process.execPath, [CLI, 'check'], { env: { HOME: box.home, USERPROFILE: box.home, PATH: env.PATH }, encoding: 'utf8' });
  assert.equal(check.status, 0, check.stdout + check.stderr);
  assert.match(check.stdout, /key-seen/);
});

test('a hook never waits for the start: the prompt returns at once and later prompts find the service', async () => {
  const box = sandbox({ serviceEnv: { STUB_DELAY_MS: 2500 } });
  const port = await freePort();
  const env = envOf(box, port);
  const saved = { PATH: process.env.PATH, HOME: process.env.HOME };
  process.env.PATH = env.PATH;
  try {
    const settings = resolveSettings({}, env, {});
    const memory = new HookMemory({ settings, log: memoryLog([]), host: 'test' });
    auto.resetAutostartMemo();
    const t0 = Date.now();
    const first = await memory.prompt({ session: 's1', turn: 't1', text: 'Where do I live?' });
    const took = Date.now() - t0;
    track(box);
    assert.ok(took < AT_ONCE_MS, `the prompt waited ${took} ms`);
    assert.equal(first.kind, 'unavailable');
    assert.ok(!first.context.includes(SETUP_HEADER), 'no setup note while it starts');
    await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the start');
    track(box);
    // The breaker did not open for a start in progress: the next question searches.
    let later = null;
    await waitFor(async () => {
      auto.resetAutostartMemo();
      later = await memory.prompt({ session: 's1', turn: `t${Date.now()}`, text: 'Where do I live now?' });
      return later.kind === 'recall';
    }, 10_000, 'a recall once the service answers');
    assert.match(later.context, /Lisbon/);
    assert.equal(starts(box).length, 1);
  } finally {
    Object.assign(process.env, saved);
    await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
  }
});

test('the MCP server starts the service on demand and tells the model it is starting', async () => {
  const box = sandbox({ serviceEnv: { STUB_MODE: 'hang' } });
  const port = await freePort();
  const env = envOf(box, port);
  const child = spawn(process.execPath, [SERVER], { env, stdio: ['pipe', 'pipe', 'pipe'] });
  let out = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } })}\n`);
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live?' } } })}\n`);
  try {
    await waitFor(() => out.includes('"id":2'), 8000, 'the tool answer');
    await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the start');
    track(box);
    const answer = JSON.parse(out.trim().split('\n').find((l) => l.includes('"id":2')));
    assert.match(JSON.stringify(answer), /starting on this computer/);
  } finally {
    child.kill();
    await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
  }
});

test('the service log is cut once it grows past its cap', () => {
  const box = sandbox();
  const log = path.join(box.home, '.quotemem', 'service.log');
  fs.writeFileSync(log, 'x'.repeat(auto.LOG_MAX_BYTES + 10));
  auto.capLog(log);
  assert.equal(fs.statSync(log).size, 0);
  assert.equal(fs.statSync(`${log}.1`).size, auto.LOG_MAX_BYTES / 2);
});

/** A running process whose command line is `uvx … quotemem serve` (the stub, hanging). */
function fakeService(box) {
  const child = spawn(box.uvx, ['--from', 'quotemem==0', 'quotemem', 'serve'], {
    detached: true, stdio: 'ignore', windowsHide: true, env: { ...process.env, STUB_RECORD: box.record, STUB_MODE: 'hang' },
  });
  started.add(child.pid);
  return child;
}

function gone(pid) {
  try { process.kill(pid, 0); return false; } catch { return true; }
}

test('a tool with another QUOTEMEM_URL port is told so once, and never waits for a start that is not its own', async () => {
  const box = sandbox();
  const [portA, portB] = [await freePort(), await freePort()];
  const lockFile = path.join(box.home, '.quotemem', 'service.lock');
  const service = fakeService(box);
  await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the stub service');
  const { bootTime } = await import(pathToFileURL(path.join(CORE, 'proc.mjs')).href);
  fs.writeFileSync(lockFile, JSON.stringify({ pid: service.pid, startedAt: Date.now(), nonce: 'nA', url: `http://127.0.0.1:${portA}`, boot: bootTime() }), { mode: 0o600 });
  const lines = [];
  const settingsB = resolveSettings({}, envOf(box, portB), {});
  auto.resetAutostartMemo();
  const other = auto.ensureService(settingsB, { env: envOf(box, portB), log: memoryLog(lines) });
  assert.equal(other.state, 'other-url');
  assert.equal(other.url, `http://127.0.0.1:${portA}`);
  assert.ok(lines.some((l) => l.includes(`127.0.0.1:${portA}`) && /same QUOTEMEM_URL/.test(l)), lines.join('\n'));
  assert.match(auto.setupNote('other-url'), /the same QuoteMem service address/);
  for (const state of ['no-setup', 'no-uvx', 'other-url']) assert.doesNotMatch(auto.setupNote(state), NOISE, state);
  // The same address (localhost names 127.0.0.1) is the running instance: starting.
  auto.resetAutostartMemo();
  const same = resolveSettings({}, envOf(box, portA, { QUOTEMEM_URL: `http://localhost:${portA}` }), {});
  assert.equal(auto.ensureService(same, { env: envOf(box, portA) }).state, 'starting');
  assert.equal(starts(box).length, 1, 'nothing more started');
  await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
  assert.ok(gone(service.pid));
});

test('the service gets its identity from service.env only, not from the tool that starts it', () => {
  const box = sandbox({ serviceEnv: { QUOTEMEM_DATA_DIR: 'store' } });
  const toolEnv = {
    HOME: box.home, USERPROFILE: box.home, PATH: '/usr/bin', LANG: 'C.UTF-8', LC_ALL: 'C.UTF-8', HTTPS_PROXY: 'http://proxy.invalid:1', no_proxy: 'localhost',
    UV_CACHE_DIR: '/tmp/uvc', TMPDIR: '/tmp',
    QUOTEMEM_TOKEN: 'tool-token', QUOTEMEM_HTTP_AUTH_SECRET: 'tool-secret',
    QUOTEMEM_DATA_DIR: '/elsewhere', XDG_DATA_HOME: '/xdg', OPENAI_API_KEY: 'sk-openai-env', DASHSCOPE_API_KEY: 'sk-dashscope-env',
    QUOTEMEM_URL: 'http://127.0.0.1:1', QUOTEMEM_FLUSHER_SETTINGS: '{}', RANDOM_TOOL_VAR: 'x',
  };
  const runEnv = auto.serviceEnvironment(toolEnv);
  for (const name of ['HOME', 'PATH', 'LANG', 'LC_ALL', 'HTTPS_PROXY', 'no_proxy', 'UV_CACHE_DIR', 'TMPDIR']) assert.equal(runEnv[name], toolEnv[name], name);
  for (const name of ['QUOTEMEM_TOKEN', 'QUOTEMEM_HTTP_AUTH_SECRET', 'XDG_DATA_HOME', 'OPENAI_API_KEY', 'DASHSCOPE_API_KEY', 'QUOTEMEM_URL', 'QUOTEMEM_FLUSHER_SETTINGS', 'RANDOM_TOOL_VAR']) {
    assert.equal(runEnv[name], undefined, name);
  }
  assert.equal(runEnv.QUOTEMEM_DATA_DIR, 'store', 'service.env wins');
  // The review is part of the store's identity: service.env's, never the tool's.
  assert.equal(auto.serviceEnvironment({ ...toolEnv, QUOTEMEM_REVIEW: 'full' }).QUOTEMEM_REVIEW, undefined, "the tool's review is not passed on");
  assert.equal(auto.serviceEnvironment({ ...toolEnv, QUOTEMEM_REVIEW: 'full' }, { QUOTEMEM_REVIEW: 'fast' }).QUOTEMEM_REVIEW, 'fast', 'the review of service.env is');
  assert.equal(runEnv.QUOTEMEM_CHAT_API_KEY, 'sk-never-printed-1');
  // The plugin looks for the token where the started service keeps it: a
  // relative folder is taken from ~/.quotemem, the folder the service runs in.
  const local = auto.localService(toolEnv, 'http://127.0.0.1:8765');
  assert.equal(local.dataDir, path.join(box.home, '.quotemem', 'store'));
  const plain = sandbox({ serviceEnv: { QUOTEMEM_DATA_DIR: '' } });
  assert.equal(auto.localService({ ...toolEnv, HOME: plain.home, USERPROFILE: plain.home }, 'http://127.0.0.1:8765').dataDir, path.join(plain.home, '.local', 'share', 'quotemem'),
    'an XDG_DATA_HOME or QUOTEMEM_DATA_DIR of the tool does not move it');
  // Windows: the home folder is USERPROFILE, as for the service.
  assert.equal(auto.homeOf({ HOME: '/msys/home/q', USERPROFILE: 'C:\\Users\\q' }, 'win32'), 'C:\\Users\\q');
  assert.equal(auto.homeOf({ HOME: '/home/q', USERPROFILE: 'C:\\Users\\q' }, 'linux'), '/home/q');
});

test('a recorded pid that another program reuses is replaced at once on POSIX, by a background check on Windows', async () => {
  const box = sandbox({ serviceEnv: { STUB_MODE: 'hang' } });
  const port = await freePort();
  const env = envOf(box, port);
  const settings = resolveSettings({}, env, {});
  const lockFile = path.join(box.home, '.quotemem', 'service.lock');
  const { bootTime } = await import(pathToFileURL(path.join(CORE, 'proc.mjs')).href);
  const other = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { detached: true, stdio: 'ignore', windowsHide: true });
  started.add(other.pid);
  const record = { pid: other.pid, startedAt: Date.now() - 2 * 60_000, nonce: 'nR', url: settings.url, boot: bootTime() };
  // Windows: the hook does not read the command line; it starts one detached check, at most every few minutes.
  fs.writeFileSync(lockFile, JSON.stringify(record), { mode: 0o600 });
  const calls = [];
  const spawnFn = (command, args, options) => { calls.push({ command, args, options }); return { pid: 1, on() {}, unref() {} }; };
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env, spawnFn, platform: 'win32' }).state, 'starting');
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env, spawnFn, platform: 'win32' }).state, 'starting');
  assert.equal(calls.length, 1, 'one background check');
  assert.equal(calls[0].command, process.execPath);
  assert.deepEqual(calls[0].args.slice(1), ['--check-lock', lockFile, 'nR', String(other.pid)]);
  assert.equal(calls[0].options.cwd, path.dirname(lockFile));
  // That check, run for real: the lock of the reused pid goes, the program stays.
  const check = spawnSync(process.execPath, calls[0].args, { encoding: 'utf8' });
  assert.equal(check.status, 0, check.stderr);
  assert.equal(fs.existsSync(lockFile), false);
  process.kill(other.pid, 0);
  // POSIX: the command line is read at once, the lock replaced and the service started.
  if (!windows) {
    fs.writeFileSync(lockFile, JSON.stringify({ ...record, startedAt: Date.now() }), { mode: 0o600 });
    auto.resetAutostartMemo();
    const result = auto.ensureService(settings, { env });
    assert.equal(result.state, 'started');
    await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the stub to start');
    track(box);
  }
  process.kill(other.pid, 0);
  process.kill(other.pid, 'SIGKILL');
  await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
});

test('a changed host name does not make the local service foreign', async () => {
  const box = sandbox({ serviceEnv: { STUB_MODE: 'hang' } });
  const port = await freePort();
  const env = envOf(box, port);
  const settings = resolveSettings({}, env, {});
  const lockFile = path.join(box.home, '.quotemem', 'service.lock');
  const { bootTime } = await import(pathToFileURL(path.join(CORE, 'proc.mjs')).href);
  const service = fakeService(box);
  await waitFor(() => starts(box).length === 1, LAUNCH_MS, 'the stub service');
  fs.writeFileSync(lockFile, JSON.stringify({ host: 'old-name.local', boot: bootTime(), pid: service.pid, startedAt: Date.now(), nonce: 'nH', url: settings.url }), { mode: 0o600 });
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env }).state, 'starting');
  assert.equal(auto.serviceState({ HOME: box.home, USERPROFILE: box.home }).running, true);
  const stop = await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
  assert.equal(stop.stopped, true, stop.message);
  assert.ok(gone(service.pid));
  // A dead instance recorded under the old name is replaced, not kept for hours.
  fs.writeFileSync(lockFile, JSON.stringify({ host: 'old-name.local', boot: bootTime(), pid: service.pid, startedAt: Date.now() - 10 * 60_000, nonce: 'nH2', url: settings.url }), { mode: 0o600 });
  auto.resetAutostartMemo();
  assert.equal(auto.ensureService(settings, { env }).state, 'started');
  await waitFor(() => starts(box).length === 2, LAUNCH_MS, 'the new start');
  track(box);
  await auto.stopService({ HOME: box.home, USERPROFILE: box.home });
});

test('a start that lost its lock, or could not record itself, stops the process it started', async () => {
  const box = sandbox();
  const port = await freePort();
  const env = envOf(box, port);
  const settings = resolveSettings({}, env, {});
  const dir = path.join(box.home, '.quotemem');
  const lockFile = path.join(dir, 'service.lock');
  const children = [];
  const sleeper = () => {
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { detached: true, stdio: 'ignore', windowsHide: true });
    started.add(child.pid);
    children.push(child);
    return child;
  };
  // Another starter took the lock while this one spawned: this one's process is stopped, the record kept.
  const taken = { pid: process.pid, nonce: 'theirs', at: Date.now() };
  auto.resetAutostartMemo();
  const lost = auto.ensureService(settings, { env, spawnFn: () => { const c = sleeper(); fs.writeFileSync(lockFile, JSON.stringify(taken)); return c; } });
  assert.equal(lost.state, 'starting');
  await waitFor(() => gone(children[0].pid), 3000, 'the extra process to stop');
  assert.equal(JSON.parse(fs.readFileSync(lockFile, 'utf8')).nonce, 'theirs');
  fs.unlinkSync(lockFile);
  // The record cannot be written: the started process is stopped too.
  // (POSIX: Windows does not make a folder unwritable by its mode.)
  if (windows) return;
  auto.resetAutostartMemo();
  let failed;
  try {
    failed = auto.ensureService(settings, { env, spawnFn: () => { const c = sleeper(); fs.chmodSync(dir, 0o500); return c; } });
  } finally {
    fs.chmodSync(dir, 0o700);
  }
  assert.equal(failed.state, 'failed');
  await waitFor(() => gone(children[1].pid), 3000, 'the unrecorded process to stop');
});

test('setup: Enter for the embedding key writes the chat key; keys never echo when only stdin is a terminal', async () => {
  const { cmdSetup } = await import(pathToFileURL(path.resolve(CORE, '..', 'cli', 'service.mjs')).href);
  const { PassThrough, Writable } = await import('node:stream');
  const box = sandbox({ setup: false });
  const collect = (isTTY) => {
    const stream = new Writable({ write(chunk, _encoding, done) { stream.text += chunk; done(); } });
    stream.text = '';
    stream.isTTY = isTTY;
    stream.columns = 80;
    return stream;
  };
  const input = new PassThrough();
  input.isTTY = true;
  let raw = false;
  input.setRawMode = (on) => { if (on) raw = true; return input; };
  const output = collect(false); // `setup | tee`
  const echo = collect(true);
  // Custom: chat URL, model, Enter, Enter; embedding: Enter (the chat URL), model; review: Enter; chat key, Enter (the chat key).
  const answers = ['3', 'https://api.openai.com/v1', 'gpt-4o-mini', '', '', '', 'text-embedding-3-small', '', 'sk-typed-chat-EEEE', ''];
  const env = { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH };
  const done = cmdSetup({ 'no-check': true }, { env, input, output, echo, release: auto.SERVICE_RELEASE });
  for (const answer of answers) {
    await sleep(30);
    input.write(`${answer}\r`);
  }
  assert.equal(await done, 0, output.text);
  assert.ok(raw, 'the terminal is in raw mode, so it echoes nothing itself');
  assert.match(echo.text, /gpt-4o-mini/, 'plain answers are echoed to stderr');
  assert.doesNotMatch(output.text + echo.text, /sk-typed-chat/);
  const saved = auto.parseServiceEnv(fs.readFileSync(path.join(box.home, '.quotemem', 'service.env'), 'utf8'));
  assert.equal(saved.QUOTEMEM_CHAT_API_KEY, 'sk-typed-chat-EEEE');
  assert.equal(saved.QUOTEMEM_EMBED_API_KEY, 'sk-typed-chat-EEEE', 'written out, so DASHSCOPE_API_KEY elsewhere cannot replace it');

  // --yes: the same, from the environment; another host needs its own key.
  const yes = sandbox({ setup: false });
  const same = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--no-check', '--chat-url', 'https://api.openai.com/v1', '--chat-model', 'gpt-4o-mini', '--embed-model', 'text-embedding-3-small'],
    { env: { HOME: yes.home, USERPROFILE: yes.home, PATH: process.env.PATH, QUOTEMEM_CHAT_API_KEY: 'sk-env-chat-FFFF', DASHSCOPE_API_KEY: 'sk-dash-GGGG' }, encoding: 'utf8' });
  assert.equal(same.status, 0, same.stderr);
  assert.equal(auto.parseServiceEnv(fs.readFileSync(path.join(yes.home, '.quotemem', 'service.env'), 'utf8')).QUOTEMEM_EMBED_API_KEY, 'sk-env-chat-FFFF');
  const otherHost = sandbox({ setup: false });
  const refusedRun = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--no-check', '--chat-url', 'https://api.deepseek.com/v1', '--chat-model', 'deepseek-chat',
    '--embed-url', 'https://api.openai.com/v1', '--embed-model', 'text-embedding-3-small'],
  { env: { HOME: otherHost.home, USERPROFILE: otherHost.home, PATH: process.env.PATH, QUOTEMEM_CHAT_API_KEY: 'sk-env-chat-FFFF' }, encoding: 'utf8' });
  assert.equal(refusedRun.status, 2);
  assert.match(refusedRun.stderr, /QUOTEMEM_EMBED_API_KEY/);
  assert.doesNotMatch(refusedRun.stdout + refusedRun.stderr, /sk-env-chat/);
});

test('setup presets: default is the competition configuration with one key, fast turns thinking off, custom keeps what is saved', async () => {
  const { CHAT_REASONING, PRESETS } = await import(pathToFileURL(path.resolve(CORE, '..', 'cli', 'service.mjs')).href);
  const dashscope = 'https://dashscope.aliyuncs.com/compatible-mode/v1';
  const competition = {
    QUOTEMEM_CHAT_BASE_URL: dashscope,
    QUOTEMEM_CHAT_MODEL: 'deepseek-v4.1-flash-1017',
    QUOTEMEM_CHAT_STRUCTURED_OUTPUT: 'json_object',
    QUOTEMEM_CHAT_MODEL_PATTERN: 'deepseek-v4\\.1-flash(?:-1017)?|deepseek-v4-1-flash-[0-9]{6}',
    QUOTEMEM_EMBED_BASE_URL: dashscope,
    QUOTEMEM_EMBED_MODEL: 'qwen3.7-text-embedding',
  };
  assert.deepEqual(PRESETS.default.values, { ...competition, QUOTEMEM_CHAT_REASONING: CHAT_REASONING.high });
  assert.deepEqual(PRESETS.fast.values, { ...competition, QUOTEMEM_CHAT_REASONING: CHAT_REASONING.off });
  const reported = new RegExp(`^(?:${competition.QUOTEMEM_CHAT_MODEL_PATTERN})$`);
  for (const id of ['deepseek-v4.1-flash-1017', 'deepseek-v4.1-flash', 'deepseek-v4-1-flash-261017']) assert.match(id, reported);
  for (const id of ['deepseek-v4-1-flash', 'deepseek-v4.1-flash-1018', 'deepseek-chat', 'xdeepseek-v4.1-flash']) assert.doesNotMatch(id, reported);

  const setup = (box, args, env = {}, input) => spawnSync(process.execPath, [CLI, 'setup', ...args], { env: { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH, ...env }, encoding: 'utf8', input });
  const saved = (box) => auto.parseServiceEnv(fs.readFileSync(path.join(box.home, '.quotemem', 'service.env'), 'utf8'));
  const text = (box) => fs.readFileSync(path.join(box.home, '.quotemem', 'service.env'), 'utf8');

  // --yes with nothing saved and no option: the default preset; one key for both models.
  const box = sandbox({ setup: false });
  const first = setup(box, ['--yes', '--no-check'], { QUOTEMEM_CHAT_API_KEY: 'sk-dash-one-HHHH' });
  assert.equal(first.status, 0, first.stderr);
  assert.match(first.stdout, /^preset: default; from the environment: QUOTEMEM_CHAT_API_KEY$/m);
  assert.match(first.stdout, /Thinking is at high effort/);
  assert.doesNotMatch(first.stdout + first.stderr, /sk-dash-one/);
  assert.deepEqual(saved(box), { ...PRESETS.default.values, QUOTEMEM_CHAT_API_KEY: 'sk-dash-one-HHHH', QUOTEMEM_EMBED_API_KEY: 'sk-dash-one-HHHH', QUOTEMEM_REVIEW: 'standard' });
  assert.match(text(box), /^# preset: default, the competition configuration/m);
  assertMode(path.join(box.home, '.quotemem', 'service.env'), 0o600);

  // --preset fast keeps the saved key (same endpoint) and turns thinking off.
  const fast = setup(box, ['--yes', '--no-check', '--preset', 'fast']);
  assert.equal(fast.status, 0, fast.stderr);
  assert.deepEqual(saved(box), { ...PRESETS.fast.values, QUOTEMEM_CHAT_API_KEY: 'sk-dash-one-HHHH', QUOTEMEM_EMBED_API_KEY: 'sk-dash-one-HHHH', QUOTEMEM_REVIEW: 'standard' });
  assert.match(text(box), /^# preset: fast$/m);
  // The default preset with --chat-reasoning off is the fast preset.
  const viaFlag = setup(box, ['--yes', '--no-check', '--preset', 'default', '--chat-reasoning', 'off']);
  assert.equal(viaFlag.status, 0, viaFlag.stderr);
  assert.equal(saved(box).QUOTEMEM_CHAT_REASONING, CHAT_REASONING.off);

  // DASHSCOPE_API_KEY serves a preset; another chat model drops the preset's pattern.
  const other = sandbox({ setup: false });
  const swapped = setup(other, ['--yes', '--no-check', '--preset', 'default', '--chat-model', 'qwen-plus'], { DASHSCOPE_API_KEY: 'sk-dash-env-IIII' });
  assert.equal(swapped.status, 0, swapped.stderr);
  assert.match(swapped.stdout, /from the environment: DASHSCOPE_API_KEY/);
  const swappedValues = saved(other);
  assert.equal(swappedValues.QUOTEMEM_CHAT_MODEL, 'qwen-plus');
  assert.equal(swappedValues.QUOTEMEM_CHAT_MODEL_PATTERN, undefined, 'the pattern belonged to the preset model');
  assert.equal(swappedValues.QUOTEMEM_CHAT_REASONING, CHAT_REASONING.high);
  assert.equal(swappedValues.QUOTEMEM_EMBED_API_KEY, 'sk-dash-env-IIII');
  assert.match(text(other), /^# preset: custom$/m);

  // Without --preset, saved settings are kept (custom): only the key changes here.
  const rerun = setup(box, ['--yes', '--no-check'], { QUOTEMEM_CHAT_API_KEY: 'sk-dash-two-JJJJ' });
  assert.equal(rerun.status, 0, rerun.stderr);
  assert.match(rerun.stdout, /^preset: custom/m);
  assert.deepEqual(saved(box), { ...PRESETS.fast.values, QUOTEMEM_CHAT_API_KEY: 'sk-dash-two-JJJJ', QUOTEMEM_EMBED_API_KEY: 'sk-dash-two-JJJJ', QUOTEMEM_REVIEW: 'standard' });

  // Custom over the saved preset: thinking at the provider's default, another chat endpoint.
  const custom = setup(box, ['--yes', '--no-check', '--preset', 'custom', '--chat-url', 'https://api.deepseek.com/v1', '--chat-model', 'deepseek-chat', '--chat-reasoning', 'default'],
    { QUOTEMEM_CHAT_API_KEY: 'sk-deepseek-KKKK' });
  assert.equal(custom.status, 0, custom.stderr);
  const customValues = saved(box);
  // An empty value would mean thinking off to the service; `none` sends no reasoning parameters.
  assert.equal(customValues.QUOTEMEM_CHAT_REASONING, CHAT_REASONING.none);
  assert.match(text(box), /^QUOTEMEM_CHAT_REASONING=none$/m);
  assert.equal(customValues.QUOTEMEM_CHAT_MODEL_PATTERN, undefined);
  assert.equal(customValues.QUOTEMEM_CHAT_API_KEY, 'sk-deepseek-KKKK');
  assert.equal(customValues.QUOTEMEM_EMBED_BASE_URL, dashscope);
  assert.equal(customValues.QUOTEMEM_EMBED_API_KEY, 'sk-dash-two-JJJJ', 'the saved embedding key, same endpoint');

  // Refusals name the option, never a value.
  for (const [args, pattern] of [[['--preset', 'turbo'], /--preset is default, fast or custom/], [['--chat-reasoning', 'max'], /--chat-reasoning is high, off or default/]]) {
    const refused = setup(box, ['--yes', '--no-check', ...args]);
    assert.equal(refused.status, 2);
    assert.match(refused.stderr, pattern);
  }

  // Asked: Enter takes the recommended preset and asks one key; Enter later keeps it.
  const asked = sandbox({ setup: false });
  const one = setup(asked, ['--no-check'], {}, '\n\nsk-typed-dash-LLLL\n');
  assert.equal(one.status, 0, one.stdout + one.stderr);
  assert.equal((one.stdout.match(/API key/g) ?? []).length, 1, 'one key question');
  assert.match(one.stdout, /Model Studio \(DashScope\) API key, for both models/);
  assert.doesNotMatch(one.stdout + one.stderr, /sk-typed-dash/);
  assert.deepEqual(saved(asked), { ...PRESETS.default.values, QUOTEMEM_CHAT_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_EMBED_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_REVIEW: 'standard' });
  const toFast = setup(asked, ['--no-check'], {}, '2\n\n\n');
  assert.equal(toFast.status, 0, toFast.stdout + toFast.stderr);
  assert.match(toFast.stdout, /4\) keep the saved settings \(preset default\)/);
  assert.deepEqual(saved(asked), { ...PRESETS.fast.values, QUOTEMEM_CHAT_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_EMBED_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_REVIEW: 'standard' });
  const kept = setup(asked, ['--no-check'], {}, '4\n\n\n');
  assert.equal(kept.status, 0, kept.stdout + kept.stderr);
  assert.deepEqual(saved(asked), { ...PRESETS.fast.values, QUOTEMEM_CHAT_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_EMBED_API_KEY: 'sk-typed-dash-LLLL', QUOTEMEM_REVIEW: 'standard' });
});

test('setup asks the review: standard by default, fixed per store; --review and QUOTEMEM_REVIEW for --yes; status and doctor show it', async () => {
  const { DEFAULT_REVIEW, PRESETS, REVIEWS, reviewOf } = await import(pathToFileURL(path.resolve(CORE, '..', 'cli', 'service.mjs')).href);
  assert.deepEqual(Object.keys(REVIEWS), ['fast', 'standard', 'full']);
  assert.equal(DEFAULT_REVIEW, 'standard');
  assert.deepEqual(reviewOf({}), { value: 'standard', source: 'default' });
  assert.deepEqual(reviewOf({ QUOTEMEM_REVIEW: 'Full' }), { value: 'full', source: 'service.env' });
  assert.deepEqual(reviewOf({ QUOTEMEM_REVIEW: 'max' }), { value: null, source: 'service.env' });
  const setup = (box, args, env = {}, input) => spawnSync(process.execPath, [CLI, 'setup', ...args], { env: { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH, ...env }, encoding: 'utf8', input });
  const file = (box) => path.join(box.home, '.quotemem', 'service.env');
  const saved = (box) => auto.parseServiceEnv(fs.readFileSync(file(box), 'utf8'));
  const key = { QUOTEMEM_CHAT_API_KEY: 'sk-review-MMMM' };

  // --yes: standard unless asked; the preset stays the preset (the review is not a model setting).
  const box = sandbox({ setup: false });
  const first = setup(box, ['--yes', '--no-check'], key);
  assert.equal(first.status, 0, first.stderr);
  assert.match(first.stdout, /preset default, review standard/);
  assert.match(fs.readFileSync(file(box), 'utf8'), /^QUOTEMEM_REVIEW=standard$/m);
  const full = setup(box, ['--yes', '--no-check', '--preset', 'default', '--review', 'FULL']);
  assert.equal(full.status, 0, full.stderr);
  assert.deepEqual(saved(box), { ...PRESETS.default.values, QUOTEMEM_CHAT_API_KEY: 'sk-review-MMMM', QUOTEMEM_EMBED_API_KEY: 'sk-review-MMMM', QUOTEMEM_REVIEW: 'full' });
  assert.match(fs.readFileSync(file(box), 'utf8'), /^# preset: default, the competition configuration/m);
  assertMode(file(box), 0o600);
  // Without --review the saved review is kept, also with another preset.
  const keep = setup(box, ['--yes', '--no-check', '--preset', 'fast']);
  assert.equal(keep.status, 0, keep.stderr);
  assert.equal(saved(box).QUOTEMEM_REVIEW, 'full');
  assert.equal(saved(box).QUOTEMEM_CHAT_REASONING, 'off', 'the fast preset is thinking off, not the fast review');
  // QUOTEMEM_REVIEW of the environment, under --review.
  const fromEnv = setup(box, ['--yes', '--no-check'], { QUOTEMEM_REVIEW: 'fast' });
  assert.equal(fromEnv.status, 0, fromEnv.stderr);
  assert.match(fromEnv.stdout, /from the environment: .*QUOTEMEM_REVIEW/);
  assert.equal(saved(box).QUOTEMEM_REVIEW, 'fast');
  const flagWins = setup(box, ['--yes', '--no-check', '--review', 'standard'], { QUOTEMEM_REVIEW: 'full' });
  assert.equal(flagWins.status, 0, flagWins.stderr);
  assert.equal(saved(box).QUOTEMEM_REVIEW, 'standard');
  // Refusals name the option or the variable; nothing is written.
  const before = fs.readFileSync(file(box), 'utf8');
  const badFlag = setup(box, ['--yes', '--no-check', '--review', 'max']);
  assert.equal(badFlag.status, 2);
  assert.match(badFlag.stderr, /--review is fast, standard or full/);
  const badEnv = setup(box, ['--yes', '--no-check'], { QUOTEMEM_REVIEW: 'thorough' });
  assert.equal(badEnv.status, 2);
  assert.match(badEnv.stderr, /QUOTEMEM_REVIEW is fast, standard or full/);
  assert.equal(fs.readFileSync(file(box), 'utf8'), before);

  // Asked: after the models, before the key; Enter keeps the saved review, else standard.
  const asked = sandbox({ setup: false });
  const one = setup(asked, ['--no-check'], {}, '\n\nsk-typed-review-NNNN\n');
  assert.equal(one.status, 0, one.stdout + one.stderr);
  assert.match(one.stdout, /Review \(how a new memory is checked[\s\S]*fixed when the memory store is created[\s\S]*quotemem rebuild --review/);
  assert.match(one.stdout, /1\) fast: local review rules only, as in the competition submission/);
  assert.match(one.stdout, /2\) standard \(recommended\): the chat model reviews only/);
  assert.match(one.stdout, /3\) full: the chat model reviews every new memory/);
  assert.match(one.stdout, /choose \[2\]: [\s\S]*API key/, 'standard is the default, asked before the key');
  assert.equal(saved(asked).QUOTEMEM_REVIEW, 'standard');
  const toFull = setup(asked, ['--no-check'], {}, '1\n3\n\n');
  assert.equal(toFull.status, 0, toFull.stdout + toFull.stderr);
  assert.equal(saved(asked).QUOTEMEM_REVIEW, 'full');
  const again = setup(asked, ['--no-check'], {}, '4\n\n\n');
  assert.equal(again.status, 0, again.stdout + again.stderr);
  assert.match(again.stdout, /choose \[3\]: /, 'Enter keeps the saved review');
  assert.equal(saved(asked).QUOTEMEM_REVIEW, 'full');
  const flagged = setup(asked, ['--no-check', '--review', 'fast'], {}, '4\n\n');
  assert.equal(flagged.status, 0, flagged.stdout + flagged.stderr);
  assert.doesNotMatch(flagged.stdout, /Review \(/, '--review skips the question');
  assert.equal(saved(asked).QUOTEMEM_REVIEW, 'fast');

  // A store already exists: changing the review says how to move it, and how to keep it.
  const store = sandbox({ setup: false });
  assert.equal(setup(store, ['--yes', '--no-check'], key).status, 0);
  const dataDir = path.join(store.home, '.local', 'share', 'quotemem');
  fs.mkdirSync(path.join(dataDir, 'workspace'), { recursive: true });
  const same = setup(store, ['--yes', '--no-check', '--review', 'standard']);
  assert.doesNotMatch(same.stdout, /keeps the review it was created with/);
  const moved = setup(store, ['--yes', '--no-check', '--review', 'full']);
  assert.equal(moved.status, 0, moved.stderr);
  assert.match(moved.stdout, /keeps the review it was created with \(standard, as saved before\)/);
  assert.ok(moved.stdout.includes(`quotemem rebuild --from ${dataDir} --to <new data dir> --review full`), moved.stdout);
  assert.match(moved.stdout, /quotemem setup --review standard/);

  // status and doctor show the saved review; an invalid one is a doctor FAIL.
  const env = { HOME: store.home, USERPROFILE: store.home, PATH: process.env.PATH, QUOTEMEM_URL: 'http://127.0.0.1:9' };
  const status = spawnSync(process.execPath, [CLI, 'status'], { env, encoding: 'utf8' });
  assert.match(status.stdout, /^review: full; fixed when a store is created, `quotemem rebuild --review` moves a store to another$/m);
  const doctor = spawnSync(process.execPath, [CLI, 'doctor'], { env, encoding: 'utf8' });
  assert.match(doctor.stdout, /^ ok {3}review: full; fixed when a store is created/m);
  fs.appendFileSync(file(store), 'QUOTEMEM_REVIEW=thorough\n');
  const bad = spawnSync(process.execPath, [CLI, 'doctor'], { env, encoding: 'utf8' });
  assert.match(bad.stdout, /^FAIL {2}review: QUOTEMEM_REVIEW=thorough in service.env is not fast, standard or full/m);
  const unset = sandbox({ serviceEnv: {} });
  const plain = spawnSync(process.execPath, [CLI, 'status'], { env: { ...env, HOME: unset.home, USERPROFILE: unset.home }, encoding: 'utf8' });
  assert.match(plain.stdout, /^review: standard \(the default\);/m);
});

test('setup refuses a model URL whose host is not plain, before writing anything', () => {
  const box = sandbox({ setup: false });
  const run = spawnSync(process.execPath, [CLI, 'setup', '--yes', '--no-check', '--chat-url', 'http://127.0.0.1\\.evil.example.com/v1', '--chat-model', 'm', '--embed-model', 'e'],
    { env: { HOME: box.home, USERPROFILE: box.home, PATH: process.env.PATH, QUOTEMEM_CHAT_API_KEY: 'sk-plain-NNNN' }, encoding: 'utf8' });
  assert.equal(run.status, 2, run.stderr);
  assert.match(run.stderr, /QUOTEMEM_CHAT_BASE_URL must be an https URL \(or http on this computer\) without credentials/);
  assert.ok(!fs.existsSync(path.join(box.home, '.quotemem', 'service.env')));
  assert.doesNotMatch(run.stdout + run.stderr, /sk-plain/);
});
