#!/usr/bin/env node
// Detached processes hold none of the hook's handles. On Windows a process
// inherits every inheritable handle of its parent, whatever its stdio: a
// caller that reads an agent tool's output through a pipe hands that pipe
// down, through cmd and PowerShell, to the hook. A flusher started by plain
// spawn kept it open until it exited, and the caller waited that long.
//
//   node --test integrations/core/test/detach.test.mjs   (Linux, macOS, Windows)
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, TOKEN, sleep, stubServer, tempDir, waitFor } from './helpers.mjs';

const CORE_URL = pathToFileURL(path.join(CORE, 'index.mjs')).href;
const core = await import(CORE_URL);
const windows = process.platform === 'win32';
const quiet = { warn() {}, warnOnce() {}, debug() {} };
const root = tempDir('qm-core-detach-');
const started = new Set();
const servers = new Set();
after(async () => {
  for (const pid of started) stop(pid);
  await Promise.all([...servers].map((s) => s.close()));
});

function stop(pid) {
  try {
    if (windows) execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    else process.kill(-pid, 'SIGKILL');
  } catch {
    // gone
  }
}

// The hook: queue a batch and start the flusher, wait until it runs, then
// start a program with a log file and a known pid, as the service's start
// does (its words hold "%" and "&", which must reach it as they are), print
// what it started and exit.
const HOOK = path.join(root, 'hook.mjs');
fs.writeFileSync(HOOK, `
import { DiskQueue, buildJobs, resolveSettings, spawnDetached } from ${JSON.stringify(CORE_URL)};
const settings = resolveSettings({}, process.env);
const queue = new DiskQueue(settings, { warn() {}, warnOnce() {}, debug() {} });
queue.enqueue(buildJobs(settings, 'test:D', 1, 1, [{ role: 'user', content: 'My kite is orange.', timestamp: 1763500000000 }]));
const flusher = queue.startFlusher();
const end = Date.now() + 15000;
while (!queue.flusherRunning() && Date.now() < end) await new Promise((resolve) => setTimeout(resolve, 20));
const program = "console.log('detached ' + process.argv.slice(1).join(' ')); setTimeout(() => {}, 30000)";
const other = spawnDetached(process.execPath, ['-e', program, 'a%PATH%&b'], { logFile: process.env.DETACH_LOG });
process.stdout.write(JSON.stringify({ flusher, running: queue.flusherRunning(), pid: other.pid }) + '\\n');
`);

test('a caller reading a hook\'s output through a pipe sees it end while the flusher and a started service still run', async (t) => {
  // A slow service: the flusher waits about 6 s for its answer.
  const server = await stubServer({ '/memory/add': () => ({ delay: 6000, body: { success: true, outcome: { added: 1 } } }) });
  servers.add(server);
  const stateDir = path.join(root, 'state');
  const log = path.join(root, 'detached.log');
  const home = path.join(root, 'pipe-home');
  fs.mkdirSync(home, { recursive: true });
  const env = { ...process.env, HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: stateDir, QUOTEMEM_AUTOSTART: '0', DETACH_LOG: log };
  // On Windows the caller's pipe reaches the hook through cmd and
  // PowerShell, as when a terminal pipes an agent tool's output.
  const powershell = path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
  const [command, args] = windows
    ? [process.env.ComSpec ?? 'cmd.exe', ['/d', '/c', powershell, '-NoProfile', '-NonInteractive', '-Command', `& '${process.execPath}' '${HOOK}'`]]
    : [process.execPath, [HOOK]];
  const begun = Date.now();
  const caller = spawn(command, args, { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let out = '';
  let err = '';
  caller.stdout.on('data', (c) => { out += c; });
  caller.stderr.on('data', (c) => { err += c; });
  await new Promise((resolve) => caller.stdout.on('end', resolve));
  const endedMs = Date.now() - begun;
  t.diagnostic(`the caller's pipe ended after ${endedMs} ms`);
  const queue = new core.DiskQueue({ stateDir, url: server.url, userId: 'quei' }, quiet);
  const flusherAtEnd = queue.flusherRunning();
  const line = out.trim().split('\n').pop();
  assert.ok(line, `the hook printed nothing (${err.trim()})`);
  const result = JSON.parse(line);
  if (Number.isInteger(result.pid)) started.add(result.pid);
  assert.equal(result.flusher, true, 'the hook started a flusher');
  assert.equal(result.running, true, 'the flusher ran before the hook ended');
  assert.equal(flusherAtEnd, true, `the pipe ended after ${endedMs} ms, while the flusher still runs`);
  assert.ok(Number.isInteger(result.pid) && core.alive(result.pid), 'the started program still runs');
  assert.equal(server.of('/memory/add').length <= 1, true);
  // The flusher then stores the batch, with the token, and exits; the
  // program's output went to the log file, its words unchanged.
  await waitFor(() => server.of('/memory/add').length === 1, 15_000, 'the batch to be sent');
  assert.match(server.of('/memory/add')[0].headers.authorization, new RegExp(`${TOKEN}$`));
  await waitFor(() => !queue.flusherRunning(), 30_000, 'the flusher to exit');
  assert.ok(core.alive(result.pid), 'the program runs on after the flusher');
  await waitFor(() => fs.existsSync(log) && /detached /.test(fs.readFileSync(log, 'utf8')), 10_000, 'the program\'s log line');
  assert.match(fs.readFileSync(log, 'utf8'), /detached a%PATH%&b/);
  stop(result.pid);
  started.delete(result.pid);
});

test('on Windows, a word with a quote is refused before anything starts', () => {
  assert.throws(() => core.spawnDetached(process.execPath, ['a"b'], { platform: 'win32' }), { code: 'EINVAL' });
  assert.throws(() => core.spawnDetached(process.execPath, [''], { platform: 'win32' }), { code: 'EINVAL' });
});

// ---------------------------------------------- the Windows launch, and the service's start

const auto = await import(pathToFileURL(path.join(CORE, 'autostart.mjs')).href);
const { resolveSettings } = await import(pathToFileURL(path.join(CORE, 'config.mjs')).href);
const { doctorLines } = await import(pathToFileURL(path.resolve(CORE, '..', 'cli', 'service.mjs')).href);
const lines = [];
const log = { warn(m) { lines.push(m); }, warnOnce(_k, m) { lines.push(m); }, debug() {} };
// No double quote in the words: a Windows command line could not pass it (EINVAL).
const program = (text) => ['-e', `console.log('${text} ' + process.cwd()); setTimeout(() => {}, 30000)`];
const readSoon = (file, pattern) => waitFor(() => fs.existsSync(file) && pattern.test(fs.readFileSync(file, 'utf8')), 10_000, `${path.basename(file)} to match ${pattern}`);

test('a detached start goes to its folder, even one named with "[", "]", "%" and "&"', async () => {
  const dir = path.join(root, 'odd [1] 100% & co');
  fs.mkdirSync(dir, { recursive: true });
  const logFile = path.join(root, 'cwd.log');
  const child = core.spawnDetached(process.execPath, program('cwd'), { cwd: dir, logFile });
  started.add(child.pid);
  await readSoon(logFile, /cwd /);
  assert.equal(fs.readFileSync(logFile, 'utf8').trim(), `cwd ${fs.realpathSync(dir)}`);
  stop(child.pid);
});

test('Windows: the launcher runs in ConstrainedLanguage mode; without PowerShell the program is started directly, said once and recorded', { skip: !windows && 'Windows only' }, async () => {
  // ConstrainedLanguage, as AppLocker or WDAC would set it: still started,
  // its pid known, through WMI and through the shell.
  for (const wmi of [true, false]) {
    const constrainedLog = path.join(root, `constrained-${wmi}.log`);
    const child = core.spawnDetached(process.execPath, program('constrained'), { logFile: constrainedLog, constrained: true, wmi, label: 'quotemem test' });
    assert.ok(Number.isInteger(child.pid) && core.alive(child.pid), `the pid of what started (wmi ${wmi})`);
    assert.equal(child.via, wmi ? 'wmi' : 'shell');
    started.add(child.pid);
    await readSoon(constrainedLog, /constrained /);
    stop(child.pid);
  }
  // WMI gave a pid, but its process never started (an endpoint rule stopped
  // it): the shell starts the program, once, and says so.
  const decoyLog = path.join(root, 'decoy.log');
  const decoy = core.spawnDetached(process.execPath, program('decoy'), { logFile: decoyLog, wmiDecoy: true });
  assert.equal(decoy.via, 'shell');
  started.add(decoy.pid);
  await readSoon(decoyLog, /decoy /);
  await sleep(500);
  assert.equal(fs.readFileSync(decoyLog, 'utf8').trim().split('\n').length, 1, 'started once');
  stop(decoy.pid);
  // The cmd WMI started is slower than 3 s (a loaded machine): the shell
  // starts the program, and that cmd starts nothing, even once it runs.
  const slowLog = path.join(root, 'slow.log');
  const slow = core.spawnDetached(process.execPath, program('slow'), { logFile: slowLog, wmiSlow: true });
  assert.equal(slow.via, 'shell');
  started.add(slow.pid);
  await readSoon(slowLog, /slow /);
  await sleep(3000);
  assert.equal(fs.readFileSync(slowLog, 'utf8').trim().split('\n').length, 1, 'started once');
  stop(slow.pid);
  // The start that does not hold the caller up: the same result, later.
  const asyncLog = path.join(root, 'async.log');
  const later = core.spawnDetachedAsync(process.execPath, program('async'), { logFile: asyncLog });
  assert.ok(later instanceof Promise);
  const done = await later;
  assert.ok(Number.isInteger(done.pid) && core.alive(done.pid));
  started.add(done.pid);
  await readSoon(asyncLog, /async /);
  stop(done.pid);
  // No PowerShell: started directly (it holds this process's handles), with one warning and a record.
  const mark = path.join(root, 'state', core.DIRECT_START_FILE);
  const directLog = path.join(root, 'direct.log');
  lines.length = 0;
  const direct = core.spawnDetached(process.execPath, program('direct'), { logFile: directLog, log, mark, powershell: 'C:\\no\\such\\powershell.exe' });
  assert.equal(direct.direct, true);
  assert.ok(Number.isInteger(direct.pid) && core.alive(direct.pid));
  started.add(direct.pid);
  await readSoon(directLog, /direct /);
  assert.equal(lines.length, 1);
  assert.match(lines[0], /PowerShell could not start a background process \(ENOENT\); it was started directly/);
  assert.equal(JSON.parse(fs.readFileSync(mark, 'utf8')).reason, 'ENOENT');
  stop(direct.pid);
  // The check doctor runs.
  assert.deepEqual(core.launcherCheck(), { ok: true, detail: 'FullLanguage' });
  assert.equal(core.launcherCheck({ powershell: 'C:\\no\\such\\powershell.exe' }).ok, false);
});

// A stub of the service, started for uvx: prints a line, then answers /health.
const STUB = path.join(root, 'stub-service.mjs');
fs.writeFileSync(STUB, `
import http from 'node:http';
const args = process.argv.slice(2);
const port = Number(args[args.indexOf('--port') + 1]);
console.log('stub service ' + args.join(' '));
http.createServer((req, res) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"status":"ok"}'); }).listen(port, '127.0.0.1');
`);

function serviceBox(name) {
  const home = path.join(root, name);
  const bin = path.join(home, 'bin');
  fs.mkdirSync(bin, { recursive: true });
  // findUvx only needs a file of that name: the stub runs instead (spawnFn).
  fs.writeFileSync(path.join(bin, windows ? 'uvx.exe' : 'uvx'), '#!/bin/sh\nexit 1\n', { mode: 0o755 });
  fs.mkdirSync(path.join(home, '.quotemem'), { recursive: true, mode: 0o700 });
  fs.writeFileSync(path.join(home, '.quotemem', 'service.env'), `QUOTEMEM_DATA_DIR=${path.join(home, 'data')}\nQUOTEMEM_CHAT_API_KEY=sk-never-printed-2\n`, { mode: 0o600 });
  return home;
}

async function freePort() {
  const net = await import('node:net');
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

test('the service starts through spawnDetached: its log, its lock with the pid, stop, and the stop of a start that lost its lock', async () => {
  const modes = windows ? ['shell', 'constrained', 'direct'] : ['shell'];
  for (const mode of modes) {
    const home = serviceBox(`service-${mode}`);
    const port = await freePort();
    const env = { ...process.env, HOME: home, USERPROFILE: home, PATH: `${path.join(home, 'bin')}${path.delimiter}${process.env.PATH}`, QUOTEMEM_URL: `http://127.0.0.1:${port}`, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: path.join(home, 'state') };
    delete env.QUOTEMEM_AUTOSTART;
    const settings = resolveSettings({}, env, {});
    const extra = mode === 'constrained' ? { constrained: true } : mode === 'direct' ? { powershell: 'C:\\no\\such\\powershell.exe' } : {};
    const spawnFn = (_uvx, args, options) => core.spawnDetached(process.execPath, [STUB, ...args], { ...options, ...extra });
    auto.resetAutostartMemo();
    const result = auto.ensureService(settings, { env, spawnFn, log });
    assert.equal(result.state, 'started', mode);
    started.add(result.pid);
    const paths = auto.servicePaths(env);
    assert.equal(JSON.parse(fs.readFileSync(paths.lock, 'utf8')).pid, result.pid, `${mode}: the lock records the pid`);
    await readSoon(paths.log, /stub service --from quotemem==\S+ quotemem serve --port \d+/);
    await waitFor(async () => (await fetch(`${settings.url}/health`).catch(() => null))?.ok, 10_000, `${mode}: the stub to answer`);
    const state = auto.serviceState(env);
    assert.equal(state.running, true, `${mode}: the recorded pid is the service's (its command line says so)`);
    const stopped = await auto.stopService(env);
    assert.equal(stopped.stopped, true, `${mode}: ${stopped.message}`);
    await waitFor(async () => !(await fetch(`${settings.url}/health`).catch(() => null)), 10_000, `${mode}: the stub to stop`);
    if (mode === 'direct') assert.ok(fs.existsSync(path.join(paths.dir, core.DIRECT_START_FILE)), 'the direct start is recorded');
    // Another starter took the lock while this one started: what it started is stopped.
    auto.resetAutostartMemo();
    let extraPid = null;
    const lost = auto.ensureService(settings, {
      env, log,
      spawnFn: (uvx, args, options) => {
        const child = spawnFn(uvx, args, options);
        extraPid = child.pid;
        started.add(child.pid);
        fs.writeFileSync(paths.lock, JSON.stringify({ pid: process.pid, nonce: 'theirs', at: Date.now() }));
        return child;
      },
    });
    assert.equal(lost.state, 'starting', mode);
    await waitFor(() => !core.alive(extraPid), 10_000, `${mode}: the extra start to be stopped`);
    await sleep(300);
    assert.equal(await fetch(`${settings.url}/health`).then(() => true, () => false), false, `${mode}: nothing answers`);
    fs.rmSync(paths.lock, { force: true });
  }
});

test('doctor reports how background processes start, and a direct start', async () => {
  const home = serviceBox('doctor');
  const env = { ...process.env, HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'http://127.0.0.1:9', QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: path.join(home, 'state') };
  const settings = resolveSettings({}, env, {});
  fs.mkdirSync(path.join(home, '.quotemem', 'cursor'), { recursive: true });
  fs.writeFileSync(path.join(home, '.quotemem', 'cursor', core.DIRECT_START_FILE), JSON.stringify({ at: Date.UTC(2026, 10, 21), reason: 'ENOENT' }));
  const out = await doctorLines(env, settings);
  const texts = out.map(([, text]) => text);
  if (windows) assert.ok(texts.some((t) => /background processes start through PowerShell \(FullLanguage\)/.test(t)), texts.join('\n'));
  assert.ok(texts.some((t) => /a background process was last started directly at 2026-11-21T00:00:00.000Z \(PowerShell: ENOENT\)/.test(t)), texts.join('\n'));
});

test('two people signed in at once: a local service that refuses this user\'s token is named as maybe another user\'s, with a port of this user\'s own', async () => {
  const server = await stubServer({
    '/health': () => ({ body: { status: 'ok' } }),
    '/memory/search': () => ({ status: 401, body: { code: 'authentication_failed' } }),
  });
  servers.add(server);
  const home = path.join(root, 'second-user');
  fs.mkdirSync(home, { recursive: true });
  const settings = resolveSettings({}, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei', QUOTEMEM_STATE_DIR: path.join(home, 'state'), HOME: home, USERPROFILE: home, QUOTEMEM_AUTOSTART: '0' }, {});
  assert.ok(settings.localService, 'a service on this computer');
  lines.length = 0;
  const block = await core.searchRecall(settings, core.recallQuery(settings, 'What do I like to drink?'), { breaker: new core.RecallBreaker(), log, report: {} });
  assert.equal(block, null);
  const port = core.userPort();
  assert.ok(port >= 20000 && port < 26000 && port === core.userPort(), 'the same port every time, below 27000');
  assert.match(lines.join('\n'), new RegExp(`refused this user's token \\(401\\); it may be another user's QuoteMem on this computer \\(two people signed in at once\\)\\. Give this user a port of its own: QUOTEMEM_URL=http://127\\.0\\.0\\.1:${port} in ~/\\.quotemem/env and in the environment Claude Code, Codex and DSH start in`));
  // The port's listener is this user's own here.
  if (process.platform === 'linux' || process.platform === 'darwin') assert.equal(core.portOwner(server.port)?.mine, true);
});

test('Windows: a host that stops its hooks\' job when it exits (Cursor\'s CLI on Ctrl+C) does not stop a detached process started through WMI', { skip: !windows && 'Windows only' }, async () => {
  // A job that kills its processes when its last handle closes, as a host's.
  const JOB = [
    'Add-Type -TypeDefinition @\'',
    'using System; using System.Runtime.InteropServices;',
    'public static class QmJob {',
    '  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr a, string name);',
    '  [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);',
    '  [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);',
    '  [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();',
    '  [StructLayout(LayoutKind.Sequential)] struct Basic { public long A; public long B; public uint LimitFlags; public UIntPtr C; public UIntPtr D; public uint E; public UIntPtr F; public uint G; public uint H; }',
    '  [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A, B, C, D, E, F; }',
    '  [StructLayout(LayoutKind.Sequential)] struct Ext { public Basic Basic; public Io Io; public UIntPtr P1, P2, P3, P4; }',
    '  public static bool KillOnClose() {',
    '    IntPtr job = CreateJobObject(IntPtr.Zero, null);',
    '    Ext info = new Ext(); info.Basic.LimitFlags = 0x2000;',
    '    int size = Marshal.SizeOf(typeof(Ext)); IntPtr p = Marshal.AllocHGlobal(size); Marshal.StructureToPtr(info, p, false);',
    '    return SetInformationJobObject(job, 9, p, (uint)size) && AssignProcessToJobObject(job, GetCurrentProcess());',
    '  }',
    '}',
    '\'@',
    'if (-not [QmJob]::KillOnClose()) { exit 3 }',
  ].join('\n');
  const hook = path.join(root, 'hook-job.mjs');
  fs.writeFileSync(hook, `
import fs from 'node:fs';
import { spawnDetached } from ${JSON.stringify(CORE_URL)};
const [mode, pidFile] = process.argv.slice(2);
const child = spawnDetached(process.execPath, ['-e', 'setTimeout(() => {}, 30000)'], { wmi: mode === 'wmi' });
fs.writeFileSync(pidFile, String(child.pid));
`);
  const powershell = path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
  const survives = {};
  for (const mode of ['wmi', 'shell']) {
    const pidFile = path.join(root, `job-${mode}.pid`);
    const script = path.join(root, `job-${mode}.ps1`);
    fs.writeFileSync(script, `${JOB}\n& '${process.execPath}' '${hook}' ${mode} '${pidFile}'\nexit $LASTEXITCODE\n`);
    const run = execFileSync(powershell, ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', script], { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    assert.equal(run.trim(), '');
    // PowerShell has exited: its job, and every process in it, is gone.
    const pid = Number(fs.readFileSync(pidFile, 'utf8'));
    started.add(pid);
    await sleep(1500);
    survives[mode] = core.alive(pid);
    stop(pid);
  }
  assert.deepEqual(survives, { wmi: true, shell: false }, 'started through WMI it outlives the job; through the shell it does not');
});
