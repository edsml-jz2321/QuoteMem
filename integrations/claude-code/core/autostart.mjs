// The local memory service, started on demand.
//
// QuoteMem runs on the user's own computer: `quotemem serve` (the Python
// package `quotemem`, run through uvx) listens on 127.0.0.1:8765, and every
// agent's plugin talks to it. After a one-time `npx quotemem setup`, which
// writes the model settings to ~/.quotemem/service.env, no one starts the
// service by hand: the first plugin call that finds nothing listening starts
// it in the background, and later calls find it.
//
//   ~/.quotemem/service.env   the service's settings (mode 0600; keys stay here)
//   ~/.quotemem/service.lock  the one started instance: pid, start time, URL
//   ~/.quotemem/service.log   its output (the previous start's in service.log.1)
//
// Rules:
// - Only a loopback service URL is ever started (QUOTEMEM_URL unset, or
//   localhost, 127.0.0.1 or [::1]), and only when ~/.quotemem/service.env
//   exists and QUOTEMEM_AUTOSTART is not off.
// - Exactly one instance across every tool and process: the starter takes
//   service.lock with O_EXCL. A lock whose process is gone is stale; one whose
//   process is no longer the service (a reused pid) is stale too.
// - Nothing waits for the start: the call that triggered it goes on without
//   memory (recall is skipped, captures stay queued), and later calls find
//   the service once it answers.
// - The service keeps running after the tools exit: every tool shares it.
//   `quotemem stop` stops it; QUOTEMEM_AUTOSTART=0 turns starting off.
//
// The token is never read into a log or a command line. When none is
// configured, the plugins use the service's own: `<data-dir>/token`, which
// the service creates on its first start.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { lockOwned, readJson, tryLock, writeFileAtomic } from './lock.mjs';
import { DIRECT_START_FILE, alive, bootTime, homeDir, hostName, spawnDetached } from './proc.mjs';

/** The service release these plugins start; check_package.mjs keeps it equal to package.json. */
export const SERVICE_RELEASE = '0.3.0';
/** Where `quotemem serve` listens unless told otherwise (the service's DEFAULT_PORT). */
export const DEFAULT_PORT = 8765;
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]']);
// A first start downloads the service and its dependencies through uvx; a
// started process that has not answered for this long gets a warning.
export const STARTUP_GRACE_MS = 15 * 60_000;
// A start whose process died within this time is not repeated before it
// passes: a service that cannot start (bad settings) is not restarted by
// every hook.
export const RESTART_BACKOFF_MS = 60_000;
// The log is moved to service.log.1 at each start, and cut when it passes
// this size while the service runs (the service appends; a cut is safe).
export const LOG_MAX_BYTES = 4 * 1024 * 1024;
const MAX_ENV_BYTES = 64 * 1024;
const NAME = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
const OFF = new Set(['0', 'off', 'false', 'no', 'none', 'disabled']);
// How long one process reuses its last answer (a retry loop asks again at once).
const MEMO_MS = 2000;
// Windows: how often a background check may ask whether a recorded pid that
// does not answer is still the service (reading a command line there is slow).
const WINDOWS_CHECK_EVERY_MS = 5 * 60_000;

/**
 * The home folder (homeDir in proc.mjs): the one the service finds too.
 */
export function homeOf(env = process.env, platform = process.platform) {
  return homeDir(env, platform);
}

/** The paths of the local service's files under ~/.quotemem. */
export function servicePaths(env = process.env) {
  return pathsIn(path.join(homeOf(env), '.quotemem'));
}

function pathsIn(dir) {
  return {
    dir,
    envFile: path.join(dir, 'service.env'),
    lock: path.join(dir, 'service.lock'),
    log: path.join(dir, 'service.log'),
  };
}

// An http(s) URL whose authority every client reads the same way: a host
// name (letters, digits, ".", "-", "_", "~"), an IPv4 address or a bracketed
// IPv6 address, an optional port, then a path, a query, a fragment or
// nothing. Node's URL parser takes a backslash in the authority as a path
// separator and drops tabs and newlines, where Python's urlsplit and httpx
// keep them in the host: `http://127.0.0.1\.evil.example` is 127.0.0.1 here
// and another host to the service's own commands. A URL with such a
// character (a space, `%`, `@` and the like too) is refused, so no token is
// sent on this reading of it.
const PLAIN_HTTP_URL = /^https?:\/\/(\[[0-9a-f:.]+\]|[a-z0-9._~-]+)(?::\d{1,5})?(?:[/?#]|$)/i;

/**
 * Whether `text` is an http(s) URL with a plain authority (PLAIN_HTTP_URL)
 * whose host Node keeps as written, case aside. Node rewrites some hosts:
 * `127.0.0.1.`, `127.1`, `0x7f.0.0.1` and `2130706433` all become
 * 127.0.0.1, `[0:0:0:0:0:0:0:1]` becomes [::1]. Python connects to the host
 * as written (`127.0.0.1.` is a DNS name to it), so such a URL is refused
 * too: both sides then read the same host.
 */
export function plainHttpUrl(text) {
  const value = String(text ?? '');
  const match = PLAIN_HTTP_URL.exec(value);
  if (!match) return false;
  try {
    return new URL(value).hostname === match[1].toLowerCase();
  } catch {
    return false;
  }
}

/** Whether a service URL points at this computer (a URL with an authority that is not plain never does). */
export function isLoopbackUrl(url) {
  try {
    return plainHttpUrl(url) && LOOPBACK.has(new URL(url).hostname);
  } catch {
    return false;
  }
}

/**
 * Parse `NAME=value` lines (`export `, comments, blank lines and one pair of
 * surrounding quotes allowed). Any variable name is taken: the service also
 * reads provider and proxy variables.
 */
export function parseServiceEnv(text) {
  const out = {};
  for (const raw of String(text ?? '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const match = /^(?:export\s+)?([^=\s]+)\s*=\s*(.*)$/.exec(line);
    if (!match || !NAME.test(match[1])) continue;
    let value = match[2].trim();
    if (value.length >= 2 && (value[0] === '"' || value[0] === "'") && value.at(-1) === value[0]) value = value.slice(1, -1);
    out[match[1]] = value;
  }
  return out;
}

/** The service settings in ~/.quotemem/service.env, or null when there is none. */
export function readServiceEnv(env = process.env) {
  return readEnvFile(servicePaths(env).envFile);
}

/** The NAME=value pairs of an env file, or null when it is missing, not a file or too large. */
export function readEnvFile(file) {
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > MAX_ENV_BYTES) return null;
    return parseServiceEnv(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

// What the started service takes from the starting tool's environment: only
// what finds programs, files, the network and the locale. Everything that
// gives the service its identity (token, data folder, model settings and
// their OPENAI_* / DASHSCOPE_* fallbacks) comes from service.env alone, so
// the service is the same whichever tool (a GUI app, a shell) starts it.
const BASE_NAMES = new Set([
  'PATH', 'HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'USER', 'USERNAME', 'LOGNAME', 'SHELL',
  'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'COMSPEC', 'PATHEXT', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA',
  'TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME', 'XDG_RUNTIME_DIR', 'LANG', 'LANGUAGE', 'TZ',
  'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
]);
const BASE_PATTERN = /^(?:LC_[A-Z]+|UV_[A-Z0-9_]+|(?:HTTP|HTTPS|ALL|NO)_PROXY)$/;

function baseName(name) {
  const upper = name.toUpperCase();
  return BASE_NAMES.has(upper) || BASE_PATTERN.test(upper);
}

/**
 * The environment the service runs with: the base variables of this one
 * (paths, locale, proxies, uv's own), overlaid with service.env.
 */
export function serviceEnvironment(env = process.env, serviceEnv = readServiceEnv(env)) {
  const out = {};
  for (const [name, value] of Object.entries(env)) if (typeof value === 'string' && baseName(name)) out[name] = value;
  Object.assign(out, serviceEnv ?? {});
  // An empty token is unset.
  for (const name of ['QUOTEMEM_TOKEN', 'QUOTEMEM_HTTP_AUTH_SECRET']) if (out[name] === '') delete out[name];
  return out;
}

/**
 * The service's data folder, as the service resolves it from its environment:
 * QUOTEMEM_DATA_DIR (`~` expanded), else $XDG_DATA_HOME/quotemem, else
 * ~/.local/share/quotemem. A relative path is taken from `cwd`, the folder
 * the service runs in (~/.quotemem when a plugin starts it).
 */
export function dataDirOf(env, cwd = process.cwd()) {
  const configured = String(env.QUOTEMEM_DATA_DIR ?? '').trim();
  if (configured) return path.resolve(cwd, configured.replace(/^~(?=$|[\\/])/, homeOf(env)));
  const base = String(env.XDG_DATA_HOME ?? '').trim();
  return path.join(base ? path.resolve(cwd, base) : path.join(homeOf(env), '.local', 'share'), 'quotemem');
}

/**
 * What the plugins know about a service on this computer, or null when the
 * URL is not a loopback one. JSON-safe (a background flusher gets a copy).
 * @returns {{dir: string, envFile: string, dataDir: string, configured: boolean,
 *   enabled: boolean, autostart: boolean, host: string, port: number} | null}
 *   `configured`: service.env exists; `enabled`: QUOTEMEM_AUTOSTART is not
 *   off; `autostart`: both, so it may be started.
 */
export function localService(env, url) {
  if (!url || !isLoopbackUrl(url)) return null;
  const parsed = new URL(url);
  const paths = servicePaths(env);
  const serviceEnv = readServiceEnv(env);
  const enabled = !OFF.has(String(env.QUOTEMEM_AUTOSTART ?? '').trim().toLowerCase());
  return {
    dir: paths.dir,
    envFile: paths.envFile,
    dataDir: dataDirOf(serviceEnvironment(env, serviceEnv), paths.dir),
    configured: serviceEnv !== null,
    enabled,
    autostart: enabled && serviceEnv !== null,
    host: parsed.hostname.replace(/^\[|\]$/g, ''),
    port: Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80)),
  };
}

/**
 * The first executable named uvx on PATH, or null. On Windows only uvx.exe:
 * a .cmd shim cannot be started without a shell.
 */
export function findUvx(env = process.env) {
  const names = process.platform === 'win32' ? ['uvx.exe'] : ['uvx'];
  const dirs = String(env.PATH ?? env.Path ?? '').split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    for (const name of names) {
      const file = path.join(dir, name);
      try {
        if (!fs.statSync(file).isFile()) continue;
        if (process.platform !== 'win32') fs.accessSync(file, fs.constants.X_OK);
        return file;
      } catch {
        // not here
      }
    }
  }
  return null;
}

/** The `uvx` arguments that run the service for a local service record. */
export function serveArgs(local) {
  const args = ['--from', `quotemem==${SERVICE_RELEASE}`, 'quotemem', 'serve'];
  if (local.host && local.host !== '127.0.0.1' && local.host !== 'localhost') args.push('--host', local.host);
  if (local.port !== DEFAULT_PORT) args.push('--port', String(local.port));
  return args;
}

/**
 * The command line of a process, or null when it cannot be read (the caller
 * then trusts the pid). Linux reads /proc; macOS and the BSDs ask ps; Windows
 * asks PowerShell (slow: only `quotemem stop` and `status` use it there).
 */
export function commandLineOf(pid, { windows = true } = {}) {
  if (!Number.isInteger(pid) || pid <= 0) return null;
  try {
    if (process.platform === 'linux') {
      return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ').trim() || null;
    }
    if (process.platform === 'win32') {
      if (!windows) return null;
      const out = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
        `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`], { encoding: 'utf8', timeout: 8000, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
      return out.trim() || null;
    }
    const out = execFileSync('ps', ['-o', 'command=', '-p', String(pid)], { encoding: 'utf8', timeout: 2000, stdio: ['ignore', 'pipe', 'ignore'] });
    return out.trim() || null;
  } catch {
    return null;
  }
}

/** Whether a command line is a QuoteMem service started through uvx (or its stand-in). */
export function isServiceCommand(commandLine) {
  return typeof commandLine === 'string' && /quotemem/i.test(commandLine) && /\bserve\b/.test(commandLine);
}

/**
 * The recorded instance and whether it still runs.
 * @returns {{record: object|null, running: boolean, ours: boolean|null}}
 *   `ours`: the pid's command line is the service's (null when unknown).
 */
export function serviceState(env = process.env, { checkCommand = true } = {}) {
  return stateOf(servicePaths(env), { checkCommand });
}

// The host name is not compared: it changes with the network on macOS, and
// the service only ever listens on this computer. The boot time and pid decide.
function stateOf({ lock }, { checkCommand }) {
  const record = readJson(lock);
  if (!record) return { record, running: false, ours: null };
  // Started, but PowerShell stopped before it told the pid (Windows): it
  // counts as starting for STARTUP_GRACE_MS, so no second one is started.
  if (!Number.isInteger(record.pid)) return { record, running: record.pid === null && Number.isFinite(record.startedAt) && Date.now() - record.startedAt < STARTUP_GRACE_MS, ours: null };
  if (Number.isFinite(record.boot) && Math.abs(record.boot - bootTime()) > 120_000) return { record, running: false, ours: null };
  if (!alive(record.pid)) return { record, running: false, ours: null };
  const ours = checkCommand ? (() => { const line = commandLineOf(record.pid); return line === null ? null : isServiceCommand(line); })() : null;
  return { record, running: ours !== false, ours };
}

function removeLock(file, nonce) {
  const data = readJson(file);
  if (data && (nonce === undefined || data.nonce === nonce)) {
    try { fs.unlinkSync(file); } catch { /* gone */ }
  }
}

/** Keep service.log under LOG_MAX_BYTES: its tail goes to service.log.1, then it is cut. */
export function capLog(file, { rotate = false } = {}) {
  try {
    const size = fs.statSync(file).size;
    if (!rotate && size <= LOG_MAX_BYTES) return;
    if (rotate) {
      fs.renameSync(file, `${file}.1`);
      return;
    }
    const keep = Math.min(size, LOG_MAX_BYTES / 2);
    const fd = fs.openSync(file, 'r');
    const buffer = Buffer.alloc(keep);
    try { fs.readSync(fd, buffer, 0, keep, size - keep); } finally { fs.closeSync(fd); }
    fs.writeFileSync(`${file}.1`, buffer, { mode: 0o600 });
    fs.truncateSync(file, 0);
  } catch {
    // no log yet, or it cannot be cut: never fail a call for it
  }
}

const NOTES = {
  'no-setup': 'The QuoteMem memory service is not running, and it is not set up to start by itself. Run `npx quotemem setup` once; after that it starts whenever an agent needs it.',
  'no-uvx': 'The QuoteMem memory service is not running and cannot start by itself: uvx is not on PATH. Install uv (https://docs.astral.sh/uv/), then it starts whenever an agent needs it.',
  'other-url': 'Another agent tool already started the QuoteMem memory service on this computer, at a different address than this tool uses (QUOTEMEM_URL). Use the same QUOTEMEM_URL in every tool, or leave it unset everywhere.',
};

/** host:port of a service URL, with localhost taken as 127.0.0.1 (where the service listens for it). */
function endpointOf(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname === 'localhost' ? '127.0.0.1' : parsed.hostname;
    return `${host}:${parsed.port || (parsed.protocol === 'https:' ? 443 : 80)}`;
  } catch {
    return null;
  }
}

/** Stop a just-started process: its whole process group on POSIX, its tree on Windows. */
function abort(pid) {
  try {
    if (process.platform === 'win32') execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true, timeout: 15_000 });
    else process.kill(-pid, 'SIGTERM');
  } catch {
    try { process.kill(pid); } catch { /* gone */ }
  }
}

/**
 * Remove service.lock when the pid it records is no longer the service (a
 * reused pid). The background check on Windows runs this; it only removes
 * the record it was asked about.
 * @returns {boolean} whether the lock was removed.
 */
export function dropReusedLock(lock, nonce, pid) {
  const record = readJson(lock);
  if (!record || record.nonce !== nonce || record.pid !== pid) return false;
  const line = commandLineOf(pid);
  if (line === null || isServiceCommand(line)) return false;
  removeLock(lock, nonce);
  return true;
}

// Windows: reading a command line takes PowerShell (about a second), which a
// hook must not wait for. A detached node (this file, run as a script) does
// it instead, at most every WINDOWS_CHECK_EVERY_MS. Like every detached
// process here, it is started by spawnDetached: it holds none of the hook's
// handles (a pipe of the agent tool's caller among them).
function checkInBackground(paths, record, spawnFn) {
  if (!/^(node|bun)(\.exe)?$/i.test(path.basename(process.execPath))) return;
  const mark = `${paths.lock}.checked`;
  try {
    if (Date.now() - fs.statSync(mark).mtimeMs < WINDOWS_CHECK_EVERY_MS) return;
  } catch {
    // never checked
  }
  try {
    fs.writeFileSync(mark, '', { mode: 0o600 });
    fs.utimesSync(mark, new Date(), new Date());
    const child = spawnFn(process.execPath, [fileURLToPath(import.meta.url), '--check-lock', paths.lock, String(record.nonce), String(record.pid)], { cwd: paths.dir, label: 'quotemem check-lock', mark: path.join(paths.dir, DIRECT_START_FILE) });
    child.on?.('error', () => {});
  } catch {
    // the next call tries again
  }
}

// The model relays a setup note to the user in its own words, so the note it
// gets names no environment variable; the log keeps the full note.
const MODEL_NOTES = {
  ...NOTES,
  'other-url': 'Another agent tool already started the QuoteMem memory service on this computer, at a different address than this tool uses. Give every agent tool the same QuoteMem service address, or leave the address unset in all of them.',
};

/** The user-facing note for an ensureService state, or null. */
export function setupNote(state) {
  return MODEL_NOTES[state] ?? null;
}

let memo = null;

/**
 * Start the local service if it may be started and is not already starting.
 * Called after a connection to the service URL was refused. Synchronous and
 * quick: it never waits for the service. The service is started by
 * spawnDetached and holds none of the caller's handles (on Windows the start
 * waits for PowerShell, about 0.2 s).
 * @param {object} settings - `url` and `localService` (from resolveSettings).
 * @param {{env?: object, log?: object, spawnFn?: Function, now?: number}} options
 *   `spawnFn(command, args, options)`: as spawnDetached (tests).
 * @returns {{state: 'remote'|'disabled'|'no-setup'|'no-uvx'|'starting'|'backoff'|'started'|'failed'|'other-url', pid?: number, note?: string, url?: string}}
 *   `remote`: not a loopback URL; `starting`: an instance is already
 *   recorded (it may still be downloading); `started`: this call started it;
 *   `other-url`: the recorded instance serves another address (`url`).
 */
export function ensureService(settings, { env = process.env, log, spawnFn = spawnDetached, platform = process.platform } = {}) {
  const local = settings?.localService;
  if (!local) return { state: 'remote' };
  if (memo && memo.key === `${local.dir}|${settings.url}` && Date.now() - memo.at < MEMO_MS) return memo.result;
  const result = startService(settings, local, { env, log, spawnFn, platform });
  if (result.note) log?.warnOnce?.(`autostart:${result.state}`, result.state === 'other-url' ? `${result.note} (it listens at ${result.url}; this tool uses ${settings.url})` : result.note);
  memo = { key: `${local.dir}|${settings.url}`, at: Date.now(), result };
  return result;
}

/** Forget the per-process memo (tests). */
export function resetAutostartMemo() {
  memo = null;
}

function startService(settings, local, { env, log, spawnFn, platform }) {
  if (local.enabled === false) return { state: 'disabled' };
  const paths = pathsIn(local.dir);
  const serviceEnv = readEnvFile(paths.envFile);
  if (serviceEnv === null) return { state: 'no-setup', note: NOTES['no-setup'] };
  const runEnv = serviceEnvironment(env, serviceEnv);
  const uvx = findUvx(runEnv);
  if (!uvx) return { state: 'no-uvx', note: NOTES['no-uvx'] };

  // An instance is recorded: still starting (uvx may be downloading), or gone.
  const current = stateOf(paths, { checkCommand: false });
  if (current.record && (Number.isInteger(current.record.pid) || current.running)) {
    // A record without startedAt is a starter's, written while it starts
    // the service (its pid is the starter's own, briefly).
    const service = Number.isFinite(current.record.startedAt);
    const age = Date.now() - (service ? current.record.startedAt : Number(current.record.at) || Date.now());
    if (current.running) {
      if (service) {
        // The recorded pid may have been reused by another program after the
        // service died. POSIX reads its command line at once (cheap); Windows
        // leaves that to a background check.
        if (platform === 'win32') {
          if (age > RESTART_BACKOFF_MS && Number.isInteger(current.record.pid)) checkInBackground(paths, current.record, spawnFn);
        } else if (Number.isInteger(current.record.pid)) {
          const line = commandLineOf(current.record.pid);
          if (line !== null && !isServiceCommand(line)) {
            removeLock(paths.lock, current.record.nonce);
            return startService(settings, local, { env, log, spawnFn, platform });
          }
        }
        // One lock serves every port: an instance for another address is not this tool's.
        const recorded = typeof current.record.url === 'string' ? endpointOf(current.record.url) : null;
        if (recorded && recorded !== endpointOf(settings.url)) {
          return { state: 'other-url', pid: current.record.pid, url: current.record.url, note: NOTES['other-url'] };
        }
        if (age > STARTUP_GRACE_MS) log?.warnOnce?.('autostart:slow', `the memory service started ${Math.round(age / 60_000)} minutes ago (pid ${current.record.pid}) still does not answer; see ${paths.log}`);
      }
      return { state: 'starting', pid: current.record.pid };
    }
    if (age >= 0 && age < RESTART_BACKOFF_MS) return { state: 'backoff', pid: current.record.pid };
  }

  let handle;
  try {
    fs.mkdirSync(paths.dir, { recursive: true, mode: 0o700 });
    handle = tryLock(paths.lock, { ignoreHost: true });
  } catch (error) {
    log?.warnOnce?.('autostart:lock', `could not take ${paths.lock} (${error?.code ?? 'error'})`);
    return { state: 'failed' };
  }
  if (!handle) return { state: 'starting' };

  const args = serveArgs(local);
  let fd = null;
  let child = null;
  try {
    // A racing starter that broke a stale lock may have taken ours meanwhile.
    if (!lockOwned(handle)) return { state: 'starting' };
    capLog(paths.log, { rotate: true });
    fd = fs.openSync(paths.log, 'a', 0o600);
    fs.writeSync(fd, `${new Date().toISOString()} quotemem: starting the memory service: uvx ${args.join(' ')} (settings from ${paths.envFile}).\n`
      + 'The first start downloads the service and its dependencies through uvx and may take a few minutes; the agents use memory once it answers.\n');
    fs.closeSync(fd);
    fd = null;
    child = spawnFn(uvx, args, { cwd: paths.dir, env: runEnv, logFile: paths.log, label: 'quotemem serve', log, mark: path.join(paths.dir, DIRECT_START_FILE) });
    child.on?.('error', (error) => log?.warnOnce?.('autostart:spawn', `could not start the memory service (${error?.code ?? 'error'})`));
    // Started with no pid known (Windows: PowerShell stopped before telling
    // it): recorded without one, as starting, so no second one is started.
    const pid = Number.isInteger(child.pid) ? child.pid : null;
    if (pid === null && child.started !== true) throw Object.assign(new Error('no pid'), { code: 'ENOPID' });
    if (!lockOwned(handle)) {
      // Another starter owns the lock now: its instance is the recorded one.
      if (pid !== null) abort(pid);
      child = null;
      return { state: 'starting' };
    }
    writeFileAtomic(paths.lock, JSON.stringify({
      host: hostName(), boot: bootTime(), pid, nonce: handle.nonce,
      startedAt: Date.now(), url: settings.url, release: SERVICE_RELEASE, log: paths.log,
      starter: process.pid, ...(child.via ? { via: child.via } : {}),
    }));
    child.unref?.();
    log?.debug?.(`started the memory service, pid ${pid ?? 'unknown'}`);
    return { state: 'started', pid };
  } catch (error) {
    // A started but unrecorded service could never be stopped: stop it now.
    if (child && Number.isInteger(child.pid)) abort(child.pid);
    removeLock(paths.lock, handle.nonce);
    log?.warnOnce?.('autostart:spawn', `could not start the memory service (${error?.code ?? 'error'}); see ${paths.log}`);
    return { state: 'failed' };
  } finally {
    if (fd !== null) try { fs.closeSync(fd); } catch { /* closed */ }
  }
}

/** Whether an error from a request means nothing listens at the URL. */
export function refused(error) {
  if (!error) return false;
  if (error.code === 'ECONNREFUSED') return true;
  return Array.isArray(error.errors) && error.errors.length > 0 && error.errors.every((e) => e?.code === 'ECONNREFUSED');
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Whether a process, or on POSIX any process of the group it leads, still runs. */
function groupAlive(pid) {
  if (process.platform !== 'win32') {
    try {
      process.kill(-pid, 0);
      return true;
    } catch (error) {
      if (error?.code === 'EPERM') return true;
    }
  }
  return alive(pid);
}

/**
 * Stop the recorded instance: only the process the lock names, and only
 * after its command line shows it is the service.
 * @returns {Promise<{stopped: boolean, message: string}>}
 */
export async function stopService(env = process.env, { waitMs = 15_000 } = {}) {
  const paths = servicePaths(env);
  const state = serviceState(env);
  if (!state.record) return { stopped: false, message: 'no started memory service is recorded (it was not started by a plugin, or it already stopped)' };
  const pid = state.record.pid;
  if (!state.running) {
    removeLock(paths.lock, state.record.nonce);
    return { stopped: false, message: `the recorded memory service (pid ${pid ?? 'unknown'}) is not running` };
  }
  if (pid === null) return { stopped: false, message: 'the memory service was started, but its pid is not known; stop the process that runs `quotemem serve` yourself' };
  if (state.ours === null) return { stopped: false, message: `could not read the command line of pid ${pid}; stop it yourself if it is the memory service` };
  if (process.platform === 'win32') {
    try {
      execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true, timeout: 15_000 });
    } catch {
      // checked below
    }
  } else {
    // The started process leads its own process group (uvx and the service
    // under it): the whole group is stopped, and waited for.
    const signal = (name) => {
      try { process.kill(-pid, name); } catch { try { process.kill(pid, name); } catch { /* gone */ } }
    };
    signal('SIGTERM');
    const end = Date.now() + waitMs;
    while (groupAlive(pid) && Date.now() < end) await sleep(100);
    if (groupAlive(pid)) signal('SIGKILL');
  }
  const end = Date.now() + 5000;
  while (groupAlive(pid) && Date.now() < end) await sleep(100);
  if (alive(pid)) return { stopped: false, message: `the memory service (pid ${pid}) did not stop` };
  removeLock(paths.lock, state.record.nonce);
  return { stopped: true, message: `stopped the memory service (pid ${pid})` };
}

// `node autostart.mjs --check-lock <lock> <nonce> <pid>`: the background check (Windows).
if (process.argv[2] === '--check-lock' && process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  dropReusedLock(process.argv[3], process.argv[4], Number(process.argv[5]));
}
