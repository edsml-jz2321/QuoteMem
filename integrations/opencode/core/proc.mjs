// Process facts for files shared between processes: who wrote a file, and
// whether that writer is gone. And the start of a detached process that
// holds none of this process's handles (spawnDetached).
import { randomBytes } from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Windows passes every inheritable handle of a process to each process it
// starts, whatever the new process's stdio is (Node's spawn always asks for
// that). A hook holds such handles of the program that ran the agent tool:
// a caller reading the tool's output through a pipe, through cmd and
// PowerShell, has its pipe inherited down to the hook. A detached child
// started by plain spawn would hold that pipe for its whole life, and the
// caller would wait for it (minutes, for a flusher sending to a slow
// service; until `quotemem stop`, for the service). So on Windows
// PowerShell starts a detached process for it, holding none of its
// handles: through WMI (Win32_Process.Create), whose service starts it
// outside every job of this process's host (a host that stops its hooks'
// job or tree when it exits, as Cursor's CLI does on Ctrl+C, does not stop
// it), or, when WMI refuses, through the shell (Start-Process,
// ShellExecuteEx), which escapes Node's own job only. What it starts is
// cmd.exe, hidden, which goes to the working folder, starts the program
// with NUL or the log file as its output and stays its parent: cmd needs
// no zone check to start a program, and the program's words come from the
// environment, so no quoting can change what runs. Only PowerShell itself
// holds the inherited handles, and this process waits for it (0.2 to
// 0.5 s): PowerShell does not run without a console, so it cannot be
// detached, and Node's job would stop it when a hook exits.
//
// The script is safe in ConstrainedLanguage mode, which PowerShell enters
// by itself under AppLocker or WDAC script rules: cmdlets, hashtables and
// property reads only, no .NET method. The started process is known from
// the pid file PowerShell writes, and from the file cmd writes first, not
// from PowerShell's exit: PowerShell stopped at LAUNCH_TIMEOUT_MS may
// already have started it. When neither file appears, nothing started, and
// the program is started directly instead (PowerShell missing or blocked):
// it then holds this process's handles, which the warning says.
const LAUNCH_VAR = 'QUOTEMEM_LAUNCH';
// First through WMI (Win32_Process.Create): the WMI service starts cmd.exe,
// so it is in no job of this process's host (a host that stops its hooks'
// process tree or job when it exits, as Cursor's CLI does on Ctrl+C, does
// not stop it), and it gets the environment it is given, as a list. Its pid
// counts only once cmd has written its `started` file (within 3 s): an
// endpoint rule that stops processes WMI creates (Defender's ASR rule, an
// EDR) leaves none. Else through the shell: Start-Process. The pid file's
// second line says which (`wmi` or `shell`).
// One start only: the cmd WMI starts runs the program only once it has
// made the `claim` folder, which only one process can make. A cmd slower
// than 3 s (a loaded machine, a scan) has not made it: PowerShell makes it
// first, so that cmd starts nothing, stops it and starts through the shell,
// whose cmd needs no claim. When PowerShell cannot make it, that cmd made it
// and runs the program: its pid stands. The claim is removed once that cmd
// is gone, or by this process once a WMI start stands.
const LAUNCH_SCRIPT = [
  `$c = $env:${LAUNCH_VAR}_CMD`,
  `$a = $env:${LAUNCH_VAR}_ARGS`,
  `$f = $env:${LAUNCH_VAR}_PID`,
  `$m = $env:${LAUNCH_VAR}_MARK`,
  `$k = $env:${LAUNCH_VAR}_CLAIM`,
  `Remove-Item -Path Env:${LAUNCH_VAR}_CMD, Env:${LAUNCH_VAR}_ARGS, Env:${LAUNCH_VAR}_PID, Env:${LAUNCH_VAR}_MARK, Env:${LAUNCH_VAR}_CLAIM -ErrorAction SilentlyContinue`,
  '$id = $null',
  "$via = 'wmi'",
  "if ($c) { try { $v = Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value }; $s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0; EnvironmentVariables = [string[]]$v }; $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $c; ProcessStartupInformation = $s } -ErrorAction Stop; if ($r.ReturnValue -eq 0) { $id = $r.ProcessId } } catch { $id = $null } }",
  '$n = 0',
  'while ($id -and -not (Test-Path -LiteralPath $m) -and $n -lt 60) { Start-Sleep -Milliseconds 50; $n = $n + 1 }',
  'if ($id -and -not (Test-Path -LiteralPath $m)) { $w = $false; try { New-Item -ItemType Directory -Path $k -ErrorAction Stop | Out-Null; $w = $true } catch { $w = $false }; if ($w) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue; Wait-Process -Id $id -Timeout 2 -ErrorAction SilentlyContinue; if (-not (Get-Process -Id $id -ErrorAction SilentlyContinue)) { Remove-Item -LiteralPath $k -ErrorAction SilentlyContinue }; $id = $null } }',
  "if (-not $id) { $via = 'shell'; $o = @{ FilePath = (Join-Path $env:SystemRoot 'System32\\cmd.exe'); ArgumentList = $a; WindowStyle = 'Hidden'; PassThru = $true }; $p = Start-Process @o; $id = $p.Id }",
  'Set-Content -LiteralPath $f -Value @($id, $via) -Encoding Ascii',
  '$id',
].join('; ');
// Tests: the launcher as AppLocker or WDAC would run it.
const CONSTRAINED = "$ExecutionContext.SessionState.LanguageMode = 'ConstrainedLanguage'; ";
const LAUNCH_TIMEOUT_MS = 10_000;
// How long to look for cmd's own file when PowerShell gave no pid.
const LAUNCH_SETTLE_MS = 1000;

function powershellPath(env) {
  return path.join(env.SystemRoot || process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
}

// The environment and cmd.exe arguments that run `command args`: every word
// is a variable cmd expands once (a "%", "&" or "[" in it stays a
// character), cleared again before the program starts. cmd first writes
// `started`, then goes to `cwd`. The cmd WMI starts first makes the `claim`
// folder, and runs nothing when it cannot (see LAUNCH_SCRIPT). `label`
// (letters, digits, spaces, "." "-" "=") titles the hidden console, so the
// command line says what runs ("quotemem serve": `quotemem stop` and the
// reused-pid check read it).
function windowsLaunch(command, args, { env, cwd, logFile, label, started, claim, pidFile, wmi, wmiDecoy, wmiSlow }) {
  const words = [command, ...args].map(String);
  const launchEnv = startEnv(env);
  for (const name of Object.keys(launchEnv)) if (name.toUpperCase().startsWith(`${LAUNCH_VAR}_`)) delete launchEnv[name];
  const extra = [started, claim, ...(cwd ? [cwd] : []), ...(logFile ? [logFile] : [])];
  const names = [];
  for (const [index, word] of [...words, ...extra].entries()) {
    if (!word || /["\0\r\n]/.test(word)) throw Object.assign(new Error('a word that cannot be passed'), { code: 'EINVAL' });
    const name = `${LAUNCH_VAR}_${index}`;
    // A closing quote follows the word: its trailing backslashes are doubled.
    launchEnv[name] = index < words.length ? word.replace(/(\\+)$/, '$1$1') : word;
    names.push(name);
  }
  const at = (index) => `"%${names[index]}%"`;
  const clear = names.map((name) => `set "${name}=" & `).join('');
  const program = words.map((_, index) => at(index)).join(' ');
  let next = words.length;
  const mark = `type NUL > ${at(next++)} & `;
  const claimAt = at(next++);
  const go = cwd ? `cd /d ${at(next++)} & ` : '';
  const output = logFile ? `>> ${at(next)} 2>&1` : '> NUL 2>&1';
  if (label && !/^[\w .=-]{1,64}$/.test(label)) throw Object.assign(new Error('a label that cannot be passed'), { code: 'EINVAL' });
  const title = label ? `title ${label} & ` : '';
  const run = `${mark}${go}${program} < NUL ${output}`;
  const cmdArgs = `/d /v:off /s /c "${clear}${title}${run}"`;
  launchEnv[`${LAUNCH_VAR}_ARGS`] = cmdArgs;
  // WMI takes the whole command line, whose cmd runs the program only after
  // making the claim folder (`wmi: false`, for tests: the shell only;
  // `wmiDecoy`, for tests: WMI starts a cmd that writes no `started` file,
  // as when an endpoint rule stops what WMI starts; `wmiSlow`, for tests:
  // its cmd waits 4 s first, as on a loaded machine).
  const cmdExe = path.join(env.SystemRoot || process.env.SystemRoot || 'C:\\Windows', 'System32', 'cmd.exe');
  const slow = wmiSlow ? 'ping -n 5 127.0.0.1 > NUL & ' : '';
  if (wmi) launchEnv[`${LAUNCH_VAR}_CMD`] = wmiDecoy ? `"${cmdExe}" /d /c exit` : `"${cmdExe}" /d /v:off /s /c "${slow}${clear}${title}mkdir ${claimAt} 2> NUL && (${run})"`;
  launchEnv[`${LAUNCH_VAR}_PID`] = pidFile;
  launchEnv[`${LAUNCH_VAR}_MARK`] = started;
  launchEnv[`${LAUNCH_VAR}_CLAIM`] = claim;
  return launchEnv;
}

function sleepSync(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/**
 * Remove a file or folder of a Windows start (the pid file, the `started`
 * mark, the claim folder), never throwing: the cmd that wrote one may still
 * hold it, and Windows then refuses its removal (EPERM, EACCES or EBUSY) for
 * a moment. It is tried again for at most about 0.3 s; one left in the
 * temporary folder is harmless (each start names its own).
 * @param {{rm?: Function, wait?: Function}} hooks - for tests.
 * @returns {boolean} whether it is gone.
 */
export function removeLaunchFile(file, { rm = fs.rmSync, wait = sleepSync } = {}) {
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    try {
      rm(file, { recursive: true, force: true });
      return true;
    } catch (error) {
      if (!['EPERM', 'EACCES', 'EBUSY'].includes(error?.code)) return false;
      if (attempt < 10) wait(5 * attempt);
    }
  }
  return false;
}

function readPid(file) {
  try {
    const pid = Number.parseInt(fs.readFileSync(file, 'ascii').trim(), 10);
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function readVia(file) {
  try {
    return fs.readFileSync(file, 'ascii').split(/\r?\n/)[1]?.trim() || 'shell';
  } catch {
    return 'shell';
  }
}

/** Where a host records that it had to start a process directly (doctor reads it). */
export const DIRECT_START_FILE = 'direct-start.json';
/**
 * Set to 1 in the environment of a process started directly: it holds the
 * starter's handles, so it keeps its life short (the background flusher then
 * sends the queue and exits instead of waiting for idle chats).
 */
export const STARTED_DIRECTLY_VAR = 'QUOTEMEM_STARTED_DIRECTLY';

// The environment of a start: never the mark of a direct start this process
// itself may carry (only startDirectly sets it, for its own child).
function startEnv(env) {
  const out = { ...env };
  delete out[STARTED_DIRECTLY_VAR];
  return out;
}

// A detached process spawned by Node itself: its own process group on
// POSIX, NUL or the log file as its output.
function spawnPlain(command, args, { env, cwd, logFile }) {
  let fd = null;
  try {
    if (logFile) fd = fs.openSync(logFile, 'a', 0o600);
    const child = spawn(command, args, { cwd, env, detached: true, windowsHide: true, stdio: fd === null ? 'ignore' : ['ignore', fd, fd] });
    child.unref();
    return { pid: Number.isInteger(child.pid) ? child.pid : null, on: (event, fn) => child.on(event, fn) };
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

// PowerShell could not start it: start the program directly, and say so
// once (log) and on disk (`mark`, for `quotemem doctor`).
function startDirectly(command, args, { env, cwd, logFile, log, mark, reason }) {
  log?.warnOnce?.('detached-direct', `PowerShell could not start a background process (${reason}); it was started directly, so a program that reads this agent's output through a pipe may wait until that process ends`);
  if (mark) {
    try {
      fs.mkdirSync(path.dirname(mark), { recursive: true, mode: 0o700 });
      fs.writeFileSync(mark, JSON.stringify({ at: Date.now(), reason }), { mode: 0o600 });
    } catch {
      // the warning is in the log
    }
  }
  const marked = { ...startEnv(env), [STARTED_DIRECTLY_VAR]: '1' };
  return { ...spawnPlain(command, args, { env: marked, cwd, logFile }), direct: true };
}

/**
 * Start a detached process that outlives this one and holds none of its
 * handles: a caller of this process that reads its output through a pipe
 * sees the pipe end when this process ends, not when the detached one does.
 * On POSIX it is spawned in its own process group with NUL or the log file
 * as its output; Node marks every other descriptor close-on-exec. On
 * Windows it is started through WMI or the shell, as above, and this call waits
 * for that (0.2 to 0.5 s); the pid is that of the cmd.exe that runs it
 * (taskkill /T stops both). When PowerShell cannot start it, the program is
 * started directly.
 * @param {string} command - the program (a path to an executable).
 * @param {string[]} args - its arguments.
 * @param {{env?: object, cwd?: string, logFile?: string, label?: string, log?: object,
 *   mark?: string, platform?: string, powershell?: string,
 *   constrained?: boolean}} options - `logFile` gets stdout and stderr
 *   (appended); `label` names it on Windows (see windowsLaunch); `log` and
 *   `mark` (a file) record a direct start;
 *   `powershell`, `constrained`, `wmi` (false: the shell only), `wmiDecoy`
 *   and `wmiSlow` are for tests.
 * @returns {{pid: number|null, on: Function, started?: boolean, direct?: boolean}}
 *   the process's pid (null, with `started`, when it started but PowerShell
 *   stopped before telling it); `direct` when PowerShell could not start it;
 *   `on('error', fn)` for a start that failed later.
 */
export function spawnDetached(command, args = [], options = {}) {
  const { env = process.env, cwd, logFile = null, platform = process.platform } = options;
  if (platform !== 'win32') return spawnPlain(command, args, { env: startEnv(env), cwd, logFile });
  const launch = prepareLaunch(command, args, options);
  const run = spawnSync(launch.shell, launch.argv, { env: launch.env, encoding: 'utf8', windowsHide: true, timeout: LAUNCH_TIMEOUT_MS, stdio: ['ignore', 'pipe', 'pipe'] });
  // PowerShell never ran (missing, blocked): nothing started.
  const ran = !run.error || run.error.code === 'ETIMEDOUT';
  const end = Date.now() + (ran ? LAUNCH_SETTLE_MS : 0);
  while (readPid(launch.pidFile) === null && !fs.existsSync(launch.started) && Date.now() < end) sleepSync(25);
  return finishLaunch(launch, command, args, options, ran ? `it exited ${run.status ?? run.error?.code}` : (run.error?.code ?? 'it could not run'));
}

// The files and the PowerShell command line of one Windows start.
function prepareLaunch(command, args, { env = process.env, cwd, logFile = null, label = null, powershell = null, constrained = false, wmi = true, wmiDecoy = false, wmiSlow = false }) {
  const nonce = randomBytes(8).toString('hex');
  const started = path.join(os.tmpdir(), `quotemem-launch-${nonce}.started`);
  const claim = path.join(os.tmpdir(), `quotemem-launch-${nonce}.claim`);
  const pidFile = path.join(os.tmpdir(), `quotemem-launch-${nonce}.pid`);
  return {
    started,
    claim,
    pidFile,
    env: windowsLaunch(command, args, { env, cwd, logFile, label, started, claim, pidFile, wmi, wmiDecoy, wmiSlow }),
    shell: powershell ?? powershellPath(env),
    argv: ['-NoProfile', '-NonInteractive', '-Command', (constrained ? CONSTRAINED : '') + LAUNCH_SCRIPT],
  };
}

// What a Windows start gave, once PowerShell is done: the pid (and which
// way it started), a start with no pid known, or, when nothing started, a
// direct start (`reason`: why PowerShell started nothing).
function finishLaunch(launch, command, args, { env = process.env, cwd, logFile = null, log = null, mark = null }, reason) {
  try {
    const pid = readPid(launch.pidFile);
    if (pid !== null) {
      // Which way it started: `wmi`, or `shell` when WMI refused or its process did not start.
      const via = readVia(launch.pidFile);
      log?.debug?.(`started a background process through ${via}, pid ${pid}`);
      // The cmd WMI started made the claim and runs the program: no other
      // cmd of this start waits for it any more.
      if (via === 'wmi') removeLaunchFile(launch.claim);
      return { pid, via, on: () => {} };
    }
    if (fs.existsSync(launch.started)) return { pid: null, started: true, on: () => {} };
    return startDirectly(command, args, { env, cwd, logFile, log, mark, reason });
  } finally {
    removeLaunchFile(launch.pidFile);
    removeLaunchFile(launch.started);
  }
}

const pending = new Set();

/**
 * spawnDetached that does not hold this process up: on Windows PowerShell
 * runs while this process goes on (a hook prints its answer meanwhile).
 * PowerShell runs in Node's own job, so this process must not exit before
 * it is done: settleLaunches waits for every start still running.
 * QUOTEMEM_TEST_LAUNCH_DELAY_MS (tests) makes every start that much later.
 * @returns {Promise<{pid: number|null, on: Function, via?: string, started?: boolean, direct?: boolean}>}
 */
export function spawnDetachedAsync(command, args = [], options = {}) {
  const task = (async () => {
    const delay = Number((options.env ?? process.env).QUOTEMEM_TEST_LAUNCH_DELAY_MS ?? process.env.QUOTEMEM_TEST_LAUNCH_DELAY_MS) || 0;
    if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
    if ((options.platform ?? process.platform) !== 'win32') return spawnDetached(command, args, options);
    const launch = prepareLaunch(command, args, options);
    const outcome = await new Promise((resolve) => {
      const child = spawn(launch.shell, launch.argv, { env: launch.env, windowsHide: true, stdio: 'ignore' });
      const timer = setTimeout(() => child.kill(), LAUNCH_TIMEOUT_MS);
      child.on('error', (error) => { clearTimeout(timer); resolve({ ran: false, why: error?.code ?? 'it could not run' }); });
      child.on('exit', (status) => { clearTimeout(timer); resolve({ ran: true, why: `it exited ${status}` }); });
    });
    const end = Date.now() + (outcome.ran ? LAUNCH_SETTLE_MS : 0);
    while (readPid(launch.pidFile) === null && !fs.existsSync(launch.started) && Date.now() < end) await new Promise((resolve) => setTimeout(resolve, 25));
    return finishLaunch(launch, command, args, options, outcome.why);
  })();
  pending.add(task);
  task.then(() => pending.delete(task), () => pending.delete(task));
  return task;
}

/**
 * Wait, at most `ms`, for the starts spawnDetachedAsync has under way (before
 * this process exits), and for those asked for while it waits: a start that
 * is done may ask for another (DiskQueue.startFlusher tries a flusher start
 * again then), which is waited for too, within the same time.
 */
export async function settleLaunches(ms) {
  const end = Date.now() + Math.max(0, ms);
  while (pending.size > 0) {
    const left = end - Date.now();
    if (left <= 0) return;
    let timer;
    await Promise.race([Promise.allSettled([...pending]), new Promise((resolve) => { timer = setTimeout(resolve, left); })]);
    clearTimeout(timer);
    // What a settled start runs once it is done (a start it asks for) runs first.
    await new Promise((resolve) => { setImmediate(resolve); });
  }
}

/**
 * Whether PowerShell starts background processes here (Windows), for
 * `quotemem doctor`: its language mode, or why it cannot run.
 * @returns {{ok: boolean, detail: string}|null} null off Windows.
 */
export function launcherCheck({ env = process.env, platform = process.platform, powershell = null } = {}) {
  if (platform !== 'win32') return null;
  const run = spawnSync(powershell ?? powershellPath(env), ['-NoProfile', '-NonInteractive', '-Command', '$ExecutionContext.SessionState.LanguageMode'], { env, encoding: 'utf8', windowsHide: true, timeout: LAUNCH_TIMEOUT_MS, stdio: ['ignore', 'pipe', 'pipe'] });
  const mode = String(run.stdout ?? '').trim();
  if (run.error || run.status !== 0 || !mode) return { ok: false, detail: run.error?.code ?? `exit ${run.status}` };
  return { ok: true, detail: mode };
}

// A file from another host cannot be checked by pid; it counts as abandoned
// only once it has not been written for this long (longer than any retry wait).
export const FOREIGN_STALE_MS = 3 * 3_600_000;
const BOOT_SLACK_MS = 120_000;

export function bootTime() {
  return Date.now() - os.uptime() * 1000;
}

export function hostName() {
  try {
    return os.hostname();
  } catch {
    return '';
  }
}

export function alive(pid) {
  if (pid === process.pid) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === 'EPERM';
  }
}

/** The owner record a shared file carries. */
export function ownerRecord(extra = {}) {
  return { host: hostName(), boot: bootTime(), pid: process.pid, ...extra };
}

/**
 * Whether the process that wrote a shared file is gone. A file records its
 * host, boot time and pid. On this host and boot, the pid is checked; a file
 * from an earlier boot is dead whatever its pid; a file from another host
 * (a shared home folder) is dead once it has not been written for hours.
 * With `ignoreHost`, the host name is not compared (it changes with the
 * network on macOS): the boot time and pid decide alone.
 */
export function ownerGone(data, pid, mtimeMs, { ignoreHost = false } = {}) {
  if (!ignoreHost && typeof data?.host === 'string' && data.host !== hostName()) {
    return Date.now() - mtimeMs > FOREIGN_STALE_MS;
  }
  if (Number.isFinite(data?.boot) && Math.abs(data.boot - bootTime()) > BOOT_SLACK_MS) return true;
  return !alive(pid);
}

/**
 * The home folder: the one function every part of QuoteMem uses, the same
 * one the service finds (Python's Path.home()). On Windows USERPROFILE, then
 * HOMEDRIVE+HOMEPATH (an MSYS or Git Bash HOME is not it); elsewhere HOME;
 * then the operating system's own.
 */
export function homeDir(env = process.env, platform = process.platform) {
  if (platform === 'win32') {
    return env?.USERPROFILE || (env?.HOMEDRIVE && env?.HOMEPATH ? `${env.HOMEDRIVE}${env.HOMEPATH}` : '') || os.homedir();
  }
  return env?.HOME || os.homedir();
}

// Two people signed in at once on one computer (fast user switching, RDP, a
// shared host) share 127.0.0.1:8765: the second one's plugins reach the
// first one's service, which refuses their token. A port of this user's
// own, the same in every tool: 20000-25999, from the user name and home.
export const USER_PORT_BASE = 20000;
export const USER_PORT_COUNT = 6000;

/** This OS user's own port for the local service (userPort), for a second user of the computer. */
export function userPort(env = process.env) {
  let name = '';
  try { name = os.userInfo().username; } catch { name = env.USER || env.USERNAME || ''; }
  const seed = `${name}\n${homeDir(env)}`;
  let hash = 2166136261;
  for (const ch of seed) {
    hash ^= ch.codePointAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return USER_PORT_BASE + (hash % USER_PORT_COUNT);
}

// Windows: a PowerShell script that prints, for each process listening on
// `port` at `address` or a wildcard, "1" or "0" (its SID is this process's
// user's), a tab and its Domain\User, or "?" when its owner cannot be read.
// The pids come from Get-NetTCPConnection, else from netstat (the CIM
// provider lists nothing on some machines); the SID, not a name rebuilt from
// USERDOMAIN, which differs under a reduced environment and for domain accounts.
function windowsOwnerScript(port, address) {
  const want = `@('${address}', '0.0.0.0', '::')`;
  return `$me = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value; $want = ${want}; `
    + `$ids = @(Get-NetTCPConnection -LocalPort ${Number(port)} -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalAddress -in $want } | ForEach-Object { $_.OwningProcess }); `
    + `if ($ids.Count -eq 0) { $ids = @(@(netstat -ano -p TCP) + @(netstat -ano -p TCPv6) | ForEach-Object { $f = -split $_; if ($f.Count -ge 5 -and $f[0] -eq 'TCP' -and $f[3] -eq 'LISTENING') { $i = $f[1].LastIndexOf(':'); if ([int]$f[1].Substring($i + 1) -eq ${Number(port)} -and $f[1].Substring(0, $i).Trim('[', ']') -in $want) { [int]$f[4] } } }) }; `
    + `$ids | Select-Object -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); $o = $null; $s = $null; if ($p) { $o = Invoke-CimMethod -InputObject $p -MethodName GetOwner; $s = Invoke-CimMethod -InputObject $p -MethodName GetOwnerSid }; if ($o -and $o.User) { $(if ($s -and $s.Sid -eq $me) { '1' } else { '0' }) + [char]9 + $o.Domain + '\\' + $o.User } else { '?' } }`;
}

/**
 * The OS user that listens on a local TCP port, when it can be told: Linux
 * reads /proc/net/tcp; macOS asks lsof; Windows asks PowerShell for the
 * listening process and its owner. For `quotemem doctor`.
 * @returns {{mine: boolean, owner: string}|null}
 */
export function portOwner(port, { platform = process.platform } = {}) {
  try {
    if (platform === 'linux') {
      const hex = Number(port).toString(16).toUpperCase().padStart(4, '0');
      for (const table of ['/proc/net/tcp', '/proc/net/tcp6']) {
        let text = '';
        try { text = fs.readFileSync(table, 'utf8'); } catch { continue; }
        for (const line of text.split('\n').slice(1)) {
          const parts = line.trim().split(/\s+/);
          // local_address ends in :PORT; state 0A is LISTEN; uid is the 8th field.
          if (parts.length > 7 && parts[1]?.endsWith(`:${hex}`) && parts[3] === '0A') {
            const uid = Number(parts[7]);
            return { mine: uid === process.getuid?.(), owner: `uid ${uid}` };
          }
        }
      }
      return null;
    }
    if (platform === 'darwin') {
      const out = spawnSync('lsof', ['-nP', `-iTCP:${Number(port)}`, '-sTCP:LISTEN', '-Fu'], { encoding: 'utf8', timeout: 5000 });
      const uid = /^u(\d+)$/m.exec(String(out.stdout ?? ''))?.[1];
      return uid === undefined ? null : { mine: Number(uid) === process.getuid?.(), owner: `uid ${uid}` };
    }
    if (platform === 'win32') {
      const out = spawnSync(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', windowsOwnerScript(port, '127.0.0.1')], { encoding: 'utf8', windowsHide: true, timeout: 10_000 });
      const line = String(out.stdout ?? '').split(/\r?\n/).map((text) => text.trim()).find(Boolean);
      if (!line || line === '?') return null;
      return { mine: line.startsWith('1\t'), owner: line.slice(2) };
    }
  } catch {
    // unknown
  }
  return null;
}

// The /proc/net hex form of an address (Linux): IPv4 as 4 little-endian
// bytes, IPv6 as four 32-bit little-endian words.
function procHex(address) {
  if (/^\d+\.\d+\.\d+\.\d+$/.test(address)) {
    return address.split('.').map((part) => Number(part).toString(16).toUpperCase().padStart(2, '0')).reverse().join('');
  }
  const halves = address.split('::');
  const head = halves[0] ? halves[0].split(':') : [];
  const tail = halves.length > 1 && halves[1] ? halves[1].split(':') : [];
  const groups = [...head, ...Array(8 - head.length - tail.length).fill('0'), ...tail].map((g) => g.padStart(4, '0'));
  const bytes = groups.join('').match(/../g);
  let out = '';
  for (let i = 0; i < 16; i += 4) out += bytes.slice(i, i + 4).reverse().join('');
  return out.toUpperCase();
}

const ANY_V4 = '00000000';
const ANY_V6 = '0'.repeat(32);

/**
 * The owners of the sockets listening on `port` that a connection to
 * `address` (an IP address) may reach: that address, or a wildcard. Every
 * one must be this user's for `mine`; another owner is named. Null when none
 * is seen or the owners cannot be read (macOS lists only the caller's own
 * sockets to a user without administrator rights; Windows does not name
 * another user's process to one). Asynchronous, and bounded by `timeoutMs`
 * and `signal`: lsof (macOS) and PowerShell (Windows) can take seconds, so
 * only `quotemem doctor` and `status` ask (listener.mjs describeRefusal).
 * @returns {Promise<{mine: boolean, owner: string}|null>}
 */
export async function portOwnerAsync(port, address, { platform = process.platform, timeoutMs = 3000, signal } = {}) {
  try {
    if (platform === 'linux') {
      const v4 = /^\d+\.\d+\.\d+\.\d+$/.test(address);
      const wanted = new Set(v4
        ? [procHex(address), ANY_V4, `0000000000000000FFFF0000${procHex(address)}`, ANY_V6]
        : [procHex(address), ANY_V6]);
      const uids = [];
      for (const table of ['/proc/net/tcp', '/proc/net/tcp6']) {
        let text = '';
        try { text = fs.readFileSync(table, 'utf8'); } catch { continue; }
        for (const line of text.split('\n').slice(1)) {
          const parts = line.trim().split(/\s+/);
          if (parts.length <= 7 || parts[3] !== '0A') continue;
          const [hex, portHex] = parts[1].split(':');
          if (Number.parseInt(portHex, 16) === Number(port) && wanted.has(hex.toUpperCase())) uids.push(Number(parts[7]));
        }
      }
      return ownersVerdict(uids.map((uid) => ({ mine: uid === process.getuid?.(), owner: `uid ${uid}` })));
    }
    if (platform === 'darwin') {
      const at = address.includes(':') ? `[${address}]` : address;
      const out = await runBounded('lsof', ['-nP', `-iTCP@${at}:${Number(port)}`, '-sTCP:LISTEN', '-Fu'], { timeoutMs, signal });
      const uids = [...String(out ?? '').matchAll(/^u(\d+)$/gm)].map((m) => Number(m[1]));
      return ownersVerdict(uids.map((uid) => ({ mine: uid === process.getuid?.(), owner: `uid ${uid}` })));
    }
    if (platform === 'win32') {
      const script = windowsOwnerScript(port, address);
      const shell = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
      const out = await runBounded(shell, ['-NoProfile', '-NonInteractive', '-Command', script], { timeoutMs, signal });
      const names = String(out ?? '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      if (names.length === 0 || names.includes('?')) return null;
      return ownersVerdict(names.map((name) => ({ mine: name.startsWith('1\t'), owner: name.slice(2) })));
    }
  } catch {
    // unknown
  }
  return null;
}

// All seen owners together: null when none, else the first that is not this user's, else this user's.
function ownersVerdict(owners) {
  if (owners.length === 0) return null;
  return owners.find((owner) => !owner.mine) ?? owners[0];
}

// A program's stdout, or null when it fails, outlasts `timeoutMs` or `signal`
// aborts. The time limit holds even when a child of the program (netstat
// under PowerShell) keeps its output open after it is stopped.
function runBounded(command, args, { timeoutMs, signal }) {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(null);
      return;
    }
    let child;
    let timer;
    let out = '';
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', stop);
      child?.stdout?.destroy();
      resolve(value);
    };
    function stop() {
      try { child?.kill(); } catch { /* already gone */ }
      finish(null);
    }
    try {
      child = spawn(command, args, { stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true });
    } catch {
      finish(null);
      return;
    }
    timer = setTimeout(stop, Math.max(1, timeoutMs));
    signal?.addEventListener('abort', stop, { once: true });
    child.stdout.on('data', (d) => { if (out.length < 65_536) out += d; });
    child.stdout.on('error', () => {});
    child.on('error', () => finish(null));
    child.on('close', (code, killed) => finish(killed ? null : out));
  });
}
