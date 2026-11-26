#!/usr/bin/env node
// quotemem: one command for every QuoteMem integration (npm package `quotemem`).
//
//   quotemem setup                      the memory service's model settings, once;
//                                       the service then starts by itself
//   quotemem status | stop              the memory service on this computer
//   quotemem install <tool> [--write]   set up one agent (dry run by default)
//   quotemem mcp                        the QuoteMem MCP server (stdio)
//   quotemem hook <agent> <event>       the hook program for one agent
//   quotemem doctor                     check the settings, uvx and the service
//   quotemem --version | --help
//
//   quotemem serve | check | memories | export | rebuild | delete-user | repair | bench ...
//                                       the service's own commands: run as
//                                       `uvx quotemem <command> <args>` (the
//                                       Python package `quotemem`), with the
//                                       settings of ~/.quotemem/service.env;
//                                       memories, repair and delete-user also
//                                       with the plugins' URL, user and token
//
// Everything runs on this computer. Plugin settings come from QUOTEMEM_*
// variables or ~/.quotemem/env; the service's model settings from
// ~/.quotemem/service.env. Keys and the token are never printed, and never
// taken from the command line.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { nativeMarkers } from '../core/native.mjs';
import { homeDir } from '../core/proc.mjs';
import { cmdSetup, cmdStatus, cmdStop, doctorLines, runServiceCommand } from './service.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = path.resolve(here, '..', '..');
const integration = (...parts) => path.join(PACKAGE_DIR, 'integrations', ...parts);
const VERSION = JSON.parse(fs.readFileSync(path.join(PACKAGE_DIR, 'package.json'), 'utf8')).version;

/** Hook programs by agent: [script, the arguments before the event]. */
const HOOKS = {
  'claude-code': [integration('claude-code', 'scripts', 'hook.mjs'), []],
  codex: [integration('codex', 'hook.mjs'), []],
  gemini: [integration('hooks', 'quotemem-hook.mjs'), ['gemini']],
  cursor: [integration('hooks', 'quotemem-hook.mjs'), ['cursor']],
  cline: [integration('hooks', 'quotemem-hook.mjs'), ['cline']],
  goose: [integration('hooks', 'quotemem-hook.mjs'), ['goose']],
  devin: [integration('hooks', 'quotemem-hook.mjs'), ['devin']],
};
const HOOK_ALIASES = { claude: 'claude-code', 'gemini-cli': 'gemini', windsurf: 'devin' };

/** Commands of the Python service, handed to `uvx quotemem` unchanged. */
const SERVICE_COMMANDS = ['serve', 'check', 'memories', 'export', 'rebuild', 'delete-user', 'repair', 'bench'];
/** Where `quotemem serve` listens unless told otherwise (the service's DEFAULT_PORT). */
const SERVICE_PORT = '8765';
/** How long a SIGINT may take to reach the service on its own before it is passed on. */
const SIGINT_GRACE_MS = 500;

/** The service release matching this package; `bench` also needs the bench extra. */
function serviceSpec(command) {
  return `quotemem${command === 'bench' ? '[bench]' : ''}==${VERSION}`;
}

const USAGE = `quotemem ${VERSION}: auditable long-term memory for agents

QuoteMem runs on your computer: the agents' plugins share one memory service
(\`quotemem serve\`). Run \`quotemem setup\` once; after that the service
starts by itself whenever an agent needs it, and keeps running for all of them.

  quotemem setup                      give the memory service its models, keys,
                                      thinking and review, once
  quotemem status                     is the memory service running? pid, URL, log
  quotemem stop                       stop the service a plugin started (it starts
                                      again when needed unless QUOTEMEM_AUTOSTART=0)

plugin commands:
  quotemem install <tool> [--write]   set up one agent; a dry run unless --write
  quotemem mcp                        run the QuoteMem MCP server (stdio)
  quotemem hook <agent> <event>       run the hook program (agents: ${Object.keys(HOOKS).join(', ')})
  quotemem doctor                     check the settings, the token, uvx and the service
  quotemem --version

service commands (run as \`uvx quotemem <command> ...\` with the settings of
~/.quotemem/service.env; they need uv):
  quotemem serve [options]            run the memory service on 127.0.0.1:${SERVICE_PORT} by hand
                                      (data: ~/.local/share/quotemem)
  quotemem check                      check its configuration and probe the model providers
  quotemem memories [--user U] [--status S] [--json | --html FILE]
                                      list what is remembered (read-only): status,
                                      statement, dates and quotes; --html FILE
                                      writes a local read-only page. The user is
                                      the plugins' (QUOTEMEM_USER_ID, else your OS
                                      user name) unless --user is given
  quotemem export | rebuild | delete-user | repair
                                      store maintenance; \`quotemem <command> --help\`
  quotemem bench ...                  the benchmark tools

The plugins use http://127.0.0.1:${SERVICE_PORT} unless QUOTEMEM_URL says otherwise.
The token is read from a file, never needed in the environment: a service on
this computer is reached with its own token, another one with the file
QUOTEMEM_TOKEN_FILE names. QUOTEMEM_USER_ID defaults to your OS user name
everywhere, so every agent shares one memory. memories, delete-user and repair
reach the service the same way, also with the settings of ~/.quotemem/env.
Run \`quotemem install --help\` for the tools.
`;

/** `quotemem setup` options that take a value (never a key). */
const SETUP_OPTIONS = ['--preset', '--chat-url', '--chat-model', '--chat-structured-output', '--chat-reasoning', '--chat-model-pattern', '--embed-url', '--embed-model', '--review'];

function parseArgs(argv) {
  const flags = {};
  const rest = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (['--write', '--help', '-h', '--yes', '--no-check'].includes(arg)) flags[arg.replace(/^-+/, '')] = true;
    else if (arg.startsWith('--') && arg.includes('=')) flags[arg.slice(2, arg.indexOf('='))] = arg.slice(arg.indexOf('=') + 1);
    else if (['--codex-home', '--dir', '--project', '--node', ...SETUP_OPTIONS].includes(arg) && i + 1 < argv.length) flags[arg.slice(2)] = argv[++i];
    else rest.push(arg);
  }
  return { flags, rest };
}

async function cmdInstall(argv) {
  const { flags, rest } = parseArgs(argv);
  const { install, toolList, toolOf } = await import('./install.mjs');
  if (flags.help || flags.h || rest.length === 0) {
    process.stdout.write([
      'usage: quotemem install <tool> [--write]',
      '',
      'Prints what it would do; --write writes the files. The token is never written.',
      'Options: --project DIR (Devin, Continue; default .), --codex-home DIR (Codex),',
      '         --dir DIR (Cline hooks folder), --node PATH (default: this node)',
      '',
      'tools:',
      ...toolList(),
      '',
    ].join('\n'));
    return rest.length === 0 && !(flags.help || flags.h) ? 2 : 0;
  }
  if (!toolOf(rest[0])) {
    process.stderr.write(`quotemem: unknown tool ${JSON.stringify(rest[0])}\ntools:\n${toolList().join('\n')}\n`);
    return 2;
  }
  const result = install(rest[0], {
    write: Boolean(flags.write),
    node: flags.node ?? process.execPath,
    project: path.resolve(flags.project ?? process.cwd()),
    args: flags,
  });
  (result.ok ? process.stdout : process.stderr).write(`${result.lines.join('\n')}\n`);
  return result.ok ? 0 : 1;
}

async function cmdMcp(argv) {
  // Complete the environment from ~/.quotemem/env, as the agents' launchers do.
  const { hostEnv } = await import(pathToFileURL(integration('hooks', 'env.mjs')).href);
  const { env, notes } = hostEnv(process.env);
  for (const [name, value] of Object.entries(env)) {
    if (name.startsWith('QUOTEMEM_') && (process.env[name] === undefined || process.env[name] === '')) process.env[name] = value;
  }
  for (const note of notes) process.stderr.write(`[quotemem-mcp] ${note}\n`);
  process.argv = [process.argv[0], integration('dsh', 'server.mjs'), ...argv];
  await import(pathToFileURL(integration('dsh', 'server.mjs')).href);
}

async function cmdHook(argv) {
  const name = String(argv[0] ?? '').toLowerCase();
  const agent = HOOK_ALIASES[name] ?? name;
  if (!Object.hasOwn(HOOKS, agent) || argv.length < 2 || argv.includes('--help')) {
    const out = argv.includes('--help') ? process.stdout : process.stderr;
    out.write(`usage: quotemem hook <agent> <event> < payload.json\nagents: ${Object.keys(HOOKS).join(', ')}\n`);
    return argv.includes('--help') ? 0 : 2;
  }
  const [script, prefix] = HOOKS[agent];
  // Each hook program reads its own argv and runs when it is the main script.
  process.argv = [process.argv[0], script, ...prefix, ...argv.slice(1)];
  await import(pathToFileURL(script).href);
  return null;
}

/** The first executable named `names` on PATH, or null. */
function findOnPath(names, env = process.env) {
  const dirs = String(env.PATH ?? env.Path ?? '').split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    for (const name of names) {
      const file = path.join(dir, name);
      try {
        if (!fs.statSync(file).isFile()) continue;
        fs.accessSync(file, fs.constants.X_OK);
        return file;
      } catch {
        // not here
      }
    }
  }
  return null;
}

function uvxPath(env = process.env) {
  return findOnPath(process.platform === 'win32' ? ['uvx.exe', 'uvx'] : ['uvx'], env);
}

/**
 * The user the plugins store under (QUOTEMEM_USER_ID, also from
 * ~/.quotemem/env, else the OS user name), so `quotemem memories` lists the
 * memories the agents saved; the service's own default user is `default`.
 */
async function pluginUser() {
  const { hostEnv } = await import(pathToFileURL(integration('hooks', 'env.mjs')).href);
  const { resolveSettings } = await import(pathToFileURL(integration('core', 'config.mjs')).href);
  return resolveSettings({}, hostEnv(process.env).env, {}).userId;
}

async function cmdService(command, argv) {
  // Pinned to this package's version, so a cached older service is not
  // reused; the settings of ~/.quotemem/service.env (from `quotemem setup`)
  // are added to the environment, and a command that calls the service gets
  // the plugins' URL and token (clientEnvironment in service.mjs).
  if (command === 'memories' && !argv.some((arg) => arg === '--user' || arg.startsWith('--user='))) {
    argv = [...argv, '--user', await pluginUser()];
  }
  const { child } = await runServiceCommand(command, argv, { release: VERSION });
  if (!child) {
    process.stderr.write(`quotemem: \`quotemem ${command}\` runs the Python service through uvx, which is not on PATH; install uv (https://docs.astral.sh/uv/) or run \`pipx run --spec '${serviceSpec(command)}' quotemem ${command}\`\n`);
    return 127;
  }
  const alive = () => child.exitCode === null && child.signalCode === null;
  const passOn = (signal) => () => { if (alive()) child.kill(signal); };
  // Ctrl-C reaches the service directly (same process group), so a SIGINT is
  // passed on only when the service is still running after a short grace: a
  // SIGINT sent to this process alone (kill -INT, a supervisor) then reaches it
  // too. It is passed on at most once.
  let sigintTimer = null;
  const onSigint = () => {
    if (sigintTimer !== null || !alive()) return;
    sigintTimer = setTimeout(passOn('SIGINT'), SIGINT_GRACE_MS);
  };
  const handlers = { SIGINT: onSigint, SIGTERM: passOn('SIGTERM'), SIGHUP: passOn('SIGHUP') };
  for (const [signal, handler] of Object.entries(handlers)) process.on(signal, handler);
  const [code, signal] = await new Promise((resolve) => {
    child.on('error', (error) => {
      process.stderr.write(`quotemem: could not start uvx (${error.code ?? error.name})\n`);
      resolve([127, null]);
    });
    child.on('exit', (exitCode, exitSignal) => resolve([exitCode, exitSignal]));
  });
  clearTimeout(sigintTimer);
  for (const [name, handler] of Object.entries(handlers)) process.off(name, handler);
  if (signal) {
    process.kill(process.pid, signal);
    return 128 + (os.constants.signals[signal] ?? 0);
  }
  return code ?? 1;
}

async function cmdDoctor() {
  const { envFilePath, hostEnv, parseEnvFile } = await import(pathToFileURL(integration('hooks', 'env.mjs')).href);
  const { resolveSettings } = await import(pathToFileURL(integration('core', 'config.mjs')).href);
  const { serviceTokenFile } = await import(pathToFileURL(integration('core', 'token.mjs')).href);
  const { readServiceEnv } = await import(pathToFileURL(integration('core', 'autostart.mjs')).href);
  const { env, notes } = hostEnv(process.env);
  const settings = resolveSettings({}, env, {});
  const lines = [];
  let ok = true;
  const line = (good, text) => {
    if (good === false) ok = false;
    lines.push(`${good === false ? 'FAIL' : good === null ? 'note' : ' ok '}  ${text}`);
  };
  const [major, minor] = process.versions.node.split('.').map(Number);
  line(major > 20 || (major === 20 && minor >= 3), `node ${process.versions.node} (needs 20.3 or later)`);
  line(true, `quotemem ${VERSION} at ${PACKAGE_DIR}`);
  const versions = adapterVersions();
  const odd = versions.filter(([, v]) => v !== VERSION);
  line(odd.length === 0, odd.length === 0 ? `all ${versions.length} adapter manifests are ${VERSION}` : `adapter versions differ: ${odd.map(([n, v]) => `${n} ${v}`).join(', ')}`);
  line(!settings.problem || !/URL/.test(settings.problem), `service URL ${settings.url ?? '(invalid)'}`);
  const pending = !settings.token && settings.localService?.autostart;
  const fromServiceEnv = settings.localService?.configured && Boolean(readServiceEnv(env)?.QUOTEMEM_TOKEN || readServiceEnv(env)?.QUOTEMEM_HTTP_AUTH_SECRET);
  const tokenFrom = process.env.QUOTEMEM_TOKEN || process.env.QUOTEMEM_HTTP_AUTH_SECRET
    ? 'the environment'
    : settings.tokenSource === 'service'
      ? (fromServiceEnv ? `the service's settings (${settings.localService.envFile})` : `the service's own token file (${serviceTokenFile(env, settings.localService)})`)
    : settings.tokenSource === 'file' || process.env.QUOTEMEM_TOKEN_FILE ? 'the file QUOTEMEM_TOKEN_FILE names'
    : '~/.quotemem/env';
  line(settings.token ? true : pending ? null : false, settings.token
    ? `token found in ${tokenFrom} (not shown)`
    : pending ? 'no token yet: the memory service creates it on its first start, and the plugins read it'
      : `no token: ${settings.problem && /TOKEN_FILE/.test(settings.problem) && !/another computer/.test(settings.problem) ? settings.problem
        : settings.localService ? 'run `quotemem setup` once (the service then starts by itself, and the plugins read its token), or start `quotemem serve` on this computer'
          : 'for a service on another computer, set QUOTEMEM_TOKEN_FILE to a private file that holds its token'}`);
  if (process.env.QUOTEMEM_TOKEN || process.env.QUOTEMEM_HTTP_AUTH_SECRET) {
    line(null, 'the token is exported in this shell: an agent started from it hands it to its own shell tool, where the model can print it; unset it, and leave the token in its file');
  }
  for (const [good, text] of await doctorLines(env, settings)) line(good, text);
  line(true, `memory owner (QUOTEMEM_USER_ID): ${settings.userId}${env.QUOTEMEM_USER_ID ? '' : ' (the OS user name, the default in every agent)'}`);
  for (const note of notes) line(null, note);
  for (const warning of settings.warnings) line(null, warning);
  const uvx = uvxPath(env);
  line(uvx ? true : null, uvx
    ? `uvx at ${uvx} (runs \`quotemem serve\` and the other service commands)`
    : 'uvx not on PATH: `quotemem serve` and the other service commands need uv (https://docs.astral.sh/uv/); a service started another way is not affected');
  if (settings.url) {
    const health = await probe(`${settings.url}/health`, { method: 'GET' });
    const down = settings.localService?.autostart
      ? `service not running at ${settings.url} (${health.error}); a plugin starts it when an agent needs it, or start it with \`${startHint(settings.url)}\``
      : `service unreachable at ${settings.url} (${health.error}); start it with \`${startHint(settings.url)}\`${settings.localService ? ', or run `quotemem setup` once so it starts by itself' : ''}`;
    line(health.status >= 200 && health.status < 300 ? true : settings.localService?.autostart ? null : false, health.status ? `service answers /health with ${health.status}` : down);
    if (health.status && settings.token) {
      // GET /agent/status needs the token and touches no memory.
      const { authHeaders } = await import(pathToFileURL(path.join(here, '..', 'core', 'http.mjs')).href);
      const auth = await probe(`${settings.url}/agent/status`, { method: 'GET', headers: authHeaders(settings) });
      if (auth.status === 401 && settings.localService) {
        // Two people signed in at once share 127.0.0.1:8765: the first one's service refuses the second one's token.
        const { portOwner, userPort } = await import(pathToFileURL(path.join(here, '..', 'core', 'proc.mjs')).href);
        const owner = portOwner(settings.localService.port);
        line(false, `the service refused the token: it must equal the service's own (its <data-dir>/token), or the service at ${settings.url} is another user's QuoteMem on this computer${owner && !owner.mine ? ` (it runs as ${owner.owner})` : ''} (two people signed in at once). Then give this user a port of its own: QUOTEMEM_URL=http://127.0.0.1:${userPort(env)} in ~/.quotemem/env and in the environment Claude Code, Codex and DSH start in (its plugins then start its own service there)`);
      } else if (auth.status === 401) line(false, 'the service refused the token (it must equal the service\'s token: the QUOTEMEM_TOKEN it was started with, or its <data-dir>/token)');
      else if (auth.status >= 200 && auth.status < 300) line(true, 'the service accepts the token');
      else line(null, `could not check the token (/agent/status answered ${auth.status || auth.error})`);
    }
  }
  const folders = stateFolders(env, envFilePath, parseEnvFile);
  for (const [good, text] of folders.lines) line(good, text);
  // Markers in ~/.quotemem and in every state folder seen, whatever this shell sets.
  const native = [...new Set(folders.dirs.flatMap((dir) => nativeMarkers({ ...env, QUOTEMEM_STATE_DIR: dir ?? '' })))].sort();
  if (native.length > 0) line(true, `agents seen with their own adapter: ${native.join(', ')}`);
  process.stdout.write(`${lines.join('\n')}\n`);
  return ok ? 0 : 1;
}

/**
 * Where the hooks and the MCP servers keep their state. The hooks read
 * QUOTEMEM_STATE_DIR from ~/.quotemem/env (QUOTEMEM_ENV_FILE); an MCP server
 * given another one in its agent's config keeps the forgets it is asked for
 * and finds the hooks' unsent turns less surely. Of the MCP configs, doctor
 * reads Cursor's (~/.cursor/mcp.json); for the others it can only say where
 * the setting belongs.
 * @returns {{lines: Array<[boolean|null, string]>, dirs: Array<string|null>}}
 *   `dirs`: the state folders seen (null: none set).
 */
function stateFolders(env, envFilePath, parseEnvFile) {
  const home = homeDir(env);
  const resolve = (dir) => (dir ? path.resolve(home, String(dir).replace(/^~(?=$|[\\/])/, home)) : null);
  // Windows paths are compared without case (`C:\Data\qm` is `c:/data/qm/`).
  const key = (dir) => {
    const full = resolve(dir);
    return full && process.platform === 'win32' ? full.toLowerCase() : full;
  };
  const read = (file) => {
    try { return fs.readFileSync(file, 'utf8'); } catch { return null; }
  };
  const envFile = envFilePath(process.env);
  const fromFile = parseEnvFile(read(envFile) ?? '').QUOTEMEM_STATE_DIR?.trim() || null;
  const fromShell = process.env.QUOTEMEM_STATE_DIR?.trim() || null;
  const hooks = key(fromFile);
  const servers = [];
  const lines = [];
  const cursorMcp = path.join(home, '.cursor', 'mcp.json');
  const cursorText = read(cursorMcp);
  let unchecked = false;
  if (cursorText !== null) {
    try {
      const value = JSON.parse(cursorText)?.mcpServers?.quotemem?.env?.QUOTEMEM_STATE_DIR;
      if (typeof value === 'string' && value.trim()) servers.push([cursorMcp, value.trim()]);
    } catch {
      unchecked = true;
      lines.push([null, `could not check the state folder of the QuoteMem MCP server in ${cursorMcp}: it is not plain JSON (comments or trailing commas)`]);
    }
  }
  const hooksAt = fromFile ? `in ${fromFile}` : 'in their own folders (~/.quotemem/<agent> and the plugins\' data folders)';
  let differs = false;
  for (const [file, value] of servers) {
    if (key(value) === hooks) continue;
    differs = true;
    lines.push([false, `the QuoteMem MCP server in ${file} keeps its state in ${value}, the hooks ${hooksAt}: set QUOTEMEM_STATE_DIR only in ${envFile}, so they share the unsent turns and the forgets`]);
  }
  if (fromShell && key(fromShell) !== hooks) {
    differs = true;
    lines.push([null, `QUOTEMEM_STATE_DIR is set in this shell (${fromShell}), not in ${envFile}: an agent or MCP server started elsewhere keeps its state ${hooksAt}. Set it only in ${envFile}`]);
  }
  if (!differs) {
    lines.push([fromFile && !unchecked ? true : null, fromFile
      ? `state folder ${fromFile} (QUOTEMEM_STATE_DIR in ${envFile}, read by the hooks and the MCP server; doctor cannot read every agent's MCP settings, so set it nowhere else)`
      : `state folders: each agent's own. To move them, set QUOTEMEM_STATE_DIR only in ${envFile}: doctor cannot read every agent's MCP settings, and an MCP server given its own folder would not share the hooks' unsent turns and forgets as surely`]);
  }
  return { lines, dirs: [...new Set([null, fromFile, fromShell, ...servers.map(([, value]) => value)])] };
}

/** The command that starts a service at `url`: `quotemem serve`, with --port when it is not the default. */
function startHint(url) {
  let port = '';
  try {
    const parsed = new URL(url);
    port = parsed.port || (parsed.protocol === 'https:' ? '443' : '80');
  } catch {
    return 'quotemem serve';
  }
  return port === SERVICE_PORT ? 'quotemem serve' : `quotemem serve --port ${port}`;
}

async function probe(url, init) {
  try {
    const res = await fetch(url, { ...init, signal: AbortSignal.timeout(3000) });
    await res.arrayBuffer().catch(() => null);
    return { status: res.status };
  } catch (error) {
    return { status: 0, error: error?.cause?.code ?? error?.name ?? 'error' };
  }
}

function adapterVersions() {
  const read = (file, pick) => {
    try {
      return pick(fs.readFileSync(integration(...file), 'utf8'));
    } catch {
      return 'missing';
    }
  };
  const fromJson = (text) => JSON.parse(text).version;
  return [
    ['claude-code', read(['claude-code', '.claude-plugin', 'plugin.json'], fromJson)],
    ['codex', read(['codex', '.codex-plugin', 'plugin.json'], fromJson)],
    ['gemini-cli', read(['gemini-cli', 'gemini-extension.json'], fromJson)],
    ['opencode', read(['opencode', 'package.json'], fromJson)],
    ['openclaw', read(['openclaw', 'package.json'], fromJson)],
    ['goose', read(['goose', 'quotemem', 'plugin.json'], fromJson)],
    ['mcp server', read(['dsh', 'server.mjs'], (text) => /const SERVER_VERSION = '([^']+)'/.exec(text)?.[1])],
  ];
}

async function main() {
  const [command, ...argv] = process.argv.slice(2);
  switch (command) {
    case 'install': return cmdInstall(argv);
    case 'mcp': return cmdMcp(argv).then(() => null);
    case 'hook': return cmdHook(argv);
    case 'doctor': return cmdDoctor();
    case 'setup': return cmdSetup(parseArgs(argv).flags, { release: VERSION });
    case 'status': return cmdStatus(process.env, { release: VERSION, probe });
    case 'stop': return cmdStop(process.env);
    case '--version': case '-v': case 'version':
      process.stdout.write(`${VERSION}\n`);
      return 0;
    case undefined: case '--help': case '-h': case 'help':
      process.stdout.write(USAGE);
      return command === undefined ? 2 : 0;
    default:
      if (SERVICE_COMMANDS.includes(command)) return cmdService(command, argv);
      process.stderr.write(`quotemem: unknown command ${JSON.stringify(command)}\n\n${USAGE}`);
      return 2;
  }
}

const code = await main();
// The MCP server and the hook programs keep running and exit on their own.
if (code !== null) process.exitCode = code;
