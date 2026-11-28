#!/usr/bin/env node
// Direct install of QuoteMem into the Codex CLI, as an alternative to the
// plugin (`codex plugin marketplace add` + `codex plugin add`).
//
//   node integrations/codex/install.mjs                  # dry run: print what would change
//   node integrations/codex/install.mjs --write          # install
//   node integrations/codex/install.mjs --write --notify # also set the legacy notify program
//   node integrations/codex/install.mjs --uninstall --write
//   options: --codex-home DIR (default $CODEX_HOME or ~/.codex), --node PATH (default this node)
//
// It changes two files in the Codex home and nothing else:
//   hooks.json   the five QuoteMem hooks are added (other hooks are kept)
//   config.toml  a marked block with [mcp_servers.quotemem]; with --notify,
//                a marked `notify = [...]` line at the top, only when the
//                file sets no notify program of its own
// Running it again replaces its own entries. The token is never written,
// and never needed in Codex's environment (where Codex's shell tool, and so
// the model, could read it): the hooks and the MCP server read the service's
// own <data-dir>/token, or the file QUOTEMEM_TOKEN_FILE names. Codex runs new
// hooks only after you trust them with /hooks.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { homeDir } from './core/proc.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const HOOK_PATH = path.join(here, 'hook.mjs');
export const SERVER_PATH = path.join(here, 'server.mjs');
export const EVENTS = { SessionStart: 3, UserPromptSubmit: 15, Stop: 3, Interrupt: 2, SessionEnd: 3 };
// What Codex forwards from its environment to the MCP server (it starts stdio
// servers with a minimal one): the same settings the hooks read, including
// where the token file is. QUOTEMEM_TOKEN is still forwarded when set, but no
// setup needs it.
export const MCP_ENV_VARS = [
  'QUOTEMEM_URL', 'QUOTEMEM_TOKEN_FILE', 'QUOTEMEM_DATA_DIR', 'XDG_DATA_HOME', 'QUOTEMEM_HTTP_AUTH_MODE',
  'QUOTEMEM_USER_ID', 'QUOTEMEM_TIMEOUT_MS', 'QUOTEMEM_CAPTURE_EVERY', 'QUOTEMEM_TOKEN', 'QUOTEMEM_HTTP_AUTH_SECRET',
];
const BEGIN = '# >>> quotemem (managed by integrations/codex/install.mjs; keep your own settings outside these lines)';
const END = '# <<< quotemem';
const NOTIFY_BEGIN = '# >>> quotemem notify (managed by integrations/codex/install.mjs)';
const NOTIFY_END = '# <<< quotemem notify';

/** Whether a hook command is one of ours (any install location). */
export function isOurs(command) {
  return typeof command === 'string' && /integrations[\\/]+codex[\\/]+hook\.mjs/.test(command);
}

// A double-quoted shell word: `"`, `\`, `$` and backticks are escaped, so
// /bin/sh never expands a path. On Windows a path is written as given, in
// quotes: a backslash there is a path separator, not an escape, and a file
// name cannot hold a quote.
const quote = (text, platform) => (platform === 'win32'
  ? `"${String(text)}"`
  : `"${String(text).replace(/(["\\$`])/g, '\\$1')}"`);
const tomlString = (text) => JSON.stringify(text);

/** The hook command line for one event. */
export function hookCommand(node, event, platform = process.platform) {
  return `${quote(node, platform)} ${quote(HOOK_PATH, platform)} ${event}`;
}

/** hooks.json with our handlers replaced (or removed with `remove`). */
export function mergeHooks(existing, node, { remove = false } = {}) {
  const file = existing && typeof existing === 'object' ? structuredClone(existing) : {};
  const hooks = file.hooks && typeof file.hooks === 'object' ? file.hooks : {};
  for (const [event, groups] of Object.entries(hooks)) {
    if (!Array.isArray(groups)) continue;
    hooks[event] = groups
      .map((group) => (group && Array.isArray(group.hooks) ? { ...group, hooks: group.hooks.filter((h) => !isOurs(h?.command)) } : group))
      .filter((group) => !(group && Array.isArray(group.hooks) && group.hooks.length === 0));
    if (hooks[event].length === 0) delete hooks[event];
  }
  if (!remove) {
    for (const [event, timeout] of Object.entries(EVENTS)) {
      hooks[event] = [...(hooks[event] ?? []), { hooks: [{ type: 'command', command: hookCommand(node, event), timeout }] }];
    }
  }
  file.hooks = hooks;
  return file;
}

function stripBlock(text, begin, end) {
  const lines = text.split('\n');
  const out = [];
  let inside = false;
  for (const line of lines) {
    if (line.trim() === begin) { inside = true; continue; }
    if (inside && line.trim() === end) { inside = false; continue; }
    if (!inside) out.push(line);
  }
  return out.join('\n');
}

/** The [mcp_servers.quotemem] block. */
export function mcpBlock(node) {
  return [
    BEGIN,
    '[mcp_servers.quotemem]',
    `command = ${tomlString(node)}`,
    `args = [${tomlString(SERVER_PATH)}]`,
    'env = { QUOTEMEM_AUTO = "1" }',
    `env_vars = [${MCP_ENV_VARS.map(tomlString).join(', ')}]`,
    'tool_timeout_sec = 600',
    END,
  ].join('\n');
}

export function notifyBlock(node) {
  return [NOTIFY_BEGIN, `notify = [${[node, HOOK_PATH, 'notify'].map(tomlString).join(', ')}]`, NOTIFY_END].join('\n');
}

/** Whether config.toml (without our blocks) sets a top-level notify. */
function hasNotify(toml) {
  for (const line of toml.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('[')) return false; // top-level keys come before the first table
    if (/^notify\s*=/.test(trimmed)) return true;
  }
  return false;
}

/**
 * The new config.toml text, or a problem.
 * @returns {{text?: string, problem?: string, notes: string[]}}
 */
export function mergeConfig(existing, node, { remove = false, notify = false } = {}) {
  const notes = [];
  let text = stripBlock(stripBlock(existing ?? '', BEGIN, END), NOTIFY_BEGIN, NOTIFY_END);
  if (!remove) {
    if (/^\s*\[\s*mcp_servers\s*\.\s*("?)quotemem\1\s*\]/m.test(text)) {
      return { problem: 'config.toml already has its own [mcp_servers.quotemem]; remove it or install the plugin instead', notes };
    }
    if (/^\s*\[\[?\s*hooks\s*\.\s*(?!state\b)[A-Za-z]/m.test(text)) {
      return { problem: 'config.toml defines hooks inline; Codex warns when hooks.json and inline hooks share a layer. Add the QuoteMem hooks to config.toml yourself, or install the plugin instead', notes };
    }
    text = `${text.replace(/\n*$/, '')}${text.trim() ? '\n\n' : ''}${mcpBlock(node)}\n`;
    if (notify) {
      if (hasNotify(text)) notes.push('config.toml already sets notify; the QuoteMem notify program was not added (the hooks do not need it)');
      else text = `${notifyBlock(node)}\n\n${text}`;
    }
  } else {
    text = text.replace(/\n{3,}/g, '\n\n');
  }
  return { text, notes };
}

function readText(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

// The file a path names, through any symlinks (a dotfile kept in a dotfiles
// repo stays a link to it).
function realTarget(file) {
  let target = file;
  for (let hops = 0; hops < 32; hops += 1) {
    let link;
    try {
      link = fs.readlinkSync(target);
    } catch {
      return target; // not a link, or missing
    }
    target = path.resolve(path.dirname(target), link);
  }
  return target;
}

/**
 * Replace a config file safely: through symlinks to the real file, keeping
 * its permission bits (a 0600 file holding other servers' keys stays 0600),
 * and after a copy of the old text to `<file>.quotemem.bak` with the same
 * bits. A new file gets `mode` (default 0644).
 * @returns {string|null} the backup path, or null when there was no old file.
 */
export function writeConfigFile(file, text, { mode = null } = {}) {
  const target = realTarget(file);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  let old = null;
  let bits = mode ?? 0o644;
  try {
    const stat = fs.statSync(target);
    bits = mode ?? (stat.mode & 0o7777);
    old = fs.readFileSync(target);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  let backup = null;
  if (old !== null) {
    backup = `${target}.quotemem.bak`;
    fs.writeFileSync(backup, old, { mode: bits & 0o666 });
    fs.chmodSync(backup, bits & 0o666);
  }
  const tmp = `${target}.quotemem-${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmp, text, { mode: bits });
    fs.chmodSync(tmp, bits);
    fs.renameSync(tmp, target);
  } catch (error) {
    try { fs.unlinkSync(tmp); } catch { /* none */ }
    throw error;
  }
  return backup;
}

function writeText(file, text) {
  return writeConfigFile(file, text);
}

/**
 * Plan (and with `write`, apply) the install.
 * @returns {{ok: boolean, lines: string[]}}
 */
export function install({ codexHome, node = process.execPath, write = false, remove = false, notify = false }) {
  const lines = [];
  const hooksFile = path.join(codexHome, 'hooks.json');
  const configFile = path.join(codexHome, 'config.toml');
  const hooksText = readText(hooksFile);
  let hooksJson = null;
  if (hooksText !== null && hooksText.trim()) {
    try {
      hooksJson = JSON.parse(hooksText);
    } catch {
      return { ok: false, lines: [`${hooksFile} is not valid JSON; fix it first (nothing was changed)`] };
    }
  }
  const configText = readText(configFile);
  const config = mergeConfig(configText, node, { remove, notify });
  if (config.problem) return { ok: false, lines: [`${config.problem} (nothing was changed)`] };
  const hooks = mergeHooks(hooksJson, node, { remove });
  const hooksOut = `${JSON.stringify(hooks, null, 2)}\n`;
  const emptyHooks = Object.keys(hooks.hooks).length === 0 && Object.keys(hooks).length === 1;

  lines.push(`${write ? '' : '[dry run] '}${remove ? 'uninstall from' : 'install into'} ${codexHome}`);
  if (remove && emptyHooks && hooksText !== null) lines.push(`${hooksFile}: removed (it held only QuoteMem hooks)`);
  else lines.push(`${hooksFile}: ${remove ? 'QuoteMem hooks removed' : `QuoteMem hooks for ${Object.keys(EVENTS).join(', ')}`}`);
  lines.push(`${configFile}: ${remove ? 'QuoteMem blocks removed' : `[mcp_servers.quotemem] block${notify && config.text.includes(NOTIFY_BEGIN) ? ' and notify program' : ''}`}`);
  lines.push(...config.notes);
  if (!write) {
    lines.push('', '--- hooks.json', hooksOut.trimEnd(), '--- config.toml', config.text.trimEnd(), '', 'Run again with --write to apply.');
    return { ok: true, lines };
  }
  const backups = [];
  if (remove && emptyHooks) {
    if (hooksText !== null) fs.rmSync(hooksFile, { force: true });
  } else {
    backups.push(writeText(hooksFile, hooksOut));
  }
  if (!(remove && configText === null)) backups.push(writeText(configFile, config.text));
  for (const backup of backups.filter(Boolean)) lines.push(`backup of the old file: ${backup}`);
  if (!remove) {
    lines.push(
      '',
      'Next:',
      '  1. For a service on this computer nothing is needed: its token file is read. For one elsewhere, set QUOTEMEM_URL and',
      '     QUOTEMEM_TOKEN_FILE (a file holding its token) in the shell that starts codex. Do not export the token itself.',
      '  2. Start codex and run /hooks: review and trust the five QuoteMem hooks. Codex skips untrusted hooks.',
    );
  }
  return { ok: true, lines };
}

function argValue(argv, name) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : null;
}

if (process.argv[1] && fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url))) {
  const argv = process.argv.slice(2);
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log('usage: node integrations/codex/install.mjs [--write] [--uninstall] [--notify] [--codex-home DIR] [--node PATH]');
    process.exit(0);
  }
  const codexHome = path.resolve(argValue(argv, '--codex-home') ?? process.env.CODEX_HOME ?? path.join(homeDir(), '.codex'));
  const result = install({
    codexHome,
    node: argValue(argv, '--node') ?? process.execPath,
    write: argv.includes('--write'),
    remove: argv.includes('--uninstall'),
    notify: argv.includes('--notify'),
  });
  for (const line of result.lines) (result.ok ? console.log : console.error)(line);
  process.exit(result.ok ? 0 : 2);
}
