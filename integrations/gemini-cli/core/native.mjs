// Native-adapter markers: keep one turn from being saved twice.
//
// Some hosts also run hooks written for another host: Cursor and Devin run
// the hooks in Claude Code's settings files next to their own. When the user
// installed both the Claude Code hooks and that host's hooks with the same
// QuoteMem settings, each turn would reach QuoteMem twice. So:
//
//   - every host's own adapter leaves a marker, ~/.quotemem/native/<host>.json
//     (under QUOTEMEM_STATE_DIR instead, when the environment sets it);
//   - an adapter that sees it runs inside another host (foreignHost) stays
//     silent when that host's own adapter is installed (nativeAdapterFor).
// Cursor's hooks are installed per user, so its marker is enough. Devin's
// are installed per project, so for Devin only its hook files count: a
// marker would silence the Claude Code hooks in projects without them.
//
// Nothing here throws. A missing marker only allows a duplicate capture.
import fs from 'node:fs';
import path from 'node:path';

import { homeDir } from './proc.mjs';

const MARKER_REFRESH_MS = 86_400_000;
const MARKER_MAX_AGE_MS = 30 * MARKER_REFRESH_MS;
const MAX_CONFIG_BYTES = 256 * 1024;
const MAX_LEVELS = 32;

function home(env) {
  return homeDir(env ?? {});
}

// Where markers are written: in QUOTEMEM_STATE_DIR when the environment sets
// it (as the kept forgets are), else in ~/.quotemem. A run given a state
// folder of its own writes nothing under the home.
function markerDirs(env) {
  const base = path.join(home(env), '.quotemem', 'native');
  const stateDir = String(env?.QUOTEMEM_STATE_DIR ?? '').trim();
  if (!stateDir) return [base];
  return [path.join(path.resolve(home(env), stateDir.replace(/^~(?=$|[\\/])/, home(env))), 'native'), base];
}

function markerName(label) {
  return `${String(label).replace(/[^a-z0-9-]/gi, '_')}.json`;
}

/** Record that the native adapter for `label` (e.g. `cursor`) is installed and ran. */
export function markNative(env, label) {
  try {
    const file = path.join(markerDirs(env)[0], markerName(label));
    try {
      if (Date.now() - fs.statSync(file).mtimeMs < MARKER_REFRESH_MS) return;
    } catch {
      // no marker yet
    }
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    fs.writeFileSync(file, JSON.stringify({ host: label, updated: new Date().toISOString() }), { mode: 0o600 });
  } catch {
    // a missing marker only allows a duplicate capture
  }
}

/**
 * True when the native adapter for `label` ran within `maxAgeMs` (default 30
 * days): a marker in QUOTEMEM_STATE_DIR, when set, or in ~/.quotemem.
 */
export function nativeInstalled(env, label, maxAgeMs = MARKER_MAX_AGE_MS) {
  return markerDirs(env).some((dir) => {
    try {
      return Date.now() - fs.statSync(path.join(dir, markerName(label))).mtimeMs < maxAgeMs;
    } catch {
      return false;
    }
  });
}

/** The labels of the native adapters with a marker (`quotemem doctor`). */
export function nativeMarkers(env) {
  const labels = new Set();
  for (const dir of markerDirs(env)) {
    try {
      for (const name of fs.readdirSync(dir)) if (name.endsWith('.json')) labels.add(name.slice(0, -5));
    } catch {
      // none here
    }
  }
  return [...labels].sort();
}

/**
 * The host that really runs a hook written in Claude Code's format, or null
 * for Claude Code itself. The payload decides: an environment variable alone
 * could be inherited by a Claude Code started from that host's terminal.
 * - Cursor puts `cursor_version` in every hook payload.
 * - Devin sends `prompt_id` but no `transcript_path` (Claude Code always
 *   sends `transcript_path`), and sets DEVIN_PROJECT_DIR for its hooks.
 *   Devin's SessionStart has no `prompt_id` yet, so it counts as Claude
 *   Code; that event only tidies up, so nothing is saved twice.
 * @param {object} payload - the hook input.
 * @param {object} [env] - the process environment.
 * @returns {string|null} `cursor`, `devin` or null.
 */
export function foreignHost(payload, env = {}) {
  if (!payload || typeof payload !== 'object') return null;
  if (typeof payload.cursor_version === 'string' && payload.cursor_version) return 'cursor';
  if (typeof payload.prompt_id === 'string' && payload.prompt_id && !('transcript_path' in payload)
    && typeof env?.DEVIN_PROJECT_DIR === 'string' && env.DEVIN_PROJECT_DIR) return 'devin';
  return null;
}

// The desktop app's prompt hook (beforeSubmitPrompt) adds
// `additional_context` to the turn from this version on. Cursor documents
// that field only for sessionStart, postToolUse and postToolUseFailure; it
// was seen to reach the model on 3.20.17, and on 3.31.15.
export const CURSOR_CONTEXT_VERSION = '3.20.17';
// The same for Cursor's CLI, whose versions are dates: in interactive mode
// 2026.11.19 runs the prompt hook, and its additional_context reached the
// model (recall, and a forget's note). Earlier dates are not trusted with
// it: nothing showed that they run the prompt hook or read the field, and
// without it the rule has the model search and forget through the tools.
// Print mode runs no prompt hook on any version.
export const CURSOR_CLI_CONTEXT_VERSION = '2026.11.19';

function versionParts(value) {
  const match = /^\s*v?(\d{1,9})\.(\d{1,9})(?:\.(\d{1,9}))?/.exec(typeof value === 'string' ? value : '');
  return match ? [Number(match[1]), Number(match[2]), Number(match[3] ?? 0)] : null;
}

/**
 * True for a version of Cursor's CLI (cursor-agent), which numbers its
 * versions by date: `2026.11.19-f005e9d`. The desktop app's are 3.20.17 and
 * the like. The CLI 2026.11.19 was seen to run only the sessionStart and
 * sessionEnd hooks in print mode, and every hook in interactive mode.
 */
export function cursorCliVersion(version) {
  const have = versionParts(version);
  return have !== null && have[0] >= 2000;
}

/**
 * True when a Cursor prompt hook may add context to the turn.
 * - `off`: never; `on`: always (a prompt hook that runs adds it).
 * - `auto`: the desktop app from CURSOR_CONTEXT_VERSION on, and the CLI from
 *   CURSOR_CLI_CONTEXT_VERSION on (its interactive mode; print mode runs no
 *   prompt hook). Never an unknown version.
 * @param {string} mode - settings.cursorPromptContext: `auto`, `on` or `off`.
 * @param {string} version - the payload's `cursor_version`.
 */
export function cursorPromptContext(mode, version) {
  if (mode === 'off') return false;
  if (mode === 'on') return true;
  const have = versionParts(version);
  if (!have) return false;
  const need = versionParts(cursorCliVersion(version) ? CURSOR_CLI_CONTEXT_VERSION : CURSOR_CONTEXT_VERSION);
  for (let i = 0; i < 3; i += 1) if (have[i] !== need[i]) return have[i] > need[i];
  return true;
}

function mentionsQuoteMem(file) {
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > MAX_CONFIG_BYTES) return false;
    return /quotemem-hook/.test(fs.readFileSync(file, 'utf8'));
  } catch {
    return false;
  }
}

// The hook files of a host that may run quotemem-hook.
function hookFiles(host, env, payload) {
  if (host === 'cursor') {
    const files = [path.join(home(env), '.cursor', 'hooks.json')];
    const roots = Array.isArray(payload?.workspace_roots) ? payload.workspace_roots : [];
    for (const root of roots.slice(0, 8)) {
      if (typeof root === 'string' && path.isAbsolute(root)) files.push(path.join(root, '.cursor', 'hooks.json'));
    }
    return files;
  }
  if (host === 'devin') {
    // Devin gives its hooks the project folder in DEVIN_PROJECT_DIR and
    // starts them there; its payload names no folder. Devin also reads
    // hook files in the folders above, up to the repository root.
    const files = [path.join(home(env), '.config', 'devin', 'config.json')];
    const starts = [env?.DEVIN_PROJECT_DIR, process.cwd()].filter((dir) => typeof dir === 'string' && path.isAbsolute(dir));
    for (const start of new Set(starts)) {
      let dir = path.resolve(start);
      for (let level = 0; level < MAX_LEVELS; level += 1) {
        files.push(path.join(dir, '.devin', 'hooks.v1.json'));
        if (fs.existsSync(path.join(dir, '.git'))) break;
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
    }
    return [...new Set(files)];
  }
  return [];
}

/**
 * True when `host`'s own QuoteMem adapter is installed here.
 * - Cursor: its marker is recent, or hooks.json in ~/.cursor or a workspace
 *   root runs quotemem-hook. The file check covers the very first turn,
 *   before the native hook has left its marker.
 * - Devin: .devin/hooks.v1.json in the project or a folder above it (up to
 *   the repository root), or ~/.config/devin/config.json, runs
 *   quotemem-hook. Its marker is not used: Devin's hooks are per project.
 * @param {string} host - from foreignHost().
 * @param {object} env - the process environment.
 * @param {object} [payload] - the hook input (Cursor lists its workspace roots).
 */
export function nativeAdapterFor(host, env, payload = null) {
  if (!host) return false;
  if (host !== 'devin' && nativeInstalled(env, host)) return true;
  try {
    return hookFiles(host, env, payload).some(mentionsQuoteMem);
  } catch {
    return false;
  }
}
