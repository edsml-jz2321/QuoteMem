#!/usr/bin/env node
// QuoteMem automatic memory for the OpenAI Codex CLI.
//
// One entry point for every Codex lifecycle hook and for the legacy `notify`
// program. Each call is a short-lived process; the shared adapter core
// (./core, a copy of integrations/core) keeps the state in files and a
// detached background flusher sends the batches.
//
//   node hook.mjs [event]     a hook: the JSON input arrives on stdin
//   node hook.mjs notify JSON the legacy notify program: the JSON is the last argument
//   node hook.mjs flush       start the background flusher if anything is queued
//
// Codex events and what they do:
//   SessionStart      queue what idle or crashed sessions left; start the flusher
//   UserPromptSubmit  record the message; silent forget on a forget request,
//                     else a gated recall; the result goes to the model as
//                     hookSpecificOutput.additionalContext (never shown in the TUI)
//   Stop              record the final assistant reply; every N user turns a
//                     batch is queued for /memory/add
//   Interrupt         close the turn without a reply (the user's words are kept)
//   SessionEnd        queue what is left
//
// The hook never blocks the user and never fails a turn: it always exits 0,
// prints nothing or one valid JSON object, never sets systemMessage, and logs
// to <state>/quotemem.log only (counts, codes and request ids; never text or
// the token).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HookMemory, fileSink, forgetWaitWithin, homeDir, makeLog, resolveSettings, settleLaunches, sha16 } from './core/index.mjs';

const HOST = 'codex';
const MAX_STDIN_BYTES = 16 * 1024 * 1024;
// Codex runs a hook under its `timeout`; these guards end the process first,
// so a stuck call never shows up as a failed hook.
const GUARD_MS = { UserPromptSubmit: 13_000, Stop: 2500, Interrupt: 1500, SessionEnd: 2500, SessionStart: 2500, notify: 30_000, flush: 2000 };
const EVENT_ALIASES = {
  prompt: 'UserPromptSubmit',
  'user-prompt-submit': 'UserPromptSubmit',
  stop: 'Stop',
  interrupt: 'Interrupt',
  'session-start': 'SessionStart',
  start: 'SessionStart',
  'session-end': 'SessionEnd',
  end: 'SessionEnd',
};
const MARKER_DIR = 'codex-hooks';
const MARKER_KEEP_MS = 30 * 86_400_000;

/** The Codex home: CODEX_HOME, else ~/.codex. */
export function codexHome(env = process.env) {
  return env.CODEX_HOME ? path.resolve(env.CODEX_HOME) : path.join(homeDir(env), '.codex');
}

/**
 * The state folder shared by the plugin hooks, a direct install and the
 * notify program: QUOTEMEM_STATE_DIR, else <CODEX_HOME>/quotemem.
 */
export function defaultStateDir(env = process.env) {
  return path.join(codexHome(env), 'quotemem');
}

function readStdin() {
  return new Promise((resolve) => {
    if (process.stdin.isTTY) {
      resolve('');
      return;
    }
    const chunks = [];
    let size = 0;
    process.stdin.on('data', (chunk) => {
      size += chunk.length;
      if (size <= MAX_STDIN_BYTES) chunks.push(chunk);
    });
    process.stdin.on('end', () => resolve(size <= MAX_STDIN_BYTES ? Buffer.concat(chunks).toString('utf8') : ''));
    process.stdin.on('error', () => resolve(''));
  });
}

function parse(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function str(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/** The marker that says Codex hooks run for a session, so the notify program stays out. */
function markerFile(settings, sessionId) {
  return path.join(settings.stateDir, MARKER_DIR, sha16(sessionId));
}

function markHooks(settings, sessionId) {
  try {
    const dir = path.join(settings.stateDir, MARKER_DIR);
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    fs.writeFileSync(markerFile(settings, sessionId), '', { mode: 0o600 });
  } catch {
    // without the marker a notify program, if also installed, may record the turn twice
  }
}

function hooksSeen(settings, sessionId) {
  try {
    return Date.now() - fs.statSync(markerFile(settings, sessionId)).mtimeMs < MARKER_KEEP_MS;
  } catch {
    return false;
  }
}

function pruneMarkers(settings) {
  const dir = path.join(settings.stateDir, MARKER_DIR);
  try {
    for (const name of fs.readdirSync(dir)) {
      const file = path.join(dir, name);
      if (Date.now() - fs.statSync(file).mtimeMs > MARKER_KEEP_MS) fs.unlinkSync(file);
    }
  } catch {
    // nothing to prune
  }
}

/** The text Codex adds to the model context for a UserPromptSubmit hook. */
export function promptOutput(context) {
  return JSON.stringify({ hookSpecificOutput: { hookEventName: 'UserPromptSubmit', additionalContext: context } });
}

/**
 * Handle one Codex hook or notify call.
 * @param {string} event - a Codex hook event name, `notify` or `flush`.
 * @param {object|null} input - the hook's stdin JSON, or the notify payload.
 * @param {{env?: object, settings?: object, log?: object}} options
 * @returns {Promise<string>} what to print on stdout ('' for nothing).
 */
export async function handle(event, input, { env = process.env } = {}) {
  const settings = resolveSettings({}, env, { defaultStateDir: defaultStateDir(env), configName: 'Codex hook config' });
  const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
  if (settings.problem) {
    if (event === 'SessionStart') log.warnOnce('problem', `automatic memory is off: ${settings.problem}`);
    return '';
  }
  if (event === 'SessionStart') for (const warning of settings.warnings) log.warnOnce(warning, warning);
  // A prompt waits for a queued forget only as long as its guard leaves room for the recall.
  const memory = new HookMemory({ settings, log, host: HOST, label: 'Codex plugin', env, forgetWaitMs: forgetWaitWithin(GUARD_MS.UserPromptSubmit, settings) });

  if (event === 'flush') {
    memory.flush();
    return '';
  }
  if (event === 'notify') return notify(settings, memory, input);
  if (!input) return '';
  const session = str(input.session_id);
  if (!session) return '';

  switch (event) {
    case 'SessionStart': {
      pruneMarkers(settings);
      await memory.sweep();
      await memory.sessionStarted({ session });
      return '';
    }
    case 'UserPromptSubmit': {
      // A subagent's prompt was written by the parent agent, not the user.
      if (str(input.agent_id) && !settings.includeSubagents) return '';
      markHooks(settings, session);
      const prompt = typeof input.prompt === 'string' ? input.prompt : '';
      const result = await memory.prompt({ session, turn: str(input.turn_id), text: prompt });
      return result.context ? promptOutput(result.context) : '';
    }
    case 'Stop': {
      const text = typeof input.last_assistant_message === 'string' ? input.last_assistant_message : null;
      await memory.reply({ session, turn: str(input.turn_id), text, completed: text !== null });
      return '';
    }
    case 'Interrupt': {
      await memory.reply({ session, turn: str(input.turn_id), completed: false });
      return '';
    }
    case 'SessionEnd': {
      await memory.end({ session });
      return '';
    }
    default:
      return '';
  }
}

// The legacy notify program: one call per finished turn, after the turn.
// It records the turn only when no Codex hook runs for the session (an older
// Codex, untrusted hooks); it cannot add context, so it never searches. A
// forget request in the turn is still carried out silently.
async function notify(settings, memory, payload) {
  if (!payload || payload.type !== 'agent-turn-complete') return '';
  const session = str(payload['thread-id']);
  if (!session || hooksSeen(settings, session)) return '';
  const inputs = Array.isArray(payload['input-messages']) ? payload['input-messages'].filter((m) => typeof m === 'string') : [];
  const reply = typeof payload['last-assistant-message'] === 'string' ? payload['last-assistant-message'] : null;
  const turn = str(payload['turn-id']);
  settings.recall = 'off';
  if (inputs.length > 0) await memory.prompt({ session, turn, text: inputs.join('\n\n') });
  await memory.reply({ session, turn, text: reply, completed: reply !== null });
  return '';
}

function eventOf(argv, input) {
  const arg = argv[0];
  if (arg === 'notify' || arg === 'flush') return arg;
  const named = input && str(input.hook_event_name);
  if (named) return named;
  return EVENT_ALIASES[arg] ?? arg ?? '';
}

// The answer first; then the background sender's start (on Windows it waits
// for PowerShell), within the event's guard.
function finish(out, untilMs) {
  const done = () => settleLaunches(untilMs - Date.now()).finally(() => process.exit(0));
  if (!out) done();
  else process.stdout.write(out, done);
}

async function main() {
  process.removeAllListeners('warning');
  process.on('warning', () => {});
  const argv = process.argv.slice(2);
  let input = null;
  if (argv[0] === 'notify') input = parse(argv.at(-1) ?? '');
  else if (argv[0] !== 'flush') input = parse(await readStdin());
  const event = eventOf(argv, input);
  const started = Date.now();
  const guard = setTimeout(() => process.exit(0), GUARD_MS[event] ?? 2500);
  guard.unref();
  let out = '';
  try {
    out = await handle(event, input);
  } catch {
    out = '';
  }
  finish(out, started + (GUARD_MS[event] ?? 2500) - 300);
}

const self = (() => {
  try {
    return fs.realpathSync(process.argv[1] ?? '') === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
})();
if (self) {
  main().catch(() => process.exit(0));
}
