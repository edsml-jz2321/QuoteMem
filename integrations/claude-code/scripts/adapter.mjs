// Claude Code hook events → the QuoteMem adapter core (core/hooks.mjs).
//
// Each hook runs in its own short-lived process. Claude Code passes the
// event's JSON on stdin; this module maps it onto HookMemory and returns the
// text the hook prints on stdout ('' prints nothing):
//
//   SessionStart      queue what idle sessions left and start the flusher.
//   UserPromptSubmit  record the message; on a forget request run the silent
//                     forget, else a gated recall. A note or a recall block
//                     goes back as hookSpecificOutput.additionalContext,
//                     which reaches the model and not the visible transcript.
//   Stop              record the final reply (`last_assistant_message`);
//                     every N user turns a batch is queued.
//   StopFailure       the turn failed (API error): keep the user's words.
//   SessionEnd        queue what is left; the detached flusher sends it.
//
// Inside Cursor or Devin (which also run Claude Code's hooks), the hooks stay
// silent when that host's own QuoteMem hooks are installed, so a turn is
// saved once. Inside a Cursor whose prompt hook reads `additional_context`,
// the prompt answer uses that field instead of Claude Code's.
//
// Nothing here throws, blocks a prompt or prints anything but the hook
// output: every failure is logged to <state>/quotemem.log and the turn goes
// on without memory.
import path from 'node:path';

import { HookMemory, cursorPromptContext, errKind, fileSink, fitContext, foreignHost, homeDir, isCommand, lastAssistantReply, makeLog, nativeAdapterFor, resolveSettings } from '../core/index.mjs';
import { HOOK_PREFIX, applyOptions } from './env.mjs';

export const HOST = 'claude-code';
export const EVENTS = Object.freeze(['SessionStart', 'UserPromptSubmit', 'Stop', 'StopFailure', 'SessionEnd']);

// Claude Code caps additionalContext at 10,000 characters (longer text is
// saved to a file and only a preview is passed), so recall stays under it.
export const MAX_CONTEXT_CHARS = 9000;
// The prompt hook must answer well inside the event's 30 s default timeout
// (hooks.json sets 30): recall and forget are bounded below that, and so is
// a prompt's wait for a queued forget (at most 8 s, before its recall).
const MAX_FORGET_WAIT_MS = 20_000;
const MAX_RECALL_WAIT_MS = 10_000;

// Prompts Claude Code submits on its own (background task reports, messages
// from teammates or other sessions, scheduled runs) arrive wrapped in one
// element. They are not the user's words: neither captured nor searched.
const HOST_TAGS = new Set([
  'task-notification', 'teammate-message', 'peer-message', 'agent-message', 'session-message',
  'scheduled-task', 'loop-iteration', 'cron-task', 'system-reminder', 'user-prompt-submit-hook',
  'local-command-stdout', 'local-command-stderr', 'local-command-caveat',
  'command-name', 'command-message', 'command-args', 'command-stdout',
  'bash-input', 'bash-stdout', 'bash-stderr',
]);

/** Remove the marker lines Claude Code puts around pasted text; the pasted text stays. */
export function cleanPrompt(text) {
  if (typeof text !== 'string') return '';
  return text.replace(/^[ \t]*<\/?pasted_content\b[^>\n]*>[ \t]*(\r?\n|$)/gm, '').trim();
}

/** True for a prompt the host submitted itself (a wrapped notification or report). */
export function isHostInjected(text) {
  const t = String(text ?? '').trim();
  const open = /^<([A-Za-z][\w-]*)(\s[^>]*)?>/.exec(t);
  if (!open) return false;
  const tag = open[1].toLowerCase();
  if (HOST_TAGS.has(tag)) return true;
  // Any other single hyphenated element wrapping the whole prompt.
  return /[-_]/.test(tag) && new RegExp(`</${open[1]}>\\s*$`, 'i').test(t);
}

// The transcript reader is shared (core/transcript.mjs); it is used only
// when the Stop input has no `last_assistant_message`, and only takes a
// reply to the open turn's own user message.
export { lastAssistantText } from '../core/index.mjs';

/**
 * The hook output that adds text to the model context for this prompt.
 * Over MAX_CONTEXT_CHARS, the oldest notes are dropped; a block is never cut.
 * @param {string|null} context
 * @param {{cursor?: boolean}} [options] - `cursor`: the hook runs inside a
 *   Cursor that reads `additional_context` from the prompt hook. Only
 *   Cursor's own field is sent then, since Cursor may read Claude Code's
 *   field too and would add the text twice.
 */
export function promptOutput(context, { cursor = false } = {}) {
  const text = fitContext(String(context ?? ''), MAX_CONTEXT_CHARS);
  if (!text) return '';
  if (cursor) return JSON.stringify({ continue: true, additional_context: text });
  return JSON.stringify({ hookSpecificOutput: { hookEventName: 'UserPromptSubmit', additionalContext: text } });
}

/** Settings, log and memory for this process, or null when automatic memory is off. */
export function open(env = process.env) {
  const hostEnv = applyOptions(env, HOOK_PREFIX);
  const data = typeof env.CLAUDE_PLUGIN_DATA === 'string' && env.CLAUDE_PLUGIN_DATA ? env.CLAUDE_PLUGIN_DATA : null;
  const defaultStateDir = data ? path.join(data, 'state') : path.join(homeDir(), '.quotemem', HOST);
  const settings = resolveSettings({}, hostEnv, { defaultStateDir, configName: 'Claude Code plugin options' });
  const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
  if (settings.problem) return { settings, log, memory: null };
  settings.maxInjectChars = Math.min(settings.maxInjectChars, MAX_CONTEXT_CHARS);
  settings.forgetTimeoutMs = Math.min(settings.forgetTimeoutMs, MAX_FORGET_WAIT_MS);
  settings.recallTimeoutMs = Math.min(settings.recallTimeoutMs, MAX_RECALL_WAIT_MS);
  // idleWait: a chat's last turns are saved once it has been idle for
  // QUOTEMEM_IDLE_FLUSH_MS, by a background flusher that waits (armIdle).
  const memory = new HookMemory({ settings, log, host: HOST, label: 'Claude Code plugin', env: hostEnv, idleWait: true });
  return { settings, log, memory };
}

// When this hook left turns that wait for the idle time, the start of the
// background flusher that saves them then (HookMemory.armIdle).
let idleArm = null;

/** After the hook's answer is written: start the flusher that waits for idle chats, if any wait. */
export function armIdle() {
  try {
    return idleArm?.() ?? false;
  } catch {
    return false; // the next hook tries again
  }
}

/**
 * Handle one hook event.
 * @param {string} event - the hook event name (the input's hook_event_name wins).
 * @param {object} input - the hook input from stdin.
 * @param {object} [env] - the process environment.
 * @returns {Promise<string>} what to print on stdout ('' for nothing).
 */
export async function runHook(event, input, env = process.env) {
  let ctx = null;
  try {
    const payload = input && typeof input === 'object' ? input : {};
    const name = typeof payload.hook_event_name === 'string' && payload.hook_event_name ? payload.hook_event_name : event;
    if (!EVENTS.includes(name)) return '';
    // Cursor and Devin also run the hooks in Claude Code's settings files.
    // When that host's own QuoteMem hooks are installed, they handle the turn.
    const host = foreignHost(payload, env);
    if (host && nativeAdapterFor(host, env, payload)) return '';
    ctx = open(env);
    const { settings, log, memory } = ctx;
    if (memory) idleArm = () => memory.armIdle();
    if (!memory) {
      if (name === 'SessionStart') log.warn(`automatic memory is off: ${settings.problem}`);
      return '';
    }
    for (const warning of name === 'SessionStart' ? settings.warnings : []) log.warn(warning);
    const session = typeof payload.session_id === 'string' && payload.session_id ? payload.session_id : null;
    // Hooks that fire inside a subagent carry its agent_id.
    const subagent = typeof payload.agent_id === 'string' && payload.agent_id.length > 0;
    if (subagent && !settings.includeSubagents) return '';
    const turn = typeof payload.prompt_id === 'string' && payload.prompt_id ? payload.prompt_id : null;

    switch (name) {
      case 'SessionStart':
        await memory.sweep();
        await memory.sessionStarted({ session });
        return '';
      case 'UserPromptSubmit': {
        if (!session) return '';
        const text = cleanPrompt(payload.prompt);
        if (!text || isCommand(text) || isHostInjected(text)) return '';
        const out = await memory.prompt({ session, turn, text, time: Date.now() });
        const cursor = host === 'cursor' && cursorPromptContext(settings.cursorPromptContext, payload.cursor_version);
        return out?.context ? promptOutput(out.context, { cursor }) : '';
      }
      case 'Stop': {
        if (!session) return '';
        let text = typeof payload.last_assistant_message === 'string' ? payload.last_assistant_message : null;
        let time = Date.now();
        if (text === null) {
          // Only a reply to this turn's own user message: the transcript may
          // not hold this turn yet.
          const open = memory.openTurn(session);
          const same = open?.head && (turn === null || turn === open.turn);
          const kept = same ? lastAssistantReply(payload.transcript_path, { head: open.head, after: open.time }) : null;
          text = kept?.text ?? null;
          time = kept?.time ?? time;
        }
        await memory.reply({ session, turn, text, time, completed: text !== null });
        return '';
      }
      case 'StopFailure':
        if (session) await memory.reply({ session, turn, completed: false });
        return '';
      case 'SessionEnd':
        if (session) await memory.end({ session });
        return '';
      default:
        return '';
    }
  } catch (error) {
    ctx?.log?.warnOnce?.(`hook:${errKind(error)}`, `hook skipped (${errKind(error)})`);
    return '';
  }
}
