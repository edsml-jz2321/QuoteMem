// What each host's hook events mean, and how each host wants an answer.
//
// normalize(host, input, eventArg) maps one hook payload to an action:
//   {action: 'prompt',  session, turn, text}           the user sent a message
//                       (Cursor: also `transcript`, for an earlier turn
//                       whose reply had not come at its stop)
//   {action: 'reply',   session, turn, text, completed} the turn ended
//                       (Devin and Cursor: also `transcript`, a transcript
//                       path to read the reply from when `text` is null)
//   {action: 'buffer',  session, turn, text}           an assistant message to keep until the turn ends
//   (each of these three also has `time`, the host's own time of the event in
//   milliseconds, when its payload gives one: Gemini CLI and Cline do)
//   {action: 'start',   session}                       a session started
//   {action: 'end',     session}                       the session ended
//                       (Cursor: also `transcript`, the conversation's
//                       transcript path, to save the turns no hook saw)
//   {action: 'ignore',  reason}
// Cursor actions also carry `version`, the payload's cursor_version.
// render(host, action, context) is the exact stdout the host expects.
//
// The payload shapes come from each host's documentation and source:
//   gemini  Gemini CLI hooks (BeforeAgent, AfterAgent, SessionEnd)
//   cursor  Cursor hooks (sessionStart, beforeSubmitPrompt, afterAgentResponse, stop, sessionEnd);
//           beforeSubmitPrompt's additional_context is not in Cursor's docs
//           (see CURSOR_CONTEXT_VERSION in core/native.mjs). Cursor's CLI
//           (date versions, 2026.11.19) was seen to run only sessionStart
//           and sessionEnd, whose transcript_path names the chat's transcript
//   cline   Cline VS Code hooks (UserPromptSubmit, TaskComplete, TaskCancel)
//   goose   goose plugin hooks (UserPromptSubmit, Stop, SessionEnd)
//   devin   Devin CLI / Devin Desktop hooks, Claude Code format
//           (UserPromptSubmit, Stop, SessionEnd)

const MAX_SESSION_CHARS = 512;
const MAX_TURN_CHARS = 256;
const MAX_PATH_CHARS = 4096;

/**
 * Host name on the command line -> the prefix of session ids sent to QuoteMem.
 * `reliableEnd: false`: a chat can close without a session end (a Cursor tab
 * or console closed; Cline has no session end), so a new chat sends the
 * other chats' turns at once when they are between turns.
 */
export const HOSTS = {
  gemini: { label: 'gemini-cli', name: 'Gemini CLI' },
  cursor: { label: 'cursor', name: 'Cursor', reliableEnd: false },
  cline: { label: 'cline', name: 'Cline', reliableEnd: false },
  goose: { label: 'goose', name: 'goose' },
  devin: { label: 'devin', name: 'Devin' },
};

/** Aliases accepted on the command line. */
const ALIASES = { 'gemini-cli': 'gemini', windsurf: 'devin' };

export function hostOf(name) {
  const key = String(name ?? '').trim().toLowerCase();
  const host = ALIASES[key] ?? key;
  return Object.hasOwn(HOSTS, host) ? host : null;
}

function str(value, max) {
  if (typeof value !== 'string' && typeof value !== 'number') return null;
  const text = String(value).trim();
  return text && text.length <= max ? text : null;
}

const id = (value) => str(value, MAX_SESSION_CHARS);
const turnId = (value) => str(value, MAX_TURN_CHARS);
const text = (value) => (typeof value === 'string' ? value : null);
// A reply text: null when missing or blank, so a fallback source can be tried.
const replyText = (value) => (typeof value === 'string' && value.trim() ? value : null);

function eventOf(input, eventArg, ...fields) {
  if (eventArg) return String(eventArg);
  for (const field of fields) if (typeof input?.[field] === 'string' && input[field]) return input[field];
  return '';
}

const ignore = (reason) => ({ action: 'ignore', reason });

function gemini(input, eventArg) {
  const event = eventOf(input, eventArg, 'hook_event_name');
  const session = id(input?.session_id);
  if (!session) return ignore('no session');
  switch (event) {
    case 'BeforeAgent':
      return { action: 'prompt', event, session, turn: null, text: text(input.prompt) ?? '' };
    case 'AfterAgent':
      // stop_hook_active: another hook made the agent continue; the turn
      // was already closed by the first AfterAgent.
      if (input.stop_hook_active === true) return ignore('continuation');
      return { action: 'reply', event, session, turn: null, text: text(input.prompt_response), completed: true };
    case 'SessionEnd':
      return { action: 'end', event, session };
    default:
      return ignore(`event ${event || 'unknown'}`);
  }
}

function cursor(input, eventArg) {
  const event = eventOf(input, eventArg, 'hook_event_name');
  const session = id(input?.conversation_id) ?? id(input?.session_id);
  if (!session) return ignore('no session');
  const turn = turnId(input.generation_id);
  // A background agent runs without the user; it is kept only with
  // QUOTEMEM_INCLUDE_SUBAGENTS (the runner decides).
  const subagent = input.is_background_agent === true;
  // The runner decides from the version whether the prompt hook may add context.
  const version = str(input.cursor_version, 64);
  switch (event) {
    case 'sessionStart': {
      // Its transcript, when Cursor names one: what it holds was said before.
      const transcript = str(input.transcript_path, MAX_PATH_CHARS);
      const own = transcript && transcript.split(/[\\/]/).at(-1) === `${session}.jsonl`;
      return { action: 'start', event, session, subagent, version, ...(own ? { transcript } : {}) };
    }
    case 'beforeSubmitPrompt': {
      // Its transcript, for an earlier turn whose stop came before its reply.
      const action = { action: 'prompt', event, session, turn, text: text(input.prompt) ?? '', subagent, version };
      const transcript = str(input.transcript_path, MAX_PATH_CHARS);
      if (transcript && transcript.split(/[\\/]/).at(-1) === `${session}.jsonl`) action.transcript = transcript;
      return action;
    }
    case 'afterAgentResponse':
      return { action: 'buffer', event, session, turn, text: text(input.text), subagent, version };
    case 'stop': {
      // Its transcript, for the turn's reply when afterAgentResponse has
      // not kept it (the two hooks run at the same time).
      const action = { action: 'reply', event, session, turn, text: null, completed: input.status === 'completed', subagent, version };
      const transcript = str(input.transcript_path, MAX_PATH_CHARS);
      if (transcript && transcript.split(/[\\/]/).at(-1) === `${session}.jsonl`) action.transcript = transcript;
      return action;
    }
    case 'sessionEnd': {
      // Cursor's CLI in print mode runs no prompt or stop hook: the turns are read from
      // the chat's transcript. Only the chat's own file is read: Cursor names
      // it <conversation id>.jsonl.
      // completed: the run ended as it should (final_status, else reason),
      // so its last reply is written.
      const completed = (input.final_status ?? input.reason) === 'completed';
      const action = { action: 'end', event, session, subagent, version, completed };
      const transcript = str(input.transcript_path, MAX_PATH_CHARS);
      const own = transcript && transcript.split(/[\\/]/).at(-1) === `${session}.jsonl`;
      if (own) action.transcript = transcript;
      return action;
    }
    default:
      return ignore(`event ${event || 'unknown'}`);
  }
}

function cline(input, eventArg) {
  const event = eventOf(input, eventArg, 'hookName');
  const session = id(input?.taskId);
  if (!session) return ignore('no session');
  switch (event) {
    case 'UserPromptSubmit':
      return { action: 'prompt', event, session, turn: null, text: text(input.userPromptSubmit?.prompt) ?? '' };
    case 'TaskComplete':
      return { action: 'reply', event, session, turn: null, text: text(input.taskComplete?.taskMetadata?.result), completed: true };
    case 'TaskCancel':
      return { action: 'reply', event, session, turn: null, text: null, completed: false };
    default:
      return ignore(`event ${event || 'unknown'}`);
  }
}

function goose(input, eventArg) {
  const event = eventOf(input, eventArg, 'event', 'hook_event_name');
  const session = id(input?.session_id);
  if (!session) return ignore('no session');
  switch (event) {
    case 'UserPromptSubmit':
      return { action: 'prompt', event, session, turn: null, text: text(input.message) ?? text(input.prompt) ?? '' };
    case 'Stop':
      return { action: 'reply', event, session, turn: null, text: text(input.last_assistant_message), completed: true };
    case 'SessionEnd':
      return { action: 'end', event, session };
    default:
      return ignore(`event ${event || 'unknown'}`);
  }
}

function devin(input, eventArg) {
  const event = eventOf(input, eventArg, 'hook_event_name');
  const session = id(input?.session_id);
  if (!session) return ignore('no session');
  const turn = turnId(input.prompt_id);
  switch (event) {
    case 'UserPromptSubmit':
      return { action: 'prompt', event, session, turn, text: text(input.prompt) ?? '' };
    case 'Stop': {
      // stop_hook_active: another hook made the agent continue; the turn
      // was already closed by the first Stop. A known limit: the reply kept
      // is the one at that first Stop, not the reply after the continuation.
      if (input.stop_hook_active === true) return ignore('continuation');
      // Devin CLI 3000.5.20 and later send the final reply as
      // last_assistant_message. Without it, a transcript path (Claude Code's
      // field, undocumented for Devin) is read by the runner, which takes
      // only a reply to this turn's own user message; without either, the
      // turn keeps the user's words only.
      const action = { action: 'reply', event, session, turn, text: replyText(input.last_assistant_message), completed: true };
      const transcript = typeof input.transcript_path === 'string' ? str(input.transcript_path, MAX_PATH_CHARS) : null;
      if (action.text === null && transcript) action.transcript = transcript;
      return action;
    }
    case 'SessionEnd':
      return { action: 'end', event, session };
    default:
      return ignore(`event ${event || 'unknown'}`);
  }
}

const NORMALIZE = { gemini, cursor, cline, goose, devin };

/** Map one hook payload to an action (see the top of this file). Never throws. */
/**
 * A host's time of an event, in milliseconds: a number or a string of digits
 * (milliseconds, or seconds when it is too small to be milliseconds) or an
 * ISO date. Null when it is missing or not a plausible time (before 2000, or
 * more than a day ahead): the message then gets the time the hook ran.
 */
export function hostTime(value, now = Date.now()) {
  let ms = null;
  if (typeof value === 'number' || (typeof value === 'string' && /^\s*\d+(?:\.\d+)?\s*$/.test(value))) {
    const n = Number(value);
    ms = n < 1e11 ? n * 1000 : n;
  } else if (typeof value === 'string' && value.trim()) {
    ms = Date.parse(value);
  }
  return Number.isFinite(ms) && ms >= Date.UTC(2000, 0, 1) && ms <= now + 86_400_000 ? Math.trunc(ms) : null;
}

export function normalize(host, input, eventArg = '') {
  const payload = input && typeof input === 'object' && !Array.isArray(input) ? input : null;
  let result;
  try {
    result = payload ? NORMALIZE[host]?.(payload, eventArg) ?? ignore('unknown host') : ignore('no payload');
  } catch {
    result = ignore('bad payload');
  }
  // The host's own time of the message, so a stored message carries when it
  // was said (and a queued retry resends that time, not the retry's).
  if (['prompt', 'reply', 'buffer'].includes(result.action)) {
    const time = hostTime(payload?.timestamp);
    if (time !== null) result.time = time;
  }
  // An ignored event still gets the host's answer for that event.
  if (!result.event) result.event = eventOf(payload, eventArg, 'hook_event_name', 'hookName', 'event');
  return result;
}

/**
 * The stdout for an action: exactly one JSON value, or '' for nothing.
 * @param {string} host
 * @param {{action: string, event?: string}} action
 * @param {string|null} context - recall block or forget note to add to the model context.
 */
export function render(host, action, context = null) {
  const event = action?.event ?? '';
  switch (host) {
    case 'gemini':
      // suppressOutput keeps the injected memory out of Gemini's own logs and telemetry.
      if (context && event === 'BeforeAgent') return JSON.stringify({ suppressOutput: true, hookSpecificOutput: { hookEventName: 'BeforeAgent', additionalContext: context } });
      return '{}';
    case 'devin':
      if (context && event === 'UserPromptSubmit') return JSON.stringify({ hookSpecificOutput: { hookEventName: 'UserPromptSubmit', additionalContext: context } });
      return '';
    case 'cline':
      if (context && event === 'UserPromptSubmit') return JSON.stringify({ cancel: false, contextModification: context });
      return JSON.stringify({ cancel: false });
    case 'cursor':
      // beforeSubmitPrompt adds context only on Cursor versions that read
      // additional_context there; the runner passes none on the others.
      if (event === 'beforeSubmitPrompt') return JSON.stringify(context ? { continue: true, additional_context: context } : { continue: true });
      if (event === 'sessionStart') return JSON.stringify(context ? { additional_context: context } : {});
      return '{}';
    case 'goose':
      // goose reads the context from the MOIM file; stdout must stay empty
      // (on Stop, any output would be read as a decision).
      return '';
    default:
      return '';
  }
}

/** The stdout when the hook does nothing (bad input, memory off, a timeout). */
export function fallback(host, eventArg = '', input = null) {
  return render(host, normalize(host, input, eventArg), null);
}
