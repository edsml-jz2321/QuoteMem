// The last assistant reply of a turn, read from a host's transcript file.
//
// Used only when a host's end-of-turn event carries no reply text but names
// a transcript file. Two formats are read:
//   - JSONL with one record per line, Claude Code's format: {type:
//     'assistant'|'user', message: {content}, timestamp, isSidechain};
//   - one JSON document in the Agent Trajectory Interchange Format (ATIF):
//     {steps: [{source: 'user'|'agent', message, timestamp}]}.
// Neither format is a documented hook contract, so anything unexpected
// gives null and the turn keeps only the user's words.
//
// A host may write its transcript after the hook runs, so the file can still
// end with an earlier turn. The caller passes `head`, the promptHead() of the
// open turn's user message: a reply is returned only when the user message
// it answers has that head. A reply of an earlier turn is then never
// returned. A reply time before `after` (the turn's own time) is dropped.
// The reader reads regular files only, at most TAIL_BYTES of a JSONL file
// and DOCUMENT_MAX_BYTES of a JSON document, and never throws. An ATIF
// document larger than DOCUMENT_MAX_BYTES gives null.
import fs from 'node:fs';

import { sha16, textOf } from './capture.mjs';

const TAIL_BYTES = 512 * 1024;
const DOCUMENT_MAX_BYTES = 2 * 1024 * 1024;
const MAX_PATH_CHARS = 4096;
const AGENT_SOURCES = new Set(['agent', 'assistant']);
const HEAD_CHARS = 200;

/**
 * A short fingerprint of a user message: the hash of its first 200
 * characters, with paste markers dropped and whitespace collapsed. Null for
 * an empty message. It tells whether a transcript's user message is the one
 * a hook saw, without keeping the text.
 */
export function promptHead(text) {
  const clean = String(text ?? '')
    .replace(/^[ \t]*<\/?pasted_content\b[^>\n]*>[ \t]*(\r?\n|$)/gm, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, HEAD_CHARS);
  return clean ? sha16(clean) : null;
}

// The reply, if the user message it answers matches `head` (undefined: no
// check) and its time is not before `after`.
function checked(reply, userText, { head, after } = {}) {
  if (head !== undefined && (userText === null || promptHead(userText) !== head)) return null;
  if (reply.time !== null && Number.isFinite(after) && reply.time < after) return { ...reply, time: null };
  return reply;
}

// A host's time of a record in milliseconds, or null when it is missing or
// not plausible (before 2000, or more than a day ahead).
function timeOf(value) {
  const ms = typeof value === 'number' ? (value < 1e11 ? value * 1000 : value) : typeof value === 'string' ? Date.parse(value) : NaN;
  return Number.isFinite(ms) && ms >= Date.UTC(2000, 0, 1) && ms <= Date.now() + 86_400_000 ? Math.trunc(ms) : null;
}

// The text of an ATIF message: a string, or text parts.
function stepText(message) {
  if (typeof message === 'string') return message.trim();
  return textOf(message);
}

/**
 * The last agent reply after the last user step of an ATIF document, or null.
 * @param {object} document
 * @param {{head?: string, after?: number}} [expect] - see the top of this file.
 */
export function replyFromSteps(document, expect = {}) {
  const steps = Array.isArray(document?.steps) ? document.steps : null;
  if (!steps) return null;
  let reply = null;
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const step = steps[i];
    if (!step || typeof step !== 'object') continue;
    const source = String(step.source ?? '').toLowerCase();
    const text = stepText(step.message);
    if (source === 'user' && text) return reply ? checked(reply, text, expect) : null; // no reply: not written yet
    if (!reply && AGENT_SOURCES.has(source) && text) reply = { text, time: timeOf(step.timestamp) };
  }
  return reply ? checked(reply, null, expect) : null;
}

// A user record that holds a message the user typed (not a tool result or a
// note the host added).
function isUserMessage(record) {
  if (record?.type !== 'user' || record.isSidechain || record.isMeta || record.isCompactSummary) return false;
  return textOf(record.message?.content) !== '';
}

/**
 * The last main-thread assistant reply after the last user message in JSONL
 * lines, or null.
 * @param {string[]} lines
 * @param {{head?: string, after?: number}} [expect] - see the top of this file.
 */
export function replyFromLines(lines, expect = {}) {
  let reply = null;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    let record;
    try {
      record = JSON.parse(lines[i]);
    } catch {
      continue;
    }
    if (isUserMessage(record)) return reply ? checked(reply, textOf(record.message?.content), expect) : null; // no reply: not written yet
    if (reply || record?.type !== 'assistant' || record.isSidechain) continue;
    const text = textOf(record.message?.content);
    if (text) reply = { text, time: timeOf(record.timestamp) };
  }
  return reply ? checked(reply, null, expect) : null;
}

/**
 * The last assistant reply in a transcript file.
 * @param {string} file - the transcript path the host gave.
 * @param {{head?: string, after?: number}} [expect] - `head`: promptHead() of
 *   the turn's user message; the reply must answer it. `after`: the turn's
 *   time; an earlier reply time is dropped. Without `head`, any last reply
 *   after the last user message is returned.
 * @returns {{text: string, time: number|null}|null}
 */
export function lastAssistantReply(file, expect = {}) {
  if (typeof file !== 'string' || file.length === 0 || file.length > MAX_PATH_CHARS) return null;
  let fd = null;
  try {
    // Only a regular file: a FIFO or a device could block or never end.
    if (!fs.statSync(file).isFile()) return null;
    fd = fs.openSync(file, fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK ?? 0));
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || stat.size === 0) return null;
    const size = stat.size;
    const length = Math.min(size, size <= DOCUMENT_MAX_BYTES ? size : TAIL_BYTES);
    const buffer = Buffer.alloc(length);
    let read = 0;
    while (read < length) {
      const n = fs.readSync(fd, buffer, read, length - read, size - length + read);
      if (n <= 0) break;
      read += n;
    }
    const text = buffer.subarray(0, read).toString('utf8');
    if (length === size && /^\s*\{/.test(text)) {
      try {
        const document = JSON.parse(text);
        if (Array.isArray(document?.steps)) return replyFromSteps(document, expect);
      } catch {
        // not one document: read it as JSONL
      }
    }
    const lines = text.split('\n');
    if (length < size) lines.shift(); // a partial first line
    return replyFromLines(lines, expect);
  } catch {
    return null;
  } finally {
    if (fd !== null) try { fs.closeSync(fd); } catch { /* closed */ }
  }
}

/** The text of lastAssistantReply, or null. */
export function lastAssistantText(file, expect = {}) {
  return lastAssistantReply(file, expect)?.text ?? null;
}

// ------------------------------------------------ a whole conversation
// Cursor's agent transcript (agent-transcripts/<chat id>/<chat id>.jsonl)
// holds every turn of a chat so far, one record per line in the order said:
//   {"role":"user","message":{"content":[{"type":"text","text":"<timestamp>Saturday,
//    Nov 21, 2026, 11:00 AM (UTC+8)</timestamp>\n<user_query>\n…\n</user_query>"}]}}
//   {"role":"assistant","message":{"content":[{"type":"text","text":"…"}]}}
// and {"type":"turn_ended"} lines. No line has an id or a machine time; the
// timestamp is the user's local time to the minute. A user line counts only
// with its <user_query> part (so a tool line in another shape is never taken
// for the user), and only text blocks are read. Claude Code's {type:
// 'user'|'assistant', message} records are read too. Any other line is
// skipped.

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
const STAMP = /([a-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4}),?\s+(\d{1,2}):(\d{2})\s*([ap]m)?\s*\(\s*(?:utc|gmt)\s*(?:([+-])(\d{1,2})(?::?(\d{2}))?)?\s*\)/i;

/** The time of a Cursor `<timestamp>` text ("Saturday, Nov 21, 2026, 11:00 AM (UTC+8)") in milliseconds, or null. */
export function cursorStampTime(text) {
  const match = STAMP.exec(String(text ?? ''));
  if (!match) return null;
  const month = MONTHS.indexOf(match[1].slice(0, 3).toLowerCase());
  let hour = Number(match[4]);
  const minute = Number(match[5]);
  const half = match[6]?.toLowerCase();
  if (month < 0 || hour > 23 || minute > 59 || (half && (hour < 1 || hour > 12))) return null;
  if (half === 'pm' && hour < 12) hour += 12;
  if (half === 'am' && hour === 12) hour = 0;
  const offset = match[7] ? (match[7] === '-' ? -1 : 1) * (Number(match[8]) * 60 + Number(match[9] ?? 0)) : 0;
  return timeOf(Date.UTC(Number(match[3]), month, Number(match[2]), hour, minute) - offset * 60_000);
}

// The user's own words of a Cursor user line, and its time; null when the
// line has no <user_query> part.
function cursorUser(text) {
  const query = /<user_query>\r?\n?([\s\S]*)<\/user_query>/.exec(text);
  if (!query) return null;
  const stamp = /<timestamp>([^<]*)<\/timestamp>/.exec(text);
  const words = query[1].trim();
  return words ? { text: words, time: stamp ? cursorStampTime(stamp[1]) : null } : null;
}

/**
 * The user turns of a whole conversation: each user message with the last
 * assistant text after it (its final reply), in order.
 * @param {string[]} lines - JSONL lines.
 * @returns {{user: {text: string, time: number|null}, assistant: {text: string, time: number|null}|null}[]}
 */
export function turnsFromLines(lines) {
  return turnsFromRecords(lines.map((line) => ({ line, start: null, end: null }))).map(({ user, assistant }) => ({ user, assistant }));
}

// turnsFromLines for lines with their byte offsets. Each turn also has
// `start` and `end`, the offsets of its first line and after its last one;
// `cursor`, whether it is in Cursor's format; and `ended`: a later user
// message or a {"type":"turn_ended"} line follows it, so its reply is final.
function turnsFromRecords(records) {
  const turns = [];
  for (const { line, start, end } of records) {
    let record;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    if (!record || typeof record !== 'object' || record.isSidechain) continue;
    if (record.role === 'user') {
      const user = cursorUser(textOf(record.message?.content));
      if (!user) continue;
      if (turns.length > 0) turns.at(-1).ended = true;
      turns.push({ user, assistant: null, start, end, cursor: true, ended: false });
    } else if (isUserMessage(record)) {
      if (turns.length > 0) turns.at(-1).ended = true;
      turns.push({ user: { text: textOf(record.message?.content), time: timeOf(record.timestamp) }, assistant: null, start, end, cursor: false, ended: false });
    } else if ((record.role === 'assistant' || record.type === 'assistant') && turns.length > 0) {
      const text = textOf(record.message?.content);
      if (text) Object.assign(turns.at(-1), { assistant: { text, time: timeOf(record.timestamp) }, end });
    } else if (record.type === 'turn_ended' && turns.length > 0) {
      turns.at(-1).ended = true;
    }
  }
  return turns;
}

/** The most of a transcript read at once: the part after the last reading. */
export const CONVERSATION_MAX_BYTES = 32 * 1024 * 1024;

/**
 * The user turns of a conversation transcript from byte `from` on (0: the
 * whole file), for reading only what was added since the last time. Only
 * complete lines are read: a last line still being written is left for the
 * next reading.
 * @param {string} file - the transcript path the host gave: a regular file.
 * @param {{from?: number, guard?: {start: number, hash: string}|null, tail?: boolean}} [options] -
 *   `guard`: the line that ended at `from` last time (markAt); when it is no
 *   longer there, the file was rewritten. `tail`: when more than
 *   CONVERSATION_MAX_BYTES follow `from`, read only the last ones, from the
 *   first line that starts in them.
 * @returns {{turns: object[], end: number, size: number, skipped: boolean, markAt: function}
 *   | {changed: true} | {tooLarge: true} | null}
 *   `turns` as turnsFromRecords; `end`: the offset after the last complete
 *   line; `skipped`: text before the tail was not read; `markAt(offset)`: the
 *   guard for a later reading from `offset` (a line end). `changed`: the
 *   file is shorter than `from`, or the guard line differs (rewritten).
 *   `tooLarge`: more than CONVERSATION_MAX_BYTES to read, and no `tail`.
 *   Null when it cannot be read.
 */
export function readConversation(file, { from = 0, guard = null, tail = false } = {}) {
  if (typeof file !== 'string' || file.length === 0 || file.length > MAX_PATH_CHARS) return null;
  let fd = null;
  try {
    if (!fs.statSync(file).isFile()) return null;
    fd = fs.openSync(file, fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK ?? 0));
    const stat = fs.fstatSync(fd);
    if (!stat.isFile()) return null;
    const size = stat.size;
    if (from > size) return { changed: true };
    let base = guard && Number.isInteger(guard.start) && guard.start >= 0 && guard.start < from ? guard.start : from;
    const skipped = size - base > CONVERSATION_MAX_BYTES;
    if (skipped && !tail) return { tooLarge: true };
    if (skipped) base = size - CONVERSATION_MAX_BYTES;
    const buffer = Buffer.alloc(size - base);
    let read = 0;
    while (read < buffer.length) {
      const n = fs.readSync(fd, buffer, read, buffer.length - read, base + read);
      if (n <= 0) break;
      read += n;
    }
    const lineAt = (start, end) => buffer.subarray(start - base, end - base - 1).toString('utf8');
    let at = from - base;
    if (skipped) {
      // The tail starts inside a line: from the next one.
      at = buffer.indexOf(0x0a) + 1;
      if (at <= 0) at = read;
    } else if (base < from) {
      // The guard line must end where the last reading stopped.
      if (buffer[from - base - 1] !== 0x0a || sha16(lineAt(base, from)) !== guard.hash) return { changed: true };
    }
    const records = [];
    const ends = new Map();
    if (!skipped && base < from) ends.set(from, base);
    for (;;) {
      const newline = buffer.indexOf(0x0a, at);
      if (newline < 0 || newline >= read) break;
      records.push({ line: buffer.subarray(at, newline).toString('utf8'), start: base + at, end: base + newline + 1 });
      ends.set(base + newline + 1, base + at);
      at = newline + 1;
    }
    const markAt = (offset) => (ends.has(offset) ? { start: ends.get(offset), hash: sha16(lineAt(ends.get(offset), offset)) } : null);
    return { turns: turnsFromRecords(records), end: base + at, size, skipped, markAt };
  } catch {
    return null;
  } finally {
    if (fd !== null) try { fs.closeSync(fd); } catch { /* closed */ }
  }
}

// The tail of a transcript turnReply reads first, then wider ones.
const TURN_WINDOWS = [TAIL_BYTES, 4 * 1024 * 1024, CONVERSATION_MAX_BYTES];

/**
 * The reply of one turn of a Cursor transcript: the last user turn whose
 * message has this promptHead. Only the transcript's last 512 KiB is read,
 * then more only when the turn is not there.
 * @param {string} file - the transcript path the host gave.
 * @param {string} head - promptHead() of the turn's user message.
 * @returns {{text: string, time: number|null, ended: boolean}|null} null
 *   when the turn is not found, or has no reply yet; `ended`: a turn_ended
 *   line or a later message follows it.
 */
export function turnReply(file, head) {
  if (!head || typeof file !== 'string' || file.length === 0 || file.length > MAX_PATH_CHARS) return null;
  let size;
  try {
    size = fs.statSync(file).size;
  } catch {
    return null;
  }
  for (const window of TURN_WINDOWS) {
    const from = Math.max(0, size - window);
    // From inside a line, that first part is not JSON and is skipped.
    const read = readConversation(file, from === 0 ? { tail: true } : { from });
    const turn = read?.turns?.findLast((t) => t.cursor && promptHead(t.user.text) === head);
    if (turn) return turn.assistant ? { text: turn.assistant.text, time: turn.assistant.time, ended: turn.ended } : null;
    if (from === 0) break;
  }
  return null;
}

/** The user turns of a whole conversation transcript (turnsFromLines), or null when it cannot be read whole. */
export function conversationTurns(file) {
  const read = readConversation(file);
  return read?.turns ? read.turns.map(({ user, assistant }) => ({ user, assistant })) : null;
}
