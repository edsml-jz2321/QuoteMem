// Turning finished turns into /memory/add batches.
//
// A capture state is a plain JSON-safe object, so a long-lived host keeps it
// in memory and a short-lived hook process keeps it in a session file. One
// finished user turn is recorded with `recordTurn`; `cutBatch` turns what is
// pending into /memory/add jobs with deterministic request ids.
import { createHash } from 'node:crypto';

import { MAX_ID_CHARS } from './config.mjs';
import { endsWithQuestion, isCommand, isTrivial } from './gate.mjs';
import { REDACT_MARGIN, redactText } from './redact.mjs';

const MAX_MESSAGES = 256;
// How long a session file is kept (hooks.mjs): its pending turns are queued
// at the latest when it is closed, a week after it was last touched.
export const SESSION_KEEP_MS = 7 * 86_400_000;
const MAX_CONTENT_BYTES = 1_048_576;
// A user message this short after an assistant question is saved with it.
const SHORT_ANSWER_CHARS = 24;

/** The text of a message content: a string, or the `text` blocks of a block list. */
export function textOf(content) {
  if (typeof content === 'string') return content.trim();
  if (!Array.isArray(content)) return '';
  return content
    .filter((block) => block && block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text)
    .join('\n')
    .trim();
}

/** The first 16 hex digits of the SHA-256 of a value's JSON. */
export function sha16(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 16);
}

/** The owner of a user's memory at one service: what keeps users and services apart in one state folder. */
export function ownerKey(url, userId) {
  return sha16(['owner', String(url ?? ''), String(userId ?? '')]);
}

/** A millisecond timestamp from a number or a date string; now when invalid. */
export function timeOf(value) {
  const t = typeof value === 'number' ? value : Date.parse(value);
  return Number.isFinite(t) && t >= 0 ? Math.trunc(t) : Date.now();
}

/**
 * One /memory/add message: redacted, cut to `maxMessageChars`. Null for a slash
 * command, an empty text or a text that is mostly a secret.
 * @param {any} time - when it was said; `null`: unknown, and the message
 *   carries no time (the service takes the time it receives it), so the
 *   same words give the same message every time.
 * @param {string} label - names the plugin in the truncation marker, e.g. "DSH plugin".
 */
export function wireMessage(settings, role, text, time, label = 'plugin') {
  if (!text || isCommand(text)) return null;
  // Cut first, so a huge paste costs no regex time in the turn's path.
  const cut = text.length > settings.maxMessageChars + REDACT_MARGIN;
  const redacted = redactText(cut ? text.slice(0, settings.maxMessageChars + REDACT_MARGIN) : text);
  if (redacted.secretOnly) return null;
  let content = redacted.text.trim();
  if (!content) return null;
  if (cut || content.length > settings.maxMessageChars) content = `${content.slice(0, settings.maxMessageChars)}\n[truncated by the QuoteMem ${label}]`;
  return time === null ? { role, content } : { role, content, timestamp: timeOf(time) };
}

/** An empty capture state. */
export function newCaptureState() {
  return { pending: [], turns: 0, firstTurn: null, lastTurn: null, question: null, questionSent: false };
}

/**
 * Record one finished user turn.
 * @param {object} state - a capture state (changed in place).
 * @param {{users: {text: string, time?: any}[], assistant?: {text: string, time?: any}|null,
 *   turn?: number|null, completed?: boolean}} input - what the user said in the
 *   turn and the final assistant reply. The reply is kept only for a
 *   completed turn; an interrupted or failed turn keeps the user's words.
 * @param {{label?: string}} options - see wireMessage.
 * @returns {boolean} true when a batch is due (`captureEvery` user turns pending).
 */
export function recordTurn(settings, state, { users = [], assistant = null, turn = null, completed = true }, { label } = {}) {
  const reply = completed && settings.captureAssistant === 'final' && assistant
    ? wireMessage(settings, 'assistant', assistant.text, assistant.time, label)
    : null;
  const question = state.question;
  const questionSent = Boolean(state.questionSent);
  state.question = reply && endsWithQuestion(assistant.text) ? reply : null;
  state.questionSent = false;
  // A short answer ("no", "对") to the assistant's question is kept, with the
  // question when it was not stored yet: neither in the pending batch nor in
  // a batch already sent (the service reads that one as context).
  const answered = question !== null && question !== undefined;
  const said = users.filter((u) => (answered ? String(u.text ?? '').trim().length > 0 : !isTrivial(u.text)))
    .map((u) => wireMessage(settings, 'user', u.text, u.time, label)).filter(Boolean);
  if (said.length === 0) return false; // nothing the user said is worth storing
  const messages = [];
  if (answered && !questionSent && said.every((m) => [...m.content].length <= SHORT_ANSWER_CHARS) && state.pending.at(-1)?.content !== question.content) {
    messages.push(question);
  }
  messages.push(...said);
  if (reply) messages.push(reply);
  state.pending.push(...messages);
  state.turns += 1;
  if (state.firstTurn === null) state.firstTurn = turn ?? state.turns;
  state.lastTurn = turn ?? state.firstTurn + state.turns - 1;
  return state.turns >= settings.captureEvery;
}

/**
 * Split collected messages into /memory/add bodies within the service limits. The
 * request_id is derived from the session, the turn range and the content, so
 * a resend or a replay after restart carries the same id.
 * @param {string} sessionId - the full session id sent to the service, e.g. `dsh:<id>`.
 * @param {{untimed?: boolean}} [options] - `untimed`: the id is derived from
 *   the words only, not the message times, and marks the range with `x`
 *   (`:x3-3:`): for turns read again from a transcript, whose times may
 *   differ from one reading to the next.
 */
export function buildJobs(settings, sessionId, firstTurn, lastTurn, messages, now = Date.now(), { untimed = false } = {}) {
  const chunks = [];
  let current = [];
  let bytes = 0;
  for (const message of messages) {
    const size = Buffer.byteLength(message.content, 'utf8');
    if (current.length > 0 && (current.length >= MAX_MESSAGES || bytes + size > MAX_CONTENT_BYTES)) {
      chunks.push(current);
      current = [];
      bytes = 0;
    }
    current.push(message);
    bytes += size;
  }
  if (current.length > 0) chunks.push(current);
  const session = String(sessionId).slice(0, MAX_ID_CHARS);
  return chunks.map((chunk, index) => {
    const part = chunks.length > 1 ? `:p${index + 1}` : '';
    const key = sha16(untimed ? chunk.map(({ role, content }) => [role, content]) : chunk);
    const suffix = `:${untimed ? 'x' : 't'}${firstTurn}-${lastTurn}${part}:${key}`;
    const requestId = `${session.slice(0, MAX_ID_CHARS - suffix.length)}${suffix}`;
    return {
      request_id: requestId,
      body: { user_id: settings.userId, session_id: session, request_id: requestId, messages: chunk },
      url: settings.url,
      created: now,
      attempts: 0,
      nextAt: 0,
      maybeSent: false,
      rotated: false,
    };
  });
}

/**
 * Turn the pending messages into /memory/add jobs and reset the batch.
 *
 * A batch holds only turns not sent before. A batch cut early (before a
 * forget, at a session end, after idle time) needs no earlier turns: the
 * service reads the session's previous batch as context by itself, and a
 * turn sent twice would be stored twice.
 * @returns {object[]} the jobs; empty when nothing is pending.
 */
export function cutBatch(settings, state, sessionId, { now = Date.now(), untimed = false } = {}) {
  if (state.pending.length === 0) return [];
  const jobs = buildJobs(settings, sessionId, state.firstTurn, state.lastTurn, state.pending, now, { untimed });
  const question = state.question;
  if (question && state.pending.some((m) => m.role === 'assistant' && m.content === question.content)) state.questionSent = true;
  state.pending = [];
  state.turns = 0;
  state.firstTurn = null;
  state.lastTurn = null;
  return jobs;
}
