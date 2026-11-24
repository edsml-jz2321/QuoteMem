#!/usr/bin/env node
// The transcript reader: the turn's last assistant reply from a JSONL
// (Claude Code) or ATIF transcript, used when a Stop event carries no reply.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { CONVERSATION_MAX_BYTES, alignTurns, conversationTurns, cursorStampTime, lastAssistantReply, lastAssistantText, promptHead, readConversation, replyFromLines, replyFromSteps, turnReply, turnsFromLines } from '../index.mjs';
import { tempDir } from './helpers.mjs';

const root = tempDir('qm-transcript-');
let seq = 0;
function file(content, name = null) {
  seq += 1;
  const out = path.join(root, name ?? `t-${seq}.jsonl`);
  fs.writeFileSync(out, content);
  return out;
}
const jsonl = (records) => `${records.map((r) => (typeof r === 'string' ? r : JSON.stringify(r))).join('\n')}\n`;
const user = (content, extra = {}) => ({ type: 'user', message: { role: 'user', content }, ...extra });
const assistant = (content, extra = {}) => ({ type: 'assistant', message: { role: 'assistant', content }, ...extra });

test('JSONL: the last main-thread reply after the turn\'s own message, with its time', () => {
  const at = '2026-11-20T10:00:05Z';
  const t = file(jsonl([
    user('First question.'),
    assistant([{ type: 'text', text: 'First answer.' }]),
    user('We picked Postgres.'),
    assistant([{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }]),
    user([{ type: 'tool_result', tool_use_id: 't1', content: 'tool output' }]),
    user('hook note', { isMeta: true }),
    assistant([{ type: 'text', text: 'Postgres it is.' }], { timestamp: at }),
    assistant([{ type: 'text', text: 'side work' }], { isSidechain: true }),
    assistant([{ type: 'image', source: {} }]),
  ]));
  assert.deepEqual(lastAssistantReply(t), { text: 'Postgres it is.', time: Date.parse(at) });
  assert.equal(lastAssistantText(t), 'Postgres it is.');
});

test('JSONL: no reply of an earlier turn when this turn\'s reply is not written yet', () => {
  const t = file(jsonl([user('First question.'), assistant('First answer.'), user('Second question.')]));
  assert.equal(lastAssistantReply(t), null);
  // A sidechain user message does not end the search.
  const side = file(jsonl([user('Q.'), assistant('A.'), user('subagent task', { isSidechain: true })]));
  assert.equal(lastAssistantText(side), 'A.');
});

test('ATIF: the last agent step after the last user step, string or text parts', () => {
  const doc = {
    schema_version: 'ATIF-v1.7',
    session_id: 's',
    agent: { name: 'devin' },
    steps: [
      { step_id: 1, source: 'user', message: 'First?' },
      { step_id: 2, source: 'agent', message: 'First answer.' },
      { step_id: 3, source: 'user', message: 'Which database did we pick?' },
      { step_id: 4, source: 'agent', message: '', tool_calls: [{ function_name: 'grep' }] },
      { step_id: 5, source: 'agent', message: [{ type: 'text', text: 'Postgres.' }], timestamp: '2026-11-20T10:00:07Z' },
      { step_id: 6, source: 'system', message: 'done' },
    ],
  };
  assert.deepEqual(lastAssistantReply(file(JSON.stringify(doc, null, 2), 'a.json')), { text: 'Postgres.', time: Date.parse('2026-11-20T10:00:07Z') });
  assert.equal(replyFromSteps({ steps: [...doc.steps.slice(0, 3)] }), null, 'the reply is not written yet');
  assert.equal(replyFromSteps({ steps: 'x' }), null);
  assert.equal(replyFromSteps(null), null);
  // A bad time is dropped; the text is kept.
  assert.deepEqual(replyFromSteps({ steps: [{ source: 'user', message: 'q' }, { source: 'agent', message: 'a', timestamp: 'later' }] }), { text: 'a', time: null });
});

test('with the turn\'s head: only a reply to that user message; a lagging transcript gives null', () => {
  const head = promptHead('and 3+3?');
  // The host has not written this turn yet: the file ends with turn 1.
  const lagging = file(jsonl([user('what is 2+2?'), assistant('FOUR (turn 1)', { timestamp: '2026-11-20T10:00:01Z' })]));
  assert.equal(lastAssistantReply(lagging, { head }), null);
  assert.equal(lastAssistantText(lagging), 'FOUR (turn 1)', 'without a head, the old behaviour');
  // Written: the reply to "and 3+3?" is taken. Whitespace and paste markers do not matter.
  const written = file(jsonl([user('what is 2+2?'), assistant('FOUR'), user('and   3+3?\n'), assistant('SIX')]));
  assert.equal(lastAssistantText(written, { head }), 'SIX');
  assert.equal(promptHead('<pasted_content>\nsome text\n</pasted_content>\n'), promptHead('some text'));
  assert.equal(promptHead('  '), null);
  assert.equal(promptHead(`${'a'.repeat(200)}b`), promptHead(`${'a'.repeat(200)}c`), 'only the first 200 characters count');
  // No user message to check against: null.
  assert.equal(replyFromLines([JSON.stringify(assistant('orphan'))], { head }), null);
  assert.equal(lastAssistantReply(written, { head: null }), null, 'an unknown head matches nothing');
  // ATIF the same way.
  const steps = [{ source: 'user', message: 'what is 2+2?' }, { source: 'agent', message: 'FOUR' }];
  assert.equal(replyFromSteps({ steps }, { head }), null);
  assert.equal(replyFromSteps({ steps: [...steps, { source: 'user', message: 'and 3+3?' }, { source: 'agent', message: 'SIX' }] }, { head })?.text, 'SIX');
});

test('a reply time before the turn\'s own time is dropped, the text kept', () => {
  const head = promptHead('Which database?');
  const t = file(jsonl([user('Which database?'), assistant('Postgres.', { timestamp: '2026-11-20T10:00:05Z' })]));
  const at = Date.parse('2026-11-20T10:00:05Z');
  assert.deepEqual(lastAssistantReply(t, { head, after: at - 1000 }), { text: 'Postgres.', time: at });
  assert.deepEqual(lastAssistantReply(t, { head, after: at + 1000 }), { text: 'Postgres.', time: null });
});

test('a large JSONL file is read from its tail only; an ATIF file over 2 MiB gives null', () => {
  const filler = JSON.stringify(assistant('x'.repeat(1000)));
  const big = file(`${Array(2400).fill(filler).join('\n')}\n${JSON.stringify(user('Last question.'))}\n${JSON.stringify(assistant('final words'))}\n`);
  assert.ok(fs.statSync(big).size > 2 * 1024 * 1024);
  assert.equal(lastAssistantText(big), 'final words');
  const steps = Array.from({ length: 3000 }, (_, i) => ({ source: i % 2 ? 'agent' : 'user', message: 'y'.repeat(800) }));
  const atif = file(JSON.stringify({ steps }), 'big.json');
  assert.ok(fs.statSync(atif).size > 2 * 1024 * 1024);
  assert.equal(lastAssistantReply(atif), null);
});

test('missing, odd or hostile files give null, fast, and never throw', () => {
  const started = Date.now();
  const cases = [
    undefined, null, '', 42, {}, 'x'.repeat(5000),
    path.join(root, 'missing.jsonl'),
    root, // a directory
    file(''),
    file('not json at all\n{"type": "assistant", "message": {"content": "cut'),
    file(jsonl([user('q'), '{"type":"assistant","message":{"content":"half'])),
    file(jsonl([assistant([{ type: 'image', source: {} }])])),
    file('{"steps": [}', 'broken.json'),
    file(jsonl([assistant('only a sidechain', { isSidechain: true })])),
    '/dev/zero',
  ];
  for (const value of cases) assert.equal(lastAssistantReply(value), null, String(value).slice(0, 60));
  // A FIFO with no writer would block an open: it is never opened.
  if (process.platform !== 'win32') {
    const fifo = path.join(root, 'fifo');
    execFileSync('mkfifo', [fifo]);
    assert.equal(lastAssistantReply(fifo), null);
    // A file the user cannot read.
    const locked = file(jsonl([user('q'), assistant('secret')]));
    fs.chmodSync(locked, 0o000);
    if (process.getuid?.() !== 0) assert.equal(lastAssistantReply(locked), null);
  }
  assert.ok(Date.now() - started < 3000, `took ${Date.now() - started} ms`);
});

// The shape of Cursor CLI 2026.11.19's agent transcript (agent-transcripts/<id>/<id>.jsonl).
const cursorLine = (role, text) => JSON.stringify({ role, message: { content: [{ type: 'text', text }] } });
const cursorUser = (stamp, words) => cursorLine('user', `<timestamp>${stamp}</timestamp>\n<user_query>\n${words}\n</user_query>`);

test('Cursor transcript: each user message with its final reply, wrappers stripped, the local time read', () => {
  const lines = [
    cursorUser('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Note for later: my kite is orange.\nReply in one short sentence.'),
    cursorLine('assistant', 'Let me think.'),
    JSON.stringify({ role: 'assistant', message: { content: [{ type: 'tool_use', name: 'x' }, { type: 'text', text: 'Got it: an orange kite.' }] } }),
    // A user-role line without <user_query> (a tool result, say) is not a user turn.
    cursorLine('user', 'tool output'),
    JSON.stringify({ role: 'user', message: { content: [{ type: 'tool_result', content: 'x' }] } }),
    cursorUser('Thursday, Nov 20, 2025, 12:05 PM (UTC-5:30)', 'What is my favourite tea called?'),
    cursorLine('assistant', 'OOLONG'),
    JSON.stringify({ type: 'turn_ended', status: 'success' }),
    'not json',
    cursorUser('Thursday, Nov 20, 2025, 12:30 AM (UTC)', 'And my bike?'),
  ];
  const turns = turnsFromLines(lines);
  assert.deepEqual(turns, [
    { user: { text: 'Note for later: my kite is orange.\nReply in one short sentence.', time: Date.UTC(2025, 10, 20, 3, 4) }, assistant: { text: 'Got it: an orange kite.', time: null } },
    { user: { text: 'What is my favourite tea called?', time: Date.UTC(2025, 10, 20, 17, 35) }, assistant: { text: 'OOLONG', time: null } },
    { user: { text: 'And my bike?', time: Date.UTC(2025, 10, 20, 0, 30) }, assistant: null },
  ]);
  // Claude Code's records are read the same way; a side chain is not.
  const claude = turnsFromLines([
    JSON.stringify({ type: 'user', message: { content: 'I live in Lisbon.' }, timestamp: '2025-11-20T10:00:00Z' }),
    JSON.stringify({ type: 'assistant', isSidechain: true, message: { content: [{ type: 'text', text: 'side' }] } }),
    JSON.stringify({ type: 'assistant', message: { content: [{ type: 'text', text: 'Noted.' }] }, timestamp: '2025-11-20T10:00:05Z' }),
  ]);
  assert.deepEqual(claude, [{ user: { text: 'I live in Lisbon.', time: Date.UTC(2025, 10, 20, 10) }, assistant: { text: 'Noted.', time: Date.UTC(2025, 10, 20, 10, 0, 5) } }]);
  assert.deepEqual(turnsFromLines([cursorLine('assistant', 'orphan'), cursorUser('x', '   ')]), [], 'no reply without a user turn; an empty query is none');
});

test('Cursor timestamps: 12 AM and PM, offsets with minutes, no offset; anything else is no time', () => {
  assert.equal(cursorStampTime('Saturday, Nov 21, 2026, 11:00 AM (UTC+8)'), Date.UTC(2026, 10, 21, 3, 0));
  assert.equal(cursorStampTime('Thu, Jan 1, 2026, 12:00 AM (UTC)'), Date.UTC(2026, 0, 1, 0, 0));
  assert.equal(cursorStampTime('Thu, Jan 1, 2026, 12:00 PM (UTC+05:45)'), Date.UTC(2026, 0, 1, 6, 15));
  assert.equal(cursorStampTime('Thu, Jan 1, 2026, 23:10 (UTC-3)'), Date.UTC(2026, 0, 2, 2, 10), 'a 24-hour clock');
  for (const bad of ['', 'yesterday', 'Thu, Foo 1, 2026, 11:00 AM (UTC)', 'Thu, Jan 1, 2026, 13:00 PM (UTC)', 'Thu, Jan 1, 1990, 11:00 AM (UTC)', 'Thu, Jan 1, 2026, 11:00 AM']) {
    assert.equal(cursorStampTime(bad), null, bad);
  }
});

test('conversationTurns: a regular file only, read whole', () => {
  const dir = tempDir('qm-conversation-');
  const file = path.join(dir, 'c.jsonl');
  fs.writeFileSync(file, `${cursorUser('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'Hello there.')}\n${cursorLine('assistant', 'Hi.')}\n`);
  assert.deepEqual(conversationTurns(file).map((turn) => [turn.user.text, turn.assistant?.text]), [['Hello there.', 'Hi.']]);
  assert.equal(conversationTurns(dir), null, 'a folder');
  assert.equal(conversationTurns(path.join(dir, 'missing.jsonl')), null);
  assert.equal(conversationTurns(''), null);
  assert.equal(conversationTurns(null), null);
});

test('readConversation: only what was added since an offset, complete lines only; a rewritten or shorter file is named', () => {
  const dir = tempDir('qm-conversation-tail-');
  const file = path.join(dir, 'c.jsonl');
  const one = `${cursorUser('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'One.')}\n${cursorLine('assistant', 'Reply one.')}\n`;
  fs.writeFileSync(file, `${one}${cursorUser('Thursday, Nov 20, 2025, 11:05 AM (UTC+8)', 'Two.')}`);
  const first = readConversation(file);
  assert.deepEqual(first.turns.map((t) => t.user.text), ['One.'], 'a last line still being written is not read');
  assert.equal(first.end, Buffer.byteLength(one));
  assert.equal(first.turns[0].start, 0);
  const guard = first.markAt(first.end);
  fs.appendFileSync(file, `\n${cursorLine('assistant', 'Reply two.')}\n`);
  const next = readConversation(file, { from: first.end, guard });
  assert.deepEqual(next.turns.map((t) => [t.user.text, t.assistant?.text, t.start]), [['Two.', 'Reply two.', first.end]]);
  // The line before the offset changed (a restore cut the chat there and it
  // went on): the file was rewritten.
  fs.writeFileSync(file, `${cursorUser('Thursday, Nov 20, 2025, 11:04 AM (UTC+8)', 'One.')}\n${cursorLine('assistant', 'Reply 1!!.')}\n${cursorUser('x', 'Two.')}\n${cursorLine('assistant', 'Reply two.')}\n`);
  assert.deepEqual(readConversation(file, { from: first.end, guard }), { changed: true });
  // Shorter than the offset.
  fs.writeFileSync(file, `${cursorUser('x', 'One.')}\n`);
  assert.deepEqual(readConversation(file, { from: first.end, guard }), { changed: true });
  // More than one reading holds.
  fs.writeFileSync(file, 'x'.repeat(CONVERSATION_MAX_BYTES + 1));
  assert.deepEqual(readConversation(file), { tooLarge: true });
});

test('alignTurns: the longest in-order match, the earliest where several fit', () => {
  assert.deepEqual(alignTurns(['A', 'C', 'A'], ['C', 'A']), [-1, 0, 1], 'a repeated message does not match past one in between');
  assert.deepEqual(alignTurns(['A', 'C', 'A'], ['A']), [0, -1, -1]);
  assert.deepEqual(alignTurns(['A', 'B'], ['X', 'A', 'Y', 'B', 'Z']), [1, 3]);
  assert.deepEqual(alignTurns([null, 'A'], ['A']), [-1, 0], 'an empty message matches nothing');
  assert.deepEqual(alignTurns(['A'], []), [-1]);
  assert.deepEqual(alignTurns([], ['A']), []);
});

test('forgets applied again are kept a week, as long as a session file: a turn said before one can still wait that long', async () => {
  const { FORGOTTEN_KEEP_MS, forgottenFile, forgottenOf } = await import('../forget.mjs');
  const { SESSION_KEEP_MS } = await import('../capture.mjs');
  assert.equal(FORGOTTEN_KEEP_MS, SESSION_KEEP_MS);
  assert.equal(FORGOTTEN_KEEP_MS, 7 * 86_400_000);
  const home = tempDir('qm-forgotten-');
  const owner = { url: 'http://127.0.0.1:9', userId: 'quei' };
  const file = forgottenFile({ HOME: home, USERPROFILE: home }, owner);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const day = 86_400_000;
  fs.writeFileSync(file, JSON.stringify({ version: 1, items: [{ statement: 'eight days ago', at: Date.now() - 8 * day }, { statement: 'three days ago', at: Date.now() - 3 * day }] }));
  assert.deepEqual(forgottenOf({ HOME: home, USERPROFILE: home }, owner).map((item) => item.statement), ['three days ago']);
});

test('the forgets to apply again follow QUOTEMEM_STATE_DIR when it is set, and ~/.quotemem keeps them too', async () => {
  const { forgottenFile, forgottenFiles } = await import('../forget.mjs');
  const owner = { url: 'http://127.0.0.1:9', userId: 'quei' };
  const u = path.resolve('/home/u');
  const at = (extra = {}) => ({ HOME: u, USERPROFILE: u, ...extra });
  const dirs = (env) => forgottenFiles(env, owner).map((file) => path.dirname(file));
  assert.equal(path.dirname(forgottenFile(at(), owner)), path.join(u, '.quotemem', 'forgotten'));
  assert.equal(path.dirname(forgottenFile(at({ QUOTEMEM_STATE_DIR: '~/state' }), owner)), path.join(u, 'state', 'forgotten'));
  assert.equal(path.dirname(forgottenFile(at({ QUOTEMEM_STATE_DIR: '/var/qm' }), owner)), path.join(path.resolve('/var/qm'), 'forgotten'));
  assert.deepEqual(dirs(at()), [path.join(u, '.quotemem', 'forgotten')]);
  assert.deepEqual(dirs(at({ QUOTEMEM_STATE_DIR: '/var/qm' })), [path.join(path.resolve('/var/qm'), 'forgotten'), path.join(u, '.quotemem', 'forgotten')]);
  assert.deepEqual(dirs(at({ QUOTEMEM_STATE_DIR: '~/.quotemem' })), [path.join(u, '.quotemem', 'forgotten')], 'one file when they are the same');
});

test('turnReply: one turn\'s reply from the transcript\'s tail, reading further back only when the turn is not there', () => {
  const dir = tempDir('qm-turn-reply-');
  const file = path.join(dir, 'c-1.jsonl');
  const line = (role, text) => `${JSON.stringify({ role, message: { content: [{ type: 'text', text }] } })}\n`;
  const turn = (user, reply) => line('user', `<user_query>\n${user}\n</user_query>`) + (reply === null ? '' : line('assistant', reply));
  const ended = `${JSON.stringify({ type: 'turn_ended' })}\n`;
  // An early turn, then more than 512 KiB of other turns, then the last one.
  const filler = Array.from({ length: 700 }, (_, i) => turn(`Filler message ${i} ${'x'.repeat(900)}`, `Reply ${i}`)).join('');
  fs.writeFileSync(file, turn('My kite is orange.', 'Orange kite, noted.') + filler + turn('My bike is blue.', 'Blue bike, noted.') + ended);
  assert.ok(fs.statSync(file).size > 600 * 1024);
  assert.deepEqual(turnReply(file, promptHead('My bike is blue.')), { text: 'Blue bike, noted.', time: null, ended: true });
  assert.deepEqual(turnReply(file, promptHead('My kite is orange.')), { text: 'Orange kite, noted.', time: null, ended: true }, 'found further back');
  assert.equal(turnReply(file, promptHead('Never said.')), null);
  // The turn is there but its reply is not written yet.
  fs.appendFileSync(file, turn('My boat is green.', null));
  assert.equal(turnReply(file, promptHead('My boat is green.')), null);
  assert.equal(turnReply(path.join(dir, 'missing.jsonl'), promptHead('My bike is blue.')), null);
});
