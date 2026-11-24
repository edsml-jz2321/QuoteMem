#!/usr/bin/env node
// QuoteMem hook for Claude Code.
//
//   node hook.mjs <EventName>          < hook input JSON on stdin
//   quotemem hook claude-code <EventName>   (the same, from the npm package)
//
// Registered by hooks/hooks.json for SessionStart, UserPromptSubmit, Stop,
// StopFailure and SessionEnd. It always exits 0 and prints only the hook
// output Claude Code reads (or nothing), so a slow or missing QuoteMem
// service never blocks or changes a prompt. Diagnostics go to
// <state>/quotemem.log, never to stdout or stderr, and never the token.
import { settleLaunches } from '../core/index.mjs';
import { armIdle, runHook } from './adapter.mjs';

const MAX_INPUT_BYTES = 16 * 1024 * 1024;
const INPUT_WAIT_MS = 5000;
const event = process.argv[2] || '';
// A hard stop well inside Claude Code's limits: the prompt hook has 30 s
// (hooks.json), and every SessionEnd hook shares a 1.5 s budget.
const HARD_STOP_MS = event === 'UserPromptSubmit' ? 26_000 : event === 'SessionEnd' ? 1300 : 9000;
const STARTED = Date.now();
setTimeout(() => process.exit(0), HARD_STOP_MS).unref();

function readInput() {
  return new Promise((resolve) => {
    const chunks = [];
    let size = 0;
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => finish(null), INPUT_WAIT_MS);
    process.stdin.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_INPUT_BYTES) {
        finish(null);
        process.stdin.destroy();
        return;
      }
      chunks.push(chunk);
    });
    process.stdin.on('end', () => {
      try {
        finish(JSON.parse(Buffer.concat(chunks).toString('utf8') || 'null'));
      } catch {
        finish(null);
      }
    });
    process.stdin.on('error', () => finish(null));
  });
}

let output = '';
try {
  const input = await readInput();
  if (input && typeof input === 'object') output = await runHook(event, input);
} catch {
  output = '';
}
// The answer first; then the start of the flusher that waits for idle chats
// (armIdle) and the background sender's start (on Windows it waits for
// PowerShell), within the hard stop.
const done = () => {
  armIdle();
  return settleLaunches(HARD_STOP_MS - (Date.now() - STARTED) - 300).finally(() => process.exit(0));
};
if (output) process.stdout.write(output, done);
else done();
