#!/usr/bin/env node
// `quotemem setup` questions on a terminal (a fake TTY with setRawMode) and on
// piped lines, and the lines that sum up a failed check. Nothing here starts
// a process or calls a provider.
//
//   node --test integrations/cli/test/setup-input.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough, Writable } from 'node:stream';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const { checkFailure, cmdSetup, prompter } = await import(pathToFileURL(path.join(here, '..', 'service.mjs')).href);

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/** A writable that keeps what it was given; `isTTY` as asked. */
function collect(isTTY) {
  const stream = new Writable({ write(chunk, _encoding, done) { stream.text += chunk; done(); } });
  stream.text = '';
  stream.isTTY = isTTY;
  stream.columns = 80;
  return stream;
}

/** A terminal's input: raw mode is recorded in `modes`. */
function terminal() {
  const input = new PassThrough();
  input.isTTY = true;
  input.modes = [];
  input.setRawMode = (on) => { input.modes.push(on); return input; };
  return input;
}

const BS = '\b \b';

test('a key shows one * per character; Backspace and Ctrl-U take them back; the key is never echoed', async () => {
  const input = terminal();
  const output = collect(true);
  const ask = prompter(input, output, collect(true));
  assert.deepEqual(input.modes, [true], 'raw mode, so the terminal echoes nothing itself');
  const answer = ask.ask('key: ', { secret: true });
  for (const ch of 'sk-abc') {
    input.write(ch);
    await sleep(2);
  }
  input.write('\x7f\x7f');
  input.write('de');
  input.write('\r');
  assert.equal(await answer, 'sk-ade');
  assert.equal(output.text, `key: ******${BS}${BS}**\n`);

  const cleared = ask.ask('key: ', { secret: true });
  input.write('wrong\x15sk-right\b\r');
  assert.equal(await cleared, 'sk-righ', 'Ctrl-U clears the answer, \\b removes one character too');
  assert.ok(output.text.endsWith(`key: *****${BS.repeat(5)}********${BS}\n`), JSON.stringify(output.text));
  assert.doesNotMatch(output.text, /sk-|wrong|right/);
  ask.close();
  assert.deepEqual(input.modes, [true, false], 'close gives the terminal back');
});

test('a paste shows one * per character and the paste brackets and arrow keys are ignored', async () => {
  const input = terminal();
  const output = collect(true);
  const ask = prompter(input, output, null);
  const pasted = ask.ask('key: ', { secret: true });
  input.write('\x1b[200~sk-pasted-5555\x1b[201~\r');
  assert.equal(await pasted, 'sk-pasted-5555');
  assert.equal(output.text, `key: ${'*'.repeat('sk-pasted-5555'.length)}\n`);

  // A plain answer is echoed as typed; an arrow key (split across writes) does nothing.
  const plain = ask.ask('url: ');
  input.write('ab\x1b[');
  await sleep(5);
  input.write('Dc\x1bOA\r');
  assert.equal(await plain, 'abc');
  assert.match(output.text, /url: abc\n$/);
  ask.close();
});

test('Enter is \\r, \\n or \\r\\n; answers typed ahead wait for their question', async () => {
  const input = terminal();
  const output = collect(true);
  const ask = prompter(input, output, null);
  input.write('1\r\n2\n');
  await sleep(5);
  assert.equal(output.text, '', 'nothing is echoed before its question');
  assert.equal(await ask.ask('a: '), '1');
  assert.equal(await ask.ask('b: '), '2');
  const third = ask.ask('c: ', { secret: true });
  input.write('k\r');
  assert.equal(await third, 'k');
  assert.equal(output.text, 'a: 1\nb: 2\nc: *\n');
  ask.close();
});

test('Ctrl-C ends the questions as ABORT and Ctrl-D on an empty answer as EOF', async () => {
  const input = terminal();
  const output = collect(true);
  const ask = prompter(input, output, null);
  const key = ask.ask('key: ', { secret: true });
  input.write('sk-half\x03');
  await assert.rejects(key, { code: 'ABORT' });
  await assert.rejects(ask.ask('next: '), { code: 'ABORT' }, 'nothing more is asked');
  assert.doesNotMatch(output.text, /sk-half|next/);
  ask.close();
  assert.deepEqual(input.modes, [true, false]);

  const other = terminal();
  const asks = prompter(other, collect(true), null);
  const typed = asks.ask('a: ');
  other.write('x\x04\x7f\x04');
  await assert.rejects(typed, { code: 'EOF' }, 'Ctrl-D counts only on an empty answer');
  asks.close();

  const ended = terminal();
  const lastAsk = prompter(ended, collect(true), null);
  const waiting = lastAsk.ask('a: ');
  ended.end();
  await assert.rejects(waiting, { code: 'EOF' });
  lastAsk.close();
});

test('when only stdin is a terminal the echo goes to stderr; piped answers are not echoed', async () => {
  const input = terminal();
  const output = collect(false); // `setup | tee`
  const echo = collect(true);
  const ask = prompter(input, output, echo);
  const answer = ask.ask('key: ', { secret: true });
  input.write('sk-tee\r');
  assert.equal(await answer, 'sk-tee');
  assert.equal(output.text, 'key: ');
  assert.equal(echo.text, '******\n');
  ask.close();

  const piped = new PassThrough();
  const out = collect(false);
  const lines = prompter(piped, out, collect(true));
  piped.write('sk-piped-1\nsecond\n');
  assert.equal(await lines.ask('key: ', { secret: true }), 'sk-piped-1');
  assert.equal(await lines.ask('a: '), 'second');
  piped.end();
  await assert.rejects(lines.ask('b: '), { code: 'EOF' });
  assert.equal(out.text, 'key: a: b: ', 'no echo and no * when the input is not a terminal');
  lines.close();
});

test('setup: Ctrl-C at the key cancels, saves nothing and gives the terminal back', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'qm-setup-input-'));
  try {
    const input = terminal();
    const output = collect(true);
    const stderr = [];
    const write = process.stderr.write;
    process.stderr.write = (text) => { stderr.push(String(text)); return true; };
    let code;
    try {
      const done = cmdSetup({ 'no-check': true }, { env: { HOME: home, USERPROFILE: home, PATH: process.env.PATH }, input, output, echo: output, release: '0' });
      for (const step of ['\r', '\r', 'sk-cancel', '\x03']) {
        await sleep(20);
        input.write(step);
      }
      code = await done;
    } finally {
      process.stderr.write = write;
    }
    assert.equal(code, 130);
    assert.match(stderr.join(''), /cancelled; nothing was saved/);
    assert.equal(fs.existsSync(path.join(home, '.quotemem', 'service.env')), false);
    assert.deepEqual(input.modes, [true, false]);
    assert.match(output.text, /Model Studio API key \(bailian\.console\.aliyun\.com, China region\): \*{9}\n/);
    assert.doesNotMatch(output.text, /sk-cancel/);
  } finally {
    fs.rmSync(home, { recursive: true, force: true });
  }
});

test('a failed check is two short lines: what failed, with the provider\'s code, and one hint', () => {
  const chat = (code, provider_code, hint = 'The chat provider rejected the request; check QUOTEMEM_CHAT_MODEL') => ({ status: 'checked', ok: false, role: 'chat', code, provider_code, hint });
  assert.deepEqual(checkFailure(chat('http_400', 'Arrearage'), 1, 'default'), [
    'Check failed: chat model, HTTP 400 Arrearage.',
    'The Alibaba Cloud account is overdue: top it up, then run `quotemem check`.',
  ]);
  const refused = checkFailure({ ok: false, role: 'embedding', code: 'http_401', provider_code: 'invalid_api_key', hint: 'x' }, 1, 'fast');
  assert.equal(refused[0], 'Check failed: embedding model, HTTP 401 invalid_api_key.');
  assert.match(refused[1], /Model Studio key \(bailian\.console\.aliyun\.com, China region\), not a DeepSeek key/);
  // A release whose report has no provider code: the HTTP status alone.
  assert.equal(checkFailure(chat('http_403'), 1, 'default')[0], 'Check failed: chat model, HTTP 403.');
  assert.deepEqual(checkFailure(chat('http_401'), 1, null), ['Check failed: chat model, HTTP 401.', 'The chat key was refused. Run `quotemem setup` again with a working key.']);
  assert.match(checkFailure(chat('http_429'), 1, 'default')[1], /rate limiting or out of quota/);
  assert.deepEqual(checkFailure(chat('ConnectError', null, 'The chat provider is unreachable; check QUOTEMEM_CHAT_BASE_URL'), 1, null),
    ['Check failed: chat model, ConnectError.', 'The chat provider is unreachable; check QUOTEMEM_CHAT_BASE_URL; the settings are saved.']);
  // A refused configuration (the check's error line) and no report at all.
  assert.deepEqual(checkFailure({ status: 'failed', code: 'schema_invalid', message: 'QUOTEMEM_CHAT_BASE_URL must be an https URL' }, 1, null),
    ['Check failed: schema_invalid.', 'QUOTEMEM_CHAT_BASE_URL must be an https URL']);
  assert.deepEqual(checkFailure(null, 3, 'default'), ['Check failed: `quotemem check` exited with 3.', 'Run `quotemem check` to see why; the settings are saved.']);
  // Only plain codes are shown.
  assert.equal(checkFailure(chat('http_400', 'two words\nline'), 1, 'default')[0], 'Check failed: chat model, HTTP 400.');
});
