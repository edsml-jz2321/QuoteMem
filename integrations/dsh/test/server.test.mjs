#!/usr/bin/env node
// The QuoteMem MCP server (server.mjs) when the service is down, and its
// forget instructions next to an automatic-memory adapter.
//
//   node --test integrations/dsh/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { TOKEN, plan, stubServer, tempDir, waitFor } from '../../core/test/helpers.mjs';
import { ownerKey } from '../../core/index.mjs';

const SERVER = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server.mjs');
// A home folder with no QuoteMem files: the server reads no token from the
// real home's service data folder.
const EMPTY_HOME = tempDir('qm-server-home-');

function startServer(env) {
  return startServerIn(undefined, env);
}

/** The server with `home` as its HOME and working folder (default: a home with no QuoteMem files). */
function startServerIn(home, env) {
  const child = spawn(process.execPath, [SERVER], {
    env: { PATH: process.env.PATH, HOME: home ?? EMPTY_HOME, USERPROFILE: home ?? EMPTY_HOME, ...(home ? { TMPDIR: home } : {}), ...env },
    cwd: home,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let out = '';
  let err = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { err += d; });
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  const reply = async (id) => {
    await waitFor(() => out.split('\n').some((line) => line.includes(`"id":${id},`)), 5000, `reply ${id}`);
    return JSON.parse(out.split('\n').find((line) => line.includes(`"id":${id},`)));
  };
  const stop = async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const exited = new Promise((resolve) => child.once('exit', resolve));
    child.stdin.end();
    await exited;
  };
  return { send, reply, stop, err: () => err, out: () => out };
}

test('service down: the model gets a short note, no address and no start command', async () => {
  const down = await stubServer({});
  const url = down.url;
  await down.close();
  const mcp = startServer({ QUOTEMEM_URL: url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: '1' });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
  const search = (await mcp.reply(2)).result;
  assert.equal(search.isError, true);
  const text = search.content[0].text;
  assert.equal(text, 'Long-term memory is temporarily unavailable. Continue without it, and do not mention this unless the user asks about memory.');
  const port = new URL(url).port;
  assert.ok(!text.includes(port) && !text.includes('127.0.0.1') && !/uv run|uvx|quotemem serve/.test(text), 'no address or command for the model to repeat');
  mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_health', arguments: {} } });
  const healthText = (await mcp.reply(3)).result.content[0].text;
  const health = JSON.parse(healthText);
  assert.equal(health.status, 'unreachable');
  assert.match(health.next, /`npx quotemem doctor`/);
  assert.ok(!healthText.includes(port) && !healthText.includes('127.0.0.1'), 'an explicit health check names no address either');
  await mcp.stop();
  assert.match(mcp.err(), new RegExp(`not reachable at ${url.replace(/[.]/g, '\\.')} \\(ECONNREFUSED\\)`), 'the details go to stderr, for the host log');
  assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
});

test('auth: QUOTEMEM_TOKEN, then QUOTEMEM_HTTP_AUTH_SECRET; QUOTEMEM_HTTP_AUTH_MODE picks the header', async () => {
  const server = await stubServer({ '/memory/search': () => ({ body: { data: [] } }) });
  const searchWith = async (env) => {
    // The service on this computer proves the token it holds: the one this call uses.
    server.setToken(env.QUOTEMEM_TOKEN || env.QUOTEMEM_HTTP_AUTH_SECRET || TOKEN);
    const mcp = startServer({ QUOTEMEM_URL: server.url, ...env });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    await mcp.reply(1);
    const before = server.of('/memory/search').length;
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
    const result = (await mcp.reply(2)).result;
    await mcp.stop();
    return { result, sent: server.of('/memory/search').slice(before), err: mcp.err() };
  };
  const both = await searchWith({ QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_SECRET: 'the-secret-2' });
  assert.equal(both.sent[0].headers.authorization, `Bearer ${TOKEN}`);
  const secret = await searchWith({ QUOTEMEM_HTTP_AUTH_SECRET: 'the-secret-2', QUOTEMEM_HTTP_AUTH_MODE: 'x-api-key' });
  assert.equal(secret.sent[0].headers['x-api-key'], 'the-secret-2');
  assert.equal(secret.sent[0].headers.authorization, undefined);
  const mode = await searchWith({ QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_MODE: 'token' });
  assert.equal(mode.sent[0].headers.authorization, `Token ${TOKEN}`);
  const oldMode = await searchWith({ QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTH_MODE: 'x-api-key' });
  assert.equal(oldMode.sent[0].headers.authorization, `Bearer ${TOKEN}`, 'the old QUOTEMEM_AUTH_MODE is not read');
  const old = await searchWith({ QUOTEMEM_SECRET: 'old-secret-3' });
  assert.equal(old.sent.length, 0, 'no token: no search is sent');
  assert.equal(old.result.isError, true);
  assert.match(old.err, /No token and no memory service set up: run `npx quotemem setup` once/);
  const badMode = await searchWith({ QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_MODE: 'basic' });
  assert.equal(badMode.sent.length, 0);
  assert.match(badMode.err, /QUOTEMEM_HTTP_AUTH_MODE must be bearer, token or x-api-key/);
  for (const run of [both, secret, mode, oldMode, old, badMode]) {
    assert.ok(![TOKEN, 'the-secret-2', 'old-secret-3'].some((t) => run.err.includes(t) || JSON.stringify(run.result).includes(t)), 'no token in the output');
  }
  await server.close();
});

test('token from a file: QUOTEMEM_TOKEN_FILE, else the local service\'s own <data-dir>/token; never printed', async (t) => {
  const server = await stubServer({ '/memory/search': () => ({ body: { data: [] } }) });
  t.after(() => server.close());
  const home = tempDir('qm-server-tokens-');
  const write = (rel, text) => {
    const file = path.join(home, rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, text, { mode: 0o600 });
  };
  write('.local/share/quotemem/token', 'service-default-token\n');
  write('xdg/quotemem/token', 'service-xdg-token\n');
  write('data/token', 'service-dir-token\n');
  write('.quotemem/token', 'named-file-token\n');
  const searchWith = async (env, holds) => {
    // The service on this computer proves the token it holds: the one this call should send.
    server.setToken(holds);
    const mcp = startServer({ HOME: home, USERPROFILE: home, QUOTEMEM_URL: server.url, ...env });
    t.after(() => mcp.stop());
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    await mcp.reply(1);
    const before = server.of('/memory/search').length;
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
    const result = (await mcp.reply(2)).result;
    await mcp.stop();
    return { result, sent: server.of('/memory/search').slice(before), all: `${mcp.out()}${mcp.err()}` };
  };
  // No token in the environment at all: the service's own token file.
  const local = await searchWith({}, 'service-default-token');
  assert.equal(local.sent[0]?.headers.authorization, 'Bearer service-default-token');
  const xdg = await searchWith({ XDG_DATA_HOME: path.join(home, 'xdg') }, 'service-xdg-token');
  assert.equal(xdg.sent[0]?.headers.authorization, 'Bearer service-xdg-token');
  const dir = await searchWith({ QUOTEMEM_DATA_DIR: '~/data', XDG_DATA_HOME: path.join(home, 'xdg') }, 'service-dir-token');
  assert.equal(dir.sent[0]?.headers.authorization, 'Bearer service-dir-token');
  // A named file wins over the service's file; QUOTEMEM_TOKEN still wins over both.
  const named = await searchWith({ QUOTEMEM_TOKEN_FILE: '~/.quotemem/token' }, 'named-file-token');
  assert.equal(named.sent[0]?.headers.authorization, 'Bearer named-file-token');
  const direct = await searchWith({ QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_TOKEN_FILE: '~/.quotemem/token' }, TOKEN);
  assert.equal(direct.sent[0]?.headers.authorization, `Bearer ${TOKEN}`);
  // A named file that cannot be read: nothing is sent, and the service's file is not used instead.
  const missing = await searchWith({ QUOTEMEM_TOKEN_FILE: '~/no-such-file' }, TOKEN);
  assert.equal(missing.sent.length, 0);
  assert.match(missing.all, /QUOTEMEM_TOKEN_FILE cannot be read/);
  for (const run of [local, xdg, dir, named, direct, missing]) {
    assert.ok(!['service-default-token', 'service-xdg-token', 'service-dir-token', 'named-file-token', TOKEN].some((token) => run.all.includes(token)), 'no token in the output');
  }
  // The local service's file is for a service on this computer only: it is
  // never sent to another host.
  const remote = startServer({ HOME: home, USERPROFILE: home, QUOTEMEM_URL: 'https://memory.invalid' });
  t.after(() => remote.stop());
  remote.send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'x' } } });
  const refused = (await remote.reply(1)).result;
  await remote.stop();
  assert.equal(refused.isError, true);
  assert.equal(refused.content[0].text, 'Long-term memory is not set up correctly. Continue without it; the user can run `npx quotemem doctor` to check the setup.');
  assert.match(remote.err(), /No token: for a service on another computer/, 'what to change goes to stderr');
});

test('one user only: no tool takes a user argument, and a call that names one is refused before anything is sent', async (t) => {
  const seen = [];
  const server = await stubServer({
    '/memory/search': (body) => { seen.push(body); return { body: { data: [] } }; },
    '/memory/history': (body) => { seen.push(body); return { body: { data: [] } }; },
  });
  t.after(() => server.close());
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'qm-alice' });
  t.after(() => mcp.stop());
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
  const tools = (await mcp.reply(2)).result.tools;
  assert.ok(tools.length >= 6);
  const identity = /^(?:user|owner|tenant|account|principal)(?:[_-]?ids?)?$/i;
  for (const tool of tools) {
    const names = Object.keys(tool.inputSchema.properties ?? {});
    assert.ok(!names.some((name) => identity.test(name)), `${tool.name} takes no user argument (${names.join(', ')})`);
    assert.ok(!JSON.stringify(tool).includes('QUOTEMEM_USER_ID'), `${tool.name} does not name the owner setting`);
    assert.equal(tool.inputSchema.additionalProperties, false, `${tool.name} accepts no other argument`);
  }
  for (const name of ['quotemem_provenance', 'quotemem_memories']) assert.ok(tools.some((tool) => tool.name === name), `${name} is listed, without a user argument`);
  const calls = [
    ['quotemem_search', { query: 'where does the other user live', user_id: 'qm-claude' }],
    ['quotemem_search', { query: 'where does the other user live', userId: 'qm-claude' }],
    ['quotemem_history', { memory_id: '01JX00000000000000000000F1', owner: 'qm-claude' }],
    ['quotemem_provenance', { memory_id: '01JX00000000000000000000F1', user_id: 'qm-alice' }],
    ['quotemem_memories', { status: 'all', user_id: 'qm-claude' }],
    ['quotemem_memories', { owner_id: 'qm-claude' }],
    ['quotemem_forget_prepare', { statement: 'I live in Lisbon', owner_id: 'qm-claude' }],
    ['quotemem_add', { messages: [{ role: 'user', content: 'I live in Lisbon.' }], user_id: 'qm-claude' }],
  ];
  let id = 10;
  for (const [name, args] of calls) {
    id += 1;
    mcp.send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
    const result = (await mcp.reply(id)).result;
    assert.equal(result.isError, true, `${name} with ${Object.keys(args).join(', ')}`);
    assert.match(result.content[0].text, /acts only on this user's own memory, so (user_id|userId|owner|owner_id) is not accepted\. Call the tool again without it\./);
  }
  assert.equal(server.requests.length, 0, 'nothing was sent for a call that names a user');
  // Without one, every call acts on the configured user.
  mcp.send({ jsonrpc: '2.0', id: 30, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
  assert.equal((await mcp.reply(30)).result.isError, false);
  mcp.send({ jsonrpc: '2.0', id: 31, method: 'tools/call', params: { name: 'quotemem_history', arguments: { memory_id: '01JX00000000000000000000F1' } } });
  await mcp.reply(31);
  await mcp.stop();
  assert.deepEqual(seen.map((body) => body.user_id), ['qm-alice', 'qm-alice']);
});

test('forget instructions: next to an adapter, the automatic note wins and the request is the consent', async () => {
  const server = await stubServer({});
  const listed = async (env) => {
    const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, ...env });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    const init = (await mcp.reply(1)).result;
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
    const tools = (await mcp.reply(2)).result.tools;
    await mcp.stop();
    return { instructions: init.instructions, prepare: tools.find((t) => t.name === 'quotemem_forget_prepare').description };
  };
  const auto = await listed({ QUOTEMEM_AUTO: '1' });
  assert.match(auto.instructions, /forgetting are automatic/);
  assert.match(auto.instructions, /\[QuoteMem: automatic forget\]" note is present, follow it and do not call the forget tools/);
  assert.match(auto.prepare, /no "\[QuoteMem: automatic forget\]" note says it was handled/);
  assert.doesNotMatch(auto.prepare, /only after the user agrees/, 'no second confirmation the automatic forget never asks for');
  const plain = await listed({});
  assert.match(plain.prepare, /call quotemem_forget_confirm only after the user agrees/);
  await server.close();
});

test('history and provenance of an id that names no memory: the service message reaches the model', async () => {
  const message = 'No memory with this id belongs to this user. History and Provenance take the id of a memory.';
  const notFound = () => ({ status: 404, body: { code: 'memory_not_found', retryable: false, detail: 'memory_not_found', message } });
  const server = await stubServer({ '/memory/history': notFound, '/memory/provenance': notFound, '/memory/search': () => ({ body: { data: [] } }) });
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  const id = '01JX00000000000000000000E1';
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_history', arguments: { memory_id: id } } });
  mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_provenance', arguments: { memory_id: id } } });
  for (const n of [2, 3]) {
    const result = (await mcp.reply(n)).result;
    assert.equal(result.isError, true);
    assert.equal(result.content[0].text, `QuoteMem returned 404 memory_not_found. ${message}`);
  }
  // Without a message (an older service), a short hint names the cause.
  server.handlers['/memory/history'] = () => ({ status: 404, body: { code: 'memory_not_found' } });
  mcp.send({ jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'quotemem_history', arguments: { memory_id: id } } });
  assert.match((await mcp.reply(4)).result.content[0].text, /^QuoteMem returned 404 memory_not_found\. No memory with this id belongs to this user\. History and Provenance take the id/);
  // Another code's message is not passed through.
  server.handlers['/memory/history'] = () => ({ status: 422, body: { code: 'invalid_request', message: 'internal detail' } });
  mcp.send({ jsonrpc: '2.0', id: 5, method: 'tools/call', params: { name: 'quotemem_history', arguments: { memory_id: id } } });
  assert.equal((await mcp.reply(5)).result.content[0].text, 'QuoteMem returned 422 invalid_request. The service refused the request fields.');
  // top_k follows the agent service: default 10, at most 20.
  mcp.send({ jsonrpc: '2.0', id: 6, method: 'tools/list' });
  const search = (await mcp.reply(6)).result.tools.find((t) => t.name === 'quotemem_search');
  assert.equal(search.inputSchema.properties.top_k.default, 10);
  assert.equal(search.inputSchema.properties.top_k.maximum, 20);
  mcp.send({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
  await mcp.reply(7);
  assert.equal(server.of('/memory/search')[0].body.top_k, 10);
  mcp.send({ jsonrpc: '2.0', id: 8, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live', top_k: 21 } } });
  assert.match((await mcp.reply(8)).result.content[0].text, /top_k must be an integer from 1 to 20/);
  await mcp.stop();
  assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
  await server.close();
});

test('forget tools: look-alikes kept, bundled memories named, sentences counted, only verified reported as forgotten', async () => {
  const M = (n) => `01JX0000000000000000000M${n}`;
  const BUNDLE = 'The user is learning Japanese and plays tennis on Sundays.';
  let plan = null;
  const server = await stubServer({
    '/memory/forget/prepare': (body) => {
      if (body.statement.includes('sister')) {
        return { body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, decision: null,
          candidates: [{ memory_id: M(4), revision_id: M(4), statement: "The user's sister lives in Berlin.", role: 'states', target: false, excluded: 'other_subject' }] } };
      }
      plan = { schema_version: 1, user_id: body.user_id, client_session_id: body.client_session_id, request_id: body.request_id,
        targets: [M(1), M(3)].map((id) => ({ memory_id: id, expected_revision_id: id, expected_content_hash: `sha256:${'a'.repeat(64)}` })),
        candidates: [M(1), M(2), M(3)].map((id) => ({ memory_id: id, revision_id: id })), erase_epoch: 1, expires_at: Date.now() + 600_000,
        plan_binding: 'b'.repeat(64), bundled: [M(1)], statement: body.statement, spans: [{ evidence_id: M(9), byte_start: 0, byte_end: 14 }] };
      return { body: { schema_version: 1, status: 'confirmation_required', confirmation: plan, expires_at: plan.expires_at, decision: null,
        candidates: [
          { memory_id: M(1), revision_id: M(1), statement: BUNDLE, role: 'states', target: true, bundled: true },
          { memory_id: M(2), revision_id: M(2), statement: "The user's brother plays tennis.", role: 'states', target: false, excluded: 'other_subject' },
          { memory_id: M(3), revision_id: M(3), statement: 'The user plays tennis.', role: 'states', target: true },
        ],
        spans: [{ evidence_id: M(9), byte_start: 0, byte_end: 14, text: 'I play tennis.', role: 'states', target: true }] } };
    },
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', scope_state: 'ready', outcomes: [
      { memory_id: M(1), outcome: 'verified', receipt_ref: 'r1' },
      { memory_id: M(3), outcome: 'forgotten_earlier' },
      { memory_id: null, withheld_spans: 2, outcome: 'verified', receipt_ref: 'r2' },
    ] } }),
  });
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  const callTool = async (id, name, args) => {
    mcp.send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
    const result = (await mcp.reply(id)).result;
    assert.ok(!result.isError, result.content[0].text);
    return JSON.parse(result.content[0].text);
  };
  const prepared = await callTool(2, 'quotemem_forget_prepare', { statement: 'I play tennis' });
  assert.deepEqual(prepared.to_forget.map((c) => [c.memory_id, c.bundled ?? false]), [[M(1), true], [M(3), false]]);
  assert.deepEqual(prepared.kept, [{ memory_id: M(2), statement: "The user's brother plays tennis.", kept_because: 'about another person or animal' }]);
  assert.deepEqual(prepared.other_candidates, [], 'a kept look-alike is not offered as a candidate');
  assert.deepEqual(prepared.sentences_to_forget, ['I play tennis.']);
  assert.match(prepared.next, /marked bundled also holds other details and is forgotten whole/);
  assert.match(prepared.retention_notice, /if the user later states it again themselves, it may be stored again/);
  const confirmed = await callTool(3, 'quotemem_forget_confirm', { token: prepared.token });
  assert.deepEqual(server.of('/memory/forget/confirm')[0].body, plan, 'confirmed exactly as prepared');
  assert.deepEqual([confirmed.memories_forgotten, confirmed.sentences_forgotten, confirmed.already_gone, confirmed.changed], [1, 2, 1, 0]);
  assert.deepEqual(confirmed.forgotten_whole, [BUNDLE]);
  assert.equal(confirmed.outcomes[2].withheld_spans, 2);
  assert.match(confirmed.next, /^Done\. .*also held other details/);

  const nothing = await callTool(4, 'quotemem_forget_prepare', { statement: 'my sister lives in Berlin' });
  assert.equal(nothing.status, 'nothing_matched');
  assert.equal(nothing.token, null);
  assert.deepEqual(nothing.candidates, []);
  assert.equal(nothing.kept[0].kept_because, 'about another person or animal');
  assert.match(nothing.next, /nothing will be forgotten.*do not forget them/);

  // Everything already gone: never reported as forgotten by this call.
  server.handlers['/memory/forget/prepare'] = (body) => ({ body: { schema_version: 1, status: 'confirmation_required', decision: null, candidates: [],
    confirmation: { ...plan, request_id: body.request_id, bundled: undefined, statement: undefined, spans: undefined }, expires_at: plan.expires_at } });
  server.handlers['/memory/forget/confirm'] = () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', scope_state: 'ready',
    outcomes: [{ memory_id: M(1), outcome: 'forgotten_earlier' }] } });
  const again = await callTool(5, 'quotemem_forget_prepare', { statement: 'I play tennis' });
  const gone = await callTool(6, 'quotemem_forget_confirm', { token: again.token });
  assert.equal(gone.memories_forgotten, 0);
  assert.equal(gone.already_gone, 1);
  assert.match(gone.next, /^It was already forgotten/);
  await mcp.stop();
  assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
  await server.close();
});

test('forget confirm: a plan confirmed again after an error reports what its first confirmation forgot, once', async () => {
  const M = (n) => `01JX0000000000000000000M${n}`;
  const BUNDLE = 'The user is learning Japanese and plays tennis on Sundays.';
  let confirms = 0;
  const server = await stubServer({
    '/memory/forget/prepare': (body) => {
      const confirmation = { schema_version: 1, user_id: body.user_id, client_session_id: body.client_session_id, request_id: body.request_id,
        targets: [{ memory_id: M(1), expected_revision_id: M(1), expected_content_hash: `sha256:${'a'.repeat(64)}` }],
        candidates: [{ memory_id: M(1), revision_id: M(1) }], erase_epoch: 1, expires_at: Date.now() + 600_000, plan_binding: 'b'.repeat(64) };
      return { body: { schema_version: 1, status: 'confirmation_required', confirmation, expires_at: confirmation.expires_at, decision: null,
        candidates: [{ memory_id: M(1), revision_id: M(1), statement: BUNDLE, role: 'states', target: true, bundled: true }],
        spans: [{ evidence_id: M(9), byte_start: 0, byte_end: 14, text: 'I play tennis.', role: 'states', target: true }] } };
    },
    // The first confirmation forgets everything, then fails before it answers.
    '/memory/forget/confirm': () => {
      confirms += 1;
      if (confirms === 1) return { status: 500, body: { code: 'internal_error' } };
      return { body: { schema_version: 1, status: 'verified', request_digest: 'x', scope_state: 'ready', outcomes: [
        { memory_id: M(1), outcome: 'forgotten_earlier' },
        { memory_id: null, outcome: 'forgotten_earlier' },
      ] } };
    },
  });
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  const call = async (id, name, args) => {
    mcp.send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
    return (await mcp.reply(id)).result;
  };
  const prepared = JSON.parse((await call(2, 'quotemem_forget_prepare', { statement: 'I play tennis' })).content[0].text);
  const failed = await call(3, 'quotemem_forget_confirm', { token: prepared.token });
  assert.equal(failed.isError, true);
  const again = JSON.parse((await call(4, 'quotemem_forget_confirm', { token: prepared.token })).content[0].text);
  assert.deepEqual([again.memories_forgotten, again.sentences_forgotten, again.already_gone], [1, 1, 0], 'forgotten for this request, not before it');
  assert.deepEqual(again.forgotten_whole, [BUNDLE]);
  assert.match(again.next, /^Done\. .*forgotten_whole also held other details/);
  assert.doesNotMatch(again.next, /already forgotten/);
  await mcp.stop();
  assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
  await server.close();
});

test('health: the agent service\'s background completions and spend cap are shown; an older service shows none', async () => {
  const server = await stubServer({
    '/health': () => ({ body: { status: 'ok' } }),
    '/agent/status': (_body, entry) => (entry.headers.authorization === `Bearer ${TOKEN}`
      ? { body: { service: 'quotemem', profile: 'agent', adds: { in_flight: 1, completing: 2, deferred: 0, refused_deleting: 0 },
        spend: { exhausted: false, resets_at_ms: Date.now() + 3_600_000, spent_usd: 1.5 } } }
      : { status: 401, body: { code: 'unauthorized' } }),
  });
  const health = async (env) => {
    const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, ...env });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    await mcp.reply(1);
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_health', arguments: {} } });
    const value = JSON.parse((await mcp.reply(2)).result.content[0].text);
    await mcp.stop();
    assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
    return value;
  };
  const busy = await health({});
  assert.equal(busy.status, 'ok');
  assert.deepEqual(busy.adds, { in_flight: 1, completing: 2, deferred: 0 });
  assert.equal(busy.spend_cap_reached, false);
  assert.match(busy.next, /still being stored/);
  assert.ok(!('spent_usd' in busy), 'counts only');
  delete server.handlers['/agent/status'];
  const older = await health({});
  assert.equal(older.status, 'ok');
  assert.equal(older.adds, undefined);
  assert.equal(older.next, undefined);
  await server.close();
});

test('health: the listener check decides it, as for every other tool; a refused program is never "ok"', async (t) => {
  const UNAVAILABLE = 'Long-term memory is temporarily unavailable. Continue without it, and do not mention this unless the user asks about memory.';
  const server = await stubServer({ '/health': () => ({ body: { status: 'ok' } }), '/memory/search': () => ({ body: { results: [] } }) });
  t.after(() => server.close());
  const tools = async () => {
    const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTOSTART: '0' });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    await mcp.reply(1);
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_health', arguments: {} } });
    const health = (await mcp.reply(2)).result;
    mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'where do I live' } } });
    const search = (await mcp.reply(3)).result;
    await mcp.stop();
    assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
    return { health, value: JSON.parse(health.content[0].text), search: search.content[0].text, err: mcp.err() };
  };
  // The user's own service passes: ok.
  assert.equal((await tools()).value.status, 'ok');
  assert.equal(server.of('/health').length, 1);
  server.requests.length = 0;

  // A service from before the check (0.3.2 answers the check route with 422),
  // started by hand: memory is off, with the same words as the other tools.
  server.handlers['/agent/listener'] = () => ({ status: 422, body: { code: 'invalid_request' } });
  const older = await tools();
  assert.equal(older.health.isError, false);
  assert.equal(older.value.status, 'memory_off');
  assert.match(older.value.next, /^Long-term memory is off: the program on the memory service's port was not started by the agents/);
  assert.equal(older.value.next, older.search);
  assert.match(older.err, /no token and no memory data were sent to .*does not answer the check \(422\)/);
  assert.deepEqual([...new Set(server.requests.map((r) => r.route))], ['/agent/listener'], 'nothing but the check');
  server.requests.length = 0;

  // Another program that answers the check without this user's token.
  delete server.handlers['/agent/listener'];
  server.setToken('not-the-token-'.padEnd(43, 'x'));
  const foreign = await tools();
  assert.equal(foreign.value.status, 'memory_off');
  assert.match(foreign.value.next, /^Long-term memory is off: another program holds the memory service's port/);
  assert.match(foreign.value.next, /`npx quotemem doctor`/);
  assert.equal(foreign.search, foreign.value.next, 'search says the port is held, not "temporarily unavailable"');
  assert.ok(!foreign.value.next.includes(String(server.port)));
  assert.deepEqual(server.requests, [], 'nothing was sent');
  server.setToken(TOKEN);

  // The user's own service that does not answer the check (frozen): unavailable.
  server.handlers['/agent/listener'] = () => new Promise(() => {});
  const stalled = await tools();
  assert.equal(stalled.value.status, 'unavailable');
  assert.equal(stalled.value.next, UNAVAILABLE);
  assert.equal(stalled.search, UNAVAILABLE);
  assert.ok(!/another program/i.test(stalled.value.next));
  assert.deepEqual([...new Set(server.requests.map((r) => r.route))], ['/agent/listener'], 'nothing but the check');
});

async function initialized(env) {
  const mcp = startServer(env);
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  return mcp;
}

test('memories: listed read-only next to an adapter and alone; the DSH bundle names every tool', async () => {
  const server = await stubServer({});
  const names = [];
  for (const env of [{ QUOTEMEM_AUTO: '1' }, {}]) {
    const mcp = await initialized({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, ...env });
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
    const tools = (await mcp.reply(2)).result.tools;
    await mcp.stop();
    const list = tools.find((t) => t.name === 'quotemem_memories');
    assert.ok(list, 'quotemem_memories is listed');
    assert.equal(list.annotations.readOnlyHint, true);
    assert.match(list.description, /only when the user explicitly asks to see or list what you remember/);
    assert.equal(list.inputSchema.properties.status.default, 'current');
    assert.deepEqual(list.inputSchema.properties.status.enum, ['current', 'past', 'possibly_outdated', 'all']);
    assert.equal(list.inputSchema.properties.limit.default, 20);
    assert.equal(list.inputSchema.properties.limit.maximum, 200);
    assert.equal(list.inputSchema.properties.cursor.maxLength, 512, 'the service accepts a cursor of at most 512 characters');
    assert.equal(list.inputSchema.required, undefined, 'every argument is optional');
    names.push(tools.map((t) => t.name));
  }
  const bundle = createRequire(import.meta.url)('../index.cjs');
  assert.deepEqual([...bundle.tools].sort(), [...names[1]].sort(), 'index.cjs lists the tools the server offers');
  assert.equal(server.requests.length, 0, 'listing tools sends nothing');
  await server.close();
});

test('memories: pages through /memory/list, shows the first quote only, and writes nothing to disk', async () => {
  const long = 'x'.repeat(700);
  const pages = {
    null: {
      memories: [
        {
          memory_id: '01JX00000000000000000000M1', statement: 'The user lives in Lisbon.', status: 'current',
          created_at_ms: 1_700_000_000_000, updated_at_ms: 1_700_000_500_000,
          quotes: [{ role: 'user', said_at_ms: 1_700_000_000_000, text: 'I moved to Lisbon last spring.' }, { role: 'user', said_at_ms: null, text: 'Lisbon is home now.' }],
          supersedes: ['01JX00000000000000000000M0'], superseded_by: null,
        },
        {
          memory_id: '01JX00000000000000000000M2', statement: 'The user likes long walks.', status: 'current',
          created_at_ms: 1_690_000_000_000, updated_at_ms: 1_690_000_000_000,
          quotes: [{ role: 'user', said_at_ms: null, text: long }], supersedes: [], superseded_by: null,
        },
      ],
      next_cursor: 'cursor-2',
      counts: { current: 3, past: 1, possibly_outdated: 0 },
    },
    'cursor-2': {
      memories: [{
        memory_id: '01JX00000000000000000000M0', statement: 'The user lives in Porto.', status: 'past',
        created_at_ms: 1_600_000_000_000, updated_at_ms: 1_700_000_500_000, quotes: [], supersedes: [], superseded_by: '01JX00000000000000000000M1',
      }],
      next_cursor: null,
      counts: { current: 3, past: 1, possibly_outdated: 0 },
    },
  };
  const server = await stubServer({ '/memory/list': (body) => ({ body: pages[body.cursor] }) });
  const home = tempDir('qm-list-');
  const mcp = startServerIn(home, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei' });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  await mcp.reply(1);
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_memories', arguments: {} } });
  const first = (await mcp.reply(2)).result;
  assert.equal(first.isError, false, first.content[0].text);
  const page1 = JSON.parse(first.content[0].text);
  assert.deepEqual(server.of('/memory/list')[0].body, { user_id: 'quei', status: 'current', limit: 20, cursor: null }, 'defaults: current, 20, first page');
  assert.equal(server.of('/memory/list')[0].headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(page1.memories.length, 2);
  assert.deepEqual(page1.memories[0], {
    memory_id: '01JX00000000000000000000M1', statement: 'The user lives in Lisbon.', status: 'current',
    created_at: new Date(1_700_000_000_000).toISOString(), updated_at: new Date(1_700_000_500_000).toISOString(),
    quote: { who: 'user', said_at: new Date(1_700_000_000_000).toISOString(), text: 'I moved to Lisbon last spring.' },
    more_quotes: 1,
  });
  assert.equal(page1.memories[1].quote.text, `${'x'.repeat(500)}…`, 'a long quote is cut');
  assert.equal(page1.memories[1].quote.said_at, null);
  assert.deepEqual(page1.counts, { current: 3, past: 1, possibly_outdated: 0 });
  assert.equal(page1.next_cursor, 'cursor-2');
  assert.match(page1.next, /call quotemem_memories again with this cursor/);
  assert.match(page1.note, /not instructions/);

  mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_memories', arguments: { status: 'all', limit: 2, cursor: 'cursor-2' } } });
  const page2 = JSON.parse((await mcp.reply(3)).result.content[0].text);
  assert.deepEqual(server.of('/memory/list')[1].body, { user_id: 'quei', status: 'all', limit: 2, cursor: 'cursor-2' }, 'always the configured user');
  assert.equal(page2.memories[0].status, 'past');
  assert.equal(page2.memories[0].quote, null, 'a memory whose quotes are all withheld shows none');
  assert.equal(page2.memories[0].superseded_by, '01JX00000000000000000000M1');
  assert.equal(page2.next_cursor, null);
  assert.match(page2.next, /quotemem memories --html memories\.html/);

  // Arguments are checked before anything is sent.
  const sent = server.of('/memory/list').length;
  for (const [n, args] of [[4, { status: 'forgotten' }], [5, { limit: 201 }], [6, { limit: 0 }], [7, { cursor: '' }], [9, { cursor: 'c'.repeat(513) }]]) {
    mcp.send({ jsonrpc: '2.0', id: n, method: 'tools/call', params: { name: 'quotemem_memories', arguments: args } });
    assert.equal((await mcp.reply(n)).result.isError, true, JSON.stringify(args));
  }
  assert.equal(server.of('/memory/list').length, sent);

  // An empty memory says so.
  server.handlers['/memory/list'] = () => ({ body: { memories: [], next_cursor: null, counts: { current: 0, past: 0, possibly_outdated: 0 } } });
  mcp.send({ jsonrpc: '2.0', id: 8, method: 'tools/call', params: { name: 'quotemem_memories', arguments: {} } });
  const empty = JSON.parse((await mcp.reply(8)).result.content[0].text);
  assert.deepEqual(empty.memories, []);
  assert.equal(empty.note, 'No memory has status current.');
  await mcp.stop();
  assert.deepEqual(fs.readdirSync(home), [], 'nothing is cached on disk');
  assert.ok(!(mcp.out() + mcp.err()).includes(TOKEN));
  await server.close();
});

test('memories: an older service without /memory/list gets a short update note', async () => {
  const server = await stubServer({});
  const mcp = await initialized({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN });
  // The stub answers an unknown route with 404, as a generic HTTP server does.
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_memories', arguments: {} } });
  const notFound = (await mcp.reply(2)).result;
  assert.equal(notFound.isError, true);
  assert.match(notFound.content[0].text, /^This QuoteMem service cannot list memories yet: it is older than this plugin\. Update it to quotemem \d+\.\d+\.\d+ and restart it; search, history and forget still work\.$/);
  // A QuoteMem service answers an unknown route with 422 invalid_request.
  server.handlers['/memory/list'] = () => ({ status: 422, body: { code: 'invalid_request', retryable: false, detail: 'invalid_request' } });
  mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_memories', arguments: {} } });
  const invalid = (await mcp.reply(3)).result;
  assert.equal(invalid.isError, true);
  assert.match(invalid.content[0].text, /^This QuoteMem service cannot list memories yet.*\(If the service is current, it refused the request fields\.\)$/);
  // With a cursor the route exists (the cursor came from a page it answered):
  // the same 422 means the cursor was refused, not that the service is old.
  mcp.send({ jsonrpc: '2.0', id: 5, method: 'tools/call', params: { name: 'quotemem_memories', arguments: { cursor: 'c'.repeat(512) } } });
  const badCursor = (await mcp.reply(5)).result;
  assert.equal(badCursor.isError, true);
  assert.equal(server.of('/memory/list').at(-1).body.cursor, 'c'.repeat(512), 'a 512-character cursor is sent');
  assert.equal(badCursor.content[0].text, 'QuoteMem refused the cursor: it is not one this service gave. List again without a cursor to start from the newest memory.');
  assert.doesNotMatch(badCursor.content[0].text, /older than this plugin|Update it/);
  // Other failures keep their usual message.
  server.handlers['/memory/list'] = () => ({ status: 401, body: { code: 'authentication_failed', retryable: false, detail: 'authentication_failed' } });
  mcp.send({ jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'quotemem_memories', arguments: {} } });
  assert.match((await mcp.reply(4)).result.content[0].text, /^QuoteMem returned 401 authentication_failed\. The service refused the token\./);
  await mcp.stop();
  assert.match(mcp.err(), /authentication_failed; the token must equal the service's token/, 'what to change goes to stderr');
  await server.close();
});

// What the smoke runs count as noise in a reply the user sees: a raw
// connection error, a service address, a setting name or a command the user
// should not run. Only the names of the tools the model can call (those
// tools/list shows in that mode) may appear.
const NOISE = /ECONNREFUSED|not reachable|无法连接|连不上|拒绝连接|quotemem serve|aml (agent )?serve|QUOTEMEM_|quotemem_|127\.0\.0\.1:\d+|\bhook/i;
const MEMORY_ID = '01HZX3J8Q6V9K2M4N7P1R5T8WY';
const EVERY_TOOL = [
  ['quotemem_search', { query: 'where do I live' }],
  ['quotemem_add', { messages: [{ role: 'user', content: 'I live in Hangzhou.' }] }],
  ['quotemem_history', { memory_id: MEMORY_ID }],
  ['quotemem_provenance', { memory_id: MEMORY_ID }],
  ['quotemem_forget_prepare', { statement: 'I live in Hangzhou' }],
  ['quotemem_forget_confirm', { token: 'no-such-token' }],
  ['quotemem_memories', {}],
  ['quotemem_health', {}],
];

/**
 * Every tool's result text under one environment (a hidden tool is called
 * too), the tools the model can call, and the server's stderr.
 */
async function everyToolResult(env) {
  const mcp = startServer({ QUOTEMEM_AUTOSTART: '0', ...env });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
  const results = [['initialize', JSON.stringify((await mcp.reply(1)).result.instructions)]];
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
  const listed = (await mcp.reply(2)).result.tools.map((tool) => tool.name);
  for (const [index, [name, args]] of EVERY_TOOL.entries()) {
    const id = index + 3;
    mcp.send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
    results.push([name, (await mcp.reply(id)).result.content.map((part) => part.text).join('\n')]);
  }
  await mcp.stop();
  return { results, listed, err: mcp.err(), text: (name) => results.find(([tool]) => tool === name)[1] };
}

function assertQuiet({ results, listed }, what) {
  const callable = new RegExp(`\\b(?:${listed.join('|')})\\b`, 'g');
  for (const [name, text] of results) {
    assert.doesNotMatch(text.replace(callable, ''), NOISE, `${what}: ${name} answered ${text}`);
  }
}

test('no tool result names an address, a setting or a command to run, whatever fails', async (t) => {
  const down = await stubServer({});
  const closed = down.url;
  await down.close();
  for (const auto of ['1', '0']) {
    const run = await everyToolResult({ QUOTEMEM_URL: closed, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: auto });
    assert.equal(run.listed.includes('quotemem_add'), auto === '0');
    assertQuiet(run, `service down, QUOTEMEM_AUTO=${auto}`);
    assert.match(run.err, /not reachable at http:\/\/127\.0\.0\.1:\d+ \(ECONNREFUSED\); run `npx quotemem doctor`/, 'the address goes to stderr');
  }
  // A setting is missing or wrong: the details go to stderr.
  for (const [what, env, logged] of [
    ['no token for a remote service', { QUOTEMEM_URL: 'https://memory.invalid' }, /No token: for a service on another computer/],
    ['a URL that is not http', { QUOTEMEM_URL: 'ftp://127.0.0.1:9', QUOTEMEM_TOKEN: TOKEN }, /QUOTEMEM_URL must be an http/],
    ['a URL with credentials', { QUOTEMEM_URL: 'http://me:pw@127.0.0.1:9', QUOTEMEM_TOKEN: TOKEN }, /QUOTEMEM_URL must not contain credentials/],
    ['an unknown auth mode', { QUOTEMEM_URL: closed, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_MODE: 'basic' }, /QUOTEMEM_HTTP_AUTH_MODE must be/],
  ]) {
    const run = await everyToolResult(env);
    assertQuiet(run, what);
    assert.match(run.err, logged, what);
    assert.match(run.text('quotemem_search'), /`npx quotemem doctor`/, what);
  }
  // A running service that refuses every request with each code it uses.
  const refusals = {
    authentication_failed: 401, scope_denied: 403, invalid_request: 422, operation_conflict: 422,
    extraction_refused: 422, secret_rejected: 422, repair_required: 409, provider_unavailable: 503,
    delivery_busy: 429, index_rebuilding: 503, erasure_in_progress: 409, authorization_unavailable: 503,
    service_unhealthy: 503, memory_not_found: 404, a_code_this_plugin_does_not_know: 500,
  };
  let refusal = null;
  const refuse = () => ({
    status: refusals[refusal], headers: { 'retry-after': '0' },
    body: { code: refusal, retryable: [429, 503].includes(refusals[refusal]), message: 'No memory with this id belongs to this user.' },
  });
  const routes = ['/memory/search', '/memory/add', '/memory/history', '/memory/provenance', '/memory/forget/prepare', '/memory/forget/confirm', '/memory/list', '/agent/status'];
  const server = await stubServer({ '/health': () => ({ body: { status: 'ok' } }), ...Object.fromEntries(routes.map((route) => [route, refuse])) });
  t.after(() => server.close());
  for (const code of Object.keys(refusals)) {
    refusal = code;
    for (const auto of ['1', '0']) assertQuiet(await everyToolResult({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: auto }), `${code}, QUOTEMEM_AUTO=${auto}`);
  }
  // A healthy service: health says so without its address.
  delete server.handlers['/agent/status'];
  const ok = await everyToolResult({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: '1' });
  assertQuiet(ok, 'healthy service');
  assert.deepEqual(JSON.parse(ok.text('quotemem_health')), { status: 'ok' });
  assert.equal(ok.text('quotemem_add'), 'Memory is saved automatically by the QuoteMem plugin, so this conversation is saved already; this tool is off while automatic capture is on.');
  assert.match(ok.err, /quotemem_add is off while automatic capture is on; start the agent with QUOTEMEM_CAPTURE_EVERY=off/, 'how to bring quotemem_add back goes to stderr');
});

test('health: a missing or wrong setting is not_set_up, as for every other tool, even when the service answers', async (t) => {
  const live = await stubServer({ '/health': () => ({ body: { status: 'ok' } }), '/agent/status': () => ({ body: { adds: { completing: 1 } } }) });
  t.after(() => live.close());
  for (const [what, env, logged] of [
    ['no token for the service on this computer', { QUOTEMEM_URL: live.url }, /No token and no memory service set up/],
    ['an unknown auth mode', { QUOTEMEM_URL: live.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_HTTP_AUTH_MODE: 'basic' }, /QUOTEMEM_HTTP_AUTH_MODE must be/],
    ['no token for a remote service', { QUOTEMEM_URL: 'https://memory.invalid' }, /No token: for a service on another computer/],
    ['a URL that is not http', { QUOTEMEM_URL: 'ftp://127.0.0.1:9', QUOTEMEM_TOKEN: TOKEN }, /QUOTEMEM_URL must be an http/],
    // Node reads this as the live service; other clients as another host.
    ['a backslash in the host', { QUOTEMEM_URL: `http://127.0.0.1:${live.port}\\.evil.example.com`, QUOTEMEM_TOKEN: TOKEN }, /QUOTEMEM_URL is not a valid URL: after http/],
    ['a backslash in the host, no token', { QUOTEMEM_URL: `http://127.0.0.1:${live.port}\\.evil.example.com` }, /QUOTEMEM_URL is not a valid URL: after http/],
    // Node rewrites these to the live service; other clients connect to the host as written.
    ['an address with a trailing dot', { QUOTEMEM_URL: `http://127.0.0.1.:${live.port}`, QUOTEMEM_TOKEN: TOKEN }, /QUOTEMEM_URL is not a valid URL: after http/],
    ['a shortened address', { QUOTEMEM_URL: `http://127.1:${live.port}` }, /QUOTEMEM_URL is not a valid URL: after http/],
  ]) {
    const run = await everyToolResult(env);
    const health = JSON.parse(run.text('quotemem_health'));
    assert.deepEqual(health, { status: 'not_set_up', next: run.text('quotemem_search') }, `${what}: the same note as every other tool`);
    assert.match(health.next, /^Long-term memory is not set up correctly\..*`npx quotemem doctor`/, what);
    assert.match(run.err, logged, `${what}: the details go to stderr`);
  }
  assert.equal(live.requests.length, 0, 'nothing is sent while a setting is missing or wrong');
});

test('instructions next to an adapter: a forget request without a note goes through the tools; nothing is claimed forgotten without a note or a tool result', async () => {
  for (const auto of ['1', '0']) {
    const mcp = startServer({ QUOTEMEM_URL: 'http://127.0.0.1:9', QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: auto, QUOTEMEM_AUTOSTART: '0' });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
    const { instructions } = (await mcp.reply(1)).result;
    await mcp.stop();
    if (auto === '1') {
      assert.match(instructions, /When the user asks to forget something and no such note came, forget it with quotemem_forget_prepare and quotemem_forget_confirm\./);
      assert.match(instructions, /Never say that something was forgotten unless such a note or a forget tool result says so\./);
    } else {
      assert.match(instructions, /forget only on request through prepare and confirm/);
    }
  }
});

test('forget through the tools next to an adapter: the statement is kept for the adapter to apply again; never without an adapter', async () => {
  const server = await stubServer({ '/memory/forget/prepare': (body) => ({ body: { schema_version: 1, status: 'nothing_matched', confirmation: null, expires_at: null, candidates: [], decision: null, request_id: body.request_id } }) });
  const forgotten = (home) => path.join(home, '.quotemem', 'forgotten', `${ownerKey(server.url, 'qm-alice')}.json`);
  for (const auto of ['1', '0']) {
    const home = tempDir('qm-server-forgotten-');
    const mcp = startServerIn(home, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'qm-alice', QUOTEMEM_AUTO: auto, QUOTEMEM_AUTOSTART: '0' });
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'my locker code, password: hunter22' } } });
    await mcp.reply(1);
    await mcp.stop();
    if (auto === '1') {
      const kept = JSON.parse(fs.readFileSync(forgotten(home), 'utf8')).items;
      assert.equal(kept.length, 1);
      assert.ok(kept[0].statement.startsWith('my locker code') && !kept[0].statement.includes('hunter22'), 'redacted');
      assert.ok(Math.abs(kept[0].at - Date.now()) < 60_000);
      // Windows keeps no POSIX mode; the file sits in the user's own profile there.
      if (process.platform !== 'win32') assert.equal(fs.statSync(forgotten(home)).mode & 0o777, 0o600);
    } else {
      assert.equal(fs.existsSync(forgotten(home)), false);
    }
  }
  await server.close();
});

test('forget through the tools next to an adapter: a plan prepared and left is not kept; a confirmed one is', async () => {
  const server = await stubServer({
    '/memory/forget/prepare': (body) => ({ body: plan(body) }),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: [{ memory_id: '01JX0000000000000000000F1', outcome: 'verified' }], scope_state: 'ready' } }),
  });
  const home = tempDir('qm-server-forgotten-plan-');
  const forgotten = path.join(home, '.quotemem', 'forgotten', `${ownerKey(server.url, 'qm-alice')}.json`);
  const mcp = startServerIn(home, { QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'qm-alice', QUOTEMEM_AUTO: '1', QUOTEMEM_AUTOSTART: '0' });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'my sister lives in Berlin' } } });
  const left = JSON.parse((await mcp.reply(1)).result.content[0].text);
  assert.equal(left.status, 'confirmation_required');
  mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'my locker code' } } });
  const prepared = JSON.parse((await mcp.reply(2)).result.content[0].text);
  assert.equal(fs.existsSync(forgotten), false, 'nothing is kept for a prepared plan');
  mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_forget_confirm', arguments: { token: prepared.token } } });
  await mcp.reply(3);
  await mcp.stop();
  assert.deepEqual(JSON.parse(fs.readFileSync(forgotten, 'utf8')).items.map((item) => item.statement), ['my locker code'], 'the confirmed one only');
  await server.close();
});

test('forget prepare shows what else the confirmation forgets or may leave unreadable (affected), statements only; confirm reports the count', async () => {
  const TARGET = '01JX0000000000000000000T01';
  const REPLACER = '01JX0000000000000000000R01';
  const PAST = '01JX0000000000000000000P01';
  const KEPT = '01JX0000000000000000000K01';
  const OTHER = '01JX0000000000000000000O01';
  const prepared = (body, withAffected) => {
    const answer = plan(body, [TARGET]);
    answer.candidates = [
      { memory_id: TARGET, revision_id: TARGET, statement: 'The user plays badminton.', role: 'states', target: true },
      { memory_id: REPLACER, revision_id: REPLACER, statement: 'The user plays tennis instead of badminton.', role: 'mentions', target: false },
      { memory_id: OTHER, revision_id: OTHER, statement: 'The user likes sports.', role: 'related', target: false },
    ];
    if (withAffected) {
      answer.affected = [
        { memory_id: REPLACER, effect: 'forgotten', statement: 'The user plays tennis instead of badminton.' },
        { memory_id: PAST, effect: 'forgotten', statement: 'The user played badminton at school.' },
        { memory_id: KEPT, effect: 'unreadable', statement: 'The user\'s club meets on Sundays.' },
      ];
    }
    return { body: answer };
  };
  let withAffected = true;
  const server = await stubServer({
    '/memory/forget/prepare': (body) => prepared(body, withAffected),
    '/memory/forget/confirm': () => ({ body: { schema_version: 1, status: 'verified', request_digest: 'x', outcomes: [{ memory_id: TARGET, outcome: 'verified' }], scope_state: 'ready', forgotten: 3, affected: [{ memory_id: KEPT, effect: 'unreadable' }] } }),
  });
  for (const auto of ['0', '1']) {
    const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTO: auto, QUOTEMEM_AUTOSTART: '0' });
    withAffected = true;
    mcp.send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'I play badminton' } } });
    const text = (await mcp.reply(1)).result.content[0].text;
    const result = JSON.parse(text);
    assert.deepEqual(result.to_forget.map((c) => c.statement), ['The user plays badminton.']);
    assert.deepEqual(result.also_forgotten, ['The user plays tennis instead of badminton.', 'The user played badminton at school.']);
    assert.deepEqual(result.may_become_unreadable, ['The user\'s club meets on Sundays.']);
    assert.deepEqual(result.other_candidates.map((c) => c.statement), ['The user likes sports.'], 'a memory forgotten with it is not offered as kept');
    for (const id of [REPLACER, PAST, KEPT]) assert.ok(!JSON.stringify([result.also_forgotten, result.may_become_unreadable]).includes(id), 'statements only');
    assert.match(result.next, auto === '1' ? /tell the user briefly what went, also_forgotten \(forgotten with it\) and may_become_unreadable/ : /^Show to_forget \(and sentences_to_forget\), also_forgotten \(forgotten with it\) and may_become_unreadable \(kept, but they may stop being readable\) to the user\. Only after the user agrees/);
    mcp.send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_forget_confirm', arguments: { token: result.token } } });
    const confirmed = JSON.parse((await mcp.reply(2)).result.content[0].text);
    assert.equal(confirmed.status, 'verified');
    assert.deepEqual(confirmed.other_memories_changed, { forgotten: 0, unreadable: 1 });
    assert.match(confirmed.next, /Other memories changed with it: 0 forgotten too, 1 no longer readable/);
    assert.ok(!JSON.stringify(confirmed).includes(KEPT));
    // Without `affected` (another mode): the output of before.
    withAffected = false;
    mcp.send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'I play badminton' } } });
    const plain = JSON.parse((await mcp.reply(3)).result.content[0].text);
    for (const key of ['also_forgotten', 'may_become_unreadable', 'also_forgotten_unnamed', 'may_become_unreadable_unnamed']) assert.ok(!(key in plain), key);
    assert.deepEqual(plain.other_candidates.map((c) => c.statement), ['The user plays tennis instead of badminton.', 'The user likes sports.']);
    assert.doesNotMatch(plain.next, /also_forgotten/);
    await mcp.stop();
  }
  await server.close();
});

test('forget prepare: a vetoed candidate the confirmation forgets all the same is listed once, as forgotten; its kept statement is never used for it', async () => {
  const TARGET = '01JX0000000000000000000T02';
  const SISTER = '01JX0000000000000000000S02';
  const OPPOSITE = '01JX0000000000000000000N02';
  const server = await stubServer({
    '/memory/forget/prepare': (body) => {
      const answer = plan(body, [TARGET]);
      answer.candidates = [
        { memory_id: TARGET, revision_id: TARGET, statement: 'The user lives in Porto.', role: 'states', target: true },
        { memory_id: SISTER, revision_id: SISTER, statement: 'The user\'s sister lives in Porto.', role: 'mentions', target: false, excluded: 'other_subject' },
        { memory_id: OPPOSITE, revision_id: OPPOSITE, statement: 'The user never lived in Porto.', role: 'mentions', target: false, excluded: 'negated' },
      ];
      answer.affected = [
        { memory_id: SISTER, effect: 'forgotten', statement: 'The user visits their sister in Porto.' },
        { memory_id: OPPOSITE, effect: 'unreadable' },
      ];
      return { body: answer };
    },
  });
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTOSTART: '0' });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'quotemem_forget_prepare', arguments: { statement: 'I live in Porto' } } });
  const result = JSON.parse((await mcp.reply(1)).result.content[0].text);
  await mcp.stop();
  assert.deepEqual(result.also_forgotten, ['The user visits their sister in Porto.']);
  assert.equal(result.may_become_unreadable, undefined, 'a vetoed candidate\'s words are not its statement here');
  assert.equal(result.may_become_unreadable_unnamed, 1);
  assert.equal(result.kept, undefined, 'no memory is both forgotten and kept');
  assert.ok(!JSON.stringify(result).includes('never lived in Porto'));
  await server.close();
});

test('quotemem_add: a user message is the user\'s own words, no message carries a guessed time; the result carries the store status and reasons', async () => {
  const answers = [
    { success: true, outcome: { added: 3 }, agent: { status: 'stored', unresolved: 0, rejected: 0, reasons: {}, items: [] } },
    { success: true, outcome: { unresolved: 1 }, agent: { status: 'partial', unresolved: 1, rejected: 0, reasons: { identity_unresolved: 1 }, items: [] } },
    { success: true, outcome: { added: 1 } },
  ];
  const server = await stubServer({ '/memory/add': () => ({ body: answers.shift() }) });
  const mcp = startServer({ QUOTEMEM_URL: server.url, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_AUTOSTART: '0', QUOTEMEM_USER_ID: 'qm-alice' });
  mcp.send({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
  const add = (await mcp.reply(1)).result.tools.find((tool) => tool.name === 'quotemem_add');
  const speaker = add.inputSchema.properties.messages.items.properties.speaker.description;
  assert.match(speaker, /^Assistant messages only/);
  assert.match(speaker, /A user message is always the user's own words, so a speaker on it is ignored/);
  assert.equal(add.inputSchema.properties.messages.items.properties.timestamp, undefined, 'the model is not asked for a time it would guess');
  const calls = [
    [{ role: 'user', content: 'My cat is called Bob, and I live in Hangzhou.', speaker: 'me' }],
    [{ role: 'user', content: 'I moved to Shanghai last month.', speaker: 'User' }, { role: 'assistant', content: 'Noted.', speaker: 'Ada' }],
    // A time the model guessed ("today at midnight") is not sent: the service takes the time it receives it.
    [{ role: 'user', content: 'I like green tea.', timestamp: Date.UTC(2026, 10, 24) }],
  ];
  const results = [];
  for (const [index, messages] of calls.entries()) {
    mcp.send({ jsonrpc: '2.0', id: 10 + index, method: 'tools/call', params: { name: 'quotemem_add', arguments: { messages } } });
    const reply = (await mcp.reply(10 + index)).result;
    assert.equal(reply.isError, false);
    results.push(JSON.parse(reply.content[0].text));
  }
  await mcp.stop();
  const sent = server.of('/memory/add').map((request) => request.body);
  assert.deepEqual(sent.map((body) => body.messages), [
    [{ role: 'user', content: 'My cat is called Bob, and I live in Hangzhou.' }],
    [{ role: 'user', content: 'I moved to Shanghai last month.' }, { role: 'assistant', content: 'Noted.', speaker: 'Ada' }],
    [{ role: 'user', content: 'I like green tea.' }],
  ]);
  assert.deepEqual(sent.map((body) => body.source_metadata_version), [undefined, 1, undefined], 'the version goes only with an assistant speaker');
  assert.deepEqual(sent.map((body) => body.user_id), ['qm-alice', 'qm-alice', 'qm-alice']);
  assert.deepEqual(results.map(({ outcome, status, reasons }) => ({ outcome, status, reasons })), [
    { outcome: { added: 3 }, status: 'stored', reasons: {} },
    { outcome: { unresolved: 1 }, status: 'partial', reasons: { identity_unresolved: 1 } },
    { outcome: { added: 1 }, status: null, reasons: null },
  ]);
  assert.ok(results.every((result) => typeof result.request_id === 'string' && typeof result.session_id === 'string'));
  await server.close();
});

test('a server copied without the adapter core sends nothing to a service on this computer: it cannot check who listens', async () => {
  // A program on the port that is not the user's service (a 422, as another program answers).
  const squat = await stubServer({});
  squat.setToken('not-the-token-'.padEnd(43, 'x'));
  const alone = tempDir('qm-server-nocore-');
  fs.copyFileSync(SERVER, path.join(alone, 'server.mjs'));
  const port = new URL(squat.url).port;
  for (const host of ['127.0.0.1', 'localhost', 'localhost.', '[::ffff:127.0.0.1]']) {
    const child = spawn(process.execPath, [path.join(alone, 'server.mjs')], {
      env: { PATH: process.env.PATH, HOME: EMPTY_HOME, USERPROFILE: EMPTY_HOME, QUOTEMEM_URL: `http://${host}:${port}`, QUOTEMEM_TOKEN: TOKEN, QUOTEMEM_USER_ID: 'quei' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
    const reply = async (id) => {
      await waitFor(() => out.split('\n').some((line) => line.includes(`"id":${id},`)), 5000, `reply ${id}`);
      return JSON.parse(out.split('\n').find((line) => line.includes(`"id":${id},`)));
    };
    let search;
    try {
      send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } } });
      await reply(1);
      send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'quotemem_search', arguments: { query: 'what is my PIN' } } });
      search = (await reply(2)).result;
    } finally {
      const exited = new Promise((resolve) => child.once('exit', resolve));
      child.stdin.end();
      await exited;
    }
    assert.equal(search.isError, true, host);
    // Without the core, settings it cannot read are 'not set up'; nothing is sent either way.
    if (host === '127.0.0.1' || host === 'localhost') {
      assert.equal(search.content[0].text, 'Long-term memory is temporarily unavailable. Continue without it, and do not mention this unless the user asks about memory.', host);
      assert.match(err, /adapter core next to this server could not be loaded/, host);
    } else {
      assert.match(search.content[0].text, /^Long-term memory is (temporarily unavailable|not set up correctly)\./, `${host}: ${search.content[0].text}`);
    }
    assert.ok(!(out + err).includes(TOKEN));
  }
  // Not even a listener check: the requests a stub records are all but those.
  assert.deepEqual(squat.requests.map((r) => r.route), []);
  await squat.close();
});
