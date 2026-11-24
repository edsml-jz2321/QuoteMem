#!/usr/bin/env node
// The HTTP client: bounded retries, Retry-After and deadlines.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { after, test } from 'node:test';

import { CORE, HIT, TOKEN, stubServer, tempDir, testSettings } from './helpers.mjs';

const { callJson, postJson, retryable } = await import(pathToFileURL(path.join(CORE, 'http.mjs')).href);
const { RecallBreaker, searchRecall } = await import(pathToFileURL(path.join(CORE, 'recall.mjs')).href);
const servers = [];
after(async () => { await Promise.all(servers.map((s) => s.close())); });

test('callJson: retries busy answers with Retry-After, returns other answers at once', async () => {
  let calls = 0;
  const server = await stubServer({
    '/memory/search': () => {
      calls += 1;
      return calls < 3 ? { status: 503, headers: { 'retry-after': '0' }, body: { code: 'busy' } } : { body: { data: [] } };
    },
    '/bad': () => ({ status: 422, body: { code: 'invalid_request' } }),
    '/slow': () => ({ delay: 1000, body: {} }),
  });
  servers.push(server);
  const settings = { url: server.url, token: TOKEN, authMode: 'x-api-key' };
  const ok = await callJson(settings, '/memory/search', { q: 1 }, { deadlineMs: 3000 });
  assert.equal(ok.status, 200);
  assert.equal(server.of('/memory/search').length, 3);
  assert.equal(server.of('/memory/search')[0].headers['x-api-key'], TOKEN);
  const bad = await callJson(settings, '/bad', {}, { deadlineMs: 3000 });
  assert.equal(bad.status, 422);
  assert.equal(server.of('/bad').length, 1, 'not retried');
  const started = Date.now();
  await assert.rejects(callJson(settings, '/slow', {}, { deadlineMs: 200 }));
  assert.ok(Date.now() - started < 800, 'stopped at the deadline');
  await assert.rejects(postJson({ url: 'not a url', token: TOKEN }, '/x', {}), 'a bad URL rejects');
});

test('retryable: busy states, other 5xx, and any answer the service marks retryable', () => {
  for (const code of ['erasure_in_progress', 'index_rebuilding', 'provider_unavailable', 'delivery_busy']) {
    assert.ok(retryable({ status: 429, data: { code, retryable: true } }), code);
  }
  assert.ok(retryable({ status: 503, data: { code: 'service_unhealthy', retryable: true } }));
  assert.ok(retryable({ status: 500, data: null }));
  assert.ok(!retryable({ status: 500, data: { retryable: false } }));
  assert.ok(retryable({ status: 422, data: { code: 'budget_exceeded', retryable: true } }), 'marked retryable by the service');
  assert.ok(!retryable({ status: 422, data: { code: 'operation_conflict', retryable: false } }));
  assert.ok(!retryable({ status: 409, data: { code: 'user_deleting' } }));
});

// Right after a forget the service rebuilds the user's index for a second or
// two, and its 429 index_rebuilding says Retry-After: 60 (measured).
const REBUILDING = { status: 429, headers: { 'retry-after': '60' }, body: { code: 'index_rebuilding', retryable: true } };

test('recall: asks again while the index is rebuilt after a forget, within its own deadline', async () => {
  let calls = 0;
  const server = await stubServer({
    '/memory/search': () => ((calls += 1) <= 3 ? REBUILDING : { body: { data: [HIT('01JX0000000000000000000CA1', 'The user has a cat.', { score: 0.8 })] } }),
  });
  servers.push(server);
  const settings = await testSettings(server.url, tempDir('qm-core-recall-'), { recallTimeoutMs: 3000 });
  const breaker = new RecallBreaker();
  const report = {};
  const started = Date.now();
  const block = await searchRecall(settings, 'Do I have a cat?', { breaker, report });
  const took = Date.now() - started;
  assert.ok(block && block.text.includes('The user has a cat.'), 'answered once the index is rebuilt');
  assert.equal(server.of('/memory/search').length, 4);
  assert.ok(took >= 700 && took < 3000, `about 250 ms between the searches, never the 60 s Retry-After (${took} ms)`);
  assert.ok(!report.unavailable && report.reached);
  assert.equal(breaker.failures + breaker.timeouts, 0);
});

test('recall: a rebuild that outlasts the deadline ends at the deadline, as unavailable, and opens no breaker', async () => {
  const server = await stubServer({ '/memory/search': () => REBUILDING });
  servers.push(server);
  const settings = await testSettings(server.url, tempDir('qm-core-recall-'), { recallTimeoutMs: 1000 });
  const breaker = new RecallBreaker();
  const report = {};
  const started = Date.now();
  for (let round = 0; round < 2; round += 1) {
    assert.equal(await searchRecall(settings, 'Do I have a cat?', { breaker, report }), null);
  }
  const took = (Date.now() - started) / 2;
  assert.ok(took < 1100, `never longer than the recall deadline (${took} ms)`);
  assert.ok(server.of('/memory/search').length >= 6, 'about four searches each');
  assert.ok(report.unavailable, 'the note says memory could not be checked');
  assert.ok(!breaker.open && breaker.failures === 0 && breaker.timeouts === 0, 'a rebuild is not a failing service');
});

test('recall: a rebuild that says Retry-After 0 is asked again at most every 50 ms, not in a tight loop', async () => {
  const server = await stubServer({ '/memory/search': () => ({ ...REBUILDING, headers: { 'retry-after': '0' } }) });
  servers.push(server);
  const settings = await testSettings(server.url, tempDir('qm-core-recall-'), { recallTimeoutMs: 600 });
  const report = {};
  assert.equal(await searchRecall(settings, 'Do I have a cat?', { breaker: new RecallBreaker(), report }), null);
  const searches = server.of('/memory/search').length;
  // 600 ms, with 150 ms kept for the last answer: at most about nine asks 50 ms apart.
  assert.ok(searches >= 2 && searches <= 12, `about one search every 50 ms (${searches})`);
  assert.ok(report.unavailable);
});

test('recall: no other error answer is asked again', async () => {
  const answers = {
    busy: { status: 429, headers: { 'retry-after': '0' }, body: { code: 'delivery_busy', retryable: true } },
    erasing: { status: 429, headers: { 'retry-after': '0' }, body: { code: 'erasure_in_progress', retryable: true } },
    provider: { status: 429, headers: { 'retry-after': '0' }, body: { code: 'provider_unavailable', retryable: true } },
    unhealthy: { status: 503, headers: { 'retry-after': '0' }, body: { code: 'service_unhealthy', retryable: true } },
    rebuilding503: { status: 503, headers: { 'retry-after': '0' }, body: { code: 'index_rebuilding', retryable: true } },
    invalid: { status: 422, body: { code: 'invalid_request' } },
  };
  for (const [name, answer] of Object.entries(answers)) {
    const server = await stubServer({ '/memory/search': () => answer });
    servers.push(server);
    const settings = await testSettings(server.url, tempDir('qm-core-recall-'), { recallTimeoutMs: 3000 });
    const report = {};
    assert.equal(await searchRecall(settings, 'Do I have a cat?', { report }), null, name);
    assert.equal(server.of('/memory/search').length, 1, `${name}: sent once`);
    assert.ok(report.unavailable, name);
  }
});
