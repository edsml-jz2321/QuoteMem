// The demo's wait for the service: what it says when /health never answers
// 200. The clock runs 600 times faster here, so its minute takes 0.1 s.
//
//   node --test examples/demo/test/demo.test.mjs
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const DEMO = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'demo.mjs');
const FAST_CLOCK = `data:text/javascript,${encodeURIComponent(`
const realNow = Date.now.bind(Date);
const start = realNow();
Date.now = () => start + (realNow() - start) * 600;
const realTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...rest) => realTimeout(fn, Math.ceil((ms ?? 0) / 600), ...rest);
`)}`;

function runDemo(url) {
  return new Promise((resolve) => {
    const args = [`--import=${FAST_CLOCK}`, DEMO, 'demo-test'];
    const child = spawn(process.execPath, args, {
      env: { PATH: process.env.PATH, QUOTEMEM_URL: url },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let err = '';
    child.stderr.on('data', (d) => { err += d; });
    child.on('exit', (code) => resolve({ code, err }));
  });
}

async function healthServer(status) {
  const server = http.createServer((req, res) => {
    res.writeHead(status, { 'content-type': 'application/json', connection: 'close' });
    res.end(JSON.stringify({ code: 'service_unhealthy' }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  return { url, close: () => new Promise((resolve) => server.close(resolve)) };
}

test('a service answering /health with 503 is running, its providers unavailable', async () => {
  const server = await healthServer(503);
  const { code, err } = await runDemo(server.url);
  await server.close();
  assert.equal(code, 1);
  assert.match(err, /is running, but answered \/health with 503 for 60 seconds/);
  assert.match(err, /providers are unavailable or it is still starting/);
  assert.doesNotMatch(err, /did not answer/);
});

test('a service that is not listening is named as not answering', async () => {
  const server = await healthServer(503);
  const { url } = server;
  await server.close();
  const { code, err } = await runDemo(url);
  assert.equal(code, 1);
  assert.match(err, /did not answer \/health within 60 seconds\. Start it with: quotemem serve/);
  assert.doesNotMatch(err, /is running/);
});
