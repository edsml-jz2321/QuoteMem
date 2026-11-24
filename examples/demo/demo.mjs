#!/usr/bin/env node
// QuoteMem demo: one short story through the service's HTTP routes, for a new
// user, so your own memories are not touched. Needs Node 20.3 or later and a
// service (see the README next to this file); it waits up to a minute for the
// service to answer GET /health, so it can start right after `quotemem serve &`.
//   node demo.mjs [user id]     (QUOTEMEM_URL and QUOTEMEM_TOKEN_FILE, if set, are used)
import { spawnSync } from 'node:child_process';
import { randomBytes, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const url = (process.env.QUOTEMEM_URL || 'http://127.0.0.1:8765').replace(/\/+$/, '');
const dataDir = process.env.QUOTEMEM_DATA_DIR
  || path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'), 'quotemem');
const tokenFile = process.env.QUOTEMEM_TOKEN_FILE || path.join(dataDir, 'token');
const user = process.argv[2] ?? `demo-${randomBytes(3).toString('hex')}`;

// Wait until the service answers GET /health with 200: every second, up to 60 seconds.
async function waitForService() {
  const deadline = Date.now() + 60_000;
  let told = false;
  for (;;) {
    try {
      const response = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2_000) });
      await response.arrayBuffer();
      if (response.status === 200) return;
    } catch {
      // Not listening yet.
    }
    if (Date.now() >= deadline) {
      console.error(`The service at ${url} did not answer /health within 60 seconds. Start it with: quotemem serve`);
      process.exit(1);
    }
    if (!told) {
      console.log(`Waiting for the service at ${url} ...`);
      told = true;
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
}
await waitForService();
const token = fs.readFileSync(tokenFile, 'utf8').trim();

async function post(route, body) {
  const response = await fetch(url + route, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const answer = await response.json();
  if (!response.ok) throw new Error(`${route}: ${response.status} ${JSON.stringify(answer)}`);
  return answer;
}
const say = (session, text) => post('/memory/add', {
  user_id: user, session_id: session, messages: [{ role: 'user', content: text, timestamp: Date.now() }],
});
const stored = (answer) => Object.entries(answer.outcome).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${k}`).join(', ');
const list = () => post('/memory/list', { user_id: user, status: 'all' });
const show = (memories) => memories.forEach((m) => {
  console.log(`   [${m.status}] ${m.statement}`);
  for (const q of m.quotes) console.log(`     quote: "${q.text}"`);
});
const search = async (query) => {
  console.log(`   search "${query}", first hit:`);
  const hits = (await post('/memory/search', { user_id: user, query, top_k: 5 })).data;
  const lines = (hits[0]?.content ?? '').split('\n');
  lines.filter((l) => !l.startsWith('<') && !l.startsWith('The following')).forEach((l) => console.log(`   ${l}`));
  return hits;
};

console.log(`QuoteMem demo at ${url}, user ${user}\n`);
const first = 'I live in Beijing. My cat is called Bob. I work as a nurse.';
console.log(`1. Chat 1, the user says: "${first}"`);
console.log(`   stored: ${stored(await say('demo-chat-1', first))}`);
const second = 'I moved from Beijing to Shanghai last month.';
console.log(`2. Chat 2, the user says: "${second}"`);
console.log(`   stored: ${stored(await say('demo-chat-2', second))}`);

console.log('3. The memories, each with its quote:');
const before = (await list()).memories;
show(before);
const city = before.find((m) => m.status === 'current' && /Shanghai/.test(m.statement));
console.log('   history of the city, newest first:');
const history = await post('/memory/history', { schema_version: 3, user_id: user, memory_id: city.memory_id });
for (const v of history.data.entries) console.log(`   ${v.change}: ${v.statement}`);

console.log('4. Recall:');
await search('Where does the user live now?');

const request = "Forget my cat's name.";
console.log(`5. Chat 3, the user says: "${request}"`);
const plan = await post('/memory/forget/prepare', {
  schema_version: 1, user_id: user, client_session_id: 'demo-chat-3', request_id: randomUUID(), statement: request,
});
for (const c of plan.candidates ?? []) if (c.target || c.role === 'same_fact') console.log(`   prepare, goes: ${c.statement}`);
for (const a of plan.affected ?? []) if (a.statement) console.log(`   prepare, ${a.effect} to: ${a.statement}`);
if (plan.status !== 'confirmation_required') throw new Error(`nothing to confirm: ${plan.status}`);
const done = await post('/memory/forget/confirm', plan.confirmation);
console.log(`   confirm: ${done.status}`);

console.log('6. After the forget:');
const after = (await list()).memories;
show(after);
const hits = await search("What is the name of the user's cat?");
const named = JSON.stringify([after, hits]).includes('Bob');
console.log(`   "Bob" in the memories or the search: ${named ? 'yes' : 'no'}`);

// The same list in a terminal, from the npm command.
console.log(`\n$ quotemem memories --user ${user}`);
const cli = spawnSync('quotemem', ['memories', '--user', user], {
  stdio: 'inherit', shell: process.platform === 'win32', env: { ...process.env, QUOTEMEM_URL: url },
});
if (cli.error) console.log('(the quotemem command is not on PATH: npm install -g quotemem)');
