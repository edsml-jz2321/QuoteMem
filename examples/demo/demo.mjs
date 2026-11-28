#!/usr/bin/env node
// QuoteMem demo: one short story through the service's HTTP routes, for a new
// user, so your own memories are not touched. Needs Node 20.3 or later and a
// service (see the README next to this file); it waits up to a minute for the
// service to answer GET /health, so it can start right after `quotemem serve &`.
//   node demo.mjs [user id] [--relearn]
// QUOTEMEM_URL and QUOTEMEM_TOKEN_FILE, if set, are used. --relearn adds a last
// step where the user states the forgotten name again.
import { spawnSync } from 'node:child_process';
import { randomBytes, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const url = (process.env.QUOTEMEM_URL || 'http://127.0.0.1:8765').replace(/\/+$/, '');
const dataDir = process.env.QUOTEMEM_DATA_DIR || path.join(
  process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'), 'quotemem');
const tokenFile = process.env.QUOTEMEM_TOKEN_FILE || path.join(dataDir, 'token');
const args = process.argv.slice(2);
const relearn = args.includes('--relearn');
const user = args.find((a) => !a.startsWith('--')) ?? `demo-${randomBytes(3).toString('hex')}`;

// Wait until the service answers GET /health with 200: every second, up to 60 seconds.
// False (after saying why) when it never did.
async function waitForService() {
  const deadline = Date.now() + 60_000;
  let told = false;
  let last = 0; // the last /health status, 0 while nothing answers
  for (;;) {
    try {
      const response = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2_000) });
      await response.arrayBuffer();
      if (response.status === 200) return true;
      last = response.status;
    } catch {
      // Not listening yet.
    }
    if (Date.now() >= deadline) {
      const running = `The service at ${url} is running, but answered /health with`;
      console.error(last === 503
        ? `${running} 503 for 60 seconds: its model providers are unavailable or it is still`
          + ' starting. See its log (quotemem status names it); quotemem check tests the providers.'
        : last
          ? `${running} ${last}, not 200, for 60 seconds.`
          : `The service at ${url} did not answer /health within 60 seconds.`
            + ' Start it with: quotemem serve');
      return false;
    }
    if (!told) {
      console.log(`Waiting for the service at ${url} ...`);
      told = true;
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
}
// The demo exits by itself, not with process.exit(): on Windows an exit while
// Node is still closing the last request's handles can crash it (0xC0000409).
if (await waitForService()) await main();
else process.exitCode = 1;

async function main() {
  const token = fs.readFileSync(tokenFile, 'utf8').trim();

  // One call: its HTTP status and its JSON body.
  async function call(method, route, body) {
    const response = await fetch(url + route, {
      method,
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { status: response.status, answer: await response.json() };
  }
  async function post(route, body) {
    const { status, answer } = await call('POST', route, body);
    if (status < 200 || status > 299) {
      throw new Error(`${route}: ${status} ${JSON.stringify(answer)}`);
    }
    return answer;
  }
  const add = (session, messages) => post('/memory/add', {
    user_id: user, session_id: session, messages,
  });
  const say = (session, text) => add(session, [
    { role: 'user', content: text, timestamp: Date.now() },
  ]);
  const stored = (answer) => Object.entries(answer.outcome).filter(([, n]) => n > 0)
    .map(([k, n]) => `${n} ${k}`).join(', ') || 'nothing new';
  const list = () => post('/memory/list', { user_id: user, status: 'all' });
  const history = async (memoryId) => (await post('/memory/history', {
    schema_version: 3, user_id: user, memory_id: memoryId,
  })).data.entries;
  const show = (memories) => memories.forEach((m) => {
    console.log(`   [${m.status}] ${m.statement}`);
    for (const q of m.quotes) console.log(`     quote: "${q.text}"`);
  });
  const searchHits = async (query) => (await post('/memory/search', {
    user_id: user, query, top_k: 5,
  })).data;
  const search = async (query) => {
    console.log(`   search "${query}", first hit:`);
    const hits = await searchHits(query);
    const lines = (hits[0]?.content ?? '').split('\n');
    lines.filter((l) => !l.startsWith('<') && !l.startsWith('The following'))
      .forEach((l) => console.log(`   ${l}`));
    return hits;
  };

  // The chats by number. A quote's byte range is cut from the message the user sent.
  const chats = new Map();
  function chatOf(q) {
    for (const [n, text] of chats) {
      if (Buffer.from(text).subarray(q.byte_start, q.byte_end).toString() === q.quote) return n;
    }
    throw new Error(`no chat holds "${q.quote}" at bytes ${q.byte_start}-${q.byte_end}`);
  }
  const showQuote = (q) => console.log(
    `   quote: bytes ${q.byte_start}-${q.byte_end} of chat ${chatOf(q)}: "${q.quote}"`);

  console.log(`QuoteMem demo at ${url}, user ${user}\n`);
  const store = (await call('GET', '/agent/status')).answer.store ?? {};
  console.log(`0. The store: Add mode ${store.add_mode}, review ${store.review}`);
  if (store.add_mode !== 'agent') {
    console.log('   (step 7 is written for the agent Add mode)');
  }

  const first = 'I live in Beijing. My cat is called Bob. I work as a nurse.';
  chats.set(1, first);
  console.log(`1. Chat 1, the user says: "${first}"`);
  console.log(`   stored: ${stored(await say('demo-chat-1', first))}`);
  const second = 'I moved from Beijing to Shanghai last month.';
  chats.set(2, second);
  console.log(`2. Chat 2, the user says: "${second}"`);
  console.log(`   stored: ${stored(await say('demo-chat-2', second))}`);

  console.log('3. The memories, each with its quote:');
  const before = (await list()).memories;
  show(before);
  const statements = new Map(before.map((m) => [m.memory_id, m.statement]));
  const city = before.find((m) => m.status === 'current' && /Shanghai/.test(m.statement));
  const cat = before.find((m) => /Bob/.test(m.statement));
  console.log('   history of the city, newest first:');
  const cityHistory = await history(city.memory_id);
  for (const v of cityHistory) console.log(`   ${v.change}: ${v.statement}`);
  const head = cityHistory[0];
  for (const link of head.supersedes) {
    const month = link.effective_at
      ? `, from ${new Date(link.effective_at).toISOString().slice(0, 7)}` : '';
    console.log(`   it replaces: "${statements.get(link.memory_id)}" (${link.relation}${month})`);
  }
  for (const q of head.quotes) showQuote(q);

  console.log('   the review behind it:');
  const why = (await post('/memory/provenance', {
    schema_version: 2, user_id: user, target: { kind: 'memory', memory_id: city.memory_id },
  })).data;
  const outcome = `${why.outcome.category}: ${why.outcome.reason_code}`;
  if (why.verdict === null) {
    console.log(`   no review verdict (${outcome})`);
  } else {
    for (const r of why.verdict.relations.filter((r) => r.relation !== 'independent')) {
      console.log(`   ${r.relation} of "${statements.get(r.memory_id)}" (${outcome})`);
    }
    console.log(`   reason: ${why.verdict.reason ?? '(withheld)'}`);
  }

  console.log('4. Recall:');
  await search('Where does the user live now?');

  const request = "Forget my cat's name.";
  console.log(`5. Chat 3, the user says: "${request}"`);
  const chat1Sources = new Set();
  for (const v of cat ? await history(cat.memory_id) : []) {
    for (const q of v.quotes) chat1Sources.add(q.evidence_id);
  }
  const plan = await post('/memory/forget/prepare', {
    schema_version: 1, user_id: user, client_session_id: 'demo-chat-3',
    request_id: randomUUID(), statement: request,
  });
  for (const c of plan.candidates ?? []) {
    if (c.target || c.role === 'same_fact') {
      console.log(`   prepare, states the fact: ${c.statement}`);
    }
  }
  for (const a of plan.affected ?? []) {
    if (a.statement) console.log(`   prepare, ${a.effect} to: ${a.statement}`);
  }
  console.log(`   prepare, notice: ${plan.retention_notice}`);
  if (plan.status !== 'confirmation_required') {
    throw new Error(`nothing to confirm: ${plan.status}`);
  }
  const done = await post('/memory/forget/confirm', plan.confirmation);
  console.log(`   confirm: ${done.status}`);
  const receipt = done.outcomes.find((o) => o.receipt)?.receipt;
  if (receipt) {
    const { counts, internal_retention: kept } = receipt;
    console.log(`   receipt: tombstones written ${counts.tombstones_written}, `
      + `sentences withheld ${counts.evidence_spans_withheld}, `
      + `readable left ${receipt.readable_residual_count}`);
    const purpose = kept.purpose.replace('_', ' ');
    console.log(`   tombstone: kept to ${purpose}, readable: ${kept.readable}`);
  }

  console.log('6. After the forget:');
  const after = (await list()).memories;
  show(after);
  const hits = await search("What is the name of the user's cat?");
  const named = JSON.stringify([after, hits]).includes('Bob');
  console.log(`   "Bob" in the memories or the search: ${named ? 'yes' : 'no'}`);
  // The history of every memory from before the forget: a deleted one answers 404, and a
  // reworded one keeps its earlier versions without their content.
  const histories = [];
  for (const m of before) {
    const { status, answer } = await call('POST', '/memory/history', {
      schema_version: 3, user_id: user, memory_id: m.memory_id,
    });
    histories.push(answer);
    if (status === 404) {
      console.log(`   history of a memory the forget deleted: ${status} ${answer.code}`);
    }
  }
  const inHistory = JSON.stringify(histories).includes('Bob');
  console.log(`   "Bob" in any history: ${inHistory ? 'yes' : 'no'}`);

  console.log('7. Chat 4 names the cat again: an old chat imported without a date,');
  console.log("   then a question and the assistant's answer:");
  const chat4 = [
    { role: 'user', content: 'My cat is called Bob.' },
    { role: 'user', content: 'Is my cat called Bob?', timestamp: Date.now() },
    { role: 'assistant', content: 'Yes, your cat is called Bob.', timestamp: Date.now() },
  ];
  for (const m of chat4) {
    console.log(`   ${m.role}${m.timestamp ? '' : ', no date'}: "${m.content}"`);
  }
  const fourth = await add('demo-chat-4', chat4);
  console.log(`   stored: ${stored(fourth)}`);
  const reasons = Object.entries(fourth.agent?.reasons ?? {})
    .map(([code, n]) => (n > 1 ? `${code} (${n})` : code));
  if (reasons.length) console.log(`   reasons: ${reasons.join(', ')}`);
  const again = [(await list()).memories, await searchHits("What is the name of the user's cat?")];
  console.log(`   "Bob" in the memories or the search: ${
    JSON.stringify(again).includes('Bob') ? 'yes' : 'no'}`);

  if (relearn) {
    const fifth = 'For the record, my cat is called Bob.';
    chats.set(5, fifth);
    console.log(`8. Chat 5, the user says, dated now: "${fifth}"`);
    console.log(`   stored: ${stored(await say('demo-chat-5', fifth))}`);
    const back = (await list()).memories
      .filter((m) => m.status === 'current' && /Bob/.test(m.statement));
    for (const m of back) {
      console.log(`   [${m.status}] ${m.statement}`);
      const quotes = (await history(m.memory_id))[0].quotes;
      for (const q of quotes) showQuote(q);
      const old = quotes.some((q) => chat1Sources.has(q.evidence_id));
      console.log(`   quotes chat 1: ${old ? 'yes' : 'no'}`);
    }
  }

  // The same list in a terminal, from the npm command.
  console.log(`\n$ quotemem memories --user ${user}`);
  const cli = spawnSync('quotemem', ['memories', '--user', user], {
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: { ...process.env, QUOTEMEM_URL: url },
  });
  if (cli.error) console.log('(the quotemem command is not on PATH: npm install -g quotemem)');
}
