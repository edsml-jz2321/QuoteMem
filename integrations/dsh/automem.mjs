// QuoteMem automatic memory for DeepSeek Harness (DSH): a native Cordis plugin.
//
// What the user never sees:
//   Capture: the human user's prompts of each turn and the final assistant
//     text of each completed turn are recorded per session. Every N user turns
//     (default 3), when the session ends, when DSH shuts down or after idle
//     time, a batch is queued on disk and a detached background sender posts
//     it to QuoteMem /memory/add, so DSH never waits and quitting loses nothing.
//   Recall: before the model runs on a user turn, a question about the past
//     or about the user is searched under a short deadline. The top hits go in
//     as one context message, marked as retrieved and untrusted.
//   Forget: a message that asks to forget something is carried out after
//     the batches queued before it (prepare, then confirm), and the model
//     gets one short note so it can confirm; a forget that takes longer than
//     the turn's short wait is reported on the next turn. The note names no
//     tool and no id. An explicit forget tool waits for the earlier batches
//     to be stored, and prepare is refused with a short reason when they are
//     not stored in time.
//
// The work itself is the shared adapter core (./core, a copy of
// integrations/core): this file maps DSH's events onto HookMemory, which keeps
// each session's turns in a file and sends batches through the disk queue.
// Every handler catches its own errors and bounds its own wait, so a slow or
// stopped QuoteMem never breaks or blocks DSH.
//
// Settings come from QUOTEMEM_* environment variables (see README.md). A patch
// row may override the behaviour settings through its `config`. The service
// URL, user id, auth mode and token are read only from the environment, so the
// MCP tools in the same bundle always use the same values; the token is never
// logged.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';

import { homeDir } from '../core/proc.mjs';

import {
  FORGET_TOOL_WAIT_MS,
  HookMemory,
  RECALL_HEADER,
  buildJobs as coreBuildJobs,
  errKind,
  formatRecall,
  isMemoryRelated,
  isTrivial,
  makeLog as coreMakeLog,
  newRecallMemo,
  redactText,
  resolveSettings as coreResolveSettings,
  textOf,
} from '../core/index.mjs';

export { RECALL_HEADER, formatRecall, isMemoryRelated, isTrivial, redactText };

const requireHost = createRequire(import.meta.url);

// The host's Schemastery copy (an optional peer dependency). Without it the
// module still loads; Cordis then passes the row's config unvalidated.
let Schema = null;
try {
  const loaded = requireHost('@deepseek-ai/schemastery');
  Schema = loaded?.default ?? loaded;
} catch {
  Schema = null;
}

export const name = 'quotemem-automem';
export const inject = ['agents'];

// Every field is optional and has no schema default, so an unset field falls
// back to its environment variable and then to the built-in default. The
// service URL, user id and auth mode are deliberately absent: the MCP server
// reads only the environment, and both must act on the same memory. `off` and
// `false` are accepted where a setting can be turned off, because YAML 1.1
// loaders read a bare `off` as false.
const OFF_VALUES = () => [Schema.const('off'), Schema.const(false)];
export const Config = Schema
  ? Schema.object({
    recall: Schema.union(['gated', 'always', ...OFF_VALUES()]).description('Automatic recall mode (QUOTEMEM_RECALL).'),
    recallGate: Schema.union(['questions', 'broad']).description('What the gated mode searches on: questions about the past or the user, or also plain statements about the user (QUOTEMEM_RECALL_GATE).'),
    topK: Schema.natural().description('Hits requested and injected at most (QUOTEMEM_TOP_K).'),
    maxInjectChars: Schema.natural().description('Character budget of the injected block (QUOTEMEM_MAX_INJECT_CHARS).'),
    recallMinScore: Schema.number().description('Relevance threshold (QUOTEMEM_RECALL_MIN_SCORE).'),
    recallTimeoutMs: Schema.natural().description('Recall deadline in ms (QUOTEMEM_RECALL_TIMEOUT_MS).'),
    recallRepeatTurns: Schema.natural().description('User turns before an injected memory may be injected again (QUOTEMEM_RECALL_REPEAT_TURNS).'),
    searchOptions: Schema.array(Schema.string()).description('Search options sent to /memory/search (QUOTEMEM_SEARCH_OPTIONS).'),
    forget: Schema.union(['auto', ...OFF_VALUES()]).description('Silent forget of a "forget …" request, or off to leave it to the tools (QUOTEMEM_FORGET).'),
    forgetTimeoutMs: Schema.natural().description('How long a turn waits for a silent forget before the note says it finishes in the background (QUOTEMEM_FORGET_TIMEOUT_MS).'),
    captureEvery: Schema.union([Schema.natural(), ...OFF_VALUES()]).description('User turns per capture batch; 0 or off turns capture off, but prefer QUOTEMEM_CAPTURE_EVERY=off (QUOTEMEM_CAPTURE_EVERY).'),
    idleFlushMs: Schema.union([Schema.natural(), ...OFF_VALUES()]).description('Idle time before a partial batch is sent; 0 or off disables (QUOTEMEM_IDLE_FLUSH_MS).'),
    captureAssistant: Schema.union(['final', 'none']).description('Assistant text captured (QUOTEMEM_CAPTURE_ASSISTANT).'),
    maxMessageChars: Schema.natural().description('Cap per captured message (QUOTEMEM_MAX_MESSAGE_CHARS).'),
    captureTimeoutMs: Schema.natural().description('Time limit of one /memory/add attempt (QUOTEMEM_CAPTURE_TIMEOUT_MS).'),
    captureMaxAttempts: Schema.natural().description('Attempts before a batch is dropped (QUOTEMEM_CAPTURE_MAX_ATTEMPTS).'),
    captureMaxAgeMs: Schema.natural().description('Age after which an unsent batch is dropped (QUOTEMEM_CAPTURE_MAX_AGE_MS).'),
    captureBusyMaxWaitMs: Schema.natural().description('How long a batch the service holds back as busy keeps waiting before it is dropped (QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS).'),
    captureQueueMax: Schema.natural().description('Batches kept in the queue at most (QUOTEMEM_CAPTURE_QUEUE_MAX).'),
    retryBaseMs: Schema.natural().description('First retry delay of a failed /memory/add (QUOTEMEM_CAPTURE_RETRY_BASE_MS).'),
    flusherMaxWaitMs: Schema.natural().description('Longest retry wait the background sender sleeps through (QUOTEMEM_FLUSHER_MAX_WAIT_MS).'),
    stateDir: Schema.string().description('Folder of the capture queue and session files (QUOTEMEM_STATE_DIR).'),
    includeSubagents: Schema.boolean().description('Also recall and capture in subagent sessions (QUOTEMEM_INCLUDE_SUBAGENTS).'),
  })
  : undefined;

const HOST = 'dsh';
// Names the plugin in the marker of a cut message.
const LABEL = 'DSH plugin';
// Turn ends whose user messages are captured; only `completed` adds the reply.
const USER_TURN_ENDS = new Set(['completed', 'aborted', 'error', 'max-tokens']);
// The bundle's forget tools, under any MCP prefix DSH gives them.
const FORGET_TOOL = /(^|[^a-z0-9])quotemem_forget_(prepare|confirm)$/i;
// How long an explicit forget tool waits for the earlier batches to be
// stored. Then prepare is refused: run now, it could miss them for good,
// since the service's in-chat check may not read the user's words.
let forgetToolWaitMs = FORGET_TOOL_WAIT_MS;
const PENDING_CAPTURE = 'The latest conversation is still being saved to memory, so a forget now could miss it. '
  + 'Tell the user it will be possible in a minute, or try again shortly.';

/**
 * Resolve the plugin settings: the row's config, then QUOTEMEM_* variables,
 * then built-in defaults. The URL, user id, auth mode and token come from the
 * environment only. Out-of-range numbers are clamped; nothing throws.
 * @param {object} config - the Cordis row config (all fields optional).
 * @param {object} env - the process environment.
 * @param {{profileDir?: string}} host - facts read from the DSH context.
 * @returns {object} settings; `problem` is set when automatic memory must stay off.
 */
export function resolveSettings(config = {}, env = process.env, host = {}) {
  return coreResolveSettings(config, env, {
    configName: 'row config',
    defaultStateDir: host.profileDir ? path.join(host.profileDir, 'quotemem') : path.join(homeDir(), '.quotemem', 'dsh'),
  });
}

/**
 * Split collected messages into /memory/add bodies within the service limits. The
 * request_id is derived from the session, the turn range and the content, so
 * a resend or a replay after restart carries the same id.
 */
export function buildJobs(settings, sessionKey, firstTurn, lastTurn, messages, now = Date.now()) {
  return coreBuildJobs(settings, `${HOST}:${sessionKey}`, firstTurn, lastTurn, messages, now);
}

// ------------------------------------------------------------- the plugin

let logSink = null;
let lastInstance = null;
let hostOverride = null;

/** Test hooks; not part of the plugin contract. */
export const _testing = {
  setLogSink(fn) { logSink = fn; },
  setHost(host) { hostOverride = host; },
  setForgetToolWaitMs(ms) { forgetToolWaitMs = ms ?? FORGET_TOOL_WAIT_MS; },
  get instance() { return lastInstance; },
};

function makeLog(settings) {
  return coreMakeLog(settings, (line) => {
    if (logSink) logSink(line);
    else console.warn(line);
  });
}

async function loadHost() {
  if (hostOverride) return hostOverride;
  const host = { createUserMessage: null, isAppendSurfaceEvent: null };
  try {
    host.createUserMessage = (await import('@deepseek-ai/dsh-llm')).createUserMessage ?? null;
  } catch {
    host.createUserMessage = null;
  }
  try {
    host.isAppendSurfaceEvent = (await import('@deepseek-ai/dsh-session')).isAppendSurfaceEvent ?? null;
  } catch {
    host.isAppendSurfaceEvent = null;
  }
  return host;
}

function profileDirOf(ctx) {
  try {
    const profile = typeof ctx.get === 'function' ? ctx.get('profileContext') : undefined;
    return typeof profile?.dir === 'string' ? profile.dir : undefined;
  } catch {
    return undefined;
  }
}

function sessionKeyOf(session) {
  const id = session?.header?.id ?? session?.id;
  return id === undefined || id === null ? null : String(id);
}

function agentKeyOf(agent) {
  return sessionKeyOf(agent?.session) ?? (agent?.id !== undefined && agent?.id !== null ? String(agent.id) : null);
}

function isSubagent(session) {
  return session?.header?.origin === 'subagent';
}

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms).unref?.(); });

/** Resolve with the promise's value, or with `fallback` once a signal aborts or `ms` passes. */
function bounded(promise, ms, signals, fallback) {
  const all = AbortSignal.any([AbortSignal.timeout(ms), ...signals.filter((s) => s instanceof AbortSignal)]);
  if (all.aborted) return Promise.resolve(fallback);
  return new Promise((resolve) => {
    const stop = () => resolve(fallback);
    all.addEventListener('abort', stop, { once: true });
    Promise.resolve(promise).then(resolve, () => resolve(fallback)).finally(() => all.removeEventListener('abort', stop));
  });
}

export function apply(ctx, config) {
  const settings = resolveSettings(config ?? {}, process.env, { profileDir: profileDirOf(ctx) });
  const log = makeLog(settings);
  if (settings.problem) {
    log.warnOnce('off', `automatic memory is off: ${settings.problem}`);
    return;
  }
  for (const warning of settings.warnings) log.warnOnce(warning, warning);
  if (settings.recall === 'off' && settings.captureEvery === 0 && settings.forget === 'off') return;

  const life = new AbortController();
  const hostReady = loadHost();
  let host = null;
  hostReady.then((h) => { host = h; }, () => {});
  const memory = new HookMemory({ settings, log, host: HOST, label: LABEL });
  // session key -> {chain, used, reply}
  const live = new Map();
  let idleTimer = null;
  let closed = false;
  const instance = { settings, memory, live };
  lastInstance = instance;

  const liveOf = (key) => {
    let s = live.get(key);
    if (!s) {
      s = { chain: Promise.resolve(), used: false, reply: null };
      live.set(key, s);
    }
    return s;
  };

  // Run one step of a session after the previous one, so a turn's end is
  // recorded before the next message even though DSH does not await events.
  const serial = (key, fn) => {
    const s = liveOf(key);
    const run = s.chain.then(fn);
    s.chain = run.catch(() => {});
    return run;
  };
  instance.idle = () => Promise.all([...live.values()].map((s) => s.chain));

  // DSH stays up while the user is away: once sessions have been idle for
  // QUOTEMEM_IDLE_FLUSH_MS, their pending turns are queued.
  const idleLater = () => {
    if (settings.idleFlushMs <= 0 || closed) return;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      idleTimer = null;
      memory.sweep().catch(() => {});
    }, settings.idleFlushMs + 250);
    idleTimer.unref?.();
  };

  // --------------------------------------------------- user turns: pre-step

  async function preStep(payload, decision) {
    if (decision?.kind !== 'enter' || !Array.isArray(decision.messages)) return decision;
    const session = payload?.agent?.session;
    if (payload?.signal?.aborted || closed) return decision;
    if (isSubagent(session) && !settings.includeSubagents) return decision;
    const users = decision.messages.filter((m) => m?.source?.kind === 'user');
    if (users.length === 0) return decision; // tool-loop steps carry no new prompt
    const key = agentKeyOf(payload?.agent);
    if (key === null) return decision;
    const text = users.map((m) => textOf(m.content)).filter(Boolean).join('\n\n');
    if (!text) return decision;
    const s = liveOf(key);
    s.used = true;
    s.reply = null;
    // One DSH turn may take steering messages in later steps; each user batch
    // is its own recorded turn, and a re-run of the same step records nothing.
    const turn = Number.isInteger(payload?.turn) ? `${payload.turn}.${payload.step ?? 0}` : null;
    const work = serial(key, () => memory.prompt({ session: key, turn, text, time: Date.now() }));
    // A prompt may first wait for a queued forget before it recalls.
    const budget = Math.max(settings.recallTimeoutMs + memory.forgetWaitMs, settings.forgetTimeoutMs) + 2000;
    const result = await bounded(work, budget, [payload?.signal, life.signal], null);
    if (!result?.context) return decision;
    if (typeof host?.createUserMessage !== 'function') {
      const loaded = await bounded(hostReady, 1000, [payload?.signal], null);
      if (typeof loaded?.createUserMessage !== 'function') {
        log.warnOnce('no-llm', 'recall and forget notes are off: @deepseek-ai/dsh-llm could not be loaded');
        return decision;
      }
    }
    // The block (or the forget note) goes in as plugin context just before
    // the user's message: the model sees it, the transcript shows it only as
    // a collapsed context row.
    const message = host.createUserMessage({
      content: [{ type: 'text', text: result.context }],
      source: { kind: 'plugin:quotemem', form: 'recall' },
    });
    const at = decision.messages.findIndex((m) => m?.source?.kind === 'user');
    return { ...decision, messages: [...decision.messages.slice(0, at), message, ...decision.messages.slice(at)] };
  }
  instance.preStep = preStep;

  ctx.on('agent/pre-step', async (payload, next) => {
    const decision = await next();
    try {
      return await preStep(payload, decision);
    } catch (error) {
      log.warnOnce(`pre-step:${errKind(error)}`, `automatic memory skipped this turn (${errKind(error)})`);
      return decision;
    }
  }, { prepend: true });

  // After compaction or /clear the earlier recall text is gone. DSH declares
  // these sources but does not emit them yet; the turn limit of
  // QUOTEMEM_RECALL_REPEAT_TURNS covers compaction until it does.
  ctx.on('agent/created', (payload) => {
    try {
      if (payload?.source !== 'compact' && payload?.source !== 'clear') return;
      const key = agentKeyOf(payload?.agent);
      if (key === null || !live.get(key)?.used) return;
      serial(key, () => memory.update(memory.sessionId(key), (state) => { state.recall = newRecallMemo(); }))
        .catch(() => {});
    } catch {
      // never fail agent creation
    }
  });

  // ------------------------------------------- turn ends: session/event

  const surface = (event) => (typeof host?.isAppendSurfaceEvent === 'function'
    ? host.isAppendSurfaceEvent(event)
    : event?.surfaceOp === undefined || event.surfaceOp === 'append');

  // An explicit forget tool: queue what is pending now, so a fact said in the
  // last few turns is stored before the forget runs.
  const cutForTool = (key) => serial(key, async () => {
    const jobs = await memory.update(memory.sessionId(key), (state) => memory.cutEarly(state));
    memory.enqueue(jobs);
    memory.flush();
  });

  const onEvent = (session, event) => {
    if (!event || typeof event.type !== 'string') return;
    if (isSubagent(session) && !settings.includeSubagents) return;
    const key = sessionKeyOf(session);
    const s = key === null ? undefined : live.get(key);
    if (!s?.used) return;
    if (event.type === 'assistant/message') {
      if (!surface(event)) return;
      const content = event.data?.message?.content;
      if (Array.isArray(content) && content.some((b) => b?.type === 'tool-call' && FORGET_TOOL.test(String(b.name ?? '')))) {
        cutForTool(key).catch(() => {});
      }
      if (event.data?.interrupted) return;
      const text = textOf(content);
      if (text) s.reply = { turn: event.data?.turn, text };
    } else if (event.type === 'turn/end') {
      const turn = Number.isInteger(event.data?.turn) ? event.data.turn : null;
      const kind = event.data?.reason?.kind;
      // Only the final reply of this very turn is kept.
      const reply = s.reply && (turn === null || s.reply.turn === undefined || s.reply.turn === turn) ? s.reply.text : null;
      s.reply = null;
      if (USER_TURN_ENDS.has(kind)) {
        // A turn the user stopped, or one that hit an error or the token
        // ceiling, still carries what the user said.
        serial(key, () => memory.reply({ session: key, text: reply, completed: kind === 'completed' })).catch(() => {});
      } else {
        // Blocked turns and the synthetic closers of crashed or forked logs
        // are not captured.
        serial(key, () => memory.update(memory.sessionId(key), (state) => { state.open = null; })).catch(() => {});
      }
      idleLater();
    }
  };

  if (settings.captureEvery > 0) {
    ctx.on('session/event', (session, event) => {
      try {
        onEvent(session, event);
      } catch (error) {
        log.warnOnce(`capture-bug:${errKind(error)}`, `capture skipped (${errKind(error)})`);
      }
    });

    // Before an explicit forget tool runs, wait until this user's batches
    // queued before it are stored, so it can find what was said last. When
    // they are not stored in time, prepare is refused with a short reason
    // instead of answering "nothing matched".
    ctx.on('tools/pre-execute', async (exec, next) => {
      const decision = await next();
      try {
        if (decision?.kind !== 'allow' || !FORGET_TOOL.test(String(exec?.name ?? ''))) return decision;
        const key = agentKeyOf(exec?.agent);
        if (key === null || !live.get(key)?.used) return decision;
        await bounded(cutForTool(key), 5000, [exec?.signal], null);
        const waitFor = memory.queue.entries().map((entry) => entry.file);
        const started = Date.now();
        const end = started + forgetToolWaitMs;
        while (Date.now() < end && !exec?.signal?.aborted && !closed && waitFor.some((file) => fs.existsSync(file))) {
          await sleep(100);
        }
        log.debug(`forget tool waited ${Date.now() - started} ms for ${waitFor.length} queued job(s)`);
        if (/prepare$/i.test(String(exec.name)) && waitFor.some((file) => fs.existsSync(file)) && !exec?.signal?.aborted) {
          log.debug('forget tool held: capture still pending');
          return { kind: 'deny', reason: PENDING_CAPTURE };
        }
      } catch (error) {
        log.warnOnce(`forget-tool:${errKind(error)}`, `could not save the conversation before a forget (${errKind(error)})`);
      }
      return decision;
    });
  }

  if (settings.idleFlushMs > 0) {
    ctx.on('agent/status', (payload) => {
      try {
        if (payload?.status === 'idle') idleLater();
      } catch {
        // never fail a status change
      }
    });
  }

  // -------------------------------------------------- session end, shutdown

  // Session end: queue the partial batch and start the background sender,
  // which outlives DSH.
  ctx.on('session/disposed', (session) => {
    try {
      const key = sessionKeyOf(session);
      const s = key === null ? undefined : live.get(key);
      if (!s) return;
      if (!s.used) {
        live.delete(key);
        return;
      }
      // The entry stays until the end is queued, so a shutdown meanwhile
      // still queues it synchronously; whichever runs second finds nothing.
      serial(key, () => memory.end({ session: key })).catch(() => {}).finally(() => {
        if (live.get(key) === s) live.delete(key);
      });
    } catch (error) {
      log.warnOnce(`end:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    }
  });

  // Shutdown: DSH may exit right after disposing the plugin, so the last
  // turns are queued synchronously and handed to the background sender.
  // What still misses the queue stays in the session files; the next start
  // queues it after the idle time.
  const endAllSync = (why) => {
    let queued = 0;
    for (const [key, s] of live) {
      live.delete(key);
      if (s.used) queued += memory.endSync({ session: key });
    }
    if (queued > 0) log.debug(`capture ${queued} batch(es) on ${why}`);
    memory.flush({ sync: true });
  };

  const onExit = () => endAllSync('exit');
  process.on('exit', onExit);

  // Batches and turns an earlier run left (a crash, a quit before the
  // sender finished) are queued and sent again.
  memory.sweep().then(() => memory.flush()).catch((error) => {
    log.warnOnce(`start:${errKind(error)}`, `could not resume saved memory batches (${errKind(error)})`);
  });

  const dispose = () => {
    if (closed) return;
    closed = true;
    clearTimeout(idleTimer);
    process.off('exit', onExit);
    try {
      endAllSync('unload');
    } finally {
      life.abort();
    }
  };
  instance.dispose = dispose;
  ctx.effect(() => dispose, 'quotemem-automem.lifecycle');
}
