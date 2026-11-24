// QuoteMem automatic memory for OpenCode: one plugin module for both plugin
// APIs. OpenCode 1.x (`opencode-ai`) calls `server(input, options)` and gets
// a hooks object; OpenCode 2.x (`@opencode/cli`) calls `setup(ctx)`, which
// registers session hooks and subscribes to events.
//
// What the user never sees:
//   Capture: the user's own text of each turn and the final assistant text
//     are recorded; every QUOTEMEM_CAPTURE_EVERY user turns (default 3), at a
//     session's end and when OpenCode shuts down, a batch is queued on disk
//     and a background sender posts it to QuoteMem /memory/add.
//   Recall: a memory-related user message is searched under a short deadline
//     and the hits are added to the model request only (an ephemeral part of
//     the user message in 1.x, a system part in 2.x), marked as retrieved,
//     untrusted data. The stored conversation and the transcript never hold it.
//   Forget: a message asking to forget something is carried out through
//     QuoteMem's forget by statement after the earlier turns are stored, and
//     the model gets one short note so it can confirm.
//
// The work itself is the shared adapter core (./core, a copy of
// integrations/core): this file maps OpenCode's hooks and events onto
// HookMemory. Every hook catches its own errors and bounds its own wait, so
// a slow or stopped QuoteMem never breaks or blocks OpenCode. Nothing is
// written to stdout or stderr (it would corrupt the TUI); warnings go to
// <state dir>/quotemem.log and never hold the token or message text.
import fs from 'node:fs';
import path from 'node:path';

import { homeDir } from './core/proc.mjs';

import {
  FORGET_HEADER,
  FORGET_TOOL_WAIT_MS,
  HookMemory,
  OFF,
  RECALL_HEADER,
  errKind,
  fileSink,
  makeLog,
  resolveSettings,
  runFlusher,
} from './core/index.mjs';
import { hostEnv } from './env.js';

export const PLUGIN_ID = 'quotemem';
const HOST = 'opencode';
const LABEL = 'OpenCode plugin';
// The explicit forget tools of the QuoteMem MCP server, under any prefix
// OpenCode gives MCP tools (`quotemem_quotemem_forget_prepare`).
const FORGET_TOOL = /(^|[^a-z0-9])quotemem_forget_(prepare|confirm)$/i;
const PENDING_CAPTURE = 'The latest conversation is still being saved to memory, so a forget now could miss it. '
  + 'Tell the user it will be possible in a minute, or try again shortly.';

// Thrown from tool.execute.before to refuse an explicit forget prepare while
// the earlier turns are still being saved: OpenCode fails that tool call with
// this message, which the model reads.
class ForgetHeld extends Error {}
const LOOKUP_MS = 1000;
const REPLY_LOOKUP_MS = 2000;
const DISPOSE_MS = 3000;
const REKICK_MS = 30_000;
const COMMAND_WINDOW_MS = 30_000;
const MAX_SESSIONS = 500;
const MAX_ASSISTANT_IDS = 200;
const MAX_PARENTS = 5000;
// OpenCode may exit on these without unloading the plugin.
const EXIT_SIGNALS = ['SIGINT', 'SIGTERM', 'SIGHUP'];
const V2_EVENTS = [
  'session.created',
  'session.deleted',
  'session.text.ended',
  'session.execution.succeeded',
  'session.execution.failed',
  'session.execution.interrupted',
  'session.idle',
];

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms).unref?.(); });

/** Resolve with the promise's value, or reject once `ms` passes. */
function withDeadline(promise, ms) {
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error('deadline'), { code: 'ETIMEDOUT' })), ms);
    timer.unref?.();
  });
  return Promise.race([Promise.resolve(promise), deadline]).finally(() => clearTimeout(timer));
}

function executable(file) {
  try {
    fs.accessSync(file, fs.constants.X_OK);
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

function which(name, env) {
  const exts = process.platform === 'win32' ? ['.exe', ''] : [''];
  for (const dir of String(env.PATH ?? env.Path ?? '').split(path.delimiter)) {
    if (!dir) continue;
    for (const ext of exts) {
      const file = path.join(dir, `${name}${ext}`);
      if (executable(file)) return file;
    }
  }
  return null;
}

/**
 * The JavaScript runtime that runs the detached background sender, or null
 * to send from inside OpenCode instead. OpenCode ships as one compiled Bun
 * executable, so its own `process.execPath` would start OpenCode, not run a
 * script: a `node` (or `bun`) on PATH is used then. QUOTEMEM_NODE names one
 * explicitly, or `off` keeps the sender in-process.
 */
export function resolveRuntime(env = process.env, { execPath = process.execPath, bun = process.versions?.bun } = {}) {
  const forced = String(env.QUOTEMEM_NODE ?? '').trim();
  if (forced) {
    if (OFF.has(forced.toLowerCase())) return null;
    return executable(forced) ? forced : null;
  }
  const base = path.basename(String(execPath ?? '')).toLowerCase().replace(/\.exe$/, '');
  if (execPath && (!bun || base === 'node' || base === 'nodejs' || base === 'bun')) return execPath;
  return which('node', env) ?? which('bun', env);
}

/** The user's own words of a 1.x user message: its non-synthetic, non-ignored text parts. */
export function userTextV1(parts) {
  if (!Array.isArray(parts)) return '';
  return parts
    .filter((part) => part && part.type === 'text' && typeof part.text === 'string' && !part.synthetic && !part.ignored)
    .map((part) => part.text)
    .join('\n')
    .trim();
}

/** The final assistant text of a 1.x `session.messages` answer ([{info, parts}]). */
export function lastReplyV1(list) {
  const items = Array.isArray(list) ? list : [];
  for (let i = items.length - 1; i >= 0; i -= 1) {
    const info = items[i]?.info;
    if (info?.role === 'user') return '';
    if (info?.role !== 'assistant' || info.error) continue;
    const text = userTextV1(items[i].parts);
    if (text) return text;
  }
  return '';
}

/** The final assistant text of a 2.x `session.context` answer (SessionMessageInfo[]). */
export function lastReplyV2(list) {
  const items = Array.isArray(list) ? list : [];
  for (let i = items.length - 1; i >= 0; i -= 1) {
    const item = items[i];
    if (item?.type === 'user') return '';
    if (item?.type !== 'assistant' || item.error) continue;
    const text = (Array.isArray(item.content) ? item.content : [])
      .filter((block) => block?.type === 'text' && typeof block.text === 'string')
      .map((block) => block.text)
      .join('\n')
      .trim();
    if (text) return text;
  }
  return '';
}

// Text QuoteMem itself put in the context is never captured or searched.
function ownText(text) {
  const t = String(text ?? '').trimStart();
  return t.startsWith(RECALL_HEADER) || t.startsWith(FORGET_HEADER);
}

/**
 * The adapter state of one OpenCode instance: per-session turn tracking on
 * top of HookMemory, whose own state lives in files.
 */
export class OpenCodeMemory {
  /**
   * @param {object} options
   * @param {object} options.settings - from resolveSettings (problem must be null).
   * @param {object} options.log - from makeLog with a file sink.
   * @param {object} [options.env] - the environment of the background sender.
   * @param {string|null} [options.runtime] - see resolveRuntime; null sends in-process.
   */
  constructor({ settings, log, env = process.env, runtime = null }) {
    this.settings = settings;
    this.log = log;
    this.env = env;
    this.runtime = runtime;
    this.memory = new HookMemory({ settings, log, host: HOST, label: LABEL, env });
    // HookMemory starts its flusher with this process's execPath; route it
    // through the runtime chosen here.
    this.memory.flush = () => this.flush();
    this.forgetToolWaitMs = FORGET_TOOL_WAIT_MS;
    this.sessions = new Map();
    this.parents = new Map();
    this.stop = new AbortController();
    this.pump = null;
    this.rekick = null;
    this.idleTimer = null;
    this.closed = false;
    this.messagesHook = false;
    // Set by the host wiring: (sessionID) => session info with parentID, and
    // (sessionID) => the final assistant text when the events carried none.
    this.lookup = null;
    this.fetchReply = null;
    this.unwatchExit = () => {};
  }

  /**
   * Queue the rest of every session at once, before OpenCode exits.
   * `opencode serve` 1.x exits on SIGINT or SIGTERM without unloading the
   * plugin, so dispose() never runs then. Synchronous, so it runs in an
   * `exit` handler; the detached sender outlives OpenCode.
   */
  endAllSync(why) {
    let queued = 0;
    for (const [id, s] of this.sessions) {
      if (s.used) queued += this.memory.endSync({ session: id });
    }
    if (queued > 0) this.log.debug(`queued ${queued} batch(es) on ${why}`);
    if (this.runtime) this.memory.queue.startFlusher({ env: this.env, execPath: this.runtime, sync: true });
    return queued;
  }

  /**
   * Run endAllSync when the process exits or gets a stop signal. Every
   * adapter of the process shares one set of listeners (exitWatch).
   * @param {object} [target] - the process (a stand-in for tests).
   */
  watchExit(target = process) {
    this.unwatchExit();
    try {
      const watch = exitWatch(target);
      watch.adapters.add(this);
      this.unwatchExit = () => {
        watch.adapters.delete(this);
        if (watch.adapters.size === 0) watch.detach();
        this.unwatchExit = () => {};
      };
    } catch (error) {
      this.log.warnOnce(`exit-watch:${errKind(error)}`, `could not watch for OpenCode's exit (${errKind(error)})`);
    }
  }

  /** Queue what earlier runs left (an `opencode run` that exited) and send it. */
  start() {
    void (async () => {
      try {
        await this.memory.sweep();
        this.flush();
      } catch (error) {
        this.log.warnOnce(`start:${errKind(error)}`, `could not resume saved memory batches (${errKind(error)})`);
      }
    })();
  }

  /** Start sending the disk queue: a detached sender, else an in-process one. */
  flush() {
    try {
      if (this.runtime) return this.memory.queue.startFlusher({ env: this.env, execPath: this.runtime });
      if (this.closed) return false;
      return this.pumpInProcess();
    } catch (error) {
      this.log.warnOnce(`flush:${errKind(error)}`, `could not start sending memory batches (${errKind(error)})`);
      return false;
    }
  }

  pumpInProcess() {
    if (this.pump || this.stop.signal.aborted) return false;
    if (this.memory.queue.entries().length === 0) return false;
    this.pump = runFlusher(this.settings, this.log, { signal: this.stop.signal })
      .then((outcome) => {
        if (outcome === 'waiting' || outcome === 'busy') this.later();
      })
      .catch((error) => this.log.warnOnce(`pump:${errKind(error)}`, `memory sender error (${errKind(error)})`))
      .finally(() => { this.pump = null; });
    return true;
  }

  later() {
    if (this.rekick || this.stop.signal.aborted) return;
    this.rekick = setTimeout(() => {
      this.rekick = null;
      this.flush();
    }, REKICK_MS);
    this.rekick.unref?.();
  }

  // OpenCode stays up while the user is away: queue a partial batch once
  // the sessions have been idle for QUOTEMEM_IDLE_FLUSH_MS.
  idleLater() {
    if (this.settings.idleFlushMs <= 0 || this.closed) return;
    clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      this.memory.sweep().catch(() => {});
    }, this.settings.idleFlushMs + 250);
    this.idleTimer.unref?.();
  }

  session(id) {
    let s = this.sessions.get(id);
    if (!s) {
      s = { open: false, used: false, turn: null, failed: false, reply: null, block: null, command: null, assistantIds: new Set(), chain: Promise.resolve() };
      this.sessions.set(id, s);
      if (this.sessions.size > MAX_SESSIONS) {
        for (const [key, value] of this.sessions) {
          if (this.sessions.size <= MAX_SESSIONS) break;
          if (!value.open && key !== id) this.sessions.delete(key);
        }
      }
    }
    return s;
  }

  // Run one step of a session after the previous one, so a turn's end is
  // recorded before the next message even when OpenCode does not await it.
  serial(id, fn) {
    const s = this.session(id);
    const run = s.chain.then(() => fn());
    s.chain = run.catch(() => {});
    return run;
  }

  promptBudget() {
    // A prompt may first wait for a queued forget before it recalls.
    return Math.max(this.settings.recallTimeoutMs + this.memory.forgetWaitMs, this.settings.forgetTimeoutMs) + 2000;
  }

  /** Remember whether a session is a subagent's (it has a parent). */
  noteSession(id, parentID) {
    if (typeof id === 'string' && id) this.rememberParent(id, Boolean(parentID));
  }

  rememberParent(id, child) {
    this.parents.delete(id);
    this.parents.set(id, child);
    if (this.parents.size > MAX_PARENTS) this.parents.delete(this.parents.keys().next().value);
  }

  async subagent(id) {
    if (this.settings.includeSubagents) return false;
    if (this.parents.has(id)) return this.parents.get(id);
    let child = false;
    if (this.lookup) {
      try {
        const info = await withDeadline(this.lookup(id), LOOKUP_MS);
        child = Boolean(info?.parentID);
      } catch (error) {
        this.log.warnOnce(`lookup:${errKind(error)}`, `could not tell whether a session is a subagent's (${errKind(error)}); it is treated as a main session`);
      }
    }
    this.rememberParent(id, child);
    return child;
  }

  /** A slash command started; its expanded prompt is not captured. */
  commandStarted(id) {
    if (typeof id === 'string' && id) this.session(id).command = Date.now();
  }

  /**
   * The user sent a message.
   * @returns {Promise<{context: string|null, kind: string|null}|null>}
   */
  async userMessage({ sessionID, messageID = null, text }) {
    try {
      if (typeof sessionID !== 'string' || !sessionID || this.closed) return null;
      const s = this.session(sessionID);
      if (s.command && Date.now() - s.command < COMMAND_WINDOW_MS) {
        s.command = null;
        return null;
      }
      s.command = null;
      const said = String(text ?? '').trim();
      if (!said || ownText(said)) return null;
      if (await this.subagent(sessionID)) return null;
      return await this.serial(sessionID, async () => {
        s.open = true;
        s.used = true;
        s.turn = messageID ?? null;
        s.failed = false;
        s.reply = null;
        s.block = null;
        let result = null;
        try {
          result = await withDeadline(this.memory.prompt({ session: sessionID, turn: s.turn, text: said }), this.promptBudget());
        } catch (error) {
          this.log.warnOnce(`prompt:${errKind(error)}`, `automatic memory skipped this turn (${errKind(error)})`);
        }
        if (result?.context && s.open && s.turn === (messageID ?? null)) {
          s.block = { messageID: messageID ?? null, text: result.context, kind: result.kind, via: null };
        }
        return result;
      });
    } catch (error) {
      this.log.warnOnce(`message:${errKind(error)}`, `automatic memory skipped this turn (${errKind(error)})`);
      return null;
    }
  }

  /** Note an assistant message id (1.x events), so its text parts can be told apart. */
  assistantMessage(sessionID, messageID) {
    const s = this.sessions.get(sessionID);
    if (!s?.open || typeof messageID !== 'string') return;
    s.assistantIds.add(messageID);
    if (s.assistantIds.size > MAX_ASSISTANT_IDS) s.assistantIds.delete(s.assistantIds.values().next().value);
  }

  isAssistant(sessionID, messageID) {
    return Boolean(this.sessions.get(sessionID)?.assistantIds.has(messageID));
  }

  /** A finished assistant text part; the last assistant message with text is the reply. */
  assistantText({ sessionID, messageID, partID, text }) {
    const s = this.sessions.get(sessionID);
    if (!s?.open || typeof text !== 'string' || !text.trim() || ownText(text)) return;
    if (!s.reply || s.reply.messageID !== messageID) s.reply = { messageID, parts: new Map() };
    s.reply.parts.set(partID ?? s.reply.parts.size, text);
  }

  /** The turn was aborted or failed: its reply is not captured. */
  turnFailed(sessionID) {
    const s = this.sessions.get(sessionID);
    if (s?.open) s.failed = true;
  }

  /** The turn ended (the session went idle, or the execution finished). */
  turnEnd(sessionID, { failed = false } = {}) {
    const s = this.sessions.get(sessionID);
    if (!s?.open) return Promise.resolve();
    return this.serial(sessionID, async () => {
      if (!s.open) return;
      const turn = s.turn;
      const completed = !(failed || s.failed);
      let text = s.reply ? [...s.reply.parts.values()].join('\n').trim() : '';
      s.open = false;
      s.block = null;
      s.reply = null;
      s.failed = false;
      s.assistantIds.clear();
      if (completed && !text && this.fetchReply) {
        try {
          text = String(await withDeadline(this.fetchReply(sessionID), REPLY_LOOKUP_MS) ?? '').trim();
        } catch (error) {
          this.log.warnOnce(`reply-lookup:${errKind(error)}`, `could not read the final reply (${errKind(error)})`);
        }
      }
      await this.memory.reply({ session: sessionID, turn, text: text || null, completed });
      this.idleLater();
    }).catch((error) => {
      this.log.warnOnce(`turn-end:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    });
  }

  /** The session was deleted: queue what is left. */
  async endSession(sessionID) {
    const s = this.sessions.get(sessionID);
    if (!s?.used) return;
    try {
      await this.serial(sessionID, () => this.memory.end({ session: sessionID }));
    } catch (error) {
      this.log.warnOnce(`end:${errKind(error)}`, `capture skipped (${errKind(error)})`);
    }
    this.sessions.delete(sessionID);
  }

  /**
   * An explicit QuoteMem forget tool is about to run: queue the session's
   * pending turns and wait (up to FORGET_TOOL_WAIT_MS) until this user's
   * queued batches are stored, so the forget can find them.
   * @returns {Promise<boolean>} whether batches are still not stored: a
   *   prepare run now could miss them for good, since the service's in-chat
   *   check may not read the user's words.
   */
  async beforeForgetTool(sessionID) {
    const s = this.sessions.get(sessionID);
    if (!s?.used) return false;
    try {
      return await this.serial(sessionID, async () => {
        const memory = this.memory;
        const jobs = await memory.update(memory.sessionId(sessionID), (state) => memory.cutEarly(state));
        memory.enqueue(jobs);
        this.flush();
        const waitFor = memory.queue.entries().map((entry) => entry.file);
        const end = Date.now() + this.forgetToolWaitMs;
        while (Date.now() < end && waitFor.some((file) => fs.existsSync(file))) await sleep(50);
        return waitFor.some((file) => fs.existsSync(file));
      });
    } catch (error) {
      this.log.warnOnce(`forget-tool:${errKind(error)}`, `could not save the conversation before a forget (${errKind(error)})`);
      return false;
    }
  }

  /** The text QuoteMem adds to this turn's model request, or null. */
  inject(sessionID, messageID) {
    const s = this.sessions.get(sessionID);
    if (!s?.open || !s.block) return null;
    if (messageID !== undefined && s.block.messageID && messageID !== s.block.messageID) return null;
    return s.block.text;
  }

  // ------------------------------------------------------------ 1.x wiring

  /** experimental.chat.messages.transform: add the block to the turn's user message. */
  injectMessagesV1(messages) {
    this.messagesHook = true;
    if (!Array.isArray(messages)) return;
    let last = null;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.info?.role === 'user') {
        last = messages[i];
        break;
      }
    }
    if (!last || !Array.isArray(last.parts)) return;
    const { id, sessionID } = last.info;
    const text = this.inject(sessionID, id);
    if (!text) return;
    if (last.parts.some((part) => part?.type === 'text' && part.text === text)) return;
    // The loop reloads the messages on every step, so this part reaches the
    // model request only; it is never stored or shown (synthetic).
    last.parts.push({ id: `prt_quotemem_${id}`, sessionID, messageID: id, type: 'text', text, synthetic: true });
    this.sessions.get(sessionID).block.via = 'messages';
  }

  /** experimental.chat.system.transform: the fallback when the messages hook is missing. */
  injectSystemV1(sessionID, system) {
    if (this.messagesHook || !Array.isArray(system)) return;
    const text = this.inject(sessionID);
    if (!text || system.includes(text)) return;
    system.push(text);
    this.sessions.get(sessionID).block.via = 'system';
  }

  async eventV1(event) {
    const type = event?.type;
    const p = event?.properties ?? {};
    switch (type) {
      case 'session.created':
      case 'session.updated':
        this.noteSession(p.info?.id, p.info?.parentID);
        break;
      case 'session.deleted':
        if (p.info?.id) await this.endSession(p.info.id);
        break;
      case 'session.error':
        if (p.sessionID) this.turnFailed(p.sessionID);
        break;
      case 'message.updated': {
        const info = p.info;
        if (info?.role !== 'assistant') break;
        this.assistantMessage(info.sessionID, info.id);
        if (info.error) this.turnFailed(info.sessionID);
        break;
      }
      case 'message.part.updated': {
        const part = p.part;
        if (part?.type === 'text' && !part.synthetic && !part.ignored && part.time?.end && this.isAssistant(part.sessionID, part.messageID)) {
          this.assistantText({ sessionID: part.sessionID, messageID: part.messageID, partID: part.id, text: part.text });
        }
        break;
      }
      case 'session.idle':
        if (p.sessionID) await this.turnEnd(p.sessionID);
        break;
      default:
        break;
    }
  }

  // ------------------------------------------------------------ 2.x wiring

  async eventV2(event) {
    const data = event?.data ?? {};
    switch (event?.type) {
      case 'session.created':
        this.noteSession(data.sessionID, data.parentID);
        break;
      case 'session.deleted':
        if (data.sessionID) await this.endSession(data.sessionID);
        break;
      case 'session.text.ended':
        this.assistantText({ sessionID: data.sessionID, messageID: data.assistantMessageID, partID: data.ordinal, text: data.text });
        break;
      case 'session.execution.failed':
      case 'session.execution.interrupted':
        if (data.sessionID) await this.turnEnd(data.sessionID, { failed: true });
        break;
      case 'session.execution.succeeded':
      case 'session.idle':
        if (data.sessionID) await this.turnEnd(data.sessionID);
        break;
      default:
        break;
    }
  }

  async listenV2(subscribe, signal) {
    try {
      const stream = subscribe({ types: V2_EVENTS, signal });
      for await (const event of stream) {
        if (signal.aborted) break;
        try {
          await this.eventV2(event);
        } catch (error) {
          this.log.warnOnce(`event:${errKind(error)}`, `memory event skipped (${errKind(error)})`);
        }
      }
    } catch (error) {
      if (!signal.aborted) this.log.warnOnce(`events:${errKind(error)}`, `OpenCode events stopped; final replies are no longer captured (${errKind(error)})`);
    }
  }

  // --------------------------------------------------------------- unload

  /** OpenCode unloads the plugin: queue every session's rest and hand it to the background sender. */
  async dispose() {
    if (this.closed) return;
    this.closed = true;
    this.unwatchExit();
    clearTimeout(this.rekick);
    clearTimeout(this.idleTimer);
    const ids = [...this.sessions].filter(([, s]) => s.used).map(([id]) => id);
    try {
      await withDeadline(Promise.allSettled(ids.map((id) => this.serial(id, () => this.memory.end({ session: id })))), DISPOSE_MS);
    } catch {
      // what is not queued yet stays in the session files; a later start sweeps it
    }
    this.sessions.clear();
    this.stop.abort();
    if (this.pump) await withDeadline(this.pump, 2000).catch(() => {});
    if (this.runtime) {
      this.flush();
    } else if (this.memory.queue.entries().length > 0) {
      this.log.warnOnce('left-queued', 'memory batches stay queued until OpenCode starts again (no node or bun on PATH for the background sender; set QUOTEMEM_NODE)');
    }
  }
}

const EXIT_WATCH = Symbol.for('quotemem.opencode.exitWatch');

/**
 * The one set of exit and signal listeners of a process, shared by every
 * adapter in it (one per OpenCode project instance, and by every copy of
 * this module). On `exit` each adapter queues its sessions. On a stop
 * signal they do the same, then the listeners are removed and the signal is
 * raised again: OpenCode's own signal handling (a "signal-exit" style
 * listener) holds back while another listener exists, so without the
 * second signal OpenCode would not stop at all.
 */
function exitWatch(target) {
  if (target[EXIT_WATCH]) return target[EXIT_WATCH];
  const watch = { adapters: new Set(), attached: true };
  const onExit = () => {
    for (const adapter of watch.adapters) adapter.endAllSync('exit');
  };
  const onSignal = (signal) => {
    for (const adapter of [...watch.adapters]) adapter.endAllSync(signal);
    watch.detach();
    setImmediate(() => {
      try {
        target.kill(target.pid, signal);
      } catch {
        // the process is already going
      }
    });
  };
  watch.detach = () => {
    if (!watch.attached) return;
    watch.attached = false;
    target.off('exit', onExit);
    for (const signal of EXIT_SIGNALS) target.off(signal, onSignal);
    if (target[EXIT_WATCH] === watch) delete target[EXIT_WATCH];
  };
  target.on('exit', onExit);
  for (const signal of EXIT_SIGNALS) target.on(signal, onSignal);
  target[EXIT_WATCH] = watch;
  return watch;
}

/**
 * Build the adapter for one plugin instance, or null when automatic memory
 * must stay off (no token, a bad URL); the reason goes to the log file.
 * @param {object} options - the plugin options from OpenCode's config (behaviour settings only).
 * @param {object} baseEnv - the environment (QUOTEMEM_*), completed from
 *   ~/.quotemem/env and QUOTEMEM_TOKEN_FILE (see env.js).
 * @param {{runtime?: string|null, watchExit?: boolean, process?: object}} overrides - for tests.
 */
export function createAdapter(options = {}, baseEnv = process.env, overrides = {}) {
  const { env, notes } = hostEnv(baseEnv);
  const settings = resolveSettings(options && typeof options === 'object' ? options : {}, env, {
    defaultStateDir: path.join(homeDir(), '.quotemem', HOST),
    configName: 'OpenCode plugin options',
  });
  const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
  for (const warning of [...notes, ...settings.warnings]) log.warnOnce(`settings:${warning}`, warning);
  if (settings.problem) {
    log.warnOnce('problem', `automatic memory is off: ${settings.problem}`);
    return null;
  }
  const runtime = 'runtime' in overrides ? overrides.runtime : resolveRuntime(env);
  if (!runtime) log.warnOnce('runtime', 'no node or bun found for the background sender; memory batches are sent from inside OpenCode');
  const adapter = new OpenCodeMemory({ settings, log, env, runtime });
  if (overrides.forgetToolWaitMs !== undefined) adapter.forgetToolWaitMs = overrides.forgetToolWaitMs;
  adapter.start();
  if (overrides.watchExit !== false) adapter.watchExit(overrides.process ?? process);
  return adapter;
}

// Every hook catches its own errors: a failed hook would fail OpenCode's turn.
// Only a refused forget prepare (ForgetHeld) fails its tool call on purpose.
function guard(adapter, name, fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      if (error instanceof ForgetHeld) throw error;
      adapter?.log.warnOnce(`hook:${name}:${errKind(error)}`, `${name} skipped (${errKind(error)})`);
    }
  };
}

/**
 * OpenCode 1.x entry point (`@opencode-ai/plugin` Hooks).
 * @param {{client?: object}} input - OpenCode's plugin input.
 * @param {object} [options] - the plugin options from the `plugin` config entry.
 */
export function server(input, options) {
  return serverWith(input, options);
}

/** server() with test overrides: `env`, `runtime`. */
export async function serverWith(input = {}, options = {}, overrides = {}) {
  let adapter = null;
  try {
    adapter = createAdapter(options, overrides.env ?? process.env, overrides);
  } catch {
    adapter = null;
  }
  if (!adapter) return {};
  const client = input?.client;
  if (typeof client?.session?.get === 'function') {
    adapter.lookup = async (id) => {
      const res = await client.session.get({ path: { id } });
      return res?.data ?? res;
    };
  }
  if (typeof client?.session?.messages === 'function') {
    adapter.fetchReply = async (id) => {
      const res = await client.session.messages({ path: { id } });
      return lastReplyV1(res?.data ?? res);
    };
  }
  return {
    'chat.message': guard(adapter, 'chat.message', async (hookInput, output) => {
      const message = output?.message;
      await adapter.userMessage({
        sessionID: hookInput?.sessionID ?? message?.sessionID,
        messageID: message?.id ?? hookInput?.messageID ?? null,
        text: userTextV1(output?.parts),
      });
    }),
    'experimental.chat.messages.transform': guard(adapter, 'messages.transform', async (_hookInput, output) => {
      adapter.injectMessagesV1(output?.messages);
    }),
    'experimental.chat.system.transform': guard(adapter, 'system.transform', async (hookInput, output) => {
      adapter.injectSystemV1(hookInput?.sessionID, output?.system);
    }),
    'experimental.text.complete': guard(adapter, 'text.complete', async (hookInput, output) => {
      adapter.assistantText({ sessionID: hookInput?.sessionID, messageID: hookInput?.messageID, partID: hookInput?.partID, text: output?.text });
    }),
    'command.execute.before': guard(adapter, 'command.execute.before', async (hookInput) => {
      adapter.commandStarted(hookInput?.sessionID);
    }),
    'tool.execute.before': guard(adapter, 'tool.execute.before', async (hookInput) => {
      const tool = String(hookInput?.tool ?? '');
      if (!FORGET_TOOL.test(tool)) return;
      const pending = await adapter.beforeForgetTool(hookInput?.sessionID);
      if (pending && /prepare$/i.test(tool)) {
        adapter.log.debug('forget tool held: capture still pending');
        throw new ForgetHeld(PENDING_CAPTURE);
      }
    }),
    event: guard(adapter, 'event', async ({ event } = {}) => {
      await adapter.eventV1(event);
    }),
    dispose: guard(adapter, 'dispose', async () => {
      await adapter.dispose();
    }),
  };
}

/**
 * OpenCode 2.x entry point (`@opencode/plugin` Promise API).
 * @param {object} ctx - the plugin context.
 * @returns {Promise<(() => Promise<void>) | undefined>} the cleanup.
 */
export function setup(ctx) {
  return setupWith(ctx);
}

/** setup() with test overrides: `env`, `runtime`. */
export async function setupWith(ctx, overrides = {}) {
  // OpenCode 1.x calls setup() as well as server(), with no session hooks:
  // server() serves it, and a second adapter here would only log noise.
  if (typeof ctx?.session?.hook !== 'function') return undefined;
  let adapter = null;
  try {
    adapter = createAdapter(ctx?.options ?? {}, overrides.env ?? process.env, overrides);
  } catch {
    adapter = null;
  }
  if (!adapter) return undefined;
  const session = ctx?.session;
  if (typeof session?.get === 'function') adapter.lookup = (id) => session.get({ sessionID: id });
  if (typeof session?.context === 'function') adapter.fetchReply = async (id) => lastReplyV2(await session.context({ sessionID: id }));
  const registrations = [];
  const hook = async (name, callback) => {
    try {
      registrations.push(await session.hook(name, guard(adapter, name, callback)));
    } catch (error) {
      adapter.log.warnOnce(`register:${name}`, `OpenCode did not accept the "${name}" session hook (${errKind(error)}); that part of automatic memory is off`);
    }
  };
  // Never edit event.prompt: what the hook leaves there is what OpenCode stores.
  await hook('prompt', async (event) => {
    await adapter.userMessage({ sessionID: event?.sessionID, messageID: event?.messageID ?? null, text: event?.prompt?.text });
  });
  await hook('context', async (event) => {
    const text = adapter.inject(event?.sessionID);
    if (text && Array.isArray(event?.system) && !event.system.some((part) => part?.text === text)) {
      event.system.push({ type: 'text', text });
    }
  });
  const stop = new AbortController();
  if (typeof ctx?.event?.subscribe === 'function') {
    void adapter.listenV2((options) => ctx.event.subscribe(options), stop.signal);
  } else {
    adapter.log.warnOnce('no-events', 'OpenCode offers no event stream; replies are not captured (user messages still are)');
  }
  return async () => {
    stop.abort();
    for (const registration of registrations) {
      try {
        await registration?.dispose?.();
      } catch {
        // already gone
      }
    }
    await adapter.dispose();
  };
}
