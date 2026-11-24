// QuoteMem for OpenClaw: an in-process plugin that gives every conversation
// long-term memory without any extra step.
//
//   before_prompt_build  silent forget when the user asks to forget, else a
//                        gated recall added as prependContext
//   agent_end            records the final assistant reply; every
//                        `captureEvery` user turns a batch is queued
//   session_end          queues what is left of the session
//   gateway_stop         starts the background flusher for anything queued
//
// Batches are written to a disk queue and sent by a detached background
// flusher, so a turn never waits for /memory/add and nothing is lost when the
// Gateway stops. Every handler swallows its own errors: a slow or missing
// QuoteMem service only means a turn without memory. The MCP tools come from
// the manifest (openclaw.plugin.json `mcpServers`).
import path from 'node:path';

import { FORGET_WAIT_MS, HookMemory, fileSink, homeDir, makeLog, resolveSettings, sha16 } from './core/index.mjs';
import { hostEnv } from './env.mjs';

const HOST = 'openclaw';
const MAX_USER_ID_CHARS = 256;

let define = (entry) => entry;
try {
  ({ definePluginEntry: define } = await import('openclaw/plugin-sdk/plugin-entry'));
} catch {
  // Loaded outside OpenClaw (tests) or from a host without the helper: the
  // plain entry object is what the helper returns.
}

/** The text parts of a message's content (string or content parts); reasoning is skipped. */
export function textOf(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((part) => part && (part.type === 'text' || part.type === 'output_text') && typeof part.text === 'string')
    .map((part) => part.text)
    .join('\n');
}

/** The text of the last message with `role`, or null. */
export function lastText(messages, role) {
  if (!Array.isArray(messages)) return null;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message?.role !== role) continue;
    const text = textOf(message.content ?? message.text).trim();
    if (text) return text;
  }
  return null;
}

function sessionOf(event, ctx) {
  const value = ctx?.sessionKey ?? ctx?.sessionId ?? event?.sessionKey ?? event?.sessionId;
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

/**
 * Register QuoteMem's hooks on an OpenClaw plugin API.
 * @param {object} api - the OpenClaw plugin API (`on`, `pluginConfig`, `logger`).
 * @param {{env?: object}} [options] - the environment (default process.env).
 * @returns {{settings: object, memories: Map}|null} null when memory stays off.
 */
export function registerQuoteMem(api, { env = process.env } = {}) {
  const config = api?.pluginConfig && typeof api.pluginConfig === 'object' ? api.pluginConfig : {};
  const { env: full, notes } = hostEnv(env);
  const settings = resolveSettings(config, full, {
    defaultStateDir: path.join(homeDir(full), '.quotemem', HOST),
    configName: 'OpenClaw plugin config (plugins.entries.quotemem.config)',
  });
  const warn = (message) => {
    try { api?.logger?.warn?.(`quotemem: ${message}`); } catch { /* never fail the host */ }
  };
  for (const message of [...notes, ...settings.warnings]) warn(message);
  if (settings.problem) {
    warn(`automatic memory is off: ${settings.problem}`);
    return null;
  }
  const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
  const perSender = config.userScope === 'sender';
  const owners = new Set((Array.isArray(config.ownerSenders) ? config.ownerSenders : []).map((id) => String(id)));
  const memories = new Map();
  const skipped = new Set();

  // One memory per user id: the owner's by default, or each channel sender's.
  // In owner mode a channel message is the owner's only when its sender is
  // listed in ownerSenders (as `<sender id>` or `<channel>:<sender id>`) or
  // the host marks it as the owner's: anyone else in a group or a DM could
  // otherwise read and forget the owner's memories, and add to them.
  const memoryFor = (ctx) => {
    let userId = settings.userId;
    const sender = ctx?.senderId ?? ctx?.channelContext?.sender?.id;
    const hasSender = sender !== undefined && sender !== null && sender !== '';
    const channelSender = `${ctx?.channel ?? ctx?.messageProvider ?? 'channel'}:${sender}`;
    if (perSender) {
      if (!hasSender) return null;
      userId = channelSender.slice(0, MAX_USER_ID_CHARS);
    } else if (hasSender && !(ctx?.senderIsOwner === true || owners.has(String(sender)) || owners.has(channelSender))) {
      if (!skipped.has(channelSender) && skipped.size < 100) {
        skipped.add(channelSender);
        warn(`no memory for channel sender ${channelSender}: it is not in ownerSenders (or set userScope to sender)`);
      }
      return null;
    }
    let memory = memories.get(userId);
    if (!memory) {
      // In a group chat several senders share one session: each sender's
      // turns are collected under its own session id.
      const host = perSender ? `${HOST}-${sha16(userId).slice(0, 8)}` : HOST;
      // OpenClaw keeps no timer for idle chats: the background flusher does
      // (idleWait), so a chat's last turns are saved after the idle time even
      // when no later turn comes, or the gateway stops.
      memory = new HookMemory({ settings: { ...settings, userId }, log, host, label: 'QuoteMem OpenClaw plugin', env: full, idleWait: true });
      memories.set(userId, memory);
    }
    return memory;
  };

  // Only the user's own turns are remembered: heartbeat, cron and system
  // runs are not, and messages from another agent session only with
  // includeSubagents.
  const fromUser = (ctx) => {
    const kind = ctx?.inputProvenance?.kind;
    if (kind === 'internal_system') return false;
    if (kind === 'inter_session' && !settings.includeSubagents) return false;
    if (typeof ctx?.trigger === 'string' && ctx.trigger !== 'user') return false;
    return true;
  };

  // A prompt either runs a forget or may wait for a queued one, then recalls.
  const turnBudget = Math.max(settings.forgetTimeoutMs, FORGET_WAIT_MS) + settings.recallTimeoutMs + 2000;
  // Each run's memory, so agent_end and session_end reach the same one.
  const runs = new Map();

  api.on('before_prompt_build', async (event, ctx) => {
    try {
      if (!fromUser(ctx)) return undefined;
      const session = sessionOf(event, ctx);
      const memory = session && memoryFor(ctx);
      if (!memory) return undefined;
      // currentUserMessage is the request itself; `prompt` may hold
      // reconstructed history. An explicit '' means no text this turn.
      let text = typeof event?.currentUserMessage === 'string' ? event.currentUserMessage : lastText(event?.messages, 'user') ?? event?.prompt;
      text = typeof text === 'string' ? text : '';
      if (!text.trim()) return undefined;
      const turn = event?.currentUserMessageId ?? ctx?.runId ?? null;
      if (ctx?.runId) {
        runs.set(ctx.runId, { session, memory });
        if (runs.size > 1000) runs.delete(runs.keys().next().value);
      }
      const result = await memory.prompt({ session, turn, text });
      memory.armIdle();
      return result.context ? { prependContext: result.context } : undefined;
    } catch {
      return undefined;
    }
  }, { timeoutMs: turnBudget });

  api.on('agent_end', async (event, ctx) => {
    try {
      const runId = event?.runId ?? ctx?.runId;
      const known = runId ? runs.get(runId) : null;
      if (runId) runs.delete(runId);
      const session = known?.session ?? sessionOf(event, ctx);
      const memory = known?.memory ?? (session && fromUser(ctx) ? memoryFor(ctx) : null);
      if (!session || !memory) return;
      const completed = event?.success !== false;
      await memory.reply({ session, text: completed ? lastText(event?.messages, 'assistant') : null, completed });
      memory.armIdle();
    } catch {
      // capture is best effort
    }
  }, { timeoutMs: 10_000 });

  api.on('session_end', async (event, ctx) => {
    try {
      const session = sessionOf(event, ctx);
      if (!session) return;
      for (const memory of memories.values()) await memory.end({ session });
    } catch {
      // the idle sweep of a later turn queues what is left
    }
  }, { timeoutMs: 2000 });

  api.on('gateway_stop', () => {
    for (const memory of memories.values()) {
      try {
        memory.flush({ sync: true });
        memory.armIdle({ sync: true });
      } catch { /* best effort */ }
    }
  }, { timeoutMs: 2000 });

  return { settings, memories };
}

export default define({
  id: 'quotemem',
  name: 'QuoteMem',
  description: 'Auditable long-term memory: silent capture, recall with source quotes, and forget on request.',
  register(api) {
    registerQuoteMem(api);
  },
});
