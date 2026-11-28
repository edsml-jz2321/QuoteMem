// Settings shared by every QuoteMem adapter.
//
// Order of precedence: the host's own plugin config (behaviour settings only),
// then QUOTEMEM_* environment variables, then built-in defaults. The service
// URL, user id, auth mode and token come from the environment (the token
// also from a file, see token.mjs) only, so the automatic layer and the MCP
// tools always act on the same memory. Nothing here throws: out-of-range
// numbers are clamped and a fatal problem is reported in `problem`.
import os from 'node:os';
import path from 'node:path';

import { localService, plainHttpUrl } from './autostart.mjs';
import { homeDir } from './proc.mjs';
import { resolveToken } from './token.mjs';

export const MAX_ID_CHARS = 1024;
// A QUOTEMEM_URL whose authority is not plain (autostart.mjs PLAIN_HTTP_URL).
export const PLAIN_URL_PROBLEM = 'QUOTEMEM_URL is not a valid URL: after http:// or https:// it needs a plain host name or address, written in full (no trailing dot on an address, no shortened or hex address), and an optional :port, with no backslash, space or other special character';
// How long a batch the service has not saved, or a forget, may keep waiting
// while the service answers `429 delivery_busy`. The service holds a job back
// so for several reasons; the longest is a session's new batch held while an
// earlier one completes after a forget, which can take about two hours.
export const CAPTURE_BUSY_MAX_WAIT_MS = 9_000_000;
const AUTH_MODES = new Set(['bearer', 'token', 'x-api-key']);

export const OFF = new Set(['0', 'off', 'false', 'no', 'none', 'disabled']);
export const ON = new Set(['1', 'on', 'true', 'yes', 'enabled']);
/** The relevance `always` recall asks of a hit by default (see resolveSettings). */
const ALWAYS_MIN_SCORE = 0.3;
// The smallest recall budget: the block's own text (its header, notice, both
// status notes and the room it keeps) takes about 2,000 characters, so a
// smaller budget could leave no room for a hit and turn recall off.
export const MIN_INJECT_CHARS = 2500;

export function present(value) {
  return value !== undefined && value !== null && value !== '';
}

export function integer(value, fallback, min, max) {
  const n = typeof value === 'number' ? value : Number.parseInt(String(value ?? '').trim(), 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

export function flag(value, fallback) {
  if (typeof value === 'boolean') return value;
  const text = String(value ?? '').trim().toLowerCase();
  if (ON.has(text)) return true;
  if (OFF.has(text)) return false;
  return fallback;
}

export function osUser() {
  try {
    const user = os.userInfo().username;
    return typeof user === 'string' && user.length > 0 && user.length <= MAX_ID_CHARS ? user : null;
  } catch {
    return null;
  }
}

/**
 * Whether a URL host names this computer other than as the listener check
 * (listener.mjs thisComputerHost) names it: this computer's own host name, or
 * a loopback alias of /etc/hosts. Such a URL gets a warning to use 127.0.0.1.
 */
function thisComputerByName(hostname) {
  const host = hostname.toLowerCase().replace(/\.$/, '');
  if (['localhost.localdomain', 'ip6-localhost', 'ip6-loopback'].includes(host)) return true;
  let own = '';
  try {
    own = os.hostname().toLowerCase().replace(/\.$/, '');
  } catch {
    return false;
  }
  return Boolean(own) && (host === own || host === own.split('.')[0]);
}

/**
 * Resolve the adapter settings.
 * @param {object} config - the host's plugin config (all fields optional).
 * @param {object} env - the process environment.
 * @param {{defaultStateDir?: string, configName?: string}} host - the folder
 *   used when QUOTEMEM_STATE_DIR is unset (default `~/.quotemem/default`),
 *   and how warnings name the host config (default "plugin config").
 * @returns {object} settings; `problem` is set when automatic memory must stay off.
 */
export function resolveSettings(config = {}, env = process.env, host = {}) {
  const cfg = config && typeof config === 'object' ? config : {};
  const configName = host.configName || 'plugin config';
  const pick = (key, envName) => (present(cfg[key]) ? cfg[key] : present(env[envName]) ? env[envName] : undefined);
  const out = { problem: null, warnings: [] };
  const envOnly = ['url', 'userId', 'authMode', 'token'].filter((key) => present(cfg[key]));
  if (envOnly.length > 0) {
    out.warnings.push(`${configName} ${envOnly.join(', ')} ignored; set QUOTEMEM_URL, QUOTEMEM_USER_ID and QUOTEMEM_HTTP_AUTH_MODE in the environment, and QUOTEMEM_TOKEN_FILE for a token file, so the memory tools use the same values`);
  }

  const rawUrl = String(present(env.QUOTEMEM_URL) ? env.QUOTEMEM_URL : 'http://127.0.0.1:8765').trim();
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      out.problem = 'QUOTEMEM_URL must be an http:// or https:// URL';
    } else if (parsed.username || parsed.password) {
      out.problem = 'QUOTEMEM_URL must not contain credentials; put the token in the file QUOTEMEM_TOKEN_FILE names instead';
    } else if (!plainHttpUrl(rawUrl)) {
      out.problem = PLAIN_URL_PROBLEM;
    } else {
      out.url = parsed.href.replace(/\/+$/, '');
      const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname) || /^(?:[a-z0-9_-]+\.)*localhost\.?$/.test(parsed.hostname);
      if (thisComputerByName(parsed.hostname)) {
        out.warnings.push(`QUOTEMEM_URL names this computer by the host name ${parsed.hostname}; use http://127.0.0.1${parsed.port ? `:${parsed.port}` : ''} instead, so the plugins check that the program on the port is your QuoteMem service before they send the token`);
      } else if (parsed.protocol === 'http:' && !loopback) {
        out.warnings.push('QUOTEMEM_URL uses plain http to a non-loopback host; the token is sent unencrypted');
      }
    }
  } catch {
    out.problem = 'QUOTEMEM_URL is not a valid URL';
  }

  // The token never comes from the host config: QUOTEMEM_TOKEN, then
  // QUOTEMEM_HTTP_AUTH_SECRET, then the file QUOTEMEM_TOKEN_FILE names, then
  // the service's own token for a service on this computer (token.mjs). So
  // it need not be exported where an agent starts. Once the service was set
  // up with `quotemem setup`, a token that does not exist yet is read after
  // the service, started on demand, creates it (`localService`, autostart.mjs).
  out.localService = out.problem ? null : localService(env, out.url);
  const found = resolveToken(env, out.url ?? null, out.localService);
  out.token = found.token;
  out.tokenSource = found.source;
  if (!out.problem && found.problem) {
    out.problem = found.problem;
  } else if (!out.problem && !out.token && !out.localService?.autostart) {
    out.problem = out.localService?.configured
      ? 'no token: the memory service has not created its token yet, and QUOTEMEM_AUTOSTART is off; start it once with `quotemem serve`'
      : out.localService
      ? 'no token and no memory service set up; run `npx quotemem setup` once (the memory service then starts by itself whenever an agent needs it), or start `quotemem serve` yourself'
      : 'no token; for a service on another computer, copy its token (`quotemem serve` keeps it in <data-dir>/token) into a private file and set QUOTEMEM_TOKEN_FILE to that file';
  } else if (!out.problem && !out.token) {
    // set up and not started yet: the token is read once the service exists
  } else if (!out.problem && !/^[\x21-\x7e]+$/.test(out.token)) {
    out.problem = 'the token must be printable ASCII without spaces';
  }
  out.authMode = String(env.QUOTEMEM_HTTP_AUTH_MODE || 'bearer').trim().toLowerCase();
  if (!out.problem && !AUTH_MODES.has(out.authMode)) out.problem = 'QUOTEMEM_HTTP_AUTH_MODE must be bearer, token or x-api-key';

  const userId = String(env.QUOTEMEM_USER_ID || osUser() || 'default');
  out.userId = userId.length <= MAX_ID_CHARS ? userId : userId.slice(0, MAX_ID_CHARS);

  const recall = String(pick('recall', 'QUOTEMEM_RECALL') ?? 'gated').trim().toLowerCase();
  out.recall = recall === 'always' ? 'always' : OFF.has(recall) ? 'off' : 'gated';
  // The agent service answers at most 20 hits (its own default is 10); recall
  // inserts 5, so the injected block stays small.
  out.topK = integer(pick('topK', 'QUOTEMEM_TOP_K'), 5, 1, 20);
  out.maxInjectChars = integer(pick('maxInjectChars', 'QUOTEMEM_MAX_INJECT_CHARS'), 4000, MIN_INJECT_CHARS, 50_000);
  out.recallTimeoutMs = integer(pick('recallTimeoutMs', 'QUOTEMEM_RECALL_TIMEOUT_MS'), 3000, 100, 15_000);
  const minScore = Number.parseFloat(String(pick('recallMinScore', 'QUOTEMEM_RECALL_MIN_SCORE') ?? ''));
  // The agent service's score is a relevance in [0, 1), not scaled to the
  // best hit, and the service delivers no hit below 0.25:
  // gated mode adds no threshold of its own (null). `always` mode searches
  // on every message, coding requests included, and asks for 0.3, a step
  // above the service's floor, so fewer hits that only share a topic go in.
  // Any number is used as given. A hit without a score passes only in gated
  // mode.
  out.recallMinScore = Number.isFinite(minScore) ? minScore : out.recall === 'always' ? ALWAYS_MIN_SCORE : null;
  // `questions`: a plain first-person statement ("I prefer Rust") does not
  // trigger a search unless the message asks something; `broad` also
  // searches on such statements.
  out.recallGate = String(pick('recallGate', 'QUOTEMEM_RECALL_GATE') ?? 'questions').trim().toLowerCase() === 'broad' ? 'broad' : 'questions';
  out.recallRepeatTurns = integer(pick('recallRepeatTurns', 'QUOTEMEM_RECALL_REPEAT_TURNS'), 10, 1, 10_000);
  out.recallMinChars = integer(env.QUOTEMEM_RECALL_MIN_CHARS, 2, 1, 1000);
  out.queryMaxChars = integer(env.QUOTEMEM_QUERY_MAX_CHARS, 2000, 100, 100_000);
  const options = pick('searchOptions', 'QUOTEMEM_SEARCH_OPTIONS');
  out.searchOptions = (Array.isArray(options) ? options : String(options ?? '').split(','))
    .map((option) => String(option).trim())
    .filter(Boolean);

  const every = pick('captureEvery', 'QUOTEMEM_CAPTURE_EVERY');
  out.captureEvery = OFF.has(String(every ?? '').trim().toLowerCase()) ? 0 : integer(every, 3, 0, 1000);
  const envEvery = String(env.QUOTEMEM_CAPTURE_EVERY ?? '').trim().toLowerCase();
  if (out.captureEvery === 0 && !(OFF.has(envEvery))) {
    // The MCP server sees only the environment: it keeps quotemem_add hidden.
    out.warnings.push(`capture is off through the ${configName}, so quotemem_add stays hidden; set QUOTEMEM_CAPTURE_EVERY=off instead`);
  }
  const idle = pick('idleFlushMs', 'QUOTEMEM_IDLE_FLUSH_MS');
  out.idleFlushMs = OFF.has(String(idle ?? '').trim().toLowerCase()) ? 0 : integer(idle, 300_000, 0, 86_400_000);
  out.captureAssistant = String(pick('captureAssistant', 'QUOTEMEM_CAPTURE_ASSISTANT') ?? 'final').trim().toLowerCase() === 'none' ? 'none' : 'final';
  out.maxMessageChars = integer(pick('maxMessageChars', 'QUOTEMEM_MAX_MESSAGE_CHARS'), 12_000, 200, 500_000);
  out.captureTimeoutMs = integer(pick('captureTimeoutMs', 'QUOTEMEM_CAPTURE_TIMEOUT_MS'), 1_860_000, 1000, 7_200_000);
  out.captureMaxAttempts = integer(pick('captureMaxAttempts', 'QUOTEMEM_CAPTURE_MAX_ATTEMPTS'), 8, 1, 100);
  out.captureMaxAgeMs = integer(pick('captureMaxAgeMs', 'QUOTEMEM_CAPTURE_MAX_AGE_MS'), 86_400_000, 60_000, 30 * 86_400_000);
  out.captureBusyMaxWaitMs = integer(pick('captureBusyMaxWaitMs', 'QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS'), CAPTURE_BUSY_MAX_WAIT_MS, 60_000, 30 * 86_400_000);
  out.captureQueueMax = integer(pick('captureQueueMax', 'QUOTEMEM_CAPTURE_QUEUE_MAX'), 200, 1, 10_000);
  out.retryBaseMs = integer(pick('retryBaseMs', 'QUOTEMEM_CAPTURE_RETRY_BASE_MS'), 5000, 10, 600_000);
  out.includeSubagents = flag(pick('includeSubagents', 'QUOTEMEM_INCLUDE_SUBAGENTS'), false);
  // Whether a Cursor prompt hook adds recall and forget notes to the turn:
  // `auto` on Cursor versions known to read it (native.mjs), `on`, `off`.
  const cursorContext = String(pick('cursorPromptContext', 'QUOTEMEM_CURSOR_PROMPT_CONTEXT') ?? 'auto').trim().toLowerCase();
  out.cursorPromptContext = ON.has(cursorContext) ? 'on' : OFF.has(cursorContext) ? 'off' : 'auto';
  out.debug = flag(env.QUOTEMEM_DEBUG, false);

  // Silent forget: a user message that asks to forget something is carried
  // out through forget by statement (`auto`), or left to the explicit tools.
  const forget = String(pick('forget', 'QUOTEMEM_FORGET') ?? 'auto').trim().toLowerCase();
  out.forget = OFF.has(forget) ? 'off' : 'auto';
  // How long the user's turn waits for its silent forget, from the moment
  // the message arrived. A forget asks the service's model (about 1 to 5 s),
  // so the wait stays well under a host's 3 s hook budget: a forget not done
  // by then finishes in the background, and the next turn says how it ended.
  out.forgetTimeoutMs = integer(pick('forgetTimeoutMs', 'QUOTEMEM_FORGET_TIMEOUT_MS'), 1800, 200, 2500);
  // How long a background flusher waits for a batch's next retry before it
  // exits; a later hook starts a new one.
  out.flusherMaxWaitMs = integer(pick('flusherMaxWaitMs', 'QUOTEMEM_FLUSHER_MAX_WAIT_MS'), 120_000, 0, 3_600_000);

  const stateDir = pick('stateDir', 'QUOTEMEM_STATE_DIR');
  out.stateDir = present(stateDir)
    ? path.resolve(String(stateDir))
    : host.defaultStateDir || path.join(homeDir(env), '.quotemem', 'default');
  return out;
}

/** The settings without the token, warnings or problem: safe to hand to a child process. */
export function publicSettings(settings) {
  const { token: _token, warnings: _warnings, problem: _problem, ...rest } = settings;
  return rest;
}
