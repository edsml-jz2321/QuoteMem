// `quotemem setup`, `status` and `stop`: the memory service on this computer.
//
// QuoteMem runs on the user's own computer. `setup` asks once for the model
// settings and writes them to ~/.quotemem/service.env (mode 0600, in a 0700
// folder); from then on the plugins start the service by themselves when an
// agent needs it (integrations/core/autostart.mjs). Keys are read from the
// terminal without echo, or from the environment, never from the command
// line, and are never printed.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { Writable } from 'node:stream';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { plainHttpUrl } from '../core/autostart.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const core = (name) => import(pathToFileURL(path.join(here, '..', 'core', name)).href);

const DASHSCOPE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1';
/** Where the key of the default and fast presets comes from. */
const MODEL_STUDIO_CONSOLE = 'https://bailian.console.aliyun.com';

/**
 * The QUOTEMEM_CHAT_REASONING values `setup` writes: `high` asks for
 * thinking at high effort (the service keeps a reasoning reply from being
 * cut off or waiting too long), `off` asks for no thinking, and `none` sends
 * no reasoning parameters, so the provider's default applies. An empty value
 * means `off` to the service. The service owns these names: change them
 * here, and only here, when it does.
 */
export const CHAT_REASONING = Object.freeze({ high: 'high', off: 'off', none: 'none' });

/**
 * The competition configuration: Alibaba Cloud Model Studio (DashScope),
 * one key for both models. The embedding size is the service's default for
 * a new store, 1024, so it is not written.
 */
const COMPETITION = Object.freeze({
  QUOTEMEM_CHAT_BASE_URL: DASHSCOPE_URL,
  QUOTEMEM_CHAT_MODEL: 'deepseek-v4.1-flash-1017',
  QUOTEMEM_CHAT_STRUCTURED_OUTPUT: 'json_object',
  // The model ids replies may report: the name sent, the name without its
  // date, and the gateway's dashed dated snapshots (deepseek-v4-1-flash-261017).
  QUOTEMEM_CHAT_MODEL_PATTERN: 'deepseek-v4\\.1-flash(?:-1017)?|deepseek-v4-1-flash-[0-9]{6}',
  QUOTEMEM_EMBED_BASE_URL: DASHSCOPE_URL,
  QUOTEMEM_EMBED_MODEL: 'qwen3.7-text-embedding',
});

/** The ready-made model settings `setup` offers; `custom` asks for each value instead. */
export const PRESETS = Object.freeze({
  default: Object.freeze({
    label: 'default (recommended), the competition configuration: DeepSeek V4.1 Flash with thinking at high effort and Qwen3.7 embeddings (1024 dimensions), both through Alibaba Cloud Model Studio with one key. Saving memories is slower (a large batch of corrections can take minutes), and thinking adds output tokens to the bill',
    values: Object.freeze({ ...COMPETITION, QUOTEMEM_CHAT_REASONING: CHAT_REASONING.high }),
  }),
  fast: Object.freeze({
    label: 'fast: the same models and key with thinking off. Memories are saved in seconds',
    values: Object.freeze({ ...COMPETITION, QUOTEMEM_CHAT_REASONING: CHAT_REASONING.off }),
  }),
});
const PRESET_NAMES = ['default', 'fast', 'custom'];
/** `--chat-reasoning` choices: thinking high, off, or the provider's default (nothing sent). */
const REASONING_CHOICES = { high: CHAT_REASONING.high, off: CHAT_REASONING.off, default: CHAT_REASONING.none };

/**
 * The QUOTEMEM_REVIEW values: how a new store reviews a new memory against
 * the stored ones. The review is frozen into a store when it is created; the
 * service refuses to open a store with another one, and `quotemem rebuild
 * --review` moves a store to another review. The service owns these names:
 * change them here, and only here, when it does.
 */
export const REVIEWS = Object.freeze({
  fast: 'fast: local review rules only, as in the competition submission. No review calls, so the fastest and cheapest; a change said outright ("Hangzhou, not Beijing") retires the old memory',
  standard: 'standard (recommended): the chat model reviews only the new memories that may repeat, update or correct a stored one',
  full: 'full: the chat model reviews every new memory. The slowest and most careful, with the most model calls',
});
export const DEFAULT_REVIEW = 'standard';
const REVIEW = 'QUOTEMEM_REVIEW';
const REVIEW_NAMES = Object.keys(REVIEWS);
const normalReview = (value) => String(value ?? '').trim().toLowerCase();

/**
 * The review service.env gives a new store: `{value, source}`, where source is
 * `service.env` or `default` (unset: the service's standard), and `value` is
 * null for a value the service would refuse.
 */
export function reviewOf(serviceEnv) {
  const saved = normalReview(serviceEnv?.[REVIEW]);
  if (!saved) return { value: DEFAULT_REVIEW, source: 'default' };
  return { value: REVIEW_NAMES.includes(saved) ? saved : null, source: 'service.env' };
}

/** The variables `setup` writes; any other line of service.env is kept as it is. */
export const MANAGED = [
  'QUOTEMEM_CHAT_BASE_URL', 'QUOTEMEM_CHAT_MODEL', 'QUOTEMEM_CHAT_API_KEY', 'QUOTEMEM_CHAT_STRUCTURED_OUTPUT',
  'QUOTEMEM_CHAT_REASONING', 'QUOTEMEM_CHAT_MODEL_PATTERN',
  'QUOTEMEM_EMBED_BASE_URL', 'QUOTEMEM_EMBED_MODEL', 'QUOTEMEM_EMBED_API_KEY',
  REVIEW,
];
const KEYS = new Set(['QUOTEMEM_CHAT_API_KEY', 'QUOTEMEM_EMBED_API_KEY']);
const STRUCTURED = new Set(['json_schema', 'json_object']);
/** The options that set one value each, with the variable they set. */
const VALUE_OPTIONS = [
  ['chat-url', 'QUOTEMEM_CHAT_BASE_URL'], ['chat-model', 'QUOTEMEM_CHAT_MODEL'],
  ['chat-structured-output', 'QUOTEMEM_CHAT_STRUCTURED_OUTPUT'], ['chat-model-pattern', 'QUOTEMEM_CHAT_MODEL_PATTERN'],
  ['embed-url', 'QUOTEMEM_EMBED_BASE_URL'], ['embed-model', 'QUOTEMEM_EMBED_MODEL'],
];
const MODEL_OPTIONS = ['chat-url', 'chat-model', 'embed-url', 'embed-model'];

export const SETUP_USAGE = `usage: quotemem setup [options]

Asks which models the memory service calls, saves the settings to
~/.quotemem/service.env (mode 600), and checks them with \`quotemem check\`.
After that the service starts by itself, on this computer, whenever an agent
needs it. Keys are typed without echo; they are never taken from the command
line.

presets (--preset NAME):
  default   recommended, the competition configuration: chat model
            deepseek-v4.1-flash-1017 with thinking at high effort, embedding
            model qwen3.7-text-embedding (1024 dimensions), both through
            Alibaba Cloud Model Studio (DashScope) with one key. Saving
            memories is slower, and thinking adds output tokens to the bill.
  fast      the same models with thinking off: memories are saved in
            seconds.
  custom    any OpenAI-compatible chat and embedding endpoints, models and
            keys.

review (--review NAME; asked after the models, Enter keeps the saved one,
else standard). It decides whether a new memory repeats, updates or corrects
a stored one, and it is fixed when the memory store is created: to change it
for an existing store, rebuild the store (\`quotemem rebuild --from <data dir>
--to <new data dir> --review NAME\`, then point QUOTEMEM_DATA_DIR at the new one).
  fast      local review rules only, as in the competition submission: no
            review calls, the fastest and cheapest.
  standard  the default: the chat model reviews only the new memories that may
            repeat, update or correct a stored one.
  full      the chat model reviews every new memory: the slowest and most
            careful.

options (each skips its question):
  --preset default|fast|custom   (the fast preset turns thinking off; it is not
               the fast review)
  --review fast|standard|full         written as QUOTEMEM_REVIEW
  --chat-url URL --chat-model NAME --chat-structured-output json_schema|json_object
  --chat-reasoning high|off|default   thinking at high effort, off, or the
               provider's default (QUOTEMEM_CHAT_REASONING=none: no
               reasoning parameters are sent)
  --chat-model-pattern REGEX          model ids the replies may report
  --embed-url URL --embed-model NAME
  --yes        ask nothing. The values come from the options, then the
               QUOTEMEM_CHAT_*, QUOTEMEM_EMBED_* and QUOTEMEM_REVIEW
               variables of the environment (keys only there; DASHSCOPE_API_KEY also serves
               the default and fast presets), then the preset. Without
               --preset: custom when a model option is given or settings are
               saved (the saved values fill the rest), else default.
  --no-check   do not run \`quotemem check\`

Every value stays editable in ~/.quotemem/service.env; for example
QUOTEMEM_CHAT_REASONING=${CHAT_REASONING.off} turns thinking off.
`;

function validUrl(value) {
  if (!plainHttpUrl(value)) return false;
  try {
    const url = new URL(value);
    const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    return (url.protocol === 'https:' || (url.protocol === 'http:' && loopback)) && !url.username && !url.password;
  } catch {
    return false;
  }
}

function sameHost(a, b) {
  try {
    return new URL(a).host === new URL(b).host;
  } catch {
    return false;
  }
}

const printable = (value) => /^[\x20-\x7e]+$/.test(value) && value.trim() === value;

/**
 * A question reader on a terminal (keys without echo) or on piped lines.
 * Whether the answers come from a terminal decides alone: readline then puts
 * the terminal in raw mode (it echoes nothing itself) and echoes the typed
 * answers to `output`, or to `echo` (stderr) when `output` is not a terminal
 * (`setup | tee`), with keys muted either way.
 */
function prompter(input, output, echo) {
  const tty = Boolean(input.isTTY);
  const echoTo = output.isTTY ? output : echo?.isTTY ? echo : new Writable({ write(_chunk, _encoding, done) { done(); } });
  const rl = readline.createInterface({ input, output: tty ? echoTo : undefined, terminal: tty });
  let muted = false;
  if (tty) {
    const write = rl._writeToOutput.bind(rl);
    rl._writeToOutput = (text) => { if (!muted) write(text); };
  }
  const lines = [];
  const waiting = [];
  let closed = false;
  rl.on('line', (line) => {
    const next = waiting.shift();
    if (next) next(line);
    else lines.push(line);
  });
  rl.on('close', () => {
    closed = true;
    for (const next of waiting.splice(0)) next(null);
  });
  return {
    async ask(question, { secret = false } = {}) {
      output.write(question);
      muted = secret;
      const line = lines.length > 0 ? lines.shift() : closed ? null : await new Promise((resolve) => waiting.push(resolve));
      muted = false;
      if (secret && tty) echoTo.write('\n');
      if (line === null) throw Object.assign(new Error('no more input'), { code: 'EOF' });
      return line.trim();
    },
    close() { rl.close(); },
  };
}

/**
 * A numbered menu; Enter takes option `first` (0-based). An answer is the
 * option's number or its name (`name`, else `value`), in any case. Returns
 * the chosen option's value.
 */
async function menu(ask, out, title, options, first = 0) {
  out.write(`\n${title}\n`);
  options.forEach((option, i) => out.write(`  ${i + 1}) ${option.label}\n`));
  const names = options.map((option) => String(option.name ?? option.value).toLowerCase());
  for (;;) {
    const answer = await ask(`choose [${first + 1}]: `);
    const index = answer === '' ? first : /^[0-9]+$/.test(answer) ? Number(answer) - 1 : names.indexOf(answer.toLowerCase());
    if (index >= 0 && index < options.length) return options[index].value;
    out.write(`  type a number from 1 to ${options.length} or its name\n`);
  }
}

async function askUrl(ask, out, question, fallback) {
  for (;;) {
    const url = await ask(question);
    if (url === '' && fallback) return fallback;
    if (validUrl(url)) return url.replace(/\/+$/, '');
    out.write('  an https URL (or http on this computer) without credentials\n');
  }
}

async function askName(ask, question, fallback) {
  for (;;) {
    const name = await ask(question);
    if (name === '' && fallback) return fallback;
    if (name && printable(name)) return name;
  }
}

async function askKey(ask, label, saved, fallbackLabel, out) {
  const hint = saved ? 'Enter keeps the saved key' : fallbackLabel ? `Enter uses ${fallbackLabel}` : null;
  for (;;) {
    const key = await ask(`  ${label} (typed without echo${hint ? `; ${hint}` : ''}): `, { secret: true });
    if (key === '' && (saved || fallbackLabel)) return null;
    if (key && printable(key) && !/\s/.test(key)) return key;
    out.write(key ? '  a key is printable text without spaces\n' : '  a key is needed\n');
  }
}

/** The preset whose model values `values` holds, keys and the review aside, else null. */
function presetOf(values) {
  for (const [name, preset] of Object.entries(PRESETS)) {
    const names = new Set([...Object.keys(preset.values), ...MANAGED.filter((n) => !KEYS.has(n) && n !== REVIEW)]);
    if ([...names].every((n) => (values[n] ?? '') === (preset.values[n] ?? ''))) return name;
  }
  return null;
}

function serialize(values, previous) {
  const preset = presetOf(values);
  const lines = [
    '# QuoteMem memory service settings, written by `quotemem setup`.',
    '# Keys stay in this file (mode 600). The plugins start the service with',
    '# these settings when an agent needs it; run `quotemem setup` to change them,',
    '# or edit a value here (each QUOTEMEM_* line overrides the preset).',
  ];
  if (preset === 'default') lines.push(`# preset: default, the competition configuration; QUOTEMEM_CHAT_REASONING=${CHAT_REASONING.off} saves memories faster.`);
  else if (preset) lines.push(`# preset: ${preset}`);
  else lines.push('# preset: custom');
  for (const name of MANAGED) if (values[name]) lines.push(`${name}=${values[name]}`);
  const kept = Object.entries(previous).filter(([name]) => !MANAGED.includes(name));
  if (kept.length > 0) {
    lines.push('', '# kept from the previous file');
    for (const [name, value] of kept) lines.push(`${name}=${value}`);
  }
  return `${lines.join('\n')}\n`;
}

/** Write service.env with mode 0600 in a 0700 folder (an existing folder is tightened). */
export function writeServiceEnv(paths, text) {
  fs.mkdirSync(paths.dir, { recursive: true, mode: 0o700 });
  if (process.platform !== 'win32') fs.chmodSync(paths.dir, 0o700);
  const tmp = `${paths.envFile}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmp, text, { mode: 0o600 });
    if (process.platform !== 'win32') fs.chmodSync(tmp, 0o600);
    fs.renameSync(tmp, paths.envFile);
  } catch (error) {
    try { fs.unlinkSync(tmp); } catch { /* none */ }
    throw error;
  }
}

// The service commands that call the running service; the others work on a
// store or check the model settings on this computer.
export const CLIENT_COMMANDS = new Set(['memories', 'repair', 'delete-user']);
// The settings the plugins take from the environment (and ~/.quotemem/env)
// only, never from service.env.
const PLUGIN_NAMES = ['QUOTEMEM_URL', 'QUOTEMEM_USER_ID', 'QUOTEMEM_TOKEN_FILE'];
const unset = (value) => value === undefined || value === null || value === '';

/**
 * Where a client command is given the service URL: each `--url` (argparse
 * also takes `--ur`), as {index, prefix, value}: the argument that holds the
 * value, the text before the value in it, and the value. The last one counts.
 */
function urlOptions(argv) {
  const found = [];
  for (const [index, arg] of argv.entries()) {
    const match = /^(--url?)(=|$)/.exec(arg);
    if (!match) continue;
    found.push(match[2] === '=' ? { index, prefix: `${match[1]}=`, value: arg.slice(match[1].length + 1) } : { index: index + 1, prefix: '', value: argv[index + 1] ?? null });
  }
  return found;
}

/**
 * The URL a client command calls, as the service's commands pick it:
 * `--url`, else QUOTEMEM_URL, else the default; an empty value counts as
 * none, and the value is trimmed.
 */
function calledUrl(argv, env) {
  return (urlOptions(argv).at(-1)?.value || env.QUOTEMEM_URL || '').trim();
}

/**
 * The environment and arguments of a client command (CLIENT_COMMANDS), so it
 * reaches the service the agents use. This environment with service.env (the data folder
 * of a service that is not running), completed with the plugins' settings
 * (hooks/env.mjs): QUOTEMEM_URL, QUOTEMEM_USER_ID and QUOTEMEM_TOKEN_FILE as
 * the plugins read them (this environment, then ~/.quotemem/env), other
 * QUOTEMEM_* values of ~/.quotemem/env that neither sets. The token is the
 * one a plugin would send to the URL the command calls (calledUrl): the
 * token of service.env, or of the service's data folder, only for a service
 * on this computer, never for another one. A URL the plugins refuse (not
 * http(s), with credentials, or with an authority that is not plain) gets no
 * token and no token file at all. The command is given the URL exactly as it
 * was checked (QUOTEMEM_URL, and every `--url` value), so it connects to the
 * host the token was chosen for.
 * @returns {Promise<{env: object, argv: string[]}>}
 */
export async function clientEnvironment(env, serviceEnv, argv = []) {
  const { hostEnv } = await import(pathToFileURL(path.join(here, '..', 'hooks', 'env.mjs')).href);
  const { resolveSettings } = await core('config.mjs');
  const plugin = hostEnv(env).env;
  const out = { ...env, ...(serviceEnv ?? {}) };
  for (const [name, value] of Object.entries(plugin)) if (name.startsWith('QUOTEMEM_') && unset(out[name])) out[name] = value;
  for (const name of PLUGIN_NAMES) {
    if (unset(plugin[name])) delete out[name];
    else out[name] = plugin[name];
  }
  // The plugins' settings for the URL the command calls: its token, or none
  // when they refuse that URL.
  const target = resolveSettings({}, { ...plugin, QUOTEMEM_URL: calledUrl(argv, plugin) }, {});
  delete out.QUOTEMEM_TOKEN;
  delete out.QUOTEMEM_HTTP_AUTH_SECRET;
  const args = [...argv];
  if (!target.url) {
    delete out.QUOTEMEM_TOKEN_FILE;
  } else {
    if (target.token) out.QUOTEMEM_TOKEN = target.token;
    out.QUOTEMEM_URL = target.url;
    for (const { index, prefix } of urlOptions(argv)) if (index < args.length) args[index] = `${prefix}${target.url}`;
  }
  return { env: out, argv: args };
}

/**
 * Run a service command through uvx with the saved service settings. By
 * default this environment, overlaid with service.env (a command typed in a
 * shell sees that shell's settings too), and for a client command the
 * plugins' URL and token (clientEnvironment); `exact`: the environment a
 * plugin starts the service with (`setup` checks what will actually run).
 */
export async function runServiceCommand(command, argv, { env = process.env, stdio = 'inherit', release, exact = false } = {}) {
  const { findUvx, readServiceEnv, serviceEnvironment } = await core('autostart.mjs');
  const serviceEnv = readServiceEnv(env);
  const client = !exact && CLIENT_COMMANDS.has(command) ? await clientEnvironment(env, serviceEnv, argv) : null;
  const runEnv = exact ? serviceEnvironment(env, serviceEnv) : client?.env ?? { ...env, ...(serviceEnv ?? {}) };
  if (client) argv = client.argv;
  delete runEnv.QUOTEMEM_FLUSHER_SETTINGS;
  const uvx = findUvx(runEnv);
  if (!uvx) return { code: 127, uvx: null };
  const spec = `quotemem${command === 'bench' ? '[bench]' : ''}==${release}`;
  const child = spawn(uvx, ['--from', spec, 'quotemem', command, ...argv], { stdio, env: runEnv });
  return { child, uvx };
}

const embedUrlOf = (values) => values.QUOTEMEM_EMBED_BASE_URL ?? values.QUOTEMEM_CHAT_BASE_URL;
const nonKeys = (values) => Object.fromEntries(MANAGED.filter((name) => !KEYS.has(name) && values[name]).map((name) => [name, values[name]]));

/** Set `name`, or remove it for an empty value (the service's default then applies). */
function put(values, name, value) {
  const text = value === undefined || value === null ? '' : String(value).trim();
  if (text) values[name] = text;
  else delete values[name];
}

/**
 * Apply the value options (and, with `env`, the QUOTEMEM_* variables) over
 * `base`. A pattern of reported model ids belongs to its model: when the chat
 * model or URL changes and no pattern is given, the base's pattern is dropped.
 * @returns {string[]} the variables taken from the environment.
 */
function override(values, base, flags, env) {
  const fromEnv = [];
  const given = (flag, name) => {
    if (flags[flag] !== undefined) return String(flags[flag]);
    if (env?.[name]?.trim()) {
      fromEnv.push(name);
      return env[name];
    }
    return undefined;
  };
  let pattern = false;
  for (const [flag, name] of VALUE_OPTIONS) {
    const value = given(flag, name);
    if (value !== undefined) {
      put(values, name, value);
      if (name === 'QUOTEMEM_CHAT_MODEL_PATTERN') pattern = true;
    }
  }
  if (flags['chat-reasoning'] !== undefined) put(values, 'QUOTEMEM_CHAT_REASONING', REASONING_CHOICES[flags['chat-reasoning']]);
  else {
    const value = given('', 'QUOTEMEM_CHAT_REASONING');
    if (value !== undefined) put(values, 'QUOTEMEM_CHAT_REASONING', value);
  }
  if (!pattern && (values.QUOTEMEM_CHAT_MODEL !== base.QUOTEMEM_CHAT_MODEL || values.QUOTEMEM_CHAT_BASE_URL !== base.QUOTEMEM_CHAT_BASE_URL)) {
    delete values.QUOTEMEM_CHAT_MODEL_PATTERN;
  }
  return fromEnv;
}

/** The saved key of `keyName`, when the saved settings send it to the same URL. */
function savedKey(previous, values, keyName) {
  const url = keyName === 'QUOTEMEM_CHAT_API_KEY' ? (v) => v.QUOTEMEM_CHAT_BASE_URL : embedUrlOf;
  return previous[keyName] && url(previous) === url(values) ? previous[keyName] : '';
}

/** `--yes`: the values from the options, the environment and the preset or saved settings. */
function settingsWithoutQuestions(presetName, flags, env, previous) {
  const preset = PRESETS[presetName];
  const base = preset ? { ...preset.values } : nonKeys(previous);
  const values = { ...base };
  const fromEnv = override(values, base, flags, env);
  // A preset's two models share one key, also when the saved settings are a preset.
  const oneKey = Boolean(preset || presetOf(values)) && sameHost(embedUrlOf(values), values.QUOTEMEM_CHAT_BASE_URL);
  const dashscope = preset ? (env.DASHSCOPE_API_KEY ?? '').trim() : '';
  if (env.QUOTEMEM_CHAT_API_KEY?.trim()) fromEnv.push('QUOTEMEM_CHAT_API_KEY');
  else if (dashscope) fromEnv.push('DASHSCOPE_API_KEY');
  put(values, 'QUOTEMEM_CHAT_API_KEY', env.QUOTEMEM_CHAT_API_KEY?.trim() || dashscope || savedKey(previous, values, 'QUOTEMEM_CHAT_API_KEY'));
  if (env.QUOTEMEM_EMBED_API_KEY?.trim()) fromEnv.push('QUOTEMEM_EMBED_API_KEY');
  // Without its own key, the embedding endpoint uses the chat key when it is
  // on the same host (written out: the service would otherwise take
  // DASHSCOPE_API_KEY from whichever environment it starts in).
  const sameKey = sameHost(embedUrlOf(values), values.QUOTEMEM_CHAT_BASE_URL) ? values.QUOTEMEM_CHAT_API_KEY : '';
  put(values, 'QUOTEMEM_EMBED_API_KEY', env.QUOTEMEM_EMBED_API_KEY?.trim() || (oneKey ? sameKey : savedKey(previous, values, 'QUOTEMEM_EMBED_API_KEY') || sameKey));
  return { values, fromEnv };
}

/** Ask for the keys of `values`: one for a preset whose two models share a host. */
async function askKeys(ask, out, values, previous, preset) {
  const sharedHost = sameHost(embedUrlOf(values), values.QUOTEMEM_CHAT_BASE_URL);
  const savedChat = savedKey(previous, values, 'QUOTEMEM_CHAT_API_KEY');
  if (preset && sharedHost) {
    const modelStudio = values.QUOTEMEM_CHAT_BASE_URL === DASHSCOPE_URL;
    const label = modelStudio ? 'Model Studio (DashScope) API key, for both models' : 'API key, for both models';
    out.write('\n');
    if (modelStudio) {
      const deepseek = /^deepseek/i.test(values.QUOTEMEM_CHAT_MODEL ?? '');
      out.write(`  Both models${deepseek ? ', DeepSeek included,' : ''} are called through Alibaba Cloud Model Studio: the key is a Model Studio key\n`);
      out.write(`  from ${MODEL_STUDIO_CONSOLE} (China region)${deepseek ? ', not a DeepSeek platform key' : ''}.\n`);
    }
    const key = (await askKey(ask, label, Boolean(savedChat), null, out)) ?? savedChat;
    values.QUOTEMEM_CHAT_API_KEY = key;
    values.QUOTEMEM_EMBED_API_KEY = key;
    return;
  }
  values.QUOTEMEM_CHAT_API_KEY = (await askKey(ask, 'chat API key', Boolean(savedChat), null, out)) ?? savedChat;
  const savedEmbed = savedKey(previous, values, 'QUOTEMEM_EMBED_API_KEY');
  const embedKey = await askKey(ask, 'embedding API key', Boolean(savedEmbed), sharedHost ? 'the chat key' : null, out);
  // Enter takes the saved key, else the chat key: written out either way.
  values.QUOTEMEM_EMBED_API_KEY = embedKey ?? (savedEmbed || values.QUOTEMEM_CHAT_API_KEY);
}

/** The custom questions; each option given skips its question. */
async function askCustom(ask, out, flags, previous) {
  const values = {};
  const keep = (name) => (previous[name] ? `; Enter keeps ${previous[name]}` : '');
  out.write('\nChat model (any OpenAI-compatible endpoint; it picks out and reviews memories)\n');
  put(values, 'QUOTEMEM_CHAT_BASE_URL', flags['chat-url'] ?? await askUrl(ask, out, `  base URL (https://…/v1${keep('QUOTEMEM_CHAT_BASE_URL')}): `, previous.QUOTEMEM_CHAT_BASE_URL));
  put(values, 'QUOTEMEM_CHAT_MODEL', flags['chat-model'] ?? await askName(ask, `  model name${previous.QUOTEMEM_CHAT_MODEL ? ` (Enter keeps ${previous.QUOTEMEM_CHAT_MODEL})` : ''}: `, previous.QUOTEMEM_CHAT_MODEL));
  put(values, 'QUOTEMEM_CHAT_STRUCTURED_OUTPUT', flags['chat-structured-output'] ?? await menu(ask, out, '  structured output:', [
    { label: 'json_schema: strict JSON schemas (OpenAI and most providers)', value: 'json_schema' },
    { label: 'json_object: JSON mode, the schema in the prompt (DeepSeek, Model Studio and others without strict schemas)', value: 'json_object' },
  ]));
  put(values, 'QUOTEMEM_CHAT_REASONING', flags['chat-reasoning'] !== undefined ? REASONING_CHOICES[flags['chat-reasoning']] : await menu(ask, out, '  thinking:', [
    { label: 'the provider\'s default (no reasoning parameters are sent)', name: 'default', value: REASONING_CHOICES.default },
    { label: 'off (faster writes)', value: REASONING_CHOICES.off },
    { label: 'high effort (slower writes, more output tokens)', value: REASONING_CHOICES.high },
  ]));
  // A saved pattern is kept for the same model on the same endpoint.
  const samePlace = previous.QUOTEMEM_CHAT_MODEL === values.QUOTEMEM_CHAT_MODEL && previous.QUOTEMEM_CHAT_BASE_URL === values.QUOTEMEM_CHAT_BASE_URL;
  put(values, 'QUOTEMEM_CHAT_MODEL_PATTERN', flags['chat-model-pattern'] ?? (samePlace ? previous.QUOTEMEM_CHAT_MODEL_PATTERN : ''));

  out.write('\nEmbedding model (any OpenAI-compatible endpoint; it searches memories)\n');
  put(values, 'QUOTEMEM_EMBED_BASE_URL', flags['embed-url'] ?? await askUrl(ask, out, '  base URL (Enter uses the chat URL): ', values.QUOTEMEM_CHAT_BASE_URL));
  const savedModel = embedUrlOf(previous) === values.QUOTEMEM_EMBED_BASE_URL ? previous.QUOTEMEM_EMBED_MODEL : undefined;
  put(values, 'QUOTEMEM_EMBED_MODEL', flags['embed-model'] ?? await askName(ask, `  model name${savedModel ? ` (Enter keeps ${savedModel})` : ''}: `, savedModel));
  return values;
}

/** Ask the review; Enter keeps the saved one, else standard. */
async function askReview(ask, out, previous) {
  const saved = reviewOf(previous).value ?? DEFAULT_REVIEW;
  const options = REVIEW_NAMES.map((name) => ({ label: REVIEWS[name], value: name }));
  return menu(ask, out, 'Review (how a new memory is checked against the stored ones; fixed when the memory store is created,\nchanging it later needs `quotemem rebuild --review`):', options, REVIEW_NAMES.indexOf(saved));
}

/**
 * `quotemem setup`.
 * @returns {Promise<number>} the exit code.
 */
export async function cmdSetup(flags, { env = process.env, input = process.stdin, output = process.stdout, echo = process.stderr, release }) {
  if (flags.help || flags.h) {
    output.write(SETUP_USAGE);
    return 0;
  }
  if (flags.preset !== undefined && !PRESET_NAMES.includes(flags.preset)) {
    process.stderr.write('quotemem setup: --preset is default, fast or custom\n');
    return 2;
  }
  if (flags['chat-reasoning'] !== undefined && !Object.hasOwn(REASONING_CHOICES, flags['chat-reasoning'])) {
    process.stderr.write('quotemem setup: --chat-reasoning is high, off or default\n');
    return 2;
  }
  if (flags.review !== undefined && !REVIEW_NAMES.includes(normalReview(flags.review))) {
    process.stderr.write('quotemem setup: --review is fast, standard or full\n');
    return 2;
  }
  const review = flags.review === undefined ? undefined : normalReview(flags.review);
  const { dataDirOf, readServiceEnv, serviceEnvironment, servicePaths } = await core('autostart.mjs');
  const paths = servicePaths(env);
  const previous = readServiceEnv(env) ?? {};
  const modelOption = MODEL_OPTIONS.some((flag) => flags[flag] !== undefined);
  let values;

  if (flags.yes) {
    const saved = Object.keys(nonKeys(previous)).length > 0;
    const presetName = flags.preset ?? (modelOption || saved ? 'custom' : 'default');
    const result = settingsWithoutQuestions(presetName, flags, env, previous);
    values = result.values;
    // The review: --review, then QUOTEMEM_REVIEW of the environment, then the saved one, then standard.
    const envReview = normalReview(env[REVIEW]);
    if (review === undefined && envReview) result.fromEnv.push(REVIEW);
    put(values, REVIEW, review ?? (envReview || normalReview(previous[REVIEW]) || DEFAULT_REVIEW));
    output.write(`preset: ${presetName}${result.fromEnv.length > 0 ? `; from the environment: ${result.fromEnv.join(', ')}` : ''}\n`);
    const missing = ['QUOTEMEM_CHAT_BASE_URL', 'QUOTEMEM_CHAT_MODEL', 'QUOTEMEM_CHAT_API_KEY', 'QUOTEMEM_EMBED_MODEL', 'QUOTEMEM_EMBED_API_KEY'].filter((name) => !values[name]);
    if (missing.length > 0) {
      process.stderr.write(`quotemem setup: missing ${missing.join(', ')} (give the options, or set them in the environment; keys only in the environment)\n`);
      return 2;
    }
  } else {
    const ask = prompter(input, output, echo);
    try {
      output.write(`QuoteMem runs on this computer and keeps its memory here. Its memory service calls one chat model\n(to pick out memories) and one embedding model (to search them). Settings go to ${paths.envFile}.\n`);
      let choice = flags.preset ?? (modelOption ? 'custom' : null);
      if (!choice) {
        const options = [
          { label: PRESETS.default.label, value: 'default' },
          { label: PRESETS.fast.label, value: 'fast' },
          { label: 'custom: your own OpenAI-compatible chat and embedding endpoints, models and keys', value: 'custom' },
        ];
        if (previous.QUOTEMEM_CHAT_BASE_URL && previous.QUOTEMEM_CHAT_MODEL) {
          const savedPreset = presetOf(nonKeys(previous));
          options.push({ label: `keep the saved settings (${savedPreset ? `preset ${savedPreset}` : `${previous.QUOTEMEM_CHAT_BASE_URL}, ${previous.QUOTEMEM_CHAT_MODEL}`})`, value: 'keep' });
        }
        choice = await menu(ask.ask, output, 'Models:', options);
      }
      let keyPreset = null;
      if (choice === 'keep') {
        values = nonKeys(previous);
        keyPreset = presetOf(values);
      } else if (PRESETS[choice]) {
        const base = { ...PRESETS[choice].values };
        values = { ...base };
        override(values, base, flags, null);
        keyPreset = choice;
      } else {
        values = await askCustom(ask.ask, output, flags, previous);
      }
      // The review after the models, then the keys.
      put(values, REVIEW, review ?? await askReview(ask.ask, output, previous));
      await askKeys(ask.ask, output, values, previous, keyPreset);
    } catch (error) {
      if (error?.code === 'EOF') {
        process.stderr.write('\nquotemem setup: input ended before every question was answered; nothing was saved\n');
        return 2;
      }
      throw error;
    } finally {
      ask.close();
    }
  }

  for (const name of ['QUOTEMEM_CHAT_BASE_URL', 'QUOTEMEM_EMBED_BASE_URL']) {
    if (values[name] && !validUrl(values[name])) {
      process.stderr.write(`quotemem setup: ${name} must be an https URL (or http on this computer) without credentials\n`);
      return 2;
    }
  }
  if (values.QUOTEMEM_CHAT_STRUCTURED_OUTPUT && !STRUCTURED.has(values.QUOTEMEM_CHAT_STRUCTURED_OUTPUT)) {
    process.stderr.write('quotemem setup: --chat-structured-output is json_schema or json_object\n');
    return 2;
  }
  if (!REVIEW_NAMES.includes(values[REVIEW])) {
    process.stderr.write(`quotemem setup: ${REVIEW} is fast, standard or full\n`);
    return 2;
  }
  for (const [name, value] of Object.entries(values)) {
    if (!printable(String(value))) {
      process.stderr.write(`quotemem setup: ${name} must be one line of printable text\n`);
      return 2;
    }
  }

  writeServiceEnv(paths, serialize(values, previous));
  const preset = presetOf(values);
  output.write(`\nsaved ${paths.envFile} (mode 600; the keys are not shown)${preset ? `: preset ${preset}` : ''}, review ${values[REVIEW]}\n`);
  if (preset === 'default') {
    output.write(`Thinking is at high effort, as in the competition configuration: saving memories takes longer.\nFor faster writes run \`quotemem setup --preset fast\`, or set QUOTEMEM_CHAT_REASONING=${CHAT_REASONING.off} in that file.\n`);
  }
  // A store keeps the review it was created with; the service refuses another one.
  const savedReview = reviewOf(previous).value;
  const dataDir = dataDirOf(serviceEnvironment(env, { ...previous, ...values }), paths.dir);
  if (savedReview !== values[REVIEW] && fs.existsSync(path.join(dataDir, 'workspace'))) {
    output.write([
      `note: the memory store in ${dataDir} keeps the review it was created with${savedReview ? ` (${savedReview}, as saved before)` : ''}, and the`,
      `service refuses to open it with review ${values[REVIEW]}. To move it: \`quotemem stop\`, then`,
      `\`quotemem rebuild --from ${dataDir} --to <new data dir> --review ${values[REVIEW]}\`, and set QUOTEMEM_DATA_DIR=<new data dir> in ${paths.envFile}.`,
      `To keep the store as it is, run \`quotemem setup --review ${savedReview ?? DEFAULT_REVIEW}\`.`,
    ].join('\n') + '\n');
  }
  if (flags['no-check']) {
    output.write('not checked (--no-check); run `quotemem check` to check the settings\n');
    return 0;
  }
  output.write('checking the settings with `quotemem check` (one call to each model; the first run downloads the service)...\n');
  // The check's report goes to the output as it comes, and is read for a refused key.
  const { child } = await runServiceCommand('check', [], { env, stdio: ['ignore', 'pipe', 'inherit'], release, exact: true });
  if (!child) {
    output.write('could not check: uvx is not on PATH. Install uv (https://docs.astral.sh/uv/); the memory service runs through it.\n');
    return 1;
  }
  let report = '';
  child.stdout.on('data', (chunk) => {
    output.write(chunk);
    if (report.length < REPORT_CHARS) report += chunk.toString('utf8');
  });
  const code = await new Promise((resolve) => {
    child.on('error', () => resolve(127));
    child.on('close', (exitCode, signal) => resolve(signal ? 1 : exitCode ?? 1));
  });
  if (code === 0) {
    output.write('ok: the settings work. The memory service now starts by itself whenever an agent needs it.\n');
    return 0;
  }
  if ((preset === 'default' || preset === 'fast') && keyRefused(report)) {
    output.write([
      `FAIL: Alibaba Cloud Model Studio refused the key. The ${preset} preset takes a Model Studio API key`,
      `from ${MODEL_STUDIO_CONSOLE} (China region); a DeepSeek platform key or an international Model Studio key`,
      'is refused there. The settings are saved; to try again, run `quotemem setup`, or set DASHSCOPE_API_KEY',
      `and run \`quotemem setup --yes --preset ${preset}\`.`,
    ].join('\n') + '\n');
    return 1;
  }
  output.write(`FAIL: \`quotemem check\` exited with ${code}; fix the settings with \`quotemem setup\` (the file is saved).\n`);
  return 1;
}

/** Characters of the check's output `setup` reads; its report is one short line. */
const REPORT_CHARS = 1 << 16;

/** Whether the check's report (one JSON line on its output) says a provider refused the key. */
function keyRefused(text) {
  for (const line of text.split('\n')) {
    let value = null;
    try { value = JSON.parse(line); } catch { continue; }
    if (value && typeof value === 'object' && ['http_401', 'http_403'].includes(value.code)) return true;
  }
  return false;
}

/** `quotemem status`: whether the service runs, where, and its log. */
export async function cmdStatus(env = process.env, { release, probe }) {
  const auto = await core('autostart.mjs');
  const { hostEnv } = await import(pathToFileURL(path.join(here, '..', 'hooks', 'env.mjs')).href);
  const { resolveSettings } = await core('config.mjs');
  const settings = resolveSettings({}, hostEnv(env).env, {});
  const paths = auto.servicePaths(env);
  const state = auto.serviceState(env);
  auto.capLog(paths.log);
  const lines = [];
  const health = settings.url ? await probe(`${settings.url}/health`, { method: 'GET' }) : { status: 0, error: 'no URL' };
  const answering = health.status >= 200 && health.status < 300;
  lines.push(`memory service: ${answering ? 'running' : state.running ? 'starting (not answering yet)' : 'not running'}`);
  lines.push(`url: ${settings.url ?? '(invalid QUOTEMEM_URL)'}${health.status ? ` (/health ${health.status})` : ''}`);
  if (state.record && state.running) {
    lines.push(`pid: ${state.record.pid ?? 'unknown'} (started ${new Date(state.record.startedAt).toISOString()} by a plugin)`);
    lines.push(`version: quotemem==${state.record.release ?? '?'}`);
  } else if (answering) {
    lines.push('pid: not recorded (started by hand, not by a plugin)');
  }
  lines.push(`log: ${paths.log}`);
  lines.push(`settings: ${fs.existsSync(paths.envFile) ? paths.envFile : 'none (run `quotemem setup`)'}`);
  lines.push(`review: ${reviewLine(auto.readServiceEnv(env))}`);
  const local = settings.localService;
  lines.push(`autostart: ${!local ? 'off (QUOTEMEM_URL is not on this computer)' : local.autostart ? `on (quotemem==${release} through uvx)` : !local.configured ? 'off (run `quotemem setup`)' : 'off (QUOTEMEM_AUTOSTART=0)'}`);
  const other = answering && local ? await otherUserNote(settings, env, probe) : null;
  if (other) lines.push(`problem: ${other}`);
  process.stdout.write(`${lines.join('\n')}\n`);
  return answering && !other ? 0 : 1;
}

/**
 * Two people signed in at once share 127.0.0.1:8765: the service there may
 * be another user's, which refuses this user's token. Null when it is not.
 */
async function otherUserNote(settings, env, probe) {
  const proc = await core('proc.mjs');
  const owner = proc.portOwner(settings.localService.port);
  let refused = false;
  if (settings.token) {
    const { authHeaders } = await core('http.mjs');
    refused = (await probe(`${settings.url}/agent/status`, { method: 'GET', headers: authHeaders(settings) })).status === 401;
  }
  if (!refused && !(owner && !owner.mine)) return null;
  return `the memory service at ${settings.url} ${owner && !owner.mine ? `runs as another user (${owner.owner})` : 'refused this user\'s token'}: it may be another user's QuoteMem on this computer (two people signed in at once). Give this user a port of its own: QUOTEMEM_URL=http://127.0.0.1:${proc.userPort(env)} in ~/.quotemem/env and in the environment Claude Code, Codex and DSH start in (its plugins then start its own service there)`;
}

/** `quotemem stop`: stop the instance a plugin started (only that one). */
export async function cmdStop(env = process.env) {
  const { stopService } = await core('autostart.mjs');
  const result = await stopService(env);
  process.stdout.write(`${result.message}${result.stopped ? '; it starts again when an agent needs it, unless QUOTEMEM_AUTOSTART=0' : ''}\n`);
  return result.stopped || /not running|no started/.test(result.message) ? 0 : 1;
}

/** The review of service.env, for `status` and `doctor`. */
function reviewLine(serviceEnv) {
  const review = reviewOf(serviceEnv);
  if (!review.value) return `${REVIEW}=${serviceEnv[REVIEW]} in service.env is not fast, standard or full; the service refuses it (run \`quotemem setup\`)`;
  return `${review.value}${review.source === 'default' ? ' (the default)' : ''}; fixed when a store is created, \`quotemem rebuild --review\` moves a store to another`;
}

/** Doctor lines about the setup and autostart: [good (true/false/null), text]. */
export async function doctorLines(env, settings) {
  const auto = await core('autostart.mjs');
  const paths = auto.servicePaths(env);
  const out = [];
  let stat = null;
  try { stat = fs.statSync(paths.envFile); } catch { /* none */ }
  if (stat) {
    const loose = process.platform !== 'win32' && (stat.mode & 0o077) !== 0;
    out.push([loose ? false : true, loose ? `${paths.envFile} can be read by other users; run chmod 600 on it` : `service settings in ${paths.envFile} (keys not shown)`]);
  } else {
    out.push([null, `no service settings (${paths.envFile}); run \`quotemem setup\` once so the memory service starts by itself`]);
  }
  const serviceEnv = auto.readServiceEnv(env);
  if (serviceEnv) out.push([reviewOf(serviceEnv).value ? true : false, `review: ${reviewLine(serviceEnv)}`]);
  const local = settings.localService;
  if (!local) out.push([null, 'autostart: off, QUOTEMEM_URL is not on this computer']);
  else if (local.autostart) out.push([true, `autostart: on; a plugin starts the service when nothing answers at ${settings.url}`]);
  else if (local.configured) out.push([null, 'autostart: off (QUOTEMEM_AUTOSTART=0)']);
  else out.push([null, 'autostart: off until `quotemem setup` is run']);
  const state = auto.serviceState(env);
  if (state.record && state.running) out.push([true, `started by a plugin: pid ${state.record.pid ?? 'unknown'}, log ${paths.log}`]);
  else if (state.record) out.push([null, `the last started service (pid ${state.record.pid ?? 'unknown'}) is not running; see ${paths.log}`]);
  // Windows: background processes start through PowerShell, so they hold
  // none of an agent's handles; without it they are started directly.
  const proc = await core('proc.mjs');
  const launcher = proc.launcherCheck({ env });
  if (launcher) {
    out.push(launcher.ok
      ? [true, `background processes start through PowerShell (${launcher.detail}), holding none of an agent's handles`]
      : [false, `PowerShell cannot start background processes here (${launcher.detail}): the plugins start them directly, so a program that reads an agent's output through a pipe may wait until they end`]);
  }
  for (const file of directStartFiles(paths.dir, settings.stateDir, proc.DIRECT_START_FILE)) {
    let record = null;
    try { record = JSON.parse(fs.readFileSync(file, 'utf8')); } catch { continue; }
    const at = Number.isFinite(record?.at) ? new Date(record.at).toISOString() : 'an unknown time';
    out.push([false, `a background process was last started directly at ${at} (PowerShell: ${record?.reason ?? 'unknown'}); see ${file}`]);
  }
  return out;
}

// The direct-start records: ~/.quotemem's, each agent's state folder in it,
// and the state folder in use.
function directStartFiles(dir, stateDir, name) {
  const dirs = new Set([dir, ...(stateDir ? [stateDir] : [])]);
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) if (entry.isDirectory()) dirs.add(path.join(dir, entry.name));
  } catch {
    // none yet
  }
  return [...dirs].map((folder) => path.join(folder, name)).filter((file) => fs.existsSync(file));
}
