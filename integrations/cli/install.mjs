// `quotemem install <tool> [--write]`: set up QuoteMem in one agent.
//
// Every installer first makes a plan: the commands to run and the files to
// write. Without --write the plan is only printed. With --write the files are
// written; commands that belong to the agent's own CLI (`claude plugin …`,
// `dsh plugin …`, `gemini extensions …`) are printed for you to run.
//
// The token is never written anywhere, and never asked for in an agent's
// environment (there the agent's shell tool, and so the model, could read
// it). Every hook and MCP server reads it from a file when it runs: the
// service's own <data-dir>/token for a service on this computer, else the
// file QUOTEMEM_TOKEN_FILE names (core/token.mjs).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { homeDir } from '../core/proc.mjs';

import { install as codexInstall, writeConfigFile } from '../codex/install.mjs';
import { envFilePath, parseEnvFile } from '../hooks/env.mjs';
import { MANAGED } from './service.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const PACKAGE_DIR = path.resolve(here, '..', '..');
const integration = (...parts) => path.join(PACKAGE_DIR, 'integrations', ...parts);
const HOOK_RUNNER = integration('hooks', 'quotemem-hook.mjs');
const REPO = 'edsml-jz2321/QuoteMem';
const OWN_MARK = 'quotemem-hook';

const q = (text) => `"${String(text).replace(/(["\\$`])/g, '\\$1')}"`;
// A path in a hook command. On Windows it is written as given, in quotes: a
// backslash there is a path separator, not an escape, and a file name
// cannot hold a quote. Elsewhere it is quoted for a POSIX shell.
const commandPath = (text) => (process.platform === 'win32' ? `"${String(text)}"` : q(text));

/** A shell command that runs the hook runner for one agent and event. */
function hookCommand(node, host, event) {
  return `${commandPath(node)} ${commandPath(HOOK_RUNNER)} ${host} ${event}`;
}

function readText(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function readJsonFile(file) {
  const text = readText(file);
  if (text === null || !text.trim()) return { value: null };
  try {
    return { value: JSON.parse(text) };
  } catch {
    return { problem: `${file} is not valid JSON; fix it first (nothing was changed)` };
  }
}

const json = (value) => `${JSON.stringify(value, null, 2)}\n`;
const isOurs = (command) => typeof command === 'string' && command.includes(OWN_MARK);

/** Claude-format hooks ({Event: [{hooks: [{command}]}]}) with our handlers replaced. */
function mergeGroupedHooks(existing, ours) {
  const out = existing && typeof existing === 'object' ? structuredClone(existing) : {};
  for (const [event, groups] of Object.entries(out)) {
    if (!Array.isArray(groups)) continue;
    out[event] = groups
      .map((group) => (group && Array.isArray(group.hooks) ? { ...group, hooks: group.hooks.filter((h) => !isOurs(h?.command)) } : group))
      .filter((group) => !(group && Array.isArray(group.hooks) && group.hooks.length === 0));
    if (out[event].length === 0) delete out[event];
  }
  for (const [event, handler] of Object.entries(ours)) out[event] = [...(out[event] ?? []), { hooks: [handler] }];
  return out;
}

/** Cursor hooks ({version, hooks: {event: [{command}]}}) with our handlers replaced. */
function mergeCursorHooks(existing, ours) {
  const out = existing && typeof existing === 'object' ? structuredClone(existing) : {};
  out.version ??= 1;
  const hooks = out.hooks && typeof out.hooks === 'object' ? out.hooks : {};
  for (const [event, list] of Object.entries(hooks)) {
    if (!Array.isArray(list)) continue;
    hooks[event] = list.filter((h) => !isOurs(h?.command));
    if (hooks[event].length === 0) delete hooks[event];
  }
  for (const [event, handler] of Object.entries(ours)) hooks[event] = [...(hooks[event] ?? []), handler];
  out.hooks = hooks;
  return out;
}

/** An MCP config ({mcpServers: {...}}) with the quotemem entry set. */
function mergeMcp(existing, entry) {
  const out = existing && typeof existing === 'object' ? structuredClone(existing) : {};
  out.mcpServers = { ...(out.mcpServers && typeof out.mcpServers === 'object' ? out.mcpServers : {}), quotemem: entry };
  return out;
}

/** A small sh script that runs the hook, or answers `fallback` when the package is gone. */
function hookScript(title, node, host, event, fallback) {
  return [
    '#!/bin/sh',
    `# ${title}, written by \`quotemem install\`.`,
    '# It runs the hook from the installed quotemem package.',
    '# If that package is gone, the turn goes on.',
    `node=${q(node)}`,
    `hook=${q(HOOK_RUNNER)}`,
    'if [ -x "$node" ] && [ -f "$hook" ]; then',
    `  exec "$node" "$hook" ${host} ${event}`,
    'fi',
    'cat >/dev/null',
    ...(fallback ? [`printf '%s\\n' '${fallback}'`] : []),
    'exit 0',
    '',
  ].join('\n');
}

// DSH, Claude Code and Codex read the settings from their environment only;
// the other adapters also read ~/.quotemem/env. None needs the token there.
const ENV_ONLY = new Set(['claude-code', 'dsh', 'codex']);
function envHint(tool) {
  return [
    'Token: nothing to set for a service on this computer (its own <data-dir>/token is read).',
    ENV_ONLY.has(tool)
      ? 'For a service elsewhere, save its token in a file (chmod 600) and set QUOTEMEM_URL and\nQUOTEMEM_TOKEN_FILE (the file\'s path) where the agent starts. Do not export the token itself:\nthe agent\'s shell tool would see it.'
      : 'For a service elsewhere, save its token in a file (chmod 600) and put QUOTEMEM_URL and\nQUOTEMEM_TOKEN_FILE (the file\'s path) in ~/.quotemem/env (NAME=value lines, chmod 600).',
    'QUOTEMEM_USER_ID defaults to your OS user name in every agent, so all agents share one memory.',
    'Check with: quotemem doctor',
  ];
}

// ------------------------------------------------------------------ tools
// Each returns {lines, files: [{file, text, mode?, guard?}], problem?}.
// `guard(existingText)` returns a problem when a file must not be replaced.

const TOOLS = {
  'claude-code': {
    summary: 'Claude Code plugin (hooks and MCP server) from the GitHub marketplace',
    plan() {
      return {
        lines: [
          'Claude Code installs plugins from a marketplace. Run:',
          `  claude plugin marketplace add ${REPO} --sparse .claude-plugin integrations`,
          '  claude plugin install quotemem@quotemem',
          'Claude Code asks for the URL, token and memory owner when the plugin is enabled;',
          'leave them empty to use QUOTEMEM_URL, the service\'s token file and QUOTEMEM_USER_ID.',
          'Nothing is written by this command.',
        ],
        files: [],
      };
    },
  },

  dsh: {
    summary: 'DeepSeek Harness bundle (automatic memory, MCP server and skill)',
    plan() {
      return {
        lines: [
          'DSH installs the quotemem npm package as a bundle. DSH needs Node 24 and pnpm for this. Run:',
          '  dsh plugin --profile quotemem add quotemem',
          'then start DSH with: dsh --profile quotemem',
          'Start DSH with QUOTEMEM_URL (and QUOTEMEM_TOKEN_FILE for a service elsewhere) in its environment.',
          'Nothing is written by this command.',
        ],
        files: [],
      };
    },
  },

  codex: {
    summary: 'Codex CLI hooks and MCP server, written into ~/.codex',
    run({ write, node, args }) {
      const codexHome = path.resolve(args['codex-home'] ?? process.env.CODEX_HOME ?? path.join(homeDir(), '.codex'));
      const result = codexInstall({ codexHome, node, write });
      const lines = [...result.lines];
      lines.push('', `Or install the plugin instead: codex plugin marketplace add ${REPO} --sparse .agents --sparse integrations/codex`, '  then: codex plugin add quotemem@quotemem (and trust its hooks with /hooks)');
      return { ok: result.ok, lines };
    },
  },

  opencode: {
    summary: 'OpenCode 1.x or 2.x plugin and MCP server (a config snippet)',
    plan({ node }) {
      const dir = integration('opencode');
      const mcp = integration('opencode', 'mcp.js');
      const v1 = {
        $schema: 'https://opencode.ai/config.json',
        plugin: [dir],
        mcp: { quotemem: { type: 'local', command: [node, mcp], enabled: true } },
      };
      const v2 = { plugins: [dir], mcp: { servers: { quotemem: { type: 'local', command: [node, mcp] } } } };
      return {
        lines: [
          'Merge the block for your OpenCode line (`opencode --version`) into',
          '~/.config/opencode/opencode.jsonc, then restart OpenCode.',
          'OpenCode keeps comments in that file, so this command does not edit it.',
          '', '--- OpenCode 1.x (opencode-ai)', JSON.stringify(v1, null, 2),
          '', '--- OpenCode 2.x (@opencode/cli)', JSON.stringify(v2, null, 2),
        ],
        files: [],
      };
    },
  },

  cursor: {
    summary: 'Cursor user hooks and MCP server, merged into ~/.cursor',
    // Cursor's CLI does not pass its own environment to the MCP server it
    // starts: only ~/.quotemem/env reaches it.
    plan({ node }) {
      const dir = path.join(homeDir(), '.cursor');
      const hooksFile = path.join(dir, 'hooks.json');
      const mcpFile = path.join(dir, 'mcp.json');
      const hooks = readJsonFile(hooksFile);
      const mcp = readJsonFile(mcpFile);
      const problem = hooks.problem ?? mcp.problem;
      if (problem) return { problem };
      const timeouts = { sessionStart: 5, beforeSubmitPrompt: 15, afterAgentResponse: 5, stop: 5, sessionEnd: 5 };
      const ours = Object.fromEntries(Object.entries(timeouts).map(([event, timeout]) => [event, { command: hookCommand(node, 'cursor', event), timeout }]));
      return {
        lines: [
          'Each session starts with a digest of what is known about you. On Cursor 3.20.17 and',
          'later, the prompt hook adds recall and forget notes to each turn. On earlier versions',
          'recall is the digest plus the MCP tools, which the rule tells the model to use.',
          'In each project, copy the rule:',
          `  mkdir -p .cursor/rules && cp ${q(integration('cursor', 'rules', 'quotemem.mdc'))} .cursor/rules/`,
          ...shellOnlyWarning(),
        ],
        files: [
          { file: hooksFile, text: json(mergeCursorHooks(hooks.value, ours)) },
          { file: mcpFile, text: json(mergeMcp(mcp.value, { command: node, args: [HOOK_RUNNER, 'mcp'] })) },
        ],
      };
    },
  },

  gemini: {
    summary: 'Gemini CLI extension (hooks, MCP server and GEMINI.md)',
    plan() {
      return {
        lines: [
          'Gemini CLI copies an extension folder. Run:',
          `  gemini extensions install ${q(integration('gemini-cli'))}`,
          'Gemini asks for the URL, token and user id; empty answers fall back to ~/.quotemem/env.',
          'Nothing is written by this command.',
        ],
        files: [],
      };
    },
  },

  cline: {
    summary: 'Cline hook files in ~/Documents/Cline/Hooks',
    plan({ node, args }) {
      const dir = path.resolve(args.dir ?? path.join(homeDir(), 'Documents', 'Cline', 'Hooks'));
      const files = ['UserPromptSubmit', 'TaskComplete', 'TaskCancel'].map((event) => ({
        file: path.join(dir, event),
        text: hookScript(`QuoteMem hook for Cline (${event})`, node, 'cline', event, '{"cancel":false}'),
        mode: 0o755,
        guard: (old) => (old.includes('QuoteMem hook for Cline') ? null : `${path.join(dir, event)} is another hook; move it first`),
      }));
      return {
        lines: [
          'Then turn on Enable Hooks in Cline\'s settings. For the MCP tools, add this server in',
          `Cline's MCP settings: "quotemem": {"command": ${JSON.stringify(node)}, "args": [${JSON.stringify(HOOK_RUNNER)}, "mcp"]}`,
        ],
        files,
      };
    },
  },

  goose: {
    summary: 'goose plugin in ~/.agents/plugins/quotemem',
    plan({ node }) {
      const dir = path.join(homeDir(), '.agents', 'plugins', 'quotemem');
      const source = integration('goose', 'quotemem');
      const files = [
        { file: path.join(dir, 'plugin.json'), text: fs.readFileSync(path.join(source, 'plugin.json'), 'utf8') },
        { file: path.join(dir, 'hooks', 'hooks.json'), text: fs.readFileSync(path.join(source, 'hooks', 'hooks.json'), 'utf8') },
        {
          file: path.join(dir, 'scripts', 'quotemem-hook.sh'),
          text: hookScript('QuoteMem hook for goose', node, 'goose', '"$1"', null),
          mode: 0o755,
        },
      ];
      return {
        lines: [
          'goose must read the QuoteMem context file. Put this in your shell profile:',
          '  export GOOSE_MOIM_MESSAGE_FILE=~/.quotemem/goose/moim.md',
          'For the MCP tools, add to ~/.config/goose/config.yaml:',
          '  extensions:',
          '    quotemem:',
          '      type: stdio',
          '      name: quotemem',
          `      cmd: ${JSON.stringify(node)}`,
          `      args: [${JSON.stringify(HOOK_RUNNER)}, mcp]`,
          '      enabled: true',
          '      timeout: 300',
        ],
        files,
      };
    },
  },

  devin: {
    summary: 'Devin hooks in the project\'s .devin/hooks.v1.json',
    plan({ node, project }) {
      const file = path.join(project, '.devin', 'hooks.v1.json');
      const existing = readJsonFile(file);
      if (existing.problem) return { problem: existing.problem };
      const timeouts = { UserPromptSubmit: 15, Stop: 5, SessionEnd: 5 };
      const ours = Object.fromEntries(Object.entries(timeouts).map(([event, timeout]) => [event, { type: 'command', command: hookCommand(node, 'devin', event), timeout }]));
      return {
        lines: [
          'For every project, put the same object under "hooks" in ~/.config/devin/config.json instead.',
          'Each turn saves your message and the final reply (Devin CLI 3000.5.20 and later send it to',
          'the Stop hook); with an older Devin, only your messages are saved.',
          `For the MCP tools, add a stdio server: command ${JSON.stringify(node)}, args [${JSON.stringify(HOOK_RUNNER)}, "mcp"].`,
        ],
        files: [{ file, text: json(mergeGroupedHooks(existing.value, ours)) }],
      };
    },
  },

  continue: {
    summary: 'Continue MCP server and rule in the project\'s .continue folder',
    plan({ node, project }) {
      const version = JSON.parse(fs.readFileSync(path.join(PACKAGE_DIR, 'package.json'), 'utf8')).version;
      const yaml = [
        '# QuoteMem MCP server for Continue, written by `quotemem install continue`.',
        'name: QuoteMem',
        `version: ${version}`,
        'schema: v1',
        'mcpServers:',
        '  - name: quotemem',
        `    command: ${JSON.stringify(node)}`,
        '    args:',
        `      - ${JSON.stringify(HOOK_RUNNER)}`,
        '      - mcp',
        '    env:',
        '      QUOTEMEM_AUTO: "0"',
        '',
      ].join('\n');
      return {
        lines: ['Continue has no per-message hooks: the model saves and searches memory through the tools, as the rule asks.'],
        files: [
          { file: path.join(project, '.continue', 'mcpServers', 'quotemem.yaml'), text: yaml },
          { file: path.join(project, '.continue', 'rules', 'quotemem.md'), text: fs.readFileSync(integration('continue', 'quotemem-rule.md'), 'utf8') },
        ],
      };
    },
  },

  openclaw: {
    summary: 'OpenClaw plugin (in-process memory and MCP server)',
    plan() {
      return {
        lines: [
          'OpenClaw installs a plugin folder. Run:',
          `  openclaw plugins install ${q(integration('openclaw'))}`,
          '  openclaw plugins inspect quotemem --runtime --json   # check that it loaded',
          'Nothing is written by this command.',
        ],
        files: [],
      };
    },
  },
};

const ALIASES = { claude: 'claude-code', 'gemini-cli': 'gemini', windsurf: 'devin', 'continue-dev': 'continue' };

/**
 * The QUOTEMEM_* settings this shell sets that ~/.quotemem/env does not, or
 * sets otherwise: their names only, sorted. The default file is read, the
 * one a process started without this shell's environment reads.
 */
export function shellOnlySettings(env = process.env) {
  return shellNames(env).filter((name) => !serviceSetting(name));
}

// The service's model settings and keys: `quotemem setup` writes them to
// ~/.quotemem/service.env, and the service takes them from there alone.
function serviceSetting(name) {
  return MANAGED.includes(name) || /^QUOTEMEM_(?:CHAT|EMBED)_/.test(name);
}

// The QUOTEMEM_* names set in this shell that ~/.quotemem/env does not hold
// with the same value.
function shellNames(env) {
  let saved = {};
  try {
    saved = parseEnvFile(fs.readFileSync(envFilePath({ HOME: env.HOME, USERPROFILE: env.USERPROFILE }), 'utf8'));
  } catch {
    // no file
  }
  return Object.keys(env).filter((name) => /^QUOTEMEM_[A-Z0-9_]+$/.test(name) && env[name] !== '' && saved[name] !== env[name]).sort();
}

function shellOnlyWarning(env = process.env) {
  const out = [];
  const service = shellNames(env).filter(serviceSetting);
  if (service.length > 0) {
    out.push(
      '',
      `note: ${service.join(', ')} ${service.length === 1 ? 'is' : 'are'} set in this shell. The service's model settings and`,
      'keys belong in ~/.quotemem/service.env (written by `quotemem setup`), not in ~/.quotemem/env.',
    );
  }
  const names = shellOnlySettings(env);
  if (names.length === 0) return out;
  return [
    ...out,
    '',
    `warning: ${names.join(', ')} ${names.length === 1 ? 'is' : 'are'} set in this shell, not in ~/.quotemem/env. Cursor's CLI`,
    'does not pass the shell\'s environment to the MCP server, so the memory tools would not see',
    'them. Put them in ~/.quotemem/env (NAME=value lines, chmod 600); for the token, put',
    'QUOTEMEM_TOKEN_FILE with the token file\'s path there, never the token itself.',
  ];
}

export function toolOf(name) {
  const key = String(name ?? '').trim().toLowerCase();
  const tool = ALIASES[key] ?? key;
  return Object.hasOwn(TOOLS, tool) ? tool : null;
}

export function toolList() {
  return Object.entries(TOOLS).map(([name, tool]) => `  ${name.padEnd(12)} ${tool.summary}`);
}

function inNpxCache(dir) {
  return /[\\/]_npx[\\/]/.test(dir);
}

/**
 * Plan, and with `write` apply, the install for one tool.
 * @returns {{ok: boolean, lines: string[]}}
 */
export function install(name, { write = false, node = process.execPath, project = process.cwd(), args = {} } = {}) {
  const tool = TOOLS[toolOf(name)];
  if (!tool) return { ok: false, lines: [`unknown tool ${JSON.stringify(name)}; one of: ${Object.keys(TOOLS).join(', ')}`] };
  const lines = [];
  if (inNpxCache(PACKAGE_DIR)) {
    lines.push('warning: this quotemem runs from the npx cache, which npm may clear. Paths written now',
      'would then break. Install it for good first: npm install -g quotemem', '');
  }
  if (tool.run) {
    const result = tool.run({ write, node, project, args });
    return { ok: result.ok, lines: [...lines, ...result.lines, '', ...envHint(toolOf(name))] };
  }
  const plan = tool.plan({ node, project, args });
  if (plan.problem) return { ok: false, lines: [...lines, plan.problem] };
  const problems = [];
  for (const entry of plan.files) {
    const old = readText(entry.file);
    const problem = old !== null && entry.guard ? entry.guard(old) : null;
    if (problem) problems.push(problem);
  }
  if (problems.length > 0) return { ok: false, lines: [...lines, ...problems.map((p) => `${p} (nothing was changed)`)] };

  if (plan.files.length > 0) {
    lines.push(`${write ? '' : '[dry run] '}files for ${toolOf(name)}:`);
    for (const entry of plan.files) lines.push(`  ${entry.file}${readText(entry.file) === null ? ' (new)' : ' (updated; other entries kept)'}`);
    lines.push('');
  }
  lines.push(...plan.lines);
  if (!write && plan.files.length > 0) {
    for (const entry of plan.files) lines.push('', `--- ${entry.file}`, entry.text.trimEnd());
    lines.push('', 'Run again with --write to write these files.');
  }
  if (write) {
    // An existing file keeps its permission bits and gets a .quotemem.bak
    // copy first; a symlinked dotfile is written through to its target.
    const backups = plan.files.map((entry) => writeConfigFile(entry.file, entry.text, { mode: entry.mode ?? null })).filter(Boolean);
    if (plan.files.length > 0) lines.push('', `wrote ${plan.files.length} file${plan.files.length === 1 ? '' : 's'}`);
    for (const backup of backups) lines.push(`backup of the old file: ${backup}`);
  }
  lines.push('', ...envHint(toolOf(name)));
  return { ok: true, lines };
}
