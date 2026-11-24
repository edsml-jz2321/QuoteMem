#!/usr/bin/env node
// Consistency check for the Codex plugin. Run before a release:
//
//   node integrations/codex/check_plugin.mjs          # check
//   node integrations/codex/check_plugin.mjs --write  # refresh the copies, then check
//
// An installed Codex plugin holds only this folder (Codex copies it into
// $CODEX_HOME/plugins/cache/...), so it ships byte-identical copies of
// integrations/core (in core/) and of the MCP server the DSH bundle uses
// (server.mjs). This script checks those copies, the manifests and the
// marketplace entry at the repository root.
import { copyFileSync, existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TARGETS, check as checkCore } from '../core/sync.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const SERVER_SOURCE = join(root, 'integrations', 'dsh', 'server.mjs');
export const EVENTS = ['SessionStart', 'UserPromptSubmit', 'Stop', 'Interrupt', 'SessionEnd'];

if (process.argv.includes('--write')) {
  copyFileSync(SERVER_SOURCE, join(here, 'server.mjs'));
  const { execFileSync } = await import('node:child_process');
  execFileSync(process.execPath, [join(root, 'integrations', 'core', 'sync.mjs')], { stdio: 'inherit' });
}

const problems = [];
const ok = (value, message) => { if (!value) problems.push(message); };
const readJson = (file) => {
  try {
    return JSON.parse(readFileSync(file, 'utf8'));
  } catch (error) {
    problems.push(`${file.slice(root.length + 1)} is not valid JSON (${error.code ?? error.name})`);
    return {};
  }
};

// Shipped copies.
ok(TARGETS.includes('codex/core'), 'integrations/core/sync.mjs TARGETS must include codex/core');
for (const problem of checkCore().filter((p) => p.startsWith('codex/'))) problems.push(problem);
ok(existsSync(join(here, 'server.mjs')) && readFileSync(join(here, 'server.mjs')).equals(readFileSync(SERVER_SOURCE)),
  'server.mjs differs from integrations/dsh/server.mjs; run with --write');
ok(!/from '\.\.\//.test(readFileSync(join(here, 'hook.mjs'), 'utf8')), 'hook.mjs must import only from its own folder');

// Manifests.
const plugin = readJson(join(here, '.codex-plugin', 'plugin.json'));
const npmPkg = readJson(join(root, 'package.json'));
ok(plugin.name === 'quotemem', '.codex-plugin/plugin.json name must be "quotemem"');
ok(plugin.version === npmPkg.version, `.codex-plugin/plugin.json version ${plugin.version} differs from the npm package version ${npmPkg.version}`);
ok(plugin.hooks === './hooks/hooks.json', '.codex-plugin/plugin.json hooks must be ./hooks/hooks.json');
ok(plugin.mcpServers === './.mcp.json', '.codex-plugin/plugin.json mcpServers must be ./.mcp.json');

// Hooks: Codex's hooks.json takes only `description` and `hooks`.
const hooks = readJson(join(here, 'hooks', 'hooks.json'));
ok(Object.keys(hooks).every((key) => key === 'description' || key === 'hooks'), 'hooks/hooks.json may hold only description and hooks');
for (const event of EVENTS) {
  const handlers = (hooks.hooks?.[event] ?? []).flatMap((group) => group.hooks ?? []);
  ok(handlers.length === 1, `hooks/hooks.json must register exactly one ${event} handler`);
  for (const handler of handlers) {
    ok(handler.type === 'command', `${event}: the handler must be a command`);
    ok(handler.command === `node "\${PLUGIN_ROOT}/hook.mjs" ${event}`, `${event}: the command must run "\${PLUGIN_ROOT}/hook.mjs" ${event}`);
    ok(!('statusMessage' in handler), `${event}: no statusMessage (it would show on every turn)`);
    ok(!('async' in handler), `${event}: no async (Codex 0.124-0.147 skip async handlers)`);
    ok(Number.isInteger(handler.timeout) && handler.timeout >= 1, `${event}: needs a timeout in seconds`);
  }
}
for (const event of Object.keys(hooks.hooks ?? {})) ok(EVENTS.includes(event), `hooks/hooks.json has an unexpected event ${event}`);

// MCP server.
const mcp = readJson(join(here, '.mcp.json')).mcpServers?.quotemem ?? {};
ok(mcp.command === 'node' && mcp.args?.[0] === './server.mjs' && mcp.cwd === '.', '.mcp.json must run node ./server.mjs in the plugin folder');
ok(Array.isArray(mcp.env_vars) && ['QUOTEMEM_URL', 'QUOTEMEM_TOKEN_FILE', 'QUOTEMEM_DATA_DIR', 'QUOTEMEM_USER_ID'].every((name) => mcp.env_vars.includes(name)),
  '.mcp.json env_vars must forward QUOTEMEM_URL, QUOTEMEM_TOKEN_FILE, QUOTEMEM_DATA_DIR and QUOTEMEM_USER_ID (Codex starts stdio servers with a minimal environment)');
ok(!JSON.stringify(mcp.env ?? {}).includes('TOKEN'), '.mcp.json must not hold a token');

// Marketplace entry: Codex reads .agents/plugins/marketplace.json before .claude-plugin/marketplace.json.
const market = readJson(join(root, '.agents', 'plugins', 'marketplace.json'));
const entry = (market.plugins ?? []).find((p) => p.name === 'quotemem');
ok(entry?.source?.source === 'local' && entry?.source?.path === './integrations/codex', '.agents/plugins/marketplace.json must list quotemem at ./integrations/codex');

if (problems.length > 0) {
  for (const problem of problems) console.error(`FAIL ${problem}`);
  process.exit(1);
}
console.log(`ok: Codex plugin quotemem@${plugin.version} is consistent`);
