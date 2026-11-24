#!/usr/bin/env node
// Consistency check for the Claude Code plugin. Run before a release:
//
//   node integrations/claude-code/check_plugin.mjs          # check
//   node integrations/claude-code/check_plugin.mjs --write  # refresh the copies, then check
//
// An installed Claude Code plugin holds only this folder, so it ships
// byte-identical copies of integrations/core (in core/) and of the MCP
// server the DSH bundle uses (server.mjs). This script checks those copies,
// the manifests and the marketplace entry at the repository root.
import { copyFileSync, existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TARGETS, check as checkCore } from '../core/sync.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const SERVER_SOURCE = join(root, 'integrations', 'dsh', 'server.mjs');
const EVENTS = ['SessionStart', 'UserPromptSubmit', 'Stop', 'StopFailure', 'SessionEnd'];

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
ok(TARGETS.includes('claude-code/core'), 'integrations/core/sync.mjs TARGETS must include claude-code/core');
for (const problem of checkCore().filter((p) => p.startsWith('claude-code/'))) problems.push(problem);
ok(existsSync(join(here, 'server.mjs')) && readFileSync(join(here, 'server.mjs')).equals(readFileSync(SERVER_SOURCE)),
  'server.mjs differs from integrations/dsh/server.mjs; run with --write');

// Manifests.
const plugin = readJson(join(here, '.claude-plugin', 'plugin.json'));
const npmPkg = readJson(join(root, 'package.json'));
ok(plugin.name === 'quotemem', 'plugin.json name must be "quotemem"');
ok(plugin.version === npmPkg.version, `plugin.json version ${plugin.version} differs from the npm package version ${npmPkg.version}`);
ok(!('engines' in plugin), 'plugin.json must not set engines (Claude Code 2.0.15-2.1.75 reject it)');
for (const [key, option] of Object.entries(plugin.userConfig ?? {})) {
  ok(!('default' in option), `userConfig.${key} must have no default, so QUOTEMEM_* from the environment still applies`);
  ok(option.type && option.title && option.description, `userConfig.${key} needs type, title and description`);
}
ok(plugin.userConfig?.token?.sensitive === true, 'userConfig.token must be sensitive');

const hooks = readJson(join(here, 'hooks', 'hooks.json'));
for (const event of EVENTS) {
  const handlers = (hooks.hooks?.[event] ?? []).flatMap((group) => group.hooks ?? []);
  ok(handlers.length === 1, `hooks.json must register exactly one ${event} handler`);
  for (const handler of handlers) {
    ok(handler.type === 'command' && handler.command === 'node' && Array.isArray(handler.args), `${event}: use exec form (command "node" with args)`);
    ok(handler.args?.[0] === '${CLAUDE_PLUGIN_ROOT}/scripts/hook.mjs' && handler.args?.[1] === event, `${event}: args must be [hook.mjs, "${event}"]`);
    ok(!handler.async, `${event}: must not be async (async hooks are killed at teardown under claude -p)`);
  }
}
for (const event of Object.keys(hooks.hooks ?? {})) ok(EVENTS.includes(event), `hooks.json registers ${event}, which hook.mjs ignores`);

const mcp = readJson(join(here, '.mcp.json'));
const server = mcp.mcpServers?.quotemem;
ok(server?.command === 'node' && server?.args?.[0] === '${CLAUDE_PLUGIN_ROOT}/scripts/mcp.mjs', '.mcp.json must start scripts/mcp.mjs');
ok(server?.env?.QUOTEMEM_AUTO === '1', '.mcp.json must set QUOTEMEM_AUTO=1 (capture is automatic)');
for (const key of Object.keys(plugin.userConfig ?? {})) {
  ok(server?.env?.[`QUOTEMEM_PLUGIN_OPTION_${key.toUpperCase()}`] === `\${user_config.${key}}`, `.mcp.json must pass user_config.${key}`);
}

const market = readJson(join(root, '.claude-plugin', 'marketplace.json'));
const entry = (market.plugins ?? []).find((p) => p.name === 'quotemem');
ok(entry && entry.source === './integrations/claude-code', 'marketplace.json must list quotemem with source ./integrations/claude-code');

// Shipped code imports only files inside the plugin.
for (const file of ['scripts/hook.mjs', 'scripts/adapter.mjs', 'scripts/mcp.mjs', 'scripts/env.mjs']) {
  const source = readFileSync(join(here, file), 'utf8');
  for (const match of source.matchAll(/(?:from\s+|import\()\s*['"]([^'"]+)['"]/g)) {
    const spec = match[1];
    if (spec.startsWith('node:')) continue;
    ok(spec.startsWith('./') || spec === '../core/index.mjs' || spec === '../server.mjs', `${file} imports ${spec}, outside the plugin`);
    if (spec.startsWith('.')) ok(existsSync(resolve(here, dirname(file), spec)), `${file} imports ${spec}, which does not exist`);
  }
}

for (const problem of problems) console.error(`FAIL ${problem}`);
if (problems.length > 0) process.exit(1);
console.log(`ok: Claude Code plugin quotemem@${plugin.version} is consistent`);
