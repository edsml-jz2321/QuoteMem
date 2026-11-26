#!/usr/bin/env node
// Consistency check for the one npm package, `quotemem`, at the repository root.
//
//   node integrations/check_package.mjs            # manifests, versions and shipped copies
//   node integrations/check_package.mjs --write    # refresh every copy first, then check
//   node integrations/check_package.mjs --pack     # also `npm pack --dry-run`: what ships
//   node integrations/check_package.mjs --install  # also pack, install the tarball in a
//                                                  # temp folder and run every bin there
//
// One version covers the package, every adapter manifest, the MCP server and
// server.json (the MCP Registry entry).
// The core (integrations/core) and the MCP server (integrations/dsh/server.mjs)
// are single-source; copies exist only where a host installs a folder on its
// own (Claude Code and Codex plugins, the Gemini CLI extension, the OpenClaw
// plugin, the OpenCode plugin folder) and must be byte-identical.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { check as checkCore, coreFiles } from './core/sync.mjs';
import { check as checkBundles } from './hooks/bundle.mjs';
import { checkServerJson, serverSources } from './check_server_json.mjs';
import { docVersionProblems } from './check_versions.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const SERVER = path.join(here, 'dsh', 'server.mjs');
const SERVER_COPIES = ['claude-code/server.mjs', 'codex/server.mjs', 'gemini-cli/server.mjs', 'openclaw/server.mjs'];
// The MCP server and its copies are executable (mode 755), as git records them; git
// reads the owner's executable bit.
const SERVER_FILES = [SERVER, ...SERVER_COPIES.map((copy) => path.join(here, copy))];
const EXECUTABLE = 0o755;
const TOOLS = ['claude-code', 'dsh', 'codex', 'opencode', 'cursor', 'gemini', 'cline', 'goose', 'devin', 'continue', 'openclaw'];

const problems = [];
const ok = (value, message) => { if (!value) problems.push(message); };
const rel = (file) => path.relative(root, file);
const readJson = (file) => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    problems.push(`${rel(file)} is not valid JSON (${error.code ?? error.name})`);
    return {};
  }
};

if (process.argv.includes('--write')) {
  execFileSync(process.execPath, [path.join(here, 'core', 'sync.mjs')], { stdio: 'inherit' });
  execFileSync(process.execPath, [path.join(here, 'hooks', 'bundle.mjs')], { stdio: 'inherit' });
  for (const copy of SERVER_COPIES) fs.copyFileSync(SERVER, path.join(here, copy));
  for (const file of SERVER_FILES) fs.chmodSync(file, EXECUTABLE);
}

// ----------------------------------------------------------- root manifest
const pkg = readJson(path.join(root, 'package.json'));
const version = pkg.version;
ok(pkg.name === 'quotemem', 'package.json name must be "quotemem"');
ok(!pkg.private, 'package.json must not be private');
ok(/^\d+\.\d+\.\d+$/.test(version ?? ''), `package.json version ${version} is not x.y.z`);
ok(pkg.engines?.node === '>=20.3', 'engines.node must be ">=20.3"');
ok(pkg.keywords?.includes('dsh-plugin'), 'keywords must include "dsh-plugin" (DSH lists bundles by it)');
// The DSH version is not locked: engines.dsh and every @deepseek-ai/* peer take a lower bound only.
const ranges = [['engines.dsh', pkg.engines?.dsh]];
for (const [name, range] of Object.entries(pkg.peerDependencies ?? {})) {
  if (name.startsWith('@deepseek-ai/')) {
    ranges.push([`peerDependencies.${name}`, range]);
    ok(pkg.peerDependenciesMeta?.[name]?.optional === true, `peerDependencies.${name} must be optional (only DSH provides it)`);
  }
}
for (const [field, range] of ranges) ok(typeof range === 'string' && /^>=\s*\S+$/.test(range.trim()), `${field} must be a lower bound only (">=x"), not ${JSON.stringify(range)}`);
ok(Object.keys(pkg.dependencies ?? {}).length === 0, 'the package must have no runtime dependencies');

const patch = pkg.dsh?.bundle?.patch;
ok(typeof patch === 'string' && fs.existsSync(path.join(root, patch)), 'dsh.bundle.patch does not resolve');
ok(pkg.main && fs.existsSync(path.join(root, pkg.main)), 'main does not exist');
ok(pkg.exports?.['.'] === `./${pkg.main}`, 'exports["."] must be main (DSH requires the package by name)');
for (const sub of ['./dsh', './core', './opencode', './openclaw', './package.json']) {
  const target = pkg.exports?.[sub];
  ok(typeof target === 'string' && fs.existsSync(path.join(root, target)), `exports["${sub}"] is missing or does not exist`);
}
for (const name of ['quotemem', 'quotemem-mcp', 'quotemem-hook']) {
  const bin = pkg.bin?.[name];
  ok(typeof bin === 'string' && fs.existsSync(path.join(root, bin)), `bin ${name} is missing or does not exist`);
  if (bin && fs.existsSync(path.join(root, bin))) ok(fs.readFileSync(path.join(root, bin), 'utf8').startsWith('#!/usr/bin/env node\n'), `bin ${name} needs a node shebang`);
}
for (const entry of pkg.files ?? []) {
  if (entry.startsWith('!') || entry.includes('*')) continue;
  ok(fs.existsSync(path.join(root, entry)), `files lists ${entry}, which does not exist`);
}

// --------------------------------------------------------------- versions
const versionOf = {
  'plugin.json': readJson(path.join(root, 'plugin.json')).version,
  'integrations/claude-code/.claude-plugin/plugin.json': readJson(path.join(here, 'claude-code', '.claude-plugin', 'plugin.json')).version,
  'integrations/codex/.codex-plugin/plugin.json': readJson(path.join(here, 'codex', '.codex-plugin', 'plugin.json')).version,
  'integrations/gemini-cli/gemini-extension.json': readJson(path.join(here, 'gemini-cli', 'gemini-extension.json')).version,
  'integrations/opencode/package.json': readJson(path.join(here, 'opencode', 'package.json')).version,
  'integrations/openclaw/package.json': readJson(path.join(here, 'openclaw', 'package.json')).version,
  'integrations/goose/quotemem/plugin.json': readJson(path.join(here, 'goose', 'quotemem', 'plugin.json')).version,
  'integrations/continue/quotemem.yaml': /^version:\s*(\S+)/m.exec(fs.readFileSync(path.join(here, 'continue', 'quotemem.yaml'), 'utf8'))?.[1],
  'integrations/dsh/server.mjs': /const SERVER_VERSION = '([^']+)'/.exec(fs.readFileSync(SERVER, 'utf8'))?.[1],
  // The service release a plugin starts on demand (uvx --from quotemem==<it>).
  'integrations/core/autostart.mjs': /export const SERVICE_RELEASE = '([^']+)'/.exec(fs.readFileSync(path.join(here, 'core', 'autostart.mjs'), 'utf8'))?.[1],
};
for (const [file, value] of Object.entries(versionOf)) ok(value === version, `${file} version ${value} differs from package.json ${version}`);
// The release commands in the documentation (README examples) name it too.
for (const problem of docVersionProblems(root, version)) ok(false, problem);
// server.json, the MCP Registry entry: the same version, and its name is mcpName.
for (const problem of checkServerJson(readJson(path.join(root, 'server.json')), pkg, serverSources())) ok(false, problem);
// Folders with their own package.json (a host loads them as a plugin
// folder) are part of this package, never separate npm packages.
for (const name of ['opencode', 'openclaw']) ok(readJson(path.join(here, name, 'package.json')).private === true, `integrations/${name}/package.json must be private`);

// ----------------------------------------------------------- shipped copies
for (const problem of checkCore()) ok(false, `${problem}; run node integrations/check_package.mjs --write`);
for (const problem of checkBundles()) ok(false, `${problem}; run node integrations/check_package.mjs --write`);
// Windows has no executable bit; git keeps the recorded mode there.
if (process.platform !== 'win32') {
  for (const file of SERVER_FILES) ok((fs.statSync(file).mode & 0o100) !== 0, `${rel(file)} must be executable (mode 755); run with --write`);
}
for (const copy of SERVER_COPIES) {
  ok(fs.readFileSync(path.join(here, copy)).equals(fs.readFileSync(SERVER)), `${copy} differs from integrations/dsh/server.mjs; run with --write`);
}
const automem = fs.readFileSync(path.join(here, 'dsh', 'automem.mjs'), 'utf8');
for (const [, spec] of automem.matchAll(/from '(\.{1,2}\/[^']+)'/g)) {
  ok(spec.startsWith('../core/') || spec.startsWith('./'), `dsh/automem.mjs imports ${spec}; import the shared core from ../core/`);
}
let legacy = '';
try {
  legacy = execFileSync('git', ['grep', '-l', 'QUOTEMEM_DSH', '--', '.', ':!integrations/check_package.mjs'], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
} catch {
  // git grep exits 1 when nothing matches
}
ok(legacy === '', `QUOTEMEM_DSH_* is renamed QUOTEMEM_AUTO; still in: ${legacy.split('\n').join(', ')}`);

// ----------------------------------------------------------------- npm pack
function packList() {
  const out = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  return JSON.parse(out)[0];
}

let packInfo = null;
if (process.argv.includes('--pack') || process.argv.includes('--install')) {
  packInfo = packList();
  const packed = new Set(packInfo.files.map((f) => f.path));
  const required = [
    'package.json', 'README.md', 'LICENSE', 'NOTICE', 'plugin.json', 'mcp.json',
    ...Object.values(pkg.bin ?? {}), pkg.main, patch.replace(/^\.\//, ''),
    ...Object.values(pkg.exports ?? {}).map((target) => target.replace(/^\.\//, '')),
    ...coreFiles().map((name) => `integrations/core/${name}`),
    'integrations/cli/install.mjs', 'integrations/cli/service.mjs', 'integrations/dsh/automem.mjs', 'integrations/dsh/skills/quotemem/SKILL.md',
    'integrations/hooks/hosts.mjs', 'integrations/hooks/env.mjs', 'integrations/hooks/mcp.mjs', 'integrations/hooks/core/index.mjs',
    'integrations/claude-code/scripts/hook.mjs', 'integrations/claude-code/core/index.mjs', 'integrations/claude-code/server.mjs',
    'integrations/codex/hook.mjs', 'integrations/codex/install.mjs', 'integrations/codex/server.mjs', 'integrations/codex/core/index.mjs',
    'integrations/opencode/adapter.js', 'integrations/opencode/mcp.js', 'integrations/opencode/package.json', 'integrations/opencode/core/index.mjs',
    'integrations/openclaw/openclaw.plugin.json', 'integrations/openclaw/server.mjs', 'integrations/openclaw/core/index.mjs',
    'integrations/gemini-cli/gemini-extension.json', 'integrations/gemini-cli/hooks/hooks.json', 'integrations/gemini-cli/core/index.mjs',
    'integrations/cursor/rules/quotemem.mdc', 'integrations/cline/hooks/UserPromptSubmit', 'integrations/goose/quotemem/hooks/hooks.json',
    'integrations/devin/hooks.v1.json', 'integrations/continue/quotemem-rule.md',
  ];
  for (const file of required) ok(packed.has(file), `npm pack would leave out ${file}`);
  for (const file of packed) {
    ok(!/(^|\/)test\//.test(file), `npm pack would include a test: ${file}`);
    ok(!/(^|\/)(check_[a-z_]+|sync|bundle)\.mjs$/.test(file), `npm pack would include a maintenance script: ${file}`);
    ok(!/^(src|benchmarks)\//.test(file) && !file.endsWith('.py'), `npm pack would include Python code: ${file}`);
  }
  // A relative link in a shipped Markdown file must name a shipped file.
  for (const file of packed) {
    if (!/\.mdc?$/.test(file)) continue;
    const text = fs.readFileSync(path.join(root, file), 'utf8').replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '');
    for (const [, link] of text.matchAll(/\]\(([^)\s]+)\)/g)) {
      if (/^[a-z]+:/i.test(link) || link.startsWith('#')) continue;
      const target = path.posix.normalize(path.posix.join(path.posix.dirname(file), link.split('#')[0]));
      ok(packed.has(target) || [...packed].some((name) => name.startsWith(`${target}/`)), `${file} links to ${link}, which npm pack leaves out`);
    }
  }
}

// ---------------------------------------------- install the tarball and run it
function run(cmd, args, options) {
  try {
    const stdout = execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 60_000, ...options });
    return { code: 0, stdout };
  } catch (error) {
    return { code: error.status ?? 1, stdout: String(error.stdout ?? ''), stderr: String(error.stderr ?? '') };
  }
}

let installed = null;
if (process.argv.includes('--install') && problems.length === 0) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'qm-pack-'));
  try {
    execFileSync('npm', ['pack', '--ignore-scripts', '--pack-destination', tmp], { cwd: root, stdio: ['ignore', 'ignore', 'ignore'] });
    const tarball = fs.readdirSync(tmp).find((name) => name.endsWith('.tgz'));
    const prefix = path.join(tmp, 'app');
    const home = path.join(tmp, 'home');
    fs.mkdirSync(prefix);
    fs.mkdirSync(home);
    fs.writeFileSync(path.join(prefix, 'package.json'), '{"name":"qm-install-check","private":true}\n');
    execFileSync('npm', ['install', '--no-audit', '--no-fund', '--ignore-scripts', '--offline', path.join(tmp, tarball)], { cwd: prefix, stdio: ['ignore', 'ignore', 'pipe'] });
    const bin = (name) => path.join(prefix, 'node_modules', '.bin', name);
    const canary = 'qm-pack-check-canary-token';
    const env = { PATH: process.env.PATH, HOME: home, CODEX_HOME: path.join(home, '.codex'), QUOTEMEM_TOKEN: canary };
    const opts = { cwd: path.join(home), env };
    const checks = [];
    const expect = (label, result, pattern) => {
      const good = result.code === 0 && pattern.test(result.stdout);
      checks.push(label);
      ok(good, `installed package: ${label} failed (exit ${result.code}): ${(result.stderr || result.stdout).trim().split('\n').slice(-3).join(' | ')}`);
    };
    expect('quotemem --help', run(bin('quotemem'), ['--help'], opts), /quotemem install <tool>/);
    expect('quotemem --version', run(bin('quotemem'), ['--version'], opts), new RegExp(`^${version.replace(/\./g, '\\.')}\\n$`));
    expect('quotemem install --help', run(bin('quotemem'), ['install', '--help'], opts), /tools:/);
    expect('quotemem setup --help', run(bin('quotemem'), ['setup', '--help'], opts), /usage: quotemem setup/);
    expect('quotemem hook --help', run(bin('quotemem'), ['hook', '--help'], opts), /usage: quotemem hook/);
    expect('quotemem mcp --help', run(bin('quotemem'), ['mcp', '--help'], opts), /quotemem-mcp .*MCP server/);
    expect('quotemem-mcp --help', run(bin('quotemem-mcp'), ['--help'], opts), /quotemem-mcp .*MCP server/);
    expect('quotemem-hook --help', run(bin('quotemem-hook'), ['--help'], opts), /usage: quotemem-hook/);
    const installedDir = fs.realpathSync(path.join(prefix, 'node_modules', 'quotemem'));
    for (const tool of TOOLS) {
      const result = run(bin('quotemem'), ['install', tool], opts);
      expect(`quotemem install ${tool}`, result, /\S/);
      ok(!(result.stdout + (result.stderr ?? '')).includes(canary), `quotemem install ${tool} printed the token`);
      ok(!result.stdout.includes(root), `quotemem install ${tool} points at the source tree, not the installed package`);
      if (/[\\/]integrations[\\/]/.test(result.stdout)) ok(result.stdout.includes(installedDir), `quotemem install ${tool} does not point at the installed package`);
    }
    ok(fs.readdirSync(home).length === 0, `a dry run wrote into HOME: ${fs.readdirSync(home).join(', ')}`);
    // The package entry points resolve as an installed dependency.
    const probe = [
      "const { createRequire } = require('node:module');",
      "const r = createRequire(process.cwd() + '/package.json');",
      "const fs = require('node:fs');",
      "const dsh = r('quotemem');",
      "for (const k of ['patch', 'autoMemory', 'mcpServer', 'skillsDir']) if (!fs.existsSync(dsh[k])) throw new Error('dsh ' + k);",
      "if (r('quotemem/dsh') !== dsh) throw new Error('quotemem/dsh');",
      '(async () => {',
      "  const core = await import('quotemem/core'); if (typeof core.HookMemory !== 'function') throw new Error('core');",
      "  const oc = (await import('quotemem/opencode')).default; if (typeof oc.server !== 'function' || typeof oc.setup !== 'function') throw new Error('opencode');",
      "  const claw = (await import('quotemem/openclaw')).default; if (!claw) throw new Error('openclaw');",
      "  console.log('exports ok');",
      '})().catch((e) => { console.error(e.message); process.exit(1); });',
    ].join('\n');
    expect('require/import of quotemem, quotemem/dsh, quotemem/core, quotemem/opencode, quotemem/openclaw', run(process.execPath, ['-e', probe], { cwd: prefix, env }), /exports ok/);
    installed = { checks: checks.length };
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

if (problems.length > 0) {
  for (const problem of problems) console.error(`FAIL ${problem}`);
  process.exit(1);
}
const extra = [];
if (packInfo) extra.push(`npm pack: ${packInfo.entryCount} files, ${(packInfo.size / 1024).toFixed(1)} kB packed, ${(packInfo.unpackedSize / 1024).toFixed(1)} kB unpacked`);
if (installed) extra.push(`installed tarball: ${installed.checks} bin and entry-point checks passed`);
console.log([`ok: quotemem@${version}: one package, one version, copies match`, ...extra].join('\n'));
