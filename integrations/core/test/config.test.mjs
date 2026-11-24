#!/usr/bin/env node
// Where the adapters take the token and the auth mode from: QUOTEMEM_TOKEN,
// then QUOTEMEM_HTTP_AUTH_SECRET (the service's own order), and
// QUOTEMEM_HTTP_AUTH_MODE. No other names are read.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { CORE, tempDir } from './helpers.mjs';

const { CAPTURE_BUSY_MAX_WAIT_MS, PLAIN_URL_PROBLEM, publicSettings, resolveSettings } = await import(pathToFileURL(path.join(CORE, 'config.mjs')).href);
const { isLoopbackUrl, plainHttpUrl } = await import(pathToFileURL(path.join(CORE, 'autostart.mjs')).href);
const { resolveToken } = await import(pathToFileURL(path.join(CORE, 'token.mjs')).href);
const INTEGRATIONS = path.resolve(CORE, '..');
const hooksEnv = await import(pathToFileURL(path.join(INTEGRATIONS, 'hooks', 'env.mjs')).href);
const opencodeEnv = await import(pathToFileURL(path.join(INTEGRATIONS, 'opencode', 'env.js')).href);

// A temp HOME: a token of a service on this computer is never read from the real one.
const ADDR = { QUOTEMEM_URL: 'http://127.0.0.1:9', HOME: tempDir('qm-config-home-') };

test('token: QUOTEMEM_TOKEN first, then QUOTEMEM_HTTP_AUTH_SECRET; nothing else', () => {
  const both = resolveSettings({}, { ...ADDR, QUOTEMEM_TOKEN: 'from-token', QUOTEMEM_HTTP_AUTH_SECRET: 'from-secret' });
  assert.equal(both.problem, null);
  assert.equal(both.token, 'from-token');
  const secret = resolveSettings({}, { ...ADDR, QUOTEMEM_HTTP_AUTH_SECRET: 'from-secret' });
  assert.equal(secret.problem, null);
  assert.equal(secret.token, 'from-secret');
  const emptyToken = resolveSettings({}, { ...ADDR, QUOTEMEM_TOKEN: '', QUOTEMEM_HTTP_AUTH_SECRET: 'from-secret' });
  assert.equal(emptyToken.token, 'from-secret', 'an empty QUOTEMEM_TOKEN counts as unset');
  const none = resolveSettings({}, { ...ADDR, HOME: tempDir('qm-core-nohome-') });
  assert.equal(none.token, '');
  assert.match(none.problem, /^no token and no memory service set up; run `npx quotemem setup` once/);
  assert.doesNotMatch(none.problem, /set QUOTEMEM_TOKEN\b|export/, 'never asks for the token in the environment');
  const remote = resolveSettings({}, { ...ADDR, QUOTEMEM_URL: 'https://memory.example.com' });
  assert.match(remote.problem, /^no token; for a service on another computer, copy its token \(`quotemem serve` keeps it in <data-dir>\/token\) into a private file and set QUOTEMEM_TOKEN_FILE to that file$/);
  assert.doesNotMatch(remote.problem, /set QUOTEMEM_TOKEN\b|export/, 'never asks for the token in the environment');
});

test('auth mode: QUOTEMEM_HTTP_AUTH_MODE, default bearer; the old QUOTEMEM_AUTH_MODE is ignored', () => {
  const base = { ...ADDR, QUOTEMEM_TOKEN: 't0k' };
  assert.equal(resolveSettings({}, base).authMode, 'bearer');
  assert.equal(resolveSettings({}, { ...base, QUOTEMEM_HTTP_AUTH_MODE: ' X-API-Key ' }).authMode, 'x-api-key');
  assert.equal(resolveSettings({}, { ...base, QUOTEMEM_HTTP_AUTH_MODE: 'token' }).authMode, 'token');
  assert.equal(resolveSettings({}, { ...base, QUOTEMEM_AUTH_MODE: 'x-api-key' }).authMode, 'bearer');
  const bad = resolveSettings({}, { ...base, QUOTEMEM_HTTP_AUTH_MODE: 'basic' });
  assert.equal(bad.problem, 'QUOTEMEM_HTTP_AUTH_MODE must be bearer, token or x-api-key');
  const row = resolveSettings({ authMode: 'token' }, base);
  assert.equal(row.authMode, 'bearer', 'the host config cannot set the mode');
  assert.match(row.warnings.join('\n'), /QUOTEMEM_HTTP_AUTH_MODE/);
});

for (const [name, env] of [['hooks/env.mjs', hooksEnv], ['opencode/env.js', opencodeEnv]]) {
  test(`${name}: a token in the environment wins over the env file and the token file`, () => {
    const home = tempDir('qm-config-');
    fs.mkdirSync(path.join(home, '.quotemem'), { mode: 0o700 });
    fs.writeFileSync(path.join(home, '.quotemem', 'env'), 'QUOTEMEM_TOKEN=from-file\n', { mode: 0o600 });
    fs.writeFileSync(path.join(home, 'service-token'), 'from-token-file\n', { mode: 0o600 });

    const secret = env.hostEnv({ HOME: home, USERPROFILE: home, QUOTEMEM_HTTP_AUTH_SECRET: 'from-secret' }).env;
    assert.equal(resolveSettings({}, { ...secret, ...ADDR }).token, 'from-secret');
    const token = env.hostEnv({ HOME: home, USERPROFILE: home, QUOTEMEM_TOKEN: 'from-token', QUOTEMEM_HTTP_AUTH_SECRET: 'from-secret' }).env;
    assert.equal(resolveSettings({}, { ...token, ...ADDR }).token, 'from-token');
    const file = env.hostEnv({ HOME: home, USERPROFILE: home }).env;
    assert.equal(resolveSettings({}, { ...file, ...ADDR }).token, 'from-file');

    // The service's own token file, named in the env file.
    fs.writeFileSync(path.join(home, '.quotemem', 'env'), 'QUOTEMEM_TOKEN_FILE=~/service-token\n', { mode: 0o600 });
    const named = env.hostEnv({ HOME: home, USERPROFILE: home }).env;
    assert.equal(resolveSettings({}, { ...named, ...ADDR }).token, 'from-token-file');
  });
}

test('manifests pass the token and mode under the new names only', async () => {
  const { MCP_ENV_VARS } = await import(pathToFileURL(path.join(INTEGRATIONS, 'codex', 'install.mjs')).href);
  const codexMcp = JSON.parse(fs.readFileSync(path.join(INTEGRATIONS, 'codex', '.mcp.json'), 'utf8')).mcpServers.quotemem.env_vars;
  assert.deepEqual(codexMcp, MCP_ENV_VARS, 'the plugin and the installer pass the same variables');
  for (const name of ['QUOTEMEM_TOKEN_FILE', 'QUOTEMEM_DATA_DIR', 'QUOTEMEM_TOKEN', 'QUOTEMEM_HTTP_AUTH_SECRET', 'QUOTEMEM_HTTP_AUTH_MODE']) assert.ok(MCP_ENV_VARS.includes(name), name);
  // XDG_DATA_HOME too: the server finds the service's own token file where the hooks do.
  assert.ok(MCP_ENV_VARS.every((name) => name.startsWith('QUOTEMEM_') || name === 'XDG_DATA_HOME') && !MCP_ENV_VARS.includes('QUOTEMEM_AUTH_MODE'));
  const patch = fs.readFileSync(path.join(INTEGRATIONS, 'dsh', 'cordis.patch.yml'), 'utf8');
  assert.match(patch, /filter\(\(\[k, v\]\) => typeof v === 'string' && k\.startsWith\('QUOTEMEM_'\)\)/, 'DSH passes the QUOTEMEM_* variables only');
});

test('default URL: the port `quotemem serve` listens on (8765) in the core, every MCP server copy and the CLI', () => {
  assert.equal(resolveSettings({}, { QUOTEMEM_TOKEN: 't0k' }).url, 'http://127.0.0.1:8765');
  for (const dir of ['claude-code', 'codex', 'dsh', 'gemini-cli', 'openclaw']) {
    const server = fs.readFileSync(path.join(INTEGRATIONS, dir, 'server.mjs'), 'utf8');
    assert.match(server, /env\.QUOTEMEM_URL \|\| 'http:\/\/127\.0\.0\.1:8765'/, `${dir}/server.mjs`);
  }
  const cli = fs.readFileSync(path.join(INTEGRATIONS, 'cli', 'quotemem.mjs'), 'utf8');
  assert.match(cli, /const SERVICE_PORT = '8765';/);
});

test('capture limits: QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS is read from the environment and kept in its range, like the other capture limits', () => {
  const base = { ...ADDR, QUOTEMEM_TOKEN: 't0k' };
  const DAY = 86_400_000;
  assert.equal(CAPTURE_BUSY_MAX_WAIT_MS, 9_000_000, '2.5 hours');
  for (const [key, name, fallback, min, max] of [
    ['captureBusyMaxWaitMs', 'QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS', CAPTURE_BUSY_MAX_WAIT_MS, 60_000, 30 * DAY],
    ['captureMaxAgeMs', 'QUOTEMEM_CAPTURE_MAX_AGE_MS', DAY, 60_000, 30 * DAY],
    ['captureMaxAttempts', 'QUOTEMEM_CAPTURE_MAX_ATTEMPTS', 8, 1, 100],
  ]) {
    const read = (value, cfg = {}) => resolveSettings(cfg, value === undefined ? base : { ...base, [name]: value })[key];
    assert.equal(read(undefined), fallback, `${name}: default`);
    assert.equal(read(` ${min + 7} `), min + 7, `${name}: read from the environment`);
    assert.equal(read(String(min - 1)), min, `${name}: below the range`);
    assert.equal(read(String(max + 1)), max, `${name}: above the range`);
    assert.equal(read(String(max)), max, `${name}: the range includes its end`);
    assert.equal(read('soon'), fallback, `${name}: not a number`);
    assert.equal(read(String(min + 7), { [key]: min + 9 }), min + 9, `${name}: the host config wins`);
  }
  // The detached flusher gets it with the other settings.
  const settings = resolveSettings({}, { ...base, QUOTEMEM_CAPTURE_BUSY_MAX_WAIT_MS: '600000' });
  assert.equal(publicSettings(settings).captureBusyMaxWaitMs, 600_000);
});

// URLs Node reads as the service on this computer (127.0.0.1 or ::1), while
// Python's urlsplit and httpx read another host, or none.
const TWO_READINGS = [
  'http://127.0.0.1\\.evil.example.com',
  'http://127.0.0.1:8765\\.evil.example.com',
  'http://127.0.0.1\\@evil.example.com',
  'http://127.0.0.1\t.evil.example.com',
  'http://127.0.0.1 .evil.example.com',
  'http://127.0.0.1%2e.evil.example.com',
  'http:/127.0.0.1:8765',
  'http:\\\\127.0.0.1:8765',
  'http://[::1]\\.evil.example.com',
  'http://127.0.0.1:8765x',
  // Hosts Node rewrites to 127.0.0.1 or [::1], which Python connects to as written.
  'http://127.0.0.1.:8765',
  'http://127.1.:8765',
  'http://127.1:8765',
  'http://0x7f.0.0.1:8765',
  'http://2130706433:8765',
  'http://0177.0.0.1:8765',
  'http://127.000.000.001:8765',
  'http://[0:0:0:0:0:0:0:1]:8765',
  'http://[::0001]:8765',
];

test('QUOTEMEM_URL: an authority with a backslash, a space or another special character is refused, and gets no token', () => {
  const home = tempDir('qm-config-url-');
  fs.mkdirSync(path.join(home, '.local', 'share', 'quotemem'), { recursive: true });
  fs.writeFileSync(path.join(home, '.local', 'share', 'quotemem', 'token'), 'local-service-token\n', { mode: 0o600 });
  for (const url of TWO_READINGS) {
    const settings = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: url });
    assert.match(settings.problem ?? '', /^QUOTEMEM_URL is not a valid URL/, JSON.stringify(url));
    assert.ok(!settings.url && !settings.token && !settings.localService, JSON.stringify(url));
    assert.equal(plainHttpUrl(url), false, JSON.stringify(url));
    assert.equal(isLoopbackUrl(url), false, JSON.stringify(url));
    assert.equal(resolveToken({ HOME: home, USERPROFILE: home }, url).token, '', `${JSON.stringify(url)}: the local service's token is never read for it`);
  }
  assert.equal(resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: TWO_READINGS[0] }).problem, PLAIN_URL_PROBLEM);
  // Plain URLs pass; one on this computer still gets the local service's token.
  for (const [url, local] of [['http://127.0.0.1:8765', true], ['http://localhost:8765/', true], ['http://[::1]:8765', true], ['HTTP://LOCALHOST:8765', true],
    ['https://memory.example.com/base/?x=1#y', false], ['http://my_host.example:1', false], ['https://Memory.Example.COM', false],
    // Kept as written, and not this computer to either side.
    ['http://localhost.:8765', false], ['http://192.168.1.10:8765', false]]) {
    const settings = resolveSettings({}, { HOME: home, USERPROFILE: home, QUOTEMEM_URL: url });
    assert.ok(settings.url, url);
    assert.doesNotMatch(settings.problem ?? '', /not a valid URL/, url);
    assert.equal(settings.token, local ? 'local-service-token' : '', url);
    assert.equal(isLoopbackUrl(settings.url), local, url);
  }
});

test('homeDir: the one home every part uses, as Python\'s Path.home() finds it', async () => {
  const { homeDir } = await import(pathToFileURL(path.join(CORE, 'proc.mjs')).href);
  // Windows: USERPROFILE, then HOMEDRIVE+HOMEPATH; an MSYS or Git Bash HOME is not it.
  assert.equal(homeDir({ USERPROFILE: 'C:\\Users\\q', HOME: '/c/msys/home/q' }, 'win32'), 'C:\\Users\\q');
  assert.equal(homeDir({ HOME: '/c/msys/home/q', HOMEDRIVE: 'D:', HOMEPATH: '\\Users\\q' }, 'win32'), 'D:\\Users\\q');
  assert.equal(homeDir({ HOME: '/c/msys/home/q' }, 'win32'), os.homedir(), 'MSYS-like, no USERPROFILE: the OS\'s own');
  // Elsewhere: HOME, then the OS's own; USERPROFILE is not read.
  assert.equal(homeDir({ HOME: '/home/q', USERPROFILE: 'C:\\Users\\q' }, 'linux'), '/home/q');
  assert.equal(homeDir({ USERPROFILE: 'C:\\Users\\q' }, 'darwin'), os.homedir());
  // The service's paths and the plugins' files agree.
  const auto = await import(pathToFileURL(path.join(CORE, 'autostart.mjs')).href);
  assert.equal(auto.homeOf({ HOME: '/c/msys/home/q', HOMEDRIVE: 'D:', HOMEPATH: '\\Users\\q' }, 'win32'), 'D:\\Users\\q');
});
