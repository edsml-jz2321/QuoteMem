// Where the OpenCode plugin and its MCP launcher find the QuoteMem settings.
//
// OpenCode 2.x runs as a background service and the desktop app may start
// without the user's shell profile, so QUOTEMEM_URL and QUOTEMEM_TOKEN may be
// missing from its environment. Besides the environment, they may come from:
//
//   ~/.quotemem/env        QUOTEMEM_*=value lines (QUOTEMEM_ENV_FILE moves it)
//   QUOTEMEM_TOKEN_FILE    a file holding only the token
//
// This is the same file the other QuoteMem adapters read
// (integrations/hooks/env.mjs). A variable already set in the environment
// always wins. Nothing here throws or prints; the token is never logged.
import fs from 'node:fs';
import path from 'node:path';

import { homeDir } from './core/proc.mjs';

const MAX_FILE_BYTES = 64 * 1024;
const NAME = /^QUOTEMEM_[A-Z0-9_]{1,64}$/;

function home(env) {
  return homeDir(env);
}

function readSmall(file) {
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > MAX_FILE_BYTES) return null;
    return { text: fs.readFileSync(file, 'utf8'), mode: stat.mode };
  } catch {
    return null;
  }
}

/** `KEY=value` lines: only QUOTEMEM_* names; `export `, `#` comments and one pair of quotes allowed. */
export function parseEnvFile(text) {
  const out = {};
  for (const raw of String(text ?? '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const match = /^(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*)$/.exec(line);
    if (!match || !NAME.test(match[1])) continue;
    let value = match[2].trim();
    if (value.length >= 2 && (value[0] === '"' || value[0] === "'") && value.at(-1) === value[0]) value = value.slice(1, -1);
    out[match[1]] = value;
  }
  return out;
}

/**
 * The environment completed (never overridden) from the env file and the token file.
 * @returns {{env: object, notes: string[]}} `notes` hold no values and are safe to log.
 */
export function hostEnv(env = process.env) {
  const out = { ...env };
  const notes = [];
  const fill = (name, value) => {
    if ((out[name] === undefined || out[name] === '') && value !== undefined && value !== '') out[name] = value;
  };
  // A token in the environment wins over every file: QUOTEMEM_TOKEN, then
  // QUOTEMEM_HTTP_AUTH_SECRET (the service's own order).
  fill('QUOTEMEM_TOKEN', env.QUOTEMEM_HTTP_AUTH_SECRET);
  const file = readSmall(env.QUOTEMEM_ENV_FILE || path.join(home(env), '.quotemem', 'env'));
  if (file) {
    const values = parseEnvFile(file.text);
    if (values.QUOTEMEM_TOKEN && (file.mode & 0o077) !== 0 && process.platform !== 'win32') {
      notes.push('the QuoteMem env file holds the token but other users can read it; run chmod 600 on it');
    }
    for (const [name, value] of Object.entries(values)) fill(name, value);
  }
  const tokenFile = out.QUOTEMEM_TOKEN_FILE;
  if (!out.QUOTEMEM_TOKEN && tokenFile) {
    const token = readSmall(path.resolve(home(env), String(tokenFile).replace(/^~(?=$|[\\/])/, home(env))));
    if (token) fill('QUOTEMEM_TOKEN', token.text.trim());
    else notes.push('QUOTEMEM_TOKEN_FILE could not be read');
  }
  return { env: out, notes };
}
