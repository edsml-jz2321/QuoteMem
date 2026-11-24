// Where a hook or the MCP launcher finds the QuoteMem settings.
//
// Some hosts start hooks without the user's shell environment: Gemini CLI
// passes an extension only HOME, PATH and the variables it declares, and
// strips names containing TOKEN, SECRET or KEY when its redaction is on; an
// editor started from the dock may never have read the shell profile. So,
// besides the environment, the settings may come from:
//
//   ~/.quotemem/env            QUOTEMEM_*=value lines (QUOTEMEM_ENV_FILE moves it)
//   QUOTEMEM_TOKEN_FILE        a file holding only the token, for example the
//                              service's own ~/.local/share/quotemem/token
//   GEMINI_CLI_QUOTEMEM_TOKEN  the token under a name Gemini CLI never strips
//
// A variable already set in the environment always wins. Nothing here throws
// or prints, and the token is never logged.
import fs from 'node:fs';
import path from 'node:path';

import { homeDir } from './core/proc.mjs';

const MAX_FILE_BYTES = 64 * 1024;
const NAME = /^QUOTEMEM_[A-Z0-9_]{1,64}$/;

function home(env) {
  return homeDir(env);
}

/** The default env file: ~/.quotemem/env. */
export function envFilePath(env = process.env) {
  return env.QUOTEMEM_ENV_FILE || path.join(home(env), '.quotemem', 'env');
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

/**
 * Parse `KEY=value` lines. Only QUOTEMEM_* names are taken; `export `, blank
 * lines, `#` comments and one pair of surrounding quotes are allowed.
 */
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
 * The environment the hook should resolve its settings from: the process
 * environment, completed (never overridden) from the env file and the token
 * sources above.
 * @returns {{env: object, notes: string[]}} `notes` are safe to log (no values).
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
  const file = readSmall(envFilePath(env));
  if (file) {
    const values = parseEnvFile(file.text);
    if (values.QUOTEMEM_TOKEN && (file.mode & 0o077) !== 0 && process.platform !== 'win32') {
      notes.push('the QuoteMem env file holds the token but other users can read it; run chmod 600 on it');
    }
    for (const [name, value] of Object.entries(values)) fill(name, value);
  }
  fill('QUOTEMEM_TOKEN', env.GEMINI_CLI_QUOTEMEM_TOKEN);
  const tokenFile = out.QUOTEMEM_TOKEN_FILE;
  if (!out.QUOTEMEM_TOKEN && tokenFile) {
    const token = readSmall(path.resolve(home(env), tokenFile.replace(/^~(?=$|[\\/])/, home(env))));
    if (token) fill('QUOTEMEM_TOKEN', token.text.trim());
    else notes.push('QUOTEMEM_TOKEN_FILE could not be read');
  }
  return { env: out, notes };
}

// Native-adapter markers (Cursor also runs Claude Code's hooks): see core/native.mjs.
export { markNative, nativeInstalled } from './core/native.mjs';
