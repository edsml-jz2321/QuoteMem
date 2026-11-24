// Where the plugin finds the service's token, without the token in an
// agent's environment.
//
// A token exported in the shell that starts an agent is also in the agent's
// own shell tool, where the model can print it (`env`). So the plugin reads
// the token from a file by default. The order:
//
//   QUOTEMEM_TOKEN, then QUOTEMEM_HTTP_AUTH_SECRET
//                         still honoured (the service's own order), never needed
//   QUOTEMEM_TOKEN_FILE   a file holding only the token; `~` and a relative
//                         path are taken from the home folder
//   the service's own   for a service on this computer (a loopback
//   token               QUOTEMEM_URL, the default). When `quotemem setup`
//                         wrote ~/.quotemem/service.env, the plugins start
//                         that service (autostart.mjs) and read the token
//                         where it keeps it: the QUOTEMEM_TOKEN (then
//                         QUOTEMEM_HTTP_AUTH_SECRET) of service.env, else
//                         <its data-dir>/token. Otherwise <data-dir>/token
//                         as `quotemem serve` picks its data folder:
//                         QUOTEMEM_DATA_DIR, else $XDG_DATA_HOME/quotemem,
//                         else ~/.local/share/quotemem
//
// This is the one token reader: the adapters (config.mjs, http.mjs), the
// doctor and the MCP server (integrations/dsh/server.mjs, which loads this
// file from its core folder) all use it. Nothing here throws, and no message
// names the token.
import fs from 'node:fs';
import path from 'node:path';

import { isLoopbackUrl, readEnvFile } from './autostart.mjs';
import { homeDir } from './proc.mjs';

const MAX_TOKEN_FILE_BYTES = 4096;

function home(env) {
  return homeDir(env);
}

function fromHome(file, env) {
  return path.resolve(home(env), String(file).trim().replace(/^~(?=$|[\\/])/, home(env)));
}

/** The text of a file that holds only a token, trimmed; '' when it is missing, unreadable or too large. */
export function readTokenFile(file) {
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > MAX_TOKEN_FILE_BYTES) return '';
    return fs.readFileSync(file, 'utf8').trim();
  } catch {
    return '';
  }
}

const printable = (token) => /^[\x21-\x7e]+$/.test(token);

/**
 * The service's own token file on this computer: <data-dir>/token. `local`
 * (autostart.localService) is the service the plugins start, when it is set
 * up: its data folder comes from service.env, not from this environment.
 */
export function serviceTokenFile(env = process.env, local = null) {
  if (local?.configured) return path.join(local.dataDir, 'token');
  const configured = String(env.QUOTEMEM_DATA_DIR ?? '').trim();
  const xdg = String(env.XDG_DATA_HOME ?? '').trim();
  const dir = configured
    ? fromHome(configured, env)
    : path.join(xdg ? fromHome(xdg, env) : path.join(home(env), '.local', 'share'), 'quotemem');
  return path.join(dir, 'token');
}

/**
 * The token of the service on this computer: for the service the plugins
 * start (`local` set up), the one service.env gives it, else the one it
 * created in its data folder; otherwise <data-dir>/token. '' when there is
 * none yet (the service has not started once) or it is not printable.
 */
export function serviceToken(env = process.env, local = null) {
  if (local?.configured) {
    const serviceEnv = readEnvFile(local.envFile);
    const given = serviceEnv?.QUOTEMEM_TOKEN || serviceEnv?.QUOTEMEM_HTTP_AUTH_SECRET || '';
    if (given && printable(given)) return given;
  }
  const token = readTokenFile(serviceTokenFile(env, local));
  return printable(token) ? token : '';
}

/**
 * The token, in the order above.
 * @param {object} env - the environment.
 * @param {string|null} url - the service URL (QUOTEMEM_URL, or the default).
 * @param {object|null} local - autostart.localService(env, url), when known.
 * @returns {{token: string, source: 'env'|'file'|'service'|null, problem: string|null}}
 *   `source`: the environment, QUOTEMEM_TOKEN_FILE or the service's own
 *   file; `problem` when QUOTEMEM_TOKEN_FILE is set but cannot be read.
 */
export function resolveToken(env = process.env, url = null, local = null) {
  const direct = env.QUOTEMEM_TOKEN || env.QUOTEMEM_HTTP_AUTH_SECRET || '';
  if (direct) return { token: direct, source: 'env', problem: null };
  const named = String(env.QUOTEMEM_TOKEN_FILE ?? '').trim();
  if (named) {
    const token = readTokenFile(fromHome(named, env));
    return token
      ? { token, source: 'file', problem: null }
      : { token: '', source: null, problem: 'QUOTEMEM_TOKEN_FILE cannot be read; it must name a file that holds only the service\'s token' };
  }
  if (url && isLoopbackUrl(url)) {
    const token = serviceToken(env, local);
    if (token) return { token, source: 'service', problem: null };
  }
  return { token: '', source: null, problem: null };
}
