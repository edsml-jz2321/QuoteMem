#!/usr/bin/env node
// The QuoteMem MCP server for OpenCode: completes the environment from
// ~/.quotemem/env or QUOTEMEM_TOKEN_FILE (see env.js), the same way the
// plugin does, then runs the QuoteMem MCP server (integrations/dsh/server.mjs,
// or a server.mjs next to this file in a self-contained copy).
//
// The plugin already saves each conversation, so quotemem_add is hidden
// (QUOTEMEM_AUTO=1) unless QUOTEMEM_CAPTURE_EVERY=off or the environment
// sets QUOTEMEM_AUTO itself.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { hostEnv } from './env.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const { env, notes } = hostEnv(process.env);
for (const [name, value] of Object.entries(env)) {
  if (name.startsWith('QUOTEMEM_') && (process.env[name] === undefined || process.env[name] === '')) process.env[name] = value;
}
for (const note of notes) process.stderr.write(`[quotemem-mcp] ${note}\n`);
if (process.env.QUOTEMEM_AUTO === undefined) process.env.QUOTEMEM_AUTO = '1';

const server = [path.join(here, 'server.mjs'), path.join(here, '..', 'dsh', 'server.mjs')].find((file) => fs.existsSync(file));
if (!server) {
  process.stderr.write('[quotemem-mcp] server.mjs not found next to the launcher\n');
  process.exit(1);
}
await import(pathToFileURL(server).href);
