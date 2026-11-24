#!/usr/bin/env node
// quotemem-mcp launcher for hook-based hosts: completes the environment from
// ~/.quotemem/env, QUOTEMEM_TOKEN_FILE or GEMINI_CLI_QUOTEMEM_TOKEN (see
// env.mjs), then runs the QuoteMem MCP server (integrations/dsh/server.mjs,
// or the copy next to this file in a self-contained bundle).
//
// The hooks already save each conversation, so quotemem_add is hidden
// (QUOTEMEM_AUTO=1) unless QUOTEMEM_CAPTURE_EVERY=off or the environment
// sets QUOTEMEM_AUTO itself.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { hostEnv } from './env.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { env } = hostEnv(process.env);
for (const [name, value] of Object.entries(env)) {
  if (name.startsWith('QUOTEMEM_') && process.env[name] === undefined) process.env[name] = value;
}
if (process.env.QUOTEMEM_AUTO === undefined) process.env.QUOTEMEM_AUTO = '1';

const server = [path.join(here, 'server.mjs'), path.join(here, '..', 'dsh', 'server.mjs')].find((file) => fs.existsSync(file));
if (!server) {
  process.stderr.write('[quotemem-mcp] server.mjs not found next to the launcher\n');
  process.exit(1);
}
await import(pathToFileURL(server).href);
