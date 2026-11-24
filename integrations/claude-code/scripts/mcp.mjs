#!/usr/bin/env node
// Start the QuoteMem MCP server (../server.mjs, the same server the DSH
// bundle ships) with the Claude Code plugin options applied.
//
// .mcp.json passes the options as QUOTEMEM_PLUGIN_OPTION_<KEY>. A set option
// wins over QUOTEMEM_* from the environment, exactly as in the hooks, so the
// explicit tools and the automatic memory use the same service and user.
import { randomBytes } from 'node:crypto';

import { MCP_PREFIX, applyOptions } from './env.mjs';

const env = applyOptions(process.env, MCP_PREFIX);
for (const key of Object.keys(process.env)) {
  if (!(key in env)) delete process.env[key];
}
Object.assign(process.env, env);
// Explicit tool calls get their own session id per server process.
if (!process.env.QUOTEMEM_SESSION_ID) {
  process.env.QUOTEMEM_SESSION_ID = `claude-code-mcp-${Date.now()}-${randomBytes(4).toString('hex')}`;
}

await import('../server.mjs');
